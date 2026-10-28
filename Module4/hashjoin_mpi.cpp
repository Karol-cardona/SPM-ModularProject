// hashjoin_mpi.cpp
//
// This file REUSES the shared header common.hpp from Modules 2-3 unchanged, so
// the mapping function (multiply_shift), the build-phase table (FlatCountMap),
// the per-partition join (join_one_partition), the checksum scheme, and the
// uniform/skewed input generators are identical to the sequential baseline. The
// MPI layer only adds: distributed input generation, the all-to-all
// redistribution, and the global reduction.
//
// ---------------------------------------------------------------------------
// STRATEGY  (scatter -> map + local reduce -> global reduce)
// ---------------------------------------------------------------------------
// The join is partition-local: equal keys always map to the same partition id,
// hence they are always co-located. We use the partition as the unit of
// distribution.
//   0. Each rank generates a disjoint contiguous slice of R and S. The Module
//      1-3 generators are stateful (splitmix64), but both the uniform and the
//      skewed generator consume exactly the same number of PRNG draws per
//      record (1 and 2 respectively, in BOTH branches of the skewed case), so
//      record i is positionally addressable and a rank can reconstruct exactly
//      the keys global indices [lo,hi) would have in the sequential generator.
//   1. Each rank maps its keys to a partition id (same function as Modules 1-3)
//      and then to an owner rank owner(pid) = pid % nprocs, buckets records by
//      destination, and exchanges them with one MPI_Alltoallv per relation.
//   2. Each rank runs the unmodified Module 2/3 local join on the data it now
//      owns (every record mapping to one of its partitions). In the hybrid
//      build this partition loop is parallelized with OpenMP for nb=0/1; nb=2
//      has its own fused build/probe loop and runs serially.
//   3. join_count and the two checksums are summed with MPI_Reduce(MPI_SUM).
//
// join_count and the checksums count only pairs of equal keys, so they are
// invariant to P, to nprocs, and to the partition function; the distributed
// output therefore matches the sequential baseline bit-for-bit.
//
// ---------------------------------------------------------------------------
// Build:   mpicxx -O3 -march=native -std=c++20 hashjoin_mpi.cpp -o hashjoin_mpi
// Hybrid:  mpicxx -O3 -march=native -std=c++20 -fopenmp hashjoin_mpi.cpp -o hashjoin_mpi_omp
// Run:     srun --mpi=pmix -N 4 --ntasks-per-node=1 ./hashjoin_mpi
//             -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256
// ---------------------------------------------------------------------------

#include <mpi.h>
#include "common.hpp"   // shared, unchanged, from Modules 2-3

#include <iomanip>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr std::uint64_t GOLDEN = 0x9e3779b97f4a7c15ULL;

// ===========================================================================
// Distributed (positional) input generation
// ===========================================================================
//
// Uniform: the i-th record of generate_relation() is splitmix64_mix(seed +
// (i+1)*GOLDEN) % max_key. Skewed: each record consumes two draws r1,r2 with
// r1 = mix(seed + (2i+1)*GOLDEN), r2 = mix(seed + (2i+2)*GOLDEN); if r1 is below
// the hot threshold the key comes from the shared hot pool, else it is uniform.
// (Verified against common.hpp's stateful generators.)
//
static inline std::uint64_t gen_uniform_key(std::uint64_t i, std::uint64_t seed,
                                             std::uint64_t max_key) {
    const std::uint64_t r = splitmix64_mix(seed + (i + 1ULL) * GOLDEN);
    return (max_key == 0) ? 0ULL : (r % max_key);
}

static std::vector<std::uint64_t> gen_slice(std::uint64_t lo, std::uint64_t hi,
                                            std::uint64_t seed, std::uint64_t max_key,
                                            std::uint32_t P, const SkewConfig& cfg,
                                            const std::vector<std::uint64_t>& hot_pool,
                                            std::uint64_t hot_threshold) {
    std::vector<std::uint64_t> out(hi - lo);
    if (cfg.skew_frac <= 0.0) {
        for (std::uint64_t i = lo; i < hi; ++i) out[i - lo] = gen_uniform_key(i, seed, max_key);
    } else {
        const std::uint64_t pool_size = hot_pool.size();
        for (std::uint64_t i = lo; i < hi; ++i) {
            const std::uint64_t r1 = splitmix64_mix(seed + (2ULL * i + 1ULL) * GOLDEN);
            const std::uint64_t r2 = splitmix64_mix(seed + (2ULL * i + 2ULL) * GOLDEN);
            out[i - lo] = (r1 < hot_threshold) ? hot_pool[r2 % pool_size]
                                               : ((max_key == 0) ? 0ULL : (r2 % max_key));
        }
    }
    (void)P;
    return out;
}

// ===========================================================================
// Local partitioning into a PartitionedRelation (Module 2 histogram/scatter)
// ===========================================================================
static PartitionedRelation partition_relation(const std::vector<std::uint64_t>& keys,
                                              std::uint32_t P) {
    PartitionedRelation pr;
    pr.begin.assign(P, 0);
    pr.end.assign(P, 0);

    std::vector<std::size_t> hist(P, 0);
    for (std::uint64_t k : keys) ++hist[compute_partition_id(k, P)];

    std::size_t running = 0;
    for (std::uint32_t p = 0; p < P; ++p) {
        pr.begin[p] = running;
        running += hist[p];
        pr.end[p] = running;
    }

    pr.data.resize(keys.size());
    std::vector<std::size_t> next(pr.begin);
    for (std::uint64_t k : keys) pr.data[next[compute_partition_id(k, P)]++].key = k;

    return pr;
}

// ===========================================================================
// Join over already-partitioned relations (reuses common.hpp join_one_partition)
// ===========================================================================
static JoinResult join_partitions(const PartitionedRelation& Rpart,
                                  const PartitionedRelation& Spart,
                                  std::uint32_t P) {
    std::uint64_t jc = 0, c1 = 0, c2 = 0;
#ifdef _OPENMP
    // schedule(dynamic,1) handles skewed partition loads, matching the Module 3
    // finding that per-iteration dispatch is what tolerates hot partitions.
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : jc, c1, c2)
#endif
    for (std::uint32_t pid = 0; pid < P; ++pid) {
        const JoinResult r = join_one_partition(Rpart, Spart, pid);
        jc += r.join_count;
        c1 += r.checksum1;
        c2 += r.checksum2;
    }
    return JoinResult{jc, c1, c2};
}

// Local join from raw key vectors: partition both, then join.
static JoinResult local_join(const std::vector<std::uint64_t>& Rkeys,
                             const std::vector<std::uint64_t>& Skeys,
                             std::uint32_t P) {
    const PartitionedRelation Rpart = partition_relation(Rkeys, P);
    const PartitionedRelation Spart = partition_relation(Skeys, P);
    return join_partitions(Rpart, Spart, P);
}

// ===========================================================================
// Redistribution by owner rank. Two strategies share the same preparation
// step (plan_exchange) and differ only in the all-to-all call:
//   - redistribute()         : blocking MPI_Alltoallv (one relation)
//   - redistribute_pair_nb() : non-blocking MPI_Ialltoallv for R and S posted
//                              together, then a single MPI_Waitall, so the two
//                              exchanges overlap each other.
// ===========================================================================
static inline int partition_owner(std::uint32_t pid, int nprocs) {
    return static_cast<int>(pid % static_cast<std::uint32_t>(nprocs));
}

// Everything needed to launch an all-to-all exchange. The buffers and the
// count/displacement arrays must stay alive until the exchange completes, so
// for the non-blocking path we keep one of these per relation until Waitall.
struct ExchangePlan {
    std::vector<std::uint64_t> sendbuf;
    std::vector<std::uint64_t> recvbuf;
    std::vector<int> sendcnt, sdispls, recvcnt, rdispls;
};

// Bucket the local keys by destination rank and exchange the per-destination
// counts so the receive buffer can be sized. This is the part common to both
// the blocking and the non-blocking strategies (it does the small, blocking
// MPI_Alltoall of counts; only the bulk data exchange is varied afterwards).
static ExchangePlan plan_exchange(const std::vector<std::uint64_t>& in,
                                  std::uint32_t P, int nprocs, MPI_Comm comm) {
    ExchangePlan e;
    e.sendcnt.assign(nprocs, 0);
    for (std::uint64_t k : in) ++e.sendcnt[partition_owner(compute_partition_id(k, P), nprocs)];

    e.sdispls.assign(nprocs, 0);
    long long running = 0;
    for (int d = 0; d < nprocs; ++d) {
        e.sdispls[d] = static_cast<int>(running);
        running += e.sendcnt[d];
        if (running > static_cast<long long>(std::numeric_limits<int>::max())) {
            std::cerr << "ERROR: send buffer exceeds INT_MAX elements; reduce the "
                         "per-rank size or use MPI-4 large-count APIs.\n";
            MPI_Abort(comm, 2);
        }
    }

    e.sendbuf.resize(in.size());
    std::vector<int> next(e.sdispls);
    for (std::uint64_t k : in) e.sendbuf[next[partition_owner(compute_partition_id(k, P), nprocs)]++] = k;

    e.recvcnt.assign(nprocs, 0);
    MPI_Alltoall(e.sendcnt.data(), 1, MPI_INT, e.recvcnt.data(), 1, MPI_INT, comm);

    e.rdispls.assign(nprocs, 0);
    long long total_recv = 0;
    for (int d = 0; d < nprocs; ++d) {
        e.rdispls[d] = static_cast<int>(total_recv);
        total_recv += e.recvcnt[d];
    }
    e.recvbuf.resize(static_cast<std::size_t>(total_recv));
    return e;
}

// Blocking redistribution of one relation.
static std::vector<std::uint64_t> redistribute(const std::vector<std::uint64_t>& in,
                                               std::uint32_t P, int nprocs, MPI_Comm comm) {
    ExchangePlan e = plan_exchange(in, P, nprocs, comm);
    MPI_Alltoallv(e.sendbuf.data(), e.sendcnt.data(), e.sdispls.data(), MPI_UINT64_T,
                  e.recvbuf.data(), e.recvcnt.data(), e.rdispls.data(), MPI_UINT64_T, comm);
    return std::move(e.recvbuf);
}

// Non-blocking redistribution of R and S together: both Ialltoallv are posted
// before either is waited on, so the two transfers overlap. The send buffer
// preparation of S is also computed while R's plan is ready, and the plans
// (buffers + count arrays) are kept alive until MPI_Waitall.
static void redistribute_pair_nb(const std::vector<std::uint64_t>& Rin,
                                 const std::vector<std::uint64_t>& Sin,
                                 std::uint32_t P, int nprocs, MPI_Comm comm,
                                 std::vector<std::uint64_t>& Rout,
                                 std::vector<std::uint64_t>& Sout) {
    ExchangePlan rp = plan_exchange(Rin, P, nprocs, comm);
    ExchangePlan sp = plan_exchange(Sin, P, nprocs, comm);

    MPI_Request req[2];
    MPI_Ialltoallv(rp.sendbuf.data(), rp.sendcnt.data(), rp.sdispls.data(), MPI_UINT64_T,
                   rp.recvbuf.data(), rp.recvcnt.data(), rp.rdispls.data(), MPI_UINT64_T, comm, &req[0]);
    MPI_Ialltoallv(sp.sendbuf.data(), sp.sendcnt.data(), sp.sdispls.data(), MPI_UINT64_T,
                   sp.recvbuf.data(), sp.recvcnt.data(), sp.rdispls.data(), MPI_UINT64_T, comm, &req[1]);
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

    Rout = std::move(rp.recvbuf);
    Sout = std::move(sp.recvbuf);
}

// Non-blocking LEVEL 2: overlap communication with independent computation.
// Post Ialltoallv for R and S, wait only on R, then partition R and build the
// per-partition hash tables (work that depends only on R) WHILE S is still in
// flight; periodic MPI_Test calls nudge the runtime to progress the S transfer.
// Finally wait on S, partition it, and probe. Redistribution and join are fused,
// so they cannot be timed separately (we report the combined t_rj).
static JoinResult redistribute_and_join_overlap(const std::vector<std::uint64_t>& Rin,
                                                const std::vector<std::uint64_t>& Sin,
                                                std::uint32_t P, int nprocs, MPI_Comm comm) {
    ExchangePlan rp = plan_exchange(Rin, P, nprocs, comm);
    ExchangePlan sp = plan_exchange(Sin, P, nprocs, comm);

    MPI_Request reqR, reqS;
    MPI_Ialltoallv(rp.sendbuf.data(), rp.sendcnt.data(), rp.sdispls.data(), MPI_UINT64_T,
                   rp.recvbuf.data(), rp.recvcnt.data(), rp.rdispls.data(), MPI_UINT64_T, comm, &reqR);
    MPI_Ialltoallv(sp.sendbuf.data(), sp.sendcnt.data(), sp.sdispls.data(), MPI_UINT64_T,
                   sp.recvbuf.data(), sp.recvcnt.data(), sp.rdispls.data(), MPI_UINT64_T, comm, &reqS);

    // R must be present before we can build on it; S may still be transferring.
    MPI_Wait(&reqR, MPI_STATUS_IGNORE);

    // ---- overlap region: partition R and build one FlatCountMap per partition,
    //      while S keeps transferring (MPI_Test nudges asynchronous progress) ----
    const PartitionedRelation Rpart = partition_relation(rp.recvbuf, P);
    std::vector<FlatCountMap> maps;
    maps.reserve(P);
    int flag = 0;
    for (std::uint32_t pid = 0; pid < P; ++pid) {
        const std::size_t rb = Rpart.begin[pid], re = Rpart.end[pid];
        maps.emplace_back(re > rb ? (re - rb) : std::size_t{1});
        for (std::size_t i = rb; i < re; ++i) maps[pid].insert(Rpart.data[i].key);
        if ((pid & 31u) == 0u) MPI_Test(&reqS, &flag, MPI_STATUS_IGNORE);
    }

    MPI_Wait(&reqS, MPI_STATUS_IGNORE);
    const PartitionedRelation Spart = partition_relation(sp.recvbuf, P);

    std::uint64_t jc = 0, c1 = 0, c2 = 0;
    for (std::uint32_t pid = 0; pid < P; ++pid) {
        const std::size_t sb = Spart.begin[pid], se = Spart.end[pid];
        if (sb == se) continue;
        const FlatCountMap& m = maps[pid];
        for (std::size_t i = sb; i < se; ++i) {
            const std::uint64_t key = Spart.data[i].key;
            const std::uint32_t mult = m.find(key);
            if (mult) {
                jc += mult;
                c1 += splitmix64(key) * mult;
                c2 += splitmix64(key ^ GOLDEN) * mult;
            }
        }
    }
    return JoinResult{jc, c1, c2};
}

// ===========================================================================
// Embedded sequential reference (strong correctness check, small N)
// ===========================================================================
static JoinResult sequential_reference(std::uint64_t NR, std::uint64_t NS,
                                       std::uint64_t seed, std::uint64_t max_key,
                                       std::uint32_t P, const SkewConfig& cfg) {
    const std::uint64_t seedS = seed ^ 0xdeadebdecdeedef1ULL;
    const auto R = generate_skewed_relation(NR, seed,  max_key, P, cfg);
    const auto S = generate_skewed_relation(NS, seedS, max_key, P, cfg);
    std::vector<std::uint64_t> Rk(NR), Sk(NS);
    for (std::uint64_t i = 0; i < NR; ++i) Rk[i] = R[i].key;
    for (std::uint64_t i = 0; i < NS; ++i) Sk[i] = S[i].key;
    return local_join(Rk, Sk, P);
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::uint64_t nr = 0, ns = 0, seed = 0, max_key = 0, p = 0, hot = 0, check = 0, nb = 0;
    double skew = 0.0;
    bool ok = read_arg_u64(argc, argv, "-nr", nr) && read_arg_u64(argc, argv, "-ns", ns) &&
              read_arg_u64(argc, argv, "-seed", seed) && read_arg_u64(argc, argv, "-max-key", max_key) &&
              read_arg_u64(argc, argv, "-p", p);
    read_arg_dbl(argc, argv, "-skew", skew);
    read_arg_u64(argc, argv, "-hot", hot);
    read_arg_u64(argc, argv, "-check", check);
    read_arg_u64(argc, argv, "-nb", nb);   // 0 blocking, 1 non-blocking (overlap R/S), 2 non-blocking (overlap S transfer with R build)

    if (!ok || p == 0 || p > std::numeric_limits<std::uint32_t>::max() ||
        !is_power_of_two(static_cast<std::uint32_t>(p))) {
        if (rank == 0) {
            usage_common(argv[0], /*show_threads=*/false);
            std::cerr << "  -nb 0|1|2     Redistribution: 0 blocking, 1 non-blocking (overlap R/S),\n"
                         "                2 non-blocking (overlap S transfer with R build; uses more memory)\n";
            std::cerr << "  -check 1      Run the embedded sequential check (small N)\n";
        }
        MPI_Finalize();
        return 1;
    }
    const std::uint32_t P = static_cast<std::uint32_t>(p);

    SkewConfig cfg;
    cfg.skew_frac = skew;
    if (hot > 0) cfg.num_hot = static_cast<std::uint32_t>(hot);

    // Hot pool and threshold are deterministic, so every rank builds them
    // identically (no communication needed).
    const std::vector<std::uint64_t> hot_pool =
        (skew > 0.0) ? build_hot_pool(P, max_key, cfg) : std::vector<std::uint64_t>{};
    const std::uint64_t hot_threshold =
        static_cast<std::uint64_t>(cfg.skew_frac * static_cast<double>(~0ULL));

    int threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    { if (omp_get_thread_num() == 0) threads = omp_get_num_threads(); }
#endif

    auto split = [&](std::uint64_t N, std::uint64_t& lo, std::uint64_t& hi) {
        lo = (N * static_cast<std::uint64_t>(rank)) / static_cast<std::uint64_t>(nprocs);
        hi = (N * static_cast<std::uint64_t>(rank + 1)) / static_cast<std::uint64_t>(nprocs);
    };
    std::uint64_t rlo, rhi, slo, shi;
    split(nr, rlo, rhi);
    split(ns, slo, shi);
    const std::uint64_t seedS = seed ^ 0xdeadebdecdeedef1ULL;

    // ---- Phase 0: distributed generation ----------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    std::vector<std::uint64_t> Rloc = gen_slice(rlo, rhi, seed,  max_key, P, cfg, hot_pool, hot_threshold);
    std::vector<std::uint64_t> Sloc = gen_slice(slo, shi, seedS, max_key, P, cfg, hot_pool, hot_threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // ---- Phase 1+2: redistribution and local join --------------------------
    // nb=0 blocking, nb=1 non-blocking (overlap R/S transfers), nb=2 non-blocking
    // with computation overlap (build of R hidden under the S transfer). For
    // nb=2 redistribution and join are fused and cannot be timed separately.
    JoinResult local;
    double t2;
    if (nb == 2) {
        local = redistribute_and_join_overlap(Rloc, Sloc, P, nprocs, MPI_COMM_WORLD);
        std::vector<std::uint64_t>().swap(Rloc);
        std::vector<std::uint64_t>().swap(Sloc);
        t2 = MPI_Wtime();
    } else {
        std::vector<std::uint64_t> Rrecv, Srecv;
        if (nb == 1) {
            redistribute_pair_nb(Rloc, Sloc, P, nprocs, MPI_COMM_WORLD, Rrecv, Srecv);
        } else {
            Rrecv = redistribute(Rloc, P, nprocs, MPI_COMM_WORLD);
            Srecv = redistribute(Sloc, P, nprocs, MPI_COMM_WORLD);
        }
        std::vector<std::uint64_t>().swap(Rloc);
        std::vector<std::uint64_t>().swap(Sloc);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        local = local_join(Rrecv, Srecv, P);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    // ---- Phase 3: global reduction ----------------------------------------
    std::uint64_t in3[3]  = {local.join_count, local.checksum1, local.checksum2};
    std::uint64_t out3[3] = {0, 0, 0};
    MPI_Reduce(in3, out3, 3, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    double locp[6] = {t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t1, t3 - t1};
    double maxp[6] = {0, 0, 0, 0, 0, 0};
    MPI_Reduce(locp, maxp, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "NR=" << nr << " NS=" << ns << " P=" << P << " seed=" << seed
                  << " max-key=" << max_key << " skew=" << skew << " hot=" << cfg.num_hot
                  << " nprocs=" << nprocs << " threads=" << threads
                  << " nb=" << nb << "\n";
        std::cout << "join_count=" << out3[0] << "\n";
        std::cout << "checksum1="  << out3[1] << "\n";
        std::cout << "checksum2="  << out3[2] << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "t_gen_sec="       << maxp[0] << "\n";
        std::cout << "t_redist_sec="    << maxp[1] << (nb == 2 ? "  (fused with join for nb=2)\n" : "\n");
        std::cout << "t_localjoin_sec=" << maxp[2] << "\n";
        std::cout << "t_reduce_sec="    << maxp[3] << "\n";
        std::cout << "t_rj_sec="        << maxp[5] << "  (redist+localjoin; the cross-mode metric)\n";
        std::cout << "t_total_sec="     << maxp[4] << "  (redist+localjoin+reduce, gen excluded)\n";
        std::cout << "time_sec="         << maxp[4] << "\n"; // alias for analysis tooling

        if (check && nr <= 200000 && ns <= 200000) {
            const JoinResult ref = sequential_reference(nr, ns, seed, max_key, P, cfg);
            const bool good = ref.join_count == out3[0] && ref.checksum1 == out3[1] && ref.checksum2 == out3[2];
            std::cout << "check=" << (good ? "PASS" : "FAIL")
                      << " (ref join_count=" << ref.join_count
                      << " checksum1=" << ref.checksum1 << " checksum2=" << ref.checksum2 << ")\n";
            if (!good) MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    MPI_Finalize();
    return 0;
}