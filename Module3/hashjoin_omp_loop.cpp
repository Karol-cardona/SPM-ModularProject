// hashjoin_omp_loop.cpp
//
// OpenMP loop-level parallel implementation of the partitioned hash join.
//
// All parallel phases use OpenMP's `parallel` construct:
//
//   - Histogram and Scatter:
//       Use a `#pragma omp parallel` region with manual chunking inside
//       (each thread processes records [tid * N/T, (tid+1) * N/T)).
//       This is functionally equivalent to `#pragma omp parallel for
//       schedule(static)` but is written explicitly so the per-thread
//       histogram rows match exactly the C++ thread version's contract,
//       which is required for the conflict-free scatter scheme to work.
//
//   - Local join:
//       Uses `#pragma omp parallel for schedule(runtime)` over the P
//       partitions. The OpenMP runtime reads the policy from the
//       OMP_SCHEDULE environment variable, so a SINGLE binary covers
//       all the policies tested in the report:
//
//           OMP_SCHEDULE=static      ./hashjoin_omp_loop ...
//           OMP_SCHEDULE=dynamic,1   ./hashjoin_omp_loop ...
//           OMP_SCHEDULE=guided      ./hashjoin_omp_loop ...
//
//       This is the only loop with non-uniform per-iteration cost
//       (partitions of very different sizes under skew), so it is the
//       only place where the choice of scheduling policy matters.
//
// Note: when OMP_SCHEDULE is unset, GCC's default for runtime is
// `static`, matching the original Module 3 design. The benchmark
// scripts set OMP_SCHEDULE explicitly so the policy is deterministic.
//
// Run examples:
//   OMP_SCHEDULE=static    ./hashjoin_omp_loop -nr 50000000 -ns 80000000  -seed 13 -max-key 1000000 -p 256 -t 16
//   OMP_SCHEDULE=dynamic,1 ./hashjoin_omp_loop -nr 50000000 -ns 80000000  -seed 13 -max-key 1000000 -p 256 -t 16 -skew 0.9
//

#include "common.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

// ------------------------------------------------------------
// Histogram result (also exposes per-thread rows used by scatter)
// ------------------------------------------------------------
struct HistogramResult {
    std::vector<std::size_t>              global;
    std::vector<std::vector<std::size_t>> local_hists;
};

// ------------------------------------------------------------
// Parallel histogram (OpenMP, manual chunking)
// ------------------------------------------------------------
//
// `#pragma omp parallel` (NOT `parallel for`) opens a parallel region
// in which each thread does its own contiguous slice of the input.
// Manual chunking is used (instead of `parallel for schedule(static)`)
// to make the per-thread row layout in `local_hists` identical to the
// C++ thread version, which is required for the conflict-free scatter
// scheme that follows.
//
// The per-record work is uniform, so static distribution is optimal.
//
static HistogramResult compute_histogram(const std::vector<Record>& rel,
                                         std::uint32_t p, int T) {
    const std::size_t N = rel.size();
    std::vector<std::vector<std::size_t>> local_hists(T, std::vector<std::size_t>(p, 0));

    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        // Manual chunking matches the C++ thread version exactly so that
        // both implementations produce the SAME local_hists rows, which is
        // required for the scatter phase to be conflict-free.
        const std::size_t chunk = N / T;
        const std::size_t b = tid * chunk;
        const std::size_t e = (tid == T - 1) ? N : b + chunk;

        auto* row = local_hists[tid].data();
        for (std::size_t i = b; i < e; i++) {
            const std::uint32_t pid = compute_partition_id(rel[i].key, p);
            ++row[pid];
        }
    }

    std::vector<std::size_t> global(p, 0);
    for (int t = 0; t < T; t++)
        for (std::uint32_t pid = 0; pid < p; pid++)
            global[pid] += local_hists[t][pid];

    return {global, std::move(local_hists)};
}

// ------------------------------------------------------------
// Exclusive prefix sum (sequential, P entries only)
// ------------------------------------------------------------
static std::vector<std::size_t> exclusive_prefix_sum(const std::vector<std::size_t>& hist) {
    std::vector<std::size_t> begin(hist.size(), 0);
    std::size_t running = 0;
    for (std::size_t p = 0; p < hist.size(); ++p) {
        begin[p] = running;
        running += hist[p];
    }
    return begin;
}

// ------------------------------------------------------------
// Parallel scatter (OpenMP loop)
// ------------------------------------------------------------
static std::vector<Record> scatter_partitioned(
        const std::vector<Record>& rel,
        std::uint32_t p, int T,
        const std::vector<std::size_t>& global_begin,
        const std::vector<std::vector<std::size_t>>& local_hists) {

    const std::size_t N = rel.size();
    std::vector<Record> out(N);

    // Pre-compute the per-thread, per-partition cursors. Each thread
    // gets a write range that does not overlap with any other thread.
    std::vector<std::vector<std::size_t>> thread_cursor(T, std::vector<std::size_t>(p, 0));
    for (std::uint32_t pid = 0; pid < p; pid++) {
        std::size_t running = global_begin[pid];
        for (int t = 0; t < T; t++) {
            thread_cursor[t][pid] = running;
            running += local_hists[t][pid];
        }
    }

    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        const std::size_t chunk = N / T;
        const std::size_t b = tid * chunk;
        const std::size_t e = (tid == T - 1) ? N : b + chunk;

        auto* cursor = thread_cursor[tid].data();
        for (std::size_t i = b; i < e; i++) {
            const std::uint32_t pid = compute_partition_id(rel[i].key, p);
            out[cursor[pid]++] = rel[i];
        }
    }

    return out;
}

// ------------------------------------------------------------
// Full parallel partitioning pipeline (OpenMP loop)
// ------------------------------------------------------------
static PartitionedRelation partition_relation(const std::vector<Record>& rel,
                                              std::uint32_t p, int T) {
    const HistogramResult hres = compute_histogram(rel, p, T);
    const auto begin = exclusive_prefix_sum(hres.global);
    auto data = scatter_partitioned(rel, p, T, begin, hres.local_hists);

    std::vector<std::size_t> end(p, 0);
    for (std::uint32_t pid = 0; pid < p; ++pid)
        end[pid] = begin[pid] + hres.global[pid];

    return PartitionedRelation{
        .data = std::move(data),
        .begin = begin,
        .end = end};
}

// ------------------------------------------------------------
// Full parallel partitioned hash join (OpenMP loop, runtime schedule)
// ------------------------------------------------------------
//
// The local-join loop uses `schedule(runtime)`, which lets the OMP_SCHEDULE
// environment variable choose the policy: static (default if unset), dynamic,
// or guided. This is what makes a single binary cover all three policies in
// the schedule-comparison experiment of the report.
//
// `reduction(+:...)` accumulates the three uint64 fields cleanly across
// threads — equivalent in semantics to the manual PaddedJoinResult array
// used by the C++ thread version, but expressed in idiomatic OpenMP.
//
static JoinResult partitioned_hash_join_omp_loop(const std::vector<Record>& R,
                                                       const std::vector<Record>& S,
                                                       std::uint32_t p, int T) {
    using Clock = std::chrono::steady_clock;

    auto t0 = Clock::now();
    const PartitionedRelation Rpart = partition_relation(R, p, T);
    const PartitionedRelation Spart = partition_relation(S, p, T);
    auto t1 = Clock::now();

    std::uint64_t total_join = 0, total_c1 = 0, total_c2 = 0;

    // schedule(runtime) reads the policy from the OMP_SCHEDULE env var.
    // Set OMP_SCHEDULE=static, dynamic,1, or guided to switch behaviour.
    #pragma omp parallel for num_threads(T) schedule(runtime) \
        reduction(+:total_join, total_c1, total_c2)
    for (std::uint32_t pid = 0; pid < p; pid++) {
        const JoinResult local = join_one_partition(Rpart, Spart, pid);
        total_join += local.join_count;
        total_c1   += local.checksum1;
        total_c2   += local.checksum2;
    }
    auto t2 = Clock::now();

    std::cerr << "[phases] partitioning="
              << std::chrono::duration<double>(t1 - t0).count()
              << "s  join="
              << std::chrono::duration<double>(t2 - t1).count()
              << "s\n";

    return JoinResult{total_join, total_c1, total_c2};
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv) {
    std::uint64_t nr = 0, ns = 0, seed = 0, max_key = 0, p = 0, t = 0, hot = 0;
    double skew = 0.0;

    if (!read_arg_u64(argc, argv, "-nr", nr) ||
        !read_arg_u64(argc, argv, "-ns", ns) ||
        !read_arg_u64(argc, argv, "-seed", seed) ||
        !read_arg_u64(argc, argv, "-max-key", max_key) ||
        !read_arg_u64(argc, argv, "-p", p)) {
        usage_common(argv[0], /*show_threads=*/true);
        return 1;
    }
    read_arg_u64(argc, argv, "-t", t);
    read_arg_dbl(argc, argv, "-skew", skew);
    read_arg_u64(argc, argv, "-hot", hot);

    const int T = (t == 0) ? omp_get_max_threads() : static_cast<int>(t);
    if (T < 1) {
        std::cerr << "Error: invalid thread count.\n";
        return 1;
    }

    if (p > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "Error: P too large.\n";
        return 1;
    }
    const std::uint32_t P = static_cast<std::uint32_t>(p);
    if (!is_power_of_two(P)) {
        std::cerr << "Error: P must be a power of two.\n";
        return 1;
    }

    SkewConfig skew_cfg;
    skew_cfg.skew_frac = skew;
    if (hot > 0) skew_cfg.num_hot = static_cast<std::uint32_t>(hot);

    const std::size_t NR = static_cast<std::size_t>(nr);
    const std::size_t NS = static_cast<std::size_t>(ns);

    const auto R = generate_skewed_relation(NR, seed,                                 max_key, P, skew_cfg);
    const auto S = generate_skewed_relation(NS, seed ^ 0xdeadebdecdeedef1ULL,         max_key, P, skew_cfg);

    const auto t0 = std::chrono::steady_clock::now();
    const JoinResult result = partitioned_hash_join_omp_loop(R, S, P, T);
    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "NR=" << NR << " NS=" << NS << " P=" << P
              << " T=" << T << " seed=" << seed
              << " max-key=" << max_key
              << " skew=" << skew
              << " hot=" << skew_cfg.num_hot;
    // Show which scheduling policy is active (read from env var)
    const char* sched_env = std::getenv("OMP_SCHEDULE");
    std::cout << " OMP_SCHEDULE=" << (sched_env ? sched_env : "(unset->static)") << "\n";

    std::cout << "join_count=" << result.join_count << "\n";
    std::cout << "checksum1="  << result.checksum1  << "\n";
    std::cout << "checksum2="  << result.checksum2  << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "time_sec="   << sec << "\n";

    if (NR <= 500 && NS <= 500) {
        const JoinResult naive = naive_join_verifier(R, S);
        std::cout << "naive_join_count=" << naive.join_count << "\n";
        std::cout << "naive_checksum1="  << naive.checksum1  << "\n";
        std::cout << "naive_checksum2="  << naive.checksum2  << "\n";
    }
    return 0;
}