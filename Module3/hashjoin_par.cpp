// hashjoin_par.cpp
//
// C++ std::thread implementation (the Module 2 "v3" version) extended
// with the `-skew F` flag, used as the C++-threads reference for the
// Module 3 OpenMP comparison.
//
// Algorithm structure (identical to Module 2 v3):
//   1. Parallel histogram   — each thread updates its private row,
//                             histograms are merged sequentially after.
//   2. Sequential prefix sum — only P (e.g. 256) entries; parallelising
//                             would cost more than it saves.
//   3. Parallel scatter      — per-thread pre-computed cursors give each
//                             thread a conflict-free output range, so
//                             no atomics or mutexes are needed.
//   4. Parallel local join   — partitions distributed across threads via
//                             stride-T round-robin; results accumulated
//                             into PaddedJoinResult slots (alignas(64)
//                             to prevent false sharing).
//
// Run example:
//   ./hashjoin_par -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8
//   ./hashjoin_par -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8 -skew 0.8
//

#include "common.hpp"

#include <chrono>
#include <iomanip>
#include <thread>

// ------------------------------------------------------------
// Histogram result (also exposes per-thread rows used by scatter)
// ------------------------------------------------------------
struct HistogramResult {
    std::vector<std::size_t>              global;
    std::vector<std::vector<std::size_t>> local_hists;
};

// ------------------------------------------------------------
// Parallel histogram (C++ threads)
// ------------------------------------------------------------
static HistogramResult compute_histogram(const std::vector<Record>& rel,
                                         std::uint32_t p, int T) {
    const std::size_t N = rel.size();
    std::vector<std::vector<std::size_t>> local_hists(T, std::vector<std::size_t>(p, 0));

    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&, t]() {
            const std::size_t chunk = N / T;
            const std::size_t b = t * chunk;
            const std::size_t e = (t == T - 1) ? N : b + chunk;
            for (std::size_t i = b; i < e; i++) {
                const std::uint32_t pid = compute_partition_id(rel[i].key, p);
                ++local_hists[t][pid];
            }
        });
    }
    for (auto& th : threads) th.join();

    std::vector<std::size_t> global(p, 0);
    for (int t = 0; t < T; t++)
        for (std::uint32_t pid = 0; pid < p; pid++)
            global[pid] += local_hists[t][pid];

    return {global, std::move(local_hists)};
}

// ------------------------------------------------------------
// Exclusive prefix sum (sequential)
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
// Parallel scatter
// ------------------------------------------------------------
static std::vector<Record> scatter_partitioned(
        const std::vector<Record>& rel,
        std::uint32_t p, int T,
        const std::vector<std::size_t>& global_begin,
        const std::vector<std::vector<std::size_t>>& local_hists) {

    const std::size_t N = rel.size();
    std::vector<Record> out(N);

    // Compute the per-thread, per-partition write cursor by stacking
    // each thread's local histogram on top of the global begin offsets.
    std::vector<std::vector<std::size_t>> thread_cursor(T, std::vector<std::size_t>(p, 0));
    for (std::uint32_t pid = 0; pid < p; pid++) {
        std::size_t running = global_begin[pid];
        for (int t = 0; t < T; t++) {
            thread_cursor[t][pid] = running;
            running += local_hists[t][pid];
        }
    }

    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&, t]() {
            const std::size_t chunk = N / T;
            const std::size_t b = t * chunk;
            const std::size_t e = (t == T - 1) ? N : b + chunk;
            for (std::size_t i = b; i < e; i++) {
                const std::uint32_t pid = compute_partition_id(rel[i].key, p);
                out[thread_cursor[t][pid]++] = rel[i];
            }
        });
    }
    for (auto& th : threads) th.join();

    return out;
}

// ------------------------------------------------------------
// Full parallel partitioning pipeline
// ------------------------------------------------------------
static PartitionedRelation partition_relation(const std::vector<Record>& rel,
                                              std::uint32_t p, int T) {
    const HistogramResult hres = compute_histogram(rel, p, T);
    const auto begin = exclusive_prefix_sum(hres.global);
    auto data = scatter_partitioned(rel, p, T, begin, hres.local_hists);

    std::vector<std::size_t> end(p, 0);
    for (std::uint32_t pid = 0; pid < p; ++pid)
        end[pid] = begin[pid] + hres.global[pid];

    return PartitionedRelation{.data = std::move(data), .begin = begin, .end = end};
}

// ------------------------------------------------------------
// False-sharing-padded per-thread accumulator
// ------------------------------------------------------------
struct alignas(64) PaddedJoinResult {
    std::uint64_t join_count = 0;
    std::uint64_t checksum1  = 0;
    std::uint64_t checksum2  = 0;
};

// ------------------------------------------------------------
// Full parallel partitioned hash join
// ------------------------------------------------------------
static JoinResult partitioned_hash_join_parallel(const std::vector<Record>& R,
                                                 const std::vector<Record>& S,
                                                 std::uint32_t p, int T) {
    using Clock = std::chrono::steady_clock;

    auto t0 = Clock::now();
    const PartitionedRelation Rpart = partition_relation(R, p, T);
    const PartitionedRelation Spart = partition_relation(S, p, T);
    auto t1 = Clock::now();

    // Stride-T round-robin assignment of partitions to threads.
    // With P >> T (e.g. P=256, T=16) each thread gets ~16 partitions,
    // spread evenly across the partition id space.
    std::vector<PaddedJoinResult> thread_results(T);

    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&, t]() {
            for (std::uint32_t pid = t; pid < p; pid += T) {
                const JoinResult local = join_one_partition(Rpart, Spart, pid);
                thread_results[t].join_count += local.join_count;
                thread_results[t].checksum1  += local.checksum1;
                thread_results[t].checksum2  += local.checksum2;
            }
        });
    }
    for (auto& th : threads) th.join();
    auto t2 = Clock::now();

    JoinResult total{};
    for (int t = 0; t < T; t++) {
        total.join_count += thread_results[t].join_count;
        total.checksum1  += thread_results[t].checksum1;
        total.checksum2  += thread_results[t].checksum2;
    }

    std::cerr << "[phases] partitioning="
              << std::chrono::duration<double>(t1 - t0).count()
              << "s  join="
              << std::chrono::duration<double>(t2 - t1).count()
              << "s\n";

    return total;
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

    const int T = (t == 0)
        ? static_cast<int>(std::thread::hardware_concurrency())
        : static_cast<int>(t);
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
    const JoinResult result = partitioned_hash_join_parallel(R, S, P, T);
    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "NR=" << NR << " NS=" << NS << " P=" << P
              << " T=" << T << " seed=" << seed
              << " max-key=" << max_key
              << " skew=" << skew
              << " hot=" << skew_cfg.num_hot << "\n";

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
