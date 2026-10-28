// hashjoin_omp_task.cpp
//
// OpenMP task-based parallel implementation of the partitioned hash join.
//
// Parallelisation choices, phase by phase:
//
//   - Histogram and Scatter:
//       The work per record is constant, so the total work splits cleanly
//       into T equal chunks. We still use tasks (one task per chunk, all
//       spawned by an `omp single` block) for stylistic consistency with
//       the rest of the implementation, but the granularity is coarse on
//       purpose: with T tasks of N/T records each, the task creation cost
//       is fully amortised. Spawning one task per record would be a
//       textbook anti-pattern.
//
//   - Local join:
//       This is where tasks shine. We spawn ONE task per partition
//       (P = 256 tasks, T = 8..16 worker threads). The OpenMP runtime
//       schedules tasks dynamically, so when the input is skewed and a
//       few partitions carry most of the work, idle threads pick up the
//       remaining small partitions while the heavy partitions run in
//       parallel on whichever threads happen to be free. This work-
//       stealing behaviour is exactly what the loop-level version with
//       schedule(static) cannot achieve.
//
// Each task accumulates into a thread-local PaddedJoinResult slot
// indexed by omp_get_thread_num() (alignas(64) prevents false sharing).
// We do not use OpenMP `task_reduction` (OpenMP 5.0) to keep the code
// portable across compilers.
//
// Run example:
//   ./hashjoin_omp_task -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8
//   ./hashjoin_omp_task -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8 -skew 0.8
//

#include "common.hpp"

#include <chrono>
#include <iomanip>
#include <omp.h>

// ------------------------------------------------------------
// Histogram result
// ------------------------------------------------------------
struct HistogramResult {
    std::vector<std::size_t>              global;
    std::vector<std::vector<std::size_t>> local_hists;
};

// ------------------------------------------------------------
// Parallel histogram (OpenMP tasks, coarse-grained)
// ------------------------------------------------------------
//
// We open a parallel region with T threads, then have a single thread
// spawn T tasks (one per chunk). The remaining threads pick them up.
// `firstprivate(t)` captures the chunk index by value.
//
static HistogramResult compute_histogram(const std::vector<Record>& rel,
                                         std::uint32_t p, int T) {
    const std::size_t N = rel.size();
    std::vector<std::vector<std::size_t>> local_hists(T, std::vector<std::size_t>(p, 0));

    #pragma omp parallel num_threads(T)
    {
        #pragma omp single
        {
            for (int t = 0; t < T; t++) {
                #pragma omp task firstprivate(t) shared(local_hists, rel)
                {
                    const std::size_t chunk = N / T;
                    const std::size_t b = t * chunk;
                    const std::size_t e = (t == T - 1) ? N : b + chunk;

                    auto* row = local_hists[t].data();
                    for (std::size_t i = b; i < e; i++) {
                        const std::uint32_t pid = compute_partition_id(rel[i].key, p);
                        ++row[pid];
                    }
                }
            }
        }
        // implicit barrier at end of parallel region waits for all tasks
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
// Parallel scatter (OpenMP tasks, coarse-grained)
// ------------------------------------------------------------
static std::vector<Record> scatter_partitioned(
        const std::vector<Record>& rel,
        std::uint32_t p, int T,
        const std::vector<std::size_t>& global_begin,
        const std::vector<std::vector<std::size_t>>& local_hists) {

    const std::size_t N = rel.size();
    std::vector<Record> out(N);

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
        #pragma omp single
        {
            for (int t = 0; t < T; t++) {
                #pragma omp task firstprivate(t) shared(out, rel, thread_cursor)
                {
                    const std::size_t chunk = N / T;
                    const std::size_t b = t * chunk;
                    const std::size_t e = (t == T - 1) ? N : b + chunk;

                    auto* cursor = thread_cursor[t].data();
                    for (std::size_t i = b; i < e; i++) {
                        const std::uint32_t pid = compute_partition_id(rel[i].key, p);
                        out[cursor[pid]++] = rel[i];
                    }
                }
            }
        }
    }

    return out;
}

// ------------------------------------------------------------
// Full parallel partitioning pipeline (OpenMP tasks)
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
// False-sharing-padded per-thread accumulator
// ------------------------------------------------------------
struct alignas(64) PaddedJoinResult {
    std::uint64_t join_count = 0;
    std::uint64_t checksum1  = 0;
    std::uint64_t checksum2  = 0;
};

// ------------------------------------------------------------
// Full parallel partitioned hash join (OpenMP tasks)
// ------------------------------------------------------------
//
// One task per partition. The OpenMP task scheduler distributes the P
// tasks dynamically across the T threads: as soon as a thread finishes
// a task it picks the next available one. With skewed input, this means
// idle threads keep working on small partitions while the heavy ones
// are still running, which is exactly what the loop-static version
// cannot do.
//
// Tasks accumulate into per-thread slots indexed by omp_get_thread_num(),
// avoiding any contention on the reduction.
//
static JoinResult partitioned_hash_join_omp_task(const std::vector<Record>& R,
                                                 const std::vector<Record>& S,
                                                 std::uint32_t p, int T) {
    using Clock = std::chrono::steady_clock;

    auto t0 = Clock::now();
    const PartitionedRelation Rpart = partition_relation(R, p, T);
    const PartitionedRelation Spart = partition_relation(S, p, T);
    auto t1 = Clock::now();

    std::vector<PaddedJoinResult> thread_results(T);

    #pragma omp parallel num_threads(T)
    {
        #pragma omp single
        {
            for (std::uint32_t pid = 0; pid < p; pid++) {
                #pragma omp task firstprivate(pid) shared(thread_results, Rpart, Spart)
                {
                    const int tid = omp_get_thread_num();
                    const JoinResult local = join_one_partition(Rpart, Spart, pid);
                    thread_results[tid].join_count += local.join_count;
                    thread_results[tid].checksum1  += local.checksum1;
                    thread_results[tid].checksum2  += local.checksum2;
                }
            }
        }
        // implicit barrier waits for all tasks to complete
    }
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
    const JoinResult result = partitioned_hash_join_omp_task(R, S, P, T);
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
