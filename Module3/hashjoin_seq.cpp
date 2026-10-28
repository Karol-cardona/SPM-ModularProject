// hashjoin_seq.cpp
//
// Sequential baseline for Module 3.
//
// This is the same improved sequential implementation used as baseline in
// Module 2 (multiply_shift partition mapping + FlatCountMap build), with
// the addition of an optional `-skew F` flag for skewed input generation.
// When `-skew 0` (default) the behaviour is identical to the Module 2
// sequential baseline.
//
// All algorithm-specific code is sequential. Shared utilities (Record,
// hash functions, FlatCountMap, generation, verifier) live in common.hpp
// and are reused by all four implementations of Module 3.
//
// Run example:
//   ./hashjoin_seq -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256
//   ./hashjoin_seq -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -skew 0.8
//

#include "common.hpp"

#include <chrono>
#include <iomanip>

// ------------------------------------------------------------
// Sequential histogram
// ------------------------------------------------------------
static std::vector<std::size_t> compute_histogram(const std::vector<Record>& rel,
                                                  std::uint32_t p) {
    std::vector<std::size_t> hist(p, 0);
    for (const auto& rec : rel) {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        ++hist[pid];
    }
    return hist;
}

// ------------------------------------------------------------
// Exclusive prefix sum
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
// Sequential scatter
// ------------------------------------------------------------
static std::vector<Record> scatter_partitioned(const std::vector<Record>& rel,
                                               std::uint32_t p,
                                               const std::vector<std::size_t>& begin) {
    std::vector<Record> out(rel.size());
    std::vector<std::size_t> next = begin;
    for (const auto& rec : rel) {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        out[next[pid]++] = rec;
    }
    return out;
}

// ------------------------------------------------------------
// Full partitioning pipeline (sequential)
// ------------------------------------------------------------
static PartitionedRelation partition_relation(const std::vector<Record>& rel, std::uint32_t p) {
    const auto hist  = compute_histogram(rel, p);
    const auto begin = exclusive_prefix_sum(hist);
    auto data = scatter_partitioned(rel, p, begin);

    std::vector<std::size_t> end(p, 0);
    for (std::uint32_t pid = 0; pid < p; ++pid)
        end[pid] = begin[pid] + hist[pid];

    return PartitionedRelation{.data = std::move(data), .begin = begin, .end = end};
}

// ------------------------------------------------------------
// Full sequential partitioned hash join
// ------------------------------------------------------------
//
// Phase 1+2: partition R and S
// Phase 3:   join one partition at a time
// Phase 4:   accumulate results
//
// The phase breakdown is printed to stderr for comparison with the
// parallel implementations. Note that join time also includes the
// per-partition build of FlatCountMap.
//
static JoinResult partitioned_hash_join_sequential(const std::vector<Record>& R,
                                                   const std::vector<Record>& S,
                                                   std::uint32_t p) {
    using Clock = std::chrono::steady_clock;

    auto t0 = Clock::now();
    const PartitionedRelation Rpart = partition_relation(R, p);
    const PartitionedRelation Spart = partition_relation(S, p);
    auto t1 = Clock::now();

    JoinResult total{};
    for (std::uint32_t pid = 0; pid < p; ++pid) {
        const JoinResult local = join_one_partition(Rpart, Spart, pid);
        total.join_count += local.join_count;
        total.checksum1  += local.checksum1;
        total.checksum2  += local.checksum2;
    }
    auto t2 = Clock::now();

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
    std::uint64_t nr = 0, ns = 0, seed = 0, max_key = 0, p = 0, hot = 0;
    double skew = 0.0;

    if (!read_arg_u64(argc, argv, "-nr", nr) ||
        !read_arg_u64(argc, argv, "-ns", ns) ||
        !read_arg_u64(argc, argv, "-seed", seed) ||
        !read_arg_u64(argc, argv, "-max-key", max_key) ||
        !read_arg_u64(argc, argv, "-p", p)) {
        usage_common(argv[0], /*show_threads=*/false);
        return 1;
    }
    read_arg_dbl(argc, argv, "-skew", skew);
    read_arg_u64(argc, argv, "-hot", hot);

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

    // R and S use different generation seeds but share the SAME hot pool seed,
    // so hot keys overlap between R and S and the join finds matches inside
    // hot partitions. (The pool seed is a fixed constant in SkewConfig.)
    const auto R = generate_skewed_relation(NR, seed, max_key, P, skew_cfg);
    const auto S = generate_skewed_relation(NS, seed ^ 0xdeadebdecdeedef1ULL, max_key, P, skew_cfg);

    const auto t0 = std::chrono::steady_clock::now();
    const JoinResult result = partitioned_hash_join_sequential(R, S, P);
    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "NR=" << NR << " NS=" << NS << " P=" << P
              << " seed=" << seed
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
