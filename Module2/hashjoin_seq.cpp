// hashjoin_seq.cpp
//
// Sequential reference for Module 2
// Partitioned Hash Join with Duplicates
//
// This code is intentionally written to be simple and readable.
// It is meant as a reference baseline and as a starting point for
// the parallel version. You can modify it for improving performance,
// provided you do not change the overall algorithm.
// 
//
// IMPORTANT:
// The function compute_partition_id(...) below is intentionally very simple.
// Students must replace it with their own mapping function from Module 1.
// The same mapping function must be used consistently in both the sequential
// and parallel versions.
//
// Run example:
//   ./hashjoin_seq -nr 5 -ns 8 -seed 13 -max-key 8 -p 4
//
// Output:
//   join_count
//   checksum1
//   checksum2
//
//
// The code follows these phases:
//
//   1. Input generation
//      Generate two relations R and S with deterministic keys.
//
//   2. Partitioning of R and S
//      The goal of this phase is to reorganize the data so that
//      records belonging to the same partition are stored contiguously.
//
//      This is done in three steps:
//
//      - mapping key -> partition id
//        Each key is mapped to a partition identifier in [0, P).
//
//      - histogram
//        Count how many records are assigned to each partition.
//        This tells us how much space each partition will occupy.
//
//      - prefix sum (offset computation)
//        Convert counts into starting positions (offsets) for each partition
//        in the output array.
//
//      - scatter
//        Move each record to its correct position so that all records
//        of the same partition are stored contiguously.
//
//      After this phase, each partition corresponds to a contiguous
//      segment of the array, and can be processed independently.
//
//   3. Local join per partition
//      For each partition p:
//
//      - build
//        Scan the R partition and count how many times each key appears.
//
//      - probe
//        Scan the corresponding S partition.
//        For each key, if it exists in R, add as many matches as its multiplicity.
//
//   4. Final output
//      Accumulate results across all partitions.
//
// The result does NOT materialize the join pairs.
// It only computes:
//   - total number of matches
//   - two checksums for correctness verification
//

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// ------------------------------------------------------------
// Record definition
// ------------------------------------------------------------
//
// For this reference implementation we only store the key.
// You may extend the record with a payload in later versions if desired.
//
struct Record {
    std::uint64_t key{};
};

// ------------------------------------------------------------
// Utility: command-line parsing
// ------------------------------------------------------------
static bool read_arg_u64(int argc, char** argv, const std::string& name, std::uint64_t& out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            out = std::strtoull(argv[i + 1], nullptr, 10);
            return true;
        }
    }
    return false;
}
static void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " -nr NR -ns NS -seed SEED -max-key K -p P\n\n"
        << "Parameters:\n"
        << "  -nr         Number of records in relation R\n"
        << "  -ns         Number of records in relation S\n"
        << "  -seed       Deterministic seed\n"
        << "  -max-key    Keys are generated in [0, max-key)\n"
        << "  -p          Number of partitions (power of two required in this reference code)\n";
}
static bool is_power_of_two(std::uint32_t x) {
    return x != 0 && (x & (x - 1U)) == 0;
}

// ------------------------------------------------------------
// Deterministic pseudo-random generation
// ------------------------------------------------------------
//
// We use splitmix64 to generate reproducible keys and also for checksum.
// https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
//
// splitmix64_next is used as a deterministic pseudo-random generator step,
// while splitmix64 is used as a stateless 64-bit mixing function for checksums. 
//
static inline std::uint64_t splitmix64_mix(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}
static inline std::uint64_t splitmix64(std::uint64_t x) {
    return splitmix64_mix(x + 0x9e3779b97f4a7c15ULL);
}
static inline std::uint64_t splitmix64_next(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    return splitmix64_mix(state);
}


static std::vector<Record> generate_relation(std::size_t n, std::uint64_t seed, std::uint64_t max_key) {
    std::vector<Record> out(n);
    std::uint64_t state = seed;

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t r = splitmix64_next(state);
        out[i].key = (max_key == 0) ? 0ULL : (r % max_key);
    }
    return out;
}


// ------------------------------------------------------------
// Partition mapping — multiply_shift (Module 1)
// ------------------------------------------------------------
//
// Replaces the placeholder function provided by the professor.
// This is the Dietzfelbinger multiply-shift universal hash function.
//
// Given a key and P (power of two), it computes:
//   partition_id = (key * MULT_PRIME) >> (64 - log2(P))
//
// This guarantees a mathematically uniform distribution across partitions,
// which ensures good load balance during the parallel join phase.
// The same function is used in hashjoin_seq.cpp for a fair comparison.
//
static constexpr std::uint64_t MULT_PRIME = 0x9E3779B97F4A7C15ULL;  // 2^64 / phi

static inline int log2_pot(std::uint32_t p) {
    return __builtin_ctzll(p);  // counts trailing zeros = log2 for power-of-two
}

static inline std::uint32_t compute_partition_id(std::uint64_t key, std::uint32_t p) {
    const int shift = 64 - log2_pot(p);
    return static_cast<std::uint32_t>((key * MULT_PRIME) >> shift);
}

// ------------------------------------------------------------
// Histogram
// ------------------------------------------------------------
//
// Count how many records go to each partition.
//
// hist[pid] = number of records whose key maps to pid
//
static std::vector<std::size_t> compute_histogram(const std::vector<Record>& rel, std::uint32_t p) {
    std::vector<std::size_t> hist(p, 0);

    for (const auto& rec : rel) {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        ++hist[pid];
    }
    return hist;
}

// ------------------------------------------------------------
// Prefix sum (exclusive scan)
// ------------------------------------------------------------
//
// Given a histogram, compute the begin offsets of each partition.
//
// Example:
//   hist  = [0, 1, 2, 5]
//   begin = [0, 0, 1, 3]
//
// Then partition p occupies [begin[p], begin[p] + hist[p]).
//
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
// Scatter into a partitioned array
// ------------------------------------------------------------
//
// Reorder records so that all records belonging to the same partition become
// contiguous in memory.
//
// We use a write cursor per partition, initialized from the begin offsets.
//
static std::vector<Record> scatter_partitioned(const std::vector<Record>& rel,
                                               std::uint32_t p,
                                               const std::vector<std::size_t>& begin) {
    std::vector<Record> out(rel.size());

    // Current write position for each partition.
    std::vector<std::size_t> next = begin;

    for (const auto& rec : rel) {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        out[next[pid]++] = rec;
    }

    return out;
}

// ------------------------------------------------------------
// Partitioned relation metadata
// ------------------------------------------------------------
//
// This stores:
//   - the partitioned array
//   - the begin offset of each partition
//   - the end offset of each partition
//
// So partition p is data[begin[p] .. end[p]).
//
struct PartitionedRelation {
    std::vector<Record> data;
    std::vector<std::size_t> begin;
    std::vector<std::size_t> end;
};

// ------------------------------------------------------------
// Full partitioning pipeline for one relation
// ------------------------------------------------------------
//
// This groups together the steps:
//   histogram -> prefix sum -> scatter -> end offsets
//
// After this phase, all records belonging to the same partition
// are stored contiguously in memory, enabling independent processing.
//
static PartitionedRelation partition_relation(const std::vector<Record>& rel, std::uint32_t p) {
    const auto hist = compute_histogram(rel, p);
    const auto begin = exclusive_prefix_sum(hist);
    auto data = scatter_partitioned(rel, p, begin);

    std::vector<std::size_t> end(p, 0);
    for (std::uint32_t pid = 0; pid < p; ++pid) {
        end[pid] = begin[pid] + hist[pid];
    }

    return PartitionedRelation{
        .data = std::move(data),
        .begin = begin,
        .end = end
    };
}

// ------------------------------------------------------------
// Join result
// ------------------------------------------------------------
struct JoinResult {
    std::uint64_t join_count = 0;
    std::uint64_t checksum1 = 0;
    std::uint64_t checksum2 = 0;
};

// ------------------------------------------------------------
// Flat open-addressing hash table for the build phase
// ------------------------------------------------------------
//
// Replaces std::unordered_map<uint64_t, uint32_t> used in the reference
// implementation. Instead of heap-allocated bucket chains, all slots are
// stored in a single contiguous array, eliminating pointer indirection
// and improving cache utilisation during both build and probe.
//
// Collision resolution: linear probing.
// Hash function: multiply-shift with MULT_PRIME (same as partition mapping).
// Load factor: kept below 0.5 by allocating at least 2 * capacity slots,
//              so the table size is always the next power of two >= 2 * capacity.
//
// Empty slots are marked with EMPTY_KEY = 0xFFFF...FFFF (all bits set).
// Keys equal to EMPTY_KEY cannot be stored; this is safe because keys are
// generated in [0, max-key) with max-key << 2^64.
//
struct FlatCountMap {
    static constexpr uint64_t EMPTY_KEY = ~0ULL;  // sentinel for an unused slot

    struct Slot {
        uint64_t key;    // stored key, or EMPTY_KEY if slot is unused
        uint32_t count;  // number of times key has been inserted
    };

    std::vector<Slot> table;
    uint64_t mask;  // table.size() - 1; used for fast modulo via bitwise AND

    // Allocate a table large enough to hold `capacity` entries at load <= 0.5.
    explicit FlatCountMap(std::size_t capacity) {
        std::size_t sz = 1;
        while (sz < capacity * 2) sz <<= 1;  // next power of two >= 2 * capacity
        mask = sz - 1;
        table.assign(sz, {EMPTY_KEY, 0});
    }

    // Insert a key, incrementing its count.
    // If the key is not yet present, a new slot is claimed and count set to 1.
    void insert(uint64_t key) {
        uint64_t idx = (key * MULT_PRIME) & mask;
        while (table[idx].key != EMPTY_KEY && table[idx].key != key)
            idx = (idx + 1) & mask;  // linear probing
        if (table[idx].key == EMPTY_KEY) table[idx].key = key;
        ++table[idx].count;
    }

    // Return the count associated with key, or 0 if not present.
    uint32_t find(uint64_t key) const {
        uint64_t idx = (key * MULT_PRIME) & mask;
        while (table[idx].key != EMPTY_KEY && table[idx].key != key)
            idx = (idx + 1) & mask;  // linear probing
        return (table[idx].key == key) ? table[idx].count : 0;
    }
};

// ------------------------------------------------------------
// Local join on one partition
// ------------------------------------------------------------
//
// We process one partition p as follows:
//
//   Build:
//     Scan R_p and count how many times each key appears,
//     storing the result in a flat open-addressing hash table.
//
//   Probe:
//     Scan S_p. For each key, look it up in the build table.
//     If found, the key contributes countR[key] matches to the result.
//
// Duplicates are handled by counting occurrences in R.
// Each record in S contributes as many matches as the multiplicity
// of its key in the corresponding partition of R.
//
static JoinResult join_one_partition(const PartitionedRelation& Rpart,
                                     const PartitionedRelation& Spart,
                                     std::uint32_t pid) {
    JoinResult result{};

    const std::size_t r_begin = Rpart.begin[pid];
    const std::size_t r_end = Rpart.end[pid];
    const std::size_t s_begin = Spart.begin[pid];
    const std::size_t s_end = Spart.end[pid];

    // Nothing to do if either partition is empty.
    if (r_begin == r_end || s_begin == s_end) {
        return result;
    }

    // Build phase:
    // scan R_p and record the multiplicity of each key.
    FlatCountMap countR(r_end - r_begin);
    for (std::size_t i = r_begin; i < r_end; ++i)
        countR.insert(Rpart.data[i].key);

    // Probe phase:
    // for each key in S_p, if it exists in countR, add countR[key] matches.
    for (std::size_t i = s_begin; i < s_end; ++i) {
        const std::uint64_t key = Spart.data[i].key;
        const std::uint32_t multiplicity = countR.find(key);
        if (multiplicity) {
            result.join_count += multiplicity;
            result.checksum1  += splitmix64(key) * multiplicity;
            result.checksum2  += splitmix64(key ^ 0x9e3779b97f4a7c15ULL) * multiplicity;
        }
    }

    return result;
}

// ------------------------------------------------------------
// Full sequential partitioned hash join
// ------------------------------------------------------------
//
// This is the end-to-end baseline:
//
//   1. Partition R
//   2. Partition S
//   3. For each partition p:
//        local build + local probe
//   4. Accumulate results
//
// Each partition can be processed independently.
// This property is the basis for parallelization in Module 2.
//
static JoinResult partitioned_hash_join_sequential(const std::vector<Record>& R,
                                                   const std::vector<Record>& S,
                                                   std::uint32_t p) {
    using Clock = std::chrono::steady_clock;

    auto t0 = Clock::now();
    // Phase 1: partition both relations
    const PartitionedRelation Rpart = partition_relation(R, p);
    const PartitionedRelation Spart = partition_relation(S, p);
    auto t1 = Clock::now();

    // Phase 2 + 3: local joins and global reduction
    JoinResult total{};

    for (std::uint32_t pid = 0; pid < p; ++pid) {
        const JoinResult local = join_one_partition(Rpart, Spart, pid);
        total.join_count += local.join_count;
        total.checksum1 += local.checksum1;
        total.checksum2 += local.checksum2;
    }
    auto t2 = Clock::now();

    // print phase breakdown to stderr so it doesn't pollute stdout
    std::cerr << "[phases] partitioning="
              << std::chrono::duration<double>(t1 - t0).count()
              << "s  join="
              << std::chrono::duration<double>(t2 - t1).count()
              << "s\n";

    return total;
}

// ------------------------------------------------------------
// Verifier for very small inputs
// ------------------------------------------------------------
//
// This is useful only for debugging and correctness testing on tiny
// datasets. It checks all pairs directly, so its complexity is O(|R|*|S|).
//
static JoinResult naive_join_verifier(const std::vector<Record>& R,
                                      const std::vector<Record>& S) {
    JoinResult result{};

    for (const auto& r : R) {
        for (const auto& s : S) {
            if (r.key == s.key) {
                result.join_count += 1;
				result.checksum1 += splitmix64(r.key);
				result.checksum2 += splitmix64(r.key ^ 0x9e3779b97f4a7c15ULL);
            }
        }
    }
    return result;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv) {
    std::uint64_t nr = 0, ns = 0, seed = 0, max_key = 0, p = 0;

    if (!read_arg_u64(argc, argv, "-nr", nr) ||
        !read_arg_u64(argc, argv, "-ns", ns) ||
        !read_arg_u64(argc, argv, "-seed", seed) ||
        !read_arg_u64(argc, argv, "-max-key", max_key) ||
        !read_arg_u64(argc, argv, "-p", p)) {
        usage(argv[0]);
        return 1;
    }

    if (p > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "Error: P too large.\n";
        return 1;
    }

    const std::uint32_t P = static_cast<std::uint32_t>(p);

	// The power-of-two constraint on P is only due to the default
	// mapping used here. It may be removed if the chosen partition
	// function correctly handles arbitrary P
    if (!is_power_of_two(P)) {
        std::cerr << "Error: in this reference implementation, P must be a power of two.\n";
        return 1;
    }

    const std::size_t NR = static_cast<std::size_t>(nr);
    const std::size_t NS = static_cast<std::size_t>(ns);

    // Deterministic generation.
    // We use two different seeds so that R and S are not identical.
    const auto R = generate_relation(NR, seed, max_key);
    const auto S = generate_relation(NS, seed ^ 0xdeadebdecdeedef1ULL, max_key);

    // Time only the join pipeline, not input generation.
    const auto t0 = std::chrono::steady_clock::now();
    const JoinResult result = partitioned_hash_join_sequential(R, S, P);
    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "NR=" << NR << " NS=" << NS << " P=" << P
			  << " seed=" << seed
              << " [0, " << max_key << ")\n";

    std::cout << "join_count=" << result.join_count << "\n";
    std::cout << "checksum1=" << result.checksum1 << "\n";
    std::cout << "checksum2=" << result.checksum2 << "\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "time_sec=" << sec << "\n";

    //Tiny debug check, only for very small datasets
    if (NR <= 500 && NS <= 500) {
        const JoinResult naive = naive_join_verifier(R, S);
        std::cout << "naive_join_count=" << naive.join_count << "\n";
        std::cout << "naive_checksum1=" << naive.checksum1 << "\n";
        std::cout << "naive_checksum2=" << naive.checksum2 << "\n";
    }

    return 0;
}
