// hashjoin_par_v2.cpp
//
// Parallel version 2: parallel join + parallel histogram.
//
// Each thread counts records in its own chunk of the input (histogram).
// The prefix sum is left sequential: its input is only P=256 values,
// so the overhead of parallelizing it exceeds any potential gain.
// The scatter phase is still sequential.
// The join phase is parallelized identically to v1.
//
// Run example:
//   ./hashjoin_par_v2 -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8
//

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <thread>

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
        << "  -p          Number of partitions (power of two)\n"
        << "  -t          Number of threads (default: hardware concurrency)\n";
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
// Histogram Result
// ------------------------------------------------------------
//
// Holds both the global histogram and the per-thread local histograms.
// The local histograms are kept here for future use.
// In this version (v2) the scatter is still sequential and uses
// only the global histogram.
//
struct HistogramResult {
    std::vector<std::size_t> global;                   // global_hist[pid] = total records in partition pid
    std::vector<std::vector<std::size_t>> local_hists; // local_hists[t][pid] = records in partition pid seen by thread t
};

// ------------------------------------------------------------
// Parallel Histogram
// ------------------------------------------------------------
//
// Each thread processes its own contiguous chunk of the input relation
// and counts how many records belong to each partition.
// Each thread writes only to its own private local histogram row,
// so no synchronization is needed during counting.
//
// After all threads complete, the main thread merges the local histograms
// into a single global histogram by summing element-wise.
//
static HistogramResult compute_histogram(const std::vector<Record>& rel, std::uint32_t p, int T) {
    const std::size_t N = rel.size();

    // Allocate one private histogram per thread, initialized to zero.
    std::vector<std::vector<std::size_t>> local_hists(T, std::vector<std::size_t>(p, 0));

    std::vector<std::thread> threads;
    for (int t = 0; t < T; t++) {
        threads.push_back(std::thread([&, t]() {
            // Each thread processes a contiguous chunk of the input.
            // The last thread takes any remaining records (when N is not divisible by T).
            const std::size_t chunk = N / T;
            const std::size_t begin = t * chunk;
            const std::size_t end   = (t == T - 1) ? N : begin + chunk;

            for (std::size_t i = begin; i < end; i++) {
                const std::uint32_t pid = compute_partition_id(rel[i].key, p);
                ++local_hists[t][pid];  // write only to this thread's private row
            }
        }));
    }
    for (auto& th : threads) th.join();

    // Merge: sum all local histograms into the global one.
    std::vector<std::size_t> global(p, 0);
    for (int t = 0; t < T; t++)
        for (std::uint32_t pid = 0; pid < p; pid++)
            global[pid] += local_hists[t][pid];

    return HistogramResult{global, local_hists};
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
// Runs the three partitioning steps in sequence:
//   1. Parallel histogram  -> counts records per partition per thread
//   2. Sequential prefix sum -> computes partition start offsets
//   3. Sequential scatter    -> reorders records into contiguous partitions
//
// After this phase, partition p is stored at data[begin[p]..end[p])
// and can be processed independently from all other partitions.
//
static PartitionedRelation partition_relation(const std::vector<Record>& rel, std::uint32_t p, int T) {
    // Step 1: parallel histogram — also returns per-thread local histograms
    const HistogramResult hres = compute_histogram(rel, p, T);

    // Step 2: prefix sum — sequential, O(P), negligible cost.
    const auto begin = exclusive_prefix_sum(hres.global);

    // Step 3: sequential scatter
    auto data = scatter_partitioned(rel, p, begin);

    std::vector<std::size_t> end(p, 0);
    for (std::uint32_t pid = 0; pid < p; ++pid) {
        end[pid] = begin[pid] + hres.global[pid];
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
    const std::size_t r_end   = Rpart.end[pid];
    const std::size_t s_begin = Spart.begin[pid];
    const std::size_t s_end   = Spart.end[pid];

    // Nothing to do if either partition is empty.
    if (r_begin == r_end || s_begin == s_end)
        return result;

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
// Full parallel partitioned hash join
// ------------------------------------------------------------
//
// This is the end-to-end baseline:
//
//   1. Partition R (parallel histogram, sequential scatter)
//   2. Partition S (parallel histogram, sequential scatter)
//   3. Parallel local join across all partitions
//   4. Sequential reduction of per-thread results
//
// Parallelization strategy for the join phase:
//   Thread t processes partitions t, t+T, t+2T, ... (stride-T round-robin).
//   This distributes the P partitions evenly across T threads.
//   Each thread writes only to its own private JoinResult,
//   so no shared mutable state exists — no atomics or mutexes needed.
//
static JoinResult partitioned_hash_join_parallel(const std::vector<Record>& R,
                                                   const std::vector<Record>& S,
                                                   std::uint32_t p,
                                                   int T) {
    using Clock = std::chrono::steady_clock;

    auto t0 = Clock::now();
    // Phase 1 & 2: partition both relations in parallel.
    const PartitionedRelation Rpart = partition_relation(R, p, T);
    const PartitionedRelation Spart = partition_relation(S, p, T);
    auto t1 = Clock::now();

    // Phase 3: parallel local join.
    // Each thread accumulates results in its own private JoinResult.
    std::vector<JoinResult> thread_results(T);

    std::vector<std::thread> threads;
    for (int t = 0; t < T; t++) {
        threads.push_back(std::thread([&, t]() {
            // Stride-T assignment: thread t takes partitions t, t+T, t+2T, ...
            for (std::uint32_t pid = t; pid < p; pid += T) {
                const JoinResult local = join_one_partition(Rpart, Spart, pid);
                thread_results[t].join_count += local.join_count;
                thread_results[t].checksum1  += local.checksum1;
                thread_results[t].checksum2  += local.checksum2;
            }

        }));
    }

    // Wait for everyone to finish
    for (auto& th : threads) th.join();
    auto t2 = Clock::now();

    // Phase 4: sequential reduction — sum all per-thread results.
    JoinResult total{};
    for (int t = 0; t < T; t++) {
        total.join_count += thread_results[t].join_count;
        total.checksum1  += thread_results[t].checksum1;
        total.checksum2  += thread_results[t].checksum2;
    }

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
    std::uint64_t nr = 0, ns = 0, seed = 0, max_key = 0, p = 0, t=0;

    if (!read_arg_u64(argc, argv, "-nr", nr) ||
        !read_arg_u64(argc, argv, "-ns", ns) ||
        !read_arg_u64(argc, argv, "-seed", seed) ||
        !read_arg_u64(argc, argv, "-max-key", max_key) ||
        !read_arg_u64(argc, argv, "-p", p)) {
        usage(argv[0]);
        return 1;
    }

    // -t is optional: if not provided, use the number of hardware cores.
    read_arg_u64(argc, argv, "-t", t);
    const int T = (t == 0)
        ? static_cast<int>(std::thread::hardware_concurrency())
        : static_cast<int>(t);

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
    const JoinResult result = partitioned_hash_join_parallel(R, S, P, T);
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
