// common.hpp
//
// Shared utilities for all four implementations of Module 3:
//   - hashjoin_seq.cpp        (sequential baseline)
//   - hashjoin_par.cpp        (C++ std::thread, Module 2 v3)
//   - hashjoin_omp_loop.cpp   (OpenMP loop-level parallelism)
//   - hashjoin_omp_task.cpp   (OpenMP task-based parallelism)
//
// Contains code that is identical across all implementations:
//   - Record / JoinResult / PartitionedRelation type definitions
//   - splitmix64 family (deterministic PRNG and stateless mixer)
//   - multiply_shift partition mapping function (Module 1)
//   - FlatCountMap: flat open-addressing hash table for the build phase
//   - Uniform and skewed input generation
//   - CLI argument parsing
//   - Naive O(|R|*|S|) verifier for very small inputs
//

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// ------------------------------------------------------------
// Record definition
// ------------------------------------------------------------
// For this project we only store the key. A payload column could
// be added without affecting any of the parallelisation choices.
//
struct Record {
    std::uint64_t key{};
};

// ------------------------------------------------------------
// Join result (used both as a per-partition and as a global result)
// ------------------------------------------------------------
struct JoinResult {
    std::uint64_t join_count = 0;
    std::uint64_t checksum1  = 0;
    std::uint64_t checksum2  = 0;
};

// ------------------------------------------------------------
// Partitioned relation metadata
// ------------------------------------------------------------
// After partitioning, partition `pid` occupies data[begin[pid] .. end[pid]).
//
struct PartitionedRelation {
    std::vector<Record>      data;
    std::vector<std::size_t> begin;
    std::vector<std::size_t> end;
};

inline bool read_arg_u64(int argc, char** argv, const std::string& name, std::uint64_t& out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            out = std::strtoull(argv[i + 1], nullptr, 10);
            return true;
        }
    }
    return false;
}
inline bool read_arg_dbl(int argc, char** argv, const std::string& name, double& out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            out = std::strtod(argv[i + 1], nullptr);
            return true;
        }
    }
    return false;
}
inline bool is_power_of_two(std::uint32_t x) {
    return x != 0 && (x & (x - 1U)) == 0;
}

// ------------------------------------------------------------
// Deterministic pseudo-random generation
// ------------------------------------------------------------
// We use splitmix64 both to generate reproducible keys (statefully)
// and to mix keys for the checksums (statelessly).
// https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
//
inline std::uint64_t splitmix64_mix(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}
inline std::uint64_t splitmix64(std::uint64_t x) {
    return splitmix64_mix(x + 0x9e3779b97f4a7c15ULL);
}
inline std::uint64_t splitmix64_next(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    return splitmix64_mix(state);
}

// ------------------------------------------------------------
// Partition mapping — multiply_shift (Module 1)
// ------------------------------------------------------------
// Dietzfelbinger universal hashing:
//   pid = (key * MULT_PRIME) >> (64 - log2(P))
//
// MULT_PRIME = floor(2^64 / phi) is also reused in FlatCountMap so that
// the same odd multiplier is applied at both partition mapping and
// build-table indexing, with no need for a second large constant.
//
inline constexpr std::uint64_t MULT_PRIME = 0x9E3779B97F4A7C15ULL;

inline int log2_pot(std::uint32_t p) {
    return __builtin_ctzll(p);
}
inline std::uint32_t compute_partition_id(std::uint64_t key, std::uint32_t p) {
    const int shift = 64 - log2_pot(p);
    return static_cast<std::uint32_t>((key * MULT_PRIME) >> shift);
}

// ------------------------------------------------------------
// FlatCountMap: flat open-addressing hash table
// -----------------------------------------------------------
// Same structure used in the Module 2 implementation.
//
// Load factor is kept <= 0.5 (table sized to the next power of two
// above 2 * capacity). Empty slots use the sentinel ~0ULL; this is
// safe because input keys fall into [0, max_key) with max_key << 2^64.
//
struct FlatCountMap {
    static constexpr std::uint64_t EMPTY_KEY = ~0ULL;

    struct Slot {
        std::uint64_t key;
        std::uint32_t count;
    };

    std::vector<Slot> table;
    std::uint64_t     mask;

    explicit FlatCountMap(std::size_t capacity) {
        std::size_t sz = 1;
        while (sz < capacity * 2) sz <<= 1;
        mask = sz - 1;
        table.assign(sz, {EMPTY_KEY, 0});
    }

    void insert(std::uint64_t key) {
        std::uint64_t idx = (key * MULT_PRIME) & mask;
        while (table[idx].key != EMPTY_KEY && table[idx].key != key)
            idx = (idx + 1) & mask;
        if (table[idx].key == EMPTY_KEY) table[idx].key = key;
        ++table[idx].count;
    }

    std::uint32_t find(std::uint64_t key) const {
        std::uint64_t idx = (key * MULT_PRIME) & mask;
        while (table[idx].key != EMPTY_KEY && table[idx].key != key)
            idx = (idx + 1) & mask;
        return (table[idx].key == key) ? table[idx].count : 0;
    }
};

// ------------------------------------------------------------
// Local join on one partition (shared by all four implementations)
// ------------------------------------------------------------
// Build phase: scan R_p and record the multiplicity of each key.
// Probe phase: scan S_p; for each key, look it up and add countR[key]
// matches to the result. Duplicates are handled naturally by counting.
//
inline JoinResult join_one_partition(const PartitionedRelation& Rpart,
                                     const PartitionedRelation& Spart,
                                     std::uint32_t pid) {
    JoinResult result{};

    const std::size_t r_begin = Rpart.begin[pid];
    const std::size_t r_end   = Rpart.end[pid];
    const std::size_t s_begin = Spart.begin[pid];
    const std::size_t s_end   = Spart.end[pid];

    if (r_begin == r_end || s_begin == s_end)
        return result;

    FlatCountMap countR(r_end - r_begin);
    for (std::size_t i = r_begin; i < r_end; ++i)
        countR.insert(Rpart.data[i].key);

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
// Uniform input generation (same as Module 1 / Module 2)
// ------------------------------------------------------------
// Keys drawn uniformly from [0, max_key). Combined with the multiply_shift
// partition mapping, this produces near-perfect partition balance
// (CV ~ 0.002 for P = 256, verified in Module 1).
//
inline std::vector<Record> generate_relation(std::size_t n,
                                             std::uint64_t seed,
                                             std::uint64_t max_key) {
    std::vector<Record> out(n);
    std::uint64_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t r = splitmix64_next(state);
        out[i].key = (max_key == 0) ? 0ULL : (r % max_key);
    }
    return out;
}

// ------------------------------------------------------------
// Skewed input generation
// ------------------------------------------------------------
// Generates an input where a controllable fraction `skew_frac` of records
// falls into a small subset of `num_hot` "hot" partitions, while the
// remaining records are uniformly distributed across all P partitions.
//
// Strategy:
//   1. Pre-compute a pool of "hot keys" by rejection sampling: draw uniform
//      keys and keep those whose partition id < num_hot. We collect a fixed
//      number of distinct keys per hot partition so that the join still
//      produces matches between R and S inside hot partitions.
//   2. For each generated record, with probability `skew_frac` pick a key
//      from the hot pool; otherwise generate a uniform key.
//
struct SkewConfig {
    double        skew_frac           = 0.0;   // fraction of records routed to hot partitions
    std::uint32_t num_hot             = 4;     // number of hot partitions
    std::uint32_t hot_keys_per_part   = 1024;  // distinct hot keys generated per hot partition
    std::uint64_t pool_seed           = 0xC0FFEEFEEDFACE01ULL; // shared seed for the hot key pool
};

inline std::vector<std::uint64_t> build_hot_pool(std::uint32_t p,
                                                 std::uint64_t max_key,
                                                 const SkewConfig& cfg) {
    if (cfg.num_hot == 0 || cfg.hot_keys_per_part == 0)
        return {};

    std::vector<std::uint64_t> pool;
    pool.reserve(static_cast<std::size_t>(cfg.num_hot) * cfg.hot_keys_per_part);

    std::vector<std::uint32_t> per_part(cfg.num_hot, 0);
    std::uint32_t filled_parts = 0;

    std::uint64_t state = cfg.pool_seed;
    while (filled_parts < cfg.num_hot) {
        const std::uint64_t r = splitmix64_next(state);
        const std::uint64_t k = (max_key == 0) ? 0ULL : (r % max_key);
        const std::uint32_t pid = compute_partition_id(k, p);
        if (pid < cfg.num_hot && per_part[pid] < cfg.hot_keys_per_part) {
            pool.push_back(k);
            if (++per_part[pid] == cfg.hot_keys_per_part)
                ++filled_parts;
        }
    }
    return pool;
}

inline std::vector<Record> generate_skewed_relation(std::size_t n,
                                                    std::uint64_t seed,
                                                    std::uint64_t max_key,
                                                    std::uint32_t p,
                                                    const SkewConfig& cfg) {
    // Falling back to the uniform generator keeps single code path
    // for the skew_frac == 0 case (used by Module 2 baseline runs).
    if (cfg.skew_frac <= 0.0)
        return generate_relation(n, seed, max_key);

    const auto hot_pool = build_hot_pool(p, max_key, cfg);
    const std::uint64_t pool_size = hot_pool.size();

    // Convert skew_frac into an integer threshold over the unsigned 64-bit range
    // to avoid double comparisons inside the hot loop.
    const std::uint64_t hot_threshold =
        static_cast<std::uint64_t>(cfg.skew_frac * static_cast<double>(~0ULL));

    std::vector<Record> out(n);
    std::uint64_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t r1 = splitmix64_next(state);
        if (r1 < hot_threshold) {
            const std::uint64_t r2 = splitmix64_next(state);
            out[i].key = hot_pool[r2 % pool_size];
        } else {
            const std::uint64_t r2 = splitmix64_next(state);
            out[i].key = (max_key == 0) ? 0ULL : (r2 % max_key);
        }
    }
    return out;
}

// ------------------------------------------------------------
// Naive verifier for very small inputs
// ------------------------------------------------------------
// O(|R| * |S|) brute-force verification, used by `make check`.
//
inline JoinResult naive_join_verifier(const std::vector<Record>& R,
                                      const std::vector<Record>& S) {
    JoinResult result{};
    for (const auto& r : R) {
        for (const auto& s : S) {
            if (r.key == s.key) {
                result.join_count += 1;
                result.checksum1  += splitmix64(r.key);
                result.checksum2  += splitmix64(r.key ^ 0x9e3779b97f4a7c15ULL);
            }
        }
    }
    return result;
}

// ------------------------------------------------------------
// Common usage banner (each .cpp can extend it)
// ------------------------------------------------------------
inline void usage_common(const char* prog, bool show_threads) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " -nr NR -ns NS -seed SEED -max-key K -p P"
        << (show_threads ? " [-t T]" : "")
        << " [-skew F] [-hot H]\n\n"
        << "Required:\n"
        << "  -nr NR        Number of records in relation R\n"
        << "  -ns NS        Number of records in relation S\n"
        << "  -seed SEED    Deterministic seed\n"
        << "  -max-key K    Keys are generated in [0, K)\n"
        << "  -p P          Number of partitions (power of two)\n";
    if (show_threads) {
        std::cerr
            << "  -t T          Number of threads (default: hardware concurrency)\n";
    }
    std::cerr
        << "Optional (skewed workload):\n"
        << "  -skew F       Fraction of records mapped to hot partitions (default 0.0 = uniform)\n"
        << "  -hot H        Number of hot partitions (default 4, ignored if skew=0)\n";
}
