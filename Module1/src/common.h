#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>
#include <functional>


using pkey_t  = uint64_t;   // partition key (64 bit)
using partid_t = uint32_t;  // partition identifier (32 bit)

// Hash function identifiers
enum class HashType : int {
    MASK = 0,            // key & (P-1)
    MULTIPLY_SHIFT,      // Dietzfelbinger: (key * PRIME) >> (64 - log2P)
    MURMUR_FMIX64,       // murmur3 fmix64 finalizer then mask
    XOR_SHIFT,           // (key ^ (key >> 33)) & (P-1)  — lightweight bit mixing
    FOLD32_MULT,         // fold 64→32 via XOR, then 32-bit multiply-shift
    NUM_TYPES
};

static const char* hash_name(HashType h) {
    switch (h) {
        case HashType::MASK:            return "mask";
        case HashType::MULTIPLY_SHIFT:  return "multiply_shift";
        case HashType::MURMUR_FMIX64:   return "murmur_fmix64";
        case HashType::XOR_SHIFT:       return "xor_shift";
        case HashType::FOLD32_MULT:     return "fold32_mult";
        default:                        return "unknown";
    }
}

// Hash constants
static constexpr uint64_t MULT_PRIME   = 0x9E3779B97F4A7C15ULL; // 2^64 / phi
static constexpr uint64_t MURMUR_C1    = 0xFF51AFD7ED558CCDULL;
static constexpr uint64_t MURMUR_C2    = 0xC4CEB9FE1A85EC53ULL;
static constexpr uint32_t MULT_PRIME32 = 0x9E3779B9u;           // 2^32 / phi (32-bit)

// log2 for power-of-two P
static inline int log2_pot(uint64_t P) {
    return __builtin_ctzll(P);
}

// Key Generation — deterministic splitmix64
static void generate_keys(pkey_t* keys, size_t N, uint64_t seed,
                          uint64_t key_space = 0)
{
    uint64_t state = seed;
    for (size_t i = 0; i < N; i++) {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        keys[i] = (key_space > 0) ? (z % key_space) : z;
    }
}

// Verification: FNV-1a checksum over partition IDs
static uint64_t compute_checksum(const partid_t* ids, size_t N) {
    uint64_t h = 0xCBF29CE484222325ULL;   //offset basis
    const uint8_t* p = reinterpret_cast<const uint8_t*>(ids);
    size_t bytes = N * sizeof(partid_t);
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 0x100000001B3ULL;
    }
    return h;
}

// Element-by-element comparison (for small N)
static size_t compare_arrays(const partid_t* ref, const partid_t* test,
                             size_t N, size_t max_print = 10)
{
    size_t mis = 0;
    for (size_t i = 0; i < N; i++) {
        if (ref[i] != test[i]) {
            if (mis < max_print)
                std::cerr << "  mismatch [" << i << "]: ref=" << ref[i]
                          << " got=" << test[i] << "\n";
            mis++;
        }
    }
    return mis;
}

// Timing Analysis
struct BenchResult {
    double median_ms;
    double mean_ms;
    double stddev_ms;
    double throughput_Mkeys_s;
};

static BenchResult run_bench(
    std::function<void()> fn, size_t N,
    int reps = 15, int warmup = 3)
{
    std::vector<double> times;
    times.reserve(reps);

    for (int i = 0; i < warmup; i++) fn();

    for (int i = 0; i < reps; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(times.begin(), times.end());
    double median = times[reps / 2];
    double mean   = std::accumulate(times.begin(), times.end(), 0.0) / reps;
    double sq = 0;
    for (auto t : times) sq += (t - mean) * (t - mean);
    double sd = std::sqrt(sq / reps);
    double tp = (N / 1e6) / (median / 1e3);  // Mkeys/s

    return {median, mean, sd, tp};
}


// Distribution quality analysis
struct DistStats {
    double chi2;
    double cv;        // coefficient of variation
    double max_dev;   // max relative deviation from expected
};

static DistStats distribution_stats(const partid_t* ids, size_t N, uint32_t P) {
    std::vector<size_t> cnt(P, 0);
    for (size_t i = 0; i < N; i++) cnt[ids[i]]++;
    double exp = static_cast<double>(N) / P;
    double chi2 = 0, max_dev = 0, ssq = 0;
    for (uint32_t p = 0; p < P; p++) {
        double d = cnt[p] - exp;
        chi2 += (d * d) / exp;
        double rd = std::abs(d) / exp;
        max_dev = std::max(max_dev, rd);
        ssq += d * d;
    }
    double cv = std::sqrt(ssq / P) / exp;
    return {chi2, cv, max_dev};
}
