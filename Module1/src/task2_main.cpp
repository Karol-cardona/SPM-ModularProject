// Compiled once:    g++ -O3 -march=native -mavx2 → task2_avx2

#include "common.h"
#include "kernels_plain.h"   // plain reference for correctness comparison
#include <immintrin.h>

// ============================================================
//                    AVX2 INTRINSICS KERNELS
//
// AVX2 registers: 256 bits → 4 × uint64_t per vector.
// Key challenge: AVX2 has NO native 64×64→64-bit multiply.
// I emulate it with three 32×32→64 multiplies.
// ============================================================

// ============================================================
// 64-bit multiply (low 64 bits of product)
//
// For a·b mod 2^64:
//   lo_lo   = a_lo * b_lo           (exact 64-bit via _mm256_mul_epu32)
//   cross   = a_lo * b_hi + a_hi * b_lo
//   result  = lo_lo + (cross << 32)
//
// The a_hi * b_hi term affects only bits ≥ 64, so we skip it.
// Cost: 3 × mul_epu32 + 2 × add + 2 × srli + 1 × slli ≈ 9 µops.
// ============================================================
static inline __m256i avx2_mul64(__m256i a, __m256i b)
{
    __m256i a_hi = _mm256_srli_epi64(a, 32);
    __m256i b_hi = _mm256_srli_epi64(b, 32);

    __m256i lo_lo     = _mm256_mul_epu32(a, b);        // a_lo * b_lo
    __m256i a_lo_b_hi = _mm256_mul_epu32(a, b_hi);     // a_lo * b_hi
    __m256i a_hi_b_lo = _mm256_mul_epu32(a_hi, b);     // a_hi * b_lo

    __m256i cross     = _mm256_add_epi64(a_lo_b_hi, a_hi_b_lo);
    __m256i cross_hi  = _mm256_slli_epi64(cross, 32);

    return _mm256_add_epi64(lo_lo, cross_hi);
}

// ============================================================
// Narrow 4 × uint64 → 4 × uint32 (extract low 32 bits)
//
// Input:  [d0 d1 | d2 d3 | d4 d5 | d6 d7]  (8 dwords)
// Want:   [d0, d2, d4, d6] packed into __m128i.
//
// Step 1: shuffle dwords within each 128-bit lane
// Step 2: permute 64-bit chunks across lanes
// ============================================================
static inline __m128i avx2_narrow64_to32(__m256i v)
{
    __m256i shuf = _mm256_shuffle_epi32(v, _MM_SHUFFLE(0, 0, 2, 0));
    __m256i perm = _mm256_permute4x64_epi64(shuf, _MM_SHUFFLE(0, 0, 2, 0));
    return _mm256_castsi256_si128(perm);
}


// MASK kernel: out[i] = key[i] & (P-1)
// 4 keys/iter. 1 AND + narrow. Memory-bound.
static void kernel_avx2_mask(const pkey_t* __restrict__ keys,
                             partid_t* __restrict__ out,
                             size_t N, uint32_t P)
{
    const __m256i vmask = _mm256_set1_epi64x(static_cast<int64_t>(P - 1));
    const size_t vec_end = N & ~3ULL;   //multiple of 4
    size_t i = 0;

    for (; i < vec_end; i += 4) {
        __m256i vk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));
        __m256i vr = _mm256_and_si256(vk, vmask);
        __m128i n  = avx2_narrow64_to32(vr);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), n);
    }
    const uint64_t m = P - 1;
    for (; i < N; i++)
        out[i] = static_cast<partid_t>(keys[i] & m);
}

// MULTIPLY_SHIFT kernel: out[i] = (key[i] * PRIME) >> shift
// 4 keys/iter. 1 emulated mul64 + 1 shift + narrow.
static void kernel_avx2_multiply_shift(const pkey_t* __restrict__ keys,
                                       partid_t* __restrict__ out,
                                       size_t N, uint32_t P)
{
    const __m256i vprime = _mm256_set1_epi64x(static_cast<int64_t>(MULT_PRIME));
    const int shift = 64 - log2_pot(P);
    const size_t vec_end = N & ~3ULL;
    size_t i = 0;

    for (; i < vec_end; i += 4) {
        __m256i vk   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));
        __m256i prod = avx2_mul64(vk, vprime);
        __m256i vr   = _mm256_srli_epi64(prod, shift);
        __m128i n    = avx2_narrow64_to32(vr);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), n);
    }
    for (; i < N; i++)
        out[i] = static_cast<partid_t>((keys[i] * MULT_PRIME) >> shift);
}

// MURMUR fmix64 kernel
// 4 keys/iter. 2 emulated mul64 + 3 xor-shift + 1 AND + narrow.
// Most compute-intensive → benefits most from SIMD.
static void kernel_avx2_murmur(const pkey_t* __restrict__ keys,
                               partid_t* __restrict__ out,
                               size_t N, uint32_t P)
{
    const __m256i vc1   = _mm256_set1_epi64x(static_cast<int64_t>(MURMUR_C1));
    const __m256i vc2   = _mm256_set1_epi64x(static_cast<int64_t>(MURMUR_C2));
    const __m256i vmask = _mm256_set1_epi64x(static_cast<int64_t>(P - 1));
    const size_t vec_end = N & ~3ULL;
    size_t i = 0;

    for (; i < vec_end; i += 4) {
        __m256i k = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));

        // fmix64 round 1
        k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 33));
        k = avx2_mul64(k, vc1);

        // fmix64 round 2
        k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 33));
        k = avx2_mul64(k, vc2);

        // fmix64 round 3 + mask
        k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 33));
        __m256i vr = _mm256_and_si256(k, vmask);

        __m128i n = avx2_narrow64_to32(vr);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), n);
    }
    const uint64_t m = P - 1;
    for (; i < N; i++) {
        uint64_t k = keys[i];
        k ^= k >> 33; k *= MURMUR_C1;
        k ^= k >> 33; k *= MURMUR_C2;
        k ^= k >> 33;
        out[i] = static_cast<partid_t>(k & m);
    }
}

// XOR_SHIFT kernel: out[i] = (key[i] ^ (key[i] >> 33)) & (P-1)
// 4 keys/iter. 1 shift + 1 xor + 1 AND + narrow. Very cheap.
static void kernel_avx2_xor_shift(const pkey_t* __restrict__ keys,
                                  partid_t* __restrict__ out,
                                  size_t N, uint32_t P)
{
    const __m256i vmask = _mm256_set1_epi64x(static_cast<int64_t>(P - 1));
    const size_t vec_end = N & ~3ULL;
    size_t i = 0;

    for (; i < vec_end; i += 4) {
        __m256i k = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));
        k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 33));
        __m256i vr = _mm256_and_si256(k, vmask);
        __m128i n = avx2_narrow64_to32(vr);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), n);
    }
    const uint64_t m = P - 1;
    for (; i < N; i++) {
        uint64_t k = keys[i];
        k ^= k >> 33;
        out[i] = static_cast<partid_t>(k & m);
    }
}

// FOLD32_MULT kernel — THE 32-BIT FAST PATH
//   1. Fold 64→32 via XOR:  k32 = (uint32_t)(key ^ (key >> 32))
//   2. 32-bit multiply-shift: (k32 * PRIME32) >> shift

static void kernel_avx2_fold32_mult(const pkey_t* __restrict__ keys,
                                    partid_t* __restrict__ out,
                                    size_t N, uint32_t P)
{
    const __m256i vprime = _mm256_set1_epi32(static_cast<int32_t>(MULT_PRIME32));
    const int shift = 32 - log2_pot(P);
    const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);

    const size_t vec_end = N & ~7ULL;  // 8 keys per iteration!
    size_t i = 0;

    for (; i < vec_end; i += 8) {
        // Load 8 × uint64 in two 256-bit registers
        __m256i lo4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));
        __m256i hi4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i + 4]));

        // Fold 64→32: XOR high and low 32-bit halves
        lo4 = _mm256_xor_si256(lo4, _mm256_srli_epi64(lo4, 32));
        hi4 = _mm256_xor_si256(hi4, _mm256_srli_epi64(hi4, 32));

        // Pack: extract low 32 bits of each 64-bit lane
        __m256i lo4_packed = _mm256_permutevar8x32_epi32(lo4, perm_idx);
        __m256i hi4_packed = _mm256_permutevar8x32_epi32(hi4, perm_idx);

        // Combine into one 256-bit register: [k0..k3 | k4..k7]
        __m256i packed = _mm256_inserti128_si256(
            lo4_packed,
            _mm256_castsi256_si128(hi4_packed),
            1
        );

        // 32-bit multiply — NATIVE, single instruction, 8-wide!
        __m256i prod = _mm256_mullo_epi32(packed, vprime);
        __m256i result = _mm256_srli_epi32(prod, shift);

        // Store 8 × uint32
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), result);
    }

    // Scalar tail (0–7 elements)
    for (; i < N; i++) {
        uint32_t k = static_cast<uint32_t>(keys[i] ^ (keys[i] >> 32));
        out[i] = static_cast<partid_t>((k * MULT_PRIME32) >> shift);
    }
}


// AVX2 dispatch
static kernel_fn_t get_kernel_avx2(HashType h) {
    switch (h) {
        case HashType::MASK:            return kernel_avx2_mask;
        case HashType::MULTIPLY_SHIFT:  return kernel_avx2_multiply_shift;
        case HashType::MURMUR_FMIX64:   return kernel_avx2_murmur;
        case HashType::XOR_SHIFT:       return kernel_avx2_xor_shift;
        case HashType::FOLD32_MULT:     return kernel_avx2_fold32_mult;
        default:                        return nullptr;
    }
}


// CORRECTNESS VERIFICATION: AVX2 vs Plain cross-check
static bool run_verification()
{
    std::cout << "===== AVX2 vs PLAIN CORRECTNESS CHECK =====\n\n";

    const uint64_t seed = 42;
    int total = 0, passed = 0, failed = 0;

    std::vector<size_t>   test_Ns = {1, 2, 3, 4, 5, 7, 15, 16, 17, 33, 100, 1000, 10000, 100000, 1000000};
    std::vector<uint32_t> test_Ps = {2, 4, 16, 64, 256, 1024, 4096};
    std::vector<HashType> hashes  = {
        HashType::MASK,
        HashType::MULTIPLY_SHIFT,
        HashType::MURMUR_FMIX64,
        HashType::XOR_SHIFT,
        HashType::FOLD32_MULT
    };

    size_t max_N = *std::max_element(test_Ns.begin(), test_Ns.end());
    pkey_t*   keys    = static_cast<pkey_t*>(aligned_alloc(64, max_N * sizeof(pkey_t)));
    partid_t* ref_out = static_cast<partid_t*>(aligned_alloc(64, max_N * sizeof(partid_t)));
    partid_t* avx_out = static_cast<partid_t*>(aligned_alloc(64, max_N * sizeof(partid_t)));

    // Test 1: random keys
    std::cout << "Test 1: Random keys...\n";
    for (size_t N : test_Ns) {
        generate_keys(keys, N, seed);
        for (uint32_t P : test_Ps) {
            for (HashType h : hashes) {
                total++;
                get_kernel(h)(keys, ref_out, N, P);
                get_kernel_avx2(h)(keys, avx_out, N, P);

                uint64_t ck_ref = compute_checksum(ref_out, N);
                uint64_t ck_avx = compute_checksum(avx_out, N);
                bool ok = (ck_ref == ck_avx);

                if (N <= 10000 && ok) {
                    if (compare_arrays(ref_out, avx_out, N, 3) > 0) ok = false;
                }
                for (size_t i = 0; i < N && ok; i++) {
                    if (avx_out[i] >= P) ok = false;
                }

                if (ok) passed++;
                else {
                    failed++;
                    std::cerr << "FAIL: N=" << N << " P=" << P
                              << " hash=" << hash_name(h) << "\n";
                }
            }
        }
    }

    // Test 2: duplicate keys
    std::cout << "Test 2: Duplicate keys...\n";
    size_t N_dup = 100000;
    for (uint64_t ks : {100000ULL, 10000ULL, 1000ULL, 100ULL}) {
        generate_keys(keys, N_dup, seed, ks);
        for (uint32_t P : {16u, 256u, 4096u}) {
            for (HashType h : hashes) {
                total++;
                get_kernel(h)(keys, ref_out, N_dup, P);
                get_kernel_avx2(h)(keys, avx_out, N_dup, P);
                bool ok = (compute_checksum(ref_out, N_dup) == compute_checksum(avx_out, N_dup));
                if (ok) passed++;
                else { failed++; std::cerr << "FAIL: dup ks=" << ks << "\n"; }
            }
        }
    }

    // Test 3: adversarial keys
    std::cout << "Test 3: Adversarial inputs...\n";
    size_t N_adv = 10000;

    for (size_t i = 0; i < N_adv; i++) keys[i] = i;  // sequential
    for (uint32_t P : {16u, 256u}) {
        for (HashType h : hashes) {
            total++;
            get_kernel(h)(keys, ref_out, N_adv, P);
            get_kernel_avx2(h)(keys, avx_out, N_adv, P);
            if (compare_arrays(ref_out, avx_out, N_adv, 3) == 0) passed++;
            else { failed++; std::cerr << "FAIL: sequential\n"; }
        }
    }

    for (size_t i = 0; i < N_adv; i++) keys[i] = i * 256;  // aligned the low 8 bits are zero (bad for fold32_mult)
    for (uint32_t P : {16u, 256u}) {
        for (HashType h : hashes) {
            total++;
            get_kernel(h)(keys, ref_out, N_adv, P);
            get_kernel_avx2(h)(keys, avx_out, N_adv, P);
            if (compare_arrays(ref_out, avx_out, N_adv, 3) == 0) passed++;
            else { failed++; std::cerr << "FAIL: aligned\n"; }
        }
    }

    for (size_t i = 0; i < N_adv; i++) keys[i] = 0xDEADBEEFCAFEBABEULL;  // all-same
    for (uint32_t P : {16u, 256u}) {
        for (HashType h : hashes) {
            total++;
            get_kernel(h)(keys, ref_out, N_adv, P);
            get_kernel_avx2(h)(keys, avx_out, N_adv, P);
            if (compare_arrays(ref_out, avx_out, N_adv, 3) == 0) passed++;
            else { failed++; std::cerr << "FAIL: all-same\n"; }
        }
    }

    // Sample output
    std::cout << "\n--- Sample (N=16, P=4) ---\n";
    generate_keys(keys, 16, seed);
    for (HashType h : hashes) {
        get_kernel(h)(keys, ref_out, 16, 4);
        get_kernel_avx2(h)(keys, avx_out, 16, 4);
        std::cout << std::setw(18) << std::left << hash_name(h) << std::right;
        std::cout << " plain: [";
        for (int j = 0; j < 16; j++) std::cout << ref_out[j] << (j < 15 ? "," : "");
        std::cout << "]\n" << std::setw(18) << "" << " avx2:  [";
        for (int j = 0; j < 16; j++) std::cout << avx_out[j] << (j < 15 ? "," : "");
        std::cout << "]\n";
    }

    std::cout << "\nTotal: " << total << "  Passed: " << passed
              << "  Failed: " << failed << "\n\n";

    free(keys); free(ref_out); free(avx_out);
    return (failed == 0);
}

static void emit(std::ostream* file, const std::string& line)
{
    std::cout << line;
    if (file) *file << line;
}


static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --N n1,n2,...    Dataset sizes (default: 1000000,10000000,50000000,100000000)\n"
        << "  --P p1,p2,...    Partition counts (default: 16,64,256,1024,4096)\n"
        << "  --hash h1,...    Hash functions\n"
        << "  --reps R         Repetitions (default: 15)\n"
        << "  --seed S         RNG seed (default: 42)\n"
        << "  --key-space KS   Key space for duplicates\n"
        << "  --out FILE       Save table output to .txt file\n"
        << "  --label LABEL    Label shown in table header (default: avx2)\n"
        << "  --skip-verify    Skip correctness check\n"
        << "  --help\n";
}

int main(int argc, char** argv)
{
    std::vector<size_t>   Ns = {10000000, 50000000, 100000000};
    std::vector<uint32_t> Ps = {16, 64, 256, 1024, 4096};
    std::vector<HashType> hashes = {
        HashType::MASK,
        HashType::MULTIPLY_SHIFT,
        HashType::MURMUR_FMIX64,
        HashType::XOR_SHIFT,
        HashType::FOLD32_MULT
    };

    int reps = 15;
    uint64_t seed = 42;
    uint64_t key_space = 0;
    std::string out_path;
    std::string label = "avx2";
    bool skip_verify = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        auto split = [](const std::string& s) {
            std::vector<std::string> v;
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) v.push_back(tok);
            return v;
        };

        if (a == "--N") {
            Ns.clear(); for (auto& t : split(next())) Ns.push_back(std::stoull(t));
        } else if (a == "--P") {
            Ps.clear(); for (auto& t : split(next())) Ps.push_back(std::stoul(t));
        } else if (a == "--hash") {
            hashes.clear();
            for (auto& t : split(next())) {
                if      (t == "mask")            hashes.push_back(HashType::MASK);
                else if (t == "multiply_shift")  hashes.push_back(HashType::MULTIPLY_SHIFT);
                else if (t == "murmur_fmix64")   hashes.push_back(HashType::MURMUR_FMIX64);
                else if (t == "xor_shift")       hashes.push_back(HashType::XOR_SHIFT);
                else if (t == "fold32_mult")     hashes.push_back(HashType::FOLD32_MULT);
                else { std::cerr << "Unknown hash: " << t << "\n"; return 1; }
            }
        } else if (a == "--reps") { reps = std::stoi(next()); }
          else if (a == "--seed") { seed = std::stoull(next()); }
          else if (a == "--key-space") { key_space = std::stoull(next()); }
          else if (a == "--out") { out_path = next(); }
          else if (a == "--label") { label = next(); }
          else if (a == "--skip-verify") { skip_verify = true; }
          else if (a == "--help") { usage(argv[0]); return 0; }
          else { std::cerr << "Unknown option: " << a << "\n"; usage(argv[0]); return 1; }
    }

    // Correctness
    if (!skip_verify) {
        if (!run_verification()) {
            std::cerr << "AVX2 VERIFICATION FAILED — aborting.\n";
            return 1;
        }
    }

    std::ofstream out_fs;
    std::ostream* out_file = nullptr;
    if (!out_path.empty()) {
        out_fs.open(out_path);
        if (!out_fs) { std::cerr << "Cannot open output file: " << out_path << "\n"; return 1; }
        out_file = &out_fs;
    }

    // Benchmark
    {
        std::ostringstream hdr;
        hdr << "===== BENCHMARK [" << label << "] =====\n";
        hdr << "Reps: " << reps << "  Seed: " << seed;
        if (key_space > 0) hdr << "  key_space: " << key_space;
        hdr << "\n\n";
        emit(out_file, hdr.str());
    }

    // Allocate buffers
    size_t max_N = *std::max_element(Ns.begin(), Ns.end());
    pkey_t*   keys = static_cast<pkey_t*>(aligned_alloc(64, max_N * sizeof(pkey_t)));
    partid_t* out  = static_cast<partid_t*>(aligned_alloc(64, max_N * sizeof(partid_t)));
    if (!keys || !out) { std::cerr << "Allocation failed\n"; return 1; }

    // Benchmark loop
    for (size_t N : Ns) {
        generate_keys(keys, N, seed, key_space);

        std::ostringstream block;
        block << "┌──────────────────────────────────────────────────────────────────────────────────┐\n";
        block << "│  N = " << std::setw(10) << N << " (" << std::fixed << std::setprecision(0)
              << N / 1e6 << "M)    label = " << label
              << std::string(std::max(0, 25 - (int)label.size()), ' ') << "│\n";
        block << "├───────┬──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┤\n";
        block << "│   P   │ hash             │ med(ms)  │ Mkeys/s  │ sd(ms)   │   chi2   │    CV    │\n";
        block << "├───────┼──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n";
        emit(out_file, block.str());

        for (uint32_t P : Ps) {
            for (HashType h : hashes) {
                kernel_fn_t fn = get_kernel_avx2(h);
                auto task = [&]() { fn(keys, out, N, P); };
                BenchResult r = run_bench(task, N, reps);

                fn(keys, out, N, P);
                DistStats ds = distribution_stats(out, N, P);

                std::ostringstream row;
                row << "│" << std::setw(6) << P << " │ "
                    << std::setw(16) << std::left << hash_name(h) << std::right
                    << " │" << std::fixed << std::setprecision(2) << std::setw(8) << r.median_ms
                    << "  │" << std::setprecision(1) << std::setw(8) << r.throughput_Mkeys_s
                    << "  │" << std::setprecision(3) << std::setw(8) << r.stddev_ms
                    << "  │" << std::setprecision(1) << std::setw(8) << ds.chi2
                    << "  │" << std::setprecision(6) << std::setw(8) << ds.cv
                    << "  │\n";
                emit(out_file, row.str());
            }

            if (P != Ps.back()) {
                emit(out_file,
                    "├───────┼──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n");
            }
        }

        emit(out_file,
            "└───────┴──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n\n");
    }

    free(keys); free(out);

    if (out_file) {
        std::cout << "\nTable saved to: " << out_path << "\n";
        out_fs.close();
    }

    return 0;
}