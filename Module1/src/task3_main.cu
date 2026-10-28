#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <functional>
#include <cassert>


using pkey_t   = uint64_t;
using partid_t = uint32_t;


// Hash constants (same as CPU)
static constexpr uint64_t MULT_PRIME   = 0x9E3779B97F4A7C15ULL;
static constexpr uint64_t MURMUR_C1    = 0xFF51AFD7ED558CCDULL;
static constexpr uint64_t MURMUR_C2    = 0xC4CEB9FE1A85EC53ULL;
static constexpr uint32_t MULT_PRIME32 = 0x9E3779B9u;

enum class HashType : int {
    MASK = 0,
    MULTIPLY_SHIFT,
    MURMUR_FMIX64,
    XOR_SHIFT,
    FOLD32_MULT,
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

static inline int log2_pot(uint64_t P) { return __builtin_ctzll(P); }


// Key generation
static void generate_keys(pkey_t* keys, size_t N, uint64_t seed) {
    uint64_t state = seed;
    for (size_t i = 0; i < N; i++) {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        keys[i] = z;
    }
}


// CPU reference kernels (for correctness check)
static void cpu_reference(const pkey_t* keys, partid_t* out,
                          size_t N, uint32_t P, HashType h)
{
    const uint64_t mask = P - 1;
    const int shift64 = 64 - log2_pot(P);
    const int shift32 = 32 - log2_pot(P);

    for (size_t i = 0; i < N; i++) {
        uint64_t k = keys[i];
        switch (h) {
            case HashType::MASK:
                out[i] = static_cast<partid_t>(k & mask);
                break;
            case HashType::MULTIPLY_SHIFT:
                out[i] = static_cast<partid_t>((k * MULT_PRIME) >> shift64);
                break;
            case HashType::MURMUR_FMIX64:
                k ^= k >> 33; k *= MURMUR_C1;
                k ^= k >> 33; k *= MURMUR_C2;
                k ^= k >> 33;
                out[i] = static_cast<partid_t>(k & mask);
                break;
            case HashType::XOR_SHIFT:
                k ^= k >> 33;
                out[i] = static_cast<partid_t>(k & mask);
                break;
            case HashType::FOLD32_MULT: {
                uint32_t f = static_cast<uint32_t>(k ^ (k >> 32));
                out[i] = static_cast<partid_t>((f * MULT_PRIME32) >> shift32);
                break;
            }
            default: break;
        }
    }
}

// Checksum
static uint64_t compute_checksum(const partid_t* ids, size_t N) {
    uint64_t h = 0xCBF29CE484222325ULL;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(ids);
    size_t bytes = N * sizeof(partid_t);
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 0x100000001B3ULL;
    }
    return h;
}


// CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)



// CUDA KERNELS — one per hash function
__global__ void kernel_cuda_mask(const pkey_t* __restrict__ keys,
                                 partid_t* __restrict__ out,
                                 size_t N, uint64_t mask)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = static_cast<partid_t>(keys[i] & mask);
    }
}

__global__ void kernel_cuda_multiply_shift(const pkey_t* __restrict__ keys,
                                           partid_t* __restrict__ out,
                                           size_t N, int shift)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = static_cast<partid_t>((keys[i] * MULT_PRIME) >> shift);
    }
}

__global__ void kernel_cuda_murmur(const pkey_t* __restrict__ keys,
                                   partid_t* __restrict__ out,
                                   size_t N, uint64_t mask)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        uint64_t k = keys[i];
        k ^= k >> 33; k *= MURMUR_C1;
        k ^= k >> 33; k *= MURMUR_C2;
        k ^= k >> 33;
        out[i] = static_cast<partid_t>(k & mask);
    }
}

__global__ void kernel_cuda_xor_shift(const pkey_t* __restrict__ keys,
                                      partid_t* __restrict__ out,
                                      size_t N, uint64_t mask)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        uint64_t k = keys[i];
        k ^= k >> 33;
        out[i] = static_cast<partid_t>(k & mask);
    }
}

__global__ void kernel_cuda_fold32_mult(const pkey_t* __restrict__ keys,
                                        partid_t* __restrict__ out,
                                        size_t N, int shift)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        uint64_t k = keys[i];
        uint32_t f = static_cast<uint32_t>(k ^ (k >> 32));
        out[i] = static_cast<partid_t>((f * MULT_PRIME32) >> shift);
    }
}


struct CudaTimings {
    float h2d_ms;       // host to device transfer
    float kernel_ms;    // kernel execution only
    float d2h_ms;       // device to host transfer
    float total_ms;     // h2d + kernel + d2h
};

static CudaTimings run_cuda_kernel(const pkey_t* h_keys, partid_t* h_out,
                                   size_t N, uint32_t P, HashType hash,
                                   int block_size = 256)
{
    size_t keys_bytes = N * sizeof(pkey_t);
    size_t out_bytes  = N * sizeof(partid_t);

    // Device allocations
    pkey_t*   d_keys;
    partid_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_keys, keys_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    // Events for timing
    cudaEvent_t start, after_h2d, after_kernel, after_d2h;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&after_h2d));
    CUDA_CHECK(cudaEventCreate(&after_kernel));
    CUDA_CHECK(cudaEventCreate(&after_d2h));

    int grid_size = (N + block_size - 1) / block_size;

    // Precompute parameters
    uint64_t mask    = P - 1;
    int      shift64 = 64 - log2_pot(P);
    int      shift32 = 32 - log2_pot(P);

    // Timed execution
    CUDA_CHECK(cudaEventRecord(start));

    // H2D
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, keys_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(after_h2d));

    // Kernel
    switch (hash) {
        case HashType::MASK:
            kernel_cuda_mask<<<grid_size, block_size>>>(d_keys, d_out, N, mask);
            break;
        case HashType::MULTIPLY_SHIFT:
            kernel_cuda_multiply_shift<<<grid_size, block_size>>>(d_keys, d_out, N, shift64);
            break;
        case HashType::MURMUR_FMIX64:
            kernel_cuda_murmur<<<grid_size, block_size>>>(d_keys, d_out, N, mask);
            break;
       case HashType::XOR_SHIFT:
            kernel_cuda_xor_shift<<<grid_size, block_size>>>(d_keys, d_out, N, mask);
            break;
        case HashType::FOLD32_MULT:
            kernel_cuda_fold32_mult<<<grid_size, block_size>>>(d_keys, d_out, N, shift32);
            break;
        default: break;
    }
    CUDA_CHECK(cudaEventRecord(after_kernel));

    // D2H
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(after_d2h));

    CUDA_CHECK(cudaEventSynchronize(after_d2h));

    // Extract timings
    CudaTimings t;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, start, after_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernel_ms, after_h2d, after_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, after_kernel, after_d2h));
    t.total_ms = t.h2d_ms + t.kernel_ms + t.d2h_ms;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(after_h2d));
    CUDA_CHECK(cudaEventDestroy(after_kernel));
    CUDA_CHECK(cudaEventDestroy(after_d2h));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_out));

    return t;
}

// Benchmark: multiple runs, median timing
struct CudaBenchResult {
    float h2d_ms;
    float kernel_ms;
    float d2h_ms;
    float total_ms;
    float throughput_total;   // Mkeys/s including transfers
    float throughput_kernel;  // Mkeys/s kernel only
};

static CudaBenchResult benchmark_cuda(const pkey_t* h_keys, partid_t* h_out,
                                      size_t N, uint32_t P, HashType hash,
                                      int reps = 10, int warmup = 2)
{
    // Warmup
    for (int i = 0; i < warmup; i++) {
        run_cuda_kernel(h_keys, h_out, N, P, hash);
    }

    std::vector<CudaTimings> runs(reps);
    for (int i = 0; i < reps; i++) {
        runs[i] = run_cuda_kernel(h_keys, h_out, N, P, hash);
    }

    // Sort by total time, take median
    std::sort(runs.begin(), runs.end(),
              [](const CudaTimings& a, const CudaTimings& b) {
                  return a.total_ms < b.total_ms;
              });

    CudaTimings med = runs[reps / 2];
    CudaBenchResult r;
    r.h2d_ms    = med.h2d_ms;
    r.kernel_ms = med.kernel_ms;
    r.d2h_ms    = med.d2h_ms;
    r.total_ms  = med.total_ms;
    r.throughput_total  = (N / 1e6) / (med.total_ms / 1e3);
    r.throughput_kernel = (N / 1e6) / (med.kernel_ms / 1e3);

    return r;
}


// Correctness verification: CUDA vs CPU reference
static bool run_verification(pkey_t* h_keys, size_t max_N)
{
    std::cout << "===== CUDA vs CPU CORRECTNESS CHECK =====\n\n";

    int total = 0, passed = 0, failed = 0;

    std::vector<size_t>   test_Ns = {1, 4, 7, 16, 100, 1000, 10000, 100000};
    std::vector<uint32_t> test_Ps = {4, 16, 256, 4096};
    std::vector<HashType> hashes  = {
        HashType::MASK,
        HashType::MULTIPLY_SHIFT,
        HashType::MURMUR_FMIX64,
        HashType::XOR_SHIFT,
        HashType::FOLD32_MULT
    };

    partid_t* cpu_out  = new partid_t[max_N];
    partid_t* cuda_out = new partid_t[max_N];

    for (size_t N : test_Ns) {
        generate_keys(h_keys, N, 42);
        for (uint32_t P : test_Ps) {
            for (HashType h : hashes) {
                total++;
                cpu_reference(h_keys, cpu_out, N, P, h);
                run_cuda_kernel(h_keys, cuda_out, N, P, h);

                uint64_t ck_cpu  = compute_checksum(cpu_out, N);
                uint64_t ck_cuda = compute_checksum(cuda_out, N);

                bool ok = (ck_cpu == ck_cuda);

                // Element check for small N
                if (N <= 10000 && ok) {
                    for (size_t i = 0; i < N; i++) {
                        if (cpu_out[i] != cuda_out[i]) {
                            ok = false;
                            if (failed < 5) {
                                fprintf(stderr, "  mismatch [%zu]: cpu=%u cuda=%u\n",
                                        i, cpu_out[i], cuda_out[i]);
                            }
                        }
                    }
                }

                if (ok) passed++;
                else {
                    failed++;
                    fprintf(stderr, "FAIL: N=%zu P=%u hash=%s\n",
                            N, P, hash_name(h));
                }
            }
        }
    }

    // Sample output
    std::cout << "\n--- Sample (N=16, P=4) ---\n";
    generate_keys(h_keys, 16, 42);
    for (HashType h : hashes) {
        cpu_reference(h_keys, cpu_out, 16, 4, h);
        run_cuda_kernel(h_keys, cuda_out, 16, 4, h);
        std::cout << std::setw(18) << std::left << hash_name(h) << std::right;
        std::cout << " cpu:  [";
        for (int i = 0; i < 16; i++) std::cout << cpu_out[i] << (i < 15 ? "," : "");
        std::cout << "]\n" << std::setw(18) << "" << " cuda: [";
        for (int i = 0; i < 16; i++) std::cout << cuda_out[i] << (i < 15 ? "," : "");
        std::cout << "]\n";
    }

    std::cout << "\nTotal: " << total << "  Passed: " << passed
              << "  Failed: " << failed << "\n\n";

    delete[] cpu_out;
    delete[] cuda_out;
    return (failed == 0);
}

static void emit(std::ostream* file, const std::string& line)
{
    std::cout << line;
    if (file) *file << line;
}


static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --N n1,n2,...    Dataset sizes (default: 10000000,50000000,100000000)\n"
        "  --P p1,p2,...    Partition counts (default: 16,64,256,1024,4096)\n"
        "  --reps R         Repetitions (default: 10)\n"
        "  --out FILE       Save table output to .txt file\n"
        "  --skip-verify    Skip correctness check\n"
        "  --help\n", prog);
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

    int      reps       = 10;
    uint64_t seed       = 42;
    std::string out_path;
    bool skip_verify    = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", a.c_str()); exit(1); }
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
        } else if (a == "--reps")        { reps = std::stoi(next()); }
          else if (a == "--out")         { out_path  = next(); }
          else if (a == "--skip-verify") { skip_verify = true; }
          else if (a == "--help")        { usage(argv[0]); return 0; }
          else { fprintf(stderr, "Unknown option: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d, %d SMs, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    // Allocate host memory
    size_t max_N = *std::max_element(Ns.begin(), Ns.end());
    pkey_t*   h_keys = static_cast<pkey_t*>(aligned_alloc(64, max_N * sizeof(pkey_t)));
    partid_t* h_out  = static_cast<partid_t*>(aligned_alloc(64, max_N * sizeof(partid_t)));

    // Correctness
    if (!skip_verify) {
        if (!run_verification(h_keys, std::max(max_N, (size_t)100000))) {
            fprintf(stderr, "CUDA VERIFICATION FAILED\n");
            return 1;
        }
    }

    std::ofstream out_fs;
    std::ostream* out_file = nullptr;
    if (!out_path.empty()) {
        out_fs.open(out_path);
        if (!out_fs) { fprintf(stderr, "Cannot open output file: %s\n", out_path.c_str()); return 1; }
        out_file = &out_fs;
    }

    // Benchmark
    {
        std::ostringstream hdr;
        hdr << "===== BENCHMARK [cuda] =====\n";
        hdr << "GPU: " << prop.name << "  (SM " << prop.major << "." << prop.minor
            << ", " << prop.multiProcessorCount << " SMs)\n";
        hdr << "Reps: " << reps << "  Seed: " << seed << "\n\n";
        emit(out_file, hdr.str());
    }

    // Benchmark loop
    for (size_t N : Ns) {
        generate_keys(h_keys, N, seed);

        std::ostringstream block;
        block << "┌──────────────────────────────────────────────────────────────────────────────────────┐\n";
        block << "│  N = " << std::setw(10) << N << " (" << std::fixed << std::setprecision(0)
              << N / 1e6 << "M)    GPU = " << prop.name
              << std::string(std::max(0, 27 - (int)strlen(prop.name)), ' ') << "│\n";
        block << "├───────┬──────────────────┬─────────┬─────────┬─────────┬─────────┬──────────┬──────────┤\n";
        block << "│   P   │ hash             │ H2D(ms) │ ker(ms) │ D2H(ms) │ tot(ms) │ Mkeys/s  │ ker only │\n";
        block << "│       │                  │         │         │         │         │  (total) │ (Mkeys/s)│\n";
        block << "├───────┼──────────────────┼─────────┼─────────┼─────────┼─────────┼──────────┼──────────┤\n";
        emit(out_file, block.str());

        for (uint32_t P : Ps) {
            for (HashType h : hashes) {
                CudaBenchResult r = benchmark_cuda(h_keys, h_out, N, P, h, reps);

                std::ostringstream row;
                row << "│" << std::setw(6) << P << " │ "
                    << std::setw(16) << std::left << hash_name(h) << std::right
                    << " │" << std::fixed << std::setprecision(2)
                    << std::setw(7) << r.h2d_ms    << "  │"
                    << std::setw(7) << r.kernel_ms << "  │"
                    << std::setw(7) << r.d2h_ms    << "  │"
                    << std::setw(7) << r.total_ms  << "  │"
                    << std::setprecision(1)
                    << std::setw(8) << r.throughput_total  << "  │"
                    << std::setw(8) << r.throughput_kernel << "  │\n";
                emit(out_file, row.str());
            }

            if (P != Ps.back()) {
                emit(out_file,
                    "├───────┼──────────────────┼─────────┼─────────┼─────────┼─────────┼──────────┼──────────┤\n");
            }
        }

        emit(out_file,
            "└───────┴──────────────────┴─────────┴─────────┴─────────┴─────────┴──────────┴──────────┘\n\n");
    }

    free(h_keys);
    free(h_out);

    if (out_file) {
        std::cout << "\nTable saved to: " << out_path << "\n";
        out_fs.close();
    }

    return 0;
}
