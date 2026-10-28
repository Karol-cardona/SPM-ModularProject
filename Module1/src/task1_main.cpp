#include "common.h"
#include "kernels_plain.h"

// ============================================================
// Computes expected output element by element using only standard C++ arithmetic.
// This is the "ground truth" used by the verifier.
// ============================================================
static void compute_reference(const pkey_t* keys, partid_t* ref,
                              size_t N, uint32_t P, HashType h)
{
    const uint64_t mask    = static_cast<uint64_t>(P) - 1;
    const int      shift64 = 64 - log2_pot(P);
    const int      shift32 = 32 - log2_pot(P);

    for (size_t i = 0; i < N; i++) {
        uint64_t k = keys[i];
        switch (h) {
            case HashType::MASK:
                ref[i] = static_cast<partid_t>(k & mask);
                break;
            case HashType::MULTIPLY_SHIFT:
                ref[i] = static_cast<partid_t>((k * MULT_PRIME) >> shift64);
                break;
            case HashType::MURMUR_FMIX64:
                k ^= k >> 33; k *= MURMUR_C1;
                k ^= k >> 33; k *= MURMUR_C2;
                k ^= k >> 33;
                ref[i] = static_cast<partid_t>(k & mask);
                break;
            case HashType::XOR_SHIFT:
                k ^= k >> 33;
                ref[i] = static_cast<partid_t>(k & mask);
                break;
            case HashType::FOLD32_MULT: {
                uint32_t f = static_cast<uint32_t>(k ^ (k >> 32));
                ref[i] = static_cast<partid_t>((f * MULT_PRIME32) >> shift32);
                break;
            }
            default: break;
        }
    }
}
// ============================================================
// Correctness verification
// ============================================================
static bool run_verification()
{
    std::cout << "===== CORRECTNESS VERIFICATION =====\n\n";

    const uint64_t seed = 42;
    int total = 0;
    int passed = 0;
    int failed = 0;

    std::vector<size_t>   test_Ns = {1, 3, 4, 7, 16, 17, 100, 1000, 10000, 100000};
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
    partid_t* out_ker = static_cast<partid_t*>(aligned_alloc(64, max_N * sizeof(partid_t)));
    partid_t* out_ref = static_cast<partid_t*>(aligned_alloc(64, max_N * sizeof(partid_t)));

    for (size_t N : test_Ns) {
        generate_keys(keys, N, seed);

        for (uint32_t P : test_Ps) {
            for (HashType h : hashes) {
                total++;
                kernel_fn_t fn = get_kernel(h);

                // Run kernel under test
                fn(keys, out_ker, N, P);

                // Run independent reference
                compute_reference(keys, out_ref, N, P, h);

                // Checksum comparison
                uint64_t ck_ker = compute_checksum(out_ker, N);
                uint64_t ck_ref = compute_checksum(out_ref, N);
                bool ok = (ck_ker == ck_ref);

                // Element-by-element comparison
                if (N <= 10000) {
                    size_t mis = compare_arrays(out_ref, out_ker, N, 3);
                    if (mis > 0) ok = false;
                }

                // Bounds check: every id must be in [0, P)
                for (size_t i = 0; i < N && ok; i++) {
                    if (out_ker[i] >= P) {
                        std::cerr << "  bounds violation [" << i << "]: "
                                  << out_ker[i] << " >= P=" << P << "\n";
                        ok = false;
                    }
                }

                if (ok) {
                    passed++;
                }else {
                    failed++;
                    std::cerr << "FAIL: N=" << N << " P=" << P
                              << " hash=" << hash_name(h) << "\n";
                }
            }
        }
    }

    std::cout << "Total: " << total << "  Passed: " << passed
              << "  Failed: " << failed << "\n\n";

    free(keys); free(out_ker); free(out_ref);
    return (failed == 0);
}


static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --N n1,n2,...    Dataset sizes (default: 10000000,50000000,100000000)\n"
        << "  --P p1,p2,...    Partition counts (default: 16,64,256,1024,4096)\n"
        << "  --hash h1,...    Hash functions\n"
        << "  --reps R         Repetitions (default: 15)\n"
        << "  --seed S         RNG seed (default: 42)\n"
        << "  --key-space KS   Key space for duplicates (0 = full 64-bit)\n"
        << "  --out FILE       Save table output to .txt file\n"
        << "  --label LABEL    Label shown in table header (default: plain)\n"
        << "  --skip-verify    Skip correctness check\n"
        << "  --help\n";
}

static void emit(std::ostream* file, const std::string& line)
{
    std::cout << line;
    if (file) *file << line;
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
    std::string label = "plain";
    bool skip_verify = false;

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
        } else if (a == "--reps")        { reps = std::stoi(next()); }
          else if (a == "--seed")        { seed = std::stoull(next()); }
          else if (a == "--key-space")   { key_space = std::stoull(next()); }
          else if (a == "--out")         { out_path = next(); }
          else if (a == "--label")       { label = next(); }
          else if (a == "--skip-verify") { skip_verify = true; }
          else if (a == "--help")        { usage(argv[0]); return 0; }
          else { std::cerr << "Unknown option: " << a << "\n"; usage(argv[0]); return 1; }
    }

    if (!skip_verify) {
        if (!run_verification()) { std::cerr << "VERIFICATION FAILED\n"; return 1; }
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
               kernel_fn_t fn = get_kernel(h);
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
