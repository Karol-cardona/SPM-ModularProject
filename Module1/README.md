# Module 1 — Partition Mapping Benchmark

Comparative benchmark of five hash functions for **64-bit key partition mapping**,
implemented at three levels of progressive optimization:

| Task | Binaries | Technology |
|------|----------|------------|
| 1 | `task1_novec` / `task1_autovec` | Plain C++, scalar vs GCC auto-vectorized |
| 2 | `task2_avx2` | Hand-written AVX2 intrinsics |
| 3 | `task3_cuda` | CUDA GPU kernel |

---

## Requirements

- **GCC ≥ 9** with C++17 support, `-march=native` (AVX2 required for task2)
- **CUDA 12.3** (`/usr/local/cuda-12.3`) — GPU with compute capability ≥ 8.0 (sm_80)
- **numactl** (optional, used in `run.sh` for NUMA pinning)
- Linux system with `aligned_alloc` available (glibc ≥ 2.16)

---

## Running the Benchmarks

### Task 1 — Plain C++

```bash
cd ~/Module1
make task1

# Scalar baseline (no vectorization)
numactl --cpunodebind=0 --membind=0 \
  ./bin/task1_novec --label novec --out results/novec.txt

# GCC auto-vectorized
numactl --cpunodebind=0 --membind=0 \
  ./bin/task1_autovec --label autovec --out results/autovec.txt
```

### Task 2 — AVX2

```bash
cd ~/Module1
make task2

numactl --cpunodebind=0 --membind=0 \
  ./bin/task2_avx2 --label avx2 --out results/avx2.txt
```

### Task 3 — CUDA

```bash
srun --partition=gpu-excl --nodelist=node09 --ntasks=1 --cpus-per-task=2 --pty bash

# Once inside the interactive session:
cd ~/Module1
make task3
./bin/task3_cuda --out results/cuda.txt
```

### Run Everything via SLURM
```bash
sbatch run.sh
```
---

## CLI Options

All binaries (task1, task2, task3) share these options:

| Option | Default | Description |
|--------|---------|-------------|
| `--N n1,n2,...` | 10M,50M,100M | Dataset sizes |
| `--P p1,p2,...` | 16,64,256,1024,4096 | Partition counts (must be powers of 2) |
| `--hash h1,...` | all | Hash functions to run |
| `--reps R` | 15 (10 for CUDA) | Benchmark repetitions |
| `--seed S` | 42 | splitmix64 RNG seed |
| `--key-space KS` | 0 (full 64-bit) | Key space size (0 = full) |
| `--out FILE` | — | Save result table to file |
| `--label LABEL` | plain | Label shown in result table |
| `--skip-verify` | — | Skip correctness check |

---

## Summary Tables

```bash
bash summary.sh
```

Produces 5 tables from the files in `results/`:

| Table | Content |
|-------|---------|
| 1 | Throughput (Mkeys/s) for all versions at N=100M, P=256 |
| 2 | Speedup over `novec` baseline |
| 3 | Effect of vectorization across N values |
| 4 | Distribution quality (chi², CV) |
| 5 | CUDA timing breakdown (H2D / kernel / D2H) |

---

## Hash Functions

| Name | Formula | Notes |
|------|----------|-------|
| `mask` | `key & (P-1)` | Baseline, works well only with uniform keys |
| `multiply_shift` | `(key * PRIME64) >> (64 - log2P)` | Dietzfelbinger universal hashing |
| `murmur_fmix64` | fmix64 finalizer from MurmurHash3 | Best distribution quality, highest cost |
| `xor_shift` | `(key ^ (key >> 33)) & (P-1)` | Lightweight, better distribution than mask |
| `fold32_mult` | fold 64→32 via XOR, then 32-bit multiply-shift | Uses native `mullo_epi32` in AVX2 |

Constant used: `PRIME64 = 0x9E3779B97F4A7C15` (2⁶⁴/φ, golden ratio)

---

## Implementation Details

### AVX2 (task2)

AVX2 has no native 64×64→64 multiplication. The emulation used is:

```
result = lo_lo + ((a_lo*b_hi + a_hi*b_lo) << 32)
```

Cost: 3× `mul_epu32` + 2× `add_epi64` + shift. The `fold32_mult` kernel avoids
this overhead by using native `_mm256_mullo_epi32`, processing 8 keys per
iteration instead of 4.

### CUDA (task3)

- Block size: 256 threads by default
- Grid size: `ceil(N / block_size)`
- Separate timing via `cudaEvent` for H2D, kernel, and D2H
- CPU vs GPU correctness check before benchmarking
- Target architecture: sm_80 (Ampere)