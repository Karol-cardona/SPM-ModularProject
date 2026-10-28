# SPM Modular Project — Partitioned Hash Join with Duplicates

**Course:** Parallel and Distributed Systems: Paradigms and Models (SPM) — A.Y. 2025/2026
**Student:** Karol Jinet Cardona Bolaños — Mat. 702023
**Institution:** University of Pisa, MSc in Computer Science

---

## Overview

This repository contains the four modules of the SPM modular project. All modules implement
the same algorithm — a **partitioned hash join with duplicates** over two relations `R` and `S`
of 64-bit keys — under progressively richer execution models, from a single-core vectorized
kernel to a distributed-memory implementation.

The algorithm is structured in phases:

1. **Mapping** — each key is assigned a partition id `h(key) mod P`, with `P` a power of two.
2. **Partitioning** — histogram, exclusive prefix sum, scatter; records of the same partition
   become contiguous in memory.
3. **Local join** — per partition, a count structure is built over `R` and `S` is probed against it;
   since duplicates are allowed, each match contributes the multiplicity of the key in `R`.
4. **Reduction** — contributions of all partitions are accumulated.

The join pairs are never materialized. Every implementation produces exactly three values:

```
join_count
checksum1
checksum2
```

These three values are identical across all modules and all execution models, for the same input
data and the same mapping function. This invariant is the backbone of the correctness strategy.

---

## Repository structure

```
.
├── module1/        Vectorization of the partition mapping kernel (SIMD / AVX2 / CUDA)
├── module2/        Full join pipeline + parallel version with C++ threads
├── module3/        OpenMP: loop-level and task-based parallelism, uniform vs skewed workloads
└── module4/        Distributed implementation with MPI
```

Each module directory is self-contained and carries its own `README`, `Makefile`, benchmark
scripts and PDF report. Refer to the per-module README for the exact build targets, command-line
options and experiment reproduction steps.

---

## Modules

### Module 1 — Vectorization of the partition mapping kernel

Five hash functions spanning a cost/quality spectrum (`mask`, `xor_shift`, `multiply_shift`,
`murmur_fmix64`, `fold32_mult`) are compared across scalar, auto-vectorized, hand-written AVX2
and CUDA implementations. Hashing quality is assessed through Pearson's χ² and the coefficient
of variation across partitions; `P = 256` and `N = 10⁸` are selected as the reference
configuration.

*Key results:* hand-written AVX2 does not beat GCC auto-vectorization for any of the five
functions — memory bandwidth saturation, packing overhead and the cost of emulated 64-bit
multiplies dominate. `fold32_mult` benefits most from vectorization, since its 32-bit arithmetic
pairs with native 8-wide SIMD multiplies. On GPU, the kernel is roughly two orders of magnitude
faster than the best CPU version, but PCIe transfers account for ~99% of end-to-end time, so
offloading pays off only if the data already resides in device memory.

### Module 2 — Full pipeline and C++ threads

The complete join pipeline is implemented starting from the provided sequential reference
(`hashjoin_seq.cpp`), with the Module 1 mapping function substituted in. The improved sequential
version produced here is the **single baseline used by Modules 2, 3 and 4**, so that the
comparison is between execution models and not between different algorithms. The parallel version
uses C++ threads; the analysis focuses on which phases are worth parallelizing and which are not.

*Key results:* the hash table layout dominates. Open addressing (`FlatCountMap`) outperforms
chaining by a wide margin, as the comparison is effectively between an L1 cache hit and a DRAM
miss on the probe path. `alignas(64)` is used on per-thread state to prevent false sharing.

### Module 3 — OpenMP under uniform and skewed workloads

Two OpenMP implementations are provided — one based on **loop-level parallelism** and one based
on **tasks** — and evaluated under both a uniform dataset (the Module 2 generation scheme) and a
purpose-built **skewed dataset**, in which a large fraction of the keys is concentrated on a small
subset of partitions.

*Key results:* under skew, `schedule(static)` collapses, and `dynamic,4` and `guided` also fail
because geometric chunk sizing hands an oversized final chunk to a single thread; `dynamic,1`
restores balance. Note that `R` and `S` must share the skew `pool_seed` (fixed in `SkewConfig`)
rather than the relation seed, otherwise the skew is not reproducible across the two relations.

### Module 4 — Distributed implementation with MPI

The computation is redistributed so that each MPI rank owns the data required for its local joins.
Evaluation covers strong and weak scaling on 1, 2, 4 and 8 nodes of the `spmcluster`, speedup
against the sequential baseline, and a breakdown of total time into communication/redistribution
and local join.

*Key results:* the bottleneck at `nb = 2` is the simultaneous materialization of all `P` hash
tables, not the lack of an asynchronous progress thread — a `UCX_PROGRESS_MODE=thread` run shows
no improvement. Weak scaling exhibits reproducible super-efficiency at 8 nodes.

---

## Building and running

All modules target the **spmcluster** (Open MPI 5.0.3, SLURM). From any module directory:

```bash
cd module<N>
make all
make check        # correctness suite
```

Important: builds use `-march=native`, so **compilation must happen on the execution node**.
Compiling on the frontend (`spmln`) and running on a compute node causes `SIGILL`. Every sbatch
script therefore runs `make clean && make all` before the benchmark.

Benchmarks are launched through SLURM:

```bash
sbatch scripts/<benchmark>.sh        # --partition=normal
```

MPI runs use the PMIx plugin:

```bash
srun --mpi=pmix -N <nodes> -n <processes> ./<binary> [options]
```

---

## Correctness

Correctness is enforced at two levels:

- **Small inputs** — results are checked element by element against a naive `O(|R|·|S|)` verifier,
  over a suite of adversarial cases: standard generation, sequential keys, all-identical keys and
  duplicate clusters.
- **Large inputs** — the triple `(join_count, checksum1, checksum2)` is compared against the
  sequential baseline. Any implementation that does not reproduce it exactly is considered wrong.

`make check` must pass before any performance measurement is taken.

---

## Experimental methodology

- Execution on the `spmcluster`, SLURM partition `normal`.
- NUMA isolation via `numactl --cpunodebind=0 --membind=0`.
- Each configuration is repeated and the **median** across repetitions is reported.
- Results are emitted in TSV format for direct ingestion by the plotting scripts.
- Hardware specifications, compiler version and optimization flags are documented in the
  per-module READMEs rather than in the reports.

---

## Reports

A separate PDF report accompanies each module and follows the same structure:
Design Choices → Correctness → Experimental Setup → Performance Results → Discussion → Conclusions.
