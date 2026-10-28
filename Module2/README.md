# Module 2 — Parallel Partitioned Hash Join with Duplicates

---

## Project Structure

```
Module2/
├── hashjoin_seq.cpp      Sequential reference
├── hashjoin_par_v1.cpp   Parallel version 1: parallel join only
├── hashjoin_par_v2.cpp   Parallel version 2: parallel join + parallel histogram
├── hashjoin_par_v3.cpp   Parallel version 3: fully parallel (final version)
├── Makefile              Build and correctness check
├── run.sh                SLURM benchmark script (strong + weak scaling)
├── run_maxkey.sh         SLURM script for max-key sensitivity analysis
└── README.md             This file
```

---

## Algorithm Overview

The algorithm implements a Partitioned Hash Join between two relations R and S.
Each record has a 64-bit key. A match occurs when r.key == s.key.

The pipeline consists of four phases:

1. Histogram   — count how many records go to each partition
2. Prefix sum  — compute start offsets for each partition
3. Scatter     — reorder records so each partition is contiguous in memory
4. Local join  — for each partition: build a count map on R, probe with S

After partitioning, all partitions are independent and can be joined in parallel.

### Hash Function

The placeholder compute_partition_id provided by the professor has been replaced
with multiply_shift (Dietzfelbinger universal hashing) from Module 1:

partition_id = (key * 0x9E3779B97F4A7C15) >> (64 - log2(P))

This guarantees a mathematically uniform distribution across partitions,
ensuring good load balance in the parallel join phase.
The same function is used consistently in all four versions.

---

## Build

```bash
make all
```

Requires: GCC >= 9, C++17, -pthread. 

This builds all four binaries:
hashjoin_seq, hashjoin_par_v1, hashjoin_par_v2, hashjoin_par_v3

---

## Running

### Sequential baseline

```bash
./hashjoin_seq -nr NR -ns NS -seed SEED -max-key K -p P
```

### Parallel versions

```bash
./hashjoin_par_v3 -nr NR -ns NS -seed SEED -max-key K -p P -t T
```

```
Parameters:
  -nr       Number of records in R (required)
  -ns       Number of records in S (required)
  -seed     Deterministic seed (required)
  -max-key  Keys generated in [0, max-key) (required)
  -p        Number of partitions, power of two (required)
  -t        Number of threads (default: hardware_concurrency)
```

Small input example — triggers naive verifier:
```bash
./hashjoin_par_v3 -nr 50 -ns 80 -seed 42 -max-key 20 -p 4 -t 4
```

Production scale example:
```bash
./hashjoin_par_v3 -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8
```

---

## Correctness Verification

### Automated check (recommended)

```bash
make check
```

Runs all three parallel versions against the sequential baseline on a small
input and verifies that join_count, checksum1, and checksum2 are identical.

### Small input — naive verifier

For inputs with NR <= 500 and NS <= 500, the program automatically runs a
brute-force O(|R|x|S|) verifier and prints the comparison inline:

```bash
./hashjoin_par_v3 -nr 50 -ns 80 -seed 42 -max-key 20 -p 4 -t 4
```

### Large input — manual comparison

Run seq and par with identical parameters and compare the three output lines:

```bash
./hashjoin_seq    -nr 10000000 -ns 15000000 -seed 13 -max-key 500000 -p 256 \
    | grep -E '^(join_count|checksum)'

./hashjoin_par_v3 -nr 10000000 -ns 15000000 -seed 13 -max-key 500000 -p 256 -t 8 \
    | grep -E '^(join_count|checksum)'
```

Both outputs must be identical.

---

## Phase Breakdown

hashjoin_par_v3 prints a per-run timing breakdown to stderr:

[phases] partitioning=1.823s  join=0.412s

- partitioning = histogram + prefix sum + scatter for both R and S
- join         = all local joins across all partitions

---

## Parallelization Strategy

Phase        | v1         | v2         | v3
-------------|------------|------------|----------
Histogram    | sequential | parallel   | parallel
Prefix sum   | sequential | sequential | sequential (intentional: only P=256 values)
Scatter      | sequential | sequential | parallel
Local join   | parallel   | parallel   | parallel

False sharing prevention (v3 only):
PaddedJoinResult uses alignas(64) to place each thread's result
accumulator on a separate cache line, avoiding cache invalidation
between threads during the reduction phase.

---

## Running Benchmarks on the Cluster

```bash
sbatch run.sh
```

The script compiles all binaries on the compute node, then runs:
- Sequential baseline
- Strong scaling: v1, v2, v3 with T = 1, 2, 4, 8, 16 (NR=50M, NS=80M)
- Weak scaling:   v3 with T = 1, 2, 4, 8, 16 (NR and NS scale proportionally)

Results are saved to output_<jobid>.txt.

For reproducible results, the script pins execution to a single NUMA node:
numactl --cpunodebind=0 --membind=0

## Impact of Key Distribution

To analyse how max-key affects performance, run:

```bash
sbatch run_maxkey.sh
```

The script tests six values of max-key (100, 10K, 100K, 1M, 10M, 50M)
with fixed NR=50M, NS=80M, P=256, T=8, comparing seq and par v3.
