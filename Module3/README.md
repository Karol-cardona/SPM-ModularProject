# SPM Modulo 3 — Partitioned Hash Join with OpenMP

OpenMP loop-based and task-based parallel implementations of the partitioned
hash join with duplicates, compared with the sequential baseline and the C++
`std::thread` version from Module 2.

## Files

### Source code

| File                     | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| `common.hpp`             | Shared utilities (Record, hashes, FlatCountMap, generation, verifier) |
| `hashjoin_seq.cpp`       | Sequential baseline (Module 2 baseline)                    |
| `hashjoin_par.cpp`       | C++ `std::thread` parallel (Module 2 v3)                   |
| `hashjoin_omp_loop.cpp`  | OpenMP loop-level parallel — `schedule(runtime)`, policy chosen via `OMP_SCHEDULE` |
| `hashjoin_omp_task.cpp`  | OpenMP task-based parallel (`#pragma omp task`)            |
| `Makefile`               | Build all four binaries; `make check` runs correctness     |

### Benchmark scripts (all SLURM-ready, runnable with `sbatch`)

| File                          | Purpose                                                         | Output TSV(s)                          |
| ----------------------------- | --------------------------------------------------------------- | -------------------------------------- |
| `bench.sh`                    | Main strong + weak scaling sweep (T=1..32, uniform + skewed)    | `bench_results_<sched>.tsv`            |
| `bench_skew_sensitivity.sh`   | Skew sensitivity: matrix of `num_hot` × `skew_frac` at T=16     | `bench_skew_results_<sched>.tsv`       |
| `bench_schedule.sh`           | Comparison of `OMP_SCHEDULE` policies (static, dynamic, guided) | `bench_schedule_results.tsv`           |
| `bench_task_variance.sh`      | Variance / bimodality study of `omp_task` (30 reps)             | `bench_task_variance.tsv`              |

The `<sched>` suffix in the output filenames is auto-derived from the
`OMP_SCHEDULE` value, allowing multiple runs with different policies to
coexist (e.g. `bench_results_static.tsv` and `bench_results_dynamic1.tsv`).

All four binaries share the same input generation, the same partition mapping
(`multiply_shift`, Module 1), and the same per-partition build/probe code. The
implementations differ only in HOW the parallelism is expressed.

## Build

```bash
make all
```

Requirements: a C++17 compiler with OpenMP 3.0+ support (GCC ≥ 9 is fine).
`omp_get_thread_num`, `parallel for schedule(runtime)`, `task` and
`reduction` — all features used here are part of OpenMP 3.0/3.1.

## Quick correctness check

```bash
make check
```

Runs each of the four binaries on a small input (NR=200, NS=300) for both
the uniform and skewed scenarios. The brute-force O(|R|×|S|) verifier
embedded in each binary is compared against the partitioned-join output;
the `make check` target then cross-checks `join_count`, `checksum1` and
`checksum2` across all four implementations and the verifier. For
`hashjoin_omp_loop` the check is run with `OMP_SCHEDULE=dynamic,1`,
which exercises the runtime scheduling path more aggressively. Output:

```
=== Uniform correctness check (NR=200, NS=300, skew=0) ===
  seq  : join_count=587 checksum1=... checksum2=... (matches naive)
  par  : join_count=587 checksum1=... checksum2=... (matches naive)
  loop : join_count=587 checksum1=... checksum2=... (matches naive)
  task : join_count=587 checksum1=... checksum2=... (matches naive)
=== Skewed correctness check (NR=200, NS=300, skew=0.8) ===
  ...
All correctness checks PASSED.
```

## Run examples

Uniform workload (Module 2 baseline parameters):

```bash
./hashjoin_seq      -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256
./hashjoin_par      -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8
./hashjoin_omp_loop -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8
./hashjoin_omp_task -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8
```

Skewed workload (90% of records routed to 4 hot partitions out of 256):

```bash
./hashjoin_omp_loop -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8 -skew 0.9
./hashjoin_omp_task -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -t 8 -skew 0.9
```

The `omp_loop` binary changes scheduling policy via `OMP_SCHEDULE`:

```bash
OMP_SCHEDULE=static    ./hashjoin_omp_loop ... -skew 0.9
OMP_SCHEDULE=dynamic,1 ./hashjoin_omp_loop ... -skew 0.9
OMP_SCHEDULE=guided    ./hashjoin_omp_loop ... -skew 0.9
```

## Command-line flags

| Flag              | Description                                                | Default              |
| ----------------- | ---------------------------------------------------------- | -------------------- |
| `-nr NR`          | Number of records in R                                     | required             |
| `-ns NS`          | Number of records in S                                     | required             |
| `-seed SEED`      | Deterministic seed                                         | required             |
| `-max-key K`      | Keys generated in [0, K)                                   | required             |
| `-p P`            | Number of partitions (power of two)                        | required             |
| `-t T`            | Number of threads (parallel binaries only)                 | hardware concurrency |
| `-skew F`         | Fraction of records routed to hot partitions (0.0 = uniform) | 0.0                |
| `-hot H`          | Number of hot partitions when `-skew > 0`                  | 4                    |

## Benchmarking on spmcluster

All four bench scripts are SLURM batch scripts. Submit with:

```bash
sbatch bench.sh
sbatch bench_schedule.sh
sbatch bench_skew_sensitivity.sh
sbatch bench_task_variance.sh
```

The scripts that depend on `OMP_SCHEDULE` (`bench.sh` and
`bench_skew_sensitivity.sh`) accept it from the environment to produce
distinct output TSVs:

```bash
sbatch bench.sh                          # bench_results_static.tsv (default)
OMP_SCHEDULE=dynamic,1 sbatch bench.sh   # bench_results_dynamic1.tsv

sbatch bench_skew_sensitivity.sh                          # static
OMP_SCHEDULE=dynamic,1 sbatch bench_skew_sensitivity.sh   # dynamic,1
```
For manual single invocations, the recommended pattern (matching Module 2)
also pins to NUMA node 0 and binds OpenMP threads to physical cores:

```bash
OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --cpunodebind=0 --membind=0 \
    ./hashjoin_omp_task -nr 50000000 -ns 80000000 -seed 13 \
                        -max-key 1000000 -p 256 -t 16
```

The strong scaling sweep covers T = 1, 2, 4, 8, 16, 32. The weak scaling
sweep uses a per-thread base size of NR_per_thread = 10M, NS_per_thread = 16M
(matching the Module 2 weak scaling setup).
