# SPM Modulo 4 — Distributed Partitioned Hash Join with MPI

Distributed-memory implementation of the partitioned hash join with duplicates
using MPI.

## Files

### Source code

| File                   | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `common.hpp`           | Shared utilities, reused unchanged from Modules 2-3       |
| `hashjoin_seq.cpp`     | Sequential baseline (Module 2/3 improved version)        |
| `hashjoin_mpi.cpp`     | MPI distributed implementation; the same source compiles to the pure (`hashjoin_mpi`) and the hybrid (`hashjoin_mpi_omp`) binary |
| `Makefile`             | Build all binaries; `make check` runs correctness         |
| `run.sh`               | SLURM benchmark sweep (sequential + strong + weak; opt. skew / nb / hybrid) |
| `run_weak.sh`          | SLURM weak-scaling-only sweep, 3 repetitions with the median (used to confirm the weak results) |

## Algorithm Overview

The join is *partition-local*: equal keys always map to the same partition id,
so they are always co-located. We make the partition the unit of distribution:

1. **Distributed generation.** Each rank generates a disjoint slice of R and S.
   Both `common.hpp` generators consume a fixed number of PRNG draws per record
   (1 uniform, 2 skewed in *both* branches), so record `i` is positionally
   addressable and a rank reconstructs exactly the keys the sequential generator
   would produce for those indices. No I/O and no broadcast.
2. **Redistribution.** Each rank maps its keys to a partition id (same
   `compute_partition_id` as Modules 1-3), then to an owner rank
   `owner(pid) = pid % nprocs`, buckets records by destination, and exchanges
   them with one `MPI_Alltoallv` per relation.
3. **Local join.** Each rank runs the unmodified Module 2/3 join on the data it
   received. With the hybrid build, the loop over partitions is parallelized
   with OpenMP `schedule(dynamic,1)` (the policy Module 3 found necessary for
   skewed loads).
4. **Global reduction.** The three scalars are summed with `MPI_Reduce(MPI_SUM)`.

This is the distributed Map+Reduce pattern from the Structured Parallel
Programming lecture (scatter → map + local reduce → global reduce).

### Hash Function and Build Table

Inherited from Modules 1-3 via `common.hpp`:

- **Partition mapping** — `multiply_shift` (Dietzfelbinger universal hashing):
  `pid = (key * 0x9E3779B97F4A7C15) >> (64 - log2(P))`
- **Build table** — `FlatCountMap` (flat open-addressing, linear probing).

The same functions are used in the sequential baseline and in the MPI version.

## Build

```bash
make all            # hashjoin_seq + hashjoin_mpi (pure MPI)
make hybrid         # hashjoin_mpi_omp (MPI + OpenMP)
```

## Quick correctness check

```bash
make check
```

Runs the MPI binary at np=2 and np=4 on small uniform and skewed inputs. With
`-check 1`, rank 0 independently regenerates the inputs and recomputes the result
with a single-process reference, then asserts `join_count`, `checksum1`,
`checksum2` are identical to the distributed output. Expected output:

```
=== Uniform correctness check (NR=10000 NS=15000 P=64) ===
  np=2 : check=PASS (ref join_count=299441 checksum1=... checksum2=...)
  np=4 : check=PASS (ref join_count=299441 checksum1=... checksum2=...)
=== Skewed correctness check (NR=20000 NS=30000 P=64 skew=0.9 hot=4) ===
  np=2 : check=PASS (ref join_count=4090370 checksum1=... checksum2=...)
  np=4 : check=PASS (ref join_count=4090370 checksum1=... checksum2=...)
All correctness checks PASSED.
```

For a stronger large-input check, run `hashjoin_seq` and `hashjoin_mpi` with
identical parameters and compare the three output lines:

```bash
./hashjoin_seq -nr 10000000 -ns 15000000 -seed 13 -max-key 1000000 -p 256 \
    | grep -E '^(join_count|checksum)'
srun --mpi=pmix -n 4 ./hashjoin_mpi -nr 10000000 -ns 15000000 -seed 13 \
    -max-key 1000000 -p 256 | grep -E '^(join_count|checksum)'
```

The two outputs must be identical (and independent of `-n`).

## Run examples

```bash
# Uniform, 4 nodes, 1 rank per node
srun --mpi=pmix -N 4 --ntasks-per-node=1 ./hashjoin_mpi \
     -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256

# Skewed, 8 nodes
srun --mpi=pmix -N 8 --ntasks-per-node=1 ./hashjoin_mpi \
     -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256 -skew 0.9 -hot 4

# Hybrid MPI+OpenMP, 4 nodes × 16 threads (threads bound to physical cores)
srun --mpi=pmix -N 4 --ntasks-per-node=1 --cpus-per-task=16 \
     --export=ALL,OMP_NUM_THREADS=16,OMP_PROC_BIND=close,OMP_PLACES=cores \
     --cpu-bind=none ./hashjoin_mpi_omp \
     -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256
```

The MPI binary prints, in addition to `join_count`/`checksum`/`time_sec`, a
per-phase breakdown (maximum over ranks, i.e. slowest-rank wall time):

```
t_gen_sec        input generation (excluded from the join time, as in the baseline)
t_redist_sec     local bucketing + the two MPI_Alltoallv (fused with the join for -nb 2)
t_localjoin_sec  build + probe on the data received by this rank
t_reduce_sec     final MPI_Reduce
t_rj_sec         redist + localjoin (the metric comparable across the -nb modes)
t_total_sec      redist + localjoin + reduce
```

## Command-line flags

| Flag           | Description                                              | Default      |
| -------------- | -------------------------------------------------------- | ------------ |
| `-nr NR`       | Number of records in R                                   | required     |
| `-ns NS`       | Number of records in S                                   | required     |
| `-seed SEED`   | Deterministic seed                                       | required     |
| `-max-key K`   | Keys generated in [0, K)                                 | required     |
| `-p P`         | Number of partitions (power of two)                      | required     |
| `-skew F`      | Fraction of records routed to hot partitions             | 0.0          |
| `-hot H`       | Number of hot partitions (used only with `-skew > 0`)    | 4            |
| `-nb 0\|1\|2`   | Redistribution: 0 blocking, 1 non-blocking (overlap R/S), 2 non-blocking (overlap S transfer with R build) | 0 |
| `-check 1`     | Embedded strong correctness check (small N only)         | off          |

## Benchmarking on spmcluster

The same `run.sh` covers every experiment; environment variables select the
workload and the extras:

```bash
sbatch run.sh                       # whole sweep, uniform pure MPI (baseline + strong + weak)
SKEW=0.9 sbatch run.sh              # run the whole sweep with the skewed workload
NB=1     sbatch run.sh              # strong sweep with all three -nb modes (0/1/2)
HYBRID=1 sbatch run.sh              # add the hybrid (MPI+OpenMP) strong sweep
SKEW=0.9 HYBRID=1 sbatch run.sh     # skewed workload + hybrid sweep
```

To re-run only the weak-scaling experiment (3 repetitions, reporting the median),
use the dedicated script:

```bash
sbatch run_weak.sh                  # weak scaling only, uniform, 3 reps + median
SKEW=0.9 sbatch run_weak.sh         # weak scaling only, skewed workload
```

This avoids re-running the strong and baseline sweeps when only the weak results
need to be refreshed.

> Note: `-nb 2` keeps all per-partition build tables of R in memory at once
> (to build them while S is still in flight), so it needs more memory than
> `-nb 0/1` — comfortably within a cluster node's RAM at these sizes.

The script allocates 8 nodes and, inside the allocation,
launches the sequential baseline once and then loops `srun -N 1,2,4,8` for the
strong and weak sweeps. It uses one MPI rank per node (`--ntasks-per-node=1`),
so all redistribution traffic crosses the inter-node 10 GbE network — the
trade-off this module is about. The hybrid sweep uses one rank per node with
`OMP_NUM_THREADS=16` (configurable via the `THREADS` env var) and the OpenMP
threads bound to physical cores.