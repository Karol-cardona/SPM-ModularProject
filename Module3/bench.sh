#!/bin/bash
#SBATCH --job-name=hj_m3
#SBATCH --output=hj_m3_%j.out
#SBATCH --error=hj_m3_%j.err
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#
# bench.sh — main benchmark sweep for SPM Modulo 3.
#
# Two scaling experiments per workload (uniform + skewed):
#
#   1. STRONG SCALING
#        Fixed problem size (NR=50M, NS=80M), thread count T = 1, 2, 4, 8, 16, 32.
#        Goal: measure speedup.
#
#   2. WEAK SCALING
#        Problem size grows linearly with T:
#            NR(T) = NR_PER_THREAD * T   (NR_PER_THREAD = 10M, matches Module 2)
#            NS(T) = NS_PER_THREAD * T   (NS_PER_THREAD = 16M, matches Module 2)
#        At T=16 the dataset reaches ~416M records (~3.3 GB), the same
#        configuration used in the Module 2 weak-scaling report.
#
# PARAMETRIZED on OMP_SCHEDULE: hashjoin_omp_loop is compiled with
# `schedule(runtime)`, so its scheduling policy is read from the env.
# The output TSV is named accordingly so multiple runs coexist.
#
# Usage:
#   sbatch bench.sh                          # static (default)
#   OMP_SCHEDULE=dynamic,1 sbatch bench.sh   # dynamic,1
#
# Output filenames:
#   bench_results_static.tsv
#   bench_results_dynamic1.tsv
#
# par, omp_task and seq ignore OMP_SCHEDULE; their behaviour is invariant.
#

set -o pipefail

# ---------------------------------------------------------------------------
# OMP_SCHEDULE handling
# ---------------------------------------------------------------------------
: "${OMP_SCHEDULE:=static}"
export OMP_SCHEDULE
SCHED_TAG=$(printf '%s' "$OMP_SCHEDULE" | tr -cd '[:alnum:]')
OUT="bench_results_${SCHED_TAG}.tsv"

# ---------------------------------------------------------------------------
# Strong scaling parameters
# ---------------------------------------------------------------------------
NR_STRONG=50000000
NS_STRONG=80000000

# ---------------------------------------------------------------------------
# Weak scaling parameters
# Per-thread base sizes match the Module 2 weak-scaling setup.
# ---------------------------------------------------------------------------
NR_PER_THREAD=10000000   # T=1: 10M ; T=16: 160M
NS_PER_THREAD=16000000   # T=1: 16M ; T=16: 256M

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------
SEED=13
MAXKEY=1000000
P=256
HOT=4
SKEW_HEAVY=0.9
REPS=3

THREAD_COUNTS=(1 2 4 8 16 32)
PARALLEL_IMPLS=(hashjoin_par hashjoin_omp_loop hashjoin_omp_task)

echo "==================================================================="
echo "  bench.sh (main)   OMP_SCHEDULE=${OMP_SCHEDULE}"
echo "                    output       = ${OUT}"
echo "                    REPS=${REPS}"
echo "                    Strong: NR=${NR_STRONG} NS=${NS_STRONG} P=${P}"
echo "                    Weak:   per-thread NR=${NR_PER_THREAD} NS=${NS_PER_THREAD}"
echo "                    Threads: ${THREAD_COUNTS[*]}"
echo "==================================================================="
echo
echo "=== Compute node info ==="
echo "  Hostname: $(hostname)"
echo "  CPU:      $(grep 'model name' /proc/cpuinfo | head -1 | sed 's/.*: //')"
echo "  Cores:    $(nproc)"
echo
echo "=== Building (clean rebuild on this compute node) ==="
make -s clean
make -s all
echo "  Build done."
echo

# NUMA pinning
if command -v numactl >/dev/null 2>&1; then
    NUMACTL="numactl --cpunodebind=0 --membind=0"
else
    echo "WARNING: numactl not found, running without NUMA pinning" >&2
    NUMACTL=""
fi

# OpenMP thread affinity (independent from OMP_SCHEDULE)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ---------------------------------------------------------------------------
# Pre-flight check: smoke test every binary with the flag set we'll use.
# ---------------------------------------------------------------------------
echo "=== Pre-flight check ==="
for impl in hashjoin_seq "${PARALLEL_IMPLS[@]}"; do
    if [[ ! -x "./$impl" ]]; then
        echo "FATAL: ./$impl not found or not executable" >&2
        exit 1
    fi
    # Smoke test: pick the right flag combo per impl
    if [[ "$impl" == "hashjoin_seq" ]]; then
        if ! ./"$impl" -nr 1000 -ns 1000 -seed 13 -max-key 100 -p 16 \
                       >/dev/null 2>/tmp/preflight_err.log; then
            echo "FATAL: ./$impl smoke test failed:" >&2
            sed 's/^/    /' /tmp/preflight_err.log >&2
            exit 1
        fi
    else
        if ! ./"$impl" -nr 1000 -ns 1000 -seed 13 -max-key 100 -p 16 -t 4 -skew 0.5 -hot 4 \
                       >/dev/null 2>/tmp/preflight_err.log; then
            echo "FATAL: ./$impl smoke test failed:" >&2
            sed 's/^/    /' /tmp/preflight_err.log >&2
            exit 1
        fi
    fi
    echo "  $impl: OK"
done
echo

# ---------------------------------------------------------------------------
# TSV header
# ---------------------------------------------------------------------------
printf '%-20s\t%-10s\t%-10s\t%-8s\t%-5s\t%-5s\t%-12s\t%-12s\t%-15s\t%-10s\t%-10s\t%-15s\n' \
    'impl' 'schedule' 'workload' 'scaling' 'T' 'rep' 'NR' 'NS' \
    'partitioning_s' 'join_s' 'total_s' 'join_count' > "$OUT"

trim() {
    printf '%s' "$1" | tr -d '\r' | tr -d '[:space:]'
}

# ---------------------------------------------------------------------------
# run_one IMPL WORKLOAD SCALING T REP SKEW_FLAG NR NS
# ---------------------------------------------------------------------------
FAILURES=0
run_one() {
    local impl="$1"; local workload="$2"; local scaling="$3"
    local T="$4"; local rep="$5"; local skew_flag="$6"
    local nr="$7"; local ns="$8"

    local t_arg=""
    if [[ "$impl" != "hashjoin_seq" ]]; then t_arg="-t $T"; fi

    # Build cmd as array for clean quoting on echo
    local cmd=( $NUMACTL ./"$impl"
                -nr $nr -ns $ns -seed $SEED -max-key $MAXKEY -p $P
                $t_arg $skew_flag )

    local out err rc
    out=$("${cmd[@]}" 2>/tmp/bench_err.log)
    rc=$?
    err=$(cat /tmp/bench_err.log)

    if [[ $rc -ne 0 ]]; then
        echo "------------------------------------------------------------" >&2
        echo "FAILED (exit=$rc): $impl  workload=$workload  scaling=$scaling  T=$T  rep=$rep" >&2
        echo "Command:"                                                      >&2
        printf '    %q ' "${cmd[@]}"                                         >&2
        echo                                                                 >&2
        echo "stdout:"; echo "$out" | sed 's/^/    /'                       >&2
        echo "stderr:"; echo "$err" | sed 's/^/    /'                       >&2
        echo "------------------------------------------------------------" >&2
        FAILURES=$((FAILURES + 1))
        return 1
    fi

    local total partitioning join_t join_count
    total=$(echo "$out"        | awk -F= '/^time_sec=/   {print $2}')
    join_count=$(echo "$out"   | awk -F= '/^join_count=/ {print $2}')
    partitioning=$(echo "$err" | awk '/\[phases\]/ {for(i=1;i<=NF;i++) if($i ~ /^partitioning=/) {sub("partitioning=","",$i); sub("s$","",$i); print $i}}')
    join_t=$(echo "$err"       | awk '/\[phases\]/ {for(i=1;i<=NF;i++) if($i ~ /^join=/)         {sub("join=","",$i);         sub("s$","",$i); print $i}}')

    total=$(trim "$total")
    join_count=$(trim "$join_count")
    partitioning=$(trim "$partitioning")
    join_t=$(trim "$join_t")

    # schedule column: real policy for omp_loop, "n/a" for everyone else
    local sched_label
    if [[ "$impl" == "hashjoin_omp_loop" ]]; then
        sched_label="$OMP_SCHEDULE"
    else
        sched_label="n/a"
    fi

    printf '%-20s\t%-10s\t%-10s\t%-8s\t%-5s\t%-5s\t%-12s\t%-12s\t%-15s\t%-10s\t%-10s\t%-15s\n' \
        "$impl" "$sched_label" "$workload" "$scaling" "$T" "$rep" "$nr" "$ns" \
        "$partitioning" "$join_t" "$total" "$join_count" >> "$OUT"
}

# =========================================================================
# STRONG SCALING SWEEP
# =========================================================================
for workload in uniform skewed; do
    if [[ "$workload" == "uniform" ]]; then
        skew_flag=""
    else
        skew_flag="-skew $SKEW_HEAVY -hot $HOT"
    fi

    echo "==================================================================================="
    echo "  STRONG  workload=$workload  NR=$NR_STRONG NS=$NS_STRONG  OMP_SCHEDULE=$OMP_SCHEDULE"
    echo "==================================================================================="

    # Sequential: T meaningless, recorded as T=1.
    for rep in $(seq 1 $REPS); do
        run_one hashjoin_seq "$workload" "strong" 1 "$rep" "$skew_flag" \
                "$NR_STRONG" "$NS_STRONG" || true
    done

    for T in "${THREAD_COUNTS[@]}"; do
        for impl in "${PARALLEL_IMPLS[@]}"; do
            for rep in $(seq 1 $REPS); do
                run_one "$impl" "$workload" "strong" "$T" "$rep" "$skew_flag" \
                        "$NR_STRONG" "$NS_STRONG" || true
            done
        done
    done
done

# =========================================================================
# WEAK SCALING SWEEP
# Baseline for efficiency = total_s at T=1 of the SAME implementation.
# =========================================================================
for workload in uniform skewed; do
    if [[ "$workload" == "uniform" ]]; then
        skew_flag=""
    else
        skew_flag="-skew $SKEW_HEAVY -hot $HOT"
    fi

    echo "==================================================================================="
    echo "  WEAK    workload=$workload  base=${NR_PER_THREAD}/${NS_PER_THREAD} per thread  OMP_SCHEDULE=$OMP_SCHEDULE"
    echo "==================================================================================="

    for T in "${THREAD_COUNTS[@]}"; do
        nr=$((NR_PER_THREAD * T))
        ns=$((NS_PER_THREAD * T))

        for impl in "${PARALLEL_IMPLS[@]}"; do
            for rep in $(seq 1 $REPS); do
                run_one "$impl" "$workload" "weak" "$T" "$rep" "$skew_flag" \
                        "$nr" "$ns" || true
            done
        done
    done
done

echo
echo "Done. Results in $OUT"
if [[ $FAILURES -gt 0 ]]; then
    echo "WARNING: $FAILURES run(s) failed. Inspect stderr above." >&2
fi
echo
echo "Median total_s per (impl, workload, scaling, T):"
echo "------------------------------------------------"
awk -F'\t' 'NR>1 {
    for (i=1; i<=NF; i++) { gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    key = $1 "\t" $3 "\t" $4 "\t" $5    # impl, workload, scaling, T
    a[key, ++n[key]] = $11 + 0          # total_s (now in column 11)
}
END {
    for (k in n) {
        cnt = n[k]
        for (i = 2; i <= cnt; i++) {
            v = a[k, i]; j = i - 1
            while (j >= 1 && a[k, j] > v) { a[k, j+1] = a[k, j]; j-- }
            a[k, j+1] = v
        }
        med = (cnt % 2 == 1) ? a[k, (cnt+1)/2] : (a[k, cnt/2] + a[k, cnt/2+1]) / 2
        split(k, parts, "\t")
        printf "%-20s\t%-10s\t%-8s\t%-5s\t%.6f\n", \
            parts[1], parts[2], parts[3], parts[4], med
    }
}' "$OUT" | sort

exit $((FAILURES > 0 ? 1 : 0))