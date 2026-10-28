#!/bin/bash
#SBATCH --job-name=hj_m3_sched
#SBATCH --output=hj_m3_sched_%j.out
#SBATCH --error=hj_m3_sched_%j.err
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#
# bench_schedule.sh — OpenMP loop scheduling policy comparison.
#
# Uses the hashjoin_omp_loop binary, which is compiled with
# `schedule(runtime)` so the policy is selected at runtime via the
# OMP_SCHEDULE environment variable. A single binary covers all four
# policies (static, dynamic,1, dynamic,4, guided), with no source-level
# fork.
#
# Setup: T=16 (peak speedup point from Module 2), uniform + skewed,
#        REPS=3 medians for stability.
#
# Output: bench_schedule_results.tsv
#

set -o pipefail

# Fixed parameters — match the main bench so results are comparable
NR=50000000
NS=80000000
SEED=13
MAXKEY=1000000
P=256
T=16
HOT=4
REPS=3

WORKLOADS=(uniform skewed)
SKEW_HEAVY=0.9

# OpenMP scheduling policies to compare.
#   static       => default loop-static (chunks of P/T contiguous partitions)
#   dynamic,1    => one-iteration chunks pulled from a shared queue
#   dynamic,4    => 4-iteration chunks (intermediate granularity)
#   guided       => geometric chunks: large first, smaller as work depletes
SCHEDULES=(static "dynamic,1" "dynamic,4" guided)

OUT=bench_schedule_results.tsv

echo "==================================================================="
echo "  bench_schedule.sh"
echo "                    output    = ${OUT}"
echo "                    REPS=${REPS}  T=${T}  NR=${NR} NS=${NS}  P=${P}"
echo "                    Schedules: ${SCHEDULES[*]}"
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

# OpenMP thread affinity (independent of OMP_SCHEDULE)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ---------------------------------------------------------------------------
# Pre-flight check: smoke test every binary we will invoke.
# ---------------------------------------------------------------------------
echo "=== Pre-flight check ==="
for impl in hashjoin_omp_loop hashjoin_omp_task; do
    if [[ ! -x "./$impl" ]]; then
        echo "FATAL: ./$impl not found or not executable" >&2
        exit 1
    fi
    if ! ./"$impl" -nr 1000 -ns 1000 -seed 13 -max-key 100 \
                   -p 16 -t 4 -skew 0.5 -hot 4 \
                   >/dev/null 2>/tmp/preflight_err.log; then
        echo "FATAL: ./$impl smoke test failed:" >&2
        sed 's/^/    /' /tmp/preflight_err.log >&2
        exit 1
    fi
    echo "  $impl: OK"
done
echo

# TSV header
printf '%-20s\t%-12s\t%-10s\t%-5s\t%-5s\t%-15s\t%-10s\t%-10s\t%-15s\n' \
    'impl' 'schedule' 'workload' 'T' 'rep' \
    'partitioning_s' 'join_s' 'total_s' 'join_count' > "$OUT"

trim() {
    printf '%s' "$1" | tr -d '\r' | tr -d '[:space:]'
}

# ---------------------------------------------------------------------------
# run_one IMPL SCHED WORKLOAD REP SKEW_FLAG
# ---------------------------------------------------------------------------
FAILURES=0
run_one() {
    local impl="$1"; local sched="$2"; local workload="$3"
    local rep="$4"; local skew_flag="$5"

    # OMP_SCHEDULE only matters for the omp_loop binary; omp_task ignores it.
    # We set it for both anyway; it has no effect on omp_task.
    local cmd=( $NUMACTL ./"$impl"
                -nr $NR -ns $NS -seed $SEED -max-key $MAXKEY -p $P -t $T
                $skew_flag )

    local out err rc
    out=$(OMP_SCHEDULE="$sched" "${cmd[@]}" 2>/tmp/bench_sched_err.log)
    rc=$?
    err=$(cat /tmp/bench_sched_err.log)

    if [[ $rc -ne 0 ]]; then
        echo "------------------------------------------------------------" >&2
        echo "FAILED (exit=$rc): impl=$impl sched=$sched workload=$workload rep=$rep" >&2
        echo "Command: OMP_SCHEDULE=$sched"                                  >&2
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

    printf '%-20s\t%-12s\t%-10s\t%-5s\t%-5s\t%-15s\t%-10s\t%-10s\t%-15s\n' \
        "$impl" "$sched" "$workload" "$T" "$rep" \
        "$partitioning" "$join_t" "$total" "$join_count" >> "$OUT"
}

# =========================================================================
# SWEEP: omp_loop across all schedules + omp_task as reference baseline
# =========================================================================
for workload in "${WORKLOADS[@]}"; do
    if [[ "$workload" == "uniform" ]]; then
        skew_flag=""
    else
        skew_flag="-skew $SKEW_HEAVY -hot $HOT"
    fi

    echo "==================================================================================="
    echo "  workload=$workload  T=$T  hot=$HOT  skew_flag=$skew_flag"
    echo "==================================================================================="

    # Loop with each scheduling policy
    for sched in "${SCHEDULES[@]}"; do
        for rep in $(seq 1 $REPS); do
            run_one "hashjoin_omp_loop" "$sched" "$workload" "$rep" "$skew_flag" || true
        done
    done

    # Task version as reference (OMP_SCHEDULE has no effect on it)
    for rep in $(seq 1 $REPS); do
        run_one "hashjoin_omp_task" "n/a" "$workload" "$rep" "$skew_flag" || true
    done
done

echo
echo "Done. Results in $OUT"
if [[ $FAILURES -gt 0 ]]; then
    echo "WARNING: $FAILURES run(s) failed. Inspect stderr above." >&2
fi
echo
echo "Median total_s per (impl, schedule, workload):"
echo "----------------------------------------------"
awk -F'\t' 'NR>1 {
    for (i=1; i<=NF; i++) { gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    key = $1 "\t" $2 "\t" $3
    a[key, ++n[key]] = $8 + 0   # total_s
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
        printf "%-25s  sched=%-10s  workload=%-10s  %.4f s\n", \
            parts[1], parts[2], parts[3], med
    }
}' "$OUT" | sort

exit $((FAILURES > 0 ? 1 : 0))