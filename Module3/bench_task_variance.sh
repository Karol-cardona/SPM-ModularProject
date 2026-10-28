#!/bin/bash
#SBATCH --job-name=hj_m3_taskvar
#SBATCH --output=hj_m3_taskvar_%j.out
#SBATCH --error=hj_m3_taskvar_%j.err
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#
# bench_task_variance.sh — variance / bimodality study for omp_task.
#
# Runs 30 repetitions of the (hot=4, skew=0.9, T=16) configuration for
# three implementations (hashjoin_par, hashjoin_omp_loop with dynamic,1,
# hashjoin_omp_task) to characterise the run-to-run latency distribution
# observed in bench_skew_results_static.tsv and bench_schedule_results.tsv.
# Results are written to bench_task_variance.tsv.

set -o pipefail

# ---------------------------------------------------------------------------
# Fixed parameters: the SAME configuration as in the main bench's
# ---------------------------------------------------------------------------
NR=50000000
NS=80000000
SEED=13
MAXKEY=1000000
P=256
T=16
HOT=4
SKEW=0.9
REPS=30

# Implementations to characterize. We compare three strategies under the
# same skewed input to isolate whether the bimodality is unique to tasks.
IMPLS=(hashjoin_par hashjoin_omp_loop hashjoin_omp_task)

OUT="bench_task_variance.tsv"

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

# OpenMP affinity
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# omp_loop uses dynamic,1 to match omp_task semantics as closely as
# possible (one work unit dispatched at a time), so any difference in
# distribution shape can be attributed to the runtime mechanism, not
# to the scheduling policy.
export OMP_SCHEDULE=dynamic,1

# ---------------------------------------------------------------------------
# Pre-flight: smoke test every binary with the exact flag set we'll use.
# Catches problems early instead of failing rep-by-rep.
# ---------------------------------------------------------------------------
echo "=== Pre-flight check ==="
for impl in "${IMPLS[@]}"; do
    if [[ ! -x "./$impl" ]]; then
        echo "FATAL: ./$impl not found or not executable" >&2
        exit 1
    fi
    if ! ./"$impl" -nr 1000 -ns 1000 -seed 13 -max-key 100 \
                   -p 16 -t 4 -skew 0.5 -hot 4 \
                   >/dev/null 2>/tmp/preflight_err.log; then
        echo "FATAL: ./$impl smoke test failed. stderr:" >&2
        sed 's/^/    /' /tmp/preflight_err.log >&2
        exit 1
    fi
    echo "  $impl: OK"
done
echo

# ---------------------------------------------------------------------------
# TSV header
# ---------------------------------------------------------------------------
printf '%-20s\t%-10s\t%-5s\t%-9s\t%-5s\t%-5s\t%-15s\t%-10s\t%-10s\t%-15s\n' \
    'impl' 'schedule' 'hot' 'skew_frac' 'T' 'rep' \
    'partitioning_s' 'join_s' 'total_s' 'join_count' > "$OUT"

trim() {
    printf '%s' "$1" | tr -d '\r' | tr -d '[:space:]'
}

# ---------------------------------------------------------------------------
# Run a single configuration. Same robust error reporting as the other
# bench scripts: on failure print the full command + stdout + stderr,
# count the failure, and continue (so one bad rep doesn't kill the sweep).
# ---------------------------------------------------------------------------
FAILURES=0
run_one() {
    local impl="$1"; local rep="$2"

    local cmd=( $NUMACTL ./"$impl"
                -nr $NR -ns $NS -seed $SEED -max-key $MAXKEY
                -p $P -t $T -skew $SKEW -hot $HOT )

    local out err rc
    out=$("${cmd[@]}" 2>/tmp/bench_taskvar_err.log)
    rc=$?
    err=$(cat /tmp/bench_taskvar_err.log)

    if [[ $rc -ne 0 ]]; then
        echo "------------------------------------------------------------" >&2
        echo "FAILED (exit=$rc): $impl  rep=$rep"                          >&2
        echo "Command:"                                                     >&2
        printf '    %q ' "${cmd[@]}"                                        >&2
        echo                                                                >&2
        echo "stdout:"; echo "$out" | sed 's/^/    /'                      >&2
        echo "stderr:"; echo "$err" | sed 's/^/    /'                      >&2
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

    local sched_label
    if [[ "$impl" == "hashjoin_omp_loop" ]]; then
        sched_label="$OMP_SCHEDULE"
    else
        sched_label="n/a"
    fi

    printf '%-20s\t%-10s\t%-5s\t%-9s\t%-5s\t%-5s\t%-15s\t%-10s\t%-10s\t%-15s\n' \
        "$impl" "$sched_label" "$HOT" "$SKEW" "$T" "$rep" \
        "$partitioning" "$join_t" "$total" "$join_count" >> "$OUT"
}

# =========================================================================
# SWEEP: REPS repetitions of each impl, identical configuration
# =========================================================================
echo "==================================================================="
echo "  TASK VARIANCE STUDY"
echo "  hot=$HOT  skew_frac=$SKEW  T=$T  REPS=$REPS"
echo "  NR=$NR  NS=$NS  P=$P  OMP_SCHEDULE=$OMP_SCHEDULE"
echo "==================================================================="

for impl in "${IMPLS[@]}"; do
    echo ""
    echo "--- Running $REPS repetitions of $impl ---"
    for rep in $(seq 1 $REPS); do
        run_one "$impl" "$rep" || true
        # progress dots every 5 reps
        if (( rep % 5 == 0 )); then
            echo "  done $rep/$REPS"
        fi
    done
done

echo
echo "Done. Results in $OUT"
if [[ $FAILURES -gt 0 ]]; then
    echo "WARNING: $FAILURES run(s) failed. Inspect stderr above." >&2
fi
echo

# ---------------------------------------------------------------------------
# Quick statistics summary
# Computes for each impl: count, min, max, mean, median, stdev.
# ---------------------------------------------------------------------------
echo "=== Summary statistics on total_s ==="
echo "(real analysis: load $OUT into Python/R for histogram + bimodality test)"
echo
awk -F'\t' '
NR > 1 {
    for (i=1; i<=NF; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i)
    impl = $1
    val  = $9 + 0   # total_s
    arr[impl, ++cnt[impl]] = val
    sum[impl] += val
    sumsq[impl] += val * val
    if (!(impl in mn) || val < mn[impl]) mn[impl] = val
    if (!(impl in mx) || val > mx[impl]) mx[impl] = val
}
END {
    printf "%-22s %4s %9s %9s %9s %9s %9s\n", \
        "impl", "n", "min", "max", "mean", "median", "stdev"
    for (impl in cnt) {
        n = cnt[impl]
        # Sort values for median
        for (i = 1; i <= n; i++) v[i] = arr[impl, i]
        for (i = 2; i <= n; i++) {
            tmp = v[i]; j = i - 1
            while (j >= 1 && v[j] > tmp) { v[j+1] = v[j]; j-- }
            v[j+1] = tmp
        }
        med = (n % 2 == 1) ? v[(n+1)/2] : (v[n/2] + v[n/2+1]) / 2
        mean   = sum[impl] / n
        # Population stdev
        var    = sumsq[impl] / n - mean * mean
        if (var < 0) var = 0
        stdev  = sqrt(var)
        printf "%-22s %4d %9.4f %9.4f %9.4f %9.4f %9.4f\n", \
            impl, n, mn[impl], mx[impl], mean, med, stdev
        delete v
    }
}' "$OUT"

echo
echo "=== Sorted total_s for each impl (look for bimodality manually) ==="
for impl in "${IMPLS[@]}"; do
    echo
    echo "$impl:"
    awk -F'\t' -v target="$impl" '
    NR > 1 {
        for (i=1; i<=NF; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if ($1 == target) print $9 + 0
    }' "$OUT" | sort -n | awk '{printf "  %.4f\n", $1}'
done

exit $((FAILURES > 0 ? 1 : 0))
