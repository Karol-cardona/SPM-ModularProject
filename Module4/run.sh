#!/bin/bash
#SBATCH --job-name=hj_m4
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=output_%j.txt
#SBATCH --error=output_%j.txt
#
# Run the Module 4 experiments: sequential baseline, strong scaling, and weak.
# Results are appended to a TSV whose name encodes the active mode.
#
# Optional extensions (enabled via environment variables):
#   SKEW=0.9   run the entire sweep (baseline, strong, weak) with the skewed workload instead of the uniform one
#   HYBRID=1   run the strong-scaling sweep also with the MPI+OpenMP hybrid
#   NB=1       run the strong-scaling sweep with BOTH blocking (-nb 0) and non-blocking redistribution (-nb 1 overlap R/S, -nb 2 overlap
#              S transfer with R build), to compare all three
#   REPS=3     repeat every measured run REPS times (default 1); a median summary per configuration is printed at the end
#
# Output TSV (one per invocation), named by the active mode:
#   run_results_uniform.tsv            sbatch run.sh
#   run_results_skewed.tsv             SKEW=0.9 sbatch run.sh
#   run_results_uniform_nb.tsv         NB=1 sbatch run.sh
#   run_results_skewed_hybrid.tsv      SKEW=0.9 HYBRID=1 sbatch run.sh
#

cd $HOME/Module4

# compile on the compute node
make -s clean
make -s all
if [ "${HYBRID:-0}" = "1" ]; then make -s hybrid; fi

#  experiment parameters (same as Module 2/3)
NR=50000000
NS=80000000
SEED=13
MAXKEY=1000000
P=256
THREADS=${THREADS:-16}     # OpenMP threads per rank, used only by HYBRID
REPS=${REPS:-1}            # repetitions per measured run (median at the end)

SKEWFLAGS=""
SKEW_LABEL="uniform"
WORKLOAD="uniform"
if [ -n "$SKEW" ] && [ "$SKEW" != "0" ]; then
    SKEWFLAGS="-skew $SKEW -hot ${HOT:-4}"
    SKEW_LABEL="skewed (skew=$SKEW hot=${HOT:-4})"
    WORKLOAD="skewed"
fi

# TSV output file, named by the active mode
TAG="$WORKLOAD"
[ "${NB:-0}" = "1" ]     && TAG="${TAG}_nb"
[ "${HYBRID:-0}" = "1" ] && TAG="${TAG}_hybrid"
OUT="run_results_${TAG}.tsv"

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    experiment workload nodes nb threads rep NR NS \
    t_gen t_redist t_localjoin t_reduce t_rj t_total join_count > "$OUT"

append_row() {
    local exp="$1" wl="$2" nodes="$3" nb="$4" th="$5" rep="$6" nr="$7" ns="$8" out="$9"
    local f
    get() { printf '%s\n' "$out" | awk -F'[= ]' -v k="$1" '$1==k{print $2; exit}'; }
    local jc tg tr tl trd trj tt
    jc=$(get join_count)
    tg=$(get t_gen_sec)
    tr=$(get t_redist_sec)
    tl=$(get t_localjoin_sec)
    trd=$(get t_reduce_sec)
    trj=$(get t_rj_sec)
    tt=$(get t_total_sec)
    [ -z "$tt" ] && tt=$(get time_sec)        # sequential baseline
    for f in tg tr tl trd trj tt jc; do
        [ -z "${!f}" ] && printf -v "$f" 'na'
    done
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$exp" "$wl" "$nodes" "$nb" "$th" "$rep" "$nr" "$ns" \
        "$tg" "$tr" "$tl" "$trd" "$trj" "$tt" "$jc" >> "$OUT"
}

echo "===================================================================="
echo "  SPM Modulo 4 - Hash Join Performance Evaluation"
echo "  host=$(hostname)  nodelist=$SLURM_JOB_NODELIST"
echo "  NR=$NR NS=$NS P=$P seed=$SEED max-key=$MAXKEY  workload=$SKEW_LABEL"
echo "  results -> $OUT   (REPS=$REPS)"
echo "===================================================================="

# ===========================================================================
# Sequential baseline
# ===========================================================================
echo ""
echo "=== SEQUENTIAL BASELINE ==="
for rep in $(seq 1 $REPS); do
    out=$(srun --mpi=pmix -N 1 -n 1 ./hashjoin_seq \
            -nr $NR -ns $NS -seed $SEED -max-key $MAXKEY -p $P $SKEWFLAGS 2>&1)
    echo "$out"
    append_row seq "$WORKLOAD" 1 na 1 "$rep" "$NR" "$NS" "$out"
done

# ===========================================================================
# Strong scaling — fixed problem size, 1 MPI rank per node (so all
# redistribution traffic crosses the inter-node network).
# With NB=1 each node count is run three times: -nb 0 (blocking MPI_Alltoallv),
# -nb 1 (non-blocking, overlap R/S), and -nb 2 (non-blocking, overlap S transfer
# with R build), to compare the redistribution strategies.
# ===========================================================================
NBMODES="0"
if [ "${NB:-0}" = "1" ]; then NBMODES="0 1 2"; fi

echo ""
echo "=== STRONG SCALING (pure MPI, 1 rank per node) — $SKEW_LABEL ==="
for N in 1 2 4 8; do
    for nbmode in $NBMODES; do
        for rep in $(seq 1 $REPS); do
            echo "--- nodes=$N  nb=$nbmode  rep=$rep ---"
            out=$(srun --mpi=pmix -N $N --ntasks-per-node=1 -n $N ./hashjoin_mpi \
                    -nr $NR -ns $NS -seed $SEED -max-key $MAXKEY -p $P -nb $nbmode $SKEWFLAGS 2>&1)
            echo "$out"
            echo ""
            append_row strong "$WORKLOAD" "$N" "$nbmode" 1 "$rep" "$NR" "$NS" "$out"
        done
    done
done

# ===========================================================================
# Weak scaling — fixed per-rank problem size; total grows with the node count
# ===========================================================================
echo ""
echo "=== WEAK SCALING (pure MPI, 1 rank per node) — $SKEW_LABEL ==="
NR_PER=10000000
NS_PER=16000000
for N in 1 2 4 8; do
    NR_W=$(( NR_PER * N ))
    NS_W=$(( NS_PER * N ))
    for rep in $(seq 1 $REPS); do
        echo "--- nodes=$N  NR=$NR_W  NS=$NS_W  rep=$rep ---"
        out=$(srun --mpi=pmix -N $N --ntasks-per-node=1 -n $N ./hashjoin_mpi \
                -nr $NR_W -ns $NS_W -seed $SEED -max-key $MAXKEY -p $P $SKEWFLAGS 2>&1)
        echo "$out"
        echo ""
        append_row weak "$WORKLOAD" "$N" 0 1 "$rep" "$NR_W" "$NS_W" "$out"
    done
done

# ===========================================================================
# Optional: hybrid MPI+OpenMP strong scaling (1 rank per node × THREADS)
# ===========================================================================
if [ "${HYBRID:-0}" = "1" ]; then
    echo ""
    echo "=== HYBRID MPI+OpenMP STRONG SCALING (1 rank/node × $THREADS threads) ==="
    for N in 1 2 4 8; do
        for rep in $(seq 1 $REPS); do
            echo "--- nodes=$N ranks=$N threads/rank=$THREADS  rep=$rep ---"
            out=$(srun --mpi=pmix -N $N --ntasks-per-node=1 --cpus-per-task=$THREADS \
                    --export=ALL,OMP_NUM_THREADS=$THREADS,OMP_PROC_BIND=close,OMP_PLACES=cores --cpu-bind=none \
                    ./hashjoin_mpi_omp \
                    -nr $NR -ns $NS -seed $SEED -max-key $MAXKEY -p $P $SKEWFLAGS 2>&1)
            echo "$out"
            echo ""
            append_row hybrid "$WORKLOAD" "$N" 0 "$THREADS" "$rep" "$NR" "$NS" "$out"
        done
    done
fi

# ===========================================================================
# Median t_total per configuration (experiment, workload, nodes, nb, threads)
# ===========================================================================
echo ""
echo "=== MEDIAN t_total PER CONFIGURATION (s) ==="
echo "experiment  workload  nodes  nb  threads  median_t_total"
awk -F'\t' 'NR>1 && $14!="na" {
    key = $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5
    a[key, ++n[key]] = $14 + 0
}
END {
    for (k in n) {
        cnt = n[k]
        for (i = 2; i <= cnt; i++) { v = a[k,i]; j = i-1;
            while (j >= 1 && a[k,j] > v) { a[k,j+1] = a[k,j]; j-- } a[k,j+1] = v }
        med = (cnt % 2 == 1) ? a[k,(cnt+1)/2] : (a[k,cnt/2] + a[k,cnt/2+1]) / 2
        split(k, p, "\t")
        printf "%-11s %-9s %-6s %-3s %-8s %.6f\n", p[1], p[2], p[3], p[4], p[5], med
    }
}' "$OUT" | sort -k1,1 -k2,2 -k3,3n -k4,4

echo ""
echo "Done. Per-run data in $OUT"