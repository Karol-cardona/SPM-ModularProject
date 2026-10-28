#!/bin/bash
#SBATCH --job-name=hj_m4_weak
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=output_weak_%j.txt
#SBATCH --error=output_weak_%j.txt

cd $HOME/Module4
make -s clean
make -s all

SEED=13; MAXKEY=1000000; P=256
NR_PER=10000000
NS_PER=16000000
REPS=3

SKEWFLAGS=""; WORKLOAD="uniform"
if [ -n "$SKEW" ] && [ "$SKEW" != "0" ]; then
    SKEWFLAGS="-skew $SKEW -hot ${HOT:-4}"; WORKLOAD="skewed"
fi

OUT="run_results_weak_${WORKLOAD}.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    experiment workload nodes nb threads rep NR NS \
    t_gen t_redist t_localjoin t_reduce t_rj t_total join_count > "$OUT"

append_row() {
    local exp="$1" wl="$2" nodes="$3" nb="$4" th="$5" rep="$6" nr="$7" ns="$8" out="$9"
    local f; get() { printf '%s\n' "$out" | awk -F'[= ]' -v k="$1" '$1==k{print $2; exit}'; }
    local jc tg tr tl trd trj tt
    jc=$(get join_count); tg=$(get t_gen_sec); tr=$(get t_redist_sec); tl=$(get t_localjoin_sec)
    trd=$(get t_reduce_sec); trj=$(get t_rj_sec); tt=$(get t_total_sec)
    [ -z "$tt" ] && tt=$(get time_sec)
    for f in tg tr tl trd trj tt jc; do [ -z "${!f}" ] && printf -v "$f" 'na'; done
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$exp" "$wl" "$nodes" "$nb" "$th" "$rep" "$nr" "$ns" \
        "$tg" "$tr" "$tl" "$trd" "$trj" "$tt" "$jc" >> "$OUT"
}

echo "=== WEAK SCALING ($WORKLOAD) — REPS=$REPS — results in $OUT ==="
for N in 1 2 4 8; do
    NR_W=$(( NR_PER * N )); NS_W=$(( NS_PER * N ))
    for rep in $(seq 1 $REPS); do
        echo "--- nodes=$N  NR=$NR_W  NS=$NS_W  rep=$rep ---"
        out=$(srun --mpi=pmix -N $N --ntasks-per-node=1 -n $N ./hashjoin_mpi \
                -nr $NR_W -ns $NS_W -seed $SEED -max-key $MAXKEY -p $P $SKEWFLAGS 2>&1)
        echo "$out"; echo ""
        append_row weak "$WORKLOAD" "$N" 0 1 "$rep" "$NR_W" "$NS_W" "$out"
    done
done

echo ""
echo "=== MEDIAN t_total PER NODE COUNT (s) ==="
awk -F'\t' 'NR>1 && $14!="na" {
    a[$3, ++n[$3]] = $14 + 0
}
END {
    for (k in n) {
        c=n[k]; for (i=2;i<=c;i++){v=a[k,i];j=i-1; while(j>=1&&a[k,j]>v){a[k,j+1]=a[k,j];j--} a[k,j+1]=v}
        med=(c%2==1)?a[k,(c+1)/2]:(a[k,c/2]+a[k,c/2+1])/2
        printf "nodes=%-2s  median_t_total=%.6f\n", k, med
    }
}' "$OUT" | sort -t= -k2 -n