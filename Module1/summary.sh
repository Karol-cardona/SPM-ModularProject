#!/bin/bash

RESDIR=results

# Extracts Mkeys/s from a CPU benchmark .txt file.
get_tp() {
    local file="$1" hash="$2" P="$3" N="$4"
    [ -f "$file" ] || { echo "N/A"; return; }
    awk -v h="$hash" -v p="$P" -v n="$N" '
        /N =/ { in_n = ($0 ~ n) ? 1 : 0; next }
        in_n && /^\xe2\x94\x82/ {               # lines starting with │ (UTF-8)
            gsub(/^\xe2\x94\x82|\xe2\x94\x82$/, "")
            split($0, cols, "│")
            pval = cols[1]+0
            hval = cols[2]; gsub(/[[:space:]]/, "", hval)
            if (pval == p+0 && hval == h) {
                val = cols[4]; gsub(/[[:space:]]/, "", val)
                print val; exit
            }
        }
    ' "$file"
}

# Extracts a field from a CUDA benchmark .txt file.
get_cuda() {
    local file="$1" hash="$2" P="$3" N="$4" field="$5"
    [ -f "$file" ] || { echo "N/A"; return; }
    awk -v h="$hash" -v p="$P" -v n="$N" -v f="$field" '
        /N =/ { in_n = ($0 ~ n) ? 1 : 0; next }
        in_n && /^\xe2\x94\x82/ {
            gsub(/^\xe2\x94\x82|\xe2\x94\x82$/, "")
            split($0, cols, "│")
            pval = cols[1]+0
            hval = cols[2]; gsub(/[[:space:]]/, "", hval)
            if (pval == p+0 && hval == h) {
                val = cols[f]; gsub(/[[:space:]]/, "", val)
                print val; exit
            }
        }
    ' "$file"
}

# Computes speedup (b/a), prints N/A if either value is missing.
speedup() {
    local a="$1" b="$2"
    if [ -z "$a" ] || [ -z "$b" ] || [ "$a" = "N/A" ] || [ "$b" = "N/A" ]; then
        echo "N/A"
    else
        awk "BEGIN { printf \"%.2fx\", $b / $a }"
    fi
}

HASHES="mask xor_shift multiply_shift fold32_mult murmur_fmix64"
N=100000000
P=256

echo "================================================================"
echo "  TABLE 1: Throughput (Mkeys/s) — N=100M, P=256"
echo "================================================================"
echo ""
printf "%-18s %10s %10s %10s %10s %10s\n" \
       "Hash" "novec" "autovec" "avx2" "cuda_tot" "cuda_ker"
printf "%-18s %10s %10s %10s %10s %10s\n" \
       "------------------" "----------" "----------" "----------" "----------" "----------"

for hash in $HASHES; do
    novec=$(  get_tp   "$RESDIR/novec.txt"   "$hash" $P $N)
    autovec=$(get_tp   "$RESDIR/autovec.txt" "$hash" $P $N)
    avx2=$(   get_tp   "$RESDIR/avx2.txt"   "$hash" $P $N)
    cuda_tot=$(get_cuda "$RESDIR/cuda.txt"  "$hash" $P $N 7)
    cuda_ker=$(get_cuda "$RESDIR/cuda.txt"  "$hash" $P $N 8)
    printf "%-18s %10s %10s %10s %10s %10s\n" \
           "$hash" "${novec:-N/A}" "${autovec:-N/A}" "${avx2:-N/A}" \
           "${cuda_tot:-N/A}" "${cuda_ker:-N/A}"
done

echo ""
echo ""
echo "================================================================"
echo "  TABLE 2: Speedup over novec baseline — N=100M, P=256"
echo "================================================================"
echo ""
printf "%-18s %10s %10s %10s %10s\n" \
       "Hash" "autovec/novec" "avx2/novec" "cuda_tot/novec" "cuda_ker/novec"
printf "%-18s %10s %10s %10s %10s\n" \
       "------------------" "----------" "----------" "----------" "----------"

for hash in $HASHES; do
    novec=$(   get_tp   "$RESDIR/novec.txt"   "$hash" $P $N)
    autovec=$( get_tp   "$RESDIR/autovec.txt" "$hash" $P $N)
    avx2=$(    get_tp   "$RESDIR/avx2.txt"    "$hash" $P $N)
    cuda_tot=$(get_cuda "$RESDIR/cuda.txt"    "$hash" $P $N 7)
    cuda_ker=$(get_cuda "$RESDIR/cuda.txt"    "$hash" $P $N 8)

    sp_av=$(  speedup "$novec" "$autovec")
    sp_avx2=$(speedup "$novec" "$avx2")
    sp_ctot=$(speedup "$novec" "$cuda_tot")
    sp_cker=$(speedup "$novec" "$cuda_ker")

    printf "%-18s %10s %10s %10s %10s\n" \
           "$hash" "$sp_av" "$sp_avx2" "$sp_ctot" "$sp_cker"
done

echo ""
echo ""
echo "================================================================"
echo "  TABLE 3: Effect of vectorization across N — P=256"
echo "  (autovec speedup over novec for each N)"
echo "================================================================"
echo ""
printf "%-18s %12s %12s %12s\n" "Hash" "N=10M" "N=50M" "N=100M"
printf "%-18s %12s %12s %12s\n" "------------------" "----------" "----------" "----------"

for hash in $HASHES; do
    results=""
    for n in 10000000 50000000 100000000; do
        nv=$(get_tp "$RESDIR/novec.txt"   "$hash" $P $n)
        av=$(get_tp "$RESDIR/autovec.txt" "$hash" $P $n)
        sp=$(speedup "$nv" "$av")
        results="$results $sp"
    done
    printf "%-18s %12s %12s %12s\n" "$hash" $results
done

echo ""
echo ""
echo "================================================================"
echo "  TABLE 4: Distribution quality — N=100M, P=256"
echo "  (from autovec.txt — distribution is identical across all versions)"
echo "================================================================"
echo ""

CSV_FILE="$RESDIR/autovec.txt"
[ -f "$CSV_FILE" ] || CSV_FILE="$RESDIR/novec.txt"

if [ -f "$CSV_FILE" ]; then
    printf "%-18s %12s %12s\n" "Hash" "chi2" "CV"
    printf "%-18s %12s %12s\n" "------------------" "----------" "----------"
    for hash in $HASHES; do

        chi2=$(awk -v h="$hash" -v p="$P" -v n="$N" '
            /N =/ { in_n = ($0 ~ n) ? 1 : 0; next }
            in_n && /^\xe2\x94\x82/ {
                gsub(/^\xe2\x94\x82|\xe2\x94\x82$/, "")
                split($0, cols, "│")
                pval = cols[1]+0
                hval = cols[2]; gsub(/[[:space:]]/, "", hval)
                if (pval == p+0 && hval == h) {
                    val = cols[6]; gsub(/[[:space:]]/, "", val)
                    print val; exit
                }
            }' "$CSV_FILE")

        cv=$(awk -v h="$hash" -v p="$P" -v n="$N" '
            /N =/ { in_n = ($0 ~ n) ? 1 : 0; next }
            in_n && /^\xe2\x94\x82/ {
                gsub(/^\xe2\x94\x82|\xe2\x94\x82$/, "")
                split($0, cols, "│")
                pval = cols[1]+0
                hval = cols[2]; gsub(/[[:space:]]/, "", hval)
                if (pval == p+0 && hval == h) {
                    val = cols[7]; gsub(/[[:space:]]/, "", val)
                    print val; exit
                }
            }' "$CSV_FILE")
        printf "%-18s %12s %12s\n" "$hash" "${chi2:-N/A}" "${cv:-N/A}"
    done
else
    echo "  (No result files found — run the benchmarks first)"
fi

echo ""
echo ""
echo "================================================================"
echo "  TABLE 5: CUDA timing breakdown — N=100M, P=256"
echo "================================================================"
echo ""
if [ -f "$RESDIR/cuda.txt" ]; then
    printf "%-18s %9s %9s %9s %9s %10s %10s\n" \
           "Hash" "H2D(ms)" "ker(ms)" "D2H(ms)" "tot(ms)" "Mkeys/s" "ker_only"
    printf "%-18s %9s %9s %9s %9s %10s %10s\n" \
           "------------------" "-------" "-------" "-------" "-------" "--------" "--------"
    for hash in $HASHES; do
        h2d=$(  get_cuda "$RESDIR/cuda.txt" "$hash" $P $N 3)
        ker=$(  get_cuda "$RESDIR/cuda.txt" "$hash" $P $N 4)
        d2h=$(  get_cuda "$RESDIR/cuda.txt" "$hash" $P $N 5)
        tot=$(  get_cuda "$RESDIR/cuda.txt" "$hash" $P $N 6)
        tp_tot=$(get_cuda "$RESDIR/cuda.txt" "$hash" $P $N 7)
        tp_ker=$(get_cuda "$RESDIR/cuda.txt" "$hash" $P $N 8)
        printf "%-18s %9s %9s %9s %9s %10s %10s\n" \
               "$hash" "${h2d:-N/A}" "${ker:-N/A}" "${d2h:-N/A}" \
               "${tot:-N/A}" "${tp_tot:-N/A}" "${tp_ker:-N/A}"
    done
else
    echo "  (No cuda.txt found — run task3 first)"
fi

echo ""
echo "Done. Raw data in: $RESDIR/"