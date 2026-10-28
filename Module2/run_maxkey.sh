#!/bin/bash
#SBATCH --job-name=hashjoin_maxkey
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:5:00
#SBATCH --output=output_maxkey_%j.txt
#SBATCH --error=output_maxkey_%j.txt

cd $HOME/Module2

g++ -std=c++17 -O3 -march=native -pthread -o hashjoin_seq    hashjoin_seq.cpp
g++ -std=c++17 -O3 -march=native -pthread -o hashjoin_par_v3 hashjoin_par_v3.cpp

echo "=== IMPACT OF MAX-KEY (NR=50M, NS=80M, P=256, T=8) ==="

for MAXKEY in 100 10000 100000 1000000 10000000 50000000; do
    echo "--- max-key=$MAXKEY ---"
    echo -n "SEQ: "
    numactl --cpunodebind=0 --membind=0 \
        ./hashjoin_seq -nr 50000000 -ns 80000000 \
        -seed 13 -max-key $MAXKEY -p 256
    echo -n "PAR T=8: "
    numactl --cpunodebind=0 --membind=0 \
        ./hashjoin_par_v3 -nr 50000000 -ns 80000000 \
        -seed 13 -max-key $MAXKEY -p 256 -t 8
    echo ""
done