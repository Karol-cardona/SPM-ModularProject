#!/bin/bash
#SBATCH --job-name=hashjoin
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
#SBATCH --output=output_%j.txt
#SBATCH --error=output_%j.txt

cd $HOME/Module2

# compile all binaries on the compute node
g++ -std=c++17 -O3 -march=native -pthread -o hashjoin_seq    hashjoin_seq.cpp
g++ -std=c++17 -O3 -march=native -pthread -o hashjoin_par_v1 hashjoin_par_v1.cpp
g++ -std=c++17 -O3 -march=native -pthread -o hashjoin_par_v2 hashjoin_par_v2.cpp
g++ -std=c++17 -O3 -march=native -pthread -o hashjoin_par_v3 hashjoin_par_v3.cpp

echo "=== SEQUENTIAL ==="
numactl --cpunodebind=0 --membind=0 \
    ./hashjoin_seq -nr 50000000 -ns 80000000 -seed 13 -max-key 1000000 -p 256

echo ""
echo "=== STRONG SCALING — v1 (parallel join only) ==="
for T in 1 2 4 8 16; do
    echo "--- T=$T ---"
    numactl --cpunodebind=0 --membind=0 \
        ./hashjoin_par_v1 -nr 50000000 -ns 80000000 \
        -seed 13 -max-key 1000000 -p 256 -t $T
    echo ""
done

echo ""
echo "=== STRONG SCALING — v2 (parallel join + histogram) ==="
for T in 1 2 4 8 16; do
    echo "--- T=$T ---"
    numactl --cpunodebind=0 --membind=0 \
        ./hashjoin_par_v2 -nr 50000000 -ns 80000000 \
        -seed 13 -max-key 1000000 -p 256 -t $T
    echo ""
done

echo ""
echo "=== STRONG SCALING — v3 (fully parallel) ==="
for T in 1 2 4 8 16; do
    echo "--- T=$T ---"
    numactl --cpunodebind=0 --membind=0 \
        ./hashjoin_par_v3 -nr 50000000 -ns 80000000 \
        -seed 13 -max-key 1000000 -p 256 -t $T
    echo ""
done

echo ""
echo "=== WEAK SCALING — v3 (final version) ==="
echo "--- T=1  NR=10M  NS=16M ---"
numactl --cpunodebind=0 --membind=0 \
    ./hashjoin_par_v3 -nr 10000000 -ns 16000000 \
    -seed 13 -max-key 1000000 -p 256 -t 1

echo "--- T=2  NR=20M  NS=32M ---"
numactl --cpunodebind=0 --membind=0 \
    ./hashjoin_par_v3 -nr 20000000 -ns 32000000 \
    -seed 13 -max-key 1000000 -p 256 -t 2

echo "--- T=4  NR=40M  NS=64M ---"
numactl --cpunodebind=0 --membind=0 \
    ./hashjoin_par_v3 -nr 40000000 -ns 64000000 \
    -seed 13 -max-key 1000000 -p 256 -t 4

echo "--- T=8  NR=80M  NS=128M ---"
numactl --cpunodebind=0 --membind=0 \
    ./hashjoin_par_v3 -nr 80000000 -ns 128000000 \
    -seed 13 -max-key 1000000 -p 256 -t 8

echo "--- T=16  NR=160M  NS=256M ---"
numactl --cpunodebind=0 --membind=0 \
    ./hashjoin_par_v3 -nr 160000000 -ns 256000000 \
    -seed 13 -max-key 1000000 -p 256 -t 16