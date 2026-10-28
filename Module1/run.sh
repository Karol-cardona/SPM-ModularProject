#!/bin/bash
#SBATCH --job-name=m1
#SBATCH --partition=gpu-excl
#SBATCH --nodelist=node09
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -e
cd ~/Module1
mkdir -p bin results

export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

echo "=== Node: $(hostname) | Date: $(date) ==="
echo "GCC: $(g++ --version | head -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
echo ""

PIN="numactl --cpunodebind=0 --membind=0"

echo "=== BUILD ==="
make clean && make all 2>&1
echo ""

# Task 1
echo "=== TASK 1: NOVEC ==="
$PIN ./bin/task1_novec --label novec --out results/novec.txt
echo ""

echo "=== TASK 1: AUTOVEC ==="
$PIN ./bin/task1_autovec --label autovec --out results/autovec.txt
echo ""

# Task 2: AVX2
echo "=== TASK 2: AVX2 ==="
$PIN ./bin/task2_avx2 --label avx2 --out results/avx2.txt
echo ""

# Task 3: CUDA
echo "=== TASK 3: CUDA ==="
./bin/task3_cuda --out results/cuda.txt
echo ""

# GCC report
echo "=== GCC VECTORIZATION ==="
make vec-report 2>&1
echo ""

echo "=== DONE at $(date) ==="
ls -lh results/
