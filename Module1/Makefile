CXX      = g++
CXXSTD   = -std=c++17
WARN     = -Wall -Wextra
NVCC     = /usr/local/cuda-12.3/bin/nvcc

SRCDIR   = src
BINDIR   = bin
RESDIR   = results

# Compiler flags
FLAGS_NOVEC       = $(CXXSTD) $(WARN) -O3 -fno-tree-vectorize -fno-tree-slp-vectorize
FLAGS_AUTOVEC     = $(CXXSTD) $(WARN) -O3 -march=native -ftree-vectorize -fopt-info-vec-optimized
FLAGS_AVX2        = $(CXXSTD) $(WARN) -O3 -march=native -mavx2
NVCC_FLAGS        = -O3 -arch=sm_80 -std=c++17

# Headers
HDRS = $(SRCDIR)/common.h $(SRCDIR)/kernels_plain.h

.PHONY: all task1 task2 task3 clean help

all: task1 task2 task3

# Task 1: 2 binaries from same source
#   novec   → scalar baseline  (-fno-tree-vectorize)
#   autovec → GCC auto-vectorized (-O3 -march=native)
task1: $(BINDIR)/task1_novec $(BINDIR)/task1_autovec

$(BINDIR)/task1_novec: $(SRCDIR)/task1_main.cpp $(HDRS) | $(BINDIR) $(RESDIR)
	$(CXX) $(FLAGS_NOVEC) -o $@ $<

$(BINDIR)/task1_autovec: $(SRCDIR)/task1_main.cpp $(HDRS) | $(BINDIR) $(RESDIR)
	$(CXX) $(FLAGS_AUTOVEC) -o $@ $< 2> $(RESDIR)/autovec_report.txt


# Task 2: AVX2 intrinsics
task2: $(BINDIR)/task2_avx2

$(BINDIR)/task2_avx2: $(SRCDIR)/task2_main.cpp $(SRCDIR)/common.h $(SRCDIR)/kernels_plain.h | $(BINDIR) $(RESDIR)
	$(CXX) $(FLAGS_AVX2) -o $@ $<


# Task 3: CUDA
task3: $(BINDIR)/task3_cuda

$(BINDIR)/task3_cuda: $(SRCDIR)/task3_main.cu | $(BINDIR) $(RESDIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<


# Directories
$(BINDIR):
	mkdir -p $(BINDIR)
$(RESDIR):
	mkdir -p $(RESDIR)


# GCC vectorization report
vec-report: | $(BINDIR) $(RESDIR)
	@echo "=== Vectorization report for kernels_plain.h ==="
	$(CXX) $(CXXSTD) $(WARN) -O3 -march=native -ftree-vectorize \
		-fopt-info-vec-all -o $(BINDIR)/task1_report $(SRCDIR)/task1_main.cpp \
		2> $(RESDIR)/vec_report.txt
	@grep "kernels_plain.h.*optimized" $(RESDIR)/vec_report.txt 2>/dev/null || echo "(none vectorized)"

clean:
	rm -rf $(BINDIR) $(RESDIR)

help:
	@echo "=== Task 1 — Plain C++ ==="
	@echo "  task1_novec     scalar baseline"
	@echo "  task1_autovec   GCC auto-vectorized"
	@echo ""
	@echo "=== Task 2 — AVX2 intrinsics ==="
	@echo "  task2_avx2      hand-written AVX2 SIMD kernels"
	@echo ""
	@echo "=== Task 3 — CUDA ==="
	@echo "  task3_cuda      CUDA GPU kernel"
	@echo ""
	@echo "  make all          Build everything"
	@echo "  make vec-report   GCC vectorization evidence for kernels_plain.h"
	@echo "  make clean        Remove bin/ and results/"