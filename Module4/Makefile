# Makefile for SPM Modulo 4 — Distributed MPI Implementation
#
# Builds:
#   hashjoin_seq        Sequential baseline (same as Modules 2-3)
#   hashjoin_mpi        Pure MPI distributed implementation
#   hashjoin_mpi_omp    Hybrid MPI+OpenMP (same source, built by `make hybrid`)
#
# Targets:
#   make all         build pure-MPI and sequential binaries
#   make hybrid      also build the MPI+OpenMP binary
#   make check       small-input correctness (uniform + skewed)
#   make clean       remove binaries

CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -march=native
OPENMP   := -fopenmp

BINARIES := hashjoin_seq hashjoin_mpi

.PHONY: all hybrid clean check check_uniform check_skewed

all: $(BINARIES)

hybrid: hashjoin_mpi_omp

hashjoin_seq: hashjoin_seq.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

hashjoin_mpi: hashjoin_mpi.cpp common.hpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

hashjoin_mpi_omp: hashjoin_mpi.cpp common.hpp
	$(MPICXX) $(CXXFLAGS) $(OPENMP) -o $@ $<

# ----------------------------------------------------------------------------
# Correctness check (small inputs, runs locally with mpirun --oversubscribe).
# The MPI binary's -check flag makes rank 0 recompute the result with the
# embedded sequential reference (common.hpp) and assert equality.
# ----------------------------------------------------------------------------
MPIRUN ?= mpirun --oversubscribe

check: check_uniform check_skewed
	@echo "All correctness checks PASSED."

check_uniform: hashjoin_mpi
	@echo "=== Uniform correctness check (NR=10000 NS=15000 P=64) ==="
	@for NP in 2 4; do \
		printf "  np=%s : " $$NP; \
		$(MPIRUN) -n $$NP ./hashjoin_mpi -nr 10000 -ns 15000 -seed 13 \
		    -max-key 500 -p 64 -check 1 2>/dev/null \
		    | grep -E '^check=' || { echo "FAIL"; exit 1; }; \
	done

check_skewed: hashjoin_mpi
	@echo "=== Skewed correctness check (NR=20000 NS=30000 P=64 skew=0.9 hot=4) ==="
	@for NP in 2 4; do \
		printf "  np=%s : " $$NP; \
		$(MPIRUN) -n $$NP ./hashjoin_mpi -nr 20000 -ns 30000 -seed 13 \
		    -max-key 2000 -p 64 -skew 0.9 -hot 4 -check 1 2>/dev/null \
		    | grep -E '^check=' || { echo "FAIL"; exit 1; }; \
	done

clean:
	rm -f hashjoin_seq hashjoin_mpi hashjoin_mpi_omp
