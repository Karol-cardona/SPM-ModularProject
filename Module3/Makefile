# Makefile for SPM Modulo 3
#
# Builds four implementations of the partitioned hash join:
#   hashjoin_seq        sequential baseline (Module 2 baseline)
#   hashjoin_par        C++ std::thread parallel (Module 2 v3)
#   hashjoin_omp_loop   OpenMP loop-level — schedule(runtime), policy
#                       selected via OMP_SCHEDULE env var
#                       (static, dynamic, or guided)
#   hashjoin_omp_task   OpenMP task-based (Module 3)
#
# Targets:
#   make all       build all binaries
#   make check     run small correctness tests (uniform + skewed)
#   make clean     remove binaries and intermediate files
#

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -march=native
LDFLAGS  ?=

# Threading libraries
PTHREAD  := -pthread
OPENMP   := -fopenmp

BINARIES := hashjoin_seq hashjoin_par hashjoin_omp_loop hashjoin_omp_task

.PHONY: all clean check check_uniform check_skewed

all: $(BINARIES)

hashjoin_seq: hashjoin_seq.cpp common.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

hashjoin_par: hashjoin_par.cpp common.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(PTHREAD) -o $@ $<

hashjoin_omp_loop: hashjoin_omp_loop.cpp common.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OPENMP) -o $@ $<

hashjoin_omp_task: hashjoin_omp_task.cpp common.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OPENMP) -o $@ $<

# ------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------
#
# Two scenarios:
#   - uniform  (skew = 0.0): standard input
#   - skewed   (skew = 0.9): 90% of records routed to 4 hot partitions
#
# Tiny inputs (NR, NS <= 500) trigger the brute-force O(|R|*|S|) verifier
# inside each binary, which prints naive_join_count / naive_checksum*.
# We then check that join_count and the two checksums match across all
# implementations and the naive verifier.
#
# For hashjoin_omp_loop we explicitly set OMP_SCHEDULE=dynamic,1: any
# scheduling policy must be correct, but dynamic exercises the runtime
# scheduling path more aggressively, so it's a stronger correctness test.
#

check: check_uniform check_skewed
	@echo ""
	@echo "All correctness checks PASSED."

check_uniform: $(BINARIES)
	@echo "=== Uniform correctness check (NR=200, NS=300, skew=0) ==="
	@./hashjoin_seq      -nr 200 -ns 300 -seed 42 -max-key 100 -p 16        > /tmp/seq_u.out  2>/dev/null
	@./hashjoin_par      -nr 200 -ns 300 -seed 42 -max-key 100 -p 16 -t 4   > /tmp/par_u.out  2>/dev/null
	@./hashjoin_omp_task -nr 200 -ns 300 -seed 42 -max-key 100 -p 16 -t 4   > /tmp/task_u.out 2>/dev/null
	@OMP_SCHEDULE=dynamic,1 ./hashjoin_omp_loop -nr 200 -ns 300 -seed 42 -max-key 100 -p 16 -t 4 > /tmp/loop_u.out 2>/dev/null
	@for tag in seq par loop task; do \
	    jc=$$(grep '^join_count='  /tmp/$${tag}_u.out | cut -d= -f2); \
	    c1=$$(grep '^checksum1='   /tmp/$${tag}_u.out | cut -d= -f2); \
	    c2=$$(grep '^checksum2='   /tmp/$${tag}_u.out | cut -d= -f2); \
	    nj=$$(grep '^naive_join_count=' /tmp/$${tag}_u.out | cut -d= -f2); \
	    n1=$$(grep '^naive_checksum1='  /tmp/$${tag}_u.out | cut -d= -f2); \
	    n2=$$(grep '^naive_checksum2='  /tmp/$${tag}_u.out | cut -d= -f2); \
	    if [ "$$jc" != "$$nj" ] || [ "$$c1" != "$$n1" ] || [ "$$c2" != "$$n2" ]; then \
	      echo "  $$tag : MISMATCH vs naive verifier"; exit 1; \
	    fi; \
	    echo "  $$tag : join_count=$$jc checksum1=$$c1 checksum2=$$c2 (matches naive)"; \
	done

check_skewed: $(BINARIES)
	@echo "=== Skewed correctness check (NR=200, NS=300, skew=0.9) ==="
	@./hashjoin_seq      -nr 200 -ns 300 -seed 42 -max-key 100 -p 16        -skew 0.9 > /tmp/seq_s.out  2>/dev/null
	@./hashjoin_par      -nr 200 -ns 300 -seed 42 -max-key 100 -p 16 -t 4   -skew 0.9 > /tmp/par_s.out  2>/dev/null
	@./hashjoin_omp_task -nr 200 -ns 300 -seed 42 -max-key 100 -p 16 -t 4   -skew 0.9 > /tmp/task_s.out 2>/dev/null
	@OMP_SCHEDULE=dynamic,1 ./hashjoin_omp_loop -nr 200 -ns 300 -seed 42 -max-key 100 -p 16 -t 4 -skew 0.9 > /tmp/loop_s.out 2>/dev/null
	@for tag in seq par loop task; do \
	    jc=$$(grep '^join_count='  /tmp/$${tag}_s.out | cut -d= -f2); \
	    c1=$$(grep '^checksum1='   /tmp/$${tag}_s.out | cut -d= -f2); \
	    c2=$$(grep '^checksum2='   /tmp/$${tag}_s.out | cut -d= -f2); \
	    nj=$$(grep '^naive_join_count=' /tmp/$${tag}_s.out | cut -d= -f2); \
	    n1=$$(grep '^naive_checksum1='  /tmp/$${tag}_s.out | cut -d= -f2); \
	    n2=$$(grep '^naive_checksum2='  /tmp/$${tag}_s.out | cut -d= -f2); \
	    if [ "$$jc" != "$$nj" ] || [ "$$c1" != "$$n1" ] || [ "$$c2" != "$$n2" ]; then \
	      echo "  $$tag : MISMATCH vs naive verifier"; exit 1; \
	    fi; \
	    echo "  $$tag : join_count=$$jc checksum1=$$c1 checksum2=$$c2 (matches naive)"; \
	done

clean:
	rm -f $(BINARIES) /tmp/seq_*.out /tmp/par_*.out /tmp/loop_*.out /tmp/task_*.out