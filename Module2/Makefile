CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -pthread -Wall -Wextra

.PHONY: all clean check

all: hashjoin_seq hashjoin_par_v1 hashjoin_par_v2 hashjoin_par_v3

hashjoin_seq: hashjoin_seq.cpp
	$(CXX) $(CXXFLAGS) -o $@ $

hashjoin_par_v1: hashjoin_par_v1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $

hashjoin_par_v2: hashjoin_par_v2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $

hashjoin_par_v3: hashjoin_par_v3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $

check: hashjoin_seq hashjoin_par_v1 hashjoin_par_v2 hashjoin_par_v3
	@echo "=== Correctness check ==="
	@SEQ=$$(./hashjoin_seq -nr 50 -ns 80 -seed 42 -max-key 20 -p 4 \
	        | grep -E '^(join_count|checksum)'); \
	 for PAR in ./hashjoin_par_v1 ./hashjoin_par_v2 ./hashjoin_par_v3; do \
	     OUT=$$($$PAR -nr 50 -ns 80 -seed 42 -max-key 20 -p 4 -t 4 \
	             | grep -E '^(join_count|checksum)'); \
	     if [ "$$SEQ" = "$$OUT" ]; then \
	         echo "PASS: seq == $$PAR"; \
	     else \
	         echo "FAIL: $$PAR"; echo "SEQ: $$SEQ"; echo "GOT: $$OUT"; exit 1; \
	     fi \
	 done

clean:
	rm -f hashjoin_seq hashjoin_par_v1 hashjoin_par_v2 hashjoin_par_v3