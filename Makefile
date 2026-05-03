BINARY = ./target/release/wcrs
TESTFILE = test_large.txt
TESTFILE_SIZE_MB = 1000

.PHONY: build bench systembench allbench check clean

build:
	cargo build --release

$(TESTFILE): test.txt
	@target_bytes=$$(($(TESTFILE_SIZE_MB) * 1024 * 1024)); \
	src_bytes=$$(wc -c < test.txt); \
	reps=$$((target_bytes / src_bytes + 1)); \
	echo "Generating $(TESTFILE) (~$(TESTFILE_SIZE_MB)MB, $$reps reps)..."; \
	for i in $$(seq 1 $$reps); do cat test.txt; done > $(TESTFILE)

testfile: $(TESTFILE)

bench: build $(TESTFILE)
	LC_ALL=C hyperfine --warmup 3 --runs 10 '$(BINARY) $(TESTFILE)'

allbench: build $(TESTFILE)
	LC_ALL=C hyperfine --warmup 3 --runs 10 'wc $(TESTFILE)' '$(BINARY) $(TESTFILE)'

systembench: $(TESTFILE)
	LC_ALL=C hyperfine --warmup 3 --runs 10 'wc $(TESTFILE)'

check: build $(TESTFILE)
	@echo "wc:   $$(LC_ALL=C wc test.txt)"
	@echo "wcrs: $$($(BINARY) test.txt)"
	@echo "wc:   $$(LC_ALL=C wc $(TESTFILE))"
	@echo "wcrs: $$($(BINARY) $(TESTFILE))"

clean:
	cargo clean
	rm -f $(TESTFILE)
