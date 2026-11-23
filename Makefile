# Full pipeline: make all   (data/all/dataset.parquet must exist; see notes/data.md)
PY ?= /usr/bin/python3
RUN = systemd-inhibit --what=sleep:idle --who=deep-causality --why=pipeline

.PHONY: all prepare discover evaluate test test-rust test-python clean

all: prepare discover evaluate

prepare:
	$(PY) scripts/prepare_data.py --data data/all/dataset.parquet --out data/split

discover:
	cd backend && cargo build --release && $(RUN) ../target/release/icu_sepsis_backend --config ../config/default.toml

evaluate:
	$(RUN) $(PY) scripts/evaluate_detection.py --split data/split --discovery results/causal_discovery.json --out results
	$(PY) scripts/plot_results.py
	$(PY) scripts/plot_surd.py

test: test-rust test-python

test-rust:
	cargo test -p icu_sepsis_backend --locked

test-python:
	$(PY) -m pytest -q tests

clean:
	cargo clean
