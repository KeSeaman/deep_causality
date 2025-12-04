.PHONY: all build test clean setup-python run-backend

all: build

build:
	cd backend && cargo build --release

test:
	cd backend && cargo test

run-backend:
	cd backend && cargo run --release

setup-python:
	uv venv .venv --python python3.12
	. .venv/bin/activate && uv pip install -r requirements.txt

clean:
	cd backend && cargo clean
	rm -rf .venv
