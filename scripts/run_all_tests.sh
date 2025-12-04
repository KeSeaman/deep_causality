#!/bin/bash
set -e

echo "Running Rust Unit Tests..."
cd backend
cargo test
cd ..

echo "Running Rust Backend..."
cd backend
cargo run --release
cd ..

echo "Running Python Verification..."
source .venv/bin/activate
python3 scripts/verify_simple.py

echo "All tests passed!"
