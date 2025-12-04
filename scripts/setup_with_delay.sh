#!/bin/bash
set -e

echo "Installing dependencies..."
uv pip install -r requirements.txt --python .venv

echo "Installation command finished. Waiting 45 seconds as requested to ensure filesystem consistency..."
sleep 45

echo "Verifying installation..."
.venv/bin/pip list

echo "Running verification script..."
.venv/bin/python scripts/verify_simple.py
