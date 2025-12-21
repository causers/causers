#!/bin/bash
# Test script for causers package

set -e

echo "Running tests for causers package..."

# Run Rust tests
echo "Running Rust tests..."
cargo test

# Build development version
echo "Building development version..."
maturin develop

# Run Python tests
echo "Running Python tests..."
python -m pytest tests/ -v

echo "All tests passed!"