#!/bin/bash
# Build script for causers package

set -e

echo "Building causers package..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf target/
rm -rf build/
rm -rf *.egg-info/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Build in release mode
echo "Building Rust extension in release mode..."
maturin build --release

# For development install
if [ "$1" == "--dev" ]; then
    echo "Installing in development mode..."
    maturin develop --release
    echo "Development build complete!"
else
    echo "Build complete! Wheel file created in target/wheels/"
    ls -la target/wheels/
fi