.PHONY: help build develop test clean format lint install

help:
	@echo "Available commands:"
	@echo "  make install   - Install package in development mode"
	@echo "  make build     - Build release version"
	@echo "  make develop   - Build and install for development"
	@echo "  make test      - Run all tests"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make format    - Format all code"
	@echo "  make lint      - Run linters"

install:
	pip install -e ".[dev]"
	maturin develop

build:
	maturin build --release

develop:
	maturin develop --release

test:
	cargo test
	pytest tests/ -v

clean:
	cargo clean
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.so" -delete
	find . -type f -name "*.pyc" -delete

format:
	cargo fmt
	black python/ tests/ examples/

lint:
	cargo clippy
	ruff check python/ tests/ examples/
	mypy python/

test-rust:
	cargo test

test-python:
	pytest tests/ -v

wheel:
	maturin build --release
	@echo "Wheel built in target/wheels/"