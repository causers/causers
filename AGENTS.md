# AI Agent Operating Manual for causers

## Build & Test Commands

### Prerequisites
```bash
# Install required tools
pip install maturin polars pytest numpy

# Ensure Rust is installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build Commands
```bash
# Development build (from project root)
maturin develop

# Release build (optimized, from project root)
maturin develop --release

# Build wheel for distribution (from project root)
maturin build --release
# Output: target/wheels/*.whl
```

### Test Commands
```bash
# Run Rust tests (from project root)
cargo test

# Run Python tests (from project root, after building)
pytest tests/

# Run with coverage
pytest tests/ --cov=causers --cov-report=html

# Run specific test file
pytest tests/test_linear_regression.py -v
```

### Lint/Format Commands
```bash
# Format Python code
black python/ tests/

# Lint Python code
ruff check python/ tests/

# Type check Python code
mypy python/

# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy
```

## Codebase Topology

```
causers/
├── src/                    # Rust source code (DO NOT modify without understanding PyO3)
│   ├── lib.rs             # Main module entry, PyO3 bindings
│   └── stats.rs           # Statistical implementations
├── python/                 # Python package
│   └── causers/
│       └── __init__.py    # Python API surface
├── tests/                  # Python tests only
├── scripts/               # Build and utility scripts
├── target/                # Build artifacts (AUTO-GENERATED, DO NOT EDIT)
├── Cargo.toml             # Rust dependencies and configuration
├── pyproject.toml         # Python package configuration
└── README.md              # User documentation
```

### Key Areas
- **Rust Core** (`src/`): Performance-critical statistical computations
- **Python API** (`python/causers/`): User-facing Python interface
- **Tests** (`tests/`): Python integration tests only

### Do Not Touch
- `target/` - Build artifacts, regenerated on each build
- `*.so`, `*.dylib`, `*.dll` - Compiled extension modules
- `__pycache__/` - Python bytecode cache

## Conventions

### Python Naming
- Functions: `snake_case` (e.g., `linear_regression`)
- Classes: `PascalCase` (e.g., `LinearRegressionResult`)
- Constants: `UPPER_SNAKE_CASE`
- Private functions: `_leading_underscore`

### Rust Naming
- Functions: `snake_case`
- Structs: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Module names: `snake_case`

### Error Handling
- Rust: Use `PyResult<T>` for functions exposed to Python
- Rust: Convert errors to appropriate Python exceptions
- Python: Let Rust exceptions bubble up with clear messages
- Never silently ignore errors

### Testing
- Every public function needs a test
- Test edge cases (empty data, mismatched sizes, etc.)
- Use `pytest` for Python tests
- Use `cargo test` for Rust unit tests
- Integration tests go in `tests/` directory

### Documentation
- All public APIs must have docstrings
- Use type hints in Python code
- Document PyO3 bindings clearly
- Include examples in docstrings

## Safe-Change Policy

### Preferred Approach
1. Make minimal, targeted changes
2. Add new functions rather than modifying existing ones
3. Preserve backward compatibility
4. Test changes incrementally

### When Modifying Rust Code
1. Understand PyO3 conventions first
2. Ensure type conversions are correct
3. Test with `maturin develop` before committing
4. Check that Python tests still pass

### When Adding Features
1. Implement in Rust (`src/stats.rs`)
2. Expose via PyO3 (`src/lib.rs`)
3. Add Python wrapper if needed (`python/causers/__init__.py`)
4. Add comprehensive tests (`tests/`)
5. Update documentation

## Dependency Policy

### Adding Rust Dependencies
1. Add to `Cargo.toml` with exact version
2. Prefer crates with:
   - Active maintenance
   - Compatible licenses (MIT/Apache)
   - Minimal transitive dependencies
3. Run `cargo tree` to check dependency graph

### Adding Python Dependencies
1. Add to `pyproject.toml` dependencies
2. Use version ranges for compatibility
3. Keep development dependencies in `[project.optional-dependencies]`
4. Test with minimum supported versions

## Release Workflow

### Local Development
1. Make changes
2. Run `maturin develop`
3. Test locally with `pytest`
4. Format with `black` and `cargo fmt`

### Pre-Release Checklist
1. Update version in `Cargo.toml` and `pyproject.toml`
2. Run full test suite: `cargo test && pytest tests/`
3. Build release wheel: `maturin build --release`
4. Test wheel installation in clean environment

### Publishing
```bash
# Build release artifacts
maturin build --release

# Upload to PyPI (requires credentials)
maturin publish
```

## Known Sharp Edges

### Common Issues
1. **Import Error**: "No module named 'causers._causers'"
   - Solution: Run `maturin develop` first
   
2. **Rust Compilation Fails**: Usually missing Rust toolchain
   - Solution: Install Rust via rustup
   
3. **Polars Version Mismatch**: Incompatible Polars versions
   - Solution: Ensure Python and Rust use compatible versions
   
4. **Memory Issues**: Large DataFrames may cause issues
   - Solution: Process in chunks or upgrade to 64-bit Python

### Development Tips
- Always run `maturin develop` after Rust changes
- Use `--release` flag for performance testing
- Check `target/` size regularly (can grow large)
- Clean builds with `cargo clean` if issues persist

## Platform-Specific Notes

### macOS
- May need to set `MACOSX_DEPLOYMENT_TARGET`
- Use `cargo build --target` for cross-compilation

### Linux
- Ensure `python3-dev` is installed
- May need to set `LD_LIBRARY_PATH`

### Windows
- Use MSVC toolchain (not MinGW)
- May need Visual Studio Build Tools

## Quick Reference

```bash
# Most common commands
maturin develop --release  # Build and install for development
pytest tests/ -v           # Run tests
cargo fmt && black .       # Format all code
maturin build --release    # Build distribution wheel