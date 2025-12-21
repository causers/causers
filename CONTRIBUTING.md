# Contributing to causers

Thank you for your interest in contributing to causers! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

1. **Python 3.8+**: Install from [python.org](https://python.org)
2. **Rust 1.70+**: Install via [rustup](https://rustup.rs/)
3. **Git**: For version control

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/causers.git
cd causers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Build the Rust extension
maturin develop
```

## Development Workflow

### Making Changes

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Edit code, add tests
3. **Test locally**: Run test suite
4. **Format code**: Apply formatters
5. **Commit**: Write clear commit messages
6. **Push**: Push to your fork
7. **PR**: Open pull request

### Code Style

#### Python
- Use `black` for formatting: `black python/ tests/`
- Use `ruff` for linting: `ruff check python/ tests/`
- Add type hints to all functions
- Write docstrings for public APIs

#### Rust
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Follow Rust naming conventions
- Document public functions

### Testing

#### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run Rust tests only
cargo test

# Run Python tests only
pytest tests/

# Run specific test
pytest tests/test_linear_regression.py::TestLinearRegression::test_perfect_linear_relationship
```

#### Writing Tests

- Test both success and error cases
- Use descriptive test names
- Test edge cases (empty data, NaN, infinity)
- Aim for >90% code coverage

### Building

```bash
# Development build
maturin develop

# Release build
maturin develop --release

# Build wheel
maturin build --release
```

## Project Structure

```
causers/
├── src/                    # Rust source code
│   ├── lib.rs             # Module definitions and PyO3 bindings
│   └── stats.rs           # Statistical implementations
├── python/                 # Python package
│   └── causers/
│       └── __init__.py    # Python API
├── tests/                  # Python tests
├── examples/              # Example scripts
└── scripts/               # Build/test scripts
```

## Adding New Features

### 1. Design Phase
- Discuss in issue before implementing
- Consider API design carefully
- Think about performance implications

### 2. Implementation
- Start with Rust implementation in `src/`
- Add PyO3 bindings in `src/lib.rs`
- Create Python wrapper if needed
- Write comprehensive tests

### 3. Documentation
- Add docstrings with examples
- Update README if needed
- Add example script to `examples/`

## Pull Request Process

### Before Submitting

1. **Tests Pass**: All tests must pass
2. **Code Formatted**: Run formatters
3. **Documentation**: Update relevant docs
4. **Changelog**: Add entry if applicable

### PR Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what and why
- **Testing**: Describe testing done
- **Breaking Changes**: Clearly marked

### Review Process

- PRs need at least one approval
- Address all review comments
- Keep PR focused on single feature/fix
- Rebase on main if needed

## Common Tasks

### Adding a New Statistical Function

1. Implement in `src/stats.rs`:
```rust
pub fn my_function(data: &[f64]) -> PyResult<f64> {
    // Implementation
}
```

2. Expose in `src/lib.rs`:
```rust
#[pyfunction]
fn my_function_py(py: Python, data: PyObject) -> PyResult<f64> {
    // Convert and call Rust function
}
```

3. Add to module:
```rust
m.add_function(wrap_pyfunction!(my_function_py, m)?)?;
```

4. Test in `tests/test_my_function.py`

### Debugging

```bash
# Build with debug symbols
maturin develop --profile dev

# Use Python debugger
python -m pdb examples/basic_regression.py

# Use Rust debugger
rust-gdb target/debug/deps/causers-*
```

## Release Process

1. Update version in `Cargo.toml` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Build release: `maturin build --release`
5. Upload to PyPI: `maturin publish`

## Getting Help

- **Issues**: Open an issue for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check docs first

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive criticism
- No harassment or discrimination

## License

By contributing, you agree that your contributions will be licensed under the MIT License.