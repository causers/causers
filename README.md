# causers

[![PyPI Version](https://img.shields.io/pypi/v/causers)](https://pypi.org/project/causers/)
[![Python Versions](https://img.shields.io/pypi/pyversions/causers)](https://pypi.org/project/causers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Coverage: 100%](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)](https://github.com/causers/causers)
[![Documentation Status](https://readthedocs.org/projects/causers/badge/?version=latest)](https://causers.readthedocs.io/en/latest/?badge=latest)

A high-performance statistical package for Polars (and pandas) DataFrames, powered by Rust.

## 🚀 Overview

`causers` provides blazing-fast statistical operations for both Polars and pandas DataFrames, leveraging Rust's performance through PyO3 bindings. Designed for data scientists and analysts who need production-grade performance without sacrificing ease of use.

### ✨ Key Features

- **🏎️ High Performance**: Linear regression on 1M rows in ~250ms with HC3 standard errors
- **📊 Multiple Regression**: Support for multiple covariates with matrix-based OLS
- **🔮 Logistic Regression**: Binary outcome regression with Newton-Raphson MLE, fixed effects via Mundlak strategy
- **📈 Robust Standard Errors**: HC3 heteroskedasticity-consistent standard errors included
- **🎯 Flexible Models**: Optional intercept for fully saturated models
- **🏢 Clustered Standard Errors**: Cluster-robust SE for panel/grouped data
- **🔄 Bootstrap Methods**: Wild cluster bootstrap (linear) and score bootstrap (logistic)
- **📋 Two-Way Fixed Effects**: Panel data regression with entity and time fixed effects
- **🧪 Synthetic DID**: Synthetic Difference-in-Differences for causal inference with panel data
- **🎯 Synthetic Control**: Classic SC with 4 method variants (traditional, penalized, robust, augmented)
- **🔬 Two-Stage Least Squares (IV)**: Instrumental variables estimation for causal inference with endogeneity
- **🤖 Double Machine Learning**: Debiased/orthogonalized ML for causal inference with cross-fitting
- **🔧 Native Polars Integration**: Zero-copy operations on Polars DataFrames
- **🐼 pandas Support**: Seamlessly pass pandas DataFrames - automatic conversion with minimal overhead
- **🦀 Rust-Powered**: Core computations in Rust for maximum throughput
- **🐍 Pythonic API**: Clean, intuitive interface with full type hints
- **🌍 Cross-Platform**: Works on Linux, macOS (Intel/ARM), and Windows

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install causers

# With pandas support
pip install causers[pandas]
```

### From Source (Development)

```bash
# Prerequisites: Python 3.8+ and Rust 1.70+
git clone https://github.com/causers/causers.git
cd causers

# Install build dependencies
pip install "maturin>=1.4,<2.0" "polars>=0.52" numpy

# Build and install in development mode
maturin develop --release
```

## Quick Start

For comprehensive examples demonstrating all causers functions with realistic data, see the  notebook:

📓 **[examples/basic_examples.ipynb](examples/basic_examples.ipynb)**

The notebook includes:

| Function | Description |
|----------|-------------|
| `dml()` | Double Machine Learning |
| `linear_regression()` | OLS with clustered standard errors |
| `logistic_regression()` | Maximum likelihood with clustered SEs |
| `two_stage_least_squares()` | IV/2SLS for causal inference with endogeneity |
| `synthetic_control()` | Synthetic control method |
| `synthetic_did()` | Synthetic difference-in-differences |

## 🛠️ Development

### Prerequisites

- Python 3.8 or higher
- Rust 1.70 or higher
- Polars 0.52 or higher

### Building from Source

```bash
# Clone the repository
git clone https://github.com/causers/causers.git
cd causers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Build the Rust extension
maturin develop --release
```

### Running Tests

```bash
# Install test dependencies (required for running tests)
pip install -e ".[test]"

# Run all tests with coverage
pytest tests/ --cov=causers --cov-report=html

# Run specific test categories
pytest tests/test_performance.py -v  # Performance benchmarks
pytest tests/test_edge_cases.py -v   # Edge case handling

# Run Rust tests
cargo test
```

### Code Quality

```bash
# Format Python code
black python/ tests/

# Lint Python code
ruff check python/ tests/

# Type check
mypy python/

# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Polars](https://github.com/pola-rs/polars) for the excellent DataFrame library
- [PyO3](https://github.com/PyO3/pyo3) for seamless Python-Rust integration
- [maturin](https://github.com/PyO3/maturin) for simplified packaging

## 📚 Resources

- [Documentation](https://causers.readthedocs.io)
- [API Reference](https://causers.readthedocs.io/en/stable/api/causers.html)
- [GitHub Issues](https://github.com/causers/causers/issues)
- [Discussions](https://github.com/causers/causers/discussions)

## 🐛 Found a Bug?

Please [open an issue](https://github.com/causers/causers/issues/new) with:
- Minimal reproducible example
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

---

Made with ❤️ and 🦀 by the causers team