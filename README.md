# causers

[![PyPI Version](https://img.shields.io/pypi/v/causers)](https://pypi.org/project/causers/)
[![Python Versions](https://img.shields.io/pypi/pyversions/causers)](https://pypi.org/project/causers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/causers/badge/?version=latest)](https://causers.readthedocs.io/en/latest/?badge=latest)

`causers` is a statistical library for Python that implements regression and causal inference methods directly on Polars DataFrames. It is written in Rust to ensure efficient performance and memory safety.

## Purpose

Data scientists working with Polars often face a friction point when they need to run statistical models: they must convert their efficient Polars DataFrames into pandas or NumPy arrays to use libraries like `statsmodels` or `scikit-learn`. This conversion can be costly in terms of memory and time, especially for large datasets.

`causers` solves this by providing native statistical routines that operate directly on Polars data. It uses Rust's linear algebra capabilities to perform computations efficiently, supporting standard errors, fixed effects, and bootstrap methods without the overhead of data conversion.

## Installation

You can install `causers` via pip. Pre-built wheels are available for Linux, macOS, and Windows.

```bash
# Standard installation
pip install causers

# To include pandas support (if you need to pass pandas DataFrames)
pip install causers[pandas]
```

## Usage

Here is a practical example of running a linear regression with robust standard errors on a Polars DataFrame.

```python
import polars as pl
import causers

# Create a sample DataFrame
df = pl.DataFrame({
    "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "x2": [0.5, 0.5, 1.0, 1.0, 1.5, 1.5],
    "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.0],
    "group": [1, 1, 2, 2, 3, 3]
})

# Run OLS regression: y ~ x1 + x2
# Using HC3 robust standard errors by default
result = causers.linear_regression(df, x_cols=["x1", "x2"], y_col="y")

print(f"R²: {result.r_squared:.4f}")
for i, (coef, se) in enumerate(zip(result.coefficients, result.standard_errors)):
    print(f"x{i+1}: {coef:.4f} ± {se:.4f}")

# Run with cluster-robust standard errors
clustered_result = causers.linear_regression(
    df, x_cols=["x1", "x2"], y_col="y", cluster="group"
)
```

The library supports Python type hints, so your IDE should provide autocompletion for function arguments and result objects.

## Features

### Regression Models
*   **Linear Regression (OLS):** Supports single and multiple covariates.
*   **Logistic Regression:** Implemented via Newton-Raphson optimization for binary outcomes.
*   **Robust Inference:** HC3 heteroskedasticity-consistent standard errors are used by default for OLS.

### Panel Data & Fixed Effects
*   **Fixed Effects:** Absorb high-dimensional fixed effects (e.g., unit and time) efficiently.
*   **Clustered Standard Errors:** Compute cluster-robust standard errors for grouped data.
*   **Bootstrap Inference:** Implements **Wild Cluster Bootstrap** for linear models and **Score Bootstrap** for logistic models (recommended for small cluster counts).
*   **Mundlak Approach:** Supports fixed effects in logistic regression via the Mundlak transformation.

### Causal Inference
*   **Synthetic Difference-in-Differences (SDID):** Implements the Arkhangelsky et al. (2021) estimator with placebo bootstrap for inference.
*   **Synthetic Control (SC):** Includes Traditional, Penalized, Robust, and Augmented variants.
*   **Double Machine Learning (DML):** Debiased inference using cross-fitting (Chernozhukov et al., 2018).
*   **Instrumental Variables (2SLS):** Two-Stage Least Squares estimation for endogeneity correction.

### Diagnostics
*   **Covariate Balance (`balance_check`):** Computes group means, variances, standard deviations, Standardized Mean Differences (SMD), variance ratios, and effective sample sizes (ESS) for treatment vs. control groups. Supports weighted analysis (e.g., inverse-propensity weights), automatic categorical expansion, and boolean covariates.

### Performance
*   **Rust Core:** All heavy lifting (matrix factorization, optimization loops) happens in Rust.
*   **Parallelism:** Bootstrap methods (like Wild Cluster Bootstrap) utilize multi-threading via Rayon.
*   **Memory Efficiency:** Zero-copy data access where possible.

## Documentation

Full documentation, including API references and theoretical background for the implemented methods, is available at [causers.readthedocs.io](https://causers.readthedocs.io).

## Development

To build `causers` from source, you will need the Rust toolchain (cargo) and a Python environment.

```bash
# Clone the repository
git clone https://github.com/causers/causers.git
cd causers

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies and build the Rust extension
pip install -e ".[dev]"
maturin develop --release
```

### Running Tests

The test suite uses `pytest`.

```bash
# Run all tests
pytest tests/

# Run performance benchmarks (skipped by default)
pytest tests/test_performance.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
