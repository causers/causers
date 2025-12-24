# Changelog

All notable changes to the causers project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-24

### ✨ Features

- **Clustered Standard Errors**: Cluster-robust standard errors for panel and grouped data
  - New `cluster` parameter: specify column containing cluster identifiers
  - Analytical clustered SE using sandwich estimator with small-sample adjustment
  - Matches statsmodels `get_robustcov_results(cov_type='cluster')` to `rtol=1e-6`

- **Wild Cluster Bootstrap**: Bootstrap-based standard errors for small cluster counts
  - New `bootstrap` parameter: enable wild cluster bootstrap
  - New `bootstrap_iterations` parameter: control number of replications (default: 1000)
  - New `seed` parameter: ensure reproducibility
  - Uses Rademacher weights (±1 with equal probability)
  - Matches wildboottest package results to `rtol=1e-2`

- **Small-Cluster Warning**: Automatic recommendation for bootstrap
  - Emits `UserWarning` when G < 42 clusters with analytical SE
  - Guides users toward more reliable inference methods

- **New Result Attributes**:
  - `n_clusters` (int | None): Number of unique clusters
  - `cluster_se_type` (str | None): "analytical" or "bootstrap"
  - `bootstrap_iterations_used` (int | None): Actual iterations used

### 🛠️ Technical Details

- New `src/cluster.rs` module for all clustering logic
- SplitMix64 PRNG for Rademacher weight generation (no external RNG dependency)
- Welford's online algorithm for O(1) memory bootstrap variance
- Condition number check (> 1e10) for numerical stability

### 📖 API Changes

**New Parameters:**
- `cluster: Optional[str] = None` — Column name for cluster identifiers
- `bootstrap: bool = False` — Enable wild cluster bootstrap
- `bootstrap_iterations: int = 1000` — Number of bootstrap replications
- `seed: Optional[int] = None` — Random seed for reproducibility

**New Result Fields:**
- `n_clusters` — Cluster count (None if not clustered)
- `cluster_se_type` — "analytical" or "bootstrap" (None if not clustered)
- `bootstrap_iterations_used` — Iterations used (None if not bootstrap)

**New Errors:**
- `ValueError`: "bootstrap=True requires cluster to be specified"
- `ValueError`: "Clustered standard errors require at least 2 clusters"
- `ValueError`: "Cluster column contains null values"

**New Warnings:**
- `UserWarning`: "Only N clusters detected. Wild cluster bootstrap is recommended when clusters < 42."
- `UserWarning`: "Cluster column 'X' is float; will be cast to string for grouping."

### 📊 Performance

With clustered SE computation:
- Analytical clustered SE: ≤2× HC3 baseline runtime
- Bootstrap (B=1000) on 100K rows: ~3-5 seconds

### ⚠️ Breaking Changes

None. All existing code continues to work unchanged.

### 📦 Dependencies

- **No new runtime dependencies**: statsmodels and wildboottest are test-only
- Install test dependencies with: `pip install causers[test]`

### 📚 References

- Cameron, A. C., & Miller, D. L. (2015). A Practitioner's Guide to Cluster-Robust Inference. *Journal of Human Resources*, 50(2), 317-372.
- MacKinnon, J. G., & Webb, M. D. (2018). The wild bootstrap for few (treated) clusters. *The Econometrics Journal*, 21(2), 114-135.

---

## [0.2.0] - 2025-12-23

### ✨ Features

- **HC3 Robust Standard Errors**: Heteroskedasticity-consistent standard errors computed automatically
  - `standard_errors` (List[float]): HC3 SE for each coefficient
  - `intercept_se` (float | None): HC3 SE for intercept (None if `include_intercept=False`)
  - Matches statsmodels HC3 implementation to 1e-6 relative tolerance
  - Based on MacKinnon & White (1985) formulation

- **Extreme Leverage Detection**: Automatic detection of high-leverage observations
  - Raises `ValueError` if any observation has leverage ≥ 0.99
  - Prevents unreliable standard error computation

### 🛠️ Technical Details

- Matrix inversion using Gauss-Jordan elimination with partial pivoting
- Singularity tolerance: 1e-10
- Leverage threshold: 0.99
- Full backward compatibility with v0.1.0 API

### 📖 API Changes

**New Result Attributes:**
- `standard_errors` (List[float]): HC3 robust standard errors for each coefficient
- `intercept_se` (float | None): HC3 standard error for intercept

**New Errors:**
- `ValueError`: "Observation X has leverage ≥ 0.99; HC3 standard errors may be unreliable"

### 📊 Performance

With HC3 computation enabled:
- 1,000 rows: ~1ms
- 100,000 rows: ~25ms
- 1,000,000 rows: ~250ms (regression + HC3 SE)
- 5,000,000 rows: ~1,200ms

### ⚠️ Breaking Changes

- Two-observation regressions with intercept now raise `ValueError` due to extreme leverage
- Very small values (1e-15) may trigger singular matrix errors due to stricter tolerance

### 📦 Dependencies

- **Test dependency**: statsmodels 0.14.0-0.16.0 (for HC3 validation tests only)
  - Install with: `pip install causers[test]`

---

## [0.1.0] - 2025-12-21

### 🎉 Initial Release

The first public release of causers, a high-performance statistical package for Polars DataFrames powered by Rust.

### ✨ Features

- **Linear Regression**: Fast Ordinary Least Squares (OLS) implementation
  - **Multiple Covariate Support**: Accepts single or multiple independent variables
    - Single covariate: `x_cols="feature"` (backward compatible)
    - Multiple covariates: `x_cols=["size", "age", "bedrooms"]`
    - Matrix-based OLS using (X'X)⁻¹ X'y formula
  - **Optional Intercept**: Control intercept with `include_intercept` parameter
    - `include_intercept=True` (default): Standard regression with intercept
    - `include_intercept=False`: Regression through origin (fully saturated models)
  - Direct operation on Polars DataFrames without format conversion
  - Returns coefficients, intercept (if included), R-squared, and sample count
  - Performance validated at ~45ms for 1 million rows (>3x faster than NumPy/pandas)
  - **Backward Compatibility**: Single covariate API unchanged, `result.slope` still available

- **Native Polars Integration**: 
  - Zero-copy operations where possible
  - Seamless DataFrame column access
  - Preserves Polars' memory efficiency

- **Production-Ready Quality**:
  - 100% test coverage (64 tests passing)
  - Comprehensive edge case handling
  - Property-based testing for mathematical correctness
  - Performance benchmarks included

- **Cross-Platform Support**:
  - Pre-built wheels for Linux, macOS (Intel/ARM), and Windows
  - Python 3.8, 3.9, 3.10, 3.11, and 3.12 support
  - Uses stable Python ABI (abi3) for compatibility

### 🔒 Security

- Memory-safe implementation with no unsafe Rust code (except PyO3 requirements)
- Input validation for all operations
- No telemetry or external network calls
- Security assessment rating: **B+**
- No critical or high-severity vulnerabilities

### 📊 Performance

Benchmarked on Apple M1 Pro (16GB RAM):
- 1,000 rows: 0.8ms
- 100,000 rows: 4.2ms
- 1,000,000 rows: 45ms (requirement: <100ms ✅)
- 5,000,000 rows: 210ms

### 🛠️ Technical Details

- Built with Rust 1.70+ and PyO3 0.21.2
- Polars 0.44.2 compatibility
- Maturin build system for reliable packaging
- IEEE-754 compliant floating-point operations

### 📦 Installation

```bash
pip install causers
```

### 🎯 API Changes & Enhancements

**New Parameters:**
- `x_cols` parameter now accepts `str | List[str]` (was `x_col: str`)
  - Single covariate: `x_cols="feature"` or legacy `x_col="feature"` (deprecated but supported)
  - Multiple covariates: `x_cols=["feature1", "feature2", "feature3"]`
- `include_intercept` parameter added (bool, default=True)
  - Set to `False` for regression through origin

**Result Object Changes:**
- `coefficients` (List[float]): New attribute containing all regression coefficients
- `slope` (float | None): Maintained for backward compatibility (single covariate only)
- `intercept` (float | None): Now `None` when `include_intercept=False`

**Backward Compatibility:**
- All v0.1.0 code continues to work without changes
- Single covariate regression API unchanged
- `result.slope` and `result.intercept` still available for single covariate models

### 📖 Documentation

- Comprehensive README with examples
- Full API documentation with type hints
- Performance benchmarks
- Security assessment report
- Contributing guidelines

### ⚠️ Known Limitations

- **NaN/Inf Handling**: Special float values (NaN, Inf) are not fully validated
  - Workaround: Pre-filter your data or check results for NaN

- **Memory Limits**: No explicit limits on DataFrame size
  - Workaround: Monitor memory usage for very large datasets (>10GB)

### 👥 Contributors

- Core development team
- PyO3 community for excellent Python-Rust bindings
- Polars team for the outstanding DataFrame library

### 🐛 Bug Reports

Please report issues at: https://github.com/causers/causers/issues

### 📜 License

MIT License - see LICENSE file for details.

---

[0.3.0]: https://github.com/causers/causers/releases/tag/v0.3.0
[0.2.0]: https://github.com/causers/causers/releases/tag/v0.2.0
[0.1.0]: https://github.com/causers/causers/releases/tag/v0.1.0