# Changelog

All notable changes to the causers project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-12-25

### ✨ Features

- **Synthetic Difference-in-Differences (SDID)**: New causal inference method
  - New `synthetic_did()` function for panel data treatment effect estimation
  - Implements Arkhangelsky et al. (2021) SDID estimator
  - Combines synthetic control weighting with difference-in-differences

- **SyntheticDIDResult**: New result class with comprehensive diagnostics
  - `att`: Average Treatment Effect on the Treated
  - `standard_error`: Bootstrap standard error
  - `unit_weights`, `time_weights`: Optimized weights for synthetic control
  - `n_units_control`, `n_units_treated`: Panel structure info
  - `n_periods_pre`, `n_periods_post`: Time structure info
  - `solver_iterations`, `solver_converged`: Optimization diagnostics
  - `pre_treatment_fit`: RMSE of pre-treatment fit
  - `bootstrap_iterations_used`: Number of successful bootstrap iterations

- **Frank-Wolfe Solver**: Simplex-constrained optimization in Rust
  - High-performance implementation for unit and time weight optimization
  - Convergence tolerance: 1e-6 with max 10,000 iterations

- **Placebo Bootstrap SE**: Standard error estimation via placebo resampling
  - Default 200 bootstrap iterations
  - Random control unit selection as placebo treated

- **Input Validation**: Comprehensive validation with clear error messages
  - Balanced panel check
  - Treatment indicator validation (0/1 only)
  - Minimum control units (≥2) and pre-periods (≥2) checks
  - Float type detection for unit/time columns

- **Weight Concentration Warnings**: Automatic detection of concentrated weights
  - Warns if any unit weight > 50%
  - Warns if any time weight > 50%
  - Warns if bootstrap_iterations < 100

### 🛠️ Technical Details

- New `src/sdid.rs` module for SDID implementation
- All numerical optimization implemented in Rust for performance
- Matches azcausal reference implementation (ATT to rtol=1e-6, SE to rtol=1e-2)
- 38+ unit tests covering SDID functionality

### 📖 API Changes

**New Functions:**
- `synthetic_did(df, unit_col, time_col, outcome_col, treatment_col, ...)` — SDID estimation

**New Classes:**
- `SyntheticDIDResult` — Container for SDID results and diagnostics

**New Parameters:**
- `unit_col: str` — Column identifying panel units
- `time_col: str` — Column identifying time periods
- `outcome_col: str` — Outcome variable column
- `treatment_col: str` — Treatment indicator column (0/1)
- `bootstrap_iterations: int = 200` — Bootstrap replications for SE
- `seed: Optional[int] = None` — Random seed for reproducibility

**New Errors:**
- `ValueError`: "Cannot perform SDID on empty DataFrame"
- `ValueError`: "Column 'X' not found in DataFrame"
- `ValueError`: "unit_col must be integer or string, not float"
- `ValueError`: "time_col must be integer or string, not float"
- `ValueError`: "outcome_col must be numeric"
- `ValueError`: "outcome_col 'X' contains null values"
- `ValueError`: "treatment_col must contain only 0 and 1 values"
- `ValueError`: "Panel is not balanced: expected N rows, found M"
- `ValueError`: "At least 2 control units required; found N"
- `ValueError`: "No treated units found in data"
- `ValueError`: "At least 2 pre-treatment periods required; found N"
- `ValueError`: "No post-treatment periods found"
- `ValueError`: "bootstrap_iterations must be at least 1"

**New Warnings:**
- `UserWarning`: "Unit weight concentration: control unit at index X has weight Y%"
- `UserWarning`: "Time weight concentration: pre-period at index X has weight Y%"
- `UserWarning`: "bootstrap_iterations=N is less than 100. Standard error estimates may be unreliable."

### 📦 Dependencies

- **New test dependency**: `azcausal>=0.2` for SDID validation tests
  - Install with: `pip install causers[test]`

### ⚠️ Breaking Changes

None. All existing code continues to work unchanged.

### 📚 References

- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
  Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088-4118.

---

## [0.3.0] - 2025-12-25

### ✨ Features

- **Logistic Regression**: Binary outcome regression with MLE estimation
  - New `logistic_regression()` function with same API pattern as linear regression
  - Newton-Raphson optimization with configurable max iterations (35)
  - McFadden's pseudo R² for model fit assessment
  - Perfect separation detection with clear error messages
  - Matches statsmodels `Logit.fit()` coefficients to `rtol=1e-6`

- **LogisticRegressionResult**: New result class with diagnostic fields
  - `coefficients`, `intercept`, `standard_errors`, `intercept_se`: Estimates and SE
  - `converged`, `iterations`: Convergence diagnostics
  - `log_likelihood`, `pseudo_r_squared`: Model fit statistics
  - `n_clusters`, `cluster_se_type`, `bootstrap_iterations_used`: Clustering info

- **Score Bootstrap for Logistic Regression**: Clustered SE via score-based resampling
  - Implements Kline & Santos (2012) methodology
  - Uses Rademacher weights (±1 with equal probability)
  - Appropriate for MLE models unlike wild bootstrap
  - See `docs/score_bootstrap.md` for methodology details

- **Clustered Standard Errors**: Cluster-robust standard errors for panel and grouped data
  - New `cluster` parameter: specify column containing cluster identifiers
  - Analytical clustered SE using sandwich estimator with small-sample adjustment
  - Matches statsmodels `get_robustcov_results(cov_type='cluster')` to `rtol=1e-6`

- **Wild Cluster Bootstrap (Linear)**: Bootstrap-based standard errors for small cluster counts
  - New `bootstrap` parameter: enable wild cluster bootstrap
  - New `bootstrap_iterations` parameter: control number of replications (default: 1000)
  - New `seed` parameter: ensure reproducibility
  - Uses Rademacher weights (±1 with equal probability)
  - Matches wildboottest package results to `rtol=1e-2`

- **Webb Weights for Bootstrap**: New `bootstrap_method` parameter for `linear_regression()` and `logistic_regression()`
  - `bootstrap_method="rademacher"` (default): Standard Rademacher weights (±1 with equal probability)
  - `bootstrap_method="webb"`: Webb six-point distribution for improved small-sample properties
  - Case-insensitive parameter values ("Webb", "WEBB", "webb" all work)
  - Recommended for very few clusters (G < 10)
  - Based on MacKinnon & Webb (2018) methodology

- **Small-Cluster Warning**: Automatic recommendation for bootstrap
  - Emits `UserWarning` when G < 42 clusters with analytical SE
  - Guides users toward more reliable inference methods

- **Cluster Balance Warning**: Detection of imbalanced clusters
  - Emits `UserWarning` when any cluster contains >50% of observations
  - Warns that clustered SE may be unreliable with such imbalance
  - Applies to both linear and logistic regression

- **New LinearRegressionResult Attributes**:
  - `n_clusters` (int | None): Number of unique clusters
  - `cluster_se_type` (str | None): "analytical", "bootstrap_rademacher", or "bootstrap_webb"
  - `bootstrap_iterations_used` (int | None): Actual iterations used

### 🛠️ Technical Details

- New `src/logistic.rs` module for logistic regression MLE
- New `src/cluster.rs` module for all clustering logic
- Newton-Raphson optimizer with step halving for stability
- SplitMix64 PRNG for Rademacher weight generation (no external RNG dependency)
- Welford's online algorithm for O(1) memory bootstrap variance
- Condition number check (> 1e10) for numerical stability
- Perfect separation detection via coefficient divergence monitoring

### 📖 API Changes

**New Functions:**
- `logistic_regression()` — Binary outcome regression with MLE

**New Classes:**
- `LogisticRegressionResult` — Container for logistic regression results

**New Parameters (both functions):**
- `cluster: Optional[str] = None` — Column name for cluster identifiers
- `bootstrap: bool = False` — Enable bootstrap SE (wild for linear, score for logistic)
- `bootstrap_iterations: int = 1000` — Number of bootstrap replications
- `seed: Optional[int] = None` — Random seed for reproducibility
- `bootstrap_method: str = "rademacher"` — Weight distribution for wild cluster/score bootstrap ("rademacher" or "webb")

**LogisticRegressionResult Fields:**
- `coefficients`, `intercept` — MLE estimates (log-odds scale)
- `standard_errors`, `intercept_se` — Robust SE (HC3 or clustered)
- `n_samples` — Observation count
- `n_clusters`, `cluster_se_type`, `bootstrap_iterations_used` — Clustering info
- `converged`, `iterations` — Optimization diagnostics
- `log_likelihood`, `pseudo_r_squared` — Model fit statistics

**New LinearRegressionResult Fields:**
- `n_clusters` — Cluster count (None if not clustered)
- `cluster_se_type` — "analytical", "bootstrap_rademacher", or "bootstrap_webb" (None if not clustered)
- `bootstrap_iterations_used` — Iterations used (None if not bootstrap)

**New Errors:**
- `ValueError`: "y_col must contain only 0 and 1 values"
- `ValueError`: "y_col must contain both 0 and 1 values"
- `ValueError`: "Perfect separation detected; logistic regression cannot converge"
- `ValueError`: "Hessian matrix is singular; check for collinearity"
- `ValueError`: "Convergence failed after 35 iterations"
- `ValueError`: "bootstrap=True requires cluster to be specified"
- `ValueError`: "Clustered standard errors require at least 2 clusters"
- `ValueError`: "Cluster column contains null values"
- `ValueError`: "bootstrap_method must be 'rademacher' or 'webb', got: '{value}'"

**New Warnings:**
- `UserWarning`: "Only N clusters detected. [Wild cluster/Score] bootstrap is recommended when clusters < 42."
- `UserWarning`: "Cluster column 'X' is float; will be cast to string for grouping."
- `UserWarning`: "Cluster 'X' contains N% of observations. Clustered standard errors may be unreliable."

### 📊 Performance

**Logistic Regression:**
- 1M rows, 1 covariate: <500ms (requirement met ✅)
- 100K rows, 10 covariates: <200ms
- Typical convergence: 5-8 iterations

**Clustered SE:**
- Analytical clustered SE: ≤2× HC3 baseline runtime
- Score bootstrap (B=1000) on 100K rows: ~3-5 seconds
- Wild cluster bootstrap (B=1000) on 100K rows: ~2-3 seconds

### ⚠️ Breaking Changes

None. All existing code continues to work unchanged.

### 📦 Dependencies

- **No new runtime dependencies**: statsmodels and wildboottest are test-only
- Install test dependencies with: `pip install causers[test]`

### 📚 References

- Kline, P., & Santos, A. (2012). A Score Based Approach to Wild Bootstrap Inference. *Journal of Econometric Methods*, 1(1), 23-41. https://doi.org/10.1515/2156-6674.1006
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

[0.4.0]: https://github.com/causers/causers/releases/tag/v0.4.0
[0.3.0]: https://github.com/causers/causers/releases/tag/v0.3.0
[0.2.0]: https://github.com/causers/causers/releases/tag/v0.2.0
[0.1.0]: https://github.com/causers/causers/releases/tag/v0.1.0