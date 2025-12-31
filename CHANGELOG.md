# Changelog

All notable changes to the causers project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.7.0] - 2025-12-30

### 🎯 Overview

Version 0.7.0 is a major feature release that adds four significant capabilities:

1. **pandas DataFrame Support**: All causers functions now accept both Polars and pandas DataFrames seamlessly
2. **Fixed Effects Estimation**: Control for entity-level or time-level unobserved heterogeneity
   - **Linear Regression (OLS)**: Within-transformation (de-meaning) for efficient fixed effects absorption
   - **Logistic Regression**: Mundlak (1978) approach for nonlinear models with fixed effects
3. **Double Machine Learning (DML)**: Debiased causal inference with cross-fitting for ATE and CATE estimation (Chernozhukov et al., 2018)
4. **Two-Stage Least Squares (IV/2SLS)**: Instrumental variables estimation for causal inference with endogenous treatments

---

### ✨ Features

#### pandas DataFrame Input Support

All main functions now accept both Polars and pandas DataFrames:

```python
import pandas as pd
import causers

# pandas DataFrames now work directly!
df_pandas = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
result = causers.linear_regression(df_pandas, "x", "y")

# Polars DataFrames continue to work unchanged
import polars as pl
df_polars = pl.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
result = causers.linear_regression(df_polars, "x", "y")
```

**Dual Extraction Paths**: Optimized column extraction based on pandas column backend:
- **Arrow-backed columns** (`pd.ArrowDtype`): Zero-copy extraction via PyArrow → Polars
- **NumPy-backed columns** (traditional): Extraction via NumPy arrays with automatic dtype conversion

**Optional Dependencies**: pandas support is optional to keep the package lightweight:
```bash
pip install causers           # Base installation (Polars only)
pip install causers[pandas]   # With pandas support
pip install causers[dev]      # All dependencies including pandas
```

#### Fixed Effects for Linear Regression

The `linear_regression()` function now accepts an optional `fixed_effects` parameter:

```python
import polars as pl
import causers

# One-way fixed effects (entity FE)
result = causers.linear_regression(
    df=data,
    x_cols=["x1", "x2"],
    y_col="y",
    fixed_effects=["entity_id"]
)

# Two-way fixed effects (entity + time FE)
result = causers.linear_regression(
    df=data,
    x_cols=["x1", "x2"],
    y_col="y",
    fixed_effects=["entity_id", "time_id"]
)
```

**Within-Transformation (De-meaning)**: For linear models, the standard within-transformation absorbs fixed effects efficiently:
1. Compute group means of outcome and all covariates
2. Subtract group means from each observation: (y_it - ȳ_i, X_it - X̄_i)
3. Run standard OLS on the de-meaned data
4. Intercept is not estimated (absorbed by fixed effects)

**New Result Fields** (when `fixed_effects` is specified):
- `fixed_effects_absorbed`: List[int] — Number of groups absorbed for each FE dimension
- `fixed_effects_names`: List[str] — Names of the FE columns absorbed

#### Fixed Effects for Logistic Regression

The `logistic_regression()` function now accepts an optional `fixed_effects` parameter:

```python
import polars as pl
import causers

# One-way fixed effects (entity FE)
result = causers.logistic_regression(
    df=data,
    x_cols=["x1", "x2"],
    y_col="y",
    fixed_effects=["entity_id"]
)

# Two-way fixed effects (entity + time FE)
result = causers.logistic_regression(
    df=data,
    x_cols=["x1", "x2"],
    y_col="y",
    fixed_effects=["entity_id", "time_id"]
)

# Combined with clustering
result = causers.logistic_regression(
    df=data,
    x_cols=["x1", "x2"],
    y_col="y",
    fixed_effects=["entity_id"],
    cluster="cluster_id"
)
```

**Mundlak Strategy**: For nonlinear models like logistic regression, standard within-transformation (demeaning) is not valid. The Mundlak (1978) approach provides an alternative:
1. Compute group means of all covariates: X̄_g for each FE group g
2. Augment the design matrix: [X | X̄_g1 | X̄_g2 | ...]
3. Run standard logistic regression on the augmented model
4. Return only the coefficients for the original X variables

**Key behavior**:
- Mundlak term coefficients are NOT reported in output (only original covariate coefficients)
- Standard errors are computed on the full augmented model, then filtered to original covariates
- Intercept is preserved (unlike linear regression demeaning)

**New Result Fields** (when `fixed_effects` is specified):
- `fixed_effects_absorbed`: List[int] — Number of groups absorbed for each FE dimension
- `fixed_effects_names`: List[str] — Names of the FE columns absorbed
- `within_pseudo_r_squared`: float — Pseudo R² computed on the Mundlak-augmented model

#### Double Machine Learning (DML)

New `dml()` function for debiased causal inference using the Chernozhukov et al. (2018) methodology:

```python
import polars as pl
import causers

result = causers.dml(
    df,
    y_col="outcome",           # Outcome variable
    d_col="treatment",         # Binary treatment
    x_cols=["age", "income"],  # Confounders/controls
    n_folds=5,                 # Cross-fitting folds
    alpha=0.05,                # Confidence level
)

print(f"ATE: {result.ate:.4f} ± {result.ate_se:.4f}")
print(f"95% CI: [{result.ate_ci_lower:.4f}, {result.ate_ci_upper:.4f}]")
```

**Cross-Fitting Procedure**: DML uses cross-fitting to avoid overfitting bias:
1. Split data into K folds (default: 5)
2. For each fold, train nuisance models (outcome and propensity) on other folds
3. Predict on held-out fold to get residualized outcomes
4. Estimate treatment effect from residualized data
5. Aggregate across folds for final ATE estimate

**DMLResult**: New result class with comprehensive output:

| Field | Type | Description |
|-------|------|-------------|
| `ate` | `float` | Average Treatment Effect estimate |
| `ate_se` | `float` | Standard error of ATE |
| `ate_ci_lower` | `float` | Lower bound of confidence interval |
| `ate_ci_upper` | `float` | Upper bound of confidence interval |
| `n_samples` | `int` | Number of observations |
| `n_treated` | `int` | Number of treated units |
| `n_control` | `int` | Number of control units |
| `n_folds` | `int` | Number of cross-fitting folds used |
| `propensity_mean` | `float` | Mean predicted propensity score |
| `outcome_r_squared` | `float` | R² of outcome model |
| `n_clusters` | `int \| None` | Number of clusters (if clustered SE) |

#### Two-Stage Least Squares (IV/2SLS)

New `two_stage_least_squares()` function for instrumental variables estimation:

```python
import polars as pl
import causers

result = causers.two_stage_least_squares(
    df,
    y_col="wage",           # Outcome variable
    d_cols="education",     # Endogenous treatment(s)
    z_cols="quarter_born",  # Excluded instrument(s)
    x_cols="age",           # Exogenous controls (optional)
    robust=True,            # HC3 standard errors
)

print(f"Coefficient: {result.coefficients[0]:.4f}")
print(f"First-stage F: {result.first_stage_f[0]:.1f}")
```

**Weak Instrument Diagnostics**: Comprehensive diagnostics for detecting weak instruments:
- **First-stage F-statistic**: Per-endogenous-variable test; F > 10 is rule-of-thumb for strong instruments
- **Cragg-Donald statistic**: Multivariate generalization for multiple endogenous variables
- **Stock-Yogo critical values**: Lookup table for 10% maximal bias threshold

```python
# Diagnostics available in result
print(f"First-stage F: {result.first_stage_f}")
print(f"Cragg-Donald: {result.cragg_donald}")
print(f"Stock-Yogo 10% critical: {result.stock_yogo_critical}")
```

**Three Standard Error Types**:
- **Conventional**: Homoskedastic SE (default, matches statsmodels)
- **HC3 Robust**: Heteroskedasticity-consistent via `robust=True`
- **Clustered**: Cluster-robust SE via `cluster="cluster_col"`

**TwoStageLSResult**: New result class with comprehensive output:

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `List[float]` | Structural equation coefficients (endogenous + exogenous) |
| `standard_errors` | `List[float]` | Standard errors for all coefficients |
| `intercept` | `float \| None` | Intercept term (if included) |
| `intercept_se` | `float \| None` | SE for intercept |
| `n_samples` | `int` | Number of observations |
| `n_endogenous` | `int` | Number of endogenous regressors |
| `n_instruments` | `int` | Number of excluded instruments |
| `first_stage_f` | `List[float]` | F-statistics per endogenous variable |
| `first_stage_coefficients` | `List[List[float]]` | First-stage coefficients (instruments only) |
| `cragg_donald` | `float \| None` | Cragg-Donald statistic (multiple endogenous) |
| `stock_yogo_critical` | `float \| None` | Stock-Yogo 10% critical value |
| `r_squared` | `float` | R² from structural equation |
| `se_type` | `str` | "conventional", "hc3", or "clustered" |
| `n_clusters` | `int \| None` | Number of clusters (if clustered) |

---

### 🛠️ Technical Details

**pandas Compatibility:**
- New `python/causers/_pandas_compat.py` module handles all pandas compatibility
- O(1) DataFrame type detection via `isinstance()` checks
- Selective column extraction (only needed columns converted, not full DataFrame)
- Input immutability guaranteed - original pandas DataFrame never modified
- Version requirements: pandas ≥2.0.0, pyarrow ≥10.0.0

**Fixed Effects:**
- New `compute_mundlak_terms()` function in [`src/fixed_effects.rs`](src/fixed_effects.rs)
- Extended `LogisticRegressionResult` struct with FE fields in [`src/logistic.rs`](src/logistic.rs)
- Validation and augmentation logic in [`src/lib.rs`](src/lib.rs)
- 24 new tests covering one-way FE, two-way FE, edge cases, and error conditions

**Two-Stage Least Squares:**
- New [`src/iv2sls.rs`](src/iv2sls.rs) module for 2SLS implementation
- All numerical computation in Rust for performance
- Matches statsmodels IV2SLS reference implementation

---

### 📖 API Changes

**New Functions:**
- `dml(df, y_col, d_col, x_cols, n_folds=5, alpha=0.05, propensity_clip=(0.01, 0.99), cluster=None, seed=None)` — Double Machine Learning for ATE estimation
- `two_stage_least_squares(df, y_col, d_cols, z_cols, x_cols=None, include_intercept=True, robust=False, cluster=None)` — 2SLS estimation

**New Classes:**
- `DMLResult` — Container for DML results (ATE, SE, CI, diagnostics)
- `TwoStageLSResult` — Container for 2SLS results and diagnostics

**Updated Function Signatures:**

All functions now accept `Union[pl.DataFrame, pd.DataFrame]` for the `df` parameter:
- `linear_regression(df, x_cols, y_col, ...)` - Now accepts pandas; new `fixed_effects` parameter
- `logistic_regression(df, x_cols, y_col, ...)` - Now accepts pandas; new `fixed_effects` parameter
- `dml(df, y_col, d_col, x_cols, ...)` - Now accepts pandas
- `synthetic_did(df, unit_col, time_col, ...)` - Now accepts pandas
- `synthetic_control(df, unit_col, time_col, ...)` - Now accepts pandas

**New Module:**
- `causers._pandas_compat` - Internal pandas compatibility utilities

**New Parameters (linear_regression, logistic_regression):**
- `fixed_effects`: Column name(s) for fixed effects to absorb. Supports 1 or 2 columns.

**New Parameters (dml):**
- `y_col`: Outcome variable column
- `d_col`: Binary treatment column
- `x_cols`: Covariate/confounder columns
- `n_folds`: Number of cross-fitting folds (2, 5, or 10; default: 5)
- `alpha`: Significance level for confidence intervals (default: 0.05)
- `propensity_clip`: Tuple for propensity score clipping (default: (0.01, 0.99))
- `cluster`: Column for clustered standard errors (optional)
- `seed`: Random seed for fold assignment (optional)

**New Result Fields (LinearRegressionResult with fixed_effects):**
- `fixed_effects_absorbed` (List[int] | None): Group counts per FE dimension
- `fixed_effects_names` (List[str] | None): FE column names in order

**New Result Fields (LogisticRegressionResult with fixed_effects):**
- `fixed_effects_absorbed` (List[int] | None): Group counts per FE dimension
- `fixed_effects_names` (List[str] | None): FE column names in order
- `within_pseudo_r_squared` (float | None): Pseudo R² on augmented model

**New Errors:**

*pandas compatibility:*
- `TypeError`: "Unsupported DataFrame type: {type}. Expected polars.DataFrame or pandas.DataFrame."
- `ValueError`: "Column 'X' not found in DataFrame"
- `ValueError`: "Column 'X' has object dtype which cannot be converted to numeric"
- `ValueError`: "Column 'X' has datetime dtype which is not supported"
- `ValueError`: "MultiIndex columns not supported. Please flatten column names."
- `ValueError`: "MultiIndex rows not supported. Please reset_index() first."
- `ValueError`: "Sparse columns not supported. Please convert to dense first."

*Fixed effects:*
- `ValueError`: "fixed_effects supports at most 2 columns"
- `ValueError`: "Fixed effect column '{name}' not found in DataFrame"
- `ValueError`: "Fixed effect column '{name}' cannot also be a covariate"
- `ValueError`: "Fixed effect column '{name}' cannot be the outcome variable"
- `ValueError`: "Fixed effect column '{name}' contains null values"
- `ValueError`: "Fixed effect column '{name}' has only one unique value"
- `ValueError`: "Mundlak terms cause collinearity; model is not identified"

*Double Machine Learning:*
- `ValueError`: "n_folds must be 2, 5, or 10; got N"
- `ValueError`: "Number of folds K must be less than sample size N"
- `ValueError`: "Treatment variable has no variation"
- `ValueError`: "Treatment fully explained by covariates"
- `ValueError`: "Propensity model failed to converge"
- `ValueError`: "Covariate matrix is singular in fold N"
- `ValueError`: "alpha must be in (0, 1)"
- `ValueError`: "propensity_clip must satisfy 0 < low < high < 1"
- `ValueError`: "Multi-valued categorical treatments not supported"
- `ValueError`: "Multiple simultaneous treatments not supported"
- `ValueError`: "Clustered standard errors require at least 2 clusters"
- `ValueError`: "Column 'X' contains null values"
- `ValueError`: "Numerical instability detected in fold N"

*Two-Stage Least Squares:*
- `ValueError`: "Number of instruments (m) must be ≥ number of endogenous variables (k₁)"
- `ValueError`: "First stage design matrix is singular; check for collinear instruments"
- `ValueError`: "Instruments too weak for reliable inference (F < 4)"
- `ValueError`: "Column 'X' contains null values"

**New Warnings:**

*Fixed effects:*
- `UserWarning`: When fixed_effects column is float type (implicit cast to integer)

*Double Machine Learning:*
- `UserWarning`: "Fold N has fewer than 100 observations; estimates may be unstable"

*Two-Stage Least Squares:*
- `UserWarning`: "Weak instruments: first-stage F-statistic (X.XX) is below 10 for endogenous variable 'D'"
- `UserWarning`: "Cragg-Donald statistic (X.XX) is below Stock-Yogo 10% critical value (X.XX)"
- `UserWarning`: "Large number of instruments (N) relative to sample size (M); consider using fewer"

---

### ✅ Validation

**pandas Support:**
- pandas and Polars produce numerically identical results (rtol=1e-6)
- Comprehensive test coverage in `tests/test_pandas_support.py`
- Benchmark tests for conversion paths in `tests/test_benchmark_performance.py`

**Fixed Effects:**
- Numerical accuracy validated against statsmodels (pyfixest does not support fixed effects for GLMs)
- Coefficients match to rtol=1e-6
- Standard errors match to rtol=1e-5
- Log-likelihood match to rtol=1e-4
- 24 new tests for fixed effects functionality
- 100% Python API coverage maintained
- Integration with clustering, bootstrap, and intercept options tested

**Two-Stage Least Squares:**
- Numerical accuracy validated against `statsmodels.regression.linear_model.IV2SLS`
- Coefficients match to rtol=1e-6
- Standard errors match to rtol=1e-5
- First-stage F-statistic match to rtol=1e-4

---

### ⚠️ Breaking Changes

None. All existing code continues to work unchanged. New parameters default to their backward-compatible values.

---

### 📦 Dependencies

**New optional dependencies:**
- `pandas>=2.0.0,<3.0.0` (optional)
- `pyarrow>=10.0.0` (optional, for Arrow extraction path)

Install with: `pip install causers[pandas]`

No new runtime dependencies required for fixed effects or 2SLS features.

---

### 📚 References

- Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data." *Econometrica*, 46(1), 69-85.
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press, Chapter 15.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1-C68.
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV Regression."
- Staiger, D., & Stock, J. H. (1997). "Instrumental Variables Regression with Weak Instruments." *Econometrica*.

---

## [0.6.0] - 2025-12-27

### 🎯 Overview: Performance Diagnostics and Improvements

Version 0.6.0 is a performance-focused release that introduces optimized matrix operations, comprehensive benchmark infrastructure, and measurable speedups across all statistical methods. This release consolidates the 0.5.x feature set while delivering faster computation and diagnostic tooling.

### ⚡ Performance Improvements

#### Triangular Matrix Multiplication

The X'WX computation in logistic regression now uses [`faer::linalg::matmul::triangular`](src/linalg.rs:7) to exploit matrix symmetry:
- Only computes lower triangle of symmetric result (~50% fewer FLOPs)
- Automatic symmetry fill for Cholesky compatibility
- Applied to all weighted Gram matrix calculations

#### Flat Data Path Optimization

New [`compute_linear_regression_flat()`](src/lib.rs:314) and [`compute_logistic_regression_flat()`](src/lib.rs:706) functions eliminate intermediate allocations:
- Build faer::Mat directly from flat array data via [`flat_to_mat_with_intercept()`](src/linalg.rs:83)
- Skip Vec<Vec<f64>> intermediate format entirely
- Reduces memory allocation overhead for large datasets

#### In-Place Newton-Raphson Operations

Dedicated in-place functions in [`src/linalg.rs`](src/linalg.rs:692) optimize the logistic regression inner loop:
- [`mat_vec_mul_inplace_direct()`](src/linalg.rs:706): Direct loops for small p, avoiding SIMD overhead
- [`compute_xtr_inplace_direct()`](src/linalg.rs:769): Zero-allocation gradient computation
- [`compute_xtwx_inplace()`](src/linalg.rs:836): Reuses pre-allocated buffers across iterations
- [`sigmoid_inplace()`](src/linalg.rs:878): In-place sigmoid with numerical stability

#### Rayon Parallelization

All matrix operations use automatic thread pool parallelization:
```rust
Parallelism::Rayon(0)  // 0 = auto-detect thread count
```

### 📊 Benchmark Infrastructure

#### Comparative Performance Suite

New [`tests/benchmark_performance.py`](tests/benchmark_performance.py) and [`tests/test_benchmark_performance.py`](tests/test_benchmark_performance.py) provide:
- Head-to-head timing vs reference packages (statsmodels, azcausal, pysyncon)
- Warmup runs and median timing for stable measurements
- Automated pass/fail based on speedup ratios
- Coverage across dataset sizes (1K–100K rows), variable counts (2–50), and SE types

```bash
python tests/benchmark_performance.py
```

#### Performance Validation Tests

[`tests/test_performance.py`](tests/test_performance.py) validates:
- **Scaling behavior**: 1K → 100K → 1M → 5M rows
- **Consistency**: CV < 20% across 10 runs
- **Webb parity**: Webb bootstrap ≤ 1.10× Rademacher time (REQ-PERF-001)
- **Worst-case data**: Large scale differences handled efficiently

#### Benchmark Configurations

Comprehensive test matrix in [`tests/test_benchmark_performance.py`](tests/test_benchmark_performance.py:144):
- Observations: 1,000 | 10,000 | 100,000
- Variables: 2 | 10 | 50
- SE types: HC3 | Clustered (balanced) | Clustered (imbalanced)
- Methods: Linear, Logistic, SDID, SC (all 4 variants)

### 📈 Performance Results

Typical speedups vs reference implementations (Apple M1 Pro):

| Method | vs Reference | Speedup |
|--------|--------------|---------|
| Linear regression (HC3) | statsmodels | 2–6× (median 4×) |
| Logistic regression (HC3) | statsmodels | 1–4× (median 1.4×) |
| Synthetic DID | azcausal | 14–28× |
| Synthetic Control | pysyncon | 137–368× |

Absolute timings (1M rows, single covariate):
- Linear regression + HC3: ~250ms
- Logistic regression + HC3: ~400ms

### ✅ Validation

All methods validated against reference implementations:
- Linear/Logistic regression: statsmodels (rtol=1e-6)
- Wild cluster bootstrap: wildboottest (rtol=1e-2)
- Synthetic DID: azcausal (ATT rtol=1e-6, SE rtol=0.5)
- Synthetic Control: pysyncon (traditional method parity)

**Test Coverage:**
- 193+ tests passing
- 100% Python API coverage
- Property-based testing with Hypothesis

### ⚠️ Breaking Changes

None. All existing code continues to work unchanged.

### 📦 Dependencies

No new runtime dependencies. Uses existing `faer::linalg::matmul::triangular` module (faer v0.20).

---

## [0.5.1] - 2025-12-26

### Documentation

- Added Sphinx documentation infrastructure with autodoc for automatic API reference generation
- Configured Read The Docs integration with Rust toolchain support for PyO3 extension builds
- API documentation now auto-generated from Python docstrings
- Updated README.md with live RTD documentation links and badge

### Removed

- `documentation/api-reference.md` (replaced by Sphinx autodoc on RTD)

---

## [0.5.0] - 2025-12-25

### ✨ Features

- **Synthetic Control (SC)**: New causal inference method for single treated unit studies
  - New `synthetic_control()` function for estimating treatment effects
  - Implements Abadie et al. (2010, 2015) Synthetic Control methodology
  - Constructs weighted combination of control units to match pre-treatment outcomes

- **Four SC Method Variants**:
  - `"traditional"`: Classic SC minimizing pre-treatment MSE with simplex constraints
  - `"penalized"`: L2 regularization for more uniform weights, auto-lambda via LOOCV
  - `"robust"`: De-meaned data to match dynamics instead of levels
  - `"augmented"`: Bias correction via ridge outcome model (Ben-Michael et al., 2021)

- **SyntheticControlResult**: New result class with comprehensive diagnostics
  - `att`: Average Treatment Effect on the Treated
  - `standard_error`: In-space placebo standard error
  - `unit_weights`: Optimized weights for control units (sum to 1)
  - `pre_treatment_rmse`, `pre_treatment_mse`: Pre-treatment fit quality metrics
  - `method`: Method used ("traditional", "penalized", "robust", "augmented")
  - `lambda_used`: Regularization parameter (for penalized/augmented)
  - `n_units_control`, `n_periods_pre`, `n_periods_post`: Panel structure info
  - `solver_converged`, `solver_iterations`: Optimization diagnostics
  - `n_placebo_used`: Number of successful placebo iterations

- **In-Space Placebo Standard Errors**: SE estimation via placebo treatment assignment
  - Each control unit treated as placebo to compute placebo ATTs
  - SE = standard deviation of placebo ATT distribution

- **Input Validation**: Comprehensive validation with clear error messages
  - Single treated unit requirement enforced
  - Balanced panel validation
  - Column type checking (no float unit/time columns)
  - Method name and lambda parameter validation

- **Weight Concentration Warnings**: Automatic detection of concentrated weights
  - Warns if any unit weight > 50%
  - Warns if pre-treatment RMSE > 10% of outcome std

### 🛠️ Technical Details

- New `src/synth_control.rs` module for SC implementation
- Frank-Wolfe optimization reused from SDID module
- All numerical computation in Rust for performance
- Matches pysyncon reference implementation for traditional method

### 📖 API Changes

**New Functions:**
- `synthetic_control(df, unit_col, time_col, outcome_col, treatment_col, ...)` — SC estimation

**New Classes:**
- `SyntheticControlResult` — Container for SC results and diagnostics

**New Parameters:**
- `unit_col: str` — Column identifying panel units
- `time_col: str` — Column identifying time periods
- `outcome_col: str` — Outcome variable column
- `treatment_col: str` — Treatment indicator column (0/1)
- `method: str = "traditional"` — SC method variant
- `lambda_param: Optional[float] = None` — Regularization for penalized/augmented
- `compute_se: bool = True` — Whether to compute standard errors
- `n_placebo: Optional[int] = None` — Number of placebo iterations
- `max_iter: int = 1000` — Maximum optimizer iterations
- `tol: float = 1e-6` — Convergence tolerance
- `seed: Optional[int] = None` — Random seed for reproducibility

**New Errors:**
- `ValueError`: "Cannot perform SC on empty DataFrame"
- `ValueError`: "Column 'X' not found in DataFrame"
- `ValueError`: "unit_col must be integer or string, not float"
- `ValueError`: "time_col must be integer or string, not float"
- `ValueError`: "outcome_col must be numeric"
- `ValueError`: "outcome_col 'X' contains null values"
- `ValueError`: "treatment_col must contain only 0 and 1 values"
- `ValueError`: "Synthetic Control requires exactly 1 treated unit; found N"
- `ValueError`: "At least 1 control unit required; found N"
- `ValueError`: "At least 1 pre-treatment period required; found N"
- `ValueError`: "No post-treatment periods found"
- `ValueError`: "Panel is not balanced"
- `ValueError`: "method must be one of {traditional, penalized, robust, augmented}"
- `ValueError`: "lambda_param must be >= 0"

**New Warnings:**
- `UserWarning`: "Unit weight concentration: control unit at index X has weight Y%"
- `UserWarning`: "Pre-treatment RMSE (X) is Y% of outcome std (Z)"

### 📦 Dependencies

- **New test dependency**: `pysyncon>=1.0` for SC validation tests
  - Install with: `pip install causers[test]`

### ⚠️ Breaking Changes

None. All existing code continues to work unchanged.

### 📚 References

- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control Methods for
  Comparative Case Studies. *Journal of the American Statistical Association*.
- Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative Politics and the
  Synthetic Control Method. *American Journal of Political Science*.
- Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented Synthetic
  Control Method. *Journal of the American Statistical Association*.

---

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

[0.7.0]: https://github.com/causers/causers/releases/tag/v0.7.0
[0.6.0]: https://github.com/causers/causers/releases/tag/v0.6.0
[0.5.1]: https://github.com/causers/causers/releases/tag/v0.5.1
[0.5.0]: https://github.com/causers/causers/releases/tag/v0.5.0
[0.4.0]: https://github.com/causers/causers/releases/tag/v0.4.0
[0.3.0]: https://github.com/causers/causers/releases/tag/v0.3.0
[0.2.0]: https://github.com/causers/causers/releases/tag/v0.2.0
[0.1.0]: https://github.com/causers/causers/releases/tag/v0.1.0