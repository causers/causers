# API Reference

## Overview

The causers package provides high-performance statistical operations for Polars DataFrames. All computations are performed in Rust for optimal performance while maintaining a clean Python API.

## Main Module

```python
import causers
```

### Package Information

#### `causers.__version__`

The current version of the package as a string.

```python
>>> import causers
>>> print(causers.__version__)
'0.5.0'
```

#### `causers.about()`

Print information about the causers package including version, features, and links.

**Signature:**
```python
def about() -> None
```

**Example:**
```python
>>> causers.about()
causers version 0.5.0
High-performance statistical operations for Polars DataFrames
Powered by Rust via PyO3/maturin

Features:
  - Linear regression with HC3 robust standard errors
  - Logistic regression with Newton-Raphson MLE
  - Cluster-robust standard errors (analytical and bootstrap)
  - Wild cluster bootstrap for small cluster counts (linear)
  - Score bootstrap for small cluster counts (logistic)
  - Synthetic Difference-in-Differences (SDID)
  - Synthetic Control (SC) with multiple method variants
```

## Statistical Functions

### `causers.linear_regression()`

Perform Ordinary Least Squares (OLS) linear regression on a Polars DataFrame. Supports both single and multiple covariate regression with optional intercept control.

**Signature:**
```python
def linear_regression(
    df: polars.DataFrame,
    x_cols: str | List[str],
    y_col: str,
    include_intercept: bool = True,
    cluster: str | None = None,
    bootstrap: bool = False,
    bootstrap_iterations: int = 1000,
    seed: int | None = None,
    bootstrap_method: str = "rademacher"
) -> LinearRegressionResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `polars.DataFrame` | Input DataFrame containing the data. Must have at least 2 rows. |
| `x_cols` | `str \| List[str]` | Name(s) of independent variable column(s). Single covariate: `"feature"`. Multiple covariates: `["size", "age", "bedrooms"]`. Must contain numeric data. |
| `y_col` | `str` | Name of the dependent variable column (response). Must contain numeric data. |
| `include_intercept` | `bool` | Whether to include an intercept term. Default: `True`. Set to `False` for regression through origin (fully saturated models). |
| `cluster` | `str \| None` | Column name for cluster identifiers. When specified, computes cluster-robust standard errors instead of HC3. Supports integer, string, or categorical columns. Default: `None`. |
| `bootstrap` | `bool` | If `True` and `cluster` is specified, use wild cluster bootstrap for standard error computation. Requires `cluster` to be specified. Recommended when number of clusters is less than 42. Default: `False`. |
| `bootstrap_iterations` | `int` | Number of bootstrap replications when `bootstrap=True`. Default: `1000`. |
| `seed` | `int \| None` | Random seed for reproducibility when using bootstrap. When `None`, uses a random seed which may produce different results each call. Default: `None`. |
| `bootstrap_method` | `str` | Weight distribution for wild cluster bootstrap. Case-insensitive. `"rademacher"` (default): Rademacher weights (±1). `"webb"`: Webb six-point distribution, recommended for very few clusters (G < 10). See MacKinnon & Webb (2018). |

**Returns:**

`LinearRegressionResult` object with the following attributes:
- `coefficients` (List[float]): Regression coefficients for each predictor (β₁, β₂, ...)
- `slope` (float | None): Single coefficient (backward compatibility, available for single covariate only)
- `intercept` (float | None): Y-intercept (β₀). `None` when `include_intercept=False`
- `r_squared` (float): Coefficient of determination (R²) ∈ [0, 1]
- `n_samples` (int): Number of data points used
- `standard_errors` (List[float]): Robust standard errors for each coefficient. Uses HC3 by default, or cluster-robust SE if `cluster` is specified.
- `intercept_se` (float | None): Robust standard error for intercept. `None` when `include_intercept=False`.
- `n_clusters` (int | None): Number of unique clusters. `None` if `cluster` not specified.
- `cluster_se_type` (str | None): Type of clustered SE: `"analytical"`, `"bootstrap_rademacher"`, or `"bootstrap_webb"`. `None` if not clustered.
- `bootstrap_iterations_used` (int | None): Number of bootstrap iterations. `None` if not bootstrap.

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Column doesn't exist, data is empty, or x values have zero variance |
| `ValueError` | Cluster column contains null values |
| `ValueError` | `bootstrap=True` without `cluster` specified |
| `ValueError` | Fewer than 2 clusters detected |
| `ValueError` | Single-observation clusters exist (analytical mode only) |
| `ValueError` | Numerical instability detected (condition number > 1e10) |
| `ValueError` | `bootstrap_iterations < 1` |
| `TypeError` | Column contains non-numeric data |
| `RuntimeError` | Unexpected error in Rust implementation |

**Warns:**

| Warning | Condition |
|---------|-----------|
| `UserWarning` | Fewer than 42 clusters with `bootstrap=False`: recommends using wild cluster bootstrap for more accurate inference. |
| `UserWarning` | Cluster column has float dtype: implicit cast to string. |
| `UserWarning` | Any cluster contains >50% of observations: clustered SE may be unreliable with imbalanced clusters. |

**Examples:**

**Single covariate (backward compatible):**
```python
import polars as pl
import causers

# Simple dataset
df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 5, 8, 10]
})

result = causers.linear_regression(df, x_cols="x", y_col="y")
print(f"y = {result.slope}x + {result.intercept}")
print(f"R² = {result.r_squared}")
# Can also use: result.coefficients[0] for the slope
```

**Multiple covariates:**
```python
# Housing price prediction with multiple features
housing_df = pl.DataFrame({
    "size_sqft": [1000, 1500, 1200, 1800, 2200],
    "age_years": [5, 10, 3, 15, 7],
    "bedrooms": [2, 3, 2, 4, 3],
    "price": [200000, 280000, 245000, 350000, 430000]
})

result = causers.linear_regression(
    housing_df,
    x_cols=["size_sqft", "age_years", "bedrooms"],
    y_col="price"
)

print(f"Coefficients: {result.coefficients}")
print(f"Intercept: {result.intercept:,.2f}")
print(f"R² = {result.r_squared:.4f}")

# Manual prediction
size, age, beds = 2000, 8, 3
predicted = (result.coefficients[0] * size +
             result.coefficients[1] * age +
             result.coefficients[2] * beds +
             result.intercept)
print(f"Predicted price: ${predicted:,.2f}")
```

**Regression without intercept:**
```python
# Fully saturated model (regression through origin)
df = pl.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [2.5, 5.0, 7.5, 10.0, 12.5]  # y = 2.5x (no intercept)
})

result = causers.linear_regression(
    df,
    x_cols="x",
    y_col="y",
    include_intercept=False
)

print(f"y = {result.coefficients[0]}x")
print(f"Intercept: {result.intercept}")  # None
print(f"R² = {result.r_squared:.4f}")
```

**Performance with large datasets:**
```python
import numpy as np

# Generate 1 million data points with multiple features
n = 1_000_000
np.random.seed(42)
x1 = np.random.randn(n)
x2 = np.random.randn(n)
y = 2 * x1 + 3 * x2 + 5 + np.random.randn(n) * 0.1

df_large = pl.DataFrame({"x1": x1, "x2": x2, "y": y})

# Multiple regression completes in ~50-60ms
result = causers.linear_regression(df_large, x_cols=["x1", "x2"], y_col="y")
print(f"Processed {result.n_samples:,} samples")
print(f"Coefficients: {result.coefficients}")
```

**Clustered standard errors (analytical):**
```python
# Panel data with firm-level clustering
panel_df = pl.DataFrame({
    "treatment": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    "outcome": [5, 6, 12, 14, 4, 7, 11, 15, 5, 13],
    "firm_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
})

result = causers.linear_regression(
    panel_df,
    x_cols="treatment",
    y_col="outcome",
    cluster="firm_id"
)

print(f"Treatment effect: {result.coefficients[0]:.4f}")
print(f"Clustered SE: {result.standard_errors[0]:.4f}")
print(f"Number of clusters: {result.n_clusters}")
print(f"SE type: {result.cluster_se_type}")
```

**Wild cluster bootstrap (recommended for <42 clusters):**
```python
# When you have few clusters, use bootstrap for more reliable inference
result = causers.linear_regression(
    panel_df,
    x_cols="treatment",
    y_col="outcome",
    cluster="firm_id",
    bootstrap=True,
    bootstrap_iterations=1000,
    seed=42  # For reproducibility
)

print(f"Treatment effect: {result.coefficients[0]:.4f}")
print(f"Bootstrap SE: {result.standard_errors[0]:.4f}")
print(f"SE type: {result.cluster_se_type}")  # "bootstrap_rademacher"
print(f"Iterations used: {result.bootstrap_iterations_used}")
```

**Webb weights for very few clusters (recommended for G < 10):**
```python
# When you have very few clusters, Webb weights may improve inference
result = causers.linear_regression(
    panel_df,
    x_cols="treatment",
    y_col="outcome",
    cluster="firm_id",
    bootstrap=True,
    bootstrap_method="webb",  # Webb six-point distribution
    bootstrap_iterations=1000,
    seed=42
)

print(f"Treatment effect: {result.coefficients[0]:.4f}")
print(f"Webb Bootstrap SE: {result.standard_errors[0]:.4f}")
print(f"SE type: {result.cluster_se_type}")  # "bootstrap_webb"
```

> **Reference**: MacKinnon, J. G., & Webb, M. D. (2018). "The wild bootstrap for few (treated) clusters." *The Econometrics Journal*, 21(2), 114-135.

**Mathematical Details:**

The linear regression fits the model:

**Single covariate:**
```
y = β₁x + β₀ + ε
```
Where:
- `β₁` (slope) = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]
- `β₀` (intercept) = ȳ - β₁x̄

**Multiple covariates:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
```
Where coefficients are computed using matrix formula:
```
β = (X'X)⁻¹ X'y
```
- X is the design matrix (n × k, or n × (k+1) with intercept)
- y is the response vector (n × 1)
- β is the coefficient vector (k × 1, or (k+1) × 1 with intercept)

**Without intercept:**
```
y = β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
```
The intercept term is omitted from the design matrix.

**R-squared:**
```
R² = 1 - (SS_res / SS_tot)
```
- SS_res = Σ[(yi - ŷi)²] (residual sum of squares)
- SS_tot = Σ[(yi - ȳ)²] (total sum of squares)

### `causers.logistic_regression()`

Perform logistic regression on binary outcomes using Maximum Likelihood Estimation (MLE) with Newton-Raphson optimization. Returns coefficient estimates (log-odds), robust standard errors, and diagnostic information.

**Signature:**
```python
def logistic_regression(
    df: polars.DataFrame,
    x_cols: str | List[str],
    y_col: str,
    include_intercept: bool = True,
    cluster: str | None = None,
    bootstrap: bool = False,
    bootstrap_iterations: int = 1000,
    seed: int | None = None,
    bootstrap_method: str = "rademacher"
) -> LogisticRegressionResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `polars.DataFrame` | Input DataFrame containing the data. Must have at least 2 rows. |
| `x_cols` | `str \| List[str]` | Name(s) of independent variable column(s). Single covariate: `"feature"`. Multiple covariates: `["age", "income", "score"]`. Must contain numeric data. |
| `y_col` | `str` | Name of the binary outcome column (must contain only 0 and 1). |
| `include_intercept` | `bool` | Whether to include an intercept term. Default: `True`. |
| `cluster` | `str \| None` | Column name for cluster identifiers. When specified, computes cluster-robust standard errors using the score-based approach. Default: `None`. |
| `bootstrap` | `bool` | If `True` and `cluster` is specified, use score bootstrap for standard error computation. Requires `cluster` to be specified. Recommended when number of clusters is less than 42. Default: `False`. |
| `bootstrap_iterations` | `int` | Number of bootstrap replications when `bootstrap=True`. Default: `1000`. |
| `seed` | `int \| None` | Random seed for reproducibility when using bootstrap. When `None`, uses a random seed which may produce different results each call. Default: `None`. |
| `bootstrap_method` | `str` | Weight distribution for score bootstrap. Case-insensitive. `"rademacher"` (default): Rademacher weights (±1). `"webb"`: Webb six-point distribution, recommended for very few clusters (G < 10). See MacKinnon & Webb (2018). |

**Returns:**

`LogisticRegressionResult` object with the following attributes:
- `coefficients` (List[float]): Coefficient estimates for each predictor (log-odds scale)
- `intercept` (float | None): Intercept term. `None` when `include_intercept=False`
- `standard_errors` (List[float]): Robust standard errors for each coefficient. Uses HC3 by default, or clustered SE if `cluster` is specified.
- `intercept_se` (float | None): Robust standard error for intercept. `None` when `include_intercept=False`.
- `n_samples` (int): Number of observations used
- `n_clusters` (int | None): Number of unique clusters. `None` if `cluster` not specified.
- `cluster_se_type` (str | None): Type of clustered SE: `"analytical"`, `"bootstrap_rademacher"`, or `"bootstrap_webb"`. `None` if not clustered.
- `bootstrap_iterations_used` (int | None): Number of bootstrap iterations used. `None` if not bootstrap.
- `converged` (bool): Whether the MLE optimizer converged
- `iterations` (int): Number of Newton-Raphson iterations used
- `log_likelihood` (float): Log-likelihood at the MLE solution
- `pseudo_r_squared` (float): McFadden's pseudo R² = 1 - (LL_model / LL_null)

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `y_col` contains values other than 0 and 1 |
| `ValueError` | `y_col` contains only 0s or only 1s (no variation) |
| `ValueError` | Column doesn't exist, data is empty |
| `ValueError` | Cluster column contains null values |
| `ValueError` | `bootstrap=True` without `cluster` specified |
| `ValueError` | Fewer than 2 clusters detected |
| `ValueError` | Perfect separation detected (predictor perfectly separates outcomes) |
| `ValueError` | Hessian is singular (collinearity in predictors) |
| `ValueError` | Convergence fails after 35 iterations |
| `ValueError` | Numerical instability detected (condition number > 1e10) |
| `ValueError` | `bootstrap_iterations < 1` |
| `TypeError` | Column contains non-numeric data |

**Warns:**

| Warning | Condition |
|---------|-----------|
| `UserWarning` | Fewer than 42 clusters with `bootstrap=False`: recommends using score bootstrap for more accurate inference. |
| `UserWarning` | Cluster column has float dtype: implicit cast to string. |

**Examples:**

**Simple logistic regression:**
```python
import polars as pl
import causers

# Binary outcome data
df = pl.DataFrame({
    "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "y": [0, 0, 0, 1, 0, 1, 1, 1]
})

result = causers.logistic_regression(df, x_cols="x", y_col="y")

print(f"Coefficient: {result.coefficients[0]:.4f}")
print(f"Intercept: {result.intercept:.4f}")
print(f"Converged: {result.converged}")
print(f"Pseudo R²: {result.pseudo_r_squared:.4f}")
```

**Multiple covariates:**
```python
# Predicting loan default from multiple features
df = pl.DataFrame({
    "income": [50, 60, 40, 80, 100, 30, 70, 90],
    "debt_ratio": [0.4, 0.3, 0.5, 0.2, 0.1, 0.6, 0.3, 0.2],
    "default": [1, 0, 1, 0, 0, 1, 0, 0]
})

result = causers.logistic_regression(
    df,
    x_cols=["income", "debt_ratio"],
    y_col="default"
)

print(f"Income coefficient: {result.coefficients[0]:.4f}")
print(f"Debt ratio coefficient: {result.coefficients[1]:.4f}")
print(f"Log-likelihood: {result.log_likelihood:.2f}")
```

**Clustered standard errors (analytical):**
```python
# Panel data with firm-level clustering
df = pl.DataFrame({
    "treatment": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    "outcome": [0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    "firm_id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
})

result = causers.logistic_regression(
    df,
    x_cols="treatment",
    y_col="outcome",
    cluster="firm_id"
)

print(f"Treatment effect (log-odds): {result.coefficients[0]:.4f}")
print(f"Clustered SE: {result.standard_errors[0]:.4f}")
print(f"Number of clusters: {result.n_clusters}")
```

**Score bootstrap (recommended for <42 clusters):**
```python
# When you have few clusters, use bootstrap for more reliable inference
result = causers.logistic_regression(
    df,
    x_cols="treatment",
    y_col="outcome",
    cluster="firm_id",
    bootstrap=True,
    bootstrap_iterations=1000,
    seed=42  # For reproducibility
)

print(f"Treatment effect: {result.coefficients[0]:.4f}")
print(f"Bootstrap SE: {result.standard_errors[0]:.4f}")
print(f"SE type: {result.cluster_se_type}")  # "bootstrap_rademacher"
print(f"Iterations used: {result.bootstrap_iterations_used}")
```

**Webb weights for very few clusters (recommended for G < 10):**
```python
# When you have very few clusters, Webb weights may improve inference
result = causers.logistic_regression(
    df,
    x_cols="treatment",
    y_col="outcome",
    cluster="firm_id",
    bootstrap=True,
    bootstrap_method="webb",  # Webb six-point distribution
    bootstrap_iterations=1000,
    seed=42
)

print(f"Treatment effect: {result.coefficients[0]:.4f}")
print(f"Webb Bootstrap SE: {result.standard_errors[0]:.4f}")
print(f"SE type: {result.cluster_se_type}")  # "bootstrap_webb"
```

> **Reference**: MacKinnon, J. G., & Webb, M. D. (2018). "The wild bootstrap for few (treated) clusters." *The Econometrics Journal*, 21(2), 114-135.

**Interpreting coefficients:**
```python
import math

result = causers.logistic_regression(df, "treatment", "outcome")

# Coefficient is in log-odds scale
log_odds = result.coefficients[0]

# Convert to odds ratio
odds_ratio = math.exp(log_odds)
print(f"Odds ratio: {odds_ratio:.2f}")

# Probability change (at mean baseline)
# For a unit increase in x, odds multiply by exp(β)
```

**Mathematical Details:**

The logistic regression fits the model:

```
P(y=1|x) = 1 / (1 + exp(-x'β))
```

Where:
- `β` are the log-odds coefficients
- `exp(β)` gives the odds ratio for a unit increase
- The model is estimated via Maximum Likelihood using Newton-Raphson

**Standard Errors:**

- **HC3 (default)**: Heteroskedasticity-consistent standard errors adapted for logistic regression, using weighted leverages
- **Analytical clustered SE**: When `cluster` is specified. Uses sandwich estimator with cluster-level scores
- **Score bootstrap SE**: When `cluster` and `bootstrap=True`. Uses Rademacher weights following Kline & Santos (2012). Recommended when G < 42 clusters.

For details on the score bootstrap methodology, see [Score Bootstrap for Logistic Regression](score_bootstrap.md).

**Pseudo R-squared:**

McFadden's pseudo R² is computed as:
```
R² = 1 - (LL_model / LL_null)
```
Where:
- LL_model = log-likelihood of the fitted model
- LL_null = log-likelihood of the intercept-only model

Unlike linear regression R², pseudo R² values are typically much lower. Values of 0.2-0.4 are considered excellent fit.

---

### `causers.synthetic_did()`

Compute Synthetic Difference-in-Differences (SDID) estimator for causal inference. Combines synthetic control weighting with difference-in-differences to estimate the Average Treatment Effect on the Treated (ATT).

**Signature:**
```python
def synthetic_did(
    df: polars.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    bootstrap_iterations: int = 200,
    seed: int | None = None,
) -> SyntheticDIDResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `polars.DataFrame` | Panel data in long format with one row per unit-time observation. Must be a balanced panel. |
| `unit_col` | `str` | Column name identifying unique units (e.g., "state", "firm_id"). Must be integer or string type. |
| `time_col` | `str` | Column name identifying time periods (e.g., "year", "quarter"). Must be integer or string type. |
| `outcome_col` | `str` | Column name for the outcome variable. Must be numeric. |
| `treatment_col` | `str` | Column name for treatment indicator. Must contain only 0 and 1 values. Value of 1 indicates the unit is treated in that period. |
| `bootstrap_iterations` | `int` | Number of placebo bootstrap iterations for standard error estimation. Must be at least 1. Values < 100 will emit a warning. Default: `200`. |
| `seed` | `int \| None` | Random seed for reproducibility. If None, uses system time. Default: `None`. |

**Returns:**

`SyntheticDIDResult` object with the following attributes:
- `att` (float): Average Treatment Effect on the Treated
- `standard_error` (float): Bootstrap standard error of the ATT
- `unit_weights` (List[float]): Weights assigned to each control unit (sums to 1)
- `time_weights` (List[float]): Weights assigned to each pre-treatment period (sums to 1)
- `n_units_control` (int): Number of control units
- `n_units_treated` (int): Number of treated units
- `n_periods_pre` (int): Number of pre-treatment periods
- `n_periods_post` (int): Number of post-treatment periods
- `solver_iterations` (Tuple[int, int]): Number of iterations for (unit_weights, time_weights) optimization
- `solver_converged` (bool): Whether the Frank-Wolfe solver converged
- `pre_treatment_fit` (float): RMSE of pre-treatment fit (lower is better)
- `bootstrap_iterations_used` (int): Number of successful bootstrap iterations

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | DataFrame is empty |
| `ValueError` | Any specified column doesn't exist |
| `ValueError` | `unit_col` or `time_col` is float type |
| `ValueError` | `outcome_col` is not numeric or contains null values |
| `ValueError` | `treatment_col` contains values other than 0 and 1 |
| `ValueError` | `bootstrap_iterations < 1` |
| `ValueError` | Fewer than 2 control units found |
| `ValueError` | Fewer than 2 pre-treatment periods found |
| `ValueError` | No treated units found |
| `ValueError` | No post-treatment periods found |
| `ValueError` | Panel is not balanced |

**Warns:**

| Warning | Condition |
|---------|-----------|
| `UserWarning` | Any unit weight > 0.5 (weight concentration on single unit) |
| `UserWarning` | Any time weight > 0.5 (weight concentration on single period) |
| `UserWarning` | `bootstrap_iterations < 100` (may be unreliable) |

**Examples:**

**Basic SDID with panel data:**
```python
import polars as pl
import causers

# Panel data with treated and control units
df = pl.DataFrame({
    'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'y': [1.0, 2.0, 5.0, 1.5, 2.5, 3.0, 1.2, 2.2, 2.8],
    'treated': [0, 0, 1, 0, 0, 0, 0, 0, 0]
})

result = causers.synthetic_did(df, 'unit', 'time', 'y', 'treated', seed=42)

print(f"ATT: {result.att:.4f} ± {result.standard_error:.4f}")
print(f"Control unit weights: {result.unit_weights}")
print(f"Pre-period weights: {result.time_weights}")
```

**Accessing diagnostics:**
```python
# Check convergence and fit quality
print(f"Converged: {result.solver_converged}")
print(f"Pre-treatment fit RMSE: {result.pre_treatment_fit:.4f}")
print(f"Solver iterations (unit, time): {result.solver_iterations}")

# Panel structure
print(f"Control units: {result.n_units_control}")
print(f"Treated units: {result.n_units_treated}")
print(f"Pre-periods: {result.n_periods_pre}")
print(f"Post-periods: {result.n_periods_post}")
```

**Multiple treated units:**
```python
# SDID supports multiple treated units (unlike SC)
df = pl.DataFrame({
    'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    'y': [1.0, 2.0, 5.0, 1.1, 2.1, 5.5,  # Units 1,2 treated
          1.5, 2.5, 3.0, 1.2, 2.2, 2.8],  # Units 3,4 control
    'treated': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
})

result = causers.synthetic_did(df, 'unit', 'time', 'y', 'treated', seed=42)
print(f"ATT with {result.n_units_treated} treated units: {result.att:.4f}")
```

**Constructing confidence interval:**
```python
result = causers.synthetic_did(df, 'unit', 'time', 'y', 'treated',
                                bootstrap_iterations=500, seed=42)

# 95% confidence interval
ci_lower = result.att - 1.96 * result.standard_error
ci_upper = result.att + 1.96 * result.standard_error
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

**Mathematical Details:**

**Algorithm:**

The SDID estimator combines synthetic control weighting with difference-in-differences:

1. **Unit weights (ω)**: Find control unit weights that match pre-treatment trends of treated units
2. **Time weights (λ)**: Find pre-period weights that predict post-period outcomes

The estimator is:
```
τ̂_sdid = (Ȳ_tr,post - Ȳ_synth,post) - Σ_t λ_t (Ȳ_tr,t - Ȳ_synth,t)
```

Where `Ȳ_synth,t = Σ_i ω_i Y_i,t` uses optimized unit weights on control units.

**Panel Structure Detection:**

The function automatically detects:
- **Control units**: Units where treatment=0 in all periods
- **Treated units**: Units where treatment=1 in at least one period
- **Pre-periods**: Periods where all observations have treatment=0
- **Post-periods**: Periods where at least one treated unit has treatment=1

**Standard Errors:**

Standard errors are computed via placebo bootstrap:
1. Randomly select a control unit as "placebo treated"
2. Re-run SDID with this unit treated
3. Repeat for `bootstrap_iterations`
4. SE = standard deviation of placebo ATTs

> **Reference:** Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). "Synthetic difference-in-differences." *American Economic Review*, 111(12), 4088-4118.

---

### `causers.synthetic_control()`

Compute Synthetic Control (SC) estimator for causal inference with a single treated unit. Supports four method variants: traditional, penalized, robust, and augmented.

**Signature:**
```python
def synthetic_control(
    df: polars.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    method: str = "traditional",
    lambda_param: float | None = None,
    compute_se: bool = True,
    n_placebo: int | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    seed: int | None = None,
) -> SyntheticControlResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `polars.DataFrame` | Panel data in long format with one row per unit-time observation. Must be a balanced panel. |
| `unit_col` | `str` | Column name identifying unique units (e.g., "state", "firm_id"). Must be integer or string type. |
| `time_col` | `str` | Column name identifying time periods (e.g., "year", "quarter"). Must be integer or string type. |
| `outcome_col` | `str` | Column name for the outcome variable. Must be numeric. |
| `treatment_col` | `str` | Column name for treatment indicator. Must contain only 0 and 1 values. Exactly one unit must be treated. |
| `method` | `str` | Method to use: `"traditional"` (default), `"penalized"`, `"robust"`, or `"augmented"`. |
| `lambda_param` | `float \| None` | Regularization parameter for penalized/augmented methods. If None, auto-selected via LOOCV for penalized. |
| `compute_se` | `bool` | Whether to compute standard errors via in-space placebo. Default: `True`. |
| `n_placebo` | `int \| None` | Number of placebo iterations for SE. If None, uses all control units. |
| `max_iter` | `int` | Maximum iterations for Frank-Wolfe optimizer. Default: `1000`. |
| `tol` | `float` | Convergence tolerance for optimizer. Default: `1e-6`. |
| `seed` | `int \| None` | Random seed for reproducibility. Default: `None`. |

**Returns:**

`SyntheticControlResult` object with the following attributes:
- `att` (float): Average Treatment Effect on the Treated
- `standard_error` (float | None): In-space placebo standard error (None if `compute_se=False`)
- `unit_weights` (List[float]): Weights assigned to each control unit (sums to 1)
- `pre_treatment_rmse` (float): Root Mean Squared Error of pre-treatment fit
- `pre_treatment_mse` (float): Mean Squared Error of pre-treatment fit
- `method` (str): Method used ("traditional", "penalized", "robust", "augmented")
- `lambda_used` (float | None): Lambda parameter used (for penalized/augmented)
- `n_units_control` (int): Number of control units
- `n_periods_pre` (int): Number of pre-treatment periods
- `n_periods_post` (int): Number of post-treatment periods
- `solver_converged` (bool): Whether Frank-Wolfe solver converged
- `solver_iterations` (int): Number of optimizer iterations
- `n_placebo_used` (int | None): Number of successful placebo iterations

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | DataFrame is empty |
| `ValueError` | Any specified column doesn't exist |
| `ValueError` | `unit_col` or `time_col` is float type |
| `ValueError` | `outcome_col` is not numeric or contains null values |
| `ValueError` | `treatment_col` contains values other than 0 and 1 |
| `ValueError` | Not exactly one treated unit found |
| `ValueError` | Fewer than 1 control unit or pre-treatment period |
| `ValueError` | Panel is not balanced |
| `ValueError` | Invalid method name |
| `ValueError` | `lambda_param < 0` |

**Warns:**

| Warning | Condition |
|---------|-----------|
| `UserWarning` | Any unit weight > 0.5 (weight concentration on single unit) |
| `UserWarning` | Pre-treatment RMSE > 10% of outcome std (poor fit) |

**Examples:**

**Basic Traditional SC:**
```python
import polars as pl
import causers

# Panel data with 1 treated unit (California) and controls
df = pl.DataFrame({
    'state': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    'year': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    'y': [10.0, 12.0, 14.0, 25.0,   # State 1: treated in year 4
          9.0, 11.0, 13.0, 15.0,    # State 2: control
          11.0, 13.0, 15.0, 17.0],  # State 3: control
    'treated': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
})

result = causers.synthetic_control(
    df, 'state', 'year', 'y', 'treated', seed=42
)

print(f"ATT: {result.att:.4f} ± {result.standard_error:.4f}")
print(f"Weights: {result.unit_weights}")
print(f"Pre-treatment RMSE: {result.pre_treatment_rmse:.4f}")
```

**Penalized SC (more uniform weights):**
```python
# Penalized method adds L2 regularization for smoother weights
result = causers.synthetic_control(
    df, 'state', 'year', 'y', 'treated',
    method="penalized",  # Auto-selects lambda via LOOCV
    seed=42
)

print(f"Lambda used: {result.lambda_used}")
print(f"Weights: {result.unit_weights}")
```

**Robust SC (matching dynamics):**
```python
# Robust method de-means data to match dynamics instead of levels
result = causers.synthetic_control(
    df, 'state', 'year', 'y', 'treated',
    method="robust",
    seed=42
)

print(f"ATT: {result.att:.4f}")
```

**Augmented SC (bias correction):**
```python
# Augmented method adds ridge regression bias correction
result = causers.synthetic_control(
    df, 'state', 'year', 'y', 'treated',
    method="augmented",
    lambda_param=0.1,  # Explicit regularization
    seed=42
)

print(f"ATT: {result.att:.4f}")
```

**Without standard errors (faster):**
```python
result = causers.synthetic_control(
    df, 'state', 'year', 'y', 'treated',
    compute_se=False  # Skip in-space placebo
)

print(f"ATT: {result.att:.4f} (SE not computed)")
```

**Mathematical Details:**

**Traditional SC:**
Finds weights ω that minimize pre-treatment MSE:
```
ω̂ = argmin_{ω ≥ 0, Σω = 1} Σ_t∈pre (Y₁ₜ - Σⱼ ωⱼ Yⱼₜ)²
```

**Penalized SC:**
Adds L2 regularization:
```
ω̂ = argmin_{ω ≥ 0, Σω = 1} Σ_t (Y₁ₜ - Σⱼ ωⱼ Yⱼₜ)² + λ Σⱼ ωⱼ²
```

**Robust SC:**
De-means outcomes before optimization to match dynamics.

**Augmented SC:**
Adds bias correction via ridge regression on pre-treatment fit residuals.

**ATT Computation:**
```
τ̂_SC = (1/|post|) Σ_{t∈post} (Y₁ₜ - Σⱼ ω̂ⱼ Yⱼₜ)
```

**Standard Errors (In-Space Placebo):**
For each control unit, treat it as placebo, compute SC and ATT. SE = std(placebo ATTs).

> **References:**
> - Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies." *JASA*.
> - Ben-Michael, E., Feller, A., & Rothstein, J. (2021). "The Augmented Synthetic Control Method." *JASA*.

---

## Result Classes

### `LinearRegressionResult`

Container for linear regression results.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `List[float]` | Regression coefficients (β₁, β₂, ..., βₖ) for each predictor |
| `slope` | `float \| None` | Single coefficient (backward compatibility). Available only for single covariate regression. For multiple covariates, use `coefficients[0]`. |
| `intercept` | `float \| None` | Y-intercept (β₀) representing predicted y when all x = 0. `None` when `include_intercept=False` |
| `r_squared` | `float` | Coefficient of determination, proportion of variance explained ∈ [0, 1] |
| `n_samples` | `int` | Number of valid data points used in regression |
| `standard_errors` | `List[float]` | Robust standard errors for each coefficient. HC3 (default) or cluster-robust when `cluster` specified. |
| `intercept_se` | `float \| None` | Robust standard error for intercept. `None` when `include_intercept=False`. |
| `n_clusters` | `int \| None` | Number of unique clusters. `None` if `cluster` not specified. |
| `cluster_se_type` | `str \| None` | Type of clustered SE: `"analytical"`, `"bootstrap_rademacher"`, or `"bootstrap_webb"`. `None` if not clustered. |
| `bootstrap_iterations_used` | `int \| None` | Number of bootstrap iterations used. `None` if not bootstrap. |

**Methods:**

#### `__repr__()`
Returns a string representation suitable for debugging.

```python
>>> result
LinearRegressionResult(slope=2.0, intercept=3.0, r_squared=0.95, n_samples=1000)
```

#### `__str__()`
Returns a human-readable summary.

```python
>>> print(result)
Linear Regression Results:
  Slope: 2.000
  Intercept: 3.000
  R-squared: 0.950
  Samples: 1000
```

**Example Usage:**

**Single covariate:**
```python
result = causers.linear_regression(df, x_cols="x", y_col="y")

# Access attributes
slope = result.slope  # or result.coefficients[0]
intercept = result.intercept
r2 = result.r_squared
n = result.n_samples

# Use in predictions
def predict(x_value):
    return result.slope * x_value + result.intercept
```

**Multiple covariates:**
```python
result = causers.linear_regression(
    df,
    x_cols=["size", "age", "bedrooms"],
    y_col="price"
)

# Access coefficients
size_coef = result.coefficients[0]
age_coef = result.coefficients[1]
beds_coef = result.coefficients[2]
intercept = result.intercept

# Use in predictions
def predict_price(size, age, bedrooms):
    return (result.coefficients[0] * size +
            result.coefficients[1] * age +
            result.coefficients[2] * bedrooms +
            result.intercept)

# Interpret results
if result.r_squared > 0.8:
    print("Strong linear relationship")
elif result.r_squared > 0.5:
    print("Moderate linear relationship")
else:
    print("Weak linear relationship")
```

**Without intercept:**
```python
result = causers.linear_regression(
    df,
    x_cols="x",
    y_col="y",
    include_intercept=False
)

# Access attributes
coef = result.coefficients[0]
intercept = result.intercept  # None
slope = result.slope  # None (use coefficients instead)

# Use in predictions
def predict(x_value):
    return result.coefficients[0] * x_value
```

### `LogisticRegressionResult`

Container for logistic regression results.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `List[float]` | Coefficient estimates for each predictor (log-odds scale) |
| `intercept` | `float \| None` | Intercept term. `None` when `include_intercept=False` |
| `standard_errors` | `List[float]` | Robust standard errors for each coefficient. HC3 (default) or cluster-robust when `cluster` specified. |
| `intercept_se` | `float \| None` | Robust standard error for intercept. `None` when `include_intercept=False`. |
| `n_samples` | `int` | Number of observations used |
| `n_clusters` | `int \| None` | Number of unique clusters. `None` if `cluster` not specified. |
| `cluster_se_type` | `str \| None` | Type of clustered SE: `"analytical"`, `"bootstrap_rademacher"`, or `"bootstrap_webb"`. `None` if not clustered. |
| `bootstrap_iterations_used` | `int \| None` | Number of bootstrap iterations used. `None` if not bootstrap. |
| `converged` | `bool` | Whether the Newton-Raphson optimizer converged |
| `iterations` | `int` | Number of iterations used to reach convergence |
| `log_likelihood` | `float` | Log-likelihood at the MLE solution (negative value) |
| `pseudo_r_squared` | `float` | McFadden's pseudo R² ∈ [0, 1] |

**Methods:**

#### `__repr__()`
Returns a string representation suitable for debugging.

```python
>>> result
LogisticRegressionResult(coefficients=[1.5], intercept=-2.3, converged=True, pseudo_r_squared=0.32)
```

**Example Usage:**

```python
import math
import polars as pl
import causers

# Binary outcome data
df = pl.DataFrame({
    "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "y": [0, 0, 0, 1, 0, 1, 1, 1]
})

result = causers.logistic_regression(df, x_cols="x", y_col="y")

# Access basic attributes
print(f"Coefficient (log-odds): {result.coefficients[0]:.4f}")
print(f"Standard error: {result.standard_errors[0]:.4f}")
print(f"Intercept: {result.intercept:.4f}")

# Check convergence
if result.converged:
    print(f"Converged in {result.iterations} iterations")
else:
    print("Warning: Did not converge!")

# Model fit
print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"McFadden R²: {result.pseudo_r_squared:.4f}")

# Convert to odds ratio
odds_ratio = math.exp(result.coefficients[0])
print(f"Odds ratio: {odds_ratio:.2f}")

# With clustered SE
df_clustered = df.with_columns(pl.Series("cluster", [1, 1, 2, 2, 3, 3, 4, 4]))
result_clustered = causers.logistic_regression(
    df_clustered, "x", "y", cluster="cluster"
)
print(f"Clustered SE: {result_clustered.standard_errors[0]:.4f}")
print(f"Number of clusters: {result_clustered.n_clusters}")
print(f"SE type: {result_clustered.cluster_se_type}")
```

### `SyntheticDIDResult`

Container for Synthetic Difference-in-Differences results.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `att` | `float` | Average Treatment Effect on the Treated |
| `standard_error` | `float` | Bootstrap standard error of the ATT |
| `unit_weights` | `List[float]` | Weights assigned to each control unit (sums to 1) |
| `time_weights` | `List[float]` | Weights assigned to each pre-treatment period (sums to 1) |
| `n_units_control` | `int` | Number of control units |
| `n_units_treated` | `int` | Number of treated units |
| `n_periods_pre` | `int` | Number of pre-treatment periods |
| `n_periods_post` | `int` | Number of post-treatment periods |
| `solver_iterations` | `Tuple[int, int]` | Number of iterations for (unit_weights, time_weights) optimization |
| `solver_converged` | `bool` | Whether the Frank-Wolfe solver converged |
| `pre_treatment_fit` | `float` | RMSE of pre-treatment fit (lower is better) |
| `bootstrap_iterations_used` | `int` | Number of successful bootstrap iterations |

**Methods:**

#### `__repr__()`
Returns a string representation suitable for debugging.

```python
>>> result
SyntheticDIDResult(att=2.5, se=0.3, n_treated=1, n_control=5)
```

#### `__str__()`
Returns a human-readable summary.

```python
>>> print(result)
Synthetic DID Results:
  ATT: 2.500 ± 0.300
  Pre-treatment fit RMSE: 0.123
  Treated units: 1, Control units: 5
  Pre-periods: 8, Post-periods: 3
```

**Example Usage:**

```python
import polars as pl
import causers

# Create panel data
df = pl.DataFrame({
    'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'y': [1.0, 2.0, 5.0, 1.5, 2.5, 3.0, 1.2, 2.2, 2.8],
    'treated': [0, 0, 1, 0, 0, 0, 0, 0, 0]
})

result = causers.synthetic_did(df, 'unit', 'time', 'y', 'treated', seed=42)

# Access basic attributes
print(f"ATT: {result.att:.4f}")
print(f"SE: {result.standard_error:.4f}")

# Access weights
print(f"Control unit weights: {result.unit_weights}")
print(f"Time weights: {result.time_weights}")
print(f"Unit weights sum to: {sum(result.unit_weights):.6f}")  # Should be 1.0
print(f"Time weights sum to: {sum(result.time_weights):.6f}")  # Should be 1.0

# Access panel structure
print(f"Treated units: {result.n_units_treated}")
print(f"Control units: {result.n_units_control}")
print(f"Pre-periods: {result.n_periods_pre}")
print(f"Post-periods: {result.n_periods_post}")

# Access diagnostics
print(f"Pre-treatment fit RMSE: {result.pre_treatment_fit:.4f}")
print(f"Converged: {result.solver_converged}")
print(f"Solver iterations (unit, time): {result.solver_iterations}")
print(f"Bootstrap iterations used: {result.bootstrap_iterations_used}")

# Construct confidence interval
ci_lower = result.att - 1.96 * result.standard_error
ci_upper = result.att + 1.96 * result.standard_error
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

### `SyntheticControlResult`

Container for Synthetic Control results.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `att` | `float` | Average Treatment Effect on the Treated |
| `standard_error` | `float \| None` | In-space placebo standard error. `None` if `compute_se=False` |
| `unit_weights` | `List[float]` | Weights assigned to each control unit (sums to 1) |
| `pre_treatment_rmse` | `float` | Root Mean Squared Error of pre-treatment fit |
| `pre_treatment_mse` | `float` | Mean Squared Error of pre-treatment fit |
| `method` | `str` | Method used: "traditional", "penalized", "robust", or "augmented" |
| `lambda_used` | `float \| None` | Regularization parameter (for penalized/augmented methods) |
| `n_units_control` | `int` | Number of control units |
| `n_periods_pre` | `int` | Number of pre-treatment periods |
| `n_periods_post` | `int` | Number of post-treatment periods |
| `solver_converged` | `bool` | Whether Frank-Wolfe solver converged |
| `solver_iterations` | `int` | Number of optimizer iterations |
| `n_placebo_used` | `int \| None` | Number of successful placebo iterations |

**Methods:**

#### `__repr__()`
Returns a string representation suitable for debugging.

```python
>>> result
SyntheticControlResult(att=5.0, se=0.5, method='traditional', n_control=10)
```

#### `__str__()`
Returns a human-readable summary.

```python
>>> print(result)
Synthetic Control Results:
  Method: traditional
  ATT: 5.000 ± 0.500
  Pre-treatment RMSE: 0.123
  Control units: 10
  Pre-periods: 8, Post-periods: 3
```

**Example Usage:**

```python
import polars as pl
import causers

# Create panel data
df = pl.DataFrame({
    'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'y': [1.0, 2.0, 8.0, 1.5, 2.5, 3.0, 1.2, 2.2, 2.8],
    'treated': [0, 0, 1, 0, 0, 0, 0, 0, 0]
})

result = causers.synthetic_control(df, 'unit', 'time', 'y', 'treated', seed=42)

# Access basic attributes
print(f"ATT: {result.att:.4f}")
print(f"SE: {result.standard_error:.4f}")
print(f"Method: {result.method}")

# Access weights
print(f"Control unit weights: {result.unit_weights}")
print(f"Weights sum to: {sum(result.unit_weights):.6f}")  # Should be 1.0

# Access diagnostics
print(f"Pre-treatment RMSE: {result.pre_treatment_rmse:.4f}")
print(f"Converged: {result.solver_converged}")
print(f"Iterations: {result.solver_iterations}")

# Construct confidence interval
if result.standard_error is not None:
    ci_lower = result.att - 1.96 * result.standard_error
    ci_upper = result.att + 1.96 * result.standard_error
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

## Type Hints

The package includes full type hints for better IDE support and static type checking:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    from causers import LinearRegressionResult, LogisticRegressionResult
    
    def analyze_linear(df: pl.DataFrame) -> LinearRegressionResult:
        return causers.linear_regression(df, "x", "y")
    
    def analyze_logistic(df: pl.DataFrame) -> LogisticRegressionResult:
        return causers.logistic_regression(df, "x", "outcome")
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|--------|
| `linear_regression()` | O(n × k²) | n = rows, k = covariates |
| `logistic_regression()` | O(n × k² × i) | i = iterations (≤35) |
| Memory usage | O(n × k) | Design matrix storage |

### Performance Benchmarks

Measured on Apple M1 Pro (16GB RAM):

**Linear Regression:**

| Rows | Time | Memory Peak |
|------|------|-------------|
| 1K | 0.8ms | ~16KB |
| 10K | 1.5ms | ~160KB |
| 100K | 4.2ms | ~1.6MB |
| 1M | 45ms | ~16MB |
| 10M | 450ms | ~160MB |

**Logistic Regression:**

| Rows | Time | Notes |
|------|------|-------|
| 1K | ~2ms | Typical convergence in 5-8 iterations |
| 10K | ~5ms | |
| 100K | ~30ms | |
| 1M | ~300ms | Requirement: <500ms ✅ |

**Bootstrap Performance:**

| Rows | B=1000 | Notes |
|------|--------|-------|
| 100K | ~3-5s | Score bootstrap for logistic |
| 100K | ~2-3s | Wild bootstrap for linear |

## Error Handling

### Common Errors and Solutions

#### ValueError: Column not found
```python
# Error
>>> causers.linear_regression(df, "missing_col", "y")
ValueError: Column 'missing_col' not found in DataFrame

# Solution: Check column names
>>> print(df.columns)
['x', 'y', 'z']
```

#### ValueError: Zero variance
```python
# Error: All x values are the same
>>> df = pl.DataFrame({"x": [1, 1, 1], "y": [2, 3, 4]})
>>> causers.linear_regression(df, "x", "y")
ValueError: Cannot perform regression: x values have zero variance

# Solution: Ensure x has variation
>>> df = pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4]})
```

#### TypeError: Non-numeric data
```python
# Error: String column
>>> df = pl.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
>>> causers.linear_regression(df, "x", "y")
TypeError: Column 'x' contains non-numeric data

# Solution: Convert to numeric or use different column
>>> df = df.with_columns(pl.col("x").cast(pl.Int64))
```

#### ValueError: Binary outcome required (logistic regression)
```python
# Error: y contains values other than 0 and 1
>>> df = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 2]})
>>> causers.logistic_regression(df, "x", "y")
ValueError: y_col must contain only 0 and 1 values

# Solution: Ensure y is binary
>>> df = df.filter(pl.col("y").is_in([0, 1]))
```

#### ValueError: Perfect separation (logistic regression)
```python
# Error: x perfectly predicts y
>>> df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [0, 0, 1, 1]})
>>> causers.logistic_regression(df, "x", "y")
ValueError: Perfect separation detected; logistic regression cannot converge

# Solution: Add noise, use different predictors, or use penalized regression
```

#### ValueError: Convergence failure (logistic regression)
```python
# Error: Optimizer fails to converge
>>> causers.logistic_regression(df, "x", "y")
ValueError: Convergence failed after 35 iterations

# Solution: Check for collinearity, scale predictors, or simplify model
```

## Best Practices

### 1. Data Validation

Always validate your data before regression:

```python
def safe_regression(df, x_col, y_col):
    # Check columns exist
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns {x_col}, {y_col} must exist")
    
    # Check for numeric types
    if not df[x_col].dtype.is_numeric():
        raise TypeError(f"{x_col} must be numeric")
    
    # Check for sufficient data
    if len(df) < 2:
        raise ValueError("Need at least 2 data points")
    
    # Filter out nulls if needed
    df_clean = df.filter(
        pl.col(x_col).is_not_null() & 
        pl.col(y_col).is_not_null()
    )
    
    return causers.linear_regression(df_clean, x_col, y_col)
```

### 2. Interpreting Results

```python
def interpret_regression(result):
    """Provide interpretation of regression results."""
    
    # Relationship strength
    if result.r_squared > 0.9:
        strength = "very strong"
    elif result.r_squared > 0.7:
        strength = "strong"
    elif result.r_squared > 0.5:
        strength = "moderate"
    elif result.r_squared > 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    
    print(f"Linear relationship: {strength}")
    print(f"Model explains {result.r_squared * 100:.1f}% of variance")
    print(f"Each unit increase in x changes y by {result.slope:.3f}")
    
    return strength
```

### 3. Large Dataset Optimization

For very large datasets, consider chunking:

```python
def regression_on_large_data(df, x_col, y_col, chunk_size=1_000_000):
    """Process very large datasets in chunks."""
    
    if len(df) <= chunk_size:
        return causers.linear_regression(df, x_col, y_col)
    
    # For datasets larger than memory, use Polars lazy evaluation
    # This is a simplified example - actual implementation would
    # need to aggregate statistics properly
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df[i:i + chunk_size]
        result = causers.linear_regression(chunk, x_col, y_col)
        results.append(result)
    
    # Combine results (simplified - actual math is more complex)
    return combine_regression_results(results)
```

## Migration from Other Libraries

### From NumPy/SciPy

**Single covariate:**
```python
# NumPy/SciPy approach
import numpy as np
from scipy import stats

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 8, 10])
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

# causers approach
import polars as pl
import causers

df = pl.DataFrame({"x": x, "y": y})
result = causers.linear_regression(df, x_cols="x", y_col="y")
# Access: result.slope, result.intercept, result.r_squared
```

### From pandas/statsmodels

**Multiple regression:**
```python
# pandas/statsmodels approach
import pandas as pd
import statsmodels.api as sm

df_pandas = pd.DataFrame({
    "x1": [1, 2, 3, 4, 5],
    "x2": [2, 3, 4, 5, 6],
    "y": [5, 8, 11, 14, 17]
})
X = sm.add_constant(df_pandas[["x1", "x2"]])
model = sm.OLS(df_pandas["y"], X).fit()
coefficients = model.params[1:].tolist()
intercept = model.params[0]
r_squared = model.rsquared

# causers approach
df = pl.from_pandas(df_pandas)
result = causers.linear_regression(df, x_cols=["x1", "x2"], y_col="y")
# Access: result.coefficients, result.intercept, result.r_squared
```

**No intercept:**
```python
# statsmodels approach (no intercept)
X = df_pandas[["x1", "x2"]]  # No constant added
model = sm.OLS(df_pandas["y"], X).fit()

# causers approach
result = causers.linear_regression(
    df,
    x_cols=["x1", "x2"],
    y_col="y",
    include_intercept=False
)
```

### From scikit-learn

**Single covariate:**
```python
# scikit-learn approach
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 8, 10])
model = LinearRegression().fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)

# causers approach
import polars as pl
import causers

df = pl.DataFrame({"x": X.flatten(), "y": y})
result = causers.linear_regression(df, x_cols="x", y_col="y")
# Access: result.slope, result.intercept, result.r_squared
```

**Multiple covariates:**
```python
# scikit-learn approach
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([5, 8, 11, 14, 17])
model = LinearRegression().fit(X, y)
coefficients = model.coef_.tolist()
intercept = model.intercept_

# causers approach
df = pl.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})
result = causers.linear_regression(df, x_cols=["x1", "x2"], y_col="y")
# Access: result.coefficients, result.intercept
```

## Limitations

### Current Limitations (v0.5.0)

1. **Single-way clustering only**: Multi-way clustering (e.g., two-way by firm and time) not yet supported
2. **No weights**: Weighted least squares not implemented
3. **No confidence intervals**: P-values and confidence intervals not included (SE available)
4. **Bootstrap parallelization**: Bootstrap iterations run sequentially (parallelization planned)
5. **Synthetic Control**: Single treated unit only (use synthetic_did for multiple treated units)

### Memory Limitations

- Maximum DataFrame size: ~80% of available RAM
- No streaming support for out-of-memory datasets
- All computations are in-memory

### Numerical Limitations

- Floating-point precision: IEEE-754 double (64-bit)
- Very small/large numbers may have precision issues
- Ill-conditioned problems may give inaccurate results

## Troubleshooting

### Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'causers._causers'`

**Solution**: The Rust extension didn't build properly.
```bash
# Rebuild from source
maturin develop --release
```

**Problem**: `ImportError: DLL load failed`

**Solution**: Missing Visual C++ redistributables on Windows.
```bash
# Install Visual C++ redistributables
# Download from Microsoft website
```

### Performance Issues

**Problem**: Slower than expected performance

**Solutions**:
1. Ensure you built with `--release` flag
2. Check you're not in debug mode
3. Verify Polars is using optimal settings

```python
# Check if using release build
import causers
print(causers.__file__)  # Should be in site-packages, not local dev

# Optimize Polars settings
import polars as pl
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_rows(10)
```

## Support

- **Documentation**: https://github.com/causers/causers
- **Issues**: https://github.com/causers/causers/issues
- **Discussions**: https://github.com/causers/causers/discussions
- **Security**: Report to security@example.com

---

Last updated: 2025-12-25 | causers v0.5.0