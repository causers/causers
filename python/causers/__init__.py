"""
causers - High-performance statistical operations for Polars DataFrames

A Python package with Rust backend for fast statistical computations
on Polars DataFrames.
"""

from typing import List as _List, Optional as _Optional, Union as _Union
import warnings as _warnings
import polars as _polars

__version__ = "0.6.0"

# Import the Rust extension module
from causers._causers import (
    LinearRegressionResult,
    LogisticRegressionResult,
    SyntheticDIDResult,
    SyntheticControlResult,
    linear_regression as _linear_regression_rust,
    logistic_regression as _logistic_regression_rust,
    synthetic_did_impl as _synthetic_did_impl,
    synthetic_control_impl as _synthetic_control_impl,
)

# Re-export main functions
__all__ = [
    "LinearRegressionResult",
    "LogisticRegressionResult",
    "SyntheticDIDResult",
    "SyntheticControlResult",
    "linear_regression",
    "logistic_regression",
    "synthetic_did",
    "synthetic_control",
]


def _check_cluster_balance(
    df: _polars.DataFrame, cluster_col: str
) -> _Optional[str]:
    """
    Check cluster balance and return warning message if any cluster has >50% observations.
    
    Returns None if balanced, otherwise returns the warning message.
    """
    try:
        value_counts = df[cluster_col].value_counts()
        total = len(df)
        threshold = total // 2  # 50%
        
        for row in value_counts.iter_rows():
            cluster_val, count = row[0], row[1]
            if count > threshold:
                pct = (count * 100) // total
                return (
                    f"Cluster '{cluster_val}' contains {pct}% of observations ({count}/{total}). "
                    f"Clustered standard errors may be unreliable with such imbalanced clusters."
                )
        return None
    except Exception:
        # If we can't check balance, don't warn
        return None


def linear_regression(
    df: _polars.DataFrame,
    x_cols: _Union[str, _List[str]],
    y_col: str,
    include_intercept: bool = True,
    cluster: _Optional[str] = None,
    bootstrap: bool = False,
    bootstrap_iterations: int = 1000,
    seed: _Optional[int] = None,
    bootstrap_method: str = "rademacher",
) -> LinearRegressionResult:
    """
    Perform linear regression on Polars DataFrame columns.
    
    Supports both single and multiple covariate regression using ordinary
    least squares (OLS). For multiple covariates, uses matrix operations:
    β = (X'X)^-1 X'y
    
    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the data
    x_cols : str or List[str]
        Name(s) of the independent variable column(s). Can be:
        - A single column name as a string
        - A list of column names for multiple covariates
    y_col : str
        Name of the dependent variable column
    include_intercept : bool, default=True
        Whether to include an intercept term in the regression.
        If False, forces the regression line through the origin.
    cluster : str, optional
        Column name for cluster identifiers. When specified, computes
        cluster-robust standard errors instead of HC3. Supports integer,
        string, or categorical columns.
    bootstrap : bool, default=False
        If True and cluster is specified, use wild cluster bootstrap
        for standard error computation. Requires cluster to be specified.
        Recommended when number of clusters is less than 42.
    bootstrap_iterations : int, default=1000
        Number of bootstrap replications when bootstrap=True.
    seed : int, optional
        Random seed for reproducibility when using bootstrap. When None,
        uses a random seed which may produce different results each call.
    bootstrap_method : str, default "rademacher"
        Weight distribution for wild bootstrap. Options:
        - "rademacher": Standard Rademacher weights (±1 with equal probability)
        - "webb": Webb's 6-point distribution for improved small-sample performance
        Only used when bootstrap=True and cluster is specified.
    
    Returns
    -------
    LinearRegressionResult
        Result object with the following attributes:
        - coefficients : List[float]
            Regression coefficients for each x variable
        - intercept : float or None
            Intercept term (None if include_intercept=False)
        - r_squared : float
            Coefficient of determination (R²)
        - n_samples : int
            Number of samples used in the regression
        - slope : float or None
            For single covariate only, same as coefficients[0]
        - standard_errors : List[float]
            Robust standard errors for each coefficient. Uses HC3 by
            default, or cluster-robust SE if cluster is specified.
        - intercept_se : float or None
            Robust standard error for intercept (None if include_intercept=False)
        - n_clusters : int or None
            Number of unique clusters (None if cluster not specified)
        - cluster_se_type : str or None
            Type of clustered SE: "analytical", "bootstrap_rademacher", or
            "bootstrap_webb" (None if not clustered)
        - bootstrap_iterations_used : int or None
            Number of bootstrap iterations (None if not bootstrap)
    
    Raises
    ------
    ValueError
        - If x_cols is empty or columns don't exist
        - If cluster column contains null values
        - If bootstrap=True without cluster specified
        - If fewer than 2 clusters detected
        - If single-observation clusters exist (analytical mode only)
        - If numerical instability detected (condition number > 1e10)
        - If bootstrap_iterations < 1
    
    Warns
    -----
    UserWarning
        - When fewer than 42 clusters with bootstrap=False: recommends using
          wild cluster bootstrap for more accurate inference.
        - When cluster column has float dtype: implicit cast to string.
    
    Examples
    --------
    Single covariate regression:
    
    >>> import polars as pl
    >>> import causers
    >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    >>> result = causers.linear_regression(df, "x", "y")
    >>> print(f"y = {result.slope:.2f}x + {result.intercept:.2f}")
    y = 2.00x + 0.00
    
    Accessing standard errors:
    
    >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2.1, 3.9, 6.2, 7.8, 10.1]})
    >>> result = causers.linear_regression(df, "x", "y")
    >>> print(f"Coefficient: {result.coefficients[0]:.4f} ± {result.standard_errors[0]:.4f}")
    Coefficient: 1.9900 ± 0.0682
    >>> print(f"Intercept: {result.intercept:.4f} ± {result.intercept_se:.4f}")
    Intercept: 0.0500 ± 0.1896
    
    Multiple covariate regression:
    
    >>> df = pl.DataFrame({
    ...     "x1": [1, 2, 3, 4, 5],
    ...     "x2": [1, 1, 2, 2, 3],
    ...     "y": [6, 8, 13, 15, 20]
    ... })
    >>> result = causers.linear_regression(df, ["x1", "x2"], "y")
    >>> print(f"Coefficients: {result.coefficients}")
    Coefficients: [2.0, 3.0]
    
    Clustered standard errors (analytical):
    
    >>> df = pl.DataFrame({
    ...     "x": [1, 2, 3, 4, 5, 6],
    ...     "y": [2, 4, 5, 8, 9, 12],
    ...     "firm_id": [1, 1, 2, 2, 3, 3]
    ... })
    >>> result = causers.linear_regression(df, "x", "y", cluster="firm_id")
    >>> print(f"Clustered SE: {result.standard_errors[0]:.4f} (G={result.n_clusters})")
    Clustered SE: ... (G=3)
    
    Wild cluster bootstrap (recommended for <42 clusters):
    
    >>> result = causers.linear_regression(
    ...     df, "x", "y",
    ...     cluster="firm_id", bootstrap=True, seed=42
    ... )
    >>> print(f"Bootstrap SE: {result.standard_errors[0]:.4f}")
    Bootstrap SE: ...
    
    Notes
    -----
    Standard errors are computed using:
    
    - **HC3 (default)**: Heteroskedasticity-consistent standard errors
      when no cluster is specified. Provides robust inference when error
      variance may not be constant (MacKinnon & White, 1985).
    
    - **Analytical clustered SE**: When cluster is specified and bootstrap=False.
      Uses the sandwich estimator with small-sample adjustment (G/(G-1) × (N-1)/(N-k)).
      Accounts for within-cluster correlation.
    
    - **Wild cluster bootstrap SE**: When cluster and bootstrap=True.
      Uses Rademacher weights (±1 with equal probability) and is recommended
      when the number of clusters is small (G < 42).
    
    The 42-cluster threshold is based on asymptotic theory and simulation
    evidence that analytical clustered SE can be unreliable with few clusters.
    
    References
    ----------
    Cameron, A. C., & Miller, D. L. (2015). A Practitioner's Guide to
    Cluster-Robust Inference. Journal of Human Resources, 50(2), 317-372.
    
    MacKinnon, J. G., & Webb, M. D. (2018). The wild bootstrap for few
    (treated) clusters. The Econometrics Journal, 21(2), 114-135.
    
    See Also
    --------
    LinearRegressionResult : Result class with coefficient estimates and diagnostics.
    """
    # Normalize x_cols to always be a list
    if isinstance(x_cols, str):
        x_cols_list = [x_cols]
    else:
        x_cols_list = list(x_cols)
    
    # Check for float cluster column and emit warning (REQ-031)
    if cluster is not None:
        try:
            cluster_dtype = df[cluster].dtype
            if cluster_dtype in (_polars.Float32, _polars.Float64):
                _warnings.warn(
                    f"Cluster column '{cluster}' is float; will be cast to string for grouping.",
                    UserWarning,
                    stacklevel=2
                )
        except Exception:
            pass  # Let the Rust layer handle column not found errors
    
    # Normalize bootstrap_method to lowercase for case-insensitive matching
    bootstrap_method_normalized = bootstrap_method.lower()
    
    # Call the Rust implementation
    result = _linear_regression_rust(
        df,
        x_cols_list,
        y_col,
        include_intercept,
        cluster,
        bootstrap,
        bootstrap_iterations,
        seed,
        bootstrap_method_normalized,
    )
    
    # Check for small cluster count and emit warning (REQ-030)
    if result.n_clusters is not None and not bootstrap:
        if result.n_clusters < 42:
            _warnings.warn(
                f"Only {result.n_clusters} clusters detected. Wild cluster bootstrap "
                f"(bootstrap=True) is recommended when clusters < 42.",
                UserWarning,
                stacklevel=2
            )
    
    # Check for cluster imbalance and emit warning (REQ-032)
    if cluster is not None:
        balance_warning = _check_cluster_balance(df, cluster)
        if balance_warning is not None:
            _warnings.warn(balance_warning, UserWarning, stacklevel=2)
    
    return result


def logistic_regression(
    df: _polars.DataFrame,
    x_cols: _Union[str, _List[str]],
    y_col: str,
    include_intercept: bool = True,
    cluster: _Optional[str] = None,
    bootstrap: bool = False,
    bootstrap_iterations: int = 1000,
    seed: _Optional[int] = None,
    bootstrap_method: str = "rademacher",
) -> LogisticRegressionResult:
    """
    Perform logistic regression on binary outcome with robust standard errors.
    
    Fits a logistic regression model using Maximum Likelihood Estimation (MLE)
    with Newton-Raphson optimization. Returns coefficient estimates (log-odds),
    robust standard errors, and diagnostic information.
    
    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the data
    x_cols : str or List[str]
        Name(s) of the independent variable column(s). Can be:
        - A single column name as a string
        - A list of column names for multiple covariates
    y_col : str
        Name of the binary outcome column (must contain only 0 and 1)
    include_intercept : bool, default=True
        Whether to include an intercept term in the regression.
    cluster : str, optional
        Column name for cluster identifiers. When specified, computes
        cluster-robust standard errors using the score-based approach.
        Supports integer, string, or categorical columns.
    bootstrap : bool, default=False
        If True and cluster is specified, use score bootstrap
        for standard error computation. Requires cluster to be specified.
        Recommended when number of clusters is less than 42.
    bootstrap_iterations : int, default=1000
        Number of bootstrap replications when bootstrap=True.
    seed : int, optional
        Random seed for reproducibility when using bootstrap. When None,
        uses a random seed which may produce different results each call.
    bootstrap_method : str, default "rademacher"
        Weight distribution for score bootstrap. Options:
        - "rademacher": Standard Rademacher weights (±1 with equal probability)
        - "webb": Webb's 6-point distribution for improved small-sample performance
        Only used when bootstrap=True and cluster is specified.
    
    Returns
    -------
    LogisticRegressionResult
        Result object with the following attributes:
        - coefficients : List[float]
            Coefficient estimates for x variables (log-odds scale)
        - intercept : float or None
            Intercept term (None if include_intercept=False)
        - standard_errors : List[float]
            Robust standard errors for each coefficient. Uses HC3 by
            default, or clustered SE if cluster is specified.
        - intercept_se : float or None
            Robust standard error for intercept (None if include_intercept=False)
        - n_samples : int
            Number of observations used
        - n_clusters : int or None
            Number of unique clusters (None if cluster not specified)
        - cluster_se_type : str or None
            Type of clustered SE: "analytical", "bootstrap_rademacher", or
            "bootstrap_webb" (None if not clustered)
        - bootstrap_iterations_used : int or None
            Number of bootstrap iterations (None if not bootstrap)
        - converged : bool
            Whether the MLE optimizer converged
        - iterations : int
            Number of Newton-Raphson iterations used
        - log_likelihood : float
            Log-likelihood at the MLE solution
        - pseudo_r_squared : float
            McFadden's pseudo R² = 1 - (LL_model / LL_null)
    
    Raises
    ------
    ValueError
        - If y_col contains values other than 0 and 1
        - If y_col contains only 0s or only 1s
        - If x_cols is empty or columns don't exist
        - If cluster column contains null values
        - If bootstrap=True without cluster specified
        - If fewer than 2 clusters detected
        - If perfect separation is detected
        - If Hessian is singular (collinearity)
        - If convergence fails after max iterations
        - If numerical instability detected (condition number > 1e10)
        - If bootstrap_iterations < 1
    
    Warns
    -----
    UserWarning
        - When fewer than 42 clusters with bootstrap=False: recommends using
          score bootstrap for more accurate inference.
        - When cluster column has float dtype: implicit cast to string.
    
    Examples
    --------
    Simple logistic regression:
    
    >>> import polars as pl
    >>> import causers
    >>> df = pl.DataFrame({
    ...     "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    ...     "y": [0, 0, 0, 1, 1, 1]
    ... })
    >>> result = causers.logistic_regression(df, "x", "y")
    >>> print(f"Coefficient: {result.coefficients[0]:.4f}")
    Coefficient: ...
    
    Accessing convergence information:
    
    >>> if result.converged:
    ...     print(f"Converged in {result.iterations} iterations")
    ...     print(f"Log-likelihood: {result.log_likelihood:.2f}")
    ...     print(f"McFadden R²: {result.pseudo_r_squared:.3f}")
    Converged in ... iterations
    Log-likelihood: ...
    McFadden R²: ...
    
    Multiple covariates:
    
    >>> df = pl.DataFrame({
    ...     "x1": [1, 2, 3, 4, 5, 6],
    ...     "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ...     "y": [0, 0, 0, 1, 1, 1]
    ... })
    >>> result = causers.logistic_regression(df, ["x1", "x2"], "y")
    >>> print(f"Coefficients: {result.coefficients}")
    Coefficients: [...]
    
    Clustered standard errors:
    
    >>> df = pl.DataFrame({
    ...     "x": [1, 2, 3, 4, 5, 6],
    ...     "y": [0, 0, 1, 0, 1, 1],
    ...     "firm_id": [1, 1, 2, 2, 3, 3]
    ... })
    >>> result = causers.logistic_regression(df, "x", "y", cluster="firm_id")
    >>> print(f"Clustered SE: {result.standard_errors[0]:.4f} (G={result.n_clusters})")
    Clustered SE: ... (G=3)
    
    Score bootstrap (recommended for <42 clusters):
    
    >>> result = causers.logistic_regression(
    ...     df, "x", "y",
    ...     cluster="firm_id", bootstrap=True, seed=42
    ... )
    >>> print(f"Bootstrap SE: {result.standard_errors[0]:.4f}")
    Bootstrap SE: ...
    
    Notes
    -----
    The logistic regression model is:
    
        P(y=1|x) = 1 / (1 + exp(-x'β))
    
    Coefficients are on the log-odds scale. To convert to odds ratios,
    use exp(coefficient).
    
    Standard errors are computed using:
    
    - **HC3 (default)**: Heteroskedasticity-consistent standard errors
      adapted for logistic regression, using weighted leverages.
    
    - **Analytical clustered SE**: When cluster is specified and bootstrap=False.
      Uses the sandwich estimator with cluster-level scores.
    
    - **Score bootstrap SE**: When cluster and bootstrap=True.
      Uses Rademacher weights (±1 with equal probability) following
      Kline & Santos (2012). Recommended for small cluster counts (G < 42).
    
    The optimizer uses Newton-Raphson with step halving for stability,
    converging when gradient norm < 1e-8 or after 35 iterations.
    
    References
    ----------
    Kline, P., & Santos, A. (2012). "A Score Based Approach to Wild Bootstrap
    Inference." Journal of Econometric Methods, 1(1), 23-41.
    https://doi.org/10.1515/2156-6674.1006
    
    MacKinnon, J. G., & White, H. (1985). "Some heteroskedasticity-consistent
    covariance matrix estimators with improved finite sample properties."
    Journal of Econometrics, 29(3), 305-325.
    
    See Also
    --------
    LogisticRegressionResult : Result class with coefficient estimates and diagnostics.
    linear_regression : For continuous outcome regression.
    """
    # Normalize x_cols to always be a list
    if isinstance(x_cols, str):
        x_cols_list = [x_cols]
    else:
        x_cols_list = list(x_cols)
    
    # Check for float cluster column and emit warning (REQ-031 equivalent)
    if cluster is not None:
        try:
            cluster_dtype = df[cluster].dtype
            if cluster_dtype in (_polars.Float32, _polars.Float64):
                _warnings.warn(
                    f"Cluster column '{cluster}' is float; will be cast to string for grouping.",
                    UserWarning,
                    stacklevel=2
                )
        except Exception:
            pass  # Let the Rust layer handle column not found errors
    
    # Normalize bootstrap_method to lowercase for case-insensitive matching
    bootstrap_method_normalized = bootstrap_method.lower()
    
    # Call the Rust implementation
    result = _logistic_regression_rust(
        df,
        x_cols_list,
        y_col,
        include_intercept,
        cluster,
        bootstrap,
        bootstrap_iterations,
        seed,
        bootstrap_method_normalized,
    )
    
    # Check for small cluster count and emit warning (REQ-060)
    if result.n_clusters is not None and not bootstrap:
        if result.n_clusters < 42:
            _warnings.warn(
                f"Only {result.n_clusters} clusters detected. Score bootstrap "
                f"(bootstrap=True) is recommended when clusters < 42.",
                UserWarning,
                stacklevel=2
            )
    
    return result


def synthetic_did(
    df: _polars.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    bootstrap_iterations: int = 200,
    seed: _Optional[int] = None,
) -> SyntheticDIDResult:
    """
    Compute Synthetic Difference-in-Differences (SDID) estimator.
    
    Implements the SDID estimator from Arkhangelsky et al. (2021), which combines
    synthetic control weighting with difference-in-differences to estimate the
    Average Treatment Effect on the Treated (ATT).
    
    The estimator uses two-stage optimization:
    1. **Unit weights**: Find control unit weights that match pre-treatment trends
    2. **Time weights**: Find pre-period weights that predict post-period outcomes
    
    Standard errors are computed via placebo bootstrap.
    
    Parameters
    ----------
    df : pl.DataFrame
        Panel data in long format with one row per unit-time observation.
        Must be a balanced panel (all units observed in all time periods).
    unit_col : str
        Column name identifying unique units (e.g., "state", "firm_id").
        Must be integer or string type.
    time_col : str
        Column name identifying time periods (e.g., "year", "quarter").
        Must be integer or string type.
    outcome_col : str
        Column name for the outcome variable. Must be numeric.
    treatment_col : str
        Column name for treatment indicator. Must contain only 0 and 1 values.
        Value of 1 indicates the unit is treated in that period.
    bootstrap_iterations : int, default=200
        Number of placebo bootstrap iterations for standard error estimation.
        Must be at least 1. Values < 100 will emit a warning.
    seed : int, optional
        Random seed for reproducibility. If None, uses system time.
    
    Returns
    -------
    SyntheticDIDResult
        Result object with the following attributes:
        
        - att : float
            Average Treatment Effect on the Treated
        - standard_error : float
            Bootstrap standard error of the ATT
        - unit_weights : List[float]
            Weights assigned to each control unit (sums to 1)
        - time_weights : List[float]
            Weights assigned to each pre-treatment period (sums to 1)
        - n_units_control : int
            Number of control units
        - n_units_treated : int
            Number of treated units
        - n_periods_pre : int
            Number of pre-treatment periods
        - n_periods_post : int
            Number of post-treatment periods
        - solver_iterations : Tuple[int, int]
            Number of iterations for (unit_weights, time_weights) optimization
        - solver_converged : bool
            Whether the Frank-Wolfe solver converged
        - pre_treatment_fit : float
            RMSE of pre-treatment fit (lower is better)
        - bootstrap_iterations_used : int
            Number of successful bootstrap iterations
    
    Raises
    ------
    ValueError
        - If DataFrame is empty
        - If any specified column doesn't exist
        - If unit_col or time_col is float type
        - If outcome_col is not numeric
        - If outcome_col contains null values
        - If treatment_col contains values other than 0 and 1
        - If bootstrap_iterations < 1
        - If fewer than 2 control units found
        - If fewer than 2 pre-treatment periods found
        - If no treated units found
        - If no post-treatment periods found
        - If panel is not balanced
    
    Warns
    -----
    UserWarning
        - If any unit weight > 0.5 (weight concentration on single unit)
        - If any time weight > 0.5 (weight concentration on single period)
        - If bootstrap_iterations < 100 (may be unreliable)
    
    Examples
    --------
    Basic usage with panel data:
    
    >>> import polars as pl
    >>> import causers
    >>> df = pl.DataFrame({
    ...     'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    ...     'y': [1.0, 2.0, 5.0, 1.5, 2.5, 3.0, 1.2, 2.2, 2.8],
    ...     'treated': [0, 0, 1, 0, 0, 0, 0, 0, 0]
    ... })
    >>> result = causers.synthetic_did(df, 'unit', 'time', 'y', 'treated', seed=42)
    >>> print(f"ATT: {result.att:.4f} ± {result.standard_error:.4f}")
    ATT: ... ± ...
    
    Accessing weights and diagnostics:
    
    >>> print(f"Control unit weights: {result.unit_weights}")
    Control unit weights: [...]
    >>> print(f"Pre-treatment fit RMSE: {result.pre_treatment_fit:.4f}")
    Pre-treatment fit RMSE: ...
    
    Notes
    -----
    **Panel Structure Detection**
    
    The function automatically detects:
    - **Control units**: Units where treatment=0 in all periods
    - **Treated units**: Units where treatment=1 in at least one period
    - **Pre-periods**: Periods where all observations have treatment=0
    - **Post-periods**: Periods where at least one treated unit has treatment=1
    
    **Algorithm**
    
    The SDID estimator is:
    
    .. math::
    
        \\hat{\\tau}_{sdid} = (\\bar{Y}_{tr,post} - \\bar{Y}_{synth,post})
                            - \\sum_t \\lambda_t (\\bar{Y}_{tr,t} - \\bar{Y}_{synth,t})
    
    where :math:`\\bar{Y}_{synth,t} = \\sum_i \\omega_i Y_{i,t}` uses optimized
    unit weights :math:`\\omega` on control units.
    
    **Standard Errors**
    
    Standard errors are computed via placebo bootstrap:
    1. Randomly select a control unit as "placebo treated"
    2. Re-run SDID with this unit treated
    3. Repeat for bootstrap_iterations
    4. SE = standard deviation of placebo ATTs
    
    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
    Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088-4118.
    
    See Also
    --------
    SyntheticDIDResult : Result class with ATT and diagnostics.
    linear_regression : For standard regression analysis.
    """
    # ========================================================================
    # Input Validation
    # ========================================================================
    
    # Check DataFrame is not empty
    if len(df) == 0:
        raise ValueError("Cannot perform SDID on empty DataFrame")
    
    # Check all required columns exist
    for col_name, col_label in [
        (unit_col, "unit_col"),
        (time_col, "time_col"),
        (outcome_col, "outcome_col"),
        (treatment_col, "treatment_col"),
    ]:
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
    
    # Check unit_col is not float
    unit_dtype = df[unit_col].dtype
    if unit_dtype in (_polars.Float32, _polars.Float64):
        raise ValueError(f"unit_col must be integer or string, not float")
    
    # Check time_col is not float
    time_dtype = df[time_col].dtype
    if time_dtype in (_polars.Float32, _polars.Float64):
        raise ValueError(f"time_col must be integer or string, not float")
    
    # Check outcome_col is numeric
    outcome_dtype = df[outcome_col].dtype
    numeric_types = (
        _polars.Float32, _polars.Float64,
        _polars.Int8, _polars.Int16, _polars.Int32, _polars.Int64,
        _polars.UInt8, _polars.UInt16, _polars.UInt32, _polars.UInt64,
    )
    if outcome_dtype not in numeric_types:
        raise ValueError(f"outcome_col must be numeric")
    
    # Check for nulls in outcome_col
    null_count = df[outcome_col].null_count()
    if null_count > 0:
        raise ValueError(f"outcome_col '{outcome_col}' contains null values")
    
    # Check treatment_col contains only 0 and 1
    treatment_values = df[treatment_col].unique().to_list()
    valid_treatment_values = {0, 1, 0.0, 1.0}
    for val in treatment_values:
        if val not in valid_treatment_values:
            raise ValueError("treatment_col must contain only 0 and 1 values")
    
    # Check bootstrap_iterations >= 0 (0 = no bootstrap, ATT only)
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be at least 0")
    
    # ========================================================================
    # Panel Structure Detection
    # ========================================================================
    
    # Get unique units and periods
    unique_units = df[unit_col].unique().sort().to_list()
    unique_periods = df[time_col].unique().sort().to_list()
    n_units = len(unique_units)
    n_periods = len(unique_periods)
    
    # Create mappings from unit/period values to indices
    unit_to_idx = {unit: idx for idx, unit in enumerate(unique_units)}
    period_to_idx = {period: idx for idx, period in enumerate(unique_periods)}
    
    # Validate balanced panel
    expected_rows = n_units * n_periods
    if len(df) != expected_rows:
        raise ValueError(
            f"Panel is not balanced: expected {expected_rows} rows "
            f"({n_units} units × {n_periods} periods), found {len(df)}"
        )
    
    # Identify control vs treated units
    # Control units: treatment=0 in ALL periods
    # Treated units: treatment=1 in at least one period
    unit_max_treatment = (
        df.group_by(unit_col)
        .agg(_polars.col(treatment_col).max().alias("max_treatment"))
    )
    
    control_units = []
    treated_units = []
    for row in unit_max_treatment.iter_rows():
        unit_val, max_treat = row[0], row[1]
        if max_treat == 0 or max_treat == 0.0:
            control_units.append(unit_val)
        else:
            treated_units.append(unit_val)
    
    # Sort for deterministic ordering
    control_units = sorted(control_units, key=lambda x: unit_to_idx[x])
    treated_units = sorted(treated_units, key=lambda x: unit_to_idx[x])
    
    # Validate sufficient control units
    n_control = len(control_units)
    if n_control < 2:
        raise ValueError(f"At least 2 control units required; found {n_control}")
    
    # Validate at least 1 treated unit
    n_treated = len(treated_units)
    if n_treated == 0:
        raise ValueError("No treated units found in data")
    
    # Identify pre vs post periods
    # Pre-periods: all observations have treatment=0
    # Post-periods: at least one treated unit has treatment=1
    period_max_treatment = (
        df.group_by(time_col)
        .agg(_polars.col(treatment_col).max().alias("max_treatment"))
    )
    
    pre_periods = []
    post_periods = []
    for row in period_max_treatment.iter_rows():
        period_val, max_treat = row[0], row[1]
        if max_treat == 0 or max_treat == 0.0:
            pre_periods.append(period_val)
        else:
            post_periods.append(period_val)
    
    # Sort for deterministic ordering
    pre_periods = sorted(pre_periods, key=lambda x: period_to_idx[x])
    post_periods = sorted(post_periods, key=lambda x: period_to_idx[x])
    
    # Validate sufficient pre-periods
    n_pre = len(pre_periods)
    if n_pre < 2:
        raise ValueError(f"At least 2 pre-treatment periods required; found {n_pre}")
    
    # Validate at least 1 post-period
    n_post = len(post_periods)
    if n_post == 0:
        raise ValueError("No post-treatment periods found")
    
    # ========================================================================
    # Extract Outcome Matrix (Row-Major Order)
    # ========================================================================
    
    # Sort DataFrame by unit, then time for consistent ordering
    df_sorted = df.sort([unit_col, time_col])
    
    # OPTIMIZED: Use vectorized extraction instead of per-unit filter loop
    # Since df is sorted by (unit, time), outcomes are already in row-major order:
    # [unit_0_period_0, unit_0_period_1, ..., unit_N-1_period_T-1]
    # This is O(1) extraction vs O(n_units × n_periods) for the filter loop
    outcomes = df_sorted[outcome_col].cast(_polars.Float64).to_list()
    
    # Create index arrays
    control_indices = [unit_to_idx[u] for u in control_units]
    treated_indices = [unit_to_idx[u] for u in treated_units]
    pre_period_indices = [period_to_idx[p] for p in pre_periods]
    post_period_indices = [period_to_idx[p] for p in post_periods]
    
    # ========================================================================
    # Call Rust Implementation
    # ========================================================================
    
    result = _synthetic_did_impl(
        outcomes=outcomes,
        n_units=n_units,
        n_periods=n_periods,
        control_indices=control_indices,
        treated_indices=treated_indices,
        pre_period_indices=pre_period_indices,
        post_period_indices=post_period_indices,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    
    # ========================================================================
    # Post-Processing Warnings
    # ========================================================================
    
    # Check for unit weight concentration
    if result.unit_weights:
        max_unit_weight = max(result.unit_weights)
        if max_unit_weight > 0.5:
            max_idx = result.unit_weights.index(max_unit_weight)
            _warnings.warn(
                f"Unit weight concentration: control unit at index {max_idx} has "
                f"weight {max_unit_weight:.2%}. Results may be sensitive to this unit.",
                UserWarning,
                stacklevel=2
            )
    
    # Check for time weight concentration
    if result.time_weights:
        max_time_weight = max(result.time_weights)
        if max_time_weight > 0.5:
            max_idx = result.time_weights.index(max_time_weight)
            _warnings.warn(
                f"Time weight concentration: pre-period at index {max_idx} has "
                f"weight {max_time_weight:.2%}. Results may be sensitive to this period.",
                UserWarning,
                stacklevel=2
            )
    
    # Check for low bootstrap iterations
    if bootstrap_iterations < 100:
        _warnings.warn(
            f"bootstrap_iterations={bootstrap_iterations} is less than 100. "
            f"Standard error estimates may be unreliable.",
            UserWarning,
            stacklevel=2
        )
    
    return result


def synthetic_control(
    df: _polars.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    method: str = "traditional",
    lambda_param: _Optional[float] = None,
    compute_se: bool = True,
    n_placebo: _Optional[int] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    seed: _Optional[int] = None,
) -> SyntheticControlResult:
    """
    Compute Synthetic Control (SC) estimator.
    
    Implements the Synthetic Control method from Abadie et al. (2010, 2015),
    which constructs a weighted combination of control units to create a
    synthetic control that matches the treated unit's pre-treatment outcomes.
    
    Supports four method variants:
    - **Traditional**: Classic SC with simplex-constrained weights (Abadie et al., 2010)
    - **Penalized**: L2 regularization for more uniform weights
    - **Robust**: De-meaned data for matching dynamics instead of levels
    - **Augmented**: Bias correction via ridge outcome model (Ben-Michael et al., 2021)
    
    Parameters
    ----------
    df : pl.DataFrame
        Panel data in long format with one row per unit-time observation.
        Must be a balanced panel (all units observed in all time periods).
    unit_col : str
        Column name identifying unique units (e.g., "state", "firm_id").
        Must be integer or string type.
    time_col : str
        Column name identifying time periods (e.g., "year", "quarter").
        Must be integer or string type.
    outcome_col : str
        Column name for the outcome variable. Must be numeric.
    treatment_col : str
        Column name for treatment indicator. Must contain only 0 and 1 values.
        Exactly one unit must be treated (have treatment=1 in post-period).
    method : str, default="traditional"
        Synthetic control method to use. Options:
        - "traditional": Classic SC minimizing pre-treatment MSE
        - "penalized": L2 regularized SC for more uniform weights
        - "robust": De-meaned SC for matching dynamics
        - "augmented": Bias-corrected SC with ridge adjustment
    lambda_param : float, optional
        Regularization parameter for penalized/augmented methods.
        If None, auto-selected via LOOCV for penalized method.
        Must be >= 0 when specified.
    compute_se : bool, default=True
        Whether to compute standard errors via in-space placebo.
    n_placebo : int, optional
        Number of placebo iterations for SE. If None, uses all control units.
    max_iter : int, default=1000
        Maximum iterations for Frank-Wolfe optimizer.
    tol : float, default=1e-6
        Convergence tolerance for optimizer.
    seed : int, optional
        Random seed for reproducibility. If None, uses system time.
    
    Returns
    -------
    SyntheticControlResult
        Result object with the following attributes:
        
        - att : float
            Average Treatment Effect on the Treated
        - standard_error : float or None
            In-space placebo standard error (None if compute_se=False)
        - unit_weights : List[float]
            Weights assigned to each control unit (sums to 1)
        - pre_treatment_rmse : float
            Root Mean Squared Error of pre-treatment fit
        - pre_treatment_mse : float
            Mean Squared Error of pre-treatment fit
        - method : str
            Method used ("traditional", "penalized", "robust", "augmented")
        - lambda_used : float or None
            Lambda parameter used (for penalized/augmented methods)
        - n_units_control : int
            Number of control units
        - n_periods_pre : int
            Number of pre-treatment periods
        - n_periods_post : int
            Number of post-treatment periods
        - solver_converged : bool
            Whether the Frank-Wolfe solver converged
        - solver_iterations : int
            Number of optimizer iterations
        - n_placebo_used : int or None
            Number of successful placebo iterations (if SE computed)
    
    Raises
    ------
    ValueError
        - If DataFrame is empty
        - If any specified column doesn't exist
        - If unit_col or time_col is float type
        - If outcome_col is not numeric
        - If outcome_col contains null values
        - If treatment_col contains values other than 0 and 1
        - If not exactly one treated unit found
        - If fewer than 1 control unit found
        - If fewer than 1 pre-treatment period found
        - If no post-treatment periods found
        - If panel is not balanced
        - If method is not recognized
        - If lambda_param < 0
    
    Warns
    -----
    UserWarning
        - If any unit weight > 0.5 (weight concentration on single unit)
        - If pre_treatment_rmse > 0.1 × outcome std (poor pre-treatment fit)
    
    Examples
    --------
    Basic usage with panel data:
    
    >>> import polars as pl
    >>> import causers
    >>> df = pl.DataFrame({
    ...     'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    ...     'y': [1.0, 2.0, 8.0, 1.5, 2.5, 3.0, 1.2, 2.2, 2.8],
    ...     'treated': [0, 0, 1, 0, 0, 0, 0, 0, 0]
    ... })
    >>> result = causers.synthetic_control(df, 'unit', 'time', 'y', 'treated', seed=42)
    >>> print(f"ATT: {result.att:.4f}")
    ATT: ...
    
    Using penalized method with auto lambda:
    
    >>> result = causers.synthetic_control(
    ...     df, 'unit', 'time', 'y', 'treated',
    ...     method="penalized", seed=42
    ... )
    >>> print(f"Lambda used: {result.lambda_used}")
    Lambda used: ...
    
    Accessing weights and diagnostics:
    
    >>> print(f"Control unit weights: {result.unit_weights}")
    Control unit weights: [...]
    >>> print(f"Pre-treatment RMSE: {result.pre_treatment_rmse:.4f}")
    Pre-treatment RMSE: ...
    
    Without standard errors (faster):
    
    >>> result = causers.synthetic_control(
    ...     df, 'unit', 'time', 'y', 'treated',
    ...     compute_se=False
    ... )
    >>> print(f"ATT: {result.att:.4f} (SE not computed)")
    ATT: ... (SE not computed)
    
    Notes
    -----
    **Key Difference from Synthetic DID**
    
    Synthetic Control requires exactly ONE treated unit, while SDID supports
    multiple treated units. If you have multiple treated units, use
    `synthetic_did()` instead.
    
    **Panel Structure Detection**
    
    The function automatically detects:
    - **Control units**: Units where treatment=0 in all periods
    - **Treated unit**: The single unit where treatment=1 in post-period
    - **Pre-periods**: Periods where treatment=0 for all units
    - **Post-periods**: Periods where the treated unit has treatment=1
    
    **Algorithm**
    
    The SC estimator finds weights ω such that:
    
    .. math::
    
        \\hat{\\omega} = \\arg\\min_{\\omega \\geq 0, \\sum \\omega = 1}
                        \\sum_{t \\in \\text{pre}} (Y_{1t} - \\sum_j \\omega_j Y_{jt})^2
    
    Then the ATT is:
    
    .. math::
    
        \\hat{\\tau}_{SC} = \\frac{1}{|\\text{post}|} \\sum_{t \\in \\text{post}}
                          (Y_{1t} - \\sum_j \\hat{\\omega}_j Y_{jt})
    
    **Standard Errors**
    
    Standard errors are computed via in-space placebo:
    1. For each control unit, treat it as the "placebo treated" unit
    2. Compute SC weights and ATT using remaining controls
    3. SE = standard deviation of placebo ATTs
    
    References
    ----------
    Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control Methods
    for Comparative Case Studies. *Journal of the American Statistical Association*.
    
    Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative Politics and
    the Synthetic Control Method. *American Journal of Political Science*.
    
    Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented Synthetic
    Control Method. *Journal of the American Statistical Association*.
    
    See Also
    --------
    SyntheticControlResult : Result class with ATT and diagnostics.
    synthetic_did : For multiple treated units with DID adjustment.
    """
    # ========================================================================
    # Input Validation
    # ========================================================================
    
    # Check DataFrame is not empty
    if len(df) == 0:
        raise ValueError("Cannot perform SC on empty DataFrame")
    
    # Check all required columns exist
    for col_name, col_label in [
        (unit_col, "unit_col"),
        (time_col, "time_col"),
        (outcome_col, "outcome_col"),
        (treatment_col, "treatment_col"),
    ]:
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
    
    # Check unit_col is not float
    unit_dtype = df[unit_col].dtype
    if unit_dtype in (_polars.Float32, _polars.Float64):
        raise ValueError("unit_col must be integer or string, not float")
    
    # Check time_col is not float
    time_dtype = df[time_col].dtype
    if time_dtype in (_polars.Float32, _polars.Float64):
        raise ValueError("time_col must be integer or string, not float")
    
    # Check outcome_col is numeric
    outcome_dtype = df[outcome_col].dtype
    numeric_types = (
        _polars.Float32, _polars.Float64,
        _polars.Int8, _polars.Int16, _polars.Int32, _polars.Int64,
        _polars.UInt8, _polars.UInt16, _polars.UInt32, _polars.UInt64,
    )
    if outcome_dtype not in numeric_types:
        raise ValueError("outcome_col must be numeric")
    
    # Check for nulls in outcome_col
    null_count = df[outcome_col].null_count()
    if null_count > 0:
        raise ValueError(f"outcome_col '{outcome_col}' contains null values")
    
    # Check treatment_col contains only 0 and 1
    treatment_values = df[treatment_col].unique().to_list()
    valid_treatment_values = {0, 1, 0.0, 1.0}
    for val in treatment_values:
        if val not in valid_treatment_values:
            raise ValueError("treatment_col must contain only 0 and 1 values")
    
    # Validate method
    valid_methods = {"traditional", "penalized", "robust", "augmented"}
    method_lower = method.lower()
    if method_lower not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )
    
    # Validate lambda_param
    if lambda_param is not None and lambda_param < 0:
        raise ValueError(f"lambda_param must be >= 0, got {lambda_param}")
    
    # ========================================================================
    # Panel Structure Detection
    # ========================================================================
    
    # Get unique units and periods
    unique_units = df[unit_col].unique().sort().to_list()
    unique_periods = df[time_col].unique().sort().to_list()
    n_units = len(unique_units)
    n_periods = len(unique_periods)
    
    # Create mappings from unit/period values to indices
    unit_to_idx = {unit: idx for idx, unit in enumerate(unique_units)}
    period_to_idx = {period: idx for idx, period in enumerate(unique_periods)}
    
    # Validate balanced panel
    expected_rows = n_units * n_periods
    if len(df) != expected_rows:
        raise ValueError(
            f"Panel is not balanced: expected {expected_rows} rows "
            f"({n_units} units × {n_periods} periods), found {len(df)}"
        )
    
    # Identify control vs treated units
    # Control units: treatment=0 in ALL periods
    # Treated unit: treatment=1 in at least one period
    unit_max_treatment = (
        df.group_by(unit_col)
        .agg(_polars.col(treatment_col).max().alias("max_treatment"))
    )
    
    control_units = []
    treated_units = []
    for row in unit_max_treatment.iter_rows():
        unit_val, max_treat = row[0], row[1]
        if max_treat == 0 or max_treat == 0.0:
            control_units.append(unit_val)
        else:
            treated_units.append(unit_val)
    
    # Sort for deterministic ordering
    control_units = sorted(control_units, key=lambda x: unit_to_idx[x])
    treated_units = sorted(treated_units, key=lambda x: unit_to_idx[x])
    
    # Validate exactly 1 treated unit (SC requirement)
    n_treated = len(treated_units)
    if n_treated == 0:
        raise ValueError("No treated units found in data")
    if n_treated > 1:
        raise ValueError(
            f"Synthetic Control requires exactly 1 treated unit; found {n_treated}. "
            f"For multiple treated units, use synthetic_did() instead."
        )
    
    # Validate sufficient control units
    n_control = len(control_units)
    if n_control < 1:
        raise ValueError(f"At least 1 control unit required; found {n_control}")
    
    # For augmented method, need at least 2 controls
    if method_lower == "augmented" and n_control < 2:
        raise ValueError(
            f"Augmented SC requires at least 2 control units; found {n_control}"
        )
    
    # Identify pre vs post periods
    # Pre-periods: all observations have treatment=0
    # Post-periods: at least one treated unit has treatment=1
    period_max_treatment = (
        df.group_by(time_col)
        .agg(_polars.col(treatment_col).max().alias("max_treatment"))
    )
    
    pre_periods = []
    post_periods = []
    for row in period_max_treatment.iter_rows():
        period_val, max_treat = row[0], row[1]
        if max_treat == 0 or max_treat == 0.0:
            pre_periods.append(period_val)
        else:
            post_periods.append(period_val)
    
    # Sort for deterministic ordering
    pre_periods = sorted(pre_periods, key=lambda x: period_to_idx[x])
    post_periods = sorted(post_periods, key=lambda x: period_to_idx[x])
    
    # Validate sufficient pre-periods
    n_pre = len(pre_periods)
    if n_pre < 1:
        raise ValueError(f"At least 1 pre-treatment period required; found {n_pre}")
    
    # For augmented method, need at least 2 pre-periods
    if method_lower == "augmented" and n_pre < 2:
        raise ValueError(
            f"Augmented SC requires at least 2 pre-treatment periods; found {n_pre}"
        )
    
    # Validate at least 1 post-period
    n_post = len(post_periods)
    if n_post == 0:
        raise ValueError("No post-treatment periods found")
    
    # ========================================================================
    # Extract Outcome Matrix (Row-Major Order)
    # ========================================================================
    
    # Sort DataFrame by unit, then time for consistent ordering
    df_sorted = df.sort([unit_col, time_col])
    
    # OPTIMIZED: Use vectorized extraction instead of per-unit filter loop
    # Since df is sorted by (unit, time), outcomes are already in row-major order:
    # [unit_0_period_0, unit_0_period_1, ..., unit_N-1_period_T-1]
    # This is O(1) extraction vs O(n_units × n_periods) for the filter loop
    outcomes = df_sorted[outcome_col].cast(_polars.Float64).to_list()
    
    # Create index arrays
    control_indices = [unit_to_idx[u] for u in control_units]
    treated_index = unit_to_idx[treated_units[0]]  # Single treated unit
    pre_period_indices = [period_to_idx[p] for p in pre_periods]
    post_period_indices = [period_to_idx[p] for p in post_periods]
    
    # Determine n_placebo (default: all control units)
    actual_n_placebo = n_placebo if n_placebo is not None else n_control
    
    # ========================================================================
    # Call Rust Implementation
    # ========================================================================
    
    result = _synthetic_control_impl(
        outcomes=outcomes,
        n_units=n_units,
        n_periods=n_periods,
        control_indices=control_indices,
        treated_index=treated_index,
        pre_period_indices=pre_period_indices,
        post_period_indices=post_period_indices,
        method=method_lower,
        lambda_param=lambda_param,
        compute_se=compute_se,
        n_placebo=actual_n_placebo,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
    )
    
    # ========================================================================
    # Post-Processing Warnings
    # ========================================================================
    
    # Check for unit weight concentration
    if result.unit_weights:
        max_unit_weight = max(result.unit_weights)
        if max_unit_weight > 0.5:
            max_idx = result.unit_weights.index(max_unit_weight)
            _warnings.warn(
                f"Unit weight concentration: control unit at index {max_idx} has "
                f"weight {max_unit_weight:.2%}. Results may be sensitive to this unit.",
                UserWarning,
                stacklevel=2
            )
    
    # Check for poor pre-treatment fit
    outcome_std = df[outcome_col].std()
    if outcome_std is not None and outcome_std > 0:
        relative_rmse = result.pre_treatment_rmse / outcome_std
        if relative_rmse > 0.1:
            _warnings.warn(
                f"Pre-treatment RMSE ({result.pre_treatment_rmse:.4f}) is "
                f"{relative_rmse:.1%} of outcome std ({outcome_std:.4f}). "
                f"Consider using a different method or checking data quality.",
                UserWarning,
                stacklevel=2
            )
    
    return result


def about():
    """Print information about the causers package."""
    print(f"causers version {__version__}")
    print("High-performance statistical operations for Polars DataFrames")
    print("Powered by Rust via PyO3/maturin")
    print("")
    print("Features:")
    print("  - Linear regression with HC3 robust standard errors")
    print("  - Logistic regression with Newton-Raphson MLE")
    print("  - Cluster-robust standard errors (analytical and bootstrap)")
    print("  - Wild cluster bootstrap for small cluster counts (linear)")
    print("  - Score bootstrap for small cluster counts (logistic)")
    print("  - Synthetic Difference-in-Differences (SDID)")
    print("  - Synthetic Control (SC) with multiple method variants")
