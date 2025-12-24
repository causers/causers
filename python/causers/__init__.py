"""
causers - High-performance statistical operations for Polars DataFrames

A Python package with Rust backend for fast statistical computations
on Polars DataFrames.
"""

from typing import List as _List, Optional as _Optional, Union as _Union
import warnings as _warnings
import polars as _polars

__version__ = "0.3.0"

# Import the Rust extension module
from causers._causers import LinearRegressionResult, linear_regression as _linear_regression_rust

# Re-export main functions
__all__ = [
    "LinearRegressionResult",
    "linear_regression",
]


def linear_regression(
    df: _polars.DataFrame,
    x_cols: _Union[str, _List[str]],
    y_col: str,
    include_intercept: bool = True,
    cluster: _Optional[str] = None,
    bootstrap: bool = False,
    bootstrap_iterations: int = 1000,
    seed: _Optional[int] = None,
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
            Type of clustered SE: "analytical" or "bootstrap" (None if not clustered)
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
    
    return result


def about():
    """Print information about the causers package."""
    print(f"causers version {__version__}")
    print("High-performance statistical operations for Polars DataFrames")
    print("Powered by Rust via PyO3/maturin")
    print("")
    print("Features:")
    print("  - Linear regression with HC3 robust standard errors")
    print("  - Cluster-robust standard errors (analytical and bootstrap)")
    print("  - Wild cluster bootstrap for small cluster counts")
