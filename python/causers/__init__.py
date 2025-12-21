"""
causers - High-performance statistical operations for Polars DataFrames

A Python package with Rust backend for fast statistical computations
on Polars DataFrames.
"""

from typing import List, Union
import polars as pl

__version__ = "0.1.0"

# Import the Rust extension module
from causers._causers import LinearRegressionResult, linear_regression as _linear_regression_rust

# Re-export main functions
__all__ = [
    "LinearRegressionResult",
    "linear_regression",
]


def linear_regression(
    df: pl.DataFrame,
    x_cols: Union[str, List[str]],
    y_col: str,
    include_intercept: bool = True
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
    
    Raises
    ------
    ValueError
        If x_cols is empty, columns don't exist, or data is invalid
    
    Examples
    --------
    Single covariate regression:
    
    >>> import polars as pl
    >>> import causers
    >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    >>> result = causers.linear_regression(df, "x", "y")
    >>> print(f"y = {result.slope:.2f}x + {result.intercept:.2f}")
    y = 2.00x + 0.00
    
    Multiple covariate regression:
    
    >>> df = pl.DataFrame({
    ...     "x1": [1, 2, 3, 4, 5],
    ...     "x2": [1, 1, 2, 2, 3],
    ...     "y": [6, 8, 13, 15, 20]
    ... })
    >>> result = causers.linear_regression(df, ["x1", "x2"], "y")
    >>> print(f"Coefficients: {result.coefficients}")
    Coefficients: [2.0, 3.0]
    
    Regression without intercept:
    
    >>> result = causers.linear_regression(df, "x", "y", include_intercept=False)
    >>> print(f"Slope: {result.coefficients[0]:.2f}, Intercept: {result.intercept}")
    Slope: 2.00, Intercept: None
    
    Notes
    -----
    - For multiple covariates, ensure they are not perfectly collinear
    - NaN and infinite values may cause errors or unexpected results
    - Requires at least as many samples as parameters (including intercept)
    """
    # Normalize x_cols to always be a list
    if isinstance(x_cols, str):
        x_cols_list = [x_cols]
    else:
        x_cols_list = list(x_cols)
    
    # Call the Rust implementation
    return _linear_regression_rust(df, x_cols_list, y_col, include_intercept)


def about():
    """Print information about the causers package."""
    print(f"causers version {__version__}")
    print("High-performance statistical operations for Polars DataFrames")
    print("Powered by Rust via PyO3/maturin")