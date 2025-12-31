"""
Pandas compatibility layer for causers.

This module provides lazy-loaded pandas support, including:
- DataFrame type detection (Polars vs pandas)
- pandas → Polars conversion with Arrow optimization
- Input validation for pandas DataFrames

The module is designed to work without pandas installed; pandas is only
imported when actually needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import pandas as pd


# Public constants
PANDAS_AVAILABLE: bool = False
PYARROW_AVAILABLE: bool = False

# Private module variables
_pd = None  # Lazy pandas module reference
_pa = None  # Lazy pyarrow module reference
_logger = logging.getLogger("causers")


def _ensure_pandas() -> bool:
    """
    Lazy import pandas. Returns True if available.
    
    This function lazily imports pandas only when needed,
    allowing causers to work without pandas installed.
    """
    global PANDAS_AVAILABLE, _pd
    if _pd is not None:
        return True
    try:
        import pandas as pd_module
        _pd = pd_module
        PANDAS_AVAILABLE = True
        return True
    except ImportError:
        return False


def _ensure_pyarrow() -> bool:
    """
    Lazy import pyarrow. Returns True if available.
    
    This function lazily imports pyarrow only when needed.
    """
    global PYARROW_AVAILABLE, _pa
    if _pa is not None:
        return True
    try:
        import pyarrow as pa_module
        _pa = pa_module
        PYARROW_AVAILABLE = True
        return True
    except ImportError:
        return False


def detect_dataframe_type(df: Any) -> Literal["polars", "pandas"]:
    """
    Detect whether df is a Polars or pandas DataFrame.
    
    Time complexity: O(1) - uses isinstance() checks only.
    
    Parameters
    ----------
    df : Any
        The object to check.
    
    Returns
    -------
    Literal["polars", "pandas"]
        The detected DataFrame type.
    
    Raises
    ------
    TypeError
        If df is neither Polars nor pandas DataFrame.
    """
    # Check Polars first (most common case, avoid pandas import)
    if isinstance(df, pl.DataFrame):
        return "polars"
    
    # Lazy check for pandas
    if _ensure_pandas() and isinstance(df, _pd.DataFrame):
        return "pandas"
    
    # Not a recognized DataFrame type
    type_name = type(df).__name__
    if not PANDAS_AVAILABLE:
        raise TypeError(
            f"Unsupported DataFrame type: {type_name}. "
            f"pandas is required to use pandas DataFrames. "
            f"Install with: pip install pandas"
        )
    raise TypeError(f"Unsupported DataFrame type: {type_name}")


def is_arrow_backed(series: pd.Series) -> bool:
    """
    Check if a pandas Series has Arrow-backed storage (pd.ArrowDtype).
    
    Returns False if pyarrow is not available.
    
    Parameters
    ----------
    series : pd.Series
        The pandas Series to check.
    
    Returns
    -------
    bool
        True if the series has ArrowDtype backing.
    """
    if not _ensure_pandas():
        return False
    return isinstance(series.dtype, _pd.ArrowDtype)


def validate_pandas_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Validate a pandas DataFrame for use with causers.
    
    Checks:
    - No MultiIndex columns
    - No MultiIndex rows
    - All required columns exist
    - No object dtype columns
    - No datetime dtype columns
    - No sparse columns
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required_columns : list[str]
        List of column names that must exist.
    
    Raises
    ------
    TypeError
        If pandas is not available.
    ValueError
        If validation fails with specific error message.
    """
    if not _ensure_pandas():
        raise TypeError("pandas is required")
    
    # Check for MultiIndex columns
    if isinstance(df.columns, _pd.MultiIndex):
        raise ValueError(
            "MultiIndex columns not supported; "
            "flatten with df.columns = df.columns.to_flat_index()"
        )
    
    # Check for MultiIndex rows
    if isinstance(df.index, _pd.MultiIndex):
        raise ValueError(
            "MultiIndex rows not supported; flatten with df.reset_index()"
        )
    
    # Check required columns exist
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Validate column types
    for col in required_columns:
        series = df[col]
        dtype = series.dtype
        
        # Check for object dtype
        if dtype == object:
            raise ValueError(
                f"Column '{col}' has object dtype; convert to float64"
            )
        
        # Check for datetime dtype
        if _pd.api.types.is_datetime64_any_dtype(dtype):
            raise ValueError(
                f"Column '{col}' has datetime dtype; only numeric types supported"
            )
        
        # Check for sparse arrays
        if isinstance(dtype, _pd.SparseDtype):
            raise ValueError(
                f"Sparse columns not supported; "
                f"convert to dense with df['{col}'].sparse.to_dense()"
            )


def extract_arrow_column(series: pd.Series) -> pl.Series:
    """
    Extract an Arrow-backed pandas Series to Polars Series.
    
    Uses PyArrow as intermediate for efficient transfer.
    
    Parameters
    ----------
    series : pd.Series
        The Arrow-backed pandas Series to extract.
    
    Returns
    -------
    pl.Series
        The extracted Polars Series.
    """
    col_name = series.name if series.name is not None else ""
    _logger.debug(f"Using Arrow zero-copy path for column '{col_name}'")
    
    # Access underlying PyArrow ChunkedArray
    arrow_array = series.array._pa_array  # type: ignore
    
    # Convert to Polars via PyArrow
    # pl.from_arrow handles ChunkedArray -> Polars Series
    pl_series = pl.from_arrow(arrow_array)
    
    # Ensure correct name
    if isinstance(pl_series, pl.Series):
        return pl_series.alias(str(col_name))
    elif isinstance(pl_series, pl.DataFrame):
        # If from_arrow returned a DataFrame, get the first column
        return pl_series.to_series(0).alias(str(col_name))
    else:
        # Fallback: convert via numpy
        _logger.debug(f"Arrow extraction returned unexpected type, falling back to NumPy")
        return extract_numpy_column(series, str(col_name))


def extract_numpy_column(
    series: pd.Series,
    col_name: str,
    preserve_int: bool = False
) -> pl.Series:
    """
    Extract a NumPy-backed pandas Series to Polars Series.
    
    Handles:
    - Type conversion to float64 (or int64 if preserve_int=True)
    - NA/null → NaN conversion
    - Non-contiguous array copying
    
    Parameters
    ----------
    series : pd.Series
        The pandas Series to extract.
    col_name : str
        The name to give the resulting Polars Series.
    preserve_int : bool, default False
        If True, preserve integer types instead of converting to float64.
        This is useful for columns like unit_col and time_col that must be int.
    
    Returns
    -------
    pl.Series
        The extracted Polars Series.
    """
    _logger.debug(f"Using NumPy fallback for column '{col_name}'")
    
    if not _ensure_pandas():
        raise TypeError("pandas is required")
    
    dtype = series.dtype
    
    # Check for nullable integer types first (before np.issubdtype which doesn't handle them)
    is_nullable_int = (
        hasattr(_pd.core.arrays, 'integer') and
        isinstance(dtype, _pd.core.arrays.integer.IntegerDtype)
    )
    
    # Check if this is an integer dtype that should be preserved
    # np.issubdtype only works with numpy dtypes, not pandas nullable dtypes
    try:
        is_numpy_int = np.issubdtype(dtype, np.integer)
    except TypeError:
        is_numpy_int = False
    
    is_integer_dtype = is_nullable_int or is_numpy_int
    
    if preserve_int and is_integer_dtype:
        _logger.debug(f"Preserving integer type for column '{col_name}'")
        # Handle nullable integer types (Int64, Int32, etc.)
        if hasattr(_pd.core.arrays, 'integer') and isinstance(
            dtype, _pd.core.arrays.integer.IntegerDtype
        ):
            # Check for NA values first
            if series.isna().any():
                _logger.debug(
                    f"Column '{col_name}' has NA values; converting to float64"
                )
                arr = series.to_numpy(dtype='float64', na_value=np.nan)
            else:
                arr = series.to_numpy(dtype='int64')
        else:
            # Standard numpy integer
            arr = series.to_numpy(dtype='int64', copy=False)
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype='int64')
        return pl.Series(name=col_name, values=arr)
    
    # Handle nullable integer types (Int64, Int32, etc.) -> float64
    if hasattr(_pd.core.arrays, 'integer') and isinstance(
        dtype, _pd.core.arrays.integer.IntegerDtype
    ):
        _logger.debug(
            f"Converting column '{col_name}' from {dtype} to float64"
        )
        arr = series.to_numpy(dtype='float64', na_value=np.nan)
    elif hasattr(_pd.core.arrays, 'floating') and isinstance(
        dtype, _pd.core.arrays.floating.FloatingDtype
    ):
        # Nullable float
        _logger.debug(
            f"Converting column '{col_name}' from nullable {dtype} to float64"
        )
        arr = series.to_numpy(dtype='float64', na_value=np.nan)
    else:
        # Standard NumPy-backed dtype
        try:
            arr = series.to_numpy(dtype='float64', na_value=np.nan, copy=False)
        except TypeError:
            # Some older pandas versions don't support na_value
            arr = series.to_numpy(dtype='float64', copy=False)
        
        # Ensure contiguous
        if not arr.flags['C_CONTIGUOUS']:
            _logger.debug(
                f"Column '{col_name}' is non-contiguous; creating contiguous copy"
            )
            arr = np.ascontiguousarray(arr, dtype='float64')
    
    return pl.Series(name=col_name, values=arr)


def convert_pandas_to_polars(
    df: pd.DataFrame,
    columns: list[str],
    int_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Convert a pandas DataFrame to Polars DataFrame.
    
    Uses Arrow path for ArrowDtype columns, NumPy path for others.
    Only extracts the specified columns for efficiency.
    
    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to convert.
    columns : list[str]
        List of column names to extract.
    int_columns : list[str] | None, default None
        List of column names that should preserve integer types.
        Use for columns like unit_col and time_col in panel data functions.
    
    Returns
    -------
    pl.DataFrame
        The converted Polars DataFrame containing only the specified columns.
    """
    series_dict: dict[str, pl.Series] = {}
    int_cols_set = set(int_columns) if int_columns else set()
    
    for col in columns:
        pd_series = df[col]
        preserve_int = col in int_cols_set
        
        if is_arrow_backed(pd_series):
            # Arrow zero-copy path
            try:
                pl_series = extract_arrow_column(pd_series)
                series_dict[col] = pl_series.alias(col)
            except Exception as e:
                # Fallback to NumPy if Arrow extraction fails
                _logger.debug(
                    f"Arrow extraction failed for '{col}': {e}; "
                    f"falling back to NumPy"
                )
                pl_series = extract_numpy_column(pd_series, col, preserve_int=preserve_int)
                series_dict[col] = pl_series
        else:
            # NumPy fallback path
            pl_series = extract_numpy_column(pd_series, col, preserve_int=preserve_int)
            series_dict[col] = pl_series
    
    return pl.DataFrame(series_dict)
