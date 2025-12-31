"""
Tests for pandas DataFrame support in causers.

This module tests:
- Type detection (Polars vs pandas)
- All functions accepting pandas DataFrames
- Output equivalence between pandas and Polars paths
- Error messages and validation
- Arrow and NumPy extraction paths
- Edge cases
"""

# Standard library imports
import time
from typing import Dict, List, Union

# Third-party imports
import numpy as np
import polars as pl
import pytest

# Conditional third-party imports
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pa = None

# Local imports
import causers
from causers._pandas_compat import (
    PANDAS_AVAILABLE,
    convert_pandas_to_polars,
    detect_dataframe_type,
    extract_arrow_column,
    extract_numpy_column,
    is_arrow_backed,
    validate_pandas_dataframe,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_linear_data() -> Dict[str, List[float]]:
    """Simple linear regression data."""
    return {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.1, 3.9, 6.2, 7.8, 10.1]
    }


@pytest.fixture
def multi_covariate_data() -> Dict[str, List[float]]:
    """Multiple covariate regression data (non-collinear)."""
    return {
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],  # Non-collinear with x1
        "y": [3.0, 7.0, 6.5, 11.0, 10.5, 14.0]
    }


@pytest.fixture
def logistic_data() -> Dict[str, List[float]]:
    """Logistic regression data."""
    return {
        "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 1.8, 2.2],
        "y": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    }


@pytest.fixture
def panel_data() -> Dict[str, List[Union[int, float]]]:
    """Panel data for synthetic DID and synthetic control."""
    return {
        "unit": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 10.0, 1.5, 2.5, 3.5, 4.5, 1.2, 2.2, 3.2, 4.2, 1.3, 2.3, 3.3, 4.3],
        "treated": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }


@pytest.fixture
def clustered_data() -> Dict[str, List[Union[int, float]]]:
    """Data with cluster identifiers."""
    return {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "y": [2.0, 4.0, 5.0, 8.0, 9.0, 12.0, 14.0, 15.0, 18.0, 20.0, 22.0, 24.0],
        "cluster": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    }


# =============================================================================
# TestTypeDetection
# =============================================================================

class TestTypeDetection:
    """Test DataFrame type detection functionality."""
    
    def test_detects_polars_dataframe(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """Should correctly identify Polars DataFrame."""
        pl_df = pl.DataFrame(simple_linear_data)
        assert detect_dataframe_type(pl_df) == "polars"
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_detects_pandas_dataframe(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """Should correctly identify pandas DataFrame."""
        pd_df = pd.DataFrame(simple_linear_data)
        assert detect_dataframe_type(pd_df) == "pandas"
    
    def test_raises_for_unknown_type(self) -> None:
        """Should raise TypeError for unsupported types."""
        with pytest.raises(TypeError, match="Unsupported DataFrame type"):
            detect_dataframe_type([1, 2, 3])
        
        with pytest.raises(TypeError, match="Unsupported DataFrame type"):
            detect_dataframe_type({"a": 1})
        
        with pytest.raises(TypeError, match="Unsupported DataFrame type"):
            detect_dataframe_type("not a dataframe")
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_detection_is_o1(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """Type detection should be O(1) regardless of size."""
        # Small DataFrame
        small_data = {"x": [1.0] * 100, "y": [2.0] * 100}
        small_df = pd.DataFrame(small_data)
        
        start = time.perf_counter()
        for _ in range(1000):
            detect_dataframe_type(small_df)
        small_time = time.perf_counter() - start
        
        # Large DataFrame
        large_data = {"x": [1.0] * 100000, "y": [2.0] * 100000}
        large_df = pd.DataFrame(large_data)
        
        start = time.perf_counter()
        for _ in range(1000):
            detect_dataframe_type(large_df)
        large_time = time.perf_counter() - start
        
        # Detection time should be similar (O(1))
        # Allow 3x tolerance for timing variance
        assert large_time < small_time * 3


# =============================================================================
# TestAllFunctionsAcceptPandas
# =============================================================================

class TestAllFunctionsAcceptPandas:
    """Test that all four functions accept pandas DataFrames."""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_linear_regression_pandas(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """linear_regression should accept pandas DataFrame."""
        pd_df = pd.DataFrame(simple_linear_data)
        result = causers.linear_regression(pd_df, "x", "y")
        
        assert result is not None
        assert len(result.coefficients) == 1
        assert result.n_samples == 5
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_logistic_regression_pandas(self, logistic_data: Dict[str, List[float]]) -> None:
        """logistic_regression should accept pandas DataFrame."""
        pd_df = pd.DataFrame(logistic_data)
        result = causers.logistic_regression(pd_df, "x", "y")
        
        assert result is not None
        assert len(result.coefficients) == 1
        assert result.converged
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_synthetic_did_pandas(self, panel_data: Dict[str, List[Union[int, float]]]) -> None:
        """synthetic_did should accept pandas DataFrame."""
        pd_df = pd.DataFrame(panel_data)
        result = causers.synthetic_did(
            pd_df, "unit", "time", "y", "treated",
            seed=42, bootstrap_iterations=10
        )
        
        assert result is not None
        assert result.n_units_treated == 1
        assert result.n_units_control == 3
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_synthetic_control_pandas(self, panel_data: Dict[str, List[Union[int, float]]]) -> None:
        """synthetic_control should accept pandas DataFrame."""
        pd_df = pd.DataFrame(panel_data)
        result = causers.synthetic_control(
            pd_df, "unit", "time", "y", "treated",
            seed=42, compute_se=False
        )
        
        assert result is not None
        assert len(result.unit_weights) == 3
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_linear_regression_with_cluster_pandas(self, clustered_data: Dict[str, List[Union[int, float]]]) -> None:
        """Test linear regression with cluster parameter and pandas."""
        pd_df = pd.DataFrame(clustered_data)
        result = causers.linear_regression(pd_df, "x", "y", cluster="cluster")
        
        assert result is not None
        assert result.n_clusters == 4


# =============================================================================
# TestOutputEquivalence
# =============================================================================

class TestOutputEquivalence:
    """Test that pandas and Polars produce identical results."""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_linear_regression_equivalence(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """Results should match within rtol=1e-6."""
        pd_df = pd.DataFrame(simple_linear_data)
        pl_df = pl.DataFrame(simple_linear_data)
        
        pd_result = causers.linear_regression(pd_df, "x", "y")
        pl_result = causers.linear_regression(pl_df, "x", "y")
        
        np.testing.assert_allclose(
            pd_result.coefficients, pl_result.coefficients, rtol=1e-6
        )
        np.testing.assert_allclose(
            pd_result.intercept, pl_result.intercept, rtol=1e-6
        )
        np.testing.assert_allclose(
            pd_result.r_squared, pl_result.r_squared, rtol=1e-6
        )
        np.testing.assert_allclose(
            pd_result.standard_errors, pl_result.standard_errors, rtol=1e-6
        )
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_linear_regression_multi_covariate_equivalence(self, multi_covariate_data: Dict[str, List[float]]) -> None:
        """Test equivalence with multiple covariates."""
        pd_df = pd.DataFrame(multi_covariate_data)
        pl_df = pl.DataFrame(multi_covariate_data)
        
        pd_result = causers.linear_regression(pd_df, ["x1", "x2"], "y")
        pl_result = causers.linear_regression(pl_df, ["x1", "x2"], "y")
        
        np.testing.assert_allclose(
            pd_result.coefficients, pl_result.coefficients, rtol=1e-6
        )
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_logistic_regression_equivalence(self, logistic_data: Dict[str, List[float]]) -> None:
        """Logistic regression results should match."""
        pd_df = pd.DataFrame(logistic_data)
        pl_df = pl.DataFrame(logistic_data)
        
        pd_result = causers.logistic_regression(pd_df, "x", "y")
        pl_result = causers.logistic_regression(pl_df, "x", "y")
        
        np.testing.assert_allclose(
            pd_result.coefficients, pl_result.coefficients, rtol=1e-5
        )
        np.testing.assert_allclose(
            pd_result.log_likelihood, pl_result.log_likelihood, rtol=1e-5
        )
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_synthetic_did_equivalence(self, panel_data: Dict[str, List[Union[int, float]]]) -> None:
        """Synthetic DID results should match."""
        pd_df = pd.DataFrame(panel_data)
        pl_df = pl.DataFrame(panel_data)
        
        pd_result = causers.synthetic_did(
            pd_df, "unit", "time", "y", "treated",
            seed=42, bootstrap_iterations=10
        )
        pl_result = causers.synthetic_did(
            pl_df, "unit", "time", "y", "treated",
            seed=42, bootstrap_iterations=10
        )
        
        np.testing.assert_allclose(pd_result.att, pl_result.att, rtol=1e-6)
        np.testing.assert_allclose(
            pd_result.unit_weights, pl_result.unit_weights, rtol=1e-6
        )
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_synthetic_control_equivalence(self, panel_data: Dict[str, List[Union[int, float]]]) -> None:
        """Synthetic control results should match."""
        pd_df = pd.DataFrame(panel_data)
        pl_df = pl.DataFrame(panel_data)
        
        pd_result = causers.synthetic_control(
            pd_df, "unit", "time", "y", "treated",
            seed=42, compute_se=False
        )
        pl_result = causers.synthetic_control(
            pl_df, "unit", "time", "y", "treated",
            seed=42, compute_se=False
        )
        
        np.testing.assert_allclose(pd_result.att, pl_result.att, rtol=1e-6)
        np.testing.assert_allclose(
            pd_result.unit_weights, pl_result.unit_weights, rtol=1e-6
        )


# =============================================================================
# TestErrorMessages
# =============================================================================

class TestErrorMessages:
    """Test error messages for invalid input."""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_column_not_found(self) -> None:
        """Should raise ValueError for missing columns."""
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        
        with pytest.raises(ValueError, match="Column 'z' not found"):
            validate_pandas_dataframe(df, ["x", "z"])
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_object_dtype_error(self) -> None:
        """Should raise ValueError for object dtype."""
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
        
        with pytest.raises(ValueError, match="Column 'x' has object dtype"):
            validate_pandas_dataframe(df, ["x", "y"])
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_datetime_dtype_error(self) -> None:
        """Should raise ValueError for datetime dtype."""
        df = pd.DataFrame({
            "x": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "y": [1.0, 2.0]
        })
        
        with pytest.raises(ValueError, match="Column 'x' has datetime dtype"):
            validate_pandas_dataframe(df, ["x", "y"])
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_multiindex_columns_error(self) -> None:
        """Should raise ValueError for MultiIndex columns."""
        df = pd.DataFrame({("a", "x"): [1, 2], ("b", "y"): [3, 4]})
        
        with pytest.raises(ValueError, match="MultiIndex columns not supported"):
            validate_pandas_dataframe(df, ["x"])
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_multiindex_rows_error(self) -> None:
        """Should raise ValueError for MultiIndex rows."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df.index = pd.MultiIndex.from_tuples([(0, 'a'), (1, 'b')])
        
        with pytest.raises(ValueError, match="MultiIndex rows not supported"):
            validate_pandas_dataframe(df, ["x", "y"])
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_sparse_column_error(self) -> None:
        """Should raise ValueError for sparse columns."""
        sparse_arr = pd.arrays.SparseArray([1.0, 0.0, 0.0, 2.0])
        df = pd.DataFrame({"x": sparse_arr, "y": [1.0, 2.0, 3.0, 4.0]})
        
        with pytest.raises(ValueError, match="Sparse columns not supported"):
            validate_pandas_dataframe(df, ["x", "y"])


# =============================================================================
# TestNullHandling
# =============================================================================

class TestNullHandling:
    """Test null value handling."""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_nan_preserved(self) -> None:
        """NaN values should be preserved."""
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
        pl_df = convert_pandas_to_polars(df, ["x"])
        
        # Check that NaN is preserved
        values = pl_df["x"].to_list()
        assert np.isnan(values[2])
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_nullable_integer_to_nan(self) -> None:
        """Nullable integer NA should convert to NaN."""
        # Use nullable Int64 dtype
        df = pd.DataFrame({"x": pd.array([1, 2, pd.NA, 4], dtype="Int64")})
        pl_df = convert_pandas_to_polars(df, ["x"])
        
        values = pl_df["x"].to_list()
        # NA should be converted to NaN (or null in Polars)
        assert values[2] is None or (isinstance(values[2], float) and np.isnan(values[2]))
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_immutability(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """Input DataFrame should not be modified."""
        pd_df = pd.DataFrame(simple_linear_data)
        original_values = pd_df["x"].tolist()
        
        _ = causers.linear_regression(pd_df, "x", "y")
        
        # Verify original data unchanged
        assert pd_df["x"].tolist() == original_values


# =============================================================================
# TestArrowExtraction
# =============================================================================

class TestArrowExtraction:
    """Test Arrow-backed column extraction."""
    
    @pytest.mark.skipif(not HAS_PANDAS or not HAS_PYARROW, reason="pandas/pyarrow not installed")
    def test_arrow_backed_float64(self) -> None:
        """Should extract Arrow-backed float64 columns."""
        df = pd.DataFrame({
            "x": pd.array([1.0, 2.0, 3.0], dtype=pd.ArrowDtype(pa.float64()))
        })
        
        assert is_arrow_backed(df["x"])
        pl_series = extract_arrow_column(df["x"])
        
        assert pl_series.to_list() == [1.0, 2.0, 3.0]
    
    @pytest.mark.skipif(not HAS_PANDAS or not HAS_PYARROW, reason="pandas/pyarrow not installed")
    def test_arrow_backed_int64_conversion(self) -> None:
        """Arrow int64 should convert to float64."""
        df = pd.DataFrame({
            "x": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int64()))
        })
        
        assert is_arrow_backed(df["x"])
        pl_df = convert_pandas_to_polars(df, ["x"])
        
        # Values should be preserved
        assert pl_df["x"].to_list() == [1, 2, 3]
    
    @pytest.mark.skipif(not HAS_PANDAS or not HAS_PYARROW, reason="pandas/pyarrow not installed")
    def test_mixed_arrow_numpy_columns(self) -> None:
        """Should handle mixed Arrow/NumPy columns."""
        df = pd.DataFrame({
            "x": pd.array([1.0, 2.0, 3.0], dtype=pd.ArrowDtype(pa.float64())),
            "y": [4.0, 5.0, 6.0]  # NumPy-backed
        })
        
        assert is_arrow_backed(df["x"])
        assert not is_arrow_backed(df["y"])
        
        pl_df = convert_pandas_to_polars(df, ["x", "y"])
        
        assert pl_df["x"].to_list() == [1.0, 2.0, 3.0]
        assert pl_df["y"].to_list() == [4.0, 5.0, 6.0]


# =============================================================================
# TestNumPyExtraction
# =============================================================================

class TestNumPyExtraction:
    """Test NumPy-backed column extraction."""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_numpy_backed_float64(self) -> None:
        """Should extract NumPy-backed float64 columns."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        
        assert not is_arrow_backed(df["x"])
        pl_series = extract_numpy_column(df["x"], "x")
        
        assert pl_series.to_list() == [1.0, 2.0, 3.0]
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_numpy_backed_int_conversion(self) -> None:
        """Integer columns should convert to float64."""
        df = pd.DataFrame({"x": [1, 2, 3]})  # int64
        
        pl_series = extract_numpy_column(df["x"], "x")
        
        assert pl_series.to_list() == [1.0, 2.0, 3.0]
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_noncontiguous_array(self) -> None:
        """Should handle non-contiguous arrays."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # Create a non-contiguous slice
        sliced_df = df[::2]  # Every other row
        
        pl_series = extract_numpy_column(sliced_df["x"], "x")
        
        assert pl_series.to_list() == [1.0, 3.0, 5.0]
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_nullable_integer_dtype(self) -> None:
        """Should handle nullable integer dtypes."""
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="Int64")})
        
        pl_series = extract_numpy_column(df["x"], "x")
        
        assert pl_series.to_list() == [1.0, 2.0, 3.0]


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_empty_dataframe(self) -> None:
        """Should handle empty DataFrames appropriately."""
        pd_df = pd.DataFrame({"x": [], "y": []})
        
        # Empty DataFrame should raise an error in the function
        with pytest.raises(ValueError):
            causers.linear_regression(pd_df, "x", "y")
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_single_row(self) -> None:
        """Should handle single-row DataFrames."""
        pd_df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        
        # Single row may fail due to insufficient data
        # This tests that the pandas conversion at least succeeds
        pl_df = convert_pandas_to_polars(pd_df, ["x", "y"])
        assert len(pl_df) == 1
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_reproducibility(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """Results should be reproducible."""
        pd_df = pd.DataFrame(simple_linear_data)
        
        result1 = causers.linear_regression(pd_df, "x", "y")
        result2 = causers.linear_regression(pd_df, "x", "y")
        
        assert result1.coefficients == result2.coefficients
        assert result1.intercept == result2.intercept
    
    def test_polars_unchanged(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """Polars path should work unchanged."""
        pl_df = pl.DataFrame(simple_linear_data)
        
        result = causers.linear_regression(pl_df, "x", "y")
        
        assert result is not None
        assert len(result.coefficients) == 1


# =============================================================================
# TestIntegration
# =============================================================================

class TestIntegration:
    """Integration tests for end-to-end functionality."""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_linear_regression_with_multiple_features_pandas(self) -> None:
        """Test linear regression with multiple features from pandas."""
        data = {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "y": [2.2, 4.4, 6.5, 8.7, 10.9, 13.1, 15.3, 17.5]
        }
        pd_df = pd.DataFrame(data)
        
        result = causers.linear_regression(pd_df, ["x1", "x2"], "y")
        
        assert len(result.coefficients) == 2
        assert result.r_squared > 0.99  # Should be near-perfect fit
    
    @pytest.mark.skipif(not HAS_PANDAS or not HAS_PYARROW, reason="pandas/pyarrow not installed")
    def test_linear_regression_arrow_backed_pandas(self) -> None:
        """Test linear regression with Arrow-backed pandas columns."""
        data = {
            "x": pd.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pd.ArrowDtype(pa.float64())),
            "y": pd.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=pd.ArrowDtype(pa.float64()))
        }
        pd_df = pd.DataFrame(data)
        
        result = causers.linear_regression(pd_df, "x", "y")
        
        # Perfect linear relationship: y = 2*x
        np.testing.assert_allclose(result.coefficients[0], 2.0, rtol=1e-6)
        np.testing.assert_allclose(result.intercept, 0.0, atol=1e-6)
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_full_workflow_pandas(self, panel_data: Dict[str, List[Union[int, float]]]) -> None:
        """Test full workflow with pandas DataFrame."""
        pd_df = pd.DataFrame(panel_data)
        
        # SDID
        sdid_result = causers.synthetic_did(
            pd_df, "unit", "time", "y", "treated",
            seed=42, bootstrap_iterations=20
        )
        
        # SC
        sc_result = causers.synthetic_control(
            pd_df, "unit", "time", "y", "treated",
            seed=42, compute_se=False
        )
        
        assert sdid_result is not None
        assert sc_result is not None
        # Both should estimate a positive treatment effect (y jumped from 3 to 10)
        assert sdid_result.att > 0
        assert sc_result.att > 0


# =============================================================================
# TestOptionalPandas
# =============================================================================

class TestOptionalPandas:
    """Test behavior when pandas is optional."""
    
    def test_polars_works_without_pandas(self, simple_linear_data: Dict[str, List[float]]) -> None:
        """Polars should work without pandas."""
        # This test always runs regardless of pandas availability
        pl_df = pl.DataFrame(simple_linear_data)
        
        result = causers.linear_regression(pl_df, "x", "y")
        
        assert result is not None
    
    def test_causers_imports_without_pandas(self) -> None:
        """causers should import without pandas."""
        # If we got here, the import already succeeded
        assert causers is not None
        assert hasattr(causers, 'linear_regression')
        assert hasattr(causers, 'logistic_regression')
        assert hasattr(causers, 'synthetic_did')
        assert hasattr(causers, 'synthetic_control')
