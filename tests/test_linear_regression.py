"""Tests for linear regression functionality."""

import pytest
import polars as pl
import numpy as np
from causers import linear_regression, LinearRegressionResult


class TestLinearRegression:
    """Test suite for linear regression."""
    
    def test_perfect_linear_relationship(self):
        """Test regression on perfectly linear data."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        
        result = linear_regression(df, "x", "y")
        
        assert isinstance(result, LinearRegressionResult)
        assert len(result.coefficients) == 1
        assert abs(result.coefficients[0] - 2.0) < 1e-10
        assert result.slope is not None
        assert abs(result.slope - 2.0) < 1e-10
        assert result.intercept is not None
        assert abs(result.intercept - 0.0) < 1e-10
        assert abs(result.r_squared - 1.0) < 1e-10
        assert result.n_samples == 5
    
    def test_linear_with_intercept(self):
        """Test regression with non-zero intercept."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 2x + 1
        })
        
        result = linear_regression(df, "x", "y")
        
        assert len(result.coefficients) == 1
        assert abs(result.coefficients[0] - 2.0) < 1e-10
        assert abs(result.intercept - 1.0) < 1e-10
        assert abs(result.r_squared - 1.0) < 1e-10
    
    def test_noisy_linear_data(self):
        """Test regression on noisy linear data."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 * x + 3 + np.random.normal(0, 0.5, n)
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        result = linear_regression(df, "x", "y")
        
        # Should be close to y = 2x + 3
        assert len(result.coefficients) == 1
        assert abs(result.coefficients[0] - 2.0) < 0.1
        assert abs(result.intercept - 3.0) < 0.2
        assert result.r_squared > 0.98
        assert result.n_samples == n
    
    def test_result_repr_and_str(self):
        """Test string representations of result."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0]
        })
        
        result = linear_regression(df, "x", "y")
        
        repr_str = repr(result)
        assert "LinearRegressionResult" in repr_str
        assert "coefficients=" in repr_str
        assert "intercept=" in repr_str
        assert "r_squared=" in repr_str
        
        str_repr = str(result)
        assert "y =" in str_repr
        assert "R²" in str_repr
    
    def test_column_not_found(self):
        """Test error when column doesn't exist."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0]
        })
        
        with pytest.raises(Exception):  # Polars will raise an error
            linear_regression(df, "nonexistent", "y")
    
    def test_empty_dataframe(self):
        """Test error handling for empty DataFrame."""
        df = pl.DataFrame({
            "x": [],
            "y": []
        })
        
        with pytest.raises(ValueError):
            linear_regression(df, "x", "y")


    def test_multiple_covariates(self):
        """Test regression with multiple independent variables."""
        # y = 2*x1 + 3*x2 + 1
        df = pl.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 1.0, 2.0, 2.0, 3.0],
            "y": [6.0, 8.0, 13.0, 15.0, 20.0]
        })
        
        result = linear_regression(df, ["x1", "x2"], "y")
        
        assert isinstance(result, LinearRegressionResult)
        assert len(result.coefficients) == 2
        assert abs(result.coefficients[0] - 2.0) < 1e-10
        assert abs(result.coefficients[1] - 3.0) < 1e-10
        assert result.intercept is not None
        assert abs(result.intercept - 1.0) < 1e-10
        assert abs(result.r_squared - 1.0) < 1e-10
        assert result.n_samples == 5
        # slope should be None for multiple covariates
        assert result.slope is None
    
    def test_regression_without_intercept(self):
        """Test regression without intercept term."""
        # y = 2*x (no intercept)
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        
        result = linear_regression(df, "x", "y", include_intercept=False)
        
        assert len(result.coefficients) == 1
        assert abs(result.coefficients[0] - 2.0) < 1e-10
        assert result.intercept is None
        assert abs(result.r_squared - 1.0) < 1e-10
    
    def test_multiple_covariates_without_intercept(self):
        """Test multiple regression without intercept."""
        # y = 2*x1 + 3*x2 (no intercept)
        df = pl.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 1.0, 2.0, 2.0, 3.0],
            "y": [5.0, 7.0, 12.0, 14.0, 19.0]
        })
        
        result = linear_regression(df, ["x1", "x2"], "y", include_intercept=False)
        
        assert len(result.coefficients) == 2
        assert abs(result.coefficients[0] - 2.0) < 1e-10
        assert abs(result.coefficients[1] - 3.0) < 1e-10
        assert result.intercept is None
        assert abs(result.r_squared - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])