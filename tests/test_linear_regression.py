"""Tests for linear regression functionality."""

import pytest
import polars as pl
import numpy as np
from numpy.testing import assert_allclose
from causers import linear_regression, LinearRegressionResult

# Optional statsmodels import for comparison tests
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    sm = None

requires_statsmodels = pytest.mark.skipif(
    not HAS_STATSMODELS,
    reason="statsmodels not installed"
)


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


class TestHC3StandardErrors:
    """Test suite for HC3 standard error computation."""
    
    def test_standard_errors_field_exists(self):
        """Verify standard_errors field is present."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1]
        })
        result = linear_regression(df, "x", "y")
        assert hasattr(result, 'standard_errors')
        assert isinstance(result.standard_errors, list)
    
    def test_intercept_se_field_exists(self):
        """Verify intercept_se field is present."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1]
        })
        result = linear_regression(df, "x", "y")
        assert hasattr(result, 'intercept_se')
    
    def test_standard_errors_length_matches_coefficients(self):
        """Verify len(standard_errors) == len(coefficients)."""
        df = pl.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 1.0, 2.0, 2.0, 3.0],
            "y": [6.1, 8.2, 12.9, 15.1, 19.8]
        })
        result = linear_regression(df, ["x1", "x2"], "y")
        assert len(result.standard_errors) == len(result.coefficients)
    
    def test_intercept_se_present_when_intercept_included(self):
        """Verify intercept_se is not None when include_intercept=True."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1]
        })
        result = linear_regression(df, "x", "y", include_intercept=True)
        assert result.intercept_se is not None
        assert isinstance(result.intercept_se, float)
    
    def test_intercept_se_none_when_intercept_excluded(self):
        """Verify intercept_se is None when include_intercept=False."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        result = linear_regression(df, "x", "y", include_intercept=False)
        assert result.intercept_se is None
    
    def test_standard_errors_non_negative(self):
        """Verify all standard errors >= 0."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 * x + 1 + np.random.randn(n) * 0.5
        
        df = pl.DataFrame({"x": x, "y": y})
        result = linear_regression(df, "x", "y")
        
        assert all(se >= 0 for se in result.standard_errors)
        assert result.intercept_se >= 0
    
    def test_perfect_fit_near_zero_standard_errors(self):
        """Verify SE is essentially zero when R² = 1."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect linear
        })
        result = linear_regression(df, "x", "y")
        assert result.r_squared == pytest.approx(1.0)
        # SE should be essentially zero (within numerical precision)
        assert all(se < 1e-10 for se in result.standard_errors)
        assert result.intercept_se < 1e-10
    
    @requires_statsmodels
    def test_hc3_matches_statsmodels_single_covariate(self):
        """Compare HC3 SE with statsmodels for single covariate."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 * x + 1 + np.random.randn(n) * 0.5
        
        # statsmodels reference
        X_sm = sm.add_constant(x)
        model = sm.OLS(y, X_sm).fit()
        hc3 = model.get_robustcov_results(cov_type='HC3')
        expected_coef_se = hc3.bse[1]  # Coefficient SE (not intercept)
        expected_intercept_se = hc3.bse[0]  # Intercept SE
        
        # causers
        df = pl.DataFrame({"x": x, "y": y})
        result = linear_regression(df, "x", "y")
        
        # Compare with 1e-6 relative tolerance
        assert_allclose(result.standard_errors[0], expected_coef_se, rtol=1e-6)
        assert_allclose(result.intercept_se, expected_intercept_se, rtol=1e-6)
    
    @requires_statsmodels
    def test_hc3_matches_statsmodels_multiple_covariates(self):
        """Compare HC3 SE with statsmodels for multiple covariates."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + 1 + np.random.randn(n) * 0.5
        
        # statsmodels reference
        X = np.column_stack([x1, x2])
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        hc3 = model.get_robustcov_results(cov_type='HC3')
        expected_coef_se = hc3.bse[1:]  # Coefficient SEs (not intercept)
        expected_intercept_se = hc3.bse[0]
        
        # causers
        df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
        result = linear_regression(df, ["x1", "x2"], "y")
        
        # Compare with 1e-6 relative tolerance
        assert_allclose(result.standard_errors, expected_coef_se, rtol=1e-6)
        assert_allclose(result.intercept_se, expected_intercept_se, rtol=1e-6)
    
    @requires_statsmodels
    def test_hc3_matches_statsmodels_no_intercept(self):
        """Compare HC3 SE with statsmodels without intercept."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.5  # No intercept in true model
        
        # statsmodels reference (no constant)
        X_sm = x.reshape(-1, 1)
        model = sm.OLS(y, X_sm).fit()
        hc3 = model.get_robustcov_results(cov_type='HC3')
        expected_coef_se = hc3.bse[0]
        
        # causers
        df = pl.DataFrame({"x": x, "y": y})
        result = linear_regression(df, "x", "y", include_intercept=False)
        
        # Compare with 1e-6 relative tolerance
        assert_allclose(result.standard_errors[0], expected_coef_se, rtol=1e-6)
        assert result.intercept_se is None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_computation_with_moderate_leverage(self):
        """Verify computation succeeds when leverage is moderate but not extreme."""
        # Spread out data to keep leverage values moderate
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1 + np.random.randn(20) * 0.5
        
        df = pl.DataFrame({"x": x, "y": y})
        result = linear_regression(df, "x", "y")  # Should not raise
        assert len(result.standard_errors) == 1
        assert result.standard_errors[0] >= 0
    
    def test_extreme_leverage_raises_error(self):
        """Verify extreme leverage raises ValueError with clear message."""
        # This data has a single extreme point
        df = pl.DataFrame({
            "x": [0.0, 0.0, 0.0, 0.0, 10.0],
            "y": [1.0, 1.1, 0.9, 1.0, 20.0]
        })
        with pytest.raises(ValueError, match="leverage"):
            linear_regression(df, "x", "y")
    
    def test_backward_compatibility_fields_unchanged(self):
        """Verify existing fields produce same values as before."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [3.0, 5.0, 7.0, 9.0, 11.0]
        })
        result = linear_regression(df, "x", "y")
        
        # These values should be unchanged from v0.1.0
        assert result.coefficients == pytest.approx([2.0], rel=1e-10)
        assert result.intercept == pytest.approx(1.0, rel=1e-10)
        assert result.r_squared == pytest.approx(1.0, rel=1e-10)
        assert result.n_samples == 5
        assert result.slope == pytest.approx(2.0, rel=1e-10)
    
    def test_repr_includes_standard_errors(self):
        """Verify __repr__ includes new standard_errors field."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1]
        })
        result = linear_regression(df, "x", "y")
        repr_str = repr(result)
        assert "standard_errors=" in repr_str
        assert "intercept_se=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])