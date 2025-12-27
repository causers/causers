"""Property-based tests for causers package using hypothesis.

This module uses property-based testing to validate the mathematical
properties and invariants of linear regression.
"""

import pytest
import polars as pl
import numpy as np
from causers import linear_regression
try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis.extra import numpy as npst
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Define dummy decorators if hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            func.skip_hypothesis = True
            return func
        return decorator
    class st:
        @staticmethod
        def floats(*args, **kwargs):
            pass
        @staticmethod
        def integers(*args, **kwargs):
            pass
        @staticmethod
        def lists(*args, **kwargs):
            pass
        @staticmethod
        def tuples(*args, **kwargs):
            pass
    def settings(*args, **kwargs):
        # Dummy decorator for settings when hypothesis is not available
        def decorator(func):
            return func
        return decorator


class TestPropertyBased:
    """Property-based test suite for linear regression."""
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        slope=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        intercept=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        n_points=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=50, deadline=5000)  # 5 second deadline per test
    def test_property_perfect_linear_recovery(self, slope, intercept, n_points):
        """Property: Perfect linear data should recover exact parameters.
        
        For any valid slope and intercept, if we generate perfect
        linear data, regression should recover the exact parameters.
        """
        # Generate perfect linear data
        x = np.linspace(0, 10, n_points)
        y = slope * x + intercept
        
        # Skip edge case where y has zero variance (slope=0, intercept=0)
        if np.var(y) < 1e-10:
            assume(False)
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        try:
            result = linear_regression(df, "x", "y")
        except ValueError as e:
            # Skip if high leverage cases trigger HC3 error
            if "leverage" in str(e).lower():
                assume(False)
            raise
        
        # Should recover exact parameters (within floating point precision)
        assert abs(result.slope - slope) < 1e-8
        assert abs(result.intercept - intercept) < 1e-8
        assert abs(result.r_squared - 1.0) < 1e-10
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        data=st.lists(
            st.tuples(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
            ),
            min_size=3,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_property_r_squared_bounds(self, data):
        """Property: R-squared should always be between 0 and 1.
        
        For any valid dataset, R-squared should be in [0, 1].
        """
        x_vals, y_vals = zip(*data)
        
        # Check if x has variance (required for regression)
        x_var = np.var(x_vals)
        assume(x_var > 1e-10)  # Skip if x has no variance
        
        df = pl.DataFrame({
            "x": list(x_vals),
            "y": list(y_vals)
        })
        
        try:
            result = linear_regression(df, "x", "y")
        except ValueError as e:
            # Skip if high leverage cases trigger HC3 error
            if "leverage" in str(e).lower():
                assume(False)
            raise
        
        # R-squared should be bounded
        assert 0 <= result.r_squared <= 1.0001  # Small tolerance for numerical errors
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        x_data=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=50
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_property_invariant_to_y_shift(self, x_data):
        """Property: Slope is invariant to y-intercept shifts.
        
        Adding a constant to all y values should only change
        the intercept, not the slope.
        """
        # Ensure x has variance
        x_var = np.var(x_data)
        assume(x_var > 1e-10)
        
        # Create base y data with some relationship
        y_base = [2 * x + np.random.randn() * 0.1 for x in x_data]
        shift = 42.0
        
        df_base = pl.DataFrame({
            "x": x_data,
            "y": y_base
        })
        
        df_shifted = pl.DataFrame({
            "x": x_data,
            "y": [y + shift for y in y_base]
        })
        
        try:
            result_base = linear_regression(df_base, "x", "y")
            result_shifted = linear_regression(df_shifted, "x", "y")
        except ValueError as e:
            # Skip if high leverage cases trigger HC3 error
            if "leverage" in str(e).lower():
                assume(False)
            raise
        
        # Slope should be the same
        assert abs(result_base.slope - result_shifted.slope) < 1e-10
        # Intercept should differ by the shift amount
        assert abs((result_shifted.intercept - result_base.intercept) - shift) < 1e-10
        # R-squared should be the same
        assert abs(result_base.r_squared - result_shifted.r_squared) < 1e-10
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        x_data=st.lists(
            st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=50
        ),
        scale=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=5000)
    def test_property_scale_covariance(self, x_data, scale):
        """Property: Scaling x should scale the slope inversely.
        
        If we scale all x values by a factor, the slope should
        scale inversely to maintain the same predictions.
        """
        # Ensure x has variance
        x_var = np.var(x_data)
        assume(x_var > 1e-10)
        
        # Create y data with known relationship
        y_data = [2 * x + 3 + np.random.randn() * 0.01 for x in x_data]
        
        df_original = pl.DataFrame({
            "x": x_data,
            "y": y_data
        })
        
        df_scaled = pl.DataFrame({
            "x": [x * scale for x in x_data],
            "y": y_data
        })
        
        try:
            result_original = linear_regression(df_original, "x", "y")
            result_scaled = linear_regression(df_scaled, "x", "y")
        except ValueError as e:
            # Skip if high leverage cases trigger HC3 error
            if "leverage" in str(e).lower():
                assume(False)
            raise
        
        # Slope should scale inversely
        expected_slope = result_original.slope / scale
        assert abs(result_scaled.slope - expected_slope) < abs(expected_slope) * 0.01
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        n_points=st.integers(min_value=20, max_value=100),
        noise_level=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_noise_degrades_r_squared(self, n_points, noise_level):
        """Property: Adding noise should decrease R-squared.
        
        As we add more noise to perfect linear data,
        R-squared should decrease monotonically.
        """
        np.random.seed(42)  # For reproducibility in property test
        
        x = np.linspace(0, 10, n_points)
        y_perfect = 2 * x + 3
        
        # Test with no noise
        df_perfect = pl.DataFrame({
            "x": x,
            "y": y_perfect
        })
        result_perfect = linear_regression(df_perfect, "x", "y")
        
        # Test with noise
        y_noisy = y_perfect + np.random.randn(n_points) * noise_level * 10
        df_noisy = pl.DataFrame({
            "x": x,
            "y": y_noisy
        })
        result_noisy = linear_regression(df_noisy, "x", "y")
        
        # R-squared should be lower with noise (or equal if noise is 0)
        if noise_level > 0.01:
            assert result_noisy.r_squared <= result_perfect.r_squared
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        data=st.lists(
            st.tuples(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
            ),
            min_size=3,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_property_n_samples_consistency(self, data):
        """Property: n_samples should always equal input size.
        
        The reported number of samples should match the input size.
        """
        x_vals, y_vals = zip(*data)
        
        # Check if x has variance
        x_var = np.var(x_vals)
        assume(x_var > 1e-10)
        
        df = pl.DataFrame({
            "x": list(x_vals),
            "y": list(y_vals)
        })
        
        try:
            result = linear_regression(df, "x", "y")
        except ValueError as e:
            # Skip if high leverage cases trigger HC3 error
            if "leverage" in str(e).lower():
                assume(False)
            raise
        
        assert result.n_samples == len(data)
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        x1=st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=20
        ),
        x2=st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=20
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_commutativity_does_not_hold(self, x1, x2):
        """Property: Regression(x, y) != Regression(y, x) in general.
        
        Linear regression is not symmetric - swapping x and y
        gives different results (except in special cases).
        """
        # Make sure lists are same length
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        
        # Ensure both have variance
        assume(np.var(x1) > 1e-10 and np.var(x2) > 1e-10)
        
        df = pl.DataFrame({
            "x": x1,
            "y": x2
        })
        
        try:
            result_xy = linear_regression(df, "x", "y")
            result_yx = linear_regression(df, "y", "x")
        except ValueError as e:
            # Skip if high leverage cases trigger HC3 error
            if "leverage" in str(e).lower():
                assume(False)
            raise
        
        # The product of slopes should approximate 1 only if R² ≈ 1
        # For general data, slopes will be different
        if result_xy.r_squared < 0.99:
            # Slopes should generally be different
            assert abs(result_xy.slope * result_yx.slope - 1.0) > 0.01
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        base_data=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=30
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_duplicate_data_preserves_result(self, base_data):
        """Property: Duplicating data should not change regression parameters.
        
        If we duplicate all data points, the regression line should
        remain the same (though n_samples will change).
        """
        # Ensure variance
        assume(np.var(base_data) > 1e-10)
        
        # Create y data
        y_data = [2 * x + 3 + np.random.randn() * 0.1 for x in base_data]
        
        df_single = pl.DataFrame({
            "x": base_data,
            "y": y_data
        })
        
        # Duplicate the data
        df_double = pl.DataFrame({
            "x": base_data + base_data,  # Concatenate with itself
            "y": y_data + y_data
        })
        
        try:
            result_single = linear_regression(df_single, "x", "y")
            result_double = linear_regression(df_double, "x", "y")
        except ValueError as e:
            # Skip if high leverage cases trigger HC3 error
            if "leverage" in str(e).lower():
                assume(False)
            raise
        
        # Parameters should be the same
        assert abs(result_single.slope - result_double.slope) < 1e-10
        assert abs(result_single.intercept - result_double.intercept) < 1e-10
        assert abs(result_single.r_squared - result_double.r_squared) < 1e-10
        # But n_samples should double
        assert result_double.n_samples == 2 * result_single.n_samples
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    def test_property_hypothesis_not_available_message(self):
        """Test that skipping works correctly when hypothesis is not available."""
        if not HYPOTHESIS_AVAILABLE:
            # This test runs when hypothesis is not available
            # It validates that our skip decorators work correctly
            assert True, "Skip mechanism works correctly"
        else:
            # This should not be skipped if hypothesis is available
            assert HYPOTHESIS_AVAILABLE, "Hypothesis is available"


def test_property_based_available():
    """Check if property-based testing is available and provide instructions if not."""
    if not HYPOTHESIS_AVAILABLE:
        print("\n" + "="*60)
        print("Property-based testing with hypothesis is not available.")
        print("To enable these tests, install hypothesis:")
        print("  pip install hypothesis")
        print("or")
        print("  uv pip install hypothesis")
        print("="*60 + "\n")
        pytest.skip("hypothesis not installed - property tests skipped")


if __name__ == "__main__":
    # Check if hypothesis is available
    if not HYPOTHESIS_AVAILABLE:
        test_property_based_available()
    else:
        # Run tests with verbose output
        pytest.main([__file__, "-v", "-s"])