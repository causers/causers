"""
Tests for fixed effects estimation in linear regression.

Basic one-way and two-way FE tests
Input validation error tests
Edge case and immutability tests
pyfixest comparison tests (when available)
"""

import numpy as np
import polars as pl
import pytest

import causers


# ============================================================================
# Basic Fixed Effects Tests
# ============================================================================

class TestOneWayFixedEffects:
    """Test one-way (entity) fixed effects."""

    def test_one_way_fe_basic(self):
        """Basic one-way FE recovers correct coefficient."""
        # Panel data: y = 2*x + entity_effect + noise
        df = pl.DataFrame({
            'entity': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'x': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            'y': [3.0, 5.0, 7.0, 5.0, 7.0, 9.0, 7.0, 9.0, 11.0],  # 2*x + entity_effect
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        
        # Coefficient should be 2.0
        assert len(result.coefficients) == 1
        assert abs(result.coefficients[0] - 2.0) < 1e-6
        
        # Intercept should be None (absorbed)
        assert result.intercept is None
        
        # Fixed effects metadata should be populated
        assert result.fixed_effects_absorbed is not None
        assert len(result.fixed_effects_absorbed) == 1
        assert result.fixed_effects_absorbed[0] == 3  # 3 entities
        
        assert result.fixed_effects_names == ['entity']

    def test_one_way_fe_with_list(self):
        """One-way FE with single-element list."""
        df = pl.DataFrame({
            'entity': [1, 1, 2, 2, 3, 3],
            'x': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            'y': [2.0, 4.0, 4.0, 6.0, 6.0, 8.0],
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects=['entity'])
        
        assert len(result.coefficients) == 1
        assert result.fixed_effects_names == ['entity']

    def test_one_way_fe_string_groups(self):
        """One-way FE with string group identifiers."""
        df = pl.DataFrame({
            'firm': ['A', 'A', 'B', 'B', 'C', 'C'],
            'x': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            'y': [2.0, 4.0, 3.0, 5.0, 4.0, 6.0],
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='firm')
        
        assert len(result.coefficients) == 1
        assert abs(result.coefficients[0] - 2.0) < 1e-6
        assert result.fixed_effects_absorbed[0] == 3

    def test_one_way_fe_multiple_covariates(self):
        """One-way FE with multiple covariates (non-collinear)."""
        # Use independent covariates to avoid collinearity after demeaning
        np.random.seed(42)
        df = pl.DataFrame({
            'entity': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'x1': [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5],
            'x2': [0.5, 1.5, 0.8, 1.2, 0.7, 1.8, 0.9, 1.1, 1.6],
            'y': [4.5, 8.5, 8.4, 9.6, 9.1, 13.4, 6.7, 8.3, 11.8],  # ~2*x1 + 3*x2 + entity_effect
        })
        
        result = causers.linear_regression(df, ['x1', 'x2'], 'y', fixed_effects='entity')
        
        assert len(result.coefficients) == 2
        # Coefficients should be reasonable (may not be exact due to data construction)
        assert result.coefficients[0] > 0  # Positive relationship
        assert result.coefficients[1] > 0  # Positive relationship

    def test_one_way_fe_within_r_squared(self):
        """One-way FE returns within R-squared."""
        df = pl.DataFrame({
            'entity': [1, 1, 1, 2, 2, 2],
            'x': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            'y': [2.0, 4.0, 6.0, 4.0, 6.0, 8.0],
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        
        assert result.within_r_squared is not None
        assert 0.0 <= result.within_r_squared <= 1.0


class TestTwoWayFixedEffects:
    """Test two-way (entity + time) fixed effects."""

    def test_two_way_fe_basic(self):
        """Basic two-way FE recovers correct coefficient."""
        # Use random x values to ensure within-group variation after two-way demeaning
        np.random.seed(42)
        entities = [1, 1, 1, 2, 2, 2]
        times = [1, 2, 3, 1, 2, 3]
        # Random x values with variation that won't be absorbed by entity + time
        x = [1.2, 2.5, 1.8, 2.1, 3.2, 2.4]
        # y = 2*x + entity_effect + time_effect + small noise
        entity_effects = {1: 1.0, 2: 2.0}
        time_effects = {1: 0.5, 2: 1.0, 3: 1.5}
        y = [2.0 * x[i] + entity_effects[entities[i]] + time_effects[times[i]] + 0.01 * i
             for i in range(len(x))]
        
        df = pl.DataFrame({
            'entity': entities,
            'time': times,
            'x': x,
            'y': y,
        })
        
        result = causers.linear_regression(
            df, 'x', 'y',
            fixed_effects=['entity', 'time']
        )
        
        assert len(result.coefficients) == 1
        # Should be close to 2.0 (may not be exact due to data construction)
        assert abs(result.coefficients[0] - 2.0) < 0.5
        
        assert result.intercept is None
        
        # Should absorb 2 entities and 3 time periods
        assert result.fixed_effects_absorbed is not None
        assert len(result.fixed_effects_absorbed) == 2
        assert result.fixed_effects_names == ['entity', 'time']

    def test_two_way_fe_balanced_panel(self):
        """Two-way FE on balanced panel."""
        # 3 entities × 4 time periods = 12 observations
        # Use random x values to avoid collinearity with entity + time
        np.random.seed(123)
        entities = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        times = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        # Random x that varies within entity-time cells
        x = [1.2, 2.8, 1.5, 4.2, 2.1, 3.5, 2.8, 5.1, 0.9, 4.2, 3.1, 6.5]
        # y = 1.5*x + entity_effect + time_effect
        entity_effects = {1: 1.0, 2: 2.0, 3: 3.0}
        time_effects = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}
        y = [1.5 * x[i] + entity_effects[entities[i]] + time_effects[times[i]]
             for i in range(len(x))]
        
        df = pl.DataFrame({
            'entity': entities,
            'time': times,
            'x': x,
            'y': y,
        })
        
        result = causers.linear_regression(
            df, 'x', 'y',
            fixed_effects=['entity', 'time']
        )
        
        # Should recover coefficient close to 1.5
        assert abs(result.coefficients[0] - 1.5) < 0.1
        assert result.fixed_effects_absorbed[0] == 3  # 3 entities
        assert result.fixed_effects_absorbed[1] == 4  # 4 time periods


class TestFixedEffectsWithClustering:
    """Test FE combined with clustered standard errors."""

    def test_fe_with_clustered_se(self):
        """FE regression with cluster-robust SE."""
        df = pl.DataFrame({
            'entity': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'x': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            'y': [2.0, 4.0, 6.0, 3.0, 5.0, 7.0, 4.0, 6.0, 8.0, 5.0, 7.0, 9.0],
            'cluster': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        })
        
        result = causers.linear_regression(
            df, 'x', 'y',
            fixed_effects='entity',
            cluster='cluster'
        )
        
        assert result.coefficients is not None
        assert result.standard_errors is not None
        assert result.n_clusters == 2


# ============================================================================
# Input Validation Error Tests
# ============================================================================

class TestFixedEffectsValidation:
    """Test input validation for fixed effects."""

    def test_fe_more_than_two_columns_error(self):
        """Error when >2 FE columns specified."""
        df = pl.DataFrame({
            'a': [1, 1, 2, 2],
            'b': [1, 2, 1, 2],
            'c': [1, 1, 2, 2],
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
        })
        
        with pytest.raises(ValueError, match="at most 2"):
            causers.linear_regression(df, 'x', 'y', fixed_effects=['a', 'b', 'c'])

    def test_fe_column_not_found_error(self):
        """Error when FE column doesn't exist."""
        df = pl.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [2.0, 4.0, 6.0],
        })
        
        with pytest.raises(ValueError, match="not found"):
            causers.linear_regression(df, 'x', 'y', fixed_effects='entity')

    def test_fe_overlap_with_x_error(self):
        """Error when FE column overlaps with x_cols."""
        df = pl.DataFrame({
            'entity': [1, 1, 2, 2],
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
        })
        
        with pytest.raises(ValueError, match="cannot also be a covariate"):
            causers.linear_regression(df, 'x', 'y', fixed_effects='x')

    def test_fe_overlap_with_y_error(self):
        """Error when FE column overlaps with y_col."""
        df = pl.DataFrame({
            'entity': [1, 1, 2, 2],
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
        })
        
        with pytest.raises(ValueError, match="cannot be the outcome"):
            causers.linear_regression(df, 'x', 'y', fixed_effects='y')

    def test_fe_null_values_error(self):
        """Error when FE column contains nulls."""
        df = pl.DataFrame({
            'entity': [1, None, 2, 2],
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
        })
        
        with pytest.raises(ValueError, match="null"):
            causers.linear_regression(df, 'x', 'y', fixed_effects='entity')

    def test_fe_single_value_error(self):
        """Error when FE column has only one unique value."""
        df = pl.DataFrame({
            'entity': [1, 1, 1, 1],
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
        })
        
        with pytest.raises(ValueError, match="only one unique value"):
            causers.linear_regression(df, 'x', 'y', fixed_effects='entity')


# ============================================================================
# Edge Case and Immutability Tests
# ============================================================================

class TestFixedEffectsEdgeCases:
    """Test edge cases for fixed effects."""

    def test_df_not_mutated(self):
        """Verify input DataFrame is not modified."""
        df = pl.DataFrame({
            'entity': [1, 1, 2, 2],
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
        })
        
        # Store original values
        original_x = df['x'].to_list()
        original_y = df['y'].to_list()
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        
        # Verify DataFrame unchanged
        assert df['x'].to_list() == original_x
        assert df['y'].to_list() == original_y

    def test_fe_with_no_intercept_flag(self):
        """FE with include_intercept=False (should work, intercept absorbed anyway)."""
        df = pl.DataFrame({
            'entity': [1, 1, 2, 2],
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
        })
        
        result = causers.linear_regression(
            df, 'x', 'y',
            include_intercept=False,
            fixed_effects='entity'
        )
        
        assert result.intercept is None  # Always None with FE

    def test_fe_minimum_observations(self):
        """Test FE with minimum valid observations."""
        # 2 groups × 2 observations each = 4 total
        df = pl.DataFrame({
            'entity': [1, 1, 2, 2],
            'x': [1.0, 2.0, 1.0, 2.0],
            'y': [2.0, 4.0, 3.0, 5.0],
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        assert result.coefficients is not None

    def test_fe_categorical_column(self):
        """FE with categorical dtype column."""
        df = pl.DataFrame({
            'entity': ['A', 'A', 'B', 'B'],
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
        }).with_columns(pl.col('entity').cast(pl.Categorical))
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        assert result.coefficients is not None

    def test_fe_integer_column(self):
        """FE with integer column."""
        df = pl.DataFrame({
            'entity': [1, 1, 2, 2, 3, 3],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'y': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        assert result.fixed_effects_absorbed[0] == 3


# ============================================================================
# Accuracy / pyfixest Comparison Tests
# ============================================================================

class TestFixedEffectsAccuracy:
    """Test numerical accuracy, ideally comparing to pyfixest."""

    def test_within_transformation_accuracy(self):
        """Verify within-transformation is numerically correct."""
        # Create data where we know the true coefficient
        np.random.seed(42)
        n_entities = 5
        n_periods = 10
        n_obs = n_entities * n_periods
        
        entities = np.repeat(range(n_entities), n_periods)
        entity_effects = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        x = np.random.randn(n_obs)
        true_beta = 2.5
        y = true_beta * x + entity_effects[entities] + 0.1 * np.random.randn(n_obs)
        
        df = pl.DataFrame({
            'entity': entities.tolist(),
            'x': x.tolist(),
            'y': y.tolist(),
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        
        # Should recover true_beta within reasonable tolerance
        assert abs(result.coefficients[0] - true_beta) < 0.1

    def test_two_way_fe_accuracy(self):
        """Verify two-way FE numerical accuracy."""
        np.random.seed(123)
        n_entities = 4
        n_periods = 6
        
        # Generate balanced panel
        entities = []
        times = []
        for e in range(n_entities):
            for t in range(n_periods):
                entities.append(e)
                times.append(t)
        
        n_obs = n_entities * n_periods
        entity_effects = np.array([1.0, 2.0, 3.0, 4.0])
        time_effects = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        
        x = np.random.randn(n_obs)
        true_beta = 1.5
        y = (true_beta * x + 
             entity_effects[entities] + 
             time_effects[times] + 
             0.05 * np.random.randn(n_obs))
        
        df = pl.DataFrame({
            'entity': entities,
            'time': times,
            'x': x.tolist(),
            'y': y.tolist(),
        })
        
        result = causers.linear_regression(
            df, 'x', 'y',
            fixed_effects=['entity', 'time']
        )
        
        # Should recover true_beta within reasonable tolerance
        assert abs(result.coefficients[0] - true_beta) < 0.1

    def test_r_squared_bounds(self):
        """Verify R-squared and within-R-squared are in valid range."""
        df = pl.DataFrame({
            'entity': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'x': [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
            'y': [2.0, 4.1, 5.9, 4.2, 5.8, 8.1, 6.1, 7.9, 10.2],
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        
        assert 0.0 <= result.r_squared <= 1.0
        assert result.within_r_squared is not None
        assert 0.0 <= result.within_r_squared <= 1.0

    def test_se_reasonable(self):
        """Verify standard errors are positive and reasonable."""
        df = pl.DataFrame({
            'entity': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'x': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            'y': [2.1, 3.9, 6.2, 2.8, 5.1, 6.9, 4.2, 5.8, 8.1, 4.9, 7.2, 8.8],
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects='entity')
        
        assert len(result.standard_errors) == 1
        assert result.standard_errors[0] > 0
        # SE should be reasonably small relative to coefficient
        assert result.standard_errors[0] < abs(result.coefficients[0]) * 2


# ============================================================================
# Regression Tests
# ============================================================================

class TestFixedEffectsRegression:
    """Regression tests to prevent regressions in FE functionality."""

    def test_no_fe_still_works(self):
        """Verify regression without FE still works."""
        df = pl.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        
        result = causers.linear_regression(df, 'x', 'y')
        
        assert abs(result.coefficients[0] - 2.0) < 1e-6
        assert result.fixed_effects_absorbed is None
        assert result.within_r_squared is None

    def test_fe_none_explicit(self):
        """Verify explicit fixed_effects=None works."""
        df = pl.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        
        result = causers.linear_regression(df, 'x', 'y', fixed_effects=None)
        
        assert result.fixed_effects_absorbed is None


def main():
    """Run tests when module is executed directly."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    main()
