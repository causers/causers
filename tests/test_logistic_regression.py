"""
Tests for logistic regression functionality in causers.

Validates coefficient estimation, standard errors, clustered inference,
and edge cases against expected behavior and statsmodels reference.
"""

import pytest
import numpy as np
import polars as pl
import warnings


class TestLogisticRegressionBasic:
    """Basic functionality tests for logistic regression."""

    def test_import(self):
        """Test that logistic_regression and LogisticRegressionResult are importable."""
        from causers import logistic_regression, LogisticRegressionResult
        assert callable(logistic_regression)
        assert LogisticRegressionResult is not None

    def test_basic_regression(self):
        """Test basic logistic regression on simple data."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0, 0, 1, 0, 1, 1, 1, 1]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y")
        
        assert result.converged
        assert result.iterations > 0
        assert len(result.coefficients) == 1
        assert result.intercept is not None
        assert len(result.standard_errors) == 1
        assert result.intercept_se is not None
        assert result.n_samples == 8

    def test_multiple_covariates(self):
        """Test logistic regression with multiple covariates."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        prob = 1 / (1 + np.exp(-(0.5 + x1 - 0.5 * x2)))
        y = (np.random.rand(n) < prob).astype(float)
        
        df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
        
        from causers import logistic_regression
        result = logistic_regression(df, ["x1", "x2"], "y")
        
        assert result.converged
        assert len(result.coefficients) == 2
        assert len(result.standard_errors) == 2
        assert result.intercept is not None

    def test_without_intercept(self):
        """Test logistic regression without intercept."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0, 0, 1, 0, 1, 1, 1, 1]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y", include_intercept=False)
        
        assert result.converged
        assert len(result.coefficients) == 1
        assert result.intercept is None
        assert result.intercept_se is None

    def test_result_repr_and_str(self):
        """Test __repr__ and __str__ methods of LogisticRegressionResult."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0, 0, 1, 0, 1, 1, 1, 1]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y")
        
        repr_str = repr(result)
        assert "LogisticRegressionResult" in repr_str
        assert "coefficients" in repr_str
        assert "converged" in repr_str
        
        str_str = str(result)
        assert "Logistic Regression" in str_str
        assert "converged" in str_str or "FAILED" in str_str
        assert "Log-likelihood" in str_str


class TestLogisticRegressionDiagnostics:
    """Tests for logistic regression diagnostic fields."""

    def test_log_likelihood_negative(self):
        """Test that log-likelihood is always negative."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0, 0, 1, 0, 1, 1, 1, 1]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y")
        
        assert result.log_likelihood < 0

    def test_pseudo_r_squared_bounds(self):
        """Test that pseudo R² is between 0 and 1."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0, 0, 1, 0, 1, 1, 1, 1]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y")
        
        assert 0 <= result.pseudo_r_squared <= 1

    def test_convergence_fields(self):
        """Test that converged and iterations fields are populated."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0, 0, 1, 0, 1, 1, 1, 1]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y")
        
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert result.iterations > 0

    def test_standard_errors_positive(self):
        """Test that all standard errors are positive."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0, 0, 1, 0, 1, 1, 1, 1]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y")
        
        assert all(se > 0 for se in result.standard_errors)
        assert result.intercept_se > 0


class TestLogisticRegressionClusteredSE:
    """Tests for clustered standard errors in logistic regression."""

    def test_clustered_se_analytical(self):
        """Test analytical clustered standard errors."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            "y": [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            "cluster": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        })
        
        from causers import logistic_regression
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = logistic_regression(df, "x", "y", cluster="cluster")
            # Should warn about small cluster count
            assert any("clusters" in str(warning.message).lower() for warning in w)
        
        assert result.n_clusters == 6
        assert result.cluster_se_type == "analytical"
        assert result.bootstrap_iterations_used is None

    def test_score_bootstrap(self):
        """Test score bootstrap standard errors."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            "y": [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            "cluster": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        })
        
        from causers import logistic_regression
        result = logistic_regression(
            df, "x", "y", 
            cluster="cluster", 
            bootstrap=True, 
            bootstrap_iterations=500,
            seed=42
        )
        
        assert result.n_clusters == 6
        assert result.cluster_se_type == "bootstrap"
        assert result.bootstrap_iterations_used == 500

    def test_bootstrap_reproducibility(self):
        """Test that same seed produces same results."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            "y": [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            "cluster": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        })
        
        from causers import logistic_regression
        
        result1 = logistic_regression(
            df, "x", "y", 
            cluster="cluster", 
            bootstrap=True, 
            seed=12345
        )
        result2 = logistic_regression(
            df, "x", "y", 
            cluster="cluster", 
            bootstrap=True, 
            seed=12345
        )
        
        assert result1.standard_errors == result2.standard_errors
        assert result1.intercept_se == result2.intercept_se


class TestLogisticRegressionErrorHandling:
    """Tests for error handling in logistic regression."""

    def test_non_binary_y(self):
        """Test that non-binary y raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 0.5, 1.0]  # 0.5 is not allowed
        })
        
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match="0 and 1"):
            logistic_regression(df, "x", "y")

    def test_single_class_y(self):
        """Test that y with only one class raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0]  # Only zeros
        })
        
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match="both 0 and 1"):
            logistic_regression(df, "x", "y")

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError."""
        df = pl.DataFrame({"x": [], "y": []}).cast({"x": pl.Float64, "y": pl.Float64})
        
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match="empty"):
            logistic_regression(df, "x", "y")

    def test_empty_x_cols(self):
        """Test that empty x_cols raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 1.0]
        })
        
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match="at least one"):
            logistic_regression(df, [], "y")

    def test_column_not_found(self):
        """Test that missing column raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 1.0]
        })
        
        from causers import logistic_regression
        
        with pytest.raises(Exception):  # Could be ValueError or other
            logistic_regression(df, "nonexistent", "y")

    def test_bootstrap_without_cluster(self):
        """Test that bootstrap=True without cluster raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 1.0, 1.0]
        })
        
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match="cluster"):
            logistic_regression(df, "x", "y", bootstrap=True)

    def test_invalid_bootstrap_iterations(self):
        """Test that bootstrap_iterations < 1 raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "cluster": [1, 1, 2, 2]
        })
        
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match="at least 1"):
            logistic_regression(
                df, "x", "y", 
                cluster="cluster", 
                bootstrap=True, 
                bootstrap_iterations=0
            )

    def test_perfect_separation(self):
        """Test that perfect separation raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # Perfect separation at x=3.5
        })
        
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match="[Pp]erfect separation"):
            logistic_regression(df, "x", "y")


class TestLogisticRegressionStatsmodelsComparison:
    """Tests comparing logistic regression results against statsmodels."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for comparison tests."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        # Generate y based on logistic model: P(y=1) = logit(0.5 + x)
        prob = 1 / (1 + np.exp(-(0.5 + x)))
        y = (np.random.rand(n) < prob).astype(float)
        return pl.DataFrame({"x": x, "y": y})

    @pytest.mark.skipif(
        not pytest.importorskip("statsmodels", reason="statsmodels not installed"),
        reason="statsmodels not installed"
    )
    def test_coefficient_accuracy_vs_statsmodels(self, sample_data):
        """Test that coefficients match statsmodels within tolerance."""
        import statsmodels.api as sm
        
        from causers import logistic_regression
        result = logistic_regression(sample_data, "x", "y")
        
        # Statsmodels comparison
        X = sm.add_constant(sample_data["x"].to_numpy())
        y = sample_data["y"].to_numpy()
        sm_model = sm.Logit(y, X).fit(disp=0)
        
        # Compare coefficients (intercept first in statsmodels)
        assert np.allclose(result.intercept, sm_model.params[0], rtol=1e-6)
        assert np.allclose(result.coefficients[0], sm_model.params[1], rtol=1e-6)

    @pytest.mark.skipif(
        not pytest.importorskip("statsmodels", reason="statsmodels not installed"),
        reason="statsmodels not installed"
    )
    def test_hc3_se_vs_statsmodels(self, sample_data):
        """Test that HC3 standard errors match statsmodels."""
        import statsmodels.api as sm
        
        from causers import logistic_regression
        result = logistic_regression(sample_data, "x", "y")
        
        # Statsmodels with HC3
        X = sm.add_constant(sample_data["x"].to_numpy())
        y = sample_data["y"].to_numpy()
        sm_model = sm.Logit(y, X).fit(disp=0, cov_type='HC3')
        
        # Compare SE (intercept first in statsmodels)
        # HC3 for logistic may have slight differences, use looser tolerance
        assert np.allclose(result.intercept_se, sm_model.bse[0], rtol=0.1)
        assert np.allclose(result.standard_errors[0], sm_model.bse[1], rtol=0.1)


class TestLogisticRegressionImmutability:
    """Tests for DataFrame immutability."""

    def test_dataframe_unchanged(self):
        """Test that input DataFrame is not mutated."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0, 0, 1, 0, 1, 1, 1, 1]
        })
        
        df_original = df.clone()
        
        from causers import logistic_regression
        _ = logistic_regression(df, "x", "y")
        
        assert df.equals(df_original)
