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
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
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
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
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
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
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
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y")
        
        assert result.log_likelihood < 0

    def test_pseudo_r_squared_bounds(self):
        """Test that pseudo R² is between 0 and 1."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        from causers import logistic_regression
        result = logistic_regression(df, "x", "y")
        
        assert 0 <= result.pseudo_r_squared <= 1

    def test_convergence_fields(self):
        """Test that converged and iterations fields are populated."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
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
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
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
            "y": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
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
        """Test score bootstrap standard errors.
        
        Note: With the bootstrap_method parameter, cluster_se_type is now
        'bootstrap_rademacher' (default) or 'bootstrap_webb'.
        """
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            "y": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
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
        assert result.cluster_se_type == "bootstrap_rademacher"
        assert result.bootstrap_iterations_used == 500

    def test_bootstrap_reproducibility(self):
        """Test that same seed produces same results."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            "y": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
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
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        df_original = df.clone()
        
        from causers import logistic_regression
        _ = logistic_regression(df, "x", "y")
        
        assert df.equals(df_original)


# =============================================================================
# TASK-016: Webb Bootstrap Tests for Logistic Regression (REQ-030)
# =============================================================================


class TestLogisticWebbBootstrap:
    """Tests for Webb bootstrap in logistic regression (REQ-030)."""
    
    @pytest.fixture
    def logistic_webb_data(self):
        """Create test data for logistic regression Webb tests."""
        np.random.seed(42)
        n_clusters = 10
        n_per_cluster = 20
        n = n_clusters * n_per_cluster
        
        cluster_ids = []
        x = []
        y = []
        
        for g in range(n_clusters):
            cluster_effect = np.random.randn() * 0.2
            for _ in range(n_per_cluster):
                cluster_ids.append(g)
                xi = np.random.randn()
                prob = 1 / (1 + np.exp(-(0.5 + xi + cluster_effect)))
                yi = float(np.random.rand() < prob)
                x.append(xi)
                y.append(yi)
        
        return pl.DataFrame({
            "x": x,
            "y": y,
            "cluster_id": cluster_ids,
        })
    
    def test_logistic_webb_produces_finite_se(self, logistic_webb_data):
        """Verify Webb bootstrap produces finite, positive SE for logistic regression."""
        from causers import logistic_regression
        
        result = logistic_regression(
            logistic_webb_data, "x", "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_method="webb",
            bootstrap_iterations=500,
            seed=42
        )
        
        assert result.converged
        assert all(np.isfinite(se) for se in result.standard_errors)
        assert all(se > 0 for se in result.standard_errors)
        assert np.isfinite(result.intercept_se)
        assert result.intercept_se > 0
    
    def test_logistic_webb_cluster_se_type(self, logistic_webb_data):
        """Verify cluster_se_type is 'bootstrap_webb' for logistic regression."""
        from causers import logistic_regression
        
        result = logistic_regression(
            logistic_webb_data, "x", "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_method="webb",
            seed=42
        )
        
        assert result.cluster_se_type == "bootstrap_webb"
    
    def test_logistic_webb_reproducibility(self, logistic_webb_data):
        """Same seed should produce identical Webb results for logistic regression."""
        from causers import logistic_regression
        
        result1 = logistic_regression(
            logistic_webb_data, "x", "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_method="webb",
            seed=12345
        )
        
        result2 = logistic_regression(
            logistic_webb_data, "x", "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_method="webb",
            seed=12345
        )
        
        np.testing.assert_array_equal(
            result1.standard_errors,
            result2.standard_errors,
            err_msg="Same seed produced different Webb SEs for logistic regression"
        )
        assert result1.intercept_se == result2.intercept_se
    
    def test_logistic_webb_different_from_rademacher(self, logistic_webb_data):
        """Webb and Rademacher should produce different SE values for logistic."""
        from causers import logistic_regression
        
        result_webb = logistic_regression(
            logistic_webb_data, "x", "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_method="webb",
            bootstrap_iterations=500,
            seed=42
        )
        
        result_rademacher = logistic_regression(
            logistic_webb_data, "x", "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_method="rademacher",
            bootstrap_iterations=500,
            seed=42
        )
        
        # SEs should differ
        assert result_webb.standard_errors != result_rademacher.standard_errors, \
            "Webb and Rademacher produced identical SEs for logistic regression"
    
    def test_logistic_webb_case_insensitive(self, logistic_webb_data):
        """Verify bootstrap_method='webb' is case-insensitive for logistic."""
        from causers import logistic_regression
        
        for method in ["webb", "Webb", "WEBB"]:
            result = logistic_regression(
                logistic_webb_data, "x", "y",
                cluster="cluster_id",
                bootstrap=True,
                bootstrap_method=method,
                bootstrap_iterations=100,
                seed=42
            )
            assert result.cluster_se_type == "bootstrap_webb", \
                f"Failed for bootstrap_method='{method}'"
    
    def test_logistic_rademacher_case_insensitive(self, logistic_webb_data):
        """Verify bootstrap_method='rademacher' is case-insensitive for logistic."""
        from causers import logistic_regression
        
        for method in ["rademacher", "Rademacher", "RADEMACHER"]:
            result = logistic_regression(
                logistic_webb_data, "x", "y",
                cluster="cluster_id",
                bootstrap=True,
                bootstrap_method=method,
                bootstrap_iterations=100,
                seed=42
            )
            assert result.cluster_se_type == "bootstrap_rademacher", \
                f"Failed for bootstrap_method='{method}'"
    
    def test_logistic_invalid_bootstrap_method(self, logistic_webb_data):
        """Invalid bootstrap_method should raise ValueError for logistic."""
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match=r"bootstrap_method"):
            logistic_regression(
                logistic_webb_data, "x", "y",
                cluster="cluster_id",
                bootstrap=True,
                bootstrap_method="invalid_method",
                seed=42
            )
    
    def test_logistic_bootstrap_method_without_bootstrap_flag(self, logistic_webb_data):
        """bootstrap_method='webb' with bootstrap=False should raise ValueError."""
        from causers import logistic_regression
        
        with pytest.raises(ValueError, match=r"bootstrap"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logistic_regression(
                    logistic_webb_data, "x", "y",
                    cluster="cluster_id",
                    bootstrap=False,
                    bootstrap_method="webb"
                )
    
    def test_logistic_default_bootstrap_method(self, logistic_webb_data):
        """Default bootstrap_method should be 'rademacher' for logistic."""
        from causers import logistic_regression
        
        result = logistic_regression(
            logistic_webb_data, "x", "y",
            cluster="cluster_id",
            bootstrap=True,
            seed=42
        )
        
        assert result.cluster_se_type == "bootstrap_rademacher"


# =============================================================================
# TASK-006: Parallel Score Bootstrap Tests (Phase 2)
# =============================================================================


class TestParallelScoreBootstrap:
    """Tests for parallel score bootstrap implementation in logistic regression.
    
    These tests verify the Rayon-parallelized bootstrap loop produces:
    1. Deterministic results with the same seed
    2. Valid (finite, positive) standard errors
    """
    
    @pytest.fixture
    def parallel_bootstrap_data(self):
        """Create test data for parallel bootstrap tests.
        
        Uses larger dataset with more clusters to exercise parallel execution.
        """
        np.random.seed(42)
        n_clusters = 20
        n_per_cluster = 30
        n = n_clusters * n_per_cluster
        
        cluster_ids = []
        x1 = []
        x2 = []
        y = []
        
        for g in range(n_clusters):
            cluster_effect = np.random.randn() * 0.3
            for _ in range(n_per_cluster):
                cluster_ids.append(g)
                x1i = np.random.randn()
                x2i = np.random.randn() * 0.5
                prob = 1 / (1 + np.exp(-(0.3 + 0.5 * x1i - 0.3 * x2i + cluster_effect)))
                yi = float(np.random.rand() < prob)
                x1.append(x1i)
                x2.append(x2i)
                y.append(yi)
        
        return pl.DataFrame({
            "x1": x1,
            "x2": x2,
            "y": y,
            "cluster_id": cluster_ids,
        })
    
    def test_parallel_bootstrap_determinism_multiple_runs(self, parallel_bootstrap_data):
        """Verify parallel bootstrap produces identical results across multiple runs.
        
        The iteration-indexed RNG seeding (seed.wrapping_add(iter_idx)) ensures
        deterministic parallel execution regardless of thread scheduling.
        """
        from causers import logistic_regression
        
        # Run the same bootstrap 3 times with same seed
        results = []
        for _ in range(3):
            result = logistic_regression(
                parallel_bootstrap_data, ["x1", "x2"], "y",
                cluster="cluster_id",
                bootstrap=True,
                bootstrap_iterations=500,
                seed=54321
            )
            results.append(result)
        
        # All runs should produce identical SEs
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0].standard_errors,
                results[i].standard_errors,
                err_msg=f"Run 0 and run {i} produced different SEs"
            )
            assert results[0].intercept_se == results[i].intercept_se, \
                f"Run 0 and run {i} produced different intercept SEs"
    
    def test_parallel_bootstrap_valid_se_values(self, parallel_bootstrap_data):
        """Verify parallel bootstrap produces finite, positive standard errors."""
        from causers import logistic_regression
        
        result = logistic_regression(
            parallel_bootstrap_data, ["x1", "x2"], "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_iterations=1000,
            seed=12345
        )
        
        # All SEs should be finite and positive
        assert all(np.isfinite(se) for se in result.standard_errors), \
            "Bootstrap produced non-finite standard errors"
        assert all(se > 0 for se in result.standard_errors), \
            "Bootstrap produced non-positive standard errors"
        assert np.isfinite(result.intercept_se), \
            "Bootstrap produced non-finite intercept SE"
        assert result.intercept_se > 0, \
            "Bootstrap produced non-positive intercept SE"
        
        # Verify bootstrap_iterations_used is correct
        assert result.bootstrap_iterations_used == 1000
    
    def test_parallel_bootstrap_different_seeds_different_results(self, parallel_bootstrap_data):
        """Verify different seeds produce different bootstrap results."""
        from causers import logistic_regression
        
        result1 = logistic_regression(
            parallel_bootstrap_data, ["x1", "x2"], "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_iterations=500,
            seed=11111
        )
        
        result2 = logistic_regression(
            parallel_bootstrap_data, ["x1", "x2"], "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_iterations=500,
            seed=22222
        )
        
        # Results should differ (at least one SE should be different)
        ses_differ = any(
            se1 != se2
            for se1, se2 in zip(result1.standard_errors, result2.standard_errors)
        )
        assert ses_differ, "Different seeds produced identical SEs (very unlikely)"
    
    def test_parallel_bootstrap_reasonable_se_magnitude(self, parallel_bootstrap_data):
        """Verify bootstrap SEs are reasonable compared to analytical SEs."""
        from causers import logistic_regression
        
        # Get analytical clustered SE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_analytical = logistic_regression(
                parallel_bootstrap_data, ["x1", "x2"], "y",
                cluster="cluster_id",
                bootstrap=False
            )
        
        # Get bootstrap SE
        result_bootstrap = logistic_regression(
            parallel_bootstrap_data, ["x1", "x2"], "y",
            cluster="cluster_id",
            bootstrap=True,
            bootstrap_iterations=1000,
            seed=42
        )
        
        # Bootstrap SEs should be in same order of magnitude as analytical
        # (within factor of 5, very loose to avoid flaky tests)
        for se_b, se_a in zip(result_bootstrap.standard_errors, result_analytical.standard_errors):
            ratio = se_b / se_a if se_a > 0 else float('inf')
            assert 0.2 < ratio < 5.0, \
                f"Bootstrap SE {se_b} differs too much from analytical SE {se_a} (ratio: {ratio})"
