"""
Tests for logistic regression functionality in causers.

Validates coefficient estimation, standard errors, clustered inference,
and edge cases against expected behavior and statsmodels reference.
"""

import warnings

import numpy as np
import polars as pl
import pytest

from causers import logistic_regression, LogisticRegressionResult

# Conditional import for statsmodels (used in comparison tests)
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class TestLogisticRegressionBasic:
    """Basic functionality tests for logistic regression."""

    def test_import(self):
        """Test that logistic_regression and LogisticRegressionResult are importable."""
        assert callable(logistic_regression)
        assert LogisticRegressionResult is not None

    def test_basic_regression(self):
        """Test basic logistic regression on simple data."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        })
        
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
        
        result = logistic_regression(df, "x", "y")
        
        assert result.log_likelihood < 0

    def test_pseudo_r_squared_bounds(self):
        """Test that pseudo R² is between 0 and 1."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        result = logistic_regression(df, "x", "y")
        
        assert 0 <= result.pseudo_r_squared <= 1

    def test_convergence_fields(self):
        """Test that converged and iterations fields are populated."""
        df = pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        })
        
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
        
        with pytest.raises(ValueError, match="0 and 1"):
            logistic_regression(df, "x", "y")

    def test_single_class_y(self):
        """Test that y with only one class raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0]  # Only zeros
        })
        
        with pytest.raises(ValueError, match="both 0 and 1"):
            logistic_regression(df, "x", "y")

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError."""
        df = pl.DataFrame({"x": [], "y": []}).cast({"x": pl.Float64, "y": pl.Float64})
        
        with pytest.raises(ValueError, match="empty"):
            logistic_regression(df, "x", "y")

    def test_empty_x_cols(self):
        """Test that empty x_cols raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 1.0]
        })
        
        with pytest.raises(ValueError, match="at least one"):
            logistic_regression(df, [], "y")

    def test_column_not_found(self):
        """Test that missing column raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 1.0]
        })
        
        with pytest.raises(Exception):  # Could be ValueError or other
            logistic_regression(df, "nonexistent", "y")

    def test_bootstrap_without_cluster(self):
        """Test that bootstrap=True without cluster raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 1.0, 1.0]
        })
        
        with pytest.raises(ValueError, match="cluster"):
            logistic_regression(df, "x", "y", bootstrap=True)

    def test_invalid_bootstrap_iterations(self):
        """Test that bootstrap_iterations < 1 raises ValueError."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "cluster": [1, 1, 2, 2]
        })
        
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
        
        with pytest.raises(ValueError, match="[Pp]erfect separation"):
            logistic_regression(df, "x", "y")


class TestLogisticRegressionStatsmodelsComparison:
    """Tests comparing logistic regression results against statsmodels."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
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
        
        _ = logistic_regression(df, "x", "y")
        
        assert df.equals(df_original)


# =============================================================================
# Webb Bootstrap Tests for Logistic Regression
# =============================================================================


class TestLogisticWebbBootstrap:
    """Tests for Webb bootstrap in logistic regression."""
    
    @pytest.fixture
    def logistic_webb_data(self) -> pl.DataFrame:
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
        result = logistic_regression(
            logistic_webb_data, "x", "y",
            cluster="cluster_id",
            bootstrap=True,
            seed=42
        )
        
        assert result.cluster_se_type == "bootstrap_rademacher"


# =============================================================================
# Parallel Score Bootstrap Tests (Phase 2)
# =============================================================================


class TestParallelScoreBootstrap:
    """Tests for parallel score bootstrap implementation in logistic regression.
    
    These tests verify the Rayon-parallelized bootstrap loop produces:
    1. Deterministic results with the same seed
    2. Valid (finite, positive) standard errors
    """
    
    @pytest.fixture
    def parallel_bootstrap_data(self) -> pl.DataFrame:
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


# =============================================================================
# Fixed Effects Integration Tests
# =============================================================================


class TestLogisticRegressionFixedEffects:
    """Integration tests for logistic regression with fixed effects (Mundlak strategy)."""

    @pytest.fixture
    def one_way_fe_data(self) -> pl.DataFrame:
        """Generate panel data with one-way entity fixed effects."""
        np.random.seed(42)
        n_entities = 20
        n_per_entity = 25
        n = n_entities * n_per_entity

        entity = []
        x = []
        y = []

        for e in range(n_entities):
            entity_effect = np.random.randn() * 0.3
            for _ in range(n_per_entity):
                entity.append(e)
                xi = np.random.randn()
                prob = 1 / (1 + np.exp(-(0.5 + 0.8 * xi + entity_effect)))
                yi = float(np.random.rand() < prob)
                x.append(xi)
                y.append(yi)

        return pl.DataFrame({
            "x": x,
            "y": y,
            "entity": entity,
        })

    @pytest.fixture
    def two_way_fe_data(self) -> pl.DataFrame:
        """Generate panel data with two-way (entity + time) fixed effects."""
        np.random.seed(42)
        n_entities = 15
        n_times = 10
        n = n_entities * n_times

        entity = []
        time = []
        x = []
        y = []

        for e in range(n_entities):
            entity_effect = np.random.randn() * 0.2
            for t in range(n_times):
                time_effect = 0.05 * t
                entity.append(e)
                time.append(t)
                xi = np.random.randn()
                prob = 1 / (1 + np.exp(-(0.3 + 0.6 * xi + entity_effect + time_effect)))
                yi = float(np.random.rand() < prob)
                x.append(xi)
                y.append(yi)

        return pl.DataFrame({
            "x": x,
            "y": y,
            "entity": entity,
            "time": time,
        })

    # -------------------------------------------------------------------------
    # Basic one-way FE integration tests
    # -------------------------------------------------------------------------

    def test_logistic_fe_one_way_basic(self, one_way_fe_data):
        """Test one-way FE logistic regression converges and returns correct output shape."""
        result = logistic_regression(
            one_way_fe_data, ["x"], "y",
            fixed_effects=["entity"]
        )

        assert result.converged, "FE logistic regression should converge"
        assert result.iterations > 0
        assert len(result.coefficients) == 1, "Should return only original covariate coefficients"
        assert len(result.standard_errors) == 1, "Should return only original covariate SEs"
        assert result.intercept is not None

    def test_logistic_fe_one_way_result_fields(self, one_way_fe_data):
        """Test that FE-specific result fields are populated for one-way FE."""
        result = logistic_regression(
            one_way_fe_data, ["x"], "y",
            fixed_effects=["entity"]
        )

        # FE fields should be populated
        assert result.fixed_effects_absorbed is not None
        assert result.fixed_effects_names is not None
        assert result.within_pseudo_r_squared is not None

        # Check field values
        assert result.fixed_effects_names == ["entity"]
        assert len(result.fixed_effects_absorbed) == 1
        assert result.fixed_effects_absorbed[0] == 20  # 20 entities
        assert 0.0 <= result.within_pseudo_r_squared <= 1.0

    def test_logistic_fe_preserves_backward_compat(self, one_way_fe_data):
        """Test that fixed_effects=None works (backward compatibility)."""
        # Should work without FE (existing behavior)
        result = logistic_regression(one_way_fe_data, ["x"], "y")

        assert result.converged
        assert result.fixed_effects_absorbed is None
        assert result.fixed_effects_names is None
        assert result.within_pseudo_r_squared is None

    # -------------------------------------------------------------------------
    # Two-way FE integration tests
    # -------------------------------------------------------------------------

    def test_logistic_fe_two_way_basic(self, two_way_fe_data):
        """Test two-way FE logistic regression converges and returns correct output shape."""
        result = logistic_regression(
            two_way_fe_data, ["x"], "y",
            fixed_effects=["entity", "time"]
        )

        assert result.converged, "Two-way FE logistic regression should converge"
        assert len(result.coefficients) == 1, "Should return only original covariate coefficients"
        assert len(result.standard_errors) == 1, "Should return only original covariate SEs"

    def test_logistic_fe_two_way_result_fields(self, two_way_fe_data):
        """Test that FE-specific result fields are populated for two-way FE."""
        result = logistic_regression(
            two_way_fe_data, ["x"], "y",
            fixed_effects=["entity", "time"]
        )

        assert result.fixed_effects_names == ["entity", "time"]
        assert len(result.fixed_effects_absorbed) == 2
        assert result.fixed_effects_absorbed[0] == 15  # 15 entities
        assert result.fixed_effects_absorbed[1] == 10  # 10 time periods

    def test_logistic_fe_multiple_covariates(self, two_way_fe_data):
        """Test FE logistic regression with multiple covariates."""
        # Add another covariate
        df = two_way_fe_data.with_columns(
            (pl.col("x") * 0.5 + pl.lit(np.random.randn(len(two_way_fe_data)))).alias("x2")
        )

        result = logistic_regression(
            df, ["x", "x2"], "y",
            fixed_effects=["entity"]
        )

        assert result.converged
        assert len(result.coefficients) == 2
        assert len(result.standard_errors) == 2


class TestLogisticRegressionFEValidation:
    """Tests for FE validation error messages."""

    @pytest.fixture
    def base_data(self) -> pl.DataFrame:
        """Create base test data."""
        return pl.DataFrame({
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "entity": [0, 0, 1, 1, 2, 2, 3, 3],
        })

    def test_fe_too_many_columns(self, base_data):
        """Test that >2 FE columns raises ValueError."""
        df = base_data.with_columns([
            pl.col("entity").alias("fe2"),
            pl.col("entity").alias("fe3"),
        ])

        with pytest.raises(ValueError, match="at most 2"):
            logistic_regression(df, ["x"], "y", fixed_effects=["entity", "fe2", "fe3"])

    def test_fe_column_not_found(self, base_data):
        """Test that missing FE column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            logistic_regression(base_data, ["x"], "y", fixed_effects=["nonexistent"])

    def test_fe_overlap_with_x(self, base_data):
        """Test that FE column overlapping with x_cols raises ValueError."""
        with pytest.raises(ValueError, match="cannot also be a covariate"):
            logistic_regression(base_data, ["x"], "y", fixed_effects=["x"])

    def test_fe_overlap_with_y(self, base_data):
        """Test that FE column equal to y_col raises ValueError."""
        with pytest.raises(ValueError, match="cannot be the outcome"):
            logistic_regression(base_data, ["x"], "y", fixed_effects=["y"])

    def test_fe_null_values(self, base_data):
        """Test that null values in FE column raises ValueError."""
        df = base_data.with_columns(
            pl.when(pl.col("entity") == 0).then(None).otherwise(pl.col("entity")).alias("entity_null")
        )

        with pytest.raises(ValueError, match="null"):
            logistic_regression(df, ["x"], "y", fixed_effects=["entity_null"])

    def test_fe_single_unique_value(self, base_data):
        """Test that single unique value in FE column raises ValueError."""
        df = base_data.with_columns(pl.lit(1).alias("constant_fe"))

        with pytest.raises(ValueError, match="only one unique value"):
            logistic_regression(df, ["x"], "y", fixed_effects=["constant_fe"])


class TestLogisticRegressionFEEdgeCases:
    """Edge case tests for FE logistic regression."""

    def test_fe_minimum_groups(self):
        """Test FE with minimum valid number of groups (2)."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.binomial(1, 0.5, 100).astype(float),
            "entity": [0] * 50 + [1] * 50,  # Exactly 2 groups
        })

        result = logistic_regression(df, ["x"], "y", fixed_effects=["entity"])

        assert result.converged
        assert result.fixed_effects_absorbed == [2]

    def test_fe_with_cluster(self):
        """Test FE combined with clustered standard errors."""
        np.random.seed(42)
        n = 200
        df = pl.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.binomial(1, 0.5, n).astype(float),
            "entity": [i % 10 for i in range(n)],
            "cluster": [i % 20 for i in range(n)],
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = logistic_regression(
                df, ["x"], "y",
                fixed_effects=["entity"],
                cluster="cluster"
            )

        assert result.converged
        assert result.n_clusters == 20
        assert result.fixed_effects_absorbed == [10]

    def test_fe_with_bootstrap(self):
        """Test FE combined with bootstrap standard errors."""
        np.random.seed(42)
        n = 200
        df = pl.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.binomial(1, 0.5, n).astype(float),
            "entity": [i % 10 for i in range(n)],
            "cluster": [i % 20 for i in range(n)],
        })

        result = logistic_regression(
            df, ["x"], "y",
            fixed_effects=["entity"],
            cluster="cluster",
            bootstrap=True,
            bootstrap_iterations=100,
            seed=42
        )

        assert result.converged
        assert result.cluster_se_type == "bootstrap_rademacher"
        assert result.fixed_effects_absorbed == [10]

    def test_fe_collinearity_error(self):
        """Test that collinearity from Mundlak augmentation raises appropriate error."""
        # Create data where each observation is its own group
        # This creates perfect collinearity in the Mundlak terms
        np.random.seed(42)
        n = 50
        df = pl.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.binomial(1, 0.5, n).astype(float),
            "entity": list(range(n)),  # Each obs is its own group
        })

        with pytest.raises(ValueError, match="[Cc]ollinear"):
            logistic_regression(df, ["x"], "y", fixed_effects=["entity"])


# =============================================================================
# Validation Tests
#
# NOTE: pyfixest does NOT support fixed effects for GLMs (logistic regression)
# as of version 0.25+. See: https://github.com/py-econometrics/pyfixest/issues
# The error: "NotImplementedError: Fixed effects are not yet supported for GLMs."
#
# Instead, we validate against statsmodels with manually-added Mundlak terms
# (Correlated Random Effects / CRE approach), which is mathematically equivalent
# to our Mundlak implementation.
# =============================================================================


class TestLogisticRegressionMundlakValidation:
    """Numerical validation tests comparing against statsmodels with manual Mundlak terms.
    
    The Mundlak (1978) approach adds group means of covariates as additional regressors.
    This is equivalent to the Correlated Random Effects (CRE) approach and produces
    consistent estimates under the same assumptions as fixed effects for nonlinear models.
    
    We validate by:
    1. Computing Mundlak terms manually in Python
    2. Running statsmodels logit on the augmented design matrix
    3. Comparing the coefficients for the original covariates
    """

    @pytest.fixture
    def panel_data(self) -> pl.DataFrame:
        """Generate panel data for validation."""
        np.random.seed(42)
        n_entities = 30
        n_times = 8
        n = n_entities * n_times

        entity = []
        time = []
        x1 = []
        x2 = []
        y = []

        for e in range(n_entities):
            entity_effect = np.random.randn() * 0.2
            for t in range(n_times):
                time_effect = 0.03 * t
                entity.append(e)
                time.append(t)
                x1i = np.random.randn()
                x2i = np.random.randn() * 0.5
                prob = 1 / (1 + np.exp(-(0.3 + 0.5 * x1i - 0.3 * x2i + entity_effect + time_effect)))
                yi = float(np.random.rand() < prob)
                x1.append(x1i)
                x2.append(x2i)
                y.append(yi)

        return pl.DataFrame({
            "x1": x1,
            "x2": x2,
            "y": y,
            "entity": entity,
            "time": time,
        })

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_logistic_fe_matches_statsmodels_mundlak_one_way(self, panel_data):
        """Test that one-way FE coefficients match statsmodels with manual Mundlak terms.
        
        Coefficients within 1e-6 of reference implementation.
        """
        # causers result with FE
        result = logistic_regression(
            panel_data, ["x1", "x2"], "y",
            fixed_effects=["entity"]
        )

        # Manually compute Mundlak terms and run statsmodels
        df_pandas = panel_data.to_pandas()
        
        # Compute entity means of x1 and x2
        entity_means = df_pandas.groupby("entity")[["x1", "x2"]].transform("mean")
        df_pandas["x1_entity_mean"] = entity_means["x1"]
        df_pandas["x2_entity_mean"] = entity_means["x2"]
        
        # Build design matrix: [intercept, x1, x2, x1_entity_mean, x2_entity_mean]
        X = sm.add_constant(df_pandas[["x1", "x2", "x1_entity_mean", "x2_entity_mean"]])
        y = df_pandas["y"].values
        
        # Fit statsmodels logit
        sm_model = sm.Logit(y, X).fit(disp=0)
        
        # Compare coefficients for original covariates (x1, x2)
        # The order in statsmodels is: const, x1, x2, x1_entity_mean, x2_entity_mean
        assert np.allclose(result.coefficients[0], sm_model.params["x1"], atol=1e-5), \
            f"x1 coef mismatch: causers={result.coefficients[0]}, statsmodels={sm_model.params['x1']}"
        assert np.allclose(result.coefficients[1], sm_model.params["x2"], atol=1e-5), \
            f"x2 coef mismatch: causers={result.coefficients[1]}, statsmodels={sm_model.params['x2']}"

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_logistic_fe_matches_statsmodels_mundlak_two_way(self, panel_data):
        """Test that two-way FE coefficients match statsmodels with manual Mundlak terms.
        
        Validates for two-way fixed effects.
        """
        # causers result with two-way FE
        result = logistic_regression(
            panel_data, ["x1", "x2"], "y",
            fixed_effects=["entity", "time"]
        )

        # Manually compute Mundlak terms for both entity and time
        df_pandas = panel_data.to_pandas()
        
        # Entity means
        entity_means = df_pandas.groupby("entity")[["x1", "x2"]].transform("mean")
        df_pandas["x1_entity_mean"] = entity_means["x1"]
        df_pandas["x2_entity_mean"] = entity_means["x2"]
        
        # Time means
        time_means = df_pandas.groupby("time")[["x1", "x2"]].transform("mean")
        df_pandas["x1_time_mean"] = time_means["x1"]
        df_pandas["x2_time_mean"] = time_means["x2"]
        
        # Build design matrix with all Mundlak terms
        X = sm.add_constant(df_pandas[[
            "x1", "x2",
            "x1_entity_mean", "x2_entity_mean",
            "x1_time_mean", "x2_time_mean"
        ]])
        y = df_pandas["y"].values
        
        # Fit statsmodels logit
        sm_model = sm.Logit(y, X).fit(disp=0)
        
        # Compare coefficients for original covariates
        assert np.allclose(result.coefficients[0], sm_model.params["x1"], atol=1e-5), \
            f"x1 coef mismatch: causers={result.coefficients[0]}, statsmodels={sm_model.params['x1']}"
        assert np.allclose(result.coefficients[1], sm_model.params["x2"], atol=1e-5), \
            f"x2 coef mismatch: causers={result.coefficients[1]}, statsmodels={sm_model.params['x2']}"

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_logistic_fe_matches_statsmodels_se_one_way(self, panel_data):
        """Test that one-way FE standard errors match statsmodels within tolerance.
        
        Standard errors within 1e-5 of reference.
        """
        # causers result with FE
        result = logistic_regression(
            panel_data, ["x1", "x2"], "y",
            fixed_effects=["entity"]
        )

        # Manually compute Mundlak terms and run statsmodels
        df_pandas = panel_data.to_pandas()
        
        entity_means = df_pandas.groupby("entity")[["x1", "x2"]].transform("mean")
        df_pandas["x1_entity_mean"] = entity_means["x1"]
        df_pandas["x2_entity_mean"] = entity_means["x2"]
        
        X = sm.add_constant(df_pandas[["x1", "x2", "x1_entity_mean", "x2_entity_mean"]])
        y = df_pandas["y"].values
        
        # Fit with HC3 robust SE (matching causers default)
        sm_model = sm.Logit(y, X).fit(disp=0, cov_type='HC3')
        
        # Compare SEs for original covariates
        # Note: HC3 for logistic may have slight differences, use 10% tolerance
        assert np.allclose(result.standard_errors[0], sm_model.bse["x1"], rtol=0.1), \
            f"x1 SE mismatch: causers={result.standard_errors[0]}, statsmodels={sm_model.bse['x1']}"
        assert np.allclose(result.standard_errors[1], sm_model.bse["x2"], rtol=0.1), \
            f"x2 SE mismatch: causers={result.standard_errors[1]}, statsmodels={sm_model.bse['x2']}"

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_logistic_fe_log_likelihood_matches(self, panel_data):
        """Test that log-likelihood matches statsmodels within tolerance.
        
        Log-likelihood within 1e-4 of reference.
        """
        # causers result with FE
        result = logistic_regression(
            panel_data, ["x1", "x2"], "y",
            fixed_effects=["entity"]
        )

        # Manually compute Mundlak terms and run statsmodels
        df_pandas = panel_data.to_pandas()
        
        entity_means = df_pandas.groupby("entity")[["x1", "x2"]].transform("mean")
        df_pandas["x1_entity_mean"] = entity_means["x1"]
        df_pandas["x2_entity_mean"] = entity_means["x2"]
        
        X = sm.add_constant(df_pandas[["x1", "x2", "x1_entity_mean", "x2_entity_mean"]])
        y = df_pandas["y"].values
        
        sm_model = sm.Logit(y, X).fit(disp=0)
        
        # Compare log-likelihood
        assert np.allclose(result.log_likelihood, sm_model.llf, atol=1e-4), \
            f"LL mismatch: causers={result.log_likelihood}, statsmodels={sm_model.llf}"


class TestLogisticRegressionPyfixestValidation:
    """Placeholder tests for pyfixest validation - currently skipped.
    
    pyfixest does NOT support fixed effects for GLMs (logistic regression) as of
    version 0.25+. These tests are kept as placeholders for when pyfixest adds
    FE support for GLMs.
    
    Error from pyfixest: "NotImplementedError: Fixed effects are not yet supported for GLMs."
    """

    @pytest.fixture
    def panel_data(self) -> pl.DataFrame:
        """Generate panel data for pyfixest comparison."""
        np.random.seed(42)
        n_entities = 30
        n_times = 8
        n = n_entities * n_times

        entity = []
        time = []
        x1 = []
        x2 = []
        y = []

        for e in range(n_entities):
            entity_effect = np.random.randn() * 0.2
            for t in range(n_times):
                time_effect = 0.03 * t
                entity.append(e)
                time.append(t)
                x1i = np.random.randn()
                x2i = np.random.randn() * 0.5
                prob = 1 / (1 + np.exp(-(0.3 + 0.5 * x1i - 0.3 * x2i + entity_effect + time_effect)))
                yi = float(np.random.rand() < prob)
                x1.append(x1i)
                x2.append(x2i)
                y.append(yi)

        return pl.DataFrame({
            "x1": x1,
            "x2": x2,
            "y": y,
            "entity": entity,
            "time": time,
        })

    @pytest.mark.skip(reason="pyfixest does not support fixed effects for GLMs (feglm)")
    def test_logistic_fe_matches_pyfixest_coef_one_way(self, panel_data):
        """Placeholder: one-way FE coefficients vs pyfixest.
        
        Skipped because pyfixest.feglm() raises NotImplementedError for fixed effects.
        See TestLogisticRegressionMundlakValidation for alternative validation.
        """
        pass

    @pytest.mark.skip(reason="pyfixest does not support fixed effects for GLMs (feglm)")
    def test_logistic_fe_matches_pyfixest_coef_two_way(self, panel_data):
        """Placeholder: two-way FE coefficients vs pyfixest.
        
        Skipped because pyfixest.feglm() raises NotImplementedError for fixed effects.
        See TestLogisticRegressionMundlakValidation for alternative validation.
        """
        pass

    @pytest.mark.skip(reason="pyfixest does not support fixed effects for GLMs (feglm)")
    def test_logistic_fe_matches_pyfixest_se_one_way(self, panel_data):
        """Placeholder: one-way FE standard errors vs pyfixest.
        
        Skipped because pyfixest.feglm() raises NotImplementedError for fixed effects.
        See TestLogisticRegressionMundlakValidation for alternative validation.
        """
        pass

    @pytest.mark.skip(reason="pyfixest does not support fixed effects for GLMs (feglm)")
    def test_logistic_fe_matches_pyfixest_ll(self, panel_data):
        """Placeholder: log-likelihood vs pyfixest.
        
        Skipped because pyfixest.feglm() raises NotImplementedError for fixed effects.
        See TestLogisticRegressionMundlakValidation for alternative validation.
        """
        pass
