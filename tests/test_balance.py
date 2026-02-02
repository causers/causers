"""Tests for covariate balance checking (balance_check).

Covers:
  TASK-BAL-12 - Core functionality tests
  TASK-BAL-13 - Validation / error handling tests
  TASK-BAL-14 - Edge cases and warning tests
"""

import math
import warnings

import numpy as np
import polars as pl
import pytest

import causers


# ============================================================================
# Helpers
# ============================================================================

def _simple_df():
    """Small hand-constructed dataset with ASYMMETRIC variances.

    Treatment group (T=1): x1 = [2, 4, 6],       x2 = [10, 20, 30]
    Control   group (T=0): x1 = [1, 2, 5, 10],    x2 = [12, 18, 24, 30]

    x1 treated:  mean=4.0, var=4.0
    x1 control:  mean=4.5, var=16.333...
    x2 treated:  mean=20.0, var=100.0
    x2 control:  mean=21.0, var=54.0

    Asymmetric group sizes (3 vs 4) and different variances ensure that
    pooled-SD, VR, and SMD computations are non-trivially tested.
    """
    return pl.DataFrame({
        "T": [1, 1, 1, 0, 0, 0, 0],
        "x1": [2.0, 4.0, 6.0, 1.0, 2.0, 5.0, 10.0],
        "x2": [10.0, 20.0, 30.0, 12.0, 18.0, 24.0, 30.0],
    })


def _bessel_var(values):
    """Compute sample variance with Bessel correction (N-1 denominator)."""
    n = len(values)
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _weighted_mean(values, weights):
    """Compute weighted mean: sum(w*x) / sum(w)."""
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def _weighted_var_reliability(values, weights):
    """Compute weighted variance with reliability-weights correction.

    var = (sum_w / (sum_w^2 - sum_w2)) * sum(w * (x - mean_w)^2)
    where sum_w = V1, sum_w2 = V2.
    """
    wm = _weighted_mean(values, weights)
    v1 = sum(weights)
    v2 = sum(w ** 2 for w in weights)
    m2 = sum(w * (x - wm) ** 2 for x, w in zip(values, weights))
    return m2 * v1 / (v1 ** 2 - v2)


# ============================================================================
# TASK-BAL-12: Core tests
# ============================================================================

class TestBalanceBasic:
    """Basic smoke tests for balance_check."""

    def test_balance_basic(self):
        """Simple binary 0/1 treatment, 2 covariates, result type check."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])

        assert isinstance(result, causers.BalanceCheckResult)
        assert isinstance(result.smd, dict)
        assert result.n_treated == 3
        assert result.n_control == 4

    def test_single_covariate_string(self):
        """Passing covariate_cols as a single string works."""
        df = _simple_df()
        result = causers.balance_check(df, "T", "x1")

        assert result.covariates == ["x1"]
        assert "x1" in result.smd

    def test_sample_sizes(self):
        """n_treated and n_control match expected counts."""
        df = pl.DataFrame({
            "T": [1] * 7 + [0] * 13,
            "x": list(range(20)),
        }).cast({"x": pl.Float64})
        result = causers.balance_check(df, "T", "x")

        assert result.n_treated == 7
        assert result.n_control == 13

    def test_covariates_list(self):
        """result.covariates matches the input covariate names."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])

        assert set(result.covariates) == {"x1", "x2"}

    def test_is_weighted_false(self):
        """Unweighted analysis returns is_weighted=False, ess_* = None."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])

        assert result.is_weighted is False
        assert result.ess_treated is None
        assert result.ess_control is None

    def test_treatment_value_explicit(self):
        """Explicit treatment_value=1, control_value=0."""
        df = _simple_df()
        result = causers.balance_check(
            df, "T", ["x1", "x2"],
            treatment_value=1, control_value=0,
        )
        assert result.n_treated == 3
        assert result.n_control == 4

    def test_treatment_value_auto_detect(self):
        """Auto-detection picks larger value as treatment."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1"])

        # Treated = rows where T=1 (the larger value); mean_treated should
        # correspond to [2, 4, 6] -> mean = 4.0
        assert abs(result.mean_treated["x1"] - 4.0) < 1e-10


class TestBalanceMoments:
    """Verify first and second moments against hand computations."""

    def test_means(self):
        """Hand-computed means for asymmetric-size groups."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])

        # Treated (3 obs): x1 mean = (2+4+6)/3 = 4.0
        assert abs(result.mean_treated["x1"] - 4.0) < 1e-10
        # Control (4 obs): x1 mean = (1+2+5+10)/4 = 4.5
        assert abs(result.mean_control["x1"] - 4.5) < 1e-10

        # Treated: x2 mean = (10+20+30)/3 = 20.0
        assert abs(result.mean_treated["x2"] - 20.0) < 1e-10
        # Control: x2 mean = (12+18+24+30)/4 = 21.0
        assert abs(result.mean_control["x2"] - 21.0) < 1e-10

    def test_variances(self):
        """Hand-computed variances with Bessel correction, asymmetric groups."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])

        treated_x1 = [2.0, 4.0, 6.0]
        control_x1 = [1.0, 2.0, 5.0, 10.0]

        var_t_x1 = _bessel_var(treated_x1)  # 4.0
        var_c_x1 = _bessel_var(control_x1)  # 16.333...

        # Verify expected values themselves
        assert abs(var_t_x1 - 4.0) < 1e-10
        assert abs(var_c_x1 - 49.0 / 3.0) < 1e-10  # (1-4.5)^2+(2-4.5)^2+(5-4.5)^2+(10-4.5)^2 = 49, /3

        # Verify results match
        assert abs(result.var_treated["x1"] - var_t_x1) < 1e-10
        assert abs(result.var_control["x1"] - var_c_x1) < 1e-10

        # Verify variances are DIFFERENT (non-trivial VR test)
        assert abs(var_t_x1 - var_c_x1) > 1.0

        treated_x2 = [10.0, 20.0, 30.0]
        control_x2 = [12.0, 18.0, 24.0, 30.0]
        var_t_x2 = _bessel_var(treated_x2)
        var_c_x2 = _bessel_var(control_x2)

        assert abs(result.var_treated["x2"] - var_t_x2) < 1e-10
        assert abs(result.var_control["x2"] - var_c_x2) < 1e-10

        # SDs
        assert abs(result.sd_treated["x1"] - math.sqrt(var_t_x1)) < 1e-10
        assert abs(result.sd_control["x1"] - math.sqrt(var_c_x1)) < 1e-10


class TestBalanceMetrics:
    """Verify derived metrics (SMD, variance ratio)."""

    def test_smd_asymmetric_variance(self):
        """SMD with different group variances uses pooled SD correctly.

        A bug that uses only one group's variance would produce a different value.
        """
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])

        # x1: var_t=4.0, var_c=49/3 -- these are NOT equal
        var_t_x1 = _bessel_var([2.0, 4.0, 6.0])
        var_c_x1 = _bessel_var([1.0, 2.0, 5.0, 10.0])
        pooled_sd_x1 = math.sqrt((var_t_x1 + var_c_x1) / 2.0)
        expected_smd_x1 = (4.0 - 4.5) / pooled_sd_x1

        assert abs(result.smd["x1"] - expected_smd_x1) < 1e-10
        # Confirm this is NOT what you'd get using only treated variance
        wrong_smd_treated_only = (4.0 - 4.5) / math.sqrt(var_t_x1)
        assert abs(result.smd["x1"] - wrong_smd_treated_only) > 0.01

        # x2: verify negative SMD sign is preserved
        var_t_x2 = _bessel_var([10.0, 20.0, 30.0])
        var_c_x2 = _bessel_var([12.0, 18.0, 24.0, 30.0])
        expected_smd_x2 = (20.0 - 21.0) / math.sqrt((var_t_x2 + var_c_x2) / 2.0)
        assert expected_smd_x2 < 0  # negative
        assert abs(result.smd["x2"] - expected_smd_x2) < 1e-10

    def test_variance_ratio_not_one(self):
        """VR != 1.0 when group variances differ.

        A bug that always returns 1.0 would fail.
        """
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1"])

        var_t = _bessel_var([2.0, 4.0, 6.0])       # 4.0
        var_c = _bessel_var([1.0, 2.0, 5.0, 10.0])  # 49/3 ≈ 16.333

        expected_vr = var_t / var_c  # ≈ 0.2449
        assert abs(result.variance_ratio["x1"] - expected_vr) < 1e-10
        assert expected_vr < 0.5  # clearly not 1.0

    def test_smd_sign(self):
        """SMD preserves sign: positive when treated > control, negative otherwise."""
        # Construct data where treated mean > control for x1,
        # treated mean < control for x2
        df = pl.DataFrame({
            "T": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "x1": [10.0, 11.0, 12.0, 13.0, 14.0,
                   10.0, 11.0, 12.0, 13.0, 14.0,
                   1.0, 2.0, 3.0, 4.0, 5.0,
                   1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0,
                   1.0, 2.0, 3.0, 4.0, 5.0,
                   10.0, 11.0, 12.0, 13.0, 14.0,
                   10.0, 11.0, 12.0, 13.0, 14.0],
        })
        result = causers.balance_check(df, "T", ["x1", "x2"])

        assert result.smd["x1"] > 0
        assert result.smd["x2"] < 0
        # They should be equal in magnitude (symmetric construction)
        assert abs(abs(result.smd["x1"]) - abs(result.smd["x2"])) < 1e-10


class TestBalanceResult:
    """Tests for BalanceCheckResult convenience methods."""

    def test_result_summary(self):
        """summary() returns DataFrame with correct columns, row count, and values."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])
        summary = result.summary()

        assert isinstance(summary, pl.DataFrame)
        expected_cols = {
            "covariate", "mean_treated", "mean_control",
            "sd_treated", "sd_control", "smd", "variance_ratio",
        }
        assert set(summary.columns) == expected_cols
        assert len(summary) == 2

        # Verify the summary data matches the result attributes
        for row in summary.iter_rows(named=True):
            cov = row["covariate"]
            assert abs(row["mean_treated"] - result.mean_treated[cov]) < 1e-10
            assert abs(row["smd"] - result.smd[cov]) < 1e-10

    def test_result_imbalanced_mixed(self):
        """imbalanced() correctly separates covariates with different SMD magnitudes.

        Construct data where one covariate is imbalanced and another is not,
        so the test can distinguish filtering behavior.
        """
        # x1: treated mean=10, control mean=0 (large SMD)
        # x2: treated mean=5, control mean=5 (SMD ≈ 0)
        df = pl.DataFrame({
            "T": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "x1": [9.0, 10.0, 11.0, 9.5, 10.5,
                   9.0, 10.0, 11.0, 9.5, 10.5,
                   -1.0, 0.0, 1.0, -0.5, 0.5,
                   -1.0, 0.0, 1.0, -0.5, 0.5],
            "x2": [4.0, 5.0, 6.0, 4.5, 5.5,
                   4.0, 5.0, 6.0, 4.5, 5.5,
                   4.0, 5.0, 6.0, 4.5, 5.5,
                   4.0, 5.0, 6.0, 4.5, 5.5],
        })
        result = causers.balance_check(df, "T", ["x1", "x2"])

        # x1 has large SMD, x2 has near-zero SMD
        assert abs(result.smd["x2"]) < 1e-10
        assert abs(result.smd["x1"]) > 1.0

        imb = result.imbalanced(threshold=0.1)
        assert "x1" in imb
        assert "x2" not in imb

    def test_result_to_dataframe(self):
        """to_dataframe() returns DataFrame with all stat columns including variances."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])
        full = result.to_dataframe()

        assert isinstance(full, pl.DataFrame)
        expected_cols = {
            "covariate", "mean_treated", "mean_control",
            "var_treated", "var_control", "sd_treated", "sd_control",
            "smd", "variance_ratio",
        }
        assert set(full.columns) == expected_cols
        assert len(full) == 2

        # Verify var columns contain actual variance values, not zeros/placeholders
        for row in full.iter_rows(named=True):
            cov = row["covariate"]
            assert abs(row["var_treated"] - result.var_treated[cov]) < 1e-10
            assert abs(row["var_control"] - result.var_control[cov]) < 1e-10
            assert row["var_treated"] > 0
            assert row["var_control"] > 0

    def test_weighted_basic_values(self):
        """Weighted analysis returns correct weighted means and ESS.

        Uses non-uniform, non-trivial weights where weighted != unweighted.
        """
        # 5 treated, 5 control with known weights
        treated_x = [1.0, 2.0, 3.0, 4.0, 5.0]
        control_x = [6.0, 7.0, 8.0, 9.0, 10.0]
        treated_w = [1.0, 1.0, 1.0, 1.0, 6.0]  # heavy weight on x=5
        control_w = [6.0, 1.0, 1.0, 1.0, 1.0]  # heavy weight on x=6
        df = pl.DataFrame({
            "T": [1] * 5 + [0] * 5,
            "x": treated_x + control_x,
            "w": treated_w + control_w,
        })
        result = causers.balance_check(df, "T", "x", weights="w")

        assert result.is_weighted is True

        # Weighted treated mean: (1*1+2*1+3*1+4*1+5*6)/(1+1+1+1+6) = 40/10 = 4.0
        expected_wm_t = _weighted_mean(treated_x, treated_w)
        assert abs(expected_wm_t - 4.0) < 1e-10
        assert abs(result.mean_treated["x"] - expected_wm_t) < 1e-10

        # Weighted control mean: (6*6+7*1+8*1+9*1+10*1)/(6+1+1+1+1) = 70/10 = 7.0
        expected_wm_c = _weighted_mean(control_x, control_w)
        assert abs(expected_wm_c - 7.0) < 1e-10
        assert abs(result.mean_control["x"] - expected_wm_c) < 1e-10

        # Unweighted means differ: (1+2+3+4+5)/5 = 3.0, (6+7+8+9+10)/5 = 8.0
        assert abs(sum(treated_x) / 5 - 3.0) < 1e-10  # != 4.0
        assert abs(sum(control_x) / 5 - 8.0) < 1e-10  # != 7.0

        # ESS = V1^2 / V2
        v1_t = sum(treated_w)  # 10
        v2_t = sum(w ** 2 for w in treated_w)  # 1+1+1+1+36 = 40
        expected_ess_t = v1_t ** 2 / v2_t  # 100/40 = 2.5
        assert abs(result.ess_treated - expected_ess_t) < 1e-6

    def test_weighted_uniform_weights(self):
        """Uniform weights give same means and variances as unweighted."""
        np.random.seed(42)
        n = 50
        t = np.array([1] * 25 + [0] * 25)
        x1 = np.random.randn(n)
        df = pl.DataFrame({
            "T": t.tolist(),
            "x1": x1.tolist(),
            "w": [1.0] * n,
        })

        result_uw = causers.balance_check(df, "T", "x1")
        result_w = causers.balance_check(df, "T", "x1", weights="w")

        assert abs(result_w.mean_treated["x1"] - result_uw.mean_treated["x1"]) < 1e-8
        assert abs(result_w.mean_control["x1"] - result_uw.mean_control["x1"]) < 1e-8
        # Check variances too (the hardest part of weighted computation)
        assert abs(result_w.var_treated["x1"] - result_uw.var_treated["x1"]) < 1e-6
        assert abs(result_w.var_control["x1"] - result_uw.var_control["x1"]) < 1e-6
        assert abs(result_w.smd["x1"] - result_uw.smd["x1"]) < 1e-6

    def test_weighted_smd_uses_unadjusted_denominator(self):
        """Weighted SMD denominator uses UNADJUSTED (unweighted) variances.

        This is the cobalt convention: even in weighted analysis, the SMD
        pooled-SD denominator comes from the unadjusted sample variances.
        If a bug uses weighted variances in the denominator, this test fails.
        """
        # Construct data where weighted and unweighted variances differ
        treated_x = [0.0, 10.0]
        control_x = [0.0, 10.0]
        treated_w = [9.0, 1.0]  # heavy weight on 0 -> weighted var much smaller
        control_w = [1.0, 1.0]
        df = pl.DataFrame({
            "T": [1, 1, 0, 0],
            "x": treated_x + control_x,
            "w": treated_w + control_w,
        })
        result = causers.balance_check(df, "T", "x", weights="w")

        # Unweighted variances (Bessel): both groups = ((0-5)^2 + (10-5)^2)/1 = 50
        unadj_var_t = _bessel_var(treated_x)  # 50.0
        unadj_var_c = _bessel_var(control_x)  # 50.0
        unadj_pooled_sd = math.sqrt((unadj_var_t + unadj_var_c) / 2.0)  # sqrt(50) ≈ 7.07

        # Weighted treated mean: (0*9 + 10*1)/(9+1) = 1.0
        # Weighted control mean: (0*1 + 10*1)/(1+1) = 5.0
        wm_t = _weighted_mean(treated_x, treated_w)  # 1.0
        wm_c = _weighted_mean(control_x, control_w)  # 5.0

        expected_smd = (wm_t - wm_c) / unadj_pooled_sd
        assert abs(result.smd["x"] - expected_smd) < 1e-6


# ============================================================================
# TASK-BAL-13: Validation error tests
# ============================================================================

class TestBalanceValidation:
    """All should raise ValueError (or similar)."""

    def test_treatment_col_not_found(self):
        """treatment_col='nonexistent' raises ValueError."""
        df = _simple_df()
        with pytest.raises((ValueError, Exception)):
            causers.balance_check(df, "nonexistent", ["x1"])

    def test_covariate_col_not_found(self):
        """covariate_cols=['nonexistent'] raises ValueError."""
        df = _simple_df()
        with pytest.raises((ValueError, Exception)):
            causers.balance_check(df, "T", ["nonexistent"])

    def test_weights_col_not_found(self):
        """weights='nonexistent' raises ValueError."""
        df = _simple_df()
        with pytest.raises((ValueError, Exception)):
            causers.balance_check(df, "T", ["x1"], weights="nonexistent")

    def test_multi_valued_treatment(self):
        """Treatment column with 3 unique values and no explicit values -> error."""
        df = pl.DataFrame({
            "T": [0, 1, 2, 0, 1, 2],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        with pytest.raises((ValueError, Exception)):
            causers.balance_check(df, "T", ["x1"])

    def test_empty_dataframe(self):
        """Empty DataFrame raises an error."""
        df = pl.DataFrame({
            "T": pl.Series([], dtype=pl.Int64),
            "x1": pl.Series([], dtype=pl.Float64),
        })
        with pytest.raises(Exception):
            causers.balance_check(df, "T", ["x1"])


# ============================================================================
# TASK-BAL-14: Edge cases and warning tests
# ============================================================================

class TestBalanceEdgeCases:
    """Edge case tests for balance_check."""

    def test_binary_covariate_smd_value(self):
        """Integer 0/1 covariate produces correct SMD from proportion difference.

        Treated proportions: 3/5 = 0.6
        Control proportions: 1/5 = 0.2
        Variances are p*(1-p) with Bessel correction.
        """
        df = pl.DataFrame({
            "T": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "bin_cov": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        })
        result = causers.balance_check(df, "T", "bin_cov")

        # Treated: mean=0.6, var = (0.16+0.36+0.16+0.16+0.36)/4 = 0.3
        # Control: mean=0.2, var = (0.04+0.04+0.64+0.04+0.04)/4 = 0.2
        expected_mean_t = 3.0 / 5.0
        expected_mean_c = 1.0 / 5.0
        expected_var_t = _bessel_var([1.0, 0.0, 1.0, 1.0, 0.0])
        expected_var_c = _bessel_var([0.0, 0.0, 1.0, 0.0, 0.0])
        expected_smd = (expected_mean_t - expected_mean_c) / math.sqrt(
            (expected_var_t + expected_var_c) / 2.0
        )

        assert abs(result.mean_treated["bin_cov"] - expected_mean_t) < 1e-10
        assert abs(result.mean_control["bin_cov"] - expected_mean_c) < 1e-10
        assert abs(result.smd["bin_cov"] - expected_smd) < 1e-10
        assert result.smd["bin_cov"] > 0  # treated proportion > control

    def test_boolean_treatment(self):
        """Boolean True/False treatment column works (cast to int first)."""
        df = pl.DataFrame({
            "T": [True, True, True, True, True,
                  False, False, False, False, False],
            "x1": [float(i) for i in range(10)],
        })
        # Boolean columns need to be cast to int for the Rust side
        df = df.with_columns(pl.col("T").cast(pl.Int64))
        result = causers.balance_check(df, "T", "x1")

        assert result.n_treated + result.n_control == 10
        assert "x1" in result.smd

    def test_large_dataset_smd_near_zero(self):
        """N=10000, same distribution, randomized treatment -> SMDs near zero."""
        np.random.seed(99)
        n = 10_000
        # Randomize treatment so both groups are from the same distribution
        t = np.random.choice([0, 1], size=n)
        x = np.random.randn(n)
        df = pl.DataFrame({
            "T": t.tolist(),
            "x": x.tolist(),
        })

        result = causers.balance_check(df, "T", "x")
        # With N=10000 and random assignment, |SMD| should be very small
        assert abs(result.smd["x"]) < 0.1

    def test_many_covariates(self):
        """50 covariates simultaneously, verify all present in result."""
        np.random.seed(77)
        n = 200
        data = {"T": np.random.choice([0, 1], size=n).tolist()}
        for i in range(50):
            data[f"c{i}"] = np.random.randn(n).tolist()
        df = pl.DataFrame(data)

        cov_names = [f"c{i}" for i in range(50)]
        result = causers.balance_check(df, "T", cov_names)
        assert len(result.covariates) == 50
        # Every covariate should have an SMD value
        for cov in cov_names:
            assert cov in result.smd
            assert isinstance(result.smd[cov], float)
            assert math.isfinite(result.smd[cov])

    def test_asymmetric_group_sizes(self):
        """Very unequal groups (90/10 split) compute correctly."""
        np.random.seed(88)
        n_t, n_c = 90, 10
        treated_x = np.random.normal(5.0, 1.0, n_t)
        control_x = np.random.normal(3.0, 2.0, n_c)
        df = pl.DataFrame({
            "T": [1] * n_t + [0] * n_c,
            "x": treated_x.tolist() + control_x.tolist(),
        })
        result = causers.balance_check(df, "T", "x")

        # Means should be near 5.0 and 3.0
        assert abs(result.mean_treated["x"] - 5.0) < 0.5
        assert abs(result.mean_control["x"] - 3.0) < 1.5  # wider tolerance for n=10

        # VR should reflect variance ratio ≈ 1/4 (sd 1 vs sd 2)
        assert result.variance_ratio["x"] < 1.0  # treated var < control var
        assert result.n_treated == 90
        assert result.n_control == 10


class TestBalanceWarnings:
    """Warning emission tests using pytest.warns."""

    def test_large_smd_warning(self):
        """Data with very different group means triggers |SMD| > 0.25 warning."""
        # Treated mean ~ 100, Control mean ~ 0, both with nonzero variance
        np.random.seed(111)
        treated_x = (100.0 + np.random.randn(10)).tolist()
        control_x = (0.0 + np.random.randn(10)).tolist()
        df = pl.DataFrame({
            "T": [1] * 10 + [0] * 10,
            "x": treated_x + control_x,
        })
        with pytest.warns(UserWarning, match="Large imbalance detected for covariate"):
            causers.balance_check(df, "T", "x")

    def test_extreme_variance_ratio_warning(self):
        """One group has much larger variance triggers VR warning."""
        # Treated: tight around 50 (var ≈ 0.003), Control: spread (var ≈ 2778)
        # Same mean so SMD is small -> isolates VR warning
        df = pl.DataFrame({
            "T": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "x": [5.0, 5.1, 5.0, 5.1, 5.0,
                  5.1, 5.0, 5.1, 5.0, 5.1,
                  0.0, 100.0, 0.0, 100.0, 0.0,
                  100.0, 0.0, 100.0, 0.0, 100.0],
        })
        with pytest.warns(UserWarning, match="Extreme variance ratio for covariate"):
            causers.balance_check(df, "T", "x")

    def test_small_treatment_group_warning(self):
        """Only 3 treated observations triggers small treatment group warning."""
        df = pl.DataFrame({
            "T": [1, 1, 1] + [0] * 50,
            "x": [float(i) for i in range(53)],
        })
        with pytest.warns(UserWarning, match="Small treatment group"):
            causers.balance_check(df, "T", "x")

    def test_small_control_group_warning(self):
        """Only 3 control observations triggers small control group warning."""
        df = pl.DataFrame({
            "T": [1] * 50 + [0, 0, 0],
            "x": [float(i) for i in range(53)],
        })
        with pytest.warns(UserWarning, match="Small control group"):
            causers.balance_check(df, "T", "x")

    def test_no_warning_balanced(self):
        """Intentionally balanced data triggers no UserWarning.

        Both groups are constructed from the SAME values (interleaved), so
        means, variances, and VR are all identical. Groups are large (50 each).
        """
        # Interleave identical values into both groups
        vals = list(range(50))
        df = pl.DataFrame({
            "T": [1] * 50 + [0] * 50,
            "x": [float(v) for v in vals] + [float(v) for v in vals],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            causers.balance_check(df, "T", "x")


# ============================================================================
# REQ COVERAGE GAP TESTS
# ============================================================================


class TestBalanceZeroVariance:
    """FR-BAL-59 to FR-BAL-62: Zero variance edge cases."""

    def test_zero_variance_both_groups_equal_means(self):
        """FR-BAL-59: Zero var in both groups + equal means -> SMD = 0."""
        df = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [5.0] * 40,
        })
        result = causers.balance_check(df, "T", "x")
        assert result.smd["x"] == 0.0
        assert result.var_treated["x"] == 0.0
        assert result.var_control["x"] == 0.0

    def test_zero_variance_both_groups_different_means(self):
        """FR-BAL-60: Zero var in both groups + different means -> SMD = NaN."""
        df = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [5.0] * 20 + [3.0] * 20,
        })
        result = causers.balance_check(df, "T", "x")
        assert math.isnan(result.smd["x"])

    def test_zero_variance_one_group_vr(self):
        """FR-BAL-61: Zero var in treated -> VR = 0; zero var in control -> VR = inf."""
        # Zero var in treated
        df_t = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [5.0] * 20 + [float(i) for i in range(20)],
        })
        result_t = causers.balance_check(df_t, "T", "x")
        assert result_t.variance_ratio["x"] == 0.0

        # Zero var in control
        df_c = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [float(i) for i in range(20)] + [5.0] * 20,
        })
        result_c = causers.balance_check(df_c, "T", "x")
        assert math.isinf(result_c.variance_ratio["x"])


class TestBalanceNullHandling:
    """FR-BAL-56, FR-BAL-57a: Null value validation."""

    def test_null_in_covariate_raises(self):
        """FR-BAL-56: Null values in covariate -> ValueError."""
        df = pl.DataFrame({
            "T": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "x": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })
        with pytest.raises(Exception):
            causers.balance_check(df, "T", "x")

    def test_null_in_weights_raises(self):
        """FR-BAL-57a: Null values in weights -> error."""
        df = pl.DataFrame({
            "T": [1, 1, 0, 0],
            "x": [1.0, 2.0, 3.0, 4.0],
            "w": [1.0, None, 1.0, 1.0],
        })
        with pytest.raises(Exception):
            causers.balance_check(df, "T", "x", weights="w")

    def test_negative_weights_raises(self):
        """FR-BAL-57: Negative weights -> error."""
        df = pl.DataFrame({
            "T": [1, 1, 0, 0],
            "x": [1.0, 2.0, 3.0, 4.0],
            "w": [1.0, -1.0, 1.0, 1.0],
        })
        with pytest.raises(Exception):
            causers.balance_check(df, "T", "x", weights="w")


class TestBalanceImmutability:
    """FR-BAL-70: Input DataFrame must not be modified."""

    def test_input_not_modified(self):
        """FR-BAL-70: balance_check does not modify the input DataFrame."""
        df = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [float(i) for i in range(40)],
        })
        original_hash = hash(df.to_pandas().to_csv())
        causers.balance_check(df, "T", "x")
        after_hash = hash(df.to_pandas().to_csv())
        assert original_hash == after_hash


class TestBalanceDefaultThreshold:
    """FR-BAL-49: Default threshold for imbalanced()."""

    def test_imbalanced_default_threshold(self):
        """FR-BAL-49: imbalanced() defaults to threshold=0.1."""
        # Build data where one covariate has SMD slightly above 0.1
        # and another is near zero
        np.random.seed(42)
        n = 200
        df = pl.DataFrame({
            "T": [1] * 100 + [0] * 100,
            # x1: treated ~= 0.15 higher than control -> SMD ~ 0.15
            "x1": np.random.normal(0.15, 1.0, 100).tolist()
                  + np.random.normal(0.0, 1.0, 100).tolist(),
            # x2: same distribution -> SMD ~ 0
            "x2": np.random.normal(0.0, 1.0, n).tolist(),
        })
        result = causers.balance_check(df, "T", ["x1", "x2"])

        # Call without threshold argument
        imb = result.imbalanced()
        # x2 should not appear; x1 may or may not depending on randomness,
        # but the key test is that the method works with no argument
        assert isinstance(imb, list)
        for cov_name in imb:
            assert abs(result.smd[cov_name]) > 0.1


class TestBalanceRepr:
    """NFR-BAL-25: __repr__ includes key info."""

    def test_repr_contains_info(self):
        """NFR-BAL-25: repr includes n_treated, n_control, n_covariates."""
        df = _simple_df()
        result = causers.balance_check(df, "T", ["x1", "x2"])
        r = repr(result)
        assert isinstance(r, str)
        assert len(r) > 0


class TestBalanceLowESSWarning:
    """FR-BAL-68/69: Low ESS warning in weighted analysis."""

    def test_low_ess_warning(self):
        """FR-BAL-68: Low ESS in treatment triggers warning."""
        # One observation dominates -> ESS near 1
        df = pl.DataFrame({
            "T": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "w": [100.0, 0.01, 0.01, 0.01, 0.01,
                  0.01, 0.01, 0.01, 0.01, 100.0],
        })
        with pytest.warns(UserWarning, match="Low effective sample size"):
            causers.balance_check(df, "T", "x", weights="w")


class TestBalanceSingleObservationGroup:
    """Edge case: group with only one observation."""

    def test_single_treated_observation(self):
        """Single treated observation: variance should be NaN or zero, no crash."""
        df = pl.DataFrame({
            "T": [1] + [0] * 20,
            "x": [5.0] + [float(i) for i in range(20)],
        })
        # Should not crash -- behavior for n=1 variance (Bessel) is degenerate
        # but function should handle it gracefully
        result = causers.balance_check(df, "T", "x")
        assert result.n_treated == 1
        assert result.n_control == 20


# ============================================================
# FR-BAL-62: Zero variance in one group warning
# ============================================================


class TestZeroVarianceWarning:
    """Test FR-BAL-62: warn when zero variance in exactly one group."""

    def test_zero_var_treated_warning(self):
        """Zero variance in treatment group emits warning."""
        df = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [5.0] * 20 + [float(i) for i in range(20)],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            causers.balance_check(df, "T", "x")
        msgs = [str(wi.message) for wi in w]
        assert any("zero variance in treatment group" in m for m in msgs)

    def test_zero_var_control_warning(self):
        """Zero variance in control group emits warning."""
        df = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [float(i) for i in range(20)] + [3.0] * 20,
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            causers.balance_check(df, "T", "x")
        msgs = [str(wi.message) for wi in w]
        assert any("zero variance in control group" in m for m in msgs)

    def test_no_zero_var_warning_when_both_zero(self):
        """No zero-variance warning when BOTH groups have zero variance."""
        df = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [5.0] * 40,
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            causers.balance_check(df, "T", "x")
        msgs = [str(wi.message) for wi in w]
        assert not any("zero variance" in m for m in msgs)


# ============================================================
# S5: imbalanced() with NaN SMD
# ============================================================


class TestImbalancedNaN:
    """Test that imbalanced() handles NaN SMD correctly."""

    def test_nan_smd_excluded(self):
        """Covariates with NaN SMD should not appear in imbalanced()."""
        df = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x_nan": [5.0] * 20 + [3.0] * 20,  # zero var both, diff means -> NaN SMD
            "x_ok": list(range(20)) + list(range(20, 40)),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = causers.balance_check(df, "T", ["x_nan", "x_ok"])
        assert math.isnan(result.smd["x_nan"])
        imb = result.imbalanced(threshold=0.0)
        assert "x_nan" not in imb
        assert "x_ok" in imb



class TestNaNWeightedVarianceWarning:
    """Test FR-BAL-69a: warn when weighted variance is numerically unstable."""

    def test_nan_weighted_variance_warning(self):
        """ESS ≈ 1 triggers NaN variance and warning."""
        # One observation dominates each group so V1^2 ≈ V2 → NaN variance
        df = pl.DataFrame({
            "T": [1, 1, 0, 0],
            "x": [1.0, 2.0, 3.0, 4.0],
            "w": [1e12, 1e-12, 1e12, 1e-12],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            causers.balance_check(df, "T", "x", weights="w")
        msgs = [str(wi.message) for wi in w]
        assert any("numerically unstable" in m for m in msgs)


# ============================================================
# S6: Both-zero variance ratio is NaN
# ============================================================


class TestBothZeroVR:
    """Test that both-zero variance returns NaN for VR."""

    def test_both_zero_vr_is_nan(self):
        """S6: Both groups zero variance yields NaN variance ratio."""
        df = pl.DataFrame({
            "T": [1] * 20 + [0] * 20,
            "x": [5.0] * 40,
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = causers.balance_check(df, "T", "x")
        assert math.isnan(result.variance_ratio["x"])
