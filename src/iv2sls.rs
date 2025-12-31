//! Two-Stage Least Squares (2SLS) Instrumental Variables Estimator.
//!
//! This module provides a high-performance implementation of the 2SLS estimator
//! for addressing endogeneity problems where treatment variables are correlated
//! with error terms.
//!
//! # Algorithm
//!
//! The 2SLS estimator proceeds in two stages:
//!
//! **First Stage**: Regress each endogenous variable on instruments and exogenous controls:
//! ```text
//! D = Z×π + X×δ + ν
//! D̂ = Z×π̂ + X×δ̂
//! ```
//!
//! **Second Stage**: Regress the outcome on predicted endogenous values and controls:
//! ```text
//! Y = D̂×β + X×γ + ε
//! ```
//!
//! **CRITICAL**: Standard errors are computed using residuals from the **original** D,
//! not the predicted D̂. Using D̂ would understate variance.
//!
//! # References
//!
//! - Angrist, J. D., & Pischke, J. S. (2009). Mostly Harmless Econometrics.
//! - Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression.

use std::error::Error;
use std::fmt;

use faer::Mat;
use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::cluster::{build_cluster_indices, compute_cluster_se_analytical_faer, ClusterError};
use crate::linalg;

// ============================================================================
// Error Types
// ============================================================================

/// Error types for 2SLS operations.
#[derive(Debug, Clone)]
pub enum IV2SLSError {
    /// Under-identified: m < k₁
    UnderIdentified {
        n_instruments: usize,
        n_endogenous: usize,
    },
    /// Instruments too weak: F < 4
    InstrumentsTooWeak { endogenous_var: String, f_stat: f64 },
    /// First-stage design matrix singular
    FirstStageSingular,
    /// Second-stage design matrix singular
    SecondStageSingular,
    /// Numerical instability (condition number > 10^10)
    NumericalInstability { message: String },
    /// Column contains null values
    NullValues { column: String },
    /// Zero variance in variable
    ZeroVariance { column: String, var_type: String },
    /// Instruments collinear with exogenous controls
    CollinearInstruments,
    /// Insufficient observations
    InsufficientObservations { n_samples: usize, n_params: usize },
}

impl fmt::Display for IV2SLSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IV2SLSError::UnderIdentified {
                n_instruments,
                n_endogenous,
            } => {
                write!(
                    f,
                    "Number of instruments ({}) must be ≥ number of endogenous variables ({})",
                    n_instruments, n_endogenous
                )
            }
            IV2SLSError::InstrumentsTooWeak {
                endogenous_var,
                f_stat,
            } => {
                write!(
                    f,
                    "Instruments too weak for reliable inference (F = {:.2} < 4) for endogenous variable '{}'",
                    f_stat, endogenous_var
                )
            }
            IV2SLSError::FirstStageSingular => {
                write!(
                    f,
                    "First stage design matrix is singular; check for collinear instruments"
                )
            }
            IV2SLSError::SecondStageSingular => {
                write!(f, "Second stage design matrix is singular")
            }
            IV2SLSError::NumericalInstability { message } => {
                write!(f, "Numerical instability detected: {}", message)
            }
            IV2SLSError::NullValues { column } => {
                write!(f, "Column '{}' contains null values", column)
            }
            IV2SLSError::ZeroVariance { column, var_type } => {
                write!(f, "{} variable '{}' has zero variance", var_type, column)
            }
            IV2SLSError::CollinearInstruments => {
                write!(f, "Instruments are collinear with exogenous controls")
            }
            IV2SLSError::InsufficientObservations {
                n_samples: _,
                n_params,
            } => {
                write!(
                    f,
                    "Need at least {} observations for {} parameters",
                    n_params + 1,
                    n_params
                )
            }
        }
    }
}

impl Error for IV2SLSError {}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for 2SLS estimation.
pub struct IV2SLSConfig<'a> {
    /// Whether to include an intercept term in both stages.
    pub include_intercept: bool,
    /// If True, compute HC3 heteroskedasticity-robust standard errors.
    pub robust: bool,
    /// Optional cluster identifiers for cluster-robust standard errors.
    pub cluster_ids: Option<&'a [i64]>,
}

impl Default for IV2SLSConfig<'_> {
    fn default() -> Self {
        Self {
            include_intercept: true,
            robust: false,
            cluster_ids: None,
        }
    }
}

// ============================================================================
// Result Struct (PyO3)
// ============================================================================

/// Result of Two-Stage Least Squares estimation.
#[pyclass]
#[derive(Debug, Clone)]
pub struct TwoStageLSResult {
    /// Structural coefficients (endogenous + exogenous, excluding intercept)
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    /// Standard errors for all coefficients
    #[pyo3(get)]
    pub standard_errors: Vec<f64>,
    /// Intercept term if included
    #[pyo3(get)]
    pub intercept: Option<f64>,
    /// Standard error for intercept
    #[pyo3(get)]
    pub intercept_se: Option<f64>,
    /// Number of observations
    #[pyo3(get)]
    pub n_samples: usize,
    /// Number of endogenous regressors
    #[pyo3(get)]
    pub n_endogenous: usize,
    /// Number of excluded instruments
    #[pyo3(get)]
    pub n_instruments: usize,
    /// F-statistics per endogenous variable
    #[pyo3(get)]
    pub first_stage_f: Vec<f64>,
    /// First-stage coefficients for instruments only (per endogenous variable)
    #[pyo3(get)]
    pub first_stage_coefficients: Vec<Vec<f64>>,
    /// Cragg-Donald statistic (multiple endogenous only)
    #[pyo3(get)]
    pub cragg_donald: Option<f64>,
    /// Stock-Yogo 10% critical value
    #[pyo3(get)]
    pub stock_yogo_critical: Option<f64>,
    /// R² from structural equation (using original D)
    #[pyo3(get)]
    pub r_squared: f64,
    /// SE type: "conventional", "hc3", or "clustered"
    #[pyo3(get)]
    pub se_type: String,
    /// Number of clusters (if clustered)
    #[pyo3(get)]
    pub n_clusters: Option<usize>,
}

#[pymethods]
impl TwoStageLSResult {
    fn __repr__(&self) -> String {
        format!(
            "TwoStageLSResult(coefficients={:?}, se={:?}, first_stage_f={:?}, \
             n_samples={}, n_endogenous={}, n_instruments={}, se_type=\"{}\")",
            self.coefficients,
            self.standard_errors,
            self.first_stage_f,
            self.n_samples,
            self.n_endogenous,
            self.n_instruments,
            self.se_type
        )
    }

    fn __str__(&self) -> String {
        let mut s = String::from("Two-Stage Least Squares Results\n");
        s.push_str(&format!("Coefficients: {:?}\n", self.coefficients));
        s.push_str(&format!("Std Errors: {:?}\n", self.standard_errors));
        s.push_str(&format!("First-stage F: {:?}\n", self.first_stage_f));
        if let Some(cd) = self.cragg_donald {
            s.push_str(&format!("Cragg-Donald: {:.2}\n", cd));
        }
        s.push_str(&format!(
            "N={}, SE type: {}\n",
            self.n_samples, self.se_type
        ));
        s
    }

    /// Compute t-statistics for all coefficients.
    ///
    /// T-statistics are computed as coefficients / standard_errors (element-wise).
    /// Returns t-statistics for slope coefficients only (intercept excluded).
    ///
    /// # Returns
    /// Vec of t-statistics, one for each coefficient.
    #[getter]
    fn t_statistics(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .zip(self.standard_errors.iter())
            .map(|(coef, se)| {
                if *se > 0.0 {
                    coef / se
                } else if *coef == 0.0 {
                    0.0
                } else {
                    f64::INFINITY * coef.signum()
                }
            })
            .collect()
    }

    /// Compute two-tailed p-values for all coefficients.
    ///
    /// P-values are computed using the t-distribution with (n_samples - n_params)
    /// degrees of freedom. Tests H₀: β = 0 against H₁: β ≠ 0.
    ///
    /// # Returns
    /// Vec of p-values, one for each coefficient.
    #[getter]
    fn p_values(&self) -> Vec<f64> {
        let df = self.degrees_of_freedom();
        let t_stats = self.t_statistics();
        
        // Create t-distribution with df degrees of freedom
        // If df <= 0, fall back to large sample (normal approximation)
        if df <= 0.0 {
            // Use normal approximation for edge cases
            return t_stats
                .iter()
                .map(|t| {
                    let abs_t = t.abs();
                    if abs_t.is_infinite() {
                        0.0
                    } else {
                        2.0 * (1.0 - normal_cdf_approx(abs_t))
                    }
                })
                .collect();
        }

        // Use t-distribution
        match StudentsT::new(0.0, 1.0, df) {
            Ok(t_dist) => t_stats
                .iter()
                .map(|t| {
                    let abs_t = t.abs();
                    if abs_t.is_infinite() {
                        0.0
                    } else {
                        2.0 * (1.0 - t_dist.cdf(abs_t))
                    }
                })
                .collect(),
            Err(_) => {
                // Fallback to normal approximation
                t_stats
                    .iter()
                    .map(|t| {
                        let abs_t = t.abs();
                        if abs_t.is_infinite() {
                            0.0
                        } else {
                            2.0 * (1.0 - normal_cdf_approx(abs_t))
                        }
                    })
                    .collect()
            }
        }
    }

    /// Compute confidence intervals for all coefficients.
    ///
    /// Confidence intervals are computed as: coef ± t_crit × se
    /// where t_crit is the critical value from the t-distribution.
    ///
    /// # Arguments
    /// * `alpha` - Significance level (default: 0.05 for 95% CI).
    ///             Must be in (0, 1).
    ///
    /// # Returns
    /// List of (lower, upper) bounds for each coefficient.
    #[pyo3(signature = (alpha=0.05))]
    fn confidence_intervals(&self, alpha: f64) -> PyResult<Vec<(f64, f64)>> {
        // Validate alpha
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("alpha must be in (0, 1); got {}", alpha)
            ));
        }

        let df = self.degrees_of_freedom();
        let t_crit = self.compute_t_critical(alpha, df);

        let intervals: Vec<(f64, f64)> = self
            .coefficients
            .iter()
            .zip(self.standard_errors.iter())
            .map(|(coef, se)| {
                let margin = t_crit * se;
                (coef - margin, coef + margin)
            })
            .collect();

        Ok(intervals)
    }

    /// Return a formatted summary table similar to statsmodels IV2SLS output.
    ///
    /// The summary includes:
    /// - Model information (N, R², SE type)
    /// - First-stage diagnostics (F-statistics, Cragg-Donald)
    /// - Coefficient table with estimates, SEs, t-stats, p-values, and CIs
    ///
    /// # Returns
    /// Formatted string suitable for printing.
    fn summary(&self) -> String {
        let t_stats = self.t_statistics();
        let p_values = self.p_values();
        let cis = self.confidence_intervals(0.05).unwrap_or_default();
        let df = self.degrees_of_freedom();

        let mut s = String::new();
        
        // Header
        s.push_str("\n");
        s.push_str("                          IV-2SLS Regression Results                           \n");
        s.push_str("==============================================================================\n");
        
        // Model info
        s.push_str(&format!("Dep. Variable:           Y          No. Observations:       {:>10}\n", self.n_samples));
        s.push_str(&format!("Model:                   IV-2SLS    Df Residuals:           {:>10.0}\n", df));
        s.push_str(&format!("R-squared:               {:.4}      Df Model:               {:>10}\n",
            self.r_squared, self.coefficients.len()));
        s.push_str(&format!("SE Type:                 {:10}\n", self.se_type));
        
        if let Some(n_clust) = self.n_clusters {
            s.push_str(&format!("No. Clusters:            {:>10}\n", n_clust));
        }
        
        s.push_str("------------------------------------------------------------------------------\n");
        
        // First-stage diagnostics
        s.push_str("First-Stage Diagnostics:\n");
        for (i, f_stat) in self.first_stage_f.iter().enumerate() {
            let weak_indicator = if *f_stat < 10.0 { " [WEAK]" } else { "" };
            s.push_str(&format!("  Endogenous {}: F-statistic = {:.2}{}\n",
                i + 1, f_stat, weak_indicator));
        }
        
        if let Some(cd) = self.cragg_donald {
            let sy_info = match self.stock_yogo_critical {
                Some(critical) if cd < critical => format!(" < {:.2} (10% bias)", critical),
                Some(critical) => format!(" >= {:.2} (10% bias)", critical),
                None => String::new(),
            };
            s.push_str(&format!("  Cragg-Donald Statistic: {:.2}{}\n", cd, sy_info));
        }
        
        s.push_str("------------------------------------------------------------------------------\n");
        
        // Coefficient table header
        s.push_str(&format!(
            "{:>12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>12}\n",
            "", "coef", "std err", "t", "P>|t|", "[0.025", "0.975]"
        ));
        s.push_str("------------------------------------------------------------------------------\n");
        
        // Intercept (if present)
        if let (Some(intercept), Some(intercept_se)) = (self.intercept, self.intercept_se) {
            let int_t = if intercept_se > 0.0 { intercept / intercept_se } else { 0.0 };
            let int_p = self.compute_p_value_single(int_t, df);
            let t_crit = self.compute_t_critical(0.05, df);
            let int_ci_low = intercept - t_crit * intercept_se;
            let int_ci_high = intercept + t_crit * intercept_se;
            
            s.push_str(&format!(
                "{:>12} {:>10.4} {:>10.4} {:>8.3} {:>8.3} {:>12.3} {:>12.3}\n",
                "const", intercept, intercept_se, int_t, int_p, int_ci_low, int_ci_high
            ));
        }
        
        // Coefficient rows
        for (i, ((((coef, se), t), p), ci)) in self
            .coefficients
            .iter()
            .zip(self.standard_errors.iter())
            .zip(t_stats.iter())
            .zip(p_values.iter())
            .zip(cis.iter())
            .enumerate()
        {
            let var_name = if i < self.n_endogenous {
                format!("x{}", i + 1)
            } else {
                format!("z{}", i + 1 - self.n_endogenous)
            };
            
            s.push_str(&format!(
                "{:>12} {:>10.4} {:>10.4} {:>8.3} {:>8.3} {:>12.3} {:>12.3}\n",
                var_name, coef, se, t, p, ci.0, ci.1
            ));
        }
        
        s.push_str("==============================================================================\n");
        
        // Significance codes
        let stars: Vec<&str> = p_values
            .iter()
            .map(|&p| {
                if p < 0.001 { "***" }
                else if p < 0.01 { "**" }
                else if p < 0.05 { "*" }
                else if p < 0.1 { "." }
                else { "" }
            })
            .collect();
        
        if stars.iter().any(|s| !s.is_empty()) {
            s.push_str("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n");
        }
        
        // Warnings
        if self.first_stage_f.iter().any(|&f| f < 10.0) {
            s.push_str("\nWarning: Weak instruments detected (first-stage F < 10).\n");
        }
        
        if let (Some(cd), Some(critical)) = (self.cragg_donald, self.stock_yogo_critical) {
            if cd < critical {
                s.push_str("\nWarning: Cragg-Donald statistic below Stock-Yogo critical value.\n");
            }
        }
        
        s
    }
}

impl TwoStageLSResult {
    /// Compute degrees of freedom: n - k where k is total number of parameters.
    fn degrees_of_freedom(&self) -> f64 {
        let n_params = self.coefficients.len() + if self.intercept.is_some() { 1 } else { 0 };
        (self.n_samples as isize - n_params as isize).max(1) as f64
    }

    /// Compute t-distribution critical value for given alpha and df.
    fn compute_t_critical(&self, alpha: f64, df: f64) -> f64 {
        if df <= 0.0 {
            // Fallback to normal approximation z_0.975 ≈ 1.96
            return normal_quantile_approx(1.0 - alpha / 2.0);
        }

        match StudentsT::new(0.0, 1.0, df) {
            Ok(t_dist) => {
                // Use inverse CDF to get critical value via binary search
                let target = 1.0 - alpha / 2.0;
                let mut low = 0.0;
                let mut high = 10.0;
                
                // Expand high if needed
                while t_dist.cdf(high) < target && high < 1000.0 {
                    high *= 2.0;
                }
                
                // Binary search
                for _ in 0..100 {
                    let mid = (low + high) / 2.0;
                    if t_dist.cdf(mid) < target {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                
                (low + high) / 2.0
            }
            Err(_) => normal_quantile_approx(1.0 - alpha / 2.0),
        }
    }

    /// Compute p-value for a single t-statistic.
    fn compute_p_value_single(&self, t_stat: f64, df: f64) -> f64 {
        if df <= 0.0 {
            let abs_t = t_stat.abs();
            return if abs_t.is_infinite() { 0.0 } else { 2.0 * (1.0 - normal_cdf_approx(abs_t)) };
        }

        match StudentsT::new(0.0, 1.0, df) {
            Ok(t_dist) => {
                let abs_t = t_stat.abs();
                if abs_t.is_infinite() { 0.0 } else { 2.0 * (1.0 - t_dist.cdf(abs_t)) }
            }
            Err(_) => {
                let abs_t = t_stat.abs();
                if abs_t.is_infinite() { 0.0 } else { 2.0 * (1.0 - normal_cdf_approx(abs_t)) }
            }
        }
    }
}

/// Standard normal CDF approximation using Abramowitz and Stegun (7.1.26).
fn normal_cdf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d1 = 0.319381530;
    let d2 = -0.356563782;
    let d3 = 1.781477937;
    let d4 = -1.821255978;
    let d5 = 1.330274429;

    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * t * (d1 + t * (d2 + t * (d3 + t * (d4 + t * d5))));

    if x >= 0.0 { cdf } else { 1.0 - cdf }
}

/// Standard normal quantile (inverse CDF) approximation using Abramowitz and Stegun (26.2.23).
fn normal_quantile_approx(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -z } else { z }
}

// ============================================================================
// Internal Result Structs
// ============================================================================

/// Internal result from first-stage regression(s).
struct FirstStageResult {
    /// Predicted endogenous values D̂ (n × k₁) in row-major
    d_hat_flat: Vec<f64>,
    /// First-stage coefficients for instruments only per endogenous variable
    instrument_coefficients: Vec<Vec<f64>>,
    /// F-statistics for each endogenous variable
    f_statistics: Vec<f64>,
}

// ============================================================================
// Stock-Yogo Critical Values
// ============================================================================

/// Stock-Yogo critical values for 10% maximal bias.
///
/// Reference: Stock & Yogo (2005), Table 5.2
fn stock_yogo_lookup(n_endog: usize, n_instruments: usize) -> Option<f64> {
    match (n_endog, n_instruments) {
        // Single endogenous variable
        (1, 1) => Some(16.38),
        (1, 2) => Some(19.93),
        (1, 3) => Some(22.30),
        (1, 4) => Some(24.58),
        (1, 5) => Some(26.87),
        (1, 6) => Some(29.18),
        (1, 7) => Some(31.50),
        (1, 8) => Some(33.84),
        (1, 9) => Some(36.19),
        (1, 10) => Some(38.54),
        // Two endogenous variables
        (2, 2) => Some(7.03),
        (2, 3) => Some(13.43),
        (2, 4) => Some(17.88),
        (2, 5) => Some(21.68),
        (2, 6) => Some(25.14),
        (2, 7) => Some(28.40),
        (2, 8) => Some(31.50),
        (2, 9) => Some(34.50),
        (2, 10) => Some(37.42),
        // Not supported configuration
        _ => None,
    }
}

// ============================================================================
// Matrix Building Functions
// ============================================================================

/// Build first-stage design matrix W = [1 | Z | X] or [Z | X].
///
/// The matrix layout depends on `include_intercept`:
/// - If true: W = [intercept | instruments | exogenous]
/// - If false: W = [instruments | exogenous]
fn build_first_stage_matrix(
    z_flat: &[f64],
    x_flat: &[f64],
    n_rows: usize,
    n_instruments: usize,
    n_exog: usize,
    include_intercept: bool,
) -> Mat<f64> {
    let n_cols_total = n_instruments + n_exog + if include_intercept { 1 } else { 0 };

    Mat::from_fn(n_rows, n_cols_total, |i, j| {
        if include_intercept {
            if j == 0 {
                1.0 // Intercept column
            } else if j <= n_instruments {
                // Instruments: columns 1 to n_instruments
                z_flat[i * n_instruments + (j - 1)]
            } else if n_exog > 0 {
                // Exogenous: columns n_instruments+1 to end
                x_flat[i * n_exog + (j - 1 - n_instruments)]
            } else {
                0.0
            }
        } else if j < n_instruments {
            // Instruments: columns 0 to n_instruments-1
            z_flat[i * n_instruments + j]
        } else if n_exog > 0 {
            // Exogenous: columns n_instruments to end
            x_flat[i * n_exog + (j - n_instruments)]
        } else {
            0.0
        }
    })
}

/// Build second-stage design matrix V = [1 | D̂ | X] or [D̂ | X].
///
/// Uses predicted endogenous values D̂ from first stage.
fn build_second_stage_matrix(
    d_hat_flat: &[f64],
    x_flat: &[f64],
    n_rows: usize,
    n_endog: usize,
    n_exog: usize,
    include_intercept: bool,
) -> Mat<f64> {
    let n_cols_total = n_endog + n_exog + if include_intercept { 1 } else { 0 };

    Mat::from_fn(n_rows, n_cols_total, |i, j| {
        if include_intercept {
            if j == 0 {
                1.0 // Intercept column
            } else if j <= n_endog {
                // Predicted endogenous: columns 1 to n_endog
                d_hat_flat[i * n_endog + (j - 1)]
            } else if n_exog > 0 {
                // Exogenous: columns n_endog+1 to end
                x_flat[i * n_exog + (j - 1 - n_endog)]
            } else {
                0.0
            }
        } else if j < n_endog {
            // Predicted endogenous: columns 0 to n_endog-1
            d_hat_flat[i * n_endog + j]
        } else if n_exog > 0 {
            // Exogenous: columns n_endog to end
            x_flat[i * n_exog + (j - n_endog)]
        } else {
            0.0
        }
    })
}

/// Build original design matrix U = [1 | D | X] or [D | X].
///
/// **CRITICAL**: This uses ORIGINAL D, not predicted D̂.
/// This is essential for correct residual computation and standard errors.
fn build_original_design_matrix(
    d_flat: &[f64],
    x_flat: &[f64],
    n_rows: usize,
    n_endog: usize,
    n_exog: usize,
    include_intercept: bool,
) -> Mat<f64> {
    // Same structure as second-stage but with original D
    let n_cols_total = n_endog + n_exog + if include_intercept { 1 } else { 0 };

    Mat::from_fn(n_rows, n_cols_total, |i, j| {
        if include_intercept {
            if j == 0 {
                1.0 // Intercept column
            } else if j <= n_endog {
                // Original endogenous: columns 1 to n_endog
                d_flat[i * n_endog + (j - 1)]
            } else if n_exog > 0 {
                // Exogenous: columns n_endog+1 to end
                x_flat[i * n_exog + (j - 1 - n_endog)]
            } else {
                0.0
            }
        } else if j < n_endog {
            // Original endogenous: columns 0 to n_endog-1
            d_flat[i * n_endog + j]
        } else if n_exog > 0 {
            // Exogenous: columns n_endog to end
            x_flat[i * n_exog + (j - n_endog)]
        } else {
            0.0
        }
    })
}

// ============================================================================
// First-Stage Regression
// ============================================================================

/// Perform first-stage regression for all endogenous variables.
///
/// For each endogenous variable D_j, regress on instruments Z and controls X:
/// D_j = Z×π + X×δ + ν
///
/// Returns predicted values D̂, instrument coefficients, and F-statistics.
#[allow(clippy::too_many_arguments)]
fn first_stage_regression(
    w_mat: &Mat<f64>,
    d_flat: &[f64],
    x_flat: &[f64],
    n_rows: usize,
    n_endog: usize,
    n_exog: usize,
    n_instruments: usize,
    include_intercept: bool,
    endog_names: &[String],
) -> Result<FirstStageResult, IV2SLSError> {
    let mut d_hat_flat = vec![0.0; n_rows * n_endog];
    let mut instrument_coefficients = Vec::with_capacity(n_endog);
    let mut f_statistics = Vec::with_capacity(n_endog);

    // Compute W'W and its inverse once (shared across all endogenous variables)
    let wtw = linalg::xtx(w_mat);
    let _wtw_inv = linalg::invert_xtx(&wtw).map_err(|_| IV2SLSError::FirstStageSingular)?;

    // For F-stat computation: need restricted model (X only, no Z)
    let x_only_mat = if n_exog > 0 || include_intercept {
        Some(linalg::flat_to_mat_with_intercept(
            x_flat,
            n_rows,
            n_exog,
            include_intercept,
        ))
    } else {
        None
    };

    let (xtx_restricted, xtx_inv_restricted) = if let Some(ref x_mat) = x_only_mat {
        let xtx_r = linalg::xtx(x_mat);
        let xtx_inv_r = linalg::invert_xtx(&xtx_r).ok();
        (Some(xtx_r), xtx_inv_r)
    } else {
        (None, None)
    };

    // First stage for each endogenous variable
    for j in 0..n_endog {
        // Extract column j of D
        let d_j: Vec<f64> = (0..n_rows).map(|i| d_flat[i * n_endog + j]).collect();

        // Compute W'D_j
        let wt_dj = linalg::xty(w_mat, &d_j);

        // Solve for first-stage coefficients: π̂ = (W'W)^-1 W'D_j
        let pi_hat = linalg::solve_normal_equations(&wtw, &wt_dj)
            .map_err(|_| IV2SLSError::FirstStageSingular)?;

        // Compute predicted values: D̂_j = W × π̂
        let d_hat_j = linalg::mat_vec_mul(w_mat, &pi_hat);

        // Store predicted values
        for i in 0..n_rows {
            d_hat_flat[i * n_endog + j] = d_hat_j[i];
        }

        // Extract instrument coefficients only (skip intercept if present)
        let instrument_start = if include_intercept { 1 } else { 0 };
        let instrument_coefs: Vec<f64> =
            pi_hat[instrument_start..instrument_start + n_instruments].to_vec();
        instrument_coefficients.push(instrument_coefs);

        // Compute F-statistic for weak instrument detection
        let f_stat = compute_first_stage_f(
            &d_j,
            &d_hat_j,
            &x_only_mat,
            &xtx_restricted,
            &xtx_inv_restricted,
            n_rows,
            n_exog,
            n_instruments,
            include_intercept,
        );
        f_statistics.push(f_stat);

        // Validate F-statistic threshold (F < 4 indicates very weak instruments)
        if f_stat < 4.0 {
            return Err(IV2SLSError::InstrumentsTooWeak {
                endogenous_var: endog_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("D{}", j)),
                f_stat,
            });
        }
    }

    Ok(FirstStageResult {
        d_hat_flat,
        instrument_coefficients,
        f_statistics,
    })
}

/// Compute first-stage F-statistic for weak instrument detection.
///
/// Formula: F = [(RSS_r - RSS_u)/m] / [RSS_u/(n - m - k₂ - 1)]
/// where RSS_r is from restricted model (no instruments), RSS_u is from unrestricted.
#[allow(clippy::too_many_arguments)]
fn compute_first_stage_f(
    d_j: &[f64],
    d_hat_j: &[f64],
    x_mat_opt: &Option<Mat<f64>>,
    xtx_restricted: &Option<Mat<f64>>,
    xtx_inv_restricted: &Option<Mat<f64>>,
    n_rows: usize,
    n_exog: usize,
    n_instruments: usize,
    include_intercept: bool,
) -> f64 {
    // RSS from unrestricted model (with instruments): RSS_u = sum((D - D̂)²)
    let rss_unrestricted: f64 = d_j
        .iter()
        .zip(d_hat_j.iter())
        .map(|(d, dh)| (d - dh).powi(2))
        .sum();

    // RSS from restricted model (no instruments, just X)
    let rss_restricted: f64 = if let (Some(x_mat), Some(xtx), Some(_xtx_inv)) =
        (x_mat_opt, xtx_restricted, xtx_inv_restricted)
    {
        // Compute X'D_j and solve for restricted coefficients
        let xt_dj = linalg::xty(x_mat, d_j);
        if let Ok(delta_hat) = linalg::solve_normal_equations(xtx, &xt_dj) {
            // Compute predicted from restricted model
            let d_hat_restricted = linalg::mat_vec_mul(x_mat, &delta_hat);
            d_j.iter()
                .zip(d_hat_restricted.iter())
                .map(|(d, dr)| (d - dr).powi(2))
                .sum()
        } else {
            // If restricted model fails, use total sum of squares
            compute_total_sum_of_squares(d_j, n_rows)
        }
    } else {
        // No exogenous controls: restricted model is just the mean
        compute_total_sum_of_squares(d_j, n_rows)
    };

    // Number of parameters in restricted model
    let k_restricted = n_exog + if include_intercept { 1 } else { 0 };

    // Degrees of freedom
    let df_num = n_instruments as f64;
    let df_denom = (n_rows - n_instruments - k_restricted) as f64;

    if df_denom <= 0.0 || rss_unrestricted <= 0.0 {
        return 0.0;
    }

    // F = [(RSS_r - RSS_u)/m] / [RSS_u/(n - m - k₂ - 1)]
    let f_stat = ((rss_restricted - rss_unrestricted) / df_num) / (rss_unrestricted / df_denom);

    f_stat.max(0.0)
}

/// Compute total sum of squares relative to mean.
#[inline]
fn compute_total_sum_of_squares(data: &[f64], n: usize) -> f64 {
    let mean = data.iter().sum::<f64>() / (n as f64);
    data.iter().map(|x| (x - mean).powi(2)).sum()
}

// ============================================================================
// Second-Stage Regression
// ============================================================================

/// Perform second-stage regression: Y = V β + ε where V = [D̂ | X | 1].
///
/// Returns structural coefficients β̂₂ₛₗₛ.
fn second_stage_regression(v_mat: &Mat<f64>, y: &[f64]) -> Result<Vec<f64>, IV2SLSError> {
    // Compute V'V and V'y
    let vtv = linalg::xtx(v_mat);
    let vty = linalg::xty(v_mat, y);

    // Solve for β̂₂ₛₗₛ = (V'V)^-1 V'y
    linalg::solve_normal_equations(&vtv, &vty).map_err(|_| IV2SLSError::SecondStageSingular)
}

// ============================================================================
// Residual Computation (CRITICAL: Use Original D)
// ============================================================================

/// Compute second-stage residuals using ORIGINAL D, not predicted D̂.
///
/// **CRITICAL**: This is the correct formula for 2SLS residuals:
/// ε̂ᵢ = Yᵢ - β̂'[D | X | 1]ᵢ (using original D)
///
/// NOT: ε̂ᵢ = Yᵢ - β̂'[D̂ | X | 1]ᵢ (this would understate variance!)
fn compute_2sls_residuals(
    y: &[f64],
    u_mat: &Mat<f64>, // ORIGINAL design matrix with D (not D̂)
    beta: &[f64],
) -> Vec<f64> {
    let fitted = linalg::mat_vec_mul(u_mat, beta);
    y.iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect()
}

/// Compute R-squared using residuals from original D.
fn compute_r_squared(y: &[f64], residuals: &[f64]) -> f64 {
    let n = y.len() as f64;
    let y_mean = y.iter().sum::<f64>() / n;
    let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

// ============================================================================
// Standard Error Computation
// ============================================================================

/// Compute conventional 2SLS standard errors: SE = sqrt(diag(σ̂²(V'V)⁻¹))
/// where σ̂² = ε̂'ε̂/(n-k) and ε̂ uses ORIGINAL D.
fn compute_conventional_se(
    v_mat: &Mat<f64>,
    residuals: &[f64], // From original D
    n_rows: usize,
    n_params: usize,
) -> Result<Vec<f64>, IV2SLSError> {
    // Compute σ̂² = ε̂'ε̂/(n-k)
    let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    let df = (n_rows - n_params) as f64;
    if df <= 0.0 {
        return Err(IV2SLSError::InsufficientObservations {
            n_samples: n_rows,
            n_params,
        });
    }
    let sigma_sq = ss_res / df;

    // Compute (V'V)⁻¹
    let vtv = linalg::xtx(v_mat);
    let vtv_inv = linalg::invert_xtx(&vtv).map_err(|_| IV2SLSError::SecondStageSingular)?;

    // SE = sqrt(σ̂² × diag((V'V)⁻¹))
    let se: Vec<f64> = (0..n_params)
        .map(|i| (sigma_sq * vtv_inv.read(i, i)).sqrt())
        .collect();

    Ok(se)
}

/// Compute HC3 robust standard errors for 2SLS.
///
/// Returns standard errors and indices of high-leverage observations.
fn compute_robust_se(
    v_mat: &Mat<f64>,
    residuals: &[f64], // From original D
    _n_rows: usize,
) -> Result<(Vec<f64>, Vec<usize>), IV2SLSError> {
    let n_params = v_mat.ncols();

    // Compute (V'V)⁻¹
    let vtv = linalg::xtx(v_mat);
    let vtv_inv = linalg::invert_xtx(&vtv).map_err(|_| IV2SLSError::SecondStageSingular)?;

    // Compute leverages
    let leverages = linalg::compute_leverages_batch(v_mat, &vtv_inv).map_err(|_| {
        IV2SLSError::NumericalInstability {
            message: "High leverage observation detected".to_string(),
        }
    })?;

    // Identify high leverage observations (≥0.99)
    let high_leverage_indices: Vec<usize> = leverages
        .iter()
        .enumerate()
        .filter_map(|(i, &h)| if h >= 0.99 { Some(i) } else { None })
        .collect();

    // Compute HC3 vcov
    let hc3_vcov = linalg::compute_hc3_vcov_faer(v_mat, residuals, &leverages, &vtv_inv);

    // SE = sqrt(diag(HC3_vcov))
    let se: Vec<f64> = (0..n_params).map(|i| hc3_vcov.read(i, i).sqrt()).collect();

    Ok((se, high_leverage_indices))
}

/// Compute clustered standard errors for 2SLS.
///
/// Returns standard errors and number of clusters.
fn compute_clustered_se(
    v_mat: &Mat<f64>,
    residuals: &[f64], // From original D
    cluster_ids: &[i64],
    include_intercept: bool,
) -> Result<(Vec<f64>, usize), IV2SLSError> {
    // Build cluster indices
    let cluster_info = build_cluster_indices(cluster_ids).map_err(|e| match e {
        ClusterError::InsufficientClusters { found } => IV2SLSError::NumericalInstability {
            message: format!(
                "Clustered standard errors require at least 2 clusters; found {}",
                found
            ),
        },
        ClusterError::SingleObservationCluster { cluster_idx } => {
            IV2SLSError::NumericalInstability {
                message: format!("Cluster {} contains only 1 observation", cluster_idx),
            }
        }
        _ => IV2SLSError::NumericalInstability {
            message: "Cluster SE computation failed".to_string(),
        },
    })?;

    // Compute (V'V)⁻¹
    let vtv = linalg::xtx(v_mat);
    let vtv_inv = linalg::invert_xtx(&vtv).map_err(|_| IV2SLSError::SecondStageSingular)?;

    // Compute clustered SE
    let (coef_se, int_se) = compute_cluster_se_analytical_faer(
        v_mat,
        residuals,
        &vtv_inv,
        &cluster_info,
        include_intercept,
    )
    .map_err(|_| IV2SLSError::NumericalInstability {
        message: "Cluster SE computation failed".to_string(),
    })?;

    // Combine intercept SE with coef SE
    let se = if include_intercept {
        let mut full_se = vec![int_se.unwrap_or(0.0)];
        full_se.extend(coef_se);
        full_se
    } else {
        coef_se
    };

    Ok((se, cluster_info.n_clusters))
}

// ============================================================================
// Cragg-Donald Statistic
// ============================================================================

/// Compute Cragg-Donald statistic for multiple endogenous variables.
///
/// The Cragg-Donald statistic tests for weak instruments in the presence of
/// multiple endogenous regressors. It is based on the minimum eigenvalue of
/// the concentration parameter matrix.
///
/// **Correct Formula (matching linearmodels):**
/// ```text
/// CD = λ_min(D̃'P_Z D̃ / (n × σ²))
/// ```
/// Where:
/// - D̃ = D residualized on exogenous controls X
/// - P_Z = Z̃(Z̃'Z̃)⁻¹Z̃' (projection onto residualized instruments)
/// - σ² = average first-stage residual variance
/// - n = number of observations
///
/// Returns None for single endogenous (use first-stage F instead).
#[allow(clippy::too_many_arguments)]
fn compute_cragg_donald(
    d_flat: &[f64],
    z_flat: &[f64],
    x_flat: &[f64],
    n_rows: usize,
    n_endog: usize,
    n_instruments: usize,
    n_exog: usize,
    include_intercept: bool,
) -> Option<f64> {
    // For single endogenous, Cragg-Donald = first-stage F
    if n_endog == 1 {
        return None;
    }

    let n = n_rows as f64;

    // Build matrices
    let d_mat = linalg::flat_to_mat(d_flat, n_rows, n_endog);
    let z_mat = linalg::flat_to_mat(z_flat, n_rows, n_instruments);

    // Residualize D and Z on X (and intercept) if there are exogenous controls
    let (d_tilde, z_tilde) = if n_exog > 0 || include_intercept {
        let x_mat = linalg::flat_to_mat_with_intercept(x_flat, n_rows, n_exog, include_intercept);
        let xtx = linalg::xtx(&x_mat);
        let _xtx_inv = linalg::invert_xtx(&xtx).ok()?;

        // Residualize both D and Z matrices
        let d_tilde = residualize_matrix(&d_mat, &x_mat, &xtx, n_rows, n_endog)?;
        let z_tilde = residualize_matrix(&z_mat, &x_mat, &xtx, n_rows, n_instruments)?;

        (d_tilde, z_tilde)
    } else {
        (d_mat.clone(), z_mat.clone())
    };

    // Compute G = D̃'P_Z D̃ = D̃'Z̃(Z̃'Z̃)⁻¹Z̃'D̃
    // This represents the variation in D̃ explained by Z̃
    let ztz = linalg::xtx(&z_tilde);
    let ztz_inv = linalg::invert_xtx(&ztz).ok()?;

    // Z̃'D̃
    let mut ztd = Mat::zeros(n_instruments, n_endog);
    for j in 0..n_endog {
        let d_col: Vec<f64> = (0..n_rows).map(|i| d_tilde.read(i, j)).collect();
        let zt_dj = linalg::xty(&z_tilde, &d_col);
        for (k, &value) in zt_dj.iter().enumerate() {
            ztd.write(k, j, value);
        }
    }

    // D̃'Z̃ = (Z̃'D̃)'
    let dtz = ztd.transpose().to_owned();

    // G = D̃'P_Z D̃ = D̃'Z̃ × (Z̃'Z̃)⁻¹ × Z̃'D̃
    let temp = mat_mul(&dtz, &ztz_inv);
    let g = mat_mul(&temp, &ztd);

    // Compute D̃'D̃ (total sum of squares for D̃)
    let dtd = linalg::xtx(&d_tilde);

    // Compute σ² = average variance of residualized D̃
    // This matches linearmodels: np.mean([np.var(d1_resid), np.var(d2_resid)])
    // where d_resid is D residualized on X (the TOTAL variance of D̃, not first-stage residuals)
    let mut total_var = 0.0;
    for j in 0..n_endog {
        // Variance of D̃_j = diag(D̃'D̃) / n
        total_var += dtd.read(j, j) / n;
    }
    let sigma2 = total_var / (n_endog as f64);  // Average variance across endogenous vars

    if sigma2 <= 0.0 {
        return None;
    }

    // Compute concentration matrix: G / (n × σ²)
    let scale = n * sigma2;
    let mut conc_matrix = Mat::zeros(n_endog, n_endog);
    for i in 0..n_endog {
        for j in 0..n_endog {
            conc_matrix.write(i, j, g.read(i, j) / scale);
        }
    }

    // Cragg-Donald = λ_min(G / (n × σ²))
    let cd = compute_min_eigenvalue(&conc_matrix, n_endog);

    Some(cd.max(0.0))
}

/// Residualize a matrix by projecting out columns of X.
///
/// For each column of `mat`, compute residuals after regressing on X.
fn residualize_matrix(
    mat: &Mat<f64>,
    x_mat: &Mat<f64>,
    xtx: &Mat<f64>,
    n_rows: usize,
    n_cols: usize,
) -> Option<Mat<f64>> {
    let mut result = Mat::zeros(n_rows, n_cols);
    for j in 0..n_cols {
        let col: Vec<f64> = (0..n_rows).map(|i| mat.read(i, j)).collect();
        let xt_col = linalg::xty(x_mat, &col);
        if let Ok(coeffs) = linalg::solve_normal_equations(xtx, &xt_col) {
            let fitted = linalg::mat_vec_mul(x_mat, &coeffs);
            for (i, (&col_val, &fit_val)) in col.iter().zip(fitted.iter()).enumerate() {
                result.write(i, j, col_val - fit_val);
            }
        } else {
            // If regression fails, use original values
            for (i, &col_val) in col.iter().enumerate() {
                result.write(i, j, col_val);
            }
        }
    }
    Some(result)
}

/// Simple matrix multiplication for small matrices.
///
/// Computes C = A × B where A is m×k and B is k×n.
fn mat_mul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    let n = b.ncols();
    let k = a.ncols();

    let mut result = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a.read(i, l) * b.read(l, j);
            }
            result.write(i, j, sum);
        }
    }
    result
}

/// Compute minimum eigenvalue using analytical formulas or approximation.
///
/// For 1×1 and 2×2 matrices, uses analytical formulas.
/// For larger matrices, uses Rayleigh quotient approximation.
fn compute_min_eigenvalue(mat: &Mat<f64>, n: usize) -> f64 {
    match n {
        1 => mat.read(0, 0),
        2 => {
            // Use analytical formula for 2×2 matrix
            let a = mat.read(0, 0);
            let b = mat.read(0, 1);
            let c = mat.read(1, 0);
            let d = mat.read(1, 1);

            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant < 0.0 {
                // Complex eigenvalues, return real part
                trace / 2.0
            } else {
                let sqrt_disc = discriminant.sqrt();
                let lambda1 = (trace + sqrt_disc) / 2.0;
                let lambda2 = (trace - sqrt_disc) / 2.0;
                lambda1.min(lambda2)
            }
        }
        _ => {
            // For larger matrices, use Rayleigh quotient approximation
            // A more robust implementation would use faer's eigenvalue routines
            let mut v: Vec<f64> = (0..n).map(|i| 1.0 / ((i + 1) as f64).sqrt()).collect();
            let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for vi in &mut v {
                *vi /= v_norm;
            }

            // Compute Rayleigh quotient: v' A v / (v' v)
            let mut av = vec![0.0; n];
            for (i, av_elem) in av.iter_mut().enumerate() {
                for (j, &v_j) in v.iter().enumerate() {
                    *av_elem += mat.read(i, j) * v_j;
                }
            }

            v.iter().zip(av.iter()).map(|(vi, avi)| vi * avi).sum()
        }
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validate F-statistics and return warning messages for weak instruments.
fn validate_first_stage_f(f_stats: &[f64], endog_names: &[String]) -> Vec<String> {
    f_stats
        .iter()
        .enumerate()
        .filter_map(|(j, &f)| {
            if (4.0..10.0).contains(&f) {
                let name = endog_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("D{}", j));
                Some(format!(
                    "Weak instruments: first-stage F-statistic ({:.2}) is below 10 for endogenous variable '{}'",
                    f, name
                ))
            } else {
                None
            }
        })
        .collect()
}

/// Validate Cragg-Donald against Stock-Yogo critical value.
fn validate_cragg_donald(
    cragg_donald: Option<f64>,
    stock_yogo_critical: Option<f64>,
) -> Option<String> {
    match (cragg_donald, stock_yogo_critical) {
        (Some(cd), Some(critical)) if cd < critical => Some(format!(
            "Cragg-Donald statistic ({:.2}) is below Stock-Yogo 10% critical value ({:.2})",
            cd, critical
        )),
        _ => None,
    }
}

/// Validate instrument count relative to sample size.
fn validate_instrument_count(n_instruments: usize, n_samples: usize) -> Option<String> {
    if n_instruments > n_samples / 10 {
        Some(format!(
            "Large number of instruments ({}) relative to sample size ({}); consider using fewer",
            n_instruments, n_samples
        ))
    } else {
        None
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Main 2SLS computation entry point.
///
/// Performs two-stage least squares estimation with weak instrument diagnostics
/// and optional robust or clustered standard errors.
#[allow(clippy::too_many_arguments)]
pub fn compute_2sls(
    y: &[f64],
    d_flat: &[f64],
    x_flat: &[f64],
    z_flat: &[f64],
    n_rows: usize,
    n_endog: usize,
    n_exog: usize,
    n_instruments: usize,
    endog_names: &[String],
    _exog_names: &[String],
    _instrument_names: &[String],
    config: &IV2SLSConfig,
) -> Result<(TwoStageLSResult, Vec<String>), IV2SLSError> {
    let mut warnings = Vec::new();

    // =========================================================================
    // Validation
    // =========================================================================

    // Check identification: m ≥ k₁
    if n_instruments < n_endog {
        return Err(IV2SLSError::UnderIdentified {
            n_instruments,
            n_endogenous: n_endog,
        });
    }

    // Check sufficient observations
    let n_params = n_endog + n_exog + if config.include_intercept { 1 } else { 0 };
    if n_rows <= n_params {
        return Err(IV2SLSError::InsufficientObservations {
            n_samples: n_rows,
            n_params,
        });
    }

    // Validate many instruments warning
    if let Some(warning) = validate_instrument_count(n_instruments, n_rows) {
        warnings.push(warning);
    }

    // =========================================================================
    // First Stage
    // =========================================================================

    // Build first-stage design matrix W = [1 | Z | X]
    let w_mat = build_first_stage_matrix(
        z_flat,
        x_flat,
        n_rows,
        n_instruments,
        n_exog,
        config.include_intercept,
    );

    // Run first-stage regressions
    let first_stage_result = first_stage_regression(
        &w_mat,
        d_flat,
        x_flat,
        n_rows,
        n_endog,
        n_exog,
        n_instruments,
        config.include_intercept,
        endog_names,
    )?;

    // Add warnings for weak instruments (F < 10 but >= 4)
    warnings.extend(validate_first_stage_f(
        &first_stage_result.f_statistics,
        endog_names,
    ));

    // =========================================================================
    // Weak Instrument Diagnostics
    // =========================================================================

    // Compute Cragg-Donald if multiple endogenous
    let cragg_donald = compute_cragg_donald(
        d_flat,
        z_flat,
        x_flat,
        n_rows,
        n_endog,
        n_instruments,
        n_exog,
        config.include_intercept,
    );

    // Look up Stock-Yogo critical value
    let stock_yogo_critical = stock_yogo_lookup(n_endog, n_instruments);

    // Add Cragg-Donald warning if below critical value
    if let Some(warning) = validate_cragg_donald(cragg_donald, stock_yogo_critical) {
        warnings.push(warning);
    }

    // =========================================================================
    // Second Stage
    // =========================================================================

    // Build second-stage design matrix V = [1 | D̂ | X]
    let v_mat = build_second_stage_matrix(
        &first_stage_result.d_hat_flat,
        x_flat,
        n_rows,
        n_endog,
        n_exog,
        config.include_intercept,
    );

    // Run second-stage regression
    let beta_full = second_stage_regression(&v_mat, y)?;

    // =========================================================================
    // Residuals (CRITICAL: Use Original D)
    // =========================================================================

    // Build original design matrix U = [1 | D | X] (with ORIGINAL D, not D̂)
    let u_mat = build_original_design_matrix(
        d_flat,
        x_flat,
        n_rows,
        n_endog,
        n_exog,
        config.include_intercept,
    );

    // Compute residuals using ORIGINAL D
    let residuals = compute_2sls_residuals(y, &u_mat, &beta_full);

    // Compute R-squared
    let r_squared = compute_r_squared(y, &residuals);

    // =========================================================================
    // Standard Errors
    // =========================================================================

    let (se_full, se_type, n_clusters) = if let Some(cluster_ids) = config.cluster_ids {
        // Clustered SE
        let (se, n_clust) =
            compute_clustered_se(&v_mat, &residuals, cluster_ids, config.include_intercept)?;
        (se, "clustered".to_string(), Some(n_clust))
    } else if config.robust {
        // HC3 robust SE
        let (se, high_leverage) = compute_robust_se(&v_mat, &residuals, n_rows)?;

        // Add warnings for high leverage observations
        for idx in high_leverage {
            warnings.push(format!(
                "Observation {} has leverage ≥ 0.99; SE may be unreliable",
                idx
            ));
        }

        (se, "hc3".to_string(), None)
    } else {
        // Conventional SE
        let se = compute_conventional_se(&v_mat, &residuals, n_rows, n_params)?;
        (se, "conventional".to_string(), None)
    };

    // =========================================================================
    // Extract Coefficients
    // =========================================================================

    let (intercept, intercept_se, coefficients, standard_errors) = if config.include_intercept {
        (
            Some(beta_full[0]),
            Some(se_full[0]),
            beta_full[1..].to_vec(),
            se_full[1..].to_vec(),
        )
    } else {
        (None, None, beta_full, se_full)
    };

    // =========================================================================
    // Build Result
    // =========================================================================

    let result = TwoStageLSResult {
        coefficients,
        standard_errors,
        intercept,
        intercept_se,
        n_samples: n_rows,
        n_endogenous: n_endog,
        n_instruments,
        first_stage_f: first_stage_result.f_statistics,
        first_stage_coefficients: first_stage_result.instrument_coefficients,
        cragg_donald,
        stock_yogo_critical,
        r_squared,
        se_type,
        n_clusters,
    };

    Ok((result, warnings))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iv2sls_config_default() {
        let config = IV2SLSConfig::default();
        assert!(config.include_intercept);
        assert!(!config.robust);
        assert!(config.cluster_ids.is_none());
    }

    #[test]
    fn test_stock_yogo_lookup_single_endog() {
        assert_eq!(stock_yogo_lookup(1, 1), Some(16.38));
        assert_eq!(stock_yogo_lookup(1, 2), Some(19.93));
        assert_eq!(stock_yogo_lookup(1, 5), Some(26.87));
        assert_eq!(stock_yogo_lookup(1, 10), Some(38.54));
    }

    #[test]
    fn test_stock_yogo_lookup_two_endog() {
        assert_eq!(stock_yogo_lookup(2, 2), Some(7.03));
        assert_eq!(stock_yogo_lookup(2, 5), Some(21.68));
        assert_eq!(stock_yogo_lookup(2, 10), Some(37.42));
    }

    #[test]
    fn test_stock_yogo_lookup_unsupported() {
        assert_eq!(stock_yogo_lookup(3, 5), None);
        assert_eq!(stock_yogo_lookup(1, 11), None);
    }

    #[test]
    fn test_build_first_stage_matrix_with_intercept() {
        let z_flat = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows, 2 instruments
        let x_flat = vec![5.0, 6.0]; // 2 rows, 1 exogenous
        let mat = build_first_stage_matrix(&z_flat, &x_flat, 2, 2, 1, true);

        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 4); // intercept + 2 instruments + 1 exog

        // Check intercept column
        assert_eq!(mat.read(0, 0), 1.0);
        assert_eq!(mat.read(1, 0), 1.0);

        // Check instruments
        assert_eq!(mat.read(0, 1), 1.0);
        assert_eq!(mat.read(0, 2), 2.0);
        assert_eq!(mat.read(1, 1), 3.0);
        assert_eq!(mat.read(1, 2), 4.0);

        // Check exogenous
        assert_eq!(mat.read(0, 3), 5.0);
        assert_eq!(mat.read(1, 3), 6.0);
    }

    #[test]
    fn test_build_first_stage_matrix_no_intercept() {
        let z_flat = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows, 2 instruments
        let x_flat = vec![5.0, 6.0]; // 2 rows, 1 exogenous
        let mat = build_first_stage_matrix(&z_flat, &x_flat, 2, 2, 1, false);

        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3); // 2 instruments + 1 exog

        // Check instruments
        assert_eq!(mat.read(0, 0), 1.0);
        assert_eq!(mat.read(0, 1), 2.0);
        assert_eq!(mat.read(1, 0), 3.0);
        assert_eq!(mat.read(1, 1), 4.0);

        // Check exogenous
        assert_eq!(mat.read(0, 2), 5.0);
        assert_eq!(mat.read(1, 2), 6.0);
    }

    #[test]
    fn test_r_squared_computation() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residuals = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // Perfect fit
        let r2 = compute_r_squared(&y, &residuals);
        assert!((r2 - 1.0).abs() < 1e-10);

        // Constant y case
        let y_const = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let r2_const = compute_r_squared(&y_const, &residuals);
        assert_eq!(r2_const, 0.0); // ss_tot = 0
    }

    #[test]
    fn test_validate_first_stage_f() {
        let f_stats = vec![15.0, 8.0, 3.0];
        let names = vec!["d1".to_string(), "d2".to_string(), "d3".to_string()];

        // Note: F < 4 is an error in the main code, so we only get warnings for 4 <= F < 10
        let warnings = validate_first_stage_f(&f_stats, &names);

        // F=15 is fine, F=8 triggers warning, F=3 would have caused error earlier
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("d2"));
        assert!(warnings[0].contains("8.00"));
    }

    #[test]
    fn test_validate_instrument_count() {
        // n/10 = 10, so m > 10 should trigger warning
        assert!(validate_instrument_count(11, 100).is_some());
        assert!(validate_instrument_count(10, 100).is_none());
        assert!(validate_instrument_count(5, 100).is_none());
    }

    #[test]
    fn test_cragg_donald_warning() {
        // CD below critical should trigger warning
        let warning = validate_cragg_donald(Some(10.0), Some(16.38));
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("10.00"));

        // CD above critical should not trigger warning
        let no_warning = validate_cragg_donald(Some(20.0), Some(16.38));
        assert!(no_warning.is_none());

        // Missing values should not trigger warning
        assert!(validate_cragg_donald(None, Some(16.38)).is_none());
        assert!(validate_cragg_donald(Some(10.0), None).is_none());
    }

    #[test]
    fn test_under_identified_error() {
        let config = IV2SLSConfig::default();
        let y = vec![1.0, 2.0, 3.0];
        let d_flat = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]; // 3 rows, 2 endogenous
        let x_flat = vec![];
        let z_flat = vec![1.0, 2.0, 3.0]; // 3 rows, 1 instrument (under-identified!)

        let result = compute_2sls(
            &y,
            &d_flat,
            &x_flat,
            &z_flat,
            3,
            2,
            0,
            1,
            &["d1".to_string(), "d2".to_string()],
            &[],
            &["z1".to_string()],
            &config,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            IV2SLSError::UnderIdentified {
                n_instruments,
                n_endogenous,
            } => {
                assert_eq!(n_instruments, 1);
                assert_eq!(n_endogenous, 2);
            }
            _ => panic!("Expected UnderIdentified error"),
        }
    }
}
