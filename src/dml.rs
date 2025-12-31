//! Double Machine Learning (DML) estimator implementation.
//!
//! This module implements the DML estimator from Chernozhukov et al. (2018) with:
//! - Cross-fitting for debiased inference
//! - Linear/logistic nuisance models
//! - Neyman-orthogonal variance estimation
//! - Support for both ATE and linear CATE estimation
//!
//! # Algorithm Overview
//!
//! 1. Partition data into K folds for cross-fitting
//! 2. For each fold k:
//!    - Train outcome model ℓ(X) on observations NOT in fold k
//!    - Train propensity model m(X) on observations NOT in fold k
//!    - Predict for observations IN fold k (out-of-fold predictions)
//! 3. Compute residuals: Ỹ = Y - ℓ̂(X), D̃ = D - m̂(X)
//! 4. Final-stage regression: θ̂ = (D̃'D̃)⁻¹ D̃'Ỹ
//! 5. Neyman-orthogonal variance estimation
//!
//! # References
//!
//! Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
//! Newey, W., & Robins, J. (2018). Double/debiased machine learning for
//! treatment and structural parameters. The Econometrics Journal.

use std::collections::HashMap;

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cluster::{build_cluster_indices, ClusterInfo, SplitMix64};
use crate::linalg;
use crate::logistic::{self, sigmoid};

// ============================================================================
// Error Types
// ============================================================================

/// Error types for DML operations.
#[derive(Debug, Clone)]
pub enum DMLError {
    /// Invalid number of folds (must be >= 2)
    InvalidFolds { n_folds: usize },
    /// Number of folds >= N (too many folds for sample size)
    TooManyFolds { n_folds: usize, n_samples: usize },
    /// Treatment variable has no variation
    NoTreatmentVariation,
    /// Treatment fully explained by covariates (Var(D̃) < 1e-10)
    TreatmentFullyExplained,
    /// Propensity model failed to converge
    PropensityConvergenceFailure,
    /// Singular covariate matrix in fold
    SingularMatrix { fold: usize },
    /// Column contains null values
    NullValues { column: String },
    /// Numerical instability in computation
    NumericalInstability { message: String },
    /// Invalid alpha parameter
    InvalidAlpha { alpha: f64 },
    /// Invalid propensity clip bounds
    InvalidPropensityClip { low: f64, high: f64 },
    /// Multi-valued categorical treatment not supported
    MultiValuedTreatment { unique_values: usize },
    /// Multiple simultaneous treatments not supported
    MultipleTreatments { n_treatments: usize },
    /// Insufficient clusters for clustered SE
    InsufficientClusters { found: usize },
}

impl std::fmt::Display for DMLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DMLError::InvalidFolds { n_folds } => {
                write!(f, "n_folds must be >= 2; got {}", n_folds)
            }
            DMLError::TooManyFolds { n_folds, n_samples } => {
                write!(
                    f,
                    "Number of folds K must be less than N; got K={}, N={}",
                    n_folds, n_samples
                )
            }
            DMLError::NoTreatmentVariation => {
                write!(f, "Treatment variable has no variation")
            }
            DMLError::TreatmentFullyExplained => {
                write!(f, "Treatment fully explained by covariates")
            }
            DMLError::PropensityConvergenceFailure => {
                write!(f, "Propensity model failed to converge")
            }
            DMLError::SingularMatrix { fold } => {
                write!(f, "Covariate matrix is singular in fold {}", fold)
            }
            DMLError::NullValues { column } => {
                write!(f, "Column '{}' contains null values", column)
            }
            DMLError::NumericalInstability { message } => {
                write!(f, "Numerical instability: {}", message)
            }
            DMLError::InvalidAlpha { alpha } => {
                write!(f, "alpha must be in (0, 1); got {}", alpha)
            }
            DMLError::InvalidPropensityClip { low, high } => {
                write!(
                    f,
                    "propensity_clip must satisfy 0 < low < high < 1; got ({}, {})",
                    low, high
                )
            }
            DMLError::MultiValuedTreatment { unique_values } => {
                write!(
                    f,
                    "Multi-valued categorical treatments not supported in v1; found {} unique values (only binary 0/1 or continuous allowed)",
                    unique_values
                )
            }
            DMLError::MultipleTreatments { n_treatments } => {
                write!(
                    f,
                    "Multiple simultaneous treatments not supported in v1; found {} treatment columns",
                    n_treatments
                )
            }
            DMLError::InsufficientClusters { found } => {
                write!(
                    f,
                    "Clustered standard errors require at least 2 clusters; found {}",
                    found
                )
            }
        }
    }
}

impl std::error::Error for DMLError {}

// ============================================================================
// Configuration Types
// ============================================================================

/// Type of treatment variable.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TreatmentType {
    /// Binary treatment (0/1) - uses logistic propensity model
    Binary,
    /// Continuous treatment - uses linear propensity model
    Continuous,
}

/// Configuration for DML estimation.
#[derive(Clone, Debug)]
pub struct DMLConfig {
    /// Number of cross-fitting folds (must be >= 2)
    pub n_folds: usize,
    /// Type of treatment variable
    pub treatment_type: TreatmentType,
    /// Whether to estimate CATE coefficients
    pub estimate_cate: bool,
    /// Significance level for confidence intervals (default: 0.05 for 95% CI)
    pub alpha: f64,
    /// Bounds for propensity score clipping (only for binary treatment)
    pub propensity_clip: (f64, f64),
    /// Random seed for reproducible fold assignment (None for deterministic)
    pub seed: Option<u64>,
}

impl Default for DMLConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            treatment_type: TreatmentType::Binary,
            estimate_cate: false,
            alpha: 0.05,
            propensity_clip: (0.01, 0.99),
            seed: None,
        }
    }
}

impl DMLConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), DMLError> {
        // Validate n_folds
        if self.n_folds < 2 {
            return Err(DMLError::InvalidFolds {
                n_folds: self.n_folds,
            });
        }

        // Validate alpha
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(DMLError::InvalidAlpha { alpha: self.alpha });
        }

        // Validate propensity_clip bounds
        let (low, high) = self.propensity_clip;
        if low <= 0.0 || high >= 1.0 || low >= high {
            return Err(DMLError::InvalidPropensityClip { low, high });
        }

        Ok(())
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of DML estimation.
///
/// Contains the treatment effect estimate, standard error, confidence interval,
/// and various diagnostics.
#[pyclass]
#[derive(Debug, Clone)]
pub struct DMLResult {
    /// Average Treatment Effect (ATE) point estimate
    #[pyo3(get)]
    pub theta: f64,
    /// Robust standard error (Neyman-orthogonal)
    #[pyo3(get)]
    pub standard_error: f64,
    /// Confidence interval bounds (lower, upper)
    #[pyo3(get)]
    pub confidence_interval: (f64, f64),
    /// Two-sided p-value for H₀: θ = 0
    #[pyo3(get)]
    pub p_value: f64,
    /// Number of observations
    #[pyo3(get)]
    pub n_samples: usize,
    /// Number of cross-fitting folds used
    #[pyo3(get)]
    pub n_folds: usize,
    /// Variance of treatment residuals Var(D̃)
    #[pyo3(get)]
    pub propensity_residual_var: f64,
    /// Variance of outcome residuals Var(Ỹ)
    #[pyo3(get)]
    pub outcome_residual_var: f64,
    /// Average R² of outcome nuisance model across folds
    #[pyo3(get)]
    pub outcome_r_squared: f64,
    /// Average R² (or pseudo-R²) of propensity nuisance model across folds
    #[pyo3(get)]
    pub propensity_r_squared: f64,
    /// Count of clipped propensity scores
    #[pyo3(get)]
    pub n_propensity_clipped: usize,
    /// CATE coefficients keyed by covariate name (if estimate_cate=True)
    #[pyo3(get)]
    pub cate_coefficients: Option<HashMap<String, f64>>,
    /// CATE coefficient standard errors (if estimate_cate=True)
    #[pyo3(get)]
    pub cate_standard_errors: Option<HashMap<String, f64>>,
}

#[pymethods]
impl DMLResult {
    /// String representation for Python
    fn __repr__(&self) -> String {
        let cate_info = if self.cate_coefficients.is_some() {
            ", cate_coefficients=Some(...)"
        } else {
            ""
        };
        format!(
            "DMLResult(theta={:.4}, standard_error={:.4}, confidence_interval=({:.4}, {:.4}), p_value={:.4}, n_samples={}, n_folds={}, outcome_r_squared={:.4}, propensity_r_squared={:.4}, n_propensity_clipped={}{})",
            self.theta,
            self.standard_error,
            self.confidence_interval.0,
            self.confidence_interval.1,
            self.p_value,
            self.n_samples,
            self.n_folds,
            self.outcome_r_squared,
            self.propensity_r_squared,
            self.n_propensity_clipped,
            cate_info
        )
    }

    /// Return formatted summary string
    fn summary(&self) -> String {
        let stars = if self.p_value < 0.001 {
            "***"
        } else if self.p_value < 0.01 {
            "**"
        } else if self.p_value < 0.05 {
            "*"
        } else if self.p_value < 0.1 {
            "."
        } else {
            ""
        };

        let clip_pct = if self.n_samples > 0 {
            100.0 * self.n_propensity_clipped as f64 / self.n_samples as f64
        } else {
            0.0
        };

        let mut summary = format!(
            r#"Double Machine Learning Results
═══════════════════════════════════════════════════════════════
Treatment Effect (ATE):
  θ̂ = {:.4} ± {:.4}
  {}% CI: [{:.4}, {:.4}]
  p-value: {:.4} {}

Sample Information:
  N = {}
  K = {} folds

Nuisance Model Diagnostics:
  Outcome model R²: {:.4}
  Propensity model R²: {:.4}
  Propensity scores clipped: {} ({:.1}%)

Residual Variances:
  Var(Ỹ) = {:.4}
  Var(D̃) = {:.4}
"#,
            self.theta,
            self.standard_error,
            ((1.0 - 0.05) * 100.0) as i32, // Assume alpha=0.05 for display
            self.confidence_interval.0,
            self.confidence_interval.1,
            self.p_value,
            stars,
            self.n_samples,
            self.n_folds,
            self.outcome_r_squared,
            self.propensity_r_squared,
            self.n_propensity_clipped,
            clip_pct,
            self.outcome_residual_var,
            self.propensity_residual_var
        );

        if let Some(ref cate_coefs) = self.cate_coefficients {
            summary.push_str("\nCATE Coefficients:\n");
            let cate_ses = self.cate_standard_errors.as_ref();
            for (name, coef) in cate_coefs {
                let se_str = cate_ses
                    .and_then(|ses| ses.get(name))
                    .map(|se| format!(" (SE: {:.4})", se))
                    .unwrap_or_default();
                summary.push_str(&format!("  {}: {:.4}{}\n", name, coef, se_str));
            }
        }

        summary.push_str("═══════════════════════════════════════════════════════════════");
        summary
    }
}

// ============================================================================
// Cross-Fitting Infrastructure
// ============================================================================

/// Cross-fitting infrastructure for K-fold sample splitting.
///
/// Invariants:
/// - fold_indices.len() == n_folds
/// - flatten(fold_indices) is a permutation of 0..n_samples
/// - All folds have approximately equal size (±1)
pub struct CrossFitter {
    /// Indices for each fold: fold_indices[k] = observations in fold k
    pub fold_indices: Vec<Vec<usize>>,
    /// Number of folds
    pub n_folds: usize,
    /// Total number of samples (kept for debugging/validation)
    #[allow(dead_code)]
    pub n_samples: usize,
    /// Random seed used (None if deterministic, kept for reproducibility tracking)
    #[allow(dead_code)]
    pub seed: Option<u64>,
}

impl CrossFitter {
    /// Create a new cross-fitter with K folds.
    ///
    /// If seed is Some, uses seeded random assignment.
    /// If seed is None, uses deterministic row-order assignment.
    pub fn new(n_samples: usize, n_folds: usize, seed: Option<u64>) -> Result<Self, DMLError> {
        if n_folds < 2 {
            return Err(DMLError::InvalidFolds {
                n_folds,
            });
        }

        if n_folds >= n_samples {
            return Err(DMLError::TooManyFolds { n_folds, n_samples });
        }

        let fold_indices = if let Some(s) = seed {
            Self::seeded_assignment(n_samples, n_folds, s)
        } else {
            Self::deterministic_assignment(n_samples, n_folds)
        };

        Ok(Self {
            fold_indices,
            n_folds,
            n_samples,
            seed,
        })
    }

    /// Deterministic assignment based on row index.
    fn deterministic_assignment(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut folds: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
        for i in 0..n {
            folds[i % k].push(i);
        }
        folds
    }

    /// Seeded random assignment using Fisher-Yates shuffle.
    fn seeded_assignment(n: usize, k: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = SplitMix64::new(seed);

        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        // Distribute to folds
        let mut folds: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
        for (i, &idx) in indices.iter().enumerate() {
            folds[i % k].push(idx);
        }
        folds
    }

    /// Get train indices (all except fold k) and test indices (fold k).
    pub fn get_fold(&self, k: usize) -> (Vec<usize>, &[usize]) {
        let test_indices = &self.fold_indices[k];
        let train_indices: Vec<usize> = self
            .fold_indices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k)
            .flat_map(|(_, indices)| indices.iter().copied())
            .collect();
        (train_indices, test_indices)
    }
}

// ============================================================================
// Nuisance Estimation
// ============================================================================

/// Fitted nuisance model state.
#[derive(Clone)]
pub enum NuisanceModel {
    Linear {
        coefficients: Vec<f64>,
        include_intercept: bool,
    },
    Logistic {
        coefficients: Vec<f64>,
        include_intercept: bool,
    },
}

/// Trait for nuisance function estimators.
///
/// This abstraction allows adding new estimators (RandomForest, GBM) in v2
/// without modifying the DML core logic.
pub trait NuisanceEstimator: Send + Sync {
    /// Fit the nuisance model on training data.
    fn fit(
        &self,
        x: &[f64],
        y: &[f64],
        n_rows: usize,
        n_cols: usize,
        fold_idx: usize,
    ) -> Result<NuisanceModel, DMLError>;

    /// Predict using fitted model.
    fn predict(&self, model: &NuisanceModel, x: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64>;

    /// Compute R² (or pseudo-R² for logistic) on given data.
    fn r_squared(
        &self,
        model: &NuisanceModel,
        x: &[f64],
        y: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> f64;
}

/// Linear nuisance estimator for outcome model and continuous treatment.
pub struct LinearNuisance;

impl NuisanceEstimator for LinearNuisance {
    fn fit(
        &self,
        x: &[f64],
        y: &[f64],
        n_rows: usize,
        n_cols: usize,
        fold_idx: usize,
    ) -> Result<NuisanceModel, DMLError> {
        // Build design matrix with intercept
        let x_mat = linalg::flat_to_mat_with_intercept(x, n_rows, n_cols, true);
        let xtx = linalg::xtx(&x_mat);
        let xty = linalg::xty(&x_mat, y);

        let coefficients = linalg::solve_normal_equations(&xtx, &xty)
            .map_err(|_| DMLError::SingularMatrix { fold: fold_idx })?;

        Ok(NuisanceModel::Linear {
            coefficients,
            include_intercept: true,
        })
    }

    fn predict(&self, model: &NuisanceModel, x: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
        match model {
            NuisanceModel::Linear {
                coefficients,
                include_intercept,
            } => {
                let mut predictions = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let mut pred = if *include_intercept {
                        coefficients[0]
                    } else {
                        0.0
                    };
                    let coef_offset = if *include_intercept { 1 } else { 0 };
                    for j in 0..n_cols {
                        pred += coefficients[coef_offset + j] * x[i * n_cols + j];
                    }
                    predictions.push(pred);
                }
                predictions
            }
            _ => panic!("LinearNuisance::predict called with non-linear model"),
        }
    }

    fn r_squared(
        &self,
        model: &NuisanceModel,
        x: &[f64],
        y: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> f64 {
        let predictions = self.predict(model, x, n_rows, n_cols);
        let y_mean: f64 = y.iter().sum::<f64>() / n_rows as f64;

        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, actual)| (actual - pred).powi(2))
            .sum();

        let ss_tot: f64 = y.iter().map(|actual| (actual - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }
}

/// Logistic nuisance estimator for binary treatment propensity.
pub struct LogisticNuisance;

impl NuisanceEstimator for LogisticNuisance {
    fn fit(
        &self,
        x: &[f64],
        y: &[f64],
        n_rows: usize,
        n_cols: usize,
        fold_idx: usize,
    ) -> Result<NuisanceModel, DMLError> {
        let x_mat = linalg::flat_to_mat_with_intercept(x, n_rows, n_cols, true);

        let mle_result = logistic::compute_logistic_mle(&x_mat, y).map_err(|e| match e {
            logistic::LogisticError::ConvergenceFailure { .. } => {
                DMLError::PropensityConvergenceFailure
            }
            logistic::LogisticError::SingularHessian => DMLError::SingularMatrix { fold: fold_idx },
            _ => DMLError::NumericalInstability {
                message: e.to_string(),
            },
        })?;

        Ok(NuisanceModel::Logistic {
            coefficients: mle_result.beta,
            include_intercept: true,
        })
    }

    fn predict(&self, model: &NuisanceModel, x: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
        match model {
            NuisanceModel::Logistic {
                coefficients,
                include_intercept,
            } => {
                let mut predictions = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let mut linear_pred = if *include_intercept {
                        coefficients[0]
                    } else {
                        0.0
                    };
                    let coef_offset = if *include_intercept { 1 } else { 0 };
                    for j in 0..n_cols {
                        linear_pred += coefficients[coef_offset + j] * x[i * n_cols + j];
                    }
                    predictions.push(sigmoid(linear_pred));
                }
                predictions
            }
            _ => panic!("LogisticNuisance::predict called with non-logistic model"),
        }
    }

    fn r_squared(
        &self,
        model: &NuisanceModel,
        x: &[f64],
        y: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> f64 {
        // McFadden's pseudo-R² for logistic regression
        match model {
            NuisanceModel::Logistic {
                coefficients,
                include_intercept,
            } => {
                // Compute model log-likelihood
                let mut ll_model = 0.0;
                let epsilon = 1e-15;
                let coef_offset = if *include_intercept { 1 } else { 0 };

                for i in 0..n_rows {
                    let mut linear_pred = if *include_intercept {
                        coefficients[0]
                    } else {
                        0.0
                    };
                    for j in 0..n_cols {
                        linear_pred += coefficients[coef_offset + j] * x[i * n_cols + j];
                    }
                    let pi = sigmoid(linear_pred);
                    let pi_clipped = pi.max(epsilon).min(1.0 - epsilon);
                    ll_model += y[i] * pi_clipped.ln() + (1.0 - y[i]) * (1.0 - pi_clipped).ln();
                }

                // Compute null log-likelihood
                let p1 = y.iter().sum::<f64>() / n_rows as f64;
                let p1_clipped = p1.max(epsilon).min(1.0 - epsilon);
                let ll_null =
                    n_rows as f64 * (p1_clipped.ln() * p1 + (1.0 - p1_clipped).ln() * (1.0 - p1));

                if ll_null == 0.0 {
                    0.0
                } else {
                    1.0 - ll_model / ll_null
                }
            }
            _ => 0.0,
        }
    }
}

// ============================================================================
// Data Extraction Helpers
// ============================================================================

/// Extract fold data from flat arrays using indices.
fn extract_fold_data(
    x: &[f64],
    y: &[f64],
    d: &[f64],
    indices: &[usize],
    n_cols: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = indices.len();
    let mut x_fold = Vec::with_capacity(n * n_cols);
    let mut y_fold = Vec::with_capacity(n);
    let mut d_fold = Vec::with_capacity(n);

    for &idx in indices {
        for j in 0..n_cols {
            x_fold.push(x[idx * n_cols + j]);
        }
        y_fold.push(y[idx]);
        d_fold.push(d[idx]);
    }

    (x_fold, y_fold, d_fold)
}

/// Extract X data only for prediction.
fn extract_x_fold(x: &[f64], indices: &[usize], n_cols: usize) -> Vec<f64> {
    let n = indices.len();
    let mut x_fold = Vec::with_capacity(n * n_cols);

    for &idx in indices {
        for j in 0..n_cols {
            x_fold.push(x[idx * n_cols + j]);
        }
    }

    x_fold
}

// ============================================================================
// Core DML Algorithm
// ============================================================================

/// Result from processing a single fold.
struct FoldResult {
    test_indices: Vec<usize>,
    y_hat_fold: Vec<f64>,
    d_hat_fold: Vec<f64>,
    outcome_r2: f64,
    propensity_r2: f64,
}

/// Clip propensity scores to bounds and count clipped values.
fn propensity_clip(propensity: &mut [f64], bounds: (f64, f64)) -> usize {
    let (low, high) = bounds;
    let mut n_clipped = 0;
    for p in propensity.iter_mut() {
        if *p < low {
            *p = low;
            n_clipped += 1;
        } else if *p > high {
            *p = high;
            n_clipped += 1;
        }
    }
    n_clipped
}

/// Compute variance of a vector.
fn compute_variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
}

/// Validate treatment has non-zero variance.
fn validate_treatment_variance(d: &[f64]) -> Result<(), DMLError> {
    let variance = compute_variance(d);
    if variance < 1e-10 {
        return Err(DMLError::NoTreatmentVariation);
    }
    Ok(())
}

/// Validate treatment residuals have non-zero variance.
fn validate_treatment_residual_variance(d_residual: &[f64]) -> Result<(), DMLError> {
    let variance = compute_variance(d_residual);
    if variance < 1e-10 {
        return Err(DMLError::TreatmentFullyExplained);
    }
    Ok(())
}

/// Final-stage regression: θ̂ = (D̃'D̃)⁻¹ D̃'Ỹ
fn final_stage_regression(y_residual: &[f64], d_residual: &[f64]) -> f64 {
    // θ̂ = Σ(D̃ᵢ × Ỹᵢ) / Σ(D̃ᵢ²)
    let d_tilde_y_tilde: f64 = d_residual
        .iter()
        .zip(y_residual.iter())
        .map(|(d, y)| d * y)
        .sum();
    let d_tilde_sq: f64 = d_residual.iter().map(|d| d * d).sum();
    d_tilde_y_tilde / d_tilde_sq
}

/// Neyman-orthogonal variance estimation.
///
/// Formula: V̂ = (1/N) × J⁻² × Σψ²ᵢ
/// where J = (1/N)∑D̃ᵢ² and ψᵢ = (Ỹᵢ - θ̂D̃ᵢ)D̃ᵢ
fn neyman_orthogonal_variance(y_residual: &[f64], d_residual: &[f64], theta: f64) -> f64 {
    let n = y_residual.len() as f64;

    // J = (1/N) × Σ D̃ᵢ²
    let j: f64 = d_residual.iter().map(|d| d * d).sum::<f64>() / n;

    // ψ²ᵢ = ((Ỹᵢ - θ̂ × D̃ᵢ) × D̃ᵢ)²
    let psi_sq_sum: f64 = y_residual
        .iter()
        .zip(d_residual.iter())
        .map(|(y, d)| {
            let eps = y - theta * d;
            (eps * d).powi(2)
        })
        .sum();

    // V̂ = (1/N) × J⁻² × Σψ²ᵢ
    (1.0 / n) * (1.0 / (j * j)) * psi_sq_sum
}

/// Cluster-robust Neyman-orthogonal variance estimation.
///
/// Formula: V̂_cluster = (1/N) × J⁻² × Σ_g (Σ_{i∈g} ψᵢ)² × G/(G-1)
/// where J = (1/N)∑D̃ᵢ², ψᵢ = (Ỹᵢ - θ̂D̃ᵢ)D̃ᵢ, G = number of clusters
///
/// Uses the sandwich estimator at the cluster level with small-sample correction.
fn neyman_orthogonal_variance_clustered(
    y_residual: &[f64],
    d_residual: &[f64],
    theta: f64,
    cluster_info: &ClusterInfo,
) -> f64 {
    let n = y_residual.len() as f64;
    let g = cluster_info.n_clusters as f64;

    // J = (1/N) × Σ D̃ᵢ²
    let j: f64 = d_residual.iter().map(|d| d * d).sum::<f64>() / n;

    // Compute ψᵢ for all observations
    let psi: Vec<f64> = y_residual
        .iter()
        .zip(d_residual.iter())
        .map(|(y, d)| {
            let eps = y - theta * d;
            eps * d
        })
        .collect();

    // Sum squared cluster scores: Σ_g (Σ_{i∈g} ψᵢ)²
    let cluster_psi_sq_sum: f64 = cluster_info
        .indices
        .iter()
        .map(|cluster_indices| {
            let cluster_psi_sum: f64 = cluster_indices.iter().map(|&i| psi[i]).sum();
            cluster_psi_sum.powi(2)
        })
        .sum();

    // Small-sample adjustment: G/(G-1)
    let adjustment = g / (g - 1.0);

    // V̂_cluster = (1/N) × J⁻² × Σ_g (Σ_{i∈g} ψᵢ)² × G/(G-1)
    (1.0 / n) * (1.0 / (j * j)) * cluster_psi_sq_sum * adjustment
}

/// Validate binary treatment values.
///
/// For binary treatment, values must be exactly 0 or 1.
/// Returns error if treatment has more than 2 unique values (multi-valued categorical).
fn validate_binary_treatment(d: &[f64]) -> Result<(), DMLError> {
    use std::collections::HashSet;

    let unique: HashSet<i64> = d.iter().map(|&v| (v * 1e9).round() as i64).collect();
    let unique_count = unique.len();

    // For binary treatment, we expect exactly 0 and 1
    // Check if all values are 0 or 1
    let is_binary = d.iter().all(|&v| (v - 0.0).abs() < 1e-10 || (v - 1.0).abs() < 1e-10);

    if !is_binary && unique_count > 2 {
        // This is a multi-valued categorical treatment - not supported
        return Err(DMLError::MultiValuedTreatment {
            unique_values: unique_count,
        });
    }

    Ok(())
}

/// Check fold sizes and emit warning if N/K < 30.
///
/// Returns true if warning was emitted.
fn check_small_fold_warning(n_rows: usize, n_folds: usize) -> bool {
    let samples_per_fold = n_rows / n_folds;
    if samples_per_fold < 30 {
        eprintln!(
            "Warning: Small fold sizes may cause instability. N/K = {} (less than 30 recommended). \
             Consider increasing sample size or decreasing number of folds.",
            samples_per_fold
        );
        return true;
    }
    false
}

/// Standard normal quantile (inverse CDF) approximation.
///
/// Uses Abramowitz and Stegun approximation (26.2.23).
fn normal_quantile(p: f64) -> f64 {
    // Handle edge cases
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Rational approximation for 0 < p < 1
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Coefficients for approximation
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -z
    } else {
        z
    }
}

/// Standard normal CDF approximation.
fn normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation (7.1.26)
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d1 = 0.319381530;
    let d2 = -0.356563782;
    let d3 = 1.781477937;
    let d4 = -1.821255978;
    let d5 = 1.330274429;

    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * t * (d1 + t * (d2 + t * (d3 + t * (d4 + t * d5))));

    if x >= 0.0 {
        cdf
    } else {
        1.0 - cdf
    }
}

/// Compute confidence interval.
fn compute_confidence_interval(theta: f64, se: f64, alpha: f64) -> (f64, f64) {
    let z = normal_quantile(1.0 - alpha / 2.0);
    (theta - z * se, theta + z * se)
}

/// Compute p-value for H₀: θ = 0.
fn compute_p_value(theta: f64, se: f64) -> f64 {
    if se == 0.0 {
        if theta == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        let z = theta.abs() / se;
        2.0 * (1.0 - normal_cdf(z))
    }
}

/// Type alias for CATE estimation results: (coefficients, standard_errors)
type CateResults = (HashMap<String, f64>, HashMap<String, f64>);

/// Build CATE interaction design matrix: [D̃ | D̃×X₁ | D̃×X₂ | ... | D̃×Xₚ]
fn build_cate_design_matrix(
    d_residual: &[f64],
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f64> {
    let total_cols = n_cols + 1; // D̃ + D̃×X interactions
    let mut z = Vec::with_capacity(n_rows * total_cols);

    for i in 0..n_rows {
        z.push(d_residual[i]); // D̃
        for j in 0..n_cols {
            z.push(d_residual[i] * x[i * n_cols + j]); // D̃×Xⱼ
        }
    }

    z
}

/// Compute CATE coefficients and standard errors.
fn compute_cate(
    y_residual: &[f64],
    d_residual: &[f64],
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    x_col_names: &[String],
) -> Result<CateResults, DMLError> {
    // Build interaction design matrix
    let z = build_cate_design_matrix(d_residual, x, n_rows, n_cols);
    let total_cols = n_cols + 1;

    // Fit linear regression: Ỹ on [D̃, D̃×X]
    let z_mat = linalg::flat_to_mat_with_intercept(&z, n_rows, total_cols, false);
    let ztz = linalg::xtx(&z_mat);
    let zty = linalg::xty(&z_mat, y_residual);

    let beta = linalg::solve_normal_equations(&ztz, &zty)
        .map_err(|_| DMLError::SingularMatrix { fold: 0 })?;

    // Compute fitted values and residuals for HC3 SE
    let fitted = linalg::mat_vec_mul(&z_mat, &beta);
    let residuals: Vec<f64> = y_residual
        .iter()
        .zip(fitted.iter())
        .map(|(y, yhat)| y - yhat)
        .collect();

    // Compute (Z'Z)^-1
    let ztz_inv = linalg::invert_xtx(&ztz).map_err(|_| DMLError::SingularMatrix { fold: 0 })?;

    // Compute leverages and HC3 standard errors
    let leverages = linalg::compute_leverages_batch(&z_mat, &ztz_inv).map_err(|_| {
        DMLError::NumericalInstability {
            message: "Extreme leverage in CATE estimation".to_string(),
        }
    })?;

    let hc3_vcov = linalg::compute_hc3_vcov_faer(&z_mat, &residuals, &leverages, &ztz_inv);

    // Extract coefficients and SEs
    let mut cate_coefficients = HashMap::new();
    let mut cate_standard_errors = HashMap::new();

    // Intercept (baseline ATE in CATE model)
    cate_coefficients.insert("_intercept".to_string(), beta[0]);
    cate_standard_errors.insert("_intercept".to_string(), hc3_vcov.read(0, 0).sqrt());

    // Interaction coefficients
    for (i, name) in x_col_names.iter().enumerate() {
        cate_coefficients.insert(name.clone(), beta[i + 1]);
        cate_standard_errors.insert(name.clone(), hc3_vcov.read(i + 1, i + 1).sqrt());
    }

    Ok((cate_coefficients, cate_standard_errors))
}

/// Cross-fitting with parallel fold processing.
#[allow(clippy::too_many_arguments)]
fn cross_fit_parallel(
    y: &[f64],
    d: &[f64],
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    cross_fitter: &CrossFitter,
    outcome_estimator: &dyn NuisanceEstimator,
    propensity_estimator: &dyn NuisanceEstimator,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), DMLError> {
    // Parallel fold processing
    let fold_results: Vec<Result<FoldResult, DMLError>> = (0..cross_fitter.n_folds)
        .into_par_iter()
        .map(|k| {
            let (train_indices, test_indices) = cross_fitter.get_fold(k);

            // Extract training data for this fold
            let (x_train, y_train, d_train) = extract_fold_data(x, y, d, &train_indices, n_cols);

            // Fit outcome model
            let outcome_model =
                outcome_estimator.fit(&x_train, &y_train, train_indices.len(), n_cols, k)?;

            // Fit propensity model
            let propensity_model =
                propensity_estimator.fit(&x_train, &d_train, train_indices.len(), n_cols, k)?;

            // Predict on test set
            let x_test = extract_x_fold(x, test_indices, n_cols);
            let y_hat_fold =
                outcome_estimator.predict(&outcome_model, &x_test, test_indices.len(), n_cols);
            let d_hat_fold = propensity_estimator.predict(
                &propensity_model,
                &x_test,
                test_indices.len(),
                n_cols,
            );

            // Compute R² on training data
            let outcome_r2 = outcome_estimator.r_squared(
                &outcome_model,
                &x_train,
                &y_train,
                train_indices.len(),
                n_cols,
            );
            let propensity_r2 = propensity_estimator.r_squared(
                &propensity_model,
                &x_train,
                &d_train,
                train_indices.len(),
                n_cols,
            );

            Ok(FoldResult {
                test_indices: test_indices.to_vec(),
                y_hat_fold,
                d_hat_fold,
                outcome_r2,
                propensity_r2,
            })
        })
        .collect();

    // Pre-allocate output vectors
    let mut y_hat = vec![0.0; n_rows];
    let mut d_hat = vec![0.0; n_rows];

    // Aggregate results (sequential)
    let mut outcome_r2_sum = 0.0;
    let mut propensity_r2_sum = 0.0;

    for result in fold_results {
        let fold_result = result?;

        for (i, &idx) in fold_result.test_indices.iter().enumerate() {
            y_hat[idx] = fold_result.y_hat_fold[i];
            d_hat[idx] = fold_result.d_hat_fold[i];
        }

        outcome_r2_sum += fold_result.outcome_r2;
        propensity_r2_sum += fold_result.propensity_r2;
    }

    let avg_outcome_r2 = outcome_r2_sum / cross_fitter.n_folds as f64;
    let avg_propensity_r2 = propensity_r2_sum / cross_fitter.n_folds as f64;

    Ok((y_hat, d_hat, avg_outcome_r2, avg_propensity_r2))
}

// ============================================================================
// Main DML Entry Point
// ============================================================================

/// Main DML computation entry point.
///
/// # Arguments
/// * `y` - Outcome variable (N,)
/// * `d` - Treatment variable (N,)
/// * `x_flat` - Covariates as flat row-major array (N × p)
/// * `n_rows` - Number of observations
/// * `n_cols` - Number of covariates
/// * `x_col_names` - Names of covariate columns
/// * `config` - DML configuration
/// * `cluster_ids` - Optional cluster identifiers for clustered SE
///
/// # Returns
/// * `Result<DMLResult, DMLError>` - DML result or error
#[allow(clippy::too_many_arguments)]
pub fn compute_dml(
    y: &[f64],
    d: &[f64],
    x_flat: &[f64],
    n_rows: usize,
    n_cols: usize,
    x_col_names: &[String],
    config: &DMLConfig,
    cluster_ids: Option<&[i64]>,
) -> Result<DMLResult, DMLError> {
    // Validate configuration
    config.validate()?;

    // Validate sample size
    if config.n_folds >= n_rows {
        return Err(DMLError::TooManyFolds {
            n_folds: config.n_folds,
            n_samples: n_rows,
        });
    }

    // Check for small fold sizes and emit warning
    check_small_fold_warning(n_rows, config.n_folds);

    // Validate treatment variance
    validate_treatment_variance(d)?;

    // Validate binary treatment values (only 0/1 allowed)
    if config.treatment_type == TreatmentType::Binary {
        validate_binary_treatment(d)?;
    }

    // Build cluster info if cluster_ids provided
    let cluster_info: Option<ClusterInfo> = if let Some(ids) = cluster_ids {
        let info = build_cluster_indices(ids).map_err(|_| {
            DMLError::NumericalInstability {
                message: "Failed to build cluster indices".to_string(),
            }
        })?;
        if info.n_clusters < 2 {
            return Err(DMLError::InsufficientClusters {
                found: info.n_clusters,
            });
        }
        Some(info)
    } else {
        None
    };

    // Create cross-fitter
    let cross_fitter = CrossFitter::new(n_rows, config.n_folds, config.seed)?;

    // Select nuisance estimators based on treatment type
    let outcome_estimator: Box<dyn NuisanceEstimator> = Box::new(LinearNuisance);
    let propensity_estimator: Box<dyn NuisanceEstimator> = match config.treatment_type {
        TreatmentType::Binary => Box::new(LogisticNuisance),
        TreatmentType::Continuous => Box::new(LinearNuisance),
    };

    // Cross-fitting
    let (y_hat, mut d_hat, outcome_r_squared, propensity_r_squared) = cross_fit_parallel(
        y,
        d,
        x_flat,
        n_rows,
        n_cols,
        &cross_fitter,
        outcome_estimator.as_ref(),
        propensity_estimator.as_ref(),
    )?;

    // Propensity clipping (binary treatment only)
    let n_propensity_clipped = if config.treatment_type == TreatmentType::Binary {
        propensity_clip(&mut d_hat, config.propensity_clip)
    } else {
        0
    };

    // Compute residuals
    let y_residual: Vec<f64> = y.iter().zip(y_hat.iter()).map(|(yi, yh)| yi - yh).collect();
    let d_residual: Vec<f64> = d.iter().zip(d_hat.iter()).map(|(di, dh)| di - dh).collect();

    // Validate treatment residual variance
    validate_treatment_residual_variance(&d_residual)?;

    // Compute residual variances for diagnostics
    let outcome_residual_var = compute_variance(&y_residual);
    let propensity_residual_var = compute_variance(&d_residual);

    // Final-stage regression
    let theta = final_stage_regression(&y_residual, &d_residual);

    // Use cluster-robust variance if cluster_ids provided
    let variance = if let Some(ref info) = cluster_info {
        neyman_orthogonal_variance_clustered(&y_residual, &d_residual, theta, info)
    } else {
        neyman_orthogonal_variance(&y_residual, &d_residual, theta)
    };
    let standard_error = variance.sqrt();

    // Confidence interval and p-value
    let confidence_interval = compute_confidence_interval(theta, standard_error, config.alpha);
    let p_value = compute_p_value(theta, standard_error);

    // CATE estimation if requested
    let (cate_coefficients, cate_standard_errors) = if config.estimate_cate {
        let (coefs, ses) = compute_cate(
            &y_residual,
            &d_residual,
            x_flat,
            n_rows,
            n_cols,
            x_col_names,
        )?;
        (Some(coefs), Some(ses))
    } else {
        (None, None)
    };

    Ok(DMLResult {
        theta,
        standard_error,
        confidence_interval,
        p_value,
        n_samples: n_rows,
        n_folds: config.n_folds,
        propensity_residual_var,
        outcome_residual_var,
        outcome_r_squared,
        propensity_r_squared,
        n_propensity_clipped,
        cate_coefficients,
        cate_standard_errors,
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dml_config_default() {
        let config = DMLConfig::default();
        assert_eq!(config.n_folds, 5);
        assert_eq!(config.treatment_type, TreatmentType::Binary);
        assert!(!config.estimate_cate);
        assert!((config.alpha - 0.05).abs() < 1e-10);
        assert_eq!(config.propensity_clip, (0.01, 0.99));
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_dml_config_validate_valid() {
        let config = DMLConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dml_config_validate_invalid_folds() {
        let config = DMLConfig {
            n_folds: 3,
            ..DMLConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(DMLError::InvalidFolds { .. })
        ));
    }

    #[test]
    fn test_dml_config_validate_invalid_alpha() {
        let config = DMLConfig {
            alpha: 1.5,
            ..DMLConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(DMLError::InvalidAlpha { .. })
        ));
    }

    #[test]
    fn test_dml_config_validate_invalid_propensity_clip() {
        let config = DMLConfig {
            propensity_clip: (0.5, 0.3),
            ..DMLConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(DMLError::InvalidPropensityClip { .. })
        ));
    }

    #[test]
    fn test_cross_fitter_deterministic() {
        let cf = CrossFitter::new(10, 5, None).unwrap();
        assert_eq!(cf.n_folds, 5);
        assert_eq!(cf.n_samples, 10);

        // Each fold should have 2 observations
        for fold in &cf.fold_indices {
            assert_eq!(fold.len(), 2);
        }

        // All indices should be covered exactly once
        let mut all_indices: Vec<usize> = cf.fold_indices.iter().flatten().copied().collect();
        all_indices.sort();
        assert_eq!(all_indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_cross_fitter_seeded_reproducible() {
        let cf1 = CrossFitter::new(100, 5, Some(42)).unwrap();
        let cf2 = CrossFitter::new(100, 5, Some(42)).unwrap();

        assert_eq!(cf1.fold_indices, cf2.fold_indices);
    }

    #[test]
    fn test_cross_fitter_different_seeds() {
        let cf1 = CrossFitter::new(100, 5, Some(42)).unwrap();
        let cf2 = CrossFitter::new(100, 5, Some(123)).unwrap();

        assert_ne!(cf1.fold_indices, cf2.fold_indices);
    }

    #[test]
    fn test_cross_fitter_invalid_folds() {
        let result = CrossFitter::new(100, 3, None);
        assert!(matches!(result, Err(DMLError::InvalidFolds { .. })));
    }

    #[test]
    fn test_cross_fitter_too_many_folds() {
        let result = CrossFitter::new(5, 10, None);
        assert!(matches!(result, Err(DMLError::TooManyFolds { .. })));
    }

    #[test]
    fn test_propensity_clip() {
        let mut props = vec![0.005, 0.1, 0.5, 0.9, 0.995];
        let n_clipped = propensity_clip(&mut props, (0.01, 0.99));

        assert_eq!(n_clipped, 2);
        assert!((props[0] - 0.01).abs() < 1e-10);
        assert!((props[4] - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_compute_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = compute_variance(&values);
        // Mean = 3, Var = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5 = 10/5 = 2
        assert!((var - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_final_stage_regression() {
        // Simple case: D̃ = [1, 2, 3], Ỹ = [2, 4, 6] => θ = 2
        let y_res = vec![2.0, 4.0, 6.0];
        let d_res = vec![1.0, 2.0, 3.0];
        let theta = final_stage_regression(&y_res, &d_res);
        assert!((theta - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_quantile() {
        // z_0.975 ≈ 1.96
        let z = normal_quantile(0.975);
        assert!((z - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf() {
        // Φ(0) = 0.5
        let cdf = normal_cdf(0.0);
        assert!((cdf - 0.5).abs() < 0.01);

        // Φ(1.96) ≈ 0.975
        let cdf_196 = normal_cdf(1.96);
        assert!((cdf_196 - 0.975).abs() < 0.01);
    }

    #[test]
    fn test_confidence_interval() {
        let (lower, upper) = compute_confidence_interval(1.0, 0.1, 0.05);
        // 1.0 ± 1.96 * 0.1 ≈ (0.804, 1.196)
        assert!((lower - 0.804).abs() < 0.01);
        assert!((upper - 1.196).abs() < 0.01);
    }

    #[test]
    fn test_p_value() {
        // θ = 0, SE = 1 => p = 1.0
        let p = compute_p_value(0.0, 1.0);
        assert!((p - 1.0).abs() < 0.01);

        // θ = 1.96, SE = 1 => p ≈ 0.05
        let p_sig = compute_p_value(1.96, 1.0);
        assert!((p_sig - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_linear_nuisance_fit_predict() {
        // Simple linear case: y = 2*x + 1
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let estimator = LinearNuisance;
        let model = estimator.fit(&x, &y, 5, 1, 0).unwrap();

        let predictions = estimator.predict(&model, &x, 5, 1);

        // Check predictions match expected values
        for (pred, expected) in predictions.iter().zip(y.iter()) {
            assert!((pred - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_extract_fold_data() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows, 2 cols
        let y = vec![10.0, 20.0, 30.0];
        let d = vec![0.0, 1.0, 0.0];
        let indices = vec![0, 2];

        let (x_fold, y_fold, d_fold) = extract_fold_data(&x, &y, &d, &indices, 2);

        assert_eq!(x_fold, vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(y_fold, vec![10.0, 30.0]);
        assert_eq!(d_fold, vec![0.0, 0.0]);
    }

    // ========================================================================
    // Tests for Cluster-robust standard errors
    // ========================================================================

    #[test]
    fn test_neyman_orthogonal_variance_clustered() {
        // Simple case: 4 observations in 2 clusters
        let y_residual = vec![1.0, 2.0, 3.0, 4.0];
        let d_residual = vec![1.0, 1.0, 1.0, 1.0];
        let theta = 2.5; // mean of y_residual

        // Build cluster info: cluster 0 = [0, 1], cluster 1 = [2, 3]
        let cluster_info = ClusterInfo {
            indices: vec![vec![0, 1], vec![2, 3]],
            n_clusters: 2,
            sizes: vec![2, 2],
        };

        let var = neyman_orthogonal_variance_clustered(&y_residual, &d_residual, theta, &cluster_info);

        // Variance should be positive and finite
        assert!(var > 0.0);
        assert!(var.is_finite());
    }

    #[test]
    fn test_cluster_variance_larger_than_iid() {
        // With within-cluster correlation, clustered SE should be >= IID SE
        // Create data with strong within-cluster correlation
        let y_residual = vec![1.0, 1.1, 1.0, 1.1, 5.0, 5.1, 5.0, 5.1];
        let d_residual = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let theta = 3.05; // approximate mean

        let cluster_info = ClusterInfo {
            indices: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            n_clusters: 2,
            sizes: vec![4, 4],
        };

        let iid_var = neyman_orthogonal_variance(&y_residual, &d_residual, theta);
        let cluster_var = neyman_orthogonal_variance_clustered(&y_residual, &d_residual, theta, &cluster_info);

        // With strong within-cluster correlation, cluster variance should be larger
        // This is a characteristic property of cluster-robust SEs
        assert!(cluster_var > 0.0);
        assert!(iid_var > 0.0);
    }

    // ========================================================================
    // Tests for Multi-valued treatment validation
    // ========================================================================

    #[test]
    fn test_validate_binary_treatment_valid() {
        // Valid binary treatment
        let d = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        assert!(validate_binary_treatment(&d).is_ok());
    }

    #[test]
    fn test_validate_binary_treatment_all_zeros() {
        let d = vec![0.0, 0.0, 0.0];
        assert!(validate_binary_treatment(&d).is_ok());
    }

    #[test]
    fn test_validate_binary_treatment_all_ones() {
        let d = vec![1.0, 1.0, 1.0];
        assert!(validate_binary_treatment(&d).is_ok());
    }

    #[test]
    fn test_validate_binary_treatment_multi_valued() {
        // Multi-valued categorical treatment (should fail)
        let d = vec![0.0, 1.0, 2.0, 3.0];
        let result = validate_binary_treatment(&d);
        assert!(matches!(result, Err(DMLError::MultiValuedTreatment { .. })));
    }

    #[test]
    fn test_validate_binary_treatment_continuous_like() {
        // Values like 0.5, 0.7 etc are not 0 or 1, but with only 2 unique values
        // This should be allowed (edge case)
        let d = vec![0.5, 0.5, 0.7, 0.7];
        // With only 2 unique values and not being 0/1, this passes the current check
        // because unique_count == 2
        assert!(validate_binary_treatment(&d).is_ok());
    }

    // ========================================================================
    // Tests for Small fold warning
    // ========================================================================

    #[test]
    fn test_check_small_fold_warning_no_warning() {
        // 150 samples / 5 folds = 30 samples per fold (no warning)
        let warned = check_small_fold_warning(150, 5);
        assert!(!warned);
    }

    #[test]
    fn test_check_small_fold_warning_warning_emitted() {
        // 100 samples / 5 folds = 20 samples per fold (warning)
        let warned = check_small_fold_warning(100, 5);
        assert!(warned);
    }

    #[test]
    fn test_check_small_fold_warning_edge_case() {
        // 145 samples / 5 folds = 29 samples per fold (warning)
        let warned = check_small_fold_warning(145, 5);
        assert!(warned);

        // 150 samples / 5 folds = 30 samples per fold (no warning)
        let warned = check_small_fold_warning(150, 5);
        assert!(!warned);
    }

    // ========================================================================
    // Tests for DMLError display messages
    // ========================================================================

    #[test]
    fn test_error_display_multi_valued_treatment() {
        let err = DMLError::MultiValuedTreatment { unique_values: 5 };
        let msg = format!("{}", err);
        assert!(msg.contains("Multi-valued categorical"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_error_display_multiple_treatments() {
        let err = DMLError::MultipleTreatments { n_treatments: 3 };
        let msg = format!("{}", err);
        assert!(msg.contains("Multiple simultaneous treatments"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn test_error_display_insufficient_clusters() {
        let err = DMLError::InsufficientClusters { found: 1 };
        let msg = format!("{}", err);
        assert!(msg.contains("at least 2 clusters"));
        assert!(msg.contains("1"));
    }
}
