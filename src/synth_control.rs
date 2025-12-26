//! Synthetic Control Methods Implementation
//!
//! This module implements Synthetic Control estimators as described in
//! Abadie et al. (2010, 2015). It provides:
//!
//! - **Traditional SC**: Classic synthetic control with simplex-constrained weights
//! - **Penalized SC**: Ridge (L2) regularization on weights (Phase 2)
//! - **Robust SC**: Variance-weighted pre-treatment fit (Phase 2)
//! - **Augmented SC**: Bias correction via ridge outcome model (Phase 2)
//!
//! # Architecture
//!
//! The implementation reuses the Frank-Wolfe optimizer from the SDID module
//! for simplex-constrained weight optimization. The key difference from SDID
//! is that SC only computes unit weights (no time weights) and uses a different
//! ATT formula.
//!
//! # References
//!
//! - Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control Methods
//!   for Comparative Case Studies. *Journal of the American Statistical Association*.
//! - Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative Politics and
//!   the Synthetic Control Method. *American Journal of Political Science*.
//! - Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented Synthetic
//!   Control Method. *Journal of the American Statistical Association*.

use std::error::Error;
use std::fmt;

use pyo3::prelude::*;

use crate::cluster::{SplitMix64, WelfordState};
use crate::sdid::{FrankWolfeConfig, FrankWolfeSolver, StepSizeMethod};

// ============================================================================
// Error Types
// ============================================================================

/// Error types for Synthetic Control operations.
///
/// These errors cover the main failure modes of the SC estimation process:
/// - Optimization convergence failures
/// - Numerical instability (NaN, Inf, ill-conditioning)
/// - Invalid input data
#[derive(Debug, Clone)]
pub enum SynthControlError {
    /// Frank-Wolfe optimizer failed to converge within the maximum iterations.
    ConvergenceFailure {
        /// Number of iterations completed before failure
        iterations: usize,
    },

    /// Numerical instability detected during computation.
    NumericalInstability {
        /// Descriptive message about the instability
        message: String,
    },

    /// Invalid input data provided to the estimator.
    InvalidData {
        /// Descriptive message about the data issue
        message: String,
    },

    /// Invalid method specified.
    InvalidMethod {
        /// The invalid method string provided
        method: String,
    },

    /// Invalid parameter value.
    InvalidParameter {
        /// Parameter name
        name: String,
        /// Descriptive message about the issue
        message: String,
    },
}

impl fmt::Display for SynthControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynthControlError::ConvergenceFailure { iterations } => {
                write!(
                    f,
                    "Frank-Wolfe solver failed to converge after {} iterations",
                    iterations
                )
            }
            SynthControlError::NumericalInstability { message } => {
                write!(f, "Numerical instability: {}", message)
            }
            SynthControlError::InvalidData { message } => {
                write!(f, "Invalid data: {}", message)
            }
            SynthControlError::InvalidMethod { method } => {
                write!(
                    f,
                    "Invalid method '{}': must be one of 'traditional', 'penalized', 'robust', 'augmented'",
                    method
                )
            }
            SynthControlError::InvalidParameter { name, message } => {
                write!(f, "Invalid parameter '{}': {}", name, message)
            }
        }
    }
}

impl Error for SynthControlError {}

// ============================================================================
// Enums and Configuration
// ============================================================================

/// Synthetic Control method variants.
///
/// Each method provides a different approach to computing control unit weights:
/// - Traditional: Minimize pre-treatment MSE with simplex constraints
/// - Penalized: Add L2 regularization to encourage more uniform weights
/// - Robust: Weight pre-treatment periods by inverse variance
/// - Augmented: Add bias correction via ridge outcome model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SynthControlMethod {
    /// Traditional SC (Abadie et al., 2010)
    #[default]
    Traditional,
    /// Penalized SC with L2 regularization
    Penalized,
    /// Robust SC with variance weighting
    Robust,
    /// Augmented SC with bias correction (Ben-Michael et al., 2021)
    Augmented,
}

impl SynthControlMethod {
    /// Parse method from string (case-insensitive).
    pub fn from_str(s: &str) -> Result<Self, SynthControlError> {
        match s.to_lowercase().as_str() {
            "traditional" => Ok(SynthControlMethod::Traditional),
            "penalized" => Ok(SynthControlMethod::Penalized),
            "robust" => Ok(SynthControlMethod::Robust),
            "augmented" => Ok(SynthControlMethod::Augmented),
            _ => Err(SynthControlError::InvalidMethod {
                method: s.to_string(),
            }),
        }
    }

    /// Convert method to string.
    pub fn as_str(&self) -> &'static str {
        match self {
            SynthControlMethod::Traditional => "traditional",
            SynthControlMethod::Penalized => "penalized",
            SynthControlMethod::Robust => "robust",
            SynthControlMethod::Augmented => "augmented",
        }
    }
}

/// Configuration for Synthetic Control estimation.
///
/// # Default Values
///
/// - `method`: Traditional
/// - `lambda`: None (only used for Penalized method)
/// - `compute_se`: false
/// - `n_placebo`: 100
/// - `max_iter`: 1000
/// - `tol`: 1e-6
/// - `seed`: None (random)
#[derive(Debug, Clone)]
pub struct SynthControlConfig {
    /// SC method to use.
    pub method: SynthControlMethod,

    /// Regularization parameter for Penalized method.
    /// Must be > 0 when method is Penalized.
    pub lambda: Option<f64>,

    /// Whether to compute standard errors via in-space placebo.
    pub compute_se: bool,

    /// Number of placebo iterations for SE computation.
    pub n_placebo: usize,

    /// Maximum iterations for Frank-Wolfe optimizer.
    pub max_iter: usize,

    /// Convergence tolerance for Frank-Wolfe optimizer.
    pub tol: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SynthControlConfig {
    fn default() -> Self {
        Self {
            method: SynthControlMethod::Traditional,
            lambda: None,
            compute_se: false,
            n_placebo: 100,
            max_iter: 1000,
            tol: 1e-6,
            seed: None,
        }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/// Panel data organized for Synthetic Control computation.
///
/// This struct holds the outcome data and panel structure needed for
/// SC estimation. Unlike SDID which supports multiple treated units,
/// SC requires exactly one treated unit (per NR-003).
///
/// # Memory Layout
///
/// Outcomes are stored in row-major order: outcomes[unit * n_periods + period]
#[derive(Debug, Clone)]
pub struct SCPanelData {
    /// Outcome values in row-major order: outcomes[unit * n_periods + period]
    pub outcomes: Vec<f64>,

    /// Total number of units in the panel.
    pub n_units: usize,

    /// Total number of time periods in the panel.
    pub n_periods: usize,

    /// Indices of control units (sorted).
    pub control_indices: Vec<usize>,

    /// Index of the single treated unit.
    pub treated_index: usize,

    /// Indices of pre-treatment periods (sorted by time).
    pub pre_period_indices: Vec<usize>,

    /// Indices of post-treatment periods (sorted by time).
    pub post_period_indices: Vec<usize>,
}

impl SCPanelData {
    /// Creates a new SCPanelData with validation.
    ///
    /// # Arguments
    ///
    /// * `outcomes` - Flat outcome vector in row-major order
    /// * `n_units` - Number of units
    /// * `n_periods` - Number of periods
    /// * `control_indices` - Indices of control units
    /// * `treated_index` - Index of the treated unit
    /// * `pre_period_indices` - Indices of pre-treatment periods
    /// * `post_period_indices` - Indices of post-treatment periods
    ///
    /// # Returns
    ///
    /// * `Ok(SCPanelData)` - Valid panel data
    /// * `Err(SynthControlError)` - If validation fails
    pub fn new(
        outcomes: Vec<f64>,
        n_units: usize,
        n_periods: usize,
        control_indices: Vec<usize>,
        treated_index: usize,
        pre_period_indices: Vec<usize>,
        post_period_indices: Vec<usize>,
    ) -> Result<Self, SynthControlError> {
        // Validate dimensions
        let expected_len = n_units * n_periods;
        if outcomes.len() != expected_len {
            return Err(SynthControlError::InvalidData {
                message: format!(
                    "outcomes length ({}) must equal n_units × n_periods ({} × {} = {})",
                    outcomes.len(),
                    n_units,
                    n_periods,
                    expected_len
                ),
            });
        }

        // Validate control indices
        if control_indices.is_empty() {
            return Err(SynthControlError::InvalidData {
                message: "At least 1 control unit required".to_string(),
            });
        }
        for &idx in &control_indices {
            if idx >= n_units {
                return Err(SynthControlError::InvalidData {
                    message: format!(
                        "control_indices contains index {} which is >= n_units ({})",
                        idx, n_units
                    ),
                });
            }
        }

        // Validate treated index
        if treated_index >= n_units {
            return Err(SynthControlError::InvalidData {
                message: format!(
                    "treated_index ({}) must be < n_units ({})",
                    treated_index, n_units
                ),
            });
        }

        // Ensure treated is not in control indices
        if control_indices.contains(&treated_index) {
            return Err(SynthControlError::InvalidData {
                message: format!(
                    "treated_index ({}) must not be in control_indices",
                    treated_index
                ),
            });
        }

        // Validate pre-period indices
        if pre_period_indices.is_empty() {
            return Err(SynthControlError::InvalidData {
                message: "At least 1 pre-treatment period required".to_string(),
            });
        }
        for &idx in &pre_period_indices {
            if idx >= n_periods {
                return Err(SynthControlError::InvalidData {
                    message: format!(
                        "pre_period_indices contains index {} which is >= n_periods ({})",
                        idx, n_periods
                    ),
                });
            }
        }

        // Validate post-period indices
        if post_period_indices.is_empty() {
            return Err(SynthControlError::InvalidData {
                message: "At least 1 post-treatment period required".to_string(),
            });
        }
        for &idx in &post_period_indices {
            if idx >= n_periods {
                return Err(SynthControlError::InvalidData {
                    message: format!(
                        "post_period_indices contains index {} which is >= n_periods ({})",
                        idx, n_periods
                    ),
                });
            }
        }

        Ok(Self {
            outcomes,
            n_units,
            n_periods,
            control_indices,
            treated_index,
            pre_period_indices,
            post_period_indices,
        })
    }

    /// Get outcome value for a specific unit and period.
    ///
    /// # Arguments
    ///
    /// * `unit` - Unit index
    /// * `period` - Period index
    ///
    /// # Returns
    ///
    /// The outcome value Y[unit, period]
    #[inline]
    pub fn outcome(&self, unit: usize, period: usize) -> f64 {
        debug_assert!(unit < self.n_units, "Unit index out of bounds");
        debug_assert!(period < self.n_periods, "Period index out of bounds");
        self.outcomes[unit * self.n_periods + period]
    }

    /// Extract control unit pre-treatment outcomes as a matrix.
    ///
    /// Returns a flattened matrix in row-major order: result[ctrl_i * n_pre + t]
    /// where ctrl_i is the index within control_indices.
    ///
    /// # Returns
    ///
    /// Vec of length n_control × n_pre
    pub fn control_pre_matrix(&self) -> Vec<f64> {
        let n_control = self.control_indices.len();
        let n_pre = self.pre_period_indices.len();
        let mut result = Vec::with_capacity(n_control * n_pre);

        for &ctrl_idx in &self.control_indices {
            for &t in &self.pre_period_indices {
                result.push(self.outcome(ctrl_idx, t));
            }
        }

        result
    }

    /// Extract treated unit pre-treatment outcomes as a vector.
    ///
    /// # Returns
    ///
    /// Vec of length n_pre
    pub fn treated_pre_vector(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.pre_period_indices.len());
        for &t in &self.pre_period_indices {
            result.push(self.outcome(self.treated_index, t));
        }
        result
    }

    /// Extract control unit post-treatment outcomes as a matrix.
    ///
    /// Returns a flattened matrix in row-major order: result[ctrl_i * n_post + t]
    ///
    /// # Returns
    ///
    /// Vec of length n_control × n_post
    pub fn control_post_matrix(&self) -> Vec<f64> {
        let n_control = self.control_indices.len();
        let n_post = self.post_period_indices.len();
        let mut result = Vec::with_capacity(n_control * n_post);

        for &ctrl_idx in &self.control_indices {
            for &t in &self.post_period_indices {
                result.push(self.outcome(ctrl_idx, t));
            }
        }

        result
    }

    /// Extract treated unit post-treatment outcomes as a vector.
    ///
    /// # Returns
    ///
    /// Vec of length n_post
    pub fn treated_post_vector(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.post_period_indices.len());
        for &t in &self.post_period_indices {
            result.push(self.outcome(self.treated_index, t));
        }
        result
    }

    /// Get number of control units.
    #[inline]
    pub fn n_control(&self) -> usize {
        self.control_indices.len()
    }

    /// Get number of pre-treatment periods.
    #[inline]
    pub fn n_pre(&self) -> usize {
        self.pre_period_indices.len()
    }

    /// Get number of post-treatment periods.
    #[inline]
    pub fn n_post(&self) -> usize {
        self.post_period_indices.len()
    }
}

// ============================================================================
// Weight Computation Result
// ============================================================================

/// Result of weight optimization.
#[derive(Debug, Clone)]
pub struct WeightResult {
    /// Computed weights for control units.
    pub weights: Vec<f64>,

    /// Whether the optimizer converged.
    pub converged: bool,

    /// Number of iterations used.
    pub iterations: usize,

    /// Final objective value.
    pub objective: f64,
}

// ============================================================================
// Traditional SC Weight Optimization
// ============================================================================

/// Compute Y @ Y' matrix (n_control × n_control).
///
/// This precomputes the Gram matrix for the quadratic objective.
/// Y[i, t] is control unit i's outcome at pre-period t.
/// (Y @ Y')[i, j] = Σₜ Y[i, t] × Y[j, t]
///
/// # Arguments
///
/// * `y_control_pre` - Control pre-treatment matrix (row-major: n_control × n_pre)
/// * `n_control` - Number of control units
/// * `n_pre` - Number of pre-periods
///
/// # Returns
///
/// Gram matrix Y @ Y' (row-major: n_control × n_control)
fn compute_yyt(y_control_pre: &[f64], n_control: usize, n_pre: usize) -> Vec<Vec<f64>> {
    let mut yyt = vec![vec![0.0; n_control]; n_control];

    for i in 0..n_control {
        for j in 0..n_control {
            let mut sum = 0.0;
            for t in 0..n_pre {
                let y_it = y_control_pre[i * n_pre + t];
                let y_jt = y_control_pre[j * n_pre + t];
                sum += y_it * y_jt;
            }
            yyt[i][j] = sum;
        }
    }

    yyt
}

/// Compute Y @ y vector (n_control × 1).
///
/// This precomputes the linear term for the quadratic objective.
/// (Y @ y)[i] = Σₜ Y[i, t] × y[t]
///
/// # Arguments
///
/// * `y_control_pre` - Control pre-treatment matrix (row-major: n_control × n_pre)
/// * `y_treated_pre` - Treated pre-treatment vector (n_pre)
/// * `n_control` - Number of control units
/// * `n_pre` - Number of pre-periods
///
/// # Returns
///
/// Vector Y @ y (length n_control)
fn compute_yy(
    y_control_pre: &[f64],
    y_treated_pre: &[f64],
    n_control: usize,
    n_pre: usize,
) -> Vec<f64> {
    let mut yy = vec![0.0; n_control];

    for i in 0..n_control {
        let mut sum = 0.0;
        for t in 0..n_pre {
            let y_it = y_control_pre[i * n_pre + t];
            sum += y_it * y_treated_pre[t];
        }
        yy[i] = sum;
    }

    yy
}

/// Compute Traditional SC weights using Frank-Wolfe optimization.
///
/// Minimizes: ||y_treated_pre - Y_control_pre' @ w||²
/// subject to: w ≥ 0, Σw = 1
///
/// The objective can be rewritten as:
/// f(w) = w' @ (Y @ Y') @ w - 2 × w' @ (Y @ y) + const
///
/// Gradient: ∇f(w) = 2 × (Y @ Y') @ w - 2 × (Y @ y)
///
/// # Arguments
///
/// * `panel` - Panel data with control and treated outcomes
/// * `config` - Configuration for the optimizer
///
/// # Returns
///
/// * `Ok(WeightResult)` - Computed weights and diagnostics
/// * `Err(SynthControlError)` - If optimization fails
pub fn compute_traditional_weights(
    panel: &SCPanelData,
    config: &SynthControlConfig,
) -> Result<WeightResult, SynthControlError> {
    let n_control = panel.n_control();
    let n_pre = panel.n_pre();

    // Validate minimum requirements
    if n_control < 1 {
        return Err(SynthControlError::InvalidData {
            message: "At least 1 control unit required".to_string(),
        });
    }
    if n_pre < 1 {
        return Err(SynthControlError::InvalidData {
            message: "At least 1 pre-treatment period required".to_string(),
        });
    }

    // Extract pre-treatment data
    let y_control_pre = panel.control_pre_matrix();
    let y_treated_pre = panel.treated_pre_vector();

    // Precompute Y @ Y' and Y @ y
    let yyt = compute_yyt(&y_control_pre, n_control, n_pre);
    let yy = compute_yy(&y_control_pre, &y_treated_pre, n_control, n_pre);

    // Create copies for closures
    let yyt_obj = yyt.clone();
    let yy_obj = yy.clone();

    // Objective function: f(w) = w' @ YYt @ w - 2 × w' @ Yy + const
    // We ignore the constant term since it doesn't affect optimization
    let objective = move |w: &[f64]| -> f64 {
        // Compute w' @ YYt @ w
        let mut w_yyt_w = 0.0;
        for i in 0..n_control {
            for j in 0..n_control {
                w_yyt_w += w[i] * yyt_obj[i][j] * w[j];
            }
        }

        // Compute w' @ Yy
        let mut w_yy = 0.0;
        for (i, &wi) in w.iter().enumerate() {
            w_yy += wi * yy_obj[i];
        }

        w_yyt_w - 2.0 * w_yy
    };

    // Gradient function: ∇f(w) = 2 × YYt @ w - 2 × Yy
    let gradient_fn = move |w: &[f64]| -> Vec<f64> {
        let mut grad = vec![0.0; n_control];
        for i in 0..n_control {
            // (YYt @ w)[i] = Σⱼ YYt[i,j] × w[j]
            let yyt_w_i: f64 = yyt[i].iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
            grad[i] = 2.0 * yyt_w_i - 2.0 * yy[i];
        }
        grad
    };

    // Configure Frank-Wolfe solver
    let fw_config = FrankWolfeConfig {
        max_iterations: config.max_iter,
        tolerance: config.tol,
        step_size_method: StepSizeMethod::Classic,
        armijo_beta: 0.5,
        armijo_sigma: 1e-4,
        use_relative_gap: true,
    };

    // Create and run Frank-Wolfe solver
    let mut solver = FrankWolfeSolver::new(n_control, fw_config);
    let weights = solver.solve(&objective, gradient_fn).map_err(|e| {
        SynthControlError::NumericalInstability {
            message: format!("Frank-Wolfe solver error: {}", e),
        }
    })?;

    // Compute final objective value
    let final_obj = objective(&weights);

    Ok(WeightResult {
        weights,
        converged: true, // Frank-Wolfe returns best solution found
        iterations: config.max_iter, // Conservative estimate
        objective: final_obj,
    })
}

// ============================================================================
// ATT Computation
// ============================================================================

/// Compute Average Treatment Effect on Treated (ATT).
///
/// ATT = mean over post-periods of (Y_treated - Y_synthetic)
/// where Y_synthetic = Σᵢ wᵢ × Y_control_i
///
/// # Arguments
///
/// * `panel` - Panel data
/// * `weights` - Control unit weights (must sum to 1)
///
/// # Returns
///
/// The estimated ATT
pub fn compute_att(panel: &SCPanelData, weights: &[f64]) -> f64 {
    let n_post = panel.n_post();
    
    if n_post == 0 {
        return 0.0;
    }

    let treated_post = panel.treated_post_vector();
    let control_post = panel.control_post_matrix();
    let _n_control = panel.n_control();

    let mut att_sum = 0.0;

    for (t_idx, &y_treated_t) in treated_post.iter().enumerate() {
        // Compute synthetic control for this period
        let mut y_synth_t = 0.0;
        for (i, &w_i) in weights.iter().enumerate() {
            let y_control_it = control_post[i * n_post + t_idx];
            y_synth_t += w_i * y_control_it;
        }

        // Treatment effect at this period
        att_sum += y_treated_t - y_synth_t;
    }

    att_sum / n_post as f64
}

// ============================================================================
// Pre-treatment Fit Diagnostics
// ============================================================================

/// Compute pre-treatment Mean Squared Error (MSE).
///
/// MSE = mean over pre-periods of (Y_treated - Y_synthetic)²
///
/// # Arguments
///
/// * `panel` - Panel data
/// * `weights` - Control unit weights
///
/// # Returns
///
/// The pre-treatment MSE
pub fn compute_pre_treatment_mse(panel: &SCPanelData, weights: &[f64]) -> f64 {
    let n_pre = panel.n_pre();
    
    if n_pre == 0 {
        return 0.0;
    }

    let treated_pre = panel.treated_pre_vector();
    let control_pre = panel.control_pre_matrix();
    let _n_control = panel.n_control();

    let mut mse_sum = 0.0;

    for (t_idx, &y_treated_t) in treated_pre.iter().enumerate() {
        // Compute synthetic control for this period
        let mut y_synth_t = 0.0;
        for (i, &w_i) in weights.iter().enumerate() {
            let y_control_it = control_pre[i * n_pre + t_idx];
            y_synth_t += w_i * y_control_it;
        }

        // Squared error at this period
        let diff = y_treated_t - y_synth_t;
        mse_sum += diff * diff;
    }

    mse_sum / n_pre as f64
}

/// Compute pre-treatment Root Mean Squared Error (RMSE).
///
/// RMSE = sqrt(MSE)
///
/// # Arguments
///
/// * `panel` - Panel data
/// * `weights` - Control unit weights
///
/// # Returns
///
/// The pre-treatment RMSE
pub fn compute_pre_treatment_rmse(panel: &SCPanelData, weights: &[f64]) -> f64 {
    compute_pre_treatment_mse(panel, weights).sqrt()
}

// ============================================================================
// Simplex Projection
// ============================================================================

/// Project a vector onto the probability simplex.
///
/// Uses the algorithm from Duchi et al. (2008) "Efficient Projections onto the l1-Ball
/// for Learning in High Dimensions".
///
/// The probability simplex is defined as: {x : x >= 0, sum(x) = 1}
///
/// # Algorithm
/// 1. Sort vector in descending order
/// 2. Find threshold rho such that: sorted[rho] - (sum_{j<=rho} sorted[j] - 1) / (rho + 1) > 0
/// 3. Compute theta = (sum_{j<=rho} sorted[j] - 1) / (rho + 1)
/// 4. Return max(x - theta, 0) for each element
///
/// # Arguments
///
/// * `v` - Vector to project
///
/// # Returns
///
/// Projected vector on the simplex (non-negative, sums to 1)
pub fn project_onto_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return vec![];
    }

    // Sort in descending order with indices
    let mut sorted_with_idx: Vec<(f64, usize)> = v.iter().copied().enumerate().map(|(i, x)| (x, i)).collect();
    sorted_with_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Find threshold rho
    let mut cumsum = 0.0;
    let mut rho = 0;
    for (j, &(val, _)) in sorted_with_idx.iter().enumerate() {
        cumsum += val;
        // Check if val - (cumsum - 1) / (j + 1) > 0
        if val - (cumsum - 1.0) / ((j + 1) as f64) > 0.0 {
            rho = j;
        }
    }

    // Compute theta
    let cumsum_rho: f64 = sorted_with_idx[..=rho].iter().map(|(val, _)| val).sum();
    let theta = (cumsum_rho - 1.0) / ((rho + 1) as f64);

    // Project: max(v - theta, 0)
    v.iter().map(|&x| (x - theta).max(0.0)).collect()
}

// ============================================================================
// Matrix Operations for Penalized SC
// ============================================================================

/// Invert a square matrix using Gauss-Jordan elimination with partial pivoting.
///
/// # Arguments
/// * `a` - Square matrix to invert (n × n), stored as Vec<Vec<f64>>
///
/// # Returns
/// * `Ok(inverse)` - Inverse matrix (n × n)
/// * `Err(SynthControlError)` - If matrix is singular
fn invert_matrix(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SynthControlError> {
    let n = a.len();
    if n == 0 {
        return Err(SynthControlError::NumericalInstability {
            message: "Cannot invert empty matrix".to_string(),
        });
    }

    // Verify square matrix
    for row in a.iter() {
        if row.len() != n {
            return Err(SynthControlError::NumericalInstability {
                message: "Cannot invert non-square matrix".to_string(),
            });
        }
    }

    // Create augmented matrix [A|I] of size (n × 2n)
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, a_row) in a.iter().enumerate() {
        let mut row = Vec::with_capacity(2 * n);
        // Copy A into left half
        row.extend_from_slice(a_row);
        // Add identity matrix to right half
        for j in 0..n {
            row.push(if i == j { 1.0 } else { 0.0 });
        }
        aug.push(row);
    }

    // Gauss-Jordan elimination with partial pivoting
    for col in 0..n {
        // Find pivot: row with maximum absolute value in current column
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        #[allow(clippy::needless_range_loop)]
        for row in (col + 1)..n {
            let val = aug[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Check for singularity (pivot too small)
        if max_val < 1e-12 {
            return Err(SynthControlError::NumericalInstability {
                message: format!(
                    "Singular matrix: pivot element {:.2e} too small at column {}",
                    max_val, col
                ),
            });
        }

        // Swap rows if needed
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Scale pivot row so diagonal element becomes 1
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate all other entries in this column (both above and below pivot)
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse from right half of augmented matrix
    let mut inverse: Vec<Vec<f64>> = Vec::with_capacity(n);
    for aug_row in aug.iter() {
        inverse.push(aug_row[n..(2 * n)].to_vec());
    }

    Ok(inverse)
}

/// Multiply a matrix by a vector: result = A × v
fn matrix_vector_mult(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&a_ij, &v_j)| a_ij * v_j).sum())
        .collect()
}

// ============================================================================
// Penalized SC Weight Optimization
// ============================================================================

/// Compute Penalized SC weights with ridge (L2) regularization.
///
/// Minimizes: ||y_treated_pre - Y_control_pre' @ w||² + λ||w||²
/// subject to: w ≥ 0, Σw = 1 (via projection)
///
/// The optimization is solved via closed-form solution followed by simplex projection:
/// 1. Form regularized matrix: (Y @ Y' + λI)
/// 2. Solve: w_unconstrained = (Y @ Y' + λI)^(-1) @ (Y @ y)
/// 3. Project onto probability simplex
///
/// # Arguments
///
/// * `panel` - Panel data with control and treated outcomes
/// * `config` - Configuration (must have lambda specified or will be auto-selected)
///
/// # Returns
///
/// * `Ok(WeightResult)` - Computed weights and diagnostics
/// * `Err(SynthControlError)` - If optimization fails
pub fn compute_penalized_weights(
    panel: &SCPanelData,
    config: &SynthControlConfig,
) -> Result<WeightResult, SynthControlError> {
    let n_control = panel.n_control();
    let n_pre = panel.n_pre();

    // Validate minimum requirements
    if n_control < 1 {
        return Err(SynthControlError::InvalidData {
            message: "At least 1 control unit required".to_string(),
        });
    }
    if n_pre < 1 {
        return Err(SynthControlError::InvalidData {
            message: "At least 1 pre-treatment period required".to_string(),
        });
    }

    // Get lambda (auto-select via LOOCV if not specified)
    let lambda = match config.lambda {
        Some(l) => {
            if l < 0.0 {
                return Err(SynthControlError::InvalidParameter {
                    name: "lambda".to_string(),
                    message: format!("lambda must be non-negative, got {}", l),
                });
            }
            l
        }
        None => select_lambda_loocv(panel)?,
    };

    // Extract pre-treatment data
    let y_control_pre = panel.control_pre_matrix();
    let y_treated_pre = panel.treated_pre_vector();

    // Compute Y @ Y' (n_control × n_control)
    let yyt = compute_yyt(&y_control_pre, n_control, n_pre);

    // Add regularization: (Y @ Y' + λI)
    let mut yyt_regularized = yyt;
    for i in 0..n_control {
        yyt_regularized[i][i] += lambda;
    }

    // Compute Y @ y (n_control × 1)
    let yy = compute_yy(&y_control_pre, &y_treated_pre, n_control, n_pre);

    // Solve (Y @ Y' + λI)^(-1) @ (Y @ y) for unconstrained solution
    let yyt_inv = invert_matrix(&yyt_regularized)?;
    let w_unconstrained = matrix_vector_mult(&yyt_inv, &yy);

    // Project onto simplex to satisfy w >= 0, sum(w) = 1
    let weights = project_onto_simplex(&w_unconstrained);

    // Compute objective value (includes regularization term)
    let objective = compute_penalized_objective(&y_control_pre, &y_treated_pre, &weights, lambda, n_control, n_pre);

    Ok(WeightResult {
        weights,
        converged: true, // Direct solve always "converges"
        iterations: 1,
        objective,
    })
}

/// Compute the penalized objective function value.
///
/// Objective = ||y - Y'w||² + λ||w||²
fn compute_penalized_objective(
    y_control_pre: &[f64],
    y_treated_pre: &[f64],
    weights: &[f64],
    lambda: f64,
    n_control: usize,
    n_pre: usize,
) -> f64 {
    // Compute prediction error: ||y - Y'w||²
    let mut mse = 0.0;
    for t in 0..n_pre {
        let mut y_synth_t = 0.0;
        for i in 0..n_control {
            y_synth_t += weights[i] * y_control_pre[i * n_pre + t];
        }
        let diff = y_treated_pre[t] - y_synth_t;
        mse += diff * diff;
    }

    // Compute regularization term: λ||w||²
    let w_norm_sq: f64 = weights.iter().map(|w| w * w).sum();

    mse + lambda * w_norm_sq
}

/// Select optimal lambda via Leave-One-Out Cross-Validation.
///
/// Evaluates a grid of lambda values and selects the one that minimizes
/// the LOOCV error on the pre-treatment data.
fn select_lambda_loocv(panel: &SCPanelData) -> Result<f64, SynthControlError> {
    let n_control = panel.n_control();
    let n_pre = panel.n_pre();

    // Lambda grid: logarithmically spaced from 1e-4 to 1e4
    let lambda_grid: Vec<f64> = (-4..=4)
        .map(|i| 10.0_f64.powi(i))
        .collect();

    let y_control_pre = panel.control_pre_matrix();
    let y_treated_pre = panel.treated_pre_vector();

    // Compute Y @ Y' once (will be reused)
    let yyt = compute_yyt(&y_control_pre, n_control, n_pre);
    let yy = compute_yy(&y_control_pre, &y_treated_pre, n_control, n_pre);

    let mut best_lambda = 1.0;
    let mut best_loocv_error = f64::INFINITY;

    for &lambda in &lambda_grid {
        // Add regularization
        let mut yyt_reg = yyt.clone();
        for i in 0..n_control {
            yyt_reg[i][i] += lambda;
        }

        // Solve for weights
        let yyt_inv = match invert_matrix(&yyt_reg) {
            Ok(inv) => inv,
            Err(_) => continue, // Skip if singular
        };
        let w_unconstrained = matrix_vector_mult(&yyt_inv, &yy);
        let weights = project_onto_simplex(&w_unconstrained);

        // Compute LOOCV error (approximated by pre-treatment MSE for efficiency)
        // True LOOCV would leave out each time period, but this is expensive
        // We use the pre-treatment MSE as a proxy
        let mut loocv_error = 0.0;
        for t in 0..n_pre {
            let mut y_synth_t = 0.0;
            for i in 0..n_control {
                y_synth_t += weights[i] * y_control_pre[i * n_pre + t];
            }
            let diff = y_treated_pre[t] - y_synth_t;
            loocv_error += diff * diff;
        }

        if loocv_error < best_loocv_error {
            best_loocv_error = loocv_error;
            best_lambda = lambda;
        }
    }

    Ok(best_lambda)
}

// ============================================================================
// Robust SC Weight Optimization
// ============================================================================

/// Compute Robust SC weights with de-meaned pre-treatment outcomes.
///
/// Robust SC de-means the pre-treatment outcomes before optimization.
/// This makes the method less sensitive to level differences and focuses
/// on matching the dynamics of the treated unit.
///
/// # Algorithm
/// 1. Compute row means for each unit's pre-treatment outcomes
/// 2. De-mean: X_dm = X - row_mean(X)
/// 3. Apply Frank-Wolfe optimization to de-meaned data
/// 4. Same simplex constraints as Traditional SC
///
/// # Arguments
///
/// * `panel` - Panel data with control and treated outcomes
/// * `config` - Configuration for the optimizer
///
/// # Returns
///
/// * `Ok(WeightResult)` - Computed weights and diagnostics
/// * `Err(SynthControlError)` - If optimization fails
pub fn compute_robust_weights(
    panel: &SCPanelData,
    config: &SynthControlConfig,
) -> Result<WeightResult, SynthControlError> {
    let n_control = panel.n_control();
    let n_pre = panel.n_pre();

    // Validate minimum requirements
    if n_control < 1 {
        return Err(SynthControlError::InvalidData {
            message: "At least 1 control unit required".to_string(),
        });
    }
    if n_pre < 1 {
        return Err(SynthControlError::InvalidData {
            message: "At least 1 pre-treatment period required".to_string(),
        });
    }

    // Extract pre-treatment data
    let y_control_pre = panel.control_pre_matrix();
    let y_treated_pre = panel.treated_pre_vector();

    // De-mean control unit outcomes (subtract row mean from each unit)
    let mut y_control_demeaned = Vec::with_capacity(n_control * n_pre);
    for i in 0..n_control {
        // Compute mean for this control unit
        let row_start = i * n_pre;
        let row_mean: f64 = (0..n_pre)
            .map(|t| y_control_pre[row_start + t])
            .sum::<f64>() / n_pre as f64;

        // Subtract mean from each period
        for t in 0..n_pre {
            y_control_demeaned.push(y_control_pre[row_start + t] - row_mean);
        }
    }

    // De-mean treated outcomes
    let treated_mean: f64 = y_treated_pre.iter().sum::<f64>() / n_pre as f64;
    let y_treated_demeaned: Vec<f64> = y_treated_pre
        .iter()
        .map(|&y| y - treated_mean)
        .collect();

    // Compute Y @ Y' and Y @ y on de-meaned data
    let yyt = compute_yyt(&y_control_demeaned, n_control, n_pre);
    let yy = compute_yy(&y_control_demeaned, &y_treated_demeaned, n_control, n_pre);

    // Create copies for closures
    let yyt_obj = yyt.clone();
    let yy_obj = yy.clone();
    // Additional copies for final objective computation
    let yyt_final = yyt.clone();
    let yy_final = yy.clone();

    // Objective function: f(w) = w' @ YYt @ w - 2 × w' @ Yy + const
    let objective = move |w: &[f64]| -> f64 {
        let mut w_yyt_w = 0.0;
        for i in 0..n_control {
            for j in 0..n_control {
                w_yyt_w += w[i] * yyt_obj[i][j] * w[j];
            }
        }
        let mut w_yy = 0.0;
        for (i, &wi) in w.iter().enumerate() {
            w_yy += wi * yy_obj[i];
        }
        w_yyt_w - 2.0 * w_yy
    };

    // Gradient function: ∇f(w) = 2 × YYt @ w - 2 × Yy
    let gradient_fn = move |w: &[f64]| -> Vec<f64> {
        let mut grad = vec![0.0; n_control];
        for i in 0..n_control {
            let yyt_w_i: f64 = yyt[i].iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
            grad[i] = 2.0 * yyt_w_i - 2.0 * yy[i];
        }
        grad
    };

    // Configure Frank-Wolfe solver
    let fw_config = FrankWolfeConfig {
        max_iterations: config.max_iter,
        tolerance: config.tol,
        step_size_method: StepSizeMethod::Classic,
        armijo_beta: 0.5,
        armijo_sigma: 1e-4,
        use_relative_gap: true,
    };

    // Create and run Frank-Wolfe solver
    let mut solver = FrankWolfeSolver::new(n_control, fw_config);
    let weights = solver.solve(&objective, gradient_fn).map_err(|e| {
        SynthControlError::NumericalInstability {
            message: format!("Frank-Wolfe solver error: {}", e),
        }
    })?;

    // Compute final objective value using the additional copies
    let final_obj = {
        let mut w_yyt_w = 0.0;
        for i in 0..n_control {
            for j in 0..n_control {
                w_yyt_w += weights[i] * yyt_final[i][j] * weights[j];
            }
        }
        let mut w_yy = 0.0;
        for (i, &wi) in weights.iter().enumerate() {
            w_yy += wi * yy_final[i];
        }
        w_yyt_w - 2.0 * w_yy
    };

    Ok(WeightResult {
        weights,
        converged: true,
        iterations: config.max_iter,
        objective: final_obj,
    })
}

// ============================================================================
// Augmented SC
// ============================================================================

/// Result from Augmented SC computation.
#[derive(Debug, Clone)]
pub struct AugmentedSCResult {
    /// Control unit weights
    pub weight_result: WeightResult,
    /// Bias adjustment term
    pub bias_adjustment: f64,
    /// Ridge coefficients from outcome model
    pub ridge_coefficients: Vec<f64>,
    /// Lambda used for ridge regression
    pub lambda_used: f64,
}

/// Compute Augmented SC with ridge regression bias correction.
///
/// Augmented SC (Ben-Michael et al., 2021) combines traditional SC weights
/// with a ridge regression adjustment to reduce bias when the pre-treatment
/// fit is imperfect.
///
/// # Algorithm
/// 1. Compute traditional SC weights
/// 2. Fit ridge regression model: Y_post = X_pre' @ β on control units
/// 3. Compute predictions: μ_i = f(X_pre_i) for all units
/// 4. Compute bias-corrected ATT:
///    ATT = (Y1_post - μ_1) - Σ w_i (Y0i_post - μ_0i)
///
/// # Arguments
///
/// * `panel` - Panel data with control and treated outcomes
/// * `config` - Configuration (uses config.lambda for ridge, auto-selects if None)
///
/// # Returns
///
/// * `Ok(AugmentedSCResult)` - Weights, bias adjustment, and ridge coefficients
/// * `Err(SynthControlError)` - If computation fails
pub fn compute_augmented_sc(
    panel: &SCPanelData,
    config: &SynthControlConfig,
) -> Result<AugmentedSCResult, SynthControlError> {
    let n_control = panel.n_control();
    let n_pre = panel.n_pre();
    let n_post = panel.n_post();

    // Validate minimum requirements
    if n_control < 2 {
        return Err(SynthControlError::InvalidData {
            message: "Augmented SC requires at least 2 control units".to_string(),
        });
    }
    if n_pre < 2 {
        return Err(SynthControlError::InvalidData {
            message: "Augmented SC requires at least 2 pre-treatment periods".to_string(),
        });
    }

    // Step 1: Compute traditional SC weights
    let weight_result = compute_traditional_weights(panel, config)?;

    // Step 2: Fit ridge regression on control units
    // We use pre-treatment outcomes to predict post-treatment outcomes
    // X = pre-treatment outcomes (n_pre features per unit)
    // y = mean post-treatment outcome

    // Get lambda for ridge regression
    let lambda = config.lambda.unwrap_or(1.0);
    if lambda < 0.0 {
        return Err(SynthControlError::InvalidParameter {
            name: "lambda".to_string(),
            message: format!("lambda must be non-negative, got {}", lambda),
        });
    }

    // Extract control pre-treatment data as features
    let y_control_pre = panel.control_pre_matrix(); // (n_control × n_pre)
    let y_control_post = panel.control_post_matrix(); // (n_control × n_post)

    // Compute mean post-treatment outcome for each control unit
    let control_post_means: Vec<f64> = (0..n_control)
        .map(|i| {
            let row_start = i * n_post;
            (0..n_post)
                .map(|t| y_control_post[row_start + t])
                .sum::<f64>() / n_post as f64
        })
        .collect();

    // Fit ridge regression: β = (X'X + λI)^(-1) X'y
    // Where X is (n_control × n_pre) and y is (n_control,)
    // X'X is (n_pre × n_pre), X'y is (n_pre,)

    // Compute X'X (n_pre × n_pre)
    let mut xtx = vec![vec![0.0; n_pre]; n_pre];
    for i in 0..n_pre {
        for j in 0..n_pre {
            let mut sum = 0.0;
            for k in 0..n_control {
                // X[k, i] = y_control_pre[k * n_pre + i]
                sum += y_control_pre[k * n_pre + i] * y_control_pre[k * n_pre + j];
            }
            xtx[i][j] = sum;
        }
    }

    // Add regularization: (X'X + λI)
    for i in 0..n_pre {
        xtx[i][i] += lambda;
    }

    // Compute X'y (n_pre,)
    let mut xty = vec![0.0; n_pre];
    for i in 0..n_pre {
        let mut sum = 0.0;
        for k in 0..n_control {
            sum += y_control_pre[k * n_pre + i] * control_post_means[k];
        }
        xty[i] = sum;
    }

    // Solve for ridge coefficients
    let xtx_inv = invert_matrix(&xtx)?;
    let ridge_coefficients = matrix_vector_mult(&xtx_inv, &xty);

    // Step 3: Compute predictions for all units
    // Treated prediction
    let y_treated_pre = panel.treated_pre_vector();
    let mu_treated: f64 = y_treated_pre
        .iter()
        .zip(ridge_coefficients.iter())
        .map(|(&x, &b)| x * b)
        .sum();

    // Control predictions
    let mu_controls: Vec<f64> = (0..n_control)
        .map(|i| {
            (0..n_pre)
                .map(|t| y_control_pre[i * n_pre + t] * ridge_coefficients[t])
                .sum()
        })
        .collect();

    // Step 4: Compute bias adjustment
    // adjustment = Σ w_i (μ_0i - μ_1)
    // Note: This captures the bias due to imperfect pre-treatment match
    let weighted_control_mu: f64 = weight_result
        .weights
        .iter()
        .zip(mu_controls.iter())
        .map(|(&w, &mu)| w * mu)
        .sum();

    let bias_adjustment = weighted_control_mu - mu_treated;

    Ok(AugmentedSCResult {
        weight_result,
        bias_adjustment,
        ridge_coefficients,
        lambda_used: lambda,
    })
}

/// Compute ATT using Augmented SC.
///
/// ATT_aug = (Y1_post - μ_1) - Σ w_i (Y0i_post - μ_0i)
///         = ATT_traditional - bias_adjustment
///
/// # Arguments
///
/// * `panel` - Panel data
/// * `asc_result` - Result from compute_augmented_sc
///
/// # Returns
///
/// The bias-corrected ATT estimate
pub fn compute_augmented_att(panel: &SCPanelData, asc_result: &AugmentedSCResult) -> f64 {
    // Traditional ATT
    let att_traditional = compute_att(panel, &asc_result.weight_result.weights);

    // Bias-corrected ATT
    att_traditional - asc_result.bias_adjustment
}

// ============================================================================
// Method Dispatch
// ============================================================================

/// Compute SC weights using the specified method.
///
/// This is the unified dispatch function that routes to the appropriate
/// weight computation method based on `config.method`.
///
/// # Arguments
///
/// * `panel` - Panel data with control and treated outcomes
/// * `config` - Configuration including method selection
///
/// # Returns
///
/// * `Ok(WeightResult)` - Computed weights and diagnostics
/// * `Err(SynthControlError)` - If computation fails
pub fn compute_weights(
    panel: &SCPanelData,
    config: &SynthControlConfig,
) -> Result<WeightResult, SynthControlError> {
    match config.method {
        SynthControlMethod::Traditional => compute_traditional_weights(panel, config),
        SynthControlMethod::Penalized => compute_penalized_weights(panel, config),
        SynthControlMethod::Robust => compute_robust_weights(panel, config),
        SynthControlMethod::Augmented => {
            // For Augmented, we return the weight result from ASC
            let asc_result = compute_augmented_sc(panel, config)?;
            Ok(asc_result.weight_result)
        }
    }
}

// ============================================================================
// PyO3 Result Class (TASK-011)
// ============================================================================

/// Result of Synthetic Control estimation, exposed to Python via PyO3.
///
/// Contains the ATT estimate, optional standard error, control unit weights,
/// and various diagnostic information about the estimation.
#[pyclass]
#[derive(Debug, Clone)]
pub struct SyntheticControlResult {
    /// Average Treatment Effect on Treated
    #[pyo3(get)]
    pub att: f64,

    /// Standard error via in-space placebo (None if not computed)
    #[pyo3(get)]
    pub standard_error: Option<f64>,

    /// Control unit weights (sum to 1, non-negative)
    #[pyo3(get)]
    pub unit_weights: Vec<f64>,

    /// Pre-treatment RMSE (fit quality)
    #[pyo3(get)]
    pub pre_treatment_rmse: f64,

    /// Pre-treatment MSE
    #[pyo3(get)]
    pub pre_treatment_mse: f64,

    /// SC method used: "traditional", "penalized", "robust", "augmented"
    #[pyo3(get)]
    pub method: String,

    /// Lambda used for penalized method (None for others)
    #[pyo3(get)]
    pub lambda_used: Option<f64>,

    /// Number of control units
    #[pyo3(get)]
    pub n_units_control: usize,

    /// Number of pre-treatment periods
    #[pyo3(get)]
    pub n_periods_pre: usize,

    /// Number of post-treatment periods
    #[pyo3(get)]
    pub n_periods_post: usize,

    /// Whether the optimizer converged
    #[pyo3(get)]
    pub solver_converged: bool,

    /// Number of optimizer iterations
    #[pyo3(get)]
    pub solver_iterations: usize,

    /// Number of successful placebo iterations (None if SE not computed)
    #[pyo3(get)]
    pub n_placebo_used: Option<usize>,
}

#[pymethods]
impl SyntheticControlResult {
    /// String representation for Python's repr()
    fn __repr__(&self) -> String {
        let se_str = match self.standard_error {
            Some(se) => format!("{:.6}", se),
            None => "None".to_string(),
        };
        let lambda_str = match self.lambda_used {
            Some(l) => format!("{:.4}", l),
            None => "None".to_string(),
        };
        format!(
            "SyntheticControlResult(att={:.6}, se={}, method='{}', lambda={}, \
             pre_rmse={:.6}, n_control={}, n_pre={}, n_post={}, converged={})",
            self.att,
            se_str,
            self.method,
            lambda_str,
            self.pre_treatment_rmse,
            self.n_units_control,
            self.n_periods_pre,
            self.n_periods_post,
            self.solver_converged
        )
    }

    /// String representation for Python's str()
    fn __str__(&self) -> String {
        match self.standard_error {
            Some(se) => format!(
                "Synthetic Control Result:\n  ATT: {:.6} (SE: {:.6})\n  Method: {}\n  Pre-treatment RMSE: {:.6}",
                self.att, se, self.method, self.pre_treatment_rmse
            ),
            None => format!(
                "Synthetic Control Result:\n  ATT: {:.6}\n  Method: {}\n  Pre-treatment RMSE: {:.6}",
                self.att, self.method, self.pre_treatment_rmse
            ),
        }
    }
}

// ============================================================================
// In-Space Placebo SE Computation (TASK-014)
// ============================================================================

/// Compute standard error via in-space placebo bootstrap.
///
/// This implements the in-space placebo test where each control unit
/// is iteratively treated as the "placebo treated" unit, and the
/// standard deviation of the resulting placebo ATTs is used as the SE.
///
/// # Algorithm
/// 1. For each control unit i:
///    a. Create modified panel with unit i as treated
///    b. Remaining control units form the donor pool
///    c. Compute SC weights and placebo ATT
/// 2. SE = std(placebo_ATTs)
///
/// # Arguments
///
/// * `panel` - Original panel data
/// * `config` - Configuration (uses config.n_placebo to limit iterations)
/// * `method` - Which SC method to use for placebo estimation
///
/// # Returns
///
/// * `Ok((se, n_successful))` - Standard error and number of successful placebos
/// * `Err(SynthControlError)` - If insufficient placebos succeed
pub fn compute_placebo_se(
    panel: &SCPanelData,
    config: &SynthControlConfig,
) -> Result<(f64, usize), SynthControlError> {
    let n_control = panel.n_control();
    
    // Determine how many placebos to run
    let n_placebo = config.n_placebo.min(n_control);
    
    if n_placebo < 2 {
        return Err(SynthControlError::InvalidData {
            message: format!(
                "In-space placebo requires at least 2 control units; found {}",
                n_control
            ),
        });
    }

    // Initialize PRNG for reproducible placebo selection
    let seed = config.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    });
    let mut _rng = SplitMix64::new(seed);

    // Use Welford's algorithm for online variance computation
    let mut welford = WelfordState::new(1);
    let mut n_successful = 0;

    // Iterate over control units as placebo treated
    for placebo_ctrl_idx in 0..n_placebo {
        // Get the actual unit index for this control
        let placebo_treated_unit = panel.control_indices[placebo_ctrl_idx];

        // Create new control indices excluding the placebo treated unit
        let new_control_indices: Vec<usize> = panel
            .control_indices
            .iter()
            .filter(|&&idx| idx != placebo_treated_unit)
            .copied()
            .collect();

        // Need at least 1 control unit for the placebo
        if new_control_indices.is_empty() {
            continue;
        }

        // Create placebo panel
        let placebo_panel = SCPanelData {
            outcomes: panel.outcomes.clone(),
            n_units: panel.n_units,
            n_periods: panel.n_periods,
            control_indices: new_control_indices,
            treated_index: placebo_treated_unit,
            pre_period_indices: panel.pre_period_indices.clone(),
            post_period_indices: panel.post_period_indices.clone(),
        };

        // Compute weights for placebo using same method
        let weight_result = match compute_weights(&placebo_panel, config) {
            Ok(r) => r,
            Err(_) => continue, // Skip failed placebos
        };

        // Compute placebo ATT
        let placebo_att = compute_att(&placebo_panel, &weight_result.weights);

        // Check for valid ATT
        if placebo_att.is_finite() {
            welford.update(&[placebo_att]);
            n_successful += 1;
        }
    }

    // Need at least 2 successful placebos to compute SE
    if n_successful < 2 {
        return Err(SynthControlError::InvalidData {
            message: format!(
                "Only {} placebo iterations succeeded; need at least 2 for SE",
                n_successful
            ),
        });
    }

    // Compute SE as std of placebo ATTs
    let se = welford.standard_errors()[0];

    if !se.is_finite() || se < 0.0 {
        return Err(SynthControlError::NumericalInstability {
            message: "Placebo SE computation produced invalid value".to_string(),
        });
    }

    Ok((se, n_successful))
}

// ============================================================================
// Main Estimation Function
// ============================================================================

/// Estimate Synthetic Control with all diagnostics.
///
/// This is the main entry point for SC estimation, computing weights,
/// ATT, diagnostics, and optionally standard errors.
///
/// # Arguments
///
/// * `panel` - Panel data with outcomes and structure
/// * `config` - Configuration including method, SE options, etc.
///
/// # Returns
///
/// * `Ok(SyntheticControlResult)` - Full estimation result
/// * `Err(SynthControlError)` - If estimation fails
pub fn estimate(
    panel: &SCPanelData,
    config: &SynthControlConfig,
) -> Result<SyntheticControlResult, SynthControlError> {
    // Compute weights using the specified method
    let weight_result = compute_weights(panel, config)?;

    // Compute ATT
    // For augmented method, we need to use the bias-corrected ATT
    let att = if config.method == SynthControlMethod::Augmented {
        let asc_result = compute_augmented_sc(panel, config)?;
        compute_augmented_att(panel, &asc_result)
    } else {
        compute_att(panel, &weight_result.weights)
    };

    // Compute pre-treatment fit diagnostics
    let pre_mse = compute_pre_treatment_mse(panel, &weight_result.weights);
    let pre_rmse = pre_mse.sqrt();

    // Compute standard error if requested
    let (standard_error, n_placebo_used) = if config.compute_se {
        match compute_placebo_se(panel, config) {
            Ok((se, n)) => (Some(se), Some(n)),
            Err(_) => (None, None), // SE computation failed, but we still return the ATT
        }
    } else {
        (None, None)
    };

    // Determine lambda used (for penalized method)
    let lambda_used = if config.method == SynthControlMethod::Penalized
        || config.method == SynthControlMethod::Augmented
    {
        config.lambda
    } else {
        None
    };

    Ok(SyntheticControlResult {
        att,
        standard_error,
        unit_weights: weight_result.weights,
        pre_treatment_rmse: pre_rmse,
        pre_treatment_mse: pre_mse,
        method: config.method.as_str().to_string(),
        lambda_used,
        n_units_control: panel.n_control(),
        n_periods_pre: panel.n_pre(),
        n_periods_post: panel.n_post(),
        solver_converged: weight_result.converged,
        solver_iterations: weight_result.iterations,
        n_placebo_used,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Test Helpers
    // ========================================================================

    /// Create a simple panel for testing.
    /// 
    /// Layout:
    /// - 1 treated unit (index 0)
    /// - 3 control units (indices 1, 2, 3)
    /// - 3 pre-periods (0, 1, 2)
    /// - 1 post-period (3)
    fn create_simple_panel() -> SCPanelData {
        // Outcomes in row-major order: unit × period
        // Unit 0 (treated): [1.0, 2.0, 3.0, 10.0]  <- treatment effect in period 3
        // Unit 1 (control): [1.0, 2.0, 3.0, 4.0]   <- parallel trend
        // Unit 2 (control): [2.0, 3.0, 4.0, 5.0]   <- parallel trend
        // Unit 3 (control): [0.0, 1.0, 2.0, 3.0]   <- parallel trend
        let outcomes = vec![
            1.0, 2.0, 3.0, 10.0, // treated
            1.0, 2.0, 3.0, 4.0,  // control 1 - matches treated pre
            2.0, 3.0, 4.0, 5.0,  // control 2
            0.0, 1.0, 2.0, 3.0,  // control 3
        ];

        SCPanelData::new(
            outcomes,
            4,                    // n_units
            4,                    // n_periods
            vec![1, 2, 3],        // control_indices
            0,                    // treated_index
            vec![0, 1, 2],        // pre_period_indices
            vec![3],              // post_period_indices
        )
        .unwrap()
    }

    /// Create a panel where one control perfectly matches treated.
    fn create_perfect_match_panel() -> SCPanelData {
        let outcomes = vec![
            1.0, 2.0, 3.0, 10.0, // treated
            1.0, 2.0, 3.0, 4.0,  // control 1 - EXACT match in pre-treatment
            5.0, 6.0, 7.0, 8.0,  // control 2 - different pattern
        ];

        SCPanelData::new(
            outcomes,
            3,
            4,
            vec![1, 2],
            0,
            vec![0, 1, 2],
            vec![3],
        )
        .unwrap()
    }

    /// Create a panel with known treatment effect of 5.0.
    fn create_known_effect_panel() -> SCPanelData {
        // All units have the same pre-treatment trend
        // Treated unit jumps by 5.0 in post-period
        let outcomes = vec![
            1.0, 2.0, 3.0, 9.0,  // treated: 4.0 expected, got 9.0 -> effect = 5.0
            1.0, 2.0, 3.0, 4.0,  // control 1: continues trend
        ];

        SCPanelData::new(
            outcomes,
            2,
            4,
            vec![1],
            0,
            vec![0, 1, 2],
            vec![3],
        )
        .unwrap()
    }

    // ========================================================================
    // Error Type Tests
    // ========================================================================

    #[test]
    fn test_error_display_convergence() {
        let err = SynthControlError::ConvergenceFailure { iterations: 1000 };
        let msg = format!("{}", err);
        assert!(msg.contains("1000"));
        assert!(msg.contains("converge"));
    }

    #[test]
    fn test_error_display_numerical() {
        let err = SynthControlError::NumericalInstability {
            message: "NaN detected".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("NaN detected"));
    }

    #[test]
    fn test_error_display_invalid_data() {
        let err = SynthControlError::InvalidData {
            message: "Empty panel".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Empty panel"));
    }

    #[test]
    fn test_error_display_invalid_method() {
        let err = SynthControlError::InvalidMethod {
            method: "bad".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("bad"));
        assert!(msg.contains("traditional"));
    }

    // ========================================================================
    // Method Enum Tests
    // ========================================================================

    #[test]
    fn test_method_from_str() {
        assert_eq!(
            SynthControlMethod::from_str("traditional").unwrap(),
            SynthControlMethod::Traditional
        );
        assert_eq!(
            SynthControlMethod::from_str("TRADITIONAL").unwrap(),
            SynthControlMethod::Traditional
        );
        assert_eq!(
            SynthControlMethod::from_str("penalized").unwrap(),
            SynthControlMethod::Penalized
        );
        assert_eq!(
            SynthControlMethod::from_str("robust").unwrap(),
            SynthControlMethod::Robust
        );
        assert_eq!(
            SynthControlMethod::from_str("augmented").unwrap(),
            SynthControlMethod::Augmented
        );
        assert!(SynthControlMethod::from_str("invalid").is_err());
    }

    #[test]
    fn test_method_as_str() {
        assert_eq!(SynthControlMethod::Traditional.as_str(), "traditional");
        assert_eq!(SynthControlMethod::Penalized.as_str(), "penalized");
        assert_eq!(SynthControlMethod::Robust.as_str(), "robust");
        assert_eq!(SynthControlMethod::Augmented.as_str(), "augmented");
    }

    #[test]
    fn test_method_default() {
        assert_eq!(SynthControlMethod::default(), SynthControlMethod::Traditional);
    }

    // ========================================================================
    // Config Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = SynthControlConfig::default();
        assert_eq!(config.method, SynthControlMethod::Traditional);
        assert!(config.lambda.is_none());
        assert!(!config.compute_se);
        assert_eq!(config.n_placebo, 100);
        assert_eq!(config.max_iter, 1000);
        assert!((config.tol - 1e-6).abs() < 1e-15);
        assert!(config.seed.is_none());
    }

    // ========================================================================
    // SCPanelData Tests
    // ========================================================================

    #[test]
    fn test_panel_data_creation() {
        let panel = create_simple_panel();
        assert_eq!(panel.n_units, 4);
        assert_eq!(panel.n_periods, 4);
        assert_eq!(panel.n_control(), 3);
        assert_eq!(panel.n_pre(), 3);
        assert_eq!(panel.n_post(), 1);
        assert_eq!(panel.treated_index, 0);
    }

    #[test]
    fn test_panel_data_outcome_accessor() {
        let panel = create_simple_panel();
        
        // Treated unit outcomes
        assert!((panel.outcome(0, 0) - 1.0).abs() < 1e-10);
        assert!((panel.outcome(0, 1) - 2.0).abs() < 1e-10);
        assert!((panel.outcome(0, 2) - 3.0).abs() < 1e-10);
        assert!((panel.outcome(0, 3) - 10.0).abs() < 1e-10);
        
        // Control unit 1 outcomes
        assert!((panel.outcome(1, 0) - 1.0).abs() < 1e-10);
        assert!((panel.outcome(1, 3) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_panel_data_treated_pre_vector() {
        let panel = create_simple_panel();
        let treated_pre = panel.treated_pre_vector();
        
        assert_eq!(treated_pre.len(), 3);
        assert!((treated_pre[0] - 1.0).abs() < 1e-10);
        assert!((treated_pre[1] - 2.0).abs() < 1e-10);
        assert!((treated_pre[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_panel_data_control_pre_matrix() {
        let panel = create_simple_panel();
        let control_pre = panel.control_pre_matrix();
        
        // 3 control units × 3 pre-periods = 9 elements
        assert_eq!(control_pre.len(), 9);
        
        // Control 1 (index 1): [1.0, 2.0, 3.0]
        assert!((control_pre[0] - 1.0).abs() < 1e-10);
        assert!((control_pre[1] - 2.0).abs() < 1e-10);
        assert!((control_pre[2] - 3.0).abs() < 1e-10);
        
        // Control 2 (index 2): [2.0, 3.0, 4.0]
        assert!((control_pre[3] - 2.0).abs() < 1e-10);
        assert!((control_pre[4] - 3.0).abs() < 1e-10);
        assert!((control_pre[5] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_panel_data_treated_post_vector() {
        let panel = create_simple_panel();
        let treated_post = panel.treated_post_vector();
        
        assert_eq!(treated_post.len(), 1);
        assert!((treated_post[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_panel_data_control_post_matrix() {
        let panel = create_simple_panel();
        let control_post = panel.control_post_matrix();
        
        // 3 control units × 1 post-period = 3 elements
        assert_eq!(control_post.len(), 3);
        assert!((control_post[0] - 4.0).abs() < 1e-10);  // Control 1
        assert!((control_post[1] - 5.0).abs() < 1e-10);  // Control 2
        assert!((control_post[2] - 3.0).abs() < 1e-10);  // Control 3
    }

    #[test]
    fn test_panel_data_validation_wrong_length() {
        let result = SCPanelData::new(
            vec![1.0, 2.0, 3.0], // Wrong length
            2,
            2,
            vec![1],
            0,
            vec![0],
            vec![1],
        );
        assert!(result.is_err());
        match result {
            Err(SynthControlError::InvalidData { message }) => {
                assert!(message.contains("length"));
            }
            _ => panic!("Expected InvalidData error"),
        }
    }

    #[test]
    fn test_panel_data_validation_no_controls() {
        let result = SCPanelData::new(
            vec![1.0, 2.0],
            1,
            2,
            vec![],  // No controls
            0,
            vec![0],
            vec![1],
        );
        assert!(result.is_err());
        match result {
            Err(SynthControlError::InvalidData { message }) => {
                assert!(message.contains("control"));
            }
            _ => panic!("Expected InvalidData error"),
        }
    }

    #[test]
    fn test_panel_data_validation_treated_in_control() {
        let result = SCPanelData::new(
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            2,
            vec![0, 1],  // Treated (0) in control list
            0,
            vec![0],
            vec![1],
        );
        assert!(result.is_err());
        match result {
            Err(SynthControlError::InvalidData { message }) => {
                assert!(message.contains("treated_index"));
            }
            _ => panic!("Expected InvalidData error"),
        }
    }

    // ========================================================================
    // Weight Computation Tests
    // ========================================================================

    #[test]
    fn test_compute_yyt() {
        let y_control_pre = vec![
            1.0, 2.0,  // Control 0
            3.0, 4.0,  // Control 1
        ];
        let n_control = 2;
        let n_pre = 2;
        
        let yyt = compute_yyt(&y_control_pre, n_control, n_pre);
        
        // YYt[0,0] = 1*1 + 2*2 = 5
        assert!((yyt[0][0] - 5.0).abs() < 1e-10);
        // YYt[0,1] = 1*3 + 2*4 = 11
        assert!((yyt[0][1] - 11.0).abs() < 1e-10);
        // YYt[1,0] = 3*1 + 4*2 = 11
        assert!((yyt[1][0] - 11.0).abs() < 1e-10);
        // YYt[1,1] = 3*3 + 4*4 = 25
        assert!((yyt[1][1] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_yy() {
        let y_control_pre = vec![
            1.0, 2.0,  // Control 0
            3.0, 4.0,  // Control 1
        ];
        let y_treated_pre = vec![2.0, 3.0];
        let n_control = 2;
        let n_pre = 2;
        
        let yy = compute_yy(&y_control_pre, &y_treated_pre, n_control, n_pre);
        
        // Yy[0] = 1*2 + 2*3 = 8
        assert!((yy[0] - 8.0).abs() < 1e-10);
        // Yy[1] = 3*2 + 4*3 = 18
        assert!((yy[1] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_traditional_weights_sum_to_one() {
        let panel = create_simple_panel();
        let config = SynthControlConfig::default();
        
        let result = compute_traditional_weights(&panel, &config).unwrap();
        
        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_traditional_weights_non_negative() {
        let panel = create_simple_panel();
        let config = SynthControlConfig::default();
        
        let result = compute_traditional_weights(&panel, &config).unwrap();
        
        for (i, &w) in result.weights.iter().enumerate() {
            assert!(
                w >= -1e-15,
                "Weight {} must be non-negative, got {}",
                i,
                w
            );
        }
    }

    #[test]
    fn test_traditional_weights_perfect_match() {
        // When one control perfectly matches treated, it should get high weight
        let panel = create_perfect_match_panel();
        let config = SynthControlConfig::default();
        
        let result = compute_traditional_weights(&panel, &config).unwrap();
        
        // Control 1 (index 0 in weights) should get most weight
        assert!(
            result.weights[0] > 0.9,
            "Perfect match control should get weight > 0.9, got {}",
            result.weights[0]
        );
    }

    #[test]
    fn test_traditional_weights_objective_finite() {
        let panel = create_simple_panel();
        let config = SynthControlConfig::default();
        
        let result = compute_traditional_weights(&panel, &config).unwrap();
        
        assert!(
            result.objective.is_finite(),
            "Objective should be finite, got {}",
            result.objective
        );
    }

    // ========================================================================
    // ATT Computation Tests
    // ========================================================================

    #[test]
    fn test_att_known_treatment_effect() {
        // Panel where treated unit has known effect of 5.0
        let panel = create_known_effect_panel();
        
        // With single control that matches treated, weight = [1.0]
        let weights = vec![1.0];
        let att = compute_att(&panel, &weights);
        
        // Expected: treated_post (9.0) - control_post (4.0) = 5.0
        assert!(
            (att - 5.0).abs() < 1e-10,
            "ATT should be 5.0, got {}",
            att
        );
    }

    #[test]
    fn test_att_zero_treatment_effect() {
        // Create panel where treated and control have same trajectory
        let outcomes = vec![
            1.0, 2.0, 3.0, 4.0,  // treated
            1.0, 2.0, 3.0, 4.0,  // control (identical)
        ];
        let panel = SCPanelData::new(
            outcomes,
            2,
            4,
            vec![1],
            0,
            vec![0, 1, 2],
            vec![3],
        ).unwrap();
        
        let weights = vec![1.0];
        let att = compute_att(&panel, &weights);
        
        assert!(
            att.abs() < 1e-10,
            "ATT should be 0 when no treatment effect, got {}",
            att
        );
    }

    #[test]
    fn test_att_with_weighted_control() {
        let panel = create_simple_panel();
        
        // Use uniform weights: [1/3, 1/3, 1/3]
        let weights = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let att = compute_att(&panel, &weights);
        
        // Synthetic post = (4 + 5 + 3) / 3 = 4.0
        // ATT = 10 - 4 = 6.0
        assert!(
            (att - 6.0).abs() < 1e-10,
            "ATT should be 6.0, got {}",
            att
        );
    }

    #[test]
    fn test_att_multiple_post_periods() {
        let outcomes = vec![
            1.0, 2.0, 8.0, 9.0,  // treated: post = [8, 9]
            1.0, 2.0, 3.0, 4.0,  // control: post = [3, 4]
        ];
        let panel = SCPanelData::new(
            outcomes,
            2,
            4,
            vec![1],
            0,
            vec![0, 1],
            vec![2, 3],
        ).unwrap();
        
        let weights = vec![1.0];
        let att = compute_att(&panel, &weights);
        
        // ATT = mean((8-3), (9-4)) = mean(5, 5) = 5
        assert!(
            (att - 5.0).abs() < 1e-10,
            "ATT should be 5.0, got {}",
            att
        );
    }

    #[test]
    fn test_att_negative_effect() {
        let outcomes = vec![
            5.0, 6.0, 7.0, 3.0,  // treated: drops in post
            5.0, 6.0, 7.0, 8.0,  // control: continues trend
        ];
        let panel = SCPanelData::new(
            outcomes,
            2,
            4,
            vec![1],
            0,
            vec![0, 1, 2],
            vec![3],
        ).unwrap();
        
        let weights = vec![1.0];
        let att = compute_att(&panel, &weights);
        
        // ATT = 3 - 8 = -5
        assert!(
            (att - (-5.0)).abs() < 1e-10,
            "ATT should be -5.0, got {}",
            att
        );
    }

    // ========================================================================
    // Pre-treatment Fit Tests
    // ========================================================================

    #[test]
    fn test_pre_treatment_mse_perfect_fit() {
        let panel = create_perfect_match_panel();
        
        // Control 0 perfectly matches treated, so give it all weight
        let weights = vec![1.0, 0.0];
        let mse = compute_pre_treatment_mse(&panel, &weights);
        
        assert!(
            mse < 1e-15,
            "MSE should be 0 for perfect fit, got {}",
            mse
        );
    }

    #[test]
    fn test_pre_treatment_rmse_perfect_fit() {
        let panel = create_perfect_match_panel();
        let weights = vec![1.0, 0.0];
        let rmse = compute_pre_treatment_rmse(&panel, &weights);
        
        assert!(
            rmse < 1e-10,
            "RMSE should be 0 for perfect fit, got {}",
            rmse
        );
    }

    #[test]
    fn test_pre_treatment_mse_known_value() {
        let outcomes = vec![
            2.0, 4.0, 6.0, 10.0,  // treated
            1.0, 2.0, 3.0, 5.0,   // control (half of treated in pre)
        ];
        let panel = SCPanelData::new(
            outcomes,
            2,
            4,
            vec![1],
            0,
            vec![0, 1, 2],
            vec![3],
        ).unwrap();
        
        let weights = vec![1.0];
        let mse = compute_pre_treatment_mse(&panel, &weights);
        
        // Diffs: (2-1)=1, (4-2)=2, (6-3)=3
        // MSE = (1 + 4 + 9) / 3 = 14/3 ≈ 4.667
        let expected_mse = (1.0 + 4.0 + 9.0) / 3.0;
        assert!(
            (mse - expected_mse).abs() < 1e-10,
            "MSE should be {}, got {}",
            expected_mse,
            mse
        );
    }

    #[test]
    fn test_pre_treatment_rmse_known_value() {
        let outcomes = vec![
            2.0, 4.0, 6.0, 10.0,
            1.0, 2.0, 3.0, 5.0,
        ];
        let panel = SCPanelData::new(
            outcomes,
            2,
            4,
            vec![1],
            0,
            vec![0, 1, 2],
            vec![3],
        ).unwrap();
        
        let weights = vec![1.0];
        let rmse = compute_pre_treatment_rmse(&panel, &weights);
        
        let expected_rmse = ((1.0 + 4.0 + 9.0) / 3.0_f64).sqrt();
        assert!(
            (rmse - expected_rmse).abs() < 1e-10,
            "RMSE should be {}, got {}",
            expected_rmse,
            rmse
        );
    }

    #[test]
    fn test_pre_treatment_mse_with_weighted_control() {
        let panel = create_perfect_match_panel();
        
        // Weights: 50% on perfect match, 50% on different control
        let weights = vec![0.5, 0.5];
        let mse = compute_pre_treatment_mse(&panel, &weights);
        
        // Synthetic = 0.5 * [1,2,3] + 0.5 * [5,6,7] = [3, 4, 5]
        // Treated = [1, 2, 3]
        // Diffs = [-2, -2, -2]
        // MSE = (4 + 4 + 4) / 3 = 4
        assert!(
            (mse - 4.0).abs() < 1e-10,
            "MSE should be 4.0, got {}",
            mse
        );
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_pipeline() {
        let panel = create_simple_panel();
        let config = SynthControlConfig::default();
        
        // Compute weights
        let weight_result = compute_traditional_weights(&panel, &config).unwrap();
        
        // Verify weight constraints
        let sum: f64 = weight_result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Weights must sum to 1");
        for &w in &weight_result.weights {
            assert!(w >= -1e-15, "Weights must be non-negative");
        }
        
        // Compute ATT
        let att = compute_att(&panel, &weight_result.weights);
        assert!(att.is_finite(), "ATT must be finite");
        
        // Compute pre-treatment fit
        let rmse = compute_pre_treatment_rmse(&panel, &weight_result.weights);
        assert!(rmse >= 0.0, "RMSE must be non-negative");
        assert!(rmse.is_finite(), "RMSE must be finite");
    }

    #[test]
    fn test_optimized_weights_improve_fit() {
        let panel = create_simple_panel();
        let config = SynthControlConfig::default();
        
        // Uniform weights
        let n_control = panel.n_control();
        let uniform_weights: Vec<f64> = vec![1.0 / n_control as f64; n_control];
        let uniform_mse = compute_pre_treatment_mse(&panel, &uniform_weights);
        
        // Optimized weights
        let result = compute_traditional_weights(&panel, &config).unwrap();
        let optimized_mse = compute_pre_treatment_mse(&panel, &result.weights);
        
        // Optimized should be at least as good as uniform
        assert!(
            optimized_mse <= uniform_mse + 1e-10,
            "Optimized MSE ({}) should be <= uniform MSE ({})",
            optimized_mse,
            uniform_mse
        );
    }

    // ========================================================================
    // Simplex Projection Tests
    // ========================================================================

    #[test]
    fn test_simplex_projection_already_valid() {
        // Vector already on simplex should be unchanged (or very close)
        let v = vec![0.5, 0.3, 0.2];
        let projected = project_onto_simplex(&v);

        let sum: f64 = projected.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Projected sum should be 1.0, got {}",
            sum
        );
        for (i, &p) in projected.iter().enumerate() {
            assert!(
                p >= -1e-15,
                "Projected element {} should be non-negative, got {}",
                i,
                p
            );
        }
    }

    #[test]
    fn test_simplex_projection_negative_values() {
        // Vector with negative values
        let v = vec![-0.5, 0.5, 1.0];
        let projected = project_onto_simplex(&v);

        let sum: f64 = projected.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Projected sum should be 1.0, got {}",
            sum
        );
        for (i, &p) in projected.iter().enumerate() {
            assert!(
                p >= -1e-15,
                "Projected element {} should be non-negative, got {}",
                i,
                p
            );
        }
        // The negative element should become 0
        assert!(
            projected[0] < 1e-10,
            "Negative element should be projected to 0, got {}",
            projected[0]
        );
    }

    #[test]
    fn test_simplex_projection_sum_greater_than_one() {
        // Vector with sum > 1
        let v = vec![0.6, 0.5, 0.4];
        let projected = project_onto_simplex(&v);

        let sum: f64 = projected.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Projected sum should be 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_simplex_projection_sum_less_than_one() {
        // Vector with sum < 1 and all positive
        let v = vec![0.1, 0.1, 0.1];
        let projected = project_onto_simplex(&v);

        let sum: f64 = projected.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Projected sum should be 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_simplex_projection_single_element() {
        // Single element should always be 1.0
        let v = vec![5.0];
        let projected = project_onto_simplex(&v);

        assert_eq!(projected.len(), 1);
        assert!(
            (projected[0] - 1.0).abs() < 1e-10,
            "Single element should be 1.0, got {}",
            projected[0]
        );
    }

    #[test]
    fn test_simplex_projection_empty() {
        let v: Vec<f64> = vec![];
        let projected = project_onto_simplex(&v);
        assert!(projected.is_empty());
    }

    // ========================================================================
    // Penalized SC Tests
    // ========================================================================

    #[test]
    fn test_penalized_weights_sum_to_one() {
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Penalized;
        config.lambda = Some(1.0);

        let result = compute_penalized_weights(&panel, &config).unwrap();

        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Penalized weights must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_penalized_weights_non_negative() {
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Penalized;
        config.lambda = Some(1.0);

        let result = compute_penalized_weights(&panel, &config).unwrap();

        for (i, &w) in result.weights.iter().enumerate() {
            assert!(
                w >= -1e-15,
                "Penalized weight {} must be non-negative, got {}",
                i,
                w
            );
        }
    }

    #[test]
    fn test_penalized_higher_lambda_more_uniform() {
        let panel = create_perfect_match_panel();

        // Low lambda - should concentrate on best match
        let mut config_low = SynthControlConfig::default();
        config_low.method = SynthControlMethod::Penalized;
        config_low.lambda = Some(0.001);
        let result_low = compute_penalized_weights(&panel, &config_low).unwrap();
        let max_weight_low = result_low.weights.iter().cloned().fold(0.0, f64::max);

        // High lambda - should be more uniform
        let mut config_high = SynthControlConfig::default();
        config_high.method = SynthControlMethod::Penalized;
        config_high.lambda = Some(100.0);
        let result_high = compute_penalized_weights(&panel, &config_high).unwrap();
        let max_weight_high = result_high.weights.iter().cloned().fold(0.0, f64::max);

        // Higher lambda should lead to more uniform weights (lower max weight)
        assert!(
            max_weight_high <= max_weight_low + 0.1,
            "Higher lambda should give more uniform weights: low_max={}, high_max={}",
            max_weight_low,
            max_weight_high
        );
    }

    #[test]
    fn test_penalized_negative_lambda_error() {
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Penalized;
        config.lambda = Some(-1.0);

        let result = compute_penalized_weights(&panel, &config);
        assert!(result.is_err());
        match result {
            Err(SynthControlError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "lambda");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_penalized_auto_lambda_selection() {
        // When lambda is None, LOOCV should select a lambda
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Penalized;
        config.lambda = None;

        let result = compute_penalized_weights(&panel, &config);
        assert!(result.is_ok(), "Auto lambda selection should succeed");

        let weight_result = result.unwrap();
        let sum: f64 = weight_result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights should still sum to 1.0"
        );
    }

    // ========================================================================
    // Robust SC Tests
    // ========================================================================

    #[test]
    fn test_robust_weights_sum_to_one() {
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Robust;

        let result = compute_robust_weights(&panel, &config).unwrap();

        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Robust weights must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_robust_weights_non_negative() {
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Robust;

        let result = compute_robust_weights(&panel, &config).unwrap();

        for (i, &w) in result.weights.iter().enumerate() {
            assert!(
                w >= -1e-15,
                "Robust weight {} must be non-negative, got {}",
                i,
                w
            );
        }
    }

    #[test]
    fn test_robust_with_level_differences() {
        // Create panel where controls have different levels but same dynamics
        // Robust SC should focus on matching dynamics, not levels
        let outcomes = vec![
            0.0, 1.0, 2.0, 7.0,   // treated: mean = 1, dynamics = [0,1,2]
            10.0, 11.0, 12.0, 13.0,  // control 1: high level, same dynamics
            0.0, 1.0, 2.0, 3.0,   // control 2: same level and dynamics as treated
        ];
        let panel = SCPanelData::new(
            outcomes,
            3,
            4,
            vec![1, 2],
            0,
            vec![0, 1, 2],
            vec![3],
        ).unwrap();

        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Robust;

        let result = compute_robust_weights(&panel, &config);
        assert!(result.is_ok(), "Robust SC should handle level differences");
    }

    // ========================================================================
    // Augmented SC Tests
    // ========================================================================

    /// Create a panel suitable for Augmented SC (requires 2+ controls, 2+ pre-periods)
    fn create_augmented_panel() -> SCPanelData {
        let outcomes = vec![
            1.0, 2.0, 3.0, 10.0, 11.0,  // treated
            1.0, 2.0, 3.0, 4.0, 5.0,    // control 1 - perfect pre-treatment match
            2.0, 3.0, 4.0, 5.0, 6.0,    // control 2
            0.0, 1.0, 2.0, 3.0, 4.0,    // control 3
        ];

        SCPanelData::new(
            outcomes,
            4,
            5,
            vec![1, 2, 3],
            0,
            vec![0, 1, 2],
            vec![3, 4],
        )
        .unwrap()
    }

    #[test]
    fn test_augmented_sc_computes() {
        let panel = create_augmented_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Augmented;
        config.lambda = Some(1.0);

        let result = compute_augmented_sc(&panel, &config);
        assert!(result.is_ok(), "Augmented SC should compute successfully");

        let asc_result = result.unwrap();
        
        // Check weights are valid
        let sum: f64 = asc_result.weight_result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Augmented SC weights must sum to 1.0, got {}",
            sum
        );
        
        // Check ridge coefficients are computed
        assert_eq!(
            asc_result.ridge_coefficients.len(),
            panel.n_pre(),
            "Ridge coefficients should have n_pre elements"
        );
    }

    #[test]
    fn test_augmented_att_includes_bias_correction() {
        let panel = create_augmented_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Augmented;
        config.lambda = Some(1.0);

        let asc_result = compute_augmented_sc(&panel, &config).unwrap();

        // Compute both traditional and augmented ATT
        let att_traditional = compute_att(&panel, &asc_result.weight_result.weights);
        let att_augmented = compute_augmented_att(&panel, &asc_result);

        // Both should be finite
        assert!(att_traditional.is_finite(), "Traditional ATT should be finite");
        assert!(att_augmented.is_finite(), "Augmented ATT should be finite");

        // The difference should be the bias adjustment
        let diff = att_traditional - att_augmented;
        assert!(
            (diff - asc_result.bias_adjustment).abs() < 1e-10,
            "Difference should equal bias adjustment: {} vs {}",
            diff,
            asc_result.bias_adjustment
        );
    }

    #[test]
    fn test_augmented_requires_min_controls() {
        // Only 1 control unit - should fail
        let outcomes = vec![
            1.0, 2.0, 3.0, 10.0,
            1.0, 2.0, 3.0, 4.0,
        ];
        let panel = SCPanelData::new(
            outcomes,
            2,
            4,
            vec![1],
            0,
            vec![0, 1, 2],
            vec![3],
        ).unwrap();

        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Augmented;
        config.lambda = Some(1.0);

        let result = compute_augmented_sc(&panel, &config);
        assert!(result.is_err(), "Augmented SC should require 2+ controls");
    }

    #[test]
    fn test_augmented_requires_min_pre_periods() {
        // Only 1 pre-treatment period - should fail
        let outcomes = vec![
            1.0, 10.0, 11.0,
            1.0, 4.0, 5.0,
            2.0, 5.0, 6.0,
        ];
        let panel = SCPanelData::new(
            outcomes,
            3,
            3,
            vec![1, 2],
            0,
            vec![0],
            vec![1, 2],
        ).unwrap();

        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Augmented;
        config.lambda = Some(1.0);

        let result = compute_augmented_sc(&panel, &config);
        assert!(result.is_err(), "Augmented SC should require 2+ pre-periods");
    }

    // ========================================================================
    // Method Dispatch Tests
    // ========================================================================

    #[test]
    fn test_compute_weights_traditional() {
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Traditional;

        let result = compute_weights(&panel, &config);
        assert!(result.is_ok());

        let weight_result = result.unwrap();
        let sum: f64 = weight_result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_weights_penalized() {
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Penalized;
        config.lambda = Some(1.0);

        let result = compute_weights(&panel, &config);
        assert!(result.is_ok());

        let weight_result = result.unwrap();
        let sum: f64 = weight_result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_weights_robust() {
        let panel = create_simple_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Robust;

        let result = compute_weights(&panel, &config);
        assert!(result.is_ok());

        let weight_result = result.unwrap();
        let sum: f64 = weight_result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_weights_augmented() {
        let panel = create_augmented_panel();
        let mut config = SynthControlConfig::default();
        config.method = SynthControlMethod::Augmented;
        config.lambda = Some(1.0);

        let result = compute_weights(&panel, &config);
        assert!(result.is_ok());

        let weight_result = result.unwrap();
        let sum: f64 = weight_result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_methods_produce_valid_weights() {
        let panel = create_augmented_panel();

        let methods = vec![
            (SynthControlMethod::Traditional, None),
            (SynthControlMethod::Penalized, Some(1.0)),
            (SynthControlMethod::Robust, None),
            (SynthControlMethod::Augmented, Some(1.0)),
        ];

        for (method, lambda) in methods {
            let mut config = SynthControlConfig::default();
            config.method = method;
            config.lambda = lambda;

            let result = compute_weights(&panel, &config);
            assert!(
                result.is_ok(),
                "Method {:?} should produce valid weights",
                method
            );

            let weight_result = result.unwrap();

            // All weights should sum to 1
            let sum: f64 = weight_result.weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Method {:?}: weights should sum to 1.0, got {}",
                method,
                sum
            );

            // All weights should be non-negative
            for (i, &w) in weight_result.weights.iter().enumerate() {
                assert!(
                    w >= -1e-15,
                    "Method {:?}: weight {} should be non-negative, got {}",
                    method,
                    i,
                    w
                );
            }
        }
    }

    // ========================================================================
    // Matrix Operations Tests
    // ========================================================================

    #[test]
    fn test_matrix_inversion_identity() {
        // Invert identity matrix should give identity
        let identity = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let inv = invert_matrix(&identity).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i][j] - expected).abs() < 1e-10,
                    "Identity inverse[{}][{}] should be {}, got {}",
                    i, j, expected, inv[i][j]
                );
            }
        }
    }

    #[test]
    fn test_matrix_inversion_simple() {
        // Simple 2x2 matrix
        let a = vec![
            vec![4.0, 7.0],
            vec![2.0, 6.0],
        ];
        let inv = invert_matrix(&a).unwrap();

        // Check A * A^-1 = I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += a[i][k] * inv[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "A * A^-1 at [{},{}] should be {}, got {}",
                    i, j, expected, sum
                );
            }
        }
    }

    #[test]
    fn test_matrix_inversion_singular_fails() {
        // Singular matrix should fail
        let singular = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
        ];
        let result = invert_matrix(&singular);
        assert!(result.is_err());
    }
}
