//! Logistic regression implementation for binary outcomes.
//!
//! This module implements logistic regression using Maximum Likelihood Estimation (MLE)
//! with Newton-Raphson optimization. It provides:
//!
//! - Newton-Raphson/IRLS optimization with step halving for numerical stability
//! - HC3 robust standard errors adapted for logistic regression
//! - Score bootstrap for clustered inference (Kline & Santos, 2012)
//! - Perfect separation detection
//! - McFadden's pseudo R² and log-likelihood computation
//!
//! # References
//! - Kline, P., & Santos, A. (2012). "A Score Based Approach to Wild Bootstrap Inference."
//! - MacKinnon, J. G., & White, H. (1985). "Some heteroskedasticity-consistent
//!   covariance matrix estimators with improved finite sample properties."

use pyo3::prelude::*;

/// Result of a logistic regression computation.
///
/// Contains coefficient estimates, robust standard errors, and diagnostic
/// information about the optimization process.
#[pyclass]
#[derive(Debug, Clone)]
pub struct LogisticRegressionResult {
    /// Coefficient estimates for x variables (log-odds scale)
    #[pyo3(get)]
    pub coefficients: Vec<f64>,

    /// Intercept term (None if include_intercept=False)
    #[pyo3(get)]
    pub intercept: Option<f64>,

    /// Robust standard errors for each coefficient
    #[pyo3(get)]
    pub standard_errors: Vec<f64>,

    /// Robust standard error for intercept (None if include_intercept=False)
    #[pyo3(get)]
    pub intercept_se: Option<f64>,

    /// Number of observations used
    #[pyo3(get)]
    pub n_samples: usize,

    /// Number of unique clusters (None if not clustered)
    #[pyo3(get)]
    pub n_clusters: Option<usize>,

    /// Type of clustered SE: "analytical" or "bootstrap" (None if not clustered)
    #[pyo3(get)]
    pub cluster_se_type: Option<String>,

    /// Number of bootstrap iterations used (None if not bootstrap)
    #[pyo3(get)]
    pub bootstrap_iterations_used: Option<usize>,

    /// Whether the optimization converged
    #[pyo3(get)]
    pub converged: bool,

    /// Number of iterations used by the optimizer
    #[pyo3(get)]
    pub iterations: usize,

    /// Log-likelihood at the MLE solution
    #[pyo3(get)]
    pub log_likelihood: f64,

    /// McFadden's pseudo R² = 1 - (LL_model / LL_null)
    #[pyo3(get)]
    pub pseudo_r_squared: f64,
}

#[pymethods]
impl LogisticRegressionResult {
    /// Full technical representation of the result object.
    fn __repr__(&self) -> String {
        let intercept_str = match self.intercept {
            Some(i) => format!("{:.6}", i),
            None => "None".to_string(),
        };
        let intercept_se_str = match self.intercept_se {
            Some(se) => format!("{:.6}", se),
            None => "None".to_string(),
        };
        let n_clusters_str = match self.n_clusters {
            Some(n) => n.to_string(),
            None => "None".to_string(),
        };
        let cluster_se_type_str = match &self.cluster_se_type {
            Some(s) => format!("\"{}\"", s),
            None => "None".to_string(),
        };
        let bootstrap_iter_str = match self.bootstrap_iterations_used {
            Some(b) => b.to_string(),
            None => "None".to_string(),
        };
        format!(
            "LogisticRegressionResult(coefficients={:?}, intercept={}, standard_errors={:?}, intercept_se={}, n_samples={}, n_clusters={}, cluster_se_type={}, bootstrap_iterations_used={}, converged={}, iterations={}, log_likelihood={:.6}, pseudo_r_squared={:.6})",
            self.coefficients,
            intercept_str,
            self.standard_errors,
            intercept_se_str,
            self.n_samples,
            n_clusters_str,
            cluster_se_type_str,
            bootstrap_iter_str,
            self.converged,
            self.iterations,
            self.log_likelihood,
            self.pseudo_r_squared
        )
    }

    /// Human-readable summary of the logistic regression result.
    fn __str__(&self) -> String {
        let convergence_str = if self.converged {
            format!("converged in {} iterations", self.iterations)
        } else {
            format!("FAILED to converge after {} iterations", self.iterations)
        };

        let mut output = format!(
            "Logistic Regression: n={}, {}\n",
            self.n_samples, convergence_str
        );

        // Print intercept if present
        if let Some(intercept) = self.intercept {
            let se_str = match self.intercept_se {
                Some(se) => format!(" ± {:.4}", se),
                None => String::new(),
            };
            output.push_str(&format!("  β₀ = {:.4}{} (intercept)\n", intercept, se_str));
        }

        // Print coefficients
        for (i, (&coef, &se)) in self.coefficients.iter().zip(&self.standard_errors).enumerate() {
            output.push_str(&format!("  β{} = {:.4} ± {:.4}\n", i + 1, coef, se));
        }

        // Print fit statistics
        output.push_str(&format!(
            "  Log-likelihood: {:.2}, McFadden R²: {:.3}",
            self.log_likelihood, self.pseudo_r_squared
        ));

        // Add clustering info if present
        if let Some(n_clusters) = self.n_clusters {
            let method = self.cluster_se_type.as_deref().unwrap_or("unknown");
            output.push_str(&format!("\n  Clustered SE ({}): {} clusters", method, n_clusters));
        }

        output
    }
}

/// Error types for logistic regression operations.
#[derive(Debug, Clone)]
pub enum LogisticError {
    /// Perfect separation detected (coefficients diverging)
    PerfectSeparation,
    /// Optimization failed to converge within max iterations
    ConvergenceFailure { iterations: usize },
    /// Hessian matrix is singular (collinearity)
    SingularHessian,
    /// Numerical instability detected (condition number too high, NaN/Inf values)
    NumericalInstability { message: String },
}

impl std::fmt::Display for LogisticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogisticError::PerfectSeparation => {
                write!(
                    f,
                    "Perfect separation detected; logistic regression cannot converge"
                )
            }
            LogisticError::ConvergenceFailure { iterations } => {
                write!(f, "Convergence failed after {} iterations", iterations)
            }
            LogisticError::SingularHessian => {
                write!(f, "Hessian matrix is singular; check for collinearity")
            }
            LogisticError::NumericalInstability { message } => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for LogisticError {}

/// Internal result from MLE optimization.
pub struct MleResult {
    /// Estimated coefficients (including intercept if applicable)
    pub beta: Vec<f64>,
    /// Whether the optimization converged
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
    /// Log-likelihood at the MLE solution
    pub log_likelihood: f64,
    /// Information matrix inverse (X'WX)^-1 for SE computation
    pub info_inv: Vec<Vec<f64>>,
    /// Predicted probabilities at convergence
    pub pi: Vec<f64>,
}

// ============================================================================
// Core mathematical functions
// ============================================================================

/// Logistic (sigmoid) function: σ(x) = 1 / (1 + exp(-x))
///
/// Handles numerical stability for extreme values of x.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Dot product of two vectors.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Compute X' × v (X transpose times vector).
///
/// X is (n × p), v is (n,), result is (p,).
pub fn xt_vector_multiply(x: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let p = x[0].len();
    let mut result = vec![0.0; p];

    for (xi, &vi) in x.iter().zip(v.iter()) {
        for (j, &xij) in xi.iter().enumerate() {
            result[j] += xij * vi;
        }
    }

    result
}

/// Compute X' W X where W is a diagonal matrix represented as a vector.
///
/// X is (n × p), W is diagonal (n,), result is (p × p).
pub fn xt_w_x(x: &[Vec<f64>], w: &[f64]) -> Vec<Vec<f64>> {
    if x.is_empty() {
        return vec![];
    }
    let p = x[0].len();
    let mut result = vec![vec![0.0; p]; p];

    for (xi, &wi) in x.iter().zip(w.iter()) {
        for j in 0..p {
            for k in 0..p {
                result[j][k] += xi[j] * wi * xi[k];
            }
        }
    }

    result
}

/// Multiply a matrix by a vector: result = A × v
pub fn matrix_vector_multiply(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let m = a.len();
    let mut result = Vec::with_capacity(m);

    for row in a.iter() {
        let mut sum = 0.0;
        for (j, &val) in row.iter().enumerate() {
            sum += val * v[j];
        }
        result.push(sum);
    }

    result
}

/// Multiply two matrices: C = A × B
pub fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return vec![];
    }
    let k = a[0].len();
    if k == 0 || b.is_empty() {
        return vec![vec![]; m];
    }
    let n = b[0].len();

    let mut result = vec![vec![0.0; n]; m];

    for (i, a_row) in a.iter().enumerate() {
        for j in 0..n {
            let mut sum = 0.0;
            for (l, &a_val) in a_row.iter().enumerate() {
                sum += a_val * b[l][j];
            }
            result[i][j] = sum;
        }
    }

    result
}

/// Invert a square matrix using Gauss-Jordan elimination with partial pivoting.
pub fn invert_matrix(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, LogisticError> {
    let n = a.len();
    if n == 0 {
        return Err(LogisticError::SingularHessian);
    }

    // Verify square matrix
    for row in a.iter() {
        if row.len() != n {
            return Err(LogisticError::SingularHessian);
        }
    }

    // Create augmented matrix [A|I]
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, a_row) in a.iter().enumerate() {
        let mut row = Vec::with_capacity(2 * n);
        row.extend_from_slice(a_row);
        for j in 0..n {
            row.push(if i == j { 1.0 } else { 0.0 });
        }
        aug.push(row);
    }

    // Gauss-Jordan elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        #[allow(clippy::needless_range_loop)]
        for row in (col + 1)..n {
            // Index needed for row swapping and in-place modification
            let val = aug[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-10 {
            return Err(LogisticError::SingularHessian);
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse
    let mut inverse: Vec<Vec<f64>> = Vec::with_capacity(n);
    for aug_row in aug.iter() {
        inverse.push(aug_row[n..(2 * n)].to_vec());
    }

    Ok(inverse)
}

// ============================================================================
// Log-likelihood computation
// ============================================================================

/// Compute log-likelihood for logistic regression.
///
/// L(β) = Σᵢ [yᵢ log(πᵢ) + (1-yᵢ) log(1-πᵢ)]
///
/// Handles numerical stability by clipping probabilities to [1e-15, 1-1e-15].
pub fn compute_log_likelihood(x: &[Vec<f64>], y: &[f64], beta: &[f64]) -> f64 {
    let mut ll = 0.0;
    let epsilon = 1e-15;

    for (xi, &yi) in x.iter().zip(y.iter()) {
        let linear_pred = dot(xi, beta);
        let pi = sigmoid(linear_pred);
        // Clip for numerical stability
        let pi_clipped = pi.max(epsilon).min(1.0 - epsilon);

        ll += yi * pi_clipped.ln() + (1.0 - yi) * (1.0 - pi_clipped).ln();
    }

    ll
}

/// Compute null model log-likelihood (intercept-only model).
///
/// For the null model, π = mean(y) for all observations.
pub fn compute_null_log_likelihood(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mean_y = y.iter().sum::<f64>() / n;

    let epsilon = 1e-15;
    let mean_clipped = mean_y.max(epsilon).min(1.0 - epsilon);

    let ll_null = y.iter().map(|&yi| {
        yi * mean_clipped.ln() + (1.0 - yi) * (1.0 - mean_clipped).ln()
    }).sum();

    ll_null
}

/// Compute McFadden's pseudo R².
///
/// R² = 1 - (LL_model / LL_null)
pub fn compute_pseudo_r_squared(ll_model: f64, ll_null: f64) -> f64 {
    if ll_null == 0.0 {
        0.0
    } else {
        1.0 - (ll_model / ll_null)
    }
}

// ============================================================================
// Newton-Raphson MLE solver
// ============================================================================

/// Maximum number of Newton-Raphson iterations (REQ-012)
const MAX_ITERATIONS: usize = 35;

/// Convergence tolerance for gradient norm (REQ-013)
const CONVERGENCE_TOL: f64 = 1e-8;

/// Minimum weight floor to prevent numerical issues
const WEIGHT_FLOOR: f64 = 1e-10;

/// Threshold for detecting perfect separation (|β| > threshold)
const SEPARATION_THRESHOLD: f64 = 20.0;

/// Maximum step halvings per iteration
const MAX_STEP_HALVINGS: i32 = 10;

/// Detect perfect separation by checking for coefficient divergence.
///
/// Returns true if separation is detected.
fn detect_separation(beta: &[f64], pi: &[f64]) -> bool {
    // Check if any coefficient is too large
    if beta.iter().any(|&b| b.abs() > SEPARATION_THRESHOLD) {
        return true;
    }

    // Check if all predictions are near 0 or 1
    let all_extreme = pi.iter().all(|&p| !(1e-7..=(1.0 - 1e-7)).contains(&p));
    if all_extreme && !pi.is_empty() {
        return true;
    }

    false
}

/// Check for NaN or Inf in a vector.
fn has_invalid_values(v: &[f64]) -> bool {
    v.iter().any(|&x| x.is_nan() || x.is_infinite())
}

/// Compute Newton step with step halving for numerical stability.
///
/// Returns true if a valid step was found, false otherwise.
fn newton_step_with_halving(
    beta: &mut [f64],
    delta: &[f64],
    x: &[Vec<f64>],
    y: &[f64],
) -> bool {
    let old_ll = compute_log_likelihood(x, y, beta);
    let p = beta.len();

    for halving in 0..MAX_STEP_HALVINGS {
        let step_size = 0.5_f64.powi(halving);

        // Compute new beta
        let beta_new: Vec<f64> = beta
            .iter()
            .zip(delta.iter())
            .map(|(&b, &d)| b + step_size * d)
            .collect();

        // Check for invalid values
        if has_invalid_values(&beta_new) {
            continue;
        }

        let new_ll = compute_log_likelihood(x, y, &beta_new);

        // Accept step if log-likelihood improves (or is close enough)
        if new_ll >= old_ll - 1e-8 || (halving == MAX_STEP_HALVINGS - 1 && !new_ll.is_nan()) {
            beta[..p].copy_from_slice(&beta_new[..p]);
            return true;
        }
    }

    false
}

/// Compute logistic regression MLE using Newton-Raphson algorithm.
///
/// # Arguments
/// * `x` - Design matrix (n × p), including intercept column if applicable
/// * `y` - Binary outcome vector (n,), values must be 0 or 1
///
/// # Returns
/// * `Result<MleResult, LogisticError>` - MLE result or error
pub fn compute_logistic_mle(x: &[Vec<f64>], y: &[f64]) -> Result<MleResult, LogisticError> {
    let n = x.len();
    if n == 0 {
        return Err(LogisticError::NumericalInstability {
            message: "Cannot perform regression on empty data".to_string(),
        });
    }

    let p = x[0].len();
    if p == 0 {
        return Err(LogisticError::NumericalInstability {
            message: "Design matrix must have at least one column".to_string(),
        });
    }

    // Initialize β = 0
    let mut beta = vec![0.0; p];
    let mut converged = false;
    let mut iterations = 0;
    let mut info_inv = vec![vec![0.0; p]; p];
    let mut pi = vec![0.0; n];

    for iter in 0..MAX_ITERATIONS {
        iterations = iter + 1;

        // Compute π = sigmoid(Xβ)
        pi = x
            .iter()
            .map(|xi| sigmoid(dot(xi, &beta)))
            .collect();

        // Check for perfect separation
        if detect_separation(&beta, &pi) {
            return Err(LogisticError::PerfectSeparation);
        }

        // Compute weights W = diag(π(1-π)) with floor for stability
        let weights: Vec<f64> = pi
            .iter()
            .map(|&p| (p * (1.0 - p)).max(WEIGHT_FLOOR))
            .collect();

        // Compute gradient: X'(y - π)
        let residuals: Vec<f64> = y.iter().zip(&pi).map(|(&yi, &pi)| yi - pi).collect();
        let gradient = xt_vector_multiply(x, &residuals);

        // Check for NaN in gradient
        if has_invalid_values(&gradient) {
            return Err(LogisticError::NumericalInstability {
                message: "NaN detected in gradient computation".to_string(),
            });
        }

        // Check convergence: ||∇L|| < tol
        let grad_norm = gradient.iter().map(|g| g.powi(2)).sum::<f64>().sqrt();
        if grad_norm < CONVERGENCE_TOL {
            converged = true;
            // Compute final info_inv before breaking
            let hessian = xt_w_x(x, &weights);
            info_inv = invert_matrix(&hessian)?;
            break;
        }

        // Compute Hessian: X'WX
        let hessian = xt_w_x(x, &weights);

        // Check condition number (simple estimate using diagonal ratio)
        let diag: Vec<f64> = (0..p).map(|i| hessian[i][i]).collect();
        let max_diag = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_diag = diag.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_diag <= 0.0 || max_diag / min_diag > 1e10 {
            return Err(LogisticError::NumericalInstability {
                message: "Numerical instability detected; check for extreme values or collinearity"
                    .to_string(),
            });
        }

        // Invert Hessian
        info_inv = invert_matrix(&hessian)?;

        // Compute Newton direction: δ = H⁻¹ × g
        let delta = matrix_vector_multiply(&info_inv, &gradient);

        // Apply Newton step with step halving
        if !newton_step_with_halving(&mut beta, &delta, x, y) {
            // Could not improve - check if we're close to optimum
            if grad_norm < CONVERGENCE_TOL * 100.0 {
                converged = true;
            }
            break;
        }

        // Check for NaN/Inf in beta
        if has_invalid_values(&beta) {
            return Err(LogisticError::NumericalInstability {
                message: "NaN/Inf detected in coefficient estimates".to_string(),
            });
        }
    }

    if !converged {
        return Err(LogisticError::ConvergenceFailure { iterations });
    }

    // Compute final log-likelihood
    let log_likelihood = compute_log_likelihood(x, y, &beta);

    // Verify no NaN/Inf in final values
    if log_likelihood.is_nan() || log_likelihood.is_infinite() {
        return Err(LogisticError::NumericalInstability {
            message: "Invalid log-likelihood at convergence".to_string(),
        });
    }

    Ok(MleResult {
        beta,
        converged,
        iterations,
        log_likelihood,
        info_inv,
        pi,
    })
}

// ============================================================================
// HC3 Standard Errors for Logistic Regression
// ============================================================================

/// Compute weighted leverages for logistic regression.
///
/// h_ii = w_i × x_i' (X'WX)⁻¹ x_i
fn compute_weighted_leverages(
    x: &[Vec<f64>],
    weights: &[f64],
    info_inv: &[Vec<f64>],
) -> Result<Vec<f64>, LogisticError> {
    let n = x.len();
    let mut leverages = Vec::with_capacity(n);

    for (i, (xi, &wi)) in x.iter().zip(weights.iter()).enumerate() {
        // Compute temp = (X'WX)⁻¹ × x_i
        let temp = matrix_vector_multiply(info_inv, xi);

        // Compute h_ii = w_i × x_i · temp
        let h_ii = wi * dot(xi, &temp);

        if h_ii >= 0.99 {
            return Err(LogisticError::NumericalInstability {
                message: format!(
                    "Observation {} has leverage ≥ 0.99; HC3 standard errors may be unreliable",
                    i
                ),
            });
        }

        leverages.push(h_ii);
    }

    Ok(leverages)
}

/// Compute HC3 standard errors for logistic regression.
///
/// Uses the sandwich formula with weighted leverages:
/// V = I⁻¹ M I⁻¹ where M = Σᵢ xᵢ xᵢ' × (yᵢ - πᵢ)² / (1 - hᵢᵢ)²
///
/// # Arguments
/// * `x` - Design matrix (n × p)
/// * `y` - Binary outcome (n,)
/// * `pi` - Predicted probabilities (n,)
/// * `info_inv` - Inverse information matrix (X'WX)⁻¹ (p × p)
///
/// # Returns
/// * `Result<Vec<f64>, LogisticError>` - Standard errors (p,)
pub fn compute_hc3_logistic(
    x: &[Vec<f64>],
    y: &[f64],
    pi: &[f64],
    info_inv: &[Vec<f64>],
) -> Result<Vec<f64>, LogisticError> {
    let _n = x.len();
    let p = info_inv.len();

    // Compute weights: W = diag(π(1-π))
    let weights: Vec<f64> = pi.iter().map(|&p| p * (1.0 - p)).collect();

    // Compute weighted leverages
    let leverages = compute_weighted_leverages(x, &weights, info_inv)?;

    // Compute HC3 meat matrix
    let mut meat = vec![vec![0.0; p]; p];
    for (i, x_row) in x.iter().enumerate() {
        let resid = y[i] - pi[i];
        let one_minus_h = 1.0 - leverages[i];
        let omega = resid.powi(2) / one_minus_h.powi(2);

        for (j, meat_row) in meat.iter_mut().enumerate() {
            for k in 0..p {
                meat_row[k] += x_row[j] * omega * x_row[k];
            }
        }
    }

    // Sandwich: V = I⁻¹ M I⁻¹
    let temp = matrix_multiply(info_inv, &meat);
    let vcov = matrix_multiply(&temp, info_inv);

    // Extract diagonal and compute SE
    let se: Vec<f64> = (0..p).map(|i| vcov[i][i].sqrt()).collect();

    // Check for invalid values
    if se.iter().any(|&s| s.is_nan() || s.is_infinite() || s < 0.0) {
        return Err(LogisticError::NumericalInstability {
            message: "HC3 standard error computation produced invalid values".to_string(),
        });
    }

    Ok(se)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_relative_eq!(dot(&a, &b), 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_bounds() {
        // Log-likelihood should always be negative (or zero for perfect fit)
        let x = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
        let y = vec![0.0, 1.0];
        let beta = vec![0.0, 1.0];

        let ll = compute_log_likelihood(&x, &y, &beta);
        assert!(ll <= 0.0);
    }

    #[test]
    fn test_null_log_likelihood() {
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let ll_null = compute_null_log_likelihood(&y);
        // For 50/50 split, null LL should be n * ln(0.5) = 4 * (-0.693) ≈ -2.77
        assert!(ll_null < 0.0);
        assert_relative_eq!(ll_null, 4.0 * 0.5_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_pseudo_r_squared_bounds() {
        // Pseudo R² should be in [0, 1] for reasonable models
        let ll_model = -10.0;
        let ll_null = -20.0;
        let r2 = compute_pseudo_r_squared(ll_model, ll_null);
        assert!(r2 >= 0.0 && r2 <= 1.0);
        assert_relative_eq!(r2, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mle_simple_convergence() {
        // Simple test: y perfectly predicted by x
        // Create data where y=0 when x<0.5, y=1 when x>0.5
        let x = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.2],
            vec![1.0, 0.4],
            vec![1.0, 0.6],
            vec![1.0, 0.8],
            vec![1.0, 1.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let result = compute_logistic_mle(&x, &y);
        assert!(result.is_ok());
        let mle = result.unwrap();
        assert!(mle.converged);
        assert!(mle.iterations <= MAX_ITERATIONS);
    }

    #[test]
    fn test_hc3_produces_positive_se() {
        let x = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.2],
            vec![1.0, 0.4],
            vec![1.0, 0.6],
            vec![1.0, 0.8],
            vec![1.0, 1.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mle = compute_logistic_mle(&x, &y).unwrap();
        let se = compute_hc3_logistic(&x, &y, &mle.pi, &mle.info_inv).unwrap();

        assert_eq!(se.len(), 2);
        assert!(se[0] > 0.0);
        assert!(se[1] > 0.0);
    }

    #[test]
    fn test_separation_detection() {
        // Large coefficients should trigger separation detection
        let beta = vec![25.0, -25.0];
        let pi = vec![0.999999, 0.000001];
        assert!(detect_separation(&beta, &pi));

        // Normal coefficients should not
        let beta_normal = vec![1.0, 2.0];
        let pi_normal = vec![0.3, 0.7];
        assert!(!detect_separation(&beta_normal, &pi_normal));
    }

    #[test]
    fn test_invert_matrix_2x2() {
        // Test matrix inversion on simple 2x2
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let a_inv = invert_matrix(&a).unwrap();

        // Expected: [[0.6, -0.7], [-0.2, 0.4]]
        assert_relative_eq!(a_inv[0][0], 0.6, epsilon = 1e-10);
        assert_relative_eq!(a_inv[0][1], -0.7, epsilon = 1e-10);
        assert_relative_eq!(a_inv[1][0], -0.2, epsilon = 1e-10);
        assert_relative_eq!(a_inv[1][1], 0.4, epsilon = 1e-10);
    }

    #[test]
    fn test_singular_matrix_error() {
        // Singular matrix should return error
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let result = invert_matrix(&a);
        assert!(result.is_err());
    }
}
