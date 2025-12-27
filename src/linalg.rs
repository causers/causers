//! Optimized linear algebra operations using faer.
//!
//! This module provides high-performance matrix operations for linear regression,
//! leveraging the faer crate for BLAS-like performance without external dependencies.

use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{matmul as tri_matmul, BlockStructure};
use faer::prelude::*;
use faer::{Col, Mat, Parallelism};
use pyo3::PyResult;

/// Error type for linear algebra operations
#[derive(Debug)]
pub enum LinalgError {
    /// Matrix is singular or not positive definite
    SingularMatrix,
    /// Numerical instability detected (NaN/Inf in result)
    NumericalInstability,
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinalgError::SingularMatrix => write!(
                f,
                "Singular matrix: cannot solve linear regression (X'X is not invertible, check for collinearity)"
            ),
            LinalgError::NumericalInstability => {
                write!(f, "Numerical instability: NaN or Inf in computation")
            }
            LinalgError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, got
                )
            }
        }
    }
}

impl std::error::Error for LinalgError {}

impl From<LinalgError> for pyo3::PyErr {
    fn from(e: LinalgError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    }
}

/// Convert a row-major Vec<Vec<f64>> to a faer Mat<f64> (column-major).
///
/// faer uses column-major storage, so we need to transpose during conversion
/// for optimal memory access patterns.
pub fn vec_to_mat(data: &[Vec<f64>]) -> Mat<f64> {
    if data.is_empty() {
        return Mat::zeros(0, 0);
    }

    let n_rows = data.len();
    let n_cols = data[0].len();

    Mat::from_fn(n_rows, n_cols, |i, j| data[i][j])
}

/// Build a faer Mat directly from a flat row-major array with optional intercept column.
///
/// This function eliminates the Vec<Vec<f64>> intermediate format by constructing
/// the faer::Mat directly from the flat numpy array data.
///
/// # Arguments
/// * `flat_data` - Flat array in row-major order (n_rows × n_cols elements)
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns in the source data (not including intercept)
/// * `include_intercept` - If true, prepend a column of 1.0s
///
/// # Returns
/// * `Mat<f64>` - Design matrix of shape (n_rows × (n_cols + intercept))
///
/// # Panics
/// * Panics if flat_data.len() != n_rows * n_cols
pub fn flat_to_mat_with_intercept(
    flat_data: &[f64],
    n_rows: usize,
    n_cols: usize,
    include_intercept: bool,
) -> Mat<f64> {
    debug_assert_eq!(
        flat_data.len(),
        n_rows * n_cols,
        "flat_data length {} != n_rows {} * n_cols {}",
        flat_data.len(),
        n_rows,
        n_cols
    );

    if include_intercept {
        // Output has n_cols + 1 columns (intercept + original columns)
        Mat::from_fn(n_rows, n_cols + 1, |i, j| {
            if j == 0 {
                1.0 // Intercept column
            } else {
                // Row-major: element [i, j-1] is at index i * n_cols + (j - 1)
                flat_data[i * n_cols + (j - 1)]
            }
        })
    } else {
        // No intercept, just convert directly
        Mat::from_fn(n_rows, n_cols, |i, j| {
            // Row-major: element [i, j] is at index i * n_cols + j
            flat_data[i * n_cols + j]
        })
    }
}

/// Build a faer Mat directly from a flat row-major array without intercept.
///
/// Convenience function when intercept is not needed.
///
/// # Arguments
/// * `flat_data` - Flat array in row-major order (n_rows × n_cols elements)
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
///
/// # Returns
/// * `Mat<f64>` - Matrix of shape (n_rows × n_cols)
#[inline]
pub fn flat_to_mat(flat_data: &[f64], n_rows: usize, n_cols: usize) -> Mat<f64> {
    flat_to_mat_with_intercept(flat_data, n_rows, n_cols, false)
}

/// Convert a faer Mat<f64> back to Vec<Vec<f64>> (row-major).
pub fn mat_to_vec(mat: &Mat<f64>) -> Vec<Vec<f64>> {
    let n_rows = mat.nrows();
    let n_cols = mat.ncols();

    (0..n_rows)
        .map(|i| (0..n_cols).map(|j| mat.read(i, j)).collect())
        .collect()
}

/// Convert a Vec<f64> to a faer column vector.
pub fn vec_to_col(data: &[f64]) -> Col<f64> {
    Col::from_fn(data.len(), |i| data[i])
}

/// Convert a faer column vector to Vec<f64>.
pub fn col_to_vec(col: &Col<f64>) -> Vec<f64> {
    (0..col.nrows()).map(|i| col.read(i)).collect()
}

/// Compute X'X (X transpose times X) efficiently using faer.
///
/// This is the core operation for computing the normal equations.
/// Uses BLAS gemm internally for optimal performance.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
///
/// # Returns
/// * `Mat<f64>` - Gram matrix X'X of shape (p × p)
pub fn xtx(x: &Mat<f64>) -> Mat<f64> {
    let p = x.ncols();
    let mut result = Mat::zeros(p, p);

    // result = X' × X
    // Using faer's matmul: C = alpha * A * B + beta * C
    // Use Rayon parallelism for large matrices (0 = auto-detect thread count)
    matmul(
        result.as_mut(),
        x.transpose(),
        x.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    result
}

/// Compute X'y (X transpose times y) efficiently using faer.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `y` - Response vector y of shape (n,)
///
/// # Returns
/// * `Vec<f64>` - Vector X'y of shape (p,)
pub fn xty(x: &Mat<f64>, y: &[f64]) -> Vec<f64> {
    let p = x.ncols();
    let y_col = vec_to_col(y);
    let mut result = Col::zeros(p);

    // result = X' × y
    // Use Rayon parallelism for large matrices
    matmul(
        result.as_mut().as_2d_mut(),
        x.transpose(),
        y_col.as_ref().as_2d(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    col_to_vec(&result)
}

/// Compute matrix-vector product Xβ efficiently using faer.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `beta` - Coefficient vector β of shape (p,)
///
/// # Returns
/// * `Vec<f64>` - Predicted values Xβ of shape (n,)
pub fn mat_vec_mul(x: &Mat<f64>, beta: &[f64]) -> Vec<f64> {
    let n = x.nrows();
    let beta_col = vec_to_col(beta);
    let mut result = Col::zeros(n);

    // Use Rayon parallelism for large matrices
    matmul(
        result.as_mut().as_2d_mut(),
        x.as_ref(),
        beta_col.as_ref().as_2d(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    col_to_vec(&result)
}

/// Solve the normal equations (X'X)β = X'y via Cholesky decomposition.
///
/// This is more numerically stable and faster than explicit matrix inversion.
/// The Cholesky decomposition requires X'X to be positive definite, which
/// is guaranteed when X has full column rank.
///
/// # Arguments
/// * `xtx` - Gram matrix X'X of shape (p × p), must be positive definite
/// * `xty` - Vector X'y of shape (p,)
///
/// # Returns
/// * `Result<Vec<f64>, LinalgError>` - Solution β or error if singular
pub fn solve_normal_equations(xtx: &Mat<f64>, xty: &[f64]) -> Result<Vec<f64>, LinalgError> {
    let p = xtx.nrows();
    if xtx.ncols() != p {
        return Err(LinalgError::DimensionMismatch {
            expected: p,
            got: xtx.ncols(),
        });
    }
    if xty.len() != p {
        return Err(LinalgError::DimensionMismatch {
            expected: p,
            got: xty.len(),
        });
    }

    // Try Cholesky decomposition (for SPD matrices, which X'X should be)
    let chol = xtx.cholesky(faer::Side::Lower);

    // If Cholesky fails, the matrix is not positive definite (singular)
    let chol = match chol {
        Ok(c) => c,
        Err(_) => return Err(LinalgError::SingularMatrix),
    };

    // Convert xty to a column matrix for solving
    let b = vec_to_col(xty);
    let b_mat = b.as_ref().as_2d();

    // Solve the system using the Cholesky factorization
    let solution = chol.solve(&b_mat);

    // Extract the solution
    let result: Vec<f64> = (0..p).map(|i| solution.read(i, 0)).collect();

    // Verify solution is valid (no NaN/Inf)
    if result.iter().any(|&b| b.is_nan() || b.is_infinite()) {
        return Err(LinalgError::NumericalInstability);
    }

    Ok(result)
}

/// Compute (X'X)^-1 via Cholesky decomposition.
///
/// This computes the explicit inverse, which is needed for standard error
/// calculations. Uses Cholesky decomposition for numerical stability.
///
/// # Arguments
/// * `xtx` - Gram matrix X'X of shape (p × p), must be positive definite
///
/// # Returns
/// * `Result<Mat<f64>, LinalgError>` - Inverse (X'X)^-1 or error if singular
pub fn invert_xtx(xtx: &Mat<f64>) -> Result<Mat<f64>, LinalgError> {
    let p = xtx.nrows();
    if xtx.ncols() != p {
        return Err(LinalgError::DimensionMismatch {
            expected: p,
            got: xtx.ncols(),
        });
    }

    // Try Cholesky decomposition
    let chol = xtx.cholesky(faer::Side::Lower);

    let chol = match chol {
        Ok(c) => c,
        Err(_) => return Err(LinalgError::SingularMatrix),
    };

    // Solve for inverse by solving A * X = I
    let identity: Mat<f64> = Mat::identity(p, p);
    let inverse: Mat<f64> = chol.solve(&identity);

    // Verify result is valid
    for i in 0..p {
        for j in 0..p {
            if inverse.read(i, j).is_nan() || inverse.read(i, j).is_infinite() {
                return Err(LinalgError::NumericalInstability);
            }
        }
    }

    Ok(inverse)
}

/// Compute leverage values h_ii = diag(X(X'X)^-1X') efficiently.
///
/// Instead of computing n separate matrix-vector products, this function
/// computes all leverages in a batch operation by computing the diagonal
/// of the hat matrix H = X(X'X)^-1X'.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `xtx_inv` - Inverse of X'X of shape (p × p)
///
/// # Returns
/// * `PyResult<Vec<f64>>` - Leverage values h_ii or error if extreme leverage
pub fn compute_leverages_batch(x: &Mat<f64>, xtx_inv: &Mat<f64>) -> PyResult<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();

    // Compute Z = X × (X'X)^-1, shape (n × p)
    // Use Rayon parallelism for large matrices
    let mut z = Mat::zeros(n, p);
    matmul(
        z.as_mut(),
        x.as_ref(),
        xtx_inv.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    // h_ii = Z[i,:] · X[i,:] = sum over j of Z[i,j] * X[i,j]
    // This is the diagonal of Z × X'
    let mut leverages = Vec::with_capacity(n);
    for i in 0..n {
        let mut h_ii = 0.0;
        for j in 0..p {
            h_ii += z.read(i, j) * x.read(i, j);
        }

        if h_ii >= 0.99 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Observation {} has leverage ≥ 0.99; HC3 standard errors may be unreliable due to extreme leverage.",
                i
            )));
        }

        leverages.push(h_ii);
    }

    Ok(leverages)
}

/// Compute HC3 variance-covariance matrix using the sandwich formula.
///
/// Formula: Var(β) = (X'X)^-1 × meat × (X'X)^-1
/// where meat = X' Ω X and Ω is diagonal with Ω_ii = e_i² / (1 - h_ii)²
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `residuals` - OLS residuals of shape (n,)
/// * `leverages` - Leverage values h_ii of shape (n,)
/// * `xtx_inv` - Inverse of X'X of shape (p × p)
///
/// # Returns
/// * `Mat<f64>` - HC3 variance-covariance matrix (p × p)
pub fn compute_hc3_vcov_faer(
    x: &Mat<f64>,
    residuals: &[f64],
    leverages: &[f64],
    xtx_inv: &Mat<f64>,
) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();

    // Compute X_weighted where X_weighted[i,:] = x[i,:] * sqrt(omega_ii)
    // omega_ii = e_i² / (1 - h_ii)²
    let mut x_weighted = Mat::zeros(n, p);
    for i in 0..n {
        let one_minus_h = 1.0 - leverages[i];
        let sqrt_omega = residuals[i].abs() / one_minus_h;
        for j in 0..p {
            x_weighted.write(i, j, x.read(i, j) * sqrt_omega);
        }
    }

    // Compute meat = X_weighted' × X_weighted = X' Ω X
    // Use Rayon parallelism for large matrices
    let mut meat = Mat::zeros(p, p);
    matmul(
        meat.as_mut(),
        x_weighted.transpose(),
        x_weighted.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    // Compute sandwich: (X'X)^-1 × meat × (X'X)^-1
    let mut temp = Mat::zeros(p, p);
    matmul(
        temp.as_mut(),
        xtx_inv.as_ref(),
        meat.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    let mut result = Mat::zeros(p, p);
    matmul(
        result.as_mut(),
        temp.as_ref(),
        xtx_inv.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    result
}

// ============================================================================
// Weighted matrix operations for logistic regression
// ============================================================================

/// Compute X'WX where W is a diagonal matrix represented as a vector.
///
/// This is the core operation for computing the Hessian in logistic regression.
/// Uses efficient diagonal scaling followed by matrix multiplication.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `w` - Diagonal weight vector (n,), representing W = diag(w)
///
/// # Returns
/// * `Mat<f64>` - Weighted Gram matrix X'WX of shape (p × p)
pub fn compute_xtwx(x: &Mat<f64>, w: &[f64]) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();

    // Create X_weighted where each row is scaled by sqrt(w_i)
    // X_weighted[i,:] = sqrt(w_i) * X[i,:]
    // Then X'WX = X_weighted' × X_weighted
    let mut x_weighted = Mat::zeros(n, p);
    for i in 0..n {
        let sqrt_w = w[i].sqrt();
        for j in 0..p {
            x_weighted.write(i, j, x.read(i, j) * sqrt_w);
        }
    }

    // Compute X_weighted' × X_weighted = X'WX
    // Use Rayon parallelism for large matrices
    let mut result = Mat::zeros(p, p);
    matmul(
        result.as_mut(),
        x_weighted.transpose(),
        x_weighted.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    result
}

/// Compute X'r (X transpose times a residual/response vector).
///
/// This is used for computing the gradient X'(y - π) in logistic regression.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `r` - Residual vector (n,)
///
/// # Returns
/// * `Vec<f64>` - Vector X'r of shape (p,)
pub fn compute_xtr(x: &Mat<f64>, r: &[f64]) -> Vec<f64> {
    // This is the same as xty, just with different semantic meaning
    xty(x, r)
}

/// Solve Hx = g via Cholesky decomposition without explicit inversion.
///
/// This is more efficient than computing H^-1 then multiplying:
/// - Cholesky solve: O(p²) after O(p³/3) decomposition
/// - Explicit inversion: O(p³) inversion + O(p²) multiply
///
/// For repeated solves with the same H, consider caching the Cholesky factor.
///
/// # Arguments
/// * `h` - Symmetric positive definite matrix H of shape (p × p)
/// * `g` - Right-hand side vector g of shape (p,)
///
/// # Returns
/// * `Result<Vec<f64>, LinalgError>` - Solution x or error if H is singular
pub fn cholesky_solve(h: &Mat<f64>, g: &[f64]) -> Result<Vec<f64>, LinalgError> {
    let p = h.nrows();
    if h.ncols() != p {
        return Err(LinalgError::DimensionMismatch {
            expected: p,
            got: h.ncols(),
        });
    }
    if g.len() != p {
        return Err(LinalgError::DimensionMismatch {
            expected: p,
            got: g.len(),
        });
    }

    // Attempt Cholesky decomposition
    let chol = match h.cholesky(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return Err(LinalgError::SingularMatrix),
    };

    // Convert g to column matrix for solving
    let b = vec_to_col(g);
    let b_mat = b.as_ref().as_2d();

    // Solve the system
    let solution = chol.solve(&b_mat);

    // Extract solution
    let result: Vec<f64> = (0..p).map(|i| solution.read(i, 0)).collect();

    // Verify solution is valid
    if result.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return Err(LinalgError::NumericalInstability);
    }

    Ok(result)
}

/// Solve using a precomputed (X'X)^-1 or (X'WX)^-1 matrix.
///
/// # Arguments
/// * `h_inv` - Precomputed inverse matrix (p × p)
/// * `g` - Right-hand side vector (p,)
///
/// # Returns
/// * `Vec<f64>` - Solution vector (p,)
pub fn solve_with_inverse(h_inv: &Mat<f64>, g: &[f64]) -> Vec<f64> {
    mat_vec_mul(h_inv, g)
}

/// Compute weighted leverages h_ii = w_i × x_i' (X'WX)^-1 x_i for logistic regression.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `weights` - Diagonal weights W = diag(π(1-π)) of shape (n,)
/// * `xtwx_inv` - Inverse of X'WX of shape (p × p)
///
/// # Returns
/// * `Result<Vec<f64>, LinalgError>` - Weighted leverage values h_ii or error
pub fn compute_weighted_leverages_batch(
    x: &Mat<f64>,
    weights: &[f64],
    xtwx_inv: &Mat<f64>,
) -> Result<Vec<f64>, LinalgError> {
    let n = x.nrows();
    let p = x.ncols();

    // Compute Z = X × (X'WX)^-1, shape (n × p)
    // Use Rayon parallelism for large matrices
    let mut z = Mat::zeros(n, p);
    matmul(
        z.as_mut(),
        x.as_ref(),
        xtwx_inv.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    // h_ii = w_i × (Z[i,:] · X[i,:]) = w_i × sum over j of Z[i,j] * X[i,j]
    let mut leverages = Vec::with_capacity(n);
    for i in 0..n {
        let mut h_ii = 0.0;
        for j in 0..p {
            h_ii += z.read(i, j) * x.read(i, j);
        }
        h_ii *= weights[i];

        if h_ii >= 0.99 {
            return Err(LinalgError::NumericalInstability);
        }

        leverages.push(h_ii);
    }

    Ok(leverages)
}

/// Compute HC3 variance-covariance matrix for logistic regression.
///
/// Formula: Var(β) = (X'WX)^-1 × meat × (X'WX)^-1
/// where meat = X' Ω X and Ω is diagonal with Ω_ii = (y_i - π_i)² / (1 - h_ii)²
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `residuals` - Pearson residuals (y - π) of shape (n,)
/// * `leverages` - Weighted leverage values h_ii of shape (n,)
/// * `xtwx_inv` - Inverse of X'WX of shape (p × p)
///
/// # Returns
/// * `Mat<f64>` - HC3 variance-covariance matrix (p × p)
pub fn compute_hc3_logistic_vcov(
    x: &Mat<f64>,
    residuals: &[f64],
    leverages: &[f64],
    xtwx_inv: &Mat<f64>,
) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();

    // Compute X_weighted where X_weighted[i,:] = x[i,:] * |residual_i| / (1 - h_ii)
    let mut x_weighted = Mat::zeros(n, p);
    for i in 0..n {
        let one_minus_h = 1.0 - leverages[i];
        let scale = residuals[i].abs() / one_minus_h;
        for j in 0..p {
            x_weighted.write(i, j, x.read(i, j) * scale);
        }
    }

    // Compute meat = X_weighted' × X_weighted = X' Ω X
    // Use Rayon parallelism for large matrices
    let mut meat = Mat::zeros(p, p);
    matmul(
        meat.as_mut(),
        x_weighted.transpose(),
        x_weighted.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    // Compute sandwich: (X'WX)^-1 × meat × (X'WX)^-1
    let mut temp = Mat::zeros(p, p);
    matmul(
        temp.as_mut(),
        xtwx_inv.as_ref(),
        meat.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    let mut result = Mat::zeros(p, p);
    matmul(
        result.as_mut(),
        temp.as_ref(),
        xtwx_inv.as_ref(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    result
}

// ============================================================================
// In-place operations for Newton-Raphson loop optimization
// ============================================================================

/// Compute matrix-vector product Xβ in-place using direct loops.
///
/// Optimized for small p (number of predictors). For p <= 8, direct loops
/// with minimal overhead outperform SIMD matmul + conversion overhead.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `beta` - Coefficient vector β of shape (p,)
/// * `result` - Pre-allocated output buffer of shape (n,)
#[inline]
pub fn mat_vec_mul_inplace_direct(x: &Mat<f64>, beta: &[f64], result: &mut [f64]) {
    let n = x.nrows();
    let p = x.ncols();
    debug_assert_eq!(result.len(), n, "result buffer must have length n");
    debug_assert_eq!(beta.len(), p, "beta must have length p");

    // Direct loop: result[i] = sum_j x[i,j] * beta[j]
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..p {
            sum += x.read(i, j) * beta[j];
        }
        result[i] = sum;
    }
}

/// Compute matrix-vector product Xβ in-place using faer matmul.
///
/// Uses faer's SIMD-optimized matmul with pre-allocated Col buffer.
/// Better for large p, but has conversion overhead.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `beta` - Coefficient vector β of shape (p,)
/// * `result` - Pre-allocated output buffer of shape (n,)
/// * `result_col` - Pre-allocated faer::Col buffer of shape (n,)
#[inline]
pub fn mat_vec_mul_inplace(x: &Mat<f64>, beta: &[f64], result: &mut [f64], result_col: &mut Col<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    debug_assert_eq!(result.len(), n, "result buffer must have length n");
    debug_assert_eq!(beta.len(), p, "beta must have length p");
    debug_assert_eq!(result_col.nrows(), n, "result_col must have length n");

    // Convert beta to Col (small allocation, size p)
    let beta_col = vec_to_col(beta);

    // Fast SIMD matmul using pre-allocated result_col
    // Use Rayon parallelism for large matrices
    matmul(
        result_col.as_mut().as_2d_mut(),
        x.as_ref(),
        beta_col.as_ref().as_2d(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    // Copy result to output buffer
    for i in 0..n {
        result[i] = result_col.read(i);
    }
}

/// Compute X'r (X transpose times vector) in-place using direct loops.
///
/// Optimized for small p (number of predictors). Avoids allocation overhead.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `r` - Vector r of shape (n,)
/// * `result` - Pre-allocated output buffer of shape (p,)
#[inline]
pub fn compute_xtr_inplace_direct(x: &Mat<f64>, r: &[f64], result: &mut [f64]) {
    let n = x.nrows();
    let p = x.ncols();
    debug_assert_eq!(result.len(), p, "result buffer must have length p");
    debug_assert_eq!(r.len(), n, "r must have length n");

    // Initialize result to zero
    for j in 0..p {
        result[j] = 0.0;
    }

    // Direct loop: result[j] = sum_i x[i,j] * r[i]
    for i in 0..n {
        let ri = r[i];
        for j in 0..p {
            result[j] += x.read(i, j) * ri;
        }
    }
}

/// Compute X'r (X transpose times vector) in-place using faer matmul.
///
/// Uses faer's SIMD-optimized matmul with pre-allocated Col buffer.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `r` - Vector r of shape (n,)
/// * `result` - Pre-allocated output buffer of shape (p,)
/// * `result_col` - Pre-allocated faer::Col buffer of shape (p,)
#[inline]
pub fn compute_xtr_inplace(x: &Mat<f64>, r: &[f64], result: &mut [f64], result_col: &mut Col<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    debug_assert_eq!(result.len(), p, "result buffer must have length p");
    debug_assert_eq!(r.len(), n, "r must have length n");
    debug_assert_eq!(result_col.nrows(), p, "result_col must have length p");

    // Convert r to Col (allocation, size n - but using SIMD matmul is still faster)
    let r_col = vec_to_col(r);

    // Fast SIMD matmul using pre-allocated result_col
    // Use Rayon parallelism for large matrices
    matmul(
        result_col.as_mut().as_2d_mut(),
        x.transpose(),
        r_col.as_ref().as_2d(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    // Copy result to output buffer
    for i in 0..p {
        result[i] = result_col.read(i);
    }
}

/// Compute X'WX in-place, reusing pre-allocated buffers.
///
/// Uses faer matmul for the O(p²n) matrix multiply which benefits from BLAS.
///
/// # Arguments
/// * `x` - Design matrix X of shape (n × p)
/// * `w` - Diagonal weight vector (n,)
/// * `x_weighted` - Pre-allocated buffer for weighted X (n × p)
/// * `result` - Pre-allocated output matrix (p × p)
#[inline]
pub fn compute_xtwx_inplace(x: &Mat<f64>, w: &[f64], x_weighted: &mut Mat<f64>, result: &mut Mat<f64>) {
    let n = x.nrows();
    let p = x.ncols();

    // Scale rows of X by sqrt(w_i)
    for i in 0..n {
        let sqrt_w = w[i].sqrt();
        for j in 0..p {
            x_weighted.write(i, j, x.read(i, j) * sqrt_w);
        }
    }

    // Compute X_weighted' × X_weighted = X'WX using triangular matmul
    // Since X'WX is symmetric, only compute lower triangle for ~50% savings
    // Use Rayon parallelism for large matrices
    tri_matmul(
        result.as_mut(),
        BlockStructure::TriangularLower,  // Only compute lower triangle
        x_weighted.transpose(),
        BlockStructure::Rectangular,
        x_weighted.as_ref(),
        BlockStructure::Rectangular,
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    // Copy lower triangle to upper for symmetry (required for Cholesky decomposition)
    for i in 0..p {
        for j in (i + 1)..p {
            result.write(i, j, result.read(j, i));
        }
    }
}

/// Apply sigmoid function in-place: result[i] = 1 / (1 + exp(-x[i]))
///
/// Uses numerically stable formula.
///
/// # Arguments
/// * `linear_pred` - Input linear predictions (n,), will be overwritten with sigmoid values
#[inline]
pub fn sigmoid_inplace(linear_pred: &mut [f64]) {
    for x in linear_pred.iter_mut() {
        *x = if *x >= 0.0 {
            1.0 / (1.0 + (-*x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        };
    }
}

/// Compute weights π(1-π) in-place from pi values.
///
/// # Arguments
/// * `pi` - Predicted probabilities (n,)
/// * `weights` - Pre-allocated output buffer (n,), will be filled with π(1-π)
/// * `floor` - Minimum weight value for numerical stability
#[inline]
pub fn compute_weights_inplace(pi: &[f64], weights: &mut [f64], floor: f64) {
    debug_assert_eq!(pi.len(), weights.len());
    for (w, &p) in weights.iter_mut().zip(pi.iter()) {
        *w = (p * (1.0 - p)).max(floor);
    }
}

/// Compute residuals (y - π) in-place.
///
/// # Arguments
/// * `y` - Binary outcomes (n,)
/// * `pi` - Predicted probabilities (n,)
/// * `residuals` - Pre-allocated output buffer (n,)
#[inline]
pub fn compute_residuals_inplace(y: &[f64], pi: &[f64], residuals: &mut [f64]) {
    debug_assert_eq!(y.len(), pi.len());
    debug_assert_eq!(y.len(), residuals.len());
    for ((r, &yi), &pi) in residuals.iter_mut().zip(y.iter()).zip(pi.iter()) {
        *r = yi - pi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vec_to_mat_roundtrip() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mat = vec_to_mat(&data);
        let back = mat_to_vec(&mat);
        assert_eq!(data, back);
    }

    #[test]
    fn test_xtx_simple() {
        // X = [[1, 2], [3, 4], [5, 6]]
        // X'X = [[35, 44], [44, 56]]
        let x = vec_to_mat(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let result = xtx(&x);

        assert_relative_eq!(result.read(0, 0), 35.0, epsilon = 1e-10);
        assert_relative_eq!(result.read(0, 1), 44.0, epsilon = 1e-10);
        assert_relative_eq!(result.read(1, 0), 44.0, epsilon = 1e-10);
        assert_relative_eq!(result.read(1, 1), 56.0, epsilon = 1e-10);
    }

    #[test]
    fn test_xty_simple() {
        let x = vec_to_mat(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let y = vec![1.0, 2.0, 3.0];
        let result = xty(&x, &y);

        // X'y = [[1*1 + 3*2 + 5*3], [2*1 + 4*2 + 6*3]] = [[22], [28]]
        assert_relative_eq!(result[0], 22.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 28.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_normal_equations_identity() {
        // X'X = I, X'y = [1, 2, 3] => β = [1, 2, 3]
        let xtx = vec_to_mat(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let xty = vec![1.0, 2.0, 3.0];
        let beta = solve_normal_equations(&xtx, &xty).unwrap();

        assert_relative_eq!(beta[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(beta[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(beta[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_normal_equations_singular() {
        // Singular matrix
        let xtx = vec_to_mat(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        let xty = vec![1.0, 2.0];
        let result = solve_normal_equations(&xtx, &xty);
        assert!(result.is_err());
    }

    #[test]
    fn test_invert_xtx_identity() {
        let xtx = vec_to_mat(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let inv = invert_xtx(&xtx).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(inv.read(i, j), expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_invert_xtx_simple() {
        // [[2, 0], [0, 3]] => inverse [[0.5, 0], [0, 1/3]]
        let xtx = vec_to_mat(&[vec![2.0, 0.0], vec![0.0, 3.0]]);
        let inv = invert_xtx(&xtx).unwrap();

        assert_relative_eq!(inv.read(0, 0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(inv.read(0, 1), 0.0, epsilon = 1e-10);
        assert_relative_eq!(inv.read(1, 0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(inv.read(1, 1), 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mat_vec_mul() {
        let x = vec_to_mat(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let beta = vec![1.0, 1.0];
        let result = mat_vec_mul(&x, &beta);

        assert_relative_eq!(result[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_regression_integration() {
        // y = 2x + 1, perfect fit
        let design_matrix = vec![
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 4.0],
            vec![1.0, 5.0],
        ];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let x = vec_to_mat(&design_matrix);
        let xtx_mat = xtx(&x);
        let xty_vec = xty(&x, &y);
        let beta = solve_normal_equations(&xtx_mat, &xty_vec).unwrap();

        // β = [1, 2] (intercept=1, slope=2)
        assert_relative_eq!(beta[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(beta[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_xtwx() {
        // Test X'WX computation
        // X = [[1, 2], [3, 4], [5, 6]]
        // W = diag([1, 2, 3])
        // X'WX = sum over i of w_i * x_i * x_i'
        //      = 1*[1,2]'*[1,2] + 2*[3,4]'*[3,4] + 3*[5,6]'*[5,6]
        //      = [[1,2],[2,4]] + [[18,24],[24,32]] + [[75,90],[90,108]]
        //      = [[94, 116], [116, 144]]
        let x = vec_to_mat(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let w = vec![1.0, 2.0, 3.0];
        let result = compute_xtwx(&x, &w);

        assert_relative_eq!(result.read(0, 0), 94.0, epsilon = 1e-10);
        assert_relative_eq!(result.read(0, 1), 116.0, epsilon = 1e-10);
        assert_relative_eq!(result.read(1, 0), 116.0, epsilon = 1e-10);
        assert_relative_eq!(result.read(1, 1), 144.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_xtwx_uniform_weights() {
        // With uniform weights, X'WX = w * X'X
        let x = vec_to_mat(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let w = vec![2.0, 2.0, 2.0];
        let result = compute_xtwx(&x, &w);
        let xtx_result = xtx(&x);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result.read(i, j), 2.0 * xtx_result.read(i, j), epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky_solve() {
        // Solve Hx = g where H = [[4, 2], [2, 3]], g = [4, 5]
        // x = H^-1 * g = [[3/8, -1/4], [-1/4, 1/2]] * [4, 5] = [0.25, 1.5]
        let h = vec_to_mat(&[vec![4.0, 2.0], vec![2.0, 3.0]]);
        let g = vec![4.0, 5.0];
        let result = cholesky_solve(&h, &g).unwrap();

        assert_relative_eq!(result[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cholesky_solve_identity() {
        // Solve Ix = g where I is identity
        let h = vec_to_mat(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let g = vec![1.0, 2.0, 3.0];
        let result = cholesky_solve(&h, &g).unwrap();

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cholesky_solve_singular() {
        // Singular matrix should fail
        let h = vec_to_mat(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        let g = vec![1.0, 2.0];
        let result = cholesky_solve(&h, &g);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_xtr() {
        // X'r should be same as xty
        let x = vec_to_mat(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let r = vec![1.0, -1.0, 2.0];
        let result = compute_xtr(&x, &r);
        let expected = xty(&x, &r);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_weighted_leverages_batch() {
        // Simple test with identity-like structure
        let x = vec_to_mat(&[
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ]);
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        
        // Compute X'WX and its inverse
        let xtwx = compute_xtwx(&x, &weights);
        let xtwx_inv = invert_xtx(&xtwx).unwrap();
        
        let leverages = compute_weighted_leverages_batch(&x, &weights, &xtwx_inv);
        assert!(leverages.is_ok());
        
        let h = leverages.unwrap();
        assert_eq!(h.len(), 4);
        
        // All leverages should be positive and < 1
        for &h_ii in &h {
            assert!(h_ii >= 0.0);
            assert!(h_ii < 1.0);
        }
    }
}
