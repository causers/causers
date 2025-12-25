use pyo3::prelude::*;

/// Result of a linear regression computation
#[pyclass]
#[derive(Debug, Clone)]
pub struct LinearRegressionResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub intercept: Option<f64>,
    #[pyo3(get)]
    pub r_squared: f64,
    #[pyo3(get)]
    pub n_samples: usize,
    // Keep slope for backward compatibility (single covariate case)
    #[pyo3(get)]
    pub slope: Option<f64>,
    // HC3 robust standard errors for each coefficient (or clustered SE if cluster specified)
    #[pyo3(get)]
    pub standard_errors: Vec<f64>,
    // HC3 robust standard error for intercept (None if include_intercept=False)
    #[pyo3(get)]
    pub intercept_se: Option<f64>,
    // NEW: Number of unique clusters (None if not clustered)
    #[pyo3(get)]
    pub n_clusters: Option<usize>,
    // NEW: Type of clustered SE ("analytical" or "bootstrap", None if not clustered)
    #[pyo3(get)]
    pub cluster_se_type: Option<String>,
    // NEW: Number of bootstrap iterations used (None if not bootstrap)
    #[pyo3(get)]
    pub bootstrap_iterations_used: Option<usize>,
}

#[pymethods]
impl LinearRegressionResult {
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
            "LinearRegressionResult(coefficients={:?}, intercept={}, r_squared={:.6}, n_samples={}, standard_errors={:?}, intercept_se={}, n_clusters={}, cluster_se_type={}, bootstrap_iterations_used={})",
            self.coefficients, intercept_str, self.r_squared, self.n_samples, self.standard_errors, intercept_se_str, n_clusters_str, cluster_se_type_str, bootstrap_iter_str
        )
    }

    fn __str__(&self) -> String {
        let intercept_str = match self.intercept {
            Some(i) => format!(" + {:.6}", i),
            None => "".to_string(),
        };

        if self.coefficients.len() == 1 {
            // For single covariate, show coefficient with SE
            let se_str = if !self.standard_errors.is_empty() {
                format!(" ± {:.6}", self.standard_errors[0])
            } else {
                "".to_string()
            };
            format!(
                "y = {:.6}{}x{}(R² = {:.6}, n = {})",
                self.coefficients[0], se_str, intercept_str, self.r_squared, self.n_samples
            )
        } else {
            let terms: Vec<String> = self
                .coefficients
                .iter()
                .enumerate()
                .map(|(i, &c)| format!("{:.6}*x{}", c, i + 1))
                .collect();
            format!(
                "y = {}{}(R² = {:.6}, n = {})",
                terms.join(" + "),
                intercept_str,
                self.r_squared,
                self.n_samples
            )
        }
    }
}

/// Compute multiple linear regression using ordinary least squares
///
/// Supports multiple covariates with matrix operations: β = (X'X)^-1 X'y
///
/// # Arguments
/// * `x` - Matrix of predictor variables (rows are observations, columns are variables)
/// * `y` - Vector of response variable
/// * `include_intercept` - Whether to include an intercept term
///
/// Note: This function is kept for standalone module use and unit tests.
/// The main library uses compute_linear_regression_with_cluster in lib.rs.
#[allow(dead_code)]
pub fn compute_linear_regression(
    x: &[Vec<f64>],
    y: &[f64],
    include_intercept: bool,
) -> PyResult<LinearRegressionResult> {
    if x.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot perform regression on empty data",
        ));
    }

    let n = x.len();
    if n != y.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x and y must have the same number of rows: x has {}, y has {}",
            n,
            y.len()
        )));
    }

    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot perform regression on empty data",
        ));
    }

    let n_vars = x[0].len();
    if n_vars == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must have at least one variable",
        ));
    }

    // Check all rows have same number of variables
    for (i, row) in x.iter().enumerate() {
        if row.len() != n_vars {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "All rows in x must have the same number of variables: row {} has {}, expected {}",
                i,
                row.len(),
                n_vars
            )));
        }
    }

    // Build design matrix X
    let mut design_matrix = Vec::new();
    for x_row in x.iter() {
        let mut row = Vec::new();
        if include_intercept {
            row.push(1.0); // Add intercept column
        }
        row.extend_from_slice(x_row);
        design_matrix.push(row);
    }

    let n_params = design_matrix[0].len();

    // Check if we have enough samples
    if n < n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Not enough samples: need at least {} samples for {} parameters",
            n_params, n_params
        )));
    }

    // Compute X'X
    let mut xtx = vec![vec![0.0; n_params]; n_params];
    for (i, xtx_row) in xtx.iter_mut().enumerate() {
        for j in 0..n_params {
            let mut sum = 0.0;
            for dm_row in design_matrix.iter() {
                sum += dm_row[i] * dm_row[j];
            }
            xtx_row[j] = sum;
        }
    }

    // Compute X'y
    let mut xty = vec![0.0; n_params];
    for (i, xty_val) in xty.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (dm_row, &y_val) in design_matrix.iter().zip(y.iter()) {
            sum += dm_row[i] * y_val;
        }
        *xty_val = sum;
    }

    // Compute (X'X)^-1 and use it for both coefficient estimation and HC3
    // This enables matrix reuse per REQ-014
    let xtx_inv = invert_matrix(&xtx)?;

    // Compute coefficients: β = (X'X)^-1 X'y
    let coefficients_full = matrix_vector_multiply(&xtx_inv, &xty);

    // Compute residuals explicitly: e = y - Xβ
    let residuals: Vec<f64> = design_matrix
        .iter()
        .zip(y.iter())
        .map(|(dm_row, &y_val)| {
            let y_pred: f64 = coefficients_full
                .iter()
                .zip(dm_row.iter())
                .map(|(&coef, &dm_val)| coef * dm_val)
                .sum();
            y_val - y_pred
        })
        .collect();

    // Calculate R-squared
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    let r_squared = if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - (ss_res / ss_tot)
    };

    // Compute HC3 standard errors
    // Handle perfect fit case: when all residuals are zero, SE should be zero
    let (intercept_se, standard_errors) = if residuals.iter().all(|&r| r == 0.0) {
        // Perfect fit case (R² = 1.0): all standard errors are zero
        if include_intercept {
            (Some(0.0), vec![0.0; n_vars])
        } else {
            (None, vec![0.0; n_vars])
        }
    } else {
        // Normal HC3 computation
        // Compute leverage values h_ii = x_i' (X'X)^-1 x_i
        let leverages = compute_leverages(&design_matrix, &xtx_inv)?;

        // Compute HC3 variance-covariance matrix
        let hc3_vcov = compute_hc3_vcov(&design_matrix, &residuals, &leverages, &xtx_inv);

        // Extract standard errors from diagonal of variance-covariance matrix
        if include_intercept {
            // intercept_se is SE for β_0, standard_errors is SE for β_1, β_2, ...
            let intercept_se_val = hc3_vcov[0][0].sqrt();
            let se_vec: Vec<f64> = (1..n_params).map(|i| hc3_vcov[i][i].sqrt()).collect();
            (Some(intercept_se_val), se_vec)
        } else {
            // No intercept: all standard errors for coefficients
            let se_vec: Vec<f64> = (0..n_params).map(|i| hc3_vcov[i][i].sqrt()).collect();
            (None, se_vec)
        }
    };

    // Extract intercept and coefficients
    let (intercept, coefficients) = if include_intercept {
        (Some(coefficients_full[0]), coefficients_full[1..].to_vec())
    } else {
        (None, coefficients_full)
    };

    // For backward compatibility with single covariate
    let slope = if coefficients.len() == 1 {
        Some(coefficients[0])
    } else {
        None
    };

    Ok(LinearRegressionResult {
        coefficients,
        intercept,
        r_squared,
        n_samples: n,
        slope,
        standard_errors,
        intercept_se,
        // Cluster-related fields default to None for non-clustered regression
        n_clusters: None,
        cluster_se_type: None,
        bootstrap_iterations_used: None,
    })
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting
///
/// Note: Kept for potential future use in the stats module.
#[allow(dead_code)]
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> PyResult<Vec<f64>> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Invalid matrix dimensions for linear system",
        ));
    }

    // Create augmented matrix [A|b]
    let mut aug = Vec::new();
    for i in 0..n {
        let mut row = a[i].clone();
        row.push(b[i]);
        aug.push(row);
    }

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = aug[i][i].abs();
        #[allow(clippy::needless_range_loop)]
        for k in (i + 1)..n {
            // Index k needed for row swapping and in-place modification
            let val = aug[k][i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Singular matrix: cannot solve linear regression (X'X is not invertible, check for collinearity)"
            ));
        }

        // Swap rows
        if max_row != i {
            aug.swap(i, max_row);
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[k][i] / aug[i][i];
            for j in i..=n {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        #[allow(clippy::needless_range_loop)]
        for j in (i + 1)..n {
            // Index j needed for random access to x
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

/// Invert a square matrix using Gauss-Jordan elimination with partial pivoting.
///
/// This function computes the inverse of a square matrix A by augmenting it with
/// the identity matrix [A|I] and performing row operations until the left side
/// becomes the identity, at which point the right side is A^-1.
///
/// # Arguments
/// * `a` - Square matrix to invert (n × n)
///
/// # Returns
/// * `PyResult<Vec<Vec<f64>>>` - Inverse matrix (n × n) or error if singular
///
/// # Errors
/// * `PyValueError` if matrix is singular (pivot element < 1e-10)
///
/// # Algorithm
/// 1. Create augmented matrix [A|I] of size (n × 2n)
/// 2. Forward elimination with partial pivoting:
///    - For each column i, find the row with maximum absolute value (pivot row)
///    - Swap current row with pivot row if needed
///    - Scale pivot row so diagonal element is 1
///    - Eliminate all other entries in column i
/// 3. After processing, left side is I, right side is A^-1
///
/// Note: Used internally by compute_linear_regression and tests.
#[allow(dead_code)]
fn invert_matrix(a: &[Vec<f64>]) -> PyResult<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot invert empty matrix",
        ));
    }

    // Verify square matrix
    for row in a.iter() {
        if row.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot invert non-square matrix",
            ));
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
            // Index row needed for row swapping and in-place modification
            let val = aug[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Check for singularity (pivot too small)
        if max_val < 1e-10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Singular matrix: cannot solve linear regression (X'X is not invertible, check for collinearity)"
            ));
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
///
/// Computes the matrix-vector product where A is (m × n) and v is (n,),
/// producing a result vector of length m.
///
/// # Arguments
/// * `a` - Matrix of shape (m × n)
/// * `v` - Vector of length n
///
/// # Returns
/// * `Vec<f64>` - Result vector of length m
///
/// Note: Used internally by compute_linear_regression and tests.
#[allow(dead_code)]
fn matrix_vector_multiply(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
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
///
/// Computes the matrix product where A is (m × k) and B is (k × n),
/// producing a result matrix of shape (m × n).
///
/// # Arguments
/// * `a` - Matrix of shape (m × k)
/// * `b` - Matrix of shape (k × n)
///
/// # Returns
/// * `Vec<Vec<f64>>` - Result matrix of shape (m × n)
///
/// Note: Used internally by compute_hc3_vcov and tests.
#[allow(dead_code)]
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Compute leverage values h_ii = x_i' (X'X)^-1 x_i for each observation.
///
/// The leverage of an observation measures its influence on the regression fit.
/// High leverage points have unusual predictor values and can unduly influence
/// the fitted model. The leverage values are the diagonal elements of the
/// hat matrix H = X(X'X)^-1X'.
///
/// This implementation uses the efficient formula h_ii = x_i' (X'X)^-1 x_i
/// which avoids forming the full hat matrix (O(n·k²) instead of O(n²)).
///
/// # Arguments
/// * `design_matrix` - Design matrix X of shape (n × p)
/// * `xtx_inv` - Inverse of X'X of shape (p × p)
///
/// # Returns
/// * `PyResult<Vec<f64>>` - Leverage values (n,) or error if extreme leverage detected
///
/// # Errors
/// * `PyValueError` if any h_ii >= 0.99 (extreme leverage makes HC3 unreliable)
///
/// Note: Used internally by compute_linear_regression and tests.
#[allow(dead_code)]
fn compute_leverages(design_matrix: &[Vec<f64>], xtx_inv: &[Vec<f64>]) -> PyResult<Vec<f64>> {
    let n = design_matrix.len();
    let mut leverages = Vec::with_capacity(n);

    for (i, x_i) in design_matrix.iter().enumerate() {
        // Compute temp = (X'X)^-1 × x_i (matrix-vector multiply)
        let temp = matrix_vector_multiply(xtx_inv, x_i);

        // Compute h_ii = x_i · temp (dot product)
        let mut h_ii = 0.0;
        for (j, &x_ij) in x_i.iter().enumerate() {
            h_ii += x_ij * temp[j];
        }

        // Check for extreme leverage (>= 0.99 makes (1 - h_ii)^2 too small)
        if h_ii >= 0.99 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "Observation {} has leverage ≥ 0.99; HC3 standard errors may be unreliable due to extreme leverage.",
                    i
                )
            ));
        }

        leverages.push(h_ii);
    }

    Ok(leverages)
}

/// Compute HC3 variance-covariance matrix using the sandwich formula.
///
/// The HC3 estimator is a heteroskedasticity-consistent covariance matrix estimator
/// that provides robust standard errors even when the error variance is not constant
/// across observations. It was proposed by MacKinnon & White (1985) and has good
/// finite-sample properties.
///
/// Formula: Var(β) = (X'X)^-1 × meat × (X'X)^-1
/// where meat = X' Ω X and Ω is diagonal with Ω_ii = e_i² / (1 - h_ii)²
///
/// The HC3 adjustment (1 - h_ii)² in the denominator gives more weight to observations
/// with high leverage, which provides better small-sample performance than HC0-HC2.
///
/// # Arguments
/// * `design_matrix` - Design matrix X of shape (n × p)
/// * `residuals` - OLS residuals e = y - Xβ of shape (n,)
/// * `leverages` - Leverage values h_ii of shape (n,)
/// * `xtx_inv` - Inverse of X'X of shape (p × p)
///
/// # Returns
/// * `Vec<Vec<f64>>` - HC3 variance-covariance matrix (p × p)
///
/// Note: Used internally by compute_linear_regression and tests.
#[allow(dead_code)]
fn compute_hc3_vcov(
    design_matrix: &[Vec<f64>],
    residuals: &[f64],
    leverages: &[f64],
    xtx_inv: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let _n = design_matrix.len();
    let p = xtx_inv.len();

    // Compute the "meat" of the sandwich: X' Ω X
    // where Ω_ii = e_i² / (1 - h_ii)²
    // This is computed efficiently in a single pass over observations
    let mut meat = vec![vec![0.0; p]; p];

    for (i, dm_row) in design_matrix.iter().enumerate() {
        // HC3 weight: e_i² / (1 - h_ii)²
        let one_minus_h = 1.0 - leverages[i];
        let omega_ii = residuals[i].powi(2) / one_minus_h.powi(2);

        // Accumulate x_i' × omega_ii × x_i into meat matrix
        // meat[j][k] += x_ij * omega_ii * x_ik
        for (j, meat_row) in meat.iter_mut().enumerate() {
            for k in 0..p {
                meat_row[k] += dm_row[j] * omega_ii * dm_row[k];
            }
        }
    }

    // Compute sandwich: (X'X)^-1 × meat × (X'X)^-1
    // First: temp = (X'X)^-1 × meat
    let temp = matrix_multiply(xtx_inv, &meat);
    // Then: result = temp × (X'X)^-1
    matrix_multiply(&temp, xtx_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_single_covariate_regression() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = compute_linear_regression(&x, &y, true).unwrap();

        assert_eq!(result.coefficients.len(), 1);
        assert_relative_eq!(result.coefficients[0], 2.0, epsilon = 1e-10);
        assert!(result.intercept.is_some());
        assert_relative_eq!(result.intercept.unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.r_squared, 1.0, epsilon = 1e-10);
        assert_eq!(result.n_samples, 5);
        assert!(result.slope.is_some());
        assert_relative_eq!(result.slope.unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multiple_covariate_regression() {
        // y = 2*x1 + 3*x2 + 1
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![3.0, 2.0],
            vec![4.0, 2.0],
            vec![5.0, 3.0],
        ];
        let y = vec![6.0, 8.0, 13.0, 15.0, 20.0];

        let result = compute_linear_regression(&x, &y, true).unwrap();

        assert_eq!(result.coefficients.len(), 2);
        assert_relative_eq!(result.coefficients[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.coefficients[1], 3.0, epsilon = 1e-10);
        assert!(result.intercept.is_some());
        assert_relative_eq!(result.intercept.unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.r_squared, 1.0, epsilon = 1e-10);
        assert_eq!(result.n_samples, 5);
    }

    #[test]
    fn test_regression_without_intercept() {
        // y = 2*x (no intercept)
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = compute_linear_regression(&x, &y, false).unwrap();

        assert_eq!(result.coefficients.len(), 1);
        assert_relative_eq!(result.coefficients[0], 2.0, epsilon = 1e-10);
        assert!(result.intercept.is_none());
        assert_relative_eq!(result.r_squared, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_data() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        let result = compute_linear_regression(&x, &y, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_lengths() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0];

        let result = compute_linear_regression(&x, &y, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_singular_matrix() {
        // x1 and x2 are perfectly correlated (collinear)
        let x = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let y = vec![1.0, 2.0, 3.0];

        let result = compute_linear_regression(&x, &y, true);
        assert!(result.is_err());
    }
}
