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
}

#[pymethods]
impl LinearRegressionResult {
    fn __repr__(&self) -> String {
        let intercept_str = match self.intercept {
            Some(i) => format!("{:.6}", i),
            None => "None".to_string(),
        };
        format!(
            "LinearRegressionResult(coefficients={:?}, intercept={}, r_squared={:.6}, n_samples={})",
            self.coefficients, intercept_str, self.r_squared, self.n_samples
        )
    }
    
    fn __str__(&self) -> String {
        let intercept_str = match self.intercept {
            Some(i) => format!(" + {:.6}", i),
            None => "".to_string(),
        };
        
        if self.coefficients.len() == 1 {
            format!(
                "y = {:.6}x{}(R² = {:.6}, n = {})",
                self.coefficients[0], intercept_str, self.r_squared, self.n_samples
            )
        } else {
            let terms: Vec<String> = self.coefficients.iter()
                .enumerate()
                .map(|(i, &c)| format!("{:.6}*x{}", c, i + 1))
                .collect();
            format!(
                "y = {}{}(R² = {:.6}, n = {})",
                terms.join(" + "), intercept_str, self.r_squared, self.n_samples
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
pub fn compute_linear_regression(
    x: &[Vec<f64>],
    y: &[f64],
    include_intercept: bool,
) -> PyResult<LinearRegressionResult> {
    if x.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot perform regression on empty data"
        ));
    }
    
    let n = x.len();
    if n != y.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("x and y must have the same number of rows: x has {}, y has {}", n, y.len())
        ));
    }
    
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot perform regression on empty data"
        ));
    }
    
    let n_vars = x[0].len();
    if n_vars == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must have at least one variable"
        ));
    }
    
    // Check all rows have same number of variables
    for (i, row) in x.iter().enumerate() {
        if row.len() != n_vars {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("All rows in x must have the same number of variables: row {} has {}, expected {}", i, row.len(), n_vars)
            ));
        }
    }
    
    // Build design matrix X
    let mut design_matrix = Vec::new();
    for i in 0..n {
        let mut row = Vec::new();
        if include_intercept {
            row.push(1.0); // Add intercept column
        }
        row.extend_from_slice(&x[i]);
        design_matrix.push(row);
    }
    
    let n_params = design_matrix[0].len();
    
    // Check if we have enough samples
    if n < n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Not enough samples: need at least {} samples for {} parameters", n_params, n_params)
        ));
    }
    
    // Compute X'X
    let mut xtx = vec![vec![0.0; n_params]; n_params];
    for i in 0..n_params {
        for j in 0..n_params {
            let mut sum = 0.0;
            for k in 0..n {
                sum += design_matrix[k][i] * design_matrix[k][j];
            }
            xtx[i][j] = sum;
        }
    }
    
    // Compute X'y
    let mut xty = vec![0.0; n_params];
    for i in 0..n_params {
        let mut sum = 0.0;
        for k in 0..n {
            sum += design_matrix[k][i] * y[k];
        }
        xty[i] = sum;
    }
    
    // Solve (X'X)β = X'y using Gaussian elimination
    let coefficients_full = solve_linear_system(&xtx, &xty)?;
    
    // Calculate R-squared using full coefficients before extracting
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    
    for i in 0..n {
        let mut y_pred = 0.0;
        for j in 0..n_params {
            y_pred += coefficients_full[j] * design_matrix[i][j];
        }
        ss_res += (y[i] - y_pred).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }
    
    // Extract intercept and coefficients
    let (intercept, coefficients) = if include_intercept {
        (Some(coefficients_full[0]), coefficients_full[1..].to_vec())
    } else {
        (None, coefficients_full)
    };
    
    let r_squared = if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - (ss_res / ss_tot)
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
    })
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> PyResult<Vec<f64>> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Invalid matrix dimensions for linear system"
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
        for k in (i + 1)..n {
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
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    
    Ok(x)
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
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
        ];
        let y = vec![1.0, 2.0, 3.0];
        
        let result = compute_linear_regression(&x, &y, true);
        assert!(result.is_err());
    }
}