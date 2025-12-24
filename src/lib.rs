use pyo3::prelude::*;

mod cluster;
mod stats;

use cluster::{
    build_cluster_indices, compute_cluster_se_analytical, compute_cluster_se_bootstrap,
    ClusterError,
};
use stats::LinearRegressionResult;

/// Main module for causers - statistical operations for Polars DataFrames
#[pymodule]
fn _causers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LinearRegressionResult>()?;
    m.add_function(wrap_pyfunction!(linear_regression, m)?)?;
    Ok(())
}

/// Validate that a column name doesn't contain control characters (REQ-400)
fn validate_column_name(name: &str) -> PyResult<()> {
    if name.bytes().any(|b| b < 0x20 || b == 0x7F) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Column name '{}' contains invalid characters",
            name
        )));
    }
    Ok(())
}

/// Perform linear regression on Polars DataFrame columns
///
/// Args:
///     df: Polars DataFrame
///     x_cols: List of names of the independent variable columns
///     y_col: Name of the dependent variable column
///     include_intercept: Whether to include an intercept term (default: True)
///     cluster: Optional column name for cluster identifiers for cluster-robust SE
///     bootstrap: Whether to use wild cluster bootstrap (requires cluster)
///     bootstrap_iterations: Number of bootstrap iterations (default: 1000)
///     seed: Random seed for reproducibility (None for random)
///
/// Returns:
///     LinearRegressionResult with coefficients, intercept, r_squared, and standard errors
#[pyfunction]
#[pyo3(signature = (df, x_cols, y_col, include_intercept=true, cluster=None, bootstrap=false, bootstrap_iterations=1000, seed=None))]
fn linear_regression(
    py: Python,
    df: PyObject,
    x_cols: Vec<String>,
    y_col: &str,
    include_intercept: bool,
    cluster: Option<&str>,
    bootstrap: bool,
    bootstrap_iterations: usize,
    seed: Option<u64>,
) -> PyResult<LinearRegressionResult> {
    // Validate inputs
    if x_cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x_cols must contain at least one column name",
        ));
    }

    // Validate bootstrap requires cluster (REQ-202)
    if bootstrap && cluster.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap=True requires cluster to be specified",
        ));
    }

    // Validate bootstrap_iterations (REQ-203)
    if bootstrap_iterations < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_iterations must be at least 1",
        ));
    }

    // Validate column names for control characters (REQ-400)
    validate_column_name(y_col)?;
    for col in &x_cols {
        validate_column_name(col)?;
    }
    if let Some(cluster_col) = cluster {
        validate_column_name(cluster_col)?;
    }

    // Extract y column from Polars DataFrame
    let y_series = df.getattr(py, "get_column")?.call1(py, (y_col,))?;
    let y_array = y_series.call_method0(py, "to_numpy")?;
    let y_vec: Vec<f64> = y_array.extract(py)?;

    // Extract x columns from Polars DataFrame
    let mut x_matrix: Vec<Vec<f64>> = Vec::new();
    let n_rows = y_vec.len();

    // Initialize x_matrix with n_rows empty vectors
    for _ in 0..n_rows {
        x_matrix.push(Vec::new());
    }

    // Extract each x column and add to the matrix
    for col_name in &x_cols {
        let x_series = df
            .getattr(py, "get_column")?
            .call1(py, (col_name.as_str(),))?;
        let x_array = x_series.call_method0(py, "to_numpy")?;
        let x_vec: Vec<f64> = x_array.extract(py)?;

        if x_vec.len() != n_rows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "All columns must have the same length: {} has {}, expected {}",
                col_name,
                x_vec.len(),
                n_rows
            )));
        }

        // Add this column's values to each row
        for (i, val) in x_vec.iter().enumerate() {
            x_matrix[i].push(*val);
        }
    }

    // Extract cluster column if specified
    let cluster_ids: Option<Vec<i64>> = if let Some(cluster_col) = cluster {
        let cluster_series = df.getattr(py, "get_column")?.call1(py, (cluster_col,))?;

        // Check for nulls
        let null_count: usize = cluster_series.call_method0(py, "null_count")?.extract(py)?;
        if null_count > 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cluster column '{}' contains null values",
                cluster_col
            )));
        }

        // Try to extract as i64 directly, or convert via unique_id mapping
        // First, try direct i64 extraction
        let cluster_array = cluster_series.call_method0(py, "to_numpy")?;

        // Try extracting as i64
        let cluster_vec_result: Result<Vec<i64>, _> = cluster_array.extract(py);

        let cluster_vec = match cluster_vec_result {
            Ok(v) => v,
            Err(_) => {
                // Try f64 and cast
                let f64_result: Result<Vec<f64>, _> = cluster_array.extract(py);
                match f64_result {
                    Ok(v) => v.iter().map(|&x| x as i64).collect(),
                    Err(_) => {
                        // For string columns, use unique encoding
                        // Cast to string and get unique integer codes
                        let unique = cluster_series.call_method0(py, "unique")?;
                        let unique_list: Vec<String> =
                            unique.call_method0(py, "to_list")?.extract(py)?;

                        let string_list: Vec<String> =
                            cluster_series.call_method0(py, "to_list")?.extract(py)?;

                        // Create mapping from string to integer ID
                        let mut mapping = std::collections::HashMap::new();
                        for (i, s) in unique_list.iter().enumerate() {
                            mapping.insert(s.clone(), i as i64);
                        }

                        string_list
                            .iter()
                            .map(|s| *mapping.get(s).unwrap())
                            .collect()
                    }
                }
            }
        };

        if cluster_vec.len() != n_rows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cluster column must have same length as data: {} has {}, expected {}",
                cluster_col,
                cluster_vec.len(),
                n_rows
            )));
        }

        Some(cluster_vec)
    } else {
        None
    };

    // Compute regression with optional clustering
    compute_linear_regression_with_cluster(
        &x_matrix,
        &y_vec,
        include_intercept,
        cluster_ids.as_deref(),
        bootstrap,
        bootstrap_iterations,
        seed,
    )
}

/// Compute linear regression with optional clustered standard errors
fn compute_linear_regression_with_cluster(
    x: &[Vec<f64>],
    y: &[f64],
    include_intercept: bool,
    cluster_ids: Option<&[i64]>,
    bootstrap: bool,
    bootstrap_iterations: usize,
    seed: Option<u64>,
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
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Not enough samples: need at least {} samples for {} parameters",
            n_params, n_params
        )));
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

    // Compute (X'X)^-1
    let xtx_inv = invert_matrix(&xtx)?;

    // Compute coefficients: β = (X'X)^-1 X'y
    let coefficients_full = matrix_vector_multiply(&xtx_inv, &xty);

    // Compute fitted values: ŷ = Xβ
    let fitted_values: Vec<f64> = (0..n)
        .map(|i| {
            let mut y_pred = 0.0;
            for j in 0..n_params {
                y_pred += coefficients_full[j] * design_matrix[i][j];
            }
            y_pred
        })
        .collect();

    // Compute residuals: e = y - ŷ
    let residuals: Vec<f64> = (0..n).map(|i| y[i] - fitted_values[i]).collect();

    // Calculate R-squared
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    let r_squared = if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - (ss_res / ss_tot)
    };

    // Compute standard errors based on clustering option
    let (
        intercept_se,
        standard_errors,
        n_clusters_opt,
        cluster_se_type_opt,
        bootstrap_iterations_opt,
    ) = if let Some(cluster_ids) = cluster_ids {
        // Build cluster info
        let cluster_info = build_cluster_indices(cluster_ids).map_err(|e| {
                match e {
                    ClusterError::InsufficientClusters { found } => {
                        pyo3::exceptions::PyValueError::new_err(
                            format!("Clustered standard errors require at least 2 clusters; found {}", found)
                        )
                    }
                    ClusterError::SingleObservationCluster { cluster_idx } => {
                        pyo3::exceptions::PyValueError::new_err(
                            format!("Cluster {} contains only 1 observation; within-cluster correlation cannot be estimated. Use bootstrap=True for single-observation clusters.", cluster_idx)
                        )
                    }
                    ClusterError::NumericalInstability { message } => {
                        pyo3::exceptions::PyValueError::new_err(message)
                    }
                    ClusterError::InvalidStandardErrors => {
                        pyo3::exceptions::PyValueError::new_err(
                            "Standard error computation produced invalid values; check for numerical issues in data"
                        )
                    }
                }
            })?;

        let n_clusters = cluster_info.n_clusters;

        if bootstrap {
            // Wild cluster bootstrap
            let (coef_se, int_se) = compute_cluster_se_bootstrap(
                    &design_matrix,
                    &fitted_values,
                    &residuals,
                    &xtx_inv,
                    &cluster_info,
                    bootstrap_iterations,
                    seed,
                    include_intercept,
                ).map_err(|e| {
                    match e {
                        ClusterError::InvalidStandardErrors => {
                            pyo3::exceptions::PyValueError::new_err(
                                "Standard error computation produced invalid values; check for numerical issues in data"
                            )
                        }
                        _ => pyo3::exceptions::PyValueError::new_err(e.to_string())
                    }
                })?;

            (
                int_se,
                coef_se,
                Some(n_clusters),
                Some("bootstrap".to_string()),
                Some(bootstrap_iterations),
            )
        } else {
            // Analytical clustered SE
            let (coef_se, int_se) = compute_cluster_se_analytical(
                    &design_matrix,
                    &residuals,
                    &xtx_inv,
                    &cluster_info,
                    include_intercept,
                ).map_err(|e| {
                    match e {
                        ClusterError::SingleObservationCluster { cluster_idx } => {
                            pyo3::exceptions::PyValueError::new_err(
                                format!("Cluster {} contains only 1 observation; within-cluster correlation cannot be estimated. Use bootstrap=True for single-observation clusters.", cluster_idx)
                            )
                        }
                        ClusterError::NumericalInstability { message } => {
                            pyo3::exceptions::PyValueError::new_err(message)
                        }
                        ClusterError::InvalidStandardErrors => {
                            pyo3::exceptions::PyValueError::new_err(
                                "Standard error computation produced invalid values; check for numerical issues in data"
                            )
                        }
                        _ => pyo3::exceptions::PyValueError::new_err(e.to_string())
                    }
                })?;

            (
                int_se,
                coef_se,
                Some(n_clusters),
                Some("analytical".to_string()),
                None,
            )
        }
    } else {
        // Non-clustered: use HC3
        if residuals.iter().all(|&r| r == 0.0) {
            // Perfect fit case
            if include_intercept {
                (Some(0.0), vec![0.0; n_vars], None, None, None)
            } else {
                (None, vec![0.0; n_vars], None, None, None)
            }
        } else {
            // Compute HC3 leverages
            let leverages = compute_leverages(&design_matrix, &xtx_inv)?;
            let hc3_vcov = compute_hc3_vcov(&design_matrix, &residuals, &leverages, &xtx_inv);

            if include_intercept {
                let intercept_se_val = hc3_vcov[0][0].sqrt();
                let se_vec: Vec<f64> = (1..n_params).map(|i| hc3_vcov[i][i].sqrt()).collect();
                (Some(intercept_se_val), se_vec, None, None, None)
            } else {
                let se_vec: Vec<f64> = (0..n_params).map(|i| hc3_vcov[i][i].sqrt()).collect();
                (None, se_vec, None, None, None)
            }
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
        n_clusters: n_clusters_opt,
        cluster_se_type: cluster_se_type_opt,
        bootstrap_iterations_used: bootstrap_iterations_opt,
    })
}

/// Invert a square matrix using Gauss-Jordan elimination with partial pivoting
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

    // Create augmented matrix [A|I]
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(2 * n);
        row.extend_from_slice(&a[i]);
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
        for row in (col + 1)..n {
            let val = aug[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Singular matrix: cannot solve linear regression (X'X is not invertible, check for collinearity)"
            ));
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
    for i in 0..n {
        inverse.push(aug[i][n..(2 * n)].to_vec());
    }

    Ok(inverse)
}

/// Multiply a matrix by a vector
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

/// Multiply two matrices
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

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i][l] * b[l][j];
            }
            result[i][j] = sum;
        }
    }

    result
}

/// Compute leverage values h_ii = x_i' (X'X)^-1 x_i
fn compute_leverages(design_matrix: &[Vec<f64>], xtx_inv: &[Vec<f64>]) -> PyResult<Vec<f64>> {
    let n = design_matrix.len();
    let mut leverages = Vec::with_capacity(n);

    for (i, x_i) in design_matrix.iter().enumerate() {
        let temp = matrix_vector_multiply(xtx_inv, x_i);

        let mut h_ii = 0.0;
        for (j, &x_ij) in x_i.iter().enumerate() {
            h_ii += x_ij * temp[j];
        }

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

/// Compute HC3 variance-covariance matrix using the sandwich formula
fn compute_hc3_vcov(
    design_matrix: &[Vec<f64>],
    residuals: &[f64],
    leverages: &[f64],
    xtx_inv: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let n = design_matrix.len();
    let p = xtx_inv.len();

    let mut meat = vec![vec![0.0; p]; p];

    for i in 0..n {
        let one_minus_h = 1.0 - leverages[i];
        let omega_ii = residuals[i].powi(2) / one_minus_h.powi(2);

        for j in 0..p {
            for k in 0..p {
                meat[j][k] += design_matrix[i][j] * omega_ii * design_matrix[i][k];
            }
        }
    }

    let temp = matrix_multiply(xtx_inv, &meat);
    matrix_multiply(&temp, xtx_inv)
}
