use pyo3::prelude::*;

mod cluster;
mod logistic;
mod sdid;
mod stats;
mod synth_control;

use cluster::{
    build_cluster_indices, compute_cluster_se_analytical, compute_cluster_se_bootstrap,
    BootstrapWeightType, ClusterError, ClusterInfo,
};
use logistic::{
    compute_hc3_logistic, compute_logistic_mle, compute_null_log_likelihood,
    compute_pseudo_r_squared, LogisticError, LogisticRegressionResult,
};
use sdid::{synthetic_did_impl, SyntheticDIDResult};
use stats::LinearRegressionResult;
use synth_control::{
    estimate as synth_control_estimate, SCPanelData, SynthControlConfig, SynthControlError,
    SynthControlMethod, SyntheticControlResult,
};

/// Main module for causers - statistical operations for Polars DataFrames
#[pymodule]
fn _causers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LinearRegressionResult>()?;
    m.add_class::<LogisticRegressionResult>()?;
    m.add_class::<SyntheticDIDResult>()?;
    m.add_class::<SyntheticControlResult>()?;
    m.add_function(wrap_pyfunction!(linear_regression, m)?)?;
    m.add_function(wrap_pyfunction!(logistic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(synthetic_did_impl, m)?)?;
    m.add_function(wrap_pyfunction!(synthetic_control_impl, m)?)?;
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

/// Parse bootstrap_method string to BootstrapWeightType enum.
///
/// Accepts case-insensitive "rademacher" or "webb".
fn parse_bootstrap_method(method: &str) -> PyResult<BootstrapWeightType> {
    match method.to_lowercase().as_str() {
        "rademacher" => Ok(BootstrapWeightType::Rademacher),
        "webb" => Ok(BootstrapWeightType::Webb),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "bootstrap_method must be 'rademacher' or 'webb', got: '{}'",
            method
        ))),
    }
}

/// Get the cluster_se_type string based on the weight type.
fn get_cluster_se_type(weight_type: BootstrapWeightType) -> String {
    match weight_type {
        BootstrapWeightType::Rademacher => "bootstrap_rademacher".to_string(),
        BootstrapWeightType::Webb => "bootstrap_webb".to_string(),
    }
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
///     bootstrap_method: Weight distribution for bootstrap ("rademacher" or "webb", default: "rademacher")
///
/// Returns:
///     LinearRegressionResult with coefficients, intercept, r_squared, and standard errors
#[pyfunction]
#[pyo3(signature = (df, x_cols, y_col, include_intercept=true, cluster=None, bootstrap=false, bootstrap_iterations=1000, seed=None, bootstrap_method="rademacher"))]
// Statistical functions commonly require many parameters for configuration.
// Refactoring into a config struct would reduce API clarity for Python users.
#[allow(clippy::too_many_arguments)]
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
    bootstrap_method: &str,
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

    // Parse and validate bootstrap_method
    let weight_type = parse_bootstrap_method(bootstrap_method)?;

    // Validate bootstrap_method requires bootstrap=True (REQ-006)
    if weight_type != BootstrapWeightType::Rademacher && !bootstrap {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_method requires bootstrap=True",
        ));
    }

    // Validate bootstrap_method requires cluster (REQ-007)
    if weight_type != BootstrapWeightType::Rademacher && cluster.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_method requires cluster to be specified",
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
        weight_type,
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
    weight_type: BootstrapWeightType,
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

    // Compute (X'X)^-1
    let xtx_inv = invert_matrix(&xtx)?;

    // Compute coefficients: β = (X'X)^-1 X'y
    let coefficients_full = matrix_vector_multiply(&xtx_inv, &xty);

    // Compute fitted values: ŷ = Xβ
    let fitted_values: Vec<f64> = design_matrix
        .iter()
        .map(|dm_row| {
            coefficients_full
                .iter()
                .zip(dm_row.iter())
                .map(|(&coef, &dm_val)| coef * dm_val)
                .sum()
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
                    weight_type,
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
                Some(get_cluster_se_type(weight_type)),
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
    for aug_row in aug.iter() {
        inverse.push(aug_row[n..(2 * n)].to_vec());
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
    let _n = design_matrix.len();
    let p = xtx_inv.len();

    let mut meat = vec![vec![0.0; p]; p];

    for (i, dm_row) in design_matrix.iter().enumerate() {
        let one_minus_h = 1.0 - leverages[i];
        let omega_ii = residuals[i].powi(2) / one_minus_h.powi(2);

        for (j, meat_row) in meat.iter_mut().enumerate() {
            for k in 0..p {
                meat_row[k] += dm_row[j] * omega_ii * dm_row[k];
            }
        }
    }

    let temp = matrix_multiply(xtx_inv, &meat);
    matrix_multiply(&temp, xtx_inv)
}

// ============================================================================
// Logistic Regression
// ============================================================================

/// Perform logistic regression on Polars DataFrame columns with binary outcome
///
/// Uses Maximum Likelihood Estimation with Newton-Raphson optimization.
/// Computes HC3 robust standard errors (or clustered SE if cluster specified).
///
/// Args:
///     df: Polars DataFrame
///     x_cols: List of names of the independent variable columns
///     y_col: Name of the binary outcome column (must contain only 0 and 1)
///     include_intercept: Whether to include an intercept term (default: True)
///     cluster: Optional column name for cluster identifiers for cluster-robust SE
///     bootstrap: Whether to use score bootstrap for SE (requires cluster)
///     bootstrap_iterations: Number of bootstrap iterations (default: 1000)
///     seed: Random seed for reproducibility (None for random)
///     bootstrap_method: Weight distribution for bootstrap ("rademacher" or "webb", default: "rademacher")
///
/// Returns:
///     LogisticRegressionResult with coefficients, standard errors, and diagnostics
#[pyfunction]
#[pyo3(signature = (df, x_cols, y_col, include_intercept=true, cluster=None, bootstrap=false, bootstrap_iterations=1000, seed=None, bootstrap_method="rademacher"))]
// Statistical functions commonly require many parameters for configuration.
// Refactoring into a config struct would reduce API clarity for Python users.
#[allow(clippy::too_many_arguments)]
fn logistic_regression(
    py: Python,
    df: PyObject,
    x_cols: Vec<String>,
    y_col: &str,
    include_intercept: bool,
    cluster: Option<&str>,
    bootstrap: bool,
    bootstrap_iterations: usize,
    seed: Option<u64>,
    bootstrap_method: &str,
) -> PyResult<LogisticRegressionResult> {
    // Validate inputs
    if x_cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x_cols must contain at least one column name",
        ));
    }

    // Validate bootstrap requires cluster (REQ-105)
    if bootstrap && cluster.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap=True requires cluster to be specified",
        ));
    }

    // Validate bootstrap_iterations (REQ-106)
    if bootstrap_iterations < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_iterations must be at least 1",
        ));
    }

    // Parse and validate bootstrap_method
    let weight_type = parse_bootstrap_method(bootstrap_method)?;

    // Validate bootstrap_method requires bootstrap=True (REQ-006)
    if weight_type != BootstrapWeightType::Rademacher && !bootstrap {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_method requires bootstrap=True",
        ));
    }

    // Validate bootstrap_method requires cluster (REQ-007)
    if weight_type != BootstrapWeightType::Rademacher && cluster.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_method requires cluster to be specified",
        ));
    }

    // Validate column names for control characters (REQ-300)
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

    let n_rows = y_vec.len();

    // Validate empty DataFrame (REQ-102)
    if n_rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot perform regression on empty data",
        ));
    }

    // Validate y contains only 0 and 1 (REQ-100)
    for &yi in &y_vec {
        if yi != 0.0 && yi != 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "y_col must contain only 0 and 1 values",
            ));
        }
    }

    // Validate y contains both 0 and 1 (REQ-101)
    let has_zero = y_vec.iter().any(|&y| y == 0.0);
    let has_one = y_vec.iter().any(|&y| y == 1.0);
    if !has_zero || !has_one {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y_col must contain both 0 and 1 values",
        ));
    }

    // Extract x columns from Polars DataFrame
    let mut x_matrix: Vec<Vec<f64>> = Vec::new();

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

        // Check for nulls (REQ-107)
        let null_count: usize = cluster_series.call_method0(py, "null_count")?.extract(py)?;
        if null_count > 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cluster column '{}' contains null values",
                cluster_col
            )));
        }

        // Try to extract as i64 directly, or convert via unique_id mapping
        let cluster_array = cluster_series.call_method0(py, "to_numpy")?;
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

    // Compute logistic regression with optional clustering
    compute_logistic_regression_with_cluster(
        &x_matrix,
        &y_vec,
        include_intercept,
        cluster_ids.as_deref(),
        bootstrap,
        bootstrap_iterations,
        seed,
        weight_type,
    )
}

/// Compute logistic regression with optional clustered standard errors
fn compute_logistic_regression_with_cluster(
    x: &[Vec<f64>],
    y: &[f64],
    include_intercept: bool,
    cluster_ids: Option<&[i64]>,
    bootstrap: bool,
    bootstrap_iterations: usize,
    seed: Option<u64>,
    weight_type: BootstrapWeightType,
) -> PyResult<LogisticRegressionResult> {
    let n = x.len();

    // Build design matrix X
    let mut design_matrix: Vec<Vec<f64>> = Vec::new();
    for x_row in x.iter() {
        let mut row = Vec::new();
        if include_intercept {
            row.push(1.0); // Add intercept column
        }
        row.extend_from_slice(x_row);
        design_matrix.push(row);
    }

    // Run MLE optimization
    let mle_result = compute_logistic_mle(&design_matrix, y).map_err(|e| match e {
        LogisticError::PerfectSeparation => pyo3::exceptions::PyValueError::new_err(
            "Perfect separation detected; logistic regression cannot converge",
        ),
        LogisticError::ConvergenceFailure { iterations } => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Convergence failed after {} iterations",
                iterations
            ))
        }
        LogisticError::SingularHessian => pyo3::exceptions::PyValueError::new_err(
            "Hessian matrix is singular; check for collinearity",
        ),
        LogisticError::NumericalInstability { message } => {
            pyo3::exceptions::PyValueError::new_err(message)
        }
    })?;

    // Compute null log-likelihood and pseudo R²
    let ll_null = compute_null_log_likelihood(y);
    let pseudo_r_squared = compute_pseudo_r_squared(mle_result.log_likelihood, ll_null);

    // Compute standard errors based on clustering option
    let (
        intercept_se,
        standard_errors,
        n_clusters_opt,
        cluster_se_type_opt,
        bootstrap_iterations_opt,
    ) = if let Some(cluster_ids) = cluster_ids {
        // Build cluster info
        let cluster_info = build_cluster_indices(cluster_ids).map_err(|e| match e {
            ClusterError::InsufficientClusters { found } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Clustered standard errors require at least 2 clusters; found {}",
                    found
                ))
            }
            ClusterError::SingleObservationCluster { cluster_idx } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Cluster {} contains only 1 observation; within-cluster correlation cannot be estimated. Use bootstrap=True for single-observation clusters.",
                    cluster_idx
                ))
            }
            ClusterError::NumericalInstability { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            ClusterError::InvalidStandardErrors => pyo3::exceptions::PyValueError::new_err(
                "Standard error computation produced invalid values; check for numerical issues in data",
            ),
        })?;

        let n_clusters = cluster_info.n_clusters;

        if bootstrap {
            // Score bootstrap for logistic regression
            let (coef_se, int_se) = compute_score_bootstrap_logistic(
                &design_matrix,
                y,
                &mle_result.beta,
                &mle_result.info_inv,
                &cluster_info,
                bootstrap_iterations,
                seed,
                include_intercept,
                weight_type,
            )
            .map_err(|e| match e {
                ClusterError::InvalidStandardErrors => pyo3::exceptions::PyValueError::new_err(
                    "Standard error computation produced invalid values; check for numerical issues in data",
                ),
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
            })?;

            (
                int_se,
                coef_se,
                Some(n_clusters),
                Some(get_cluster_se_type(weight_type)),
                Some(bootstrap_iterations),
            )
        } else {
            // Analytical clustered SE for logistic regression
            let (coef_se, int_se) = compute_cluster_se_logistic(
                &design_matrix,
                y,
                &mle_result.beta,
                &mle_result.info_inv,
                &cluster_info,
                include_intercept,
            )
            .map_err(|e| match e {
                ClusterError::SingleObservationCluster { cluster_idx } => {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Cluster {} contains only 1 observation; within-cluster correlation cannot be estimated. Use bootstrap=True for single-observation clusters.",
                        cluster_idx
                    ))
                }
                ClusterError::NumericalInstability { message } => {
                    pyo3::exceptions::PyValueError::new_err(message)
                }
                ClusterError::InvalidStandardErrors => pyo3::exceptions::PyValueError::new_err(
                    "Standard error computation produced invalid values; check for numerical issues in data",
                ),
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
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
        let se = compute_hc3_logistic(&design_matrix, y, &mle_result.pi, &mle_result.info_inv)
            .map_err(|e| match e {
                LogisticError::NumericalInstability { message } => {
                    pyo3::exceptions::PyValueError::new_err(message)
                }
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
            })?;

        if include_intercept {
            (Some(se[0]), se[1..].to_vec(), None, None, None)
        } else {
            (None, se, None, None, None)
        }
    };

    // Extract intercept and coefficients
    let (intercept, coefficients) = if include_intercept {
        (Some(mle_result.beta[0]), mle_result.beta[1..].to_vec())
    } else {
        (None, mle_result.beta)
    };

    Ok(LogisticRegressionResult {
        coefficients,
        intercept,
        standard_errors,
        intercept_se,
        n_samples: n,
        n_clusters: n_clusters_opt,
        cluster_se_type: cluster_se_type_opt,
        bootstrap_iterations_used: bootstrap_iterations_opt,
        converged: mle_result.converged,
        iterations: mle_result.iterations,
        log_likelihood: mle_result.log_likelihood,
        pseudo_r_squared,
    })
}

// ============================================================================
// Clustered SE for Logistic Regression (Score-based)
// ============================================================================

/// Compute analytical clustered standard errors for logistic regression.
///
/// Uses the sandwich estimator with cluster-level scores.
fn compute_cluster_se_logistic(
    design_matrix: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
    info_inv: &[Vec<f64>],
    cluster_info: &ClusterInfo,
    include_intercept: bool,
) -> Result<(Vec<f64>, Option<f64>), ClusterError> {
    let n = y.len();
    let p = info_inv.len();
    let g = cluster_info.n_clusters;

    // Check for single-observation clusters in analytical mode
    for (cluster_idx, size) in cluster_info.sizes.iter().enumerate() {
        if *size == 1 {
            return Err(ClusterError::SingleObservationCluster { cluster_idx });
        }
    }

    // Compute predicted probabilities
    let pi: Vec<f64> = design_matrix
        .iter()
        .map(|xi| logistic::sigmoid(logistic::dot(xi, beta)))
        .collect();

    // Compute meat matrix: Σ_g S_g S_g' where S_g = Σᵢ∈g xᵢ(yᵢ - πᵢ)
    let mut meat = vec![vec![0.0; p]; p];

    for cluster_indices in &cluster_info.indices {
        // Compute score for cluster g
        let mut score_g = vec![0.0; p];
        for &i in cluster_indices {
            let resid = y[i] - pi[i];
            for (j, score_val) in score_g.iter_mut().enumerate() {
                *score_val += design_matrix[i][j] * resid;
            }
        }

        // Accumulate outer product: score_g × score_g'
        for j in 0..p {
            for k in 0..p {
                meat[j][k] += score_g[j] * score_g[k];
            }
        }
    }

    // Small-sample adjustment: G/(G-1) × (n-1)/(n-k)
    let adjustment = (g as f64 / (g - 1) as f64) * ((n - 1) as f64 / (n - p) as f64);
    for row in &mut meat {
        for val in row.iter_mut() {
            *val *= adjustment;
        }
    }

    // Sandwich: V = I⁻¹ × meat × I⁻¹
    let temp = logistic::matrix_multiply(info_inv, &meat);
    let v = logistic::matrix_multiply(&temp, info_inv);

    // Check condition number for numerical stability
    let diag: Vec<f64> = (0..p).map(|i| v[i][i]).collect();
    let max_diag = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_diag = diag.iter().cloned().fold(f64::INFINITY, f64::min);

    if min_diag <= 0.0 || max_diag / min_diag > 1e10 {
        return Err(ClusterError::NumericalInstability {
            message: "Cluster covariance matrix is nearly singular (condition number > 1e10); standard errors may be unreliable".to_string()
        });
    }

    // Extract standard errors from diagonal
    let se: Vec<f64> = diag.iter().map(|&d| d.sqrt()).collect();

    // Check for NaN/Inf values
    if se.iter().any(|&s| s.is_nan() || s.is_infinite()) {
        return Err(ClusterError::InvalidStandardErrors);
    }

    // Split into intercept_se and coefficient_se
    if include_intercept {
        let intercept_se = Some(se[0]);
        let coefficient_se = se[1..].to_vec();
        Ok((coefficient_se, intercept_se))
    } else {
        Ok((se, None))
    }
}

/// Compute score bootstrap standard errors for logistic regression.
///
/// Implements the Kline & Santos (2012) score bootstrap with configurable weight distribution.
// Score bootstrap requires all statistical context parameters. Struct would reduce clarity.
#[allow(clippy::too_many_arguments)]
fn compute_score_bootstrap_logistic(
    design_matrix: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
    info_inv: &[Vec<f64>],
    cluster_info: &ClusterInfo,
    bootstrap_iterations: usize,
    seed: Option<u64>,
    include_intercept: bool,
    weight_type: BootstrapWeightType,
) -> Result<(Vec<f64>, Option<f64>), ClusterError> {
    let p = info_inv.len();
    let g = cluster_info.n_clusters;

    // Compute predicted probabilities
    let pi: Vec<f64> = design_matrix
        .iter()
        .map(|xi| logistic::sigmoid(logistic::dot(xi, beta)))
        .collect();

    // Compute cluster-level scores: S_g = Σᵢ∈g xᵢ(yᵢ - πᵢ)
    let cluster_scores: Vec<Vec<f64>> = cluster_info
        .indices
        .iter()
        .map(|idx| {
            let mut score = vec![0.0; p];
            for &i in idx {
                let resid = y[i] - pi[i];
                for (j, score_val) in score.iter_mut().enumerate() {
                    *score_val += design_matrix[i][j] * resid;
                }
            }
            score
        })
        .collect();

    // Initialize RNG
    let actual_seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    });
    let mut rng = cluster::SplitMix64::new(actual_seed);

    // Initialize Welford's online algorithm state
    let mut welford = cluster::WelfordState::new(p);

    // Pre-allocate buffers
    let mut weights = vec![0.0; g];
    let mut perturbed_score = vec![0.0; p];

    for _ in 0..bootstrap_iterations {
        // Generate weights for each cluster using specified distribution
        for w in weights.iter_mut() {
            *w = rng.weight(weight_type);
        }

        // Compute perturbed score: S* = Σ_g w_g S_g
        for val in perturbed_score.iter_mut() {
            *val = 0.0;
        }
        for (c, score) in cluster_scores.iter().enumerate() {
            for j in 0..p {
                perturbed_score[j] += weights[c] * score[j];
            }
        }

        // Coefficient perturbation: δ* = I⁻¹ S*
        let delta = logistic::matrix_vector_multiply(info_inv, &perturbed_score);

        // Update Welford state
        welford.update(&delta);
    }

    // Compute standard errors from Welford state
    let se = welford.standard_errors();

    // Check for NaN/Inf values
    if se.iter().any(|&s| s.is_nan() || s.is_infinite()) {
        return Err(ClusterError::InvalidStandardErrors);
    }

    // Split into intercept_se and coefficient_se
    if include_intercept {
        let intercept_se = Some(se[0]);
        let coefficient_se = se[1..].to_vec();
        Ok((coefficient_se, intercept_se))
    } else {
        Ok((se, None))
    }
}

// ============================================================================
// Cluster Balance Check
// ============================================================================

/// Check cluster balance and return warning message if any cluster has >50% observations.
///
/// Returns Some(warning_message) if imbalanced, None otherwise.
pub fn check_cluster_balance(cluster_info: &ClusterInfo) -> Option<String> {
    let total: usize = cluster_info.sizes.iter().sum();
    let threshold = total / 2; // 50%

    for (i, &size) in cluster_info.sizes.iter().enumerate() {
        if size > threshold {
            return Some(format!(
                "Cluster {} contains {}% of observations ({}/{}). \
                 Clustered standard errors may be unreliable with such imbalanced clusters.",
                i,
                (size * 100) / total,
                size,
                total
            ));
        }
    }
    None
}

// ============================================================================
// Synthetic Control Implementation (TASK-012)
// ============================================================================

/// Convert SynthControlError to PyErr
impl From<SynthControlError> for PyErr {
    fn from(err: SynthControlError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

/// Synthetic Control implementation exposed to Python.
///
/// This function is called from the Python wrapper after input validation
/// and panel structure detection.
///
/// # Arguments
///
/// * `outcomes` - Flat outcome matrix in row-major order (n_units × n_periods)
/// * `n_units` - Number of units in the panel
/// * `n_periods` - Number of time periods
/// * `control_indices` - Indices of control units
/// * `treated_index` - Index of the single treated unit
/// * `pre_period_indices` - Indices of pre-treatment periods
/// * `post_period_indices` - Indices of post-treatment periods
/// * `method` - SC method: "traditional", "penalized", "robust", "augmented"
/// * `lambda_param` - Regularization parameter for penalized method (None for auto)
/// * `compute_se` - Whether to compute standard errors via in-space placebo
/// * `n_placebo` - Number of placebo iterations for SE (None = use all controls)
/// * `max_iter` - Maximum Frank-Wolfe iterations
/// * `tol` - Convergence tolerance
/// * `seed` - Random seed for reproducibility (None for random)
///
/// # Returns
///
/// SyntheticControlResult with ATT, SE, weights, and diagnostics
#[pyfunction]
#[pyo3(signature = (
    outcomes,
    n_units,
    n_periods,
    control_indices,
    treated_index,
    pre_period_indices,
    post_period_indices,
    method,
    lambda_param,
    compute_se,
    n_placebo,
    max_iter,
    tol,
    seed
))]
#[allow(clippy::too_many_arguments)]
fn synthetic_control_impl(
    outcomes: Vec<f64>,
    n_units: usize,
    n_periods: usize,
    control_indices: Vec<usize>,
    treated_index: usize,
    pre_period_indices: Vec<usize>,
    post_period_indices: Vec<usize>,
    method: &str,
    lambda_param: Option<f64>,
    compute_se: bool,
    n_placebo: Option<usize>,
    max_iter: usize,
    tol: f64,
    seed: Option<u64>,
) -> PyResult<SyntheticControlResult> {
    // Parse method string to enum
    let sc_method = SynthControlMethod::from_str(method)?;

    // Build panel data structure
    let panel = SCPanelData::new(
        outcomes,
        n_units,
        n_periods,
        control_indices,
        treated_index,
        pre_period_indices,
        post_period_indices,
    )?;

    // Build configuration
    let config = SynthControlConfig {
        method: sc_method,
        lambda: lambda_param,
        compute_se,
        n_placebo: n_placebo.unwrap_or(panel.n_control()),
        max_iter,
        tol,
        seed,
    };

    // Run estimation
    let result = synth_control_estimate(&panel, &config)?;

    Ok(result)
}
