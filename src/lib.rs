use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use polars::prelude::*;

mod cluster;
mod linalg;
mod logistic;
mod sdid;
mod stats;
mod synth_control;

use cluster::{
    build_cluster_indices, compute_cluster_se_analytical_faer, compute_cluster_se_bootstrap_faer,
    BootstrapWeightType, ClusterError, ClusterInfo,
};
use logistic::{
    compute_hc3_logistic_faer, compute_logistic_mle, compute_null_log_likelihood,
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

// ============================================================================
// Arrow-based Data Extraction Helpers (Phase 3)
// ============================================================================

/// Extract a single f64 column from a Polars DataFrame
fn extract_f64_column(df: &PyDataFrame, col_name: &str) -> PyResult<Vec<f64>> {
    let series = df.as_ref().column(col_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let ca = series.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let vec: Vec<f64> = ca.into_iter()
        .map(|opt| opt.unwrap_or(f64::NAN))
        .collect();
    Ok(vec)
}

/// Extract multiple columns as flat row-major Vec<f64>
fn extract_f64_columns_flat(df: &PyDataFrame, col_names: &[String]) -> PyResult<(Vec<f64>, usize, usize)> {
    let n_rows = df.as_ref().height();
    let n_cols = col_names.len();
    let mut flat = Vec::with_capacity(n_rows * n_cols);
    
    // Pre-extract all columns ONCE (not n_rows times!)
    let columns: Vec<_> = col_names.iter()
        .map(|name| {
            let series = df.as_ref().column(name)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            series.f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    
    // Interleave into row-major order
    for i in 0..n_rows {
        for col_ca in &columns {
            flat.push(col_ca.get(i).unwrap_or(f64::NAN));
        }
    }
    
    Ok((flat, n_rows, n_cols))
}

/// Extract cluster column as Vec<i64> from a Polars DataFrame
/// Handles integer, float, string, and categorical columns with automatic encoding
fn extract_cluster_column(df: &PyDataFrame, col_name: &str, expected_len: usize) -> PyResult<Vec<i64>> {
    let series = df.as_ref().column(col_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    // Check for nulls
    if series.null_count() > 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cluster column '{}' contains null values",
            col_name
        )));
    }
    
    // Check if it's a categorical type and handle by casting to string
    let is_categorical = matches!(series.dtype(), DataType::Categorical(_, _));
    
    // Try to extract as i64, handling various types including categorical
    let cluster_vec: Vec<i64> = if let Ok(ca) = series.i64() {
        ca.into_iter().map(|opt| opt.unwrap_or(0)).collect()
    } else if let Ok(ca) = series.i32() {
        ca.into_iter().map(|opt| opt.unwrap_or(0) as i64).collect()
    } else if let Ok(ca) = series.f64() {
        ca.into_iter().map(|opt| opt.unwrap_or(0.0) as i64).collect()
    } else if let Ok(ca) = series.str() {
        // For string columns, create unique integer encoding
        let mut mapping = std::collections::HashMap::new();
        let mut next_id = 0i64;
        ca.into_iter()
            .map(|opt| {
                let s = opt.unwrap_or("");
                *mapping.entry(s.to_string()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else if is_categorical {
        // For categorical columns, cast to string first then encode
        let str_series = series.cast(&DataType::String)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let ca = str_series.str()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let mut mapping = std::collections::HashMap::new();
        let mut next_id = 0i64;
        ca.into_iter()
            .map(|opt| {
                let s = opt.unwrap_or("");
                *mapping.entry(s.to_string()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cluster column '{}' has unsupported dtype; expected integer, float, string, or categorical",
            col_name
        )));
    };
    
    if cluster_vec.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cluster column must have same length as data: {} has {}, expected {}",
            col_name,
            cluster_vec.len(),
            expected_len
        )));
    }
    
    Ok(cluster_vec)
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
    df: PyDataFrame,
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

    // OPTIMIZED: Extract y column using native Arrow path (Phase 3)
    let y_vec = extract_f64_column(&df, y_col)?;
    let n_rows = y_vec.len();

    // OPTIMIZED: Extract all x columns using native Arrow path (Phase 3)
    let (x_flat, _, n_x_cols) = extract_f64_columns_flat(&df, &x_cols)?;

    // Validate dimensions
    if x_flat.len() != n_rows * n_x_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "X matrix dimension mismatch: expected {} elements ({}×{}), got {}",
            n_rows * n_x_cols,
            n_rows,
            n_x_cols,
            x_flat.len()
        )));
    }

    // Extract cluster column if specified using native Arrow path
    let cluster_ids: Option<Vec<i64>> = if let Some(cluster_col) = cluster {
        Some(extract_cluster_column(&df, cluster_col, n_rows)?)
    } else {
        None
    };

    // Compute regression with optional clustering using optimized flat data path
    compute_linear_regression_flat(
        &x_flat,
        n_rows,
        n_x_cols,
        &y_vec,
        include_intercept,
        cluster_ids.as_deref(),
        bootstrap,
        bootstrap_iterations,
        seed,
        weight_type,
    )
}

/// Optimized linear regression computation using flat data directly.
///
/// This function builds the faer::Mat directly from flat array data,
/// eliminating intermediate Vec<Vec<f64>> allocations for the common non-clustered case.
#[allow(clippy::too_many_arguments)]
fn compute_linear_regression_flat(
    x_flat: &[f64],
    n_rows: usize,
    n_x_cols: usize,
    y: &[f64],
    include_intercept: bool,
    cluster_ids: Option<&[i64]>,
    bootstrap: bool,
    bootstrap_iterations: usize,
    seed: Option<u64>,
    weight_type: BootstrapWeightType,
) -> PyResult<LinearRegressionResult> {
    if n_rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot perform regression on empty data",
        ));
    }

    if n_rows != y.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x and y must have the same number of rows: x has {}, y has {}",
            n_rows,
            y.len()
        )));
    }

    if n_x_cols == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must have at least one variable",
        ));
    }

    let n = n_rows;
    let n_vars = n_x_cols;
    let n_params = if include_intercept { n_x_cols + 1 } else { n_x_cols };

    // Check if we have enough samples
    if n < n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Not enough samples: need at least {} samples for {} parameters",
            n_params, n_params
        )));
    }

    // OPTIMIZED: Build design matrix directly from flat data using faer::Mat
    // This skips the Vec<Vec<f64>> intermediate format entirely
    let x_faer = linalg::flat_to_mat_with_intercept(x_flat, n_rows, n_x_cols, include_intercept);

    // Compute X'X using faer (BLAS gemm)
    let xtx_faer = linalg::xtx(&x_faer);

    // Compute X'y using faer
    let xty_vec = linalg::xty(&x_faer, y);

    // Compute (X'X)^-1 using Cholesky decomposition
    let xtx_inv_faer = linalg::invert_xtx(&xtx_faer)?;

    // Compute coefficients: β = (X'X)^-1 X'y via Cholesky solve
    let coefficients_full = linalg::solve_normal_equations(&xtx_faer, &xty_vec)?;

    // Compute fitted values: ŷ = Xβ using faer
    let fitted_values = linalg::mat_vec_mul(&x_faer, &coefficients_full);

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

        // OPTIMIZED: Use faer matrices directly (no Vec<Vec<f64>> design_matrix overhead)
        if bootstrap {
            // Wild cluster bootstrap with faer matrices
            let (coef_se, int_se) = compute_cluster_se_bootstrap_faer(
                    &x_faer,
                    &fitted_values,
                    &residuals,
                    &xtx_inv_faer,
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
            // Analytical clustered SE with faer matrices
            let (coef_se, int_se) = compute_cluster_se_analytical_faer(
                    &x_faer,
                    &residuals,
                    &xtx_inv_faer,
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
            // Compute HC3 leverages using optimized faer batch computation
            let leverages = linalg::compute_leverages_batch(&x_faer, &xtx_inv_faer)?;

            // Compute HC3 variance-covariance matrix using faer
            let hc3_vcov_faer =
                linalg::compute_hc3_vcov_faer(&x_faer, &residuals, &leverages, &xtx_inv_faer);

            if include_intercept {
                let intercept_se_val = hc3_vcov_faer.read(0, 0).sqrt();
                let se_vec: Vec<f64> = (1..n_params)
                    .map(|i| hc3_vcov_faer.read(i, i).sqrt())
                    .collect();
                (Some(intercept_se_val), se_vec, None, None, None)
            } else {
                let se_vec: Vec<f64> = (0..n_params)
                    .map(|i| hc3_vcov_faer.read(i, i).sqrt())
                    .collect();
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
    df: PyDataFrame,
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

    // OPTIMIZED: Extract y column using native Arrow path (Phase 3)
    let y_vec = extract_f64_column(&df, y_col)?;
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

    // OPTIMIZED: Extract all x columns using native Arrow path (Phase 3)
    let (x_flat, _, n_x_cols) = extract_f64_columns_flat(&df, &x_cols)?;

    // Validate dimensions
    if x_flat.len() != n_rows * n_x_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "X matrix dimension mismatch: expected {} elements ({}×{}), got {}",
            n_rows * n_x_cols,
            n_rows,
            n_x_cols,
            x_flat.len()
        )));
    }

    // Extract cluster column if specified using native Arrow path
    let cluster_ids: Option<Vec<i64>> = if let Some(cluster_col) = cluster {
        Some(extract_cluster_column(&df, cluster_col, n_rows)?)
    } else {
        None
    };

    // Compute logistic regression with optional clustering using optimized flat data path
    compute_logistic_regression_flat(
        &x_flat,
        n_rows,
        n_x_cols,
        &y_vec,
        include_intercept,
        cluster_ids.as_deref(),
        bootstrap,
        bootstrap_iterations,
        seed,
        weight_type,
    )
}

/// Optimized logistic regression computation using flat data directly.
///
/// This function builds the faer::Mat directly from flat array data,
/// eliminating intermediate Vec<Vec<f64>> allocations for the common non-clustered case.
#[allow(clippy::too_many_arguments)]
fn compute_logistic_regression_flat(
    x_flat: &[f64],
    n_rows: usize,
    n_x_cols: usize,
    y: &[f64],
    include_intercept: bool,
    cluster_ids: Option<&[i64]>,
    bootstrap: bool,
    bootstrap_iterations: usize,
    seed: Option<u64>,
    weight_type: BootstrapWeightType,
) -> PyResult<LogisticRegressionResult> {
    let n = n_rows;

    // OPTIMIZED: Build design matrix directly from flat data using faer::Mat
    // This skips the Vec<Vec<f64>> intermediate format entirely
    let design_mat_faer = linalg::flat_to_mat_with_intercept(x_flat, n_rows, n_x_cols, include_intercept);

    // Run MLE optimization with faer::Mat directly
    let mle_result = compute_logistic_mle(&design_mat_faer, y).map_err(|e| match e {
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

        // Build design matrix Vec<Vec<f64>> for cluster SE functions (only when clustering)
        let design_matrix: Vec<Vec<f64>> = (0..n_rows)
            .map(|i| {
                let row_start = i * n_x_cols;
                if include_intercept {
                    let mut row = Vec::with_capacity(n_x_cols + 1);
                    row.push(1.0);
                    row.extend_from_slice(&x_flat[row_start..row_start + n_x_cols]);
                    row
                } else {
                    x_flat[row_start..row_start + n_x_cols].to_vec()
                }
            })
            .collect();

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
        // Non-clustered: use HC3 with faer matrices directly (no Vec<Vec<f64>> overhead)
        // Convert info_inv from Vec<Vec<f64>> to faer::Mat for HC3 computation
        let info_inv_mat = linalg::vec_to_mat(&mle_result.info_inv);

        let se = compute_hc3_logistic_faer(&design_mat_faer, y, &mle_result.pi, &info_inv_mat)
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
