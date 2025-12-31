//! Causers - High-performance causal inference library.
//!
//! This module provides Python bindings via PyO3 for statistical methods including:
//! - Linear regression with fixed effects and clustered standard errors
//! - Logistic regression with robust inference
//! - Synthetic control methods
//! - Synthetic Difference-in-Differences (SDID)
//! - Double Machine Learning (DML)
//! - Two-Stage Least Squares (2SLS) instrumental variables
//!
//! The implementation uses Rust for performance-critical computations, leveraging
//! the faer linear algebra library and Polars for efficient data operations.

// ============================================================================
// Standard Library Imports
// ============================================================================
use std::collections::HashMap;

// ============================================================================
// External Crate Imports
// ============================================================================
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

// ============================================================================
// Module Declarations
// ============================================================================
mod cluster;
mod dml;
mod fixed_effects;
mod iv2sls;
mod linalg;
mod logistic;
mod sdid;
mod stats;
mod synth_control;

// ============================================================================
// Local Module Imports
// ============================================================================
use cluster::{
    build_cluster_indices, compute_cluster_se_analytical_faer, compute_cluster_se_bootstrap_faer,
    BootstrapWeightType, ClusterError, ClusterInfo,
};
use dml::{compute_dml, DMLConfig, DMLError, DMLResult, TreatmentType};
use fixed_effects::{
    build_fe_indices, check_collinearity, compute_mundlak_terms, compute_within_r_squared,
    demean_one_way, demean_two_way, FixedEffectError, FixedEffectInfo,
};
use iv2sls::{compute_2sls, IV2SLSConfig, IV2SLSError, TwoStageLSResult};
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

// ============================================================================
// Module Configuration Constants
// ============================================================================

/// Convergence tolerance for two-way fixed effects demeaning
const FE_TOLERANCE: f64 = 1e-8;

/// Maximum iterations for two-way fixed effects demeaning
const FE_MAX_ITERATIONS: usize = 1000;

// ============================================================================
// PyO3 Module Configuration and Setup
// ============================================================================

/// Main module for causers - statistical operations for Polars DataFrames
#[pymodule]
fn _causers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LinearRegressionResult>()?;
    m.add_class::<LogisticRegressionResult>()?;
    m.add_class::<SyntheticDIDResult>()?;
    m.add_class::<SyntheticControlResult>()?;
    m.add_class::<DMLResult>()?;
    m.add_class::<TwoStageLSResult>()?;
    m.add_function(wrap_pyfunction!(linear_regression, m)?)?;
    m.add_function(wrap_pyfunction!(logistic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(synthetic_did_impl, m)?)?;
    m.add_function(wrap_pyfunction!(synthetic_control_impl, m)?)?;
    m.add_function(wrap_pyfunction!(dml_impl, m)?)?;
    m.add_function(wrap_pyfunction!(two_stage_least_squares, m)?)?;
    Ok(())
}

// ============================================================================
// Input Validation Utilities
// ============================================================================

/// Validate that a column name doesn't contain control characters
fn validate_column_name(name: &str) -> PyResult<()> {
    if name.bytes().any(|b| b < 0x20 || b == 0x7F) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Column name '{}' contains invalid characters",
            name
        )));
    }
    Ok(())
}

/// Validate multiple column names for control characters
#[allow(dead_code)]
fn validate_column_names(names: &[impl AsRef<str>]) -> PyResult<()> {
    for name in names {
        validate_column_name(name.as_ref())?;
    }
    Ok(())
}

// ============================================================================
// Data Extraction Utilities
// ============================================================================

/// Extract a single f64 column from a Polars DataFrame
fn extract_f64_column(df: &PyDataFrame, col_name: &str) -> PyResult<Vec<f64>> {
    let series = df
        .as_ref()
        .column(col_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let ca = series
        .f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let vec: Vec<f64> = ca.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect();
    Ok(vec)
}

/// Extract multiple columns as flat row-major Vec<f64>
fn extract_f64_columns_flat(
    df: &PyDataFrame,
    col_names: &[String],
) -> PyResult<(Vec<f64>, usize, usize)> {
    let n_rows = df.as_ref().height();
    let n_cols = col_names.len();
    let mut flat = Vec::with_capacity(n_rows * n_cols);

    // Pre-extract all columns ONCE (not n_rows times!)
    let columns: Vec<_> = col_names
        .iter()
        .map(|name| {
            let series = df
                .as_ref()
                .column(name)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            series
                .f64()
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

/// Generic helper to encode a Polars Series column to integer indices.
///
/// This function handles integer, float, string, and categorical columns by automatically
/// encoding unique values to sequential integer indices. The output type is controlled by
/// the generic parameter `T`, which must implement `From<usize>` and `Copy`.
///
/// # Type Parameters
///
/// * `T` - Target integer type (e.g., `i64` for clusters, `usize` for fixed effects)
///
/// # Arguments
///
/// * `series` - Polars Series to encode
/// * `col_name` - Column name for error messages
/// * `context` - Description of column usage (e.g., "Cluster", "Fixed effect") for error messages
///
/// # Returns
///
/// A vector of encoded indices
fn encode_column_to_indices<T>(series: &Series, col_name: &str, context: &str) -> PyResult<Vec<T>>
where
    T: From<usize> + Copy,
{
    // Check for nulls
    if series.null_count() > 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} column '{}' contains null values",
            context, col_name
        )));
    }

    // Check if it's a categorical type
    let is_categorical = matches!(series.dtype(), DataType::Categorical(_, _));

    // Try to extract and encode to indices
    let indices: Vec<T> = if let Ok(ca) = series.i64() {
        // For integer columns, create unique group encoding
        let mut mapping = HashMap::new();
        let mut next_id = 0usize;
        ca.into_iter()
            .map(|opt| {
                let val = opt.unwrap_or(0);
                let id = *mapping.entry(val).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                T::from(id)
            })
            .collect()
    } else if let Ok(ca) = series.i32() {
        let mut mapping = HashMap::new();
        let mut next_id = 0usize;
        ca.into_iter()
            .map(|opt| {
                let val = opt.unwrap_or(0) as i64;
                let id = *mapping.entry(val).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                T::from(id)
            })
            .collect()
    } else if let Ok(ca) = series.f64() {
        // For float columns, cast to integer then encode
        let mut mapping = HashMap::new();
        let mut next_id = 0usize;
        ca.into_iter()
            .map(|opt| {
                let val = opt.unwrap_or(0.0) as i64;
                let id = *mapping.entry(val).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                T::from(id)
            })
            .collect()
    } else if let Ok(ca) = series.str() {
        // For string columns, create unique integer encoding
        let mut mapping = HashMap::new();
        let mut next_id = 0usize;
        ca.into_iter()
            .map(|opt| {
                let s = opt.unwrap_or("");
                let id = *mapping.entry(s.to_string()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                T::from(id)
            })
            .collect()
    } else if is_categorical {
        // For categorical columns, cast to string first then encode
        let str_series = series
            .cast(&DataType::String)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let ca = str_series
            .str()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let mut mapping = HashMap::new();
        let mut next_id = 0usize;
        ca.into_iter()
            .map(|opt| {
                let s = opt.unwrap_or("");
                let id = *mapping.entry(s.to_string()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                T::from(id)
            })
            .collect()
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} column '{}' has unsupported dtype; expected integer, float, string, or categorical",
            context, col_name
        )));
    };

    Ok(indices)
}

/// Extract cluster column as Vec<i64> from a Polars DataFrame
///
/// Handles integer, float, string, and categorical columns with automatic encoding.
fn extract_cluster_column(
    df: &PyDataFrame,
    col_name: &str,
    expected_len: usize,
) -> PyResult<Vec<i64>> {
    let column = df
        .as_ref()
        .column(col_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let series = column.as_materialized_series();

    let cluster_vec: Vec<i64> = encode_column_to_indices::<usize>(series, col_name, "Cluster")?
        .into_iter()
        .map(|x| x as i64)
        .collect();

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

/// Extract fixed effect column as Vec<usize> (group indices) from a Polars DataFrame.
///
/// Handles integer, float, string, and categorical columns with automatic encoding.
/// Returns group indices suitable for use with FE demeaning functions.
fn extract_fe_column(
    df: &PyDataFrame,
    col_name: &str,
    expected_len: usize,
) -> PyResult<Vec<usize>> {
    let column = df.as_ref().column(col_name).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Fixed effect column '{}' not found in DataFrame",
            col_name
        ))
    })?;
    let series = column.as_materialized_series();

    let group_indices = encode_column_to_indices::<usize>(series, col_name, "Fixed effect")?;

    if group_indices.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Fixed effect column must have same length as data: {} has {}, expected {}",
            col_name,
            group_indices.len(),
            expected_len
        )));
    }

    Ok(group_indices)
}

// ============================================================================
// Bootstrap Configuration Helpers
// ============================================================================

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

// ============================================================================
// Linear Regression
// ============================================================================

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
///     fixed_effects: Optional list of column names for fixed effects to absorb (max 2)
///
/// Returns:
///     LinearRegressionResult with coefficients, intercept, r_squared, and standard errors
#[pyfunction]
#[pyo3(signature = (df, x_cols, y_col, include_intercept=true, cluster=None, bootstrap=false, bootstrap_iterations=1000, seed=None, bootstrap_method="rademacher", fixed_effects=None))]
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
    fixed_effects: Option<Vec<String>>,
) -> PyResult<LinearRegressionResult> {
    // Validate inputs
    if x_cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x_cols must contain at least one column name",
        ));
    }

    // Validate bootstrap requires cluster
    if bootstrap && cluster.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap=True requires cluster to be specified",
        ));
    }

    // Validate bootstrap_iterations
    if bootstrap_iterations < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_iterations must be at least 1",
        ));
    }

    // Parse and validate bootstrap_method
    let weight_type = parse_bootstrap_method(bootstrap_method)?;

    // Validate bootstrap_method requires bootstrap=True
    if weight_type != BootstrapWeightType::Rademacher && !bootstrap {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_method requires bootstrap=True",
        ));
    }

    // Validate bootstrap_method requires cluster
    if weight_type != BootstrapWeightType::Rademacher && cluster.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_method requires cluster to be specified",
        ));
    }

    // Validate column names for control characters
    validate_column_name(y_col)?;
    for col in &x_cols {
        validate_column_name(col)?;
    }
    if let Some(cluster_col) = cluster {
        validate_column_name(cluster_col)?;
    }

    // Validate fixed_effects columns
    if let Some(ref fe_cols) = fixed_effects {
        // Max 2 FE columns
        if fe_cols.len() > 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fixed_effects supports at most 2 columns",
            ));
        }

        // Validate FE column names
        for col in fe_cols {
            validate_column_name(col)?;
        }

        // FE columns must not overlap with x_cols
        for fe_col in fe_cols {
            if x_cols.contains(fe_col) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Fixed effect column '{}' cannot also be a covariate",
                    fe_col
                )));
            }
        }

        // FE columns must not overlap with y_col
        for fe_col in fe_cols {
            if fe_col == y_col {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Fixed effect column '{}' cannot be the outcome variable",
                    fe_col
                )));
            }
        }
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

    // Extract and build fixed effects info if specified
    let fe_info: Option<Vec<FixedEffectInfo>> = if let Some(ref fe_cols) = fixed_effects {
        let mut infos = Vec::with_capacity(fe_cols.len());
        for fe_col in fe_cols {
            let group_indices = extract_fe_column(&df, fe_col, n_rows)?;

            // Build FE info using build_fe_indices
            let info = build_fe_indices(&group_indices, fe_col).map_err(|e| match e {
                FixedEffectError::SingleUniqueValue { name } => {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Fixed effect column '{}' has only one unique value; cannot absorb",
                        name
                    ))
                }
                FixedEffectError::NullValues { name } => pyo3::exceptions::PyValueError::new_err(
                    format!("Fixed effect column '{}' contains null values", name),
                ),
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
            })?;
            infos.push(info);
        }
        Some(infos)
    } else {
        None
    };

    // Compute regression with optional clustering and FE using optimized flat data path
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
        fe_info.as_deref(),
        fixed_effects
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect()),
        &x_cols,
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
    fe_info: Option<&[FixedEffectInfo]>,
    fe_names: Option<Vec<&str>>,
    x_col_names: &[String],
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

    // Track absorbed FE groups and total absorbed DoF
    let (fe_absorbed, fe_names_out, total_fe_dof): (
        Option<Vec<usize>>,
        Option<Vec<String>>,
        usize,
    ) = if let Some(fe_infos) = fe_info {
        let absorbed: Vec<usize> = fe_infos.iter().map(|info| info.n_groups).collect();
        let names: Vec<String> = fe_names
            .as_ref()
            .map(|n| n.iter().map(|s| s.to_string()).collect())
            .unwrap_or_else(|| {
                vec!["fe1".to_string(), "fe2".to_string()][..absorbed.len()].to_vec()
            });
        // Total DoF absorbed = sum(n_groups - 1) for each FE (one group normalized to 0)
        let dof: usize = absorbed.iter().map(|&g| g.saturating_sub(1)).sum();
        (Some(absorbed), Some(names), dof)
    } else {
        (None, None, 0)
    };

    // Determine if we use intercept after FE: when FE are absorbed, intercept is implicitly absorbed
    let effective_intercept = if fe_info.is_some() {
        false
    } else {
        include_intercept
    };
    let n_params = if effective_intercept {
        n_x_cols + 1
    } else {
        n_x_cols
    };

    // Check if we have enough samples (accounting for FE DoF)
    let total_params = n_params + total_fe_dof;
    if n < total_params {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Not enough samples: need at least {} samples for {} parameters ({} covariates + {} FE DoF)",
            total_params, total_params, n_params, total_fe_dof
        )));
    }

    // Apply fixed effects demeaning if specified
    let (x_working, y_working): (Vec<f64>, Vec<f64>) = if let Some(fe_infos) = fe_info {
        // Convert x_flat to column-major format for demeaning (each column is a variable)
        let mut x_cols: Vec<Vec<f64>> = (0..n_x_cols)
            .map(|col| {
                (0..n_rows)
                    .map(|row| x_flat[row * n_x_cols + col])
                    .collect()
            })
            .collect();
        let mut y_dem = y.to_vec();

        if fe_infos.len() == 1 {
            // One-way FE demeaning
            let fe = &fe_infos[0];
            for x_col in &mut x_cols {
                *x_col = demean_one_way(x_col, fe);
            }
            y_dem = demean_one_way(&y_dem, fe);
        } else if fe_infos.len() == 2 {
            // Two-way FE demeaning using alternating projections
            // Uses module-level constants: FE_TOLERANCE and FE_MAX_ITERATIONS
            let fe1 = &fe_infos[0];
            let fe2 = &fe_infos[1];
            for x_col in &mut x_cols {
                *x_col = demean_two_way(x_col, fe1, fe2, FE_TOLERANCE, FE_MAX_ITERATIONS).map_err(
                    |e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Two-way demeaning failed: {}",
                            e
                        ))
                    },
                )?;
            }
            y_dem =
                demean_two_way(&y_dem, fe1, fe2, FE_TOLERANCE, FE_MAX_ITERATIONS).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Two-way demeaning failed: {}",
                        e
                    ))
                })?;
        }

        // Convert back to row-major flat format
        let mut x_flat_dem = Vec::with_capacity(n_rows * n_x_cols);
        for row in 0..n_rows {
            for x_col in x_cols.iter().take(n_x_cols) {
                x_flat_dem.push(x_col[row]);
            }
        }

        // Check for collinearity in demeaned X
        check_collinearity(&x_flat_dem, n_rows, n_x_cols, x_col_names, 1e-10).map_err(|e| {
            match e {
                FixedEffectError::Collinearity { covariate } => {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Covariate '{}' is collinear with fixed effects (near-zero within-group variance)",
                        covariate
                    ))
                }
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string())
            }
        })?;

        (x_flat_dem, y_dem)
    } else {
        (x_flat.to_vec(), y.to_vec())
    };

    // OPTIMIZED: Build design matrix directly from flat data using faer::Mat
    // For FE regression, don't add intercept (it's absorbed)
    let x_faer =
        linalg::flat_to_mat_with_intercept(&x_working, n_rows, n_x_cols, effective_intercept);

    // Compute X'X using faer (BLAS gemm)
    let xtx_faer = linalg::xtx(&x_faer);

    // Compute X'y using faer
    let xty_vec = linalg::xty(&x_faer, &y_working);

    // Compute (X'X)^-1 using Cholesky decomposition
    let xtx_inv_faer = linalg::invert_xtx(&xtx_faer)?;

    // Compute coefficients: β = (X'X)^-1 X'y via Cholesky solve
    let coefficients_full = linalg::solve_normal_equations(&xtx_faer, &xty_vec)?;

    // Compute fitted values: ŷ = Xβ using faer
    let fitted_values = linalg::mat_vec_mul(&x_faer, &coefficients_full);

    // Compute residuals: e = y - ŷ (using demeaned y)
    let residuals: Vec<f64> = (0..n).map(|i| y_working[i] - fitted_values[i]).collect();

    // Calculate R-squared (using original y for total R², demeaned y for within R²)
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    let r_squared = if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - (ss_res / ss_tot)
    };

    // Compute within R-squared for FE models
    let within_r_squared_opt = if fe_info.is_some() {
        Some(compute_within_r_squared(&y_working, &fitted_values))
    } else {
        None
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
                    effective_intercept,
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
                    effective_intercept,
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
            if effective_intercept {
                (Some(0.0), vec![0.0; n_vars], None, None, None)
            } else {
                (None, vec![0.0; n_vars], None, None, None)
            }
        } else {
            // Compute HC3 leverages using optimized faer batch computation
            let leverages = linalg::compute_leverages_batch(&x_faer, &xtx_inv_faer)?;

            // Compute HC3 variance-covariance matrix using faer
            // Note: For FE models, DoF adjustment is handled implicitly through demeaned data
            let hc3_vcov_faer =
                linalg::compute_hc3_vcov_faer(&x_faer, &residuals, &leverages, &xtx_inv_faer);

            if effective_intercept {
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
    // For FE models, intercept is absorbed, so we don't return one
    let (intercept, coefficients) = if effective_intercept {
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
        // Fixed effects fields
        fixed_effects_absorbed: fe_absorbed,
        fixed_effects_names: fe_names_out,
        within_r_squared: within_r_squared_opt,
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
///     fixed_effects: Optional list of column names for fixed effects (max 2). Uses Mundlak strategy.
///
/// Returns:
///     LogisticRegressionResult with coefficients, standard errors, and diagnostics
#[pyfunction]
#[pyo3(signature = (df, x_cols, y_col, include_intercept=true, cluster=None, bootstrap=false, bootstrap_iterations=1000, seed=None, bootstrap_method="rademacher", fixed_effects=None))]
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
    fixed_effects: Option<Vec<String>>,
) -> PyResult<LogisticRegressionResult> {
    // Validate inputs
    if x_cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x_cols must contain at least one column name",
        ));
    }

    // Validate bootstrap requires cluster
    if bootstrap && cluster.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap=True requires cluster to be specified",
        ));
    }

    // Validate bootstrap_iterations
    if bootstrap_iterations < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_iterations must be at least 1",
        ));
    }

    // Parse and validate bootstrap_method
    let weight_type = parse_bootstrap_method(bootstrap_method)?;

    // Validate bootstrap_method requires bootstrap=True
    if weight_type != BootstrapWeightType::Rademacher && !bootstrap {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_method requires bootstrap=True",
        ));
    }

    // Validate bootstrap_method requires cluster
    if weight_type != BootstrapWeightType::Rademacher && cluster.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bootstrap_method requires cluster to be specified",
        ));
    }

    // Validate column names for control characters
    validate_column_name(y_col)?;
    for col in &x_cols {
        validate_column_name(col)?;
    }
    if let Some(cluster_col) = cluster {
        validate_column_name(cluster_col)?;
    }

    // Validate fixed_effects columns
    if let Some(ref fe_cols) = fixed_effects {
        // Max 2 FE columns
        if fe_cols.len() > 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fixed_effects supports at most 2 columns",
            ));
        }

        // Validate FE column names for control characters
        for col in fe_cols {
            validate_column_name(col)?;
        }

        // FE columns must not overlap with x_cols
        for fe_col in fe_cols {
            if x_cols.contains(fe_col) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Fixed effect column '{}' cannot also be a covariate",
                    fe_col
                )));
            }
        }

        // FE columns must not equal y_col
        for fe_col in fe_cols {
            if fe_col == y_col {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Fixed effect column '{}' cannot be the outcome variable",
                    fe_col
                )));
            }
        }
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
    let has_zero = y_vec.contains(&0.0);
    let has_one = y_vec.contains(&1.0);
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

    // Extract and build fixed effects info if specified
    let fe_info: Option<Vec<FixedEffectInfo>> = if let Some(ref fe_cols) = fixed_effects {
        let mut infos = Vec::with_capacity(fe_cols.len());
        for fe_col in fe_cols {
            let group_indices = extract_fe_column(&df, fe_col, n_rows)?;

            // Build FE info using build_fe_indices
            let info = build_fe_indices(&group_indices, fe_col).map_err(|e| match e {
                FixedEffectError::SingleUniqueValue { name } => {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Fixed effect column '{}' has only one unique value; cannot absorb",
                        name
                    ))
                }
                FixedEffectError::NullValues { name } => pyo3::exceptions::PyValueError::new_err(
                    format!("Fixed effect column '{}' contains null values", name),
                ),
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
            })?;
            infos.push(info);
        }
        Some(infos)
    } else {
        None
    };

    // Dispatch to FE-aware function or standard function
    if let Some(ref fe_infos) = fe_info {
        let fe_names = fixed_effects
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<&str>>())
            .unwrap_or_default();

        compute_logistic_regression_with_fe(
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
            fe_infos,
            fe_names,
            &x_cols,
        )
    } else {
        // Standard path without FE
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
    let design_mat_faer =
        linalg::flat_to_mat_with_intercept(x_flat, n_rows, n_x_cols, include_intercept);

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
        // FE fields are None when not using fixed_effects
        fixed_effects_absorbed: None,
        fixed_effects_names: None,
        within_pseudo_r_squared: None,
    })
}

// ============================================================================
// Logistic Regression with Fixed Effects (Mundlak Strategy)
// ============================================================================

/// Compute logistic regression with Mundlak fixed effects approximation.
///
/// Uses the Mundlak strategy: instead of demeaning (which doesn't work for nonlinear models),
/// we add group means of covariates as additional regressors. The augmented model:
///   logit(P(y=1)) = β₀ + X'β + X̄_group'γ
///
/// where X̄_group are the within-group means of X for each FE dimension.
///
/// This produces coefficients that approximate the fixed effects model while allowing
/// for estimation via standard Newton-Raphson MLE.
///
/// # Output
/// Only the original covariate coefficients β are returned (not the Mundlak term coefficients γ).
/// Standard errors are computed on the full augmented model but filtered to match.
#[allow(clippy::too_many_arguments)]
fn compute_logistic_regression_with_fe(
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
    fe_infos: &[FixedEffectInfo],
    fe_names: Vec<&str>,
    _x_col_names: &[String], // Reserved for future use (collinearity error messages)
) -> PyResult<LogisticRegressionResult> {
    let n = n_rows;
    let n_fe = fe_infos.len();

    // TASK-006 Step 1: Compute Mundlak terms (group means of covariates)
    let mundlak_flat = compute_mundlak_terms(x_flat, n_rows, n_x_cols, fe_infos);
    let n_mundlak_cols = n_x_cols * n_fe;

    // Total augmented columns: original X + Mundlak terms
    let n_aug_cols = n_x_cols + n_mundlak_cols;

    // TASK-006 Step 2: Build augmented design matrix [intercept | X | Mundlak]
    // Using faer::Mat for efficiency
    let n_total_cols = if include_intercept {
        1 + n_aug_cols
    } else {
        n_aug_cols
    };

    let mut design_mat_aug = faer::Mat::<f64>::zeros(n_rows, n_total_cols);

    for i in 0..n_rows {
        let mut col_idx = 0;

        // Add intercept column (all 1s)
        if include_intercept {
            design_mat_aug.write(i, col_idx, 1.0);
            col_idx += 1;
        }

        // Add original covariates
        for j in 0..n_x_cols {
            design_mat_aug.write(i, col_idx + j, x_flat[i * n_x_cols + j]);
        }
        col_idx += n_x_cols;

        // Add Mundlak terms
        for m in 0..n_mundlak_cols {
            design_mat_aug.write(i, col_idx + m, mundlak_flat[i * n_mundlak_cols + m]);
        }
    }

    // TASK-007: Collinearity check on augmented matrix
    // Check condition number via diagonal ratio of X'X
    let xtx_aug = linalg::xtx(&design_mat_aug);
    let p_aug = design_mat_aug.ncols();

    let diag: Vec<f64> = (0..p_aug).map(|i| xtx_aug.read(i, i)).collect();
    let max_diag = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_diag = diag.iter().cloned().fold(f64::INFINITY, f64::min);

    if min_diag <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Mundlak terms cause collinearity; model is not identified. \
             Consider removing covariates constant within FE groups.",
        ));
    }

    let condition_estimate = max_diag / min_diag;
    if condition_estimate > 1e10 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Mundlak terms cause collinearity (condition number ≈ {:.2e}); \
             model is not identified. Consider removing covariates constant within FE groups.",
            condition_estimate
        )));
    }

    // TASK-006 Step 3: Run MLE on augmented matrix
    let mle_result = compute_logistic_mle(&design_mat_aug, y).map_err(|e| match e {
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
            "Hessian matrix is singular; check for collinearity in augmented model",
        ),
        LogisticError::NumericalInstability { message } => {
            pyo3::exceptions::PyValueError::new_err(message)
        }
    })?;

    // Compute null log-likelihood and pseudo R²
    let ll_null = compute_null_log_likelihood(y);
    let pseudo_r_squared = compute_pseudo_r_squared(mle_result.log_likelihood, ll_null);

    // Compute within pseudo R² (same as overall for Mundlak model, but could be refined)
    let within_pseudo_r_squared = pseudo_r_squared;

    // Compute standard errors on full augmented model
    let se_full: Vec<f64> = if let Some(cluster_ids) = cluster_ids {
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
                    "Cluster {} contains only 1 observation; use bootstrap=True",
                    cluster_idx
                ))
            }
            ClusterError::NumericalInstability { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            ClusterError::InvalidStandardErrors => pyo3::exceptions::PyValueError::new_err(
                "Standard error computation produced invalid values",
            ),
        })?;

        // Build design matrix Vec<Vec<f64>> for cluster SE functions
        let design_matrix: Vec<Vec<f64>> = (0..n_rows)
            .map(|i| {
                let mut row = Vec::with_capacity(n_total_cols);
                for j in 0..n_total_cols {
                    row.push(design_mat_aug.read(i, j));
                }
                row
            })
            .collect();

        if bootstrap {
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
                    "Standard error computation produced invalid values",
                ),
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
            })?;

            // Reconstruct full SE vector
            if include_intercept {
                let mut full_se = vec![int_se.unwrap_or(0.0)];
                full_se.extend(coef_se);
                full_se
            } else {
                coef_se
            }
        } else {
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
                        "Cluster {} contains only 1 observation; use bootstrap=True",
                        cluster_idx
                    ))
                }
                ClusterError::NumericalInstability { message } => {
                    pyo3::exceptions::PyValueError::new_err(message)
                }
                ClusterError::InvalidStandardErrors => pyo3::exceptions::PyValueError::new_err(
                    "Standard error computation produced invalid values",
                ),
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
            })?;

            // Reconstruct full SE vector
            if include_intercept {
                let mut full_se = vec![int_se.unwrap_or(0.0)];
                full_se.extend(coef_se);
                full_se
            } else {
                coef_se
            }
        }
    } else {
        // Non-clustered: use HC3 with faer matrices directly
        let info_inv_mat = linalg::vec_to_mat(&mle_result.info_inv);

        compute_hc3_logistic_faer(&design_mat_aug, y, &mle_result.pi, &info_inv_mat).map_err(
            |e| match e {
                LogisticError::NumericalInstability { message } => {
                    pyo3::exceptions::PyValueError::new_err(message)
                }
                _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
            },
        )?
    };

    // TASK-008: Filter output to original K coefficients only
    // Layout with intercept: [intercept, x1, x2, ..., xK, mundlak_fe1_x1, ..., mundlak_feD_xK]
    // Layout without intercept: [x1, x2, ..., xK, mundlak_fe1_x1, ..., mundlak_feD_xK]

    let offset = if include_intercept { 1 } else { 0 };

    let intercept = if include_intercept {
        Some(mle_result.beta[0])
    } else {
        None
    };

    let intercept_se = if include_intercept {
        Some(se_full[0])
    } else {
        None
    };

    // Extract original covariate coefficients (skip intercept, take K columns)
    let coefficients = mle_result.beta[offset..offset + n_x_cols].to_vec();
    let standard_errors = se_full[offset..offset + n_x_cols].to_vec();

    // Build FE metadata
    let fe_absorbed: Vec<usize> = fe_infos.iter().map(|info| info.n_groups).collect();
    let fe_names_owned: Vec<String> = fe_names.iter().map(|s| s.to_string()).collect();

    // Determine cluster info for result
    let (n_clusters_opt, cluster_se_type_opt, bootstrap_iterations_opt) = if cluster_ids.is_some() {
        let cluster_info = build_cluster_indices(cluster_ids.unwrap()).ok();
        let n_clusters = cluster_info.map(|c| c.n_clusters);
        let se_type = if bootstrap {
            Some(get_cluster_se_type(weight_type))
        } else {
            Some("analytical".to_string())
        };
        let boot_iter = if bootstrap {
            Some(bootstrap_iterations)
        } else {
            None
        };
        (n_clusters, se_type, boot_iter)
    } else {
        (None, None, None)
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
        // FE fields populated for Mundlak model
        fixed_effects_absorbed: Some(fe_absorbed),
        fixed_effects_names: Some(fe_names_owned),
        within_pseudo_r_squared: Some(within_pseudo_r_squared),
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
    let temp = linalg::matrix_multiply(info_inv, &meat);
    let v = linalg::matrix_multiply(&temp, info_inv);

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
        let delta = linalg::matrix_vector_multiply(info_inv, &perturbed_score);

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

// ============================================================================
// Double Machine Learning (DML) Implementation
// ============================================================================

/// Convert DMLError to PyErr
impl From<DMLError> for PyErr {
    fn from(err: DMLError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

/// Double Machine Learning estimation for causal treatment effects.
///
/// Implements the DML estimator from Chernozhukov et al. (2018) with cross-fitting
/// for debiased inference. Uses linear/logistic regression for nuisance models.
///
/// # Arguments
///
/// * `df` - Polars DataFrame containing the data
/// * `y_col` - Name of the outcome variable column
/// * `d_col` - Name of the treatment variable column
/// * `x_cols` - Names of the covariate columns to control for
/// * `n_folds` - Number of cross-fitting folds (must be 2, 5, or 10; default: 5)
/// * `treatment_type` - "binary" or "continuous" (default: "binary")
/// * `estimate_cate` - Whether to estimate CATE coefficients (default: false)
/// * `alpha` - Significance level for CI (default: 0.05 for 95% CI)
/// * `propensity_clip` - Bounds for propensity clipping (default: (0.01, 0.99))
/// * `cluster` - Optional column name for cluster-robust SE
/// * `seed` - Random seed for reproducible fold assignment
///
/// # Returns
///
/// DMLResult with treatment effect estimates and diagnostics
#[pyfunction]
#[pyo3(signature = (
    df,
    y_col,
    d_col,
    x_cols,
    n_folds=5,
    treatment_type="binary",
    estimate_cate=false,
    alpha=0.05,
    propensity_clip_low=0.01,
    propensity_clip_high=0.99,
    cluster=None,
    seed=None
))]
#[allow(clippy::too_many_arguments)]
fn dml_impl(
    df: PyDataFrame,
    y_col: &str,
    d_col: &str,
    x_cols: Vec<String>,
    n_folds: usize,
    treatment_type: &str,
    estimate_cate: bool,
    alpha: f64,
    propensity_clip_low: f64,
    propensity_clip_high: f64,
    cluster: Option<&str>,
    seed: Option<u64>,
) -> PyResult<DMLResult> {
    // Validate inputs
    if x_cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x_cols must contain at least one column name",
        ));
    }

    // Validate column names
    validate_column_name(y_col)?;
    validate_column_name(d_col)?;
    for col in &x_cols {
        validate_column_name(col)?;
    }
    if let Some(cluster_col) = cluster {
        validate_column_name(cluster_col)?;
    }

    // Parse treatment type
    let treatment = match treatment_type.to_lowercase().as_str() {
        "binary" => TreatmentType::Binary,
        "continuous" => TreatmentType::Continuous,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "treatment_type must be 'binary' or 'continuous'; got '{}'",
                treatment_type
            )));
        }
    };

    // Extract y column
    let y_vec = extract_f64_column(&df, y_col)?;
    let n_rows = y_vec.len();

    // Validate empty DataFrame
    if n_rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot perform DML on empty data",
        ));
    }

    // Extract d column
    let d_vec = extract_f64_column(&df, d_col)?;

    // Validate binary treatment has 0/1 values
    if treatment == TreatmentType::Binary {
        for &di in &d_vec {
            if di != 0.0 && di != 1.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Binary treatment must contain only 0 and 1 values",
                ));
            }
        }

        // Check both values present
        let has_zero = d_vec.contains(&0.0);
        let has_one = d_vec.contains(&1.0);
        if !has_zero || !has_one {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Binary treatment must contain both 0 and 1 values",
            ));
        }
    }

    // Extract x columns
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

    // Extract cluster column if specified
    let cluster_ids: Option<Vec<i64>> = if let Some(cluster_col) = cluster {
        Some(extract_cluster_column(&df, cluster_col, n_rows)?)
    } else {
        None
    };

    // Build DML config
    let config = DMLConfig {
        n_folds,
        treatment_type: treatment,
        estimate_cate,
        alpha,
        propensity_clip: (propensity_clip_low, propensity_clip_high),
        seed,
    };

    // Run DML computation
    let result = compute_dml(
        &y_vec,
        &d_vec,
        &x_flat,
        n_rows,
        n_x_cols,
        &x_cols,
        &config,
        cluster_ids.as_deref(),
    )?;

    Ok(result)
}

// ============================================================================
// Two-Stage Least Squares (IV2SLS) Implementation
// ============================================================================

/// Convert IV2SLSError to PyErr
impl From<IV2SLSError> for PyErr {
    fn from(err: IV2SLSError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

/// Two-Stage Least Squares (2SLS) instrumental variables estimation.
///
/// Estimates causal effects when treatment is endogenous using instrumental
/// variables. Implements the standard 2SLS estimator with weak instrument
/// diagnostics.
///
/// # Arguments
///
/// * `df` - Polars DataFrame containing the data
/// * `y_col` - Name of the outcome variable column
/// * `d_cols` - Names of the endogenous treatment variable columns
/// * `z_cols` - Names of the excluded instrument columns
/// * `x_cols` - Optional names of exogenous control variable columns
/// * `include_intercept` - Whether to include an intercept term (default: true)
/// * `robust` - Whether to compute HC3 robust standard errors (default: false)
/// * `cluster` - Optional column name for cluster-robust standard errors
///
/// # Returns
///
/// TwoStageLSResult with treatment effect estimates and diagnostics
///
/// # Example
///
/// ```python
/// import polars as pl
/// import causers
///
/// df = pl.DataFrame({
///     "wage": [...],      # outcome
///     "educ": [...],      # endogenous treatment
///     "quarter": [...],   # instrument (birth quarter)
///     "age": [...],       # exogenous control
/// })
///
/// result = causers.two_stage_least_squares(
///     df,
///     y_col="wage",
///     d_cols=["educ"],
///     z_cols=["quarter"],
///     x_cols=["age"],
/// )
/// print(f"Returns to education: {result.coefficients[0]:.3f}")
/// print(f"First-stage F-stat: {result.first_stage_f[0]:.1f}")
/// ```
#[pyfunction]
#[pyo3(signature = (
    df,
    y_col,
    d_cols,
    z_cols,
    x_cols=None,
    include_intercept=true,
    robust=false,
    cluster=None
))]
#[allow(clippy::too_many_arguments)]
fn two_stage_least_squares(
    df: PyDataFrame,
    y_col: &str,
    d_cols: Vec<String>,
    z_cols: Vec<String>,
    x_cols: Option<Vec<String>>,
    include_intercept: bool,
    robust: bool,
    cluster: Option<&str>,
) -> PyResult<TwoStageLSResult> {
    // Validate inputs
    if d_cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "d_cols must contain at least one endogenous variable",
        ));
    }

    if z_cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "z_cols must contain at least one instrument",
        ));
    }

    // Validate column names for control characters
    validate_column_name(y_col)?;
    for col in &d_cols {
        validate_column_name(col)?;
    }
    for col in &z_cols {
        validate_column_name(col)?;
    }
    if let Some(ref x) = x_cols {
        for col in x {
            validate_column_name(col)?;
        }
    }
    if let Some(cluster_col) = cluster {
        validate_column_name(cluster_col)?;
    }

    // Extract y column
    let y_vec = extract_f64_column(&df, y_col)?;
    let n_rows = y_vec.len();

    // Validate empty DataFrame
    if n_rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot perform 2SLS on empty data",
        ));
    }

    // Extract endogenous variables (D)
    let (d_flat, _, n_endog) = extract_f64_columns_flat(&df, &d_cols)?;
    if d_flat.len() != n_rows * n_endog {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Endogenous variable matrix dimension mismatch: expected {} elements ({}×{}), got {}",
            n_rows * n_endog,
            n_rows,
            n_endog,
            d_flat.len()
        )));
    }

    // Extract instruments (Z)
    let (z_flat, _, n_instruments) = extract_f64_columns_flat(&df, &z_cols)?;
    if z_flat.len() != n_rows * n_instruments {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Instrument matrix dimension mismatch: expected {} elements ({}×{}), got {}",
            n_rows * n_instruments,
            n_rows,
            n_instruments,
            z_flat.len()
        )));
    }

    // Extract exogenous controls (X) if specified
    let x_cols_vec = x_cols.unwrap_or_default();
    let n_exog = x_cols_vec.len();
    let x_flat = if n_exog > 0 {
        let (flat, _, _) = extract_f64_columns_flat(&df, &x_cols_vec)?;
        flat
    } else {
        vec![]
    };

    // Extract cluster column if specified
    let cluster_ids: Option<Vec<i64>> = if let Some(cluster_col) = cluster {
        Some(extract_cluster_column(&df, cluster_col, n_rows)?)
    } else {
        None
    };

    // Build configuration
    let config = IV2SLSConfig {
        include_intercept,
        robust,
        cluster_ids: cluster_ids.as_deref(),
    };

    // Run 2SLS computation
    let (result, warnings) = compute_2sls(
        &y_vec,
        &d_flat,
        &x_flat,
        &z_flat,
        n_rows,
        n_endog,
        n_exog,
        n_instruments,
        &d_cols,
        &x_cols_vec,
        &z_cols,
        &config,
    )?;

    // Log warnings (in production, these would go to a logger)
    for warning in warnings {
        // For now, we silently ignore warnings, but they could be exposed
        // through a Python logging mechanism
        let _ = warning;
    }

    Ok(result)
}
