use pyo3::prelude::*;

mod stats;

use stats::{LinearRegressionResult, compute_linear_regression};

/// Main module for causers - statistical operations for Polars DataFrames
#[pymodule]
fn _causers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LinearRegressionResult>()?;
    m.add_function(wrap_pyfunction!(linear_regression, m)?)?;
    Ok(())
}

/// Perform linear regression on Polars DataFrame columns
///
/// Args:
///     df: Polars DataFrame
///     x_cols: List of names of the independent variable columns
///     y_col: Name of the dependent variable column
///     include_intercept: Whether to include an intercept term (default: True)
///
/// Returns:
///     LinearRegressionResult with coefficients, intercept, and r_squared
#[pyfunction]
#[pyo3(signature = (df, x_cols, y_col, include_intercept=true))]
fn linear_regression(
    py: Python,
    df: PyObject,
    x_cols: Vec<String>,
    y_col: &str,
    include_intercept: bool,
) -> PyResult<LinearRegressionResult> {
    if x_cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x_cols must contain at least one column name"
        ));
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
        let x_series = df.getattr(py, "get_column")?.call1(py, (col_name.as_str(),))?;
        let x_array = x_series.call_method0(py, "to_numpy")?;
        let x_vec: Vec<f64> = x_array.extract(py)?;
        
        if x_vec.len() != n_rows {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("All columns must have the same length: {} has {}, expected {}",
                    col_name, x_vec.len(), n_rows)
            ));
        }
        
        // Add this column's values to each row
        for (i, val) in x_vec.iter().enumerate() {
            x_matrix[i].push(*val);
        }
    }
    
    // Compute regression
    compute_linear_regression(&x_matrix, &y_vec, include_intercept)
}