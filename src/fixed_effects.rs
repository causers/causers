//! Fixed effects estimation for linear regression.
//!
//! This module implements one-way and two-way fixed effects using an alternating
//! projections (Gauss-Seidel) algorithm for demeaning. The approach matches
//! `pyfixest` output exactly while achieving efficient O(N) memory usage.
//!
//! # Algorithm
//!
//! For one-way fixed effects, we demean each variable by group means:
//! ```text
//! x_demeaned[i] = x[i] - mean(x[j] for j in group g(i))
//! ```
//!
//! For two-way fixed effects, we use alternating projections:
//! 1. Demean by FE1 groups
//! 2. Demean by FE2 groups
//! 3. Repeat until convergence (max change < tolerance)
//!
//! # References
//!
//! - Guimarães, P., & Portugal, P. (2010). A simple feasible procedure to fit
//!   models with high-dimensional fixed effects. The Stata Journal.

use std::fmt;

/// Error type for fixed effects operations.
#[derive(Debug, Clone)]
pub enum FixedEffectError {
    /// Fixed effect column not found in DataFrame.
    ColumnNotFound { name: String },

    /// Fixed effect column contains null values.
    NullValues { name: String },

    /// Fixed effect column has only one unique value (need at least 2 groups).
    SingleUniqueValue { name: String },

    /// More than 2 fixed effect columns provided (v1 limit).
    TooManyColumns { count: usize },

    /// Fixed effect column overlaps with covariate columns.
    OverlapWithCovariate { name: String },

    /// Fixed effect column equals the dependent variable column.
    OverlapWithY { name: String },

    /// Covariate is collinear with fixed effects (no within-group variation).
    Collinearity { covariate: String },

    /// Two-way FE demeaning did not converge within max iterations.
    ConvergenceFailure { iterations: usize },
}

impl fmt::Display for FixedEffectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixedEffectError::ColumnNotFound { name } => {
                write!(f, "Fixed effect column '{}' not found in DataFrame", name)
            }
            FixedEffectError::NullValues { name } => {
                write!(f, "Fixed effect column '{}' contains null values", name)
            }
            FixedEffectError::SingleUniqueValue { name } => {
                write!(
                    f,
                    "Fixed effect column '{}' has only one unique value",
                    name
                )
            }
            FixedEffectError::TooManyColumns { count } => {
                write!(
                    f,
                    "Fixed effects limited to 2 columns maximum in v1. Provided: {}",
                    count
                )
            }
            FixedEffectError::OverlapWithCovariate { name } => {
                write!(
                    f,
                    "Fixed effect column '{}' cannot also be a covariate",
                    name
                )
            }
            FixedEffectError::OverlapWithY { name } => {
                write!(
                    f,
                    "Fixed effect column '{}' cannot be the dependent variable",
                    name
                )
            }
            FixedEffectError::Collinearity { covariate } => {
                write!(
                    f,
                    "Covariate '{}' is collinear with fixed effects (no within-group variation)",
                    covariate
                )
            }
            FixedEffectError::ConvergenceFailure { iterations } => {
                write!(
                    f,
                    "Two-way FE demeaning did not converge after {} iterations",
                    iterations
                )
            }
        }
    }
}

impl std::error::Error for FixedEffectError {}

/// Fixed effect group information for a single FE variable.
///
/// Mirrors the ClusterInfo pattern from cluster.rs for consistency.
///
/// # Invariants
///
/// - `indices.len() == n_groups`
/// - `sizes.len() == n_groups`
/// - `sum(sizes) == total observations`
/// - `flatten(indices)` is a permutation of `0..n`
/// - `obs_to_group.len() == total observations`
/// - `obs_to_group[i]` is in `0..n_groups` for all `i`
#[derive(Debug, Clone)]
pub struct FixedEffectInfo {
    /// indices[g] = vector of observation indices in group g.
    pub indices: Vec<Vec<usize>>,

    /// Number of unique groups.
    pub n_groups: usize,

    /// sizes[g] = number of observations in group g.
    pub sizes: Vec<usize>,

    /// Original column name (for error messages and result reporting).
    pub name: String,

    /// Observation-to-group mapping for O(1) lookup during demeaning.
    /// obs_to_group[i] = group index for observation i.
    pub obs_to_group: Vec<usize>,
}

impl FixedEffectInfo {
    /// Create a new FixedEffectInfo from group assignments.
    ///
    /// # Arguments
    ///
    /// * `group_ids` - Group ID for each observation (0-indexed consecutive)
    /// * `column_name` - Name of the FE column (for error messages)
    ///
    /// # Returns
    ///
    /// * `Result<FixedEffectInfo, FixedEffectError>` - Info struct or error
    ///
    /// # Errors
    ///
    /// * `FixedEffectError::SingleUniqueValue` if there is only 1 unique group
    pub fn new(group_ids: &[usize], column_name: &str) -> Result<Self, FixedEffectError> {
        let n = group_ids.len();

        if n == 0 {
            return Err(FixedEffectError::SingleUniqueValue {
                name: column_name.to_string(),
            });
        }

        // Find number of groups
        let n_groups = group_ids.iter().max().map(|&m| m + 1).unwrap_or(0);

        // Validate: need at least 2 groups
        if n_groups < 2 {
            return Err(FixedEffectError::SingleUniqueValue {
                name: column_name.to_string(),
            });
        }

        // Build indices grouped by group ID
        let mut indices: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
        for (i, &g) in group_ids.iter().enumerate() {
            indices[g].push(i);
        }

        // Compute sizes
        let sizes: Vec<usize> = indices.iter().map(|idx| idx.len()).collect();

        // obs_to_group is just the input group_ids
        let obs_to_group = group_ids.to_vec();

        // Debug assertions for invariants
        debug_assert_eq!(indices.len(), n_groups);
        debug_assert_eq!(sizes.len(), n_groups);
        debug_assert_eq!(sizes.iter().sum::<usize>(), n);
        debug_assert_eq!(obs_to_group.len(), n);

        Ok(FixedEffectInfo {
            indices,
            n_groups,
            sizes,
            name: column_name.to_string(),
            obs_to_group,
        })
    }
}

/// Build FixedEffectInfo from observation group assignments.
///
/// This is the main entry point for creating FE index structures.
///
/// # Arguments
///
/// * `group_ids` - Group ID for each observation (consecutive 0-indexed integers)
/// * `column_name` - Name of the FE column for error messages
///
/// # Returns
///
/// * `Result<FixedEffectInfo, FixedEffectError>` - FE info or error
///
/// # Errors
///
/// * `FixedEffectError::SingleUniqueValue` if `n_groups < 2`
///
/// # Example
///
/// ```ignore
/// let group_ids = vec![0, 0, 1, 1, 2];
/// let fe_info = build_fe_indices(&group_ids, "firm_id")?;
/// assert_eq!(fe_info.n_groups, 3);
/// assert_eq!(fe_info.sizes, vec![2, 2, 1]);
/// ```
pub fn build_fe_indices(
    group_ids: &[usize],
    column_name: &str,
) -> Result<FixedEffectInfo, FixedEffectError> {
    FixedEffectInfo::new(group_ids, column_name)
}

/// Demean a data vector by group means (one-way fixed effects).
///
/// For each observation i in group g:
/// ```text
/// x_demeaned[i] = x[i] - mean(x[j] for j in group g)
/// ```
///
/// # Time Complexity
///
/// O(N) where N is the number of observations.
///
/// # Space Complexity
///
/// O(N + G) where G is the number of groups.
///
/// # Arguments
///
/// * `data` - Input data vector (length N)
/// * `fe_info` - Fixed effect group information
///
/// # Returns
///
/// * `Vec<f64>` - Demeaned data vector (length N)
pub fn demean_one_way(data: &[f64], fe_info: &FixedEffectInfo) -> Vec<f64> {
    // Compute group sums in O(N)
    let mut group_sums = vec![0.0; fe_info.n_groups];
    for (i, &val) in data.iter().enumerate() {
        group_sums[fe_info.obs_to_group[i]] += val;
    }

    // Compute group means
    let group_means: Vec<f64> = group_sums
        .iter()
        .zip(&fe_info.sizes)
        .map(|(&sum, &size)| sum / size as f64)
        .collect();

    // Demean in O(N)
    data.iter()
        .enumerate()
        .map(|(i, &val)| val - group_means[fe_info.obs_to_group[i]])
        .collect()
}

/// Compute group means into a pre-allocated buffer.
///
/// # Arguments
///
/// * `data` - Input data vector (length N)
/// * `fe_info` - Fixed effect group information
/// * `means` - Output buffer for group means (length G)
fn compute_group_means(data: &[f64], fe_info: &FixedEffectInfo, means: &mut [f64]) {
    // Reset means to zero
    means.fill(0.0);

    // Sum values per group
    for (i, &val) in data.iter().enumerate() {
        means[fe_info.obs_to_group[i]] += val;
    }

    // Divide by group size
    for (g, m) in means.iter_mut().enumerate() {
        *m /= fe_info.sizes[g] as f64;
    }
}

/// Subtract group means in-place.
///
/// # Arguments
///
/// * `data` - Data vector to modify in-place (length N)
/// * `fe_info` - Fixed effect group information
/// * `means` - Group means (length G)
fn subtract_group_means(data: &mut [f64], fe_info: &FixedEffectInfo, means: &[f64]) {
    for (i, val) in data.iter_mut().enumerate() {
        *val -= means[fe_info.obs_to_group[i]];
    }
}

/// Demean data by two fixed effects using alternating projections.
///
/// # Algorithm (Gauss-Seidel)
///
/// 1. Initialize: `data_demeaned = data`
/// 2. Loop until convergence:
///    a. Demean by FE1: `data_demeaned -= group_means_fe1`
///    b. Demean by FE2: `data_demeaned -= group_means_fe2`
///    c. Check convergence: `max|data_demeaned - data_prev| < tol`
///
/// Convergence is guaranteed for balanced panels; typically 5-20 iterations.
///
/// # Time Complexity
///
/// O(N × k) where k is the iteration count (typically 5-20).
///
/// # Space Complexity
///
/// O(N + G1 + G2) where G1, G2 are the FE group counts.
///
/// # Arguments
///
/// * `data` - Input data vector (length N)
/// * `fe1_info` - Fixed effect info for first dimension
/// * `fe2_info` - Fixed effect info for second dimension
/// * `tol` - Convergence tolerance (default 1e-8)
/// * `max_iter` - Maximum iterations (default 1000)
///
/// # Returns
///
/// * `Result<Vec<f64>, FixedEffectError>` - Demeaned data or convergence error
///
/// # Errors
///
/// * `FixedEffectError::ConvergenceFailure` if not converged after `max_iter` iterations
pub fn demean_two_way(
    data: &[f64],
    fe1_info: &FixedEffectInfo,
    fe2_info: &FixedEffectInfo,
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, FixedEffectError> {
    let n = data.len();
    let mut demeaned = data.to_vec();
    let mut prev = vec![0.0; n];

    // Pre-allocate mean buffers
    let mut means_fe1 = vec![0.0; fe1_info.n_groups];
    let mut means_fe2 = vec![0.0; fe2_info.n_groups];

    for _ in 0..max_iter {
        // Save previous values for convergence check
        prev.copy_from_slice(&demeaned);

        // Demean by FE1
        compute_group_means(&demeaned, fe1_info, &mut means_fe1);
        subtract_group_means(&mut demeaned, fe1_info, &means_fe1);

        // Demean by FE2
        compute_group_means(&demeaned, fe2_info, &mut means_fe2);
        subtract_group_means(&mut demeaned, fe2_info, &means_fe2);

        // Check convergence (L∞ norm)
        let max_delta = demeaned
            .iter()
            .zip(prev.iter())
            .map(|(&new, &old)| (new - old).abs())
            .fold(0.0f64, f64::max);

        if max_delta < tol {
            // Converged
            return Ok(demeaned);
        }
    }

    Err(FixedEffectError::ConvergenceFailure {
        iterations: max_iter,
    })
}

/// Validate fixed effects inputs.
///
/// Checks:
/// - Number of FE columns <= 2
/// - FE columns don't overlap with X columns
/// - FE columns don't equal Y column
///
/// # Arguments
///
/// * `fe_cols` - Fixed effect column names
/// * `x_cols` - Covariate column names
/// * `y_col` - Dependent variable column name
///
/// # Returns
///
/// * `Result<(), FixedEffectError>` - Ok or specific error
pub fn validate_fe_inputs(
    fe_cols: &[String],
    x_cols: &[String],
    y_col: &str,
) -> Result<(), FixedEffectError> {
    // Check: More than 2 FE columns → error
    if fe_cols.len() > 2 {
        return Err(FixedEffectError::TooManyColumns {
            count: fe_cols.len(),
        });
    }

    // Check each FE column for overlaps
    for fe_col in fe_cols {
        // Check: FE column overlaps with X columns → error
        if x_cols.contains(fe_col) {
            return Err(FixedEffectError::OverlapWithCovariate {
                name: fe_col.clone(),
            });
        }

        // Check: FE column equals Y column → error
        if fe_col == y_col {
            return Err(FixedEffectError::OverlapWithY {
                name: fe_col.clone(),
            });
        }
    }

    Ok(())
}

/// Check if any covariate has zero within-group variation after demeaning.
///
/// A covariate is collinear with FE if its demeaned values are all approximately zero.
///
/// # Arguments
///
/// * `x_demeaned` - Flat row-major demeaned X matrix (N × K)
/// * `n_rows` - Number of observations
/// * `n_cols` - Number of covariates
/// * `x_col_names` - Covariate column names
/// * `tol` - Variance tolerance (default 1e-10)
///
/// # Returns
///
/// * `Result<(), FixedEffectError>` - Ok or collinearity error
pub fn check_collinearity(
    x_demeaned: &[f64],
    n_rows: usize,
    n_cols: usize,
    x_col_names: &[String],
    tol: f64,
) -> Result<(), FixedEffectError> {
    for j in 0..n_cols {
        // Compute variance of column j using numerically stable algorithm
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n_rows {
            let val = x_demeaned[i * n_cols + j];
            sum += val;
            sum_sq += val * val;
        }
        let mean = sum / n_rows as f64;
        let variance = (sum_sq / n_rows as f64) - mean * mean;

        if variance < tol {
            return Err(FixedEffectError::Collinearity {
                covariate: x_col_names[j].clone(),
            });
        }
    }

    Ok(())
}

/// Compute within-R² (R² on demeaned data).
///
/// Formula:
/// ```text
/// within_R² = 1 - SS_res / SS_tot_demeaned
/// ```
///
/// where:
/// - `SS_res = Σ(y_demeaned - fitted_demeaned)²`
/// - `SS_tot_demeaned = Σ(y_demeaned - mean(y_demeaned))²`
///
/// Note: `mean(y_demeaned) ≈ 0` after demeaning, so `SS_tot_demeaned ≈ Σ(y_demeaned)²`
///
/// # Arguments
///
/// * `y_demeaned` - Demeaned outcome vector
/// * `fitted_demeaned` - Fitted values on demeaned data
///
/// # Returns
///
/// * `f64` - Within-R² value (can be negative if model fits poorly)
pub fn compute_within_r_squared(y_demeaned: &[f64], fitted_demeaned: &[f64]) -> f64 {
    let n = y_demeaned.len();

    if n == 0 {
        return 0.0;
    }

    // Compute SS_res
    let ss_res: f64 = y_demeaned
        .iter()
        .zip(fitted_demeaned)
        .map(|(&y, &f)| (y - f).powi(2))
        .sum();

    // Compute SS_tot on demeaned y
    let y_mean = y_demeaned.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y_demeaned.iter().map(|&y| (y - y_mean).powi(2)).sum();

    if ss_tot == 0.0 {
        // Edge case: no variation in demeaned y
        0.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Compute Mundlak terms (group means of covariates) for fixed effects approximation.
///
/// For each FE dimension d and each covariate j, computes:
///   mundlak[i, d * K + j] = mean(x[k, j] for k in group(i, d))
///
/// This is used for the Mundlak strategy in logistic regression with fixed effects,
/// where group means of covariates are added as additional regressors to approximate
/// the fixed effects without demeaning (which is not valid for nonlinear models).
///
/// # Arguments
/// * `x_flat` - Covariate matrix in row-major flat format (N × K)
/// * `n_rows` - Number of observations (N)
/// * `n_cols` - Number of covariates (K)
/// * `fe_infos` - Fixed effect info for each dimension (1 or 2 elements)
///
/// # Returns
/// * `Vec<f64>` - Mundlak terms in row-major flat format (N × K × D)
///   Layout: For each row i, for each FE dimension d, K covariate means
///
/// # Time Complexity
/// O(N × K × D) where D is the number of FE dimensions
///
/// # Space Complexity
/// O(N × K × D) for the output + O(max(G_1, G_2) × K) temporary for group sums
///
/// # Example
///
/// ```ignore
/// let x_flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4 obs × 2 cols
/// let group_ids = vec![0, 0, 1, 1];
/// let fe_info = build_fe_indices(&group_ids, "entity").unwrap();
/// let mundlak = compute_mundlak_terms(&x_flat, 4, 2, &[fe_info]);
/// // Group 0 means: (1+3)/2=2, (2+4)/2=3
/// // Group 1 means: (5+7)/2=6, (6+8)/2=7
/// // mundlak = [2, 3, 2, 3, 6, 7, 6, 7]
/// ```
pub fn compute_mundlak_terms(
    x_flat: &[f64],
    n_rows: usize,
    n_cols: usize,
    fe_infos: &[FixedEffectInfo],
) -> Vec<f64> {
    let n_fe = fe_infos.len();
    
    // Early return for empty input
    if n_rows == 0 || n_cols == 0 || n_fe == 0 {
        return vec![];
    }
    
    // Output: N × (K × D) in row-major order
    let mut mundlak = vec![0.0; n_rows * n_cols * n_fe];
    
    for (d, fe_info) in fe_infos.iter().enumerate() {
        // Allocate temporary for group sums (will be converted to means)
        let mut group_sums = vec![0.0; fe_info.n_groups * n_cols];
        
        // First pass: accumulate sums per group
        for i in 0..n_rows {
            let g = fe_info.obs_to_group[i];
            for j in 0..n_cols {
                group_sums[g * n_cols + j] += x_flat[i * n_cols + j];
            }
        }
        
        // Convert sums to means
        for g in 0..fe_info.n_groups {
            let size = fe_info.sizes[g] as f64;
            for j in 0..n_cols {
                group_sums[g * n_cols + j] /= size;
            }
        }
        
        // Second pass: map means back to observations
        for i in 0..n_rows {
            let g = fe_info.obs_to_group[i];
            for j in 0..n_cols {
                mundlak[i * (n_cols * n_fe) + d * n_cols + j] = group_sums[g * n_cols + j];
            }
        }
    }
    
    mundlak
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ========================================================================
    // build_fe_indices() tests
    // ========================================================================

    #[test]
    fn test_build_fe_indices_basic() {
        let group_ids = vec![0, 0, 1, 1, 2];
        let fe_info = build_fe_indices(&group_ids, "firm_id").unwrap();

        assert_eq!(fe_info.n_groups, 3);
        assert_eq!(fe_info.sizes, vec![2, 2, 1]);
        assert_eq!(fe_info.indices[0], vec![0, 1]);
        assert_eq!(fe_info.indices[1], vec![2, 3]);
        assert_eq!(fe_info.indices[2], vec![4]);
        assert_eq!(fe_info.obs_to_group, group_ids);
        assert_eq!(fe_info.name, "firm_id");
    }

    #[test]
    fn test_build_fe_indices_two_groups() {
        let group_ids = vec![0, 0, 1, 1];
        let fe_info = build_fe_indices(&group_ids, "year").unwrap();

        assert_eq!(fe_info.n_groups, 2);
        assert_eq!(fe_info.sizes, vec![2, 2]);
    }

    #[test]
    fn test_build_fe_indices_single_group_error() {
        let group_ids = vec![0, 0, 0, 0];
        let result = build_fe_indices(&group_ids, "region");

        assert!(result.is_err());
        match result.unwrap_err() {
            FixedEffectError::SingleUniqueValue { name } => assert_eq!(name, "region"),
            _ => panic!("Expected SingleUniqueValue error"),
        }
    }

    #[test]
    fn test_build_fe_indices_empty_error() {
        let group_ids: Vec<usize> = vec![];
        let result = build_fe_indices(&group_ids, "empty");

        assert!(result.is_err());
        match result.unwrap_err() {
            FixedEffectError::SingleUniqueValue { name } => assert_eq!(name, "empty"),
            _ => panic!("Expected SingleUniqueValue error"),
        }
    }

    #[test]
    fn test_obs_to_group_lookup() {
        let group_ids = vec![0, 0, 1, 1, 2];
        let fe_info = build_fe_indices(&group_ids, "test").unwrap();

        // Verify O(1) lookup
        assert_eq!(fe_info.obs_to_group[0], 0);
        assert_eq!(fe_info.obs_to_group[2], 1);
        assert_eq!(fe_info.obs_to_group[4], 2);
    }

    // ========================================================================
    // One-Way Demeaning tests
    // ========================================================================

    #[test]
    fn test_demean_one_way_simple() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let group_ids = vec![0, 0, 1, 1];
        let fe_info = build_fe_indices(&group_ids, "test").unwrap();

        let demeaned = demean_one_way(&data, &fe_info);

        // Group 0 mean = 1.5, group 1 mean = 3.5
        assert_relative_eq!(demeaned[0], -0.5, epsilon = 1e-10);
        assert_relative_eq!(demeaned[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(demeaned[2], -0.5, epsilon = 1e-10);
        assert_relative_eq!(demeaned[3], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_demean_one_way_unbalanced() {
        // Group 0: [1, 2, 3] mean = 2
        // Group 1: [4, 5] mean = 4.5
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_ids = vec![0, 0, 0, 1, 1];
        let fe_info = build_fe_indices(&group_ids, "test").unwrap();

        let demeaned = demean_one_way(&data, &fe_info);

        // Group 0 demeaned: [1-2, 2-2, 3-2] = [-1, 0, 1]
        assert_relative_eq!(demeaned[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(demeaned[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(demeaned[2], 1.0, epsilon = 1e-10);
        // Group 1 demeaned: [4-4.5, 5-4.5] = [-0.5, 0.5]
        assert_relative_eq!(demeaned[3], -0.5, epsilon = 1e-10);
        assert_relative_eq!(demeaned[4], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_demean_one_way_many_groups() {
        // 100 groups
        let n = 1000;
        let n_groups = 100;
        let group_ids: Vec<usize> = (0..n).map(|i| i % n_groups).collect();
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let fe_info = build_fe_indices(&group_ids, "large").unwrap();
        let demeaned = demean_one_way(&data, &fe_info);

        // Verify output length
        assert_eq!(demeaned.len(), n);

        // Verify each group sums to approximately zero
        for g in 0..n_groups {
            let group_sum: f64 = fe_info.indices[g].iter().map(|&i| demeaned[i]).sum();
            assert_relative_eq!(group_sum, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_demean_one_way_preserves_length() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_ids = vec![0, 0, 1, 1, 1];
        let fe_info = build_fe_indices(&group_ids, "test").unwrap();

        let demeaned = demean_one_way(&data, &fe_info);

        assert_eq!(demeaned.len(), data.len());
    }

    #[test]
    fn test_demean_one_way_sum_zero_property() {
        // Property test: demeaned values within each group sum to zero
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let group_ids = vec![0, 0, 1, 1, 2, 2];
        let fe_info = build_fe_indices(&group_ids, "test").unwrap();

        let demeaned = demean_one_way(&data, &fe_info);

        for g in 0..fe_info.n_groups {
            let group_sum: f64 = fe_info.indices[g].iter().map(|&i| demeaned[i]).sum();
            assert_relative_eq!(group_sum, 0.0, epsilon = 1e-10);
        }
    }

    // ========================================================================
    // Two-Way Demeaning tests
    // ========================================================================

    #[test]
    fn test_demean_two_way_balanced_panel() {
        // 2×2 balanced panel: 2 firms, 2 years
        // firm=0: obs 0, 1 (years 0, 1)
        // firm=1: obs 2, 3 (years 0, 1)
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let firm_ids = vec![0, 0, 1, 1];
        let year_ids = vec![0, 1, 0, 1];

        let fe1_info = build_fe_indices(&firm_ids, "firm").unwrap();
        let fe2_info = build_fe_indices(&year_ids, "year").unwrap();

        let demeaned = demean_two_way(&data, &fe1_info, &fe2_info, 1e-8, 1000).unwrap();

        // Verify output length
        assert_eq!(demeaned.len(), 4);

        // Verify convergence - sum within each dimension should be approximately zero
        // Firm dimension
        for g in 0..fe1_info.n_groups {
            let group_sum: f64 = fe1_info.indices[g].iter().map(|&i| demeaned[i]).sum();
            assert_relative_eq!(group_sum, 0.0, epsilon = 1e-6);
        }
        // Year dimension
        for g in 0..fe2_info.n_groups {
            let group_sum: f64 = fe2_info.indices[g].iter().map(|&i| demeaned[i]).sum();
            assert_relative_eq!(group_sum, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_demean_two_way_larger_panel() {
        // 10×100 panel: 10 firms, 100 years = 1000 obs
        let n_firms = 10;
        let n_years = 100;
        let n = n_firms * n_years;

        let mut firm_ids = Vec::with_capacity(n);
        let mut year_ids = Vec::with_capacity(n);
        let mut data = Vec::with_capacity(n);

        for firm in 0..n_firms {
            for year in 0..n_years {
                firm_ids.push(firm);
                year_ids.push(year);
                data.push((firm + year) as f64 + 0.1 * (firm * year) as f64);
            }
        }

        let fe1_info = build_fe_indices(&firm_ids, "firm").unwrap();
        let fe2_info = build_fe_indices(&year_ids, "year").unwrap();

        let demeaned = demean_two_way(&data, &fe1_info, &fe2_info, 1e-8, 1000).unwrap();

        assert_eq!(demeaned.len(), n);

        // Verify convergence
        for g in 0..fe1_info.n_groups {
            let group_sum: f64 = fe1_info.indices[g].iter().map(|&i| demeaned[i]).sum();
            assert_relative_eq!(group_sum, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_demean_two_way_unbalanced() {
        // Unbalanced panel
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let firm_ids = vec![0, 0, 0, 1, 1];
        let year_ids = vec![0, 1, 2, 1, 2];

        let fe1_info = build_fe_indices(&firm_ids, "firm").unwrap();
        let fe2_info = build_fe_indices(&year_ids, "year").unwrap();

        let result = demean_two_way(&data, &fe1_info, &fe2_info, 1e-8, 1000);
        assert!(result.is_ok());

        let demeaned = result.unwrap();
        assert_eq!(demeaned.len(), 5);
    }

    #[test]
    fn test_demean_two_way_converges_quickly() {
        // Balanced panel should converge in few iterations
        let n_firms = 5;
        let n_years = 10;
        let n = n_firms * n_years;

        let firm_ids: Vec<usize> = (0..n).map(|i| i / n_years).collect();
        let year_ids: Vec<usize> = (0..n).map(|i| i % n_years).collect();
        let data: Vec<f64> = (0..n).map(|i| (i * i) as f64).collect();

        let fe1_info = build_fe_indices(&firm_ids, "firm").unwrap();
        let fe2_info = build_fe_indices(&year_ids, "year").unwrap();

        // With a balanced panel, should converge well before 20 iterations
        let result = demean_two_way(&data, &fe1_info, &fe2_info, 1e-8, 20);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Validation tests
    // ========================================================================

    #[test]
    fn test_validate_fe_inputs_too_many_columns() {
        let fe_cols = vec!["firm".to_string(), "year".to_string(), "region".to_string()];
        let x_cols = vec!["x1".to_string()];
        let y_col = "y";

        let result = validate_fe_inputs(&fe_cols, &x_cols, y_col);
        assert!(result.is_err());
        match result.unwrap_err() {
            FixedEffectError::TooManyColumns { count } => assert_eq!(count, 3),
            _ => panic!("Expected TooManyColumns error"),
        }
    }

    #[test]
    fn test_validate_fe_inputs_overlap_with_x() {
        let fe_cols = vec!["x1".to_string()];
        let x_cols = vec!["x1".to_string(), "x2".to_string()];
        let y_col = "y";

        let result = validate_fe_inputs(&fe_cols, &x_cols, y_col);
        assert!(result.is_err());
        match result.unwrap_err() {
            FixedEffectError::OverlapWithCovariate { name } => assert_eq!(name, "x1"),
            _ => panic!("Expected OverlapWithCovariate error"),
        }
    }

    #[test]
    fn test_validate_fe_inputs_overlap_with_y() {
        let fe_cols = vec!["y".to_string()];
        let x_cols = vec!["x1".to_string()];
        let y_col = "y";

        let result = validate_fe_inputs(&fe_cols, &x_cols, y_col);
        assert!(result.is_err());
        match result.unwrap_err() {
            FixedEffectError::OverlapWithY { name } => assert_eq!(name, "y"),
            _ => panic!("Expected OverlapWithY error"),
        }
    }

    #[test]
    fn test_validate_fe_inputs_valid() {
        let fe_cols = vec!["firm".to_string(), "year".to_string()];
        let x_cols = vec!["x1".to_string(), "x2".to_string()];
        let y_col = "y";

        let result = validate_fe_inputs(&fe_cols, &x_cols, y_col);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Collinearity tests
    // ========================================================================

    #[test]
    fn test_check_collinearity_constant_column() {
        // Column with zero variance (all values are the same after demeaning)
        let x_demeaned = vec![0.0, 0.0, 0.0, 0.0];
        let n_rows = 4;
        let n_cols = 1;
        let x_col_names = vec!["x1".to_string()];

        let result = check_collinearity(&x_demeaned, n_rows, n_cols, &x_col_names, 1e-10);
        assert!(result.is_err());
        match result.unwrap_err() {
            FixedEffectError::Collinearity { covariate } => assert_eq!(covariate, "x1"),
            _ => panic!("Expected Collinearity error"),
        }
    }

    #[test]
    fn test_check_collinearity_near_constant() {
        // Column with very small variance
        let x_demeaned = vec![1e-12, -1e-12, 1e-12, -1e-12];
        let n_rows = 4;
        let n_cols = 1;
        let x_col_names = vec!["x1".to_string()];

        let result = check_collinearity(&x_demeaned, n_rows, n_cols, &x_col_names, 1e-10);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_collinearity_valid_variation() {
        // Column with legitimate variation
        let x_demeaned = vec![-1.0, 1.0, -1.0, 1.0];
        let n_rows = 4;
        let n_cols = 1;
        let x_col_names = vec!["x1".to_string()];

        let result = check_collinearity(&x_demeaned, n_rows, n_cols, &x_col_names, 1e-10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_collinearity_multiple_columns() {
        // Two columns: first valid, second constant
        let x_demeaned = vec![
            -1.0, 0.0, // row 0
            1.0, 0.0, // row 1
            -1.0, 0.0, // row 2
            1.0, 0.0, // row 3
        ];
        let n_rows = 4;
        let n_cols = 2;
        let x_col_names = vec!["x1".to_string(), "x2".to_string()];

        let result = check_collinearity(&x_demeaned, n_rows, n_cols, &x_col_names, 1e-10);
        assert!(result.is_err());
        match result.unwrap_err() {
            FixedEffectError::Collinearity { covariate } => assert_eq!(covariate, "x2"),
            _ => panic!("Expected Collinearity error"),
        }
    }

    // ========================================================================
    // Within R² tests
    // ========================================================================

    #[test]
    fn test_within_r_squared_perfect_fit() {
        let y_demeaned = vec![1.0, 2.0, 3.0, 4.0];
        let fitted_demeaned = vec![1.0, 2.0, 3.0, 4.0];

        let r2 = compute_within_r_squared(&y_demeaned, &fitted_demeaned);
        assert_relative_eq!(r2, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_within_r_squared_no_fit() {
        // Fitted values all zero, y has variation
        let y_demeaned = vec![-1.0, 1.0, -1.0, 1.0];
        let fitted_demeaned = vec![0.0, 0.0, 0.0, 0.0];

        let r2 = compute_within_r_squared(&y_demeaned, &fitted_demeaned);
        assert_relative_eq!(r2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_within_r_squared_partial_fit() {
        // SS_res = 4, SS_tot = 10, R² = 1 - 4/10 = 0.6
        let y_demeaned = vec![1.0, 2.0, 3.0, 4.0, 0.0]; // mean = 2, SS_tot = 10
        let fitted_demeaned = vec![1.0, 2.0, 3.0, 4.0, 2.0]; // residuals = [0, 0, 0, 0, -2]

        let r2 = compute_within_r_squared(&y_demeaned, &fitted_demeaned);
        assert_relative_eq!(r2, 0.6, epsilon = 1e-10);
    }

    #[test]
    fn test_within_r_squared_zero_total_variance() {
        // All y values are the same
        let y_demeaned = vec![0.0, 0.0, 0.0];
        let fitted_demeaned = vec![0.0, 0.0, 0.0];

        let r2 = compute_within_r_squared(&y_demeaned, &fitted_demeaned);
        assert_relative_eq!(r2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_within_r_squared_empty() {
        let y_demeaned: Vec<f64> = vec![];
        let fitted_demeaned: Vec<f64> = vec![];

        let r2 = compute_within_r_squared(&y_demeaned, &fitted_demeaned);
        assert_relative_eq!(r2, 0.0, epsilon = 1e-10);
    }

    // ========================================================================
    // Helper function tests
    // ========================================================================

    #[test]
    fn test_group_mean_helpers() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let group_ids = vec![0, 0, 1, 1];
        let fe_info = build_fe_indices(&group_ids, "test").unwrap();

        let mut means = vec![0.0; 2];
        compute_group_means(&data, &fe_info, &mut means);

        assert_relative_eq!(means[0], 1.5, epsilon = 1e-10);
        assert_relative_eq!(means[1], 3.5, epsilon = 1e-10);

        let mut data_copy = data.clone();
        subtract_group_means(&mut data_copy, &fe_info, &means);

        assert_relative_eq!(data_copy[0], -0.5, epsilon = 1e-10);
        assert_relative_eq!(data_copy[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(data_copy[2], -0.5, epsilon = 1e-10);
        assert_relative_eq!(data_copy[3], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_group_mean_helpers_single_obs_groups() {
        // Groups with single observations
        let data = vec![1.0, 2.0, 3.0];
        let group_ids = vec![0, 1, 2];
        let fe_info = build_fe_indices(&group_ids, "test").unwrap();

        let mut means = vec![0.0; 3];
        compute_group_means(&data, &fe_info, &mut means);

        // Each group mean equals its single value
        assert_relative_eq!(means[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(means[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(means[2], 3.0, epsilon = 1e-10);

        let mut data_copy = data.clone();
        subtract_group_means(&mut data_copy, &fe_info, &means);

        // Single-observation groups demean to zero
        assert_relative_eq!(data_copy[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(data_copy[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(data_copy[2], 0.0, epsilon = 1e-10);
    }

    // ========================================================================
    // Mundlak Terms Computation tests
    // ========================================================================

    mod mundlak_tests {
        use super::*;

        #[test]
        fn test_compute_mundlak_terms_one_way() {
            // 4 observations, 2 covariates, 2 groups
            // Group 0: obs 0, 1  |  Group 1: obs 2, 3
            let x_flat = vec![
                1.0, 2.0, // obs 0: x1=1, x2=2
                3.0, 4.0, // obs 1: x1=3, x2=4
                5.0, 6.0, // obs 2: x1=5, x2=6
                7.0, 8.0, // obs 3: x1=7, x2=8
            ];
            let n_rows = 4;
            let n_cols = 2;
            let group_ids = vec![0, 0, 1, 1];
            let fe_info = build_fe_indices(&group_ids, "entity").unwrap();

            let mundlak = compute_mundlak_terms(&x_flat, n_rows, n_cols, &[fe_info]);

            // Expected: Group 0 means: x1=(1+3)/2=2, x2=(2+4)/2=3
            //           Group 1 means: x1=(5+7)/2=6, x2=(6+8)/2=7
            // For obs 0 (group 0): [2.0, 3.0]
            // For obs 1 (group 0): [2.0, 3.0]
            // For obs 2 (group 1): [6.0, 7.0]
            // For obs 3 (group 1): [6.0, 7.0]
            assert_eq!(mundlak.len(), 8); // 4 obs × 2 cols × 1 FE
            assert_relative_eq!(mundlak[0], 2.0, epsilon = 1e-10); // obs 0, x1 mean
            assert_relative_eq!(mundlak[1], 3.0, epsilon = 1e-10); // obs 0, x2 mean
            assert_relative_eq!(mundlak[2], 2.0, epsilon = 1e-10); // obs 1, x1 mean
            assert_relative_eq!(mundlak[3], 3.0, epsilon = 1e-10); // obs 1, x2 mean
            assert_relative_eq!(mundlak[4], 6.0, epsilon = 1e-10); // obs 2, x1 mean
            assert_relative_eq!(mundlak[5], 7.0, epsilon = 1e-10); // obs 2, x2 mean
            assert_relative_eq!(mundlak[6], 6.0, epsilon = 1e-10); // obs 3, x1 mean
            assert_relative_eq!(mundlak[7], 7.0, epsilon = 1e-10); // obs 3, x2 mean
        }

        #[test]
        fn test_compute_mundlak_terms_two_way() {
            // 4 observations, 2 covariates, 2 FE dimensions
            // FE1 (entity): Group 0: obs 0, 1 | Group 1: obs 2, 3
            // FE2 (time):   Group 0: obs 0, 2 | Group 1: obs 1, 3
            let x_flat = vec![
                1.0, 2.0, // obs 0: x1=1, x2=2
                3.0, 4.0, // obs 1: x1=3, x2=4
                5.0, 6.0, // obs 2: x1=5, x2=6
                7.0, 8.0, // obs 3: x1=7, x2=8
            ];
            let n_rows = 4;
            let n_cols = 2;
            let entity_ids = vec![0, 0, 1, 1];
            let time_ids = vec![0, 1, 0, 1];

            let fe1_info = build_fe_indices(&entity_ids, "entity").unwrap();
            let fe2_info = build_fe_indices(&time_ids, "time").unwrap();

            let mundlak = compute_mundlak_terms(&x_flat, n_rows, n_cols, &[fe1_info, fe2_info]);

            // Output layout: For each row, [FE1 means K cols | FE2 means K cols]
            // FE1: Group 0 means: x1=(1+3)/2=2, x2=(2+4)/2=3
            //      Group 1 means: x1=(5+7)/2=6, x2=(6+8)/2=7
            // FE2: Group 0 means: x1=(1+5)/2=3, x2=(2+6)/2=4
            //      Group 1 means: x1=(3+7)/2=5, x2=(4+8)/2=6

            assert_eq!(mundlak.len(), 16); // 4 obs × 2 cols × 2 FE

            // Obs 0 (entity=0, time=0): [entity_mean_x1, entity_mean_x2, time_mean_x1, time_mean_x2]
            assert_relative_eq!(mundlak[0], 2.0, epsilon = 1e-10); // entity mean x1
            assert_relative_eq!(mundlak[1], 3.0, epsilon = 1e-10); // entity mean x2
            assert_relative_eq!(mundlak[2], 3.0, epsilon = 1e-10); // time mean x1
            assert_relative_eq!(mundlak[3], 4.0, epsilon = 1e-10); // time mean x2

            // Obs 1 (entity=0, time=1): [entity_mean_x1, entity_mean_x2, time_mean_x1, time_mean_x2]
            assert_relative_eq!(mundlak[4], 2.0, epsilon = 1e-10); // entity mean x1
            assert_relative_eq!(mundlak[5], 3.0, epsilon = 1e-10); // entity mean x2
            assert_relative_eq!(mundlak[6], 5.0, epsilon = 1e-10); // time mean x1
            assert_relative_eq!(mundlak[7], 6.0, epsilon = 1e-10); // time mean x2

            // Obs 2 (entity=1, time=0)
            assert_relative_eq!(mundlak[8], 6.0, epsilon = 1e-10);  // entity mean x1
            assert_relative_eq!(mundlak[9], 7.0, epsilon = 1e-10);  // entity mean x2
            assert_relative_eq!(mundlak[10], 3.0, epsilon = 1e-10); // time mean x1
            assert_relative_eq!(mundlak[11], 4.0, epsilon = 1e-10); // time mean x2

            // Obs 3 (entity=1, time=1)
            assert_relative_eq!(mundlak[12], 6.0, epsilon = 1e-10); // entity mean x1
            assert_relative_eq!(mundlak[13], 7.0, epsilon = 1e-10); // entity mean x2
            assert_relative_eq!(mundlak[14], 5.0, epsilon = 1e-10); // time mean x1
            assert_relative_eq!(mundlak[15], 6.0, epsilon = 1e-10); // time mean x2
        }

        #[test]
        fn test_compute_mundlak_terms_single_obs_group() {
            // Each observation is its own group - Mundlak term = observation value
            let x_flat = vec![
                1.0, 2.0, // obs 0
                3.0, 4.0, // obs 1
                5.0, 6.0, // obs 2
            ];
            let n_rows = 3;
            let n_cols = 2;
            let group_ids = vec![0, 1, 2]; // 3 singleton groups
            let fe_info = build_fe_indices(&group_ids, "entity").unwrap();

            let mundlak = compute_mundlak_terms(&x_flat, n_rows, n_cols, &[fe_info]);

            // Single obs groups: mean = observation value
            assert_eq!(mundlak.len(), 6);
            assert_relative_eq!(mundlak[0], 1.0, epsilon = 1e-10);
            assert_relative_eq!(mundlak[1], 2.0, epsilon = 1e-10);
            assert_relative_eq!(mundlak[2], 3.0, epsilon = 1e-10);
            assert_relative_eq!(mundlak[3], 4.0, epsilon = 1e-10);
            assert_relative_eq!(mundlak[4], 5.0, epsilon = 1e-10);
            assert_relative_eq!(mundlak[5], 6.0, epsilon = 1e-10);
        }

        #[test]
        fn test_compute_mundlak_terms_dimensions() {
            // Test that output dimensions are correct: N × K × D
            let n_rows = 100;
            let n_cols = 5;
            let x_flat: Vec<f64> = (0..n_rows * n_cols).map(|i| i as f64).collect();

            // One-way FE
            let group_ids: Vec<usize> = (0..n_rows).map(|i| i % 10).collect();
            let fe_info = build_fe_indices(&group_ids, "entity").unwrap();

            let mundlak_1way = compute_mundlak_terms(&x_flat, n_rows, n_cols, &[fe_info.clone()]);
            assert_eq!(mundlak_1way.len(), n_rows * n_cols * 1);

            // Two-way FE
            let time_ids: Vec<usize> = (0..n_rows).map(|i| i % 20).collect();
            let fe2_info = build_fe_indices(&time_ids, "time").unwrap();

            let mundlak_2way =
                compute_mundlak_terms(&x_flat, n_rows, n_cols, &[fe_info, fe2_info]);
            assert_eq!(mundlak_2way.len(), n_rows * n_cols * 2);
        }

        #[test]
        fn test_compute_mundlak_terms_unbalanced_groups() {
            // Unbalanced groups: Group 0 has 1 obs, Group 1 has 3 obs
            let x_flat = vec![
                1.0, 2.0, // obs 0 (group 0)
                3.0, 4.0, // obs 1 (group 1)
                5.0, 6.0, // obs 2 (group 1)
                7.0, 8.0, // obs 3 (group 1)
            ];
            let n_rows = 4;
            let n_cols = 2;
            let group_ids = vec![0, 1, 1, 1];
            let fe_info = build_fe_indices(&group_ids, "entity").unwrap();

            let mundlak = compute_mundlak_terms(&x_flat, n_rows, n_cols, &[fe_info]);

            // Group 0 mean (single obs): x1=1, x2=2
            // Group 1 mean: x1=(3+5+7)/3=5, x2=(4+6+8)/3=6
            assert_eq!(mundlak.len(), 8);
            assert_relative_eq!(mundlak[0], 1.0, epsilon = 1e-10); // obs 0
            assert_relative_eq!(mundlak[1], 2.0, epsilon = 1e-10);
            assert_relative_eq!(mundlak[2], 5.0, epsilon = 1e-10); // obs 1
            assert_relative_eq!(mundlak[3], 6.0, epsilon = 1e-10);
            assert_relative_eq!(mundlak[4], 5.0, epsilon = 1e-10); // obs 2
            assert_relative_eq!(mundlak[5], 6.0, epsilon = 1e-10);
            assert_relative_eq!(mundlak[6], 5.0, epsilon = 1e-10); // obs 3
            assert_relative_eq!(mundlak[7], 6.0, epsilon = 1e-10);
        }

        #[test]
        fn test_compute_mundlak_terms_empty_input() {
            let x_flat: Vec<f64> = vec![];
            let mundlak = compute_mundlak_terms(&x_flat, 0, 0, &[]);
            assert!(mundlak.is_empty());
        }

        #[test]
        fn test_compute_mundlak_terms_single_covariate() {
            // Single covariate
            let x_flat = vec![1.0, 2.0, 3.0, 4.0];
            let n_rows = 4;
            let n_cols = 1;
            let group_ids = vec![0, 0, 1, 1];
            let fe_info = build_fe_indices(&group_ids, "entity").unwrap();

            let mundlak = compute_mundlak_terms(&x_flat, n_rows, n_cols, &[fe_info]);

            // Group 0 mean: (1+2)/2 = 1.5
            // Group 1 mean: (3+4)/2 = 3.5
            assert_eq!(mundlak.len(), 4);
            assert_relative_eq!(mundlak[0], 1.5, epsilon = 1e-10);
            assert_relative_eq!(mundlak[1], 1.5, epsilon = 1e-10);
            assert_relative_eq!(mundlak[2], 3.5, epsilon = 1e-10);
            assert_relative_eq!(mundlak[3], 3.5, epsilon = 1e-10);
        }
    }
}
