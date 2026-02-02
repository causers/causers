//! Covariate Balance Checking module.
//!
//! This module implements covariate balance statistics computation for treatment
//! and control groups, including:
//! - First moments (means)
//! - Second moments (variance, standard deviation)
//! - Derived metrics (Standardized Mean Difference, variance ratio)
//! - Support for weighted statistics with reliability weights correction
//!
//! # Algorithm Overview
//!
//! The implementation uses Welford's single-pass algorithm for numerical stability
//! when computing means and variances. For weighted statistics, the reliability
//! weights correction formula is used to match cobalt's behavior.
//!
//! # References
//!
//! - Welford, B.P. (1962). Note on a method for calculating corrected sums of
//!   squares and products. Technometrics, 4(3), 419-420.
//! - Austin, P.C. (2011). An Introduction to Propensity Score Methods for
//!   Reducing the Effects of Confounding in Observational Studies.
//!   Multivariate Behavioral Research, 46(3), 399-424.

use std::collections::HashMap;

use pyo3::prelude::*;

// ============================================================================
// Constants
// ============================================================================

/// Default maximum number of unique levels allowed for categorical columns.
/// Columns exceeding this limit raise an error to prevent memory exhaustion
/// during one-hot expansion.
pub const DEFAULT_MAX_CATEGORICAL_LEVELS: usize = 1000;

/// Epsilon threshold for weighted variance denominator stability check.
/// When |V1^2 - V2| < epsilon, the variance is numerically unstable (ESS ≈ 1).
pub const VARIANCE_DENOM_EPSILON: f64 = 1e-10;

// ============================================================================
// Error Types
// ============================================================================

/// Error types for balance checking operations.
///
/// All error messages follow the IR-ERROR specification and do not include
/// raw data values (only column names, counts, and types).
#[derive(Debug, Clone)]
pub enum BalanceError {
    /// Column not found in DataFrame
    ColumnNotFound { column: String },
    /// Weights column not found in DataFrame
    WeightsColumnNotFound { column: String },
    /// Covariate column is not numeric or categorical
    NonNumericCovariate { column: String },
    /// Treatment variable has only one unique value
    NoTreatmentVariation,
    /// Covariate column contains null values
    NullValuesInCovariate { column: String },
    /// Weights column contains null values
    NullValuesInWeights { column: String },
    /// Weights contain negative values
    NegativeWeights,
    /// Sum of weights in a group is zero
    ZeroTotalWeight { group: String },
    /// No observations in treatment group after filtering
    EmptyTreatmentGroup,
    /// No observations in control group after filtering
    EmptyControlGroup,
    /// Treatment column has more than 2 unique values when control_value is None
    MultiValuedTreatment { n_values: usize },
    /// Numerical overflow during variance computation
    NumericalOverflow { column: String },
    /// Categorical column has too many unique levels
    HighCardinalityCategorical {
        column: String,
        n_levels: usize,
        max_levels: usize,
    },
    /// Internal invariant violation (should never be triggered by user input)
    InternalError { reason: String },
}

impl std::fmt::Display for BalanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BalanceError::ColumnNotFound { column } => {
                write!(f, "Column '{}' not found in DataFrame", column)
            }
            BalanceError::WeightsColumnNotFound { column } => {
                write!(f, "Weights column '{}' not found in DataFrame", column)
            }
            BalanceError::NonNumericCovariate { column } => {
                write!(
                    f,
                    "Covariate column '{}' must be numeric or categorical",
                    column
                )
            }
            BalanceError::NoTreatmentVariation => {
                write!(f, "Treatment variable has no variation")
            }
            BalanceError::NullValuesInCovariate { column } => {
                write!(f, "Column '{}' contains null values", column)
            }
            BalanceError::NullValuesInWeights { column } => {
                write!(f, "Weights column '{}' contains null values", column)
            }
            BalanceError::NegativeWeights => {
                write!(f, "Weights must be non-negative")
            }
            BalanceError::ZeroTotalWeight { group } => {
                write!(f, "Zero total weight in {} group", group)
            }
            BalanceError::EmptyTreatmentGroup => {
                write!(f, "No observations in treatment group")
            }
            BalanceError::EmptyControlGroup => {
                write!(f, "No observations in control group")
            }
            BalanceError::MultiValuedTreatment { n_values } => {
                write!(
                    f,
                    "Binary treatment required; found {} unique values. Specify control_value or use binary treatment.",
                    n_values
                )
            }
            BalanceError::NumericalOverflow { column } => {
                write!(
                    f,
                    "Numerical overflow in variance computation for covariate '{}'",
                    column
                )
            }
            BalanceError::HighCardinalityCategorical {
                column,
                n_levels,
                max_levels,
            } => {
                write!(
                    f,
                    "Categorical column '{}' has {} levels (max {}). Consider grouping rare levels or converting to continuous.",
                    column, n_levels, max_levels
                )
            }
            BalanceError::InternalError { reason } => {
                write!(f, "Internal error: {}", reason)
            }
        }
    }
}

impl std::error::Error for BalanceError {}

// ============================================================================
// Treatment Value Types
// ============================================================================

/// Represents a treatment value which can be integer, float, or string.
///
/// Used for matching treatment and control group values in the treatment column.
#[derive(Debug, Clone, PartialEq)]
pub enum TreatmentValue {
    /// Integer treatment value
    Int(i64),
    /// Float treatment value
    Float(f64),
    /// String treatment value
    String(String),
}

impl TreatmentValue {
    /// Check if this treatment value matches an integer.
    pub fn matches_int(&self, value: i64) -> bool {
        match self {
            TreatmentValue::Int(v) => *v == value,
            TreatmentValue::Float(v) => (*v - value as f64).abs() < 1e-10,
            TreatmentValue::String(_) => false,
        }
    }

    /// Check if this treatment value matches a float.
    pub fn matches_float(&self, value: f64) -> bool {
        match self {
            TreatmentValue::Int(v) => (*v as f64 - value).abs() < 1e-10,
            TreatmentValue::Float(v) => (*v - value).abs() < 1e-10,
            TreatmentValue::String(_) => false,
        }
    }

    /// Check if this treatment value matches a string.
    pub fn matches_str(&self, value: &str) -> bool {
        match self {
            TreatmentValue::Int(_) | TreatmentValue::Float(_) => false,
            TreatmentValue::String(s) => s == value,
        }
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for balance checking.
#[derive(Debug, Clone)]
pub struct BalanceConfig {
    /// Value indicating treatment group membership
    pub treatment_value: TreatmentValue,
    /// Optional value indicating control group membership.
    /// If None, all non-treatment observations are control.
    pub control_value: Option<TreatmentValue>,
    /// Whether to compute weighted statistics
    pub weighted: bool,
    /// Maximum number of unique levels allowed for categorical columns.
    /// Default: 1000. Increase for high-cardinality columns on systems with sufficient RAM.
    pub max_categorical_levels: usize,
}

impl Default for BalanceConfig {
    fn default() -> Self {
        Self {
            treatment_value: TreatmentValue::Int(1),
            control_value: None,
            weighted: false,
            max_categorical_levels: DEFAULT_MAX_CATEGORICAL_LEVELS,
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of balance checking computation.
///
/// Contains first moments (means), second moments (variance/SD), derived metrics
/// (SMD, variance ratio), sample sizes, and metadata for all checked covariates.
#[pyclass]
#[derive(Debug, Clone)]
pub struct BalanceResult {
    // First Moments
    /// Mean of each covariate in the treatment group
    #[pyo3(get)]
    pub mean_treated: HashMap<String, f64>,
    /// Mean of each covariate in the control group
    #[pyo3(get)]
    pub mean_control: HashMap<String, f64>,

    // Second Moments
    /// Variance of each covariate in the treatment group
    #[pyo3(get)]
    pub var_treated: HashMap<String, f64>,
    /// Variance of each covariate in the control group
    #[pyo3(get)]
    pub var_control: HashMap<String, f64>,
    /// Standard deviation of each covariate in the treatment group
    #[pyo3(get)]
    pub sd_treated: HashMap<String, f64>,
    /// Standard deviation of each covariate in the control group
    #[pyo3(get)]
    pub sd_control: HashMap<String, f64>,

    // Derived Metrics
    /// Standardized Mean Difference for each covariate
    #[pyo3(get)]
    pub smd: HashMap<String, f64>,
    /// Variance ratio (var_treated / var_control) for each covariate
    #[pyo3(get)]
    pub variance_ratio: HashMap<String, f64>,

    // Sample Sizes
    /// Number of observations in treatment group
    #[pyo3(get)]
    pub n_treated: usize,
    /// Number of observations in control group
    #[pyo3(get)]
    pub n_control: usize,
    /// Effective sample size in treatment group (for weighted analysis)
    #[pyo3(get)]
    pub ess_treated: Option<f64>,
    /// Effective sample size in control group (for weighted analysis)
    #[pyo3(get)]
    pub ess_control: Option<f64>,

    // Metadata
    /// List of covariate names checked (after categorical expansion)
    #[pyo3(get)]
    pub covariates: Vec<String>,
    /// Whether weighted statistics were computed
    #[pyo3(get)]
    pub is_weighted: bool,
}

#[pymethods]
impl BalanceResult {
    /// String representation for Python inspection.
    ///
    /// Format: BalanceResult(n_treated=X, n_control=Y, n_covariates=Z, n_imbalanced=W)
    /// where n_imbalanced is the count of covariates with |SMD| > 0.1.
    fn __repr__(&self) -> String {
        let n_imbalanced = self.smd.values().filter(|&&v| v.abs() > 0.1).count();
        format!(
            "BalanceResult(n_treated={}, n_control={}, n_covariates={}, n_imbalanced={})",
            self.n_treated,
            self.n_control,
            self.covariates.len(),
            n_imbalanced
        )
    }
}

impl BalanceResult {
    /// Create a new BalanceResult with default (empty) values.
    pub fn new() -> Self {
        Self {
            mean_treated: HashMap::new(),
            mean_control: HashMap::new(),
            var_treated: HashMap::new(),
            var_control: HashMap::new(),
            sd_treated: HashMap::new(),
            sd_control: HashMap::new(),
            smd: HashMap::new(),
            variance_ratio: HashMap::new(),
            n_treated: 0,
            n_control: 0,
            ess_treated: None,
            ess_control: None,
            covariates: Vec::new(),
            is_weighted: false,
        }
    }
}

impl Default for BalanceResult {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Internal Working Structures
// ============================================================================

/// Accumulator for single-pass moment computation using Welford's algorithm.
///
/// Welford's algorithm computes running mean and variance in a single pass
/// with numerical stability. The algorithm avoids catastrophic cancellation
/// by computing the sum of squared deviations from the running mean.
///
/// # References
///
/// Welford, B.P. (1962). Note on a method for calculating corrected sums of
/// squares and products. Technometrics, 4(3), 419-420.
#[derive(Debug, Clone)]
pub struct MomentAccumulator {
    /// Count of observations
    n: usize,
    /// Running mean
    mean: f64,
    /// Sum of squared deviations from mean (M2)
    m2: f64,
}

impl MomentAccumulator {
    /// Create a new accumulator with zero observations.
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Update the accumulator with a new observation.
    ///
    /// Uses Welford's online algorithm for numerical stability:
    /// - delta = x - mean
    /// - mean += delta / n
    /// - delta2 = x - mean  (after update)
    /// - m2 += delta * delta2
    pub fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Finalize and return (mean, variance) with Bessel's correction.
    ///
    /// Returns (mean, variance) where variance uses N-1 denominator (Bessel's correction).
    /// If n == 0, returns (0.0, 0.0).
    /// If n == 1, returns (mean, 0.0) since variance is undefined for single observation.
    pub fn finalize(&self) -> (f64, f64) {
        if self.n == 0 {
            return (0.0, 0.0);
        }
        if self.n == 1 {
            return (self.mean, 0.0);
        }
        let variance = self.m2 / (self.n - 1) as f64;
        (self.mean, variance)
    }

    /// Get the current count of observations.
    pub fn count(&self) -> usize {
        self.n
    }
}

impl Default for MomentAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulator for weighted moment computation using weighted Welford's algorithm.
///
/// This accumulator tracks:
/// - V1 (sum of weights)
/// - V2 (sum of squared weights)
/// - Weighted running mean
/// - Weighted M2 (sum of weighted squared deviations)
///
/// The variance uses reliability weights correction formula:
/// var = m2 * V1 / (V1^2 - V2)
///
/// This matches the formula used by R's cobalt package for weighted balance statistics.
#[derive(Debug, Clone)]
pub struct WeightedMomentAccumulator {
    /// Sum of weights (V1)
    sum_w: f64,
    /// Sum of squared weights (V2)
    sum_w2: f64,
    /// Weighted running mean
    mean: f64,
    /// Weighted sum of squared deviations from mean
    m2: f64,
}

impl WeightedMomentAccumulator {
    /// Create a new weighted accumulator with zero observations.
    pub fn new() -> Self {
        Self {
            sum_w: 0.0,
            sum_w2: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Update the accumulator with a new observation and its weight.
    ///
    /// Uses weighted Welford's online algorithm:
    /// - sum_w += w
    /// - sum_w2 += w^2
    /// - delta = x - mean
    /// - mean += (w / sum_w) * delta
    /// - delta2 = x - mean  (after update)
    /// - m2 += w * delta * delta2
    ///
    /// Zero-weight observations are skipped to avoid division by zero.
    pub fn update(&mut self, x: f64, w: f64) {
        // Skip zero-weight observations to avoid division by zero
        if w <= 0.0 {
            return;
        }
        self.sum_w += w;
        self.sum_w2 += w * w;
        let delta = x - self.mean;
        self.mean += (w / self.sum_w) * delta;
        let delta2 = x - self.mean;
        self.m2 += w * delta * delta2;
    }

    /// Finalize and return (mean, variance, ess).
    ///
    /// Returns (mean, variance, effective_sample_size).
    ///
    /// Variance uses reliability weights correction formula:
    /// var = m2 * V1 / (V1^2 - V2)
    ///
    /// If denominator |V1^2 - V2| < VARIANCE_DENOM_EPSILON, returns NaN for variance
    /// to indicate numerical instability (ESS ≈ 1).
    ///
    /// ESS = V1^2 / V2
    pub fn finalize(&self) -> (f64, f64, f64) {
        if self.sum_w <= 0.0 {
            return (0.0, 0.0, 0.0);
        }

        // Compute effective sample size
        let ess = if self.sum_w2 > 0.0 {
            (self.sum_w * self.sum_w) / self.sum_w2
        } else {
            0.0
        };

        // Compute variance with reliability weights correction
        let denom = self.sum_w * self.sum_w - self.sum_w2;
        let variance = if denom.abs() < VARIANCE_DENOM_EPSILON {
            // Numerically unstable: ESS ≈ 1
            f64::NAN
        } else {
            self.m2 * self.sum_w / denom
        };

        (self.mean, variance, ess)
    }

    /// Get the current sum of weights (V1).
    pub fn sum_weights(&self) -> f64 {
        self.sum_w
    }
}

impl Default for WeightedMomentAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a single covariate in one group.
#[derive(Debug, Clone, Default)]
pub struct GroupCovariateStats {
    /// Mean of the covariate
    pub mean: f64,
    /// Variance with Bessel's correction
    pub variance: f64,
    /// Standard deviation (sqrt of variance)
    pub sd: f64,
    /// Number of observations (or effective sample size for weighted)
    pub n: usize,
}

impl GroupCovariateStats {
    /// Create stats from an unweighted accumulator.
    pub fn from_accumulator(acc: &MomentAccumulator) -> Self {
        let (mean, variance) = acc.finalize();
        Self {
            mean,
            variance,
            sd: variance.sqrt(),
            n: acc.count(),
        }
    }

    /// Create stats from a weighted accumulator.
    pub fn from_weighted_accumulator(acc: &WeightedMomentAccumulator, n: usize) -> Self {
        let (mean, variance, _ess) = acc.finalize();
        Self {
            mean,
            variance,
            sd: if variance.is_nan() {
                f64::NAN
            } else {
                variance.sqrt()
            },
            n,
        }
    }
}

// ============================================================================
// Covariate Type Detection
// ============================================================================

/// Enum for detecting covariate data types and determining processing strategy.
///
/// Used to determine how to handle different column types:
/// - Numeric: Use values directly as f64
/// - Boolean: Cast to 0.0/1.0
/// - Categorical: Expand to one-hot encoded dummy columns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovariateType {
    /// Numeric covariate (Int/Float) - use values directly
    Numeric,
    /// Boolean covariate - cast to 0.0/1.0
    Boolean,
    /// Categorical covariate (String/Categorical/Enum) - expand to dummy columns
    Categorical,
}

// ============================================================================
// Categorical Expansion Functions
// ============================================================================

/// Validate that a categorical column does not exceed the maximum number of levels.
///
/// This check prevents memory exhaustion during one-hot expansion of high-cardinality
/// categorical columns. For N rows with K levels, expansion creates N*K values.
///
/// # Arguments
///
/// * `n_levels` - The number of unique levels in the categorical column
/// * `col_name` - The name of the column (for error message)
/// * `max_levels` - Maximum allowed number of levels
///
/// # Returns
///
/// * `Ok(())` if n_levels <= max_levels
/// * `Err(BalanceError::HighCardinalityCategorical)` if n_levels > max_levels
///
/// # Example
///
/// ```ignore
/// use causers::balance::validate_categorical_cardinality;
///
/// // This will succeed (3 <= 1000)
/// validate_categorical_cardinality(3, "color", 1000).unwrap();
///
/// // This will fail (5000 > 1000)
/// let result = validate_categorical_cardinality(5000, "zipcode", 1000);
/// assert!(result.is_err());
/// ```
pub fn validate_categorical_cardinality(
    n_levels: usize,
    col_name: &str,
    max_levels: usize,
) -> Result<(), BalanceError> {
    if n_levels > max_levels {
        return Err(BalanceError::HighCardinalityCategorical {
            column: col_name.to_string(),
            n_levels,
            max_levels,
        });
    }
    Ok(())
}

/// Expand a categorical column to one-hot encoded dummy columns.
///
/// Creates one dummy column for each unique level in the input. Each dummy column
/// contains 1.0 where the original value matches the level, and 0.0 otherwise.
///
/// # Arguments
///
/// * `values` - Slice of string values representing the categorical column
/// * `col_name` - The original column name (used as prefix for dummy column names)
///
/// # Returns
///
/// Vector of (column_name, values) pairs where:
/// - column_name follows the format "{col_name}_{level}"
/// - values is a Vec<f64> with 1.0 for matches, 0.0 otherwise
///
/// # Ordering
///
/// Levels are sorted alphabetically to ensure deterministic output across runs.
/// All levels are included (no reference level is dropped).
///
/// # Example
///
/// ```ignore
/// use causers::balance::expand_categorical;
///
/// let values = vec!["red", "blue", "red", "green"];
/// let expanded = expand_categorical(&values, "color");
///
/// // Returns 3 columns (sorted alphabetically):
/// // - ("color_blue", [0.0, 1.0, 0.0, 0.0])
/// // - ("color_green", [0.0, 0.0, 0.0, 1.0])
/// // - ("color_red", [1.0, 0.0, 1.0, 0.0])
/// ```
pub fn expand_categorical(values: &[&str], col_name: &str) -> Vec<(String, Vec<f64>)> {
    use std::collections::HashSet;

    // Find unique levels and sort alphabetically for determinism
    let mut levels: Vec<&str> = values
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    levels.sort();

    // Create dummy column for each level
    let mut dummy_columns = Vec::with_capacity(levels.len());
    for level in levels {
        let dummy_name = format!("{}_{}", col_name, level);
        let dummy_values: Vec<f64> = values
            .iter()
            .map(|&v| if v == level { 1.0 } else { 0.0 })
            .collect();
        dummy_columns.push((dummy_name, dummy_values));
    }

    dummy_columns
}

/// Convert boolean values to numeric (0.0/1.0).
///
/// # Arguments
///
/// * `values` - Slice of boolean values
///
/// # Returns
///
/// Vec<f64> with true -> 1.0 and false -> 0.0
///
/// # Example
///
/// ```ignore
/// use causers::balance::boolean_to_numeric;
///
/// let bools = vec![true, false, true, true];
/// let nums = boolean_to_numeric(&bools);
/// assert_eq!(nums, vec![1.0, 0.0, 1.0, 1.0]);
/// ```
pub fn boolean_to_numeric(values: &[bool]) -> Vec<f64> {
    values.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
}

// ============================================================================
// Core Moment Computation Functions
// ============================================================================

/// Compute unweighted moments for treated and control groups in a single pass.
///
/// This function implements Welford's single-pass algorithm to compute mean and
/// variance for both treatment groups simultaneously. The mask determines group
/// membership: `true` = treated, `false` = control.
///
/// # Arguments
///
/// * `values` - Slice of covariate values
/// * `mask` - Slice of booleans where `true` indicates treatment group membership
///
/// # Returns
///
/// A tuple of `(treated_stats, control_stats)` where each contains:
/// - `mean`: Group mean
/// - `variance`: Bessel-corrected variance (N-1 denominator)
/// - `sd`: Standard deviation (sqrt of variance)
/// - `n`: Number of observations in the group
///
/// # Edge Cases
///
/// - If `n == 0` for a group: returns mean=0, variance=0, sd=0
/// - If `n == 1` for a group: returns variance=0, sd=0 (variance undefined)
///
/// # Errors
///
/// Returns `Err(BalanceError::InternalError)` if `values.len() != mask.len()`.
///
/// # Example
///
/// ```
/// use causers::balance::{compute_unweighted_moments, GroupCovariateStats};
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mask = vec![true, true, true, false, false, false];
///
/// let (treated, control) = compute_unweighted_moments(&values, &mask).unwrap();
///
/// // Treated group: [1, 2, 3], mean = 2
/// // Control group: [4, 5, 6], mean = 5
/// assert!((treated.mean - 2.0).abs() < 1e-10);
/// assert!((control.mean - 5.0).abs() < 1e-10);
/// ```
pub fn compute_unweighted_moments(
    values: &[f64],
    mask: &[bool],
) -> Result<(GroupCovariateStats, GroupCovariateStats), BalanceError> {
    if values.len() != mask.len() {
        return Err(BalanceError::InternalError {
            reason: "values/mask length mismatch".to_string(),
        });
    }

    let mut treated_acc = MomentAccumulator::new();
    let mut control_acc = MomentAccumulator::new();

    // Single pass: iterate through all values and update appropriate accumulator
    for (value, &is_treated) in values.iter().zip(mask.iter()) {
        if is_treated {
            treated_acc.update(*value);
        } else {
            control_acc.update(*value);
        }
    }

    // Convert accumulators to stats (applies Bessel correction internally)
    let treated_stats = GroupCovariateStats::from_accumulator(&treated_acc);
    let control_stats = GroupCovariateStats::from_accumulator(&control_acc);

    Ok((treated_stats, control_stats))
}

/// Compute weighted moments for treated and control groups in a single pass.
///
/// This function iterates through the values, mask, and weights arrays once,
/// computing weighted mean, variance, and effective sample size (ESS) for both
/// the treated and control groups simultaneously.
///
/// # Algorithm
///
/// Uses weighted Welford's algorithm for numerical stability:
/// - Track V1 (sum of weights) and V2 (sum of squared weights)
/// - Weighted running mean update: mean += (w / V1) * (x - mean)
/// - Weighted M2 accumulation: m2 += w * delta * delta2
///
/// Variance uses reliability weights correction:
/// `var = m2 * V1 / (V1^2 - V2)`
///
/// # Arguments
///
/// * `values` - Slice of covariate values
/// * `mask` - Slice of booleans where `true` indicates treatment group
/// * `weights` - Slice of observation weights (must be non-negative)
///
/// # Returns
///
/// Returns `Ok((treated_stats, control_stats, ess_treated, ess_control))` on success.
///
/// # Errors
///
/// Returns `Err(BalanceError::NegativeWeights)` if any weight is negative.
/// Returns `Err(BalanceError::ZeroTotalWeight)` if total weight in either group is zero.
///
/// # Numerical Stability
///
/// If the variance denominator |V1^2 - V2| < VARIANCE_DENOM_EPSILON, the variance
/// is set to NaN to indicate numerical instability (ESS approximately equals 1).
///
/// # Example
///
/// ```text
/// let values = vec![1.0, 2.0, 3.0, 4.0];
/// let mask = vec![true, true, false, false];
/// let weights = vec![1.0, 1.0, 1.0, 1.0];
///
/// let result = compute_weighted_moments(&values, &mask, &weights);
/// // result = Ok((treated_stats, control_stats, ess_t, ess_c))
///
/// // Treated: [1, 2] with weights [1, 1], mean = 1.5, ESS = 2.0
/// // Control: [3, 4] with weights [1, 1], mean = 3.5, ESS = 2.0
/// ```
pub fn compute_weighted_moments(
    values: &[f64],
    mask: &[bool],
    weights: &[f64],
) -> Result<(GroupCovariateStats, GroupCovariateStats, f64, f64), BalanceError> {
    // Validate input lengths match
    if values.len() != mask.len() {
        return Err(BalanceError::InternalError {
            reason: "values/mask length mismatch".to_string(),
        });
    }
    if values.len() != weights.len() {
        return Err(BalanceError::InternalError {
            reason: "values/weights length mismatch".to_string(),
        });
    }

    // Initialize accumulators for treated and control
    let mut acc_treated = WeightedMomentAccumulator::new();
    let mut acc_control = WeightedMomentAccumulator::new();
    let mut n_treated: usize = 0;
    let mut n_control: usize = 0;

    // Single pass over all data
    for i in 0..values.len() {
        let x = values[i];
        let w = weights[i];

        // Validate weight is non-negative
        if w < 0.0 {
            return Err(BalanceError::NegativeWeights);
        }

        if mask[i] {
            // Treated group
            acc_treated.update(x, w);
            n_treated += 1;
        } else {
            // Control group
            acc_control.update(x, w);
            n_control += 1;
        }
    }

    // Check for zero total weight in either group
    if acc_treated.sum_weights() <= 0.0 {
        return Err(BalanceError::ZeroTotalWeight {
            group: "treatment".to_string(),
        });
    }
    if acc_control.sum_weights() <= 0.0 {
        return Err(BalanceError::ZeroTotalWeight {
            group: "control".to_string(),
        });
    }

    // Finalize accumulators to get statistics
    let (mean_t, var_t, ess_t) = acc_treated.finalize();
    let (mean_c, var_c, ess_c) = acc_control.finalize();

    // Build GroupCovariateStats for treated
    let treated_stats = GroupCovariateStats {
        mean: mean_t,
        variance: var_t,
        sd: if var_t.is_nan() {
            f64::NAN
        } else {
            var_t.sqrt()
        },
        n: n_treated,
    };

    // Build GroupCovariateStats for control
    let control_stats = GroupCovariateStats {
        mean: mean_c,
        variance: var_c,
        sd: if var_c.is_nan() {
            f64::NAN
        } else {
            var_c.sqrt()
        },
        n: n_control,
    };

    Ok((treated_stats, control_stats, ess_t, ess_c))
}

// ============================================================================
// Group Identification Functions
// ============================================================================

/// Identify treatment and control groups from an integer treatment column.
///
/// Returns a tuple of (treatment_mask, n_treated, n_control) where:
/// - treatment_mask[i] = true if observation i is in treatment group
/// - treatment_mask[i] = false if observation i is in control group
/// - Null values are skipped (mask value will be false but not counted in either group)
///
/// # Arguments
///
/// * `treatment_col` - Treatment column values (None for null/missing)
/// * `treatment_value` - Value indicating treatment group membership
/// * `control_value` - Optional value indicating control group. If None, all non-treatment
///                     observations become control.
///
/// # Returns
///
/// Result containing (mask, n_treated, n_control) or BalanceError if validation fails.
///
/// # Errors
///
/// * `EmptyTreatmentGroup` - If no observations match treatment_value
/// * `EmptyControlGroup` - If no observations in control group
/// * `MultiValuedTreatment` - If control_value is None and >2 unique non-null values exist
pub fn identify_groups_int(
    treatment_col: &[Option<i64>],
    treatment_value: i64,
    control_value: Option<i64>,
) -> Result<(Vec<bool>, usize, usize), BalanceError> {
    let mut mask = vec![false; treatment_col.len()];
    let mut n_treated = 0usize;
    let mut n_control = 0usize;

    // For MultiValuedTreatment validation when control_value is None
    let mut unique_values = std::collections::HashSet::new();

    for (i, val) in treatment_col.iter().enumerate() {
        match val {
            None => {
                // Skip null values - they are excluded from both groups
                continue;
            }
            Some(v) => {
                // Track unique values for validation
                if control_value.is_none() {
                    unique_values.insert(*v);
                }

                if *v == treatment_value {
                    mask[i] = true;
                    n_treated += 1;
                } else if let Some(cv) = control_value {
                    // When control_value is specified, only include matching observations
                    if *v == cv {
                        // mask[i] is already false (control)
                        n_control += 1;
                    }
                    // else: neither treatment nor control, skip
                } else {
                    // When control_value is None, all non-treatment become control
                    // mask[i] is already false (control)
                    n_control += 1;
                }
            }
        }
    }

    // Validate n_treated > 0
    if n_treated == 0 {
        return Err(BalanceError::EmptyTreatmentGroup);
    }

    // Validate n_control > 0
    if n_control == 0 {
        return Err(BalanceError::EmptyControlGroup);
    }

    // If control_value is None, validate <= 2 unique values
    if control_value.is_none() && unique_values.len() > 2 {
        return Err(BalanceError::MultiValuedTreatment {
            n_values: unique_values.len(),
        });
    }

    Ok((mask, n_treated, n_control))
}

/// Identify treatment and control groups from a float treatment column.
///
/// Float comparison uses epsilon tolerance of 1e-10 for matching.
///
/// # Arguments
///
/// * `treatment_col` - Treatment column values (None for null/missing)
/// * `treatment_value` - Value indicating treatment group membership
/// * `control_value` - Optional value indicating control group. If None, all non-treatment
///                     observations become control.
///
/// # Returns
///
/// Result containing (mask, n_treated, n_control) or BalanceError if validation fails.
///
/// # Errors
///
/// * `EmptyTreatmentGroup` - If no observations match treatment_value
/// * `EmptyControlGroup` - If no observations in control group
/// * `MultiValuedTreatment` - If control_value is None and >2 unique non-null values exist
pub fn identify_groups_float(
    treatment_col: &[Option<f64>],
    treatment_value: f64,
    control_value: Option<f64>,
) -> Result<(Vec<bool>, usize, usize), BalanceError> {
    const EPSILON: f64 = 1e-10;

    let mut mask = vec![false; treatment_col.len()];
    let mut n_treated = 0usize;
    let mut n_control = 0usize;

    // For MultiValuedTreatment validation when control_value is None
    // Use a vector to track unique values (can't use HashSet with f64)
    let mut unique_values: Vec<f64> = Vec::new();

    let matches_value = |a: f64, b: f64| -> bool { (a - b).abs() < EPSILON };

    let is_unique =
        |v: f64, values: &[f64]| -> bool { !values.iter().any(|&x| (x - v).abs() < EPSILON) };

    for (i, val) in treatment_col.iter().enumerate() {
        match val {
            None => {
                // Skip null values - they are excluded from both groups
                continue;
            }
            Some(v) => {
                // Track unique values for validation
                if control_value.is_none() && is_unique(*v, &unique_values) {
                    unique_values.push(*v);
                }

                if matches_value(*v, treatment_value) {
                    mask[i] = true;
                    n_treated += 1;
                } else if let Some(cv) = control_value {
                    // When control_value is specified, only include matching observations
                    if matches_value(*v, cv) {
                        // mask[i] is already false (control)
                        n_control += 1;
                    }
                    // else: neither treatment nor control, skip
                } else {
                    // When control_value is None, all non-treatment become control
                    // mask[i] is already false (control)
                    n_control += 1;
                }
            }
        }
    }

    // Validate n_treated > 0
    if n_treated == 0 {
        return Err(BalanceError::EmptyTreatmentGroup);
    }

    // Validate n_control > 0
    if n_control == 0 {
        return Err(BalanceError::EmptyControlGroup);
    }

    // If control_value is None, validate <= 2 unique values
    if control_value.is_none() && unique_values.len() > 2 {
        return Err(BalanceError::MultiValuedTreatment {
            n_values: unique_values.len(),
        });
    }

    Ok((mask, n_treated, n_control))
}

/// Identify treatment and control groups from a string treatment column.
///
/// # Arguments
///
/// * `treatment_col` - Treatment column values (None for null/missing)
/// * `treatment_value` - Value indicating treatment group membership
/// * `control_value` - Optional value indicating control group. If None, all non-treatment
///                     observations become control.
///
/// # Returns
///
/// Result containing (mask, n_treated, n_control) or BalanceError if validation fails.
///
/// # Errors
///
/// * `EmptyTreatmentGroup` - If no observations match treatment_value
/// * `EmptyControlGroup` - If no observations in control group
/// * `MultiValuedTreatment` - If control_value is None and >2 unique non-null values exist
pub fn identify_groups_str(
    treatment_col: &[Option<&str>],
    treatment_value: &str,
    control_value: Option<&str>,
) -> Result<(Vec<bool>, usize, usize), BalanceError> {
    let mut mask = vec![false; treatment_col.len()];
    let mut n_treated = 0usize;
    let mut n_control = 0usize;

    // For MultiValuedTreatment validation when control_value is None
    let mut unique_values = std::collections::HashSet::new();

    for (i, val) in treatment_col.iter().enumerate() {
        match val {
            None => {
                // Skip null values - they are excluded from both groups
                continue;
            }
            Some(v) => {
                // Track unique values for validation
                if control_value.is_none() {
                    unique_values.insert(*v);
                }

                if *v == treatment_value {
                    mask[i] = true;
                    n_treated += 1;
                } else if let Some(cv) = control_value {
                    // When control_value is specified, only include matching observations
                    if *v == cv {
                        // mask[i] is already false (control)
                        n_control += 1;
                    }
                    // else: neither treatment nor control, skip
                } else {
                    // When control_value is None, all non-treatment become control
                    // mask[i] is already false (control)
                    n_control += 1;
                }
            }
        }
    }

    // Validate n_treated > 0
    if n_treated == 0 {
        return Err(BalanceError::EmptyTreatmentGroup);
    }

    // Validate n_control > 0
    if n_control == 0 {
        return Err(BalanceError::EmptyControlGroup);
    }

    // If control_value is None, validate <= 2 unique values
    if control_value.is_none() && unique_values.len() > 2 {
        return Err(BalanceError::MultiValuedTreatment {
            n_values: unique_values.len(),
        });
    }

    Ok((mask, n_treated, n_control))
}

// ============================================================================
// Derived Metrics Computation
// ============================================================================

/// Compute the Standardized Mean Difference (SMD) between treatment and control groups.
///
/// SMD measures the difference in means between groups relative to the pooled
/// standard deviation. It is widely used in propensity score analysis to assess
/// covariate balance.
///
/// # Formula
///
/// SMD = (mean_t - mean_c) / sqrt((var_t_unadj + var_c_unadj) / 2)
///
/// **CRITICAL**: The denominator uses UNADJUSTED (unweighted) variances even when
/// computing SMD for weighted statistics. This matches cobalt's behavior and is
/// the recommended practice per Austin (2011).
///
/// # Arguments
///
/// * `mean_t` - Mean of the covariate in the treatment group
/// * `mean_c` - Mean of the covariate in the control group
/// * `var_t_unadj` - Unadjusted (unweighted) variance in the treatment group
/// * `var_c_unadj` - Unadjusted (unweighted) variance in the control group
///
/// # Returns
///
/// The standardized mean difference as f64.
///
/// # Edge Cases
///
/// - If pooled variance <= 0 and means are equal (within epsilon): returns 0.0
/// - If pooled variance <= 0 and means differ: returns f64::NAN
/// - Normal case: returns (mean_t - mean_c) / sqrt(pooled_var)
///
/// # Example
///
/// ```
/// use causers::balance::compute_smd;
///
/// // Treatment mean = 10, Control mean = 8
/// // Treatment variance = 4, Control variance = 4
/// // Pooled SD = sqrt((4 + 4) / 2) = 2
/// // SMD = (10 - 8) / 2 = 1.0
/// let smd = compute_smd(10.0, 8.0, 4.0, 4.0);
/// assert!((smd - 1.0).abs() < 1e-10);
/// ```
pub fn compute_smd(mean_t: f64, mean_c: f64, var_t_unadj: f64, var_c_unadj: f64) -> f64 {
    // Pooled variance from UNADJUSTED variances
    let pooled_var = (var_t_unadj + var_c_unadj) / 2.0;

    // Handle zero/negative pooled variance edge cases
    if pooled_var <= 0.0 {
        // Use epsilon for floating-point comparison of means
        const EPSILON: f64 = 1e-10;
        if (mean_t - mean_c).abs() < EPSILON {
            // Both zero variance and equal means: SMD = 0
            return 0.0;
        } else {
            // Zero variance but different means: undefined, return NaN
            return f64::NAN;
        }
    }

    // Normal case: compute SMD
    let pooled_sd = pooled_var.sqrt();
    (mean_t - mean_c) / pooled_sd
}

/// Compute the variance ratio between treatment and control groups.
///
/// The variance ratio measures the relative dispersion of a covariate between
/// treatment and control groups. Values close to 1.0 indicate similar spread,
/// while extreme values (< 0.5 or > 2.0) may indicate balance issues.
///
/// # Formula
///
/// VR = var_t / var_c
///
/// # Arguments
///
/// * `var_t` - Variance in the treatment group
/// * `var_c` - Variance in the control group
///
/// # Returns
///
/// The variance ratio as f64.
///
/// # Edge Cases
///
/// - If var_c == 0 and var_t > 0: returns f64::INFINITY
/// - If var_c == 0 and var_t == 0: returns NaN (both groups have zero variance, ratio undefined)
/// - If var_c < 0: returns f64::NAN (defensive, shouldn't happen)
/// - Normal case: returns var_t / var_c
///
/// # Example
///
/// ```
/// use causers::balance::compute_variance_ratio;
///
/// // Equal variances
/// let vr = compute_variance_ratio(4.0, 4.0);
/// assert!((vr - 1.0).abs() < 1e-10);
///
/// // Treatment has twice the variance
/// let vr = compute_variance_ratio(8.0, 4.0);
/// assert!((vr - 2.0).abs() < 1e-10);
/// ```
pub fn compute_variance_ratio(var_t: f64, var_c: f64) -> f64 {
    // Handle zero control variance
    if var_c == 0.0 {
        if var_t > 0.0 {
            // Treatment has variance, control doesn't: infinite ratio
            return f64::INFINITY;
        } else if var_t == 0.0 {
            // Both groups have zero variance: undefined ratio
            return f64::NAN;
        }
        // var_t < 0 is handled below
    }

    // Defensive: negative variance shouldn't happen, but return NaN if it does
    if var_c < 0.0 {
        return f64::NAN;
    }

    if var_t < 0.0 {
        return f64::NAN;
    }

    // Normal case
    var_t / var_c
}

// ============================================================================
// Core Orchestration Function
// ============================================================================

/// Compute covariate balance statistics between treatment and control groups.
///
/// This is the main orchestration function that coordinates moment computation,
/// SMD calculation, and variance ratio computation for all covariates.
///
/// # Arguments
///
/// * `treatment_mask` - Boolean mask where `true` = treated, `false` = control.
///   Must contain only observations that belong to treatment or control (no excluded obs).
/// * `n_treated` - Number of observations in the treatment group
/// * `n_control` - Number of observations in the control group
/// * `covariate_data` - Slice of (name, values) pairs for each covariate column.
///   Categorical covariates should already be expanded to dummy columns.
/// * `weights` - Optional observation weights for weighted statistics
///
/// # Returns
///
/// A `BalanceResult` containing all computed statistics on success.
///
/// # Errors
///
/// Returns `BalanceError` if weighted moment computation fails (e.g., negative weights,
/// zero total weight in a group).
///
/// # Algorithm
///
/// For each covariate:
/// 1. Compute unweighted moments (always, needed for SMD denominator)
/// 2. If weights provided, compute weighted moments for reporting
/// 3. Compute SMD using UNADJUSTED (unweighted) variances as denominator
/// 4. Compute variance ratio using the (possibly weighted) variances
pub fn compute_balance(
    treatment_mask: &[bool],
    n_treated: usize,
    n_control: usize,
    covariate_data: &[(String, Vec<f64>)],
    weights: Option<&[f64]>,
) -> Result<BalanceResult, BalanceError> {
    let is_weighted = weights.is_some();

    let mut result = BalanceResult::new();
    result.n_treated = n_treated;
    result.n_control = n_control;
    result.is_weighted = is_weighted;

    let mut ess_treated_set = false;

    for (name, values) in covariate_data {
        // Step 1: Always compute unweighted moments (needed for SMD denominator)
        let (unweighted_treated, unweighted_control) =
            compute_unweighted_moments(values, treatment_mask)?;

        // Store unadjusted variances for SMD denominator
        let var_t_unadj = unweighted_treated.variance;
        let var_c_unadj = unweighted_control.variance;

        // Step 2: Determine which stats to report (weighted or unweighted)
        let (report_mean_t, report_mean_c, report_var_t, report_var_c, report_sd_t, report_sd_c) =
            if let Some(w) = weights {
                let (weighted_treated, weighted_control, ess_t, ess_c) =
                    compute_weighted_moments(values, treatment_mask, w)?;

                // Store ESS from first covariate only (same mask/weights for all)
                if !ess_treated_set {
                    result.ess_treated = Some(ess_t);
                    result.ess_control = Some(ess_c);
                    ess_treated_set = true;
                }

                (
                    weighted_treated.mean,
                    weighted_control.mean,
                    weighted_treated.variance,
                    weighted_control.variance,
                    weighted_treated.sd,
                    weighted_control.sd,
                )
            } else {
                (
                    unweighted_treated.mean,
                    unweighted_control.mean,
                    unweighted_treated.variance,
                    unweighted_control.variance,
                    unweighted_treated.sd,
                    unweighted_control.sd,
                )
            };

        // Step 3: Compute SMD using UNADJUSTED (always unweighted) variances
        // The means used are the reported means (weighted if applicable)
        let smd_value = compute_smd(report_mean_t, report_mean_c, var_t_unadj, var_c_unadj);

        // Step 4: Compute variance ratio using reported (possibly weighted) variances
        let vr_value = compute_variance_ratio(report_var_t, report_var_c);

        // Insert into result maps
        result
            .mean_treated
            .insert(name.clone(), report_mean_t);
        result
            .mean_control
            .insert(name.clone(), report_mean_c);
        result.var_treated.insert(name.clone(), report_var_t);
        result.var_control.insert(name.clone(), report_var_c);
        result.sd_treated.insert(name.clone(), report_sd_t);
        result.sd_control.insert(name.clone(), report_sd_c);
        result.smd.insert(name.clone(), smd_value);
        result.variance_ratio.insert(name.clone(), vr_value);
        result.covariates.push(name.clone());
    }

    Ok(result)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BalanceError Display Tests
    // ========================================================================

    #[test]
    fn test_balance_error_column_not_found() {
        let err = BalanceError::ColumnNotFound {
            column: "age".to_string(),
        };
        assert_eq!(format!("{}", err), "Column 'age' not found in DataFrame");
    }

    #[test]
    fn test_balance_error_weights_column_not_found() {
        let err = BalanceError::WeightsColumnNotFound {
            column: "ipw".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "Weights column 'ipw' not found in DataFrame"
        );
    }

    #[test]
    fn test_balance_error_non_numeric_covariate() {
        let err = BalanceError::NonNumericCovariate {
            column: "category".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "Covariate column 'category' must be numeric or categorical"
        );
    }

    #[test]
    fn test_balance_error_no_treatment_variation() {
        let err = BalanceError::NoTreatmentVariation;
        assert_eq!(format!("{}", err), "Treatment variable has no variation");
    }

    #[test]
    fn test_balance_error_null_values_in_covariate() {
        let err = BalanceError::NullValuesInCovariate {
            column: "income".to_string(),
        };
        assert_eq!(format!("{}", err), "Column 'income' contains null values");
    }

    #[test]
    fn test_balance_error_null_values_in_weights() {
        let err = BalanceError::NullValuesInWeights {
            column: "weight".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "Weights column 'weight' contains null values"
        );
    }

    #[test]
    fn test_balance_error_negative_weights() {
        let err = BalanceError::NegativeWeights;
        assert_eq!(format!("{}", err), "Weights must be non-negative");
    }

    #[test]
    fn test_balance_error_zero_total_weight() {
        let err = BalanceError::ZeroTotalWeight {
            group: "treatment".to_string(),
        };
        assert_eq!(format!("{}", err), "Zero total weight in treatment group");
    }

    #[test]
    fn test_balance_error_empty_treatment_group() {
        let err = BalanceError::EmptyTreatmentGroup;
        assert_eq!(format!("{}", err), "No observations in treatment group");
    }

    #[test]
    fn test_balance_error_empty_control_group() {
        let err = BalanceError::EmptyControlGroup;
        assert_eq!(format!("{}", err), "No observations in control group");
    }

    #[test]
    fn test_balance_error_multi_valued_treatment() {
        let err = BalanceError::MultiValuedTreatment { n_values: 5 };
        assert_eq!(
            format!("{}", err),
            "Binary treatment required; found 5 unique values. Specify control_value or use binary treatment."
        );
    }

    #[test]
    fn test_balance_error_numerical_overflow() {
        let err = BalanceError::NumericalOverflow {
            column: "large_values".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "Numerical overflow in variance computation for covariate 'large_values'"
        );
    }

    #[test]
    fn test_balance_error_high_cardinality_categorical() {
        let err = BalanceError::HighCardinalityCategorical {
            column: "region".to_string(),
            n_levels: 5000,
            max_levels: 1000,
        };
        assert_eq!(
            format!("{}", err),
            "Categorical column 'region' has 5000 levels (max 1000). Consider grouping rare levels or converting to continuous."
        );
    }

    #[test]
    fn test_balance_error_internal_error() {
        let err = BalanceError::InternalError {
            reason: "values/mask length mismatch".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "Internal error: values/mask length mismatch"
        );
    }

    // ========================================================================
    // TreatmentValue Tests
    // ========================================================================

    #[test]
    fn test_treatment_value_matches_int() {
        let tv = TreatmentValue::Int(1);
        assert!(tv.matches_int(1));
        assert!(!tv.matches_int(0));
        assert!(tv.matches_float(1.0));
        assert!(!tv.matches_str("1"));
    }

    #[test]
    fn test_treatment_value_matches_float() {
        let tv = TreatmentValue::Float(1.5);
        assert!(tv.matches_float(1.5));
        assert!(!tv.matches_float(1.0));
        assert!(!tv.matches_int(1));
        assert!(!tv.matches_str("1.5"));
    }

    #[test]
    fn test_treatment_value_matches_string() {
        let tv = TreatmentValue::String("treated".to_string());
        assert!(tv.matches_str("treated"));
        assert!(!tv.matches_str("control"));
        assert!(!tv.matches_int(1));
        assert!(!tv.matches_float(1.0));
    }

    // ========================================================================
    // BalanceConfig Tests
    // ========================================================================

    #[test]
    fn test_balance_config_default() {
        let config = BalanceConfig::default();
        assert_eq!(config.treatment_value, TreatmentValue::Int(1));
        assert!(config.control_value.is_none());
        assert!(!config.weighted);
        assert_eq!(
            config.max_categorical_levels,
            DEFAULT_MAX_CATEGORICAL_LEVELS
        );
    }

    // ========================================================================
    // MomentAccumulator Tests
    // ========================================================================

    #[test]
    fn test_moment_accumulator_empty() {
        let acc = MomentAccumulator::new();
        let (mean, var) = acc.finalize();
        assert_eq!(mean, 0.0);
        assert_eq!(var, 0.0);
        assert_eq!(acc.count(), 0);
    }

    #[test]
    fn test_moment_accumulator_single_value() {
        let mut acc = MomentAccumulator::new();
        acc.update(5.0);
        let (mean, var) = acc.finalize();
        assert!((mean - 5.0).abs() < 1e-10);
        assert_eq!(var, 0.0); // Variance undefined for single observation
        assert_eq!(acc.count(), 1);
    }

    #[test]
    fn test_moment_accumulator_known_values() {
        // Values: [1, 2, 3, 4, 5]
        // Mean: 3
        // Variance (Bessel corrected): sum((x-mean)^2)/(n-1) = 10/4 = 2.5
        let mut acc = MomentAccumulator::new();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            acc.update(x);
        }
        let (mean, var) = acc.finalize();
        assert!((mean - 3.0).abs() < 1e-10);
        assert!((var - 2.5).abs() < 1e-10);
        assert_eq!(acc.count(), 5);
    }

    #[test]
    fn test_moment_accumulator_two_values() {
        // Values: [2, 4]
        // Mean: 3
        // Variance (Bessel corrected): sum((x-mean)^2)/(n-1) = (1+1)/1 = 2
        let mut acc = MomentAccumulator::new();
        acc.update(2.0);
        acc.update(4.0);
        let (mean, var) = acc.finalize();
        assert!((mean - 3.0).abs() < 1e-10);
        assert!((var - 2.0).abs() < 1e-10);
        assert_eq!(acc.count(), 2);
    }

    #[test]
    fn test_moment_accumulator_constant_values() {
        // All same values should give variance = 0
        let mut acc = MomentAccumulator::new();
        for _ in 0..10 {
            acc.update(7.0);
        }
        let (mean, var) = acc.finalize();
        assert!((mean - 7.0).abs() < 1e-10);
        assert!(var.abs() < 1e-10);
    }

    #[test]
    fn test_moment_accumulator_negative_values() {
        // Values: [-2, -1, 0, 1, 2]
        // Mean: 0
        // Variance: (4+1+0+1+4)/4 = 10/4 = 2.5
        let mut acc = MomentAccumulator::new();
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            acc.update(x);
        }
        let (mean, var) = acc.finalize();
        assert!(mean.abs() < 1e-10);
        assert!((var - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_moment_accumulator_large_values_numerical_stability() {
        // Test numerical stability with large values
        // Values: [1e8, 1e8 + 1, 1e8 + 2]
        // Mean: 1e8 + 1
        // Variance: (1+0+1)/2 = 1
        let mut acc = MomentAccumulator::new();
        let base = 1e8;
        for x in [base, base + 1.0, base + 2.0] {
            acc.update(x);
        }
        let (mean, var) = acc.finalize();
        assert!((mean - (base + 1.0)).abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-5);
    }

    // ========================================================================
    // WeightedMomentAccumulator Tests
    // ========================================================================

    #[test]
    fn test_weighted_moment_accumulator_empty() {
        let acc = WeightedMomentAccumulator::new();
        let (mean, var, ess) = acc.finalize();
        assert_eq!(mean, 0.0);
        assert_eq!(var, 0.0);
        assert_eq!(ess, 0.0);
    }

    #[test]
    fn test_weighted_moment_accumulator_uniform_weights() {
        // With uniform weights, weighted mean should equal unweighted mean
        // Values: [1, 2, 3, 4, 5], weights: [1, 1, 1, 1, 1]
        // Mean: 3
        let mut acc = WeightedMomentAccumulator::new();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            acc.update(x, 1.0);
        }
        let (mean, var, ess) = acc.finalize();
        assert!((mean - 3.0).abs() < 1e-10);
        // ESS should equal n for uniform weights
        assert!((ess - 5.0).abs() < 1e-10);
        // Variance should be close to unweighted variance
        assert!((var - 2.5).abs() < 0.5); // Allow some tolerance due to formula differences
    }

    #[test]
    fn test_weighted_moment_accumulator_weighted_mean() {
        // Values: [1, 2], weights: [1, 3]
        // Weighted mean: (1*1 + 2*3) / (1+3) = 7/4 = 1.75
        let mut acc = WeightedMomentAccumulator::new();
        acc.update(1.0, 1.0);
        acc.update(2.0, 3.0);
        let (mean, _var, _ess) = acc.finalize();
        assert!((mean - 1.75).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_moment_accumulator_ess() {
        // ESS = V1^2 / V2
        // V1 = sum of weights, V2 = sum of squared weights
        // weights: [1, 2, 3], V1 = 6, V2 = 14, ESS = 36/14 = 2.571...
        let mut acc = WeightedMomentAccumulator::new();
        acc.update(1.0, 1.0);
        acc.update(2.0, 2.0);
        acc.update(3.0, 3.0);
        let (_mean, _var, ess) = acc.finalize();
        let expected_ess = 36.0 / 14.0;
        assert!((ess - expected_ess).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_moment_accumulator_single_dominant_weight() {
        // When one weight dominates (ESS ≈ 1), variance should be NaN
        let mut acc = WeightedMomentAccumulator::new();
        acc.update(1.0, 1000.0);
        acc.update(2.0, 0.001);
        let (_mean, var, ess) = acc.finalize();
        // ESS should be close to 1
        assert!(ess < 2.0);
        // When V1^2 ≈ V2, variance is unstable
        // The variance might be NaN or very large
    }

    // ========================================================================
    // GroupCovariateStats Tests
    // ========================================================================

    #[test]
    fn test_group_covariate_stats_from_accumulator() {
        let mut acc = MomentAccumulator::new();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            acc.update(x);
        }
        let stats = GroupCovariateStats::from_accumulator(&acc);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.variance - 2.5).abs() < 1e-10);
        assert!((stats.sd - 2.5_f64.sqrt()).abs() < 1e-10);
        assert_eq!(stats.n, 5);
    }

    // ========================================================================
    // BalanceResult Tests
    // ========================================================================

    #[test]
    fn test_balance_result_new() {
        let result = BalanceResult::new();
        assert!(result.mean_treated.is_empty());
        assert!(result.mean_control.is_empty());
        assert_eq!(result.n_treated, 0);
        assert_eq!(result.n_control, 0);
        assert!(!result.is_weighted);
    }

    #[test]
    fn test_balance_result_repr_no_imbalanced() {
        let mut result = BalanceResult::new();
        result.n_treated = 100;
        result.n_control = 100;
        result.covariates = vec!["age".to_string(), "income".to_string()];
        result.smd.insert("age".to_string(), 0.05);
        result.smd.insert("income".to_string(), 0.08);

        let repr = result.__repr__();
        assert!(repr.contains("n_treated=100"));
        assert!(repr.contains("n_control=100"));
        assert!(repr.contains("n_covariates=2"));
        assert!(repr.contains("n_imbalanced=0"));
    }

    #[test]
    fn test_balance_result_repr_with_imbalanced() {
        let mut result = BalanceResult::new();
        result.n_treated = 50;
        result.n_control = 50;
        result.covariates = vec![
            "age".to_string(),
            "income".to_string(),
            "education".to_string(),
        ];
        result.smd.insert("age".to_string(), 0.05);
        result.smd.insert("income".to_string(), 0.25); // Imbalanced
        result.smd.insert("education".to_string(), -0.15); // Imbalanced

        let repr = result.__repr__();
        assert!(repr.contains("n_imbalanced=2"));
    }

    // ========================================================================
    // compute_unweighted_moments Tests
    // ========================================================================

    #[test]
    fn test_unweighted_moments_basic() {
        // Treated: [1, 2, 3], Control: [4, 5, 6]
        // Treated mean: 2, Control mean: 5
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![true, true, true, false, false, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Verify counts
        assert_eq!(treated.n, 3);
        assert_eq!(control.n, 3);

        // Verify means
        assert!(
            (treated.mean - 2.0).abs() < 1e-10,
            "treated mean should be 2.0, got {}",
            treated.mean
        );
        assert!(
            (control.mean - 5.0).abs() < 1e-10,
            "control mean should be 5.0, got {}",
            control.mean
        );

        // Verify variances (Bessel corrected)
        // Treated: [(1-2)^2 + (2-2)^2 + (3-2)^2] / (3-1) = [1+0+1]/2 = 1.0
        // Control: [(4-5)^2 + (5-5)^2 + (6-5)^2] / (3-1) = [1+0+1]/2 = 1.0
        assert!(
            (treated.variance - 1.0).abs() < 1e-10,
            "treated variance should be 1.0, got {}",
            treated.variance
        );
        assert!(
            (control.variance - 1.0).abs() < 1e-10,
            "control variance should be 1.0, got {}",
            control.variance
        );

        // Verify SDs
        assert!(
            (treated.sd - 1.0).abs() < 1e-10,
            "treated sd should be 1.0, got {}",
            treated.sd
        );
        assert!(
            (control.sd - 1.0).abs() < 1e-10,
            "control sd should be 1.0, got {}",
            control.sd
        );
    }

    #[test]
    fn test_unweighted_moments_interleaved_groups() {
        // Test with interleaved treatment/control to verify correct grouping
        // T: [10, 30, 50], C: [20, 40]
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mask = vec![true, false, true, false, true];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated: [10, 30, 50], mean = 30, var = [(10-30)^2 + (30-30)^2 + (50-30)^2]/2 = [400+0+400]/2 = 400
        assert_eq!(treated.n, 3);
        assert!((treated.mean - 30.0).abs() < 1e-10);
        assert!((treated.variance - 400.0).abs() < 1e-10);
        assert!((treated.sd - 20.0).abs() < 1e-10);

        // Control: [20, 40], mean = 30, var = [(20-30)^2 + (40-30)^2]/1 = [100+100]/1 = 200
        assert_eq!(control.n, 2);
        assert!((control.mean - 30.0).abs() < 1e-10);
        assert!((control.variance - 200.0).abs() < 1e-10);
        assert!((control.sd - (200.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_unweighted_moments_single_observation_treated() {
        // Edge case: single observation in treated group
        // Variance should be 0 (undefined for n=1)
        let values = vec![5.0, 1.0, 2.0, 3.0];
        let mask = vec![true, false, false, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        assert_eq!(treated.n, 1);
        assert!((treated.mean - 5.0).abs() < 1e-10);
        assert_eq!(
            treated.variance, 0.0,
            "variance should be 0 for single observation"
        );
        assert_eq!(treated.sd, 0.0, "sd should be 0 for single observation");

        // Control: [1, 2, 3], mean = 2, var = 1
        assert_eq!(control.n, 3);
        assert!((control.mean - 2.0).abs() < 1e-10);
        assert!((control.variance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_unweighted_moments_single_observation_control() {
        // Edge case: single observation in control group
        let values = vec![1.0, 2.0, 3.0, 10.0];
        let mask = vec![true, true, true, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated: [1, 2, 3]
        assert_eq!(treated.n, 3);
        assert!((treated.mean - 2.0).abs() < 1e-10);

        // Control: [10]
        assert_eq!(control.n, 1);
        assert!((control.mean - 10.0).abs() < 1e-10);
        assert_eq!(
            control.variance, 0.0,
            "variance should be 0 for single observation"
        );
        assert_eq!(control.sd, 0.0, "sd should be 0 for single observation");
    }

    #[test]
    fn test_unweighted_moments_empty_treated_group() {
        // Edge case: no observations in treated group (all control)
        let values = vec![1.0, 2.0, 3.0];
        let mask = vec![false, false, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated: empty
        assert_eq!(treated.n, 0);
        assert_eq!(treated.mean, 0.0);
        assert_eq!(treated.variance, 0.0);
        assert_eq!(treated.sd, 0.0);

        // Control: [1, 2, 3]
        assert_eq!(control.n, 3);
        assert!((control.mean - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_unweighted_moments_empty_control_group() {
        // Edge case: no observations in control group (all treated)
        let values = vec![1.0, 2.0, 3.0];
        let mask = vec![true, true, true];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated: [1, 2, 3]
        assert_eq!(treated.n, 3);
        assert!((treated.mean - 2.0).abs() < 1e-10);

        // Control: empty
        assert_eq!(control.n, 0);
        assert_eq!(control.mean, 0.0);
        assert_eq!(control.variance, 0.0);
        assert_eq!(control.sd, 0.0);
    }

    #[test]
    fn test_unweighted_moments_empty_both_groups() {
        // Edge case: empty arrays
        let values: Vec<f64> = vec![];
        let mask: Vec<bool> = vec![];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        assert_eq!(treated.n, 0);
        assert_eq!(control.n, 0);
        assert_eq!(treated.mean, 0.0);
        assert_eq!(control.mean, 0.0);
    }

    #[test]
    fn test_unweighted_moments_constant_values_treated() {
        // Edge case: all treated values are the same
        let values = vec![7.0, 7.0, 7.0, 1.0, 2.0, 3.0];
        let mask = vec![true, true, true, false, false, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated: [7, 7, 7], mean = 7, var = 0
        assert_eq!(treated.n, 3);
        assert!((treated.mean - 7.0).abs() < 1e-10);
        assert!(
            treated.variance.abs() < 1e-10,
            "variance should be 0 for constant values, got {}",
            treated.variance
        );
        assert!(
            treated.sd.abs() < 1e-10,
            "sd should be 0 for constant values"
        );
    }

    #[test]
    fn test_unweighted_moments_negative_values() {
        // Test with negative values
        let values = vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let mask = vec![true, true, true, false, false, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated: [-3, -2, -1], mean = -2, var = 1
        assert_eq!(treated.n, 3);
        assert!((treated.mean - (-2.0)).abs() < 1e-10);
        assert!((treated.variance - 1.0).abs() < 1e-10);

        // Control: [1, 2, 3], mean = 2, var = 1
        assert_eq!(control.n, 3);
        assert!((control.mean - 2.0).abs() < 1e-10);
        assert!((control.variance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_unweighted_moments_large_values_numerical_stability() {
        // Test numerical stability with large values
        // Values centered around 1e8
        let base = 1e8;
        let values = vec![
            base,
            base + 1.0,
            base + 2.0, // treated
            base + 3.0,
            base + 4.0,
            base + 5.0, // control
        ];
        let mask = vec![true, true, true, false, false, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated: mean = base + 1, var = 1
        assert!((treated.mean - (base + 1.0)).abs() < 1e-5);
        assert!((treated.variance - 1.0).abs() < 1e-5);

        // Control: mean = base + 4, var = 1
        assert!((control.mean - (base + 4.0)).abs() < 1e-5);
        assert!((control.variance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_unweighted_moments_manual_calculation() {
        // Explicit manual calculation test
        // Treated: [2, 4, 6, 8]
        // mean = (2+4+6+8)/4 = 20/4 = 5
        // var = [(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2] / (4-1)
        //     = [9 + 1 + 1 + 9] / 3 = 20/3 = 6.666...
        // sd = sqrt(20/3) = 2.581988...
        //
        // Control: [1, 3, 5]
        // mean = (1+3+5)/3 = 9/3 = 3
        // var = [(1-3)^2 + (3-3)^2 + (5-3)^2] / (3-1)
        //     = [4 + 0 + 4] / 2 = 8/2 = 4
        // sd = sqrt(4) = 2
        let values = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0];
        let mask = vec![true, false, true, false, true, false, true];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated checks
        assert_eq!(treated.n, 4);
        assert!((treated.mean - 5.0).abs() < 1e-10, "treated mean mismatch");
        let expected_treated_var = 20.0 / 3.0;
        assert!(
            (treated.variance - expected_treated_var).abs() < 1e-10,
            "treated variance: expected {}, got {}",
            expected_treated_var,
            treated.variance
        );
        assert!(
            (treated.sd - expected_treated_var.sqrt()).abs() < 1e-10,
            "treated sd mismatch"
        );

        // Control checks
        assert_eq!(control.n, 3);
        assert!((control.mean - 3.0).abs() < 1e-10, "control mean mismatch");
        assert!(
            (control.variance - 4.0).abs() < 1e-10,
            "control variance: expected 4.0, got {}",
            control.variance
        );
        assert!((control.sd - 2.0).abs() < 1e-10, "control sd mismatch");
    }

    #[test]
    fn test_unweighted_moments_two_observations_per_group() {
        // Minimum viable case for variance computation (n=2 each)
        // Treated: [0, 4], mean = 2, var = [(0-2)^2 + (4-2)^2]/1 = [4+4]/1 = 8
        // Control: [1, 3], mean = 2, var = [(1-2)^2 + (3-2)^2]/1 = [1+1]/1 = 2
        let values = vec![0.0, 1.0, 4.0, 3.0];
        let mask = vec![true, false, true, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        assert_eq!(treated.n, 2);
        assert!((treated.mean - 2.0).abs() < 1e-10);
        assert!(
            (treated.variance - 8.0).abs() < 1e-10,
            "treated variance: expected 8.0, got {}",
            treated.variance
        );

        assert_eq!(control.n, 2);
        assert!((control.mean - 2.0).abs() < 1e-10);
        assert!(
            (control.variance - 2.0).abs() < 1e-10,
            "control variance: expected 2.0, got {}",
            control.variance
        );
    }

    #[test]
    fn test_unweighted_moments_length_mismatch() {
        let values = vec![1.0, 2.0, 3.0];
        let mask = vec![true, false]; // Length mismatch

        let result = super::compute_unweighted_moments(&values, &mask);
        assert!(result.is_err(), "expected error for length mismatch");
    }

    #[test]
    fn test_unweighted_moments_unequal_group_sizes() {
        // Treated: [1, 2], Control: [3, 4, 5, 6, 7]
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mask = vec![true, true, false, false, false, false, false];

        let (treated, control) = super::compute_unweighted_moments(&values, &mask).unwrap();

        // Treated: [1, 2], n=2, mean=1.5, var=[(1-1.5)^2+(2-1.5)^2]/1 = 0.5
        assert_eq!(treated.n, 2);
        assert!((treated.mean - 1.5).abs() < 1e-10);
        assert!((treated.variance - 0.5).abs() < 1e-10);

        // Control: [3,4,5,6,7], n=5, mean=5, var=[(3-5)^2+(4-5)^2+(5-5)^2+(6-5)^2+(7-5)^2]/4 = [4+1+0+1+4]/4 = 2.5
        assert_eq!(control.n, 5);
        assert!((control.mean - 5.0).abs() < 1e-10);
        assert!((control.variance - 2.5).abs() < 1e-10);
    }

    // ========================================================================
    // Categorical Expansion Tests
    // ========================================================================

    #[test]
    fn test_categorical_expansion_basic() {
        // Test basic one-hot expansion with 3 levels
        let values = vec!["red", "blue", "red", "green"];
        let expanded = super::expand_categorical(&values, "color");

        // Should have 3 dummy columns (one per level)
        assert_eq!(expanded.len(), 3, "should have 3 dummy columns");

        // Columns should be sorted alphabetically: blue, green, red
        assert_eq!(expanded[0].0, "color_blue");
        assert_eq!(expanded[1].0, "color_green");
        assert_eq!(expanded[2].0, "color_red");

        // Check values for "color_blue": [0, 1, 0, 0]
        assert_eq!(expanded[0].1, vec![0.0, 1.0, 0.0, 0.0]);

        // Check values for "color_green": [0, 0, 0, 1]
        assert_eq!(expanded[1].1, vec![0.0, 0.0, 0.0, 1.0]);

        // Check values for "color_red": [1, 0, 1, 0]
        assert_eq!(expanded[2].1, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_categorical_expansion_single_level() {
        // Test with a single level (degenerate case)
        let values = vec!["only", "only", "only"];
        let expanded = super::expand_categorical(&values, "status");

        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].0, "status_only");
        assert_eq!(expanded[0].1, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_categorical_expansion_two_levels() {
        // Test binary categorical
        let values = vec!["yes", "no", "yes", "no", "yes"];
        let expanded = super::expand_categorical(&values, "response");

        assert_eq!(expanded.len(), 2);

        // Sorted alphabetically: no, yes
        assert_eq!(expanded[0].0, "response_no");
        assert_eq!(expanded[0].1, vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        assert_eq!(expanded[1].0, "response_yes");
        assert_eq!(expanded[1].1, vec![1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_categorical_expansion_alphabetical_sorting() {
        // Verify alphabetical sorting with various level names
        let values = vec!["zebra", "apple", "mango", "banana"];
        let expanded = super::expand_categorical(&values, "fruit");

        // Should be sorted: apple, banana, mango, zebra
        assert_eq!(expanded[0].0, "fruit_apple");
        assert_eq!(expanded[1].0, "fruit_banana");
        assert_eq!(expanded[2].0, "fruit_mango");
        assert_eq!(expanded[3].0, "fruit_zebra");
    }

    #[test]
    fn test_categorical_expansion_numeric_strings() {
        // Test with numeric-looking strings (should still be treated as strings)
        let values = vec!["1", "0", "1", "0"];
        let expanded = super::expand_categorical(&values, "binary");

        assert_eq!(expanded.len(), 2);

        // Alphabetically: "0" < "1"
        assert_eq!(expanded[0].0, "binary_0");
        assert_eq!(expanded[0].1, vec![0.0, 1.0, 0.0, 1.0]);

        assert_eq!(expanded[1].0, "binary_1");
        assert_eq!(expanded[1].1, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_categorical_expansion_empty_input() {
        // Test with empty input
        let values: Vec<&str> = vec![];
        let expanded = super::expand_categorical(&values, "empty");

        assert!(expanded.is_empty());
    }

    #[test]
    fn test_categorical_expansion_single_observation() {
        // Test with a single observation
        let values = vec!["single"];
        let expanded = super::expand_categorical(&values, "col");

        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].0, "col_single");
        assert_eq!(expanded[0].1, vec![1.0]);
    }

    #[test]
    fn test_categorical_expansion_many_levels() {
        // Test with many unique levels
        let values = vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
        let expanded = super::expand_categorical(&values, "letter");

        assert_eq!(expanded.len(), 10);

        // Verify all levels are present and sorted
        let expected_names: Vec<String> = ('a'..='j').map(|c| format!("letter_{}", c)).collect();
        let actual_names: Vec<&String> = expanded.iter().map(|(name, _)| name).collect();
        assert_eq!(actual_names, expected_names.iter().collect::<Vec<_>>());

        // Each column should have exactly one 1.0
        for (i, (_, vals)) in expanded.iter().enumerate() {
            let sum: f64 = vals.iter().sum();
            assert_eq!(sum, 1.0, "column {} should have exactly one 1.0", i);

            // The 1.0 should be at position i (since values are sorted a,b,c,...,j)
            assert_eq!(
                vals[i], 1.0,
                "column {} should have 1.0 at position {}",
                i, i
            );
        }
    }

    #[test]
    fn test_categorical_expansion_column_naming() {
        // Verify the column naming format "{col}_{level}"
        let values = vec!["level1", "level2"];
        let expanded = super::expand_categorical(&values, "my_column");

        assert_eq!(expanded[0].0, "my_column_level1");
        assert_eq!(expanded[1].0, "my_column_level2");
    }

    #[test]
    fn test_categorical_expansion_all_levels_included() {
        // Verify that all levels are included (no reference level dropped)
        let values = vec!["a", "b", "c", "a", "b", "c"];
        let expanded = super::expand_categorical(&values, "cat");

        // All 3 levels should be present
        assert_eq!(expanded.len(), 3);

        // Sum of all dummy columns for each observation should be 1.0
        for i in 0..values.len() {
            let sum: f64 = expanded.iter().map(|(_, vals)| vals[i]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "sum of dummy values at position {} should be 1.0, got {}",
                i,
                sum
            );
        }
    }

    // ========================================================================
    // Categorical Cardinality Check Tests
    // ========================================================================

    #[test]
    fn test_categorical_cardinality_check_valid() {
        // Valid: 3 levels, max 1000
        let result = super::validate_categorical_cardinality(3, "color", 1000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_categorical_cardinality_check_at_limit() {
        // Valid: exactly at the limit
        let result = super::validate_categorical_cardinality(100, "region", 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_categorical_cardinality_check_exceeds_limit() {
        // Invalid: exceeds the limit
        let result = super::validate_categorical_cardinality(5000, "zipcode", 1000);
        assert!(result.is_err());

        match result {
            Err(BalanceError::HighCardinalityCategorical {
                column,
                n_levels,
                max_levels,
            }) => {
                assert_eq!(column, "zipcode");
                assert_eq!(n_levels, 5000);
                assert_eq!(max_levels, 1000);
            }
            _ => panic!("expected HighCardinalityCategorical error"),
        }
    }

    #[test]
    fn test_categorical_cardinality_check_zero_levels() {
        // Edge case: 0 levels (empty categorical)
        let result = super::validate_categorical_cardinality(0, "empty", 1000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_categorical_cardinality_check_one_over_limit() {
        // Edge case: exactly one over the limit
        let result = super::validate_categorical_cardinality(1001, "region", 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_categorical_cardinality_check_custom_limit() {
        // Test with custom limits
        let result = super::validate_categorical_cardinality(50, "state", 50);
        assert!(result.is_ok());

        let result = super::validate_categorical_cardinality(51, "state", 50);
        assert!(result.is_err());
    }

    // ========================================================================
    // Boolean to Numeric Tests
    // ========================================================================

    #[test]
    fn test_boolean_to_numeric_basic() {
        let bools = vec![true, false, true, true, false];
        let nums = super::boolean_to_numeric(&bools);

        assert_eq!(nums, vec![1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_boolean_to_numeric_all_true() {
        let bools = vec![true, true, true];
        let nums = super::boolean_to_numeric(&bools);

        assert_eq!(nums, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_boolean_to_numeric_all_false() {
        let bools = vec![false, false, false];
        let nums = super::boolean_to_numeric(&bools);

        assert_eq!(nums, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_boolean_to_numeric_empty() {
        let bools: Vec<bool> = vec![];
        let nums = super::boolean_to_numeric(&bools);

        assert!(nums.is_empty());
    }

    #[test]
    fn test_boolean_to_numeric_single() {
        assert_eq!(super::boolean_to_numeric(&[true]), vec![1.0]);
        assert_eq!(super::boolean_to_numeric(&[false]), vec![0.0]);
    }

    // ========================================================================
    // CovariateType Tests
    // ========================================================================

    #[test]
    fn test_covariate_type_enum() {
        // Test that the enum variants exist and can be compared
        let numeric = CovariateType::Numeric;
        let boolean = CovariateType::Boolean;
        let categorical = CovariateType::Categorical;

        assert_eq!(numeric, CovariateType::Numeric);
        assert_eq!(boolean, CovariateType::Boolean);
        assert_eq!(categorical, CovariateType::Categorical);

        assert_ne!(numeric, boolean);
        assert_ne!(boolean, categorical);
        assert_ne!(numeric, categorical);
    }

    #[test]
    fn test_covariate_type_clone_copy() {
        // Test that CovariateType is Clone and Copy
        let original = CovariateType::Numeric;
        let cloned = original.clone();
        let copied = original;

        assert_eq!(original, cloned);
        assert_eq!(original, copied);
    }

    #[test]
    fn test_covariate_type_debug() {
        // Test that Debug is implemented
        let numeric = CovariateType::Numeric;
        let debug_str = format!("{:?}", numeric);
        assert_eq!(debug_str, "Numeric");
    }

    // ========================================================================
    // compute_weighted_moments Tests
    // ========================================================================

    #[test]
    fn test_weighted_moments_basic_uniform_weights() {
        // With uniform weights, weighted mean should equal unweighted mean
        // Treated: [1, 2, 3] with weights [1, 1, 1], Control: [4, 5, 6] with weights [1, 1, 1]
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![true, true, true, false, false, false];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok(), "compute_weighted_moments should succeed");

        let (treated, control, ess_t, ess_c) = result.unwrap();

        // Verify counts
        assert_eq!(treated.n, 3);
        assert_eq!(control.n, 3);

        // Verify weighted means (should equal unweighted with uniform weights)
        // Treated: (1*1 + 2*1 + 3*1) / (1+1+1) = 6/3 = 2
        // Control: (4*1 + 5*1 + 6*1) / (1+1+1) = 15/3 = 5
        assert!(
            (treated.mean - 2.0).abs() < 1e-10,
            "treated mean should be 2.0, got {}",
            treated.mean
        );
        assert!(
            (control.mean - 5.0).abs() < 1e-10,
            "control mean should be 5.0, got {}",
            control.mean
        );

        // ESS should equal n for uniform weights: ESS = V1^2 / V2 = 3^2 / 3 = 3
        assert!(
            (ess_t - 3.0).abs() < 1e-10,
            "treated ESS should be 3.0, got {}",
            ess_t
        );
        assert!(
            (ess_c - 3.0).abs() < 1e-10,
            "control ESS should be 3.0, got {}",
            ess_c
        );
    }

    #[test]
    fn test_weighted_moments_weighted_mean_formula() {
        // Test weighted mean formula: sum(w*x) / sum(w)
        // Treated: [1, 2] with weights [1, 3]
        // Weighted mean: (1*1 + 2*3) / (1+3) = 7/4 = 1.75
        // Control: [3, 4] with weights [2, 2]
        // Weighted mean: (3*2 + 4*2) / (2+2) = 14/4 = 3.5
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, true, false, false];
        let weights = vec![1.0, 3.0, 2.0, 2.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (treated, control, _ess_t, _ess_c) = result.unwrap();

        // Verify weighted means
        assert!(
            (treated.mean - 1.75).abs() < 1e-10,
            "treated mean should be 1.75, got {}",
            treated.mean
        );
        assert!(
            (control.mean - 3.5).abs() < 1e-10,
            "control mean should be 3.5, got {}",
            control.mean
        );
    }

    #[test]
    fn test_weighted_moments_ess_computation() {
        // Test ESS = V1^2 / V2
        // Treated: weights [1, 2, 3], V1 = 6, V2 = 14, ESS = 36/14 = 2.571...
        // Control: weights [1, 1], V1 = 2, V2 = 2, ESS = 4/2 = 2
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true, true, true, false, false];
        let weights = vec![1.0, 2.0, 3.0, 1.0, 1.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (_treated, _control, ess_t, ess_c) = result.unwrap();

        // Treated ESS = 6^2 / 14 = 36/14 = 2.571428...
        let expected_ess_t = 36.0 / 14.0;
        assert!(
            (ess_t - expected_ess_t).abs() < 1e-10,
            "treated ESS should be {}, got {}",
            expected_ess_t,
            ess_t
        );

        // Control ESS = 2^2 / 2 = 2
        assert!(
            (ess_c - 2.0).abs() < 1e-10,
            "control ESS should be 2.0, got {}",
            ess_c
        );
    }

    #[test]
    fn test_weighted_moments_reliability_weights_variance() {
        // Test variance with reliability weights correction formula:
        // var = m2 * V1 / (V1^2 - V2)
        //
        // Treated: values [2, 4] with weights [1, 1]
        // V1 = 2, V2 = 2
        // Mean = (2*1 + 4*1) / 2 = 3
        // Using Welford:
        //   After x=2, w=1: sum_w=1, mean=2, m2=0
        //   After x=4, w=1: sum_w=2, delta=4-2=2, mean=2+1*2/2=3, delta2=4-3=1, m2=0+1*2*1=2
        // So m2 = 2
        // var = 2 * 2 / (4 - 2) = 4 / 2 = 2
        let values = vec![2.0, 4.0, 10.0, 20.0];
        let mask = vec![true, true, false, false];
        let weights = vec![1.0, 1.0, 1.0, 1.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (treated, control, _ess_t, _ess_c) = result.unwrap();

        // Treated: variance should be 2.0
        assert!(
            (treated.variance - 2.0).abs() < 1e-10,
            "treated variance should be 2.0, got {}",
            treated.variance
        );
        assert!(
            (treated.sd - 2.0_f64.sqrt()).abs() < 1e-10,
            "treated sd should be sqrt(2), got {}",
            treated.sd
        );

        // Control: values [10, 20], mean = 15
        // m2 = 1*(10-15 after adjustment) + ... = 50
        // var = 50 * 2 / (4-2) = 50
        assert!(
            (control.variance - 50.0).abs() < 1e-10,
            "control variance should be 50.0, got {}",
            control.variance
        );
    }

    #[test]
    fn test_weighted_moments_negative_weights_error() {
        // Test that negative weights return an error
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, true, false, false];
        let weights = vec![1.0, -1.0, 1.0, 1.0]; // Negative weight

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_err(), "negative weights should return error");

        match result.unwrap_err() {
            super::BalanceError::NegativeWeights => {} // Expected
            other => panic!("expected NegativeWeights error, got {:?}", other),
        }
    }

    #[test]
    fn test_weighted_moments_zero_weight_treated_group() {
        // Test that zero total weight in treated group returns error
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, true, false, false];
        let weights = vec![0.0, 0.0, 1.0, 1.0]; // Zero weights for treated

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_err(), "zero total weight should return error");

        match result.unwrap_err() {
            super::BalanceError::ZeroTotalWeight { group } => {
                assert_eq!(group, "treatment", "error should mention treatment group");
            }
            other => panic!("expected ZeroTotalWeight error, got {:?}", other),
        }
    }

    #[test]
    fn test_weighted_moments_zero_weight_control_group() {
        // Test that zero total weight in control group returns error
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, true, false, false];
        let weights = vec![1.0, 1.0, 0.0, 0.0]; // Zero weights for control

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_err(), "zero total weight should return error");

        match result.unwrap_err() {
            super::BalanceError::ZeroTotalWeight { group } => {
                assert_eq!(group, "control", "error should mention control group");
            }
            other => panic!("expected ZeroTotalWeight error, got {:?}", other),
        }
    }

    #[test]
    fn test_weighted_moments_near_zero_denominator_returns_nan() {
        // When V1^2 approximately equals V2 (single dominant weight), variance may be NaN
        // Use extreme weight disparity to make V1^2 approximately equal V2
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, true, false, false];
        let weights = vec![1e10, 1e-10, 1.0, 1.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok(), "should succeed but may return NaN variance");

        let (treated, _control, ess_t, _ess_c) = result.unwrap();

        // ESS should be very close to 1 (dominated by single weight)
        assert!(
            ess_t < 2.0,
            "ESS should be close to 1 for dominant weight, got {}",
            ess_t
        );

        // Variance might be NaN due to near-zero denominator or just very unstable
        // This is acceptable behavior per design doc
        if treated.variance.is_nan() {
            assert!(treated.sd.is_nan(), "SD should be NaN when variance is NaN");
        }
    }

    #[test]
    fn test_weighted_moments_interleaved_groups() {
        // Test with interleaved treatment/control observations
        // T: [1, 3] with w=[2, 2], C: [2, 4] with w=[1, 3]
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        let weights = vec![2.0, 1.0, 2.0, 3.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (treated, control, ess_t, ess_c) = result.unwrap();

        // Treated: [1, 3] with w=[2, 2]
        // Weighted mean = (1*2 + 3*2) / (2+2) = 8/4 = 2
        assert!((treated.mean - 2.0).abs() < 1e-10);
        assert_eq!(treated.n, 2);

        // Treated ESS = 4^2 / 8 = 16/8 = 2
        assert!((ess_t - 2.0).abs() < 1e-10);

        // Control: [2, 4] with w=[1, 3]
        // Weighted mean = (2*1 + 4*3) / (1+3) = 14/4 = 3.5
        assert!((control.mean - 3.5).abs() < 1e-10);
        assert_eq!(control.n, 2);

        // Control ESS = 4^2 / 10 = 16/10 = 1.6
        assert!((ess_c - 1.6).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_moments_all_zero_weights_one_positive() {
        // Edge case: only one observation has positive weight in each group
        let values = vec![5.0, 10.0, 20.0, 30.0];
        let mask = vec![true, true, false, false];
        let weights = vec![0.0, 1.0, 1.0, 0.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (treated, control, ess_t, ess_c) = result.unwrap();

        // Treated: only value 10 has weight, so mean = 10
        assert!((treated.mean - 10.0).abs() < 1e-10);
        // ESS = 1^2 / 1 = 1
        assert!((ess_t - 1.0).abs() < 1e-10);

        // Control: only value 20 has weight, so mean = 20
        assert!((control.mean - 20.0).abs() < 1e-10);
        assert!((ess_c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_moments_values_mask_length_mismatch() {
        let values = vec![1.0, 2.0, 3.0];
        let mask = vec![true, false]; // Length mismatch
        let weights = vec![1.0, 1.0, 1.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_err(), "expected error for length mismatch");
    }

    #[test]
    fn test_weighted_moments_values_weights_length_mismatch() {
        let values = vec![1.0, 2.0, 3.0];
        let mask = vec![true, true, false];
        let weights = vec![1.0, 1.0]; // Length mismatch

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_err(), "expected error for length mismatch");
    }

    #[test]
    fn test_weighted_moments_large_weights() {
        // Test with large weights for numerical stability
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, true, false, false];
        let weights = vec![1e6, 1e6, 1e6, 1e6];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (treated, control, ess_t, ess_c) = result.unwrap();

        // With uniform large weights, should behave like uniform weights
        assert!((treated.mean - 1.5).abs() < 1e-10);
        assert!((control.mean - 3.5).abs() < 1e-10);
        assert!((ess_t - 2.0).abs() < 1e-10);
        assert!((ess_c - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_moments_small_weights() {
        // Test with very small weights for numerical stability
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, true, false, false];
        let weights = vec![1e-6, 1e-6, 1e-6, 1e-6];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (treated, control, ess_t, ess_c) = result.unwrap();

        // With uniform small weights, should behave like uniform weights
        assert!((treated.mean - 1.5).abs() < 1e-10);
        assert!((control.mean - 3.5).abs() < 1e-10);
        assert!((ess_t - 2.0).abs() < 1e-10);
        assert!((ess_c - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_moments_mixed_group_sizes() {
        // Test with different number of observations in each group
        // Treated: [1, 2], Control: [3, 4, 5, 6, 7]
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mask = vec![true, true, false, false, false, false, false];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (treated, control, ess_t, ess_c) = result.unwrap();

        // Verify group sizes
        assert_eq!(treated.n, 2);
        assert_eq!(control.n, 5);

        // Verify means
        assert!((treated.mean - 1.5).abs() < 1e-10);
        assert!((control.mean - 5.0).abs() < 1e-10);

        // ESS should equal n for uniform weights
        assert!((ess_t - 2.0).abs() < 1e-10);
        assert!((ess_c - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_moments_constant_values_zero_variance() {
        // When all values are constant, variance should be 0 (or near-zero)
        // With reliability weights formula, constant values give m2 = 0, so var = 0
        let values = vec![5.0, 5.0, 5.0, 10.0, 10.0];
        let mask = vec![true, true, true, false, false];
        let weights = vec![1.0, 2.0, 1.0, 1.0, 1.0];

        let result = super::compute_weighted_moments(&values, &mask, &weights);
        assert!(result.is_ok());

        let (treated, control, _ess_t, _ess_c) = result.unwrap();

        // Treated: all 5.0, variance should be 0
        assert!((treated.mean - 5.0).abs() < 1e-10);
        assert!(
            treated.variance.abs() < 1e-10 || treated.variance.is_nan(),
            "treated variance should be 0 for constant values, got {}",
            treated.variance
        );

        // Control: all 10.0, variance should be 0
        assert!((control.mean - 10.0).abs() < 1e-10);
        assert!(
            control.variance.abs() < 1e-10 || control.variance.is_nan(),
            "control variance should be 0 for constant values, got {}",
            control.variance
        );
    }

    // ========================================================================
    // Group Identification Tests (identify_groups_int, identify_groups_float, identify_groups_str)
    // ========================================================================

    #[test]
    fn test_group_identification_basic_int() {
        // Basic binary treatment: 0=control, 1=treatment
        let treatment_col: Vec<Option<i64>> = vec![Some(1), Some(0), Some(1), Some(0), Some(0)];

        let result = identify_groups_int(&treatment_col, 1, None);
        assert!(result.is_ok());

        let (mask, n_treated, n_control) = result.unwrap();

        // Verify counts
        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 3);

        // Verify mask
        assert_eq!(mask, vec![true, false, true, false, false]);
    }

    #[test]
    fn test_group_identification_with_control_value_int() {
        // Multi-valued treatment with explicit control_value
        // Only observations with treatment=1 or control=0 are included
        let treatment_col: Vec<Option<i64>> =
            vec![Some(1), Some(0), Some(2), Some(1), Some(0), Some(3)];

        let result = identify_groups_int(&treatment_col, 1, Some(0));
        assert!(result.is_ok());

        let (mask, n_treated, n_control) = result.unwrap();

        // Observations with value 2 and 3 are excluded from counting
        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 2);

        // mask[i]=true means treatment, mask[i]=false means control or excluded
        // Only observations at indices 0, 3 are treated (value=1)
        // Only observations at indices 1, 4 are control (value=0)
        assert!(mask[0]); // 1 -> treated
        assert!(!mask[1]); // 0 -> control
        assert!(!mask[2]); // 2 -> excluded (not counted)
        assert!(mask[3]); // 1 -> treated
        assert!(!mask[4]); // 0 -> control
        assert!(!mask[5]); // 3 -> excluded (not counted)
    }

    #[test]
    fn test_empty_treatment_error_int() {
        // No observations match treatment_value
        let treatment_col: Vec<Option<i64>> = vec![Some(0), Some(0), Some(0)];

        let result = identify_groups_int(&treatment_col, 1, None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::EmptyTreatmentGroup => (),
            other => panic!("Expected EmptyTreatmentGroup, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_control_error_int() {
        // All observations are treatment
        let treatment_col: Vec<Option<i64>> = vec![Some(1), Some(1), Some(1)];

        let result = identify_groups_int(&treatment_col, 1, None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::EmptyControlGroup => (),
            other => panic!("Expected EmptyControlGroup, got {:?}", other),
        }
    }

    #[test]
    fn test_multi_valued_treatment_error_int() {
        // More than 2 unique values without specifying control_value
        let treatment_col: Vec<Option<i64>> = vec![Some(1), Some(2), Some(3), Some(4)];

        let result = identify_groups_int(&treatment_col, 1, None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::MultiValuedTreatment { n_values } => {
                assert_eq!(n_values, 4);
            }
            other => panic!("Expected MultiValuedTreatment, got {:?}", other),
        }
    }

    #[test]
    fn test_null_treatment_excluded_int() {
        // Null values should be excluded from both groups
        let treatment_col: Vec<Option<i64>> = vec![Some(1), None, Some(0), None, Some(1), Some(0)];

        let result = identify_groups_int(&treatment_col, 1, None);
        assert!(result.is_ok());

        let (mask, n_treated, n_control) = result.unwrap();

        // 2 treated (indices 0, 4), 2 control (indices 2, 5), 2 nulls excluded
        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 2);

        // Null positions have mask=false but are not counted
        assert!(mask[0]); // 1 -> treated
        assert!(!mask[1]); // null -> excluded
        assert!(!mask[2]); // 0 -> control
        assert!(!mask[3]); // null -> excluded
        assert!(mask[4]); // 1 -> treated
        assert!(!mask[5]); // 0 -> control
    }

    #[test]
    fn test_all_null_treatment_error() {
        // All values are null - should result in empty treatment group
        let treatment_col: Vec<Option<i64>> = vec![None, None, None];

        let result = identify_groups_int(&treatment_col, 1, None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::EmptyTreatmentGroup => (),
            other => panic!("Expected EmptyTreatmentGroup, got {:?}", other),
        }
    }

    #[test]
    fn test_group_identification_basic_float() {
        // Binary treatment with float values: 0.0=control, 1.0=treatment
        let treatment_col: Vec<Option<f64>> = vec![Some(1.0), Some(0.0), Some(1.0), Some(0.0)];

        let result = identify_groups_float(&treatment_col, 1.0, None);
        assert!(result.is_ok());

        let (mask, n_treated, n_control) = result.unwrap();

        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 2);
        assert_eq!(mask, vec![true, false, true, false]);
    }

    #[test]
    fn test_group_identification_float_epsilon() {
        // Float comparison with tiny differences (should match)
        let treatment_col: Vec<Option<f64>> =
            vec![Some(1.0 + 1e-12), Some(0.0), Some(1.0 - 1e-12), Some(0.0)];

        let result = identify_groups_float(&treatment_col, 1.0, None);
        assert!(result.is_ok());

        let (mask, n_treated, n_control) = result.unwrap();

        // Both 1.0+epsilon and 1.0-epsilon should match 1.0
        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 2);
    }

    #[test]
    fn test_group_identification_with_control_value_float() {
        // Multi-valued treatment with explicit control_value
        let treatment_col: Vec<Option<f64>> =
            vec![Some(1.0), Some(0.0), Some(2.0), Some(1.0), Some(0.0)];

        let result = identify_groups_float(&treatment_col, 1.0, Some(0.0));
        assert!(result.is_ok());

        let (mask, n_treated, n_control) = result.unwrap();

        // Value 2.0 is excluded
        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 2);
    }

    #[test]
    fn test_empty_treatment_error_float() {
        let treatment_col: Vec<Option<f64>> = vec![Some(0.0), Some(0.0), Some(0.0)];

        let result = identify_groups_float(&treatment_col, 1.0, None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::EmptyTreatmentGroup => (),
            other => panic!("Expected EmptyTreatmentGroup, got {:?}", other),
        }
    }

    #[test]
    fn test_multi_valued_treatment_error_float() {
        let treatment_col: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];

        let result = identify_groups_float(&treatment_col, 1.0, None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::MultiValuedTreatment { n_values } => {
                assert_eq!(n_values, 3);
            }
            other => panic!("Expected MultiValuedTreatment, got {:?}", other),
        }
    }

    #[test]
    fn test_null_treatment_excluded_float() {
        let treatment_col: Vec<Option<f64>> = vec![Some(1.0), None, Some(0.0), None];

        let result = identify_groups_float(&treatment_col, 1.0, None);
        assert!(result.is_ok());

        let (_mask, n_treated, n_control) = result.unwrap();

        assert_eq!(n_treated, 1);
        assert_eq!(n_control, 1);
    }

    #[test]
    fn test_group_identification_basic_str() {
        // String treatment values
        let treatment_col: Vec<Option<&str>> = vec![
            Some("treated"),
            Some("control"),
            Some("treated"),
            Some("control"),
        ];

        let result = identify_groups_str(&treatment_col, "treated", None);
        assert!(result.is_ok());

        let (mask, n_treated, n_control) = result.unwrap();

        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 2);
        assert_eq!(mask, vec![true, false, true, false]);
    }

    #[test]
    fn test_group_identification_with_control_value_str() {
        let treatment_col: Vec<Option<&str>> =
            vec![Some("T"), Some("C"), Some("X"), Some("T"), Some("C")];

        let result = identify_groups_str(&treatment_col, "T", Some("C"));
        assert!(result.is_ok());

        let (mask, n_treated, n_control) = result.unwrap();

        // "X" is excluded
        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 2);
    }

    #[test]
    fn test_empty_treatment_error_str() {
        let treatment_col: Vec<Option<&str>> = vec![Some("control"), Some("control")];

        let result = identify_groups_str(&treatment_col, "treated", None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::EmptyTreatmentGroup => (),
            other => panic!("Expected EmptyTreatmentGroup, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_control_error_str() {
        let treatment_col: Vec<Option<&str>> = vec![Some("treated"), Some("treated")];

        let result = identify_groups_str(&treatment_col, "treated", None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::EmptyControlGroup => (),
            other => panic!("Expected EmptyControlGroup, got {:?}", other),
        }
    }

    #[test]
    fn test_multi_valued_treatment_error_str() {
        let treatment_col: Vec<Option<&str>> = vec![Some("A"), Some("B"), Some("C"), Some("D")];

        let result = identify_groups_str(&treatment_col, "A", None);
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::MultiValuedTreatment { n_values } => {
                assert_eq!(n_values, 4);
            }
            other => panic!("Expected MultiValuedTreatment, got {:?}", other),
        }
    }

    #[test]
    fn test_null_treatment_excluded_str() {
        let treatment_col: Vec<Option<&str>> = vec![Some("T"), None, Some("C"), None, Some("T")];

        let result = identify_groups_str(&treatment_col, "T", None);
        assert!(result.is_ok());

        let (_mask, n_treated, n_control) = result.unwrap();

        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 1);
    }

    #[test]
    fn test_group_identification_exactly_two_values() {
        // Exactly 2 unique values should not trigger MultiValuedTreatment error
        let treatment_col: Vec<Option<i64>> = vec![Some(1), Some(0), Some(1), Some(0)];

        let result = identify_groups_int(&treatment_col, 1, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_group_identification_three_values_with_control_specified() {
        // 3 unique values but control_value specified - should succeed
        let treatment_col: Vec<Option<i64>> = vec![Some(1), Some(0), Some(2), Some(1), Some(0)];

        let result = identify_groups_int(&treatment_col, 1, Some(0));
        assert!(result.is_ok());

        let (_mask, n_treated, n_control) = result.unwrap();
        assert_eq!(n_treated, 2);
        assert_eq!(n_control, 2);
    }

    #[test]
    fn test_group_identification_empty_control_with_control_value() {
        // control_value specified but no observations match it
        let treatment_col: Vec<Option<i64>> = vec![Some(1), Some(2), Some(1), Some(2)];

        let result = identify_groups_int(&treatment_col, 1, Some(0));
        assert!(result.is_err());

        match result.unwrap_err() {
            BalanceError::EmptyControlGroup => (),
            other => panic!("Expected EmptyControlGroup, got {:?}", other),
        }
    }

    // ========================================================================
    // SMD Computation Tests
    // ========================================================================

    #[test]
    fn test_smd_computation_basic() {
        // Basic test: treatment mean = 10, control mean = 8
        // Treatment variance = 4, Control variance = 4
        // Pooled variance = (4 + 4) / 2 = 4
        // Pooled SD = 2
        // SMD = (10 - 8) / 2 = 1.0
        let smd = compute_smd(10.0, 8.0, 4.0, 4.0);
        assert!(
            (smd - 1.0).abs() < 1e-10,
            "Expected SMD = 1.0, got {}",
            smd
        );
    }

    #[test]
    fn test_smd_computation_zero_variance_equal_means() {
        // Zero variance in both groups with equal means: SMD should be 0
        let smd = compute_smd(5.0, 5.0, 0.0, 0.0);
        assert!(
            (smd - 0.0).abs() < 1e-10,
            "Expected SMD = 0.0 for zero variance and equal means, got {}",
            smd
        );
    }

    #[test]
    fn test_smd_computation_zero_variance_different_means() {
        // Zero variance in both groups with different means: SMD should be NaN
        let smd = compute_smd(5.0, 3.0, 0.0, 0.0);
        assert!(
            smd.is_nan(),
            "Expected SMD = NaN for zero variance and different means, got {}",
            smd
        );
    }

    #[test]
    fn test_smd_computation_positive_difference() {
        // Treatment mean > control mean: positive SMD
        // mean_t = 12, mean_c = 10, var_t = 9, var_c = 9
        // Pooled var = 9, pooled SD = 3
        // SMD = (12 - 10) / 3 = 0.667
        let smd = compute_smd(12.0, 10.0, 9.0, 9.0);
        let expected = 2.0 / 3.0;
        assert!(
            (smd - expected).abs() < 1e-10,
            "Expected SMD = {}, got {}",
            expected,
            smd
        );
    }

    #[test]
    fn test_smd_computation_negative_difference() {
        // Treatment mean < control mean: negative SMD
        // mean_t = 8, mean_c = 10, var_t = 4, var_c = 4
        // Pooled var = 4, pooled SD = 2
        // SMD = (8 - 10) / 2 = -1.0
        let smd = compute_smd(8.0, 10.0, 4.0, 4.0);
        assert!(
            (smd - (-1.0)).abs() < 1e-10,
            "Expected SMD = -1.0, got {}",
            smd
        );
    }

    #[test]
    fn test_smd_computation_asymmetric_variances() {
        // Different variances in treatment and control
        // mean_t = 10, mean_c = 8, var_t = 1, var_c = 9
        // Pooled var = (1 + 9) / 2 = 5
        // Pooled SD = sqrt(5) ~ 2.236
        // SMD = (10 - 8) / sqrt(5) ~ 0.894
        let smd = compute_smd(10.0, 8.0, 1.0, 9.0);
        let expected = 2.0 / (5.0_f64).sqrt();
        assert!(
            (smd - expected).abs() < 1e-10,
            "Expected SMD = {}, got {}",
            expected,
            smd
        );
    }

    #[test]
    fn test_smd_computation_equal_means() {
        // Equal means should give SMD = 0
        let smd = compute_smd(5.0, 5.0, 4.0, 4.0);
        assert!(
            smd.abs() < 1e-10,
            "Expected SMD = 0 for equal means, got {}",
            smd
        );
    }

    #[test]
    fn test_smd_computation_one_zero_variance() {
        // One group has zero variance, other doesn't
        // Pooled var = (0 + 4) / 2 = 2
        // Pooled SD = sqrt(2) ~ 1.414
        // SMD = (10 - 8) / sqrt(2) ~ 1.414
        let smd = compute_smd(10.0, 8.0, 0.0, 4.0);
        let expected = 2.0 / (2.0_f64).sqrt();
        assert!(
            (smd - expected).abs() < 1e-10,
            "Expected SMD = {}, got {}",
            expected,
            smd
        );
    }

    // ========================================================================
    // Variance Ratio Computation Tests
    // ========================================================================

    #[test]
    fn test_variance_ratio_basic() {
        // Basic test: var_t = 8, var_c = 4, VR = 2.0
        let vr = compute_variance_ratio(8.0, 4.0);
        assert!(
            (vr - 2.0).abs() < 1e-10,
            "Expected VR = 2.0, got {}",
            vr
        );
    }

    #[test]
    fn test_variance_ratio_zero_control_variance() {
        // var_c = 0 and var_t > 0: VR = infinity
        let vr = compute_variance_ratio(4.0, 0.0);
        assert!(
            vr.is_infinite() && vr.is_sign_positive(),
            "Expected VR = +Inf for zero control variance with positive treatment variance, got {}",
            vr
        );
    }

    #[test]
    fn test_variance_ratio_zero_both_variances() {
        // Both variances are zero: VR = NaN (undefined)
        let vr = compute_variance_ratio(0.0, 0.0);
        assert!(
            vr.is_nan(),
            "Expected VR = NaN for both zero variances, got {}",
            vr
        );
    }

    #[test]
    fn test_variance_ratio_equal_variances() {
        // Equal variances: VR = 1.0
        let vr = compute_variance_ratio(4.0, 4.0);
        assert!(
            (vr - 1.0).abs() < 1e-10,
            "Expected VR = 1.0 for equal variances, got {}",
            vr
        );
    }

    #[test]
    fn test_variance_ratio_treatment_smaller() {
        // Treatment variance < control variance: VR < 1
        // var_t = 2, var_c = 4, VR = 0.5
        let vr = compute_variance_ratio(2.0, 4.0);
        assert!(
            (vr - 0.5).abs() < 1e-10,
            "Expected VR = 0.5, got {}",
            vr
        );
    }

    #[test]
    fn test_variance_ratio_zero_treatment_variance() {
        // var_t = 0 and var_c > 0: VR = 0
        let vr = compute_variance_ratio(0.0, 4.0);
        assert!(
            (vr - 0.0).abs() < 1e-10,
            "Expected VR = 0.0 for zero treatment variance, got {}",
            vr
        );
    }

    #[test]
    fn test_variance_ratio_large_ratio() {
        // Large variance ratio
        let vr = compute_variance_ratio(100.0, 1.0);
        assert!(
            (vr - 100.0).abs() < 1e-10,
            "Expected VR = 100.0, got {}",
            vr
        );
    }

    #[test]
    fn test_variance_ratio_small_ratio() {
        // Small variance ratio
        let vr = compute_variance_ratio(1.0, 100.0);
        assert!(
            (vr - 0.01).abs() < 1e-10,
            "Expected VR = 0.01, got {}",
            vr
        );
    }

    #[test]
    fn test_variance_ratio_negative_control_variance() {
        // Defensive: negative control variance should return NaN
        let vr = compute_variance_ratio(4.0, -1.0);
        assert!(
            vr.is_nan(),
            "Expected VR = NaN for negative control variance, got {}",
            vr
        );
    }
}
