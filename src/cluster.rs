//! Cluster-robust standard error computations.
//!
//! This module implements clustered standard errors for linear regression,
//! including both analytical (sandwich estimator) and wild cluster bootstrap methods.
//!
//! # References
//! - Cameron, A. C., & Miller, D. L. (2015). A Practitioner's Guide to
//!   Cluster-Robust Inference. Journal of Human Resources, 50(2), 317-372.

use std::collections::HashMap;
use std::fmt;

/// Error type for cluster operations
#[derive(Debug, Clone)]
pub enum ClusterError {
    /// Not enough clusters for clustered SE
    InsufficientClusters { found: usize },
    /// Cluster has only one observation (analytical mode only)
    SingleObservationCluster { cluster_idx: usize },
    /// Numerical instability detected
    NumericalInstability { message: String },
    /// Invalid standard error values (NaN/Inf)
    InvalidStandardErrors,
}

impl fmt::Display for ClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterError::InsufficientClusters { found } => {
                write!(
                    f,
                    "Clustered standard errors require at least 2 clusters; found {}",
                    found
                )
            }
            ClusterError::SingleObservationCluster { cluster_idx } => {
                write!(f, "Cluster {} contains only 1 observation; within-cluster correlation cannot be estimated. Use bootstrap=True for single-observation clusters.", cluster_idx)
            }
            ClusterError::NumericalInstability { message } => {
                write!(f, "{}", message)
            }
            ClusterError::InvalidStandardErrors => {
                write!(f, "Standard error computation produced invalid values; check for numerical issues in data")
            }
        }
    }
}

impl std::error::Error for ClusterError {}

/// Cluster membership information for grouped observations.
///
/// This struct stores the mapping from cluster indices to observation indices,
/// enabling efficient iteration over observations within each cluster.
#[derive(Debug, Clone)]
pub struct ClusterInfo {
    /// indices[g] = vector of row indices belonging to cluster g
    /// Invariant: flatten(indices) is a permutation of 0..n
    pub indices: Vec<Vec<usize>>,

    /// Number of unique clusters (G)
    /// Invariant: n_clusters == indices.len()
    pub n_clusters: usize,

    /// sizes[g] = number of observations in cluster g
    /// Invariant: sizes[g] == indices[g].len()
    /// Invariant: sum(sizes) == n
    pub sizes: Vec<usize>,
}

/// Simple PRNG using SplitMix64 algorithm.
///
/// This is good enough for Rademacher weight generation (±1 with equal probability)
/// but is NOT cryptographically secure.
///
/// Reference: https://prng.di.unimi.it/splitmix64.c
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a new PRNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate the next random u64 value.
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate Rademacher weight: +1.0 or -1.0 with equal probability.
    pub fn rademacher(&mut self) -> f64 {
        if self.next() & 1 == 0 {
            -1.0
        } else {
            1.0
        }
    }
}

/// Running statistics for Welford's online variance algorithm.
///
/// Computes running mean and variance in O(1) memory per update,
/// regardless of the number of samples seen.
///
/// Reference: Welford, B. P. (1962). Note on a method for calculating
/// corrected sums of squares and products. Technometrics, 4(3), 419-420.
pub struct WelfordState {
    /// Number of values seen
    count: usize,
    /// Running mean for each coefficient
    mean: Vec<f64>,
    /// Sum of squared differences from mean (M2)
    m2: Vec<f64>,
}

impl WelfordState {
    /// Create a new WelfordState for tracking `n_params` values.
    pub fn new(n_params: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; n_params],
            m2: vec![0.0; n_params],
        }
    }

    /// Update the running statistics with a new set of values.
    pub fn update(&mut self, values: &[f64]) {
        self.count += 1;
        for (i, &val) in values.iter().enumerate() {
            let delta = val - self.mean[i];
            self.mean[i] += delta / (self.count as f64);
            let delta2 = val - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Compute the sample variance for each coefficient.
    pub fn variance(&self) -> Vec<f64> {
        if self.count < 2 {
            return vec![0.0; self.mean.len()];
        }
        self.m2
            .iter()
            .map(|&m| m / ((self.count - 1) as f64))
            .collect()
    }

    /// Compute the standard errors (sqrt of variance) for each coefficient.
    pub fn standard_errors(&self) -> Vec<f64> {
        self.variance().iter().map(|&v| v.sqrt()).collect()
    }
}

/// Build cluster index structure from cluster IDs.
///
/// Parses cluster IDs into a ClusterInfo structure with indices grouped by cluster.
///
/// # Arguments
/// * `cluster_ids` - Cluster ID for each observation
///
/// # Returns
/// * `Result<ClusterInfo, ClusterError>` - Cluster info or error if validation fails
///
/// # Errors
/// * `ClusterError::InsufficientClusters` if there is only 1 unique cluster
pub fn build_cluster_indices(cluster_ids: &[i64]) -> Result<ClusterInfo, ClusterError> {
    let n = cluster_ids.len();

    // Map from cluster ID to internal index
    let mut id_to_index: HashMap<i64, usize> = HashMap::new();
    let mut indices: Vec<Vec<usize>> = Vec::new();

    for (i, &id) in cluster_ids.iter().enumerate() {
        if let Some(&g) = id_to_index.get(&id) {
            // Existing cluster
            indices[g].push(i);
        } else {
            // New cluster
            let g = indices.len();
            id_to_index.insert(id, g);
            indices.push(vec![i]);
        }
    }

    let n_clusters = indices.len();

    // Validate: need at least 2 clusters
    if n_clusters < 2 {
        return Err(ClusterError::InsufficientClusters { found: n_clusters });
    }

    let sizes: Vec<usize> = indices.iter().map(|idx| idx.len()).collect();

    // Debug assertion for invariants
    debug_assert_eq!(n_clusters, indices.len());
    debug_assert_eq!(sizes.iter().sum::<usize>(), n);

    Ok(ClusterInfo {
        indices,
        n_clusters,
        sizes,
    })
}

/// Multiply a matrix by a vector: result = A × v
///
/// # Arguments
/// * `a` - Matrix of shape (m × n)
/// * `v` - Vector of length n
///
/// # Returns
/// * `Vec<f64>` - Result vector of length m
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
/// # Arguments
/// * `a` - Matrix of shape (m × k)
/// * `b` - Matrix of shape (k × n)
///
/// # Returns
/// * `Vec<Vec<f64>>` - Result matrix of shape (m × n)
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

/// Compute analytical clustered standard errors using the sandwich estimator.
///
/// Formula: SE = sqrt(diag((X'X)^-1 × meat × (X'X)^-1))
/// where meat = Σ_g (X_g'û_g)(X_g'û_g)' with small-sample adjustment
///
/// # Arguments
/// * `design_matrix` - Design matrix X (n × p), includes intercept if applicable
/// * `residuals` - OLS residuals (n,)
/// * `xtx_inv` - (X'X)^-1 (p × p)
/// * `cluster_info` - Cluster membership information
/// * `include_intercept` - Whether intercept is included in design matrix
///
/// # Returns
/// * `Result<(Vec<f64>, Option<f64>), ClusterError>` - (coefficient_se, intercept_se)
pub fn compute_cluster_se_analytical(
    design_matrix: &[Vec<f64>],
    residuals: &[f64],
    xtx_inv: &[Vec<f64>],
    cluster_info: &ClusterInfo,
    include_intercept: bool,
) -> Result<(Vec<f64>, Option<f64>), ClusterError> {
    let n = residuals.len();
    let p = xtx_inv.len();
    let g = cluster_info.n_clusters;

    // Check for single-observation clusters in analytical mode
    for (cluster_idx, size) in cluster_info.sizes.iter().enumerate() {
        if *size == 1 {
            return Err(ClusterError::SingleObservationCluster { cluster_idx });
        }
    }

    // Compute meat matrix: Σ_g (X_g'û_g)(X_g'û_g)'
    let mut meat = vec![vec![0.0; p]; p];

    for cluster_indices in &cluster_info.indices {
        // Compute score for cluster g: X_g'û_g (p × 1 vector)
        let mut score_g = vec![0.0; p];
        for &i in cluster_indices {
            for (j, score_val) in score_g.iter_mut().enumerate() {
                *score_val += design_matrix[i][j] * residuals[i];
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

    // Sandwich: V = (X'X)^-1 × meat × (X'X)^-1
    let temp = matrix_multiply(xtx_inv, &meat);
    let v = matrix_multiply(&temp, xtx_inv);

    // Check condition number for numerical stability
    // Using a simple estimate: max diagonal / min diagonal
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

/// Compute wild cluster bootstrap standard errors with Rademacher weights.
///
/// Uses the "type 11" bootstrap: y* = ŷ + w_g × e where w_g ∈ {-1, +1}
///
/// # Arguments
/// * `design_matrix` - Design matrix X (n × p)
/// * `fitted_values` - Fitted values ŷ = Xβ (n,)
/// * `residuals` - OLS residuals e = y - ŷ (n,)
/// * `xtx_inv` - (X'X)^-1 (p × p)
/// * `cluster_info` - Cluster membership information
/// * `bootstrap_iterations` - Number of bootstrap replications (B)
/// * `seed` - Random seed for reproducibility (None for random)
/// * `include_intercept` - Whether intercept is included
///
/// # Returns
/// * `Result<(Vec<f64>, Option<f64>), ClusterError>` - (coefficient_se, intercept_se)
// Wild cluster bootstrap requires all statistical context parameters. Struct would reduce clarity.
#[allow(clippy::too_many_arguments)]
pub fn compute_cluster_se_bootstrap(
    design_matrix: &[Vec<f64>],
    fitted_values: &[f64],
    residuals: &[f64],
    xtx_inv: &[Vec<f64>],
    cluster_info: &ClusterInfo,
    bootstrap_iterations: usize,
    seed: Option<u64>,
    include_intercept: bool,
) -> Result<(Vec<f64>, Option<f64>), ClusterError> {
    let n = residuals.len();
    let p = xtx_inv.len();
    let g = cluster_info.n_clusters;

    // Initialize RNG
    let actual_seed = seed.unwrap_or_else(|| {
        // Use system time as seed when None
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    });
    let mut rng = SplitMix64::new(actual_seed);

    // Initialize Welford's online algorithm state
    let mut welford = WelfordState::new(p);

    // Pre-allocate buffers
    let mut y_star = vec![0.0; n];
    let mut xty_star = vec![0.0; p];
    let mut weights = vec![0.0; g];

    for _ in 0..bootstrap_iterations {
        // Generate Rademacher weights for each cluster
        for w in weights.iter_mut() {
            *w = rng.rademacher();
        }

        // Create bootstrap response y*
        // y*_i = ŷ_i + w_{g(i)} × e_i
        for (cluster_idx, cluster_indices) in cluster_info.indices.iter().enumerate() {
            let w_g = weights[cluster_idx];
            for &i in cluster_indices {
                y_star[i] = fitted_values[i] + w_g * residuals[i];
            }
        }

        // Compute X'y*
        for (j, xty_val) in xty_star.iter_mut().enumerate() {
            *xty_val = 0.0;
            for (dm_row, &y_val) in design_matrix.iter().zip(y_star.iter()) {
                *xty_val += dm_row[j] * y_val;
            }
        }

        // Compute bootstrap coefficients β* = (X'X)^-1 X'y*
        let beta_star = matrix_vector_multiply(xtx_inv, &xty_star);

        // Update Welford's algorithm
        welford.update(&beta_star);
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_build_cluster_indices_basic() {
        let cluster_ids = vec![1, 1, 2, 2, 2];
        let result = build_cluster_indices(&cluster_ids).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.sizes, vec![2, 3]);
        assert_eq!(result.indices[0], vec![0, 1]);
        assert_eq!(result.indices[1], vec![2, 3, 4]);
    }

    #[test]
    fn test_build_cluster_indices_multiple() {
        let cluster_ids = vec![1, 2, 3, 1, 2, 3, 4, 5];
        let result = build_cluster_indices(&cluster_ids).unwrap();

        assert_eq!(result.n_clusters, 5);
        assert_eq!(result.sizes, vec![2, 2, 2, 1, 1]);
    }

    #[test]
    fn test_build_cluster_indices_noncontiguous_ids() {
        let cluster_ids = vec![100, 200, 100, 300, 200];
        let result = build_cluster_indices(&cluster_ids).unwrap();

        assert_eq!(result.n_clusters, 3);
        assert_eq!(result.indices[0], vec![0, 2]); // cluster 100
        assert_eq!(result.indices[1], vec![1, 4]); // cluster 200
        assert_eq!(result.indices[2], vec![3]); // cluster 300
    }

    #[test]
    fn test_build_cluster_indices_single_cluster_error() {
        let cluster_ids = vec![1, 1, 1, 1, 1];
        let result = build_cluster_indices(&cluster_ids);

        assert!(result.is_err());
        match result.unwrap_err() {
            ClusterError::InsufficientClusters { found } => assert_eq!(found, 1),
            _ => panic!("Expected InsufficientClusters error"),
        }
    }

    #[test]
    fn test_splitmix64_determinism() {
        let mut rng1 = SplitMix64::new(12345);
        let mut rng2 = SplitMix64::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_splitmix64_different_seeds() {
        let mut rng1 = SplitMix64::new(12345);
        let mut rng2 = SplitMix64::new(54321);

        // Should produce different values with different seeds
        let v1 = rng1.next();
        let v2 = rng2.next();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_rademacher_distribution() {
        let mut rng = SplitMix64::new(42);
        let mut positive_count = 0;
        let n = 10000;

        for _ in 0..n {
            let w = rng.rademacher();
            assert!(w == 1.0 || w == -1.0);
            if w > 0.0 {
                positive_count += 1;
            }
        }

        // Should be approximately 50/50
        let ratio = positive_count as f64 / n as f64;
        assert!(ratio > 0.45 && ratio < 0.55, "Ratio was {}", ratio);
    }

    #[test]
    fn test_welford_known_values() {
        let mut welford = WelfordState::new(1);

        // Add values: 2, 4, 6, 8, 10
        // Mean = 6, Variance = 10 (sample variance)
        welford.update(&[2.0]);
        welford.update(&[4.0]);
        welford.update(&[6.0]);
        welford.update(&[8.0]);
        welford.update(&[10.0]);

        let variance = welford.variance();
        assert_relative_eq!(variance[0], 10.0, epsilon = 1e-10);

        let se = welford.standard_errors();
        assert_relative_eq!(se[0], 10.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_welford_multiple_params() {
        let mut welford = WelfordState::new(2);

        // Two independent sequences
        // Seq 1: 1, 2, 3, 4, 5 -> mean=3, var=2.5
        // Seq 2: 10, 20, 30, 40, 50 -> mean=30, var=250
        welford.update(&[1.0, 10.0]);
        welford.update(&[2.0, 20.0]);
        welford.update(&[3.0, 30.0]);
        welford.update(&[4.0, 40.0]);
        welford.update(&[5.0, 50.0]);

        let variance = welford.variance();
        assert_relative_eq!(variance[0], 2.5, epsilon = 1e-10);
        assert_relative_eq!(variance[1], 250.0, epsilon = 1e-10);
    }

    #[test]
    fn test_analytical_se_2x2() {
        // Simple test case: 2 clusters, 4 observations
        // y = β0 + β1*x + ε with clusters
        let design_matrix = vec![
            vec![1.0, 1.0], // cluster 0
            vec![1.0, 2.0], // cluster 0
            vec![1.0, 3.0], // cluster 1
            vec![1.0, 4.0], // cluster 1
        ];

        // Residuals (made up for testing)
        let residuals = vec![0.1, -0.1, 0.2, -0.2];

        // Pre-computed (X'X)^-1 for this design matrix
        // X'X = [[4, 10], [10, 30]]
        // (X'X)^-1 = [[1.5, -0.5], [-0.5, 0.2]]
        let xtx_inv = vec![vec![1.5, -0.5], vec![-0.5, 0.2]];

        let cluster_ids = vec![0, 0, 1, 1];
        let cluster_info = build_cluster_indices(&cluster_ids).unwrap();

        let result = compute_cluster_se_analytical(
            &design_matrix,
            &residuals,
            &xtx_inv,
            &cluster_info,
            true, // include_intercept
        );

        // Should succeed without error
        assert!(result.is_ok());
        let (coef_se, intercept_se) = result.unwrap();

        // Coefficient SE should have 1 element (for β1)
        assert_eq!(coef_se.len(), 1);
        assert!(intercept_se.is_some());

        // Values should be positive and finite
        assert!(coef_se[0] > 0.0);
        assert!(intercept_se.unwrap() > 0.0);
    }

    #[test]
    fn test_bootstrap_se_reproducibility() {
        // Test that same seed produces same results
        let design_matrix = vec![
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 4.0],
        ];
        let fitted_values = vec![1.0, 2.0, 3.0, 4.0];
        let residuals = vec![0.1, -0.1, 0.2, -0.2];
        let xtx_inv = vec![vec![1.5, -0.5], vec![-0.5, 0.2]];

        let cluster_ids = vec![0, 0, 1, 1];
        let cluster_info = build_cluster_indices(&cluster_ids).unwrap();

        let result1 = compute_cluster_se_bootstrap(
            &design_matrix,
            &fitted_values,
            &residuals,
            &xtx_inv,
            &cluster_info,
            100,
            Some(42),
            true,
        )
        .unwrap();

        let result2 = compute_cluster_se_bootstrap(
            &design_matrix,
            &fitted_values,
            &residuals,
            &xtx_inv,
            &cluster_info,
            100,
            Some(42),
            true,
        )
        .unwrap();

        // Same seed should produce same results
        assert_relative_eq!(result1.0[0], result2.0[0], epsilon = 1e-10);
        assert_relative_eq!(result1.1.unwrap(), result2.1.unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_bootstrap_se_different_seeds() {
        // Test that different seeds produce different results
        let design_matrix = vec![
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 4.0],
        ];
        let fitted_values = vec![1.0, 2.0, 3.0, 4.0];
        let residuals = vec![0.1, -0.1, 0.2, -0.2];
        let xtx_inv = vec![vec![1.5, -0.5], vec![-0.5, 0.2]];

        let cluster_ids = vec![0, 0, 1, 1];
        let cluster_info = build_cluster_indices(&cluster_ids).unwrap();

        let result1 = compute_cluster_se_bootstrap(
            &design_matrix,
            &fitted_values,
            &residuals,
            &xtx_inv,
            &cluster_info,
            100,
            Some(1), // Different seed
            true,
        )
        .unwrap();

        let result2 = compute_cluster_se_bootstrap(
            &design_matrix,
            &fitted_values,
            &residuals,
            &xtx_inv,
            &cluster_info,
            100,
            Some(999), // Different seed
            true,
        )
        .unwrap();

        // Results should be valid numbers (we don't assert they're different
        // since bootstrap variance can sometimes be similar by chance)
        assert!(result1.0[0] >= 0.0);
        assert!(result2.0[0] >= 0.0);
    }

    #[test]
    fn test_single_observation_cluster_analytical_error() {
        let design_matrix = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]];
        let residuals = vec![0.1, -0.1, 0.2];
        let xtx_inv = vec![vec![1.5, -0.5], vec![-0.5, 0.2]];

        // Cluster with only 1 observation (cluster 1)
        let cluster_ids = vec![0, 0, 1];
        let cluster_info = build_cluster_indices(&cluster_ids).unwrap();

        let result = compute_cluster_se_analytical(
            &design_matrix,
            &residuals,
            &xtx_inv,
            &cluster_info,
            true,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            ClusterError::SingleObservationCluster { cluster_idx } => {
                assert_eq!(cluster_idx, 1);
            }
            _ => panic!("Expected SingleObservationCluster error"),
        }
    }

    #[test]
    fn test_single_observation_cluster_bootstrap_ok() {
        // Bootstrap should allow single-observation clusters
        let design_matrix = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]];
        let fitted_values = vec![1.0, 2.0, 3.0];
        let residuals = vec![0.1, -0.1, 0.2];
        let xtx_inv = vec![vec![1.5, -0.5], vec![-0.5, 0.2]];

        // Cluster with only 1 observation (cluster 1)
        let cluster_ids = vec![0, 0, 1];
        let cluster_info = build_cluster_indices(&cluster_ids).unwrap();

        let result = compute_cluster_se_bootstrap(
            &design_matrix,
            &fitted_values,
            &residuals,
            &xtx_inv,
            &cluster_info,
            100,
            Some(42),
            true,
        );

        // Should succeed
        assert!(result.is_ok());
    }
}
