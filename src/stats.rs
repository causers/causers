use pyo3::prelude::*;

/// Result of a linear regression computation.
///
/// This struct contains the results of fitting a linear regression model,
/// including coefficients, standard errors, goodness-of-fit metrics, and
/// optional information about clustering and fixed effects.
///
/// # Fields
///
/// * `coefficients` - Regression coefficients for each covariate
/// * `intercept` - Intercept term (None if `include_intercept=False`)
/// * `r_squared` - Coefficient of determination (R²)
/// * `n_samples` - Number of observations used in the regression
/// * `slope` - Deprecated: Same as `coefficients[0]` for single covariate (backward compatibility)
/// * `standard_errors` - HC3 robust or clustered standard errors for coefficients
/// * `intercept_se` - Standard error for intercept (None if no intercept)
/// * `n_clusters` - Number of unique clusters (None if not clustered)
/// * `cluster_se_type` - Type of clustered SE: "analytical" or "bootstrap" (None if not clustered)
/// * `bootstrap_iterations_used` - Number of bootstrap iterations (None if not bootstrap)
/// * `fixed_effects_absorbed` - Number of groups absorbed per FE variable
/// * `fixed_effects_names` - Column names used for fixed effects
/// * `within_r_squared` - R² computed on demeaned data (within-R²)
#[pyclass]
#[derive(Debug, Clone)]
pub struct LinearRegressionResult {
    /// Regression coefficients for each covariate.
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    /// Intercept term (None if `include_intercept=False`).
    #[pyo3(get)]
    pub intercept: Option<f64>,
    /// Coefficient of determination (R²).
    #[pyo3(get)]
    pub r_squared: f64,
    /// Number of observations used in the regression.
    #[pyo3(get)]
    pub n_samples: usize,
    /// Deprecated: Use `coefficients[0]` instead. Kept for backward compatibility.
    #[pyo3(get)]
    pub slope: Option<f64>,
    /// HC3 robust standard errors for each coefficient (or clustered SE if cluster specified).
    #[pyo3(get)]
    pub standard_errors: Vec<f64>,
    /// HC3 robust standard error for intercept (None if `include_intercept=False`).
    #[pyo3(get)]
    pub intercept_se: Option<f64>,
    /// Number of unique clusters (None if not clustered).
    #[pyo3(get)]
    pub n_clusters: Option<usize>,
    /// Type of clustered SE: "analytical" or "bootstrap" (None if not clustered).
    #[pyo3(get)]
    pub cluster_se_type: Option<String>,
    /// Number of bootstrap iterations used (None if not bootstrap).
    #[pyo3(get)]
    pub bootstrap_iterations_used: Option<usize>,
    /// Number of groups absorbed per FE variable (e.g., [100, 10] for firm+year).
    #[pyo3(get)]
    pub fixed_effects_absorbed: Option<Vec<usize>>,
    /// Column names used for fixed effects (e.g., ["firm_id", "year"]).
    #[pyo3(get)]
    pub fixed_effects_names: Option<Vec<String>>,
    /// R² computed on demeaned data (within-R²).
    #[pyo3(get)]
    pub within_r_squared: Option<f64>,
}

#[pymethods]
impl LinearRegressionResult {
    /// Python repr() implementation.
    ///
    /// Returns a detailed representation of the regression result including
    /// all fields and their values.
    fn __repr__(&self) -> String {
        let intercept_str = self
            .intercept
            .map_or_else(|| "None".to_string(), |i| format!("{:.6}", i));
        let intercept_se_str = self
            .intercept_se
            .map_or_else(|| "None".to_string(), |se| format!("{:.6}", se));
        let n_clusters_str = self
            .n_clusters
            .map_or_else(|| "None".to_string(), |n| n.to_string());
        let cluster_se_type_str = self
            .cluster_se_type
            .as_ref()
            .map_or_else(|| "None".to_string(), |s| format!("\"{}\"", s));
        let bootstrap_iter_str = self
            .bootstrap_iterations_used
            .map_or_else(|| "None".to_string(), |b| b.to_string());
        let fe_absorbed_str = self
            .fixed_effects_absorbed
            .as_ref()
            .map_or_else(|| "None".to_string(), |v| format!("{:?}", v));
        let fe_names_str = self
            .fixed_effects_names
            .as_ref()
            .map_or_else(|| "None".to_string(), |v| format!("{:?}", v));
        let within_r2_str = self
            .within_r_squared
            .map_or_else(|| "None".to_string(), |r2| format!("{:.6}", r2));

        format!(
            "LinearRegressionResult(coefficients={:?}, intercept={}, r_squared={:.6}, n_samples={}, standard_errors={:?}, intercept_se={}, n_clusters={}, cluster_se_type={}, bootstrap_iterations_used={}, fixed_effects_absorbed={}, fixed_effects_names={}, within_r_squared={})",
            self.coefficients,
            intercept_str,
            self.r_squared,
            self.n_samples,
            self.standard_errors,
            intercept_se_str,
            n_clusters_str,
            cluster_se_type_str,
            bootstrap_iter_str,
            fe_absorbed_str,
            fe_names_str,
            within_r2_str
        )
    }

    /// Python str() implementation.
    ///
    /// Returns a human-readable equation representation of the regression.
    /// For single covariate: `y = β₀ ± SE*x + intercept`
    /// For multiple covariates: `y = β₁*x1 + β₂*x2 + ... + intercept`
    fn __str__(&self) -> String {
        let intercept_str = self
            .intercept
            .map_or_else(String::new, |i| format!(" + {:.6}", i));

        if self.coefficients.len() == 1 {
            // For single covariate, show coefficient with SE
            let se_str = if !self.standard_errors.is_empty() {
                format!(" ± {:.6}", self.standard_errors[0])
            } else {
                String::new()
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
