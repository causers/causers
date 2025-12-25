# Score Bootstrap for Logistic Regression

This document describes the score bootstrap methodology used in `causers` for computing clustered standard errors in logistic regression.

## Overview

The score bootstrap is a resampling technique specifically designed for clustered data in non-linear models like logistic regression. Unlike the wild cluster bootstrap used for linear regression, the score bootstrap operates on the score function (first derivative of the log-likelihood), making it appropriate for maximum likelihood estimators.

## Why Score Bootstrap for Logistic Regression?

For **linear regression**, the wild cluster bootstrap is appropriate because:
- The model is linear: y = Xβ + ε
- Residuals can be resampled directly
- The estimator is a closed-form function of the residuals

For **logistic regression**, the wild cluster bootstrap is **not appropriate** because:
- The model is non-linear: P(y=1|x) = 1/(1+exp(-x'β))
- Coefficients are estimated via iterative MLE (Newton-Raphson)
- Resampling residuals would require re-fitting the model each iteration

The **score bootstrap** solves this by:
- Operating on the score (gradient) function
- Avoiding model re-estimation in each bootstrap iteration
- Providing asymptotically valid inference for MLE estimators

## Algorithm

### Step 1: Estimate the Model

Fit the logistic regression model to obtain:
- MLE coefficients: β̂
- Information matrix inverse: I⁻¹ = (X'WX)⁻¹
- Predicted probabilities: π̂ᵢ = 1/(1+exp(-xᵢ'β̂))

### Step 2: Compute Cluster-Level Scores

For each cluster g, compute the cluster score:

```
Sᵍ = Σᵢ∈ᵍ xᵢ(yᵢ - π̂ᵢ)
```

where the sum is over all observations i belonging to cluster g.

### Step 3: Bootstrap Loop

For b = 1, ..., B (default B = 1000):

1. **Generate Rademacher weights** for each cluster:
   ```
   wᵍ ∈ {-1, +1} with probability 0.5 each
   ```

2. **Compute perturbed score**:
   ```
   S* = Σᵍ wᵍ Sᵍ
   ```

3. **Compute coefficient perturbation**:
   ```
   δ* = I⁻¹ S*
   ```

4. **Update running variance** using Welford's online algorithm

### Step 4: Extract Standard Errors

Compute standard errors from the bootstrap distribution of δ*:

```
SE(β̂ⱼ) = √(Var(δ*ⱼ))
```

## Rademacher Weights

The score bootstrap uses **Rademacher weights**:

```
wᵍ = { +1  with probability 0.5
     { -1  with probability 0.5
```

### Why Rademacher?

1. **Simplicity**: Binary weights are computationally efficient
2. **Symmetry**: Equal probability of +1 and -1 ensures E[wᵍ] = 0
3. **Variance**: Var(wᵍ) = 1, which preserves the variance structure
4. **Theory**: Kline & Santos (2012) show these provide valid inference

### Alternative: Mammen Weights

Mammen (1993) proposed an alternative two-point distribution:
```
wᵍ = { (√5 + 1)/2     with probability (√5 - 1)/(2√5)
     { -(√5 - 1)/2    with probability (√5 + 1)/(2√5)
```

The `causers` package **does not implement Mammen weights** for the following reasons:

1. **Simplicity**: Rademacher weights are simpler to implement and explain
2. **Performance**: No significant improvement in simulation studies for logistic regression
3. **Kline & Santos recommendation**: Their paper uses Rademacher weights

## Mathematical Foundation

The score bootstrap is based on the following asymptotic result:

For a maximum likelihood estimator with clustered data, the score-based perturbation:

```
δ* = I⁻¹ Σᵍ wᵍ Sᵍ
```

has the same asymptotic distribution as:

```
√n (β̂ - β₀)
```

under the null hypothesis, where β₀ is the true parameter value.

This result holds because:
1. The score evaluated at the true parameter has mean zero
2. The cluster-level scores are independent across clusters
3. The information matrix consistently estimates the variance of the score

## Implementation Details

### Welford's Algorithm

To avoid storing all B bootstrap replicates, `causers` uses Welford's online algorithm for computing variance:

```rust
struct WelfordState {
    count: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,  // Sum of squared deviations
}

fn update(&mut self, delta: &[f64]) {
    self.count += 1;
    for i in 0..delta.len() {
        let diff = delta[i] - self.mean[i];
        self.mean[i] += diff / self.count as f64;
        self.m2[i] += diff * (delta[i] - self.mean[i]);
    }
}

fn standard_errors(&self) -> Vec<f64> {
    self.m2.iter()
        .map(|&m| (m / self.count as f64).sqrt())
        .collect()
}
```

This provides O(k) memory usage instead of O(B × k).

### Random Number Generation

For reproducibility, `causers` uses the SplitMix64 PRNG:
- Fast and high-quality
- Seeded from user-provided seed or system entropy
- Same seed produces identical results

## Usage Example

```python
import polars as pl
from causers import logistic_regression

df = pl.DataFrame({
    "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    "y": [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    "firm_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
})

# Score bootstrap with 1000 iterations
result = logistic_regression(
    df, "x", "y",
    cluster="firm_id",
    bootstrap=True,
    bootstrap_iterations=1000,
    seed=42  # For reproducibility
)

print(f"Coefficient: {result.coefficients[0]:.4f}")
print(f"Bootstrap SE: {result.standard_errors[0]:.4f}")
print(f"Method: {result.cluster_se_type}")
```

## When to Use Score Bootstrap

The score bootstrap is recommended when:

1. **Small number of clusters** (G < 42): Analytical clustered SE can be unreliable with few clusters. The 42-cluster threshold is based on simulation evidence.

2. **Unbalanced clusters**: When cluster sizes vary significantly, bootstrap methods tend to perform better.

3. **Reproducible inference**: With a fixed seed, bootstrap results are exactly reproducible.

## Comparison with Analytical Clustered SE

| Aspect | Analytical | Score Bootstrap |
|--------|------------|-----------------|
| Computation | Faster | Slower (B iterations) |
| Small clusters | Unreliable for G < 42 | Robust |
| Reproducibility | Deterministic | Requires seed |
| Small sample adjustment | G/(G-1) × (n-1)/(n-k) | Built into resampling |

## References

**Kline, P., & Santos, A. (2012).** "A Score Based Approach to Wild Bootstrap Inference."
*Journal of Econometric Methods*, 1(1), 23-41.
https://doi.org/10.1515/2156-6674.1006

This paper:
- Develops the theoretical foundation for score-based bootstrap
- Proves asymptotic validity for MLE estimators
- Provides simulation evidence for logistic regression
- Recommends Rademacher weights

**Cameron, A. C., & Miller, D. L. (2015).** "A Practitioner's Guide to Cluster-Robust Inference."
*Journal of Human Resources*, 50(2), 317-372.

General reference for clustered standard errors and when to use bootstrap methods.

**Mammen, E. (1993).** "Bootstrap and Wild Bootstrap for High Dimensional Linear Models."
*The Annals of Statistics*, 21(1), 255-285.

Original paper on wild bootstrap weights (Mammen vs Rademacher).

## Technical Notes

### Numerical Stability

The implementation includes several safeguards:
- Weights in IRLS are floored at 1e-10 to prevent division by zero
- Condition number check on the information matrix
- NaN/Inf detection in output

### Memory Efficiency

Memory usage is O(n × k + G × k) where:
- n = number of observations
- k = number of parameters
- G = number of clusters

The B bootstrap iterations do not increase memory usage due to Welford's algorithm.
