#!/usr/bin/env python3
"""
Performance benchmark: causers vs reference packages.

Compares:
- linear_regression vs statsmodels.OLS
- logistic_regression vs statsmodels.Logit
- synthetic_did vs azcausal.SDID

Target: causers should be faster than reference packages.

Usage:
    python tests/test_benchmark_performance.py
"""

import sys
import time
import warnings
from typing import Callable, Dict, Any, List, Tuple, Optional

import numpy as np
import polars as pl


# ============================================================
# Timing utilities
# ============================================================

def time_function(
    func: Callable,
    *args,
    n_iter: int = 5,
    warmup: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """Time function execution with warmup.
    
    Args:
        func: Function to time
        n_iter: Number of timed iterations
        warmup: Number of warmup iterations (discarded)
    
    Returns:
        Dict with 'result', 'median_ms', 'min_ms', 'max_ms'
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    result = None
    for _ in range(n_iter):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    
    return {
        "result": result,
        "median_ms": np.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


# ============================================================
# Data generation
# ============================================================

SEED = 42


def generate_linear_data(n_samples: int, seed: int = SEED) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Generate linear regression test data (single variable)."""
    np.random.seed(seed)
    x = np.random.randn(n_samples)
    y = 2 * x + 1 + np.random.randn(n_samples) * 0.5
    df = pl.DataFrame({"x": x, "y": y})
    return df, x, y


def generate_linear_regression_data(
    n_obs: int,
    n_vars: int,
    cluster_type: Optional[str] = None,
    seed: int = SEED
) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Generate data for comprehensive linear regression benchmark.
    
    Args:
        n_obs: Number of observations
        n_vars: Number of control variables
        cluster_type: None for HC3, "balanced" for balanced clusters, "imbalanced" for imbalanced
        seed: Random seed
    
    Returns:
        Tuple of (polars DataFrame, X numpy array, y numpy array, cluster array or None)
    """
    np.random.seed(seed)
    
    # Generate X variables
    x_data = {f"x{i}": np.random.randn(n_obs) for i in range(n_vars)}
    
    # Generate y with some relationship to X
    y = sum(x_data.values()) + np.random.randn(n_obs) * 0.5
    
    df = pl.DataFrame({"y": y, **x_data})
    
    # Create numpy X array for statsmodels
    X = np.column_stack(list(x_data.values()))
    
    cluster_ids = None
    if cluster_type is not None:
        n_clusters = 100
        if cluster_type == "balanced":
            # Equal observations per cluster
            cluster_ids = np.repeat(range(n_clusters), n_obs // n_clusters)
            # Handle remainder
            remainder = n_obs - len(cluster_ids)
            if remainder > 0:
                cluster_ids = np.concatenate([cluster_ids, np.arange(remainder)])
        else:  # imbalanced
            # Some clusters have 5x more observations
            cluster_sizes = []
            for i in range(n_clusters):
                size = n_obs // n_clusters
                if i < 10:  # First 10 clusters are 5x larger
                    size *= 5
                cluster_sizes.append(size)
            # Normalize to sum to n_obs
            cluster_sizes = np.array(cluster_sizes)
            cluster_sizes = (cluster_sizes / cluster_sizes.sum() * n_obs).astype(int)
            cluster_sizes[-1] = n_obs - cluster_sizes[:-1].sum()  # Ensure exact sum
            cluster_ids = np.repeat(range(n_clusters), cluster_sizes)
        
        df = df.with_columns(pl.Series("cluster", cluster_ids[:n_obs]))
    
    return df, X, np.array(y), cluster_ids


# Linear regression benchmark configurations
# (n_obs, n_vars, se_type, cluster_type, label)
LINEAR_REGRESSION_CONFIGS = [
    # Vary observations (with 2 variables, HC3)
    (1000, 2, "hc3", None, "1K obs, 2 vars, HC3"),
    (10000, 2, "hc3", None, "10K obs, 2 vars, HC3"),
    (100000, 2, "hc3", None, "100K obs, 2 vars, HC3"),
    
    # Vary variables (with 10K observations, HC3)
    (10000, 2, "hc3", None, "10K obs, 2 vars, HC3"),
    (10000, 10, "hc3", None, "10K obs, 10 vars, HC3"),
    (10000, 50, "hc3", None, "10K obs, 50 vars, HC3"),
    
    # Vary SE type (with 10K observations, 10 variables)
    (10000, 10, "hc3", None, "10K obs, 10 vars, HC3"),
    (10000, 10, "cluster", "balanced", "10K obs, 10 vars, Cluster (balanced)"),
    (10000, 10, "cluster", "imbalanced", "10K obs, 10 vars, Cluster (imbalanced)"),
]


# Logistic regression benchmark configurations
# (n_obs, n_vars, se_type, cluster_type, label)
LOGISTIC_REGRESSION_CONFIGS = [
    # Vary observations (with 2 variables, HC3)
    (1000, 2, "hc3", None, "1K obs, 2 vars, HC3"),
    (10000, 2, "hc3", None, "10K obs, 2 vars, HC3"),
    (100000, 2, "hc3", None, "100K obs, 2 vars, HC3"),
    
    # Vary variables (with 10K observations, HC3)
    (10000, 2, "hc3", None, "10K obs, 2 vars, HC3"),
    (10000, 10, "hc3", None, "10K obs, 10 vars, HC3"),
    (10000, 50, "hc3", None, "10K obs, 50 vars, HC3"),
    
    # Vary SE type (with 10K observations, 10 variables)
    (10000, 10, "hc3", None, "10K obs, 10 vars, HC3"),
    (10000, 10, "cluster", "balanced", "10K obs, 10 vars, Cluster (balanced)"),
    (10000, 10, "cluster", "imbalanced", "10K obs, 10 vars, Cluster (imbalanced)"),
]


def generate_logistic_data(n_samples: int, seed: int = SEED) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Generate logistic regression test data."""
    np.random.seed(seed)
    x = np.random.randn(n_samples)
    prob = 1 / (1 + np.exp(-(0.5 + x)))
    y = (np.random.rand(n_samples) < prob).astype(float)
    df = pl.DataFrame({"x": x, "y": y})
    return df, x, y


def generate_logistic_regression_data(
    n_obs: int,
    n_vars: int,
    cluster_type: Optional[str] = None,
    seed: int = SEED
) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Generate data for comprehensive logistic regression benchmark.
    
    Args:
        n_obs: Number of observations
        n_vars: Number of control variables
        cluster_type: None for HC3, "balanced" for balanced clusters, "imbalanced" for imbalanced
        seed: Random seed
    
    Returns:
        Tuple of (polars DataFrame, X numpy array, y numpy array, cluster array or None)
    """
    np.random.seed(seed)
    
    # Generate X variables
    x_data = {f"x{i}": np.random.randn(n_obs) for i in range(n_vars)}
    
    # Generate binary y with logistic relationship to X
    linear_pred = sum(x_data.values()) * 0.5
    prob = 1 / (1 + np.exp(-linear_pred))
    y = (np.random.random(n_obs) < prob).astype(float)
    
    df = pl.DataFrame({"y": y, **x_data})
    
    # Create numpy X array for statsmodels
    X = np.column_stack(list(x_data.values()))
    
    cluster_ids = None
    if cluster_type is not None:
        n_clusters = 100
        if cluster_type == "balanced":
            # Equal observations per cluster
            cluster_ids = np.repeat(range(n_clusters), n_obs // n_clusters)
            # Handle remainder
            remainder = n_obs - len(cluster_ids)
            if remainder > 0:
                cluster_ids = np.concatenate([cluster_ids, np.arange(remainder)])
        else:  # imbalanced
            # Some clusters have 5x more observations
            cluster_sizes = []
            for i in range(n_clusters):
                size = n_obs // n_clusters
                if i < 10:  # First 10 clusters are 5x larger
                    size *= 5
                cluster_sizes.append(size)
            # Normalize to sum to n_obs
            cluster_sizes = np.array(cluster_sizes)
            cluster_sizes = (cluster_sizes / cluster_sizes.sum() * n_obs).astype(int)
            cluster_sizes[-1] = n_obs - cluster_sizes[:-1].sum()  # Ensure exact sum
            cluster_ids = np.repeat(range(n_clusters), cluster_sizes)
        
        df = df.with_columns(pl.Series("cluster", cluster_ids[:n_obs]))
    
    return df, X, np.array(y), cluster_ids


def generate_sdid_panel(
    n_units: int, 
    n_periods: int, 
    n_treated: int, 
    n_pre: int, 
    effect: float = 5.0, 
    seed: int = SEED
) -> pl.DataFrame:
    """Generate synthetic DID panel data in long format."""
    np.random.seed(seed)
    data = {"unit": [], "time": [], "y": [], "treated": []}
    
    for unit in range(n_units):
        is_treated = unit < n_treated
        unit_effect = np.random.uniform(-1, 1)
        
        for t in range(n_periods):
            base = unit_effect + t * 0.5
            if is_treated and t >= n_pre:
                data["y"].append(base + effect)
                data["treated"].append(1)
            else:
                data["y"].append(base)
                data["treated"].append(0)
            data["unit"].append(unit)
            data["time"].append(t)
    
    return pl.DataFrame(data)


def generate_sc_panel(
    n_control: int,
    n_pre: int,
    n_post: int,
    effect: float = 5.0,
    seed: int = SEED
) -> pl.DataFrame:
    """Generate synthetic control panel data.
    
    Args:
        n_control: Number of control units
        n_pre: Number of pre-treatment periods
        n_post: Number of post-treatment periods
        effect: Treatment effect
        seed: Random seed
    
    Returns:
        Panel DataFrame with unit, time, y, treated columns
    """
    np.random.seed(seed)
    n_periods = n_pre + n_post
    data = {"unit": [], "time": [], "y": [], "treated": []}
    
    # Treated unit (unit 0)
    for t in range(n_periods):
        data["unit"].append(0)
        data["time"].append(t)
        base = 1.0 + t * 0.5
        if t >= n_pre:
            data["y"].append(base + effect)
            data["treated"].append(1)
        else:
            data["y"].append(base)
            data["treated"].append(0)
    
    # Control units
    for unit_id in range(1, n_control + 1):
        unit_effect = np.random.uniform(-0.5, 0.5)
        for t in range(n_periods):
            data["unit"].append(unit_id)
            data["time"].append(t)
            data["y"].append(1.0 + unit_effect + t * 0.5)
            data["treated"].append(0)
    
    return pl.DataFrame(data)


# Synthetic Control benchmark configurations
# (n_control, n_pre, n_post, label)
SYNTHETIC_CONTROL_CONFIGS = [
    (10, 16, 4, "Small (10 controls × 20 periods)"),
    (50, 40, 10, "Medium (50 controls × 50 periods)"),
    (100, 80, 20, "Large (100 controls × 100 periods)"),
]


def generate_azcausal_panel(
    n_units: int, 
    n_periods: int, 
    n_treated: int, 
    n_pre: int, 
    effect: float = 5.0, 
    seed: int = SEED
):
    """Generate panel data in azcausal format (wide)."""
    import pandas as pd
    
    np.random.seed(seed)
    outcome_data = {}
    intervention_data = {}
    
    for unit in range(n_units):
        is_treated = unit < n_treated
        unit_effect = np.random.uniform(-1, 1)
        outcomes = []
        interventions = []
        
        for t in range(n_periods):
            base = unit_effect + t * 0.5
            if is_treated and t >= n_pre:
                outcomes.append(base + effect)
                interventions.append(1)
            else:
                outcomes.append(base)
                interventions.append(0)
        
        outcome_data[unit] = outcomes
        intervention_data[unit] = interventions
    
    outcome_wide = pd.DataFrame(outcome_data, index=range(n_periods))
    intervention_wide = pd.DataFrame(intervention_data, index=range(n_periods))
    return outcome_wide, intervention_wide


# ============================================================
# Benchmark functions
# ============================================================

def benchmark_linear_regression(sizes: Dict[str, int]) -> List[Dict[str, Any]]:
    """Benchmark linear_regression vs statsmodels.OLS (simple single-variable)."""
    from causers import linear_regression
    
    try:
        import statsmodels.api as sm
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False
        print("⚠️  statsmodels not installed - skipping linear_regression comparison")
        return []
    
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION (Simple): causers vs statsmodels.OLS")
    print("=" * 60)
    
    def run_causers(df):
        return linear_regression(df, "x", "y")
    
    def run_statsmodels(x, y):
        X_sm = sm.add_constant(x)
        model = sm.OLS(y, X_sm).fit()
        return model.get_robustcov_results(cov_type='HC3')
    
    results = []
    for size_name, n in sizes.items():
        print(f"  {size_name}...", end=" ", flush=True)
        df, x, y = generate_linear_data(n)
        
        causers_timing = time_function(run_causers, df)
        ref_timing = time_function(run_statsmodels, x, y)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        result = {
            "Dataset": size_name,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        }
        results.append(result)
        
        status = "✅ FASTER" if speedup > 1.0 else "❌ SLOWER"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
    
    return results


def benchmark_linear_regression_comprehensive() -> List[Dict[str, Any]]:
    """Comprehensive benchmark for linear_regression covering:
    - Different numbers of observations (n)
    - Different numbers of control variables (p)
    - Different standard error types (HC3, clustered balanced, clustered imbalanced)
    """
    from causers import linear_regression
    
    try:
        import statsmodels.api as sm
    except ImportError:
        print("⚠️  statsmodels not installed - skipping comprehensive linear_regression comparison")
        return []
    
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION (Comprehensive): causers vs statsmodels.OLS")
    print("=" * 80)
    print("\nBenchmark dimensions:")
    print("  - Observations (n): 1,000 | 10,000 | 100,000")
    print("  - Variables (p): 2 | 10 | 50")
    print("  - SE types: HC3 | Clustered (balanced) | Clustered (imbalanced)")
    print()
    
    results = []
    seen_configs = set()  # Track to avoid duplicate configs
    
    for n_obs, n_vars, se_type, cluster_type, label in LINEAR_REGRESSION_CONFIGS:
        # Skip duplicates (some configs appear in multiple dimension groups)
        config_key = (n_obs, n_vars, se_type, cluster_type)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        
        print(f"  {label}...", end=" ", flush=True)
        
        # Generate data
        df, X, y, cluster_ids = generate_linear_regression_data(n_obs, n_vars, cluster_type)
        x_cols = [f"x{i}" for i in range(n_vars)]
        
        # Define causers runner - capture df, x_cols, cluster_type via default args
        def run_causers(_df=df, _x_cols=x_cols, _cluster_type=cluster_type):
            if _cluster_type is not None:
                return linear_regression(_df, _x_cols, "y", cluster="cluster")
            else:
                return linear_regression(_df, _x_cols, "y")
        
        # Define statsmodels runner - capture values via default args
        def run_statsmodels(_X=X, _y=y, _cluster_type=cluster_type, _cluster_ids=cluster_ids, _n_obs=n_obs):
            X_sm = sm.add_constant(_X)
            if _cluster_type is not None:
                return sm.OLS(_y, X_sm).fit(cov_type='cluster',
                                            cov_kwds={'groups': _cluster_ids[:_n_obs]})
            else:
                return sm.OLS(_y, X_sm).fit(cov_type='HC3')
        
        # Time both
        causers_timing = time_function(run_causers)
        ref_timing = time_function(run_statsmodels)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        result = {
            "Config": label,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "se_type": se_type,
            "cluster_type": cluster_type,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        }
        results.append(result)
        
        status = "✅" if speedup > 1.0 else "❌"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
    
    return results


def print_comprehensive_lr_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted summary table for comprehensive linear regression benchmarks."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION COMPREHENSIVE BENCHMARKS")
    print("=" * 80)
    
    # Header
    print(f"{'Config':<40} | {'causers (ms)':<12} | {'statsmodels (ms)':<16} | {'Speedup':<10}")
    print("-" * 40 + "-|-" + "-" * 12 + "-|-" + "-" * 16 + "-|-" + "-" * 10)
    
    for r in results:
        status = "✅" if r["faster"] else "❌"
        print(f"{r['Config']:<40} | {r['causers_ms']:<12.2f} | {r['reference_ms']:<16.2f} | {r['speedup']:.2f}x {status}")
    
    # Summary statistics by dimension
    print("\n" + "-" * 80)
    print("Summary by dimension:")
    
    # By observations (filter to 2 vars, HC3)
    obs_results = [r for r in results if r["n_vars"] == 2 and r["se_type"] == "hc3"]
    if obs_results:
        print(f"\n  Scaling with observations (p=2, HC3):")
        for r in sorted(obs_results, key=lambda x: x["n_obs"]):
            print(f"    n={r['n_obs']:>6}: {r['speedup']:.2f}x")
    
    # By variables (filter to 10K obs, HC3)
    var_results = [r for r in results if r["n_obs"] == 10000 and r["se_type"] == "hc3"]
    if var_results:
        print(f"\n  Scaling with variables (n=10K, HC3):")
        for r in sorted(var_results, key=lambda x: x["n_vars"]):
            print(f"    p={r['n_vars']:>2}: {r['speedup']:.2f}x")
    
    # By SE type (filter to 10K obs, 10 vars)
    se_results = [r for r in results if r["n_obs"] == 10000 and r["n_vars"] == 10]
    if se_results:
        print(f"\n  By SE type (n=10K, p=10):")
        for r in se_results:
            se_label = r["se_type"].upper() if r["cluster_type"] is None else f"Cluster ({r['cluster_type']})"
            print(f"    {se_label}: {r['speedup']:.2f}x")


def benchmark_logistic_regression(sizes: Dict[str, int]) -> List[Dict[str, Any]]:
    """Benchmark logistic_regression vs statsmodels.Logit."""
    from causers import logistic_regression
    
    try:
        import statsmodels.api as sm
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False
        print("⚠️  statsmodels not installed - skipping logistic_regression comparison")
        return []
    
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: causers vs statsmodels.Logit")
    print("=" * 60)
    
    def run_causers(df):
        return logistic_regression(df, "x", "y")
    
    def run_statsmodels(x, y):
        X_sm = sm.add_constant(x)
        return sm.Logit(y, X_sm).fit(disp=0, cov_type="HC3")
    
    results = []
    for size_name, n in sizes.items():
        print(f"  {size_name}...", end=" ", flush=True)
        df, x, y = generate_logistic_data(n)
        
        # Use warmup=3 for iterative/optimization-heavy methods
        causers_timing = time_function(run_causers, df, warmup=3)
        ref_timing = time_function(run_statsmodels, x, y, warmup=3)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        result = {
            "Dataset": size_name,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        }
        results.append(result)
        
        status = "✅ FASTER" if speedup > 1.0 else "❌ SLOWER"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
    
    return results


def benchmark_logistic_regression_comprehensive() -> List[Dict[str, Any]]:
    """Comprehensive benchmark for logistic_regression covering:
    - Different numbers of observations (n)
    - Different numbers of control variables (p)
    - Different standard error types (HC3, clustered balanced, clustered imbalanced)
    """
    from causers import logistic_regression
    
    try:
        import statsmodels.api as sm
    except ImportError:
        print("⚠️  statsmodels not installed - skipping comprehensive logistic_regression comparison")
        return []
    
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION (Comprehensive): causers vs statsmodels.Logit")
    print("=" * 80)
    print("\nBenchmark dimensions:")
    print("  - Observations (n): 1,000 | 10,000 | 100,000")
    print("  - Variables (p): 2 | 10 | 50")
    print("  - SE types: HC3 | Clustered (balanced) | Clustered (imbalanced)")
    print()
    
    results = []
    seen_configs = set()  # Track to avoid duplicate configs
    
    for n_obs, n_vars, se_type, cluster_type, label in LOGISTIC_REGRESSION_CONFIGS:
        # Skip duplicates (some configs appear in multiple dimension groups)
        config_key = (n_obs, n_vars, se_type, cluster_type)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        
        print(f"  {label}...", end=" ", flush=True)
        
        # Generate data
        df, X, y, cluster_ids = generate_logistic_regression_data(n_obs, n_vars, cluster_type)
        x_cols = [f"x{i}" for i in range(n_vars)]
        
        # Define causers runner - capture df, x_cols, cluster_type via default args
        def run_causers(_df=df, _x_cols=x_cols, _cluster_type=cluster_type):
            if _cluster_type is not None:
                return logistic_regression(_df, _x_cols, "y", cluster="cluster")
            else:
                return logistic_regression(_df, _x_cols, "y")
        
        # Define statsmodels runner - capture values via default args
        def run_statsmodels(_X=X, _y=y, _cluster_type=cluster_type, _cluster_ids=cluster_ids, _n_obs=n_obs):
            X_sm = sm.add_constant(_X)
            if _cluster_type is not None:
                return sm.Logit(_y, X_sm).fit(disp=0, cov_type='cluster',
                                              cov_kwds={'groups': _cluster_ids[:_n_obs]})
            else:
                return sm.Logit(_y, X_sm).fit(disp=0, cov_type='HC3')
        
        # Time both - use warmup=3 for iterative/optimization-heavy methods
        causers_timing = time_function(run_causers, warmup=3)
        ref_timing = time_function(run_statsmodels, warmup=3)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        result = {
            "Config": label,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "se_type": se_type,
            "cluster_type": cluster_type,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        }
        results.append(result)
        
        status = "✅" if speedup > 1.0 else "❌"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
    
    return results


def print_comprehensive_logit_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted summary table for comprehensive logistic regression benchmarks."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION COMPREHENSIVE BENCHMARKS")
    print("=" * 80)
    
    # Header
    print(f"{'Config':<40} | {'causers (ms)':<12} | {'statsmodels (ms)':<16} | {'Speedup':<10}")
    print("-" * 40 + "-|-" + "-" * 12 + "-|-" + "-" * 16 + "-|-" + "-" * 10)
    
    for r in results:
        status = "✅" if r["faster"] else "❌"
        print(f"{r['Config']:<40} | {r['causers_ms']:<12.2f} | {r['reference_ms']:<16.2f} | {r['speedup']:.2f}x {status}")
    
    # Summary statistics by dimension
    print("\n" + "-" * 80)
    print("Summary by dimension:")
    
    # By observations (filter to 2 vars, HC3)
    obs_results = [r for r in results if r["n_vars"] == 2 and r["se_type"] == "hc3"]
    if obs_results:
        print(f"\n  Scaling with observations (p=2, HC3):")
        for r in sorted(obs_results, key=lambda x: x["n_obs"]):
            print(f"    n={r['n_obs']:>6}: {r['speedup']:.2f}x")
    
    # By variables (filter to 10K obs, HC3)
    var_results = [r for r in results if r["n_obs"] == 10000 and r["se_type"] == "hc3"]
    if var_results:
        print(f"\n  Scaling with variables (n=10K, HC3):")
        for r in sorted(var_results, key=lambda x: x["n_vars"]):
            print(f"    p={r['n_vars']:>2}: {r['speedup']:.2f}x")
    
    # By SE type (filter to 10K obs, 10 vars)
    se_results = [r for r in results if r["n_obs"] == 10000 and r["n_vars"] == 10]
    if se_results:
        print(f"\n  By SE type (n=10K, p=10):")
        for r in se_results:
            se_label = r["se_type"].upper() if r["cluster_type"] is None else f"Cluster ({r['cluster_type']})"
            print(f"    {se_label}: {r['speedup']:.2f}x")


def benchmark_synthetic_did(sizes: Dict[str, Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
    """Benchmark synthetic_did vs azcausal.SDID."""
    from causers import synthetic_did
    
    try:
        from azcausal.core.panel import Panel
        from azcausal.estimators.panel.sdid import SDID
        HAS_AZCAUSAL = True
    except ImportError:
        HAS_AZCAUSAL = False
        print("⚠️  azcausal not installed - skipping synthetic_did comparison")
        return []
    
    print("\n" + "=" * 60)
    print("SYNTHETIC DID: causers vs azcausal.SDID")
    print("=" * 60)
    
    def run_causers(panel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return synthetic_did(panel, "unit", "time", "y", "treated",
                               bootstrap_iterations=0, seed=SEED)
    
    def run_azcausal(outcome_wide, intervention_wide):
        az_panel = Panel(data={"outcome": outcome_wide, "intervention": intervention_wide})
        estimator = SDID()
        return estimator.fit(az_panel)
    
    results = []
    for size_name, (n_units, n_periods, n_treated, n_pre) in sizes.items():
        print(f"  {size_name}...", end=" ", flush=True)
        
        # Generate data for both
        panel = generate_sdid_panel(n_units, n_periods, n_treated, n_pre)
        outcome_wide, intervention_wide = generate_azcausal_panel(n_units, n_periods, n_treated, n_pre)
        
        # Use warmup=3 for iterative/optimization-heavy methods
        causers_timing = time_function(run_causers, panel, warmup=3)
        ref_timing = time_function(run_azcausal, outcome_wide, intervention_wide, warmup=3)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        result = {
            "Dataset": size_name,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        }
        results.append(result)
        
        status = "✅ FASTER" if speedup > 1.0 else "❌ SLOWER"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
    
    return results


def benchmark_synthetic_control() -> List[Dict[str, Any]]:
    """Benchmark synthetic_control vs pysyncon for all methods.
    
    Compares causers.synthetic_control with pysyncon.Synth for:
    - traditional: Classic synthetic control
    - penalized: L2 regularized weights
    - robust: De-meaned SC matching dynamics
    - augmented: Bias-corrected SC
    """
    from causers import synthetic_control
    
    try:
        from pysyncon import Synth, AugSynth, PenalizedSynth
        HAS_PYSYNCON = True
    except ImportError:
        HAS_PYSYNCON = False
        print("⚠️  pysyncon not installed - skipping synthetic_control comparison")
        print("   Install with: pip install pysyncon")
        return []
    
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("SYNTHETIC CONTROL: causers vs pysyncon")
    print("=" * 80)
    print("\nMethods tested:")
    print("  - Traditional SC (causers vs pysyncon.Synth)")
    print("  - Penalized SC (causers vs pysyncon.PenalizedSynth)")
    print("  - Augmented SC (causers vs pysyncon.AugSynth)")
    print("  - Robust SC (causers only - pysyncon RobustSynth has bugs)")
    print()
    
    results = []
    
    # Test each panel size
    for n_control, n_pre, n_post, label in SYNTHETIC_CONTROL_CONFIGS:
        print(f"\n  {label}:")
        
        # Generate data
        panel = generate_sc_panel(n_control, n_pre, n_post, effect=5.0, seed=SEED)
        df_pandas = panel.to_pandas()
        
        # Prepare pysyncon matrices
        Z0_pre = df_pandas[(df_pandas["unit"] > 0) & (df_pandas["time"] < n_pre)].pivot(
            index="time", columns="unit", values="y")
        Z1_pre = df_pandas[(df_pandas["unit"] == 0) & (df_pandas["time"] < n_pre)].set_index("time")["y"]
        Z0_all = df_pandas[df_pandas["unit"] > 0].pivot(index="time", columns="unit", values="y")
        Z1_all = df_pandas[df_pandas["unit"] == 0].set_index("time")["y"]
        post_periods = list(range(n_pre, n_pre + n_post))
        
        # =============================================================
        # Traditional SC benchmark
        # =============================================================
        def run_causers_traditional():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return synthetic_control(
                    panel, "unit", "time", "y", "treated",
                    method="traditional", compute_se=False, seed=SEED
                )
        
        def run_pysyncon_traditional():
            synth = Synth()
            synth.fit(X0=Z0_pre, X1=Z1_pre, Z0=Z0_all, Z1=Z1_all)
            return synth.att(time_period=post_periods, Z0=Z0_all, Z1=Z1_all)
        
        print(f"    Traditional...", end=" ", flush=True)
        causers_timing = time_function(run_causers_traditional)
        ref_timing = time_function(run_pysyncon_traditional)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        results.append({
            "Config": f"{label} - Traditional",
            "method": "traditional",
            "panel_size": label,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        })
        
        status = "✅" if speedup > 1.0 else "❌"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.1f}x {status}")
        
        # =============================================================
        # Penalized SC benchmark
        # =============================================================
        def run_causers_penalized():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return synthetic_control(
                    panel, "unit", "time", "y", "treated",
                    method="penalized", lambda_param=0.1, compute_se=False, seed=SEED
                )
        
        def run_pysyncon_penalized():
            penalized = PenalizedSynth()
            penalized.fit(X0=Z0_pre, X1=Z1_pre, lambda_=0.1)
            return penalized.att(time_period=post_periods, Z0=Z0_all, Z1=Z1_all)
        
        print(f"    Penalized...", end=" ", flush=True)
        causers_timing = time_function(run_causers_penalized)
        ref_timing = time_function(run_pysyncon_penalized)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        results.append({
            "Config": f"{label} - Penalized",
            "method": "penalized",
            "panel_size": label,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        })
        
        status = "✅" if speedup > 1.0 else "❌"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.1f}x {status}")
        
        # =============================================================
        # Augmented SC benchmark
        # =============================================================
        # Prepare dataprep for AugSynth
        from pysyncon import Dataprep
        dataprep = Dataprep(
            foo=df_pandas,
            predictors=["y"],
            predictors_op="mean",
            dependent="y",
            unit_variable="unit",
            time_variable="time",
            treatment_identifier=0,
            controls_identifier=list(range(1, n_control + 1)),
            time_predictors_prior=list(range(n_pre)),
            time_optimize_ssr=list(range(n_pre)),
        )
        
        def run_causers_augmented():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return synthetic_control(
                    panel, "unit", "time", "y", "treated",
                    method="augmented", lambda_param=0.1, compute_se=False, seed=SEED
                )
        
        def run_pysyncon_augmented():
            augsynth = AugSynth()
            augsynth.fit(dataprep=dataprep, lambda_=0.1)
            return augsynth.att(time_period=post_periods)
        
        print(f"    Augmented...", end=" ", flush=True)
        # Use warmup=3 for augmented SC (optimization-heavy method)
        causers_timing = time_function(run_causers_augmented, warmup=3)
        ref_timing = time_function(run_pysyncon_augmented, warmup=3)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        results.append({
            "Config": f"{label} - Augmented",
            "method": "augmented",
            "panel_size": label,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        })
        
        status = "✅" if speedup > 1.0 else "❌"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.1f}x {status}")
        
        # =============================================================
        # Robust SC benchmark (causers only - pysyncon has bugs)
        # =============================================================
        def run_causers_robust():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return synthetic_control(
                    panel, "unit", "time", "y", "treated",
                    method="robust", compute_se=False, seed=SEED
                )
        
        print(f"    Robust (causers only)...", end=" ", flush=True)
        causers_timing = time_function(run_causers_robust)
        
        # For robust, we don't have a valid pysyncon comparison
        # so we report causers timing only
        results.append({
            "Config": f"{label} - Robust",
            "method": "robust",
            "panel_size": label,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": float('nan'),  # No valid reference
            "speedup": float('nan'),
            "faster": True,  # Causers is the only working implementation
        })
        
        print(f"{causers_timing['median_ms']:.2f}ms (pysyncon RobustSynth has bugs)")
    
    return results


def print_synthetic_control_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted summary table for synthetic control benchmarks."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("SYNTHETIC CONTROL BENCHMARKS")
    print("=" * 80)
    
    # Header
    print(f"{'Config':<45} | {'causers (ms)':<12} | {'pysyncon (ms)':<13} | {'Speedup':<10}")
    print("-" * 45 + "-|-" + "-" * 12 + "-|-" + "-" * 13 + "-|-" + "-" * 10)
    
    for r in results:
        status = "✅" if r["faster"] else "❌"
        ref_str = f"{r['reference_ms']:.2f}" if not np.isnan(r['reference_ms']) else "N/A"
        speedup_str = f"{r['speedup']:.1f}x" if not np.isnan(r['speedup']) else "N/A"
        print(f"{r['Config']:<45} | {r['causers_ms']:<12.2f} | {ref_str:<13} | {speedup_str:<8} {status}")
    
    # Summary by method
    print("\n" + "-" * 80)
    print("Summary by method:")
    
    methods = ["traditional", "penalized", "augmented", "robust"]
    for method in methods:
        method_results = [r for r in results if r["method"] == method and not np.isnan(r["speedup"])]
        if method_results:
            avg_speedup = np.mean([r["speedup"] for r in method_results])
            min_speedup = min(r["speedup"] for r in method_results)
            max_speedup = max(r["speedup"] for r in method_results)
            print(f"\n  {method.capitalize()} SC:")
            print(f"    Average speedup: {avg_speedup:.1f}x")
            print(f"    Range: {min_speedup:.1f}x - {max_speedup:.1f}x")
        elif method == "robust":
            print(f"\n  Robust SC:")
            print(f"    Note: pysyncon RobustSynth has bugs, causers-only benchmarks")
    
    # Summary by panel size
    print("\n" + "-" * 80)
    print("Summary by panel size:")
    
    panel_sizes = list(dict.fromkeys([r["panel_size"] for r in results]))
    for size in panel_sizes:
        size_results = [r for r in results if r["panel_size"] == size and not np.isnan(r["speedup"])]
        if size_results:
            avg_speedup = np.mean([r["speedup"] for r in size_results])
            print(f"  {size}: {avg_speedup:.1f}x average speedup")


# ============================================================
# Summary and reporting
# ============================================================

def print_summary(
    lr_results: List[Dict],
    logit_results: List[Dict],
    sdid_results: List[Dict]
) -> bool:
    """Print summary table and return overall pass/fail."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    all_results = []
    
    if lr_results:
        print("\nLinear Regression:")
        print(f"  {'Dataset':<15} {'causers':<12} {'statsmodels':<12} {'Speedup':<10} {'Status'}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for r in lr_results:
            status = "✅ PASS" if r["faster"] else "❌ FAIL"
            print(f"  {r['Dataset']:<15} {r['causers_ms']:<12.2f} {r['reference_ms']:<12.2f} {r['speedup']:<10.2f}x {status}")
            all_results.append(r)
    
    if logit_results:
        print("\nLogistic Regression:")
        print(f"  {'Dataset':<15} {'causers':<12} {'statsmodels':<12} {'Speedup':<10} {'Status'}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for r in logit_results:
            status = "✅ PASS" if r["faster"] else "❌ FAIL"
            print(f"  {r['Dataset']:<15} {r['causers_ms']:<12.2f} {r['reference_ms']:<12.2f} {r['speedup']:<10.2f}x {status}")
            all_results.append(r)
    
    if sdid_results:
        print("\nSynthetic DID:")
        print(f"  {'Dataset':<15} {'causers':<12} {'azcausal':<12} {'Speedup':<10} {'Status'}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for r in sdid_results:
            status = "✅ PASS" if r["faster"] else "❌ FAIL"
            print(f"  {r['Dataset']:<15} {r['causers_ms']:<12.2f} {r['reference_ms']:<12.2f} {r['speedup']:<10.2f}x {status}")
            all_results.append(r)
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)
    
    if not all_results:
        print("⚠️  No reference packages installed - cannot assess performance")
        return False
    
    total = len(all_results)
    faster_count = sum(1 for r in all_results if r["faster"])
    
    print(f"Benchmarks where causers is faster: {faster_count}/{total}")
    
    if faster_count == total:
        print("✅ PASS: All benchmarks show causers is faster than reference packages")
        return True
    else:
        slower = [r for r in all_results if not r["faster"]]
        print(f"❌ FAIL: causers is slower in {len(slower)} benchmark(s):")
        for r in slower:
            print(f"    - {r['Dataset']}: {r['speedup']:.2f}x")
        return False


def print_summary_extended(
    lr_results: List[Dict],
    lr_comprehensive_results: List[Dict],
    logit_results: List[Dict],
    logit_comprehensive_results: List[Dict],
    sdid_results: List[Dict],
    sc_results: List[Dict] = None
) -> bool:
    """Print summary table and return overall pass/fail."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    all_results = []
    
    if lr_results:
        print("\nLinear Regression (Simple):")
        print(f"  {'Dataset':<15} {'causers':<12} {'statsmodels':<12} {'Speedup':<10} {'Status'}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for r in lr_results:
            status = "✅ PASS" if r["faster"] else "❌ FAIL"
            print(f"  {r['Dataset']:<15} {r['causers_ms']:<12.2f} {r['reference_ms']:<12.2f} {r['speedup']:<10.2f}x {status}")
            all_results.append(r)
    
    if lr_comprehensive_results:
        print_comprehensive_lr_summary(lr_comprehensive_results)
        for r in lr_comprehensive_results:
            all_results.append(r)
    
    if logit_results:
        print("\nLogistic Regression (Simple):")
        print(f"  {'Dataset':<15} {'causers':<12} {'statsmodels':<12} {'Speedup':<10} {'Status'}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for r in logit_results:
            status = "✅ PASS" if r["faster"] else "❌ FAIL"
            print(f"  {r['Dataset']:<15} {r['causers_ms']:<12.2f} {r['reference_ms']:<12.2f} {r['speedup']:<10.2f}x {status}")
            all_results.append(r)
    
    if logit_comprehensive_results:
        print_comprehensive_logit_summary(logit_comprehensive_results)
        for r in logit_comprehensive_results:
            all_results.append(r)
    
    if sdid_results:
        print("\nSynthetic DID:")
        print(f"  {'Dataset':<15} {'causers':<12} {'azcausal':<12} {'Speedup':<10} {'Status'}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for r in sdid_results:
            status = "✅ PASS" if r["faster"] else "❌ FAIL"
            print(f"  {r['Dataset']:<15} {r['causers_ms']:<12.2f} {r['reference_ms']:<12.2f} {r['speedup']:<10.2f}x {status}")
            all_results.append(r)
    
    if sc_results:
        print_synthetic_control_summary(sc_results)
        # Only add results with valid speedups to all_results for overall assessment
        for r in sc_results:
            if not np.isnan(r.get("speedup", float('nan'))):
                all_results.append(r)
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    if not all_results:
        print("⚠️  No reference packages installed - cannot assess performance")
        return False
    
    total = len(all_results)
    faster_count = sum(1 for r in all_results if r["faster"])
    
    print(f"Benchmarks where causers is faster: {faster_count}/{total}")
    
    if faster_count == total:
        print("✅ PASS: All benchmarks show causers is faster than reference packages")
        return True
    else:
        slower = [r for r in all_results if not r["faster"]]
        print(f"❌ FAIL: causers is slower in {len(slower)} benchmark(s):")
        for r in slower:
            label = r.get('Dataset', r.get('Config', 'Unknown'))
            print(f"    - {label}: {r['speedup']:.2f}x")
        return False


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("CAUSERS PERFORMANCE BENCHMARK")
    print("=" * 80)
    print("Comparing causers against reference packages...")
    print("Build: maturin develop --release")
    
    # Define dataset sizes for simple benchmarks
    regression_sizes = {
        "1K": 1_000,
        "10K": 10_000,
        "100K": 100_000,
    }
    
    sdid_sizes = {
        "10x20": (10, 20, 2, 16),      # n_units, n_periods, n_treated, n_pre
        "50x50": (50, 50, 10, 40),
    }
    
    # Run simple benchmarks
    lr_results = benchmark_linear_regression(regression_sizes)
    
    # Run comprehensive linear regression benchmarks
    lr_comprehensive_results = benchmark_linear_regression_comprehensive()
    
    # Run simple logistic regression benchmarks
    logit_results = benchmark_logistic_regression(regression_sizes)
    
    # Run comprehensive logistic regression benchmarks
    logit_comprehensive_results = benchmark_logistic_regression_comprehensive()
    
    # Run synthetic DID benchmarks
    sdid_results = benchmark_synthetic_did(sdid_sizes)
    
    # Run synthetic control benchmarks (causers vs pysyncon)
    sc_results = benchmark_synthetic_control()
    
    # Print summary and get pass/fail
    passed = print_summary_extended(
        lr_results, lr_comprehensive_results,
        logit_results, logit_comprehensive_results,
        sdid_results, sc_results
    )
    
    return 0 if passed else 1


# ============================================================
# Pytest test functions for synthetic control benchmarks
# ============================================================

def test_synthetic_control_benchmark_traditional():
    """Benchmark test: causers vs pysyncon for traditional SC."""
    from causers import synthetic_control
    
    try:
        from pysyncon import Synth
    except ImportError:
        import pytest
        pytest.skip("pysyncon not installed")
    
    # Use small panel for test speed
    n_control, n_pre, n_post = 10, 16, 4
    panel = generate_sc_panel(n_control, n_pre, n_post, effect=5.0, seed=SEED)
    df_pandas = panel.to_pandas()
    
    # Prepare pysyncon matrices
    Z0_pre = df_pandas[(df_pandas["unit"] > 0) & (df_pandas["time"] < n_pre)].pivot(
        index="time", columns="unit", values="y")
    Z1_pre = df_pandas[(df_pandas["unit"] == 0) & (df_pandas["time"] < n_pre)].set_index("time")["y"]
    Z0_all = df_pandas[df_pandas["unit"] > 0].pivot(index="time", columns="unit", values="y")
    Z1_all = df_pandas[df_pandas["unit"] == 0].set_index("time")["y"]
    post_periods = list(range(n_pre, n_pre + n_post))
    
    def run_causers():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return synthetic_control(
                panel, "unit", "time", "y", "treated",
                method="traditional", compute_se=False, seed=SEED
            )
    
    def run_pysyncon():
        synth = Synth()
        synth.fit(X0=Z0_pre, X1=Z1_pre, Z0=Z0_all, Z1=Z1_all)
        return synth.att(time_period=post_periods, Z0=Z0_all, Z1=Z1_all)
    
    causers_timing = time_function(run_causers, n_iter=3, warmup=1)
    ref_timing = time_function(run_pysyncon, n_iter=3, warmup=1)
    speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
    
    print(f"\n  Traditional SC: causers {causers_timing['median_ms']:.2f}ms vs pysyncon {ref_timing['median_ms']:.2f}ms = {speedup:.1f}x")
    
    # Log the speedup but don't fail if slower (benchmark info only)
    assert causers_timing["median_ms"] > 0, "causers should complete successfully"


def test_synthetic_control_benchmark_penalized():
    """Benchmark test: causers vs pysyncon for penalized SC."""
    from causers import synthetic_control
    
    try:
        from pysyncon import PenalizedSynth
    except ImportError:
        import pytest
        pytest.skip("pysyncon not installed")
    
    # Use small panel for test speed
    n_control, n_pre, n_post = 10, 16, 4
    panel = generate_sc_panel(n_control, n_pre, n_post, effect=5.0, seed=SEED)
    df_pandas = panel.to_pandas()
    
    # Prepare pysyncon matrices
    Z0_pre = df_pandas[(df_pandas["unit"] > 0) & (df_pandas["time"] < n_pre)].pivot(
        index="time", columns="unit", values="y")
    Z1_pre = df_pandas[(df_pandas["unit"] == 0) & (df_pandas["time"] < n_pre)].set_index("time")["y"]
    Z0_all = df_pandas[df_pandas["unit"] > 0].pivot(index="time", columns="unit", values="y")
    Z1_all = df_pandas[df_pandas["unit"] == 0].set_index("time")["y"]
    post_periods = list(range(n_pre, n_pre + n_post))
    
    def run_causers():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return synthetic_control(
                panel, "unit", "time", "y", "treated",
                method="penalized", lambda_param=0.1, compute_se=False, seed=SEED
            )
    
    def run_pysyncon():
        penalized = PenalizedSynth()
        penalized.fit(X0=Z0_pre, X1=Z1_pre, lambda_=0.1)
        return penalized.att(time_period=post_periods, Z0=Z0_all, Z1=Z1_all)
    
    causers_timing = time_function(run_causers, n_iter=3, warmup=1)
    ref_timing = time_function(run_pysyncon, n_iter=3, warmup=1)
    speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
    
    print(f"\n  Penalized SC: causers {causers_timing['median_ms']:.2f}ms vs pysyncon {ref_timing['median_ms']:.2f}ms = {speedup:.1f}x")
    
    assert causers_timing["median_ms"] > 0, "causers should complete successfully"


def test_synthetic_control_benchmark_augmented():
    """Benchmark test: causers vs pysyncon for augmented SC."""
    from causers import synthetic_control
    
    try:
        from pysyncon import AugSynth, Dataprep
    except ImportError:
        import pytest
        pytest.skip("pysyncon not installed")
    
    # Use small panel for test speed
    n_control, n_pre, n_post = 10, 16, 4
    panel = generate_sc_panel(n_control, n_pre, n_post, effect=5.0, seed=SEED)
    df_pandas = panel.to_pandas()
    
    # Prepare dataprep for AugSynth
    dataprep = Dataprep(
        foo=df_pandas,
        predictors=["y"],
        predictors_op="mean",
        dependent="y",
        unit_variable="unit",
        time_variable="time",
        treatment_identifier=0,
        controls_identifier=list(range(1, n_control + 1)),
        time_predictors_prior=list(range(n_pre)),
        time_optimize_ssr=list(range(n_pre)),
    )
    
    post_periods = list(range(n_pre, n_pre + n_post))
    
    def run_causers():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return synthetic_control(
                panel, "unit", "time", "y", "treated",
                method="augmented", lambda_param=0.1, compute_se=False, seed=SEED
            )
    
    def run_pysyncon():
        augsynth = AugSynth()
        augsynth.fit(dataprep=dataprep, lambda_=0.1)
        return augsynth.att(time_period=post_periods)
    
    # Use warmup=3 for augmented SC (optimization-heavy method)
    causers_timing = time_function(run_causers, n_iter=3, warmup=3)
    ref_timing = time_function(run_pysyncon, n_iter=3, warmup=3)
    speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
    
    print(f"\n  Augmented SC: causers {causers_timing['median_ms']:.2f}ms vs pysyncon {ref_timing['median_ms']:.2f}ms = {speedup:.1f}x")
    
    assert causers_timing["median_ms"] > 0, "causers should complete successfully"


def test_synthetic_control_benchmark_robust():
    """Benchmark test: causers robust SC (no pysyncon comparison - has bugs)."""
    from causers import synthetic_control
    
    # Use small panel for test speed
    n_control, n_pre, n_post = 10, 16, 4
    panel = generate_sc_panel(n_control, n_pre, n_post, effect=5.0, seed=SEED)
    
    def run_causers():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return synthetic_control(
                panel, "unit", "time", "y", "treated",
                method="robust", compute_se=False, seed=SEED
            )
    
    causers_timing = time_function(run_causers, n_iter=3, warmup=1)
    
    print(f"\n  Robust SC: causers {causers_timing['median_ms']:.2f}ms (pysyncon RobustSynth has bugs)")
    
    assert causers_timing["median_ms"] > 0, "causers should complete successfully"


def test_synthetic_control_benchmark_all_methods():
    """Benchmark test: run all methods and report comprehensive speedups."""
    from causers import synthetic_control
    
    try:
        from pysyncon import Synth, PenalizedSynth, AugSynth
        HAS_PYSYNCON = True
    except ImportError:
        HAS_PYSYNCON = False
    
    # Use medium panel for comprehensive test
    n_control, n_pre, n_post = 50, 40, 10
    panel = generate_sc_panel(n_control, n_pre, n_post, effect=5.0, seed=SEED)
    
    methods = ["traditional", "penalized", "robust", "augmented"]
    results = {}
    
    print("\n  Comprehensive SC benchmark (50 controls × 50 periods):")
    
    for method in methods:
        def run_causers(_method=method):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return synthetic_control(
                    panel, "unit", "time", "y", "treated",
                    method=_method, lambda_param=0.1 if _method in ["penalized", "augmented"] else None,
                    compute_se=False, seed=SEED
                )
        
        timing = time_function(run_causers, n_iter=3, warmup=1)
        results[method] = timing["median_ms"]
        print(f"    {method}: {timing['median_ms']:.2f}ms")
    
    # Verify all methods complete
    for method in methods:
        assert results[method] > 0, f"causers {method} should complete successfully"
    
    if HAS_PYSYNCON:
        print("  ✅ pysyncon comparison available")
    else:
        print("  ⚠️  pysyncon not installed - skipping comparison")


if __name__ == "__main__":
    sys.exit(main())
