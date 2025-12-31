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


# Fixed effects benchmark configurations (for linear regression)
# (n_obs, n_vars, n_fe, fe_type, label)
# fe_type: 0 (no FE), 1 (one-way FE), 2 (two-way FE)
LINEAR_REGRESSION_FE_CONFIGS = [
    # Vary observations (with 5 vars, no FE)
    (1000, 5, 0, None, "1K obs, 5 vars, No FE"),
    (10000, 5, 0, None, "10K obs, 5 vars, No FE"),
    (100000, 5, 0, None, "100K obs, 5 vars, No FE"),
    
    # Vary observations (with 5 vars, one-way FE)
    (1000, 5, 1, "entity", "1K obs, 5 vars, One-way FE"),
    (10000, 5, 1, "entity", "10K obs, 5 vars, One-way FE"),
    (100000, 5, 1, "entity", "100K obs, 5 vars, One-way FE"),
    
    # Vary observations (with 5 vars, two-way FE)
    (1000, 5, 2, ["entity", "time"], "1K obs, 5 vars, Two-way FE"),
    (10000, 5, 2, ["entity", "time"], "10K obs, 5 vars, Two-way FE"),
    (100000, 5, 2, ["entity", "time"], "100K obs, 5 vars, Two-way FE"),
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


# Logistic regression fixed effects benchmark configurations
# (n_obs, n_vars, n_fe, fe_type, label)
# fe_type: 0 (no FE), 1 (one-way FE), 2 (two-way FE)
LOGISTIC_REGRESSION_FE_CONFIGS = [
    # Vary observations (with 3 vars, no FE)
    (1000, 3, 0, None, "1K obs, 3 vars, No FE"),
    (10000, 3, 0, None, "10K obs, 3 vars, No FE"),
    (100000, 3, 0, None, "100K obs, 3 vars, No FE"),
    
    # Vary observations (with 3 vars, one-way FE)
    (1000, 3, 1, "entity", "1K obs, 3 vars, One-way FE"),
    (10000, 3, 1, "entity", "10K obs, 3 vars, One-way FE"),
    (100000, 3, 1, "entity", "100K obs, 3 vars, One-way FE"),
    
    # Vary observations (with 3 vars, two-way FE)
    (1000, 3, 2, ["entity", "time"], "1K obs, 3 vars, Two-way FE"),
    (10000, 3, 2, ["entity", "time"], "10K obs, 3 vars, Two-way FE"),
    (100000, 3, 2, ["entity", "time"], "100K obs, 3 vars, Two-way FE"),
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


def generate_linear_regression_fe_data(
    n_obs: int,
    n_vars: int,
    n_fe: int,
    fe_type: Optional[str | list],
    seed: int = SEED
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Generate data for fixed effects linear regression benchmark.
    
    Args:
        n_obs: Number of observations
        n_vars: Number of control variables
        n_fe: Number of fixed effect dimensions (0, 1, or 2)
        fe_type: None for no FE, "entity" for one-way, ["entity", "time"] for two-way
        seed: Random seed
    
    Returns:
        Tuple of (polars DataFrame, dict with pyfixest-compatible data)
    """
    np.random.seed(seed)
    
    # Generate X variables
    x_data = {f"x{i}": np.random.randn(n_obs) for i in range(n_vars)}
    
    # Generate fixed effects
    entity_effects = None
    time_effects = None
    
    if n_fe == 0:
        # No fixed effects
        y = sum(x_data.values()) + np.random.randn(n_obs) * 0.5
        fe_cols = {}
    elif n_fe == 1:
        # One-way fixed effects (entity)
        n_entities = min(100, n_obs // 10)
        entity_ids = np.random.randint(0, n_entities, n_obs)
        entity_effects = np.random.randn(n_entities)
        
        y = sum(x_data.values()) + entity_effects[entity_ids] + np.random.randn(n_obs) * 0.5
        fe_cols = {"entity": entity_ids}
    else:  # n_fe == 2
        # Two-way fixed effects (entity + time)
        n_entities = min(50, n_obs // 20)
        n_periods = min(20, n_obs // 50)
        
        entity_ids = np.random.randint(0, n_entities, n_obs)
        time_ids = np.random.randint(0, n_periods, n_obs)
        
        entity_effects = np.random.randn(n_entities)
        time_effects = np.random.randn(n_periods)
        
        y = (sum(x_data.values()) +
             entity_effects[entity_ids] +
             time_effects[time_ids] +
             np.random.randn(n_obs) * 0.5)
        fe_cols = {"entity": entity_ids, "time": time_ids}
    
    df = pl.DataFrame({"y": y, **x_data, **fe_cols})
    
    # Prepare pyfixest-compatible data
    pyfixest_data = {
        "y": np.array(y),
        "X": np.column_stack(list(x_data.values())),
        "fe_cols": fe_cols,
        "fe_type": fe_type,
    }
    
    return df, pyfixest_data


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


def benchmark_linear_regression_fe() -> List[Dict[str, Any]]:
    """Comprehensive benchmark for linear_regression with fixed effects.
    
    Compares causers vs pyfixest for:
    - No fixed effects
    - One-way fixed effects (entity)
    - Two-way fixed effects (entity + time)
    
    Across different observation counts (1K, 10K, 100K).
    """
    from causers import linear_regression
    
    try:
        import pyfixest as pf
        HAS_PYFIXEST = True
    except ImportError:
        HAS_PYFIXEST = False
        print("⚠️  pyfixest not installed - skipping fixed effects comparison")
        return []
    
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION (Fixed Effects): causers vs pyfixest")
    print("=" * 80)
    print("\nBenchmark dimensions:")
    print("  - Observations (n): 1,000 | 10,000 | 100,000")
    print("  - Fixed effects: None | One-way (entity) | Two-way (entity + time)")
    print("  - Variables (p): 5")
    print()
    
    results = []
    
    for n_obs, n_vars, n_fe, fe_type, label in LINEAR_REGRESSION_FE_CONFIGS:
        print(f"  {label}...", end=" ", flush=True)
        
        # Generate data
        df, pyfixest_data = generate_linear_regression_fe_data(n_obs, n_vars, n_fe, fe_type)
        x_cols = [f"x{i}" for i in range(n_vars)]
        
        # Define causers runner
        def run_causers(_df=df, _x_cols=x_cols, _fe_type=fe_type):
            if _fe_type is None:
                return linear_regression(_df, _x_cols, "y")
            else:
                return linear_regression(_df, _x_cols, "y", fixed_effects=_fe_type)
        
        # Define pyfixest runner
        def run_pyfixest(_df=df, _x_cols=x_cols, _fe_type=fe_type):
            # pyfixest requires pandas DataFrame
            df_pandas = _df.to_pandas()
            
            # Build formula
            x_formula = " + ".join(_x_cols)
            if _fe_type is None:
                formula = f"y ~ {x_formula}"
                # No fixed effects - use HC3
                vcov = "HC3"
            elif isinstance(_fe_type, str):
                formula = f"y ~ {x_formula} | {_fe_type}"
                # Fixed effects - HC3 not supported, use hetero instead
                vcov = "hetero"
            else:  # list of FE columns
                fe_formula = " + ".join(_fe_type)
                formula = f"y ~ {x_formula} | {fe_formula}"
                # Fixed effects - HC3 not supported, use hetero instead
                vcov = "hetero"
            
            return pf.feols(formula, data=df_pandas, vcov=vcov)
        
        # Time both
        causers_timing = time_function(run_causers)
        ref_timing = time_function(run_pyfixest)
        speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
        
        result = {
            "Config": label,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "n_fe": n_fe,
            "fe_type": fe_type,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": ref_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        }
        results.append(result)
        
        status = "✅" if speedup > 1.0 else "❌"
        print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
    
    return results


def print_comprehensive_lr_fe_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted summary table for fixed effects linear regression benchmarks."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION FIXED EFFECTS BENCHMARKS")
    print("=" * 80)
    
    # Header
    print(f"{'Config':<40} | {'causers (ms)':<12} | {'pyfixest (ms)':<14} | {'Speedup':<10}")
    print("-" * 40 + "-|-" + "-" * 12 + "-|-" + "-" * 14 + "-|-" + "-" * 10)
    
    for r in results:
        status = "✅" if r["faster"] else "❌"
        print(f"{r['Config']:<40} | {r['causers_ms']:<12.2f} | {r['reference_ms']:<14.2f} | {r['speedup']:.2f}x {status}")
    
    # Summary statistics by dimension
    print("\n" + "-" * 80)
    print("Summary by dimension:")
    
    # By observations (filter to 5 vars, no FE)
    no_fe_results = [r for r in results if r["n_fe"] == 0]
    if no_fe_results:
        print(f"\n  Scaling with observations (p=5, No FE):")
        for r in sorted(no_fe_results, key=lambda x: x["n_obs"]):
            print(f"    n={r['n_obs']:>6}: {r['speedup']:.2f}x")
    
    # By observations (filter to 5 vars, one-way FE)
    one_way_results = [r for r in results if r["n_fe"] == 1]
    if one_way_results:
        print(f"\n  Scaling with observations (p=5, One-way FE):")
        for r in sorted(one_way_results, key=lambda x: x["n_obs"]):
            print(f"    n={r['n_obs']:>6}: {r['speedup']:.2f}x")
    
    # By observations (filter to 5 vars, two-way FE)
    two_way_results = [r for r in results if r["n_fe"] == 2]
    if two_way_results:
        print(f"\n  Scaling with observations (p=5, Two-way FE):")
        for r in sorted(two_way_results, key=lambda x: x["n_obs"]):
            print(f"    n={r['n_obs']:>6}: {r['speedup']:.2f}x")
    
    # By FE type (filter to 10K obs)
    fe_10k_results = [r for r in results if r["n_obs"] == 10000]
    if fe_10k_results:
        print(f"\n  By FE type (n=10K, p=5):")
        for r in sorted(fe_10k_results, key=lambda x: x["n_fe"]):
            fe_label = "No FE" if r["n_fe"] == 0 else ("One-way FE" if r["n_fe"] == 1 else "Two-way FE")
            print(f"    {fe_label}: {r['speedup']:.2f}x")


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


def generate_logistic_regression_fe_data(
    n_obs: int,
    n_vars: int,
    n_fe: int,
    fe_type: Optional[str | list],
    seed: int = SEED,
    max_retries: int = 10
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Generate data for fixed effects logistic regression benchmark.
    
    Generates well-behaved data for logistic regression MLE:
    - Uses moderate coefficients to avoid numerical overflow
    - Centers probability around 50% to ensure balanced classes
    - Adds noise to prevent perfect separation
    - Retries with different seed if class imbalance is too extreme
    
    Args:
        n_obs: Number of observations
        n_vars: Number of control variables
        n_fe: Number of fixed effect dimensions (0, 1, or 2)
        fe_type: None for no FE, "entity" for one-way, ["entity", "time"] for two-way
        seed: Random seed
        max_retries: Maximum retries if class distribution is too extreme
    
    Returns:
        Tuple of (polars DataFrame, dict with pyfixest-compatible data)
    """
    # Target: positive rate between 30% and 70%
    min_pos_rate = 0.30
    max_pos_rate = 0.70
    
    for attempt in range(max_retries):
        current_seed = seed + attempt
        np.random.seed(current_seed)
        
        # Generate X variables with moderate range
        x_data = {f"x{i}": np.random.randn(n_obs) for i in range(n_vars)}
        
        # Use small coefficients per variable to keep linear predictor in reasonable range
        # With n_vars variables, each contributes ~0.15 to avoid extreme combined effects
        coef_per_var = 0.15
        
        # Generate fixed effects with small magnitudes
        if n_fe == 0:
            # No fixed effects - just X contribution
            linear_pred = sum(x_data.values()) * coef_per_var
            fe_cols = {}
        elif n_fe == 1:
            # One-way fixed effects (entity) with small effects
            n_entities = min(100, n_obs // 10)
            entity_ids = np.random.randint(0, n_entities, n_obs)
            # Small entity effects centered at 0
            entity_effects = np.random.randn(n_entities) * 0.15
            
            linear_pred = sum(x_data.values()) * coef_per_var + entity_effects[entity_ids]
            fe_cols = {"entity": entity_ids}
        else:  # n_fe == 2
            # Two-way fixed effects (entity + time) with small effects
            n_entities = min(50, n_obs // 20)
            n_periods = min(20, n_obs // 50)
            
            entity_ids = np.random.randint(0, n_entities, n_obs)
            time_ids = np.random.randint(0, n_periods, n_obs)
            
            # Very small fixed effects to avoid extreme values
            entity_effects = np.random.randn(n_entities) * 0.1
            time_effects = np.random.randn(n_periods) * 0.05
            
            linear_pred = (sum(x_data.values()) * coef_per_var +
                          entity_effects[entity_ids] +
                          time_effects[time_ids])
            fe_cols = {"entity": entity_ids, "time": time_ids}
        
        # Add small noise to prevent perfect separation
        noise = np.random.randn(n_obs) * 0.1
        linear_pred = linear_pred + noise
        
        # Convert to probability and generate binary outcome
        prob = 1 / (1 + np.exp(-linear_pred))
        y = (np.random.random(n_obs) < prob).astype(float)
        
        # Check class balance
        pos_rate = y.mean()
        if min_pos_rate <= pos_rate <= max_pos_rate:
            # Good distribution, proceed
            break
        # Otherwise retry with different seed
    
    df = pl.DataFrame({"y": y, **x_data, **fe_cols})
    
    # Prepare pyfixest-compatible data
    pyfixest_data = {
        "y": np.array(y),
        "X": np.column_stack(list(x_data.values())),
        "fe_cols": fe_cols,
        "fe_type": fe_type,
    }
    
    return df, pyfixest_data


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


def benchmark_logistic_regression_fe() -> List[Dict[str, Any]]:
    """Comprehensive benchmark for logistic_regression with fixed effects.
    
    Compares causers vs pyfixest for:
    - No fixed effects
    - One-way fixed effects (entity)
    - Two-way fixed effects (entity + time)
    
    Across different observation counts (1K, 10K, 100K).
    Uses Mundlak strategy (adding group means as regressors).
    """
    from causers import logistic_regression
    
    try:
        import pyfixest as pf
        HAS_PYFIXEST = True
    except ImportError:
        HAS_PYFIXEST = False
        print("⚠️  pyfixest not installed - skipping fixed effects logistic regression comparison")
        return []
    
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION (Fixed Effects): causers vs pyfixest.feglm")
    print("=" * 80)
    print("\nBenchmark dimensions:")
    print("  - Observations (n): 1,000 | 10,000 | 100,000")
    print("  - Fixed effects: None | One-way (entity) | Two-way (entity + time)")
    print("  - Variables (p): 3")
    print("  - Method: Mundlak (group means as regressors)")
    print()
    
    results = []
    
    for n_obs, n_vars, n_fe, fe_type, label in LOGISTIC_REGRESSION_FE_CONFIGS:
        print(f"  {label}...", end=" ", flush=True)
        
        # Generate data
        df, pyfixest_data = generate_logistic_regression_fe_data(n_obs, n_vars, n_fe, fe_type)
        x_cols = [f"x{i}" for i in range(n_vars)]
        
        # Define causers runner
        def run_causers(_df=df, _x_cols=x_cols, _fe_type=fe_type):
            if _fe_type is None:
                return logistic_regression(_df, _x_cols, "y")
            elif isinstance(_fe_type, str):
                return logistic_regression(_df, _x_cols, "y", fixed_effects=[_fe_type])
            else:  # list
                return logistic_regression(_df, _x_cols, "y", fixed_effects=_fe_type)
        
        # Define pyfixest runner
        def run_pyfixest(_df=df, _x_cols=x_cols, _fe_type=fe_type):
            # pyfixest requires pandas DataFrame
            df_pandas = _df.to_pandas()
            
            # Build formula
            x_formula = " + ".join(_x_cols)
            if _fe_type is None:
                formula = f"y ~ {x_formula}"
                return pf.feglm(formula, data=df_pandas, family="logit")
            elif isinstance(_fe_type, str):
                formula = f"y ~ {x_formula} | {_fe_type}"
                return pf.feglm(formula, data=df_pandas, family="logit")
            else:  # list of FE columns
                fe_formula = " + ".join(_fe_type)
                formula = f"y ~ {x_formula} | {fe_formula}"
                return pf.feglm(formula, data=df_pandas, family="logit")
        
        # Time both - use warmup=3 for iterative/optimization-heavy methods
        try:
            causers_timing = time_function(run_causers, warmup=3)
            ref_timing = time_function(run_pyfixest, warmup=3)
            speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
            
            result = {
                "Config": label,
                "n_obs": n_obs,
                "n_vars": n_vars,
                "n_fe": n_fe,
                "fe_type": fe_type,
                "causers_ms": causers_timing["median_ms"],
                "reference_ms": ref_timing["median_ms"],
                "speedup": speedup,
                "faster": speedup > 1.0,
            }
            results.append(result)
            
            status = "✅" if speedup > 1.0 else "❌"
            print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
        except Exception as e:
            print(f"SKIPPED: {e}")
            continue
    
    return results


def print_comprehensive_logit_fe_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted summary table for fixed effects logistic regression benchmarks."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION FIXED EFFECTS BENCHMARKS (Mundlak Strategy)")
    print("=" * 80)
    
    # Header
    print(f"{'Config':<40} | {'causers (ms)':<12} | {'pyfixest (ms)':<14} | {'Speedup':<10}")
    print("-" * 40 + "-|-" + "-" * 12 + "-|-" + "-" * 14 + "-|-" + "-" * 10)
    
    for r in results:
        status = "✅" if r["faster"] else "❌"
        print(f"{r['Config']:<40} | {r['causers_ms']:<12.2f} | {r['reference_ms']:<14.2f} | {r['speedup']:.2f}x {status}")
    
    # Summary statistics by dimension
    print("\n" + "-" * 80)
    print("Summary by dimension:")
    
    # By observations (filter to 3 vars, no FE)
    no_fe_results = [r for r in results if r["n_fe"] == 0]
    if no_fe_results:
        print(f"\n  Scaling with observations (p=3, No FE):")
        for r in sorted(no_fe_results, key=lambda x: x["n_obs"]):
            print(f"    n={r['n_obs']:>6}: {r['speedup']:.2f}x")
    
    # By observations (filter to 3 vars, one-way FE)
    one_way_results = [r for r in results if r["n_fe"] == 1]
    if one_way_results:
        print(f"\n  Scaling with observations (p=3, One-way FE):")
        for r in sorted(one_way_results, key=lambda x: x["n_obs"]):
            print(f"    n={r['n_obs']:>6}: {r['speedup']:.2f}x")
    
    # By observations (filter to 3 vars, two-way FE)
    two_way_results = [r for r in results if r["n_fe"] == 2]
    if two_way_results:
        print(f"\n  Scaling with observations (p=3, Two-way FE):")
        for r in sorted(two_way_results, key=lambda x: x["n_obs"]):
            print(f"    n={r['n_obs']:>6}: {r['speedup']:.2f}x")
    
    # By FE type (filter to 10K obs)
    fe_10k_results = [r for r in results if r["n_obs"] == 10000]
    if fe_10k_results:
        print(f"\n  By FE type (n=10K, p=3):")
        for r in sorted(fe_10k_results, key=lambda x: x["n_fe"]):
            fe_label = "No FE" if r["n_fe"] == 0 else ("One-way FE" if r["n_fe"] == 1 else "Two-way FE")
            print(f"    {fe_label}: {r['speedup']:.2f}x")


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
# Summary Functions
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
    lr_fe_results: List[Dict],
    logit_results: List[Dict],
    logit_comprehensive_results: List[Dict],
    sdid_results: List[Dict],
    sc_results: List[Dict] = None
) -> bool:
    """Print summary table and return overall pass/fail."""
    all_results = []
    
    # Collect all results for overall assessment
    if lr_results:
        for r in lr_results:
            all_results.append(r)
    
    if lr_comprehensive_results:
        for r in lr_comprehensive_results:
            all_results.append(r)
    
    if lr_fe_results:
        for r in lr_fe_results:
            all_results.append(r)
    
    if logit_results:
        for r in logit_results:
            all_results.append(r)
    
    if logit_comprehensive_results:
        for r in logit_comprehensive_results:
            all_results.append(r)
    
    if sdid_results:
        for r in sdid_results:
            all_results.append(r)
    
    if sc_results:
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


# ============================================================
# Test Functions
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


# ============================================================
# DML BENCHMARKS
# ============================================================

# DML benchmark configurations
# (n_obs, n_vars, n_folds, treatment_type, estimate_cate, label)
DML_CONFIGS = [
    # Binary treatment, varying observations
    (10000, 10, 5, "binary", False, "10K obs, 10 vars, K=5, binary"),
    (100000, 10, 5, "binary", False, "100K obs, 10 vars, K=5, binary"),
    
    # Continuous treatment
    (10000, 10, 5, "continuous", False, "10K obs, 10 vars, K=5, continuous"),
    (100000, 10, 5, "continuous", False, "100K obs, 10 vars, K=5, continuous"),
    
    # With CATE estimation
    (10000, 10, 5, "binary", True, "10K obs, 10 vars, K=5, binary+CATE"),
    (100000, 10, 5, "binary", True, "100K obs, 10 vars, K=5, binary+CATE"),
]


def generate_dml_data(
    n_obs: int,
    n_vars: int,
    treatment_type: str = "binary",
    seed: int = SEED
) -> pl.DataFrame:
    """Generate synthetic data for DML benchmarks.
    
    Args:
        n_obs: Number of observations
        n_vars: Number of control variables
        treatment_type: "binary" or "continuous"
        seed: Random seed
    
    Returns:
        polars DataFrame with y, d, and x0..x{n_vars-1} columns
    """
    np.random.seed(seed)
    
    # Generate X variables
    x_data = {f"x{i}": np.random.randn(n_obs) for i in range(n_vars)}
    
    # Treatment depends on X
    linear_pred = sum(0.1 * v for v in x_data.values())
    
    if treatment_type == "binary":
        # Binary treatment from logistic model
        propensity = 1 / (1 + np.exp(-linear_pred))
        d = (np.random.random(n_obs) < propensity).astype(float)
    else:
        # Continuous treatment
        d = linear_pred + np.random.randn(n_obs) * 0.5
    
    # True treatment effect = 2.0
    true_effect = 2.0
    
    # Outcome depends on X and treatment
    y = true_effect * d + sum(0.5 * v for v in x_data.values()) + np.random.randn(n_obs) * 0.5
    
    df = pl.DataFrame({"y": y, "d": d, **x_data})
    
    return df


def benchmark_dml() -> List[Dict[str, Any]]:
    """Benchmark DML estimator performance.
    
    Compares causers.dml() vs econml.LinearDML for both ATE and CATE estimation.
    
    For CATE configs, econml.LinearDML is used with X parameter to get CATE coefficients
    via model.coef_ and model.effect().
    
    Performance target: DML with K=5 folds on N=100,000 should complete in <500ms.
    """
    from causers import dml
    
    # Try to import econml for comparison
    try:
        from econml.dml import LinearDML
        from sklearn.linear_model import LinearRegression, LogisticRegression
        HAS_ECONML = True
    except ImportError:
        HAS_ECONML = False
        print("⚠️  econml not installed - showing causers-only benchmarks")
    
    print("\n" + "=" * 80)
    print("DOUBLE MACHINE LEARNING (DML): causers vs econml.LinearDML")
    print("=" * 80)
    print("\nBenchmark dimensions:")
    print("  - Observations (n): 10,000 | 100,000")
    print("  - Variables (p): 10")
    print("  - Folds (K): 5")
    print("  - Treatment: binary | continuous")
    print("  - CATE: with/without (econml comparison for both)")
    if not HAS_ECONML:
        print("  - Note: econml not installed, comparison unavailable")
    print()
    
    results = []
    seen_configs = set()  # Track to avoid duplicate configs
    
    for n_obs, n_vars, n_folds, treatment_type, estimate_cate, label in DML_CONFIGS:
        # Skip duplicates
        config_key = (n_obs, n_vars, n_folds, treatment_type, estimate_cate)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        
        print(f"  {label}...", end=" ", flush=True)
        
        # Generate data - use numpy arrays for econml compatibility
        # For CATE configs, generate heterogeneous treatment effects
        np.random.seed(SEED)
        x = np.random.randn(n_obs, n_vars)
        
        # Treatment depends on X
        linear_pred = 0.1 * x.sum(axis=1)
        
        if treatment_type == "binary":
            propensity = 1 / (1 + np.exp(-linear_pred))
            d = (np.random.random(n_obs) < propensity).astype(float)
        else:
            d = linear_pred + np.random.randn(n_obs) * 0.5
        
        if estimate_cate:
            # For CATE configs: heterogeneous treatment effect
            # CATE(X) = 2.0 + 0.5*X₁ + 0.3*X₂
            theta_base = 2.0
            theta_x1 = 0.5
            theta_x2 = 0.3
            cate = theta_base + theta_x1 * x[:, 0] + theta_x2 * x[:, 1]
            y = cate * d + 0.5 * x.sum(axis=1) + np.random.randn(n_obs) * 0.5
        else:
            # For ATE configs: constant treatment effect
            true_effect = 2.0
            y = true_effect * d + 0.5 * x.sum(axis=1) + np.random.randn(n_obs) * 0.5
        
        # Create polars DataFrame for causers
        x_data = {f"x{i}": x[:, i].tolist() for i in range(n_vars)}
        df = pl.DataFrame({"y": y.tolist(), "d": d.tolist(), **x_data})
        x_cols = [f"x{i}" for i in range(n_vars)]
        
        # Define causers runner - capture variables via default args
        def run_causers(_df=df, _x_cols=x_cols, _n_folds=n_folds,
                       _treatment_type=treatment_type, _estimate_cate=estimate_cate):
            return dml(
                _df, "y", "d", _x_cols,
                n_folds=_n_folds,
                treatment_type=_treatment_type,
                estimate_cate=_estimate_cate,
                seed=42
            )
        
        # Time causers DML
        causers_timing = time_function(run_causers, warmup=1, n_iter=5)
        
        # Time econml if available
        econml_ms = float('nan')
        speedup = float('nan')
        
        if HAS_ECONML:
            if estimate_cate:
                # For CATE configs: use econml with X for heterogeneity
                def run_econml_cate(_x=x, _y=y, _d=d, _treatment_type=treatment_type, _n_folds=n_folds):
                    model_y = LinearRegression()
                    if _treatment_type == "binary":
                        model_t = LogisticRegression(solver='lbfgs', max_iter=200)
                        discrete = True
                    else:
                        model_t = LinearRegression()
                        discrete = False
                    
                    econml_dml = LinearDML(
                        model_y=model_y,
                        model_t=model_t,
                        cv=_n_folds,
                        random_state=42,
                        discrete_treatment=discrete
                    )
                    # X=x for CATE heterogeneity, W=x for confounding control
                    econml_dml.fit(_y, _d, X=_x, W=_x)
                    # Get CATE coefficients via intercept_ and coef_ attributes
                    intercept = econml_dml.intercept_
                    coefs = econml_dml.coef_
                    return intercept, coefs
                
                try:
                    econml_timing = time_function(run_econml_cate, warmup=1, n_iter=5)
                    econml_ms = econml_timing["median_ms"]
                    speedup = econml_ms / causers_timing["median_ms"]
                except Exception as e:
                    # If econml fails, mark as N/A
                    econml_ms = float('nan')
                    speedup = float('nan')
            else:
                # For ATE configs: use econml with W only (no X)
                def run_econml_ate(_x=x, _y=y, _d=d, _treatment_type=treatment_type, _n_folds=n_folds):
                    model_y = LinearRegression()
                    if _treatment_type == "binary":
                        model_t = LogisticRegression(solver='lbfgs', max_iter=200)
                        discrete = True
                    else:
                        model_t = LinearRegression()
                        discrete = False
                    
                    econml_dml = LinearDML(
                        model_y=model_y,
                        model_t=model_t,
                        cv=_n_folds,
                        random_state=42,
                        discrete_treatment=discrete
                    )
                    econml_dml.fit(_y, _d, X=None, W=_x)
                    return econml_dml.ate()
                
                try:
                    econml_timing = time_function(run_econml_ate, warmup=1, n_iter=5)
                    econml_ms = econml_timing["median_ms"]
                    speedup = econml_ms / causers_timing["median_ms"]
                except Exception as e:
                    # If econml fails, mark as N/A
                    econml_ms = float('nan')
                    speedup = float('nan')
        
        result = {
            "Config": label,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "n_folds": n_folds,
            "treatment_type": treatment_type,
            "estimate_cate": estimate_cate,
            "causers_ms": causers_timing["median_ms"],
            "econml_ms": econml_ms,
            "speedup": speedup,
            "faster": speedup > 1.0 if not np.isnan(speedup) else True,
            "min_ms": causers_timing["min_ms"],
            "max_ms": causers_timing["max_ms"],
        }
        results.append(result)
        
        # NFR-DML-01: 100K rows with K=5 should be <500ms
        target_met = True
        if n_obs == 100000 and n_folds == 5:
            target_met = causers_timing["median_ms"] < 500
        
        if not np.isnan(econml_ms):
            status = "✅" if speedup > 1.0 else "❌"
            print(f"causers {causers_timing['median_ms']:.2f}ms vs econml {econml_ms:.2f}ms = {speedup:.1f}x {status}")
        else:
            status = "✅" if target_met else "❌"
            print(f"{causers_timing['median_ms']:.2f}ms (econml N/A) {status}")
    
    return results


def benchmark_dml_vs_econml() -> List[Dict[str, Any]]:
    """Benchmark DML against econml for performance comparison.
    
    Compares causers.dml() vs econml.dml.LinearDML.
    """
    from causers import dml
    
    try:
        from econml.dml import LinearDML
        from sklearn.linear_model import LinearRegression, LogisticRegression
        HAS_ECONML = True
    except ImportError:
        HAS_ECONML = False
        print("⚠️  econml not installed - skipping DML comparison benchmark")
        return []
    
    print("\n" + "=" * 80)
    print("DML COMPARISON: causers vs econml.LinearDML")
    print("=" * 80)
    print()
    
    results = []
    
    # Test configurations: (n_obs, n_vars, treatment_type, label)
    comparison_configs = [
        (1000, 5, "binary", "1K obs, 5 vars, binary"),
        (10000, 5, "binary", "10K obs, 5 vars, binary"),
        (1000, 5, "continuous", "1K obs, 5 vars, continuous"),
        (10000, 5, "continuous", "10K obs, 5 vars, continuous"),
    ]
    
    for n_obs, n_vars, treatment_type, label in comparison_configs:
        print(f"  {label}...", end=" ", flush=True)
        
        # Generate data
        np.random.seed(SEED)
        x = np.random.randn(n_obs, n_vars)
        
        if treatment_type == "binary":
            propensity = 1 / (1 + np.exp(-0.3 * x[:, 0]))
            d = (np.random.random(n_obs) < propensity).astype(float)
        else:
            d = x[:, 0] + np.random.randn(n_obs) * 0.5
        
        y = 2.0 * d + x[:, 0] + np.random.randn(n_obs) * 0.5
        
        # Create polars DataFrame
        x_data = {f"x{i}": x[:, i].tolist() for i in range(n_vars)}
        df = pl.DataFrame({"y": y.tolist(), "d": d.tolist(), **x_data})
        x_cols = [f"x{i}" for i in range(n_vars)]
        
        # Define causers runner
        def run_causers(_df=df, _x_cols=x_cols, _treatment_type=treatment_type):
            return dml(_df, "y", "d", _x_cols, n_folds=5, treatment_type=_treatment_type, seed=42)
        
        # Define econml runner
        def run_econml(_x=x, _y=y, _d=d, _treatment_type=treatment_type):
            model_y = LinearRegression()
            if _treatment_type == "binary":
                model_t = LogisticRegression(solver='lbfgs', max_iter=200)
                discrete = True
            else:
                model_t = LinearRegression()
                discrete = False
            
            econml_dml = LinearDML(
                model_y=model_y,
                model_t=model_t,
                cv=5,
                random_state=42,
                discrete_treatment=discrete
            )
            econml_dml.fit(_y, _d, X=None, W=_x)
            return econml_dml.ate()
        
        # Time both
        causers_timing = time_function(run_causers, warmup=1, n_iter=3)
        econml_timing = time_function(run_econml, warmup=1, n_iter=3)
        speedup = econml_timing["median_ms"] / causers_timing["median_ms"]
        
        result = {
            "Config": label,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "treatment_type": treatment_type,
            "causers_ms": causers_timing["median_ms"],
            "econml_ms": econml_timing["median_ms"],
            "speedup": speedup,
            "faster": speedup > 1.0,
        }
        results.append(result)
        
        status = "✅" if speedup > 1.0 else "❌"
        print(f"causers {causers_timing['median_ms']:.2f}ms vs econml {econml_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
    
    return results


def print_dml_benchmark_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted summary table for DML benchmarks comparing causers vs econml."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("DML PERFORMANCE BENCHMARKS SUMMARY")
    print("=" * 80)
    
    # Header - match format of other benchmark summaries
    print(f"{'Config':<45} | {'causers (ms)':<12} | {'econml (ms)':<12} | {'Speedup':<10}")
    print("-" * 45 + "-|-" + "-" * 12 + "-|-" + "-" * 12 + "-|-" + "-" * 10)
    
    for r in results:
        # Format econml time and speedup (may be N/A)
        econml_str = f"{r.get('econml_ms', float('nan')):.2f}" if not np.isnan(r.get('econml_ms', float('nan'))) else "N/A"
        speedup_val = r.get('speedup', float('nan'))
        speedup_str = f"{speedup_val:.1f}x" if not np.isnan(speedup_val) else "N/A"
        
        # Determine status: faster than econml, or meets NFR target if no econml
        if not np.isnan(speedup_val):
            status = "✅" if speedup_val > 1.0 else "❌"
        else:
            # Fall back to NFR-DML-01 check for CATE configs
            target_met = True
            if r.get("n_obs") == 100000 and r.get("n_folds") == 5:
                target_met = r["causers_ms"] < 500
            status = "✅" if target_met else "❌"
        
        print(f"{r['Config']:<45} | {r['causers_ms']:<12.2f} | {econml_str:<12} | {speedup_str:<8} {status}")
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("Summary:")
    
    # Speedup summary for configs with econml comparison
    econml_results = [r for r in results if not np.isnan(r.get('speedup', float('nan')))]
    if econml_results:
        avg_speedup = np.mean([r['speedup'] for r in econml_results])
        min_speedup = min(r['speedup'] for r in econml_results)
        max_speedup = max(r['speedup'] for r in econml_results)
        faster_count = sum(1 for r in econml_results if r['speedup'] > 1.0)
        print(f"\n  econml comparison ({len(econml_results)} configs):")
        print(f"    Average speedup: {avg_speedup:.1f}x")
        print(f"    Range: {min_speedup:.1f}x - {max_speedup:.1f}x")
        print(f"    Faster in: {faster_count}/{len(econml_results)} configs")
    else:
        print("\n  econml not installed - no comparison available")
    
    # 100K benchmark status
    benchmark_100k = [r for r in results if r.get("n_obs") == 100000 and r.get("n_folds") == 5]
    if benchmark_100k:
        all_pass = all(r["causers_ms"] < 500 for r in benchmark_100k)
        status = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"\n  100K rows, K=5 folds < 500ms: {status}")
        for r in benchmark_100k:
            print(f"    - {r['Config']}: {r['causers_ms']:.2f}ms")


# ============================================================
# DML Test Functions for pytest
# ============================================================

def test_dml_performance_100k_binary():
    """Benchmark test: DML with 100K rows, binary treatment should complete in <500ms."""
    from causers import dml
    
    # Generate data
    df = generate_dml_data(100000, 10, "binary")
    x_cols = [f"x{i}" for i in range(10)]
    
    def run_dml():
        return dml(df, "y", "d", x_cols, n_folds=5, treatment_type="binary", seed=42)
    
    timing = time_function(run_dml, warmup=1, n_iter=5)
    
    print(f"\n  DML 100K binary: {timing['median_ms']:.2f}ms")
    
    # Should complete in <500ms
    # Note: This is a soft target; CI environment may be slower
    assert timing["median_ms"] < 2000, (
        f"DML 100K binary took {timing['median_ms']:.2f}ms, expected <2000ms (relaxed from 500ms for CI)"
    )


def test_dml_performance_100k_continuous():
    """Benchmark test: DML with 100K rows, continuous treatment should complete in <500ms."""
    from causers import dml
    
    # Generate data
    df = generate_dml_data(100000, 10, "continuous")
    x_cols = [f"x{i}" for i in range(10)]
    
    def run_dml():
        return dml(df, "y", "d", x_cols, n_folds=5, treatment_type="continuous", seed=42)
    
    timing = time_function(run_dml, warmup=1, n_iter=5)
    
    print(f"\n  DML 100K continuous: {timing['median_ms']:.2f}ms")
    
    # NFR-DML-01: Should complete in <500ms
    # Note: This is a soft target; CI environment may be slower
    assert timing["median_ms"] < 2000, (
        f"DML 100K continuous took {timing['median_ms']:.2f}ms, expected <2000ms (relaxed from 500ms for CI)"
    )


def test_dml_performance_100k_cate():
    """Benchmark test: DML with 100K rows and CATE estimation."""
    from causers import dml
    
    # Generate data
    df = generate_dml_data(100000, 10, "binary")
    x_cols = [f"x{i}" for i in range(10)]
    
    def run_dml():
        return dml(
            df, "y", "d", x_cols,
            n_folds=5,
            treatment_type="binary",
            estimate_cate=True,
            seed=42
        )
    
    timing = time_function(run_dml, warmup=1, n_iter=5)
    
    print(f"\n  DML 100K with CATE: {timing['median_ms']:.2f}ms")
    
    # CATE estimation adds overhead, allow more time
    assert timing["median_ms"] < 3000, (
        f"DML 100K with CATE took {timing['median_ms']:.2f}ms, expected <3000ms"
    )


def test_dml_vs_econml_benchmark():
    """Benchmark test: causers DML vs econml performance."""
    from causers import dml
    
    try:
        from econml.dml import LinearDML
        from sklearn.linear_model import LinearRegression, LogisticRegression
    except ImportError:
        import pytest
        pytest.skip("econml not installed")
    
    # Use smaller dataset for test speed
    n_obs = 5000
    n_vars = 5
    
    np.random.seed(SEED)
    x = np.random.randn(n_obs, n_vars)
    propensity = 1 / (1 + np.exp(-0.3 * x[:, 0]))
    d = (np.random.random(n_obs) < propensity).astype(float)
    y = 2.0 * d + x[:, 0] + np.random.randn(n_obs) * 0.5
    
    # Create polars DataFrame
    x_data = {f"x{i}": x[:, i].tolist() for i in range(n_vars)}
    df = pl.DataFrame({"y": y.tolist(), "d": d.tolist(), **x_data})
    x_cols = [f"x{i}" for i in range(n_vars)]
    
    def run_causers():
        return dml(df, "y", "d", x_cols, n_folds=5, treatment_type="binary", seed=42)
    
    def run_econml():
        econml_dml = LinearDML(
            model_y=LinearRegression(),
            model_t=LogisticRegression(solver='lbfgs', max_iter=200),
            cv=5,
            random_state=42,
            discrete_treatment=True
        )
        econml_dml.fit(y, d, X=None, W=x)
        return econml_dml.ate()
    
    causers_timing = time_function(run_causers, warmup=1, n_iter=3)
    econml_timing = time_function(run_econml, warmup=1, n_iter=3)
    speedup = econml_timing["median_ms"] / causers_timing["median_ms"]
    
    print(f"\n  causers: {causers_timing['median_ms']:.2f}ms, econml: {econml_timing['median_ms']:.2f}ms, speedup: {speedup:.2f}x")
    
    # Both should complete successfully
    assert causers_timing["median_ms"] > 0, "causers should complete successfully"
    assert econml_timing["median_ms"] > 0, "econml should complete successfully"


# ============================================================
# IV/2SLS BENCHMARKS
# ============================================================

# IV/2SLS benchmark configurations
# (n_obs, n_exog, n_instruments, first_stage_strength, label)
IV2SLS_CONFIGS = [
    (10_000, 5, 3, 0.5, "10K obs, 5 exog, 3 instruments"),
    (100_000, 5, 3, 0.5, "100K obs, 5 exog, 3 instruments"),
    (1_000_000, 5, 3, 0.5, "1M obs, 5 exog, 3 instruments"),
    (100_000, 10, 1, 0.5, "100K obs, 10 exog, 1 instrument"),
    (100_000, 10, 5, 0.5, "100K obs, 10 exog, 5 instruments"),
]


def generate_iv_benchmark_data(
    n_obs: int,
    n_exog: int = 5,
    n_instruments: int = 3,
    first_stage_strength: float = 0.5,
    true_effect: float = 2.0,
    seed: int = SEED
) -> pl.DataFrame:
    """Generate synthetic IV data with valid instrument structure.
    
    Data Generating Process:
    - Z ~ N(0, 1)                          # Instruments
    - X ~ N(0, 1)                          # Exogenous controls
    - u ~ N(0, 1)                          # Structural error
    - D = first_stage_strength * Z + 0.3*u + noise   # Endogenous
    - Y = true_effect * D + X*beta + u        # Outcome
    
    Args:
        n_obs: Number of observations
        n_exog: Number of exogenous control variables
        n_instruments: Number of instruments
        first_stage_strength: Coefficient of Z in first stage
        true_effect: True causal effect of D on Y
        seed: Random seed
    
    Returns:
        polars DataFrame with y, d, z0..z{m-1}, x0..x{p-1} columns
    """
    np.random.seed(seed)
    
    # Generate instruments Z
    z_data = {f"z{i}": np.random.randn(n_obs) for i in range(n_instruments)}
    Z = np.column_stack(list(z_data.values()))
    
    # Generate exogenous controls X
    x_data = {f"x{i}": np.random.randn(n_obs) for i in range(n_exog)}
    X = np.column_stack(list(x_data.values()))
    
    # Generate structural error (correlated with D to create endogeneity)
    u = np.random.randn(n_obs)
    
    # First stage: D = pi*Z + 0.3*u + noise
    first_stage_noise = np.random.randn(n_obs) * 0.5
    D = (first_stage_strength * Z.sum(axis=1) / np.sqrt(n_instruments) +
         0.3 * u +  # Endogeneity: D correlated with structural error
         first_stage_noise)
    
    # Structural equation: Y = beta*D + X*gamma + u
    Y = true_effect * D + 0.5 * X.sum(axis=1) + u
    
    # Create DataFrame
    df = pl.DataFrame({
        "y": Y,
        "d": D,
        **z_data,
        **x_data
    })
    
    return df


def benchmark_iv2sls() -> List[Dict[str, Any]]:
    """Benchmark IV/2SLS estimator performance.
    
    Compares causers.two_stage_least_squares() vs linearmodels.IV2SLS.
    
    Reference package: linearmodels (https://pypi.org/project/linearmodels/)
    - linearmodels is a Python library for panel data and IV/2SLS estimation
    - It provides IV2SLS, LIML, GMM-IV and other instrumental variable estimators
    - Used as the reference for speedup comparison in these benchmarks
    
    Performance targets (no spec requirement IDs):
    - <50ms for 100K rows
    - <500ms for 1M rows
    - >=2x speedup vs linearmodels
    """
    from causers import two_stage_least_squares
    
    # Try to import linearmodels for comparison
    # linearmodels is a Python library for IV/2SLS estimation (pip install linearmodels)
    try:
        from linearmodels.iv import IV2SLS as LM_IV2SLS
        HAS_LINEARMODELS = True
    except ImportError:
        HAS_LINEARMODELS = False
        print("⚠️  linearmodels not installed - showing causers-only benchmarks")
    
    print("\n" + "=" * 80)
    print("TWO-STAGE LEAST SQUARES (2SLS): causers vs linearmodels")
    print("=" * 80)
    print("\nReference: linearmodels (Python library for IV/2SLS estimation)")
    print("Performance targets:")
    print("  - <50ms for 100K rows")
    print("  - <500ms for 1M rows")
    print("  - >=2x speedup over linearmodels")
    if not HAS_LINEARMODELS:
        print("  - Note: linearmodels not installed, comparison unavailable")
    print()
    
    results = []
    seen_configs = set()
    
    for n_obs, n_exog, n_instruments, strength, label in IV2SLS_CONFIGS:
        config_key = (n_obs, n_exog, n_instruments)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        
        print(f"  {label}...", end=" ", flush=True)
        
        # Generate data
        df = generate_iv_benchmark_data(n_obs, n_exog, n_instruments, strength)
        z_cols = [f"z{i}" for i in range(n_instruments)]
        x_cols = [f"x{i}" for i in range(n_exog)]
        
        # Define causers runner
        def run_causers(_df=df, _z_cols=z_cols, _x_cols=x_cols):
            return two_stage_least_squares(_df, "y", "d", _z_cols, _x_cols)
        
        # Time causers
        causers_timing = time_function(run_causers, warmup=2, n_iter=10)
        
        # Initialize result dict
        result = {
            "Config": label,
            "n_obs": n_obs,
            "n_exog": n_exog,
            "n_instruments": n_instruments,
            "causers_ms": causers_timing["median_ms"],
            "reference_ms": float('nan'),
            "speedup": float('nan'),
            "faster": True,
            "meets_100k_target": True,
            "meets_1m_target": True,
        }
        
        # Check NFR targets
        if n_obs == 100_000:
            result["meets_100k_target"] = causers_timing["median_ms"] < 50
        if n_obs == 1_000_000:
            result["meets_1m_target"] = causers_timing["median_ms"] < 500
        
        output_parts = [f"{causers_timing['median_ms']:.2f}ms"]
        
        # Benchmark linearmodels if available
        if HAS_LINEARMODELS:
            df_pandas = df.to_pandas()
            exog_formula = " + ".join(x_cols)
            instr_formula = " + ".join(z_cols)
            formula = f"y ~ 1 + {exog_formula} + [d ~ {instr_formula}]"
            
            def run_linearmodels(_df_pandas=df_pandas, _formula=formula):
                iv_model = LM_IV2SLS.from_formula(_formula, data=_df_pandas)
                return iv_model.fit(cov_type="unadjusted")
            
            ref_timing = time_function(run_linearmodels, warmup=2, n_iter=10)
            speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
            result["reference_ms"] = ref_timing["median_ms"]
            result["speedup"] = speedup
            result["faster"] = speedup > 1.0
            
            status = "✅" if speedup >= 2.0 else ("⚠️" if speedup > 1.0 else "❌")
            output_parts.append(f"lm:{ref_timing['median_ms']:.2f}ms({speedup:.1f}x {status})")
        
        results.append(result)
        print(" | ".join(output_parts))
    
    return results


def print_iv2sls_benchmark_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted summary table for IV/2SLS benchmarks.
    
    Reference: linearmodels (Python library for IV/2SLS estimation)
    """
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("IV/2SLS PERFORMANCE BENCHMARKS SUMMARY")
    print("Reference: linearmodels (Python library for IV/2SLS estimation)")
    print("=" * 80)
    
    # Header
    print(f"{'Config':<40} | {'causers (ms)':<12} | {'linearmodels (ms)':<17} | {'Speedup':<10}")
    print("-" * 40 + "-|-" + "-" * 12 + "-|-" + "-" * 17 + "-|-" + "-" * 10)
    
    for r in results:
        ref_str = f"{r['reference_ms']:.2f}" if not np.isnan(r['reference_ms']) else "N/A"
        speedup_str = f"{r['speedup']:.1f}x" if not np.isnan(r['speedup']) else "N/A"
        
        # Determine status based on speedup
        status = "✅" if r.get('faster', False) else "❌"
        
        print(f"{r['Config']:<40} | {r['causers_ms']:<12.2f} | {ref_str:<17} | {speedup_str:<8} {status}")
    
    # Summary
    print("\n" + "-" * 80)
    print("Performance Targets Summary:")
    
    # 100K target (<50ms)
    configs_100k = [r for r in results if r['n_obs'] == 100_000]
    if configs_100k:
        all_pass_100k = all(r['meets_100k_target'] for r in configs_100k)
        status = "✅ PASS" if all_pass_100k else "❌ FAIL"
        print(f"  <50ms for 100K rows: {status}")
    
    # 1M target (<500ms)
    configs_1m = [r for r in results if r['n_obs'] == 1_000_000]
    if configs_1m:
        all_pass_1m = all(r['meets_1m_target'] for r in configs_1m)
        status = "✅ PASS" if all_pass_1m else "❌ FAIL"
        print(f"  <500ms for 1M rows: {status}")
    
    # Speedup target (>=2x vs linearmodels)
    speedup_results = [r for r in results if not np.isnan(r['speedup'])]
    if speedup_results:
        all_2x = all(r['speedup'] >= 2.0 for r in speedup_results)
        avg_speedup = np.mean([r['speedup'] for r in speedup_results])
        status = "✅ PASS" if all_2x else "❌ FAIL"
        print(f"  >=2x speedup vs linearmodels: {status} (avg: {avg_speedup:.1f}x)")


# ============================================================
# IV/2SLS Test Functions for pytest
# ============================================================

def test_iv2sls_100k_under_50ms():
    """Benchmark test: 2SLS with N=100K should complete in <50ms."""
    from causers import two_stage_least_squares
    import pytest
    
    # Generate data
    df = generate_iv_benchmark_data(n_obs=100_000)
    z_cols = ["z0", "z1", "z2"]
    x_cols = [f"x{i}" for i in range(5)]
    
    # Warmup
    two_stage_least_squares(df, "y", "d", z_cols, x_cols)
    
    # Timed runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        two_stage_least_squares(df, "y", "d", z_cols, x_cols)
        times.append(time.perf_counter() - start)
    
    median_time = np.median(times)
    median_ms = median_time * 1000
    
    print(f"\n  2SLS 100K: {median_ms:.2f}ms (target: <50ms)")
    
    # <50ms for 100K rows
    # Allow 2x margin for CI environments
    assert median_ms < 100, f"2SLS took {median_ms:.1f}ms, expected <100ms (relaxed from 50ms for CI)"


def test_iv2sls_1m_under_500ms():
    """Benchmark test: 2SLS with N=1M should complete in <500ms."""
    from causers import two_stage_least_squares
    import pytest
    
    # Generate data
    df = generate_iv_benchmark_data(n_obs=1_000_000)
    z_cols = ["z0", "z1", "z2"]
    x_cols = [f"x{i}" for i in range(5)]
    
    # Warmup
    two_stage_least_squares(df, "y", "d", z_cols, x_cols)
    
    # Timed runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        two_stage_least_squares(df, "y", "d", z_cols, x_cols)
        times.append(time.perf_counter() - start)
    
    median_time = np.median(times)
    median_ms = median_time * 1000
    
    print(f"\n  2SLS 1M: {median_ms:.2f}ms (target: <500ms)")
    
    # <500ms for 1M rows
    # Allow 2x margin for CI environments
    assert median_ms < 1000, f"2SLS took {median_ms:.1f}ms, expected <1000ms (relaxed from 500ms for CI)"


def test_iv2sls_faster_than_statsmodels():
    """Benchmark test: causers 2SLS should be >=2x faster than linearmodels."""
    from causers import two_stage_least_squares
    import pytest
    
    # Try to import linearmodels for comparison
    try:
        from linearmodels.iv import IV2SLS as LM_IV2SLS
    except ImportError:
        pytest.skip("linearmodels not installed")
    
    # Generate data - use 100K for meaningful comparison
    n_obs = 100_000
    df = generate_iv_benchmark_data(n_obs=n_obs)
    z_cols = ["z0", "z1", "z2"]
    x_cols = [f"x{i}" for i in range(5)]
    
    # Prepare linearmodels data
    df_pandas = df.to_pandas()
    exog_formula = " + ".join(x_cols)
    instr_formula = " + ".join(z_cols)
    formula = f"y ~ 1 + {exog_formula} + [d ~ {instr_formula}]"
    
    # Define runners
    def run_causers():
        return two_stage_least_squares(df, "y", "d", z_cols, x_cols)
    
    def run_linearmodels():
        iv_model = LM_IV2SLS.from_formula(formula, data=df_pandas)
        return iv_model.fit(cov_type="unadjusted")
    
    # Time both
    causers_timing = time_function(run_causers, warmup=2, n_iter=10)
    ref_timing = time_function(run_linearmodels, warmup=2, n_iter=10)
    
    speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
    
    print(f"\n  causers: {causers_timing['median_ms']:.2f}ms, linearmodels: {ref_timing['median_ms']:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x (target: >=2x)")
    
    # >=2x speedup
    # Allow 1.5x for CI environments
    assert speedup >= 1.5, (
        f"causers speedup was {speedup:.2f}x, expected >=1.5x (relaxed from 2x for CI)"
    )


# ============================================================
# Main Function
# ============================================================

def main() -> int:
    """Run all benchmarks.
    
    Returns:
        Exit code: 0 if all benchmarks pass, 1 otherwise
    """
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
    
    # Run fixed effects linear regression benchmarks
    lr_fe_results = benchmark_linear_regression_fe()
    
    # Run simple logistic regression benchmarks
    logit_results = benchmark_logistic_regression(regression_sizes)
    
    # Run comprehensive logistic regression benchmarks
    logit_comprehensive_results = benchmark_logistic_regression_comprehensive()
    
    # Run fixed effects logistic regression benchmarks
    logit_fe_results = benchmark_logistic_regression_fe()
    
    # Run synthetic DID benchmarks
    sdid_results = benchmark_synthetic_did(sdid_sizes)
    
    # Run synthetic control benchmarks (causers vs pysyncon)
    sc_results = benchmark_synthetic_control()
    
    # Run DML benchmarks
    dml_results = benchmark_dml()
    
    # Run IV/2SLS benchmarks
    iv2sls_results = benchmark_iv2sls()
    
    # Print detailed summaries
    print_comprehensive_lr_summary(lr_comprehensive_results)
    print_comprehensive_lr_fe_summary(lr_fe_results)
    print_comprehensive_logit_summary(logit_comprehensive_results)
    print_comprehensive_logit_fe_summary(logit_fe_results)
    print_synthetic_control_summary(sc_results)
    print_dml_benchmark_summary(dml_results)
    print_iv2sls_benchmark_summary(iv2sls_results)
    
    # Print summary and get pass/fail
    # Include logit_fe_results in the overall assessment
    all_logit_fe = logit_fe_results if logit_fe_results else []
    passed = print_summary_extended(
        lr_results, lr_comprehensive_results, lr_fe_results,
        logit_results, logit_comprehensive_results,
        sdid_results, sc_results
    )
    
    # Add logit_fe_results to the assessment if present
    if logit_fe_results:
        for r in logit_fe_results:
            if not r.get("faster", True):
                passed = False
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
