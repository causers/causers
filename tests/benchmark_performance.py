#!/usr/bin/env python3
"""
Performance benchmark: causers vs reference packages.

Compares:
- linear_regression vs statsmodels.OLS
- logistic_regression vs statsmodels.Logit
- synthetic_did vs azcausal.SDID
- synthetic_control vs pysyncon (all four methods)

Target: causers should be faster than reference packages.

Usage:
    python tests/benchmark_performance.py
"""

import sys
import time
import warnings
from typing import Callable, Dict, Any, List, Tuple

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
    """Generate linear regression test data."""
    np.random.seed(seed)
    x = np.random.randn(n_samples)
    y = 2 * x + 1 + np.random.randn(n_samples) * 0.5
    df = pl.DataFrame({"x": x, "y": y})
    return df, x, y


def generate_logistic_data(n_samples: int, seed: int = SEED) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Generate logistic regression test data."""
    np.random.seed(seed)
    x = np.random.randn(n_samples)
    prob = 1 / (1 + np.exp(-(0.5 + x)))
    y = (np.random.rand(n_samples) < prob).astype(float)
    df = pl.DataFrame({"x": x, "y": y})
    return df, x, y


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


def generate_sc_panel(
    n_control: int,
    n_pre: int,
    n_post: int,
    effect: float = 5.0,
    seed: int = SEED
) -> pl.DataFrame:
    """Generate synthetic control panel data in long format.
    
    Creates panel data with exactly 1 treated unit (unit 0) and
    n_control control units, suitable for synthetic control estimation.
    
    Args:
        n_control: Number of control units
        n_pre: Number of pre-treatment periods
        n_post: Number of post-treatment periods
        effect: Treatment effect to apply in post-period
        seed: Random seed for reproducibility
        
    Returns:
        Polars DataFrame with columns: unit, time, y, treated
    """
    np.random.seed(seed)
    n_periods = n_pre + n_post
    
    data = {"unit": [], "time": [], "y": [], "treated": []}
    
    # Treated unit (unit 0)
    for t in range(n_periods):
        data["unit"].append(0)
        data["time"].append(t)
        base = 10.0 + t * 0.5
        if t >= n_pre:
            data["y"].append(base + effect)
            data["treated"].append(1)
        else:
            data["y"].append(base)
            data["treated"].append(0)
    
    # Control units (units 1 to n_control)
    for unit_id in range(1, n_control + 1):
        unit_effect = np.random.uniform(-2.0, 2.0)
        for t in range(n_periods):
            data["unit"].append(unit_id)
            data["time"].append(t)
            base = 10.0 + unit_effect + t * 0.5
            data["y"].append(base)
            data["treated"].append(0)
    
    return pl.DataFrame(data)


def generate_pysyncon_matrices(
    df: pl.DataFrame,
    n_pre: int,
    n_post: int,
    n_control: int
):
    """Convert polars panel to pysyncon matrix format.
    
    Args:
        df: Polars panel DataFrame
        n_pre: Number of pre-treatment periods
        n_post: Number of post-treatment periods
        n_control: Number of control units
        
    Returns:
        Tuple of (Z0_pre, Z1_pre, Z0_all, Z1_all, post_periods)
    """
    import pandas as pd
    
    # Convert to pandas
    df_pd = pd.DataFrame({col: df[col].to_list() for col in df.columns})
    
    # Pre-period outcomes for controls
    Z0_pre = df_pd[(df_pd['unit'] > 0) & (df_pd['time'] < n_pre)].pivot(
        index='time', columns='unit', values='y'
    )
    Z1_pre = df_pd[(df_pd['unit'] == 0) & (df_pd['time'] < n_pre)].set_index('time')['y']
    
    # All outcomes
    Z0_all = df_pd[df_pd['unit'] > 0].pivot(index='time', columns='unit', values='y')
    Z1_all = df_pd[df_pd['unit'] == 0].set_index('time')['y']
    
    post_periods = list(range(n_pre, n_pre + n_post))
    
    return Z0_pre, Z1_pre, Z0_all, Z1_all, post_periods


# ============================================================
# Benchmark functions
# ============================================================

def benchmark_linear_regression(sizes: Dict[str, int]) -> List[Dict[str, Any]]:
    """Benchmark linear_regression vs statsmodels.OLS."""
    from causers import linear_regression
    
    try:
        import statsmodels.api as sm
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False
        print("⚠️  statsmodels not installed - skipping linear_regression comparison")
        return []
    
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION: causers vs statsmodels.OLS")
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
                               bootstrap_iterations=50, seed=SEED)
    
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
        
        causers_timing = time_function(run_causers, panel)
        ref_timing = time_function(run_azcausal, outcome_wide, intervention_wide)
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


def benchmark_synthetic_control(
    sizes: Dict[str, Tuple[int, int, int]]
) -> List[Dict[str, Any]]:
    """Benchmark synthetic_control vs pysyncon for all four method variants.
    
    Methods benchmarked:
    - traditional: Classic SC (Abadie et al., 2010)
    - penalized: L2 regularized SC
    - robust: De-meaned/variance-weighted SC
    - augmented: Bias-corrected SC (Ben-Michael et al., 2021)
    
    Args:
        sizes: Dict mapping size name to (n_control, n_pre, n_post) tuples
        
    Returns:
        List of benchmark results with method, size, timing, and speedup info
    """
    from causers import synthetic_control
    
    # Check if pysyncon is available
    try:
        from pysyncon import Synth, PenalizedSynth, AugSynth, Dataprep
        HAS_PYSYNCON = True
    except ImportError:
        HAS_PYSYNCON = False
        print("⚠️  pysyncon not installed - skipping synthetic_control comparison")
        return []
    
    print("\n" + "=" * 60)
    print("SYNTHETIC CONTROL: causers vs pysyncon")
    print("=" * 60)
    
    # Method configurations
    # Note: robust SC is causers-only (pysyncon RobustSynth has bugs)
    methods = ["traditional", "penalized", "robust", "augmented"]
    
    results = []
    
    for size_name, (n_control, n_pre, n_post) in sizes.items():
        print(f"\n  Panel size: {n_control + 1} units × {n_pre + n_post} periods")
        
        # Generate panel data (same for all methods)
        panel = generate_sc_panel(n_control, n_pre, n_post, effect=5.0, seed=SEED)
        
        # Prepare pysyncon matrices once
        Z0_pre, Z1_pre, Z0_all, Z1_all, post_periods = generate_pysyncon_matrices(
            panel, n_pre, n_post, n_control
        )
        
        for method in methods:
            print(f"    {method}...", end=" ", flush=True)
            
            # Define causers runner
            def run_causers():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return synthetic_control(
                        panel, "unit", "time", "y", "treated",
                        method=method,
                        lambda_param=0.1 if method in ("penalized", "augmented") else None,
                        compute_se=False,
                        seed=SEED
                    )
            
            # Time causers
            causers_timing = time_function(run_causers)
            
            # Define and time pysyncon runner (method-specific)
            ref_timing = None
            pysyncon_method_name = method
            
            if method == "traditional":
                def run_pysyncon():
                    synth = Synth()
                    synth.fit(X0=Z0_pre, X1=Z1_pre, Z0=Z0_all, Z1=Z1_all)
                    return synth.att(time_period=post_periods, Z0=Z0_all, Z1=Z1_all)
                ref_timing = time_function(run_pysyncon)
                
            elif method == "penalized":
                def run_pysyncon():
                    penalized = PenalizedSynth()
                    penalized.fit(X0=Z0_pre, X1=Z1_pre, lambda_=0.1)
                    return penalized.att(time_period=post_periods, Z0=Z0_all, Z1=Z1_all)
                ref_timing = time_function(run_pysyncon)
                
            elif method == "augmented":
                # AugSynth requires Dataprep object
                import pandas as pd
                df_pd = pd.DataFrame({col: panel[col].to_list() for col in panel.columns})
                dataprep = Dataprep(
                    foo=df_pd,
                    predictors=['y'],
                    predictors_op='mean',
                    dependent='y',
                    unit_variable='unit',
                    time_variable='time',
                    treatment_identifier=0,
                    controls_identifier=list(range(1, n_control + 1)),
                    time_predictors_prior=list(range(n_pre)),
                    time_optimize_ssr=list(range(n_pre)),
                )
                def run_pysyncon():
                    augsynth = AugSynth()
                    augsynth.fit(dataprep=dataprep, lambda_=0.1)
                    return augsynth.att(time_period=post_periods)
                ref_timing = time_function(run_pysyncon)
                
            elif method == "robust":
                # pysyncon RobustSynth has bugs, benchmark causers only
                pysyncon_method_name = "(causers only)"
                ref_timing = None
            
            # Calculate speedup and record result
            if ref_timing is not None:
                speedup = ref_timing["median_ms"] / causers_timing["median_ms"]
                result = {
                    "Dataset": f"{size_name}/{method}",
                    "Method": method,
                    "Size": size_name,
                    "causers_ms": causers_timing["median_ms"],
                    "reference_ms": ref_timing["median_ms"],
                    "speedup": speedup,
                    "faster": speedup > 1.0,
                }
                results.append(result)
                
                status = "✅ FASTER" if speedup > 1.0 else "❌ SLOWER"
                print(f"{causers_timing['median_ms']:.2f}ms vs {ref_timing['median_ms']:.2f}ms = {speedup:.2f}x {status}")
            else:
                # Robust method - no pysyncon reference
                result = {
                    "Dataset": f"{size_name}/{method}",
                    "Method": method,
                    "Size": size_name,
                    "causers_ms": causers_timing["median_ms"],
                    "reference_ms": None,
                    "speedup": None,
                    "faster": None,
                }
                results.append(result)
                print(f"{causers_timing['median_ms']:.2f}ms {pysyncon_method_name}")
    
    return results


# ============================================================
# Summary and reporting
# ============================================================

def print_summary(
    lr_results: List[Dict],
    logit_results: List[Dict],
    sdid_results: List[Dict],
    sc_results: List[Dict] = None
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
    
    if sc_results:
        print("\nSynthetic Control:")
        print(f"  {'Dataset':<20} {'causers':<12} {'pysyncon':<12} {'Speedup':<10} {'Status'}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for r in sc_results:
            if r["faster"] is not None:
                status = "✅ PASS" if r["faster"] else "❌ FAIL"
                print(f"  {r['Dataset']:<20} {r['causers_ms']:<12.2f} {r['reference_ms']:<12.2f} {r['speedup']:<10.2f}x {status}")
                all_results.append(r)
            else:
                # Robust method has no reference
                print(f"  {r['Dataset']:<20} {r['causers_ms']:<12.2f} {'(no ref)':<12} {'N/A':<10} ⏭️ SKIP")
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)
    
    if not all_results:
        print("⚠️  No reference packages installed - cannot assess performance")
        return False
    
    total = len(all_results)
    faster_count = sum(1 for r in all_results if r.get("faster"))
    
    print(f"Benchmarks where causers is faster: {faster_count}/{total}")
    
    if faster_count == total:
        print("✅ PASS: All benchmarks show causers is faster than reference packages")
        return True
    else:
        slower = [r for r in all_results if r.get("faster") is False]
        print(f"❌ FAIL: causers is slower in {len(slower)} benchmark(s):")
        for r in slower:
            print(f"    - {r['Dataset']}: {r['speedup']:.2f}x")
        return False


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("CAUSERS PERFORMANCE BENCHMARK")
    print("=" * 60)
    print("Comparing causers against reference packages...")
    print("Build: maturin develop --release")
    
    # Define dataset sizes
    regression_sizes = {
        "1K": 1_000,
        "10K": 10_000,
        "100K": 100_000,
    }
    
    sdid_sizes = {
        "10x20": (10, 20, 2, 16),      # n_units, n_periods, n_treated, n_pre
        "50x50": (50, 50, 10, 40),
    }
    
    # Synthetic control sizes: (n_control, n_pre, n_post)
    sc_sizes = {
        "20x10": (20, 8, 2),           # 21 units × 10 periods
        "50x20": (50, 16, 4),          # 51 units × 20 periods
    }
    
    # Run benchmarks
    lr_results = benchmark_linear_regression(regression_sizes)
    logit_results = benchmark_logistic_regression(regression_sizes)
    sdid_results = benchmark_synthetic_did(sdid_sizes)
    sc_results = benchmark_synthetic_control(sc_sizes)
    
    # Print summary and get pass/fail
    passed = print_summary(lr_results, logit_results, sdid_results, sc_results)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
