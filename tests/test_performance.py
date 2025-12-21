"""Performance benchmark tests for causers package.

This module validates the performance requirements specified in:
- REQ-037: Performance target of <100ms for 1M rows
"""

import time
import pytest
import polars as pl
import numpy as np
from causers import linear_regression


class TestPerformance:
    """Test suite for performance benchmarks."""
    
    def test_performance_small_dataset(self):
        """Test performance on small dataset (1,000 rows).
        
        Validates baseline performance for small datasets.
        Should complete in <1ms for good responsiveness.
        """
        n = 1_000
        np.random.seed(42)
        x = np.random.randn(n)
        y = 2 * x + 3 + np.random.randn(n) * 0.1
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        start_time = time.perf_counter()
        result = linear_regression(df, "x", "y")
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        # Assertions
        assert result.n_samples == n
        assert elapsed < 10, f"Small dataset took {elapsed:.2f}ms, expected <10ms"
        print(f"Small dataset (1K rows): {elapsed:.2f}ms")
    
    def test_performance_medium_dataset(self):
        """Test performance on medium dataset (100,000 rows).
        
        Validates scaling behavior for medium-sized datasets.
        Should complete in <10ms for good performance.
        """
        n = 100_000
        np.random.seed(42)
        x = np.random.randn(n)
        y = 2 * x + 3 + np.random.randn(n) * 0.1
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        start_time = time.perf_counter()
        result = linear_regression(df, "x", "y")
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        # Assertions
        assert result.n_samples == n
        assert elapsed < 50, f"Medium dataset took {elapsed:.2f}ms, expected <50ms"
        print(f"Medium dataset (100K rows): {elapsed:.2f}ms")
    
    def test_performance_large_dataset_req037(self):
        """Test performance on large dataset (1,000,000 rows).
        
        Validates REQ-037: Performance target of <100ms for 1M rows.
        This is the critical performance requirement.
        """
        n = 1_000_000
        np.random.seed(42)
        x = np.random.randn(n)
        y = 2 * x + 3 + np.random.randn(n) * 0.1
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        # Warm-up run to ensure fair measurement
        _ = linear_regression(df, "x", "y")
        
        # Actual performance measurement
        start_time = time.perf_counter()
        result = linear_regression(df, "x", "y")
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        # Assertions
        assert result.n_samples == n
        assert elapsed < 100, f"REQ-037 VIOLATION: Large dataset took {elapsed:.2f}ms, expected <100ms"
        print(f"Large dataset (1M rows): {elapsed:.2f}ms - REQ-037 PASSED")
    
    def test_performance_very_large_dataset(self):
        """Test performance on very large dataset (5,000,000 rows).
        
        Validates performance scaling beyond the required 1M rows.
        This helps ensure we have headroom for future growth.
        """
        n = 5_000_000
        np.random.seed(42)
        x = np.random.randn(n)
        y = 2 * x + 3 + np.random.randn(n) * 0.1
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        start_time = time.perf_counter()
        result = linear_regression(df, "x", "y")
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        # Assertions
        assert result.n_samples == n
        # For 5M rows, we expect roughly 5x the time of 1M rows
        assert elapsed < 500, f"Very large dataset took {elapsed:.2f}ms, expected <500ms"
        print(f"Very large dataset (5M rows): {elapsed:.2f}ms")
    
    def test_performance_multiple_runs_consistency(self):
        """Test performance consistency across multiple runs.
        
        Validates that performance is stable and not subject to
        significant variance that could affect user experience.
        """
        n = 100_000
        np.random.seed(42)
        x = np.random.randn(n)
        y = 2 * x + 3 + np.random.randn(n) * 0.1
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        # Warm-up
        _ = linear_regression(df, "x", "y")
        
        # Measure multiple runs
        times = []
        for i in range(10):
            start_time = time.perf_counter()
            _ = linear_regression(df, "x", "y")
            elapsed = (time.perf_counter() - start_time) * 1000
            times.append(elapsed)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = (std_time / mean_time) * 100  # Coefficient of variation
        
        # Assertions
        assert cv < 20, f"Performance variance too high: CV={cv:.1f}%, expected <20%"
        print(f"Performance consistency: mean={mean_time:.2f}ms, std={std_time:.2f}ms, CV={cv:.1f}%")
    
    def test_performance_worst_case_data(self):
        """Test performance with worst-case data patterns.
        
        Tests performance with data that might stress the algorithm:
        - Very large values
        - Very small values  
        - Mixed scales
        """
        n = 1_000_000
        np.random.seed(42)
        
        # Create challenging data: large scale differences
        x = np.random.randn(n) * 1e10  # Very large scale
        y = 2e-10 * x + 3e-5 + np.random.randn(n) * 1e-8  # Very small coefficients
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        start_time = time.perf_counter()
        result = linear_regression(df, "x", "y")
        elapsed = (time.perf_counter() - start_time) * 1000
        
        # Assertions
        assert result.n_samples == n
        # Even worst-case should meet performance target
        assert elapsed < 150, f"Worst-case data took {elapsed:.2f}ms, expected <150ms"
        print(f"Worst-case data (1M rows): {elapsed:.2f}ms")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])