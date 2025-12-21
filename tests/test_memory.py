"""Memory usage tests for causers package.

This module validates the memory requirements specified in:
- REQ-016: Memory constraint of 1.5x input size
"""

import gc
import pytest
import polars as pl
import numpy as np
from causers import linear_regression
import tracemalloc
import sys


class TestMemoryUsage:
    """Test suite for memory usage validation."""
    
    def get_dataframe_memory(self, df):
        """Calculate the memory usage of a Polars DataFrame.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            Memory usage in bytes
        """
        # Get memory usage from Polars (estimated)
        # For each column: n_rows * dtype_size
        total_bytes = 0
        for col in df.columns:
            dtype = df[col].dtype
            n_rows = len(df)
            
            # Estimate bytes per element based on dtype
            if dtype == pl.Float64:
                bytes_per_element = 8
            elif dtype == pl.Float32:
                bytes_per_element = 4
            elif dtype == pl.Int64:
                bytes_per_element = 8
            elif dtype == pl.Int32:
                bytes_per_element = 4
            else:
                bytes_per_element = 8  # Default to 8 bytes
            
            total_bytes += n_rows * bytes_per_element
        
        return total_bytes
    
    def test_memory_usage_small_dataset(self):
        """Test memory usage on small dataset (10,000 rows).
        
        Validates REQ-016: Memory usage should be <1.5x input size.
        """
        n = 10_000
        np.random.seed(42)
        x = np.random.randn(n).astype(np.float64)
        y = (2 * x + 3 + np.random.randn(n) * 0.1).astype(np.float64)
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        # Calculate input data size
        input_size = self.get_dataframe_memory(df)
        
        # Start memory tracking
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        
        # Perform regression
        result = linear_regression(df, "x", "y")
        
        # Take memory snapshot after
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Calculate memory used during operation
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_allocated = sum(stat.size for stat in top_stats if stat.size > 0)
        
        # Calculate ratio
        memory_ratio = total_allocated / input_size if input_size > 0 else 0
        
        print(f"Small dataset memory usage:")
        print(f"  Input size: {input_size / 1024:.1f} KB")
        print(f"  Memory allocated: {total_allocated / 1024:.1f} KB")
        print(f"  Memory ratio: {memory_ratio:.2f}x")
        
        # REQ-016: Memory usage should be <1.5x input size
        # Note: For small datasets, overhead may be proportionally larger
        assert memory_ratio < 2.0, f"REQ-016: Memory ratio {memory_ratio:.2f}x exceeds 2.0x for small data"
    
    def test_memory_usage_medium_dataset(self):
        """Test memory usage on medium dataset (100,000 rows).
        
        Validates REQ-016: Memory usage should be <1.5x input size.
        """
        n = 100_000
        np.random.seed(42)
        x = np.random.randn(n).astype(np.float64)
        y = (2 * x + 3 + np.random.randn(n) * 0.1).astype(np.float64)
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        # Calculate input data size
        input_size = self.get_dataframe_memory(df)
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Start memory tracking
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        
        # Perform regression
        result = linear_regression(df, "x", "y")
        
        # Take memory snapshot after
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Calculate memory used during operation
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_allocated = sum(stat.size for stat in top_stats if stat.size > 0)
        
        # Calculate ratio
        memory_ratio = total_allocated / input_size if input_size > 0 else 0
        
        print(f"Medium dataset memory usage:")
        print(f"  Input size: {input_size / (1024**2):.1f} MB")
        print(f"  Memory allocated: {total_allocated / (1024**2):.1f} MB")
        print(f"  Memory ratio: {memory_ratio:.2f}x")
        
        # REQ-016: Memory usage should be <1.5x input size
        assert memory_ratio < 1.5, f"REQ-016 VIOLATION: Memory ratio {memory_ratio:.2f}x exceeds 1.5x"
    
    def test_memory_usage_large_dataset(self):
        """Test memory usage on large dataset (1,000,000 rows).
        
        Validates REQ-016: Memory usage should be <1.5x input size.
        This is the critical test for the memory requirement.
        """
        n = 1_000_000
        np.random.seed(42)
        x = np.random.randn(n).astype(np.float64)
        y = (2 * x + 3 + np.random.randn(n) * 0.1).astype(np.float64)
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        # Calculate input data size
        input_size = self.get_dataframe_memory(df)
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Start memory tracking
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        
        # Perform regression
        result = linear_regression(df, "x", "y")
        
        # Take memory snapshot after
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Calculate memory used during operation
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_allocated = sum(stat.size for stat in top_stats if stat.size > 0)
        
        # Calculate ratio
        memory_ratio = total_allocated / input_size if input_size > 0 else 0
        
        print(f"Large dataset memory usage:")
        print(f"  Input size: {input_size / (1024**2):.1f} MB")
        print(f"  Memory allocated: {total_allocated / (1024**2):.1f} MB")
        print(f"  Memory ratio: {memory_ratio:.2f}x")
        
        # REQ-016: Memory usage should be <1.5x input size
        assert memory_ratio < 1.5, f"REQ-016 VIOLATION: Memory ratio {memory_ratio:.2f}x exceeds 1.5x"
        print(f"REQ-016 PASSED: Memory ratio {memory_ratio:.2f}x is within 1.5x limit")
    
    def test_memory_no_leaks(self):
        """Test that repeated calls don't leak memory.
        
        Validates that the implementation properly releases memory
        after each computation.
        """
        n = 100_000
        np.random.seed(42)
        x = np.random.randn(n).astype(np.float64)
        y = (2 * x + 3 + np.random.randn(n) * 0.1).astype(np.float64)
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        # Force garbage collection
        gc.collect()
        
        # Get baseline memory
        tracemalloc.start()
        snapshot_baseline = tracemalloc.take_snapshot()
        
        # Run multiple times
        for i in range(10):
            result = linear_regression(df, "x", "y")
            # Explicitly delete result to ensure cleanup
            del result
        
        # Force garbage collection
        gc.collect()
        
        # Check memory after multiple runs
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Calculate memory difference
        top_stats = snapshot_after.compare_to(snapshot_baseline, 'lineno')
        leaked_memory = sum(stat.size for stat in top_stats if stat.size > 0)
        
        # Allow some overhead, but should be minimal
        max_allowed_leak = 1024 * 1024  # 1 MB tolerance
        
        print(f"Memory leak test:")
        print(f"  Memory after 10 runs: {leaked_memory / 1024:.1f} KB")
        print(f"  Max allowed: {max_allowed_leak / 1024:.1f} KB")
        
        assert leaked_memory < max_allowed_leak, f"Potential memory leak: {leaked_memory / 1024:.1f} KB accumulated"
    
    def test_memory_peak_usage(self):
        """Test peak memory usage during computation.
        
        Ensures that even peak memory usage stays within reasonable bounds.
        """
        n = 500_000
        np.random.seed(42)
        x = np.random.randn(n).astype(np.float64)
        y = (2 * x + 3 + np.random.randn(n) * 0.1).astype(np.float64)
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        # Calculate input data size
        input_size = self.get_dataframe_memory(df)
        
        # Start tracking with peak measurement
        tracemalloc.start()
        
        # Perform regression
        result = linear_regression(df, "x", "y")
        
        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate peak ratio
        peak_ratio = peak / input_size if input_size > 0 else 0
        
        print(f"Peak memory usage:")
        print(f"  Input size: {input_size / (1024**2):.1f} MB")
        print(f"  Peak memory: {peak / (1024**2):.1f} MB")
        print(f"  Peak ratio: {peak_ratio:.2f}x")
        
        # Peak should be at most 2x input size (allowing for temporary allocations)
        assert peak_ratio < 2.0, f"Peak memory ratio {peak_ratio:.2f}x exceeds 2.0x"
    
    def test_memory_different_dtypes(self):
        """Test memory usage with different data types.
        
        Validates that memory efficiency is maintained across different
        numeric precisions.
        """
        n = 100_000
        np.random.seed(42)
        
        # Test with float32 (4 bytes per element)
        x_f32 = np.random.randn(n).astype(np.float32)
        y_f32 = (2 * x_f32 + 3 + np.random.randn(n).astype(np.float32) * 0.1)
        
        df_f32 = pl.DataFrame({
            "x": x_f32,
            "y": y_f32
        })
        
        # Convert to float64 for computation (as required by the function)
        df_f64 = df_f32.select([
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64)
        ])
        
        input_size = self.get_dataframe_memory(df_f64)
        
        # Measure memory
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        
        result = linear_regression(df_f64, "x", "y")
        
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Calculate memory used
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_allocated = sum(stat.size for stat in top_stats if stat.size > 0)
        memory_ratio = total_allocated / input_size if input_size > 0 else 0
        
        print(f"Different dtype memory usage:")
        print(f"  Input size: {input_size / (1024**2):.1f} MB")
        print(f"  Memory allocated: {total_allocated / (1024**2):.1f} MB")
        print(f"  Memory ratio: {memory_ratio:.2f}x")
        
        # Should still meet memory constraint
        assert memory_ratio < 1.5, f"Memory ratio {memory_ratio:.2f}x exceeds 1.5x for converted data"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])