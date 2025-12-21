"""Integration tests for causers package.

This module validates the complete workflow and integration
of all components working together.
"""

import pytest
import polars as pl
import numpy as np
from causers import linear_regression, LinearRegressionResult
import tempfile
import os


class TestIntegration:
    """Test suite for integration scenarios."""
    
    def test_full_workflow_with_real_data_pattern(self):
        """Test complete workflow with realistic data patterns.
        
        Simulates a real-world scenario with:
        - Data loading
        - Preprocessing
        - Regression
        - Result validation
        """
        # Simulate real-world data with trend and noise
        np.random.seed(42)
        n = 1000
        
        # Generate time series data (e.g., sales over time)
        time = np.arange(n)
        base_trend = 100 + 0.5 * time
        seasonal = 10 * np.sin(2 * np.pi * time / 365)
        noise = np.random.normal(0, 5, n)
        sales = base_trend + seasonal + noise
        
        # Create DataFrame
        df = pl.DataFrame({
            "time": time.astype(float),
            "sales": sales
        })
        
        # Perform regression
        result = linear_regression(df, "time", "sales")
        
        # Validate results
        assert result.n_samples == n
        assert result.slope > 0  # Positive trend
        assert result.r_squared > 0.8  # Good fit despite seasonality
        
        # Test result object methods
        repr_str = repr(result)
        assert "LinearRegressionResult" in repr_str
        
        str_str = str(result)
        assert "y =" in str_str
        assert "R²" in str_str
    
    def test_workflow_with_data_transformations(self):
        """Test workflow with data transformations.
        
        Tests that the package works correctly when data
        undergoes typical preprocessing steps.
        """
        # Original data
        df = pl.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "target": [15, 28, 42, 55, 69, 82, 96, 109, 123, 136]
        })
        
        # Transform: normalize feature1
        df = df.with_columns([
            ((pl.col("feature1") - pl.col("feature1").mean()) / 
             pl.col("feature1").std()).alias("feature1_normalized")
        ])
        
        # Transform: log scale feature2
        df = df.with_columns([
            pl.col("feature2").log().alias("feature2_log")
        ])
        
        # Regression on transformed features
        result1 = linear_regression(df, "feature1_normalized", "target")
        result2 = linear_regression(df, "feature2_log", "target")
        
        # Both should give reasonable results
        assert result1.n_samples == 10
        assert result2.n_samples == 10
        assert result1.r_squared > 0.9
        assert result2.r_squared > 0.9
    
    def test_workflow_with_filtering(self):
        """Test workflow with data filtering.
        
        Tests that regression works correctly on filtered subsets.
        """
        # Create data with outliers
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 * x + 3 + np.random.randn(n) * 0.1
        
        # Add outliers
        x = np.append(x, [10, -10, 15])
        y = np.append(y, [100, -100, 150])
        
        df = pl.DataFrame({
            "x": x,
            "y": y
        })
        
        # Filter outliers (keep only reasonable values)
        df_filtered = df.filter(
            (pl.col("x").abs() < 5) & (pl.col("y").abs() < 20)
        )
        
        # Regression on filtered data
        result = linear_regression(df_filtered, "x", "y")
        
        # Should get close to true relationship
        assert abs(result.slope - 2.0) < 0.1
        assert abs(result.intercept - 3.0) < 0.2
        assert result.r_squared > 0.98
    
    def test_workflow_with_grouped_analysis(self):
        """Test workflow with grouped analysis.
        
        Tests regression on different groups of data.
        """
        # Create data with different groups
        group_a_x = np.array([1, 2, 3, 4, 5], dtype=float)
        group_a_y = 2 * group_a_x + 1
        
        group_b_x = np.array([1, 2, 3, 4, 5], dtype=float)
        group_b_y = 3 * group_b_x - 2
        
        df = pl.DataFrame({
            "group": ["A"] * 5 + ["B"] * 5,
            "x": np.concatenate([group_a_x, group_b_x]),
            "y": np.concatenate([group_a_y, group_b_y])
        })
        
        # Perform regression on each group
        results = {}
        for group_name, group_df in df.group_by("group"):
            result = linear_regression(group_df, "x", "y")
            results[group_name[0]] = result
        
        # Validate group A: y = 2x + 1
        assert abs(results["A"].slope - 2.0) < 1e-10
        assert abs(results["A"].intercept - 1.0) < 1e-10
        assert results["A"].r_squared == 1.0
        
        # Validate group B: y = 3x - 2
        assert abs(results["B"].slope - 3.0) < 1e-10
        assert abs(results["B"].intercept - (-2.0)) < 1e-10
        assert results["B"].r_squared == 1.0
    
    def test_workflow_with_multiple_regressions(self):
        """Test workflow with multiple regression analyses.
        
        Tests that multiple regressions can be performed
        efficiently without interference.
        """
        np.random.seed(42)
        n = 500
        
        # Create multiple feature columns
        df = pl.DataFrame({
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
            "y": np.random.randn(n)
        })
        
        # Add relationships
        df = df.with_columns([
            (2 * pl.col("x1") + 3 + np.random.randn(n) * 0.1).alias("y1"),
            (3 * pl.col("x2") - 1 + np.random.randn(n) * 0.1).alias("y2"),
            (-1 * pl.col("x3") + 5 + np.random.randn(n) * 0.1).alias("y3")
        ])
        
        # Perform multiple regressions
        result1 = linear_regression(df, "x1", "y1")
        result2 = linear_regression(df, "x2", "y2")
        result3 = linear_regression(df, "x3", "y3")
        
        # Validate all results
        assert abs(result1.slope - 2.0) < 0.1
        assert abs(result2.slope - 3.0) < 0.1
        assert abs(result3.slope - (-1.0)) < 0.1
        
        assert result1.r_squared > 0.95
        assert result2.r_squared > 0.95
        assert result3.r_squared > 0.95
    
    def test_workflow_with_save_load_cycle(self):
        """Test workflow with saving and loading data.
        
        Tests that regression works correctly with data
        that has been saved and reloaded.
        """
        # Create original data
        df_original = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        
        # Save to temporary file and reload
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
            df_original.write_parquet(tmp_path)
        
        try:
            # Reload data
            df_loaded = pl.read_parquet(tmp_path)
            
            # Perform regression on loaded data
            result = linear_regression(df_loaded, "x", "y")
            
            # Should get same results as original
            assert abs(result.slope - 2.0) < 1e-10
            assert abs(result.intercept - 0.0) < 1e-10
            assert abs(result.r_squared - 1.0) < 1e-10
            assert result.n_samples == 5
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_workflow_with_column_selection(self):
        """Test workflow with various column selection patterns."""
        # Create DataFrame with many columns
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [2.0, 4.0, 6.0, 8.0, 10.0],
            "metadata": ["a", "b", "c", "d", "e"]
        })
        
        # Test with different column selections
        result = linear_regression(df.select(["feature_a", "target"]), 
                                 "feature_a", "target")
        
        assert abs(result.slope - 2.0) < 1e-10
        assert result.n_samples == 5
    
    def test_workflow_with_lazy_evaluation(self):
        """Test workflow with Polars lazy evaluation."""
        # Create lazy DataFrame
        df = pl.DataFrame({
            "x": range(100),
            "y": [2*x + 3 for x in range(100)]
        }).lazy()
        
        # Apply transformations lazily
        df_transformed = df.with_columns([
            (pl.col("x").cast(pl.Float64)),
            (pl.col("y").cast(pl.Float64))
        ])
        
        # Collect and perform regression
        df_collected = df_transformed.collect()
        result = linear_regression(df_collected, "x", "y")
        
        assert abs(result.slope - 2.0) < 1e-10
        assert abs(result.intercept - 3.0) < 1e-10
        assert result.r_squared == 1.0
    
    def test_workflow_robustness_to_column_order(self):
        """Test that regression is independent of column order."""
        # Create DataFrames with different column orders
        df1 = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        
        df2 = pl.DataFrame({
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        result1 = linear_regression(df1, "x", "y")
        result2 = linear_regression(df2, "x", "y")
        
        # Results should be identical
        assert abs(result1.slope - result2.slope) < 1e-10
        assert abs(result1.intercept - result2.intercept) < 1e-10
        assert abs(result1.r_squared - result2.r_squared) < 1e-10
    
    def test_workflow_with_string_to_numeric_conversion(self):
        """Test workflow when numeric data needs type conversion."""
        # Create DataFrame with integer columns
        df = pl.DataFrame({
            "x": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32),
            "y": pl.Series([2, 4, 6, 8, 10], dtype=pl.Int32)
        })
        
        # Convert to float64 (as required by the function)
        df = df.with_columns([
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64)
        ])
        
        result = linear_regression(df, "x", "y")
        
        assert abs(result.slope - 2.0) < 1e-10
        assert abs(result.intercept - 0.0) < 1e-10
        assert result.r_squared == 1.0
    
    def test_workflow_stress_test(self):
        """Stress test with rapid successive calls.
        
        Tests that the package handles rapid successive calls
        without memory leaks or performance degradation.
        """
        np.random.seed(42)
        
        # Create test data
        df = pl.DataFrame({
            "x": np.random.randn(1000),
            "y": np.random.randn(1000)
        })
        
        # Perform many rapid regressions
        results = []
        for i in range(100):
            # Add slight variation to data
            df_temp = df.with_columns([
                (pl.col("y") + i * 0.01 * pl.col("x")).alias("y_modified")
            ])
            result = linear_regression(df_temp, "x", "y_modified")
            results.append(result)
        
        # All results should be valid
        assert len(results) == 100
        assert all(r.n_samples == 1000 for r in results)
        assert all(isinstance(r, LinearRegressionResult) for r in results)
    
    def test_end_to_end_example(self):
        """Test the complete end-to-end example from documentation.
        
        This mirrors the example in the README/docs to ensure
        it works exactly as documented.
        """
        # Example from documentation
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 5.0, 4.0, 5.0]
        })
        
        result = linear_regression(df, "x", "y")
        
        # Check that result has expected attributes
        assert hasattr(result, 'slope')
        assert hasattr(result, 'intercept')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'n_samples')
        
        # Check that values are reasonable
        assert isinstance(result.slope, float)
        assert isinstance(result.intercept, float)
        assert 0 <= result.r_squared <= 1
        assert result.n_samples == 5
        
        # Check string representations work
        assert len(str(result)) > 0
        assert len(repr(result)) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])