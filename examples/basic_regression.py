#!/usr/bin/env python3
"""
Basic example of using causers for linear regression with Polars DataFrames.
"""

import polars as pl
import numpy as np

# This will work after running: maturin develop
import causers


def main():
    """Run basic regression examples."""
    
    # Example 1: Perfect linear relationship
    print("Example 1: Perfect Linear Relationship")
    print("-" * 40)
    
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0]
    })
    
    print("Data:")
    print(df)
    
    result = causers.linear_regression(df, "x", "y")
    print(f"\nRegression Result: {result}")
    print(f"Equation: y = {result.slope:.2f}x + {result.intercept:.2f}")
    print(f"R-squared: {result.r_squared:.4f}")
    
    # Example 2: Noisy data
    print("\n\nExample 2: Noisy Linear Data")
    print("-" * 40)
    
    np.random.seed(42)
    n_points = 50
    x = np.linspace(0, 10, n_points)
    # y = 3x + 5 + noise
    y = 3 * x + 5 + np.random.normal(0, 2, n_points)
    
    df_noisy = pl.DataFrame({
        "x": x,
        "y": y
    })
    
    print(f"Generated {n_points} noisy data points")
    print(f"True relationship: y = 3x + 5 (plus noise)")
    
    result_noisy = causers.linear_regression(df_noisy, "x", "y")
    print(f"\nRegression Result: {result_noisy}")
    print(f"Estimated equation: y = {result_noisy.slope:.2f}x + {result_noisy.intercept:.2f}")
    print(f"R-squared: {result_noisy.r_squared:.4f}")
    
    # Example 3: Real-world style data
    print("\n\nExample 3: Housing Price vs Size")
    print("-" * 40)
    
    # Simulating house size (sq ft) vs price relationship
    sizes = np.array([750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500])
    # Roughly $200 per sq ft + $50,000 base
    prices = sizes * 200 + 50000 + np.random.normal(0, 10000, len(sizes))
    
    df_housing = pl.DataFrame({
        "size_sqft": sizes,
        "price_usd": prices
    })
    
    print("Housing Data Sample:")
    print(df_housing.head())
    
    result_housing = causers.linear_regression(df_housing, "size_sqft", "price_usd")
    print(f"\nRegression Result:")
    print(f"Price = ${result_housing.slope:.2f} per sq ft + ${result_housing.intercept:,.2f}")
    print(f"R-squared: {result_housing.r_squared:.4f}")
    print(f"Based on {result_housing.n_samples} samples")
    
    # Example 4: Multiple Covariate Regression
    print("\n\nExample 4: Multiple Covariate Regression")
    print("-" * 40)
    
    # Simulating price based on size and age
    # Price = 200*size - 5000*age + 50000
    sizes = np.array([750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400])
    ages = np.array([5, 10, 3, 15, 7, 2, 20, 8, 12, 4])
    prices = 200 * sizes - 5000 * ages + 50000 + np.random.normal(0, 5000, len(sizes))
    
    df_multi = pl.DataFrame({
        "size_sqft": sizes,
        "age_years": ages,
        "price_usd": prices
    })
    
    print("Data Sample:")
    print(df_multi.head())
    
    result_multi = causers.linear_regression(df_multi, ["size_sqft", "age_years"], "price_usd")
    print(f"\nRegression Result:")
    print(f"Coefficients: {result_multi.coefficients}")
    print(f"Price = ${result_multi.coefficients[0]:.2f} * size + ${result_multi.coefficients[1]:.2f} * age + ${result_multi.intercept:,.2f}")
    print(f"R-squared: {result_multi.r_squared:.4f}")
    print(f"Based on {result_multi.n_samples} samples")
    
    # Example 5: Regression Without Intercept
    print("\n\nExample 5: Regression Without Intercept")
    print("-" * 40)
    
    df_no_intercept = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.5, 5.0, 7.5, 10.0, 12.5]  # y = 2.5x (no intercept)
    })
    
    print("Data:")
    print(df_no_intercept)
    
    result_no_int = causers.linear_regression(df_no_intercept, "x", "y", include_intercept=False)
    print(f"\nRegression Result (forced through origin):")
    print(f"y = {result_no_int.coefficients[0]:.2f}x")
    print(f"Intercept: {result_no_int.intercept}")
    print(f"R-squared: {result_no_int.r_squared:.4f}")


if __name__ == "__main__":
    main()