causers Documentation
=====================

High-performance statistical operations for Polars DataFrames, powered by Rust.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/causers
   benchmarks

Quick Start
-----------

Install causers from PyPI::

    pip install causers

Basic usage::

    import polars as pl
    import causers

    df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    result = causers.linear_regression(df, "x", "y")
    print(f"y = {result.slope:.2f}x + {result.intercept:.2f}")

Features
--------

* Linear regression with HC3 robust standard errors
* Logistic regression with Newton-Raphson MLE
* Cluster-robust standard errors (analytical and bootstrap)
* Fixed effects (within-transformation for OLS, Mundlak for logistic)
* Synthetic Difference-in-Differences (SDID)
* Synthetic Control (SC) with 4 method variants
* Double Machine Learning (DML) with cross-fitting
* Two-Stage Least Squares (IV/2SLS) with weak instrument diagnostics
* Covariate balance diagnostics (SMD, variance ratios, weighted ESS)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
