Benchmarks
==========

These notebooks demonstrate that ``causers`` produces results equivalent to 
established reference implementations while achieving significant speedup.

Overview
--------

Each benchmark notebook compares a ``causers`` function against its reference implementation:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Function
     - Reference Package
     - Coefficient Tolerance
     - SE Tolerance
   * - ``linear_regression``
     - statsmodels OLS
     - rtol=1e-6
     - rtol=1e-6 (HC3)
   * - ``logistic_regression``
     - statsmodels Logit
     - rtol=1e-6
     - rtol=0.1 (HC3)
   * - ``synthetic_control``
     - pysyncon
     - rtol=1e-6
     - rtol=1e-2
   * - ``synthetic_did``
     - azcausal SDID
     - rtol=1e-6
     - rtol=0.5 (bootstrap)

Methodology
-----------

**Parity Tests**:

- Compare numerical outputs within documented tolerances
- Use ``numpy.isclose()`` with specified relative tolerance (rtol)
- Report PASS/FAIL status for each comparison

**Timing Benchmarks**:

- Warm-up: 1 iteration (discarded)
- Measurement: 5 iterations
- Reported: Median execution time
- Speedup: reference_time / causers_time

Running Locally
---------------

Install reference packages (optional, for parity tests)::

    pip install statsmodels pysyncon azcausal

Run notebooks::

    jupyter notebook examples/benchmarks/

Benchmark Notebooks
-------------------

.. toctree::
   :maxdepth: 1

   ../../examples/benchmarks/linear_regression
   ../../examples/benchmarks/logistic_regression
   ../../examples/benchmarks/synthetic_control
   ../../examples/benchmarks/synthetic_did
