"""
PyCFX Benchmarker: counterfactual benchmarker, metrics, factories for generators and models

Benchmarking overview:
- `CFBenchmarker` allows you to specify a dataset, the scale of benchmark (number of test points), metrics to use, models to use and generators to benchmark. See usage in `results/`
- Produces results DataFrames for programmatic use of results, stores results in JSON and LATEX tables, and can produce figures.
- Available metrics: Validity, Failures, Implausibility, Plausibility (LOF), Distance, Sensitivity, Stability.
"""

from pycfx.benchmarker.counterfactual_benchmarker import CFBenchmarker
from pycfx.benchmarker.factories import ModelFactory, GeneratorFactory
from pycfx.benchmarker.metrics import *
