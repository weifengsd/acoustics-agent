"""Reusable AT/native benchmark case presets."""

from pyacoustics.benchmarks.at_presets import (
    bellhop3d_smoke_thresholds,
    bellhop_smoke_thresholds,
    canonical_bellhop3d_arrival_benchmark_cases,
    canonical_bellhop3d_benchmark_cases,
    canonical_bellhop3d_ray_benchmark_cases,
    canonical_bellhop_benchmark_cases,
    canonical_kraken_benchmark_cases,
    kraken_smoke_thresholds,
)
from pyacoustics.benchmarks.file_formats import (
    at_file_format_smoke_thresholds,
    run_at_file_format_benchmark_matrix,
)
from pyacoustics.benchmarks.field3d_runtime import (
    canonical_field3d_runtime_benchmark_cases,
    field3d_runtime_smoke_thresholds,
    run_field3d_runtime_benchmark_matrix,
)
from pyacoustics.benchmarks.runner import BenchmarkMatrixRun, run_canonical_at_benchmark_matrix
from pyacoustics.postprocessing.benchmark_matrix import (
    postprocessing_smoke_thresholds,
    run_postprocessing_benchmark_matrix,
)

__all__ = [
    "BenchmarkMatrixRun",
    "at_file_format_smoke_thresholds",
    "bellhop3d_smoke_thresholds",
    "bellhop_smoke_thresholds",
    "canonical_bellhop3d_arrival_benchmark_cases",
    "canonical_bellhop3d_benchmark_cases",
    "canonical_bellhop3d_ray_benchmark_cases",
    "canonical_bellhop_benchmark_cases",
    "canonical_field3d_runtime_benchmark_cases",
    "canonical_kraken_benchmark_cases",
    "kraken_smoke_thresholds",
    "field3d_runtime_smoke_thresholds",
    "postprocessing_smoke_thresholds",
    "run_at_file_format_benchmark_matrix",
    "run_canonical_at_benchmark_matrix",
    "run_field3d_runtime_benchmark_matrix",
    "run_postprocessing_benchmark_matrix",
]
