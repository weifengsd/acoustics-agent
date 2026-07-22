"""Batch runners for Bellhop AT/native parity benchmark rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.benchmark_thresholds import BenchmarkThresholds, evaluate_benchmark_row
from pyacoustics.solvers.bellhop import arrival_parity, ray_parity, tl_parity


BellhopBenchmarkKind = Literal["tl", "arrival", "ray"]


@dataclass(frozen=True)
class BellhopBenchmarkCase:
    """One Bellhop parity benchmark scenario."""

    name: str
    kind: BellhopBenchmarkKind
    config: SimulationConfig
    r_grid: np.ndarray | None = None
    z_grid: np.ndarray | None = None
    run_type: str | None = None
    at_bin_path: str | Path | None = None
    options: dict[str, object] = field(default_factory=dict)
    thresholds: BenchmarkThresholds | None = None


@dataclass(frozen=True)
class BellhopBenchmarkMatrix:
    """Batch Bellhop parity rows grouped by benchmark case."""

    rows: tuple[dict[str, int | float | str | bool | None], ...]


def _receiver_ranges(config: SimulationConfig, r_grid: np.ndarray | None) -> np.ndarray:
    if r_grid is not None:
        return np.asarray(r_grid, dtype=np.float64)
    return np.asarray(config.geometry.receivers.ranges, dtype=np.float64)


def _receiver_depths(config: SimulationConfig, z_grid: np.ndarray | None) -> np.ndarray:
    if z_grid is not None:
        return np.asarray(z_grid, dtype=np.float64)
    return np.asarray(config.geometry.receivers.depths, dtype=np.float64)


def _with_kind(row: dict[str, int | float | str | bool | None], kind: BellhopBenchmarkKind):
    return {"kind": kind, **row}


def _run_tl_case(
    case: BellhopBenchmarkCase,
    *,
    at_bin_path: str | Path | None,
) -> dict[str, int | float | str | bool | None]:
    run_type = str(case.run_type or case.options.get("run_type", "C")).upper()
    r_grid = _receiver_ranges(case.config, case.r_grid)
    z_grid = _receiver_depths(case.config, case.z_grid)
    at_tl = tl_parity.run_at_tl_grid(case.config, r_grid, z_grid, run_type=run_type, at_bin_path=at_bin_path)
    native_tl = tl_parity.compute_native_tl_grid(case.config, r_grid, z_grid, run_type=run_type)
    metrics = tl_parity.compare_tl_grids(at_tl, native_tl)
    return _with_kind(tl_parity.tl_benchmark_row(case.name, metrics, run_type=run_type), "tl")


def _run_arrival_case(
    case: BellhopBenchmarkCase,
    *,
    at_bin_path: str | Path | None,
) -> dict[str, int | float | str | bool | None]:
    delay_tolerance = float(case.options.get("delay_tolerance", 0.02))
    method = str(case.options.get("method", "beam"))
    run_type = str(case.run_type or case.options.get("run_type", "A"))
    at_data = arrival_parity.run_at_arrival_data(case.config, at_bin_path=at_bin_path, run_type=run_type)
    native_data = arrival_parity.compute_native_arrival_data(
        case.config,
        max_arrivals=int(case.options.get("max_arrivals", 200)),
        beam_shape=str(case.options.get("beam_shape", "hat")),
        method=method,
        eigenray_tolerance_m=float(case.options.get("eigenray_tolerance_m", 0.25)),
        eigenray_angle_samples=case.options.get("eigenray_angle_samples"),
    )
    metrics = arrival_parity.compare_arrival_data(at_data, native_data, delay_tolerance=delay_tolerance)
    return _with_kind(
        arrival_parity.arrival_benchmark_row(
            case.name,
            metrics,
            method=method,
            delay_tolerance=delay_tolerance,
        )
        | {"run_type": run_type},
        "arrival",
    )


def _run_ray_case(
    case: BellhopBenchmarkCase,
    *,
    at_bin_path: str | Path | None,
) -> dict[str, int | float | str | bool | None]:
    run_type = str(case.run_type or case.options.get("run_type", "R")).upper()
    at_records = ray_parity.run_at_ray_records(case.config, run_type=run_type, at_bin_path=at_bin_path)
    receiver_ranges = _receiver_ranges(case.config, case.r_grid)
    trace_range = case.options.get("trace_range_m")
    if trace_range is None and receiver_ranges.size:
        trace_range = float(np.max(receiver_ranges)) * 1.1
    if run_type == "E":
        native_records = ray_parity.compute_native_eigenray_records(
            case.config,
            max_arrivals=int(case.options.get("max_arrivals", 200)),
            beam_shape=str(case.options.get("beam_shape", "hat")),
            trace_range_m=float(trace_range) if trace_range is not None else None,
        )
    else:
        native_records = ray_parity.compute_native_ray_records(
            case.config,
            trace_range_m=float(trace_range) if trace_range is not None else None,
        )
    max_range = case.options.get("max_range_m")
    metrics = ray_parity.compare_ray_records(
        at_records,
        native_records,
        max_range_m=float(max_range) if max_range is not None else None,
        sample_count=int(case.options.get("sample_count", 101)),
    )
    return _with_kind(ray_parity.ray_benchmark_row(case.name, metrics, run_type=run_type), "ray")


def run_bellhop_benchmark_matrix(
    cases: list[BellhopBenchmarkCase] | tuple[BellhopBenchmarkCase, ...],
    *,
    at_bin_path: str | Path | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> BellhopBenchmarkMatrix:
    """Run multiple Bellhop parity cases and collect flattened benchmark rows."""

    rows: list[dict[str, int | float | str | bool | None]] = []
    for case in cases:
        case_at_bin_path = at_bin_path if at_bin_path is not None else case.at_bin_path
        if case.kind == "tl":
            row = _run_tl_case(case, at_bin_path=case_at_bin_path)
        elif case.kind == "arrival":
            row = _run_arrival_case(case, at_bin_path=case_at_bin_path)
        elif case.kind == "ray":
            row = _run_ray_case(case, at_bin_path=case_at_bin_path)
        else:
            raise ValueError("Bellhop benchmark kind must be one of 'tl', 'arrival', or 'ray'")
        effective_thresholds = case.thresholds or thresholds
        if effective_thresholds is not None:
            row = evaluate_benchmark_row(row, effective_thresholds)
        rows.append(row)
    matrix_rows = tuple(rows)
    return BellhopBenchmarkMatrix(rows=matrix_rows)


__all__ = [
    "BenchmarkThresholds",
    "BellhopBenchmarkCase",
    "BellhopBenchmarkKind",
    "BellhopBenchmarkMatrix",
    "run_bellhop_benchmark_matrix",
]
