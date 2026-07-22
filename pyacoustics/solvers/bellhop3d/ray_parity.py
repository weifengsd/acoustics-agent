"""Utilities for Bellhop3D AT/native ray path parity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop3d.core import PyBellhop3D
from pyacoustics.solvers.benchmark_thresholds import BenchmarkThresholds, evaluate_benchmark_rows
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.external_io import ATRayRecord, generate_at_bellhop3d_env, read_at_ray_records


@dataclass(frozen=True)
class Bellhop3DRayParityMetrics:
    """Summary metrics for AT/native Bellhop3D ray path comparison."""

    at_ray_count: int
    native_ray_count: int
    matched_ray_count: int
    unmatched_at_ray_count: int
    sample_count: int
    depth_mae_m: float
    depth_p90_abs_m: float
    depth_max_abs_m: float
    y_mae_m: float
    y_p90_abs_m: float
    y_max_abs_m: float
    launch_angle_mae_deg: float
    launch_azimuth_mae_deg: float
    bounce_match_count: int
    bounce_match_fraction: float
    top_bounce_match_fraction: float
    bottom_bounce_match_fraction: float


@dataclass(frozen=True)
class Bellhop3DRayBenchmarkCase:
    """One Bellhop3D AT/native ray path parity benchmark scenario."""

    name: str
    config: SimulationConfig
    run_type: str = "R"
    max_range_m: float | None = None
    at_bin_path: str | Path | None = None
    thresholds: BenchmarkThresholds | None = None


@dataclass(frozen=True)
class Bellhop3DRayBenchmarkMatrix:
    """Batch Bellhop3D ray parity rows grouped by benchmark case."""

    rows: tuple[dict[str, int | float | str | bool | None], ...]


def _angle_grid(start_end: tuple[float, float] | list[float], count: int) -> np.ndarray:
    return np.linspace(float(start_end[0]), float(start_end[1]), max(int(count), 1))


NativeRayCandidate = tuple[float, float, float, tuple]


def _native_ray_lookup(config: SimulationConfig) -> dict[tuple[float, float], list[NativeRayCandidate]]:
    solver = PyBellhop3D(config)
    rays = solver.run_ray_tracing(include_phi=False)
    elevations = _angle_grid(config.solver.elevation_angles, config.solver.num_elevation_beams)
    azimuths = _angle_grid(config.solver.azimuth_angles, config.solver.num_azimuth_beams)
    source_depths = np.asarray(config.geometry.source.depths, dtype=np.float64)
    source_x = np.asarray(config.geometry.source.x or [0.0], dtype=np.float64)
    source_y = np.asarray(config.geometry.source.y or [0.0], dtype=np.float64)

    lookup: dict[tuple[float, float], list[NativeRayCandidate]] = {}
    num_elev = max(config.solver.num_elevation_beams, 1)
    num_az = max(config.solver.num_azimuth_beams, 1)
    rays_per_source = num_elev * num_az
    for ray_idx, ray in enumerate(rays):
        source_idx = ray_idx // rays_per_source
        source_depth_idx = source_idx // max(source_x.size * source_y.size, 1)
        source_xy_idx = source_idx % max(source_x.size * source_y.size, 1)
        source_x_idx = source_xy_idx // max(source_y.size, 1)
        source_y_idx = source_xy_idx % max(source_y.size, 1)
        azimuth_idx = (ray_idx // num_elev) % num_az
        elevation_idx = ray_idx % num_elev
        key = (round(float(elevations[elevation_idx]), 6), round(float(azimuths[azimuth_idx]), 6))
        candidate = (
            float(source_x[source_x_idx]),
            float(source_y[source_y_idx]),
            float(source_depths[source_depth_idx]),
            ray,
        )
        lookup.setdefault(key, []).append(candidate)
    return lookup


def _candidate_list(value) -> list[NativeRayCandidate]:
    if isinstance(value, list):
        return value
    ray = value
    x = np.asarray(ray[0], dtype=np.float64)
    y = np.asarray(ray[1], dtype=np.float64)
    z = np.asarray(ray[2], dtype=np.float64)
    return [(float(x[0]), float(y[0]), float(z[0]), ray)]


def _select_native_candidate(
    candidates: list[NativeRayCandidate],
    x_at: np.ndarray,
    y_at: np.ndarray,
    z_at: np.ndarray,
) -> NativeRayCandidate:
    if len(candidates) == 1:
        return candidates[0]
    start_x = float(x_at[0])
    start_y = float(y_at[0])
    start_z = float(z_at[0])
    return min(
        candidates,
        key=lambda item: abs(item[0] - start_x) + abs(item[1] - start_y) + abs(item[2] - start_z),
    )


def _horizontal_range_from_source(x_m: np.ndarray, y_m: np.ndarray, source_x_m: float, source_y_m: float) -> np.ndarray:
    return np.hypot(x_m - source_x_m, y_m - source_y_m)


def _sorted_unique_samples(
    ranges_m: np.ndarray,
    *values: np.ndarray,
) -> tuple[np.ndarray, ...]:
    order = np.argsort(ranges_m, kind="mergesort")
    sorted_ranges = ranges_m[order]
    unique_mask = np.concatenate(([True], np.diff(sorted_ranges) > 1.0e-9))
    outputs: list[np.ndarray] = [sorted_ranges[unique_mask]]
    outputs.extend(np.asarray(value, dtype=np.float64)[order][unique_mask] for value in values)
    return tuple(outputs)


def run_at_ray_records_3d(
    config: SimulationConfig,
    *,
    at_bin_path: str | Path | None = None,
    run_type: str = "R",
) -> list[ATRayRecord]:
    """Run AT Bellhop3D ray tracing and parse its ``.ray`` records."""

    external = ExternalSolver(config, bin_path=str(at_bin_path) if at_bin_path else None)
    if not external.is_available("bellhop3d"):
        raise RuntimeError(f"Legacy bellhop3d executable not found in {external.bin_path}.")
    bellhop3d_exe = external.bin_path / "bellhop3d"
    if not bellhop3d_exe.exists():
        bellhop3d_exe = external.bin_path / "bellhop3d.exe"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_name = "bellhop3d_ray_parity"
        generate_at_bellhop3d_env(config, tmp_path / f"{case_name}.env", run_type=run_type)
        external.run_command([str(bellhop3d_exe), case_name], cwd=str(tmp_path))
        ray_path = tmp_path / f"{case_name}.ray"
        if not ray_path.exists():
            raise RuntimeError(f"AT Bellhop3D did not generate {ray_path.name}.")
        return read_at_ray_records(ray_path)


def compare_ray_records_3d(
    at_records: list[ATRayRecord],
    config: SimulationConfig,
    *,
    max_range_m: float | None = None,
) -> Bellhop3DRayParityMetrics:
    """Compare parsed AT Bellhop3D ray records against native traced rays."""

    native_lookup = _native_ray_lookup(config)
    depth_errors: list[float] = []
    y_errors: list[float] = []
    launch_errors: list[float] = []
    azimuth_errors: list[float] = []
    matched = 0
    top_matches = 0
    bottom_matches = 0
    bounce_matches = 0

    for record in at_records:
        at_azimuth = 0.0 if record.launch_azimuth_deg is None else float(record.launch_azimuth_deg)
        key = (round(float(record.launch_angle_deg), 6), round(at_azimuth, 6))
        native_candidates = native_lookup.get(key)
        if native_candidates is None:
            continue
        x_at = np.asarray(record.ranges_m, dtype=np.float64)
        y_at = np.asarray(record.y_m if record.y_m is not None else np.zeros_like(record.ranges_m), dtype=np.float64)
        z_at = np.asarray(record.depths_m, dtype=np.float64)
        source_x_m, source_y_m, _source_depth_m, native_ray = _select_native_candidate(
            _candidate_list(native_candidates),
            x_at,
            y_at,
            z_at,
        )
        x_native = np.asarray(native_ray[0], dtype=np.float64)
        y_native = np.asarray(native_ray[1], dtype=np.float64)
        z_native = np.asarray(native_ray[2], dtype=np.float64)
        if x_native.size < 2:
            continue
        r_native = _horizontal_range_from_source(x_native, y_native, source_x_m, source_y_m)
        r_at = _horizontal_range_from_source(x_at, y_at, source_x_m, source_y_m)
        r_native, y_native, z_native = _sorted_unique_samples(r_native, y_native, z_native)
        if r_native.size < 2:
            continue
        mask = (r_at >= float(r_native[0])) & (r_at <= float(r_native[-1]))
        if max_range_m is not None:
            mask &= r_at <= float(max_range_m)
        if not np.any(mask):
            continue

        z_interp = np.interp(r_at[mask], r_native, z_native)
        y_interp = np.interp(r_at[mask], r_native, y_native)
        depth_errors.extend(np.abs(z_interp - z_at[mask]).tolist())
        y_errors.extend(np.abs(y_interp - y_at[mask]).tolist())
        launch_errors.append(abs(float(record.launch_angle_deg) - key[0]))
        azimuth_errors.append(abs(at_azimuth - key[1]))

        matched += 1
        native_top = int(native_ray[-3])
        native_bottom = int(native_ray[-2])
        top_match = native_top == int(record.num_top_bounces)
        bottom_match = native_bottom == int(record.num_bottom_bounces)
        top_matches += int(top_match)
        bottom_matches += int(bottom_match)
        bounce_matches += int(top_match and bottom_match)

    sample_count = len(depth_errors)
    depth_array = np.asarray(depth_errors, dtype=np.float64)
    y_array = np.asarray(y_errors, dtype=np.float64)
    if sample_count == 0:
        depth_mae = depth_p90 = depth_max = float("nan")
        y_mae = y_p90 = y_max = float("nan")
    else:
        depth_mae = float(np.mean(depth_array))
        depth_p90 = float(np.percentile(depth_array, 90.0))
        depth_max = float(np.max(depth_array))
        y_mae = float(np.mean(y_array))
        y_p90 = float(np.percentile(y_array, 90.0))
        y_max = float(np.max(y_array))

    return Bellhop3DRayParityMetrics(
        at_ray_count=len(at_records),
        native_ray_count=len(native_lookup),
        matched_ray_count=matched,
        unmatched_at_ray_count=max(len(at_records) - matched, 0),
        sample_count=sample_count,
        depth_mae_m=depth_mae,
        depth_p90_abs_m=depth_p90,
        depth_max_abs_m=depth_max,
        y_mae_m=y_mae,
        y_p90_abs_m=y_p90,
        y_max_abs_m=y_max,
        launch_angle_mae_deg=float(np.mean(launch_errors)) if launch_errors else float("nan"),
        launch_azimuth_mae_deg=float(np.mean(azimuth_errors)) if azimuth_errors else float("nan"),
        bounce_match_count=bounce_matches,
        bounce_match_fraction=bounce_matches / matched if matched else float("nan"),
        top_bounce_match_fraction=top_matches / matched if matched else float("nan"),
        bottom_bounce_match_fraction=bottom_matches / matched if matched else float("nan"),
    )


def bellhop3d_ray_benchmark_row(
    case_name: str,
    metrics: Bellhop3DRayParityMetrics,
    *,
    run_type: str = "R",
) -> dict[str, int | float | str | None]:
    """Flatten Bellhop3D ray parity metrics into one report/CSV-friendly row."""

    return {
        "case": case_name,
        "kind": "bellhop3d_ray",
        "run_type": run_type,
        "at_ray_count": metrics.at_ray_count,
        "native_ray_count": metrics.native_ray_count,
        "matched_ray_count": metrics.matched_ray_count,
        "unmatched_at_ray_count": metrics.unmatched_at_ray_count,
        "sample_count": metrics.sample_count,
        "depth_mae_m": metrics.depth_mae_m,
        "depth_p90_abs_m": metrics.depth_p90_abs_m,
        "depth_max_abs_m": metrics.depth_max_abs_m,
        "y_mae_m": metrics.y_mae_m,
        "y_p90_abs_m": metrics.y_p90_abs_m,
        "y_max_abs_m": metrics.y_max_abs_m,
        "launch_angle_mae_deg": metrics.launch_angle_mae_deg,
        "launch_azimuth_mae_deg": metrics.launch_azimuth_mae_deg,
        "bounce_match_count": metrics.bounce_match_count,
        "bounce_match_fraction": metrics.bounce_match_fraction,
        "top_bounce_match_fraction": metrics.top_bounce_match_fraction,
        "bottom_bounce_match_fraction": metrics.bottom_bounce_match_fraction,
    }


def run_bellhop3d_ray_parity(
    config: SimulationConfig,
    *,
    at_bin_path: str | Path | None = None,
    run_type: str = "R",
    max_range_m: float | None = None,
) -> Bellhop3DRayParityMetrics:
    at_records = run_at_ray_records_3d(config, at_bin_path=at_bin_path, run_type=run_type)
    return compare_ray_records_3d(at_records, config, max_range_m=max_range_m)


def run_bellhop3d_ray_benchmark_matrix(
    cases: list[Bellhop3DRayBenchmarkCase] | tuple[Bellhop3DRayBenchmarkCase, ...],
    *,
    at_bin_path: str | Path | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> Bellhop3DRayBenchmarkMatrix:
    """Run multiple Bellhop3D ray parity benchmark cases."""

    rows: list[dict[str, int | float | str | bool | None]] = []
    for case in cases:
        case_at_bin_path = at_bin_path if at_bin_path is not None else case.at_bin_path
        metrics = run_bellhop3d_ray_parity(
            case.config,
            at_bin_path=case_at_bin_path,
            run_type=case.run_type,
            max_range_m=case.max_range_m,
        )
        rows.append(bellhop3d_ray_benchmark_row(case.name, metrics, run_type=case.run_type))

    if thresholds is None and all(case.thresholds is None for case in cases):
        matrix_rows = tuple(rows)
    elif all(case.thresholds is None for case in cases):
        matrix_rows = evaluate_benchmark_rows(rows, thresholds)
    else:
        matrix_rows = tuple(
            evaluate_benchmark_rows([row], case.thresholds or thresholds)[0]
            if (case.thresholds or thresholds) is not None
            else row
            for row, case in zip(rows, cases)
        )
    return Bellhop3DRayBenchmarkMatrix(rows=matrix_rows)


__all__ = [
    "Bellhop3DRayBenchmarkCase",
    "Bellhop3DRayBenchmarkMatrix",
    "Bellhop3DRayParityMetrics",
    "bellhop3d_ray_benchmark_row",
    "compare_ray_records_3d",
    "run_at_ray_records_3d",
    "run_bellhop3d_ray_benchmark_matrix",
    "run_bellhop3d_ray_parity",
]
