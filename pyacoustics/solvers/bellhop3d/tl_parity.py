"""Utilities for Bellhop3D AT/native receiver-grid TL parity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.benchmark_thresholds import BenchmarkThresholds, evaluate_benchmark_rows
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.external_io import generate_at_bellhop3d_env, read_at_shd_full


@dataclass(frozen=True)
class Bellhop3DTLParityMetrics:
    """Summary metrics for native Bellhop3D vs AT Bellhop3D TL grids."""

    sample_count: int
    at_active_count: int
    native_active_count: int
    at_only_active_count: int
    native_only_active_count: int
    native_weak_tail_count: int
    active_overlap_fraction: float
    mae_db: float
    median_abs_db: float
    p90_abs_db: float
    max_abs_db: float
    bias_db: float
    rmse_db: float


@dataclass(frozen=True)
class Bellhop3DBenchmarkCase:
    """One Bellhop3D receiver-grid TL parity benchmark scenario."""

    name: str
    config: SimulationConfig
    theta_grid_deg: np.ndarray | None = None
    r_grid: np.ndarray | None = None
    z_grid: np.ndarray | None = None
    run_type: str = "C"
    min_range_m: float | None = None
    active_tl_max_db: float = 190.0
    at_bin_path: str | Path | None = None
    thresholds: BenchmarkThresholds | None = None


@dataclass(frozen=True)
class Bellhop3DBenchmarkMatrix:
    """Batch Bellhop3D parity rows grouped by benchmark case."""

    rows: tuple[dict[str, int | float | str | bool | None], ...]


def _with_receiver_grid(
    config: SimulationConfig,
    theta_grid_deg: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> SimulationConfig:
    updated = deepcopy(config)
    updated.geometry.receivers.ranges = np.asarray(r_grid, dtype=np.float64).tolist()
    updated.geometry.receivers.depths = np.asarray(z_grid, dtype=np.float64).tolist()
    updated.solver.receiver_bearings = np.asarray(theta_grid_deg, dtype=np.float64).tolist()
    return updated


def _pressure_to_tl(pressure: np.ndarray) -> np.ndarray:
    pressure_abs = np.abs(np.asarray(pressure, dtype=np.complex128))
    tl = -20.0 * np.log10(np.maximum(pressure_abs, 1.0e-20))
    tl[pressure_abs < 1.0e-20] = 200.0
    return np.asarray(tl, dtype=np.float64)


def run_at_tl_grid_3d(
    config: SimulationConfig,
    theta_grid_deg: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    at_bin_path: str | Path | None = None,
    run_type: str = "C",
) -> np.ndarray:
    """Run AT Bellhop3D and return TL shaped like native receiver-grid output."""

    at_config = _with_receiver_grid(config, theta_grid_deg, r_grid, z_grid)
    external = ExternalSolver(at_config, bin_path=str(at_bin_path) if at_bin_path else None)
    if not external.is_available("bellhop3d"):
        raise RuntimeError(f"Legacy bellhop3d executable not found in {external.bin_path}.")
    bellhop3d_exe = external.bin_path / "bellhop3d"
    if not bellhop3d_exe.exists():
        bellhop3d_exe = external.bin_path / "bellhop3d.exe"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_name = "bellhop3d_tl_parity"
        generate_at_bellhop3d_env(at_config, tmp_path / f"{case_name}.env", run_type=run_type)
        external.run_command([str(bellhop3d_exe), case_name], cwd=str(tmp_path))
        shd_path = tmp_path / f"{case_name}.shd"
        if not shd_path.exists():
            raise RuntimeError(f"AT Bellhop3D did not generate {shd_path.name}.")
        shade = read_at_shd_full(shd_path)

    tl = _pressure_to_tl(shade.pressure)
    if tl.shape[1] == 1:
        return tl[:, 0]
    return tl


def compute_native_tl_grid_3d(
    config: SimulationConfig,
    theta_grid_deg: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Compute native Bellhop3D receiver-grid TL on an explicit grid."""

    from pyacoustics.simulation import Simulation

    native_config = _with_receiver_grid(config, theta_grid_deg, r_grid, z_grid)
    sim = Simulation(native_config)
    return sim.compute_coherent_tl_3d_receiver_grid(
        theta_grid_deg=np.asarray(theta_grid_deg, dtype=np.float64),
        r_grid=np.asarray(r_grid, dtype=np.float64),
        z_grid=np.asarray(z_grid, dtype=np.float64),
    )


def _receiver_range_mask(tl_shape: tuple[int, ...], r_grid: np.ndarray, min_range_m: float | None) -> np.ndarray:
    if min_range_m is None:
        return np.ones(tl_shape, dtype=bool)
    ranges = np.asarray(r_grid, dtype=np.float64)
    if ranges.ndim != 1 or ranges.size != tl_shape[-1]:
        raise ValueError("r_grid must be one-dimensional and match the TL range axis")
    range_mask = ranges >= float(min_range_m)
    return np.broadcast_to(range_mask.reshape((1,) * (len(tl_shape) - 1) + (ranges.size,)), tl_shape)


def compare_tl_grids_3d(
    at_tl: np.ndarray,
    native_tl: np.ndarray,
    *,
    r_grid: np.ndarray | None = None,
    min_range_m: float | None = None,
    active_tl_max_db: float = 190.0,
) -> Bellhop3DTLParityMetrics:
    """Compare AT and native Bellhop3D TL grids after masking inactive points."""

    at_arr = np.asarray(at_tl, dtype=np.float64)
    native_arr = np.asarray(native_tl, dtype=np.float64)
    if at_arr.shape != native_arr.shape:
        raise ValueError("Bellhop3D TL grids must have identical shapes")
    active_tl_max = float(active_tl_max_db)
    finite_mask = np.isfinite(at_arr) & np.isfinite(native_arr)
    range_mask = np.ones(at_arr.shape, dtype=bool)
    if min_range_m is not None:
        if r_grid is None:
            raise ValueError("r_grid is required when min_range_m is provided")
        range_mask = _receiver_range_mask(at_arr.shape, r_grid, min_range_m)
    at_active = finite_mask & range_mask & (at_arr < active_tl_max)
    native_active = finite_mask & range_mask & (native_arr < active_tl_max)
    native_weak_tail = finite_mask & range_mask & (native_arr >= active_tl_max) & (native_arr < 190.0) & ~at_active
    mask = at_active & native_active
    if not np.any(mask):
        raise ValueError("No finite active Bellhop3D TL samples to compare")
    union_active = at_active | native_active
    diff = native_arr[mask] - at_arr[mask]
    abs_diff = np.abs(diff)
    return Bellhop3DTLParityMetrics(
        sample_count=int(abs_diff.size),
        at_active_count=int(np.count_nonzero(at_active)),
        native_active_count=int(np.count_nonzero(native_active)),
        at_only_active_count=int(np.count_nonzero(at_active & ~native_active)),
        native_only_active_count=int(np.count_nonzero(native_active & ~at_active)),
        native_weak_tail_count=int(np.count_nonzero(native_weak_tail)),
        active_overlap_fraction=float(abs_diff.size / np.count_nonzero(union_active)),
        mae_db=float(np.mean(abs_diff)),
        median_abs_db=float(np.median(abs_diff)),
        p90_abs_db=float(np.percentile(abs_diff, 90.0)),
        max_abs_db=float(np.max(abs_diff)),
        bias_db=float(np.mean(diff)),
        rmse_db=float(np.sqrt(np.mean(diff * diff))),
    )


def bellhop3d_tl_benchmark_row(
    case_name: str,
    metrics: Bellhop3DTLParityMetrics,
    *,
    run_type: str = "C",
    min_range_m: float | None = None,
    active_tl_max_db: float | None = None,
) -> dict[str, int | float | str | None]:
    """Flatten Bellhop3D TL metrics into one report/CSV-friendly row."""

    row: dict[str, int | float | str | None] = {
        "case": case_name,
        "kind": "bellhop3d_tl",
        "run_type": run_type,
        "sample_count": metrics.sample_count,
        "at_active_count": metrics.at_active_count,
        "native_active_count": metrics.native_active_count,
        "at_only_active_count": metrics.at_only_active_count,
        "native_only_active_count": metrics.native_only_active_count,
        "native_weak_tail_count": metrics.native_weak_tail_count,
        "active_overlap_fraction": metrics.active_overlap_fraction,
        "tl_mae_db": metrics.mae_db,
        "tl_median_abs_db": metrics.median_abs_db,
        "tl_p90_abs_db": metrics.p90_abs_db,
        "tl_max_abs_db": metrics.max_abs_db,
        "tl_bias_db": metrics.bias_db,
        "tl_rmse_db": metrics.rmse_db,
    }
    if min_range_m is not None:
        row["min_range_m"] = float(min_range_m)
    if active_tl_max_db is not None:
        row["active_tl_max_db"] = float(active_tl_max_db)
    return row


def _case_theta_grid(case: Bellhop3DBenchmarkCase) -> np.ndarray:
    if case.theta_grid_deg is not None:
        return np.asarray(case.theta_grid_deg, dtype=np.float64)
    bearings = getattr(case.config.solver, "receiver_bearings", None) or [0.0]
    return np.asarray(bearings, dtype=np.float64)


def _case_receiver_ranges(case: Bellhop3DBenchmarkCase) -> np.ndarray:
    if case.r_grid is not None:
        return np.asarray(case.r_grid, dtype=np.float64)
    return np.asarray(case.config.geometry.receivers.ranges, dtype=np.float64)


def _case_receiver_depths(case: Bellhop3DBenchmarkCase) -> np.ndarray:
    if case.z_grid is not None:
        return np.asarray(case.z_grid, dtype=np.float64)
    return np.asarray(case.config.geometry.receivers.depths, dtype=np.float64)


def run_bellhop3d_benchmark_matrix(
    cases: list[Bellhop3DBenchmarkCase] | tuple[Bellhop3DBenchmarkCase, ...],
    *,
    at_bin_path: str | Path | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> Bellhop3DBenchmarkMatrix:
    """Run multiple Bellhop3D receiver-grid TL parity cases."""

    rows: list[dict[str, int | float | str | bool | None]] = []
    for case in cases:
        theta_grid = _case_theta_grid(case)
        r_grid = _case_receiver_ranges(case)
        z_grid = _case_receiver_depths(case)
        case_at_bin_path = at_bin_path if at_bin_path is not None else case.at_bin_path
        at_tl = run_at_tl_grid_3d(
            case.config,
            theta_grid,
            r_grid,
            z_grid,
            at_bin_path=case_at_bin_path,
            run_type=case.run_type,
        )
        native_tl = compute_native_tl_grid_3d(case.config, theta_grid, r_grid, z_grid)
        metrics = compare_tl_grids_3d(
            at_tl,
            native_tl,
            r_grid=r_grid,
            min_range_m=case.min_range_m,
            active_tl_max_db=case.active_tl_max_db,
        )
        rows.append(
            bellhop3d_tl_benchmark_row(
                case.name,
                metrics,
                run_type=case.run_type,
                min_range_m=case.min_range_m,
                active_tl_max_db=case.active_tl_max_db,
            )
        )
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
    return Bellhop3DBenchmarkMatrix(rows=matrix_rows)


__all__ = [
    "Bellhop3DBenchmarkCase",
    "Bellhop3DBenchmarkMatrix",
    "Bellhop3DTLParityMetrics",
    "bellhop3d_tl_benchmark_row",
    "compare_tl_grids_3d",
    "compute_native_tl_grid_3d",
    "run_at_tl_grid_3d",
    "run_bellhop3d_benchmark_matrix",
]
