"""Utilities for comparing native Bellhop TL grids against AT ``.shd`` output."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop.beam import (
    compute_tl_field_hat_coherent,
    compute_tl_field_coherent,
    compute_tl_field_hat_incoherent,
    compute_tl_field_hat_semicoherent,
    compute_tl_field_incoherent,
    compute_tl_field_simple_gaussian_coherent,
    compute_tl_field_semicoherent,
)
from pyacoustics.solvers.bellhop.core import PyBellhop
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.external_io import generate_at_bellhop_env, read_at_prt, read_at_shd_full


@dataclass(frozen=True)
class TLParityMetrics:
    """Summary metrics for a TL grid comparison."""

    sample_count: int
    mae_db: float
    median_abs_db: float
    p90_abs_db: float
    max_abs_db: float
    bias_db: float
    rmse_db: float


def _float_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _raise_for_at_prt_errors(prt_path: Path) -> None:
    if not prt_path.exists():
        return
    log = read_at_prt(prt_path)
    if not log.errors:
        return
    context: list[str] = []
    for idx, line in enumerate(log.lines):
        if "fatal" not in line.lower() and "error" not in line.lower():
            continue
        for context_line in log.lines[idx : idx + 4]:
            stripped = context_line.strip()
            if stripped and stripped not in context:
                context.append(stripped)
        break
    message = "; ".join(context or log.errors[-3:])
    raise RuntimeError(f"AT Bellhop TL run failed: {message}")


def tl_benchmark_row(
    case_name: str,
    metrics: TLParityMetrics,
    *,
    run_type: str | None = None,
) -> dict[str, int | float | str | None]:
    """Flatten Bellhop TL parity metrics into one report/CSV-friendly row."""

    return {
        "case": case_name,
        "run_type": run_type.upper() if run_type is not None else None,
        "sample_count": int(metrics.sample_count),
        "tl_mae_db": _float_or_none(metrics.mae_db),
        "tl_median_abs_db": _float_or_none(metrics.median_abs_db),
        "tl_p90_abs_db": _float_or_none(metrics.p90_abs_db),
        "tl_max_abs_db": _float_or_none(metrics.max_abs_db),
        "tl_bias_db": _float_or_none(metrics.bias_db),
        "tl_rmse_db": _float_or_none(metrics.rmse_db),
    }


def run_at_tl_grid(
    config: SimulationConfig,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    run_type: str = "C",
    at_bin_path: str | Path | None = None,
) -> np.ndarray:
    """Run AT Bellhop TL mode and return a grid shaped ``(range, depth)``."""

    run_type = run_type.upper()
    if run_type not in {"C", "I", "S"}:
        raise ValueError("run_type must be one of 'C', 'I', or 'S'")

    external = ExternalSolver(config, bin_path=str(at_bin_path) if at_bin_path else None)
    if not external.is_available("bellhop"):
        raise RuntimeError(f"Legacy bellhop executable not found in {external.bin_path}.")

    exe_path = external.bin_path / "bellhop"
    if not exe_path.exists():
        exe_path = external.bin_path / "bellhop.exe"

    at_config = deepcopy(config)
    at_config.geometry.receivers.ranges = np.asarray(r_grid, dtype=np.float64).tolist()
    at_config.geometry.receivers.depths = np.asarray(z_grid, dtype=np.float64).tolist()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        env_path = tmp_path / "tl_parity.env"
        generate_at_bellhop_env(at_config, env_path, run_type=run_type)
        external.run_command([str(exe_path), "tl_parity"], cwd=str(tmp_path))
        prt_path = tmp_path / "tl_parity.prt"
        _raise_for_at_prt_errors(prt_path)
        try:
            shade = read_at_shd_full(tmp_path / "tl_parity.shd")
        except ValueError as exc:
            _raise_for_at_prt_errors(prt_path)
            raise RuntimeError(f"AT Bellhop TL run produced an unreadable .shd file: {exc}") from exc
        pressure = shade.pressure[0]
        tl = -20.0 * np.log10(np.abs(pressure) + 1.0e-20)
        if tl.shape[0] == 1:
            return tl[0].T
        return tl.transpose(0, 2, 1)


def compute_native_tl_grid(
    config: SimulationConfig,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    run_type: str = "C",
) -> np.ndarray:
    """Compute native Gaussian-beam TL for coherent or intensity modes."""

    run_type = run_type.upper()
    if run_type not in {"C", "I", "S"}:
        raise ValueError("run_type must be one of 'C', 'I', or 'S'")

    native_config = deepcopy(config)
    native_config.geometry.receivers.ranges = np.asarray(r_grid, dtype=np.float64).tolist()
    native_config.geometry.receivers.depths = np.asarray(z_grid, dtype=np.float64).tolist()

    angles = native_config.solver.angles
    d_alpha = np.radians(angles[1] - angles[0]) / max(native_config.solver.num_beams - 1, 1)
    use_thorp = native_config.environment.absorption_model == "thorp"
    beam_type = getattr(native_config.solver, "beam_type", "default")
    line_source = getattr(native_config.solver, "source_type", "point") == "line"
    use_hat_kernel = beam_type in {"default", "geometric_hat", "geometric_hat_ray_centered"}
    solver = PyBellhop(native_config)
    source_depths = np.asarray(native_config.geometry.source.depths, dtype=np.float64)
    tl_by_source = []

    for source_depth in source_depths:
        beam_data = solver.run_beam_tracing(source_depth=float(source_depth))
        args = (
            np.asarray(r_grid, dtype=np.float64),
            np.asarray(z_grid, dtype=np.float64),
            beam_data["ray_r"],
            beam_data["ray_z"],
            beam_data["ray_c"],
            beam_data["ray_tau"],
            beam_data["ray_q"],
            beam_data["ray_amp"],
            beam_data["ray_npts"],
            beam_data["ray_alphas"],
            native_config.frequency,
            beam_data["c_src"],
            d_alpha,
        )
        if run_type == "C":
            if use_hat_kernel:
                tl_by_source.append(compute_tl_field_hat_coherent(*args, use_thorp=use_thorp, line_source=line_source))
            elif beam_type == "simple_gaussian":
                tl_by_source.append(
                    compute_tl_field_simple_gaussian_coherent(
                        np.asarray(r_grid, dtype=np.float64),
                        np.asarray(z_grid, dtype=np.float64),
                        beam_data["ray_r"],
                        beam_data["ray_z"],
                        beam_data["ray_tau"],
                        beam_data["ray_amp"],
                        beam_data["ray_phase"],
                        beam_data["ray_q"],
                        beam_data["ray_npts"],
                        beam_data["ray_alphas"],
                        native_config.frequency,
                        beam_data["c_src"],
                        d_alpha,
                        float(native_config.solver.step_size),
                        line_source=line_source,
                    )
                )
            else:
                tl_by_source.append(compute_tl_field_coherent(*args, use_thorp=use_thorp, line_source=line_source))
        elif run_type == "S":
            if use_hat_kernel:
                tl_by_source.append(
                    compute_tl_field_hat_semicoherent(
                        *args,
                        float(source_depth),
                        use_thorp=use_thorp,
                        line_source=line_source,
                    )
                )
            else:
                tl_by_source.append(
                    compute_tl_field_semicoherent(
                        *args,
                        float(source_depth),
                        use_thorp=use_thorp,
                        line_source=line_source,
                    )
                )
        elif use_hat_kernel:
            tl_by_source.append(compute_tl_field_hat_incoherent(*args, use_thorp=use_thorp, line_source=line_source))
        else:
            tl_by_source.append(compute_tl_field_incoherent(*args, use_thorp=use_thorp, line_source=line_source))

    if source_depths.size == 1:
        return tl_by_source[0]
    return np.stack(tl_by_source, axis=0)


def compare_tl_grids(
    at_tl: np.ndarray,
    native_tl: np.ndarray,
    *,
    valid_min_db: float = 0.0,
    valid_max_db: float = 119.0,
) -> TLParityMetrics:
    """Compare TL grids on finite, active water-column samples."""

    if at_tl.shape != native_tl.shape:
        raise ValueError("AT and native TL grids must have the same shape")

    mask = (
        np.isfinite(at_tl)
        & np.isfinite(native_tl)
        & (at_tl >= valid_min_db)
        & (native_tl >= valid_min_db)
        & (at_tl < valid_max_db)
        & (native_tl < valid_max_db)
    )
    signed = native_tl[mask] - at_tl[mask]
    if signed.size == 0:
        nan = float("nan")
        return TLParityMetrics(0, nan, nan, nan, nan, nan, nan)

    diffs = np.abs(signed)
    return TLParityMetrics(
        sample_count=int(diffs.size),
        mae_db=float(np.mean(diffs)),
        median_abs_db=float(np.median(diffs)),
        p90_abs_db=float(np.percentile(diffs, 90.0)),
        max_abs_db=float(np.max(diffs)),
        bias_db=float(np.mean(signed)),
        rmse_db=float(np.sqrt(np.mean(signed * signed))),
    )
