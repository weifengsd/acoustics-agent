"""Utilities for Bellhop3D AT/native arrivals parity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from pyacoustics.postprocessing.delay_and_sum import ArrivalData, ArrivalRecord
from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop3d.arrivals import compute_arrival_grid_3d, write_arrival_grid_3d
from pyacoustics.solvers.benchmark_thresholds import BenchmarkThresholds, evaluate_benchmark_rows
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.external_io import generate_at_bellhop3d_env, read_at_arr


@dataclass(frozen=True)
class Bellhop3DArrivalParityMetrics:
    """Summary metrics for an AT/native Bellhop3D arrivals comparison."""

    at_count: int
    native_count: int
    matched_count: int
    unmatched_at_count: int
    unmatched_native_count: int
    match_fraction: float
    delay_mae_s: float
    delay_p90_abs_s: float
    delay_max_abs_s: float
    amplitude_median_relative_error: float
    amplitude_p90_relative_error: float
    phase_mae_rad: float
    phase_max_abs_rad: float
    launch_declination_mae_deg: float
    launch_azimuth_mae_deg: float
    receiver_declination_mae_deg: float
    receiver_azimuth_mae_deg: float
    bounce_match_count: int
    bounce_match_fraction: float
    top_bounce_match_count: int
    top_bounce_match_fraction: float
    bottom_bounce_match_count: int
    bottom_bounce_match_fraction: float


@dataclass(frozen=True)
class Bellhop3DArrivalBenchmarkCase:
    """One Bellhop3D AT/native arrivals parity benchmark scenario."""

    name: str
    config: SimulationConfig
    run_type: str = "A"
    delay_tolerance_s: float = 0.02
    angle_tolerance_deg: float | None = None
    tolerance_m: float | None = None
    max_arrivals: int = 200
    at_bin_path: str | Path | None = None
    thresholds: BenchmarkThresholds | None = None


@dataclass(frozen=True)
class Bellhop3DArrivalBenchmarkMatrix:
    """Batch Bellhop3D arrivals parity rows grouped by benchmark case."""

    rows: tuple[dict[str, int | float | str | bool | None], ...]


def _float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _angle_error_deg(native_deg: float, at_deg: float) -> float:
    return float(np.degrees(np.arctan2(np.sin(np.deg2rad(native_deg - at_deg)), np.cos(np.deg2rad(native_deg - at_deg)))))


def run_at_arrival_data_3d(
    config: SimulationConfig,
    *,
    at_bin_path: str | Path | None = None,
    run_type: str = "A",
) -> ArrivalData:
    """Run AT Bellhop3D in arrivals mode and parse its ASCII ``.arr`` output."""

    external = ExternalSolver(config, bin_path=str(at_bin_path) if at_bin_path else None)
    if not external.is_available("bellhop3d"):
        raise RuntimeError(f"Legacy bellhop3d executable not found in {external.bin_path}.")
    exe_path = external.bin_path / "bellhop3d"
    if not exe_path.exists():
        exe_path = external.bin_path / "bellhop3d.exe"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_name = "bellhop3d_arrival_parity"
        generate_at_bellhop3d_env(config, tmp_path / f"{case_name}.env", run_type=run_type)
        external.run_command([str(exe_path), case_name], cwd=str(tmp_path))
        arr_path = tmp_path / f"{case_name}.arr"
        if not arr_path.exists():
            raise RuntimeError(f"AT Bellhop3D did not generate {arr_path.name}.")
        return read_at_arr(arr_path)


def compute_native_arrival_data_3d(
    config: SimulationConfig,
    *,
    receiver_ranges: np.ndarray | None = None,
    receiver_depths: np.ndarray | None = None,
    receiver_bearings: np.ndarray | None = None,
    source_depths: np.ndarray | None = None,
    source_x: np.ndarray | None = None,
    source_y: np.ndarray | None = None,
    max_arrivals: int = 200,
    tolerance_m: float | None = None,
) -> ArrivalData:
    """Compute native Bellhop3D arrivals and parse the AT-compatible ASCII form."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        arr_path = Path(tmp_dir) / "native_3d.arr"
        grid = compute_arrival_grid_3d(
            config,
            receiver_ranges=receiver_ranges,
            receiver_depths=receiver_depths,
            receiver_bearings=receiver_bearings,
            source_depths=source_depths,
            source_x=source_x,
            source_y=source_y,
            max_arrivals=max_arrivals,
            tolerance_m=tolerance_m,
        )
        write_arrival_grid_3d(arr_path, grid)
        return read_at_arr(arr_path)


def _match_record_by_delay_3d(
    at_record: ArrivalRecord,
    native_record: ArrivalRecord,
    *,
    delay_tolerance_s: float,
    angle_tolerance_deg: float | None = None,
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    used_native: set[int] = set()
    native_delays = np.real(native_record.delays)

    for at_idx, at_delay in enumerate(np.real(at_record.delays)):
        candidates: list[tuple[float, float, int]] = []
        for native_idx in range(native_record.count):
            native_idx = int(native_idx)
            if native_idx in used_native:
                continue
            if native_record.num_top_bounces[native_idx] != at_record.num_top_bounces[at_idx]:
                continue
            if native_record.num_bottom_bounces[native_idx] != at_record.num_bottom_bounces[at_idx]:
                continue
            delay_abs = float(abs(native_delays[native_idx] - at_delay))
            if delay_abs > delay_tolerance_s:
                continue
            launch_decl_error = abs(float(native_record.src_decl_angles[native_idx] - at_record.src_decl_angles[at_idx]))
            launch_azim_error = abs(_angle_error_deg(native_record.src_azim_angles[native_idx], at_record.src_azim_angles[at_idx]))
            receiver_decl_error = abs(float(native_record.rcvr_decl_angles[native_idx] - at_record.rcvr_decl_angles[at_idx]))
            receiver_azim_error = abs(
                _angle_error_deg(native_record.rcvr_azim_angles[native_idx], at_record.rcvr_azim_angles[at_idx])
            )
            if angle_tolerance_deg is not None and max(
                launch_decl_error,
                launch_azim_error,
                receiver_decl_error,
                receiver_azim_error,
            ) > angle_tolerance_deg:
                continue
            angle_penalty = launch_decl_error + launch_azim_error + receiver_decl_error + receiver_azim_error
            candidates.append((delay_abs, angle_penalty, native_idx))
        if not candidates:
            continue
        _, _, native_idx = min(candidates)
        used_native.add(native_idx)
        matches.append((int(at_idx), int(native_idx)))
    return matches


def compare_arrival_data_3d(
    at_data: ArrivalData,
    native_data: ArrivalData,
    *,
    delay_tolerance_s: float = 0.02,
    angle_tolerance_deg: float | None = None,
) -> Bellhop3DArrivalParityMetrics:
    """Compare AT/native Bellhop3D arrivals using greedy nearest-delay matching."""

    if delay_tolerance_s < 0.0:
        raise ValueError("delay_tolerance_s must be non-negative")
    if angle_tolerance_deg is not None and angle_tolerance_deg < 0.0:
        raise ValueError("angle_tolerance_deg must be non-negative")
    if not at_data.is_3d or not native_data.is_3d:
        raise ValueError("Bellhop3D arrival parity requires 3D ArrivalData")
    if at_data.arrivals.shape != native_data.arrivals.shape:
        raise ValueError("AT and native Bellhop3D arrivals must share the same receiver/source grid")

    at_count = 0
    native_count = 0
    delay_errors: list[float] = []
    amp_rel_errors: list[float] = []
    phase_errors: list[float] = []
    launch_decl_errors: list[float] = []
    launch_azim_errors: list[float] = []
    receiver_decl_errors: list[float] = []
    receiver_azim_errors: list[float] = []
    top_bounce_match_count = 0
    bottom_bounce_match_count = 0
    bounce_match_count = 0

    for receiver_index, at_record in np.ndenumerate(at_data.arrivals):
        native_record = native_data.arrivals[receiver_index]
        at_count += at_record.count
        native_count += native_record.count
        for at_idx, native_idx in _match_record_by_delay_3d(
            at_record,
            native_record,
            delay_tolerance_s=delay_tolerance_s,
            angle_tolerance_deg=angle_tolerance_deg,
        ):
            at_delay = float(np.real(at_record.delays[at_idx]))
            native_delay = float(np.real(native_record.delays[native_idx]))
            delay_errors.append(native_delay - at_delay)

            at_amp = abs(at_record.amplitudes[at_idx])
            native_amp = abs(native_record.amplitudes[native_idx])
            if at_amp > np.finfo(float).eps:
                amp_rel_errors.append(float(abs(native_amp - at_amp) / at_amp))
            phase_errors.append(float(np.angle(native_record.amplitudes[native_idx] * np.conj(at_record.amplitudes[at_idx]))))
            launch_decl_errors.append(float(native_record.src_decl_angles[native_idx] - at_record.src_decl_angles[at_idx]))
            launch_azim_errors.append(_angle_error_deg(native_record.src_azim_angles[native_idx], at_record.src_azim_angles[at_idx]))
            receiver_decl_errors.append(float(native_record.rcvr_decl_angles[native_idx] - at_record.rcvr_decl_angles[at_idx]))
            receiver_azim_errors.append(
                _angle_error_deg(native_record.rcvr_azim_angles[native_idx], at_record.rcvr_azim_angles[at_idx])
            )

            top_match = bool(native_record.num_top_bounces[native_idx] == at_record.num_top_bounces[at_idx])
            bottom_match = bool(native_record.num_bottom_bounces[native_idx] == at_record.num_bottom_bounces[at_idx])
            top_bounce_match_count += int(top_match)
            bottom_bounce_match_count += int(bottom_match)
            bounce_match_count += int(top_match and bottom_match)

    matched_count = len(delay_errors)
    if delay_errors:
        delay_abs = np.abs(np.asarray(delay_errors, dtype=np.float64))
        delay_mae = float(np.mean(delay_abs))
        delay_p90 = float(np.percentile(delay_abs, 90.0))
        delay_max = float(np.max(delay_abs))
    else:
        delay_mae = delay_p90 = delay_max = float("nan")
    if amp_rel_errors:
        amp_rel = np.asarray(amp_rel_errors, dtype=np.float64)
        amp_median = float(np.median(amp_rel))
        amp_p90 = float(np.percentile(amp_rel, 90.0))
    else:
        amp_median = amp_p90 = float("nan")
    if phase_errors:
        phase_abs = np.abs(np.asarray(phase_errors, dtype=np.float64))
        phase_mae = float(np.mean(phase_abs))
        phase_max = float(np.max(phase_abs))
    else:
        phase_mae = phase_max = float("nan")

    return Bellhop3DArrivalParityMetrics(
        at_count=at_count,
        native_count=native_count,
        matched_count=matched_count,
        unmatched_at_count=max(at_count - matched_count, 0),
        unmatched_native_count=max(native_count - matched_count, 0),
        match_fraction=matched_count / at_count if at_count else float("nan"),
        delay_mae_s=delay_mae,
        delay_p90_abs_s=delay_p90,
        delay_max_abs_s=delay_max,
        amplitude_median_relative_error=amp_median,
        amplitude_p90_relative_error=amp_p90,
        phase_mae_rad=phase_mae,
        phase_max_abs_rad=phase_max,
        launch_declination_mae_deg=float(np.mean(np.abs(launch_decl_errors))) if launch_decl_errors else float("nan"),
        launch_azimuth_mae_deg=float(np.mean(np.abs(launch_azim_errors))) if launch_azim_errors else float("nan"),
        receiver_declination_mae_deg=float(np.mean(np.abs(receiver_decl_errors))) if receiver_decl_errors else float("nan"),
        receiver_azimuth_mae_deg=float(np.mean(np.abs(receiver_azim_errors))) if receiver_azim_errors else float("nan"),
        bounce_match_count=bounce_match_count,
        bounce_match_fraction=bounce_match_count / matched_count if matched_count else float("nan"),
        top_bounce_match_count=top_bounce_match_count,
        top_bounce_match_fraction=top_bounce_match_count / matched_count if matched_count else float("nan"),
        bottom_bounce_match_count=bottom_bounce_match_count,
        bottom_bounce_match_fraction=bottom_bounce_match_count / matched_count if matched_count else float("nan"),
    )


def bellhop3d_arrival_benchmark_row(
    case_name: str,
    metrics: Bellhop3DArrivalParityMetrics,
    *,
    run_type: str = "A",
    delay_tolerance_s: float | None = None,
    angle_tolerance_deg: float | None = None,
    tolerance_m: float | None = None,
) -> dict[str, int | float | str | None]:
    """Flatten Bellhop3D arrivals parity metrics into one report/CSV-friendly row."""

    return {
        "case": case_name,
        "kind": "bellhop3d_arrival",
        "run_type": run_type,
        "delay_tolerance_s": _float_or_none(delay_tolerance_s),
        "angle_tolerance_deg": _float_or_none(angle_tolerance_deg),
        "tolerance_m": _float_or_none(tolerance_m),
        "at_count": metrics.at_count,
        "native_count": metrics.native_count,
        "matched_count": metrics.matched_count,
        "unmatched_at_count": metrics.unmatched_at_count,
        "unmatched_native_count": metrics.unmatched_native_count,
        "match_fraction": _float_or_none(metrics.match_fraction),
        "delay_mae_s": _float_or_none(metrics.delay_mae_s),
        "delay_p90_abs_s": _float_or_none(metrics.delay_p90_abs_s),
        "delay_max_abs_s": _float_or_none(metrics.delay_max_abs_s),
        "amplitude_median_relative_error": _float_or_none(metrics.amplitude_median_relative_error),
        "amplitude_p90_relative_error": _float_or_none(metrics.amplitude_p90_relative_error),
        "phase_mae_rad": _float_or_none(metrics.phase_mae_rad),
        "phase_max_abs_rad": _float_or_none(metrics.phase_max_abs_rad),
        "launch_declination_mae_deg": _float_or_none(metrics.launch_declination_mae_deg),
        "launch_azimuth_mae_deg": _float_or_none(metrics.launch_azimuth_mae_deg),
        "receiver_declination_mae_deg": _float_or_none(metrics.receiver_declination_mae_deg),
        "receiver_azimuth_mae_deg": _float_or_none(metrics.receiver_azimuth_mae_deg),
        "bounce_match_count": metrics.bounce_match_count,
        "bounce_match_fraction": _float_or_none(metrics.bounce_match_fraction),
        "top_bounce_match_count": metrics.top_bounce_match_count,
        "top_bounce_match_fraction": _float_or_none(metrics.top_bounce_match_fraction),
        "bottom_bounce_match_count": metrics.bottom_bounce_match_count,
        "bottom_bounce_match_fraction": _float_or_none(metrics.bottom_bounce_match_fraction),
    }


def run_bellhop3d_arrival_parity(
    config: SimulationConfig,
    *,
    at_bin_path: str | Path | None = None,
    run_type: str = "A",
    delay_tolerance_s: float = 0.02,
    angle_tolerance_deg: float | None = None,
    tolerance_m: float | None = None,
    max_arrivals: int = 200,
) -> Bellhop3DArrivalParityMetrics:
    at_data = run_at_arrival_data_3d(config, at_bin_path=at_bin_path, run_type=run_type)
    native_data = compute_native_arrival_data_3d(
        config,
        receiver_ranges=at_data.receiver_ranges,
        receiver_depths=at_data.receiver_depths,
        receiver_bearings=at_data.receiver_bearings,
        source_depths=at_data.source_depths,
        source_x=at_data.source_x,
        source_y=at_data.source_y,
        max_arrivals=max_arrivals,
        tolerance_m=tolerance_m,
    )
    return compare_arrival_data_3d(
        at_data,
        native_data,
        delay_tolerance_s=delay_tolerance_s,
        angle_tolerance_deg=angle_tolerance_deg,
    )


def run_bellhop3d_arrival_benchmark_matrix(
    cases: list[Bellhop3DArrivalBenchmarkCase] | tuple[Bellhop3DArrivalBenchmarkCase, ...],
    *,
    at_bin_path: str | Path | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> Bellhop3DArrivalBenchmarkMatrix:
    """Run multiple Bellhop3D arrivals parity benchmark cases."""

    rows: list[dict[str, int | float | str | bool | None]] = []
    for case in cases:
        case_at_bin_path = at_bin_path if at_bin_path is not None else case.at_bin_path
        metrics = run_bellhop3d_arrival_parity(
            case.config,
            at_bin_path=case_at_bin_path,
            run_type=case.run_type,
            delay_tolerance_s=case.delay_tolerance_s,
            angle_tolerance_deg=case.angle_tolerance_deg,
            tolerance_m=case.tolerance_m,
            max_arrivals=case.max_arrivals,
        )
        rows.append(
            bellhop3d_arrival_benchmark_row(
                case.name,
                metrics,
                run_type=case.run_type,
                delay_tolerance_s=case.delay_tolerance_s,
                angle_tolerance_deg=case.angle_tolerance_deg,
                tolerance_m=case.tolerance_m,
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
    return Bellhop3DArrivalBenchmarkMatrix(rows=matrix_rows)


__all__ = [
    "Bellhop3DArrivalBenchmarkCase",
    "Bellhop3DArrivalBenchmarkMatrix",
    "Bellhop3DArrivalParityMetrics",
    "bellhop3d_arrival_benchmark_row",
    "compare_arrival_data_3d",
    "compute_native_arrival_data_3d",
    "run_at_arrival_data_3d",
    "run_bellhop3d_arrival_benchmark_matrix",
    "run_bellhop3d_arrival_parity",
]
