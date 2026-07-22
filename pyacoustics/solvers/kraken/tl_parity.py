"""Utilities for Kraken/Field AT parity metrics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.benchmark_thresholds import BenchmarkThresholds, evaluate_benchmark_rows
from pyacoustics.solvers.external import ExternalSolver, KrakenExternal
from pyacoustics.solvers.external_io import (
    ATPRTLog,
    generate_at_field_flp,
    generate_at_kraken_env,
    read_at_prt,
    read_at_shd_full,
)
from pyacoustics.solvers.kraken import PyKraken


@dataclass(frozen=True)
class KrakenTLParityMetrics:
    """Summary metrics for native Kraken vs AT Field TL comparisons."""

    sample_count: int
    mae_db: float
    median_abs_db: float
    p90_abs_db: float
    max_abs_db: float
    bias_db: float
    rmse_db: float


@dataclass(frozen=True)
class KrakenATTLResult:
    """AT Kraken/Field TL grid and parsed run logs."""

    tl_grid: np.ndarray
    receiver_ranges_m: np.ndarray
    receiver_depths_m: np.ndarray
    kraken_log: ATPRTLog | None
    field_log: ATPRTLog | None


@dataclass(frozen=True)
class KrakenTLParityReport:
    """Complete Kraken/Field parity result for benchmark matrix reporting."""

    tl_metrics: KrakenTLParityMetrics
    mode_metrics: KrakenModeParityMetrics | None
    at_result: KrakenATTLResult
    benchmark_row: dict[str, int | float | str | bool | None]


@dataclass(frozen=True)
class KrakenBenchmarkCase:
    """One Kraken/Field parity benchmark scenario."""

    name: str
    config: SimulationConfig
    r_grid: np.ndarray | None = None
    z_grid: np.ndarray | None = None
    at_bin_path: str | Path | None = None


@dataclass(frozen=True)
class KrakenBenchmarkMatrix:
    """Batch Kraken/Field parity reports plus flattened rows."""

    reports: tuple[KrakenTLParityReport, ...]
    rows: tuple[dict[str, int | float | str | bool | None], ...]


@dataclass(frozen=True)
class KrakenModeSummary:
    """Compact mode-table summary for native Kraken or parsed AT ``.prt`` data."""

    mode_count: int | None
    wavenumber_min_1pm: float | None
    wavenumber_max_1pm: float | None
    attenuation_min_1pm: float | None
    attenuation_max_1pm: float | None
    phase_speed_min_mps: float | None
    phase_speed_max_mps: float | None


def _with_receiver_grid(
    config: SimulationConfig,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> SimulationConfig:
    updated = deepcopy(config)
    updated.geometry.receivers.ranges = np.asarray(r_grid, dtype=np.float64).tolist()
    updated.geometry.receivers.depths = np.asarray(z_grid, dtype=np.float64).tolist()
    updated.solver.acoustic_media_model = "single"
    return updated


def _resample_tl_grid(
    tl_grid: np.ndarray,
    source_ranges_m: np.ndarray,
    source_depths_m: np.ndarray,
    target_ranges_m: np.ndarray,
    target_depths_m: np.ndarray,
) -> np.ndarray:
    """Interpolate a TL grid onto an explicit receiver grid.

    ``tl_grid`` may be shaped ``(depth, range)`` for a single source depth or
    ``(source_depth, depth, range)`` when Field carries a source-depth axis.
    """

    tl_grid = np.asarray(tl_grid, dtype=np.float64)
    source_ranges = np.asarray(source_ranges_m, dtype=np.float64)
    source_depths = np.asarray(source_depths_m, dtype=np.float64)
    target_ranges = np.asarray(target_ranges_m, dtype=np.float64)
    target_depths = np.asarray(target_depths_m, dtype=np.float64)

    if tl_grid.ndim == 3:
        return np.stack(
            [
                _resample_tl_grid(source_tl, source_ranges, source_depths, target_ranges, target_depths)
                for source_tl in tl_grid
            ],
            axis=0,
        )

    if tl_grid.shape != (source_depths.size, source_ranges.size):
        raise ValueError("TL grid shape must match source depth/range axes")

    range_interp = np.vstack([np.interp(target_ranges, source_ranges, row) for row in tl_grid])
    depth_interp = np.vstack(
        [np.interp(target_depths, source_depths, range_interp[:, col]) for col in range(target_ranges.size)]
    ).T
    return np.asarray(depth_interp, dtype=np.float64)


def compute_native_tl_grid(
    config: SimulationConfig,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Compute a native Kraken TL grid shaped ``(depth, range)``."""

    result = compute_native_result(config, r_grid, z_grid)
    return _resample_tl_grid(
        result["tl_grid"],
        result["r_bins"],
        result["z_bins"],
        r_grid,
        z_grid,
    )


def compute_native_result(
    config: SimulationConfig,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> dict:
    """Compute native Kraken result dict on an explicit receiver grid."""

    return PyKraken(_with_receiver_grid(config, r_grid, z_grid)).run_normal_modes()


def run_at_tl_grid(
    config: SimulationConfig,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    at_bin_path: str | Path | None = None,
) -> KrakenATTLResult:
    """Run AT Kraken/KRAKENC + Field and return a comparable TL grid.

    The returned TL grid follows Kraken's native layout: ``(depth, range)``.
    """

    at_config = _with_receiver_grid(config, r_grid, z_grid)
    external = ExternalSolver(at_config, bin_path=str(at_bin_path) if at_bin_path else None)
    if not external.is_available("field"):
        raise RuntimeError(f"Legacy field executable not found in {external.bin_path}.")

    is_complex = KrakenExternal._needs_krakenc(at_config)
    exe_name = "krakenc" if is_complex else "kraken"
    if not external.is_available(exe_name):
        if is_complex and external.is_available("kraken"):
            exe_name = "kraken"
        else:
            raise RuntimeError(f"Legacy {exe_name} executable not found in {external.bin_path}.")

    kraken_exe = external.bin_path / exe_name
    if not kraken_exe.exists():
        kraken_exe = external.bin_path / f"{exe_name}.exe"
    field_exe = external.bin_path / "field"
    if not field_exe.exists():
        field_exe = external.bin_path / "field.exe"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_name = "kraken_tl_parity"
        generate_at_kraken_env(at_config, tmp_path / f"{case_name}.env")
        generate_at_field_flp(at_config, tmp_path / f"{case_name}.flp")

        external.run_command([str(kraken_exe), case_name], cwd=str(tmp_path), stdin_input=f"{case_name}\n")
        prt_path = tmp_path / f"{case_name}.prt"
        kraken_log = read_at_prt(prt_path) if prt_path.exists() else None
        external.run_command([str(field_exe), case_name], cwd=str(tmp_path), stdin_input=f"{case_name}\n")

        shd_path = tmp_path / f"{case_name}.shd"
        if not shd_path.exists():
            field_prt = tmp_path / "field.prt"
            if not field_prt.exists():
                field_prt = tmp_path / f"{case_name}.prt"
            prt_tail = field_prt.read_text(errors="replace")[-2000:] if field_prt.exists() else ""
            raise RuntimeError(f"AT Field did not generate {shd_path.name}.\n{prt_tail}")

        shade = read_at_shd_full(shd_path)
        pressure = shade.pressure[0]
        raw_tl_grid = -20.0 * np.log10(np.abs(pressure) + 1.0e-20)
        tl_grid = _resample_tl_grid(raw_tl_grid, shade.receiver_ranges_m, shade.receiver_depths_m, r_grid, z_grid)
        if tl_grid.shape[0] == 1:
            tl_grid = tl_grid[0]
        field_prt_path = tmp_path / "field.prt"
        if not field_prt_path.exists():
            field_prt_path = prt_path
        field_log = read_at_prt(field_prt_path) if field_prt_path.exists() else None
        return KrakenATTLResult(
            tl_grid=np.asarray(tl_grid, dtype=np.float64),
            receiver_ranges_m=np.asarray(r_grid, dtype=np.float64),
            receiver_depths_m=np.asarray(z_grid, dtype=np.float64),
            kraken_log=kraken_log,
            field_log=field_log,
        )


@dataclass(frozen=True)
class KrakenModeParityMetrics:
    """Summary differences between native modal diagnostics and AT ``.prt`` metadata."""

    native_mode_count: int | None
    at_mode_count: int | None
    mode_count_delta: int | None
    wavenumber_min_error_1pm: float | None
    wavenumber_max_error_1pm: float | None
    attenuation_min_error_1pm: float | None
    attenuation_max_error_1pm: float | None
    phase_speed_min_error_mps: float | None
    phase_speed_max_error_mps: float | None


def _float_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def kraken_benchmark_row(
    case_name: str,
    tl_metrics: KrakenTLParityMetrics,
    mode_metrics: KrakenModeParityMetrics | None = None,
    *,
    at_log: ATPRTLog | None = None,
    field_log: ATPRTLog | None = None,
) -> dict[str, int | float | str | bool | None]:
    """Flatten Kraken parity metrics into one report/CSV-friendly row."""

    row: dict[str, int | float | str | bool | None] = {
        "case": case_name,
        "kind": "kraken_tl",
        "sample_count": int(tl_metrics.sample_count),
        "tl_mae_db": _float_or_none(tl_metrics.mae_db),
        "tl_median_abs_db": _float_or_none(tl_metrics.median_abs_db),
        "tl_p90_abs_db": _float_or_none(tl_metrics.p90_abs_db),
        "tl_max_abs_db": _float_or_none(tl_metrics.max_abs_db),
        "tl_bias_db": _float_or_none(tl_metrics.bias_db),
        "tl_rmse_db": _float_or_none(tl_metrics.rmse_db),
    }
    if mode_metrics is not None:
        row.update(
            {
                "native_mode_count": mode_metrics.native_mode_count,
                "at_mode_count": mode_metrics.at_mode_count,
                "mode_count_delta": mode_metrics.mode_count_delta,
                "wavenumber_min_error_1pm": _float_or_none(mode_metrics.wavenumber_min_error_1pm),
                "wavenumber_max_error_1pm": _float_or_none(mode_metrics.wavenumber_max_error_1pm),
                "attenuation_min_error_1pm": _float_or_none(mode_metrics.attenuation_min_error_1pm),
                "attenuation_max_error_1pm": _float_or_none(mode_metrics.attenuation_max_error_1pm),
                "phase_speed_min_error_mps": _float_or_none(mode_metrics.phase_speed_min_error_mps),
                "phase_speed_max_error_mps": _float_or_none(mode_metrics.phase_speed_max_error_mps),
            }
        )
    if at_log is not None:
        row.update(
            {
                "at_program": at_log.program,
                "at_completed_successfully": at_log.completed_successfully,
                "at_completed_without_errors": at_log.completed_without_errors,
                "at_warning_count": at_log.warning_count,
                "at_error_count": at_log.error_count,
                "at_fatal_error_count": at_log.fatal_error_count,
                "at_cpu_time_s": _float_or_none(at_log.cpu_time_s),
                "at_mode_count": at_log.mode_count,
            }
        )
    if field_log is not None:
        row.update(
            {
                "field_program": field_log.program,
                "field_completed_successfully": field_log.completed_successfully,
                "field_completed_without_errors": field_log.completed_without_errors,
                "field_warning_count": field_log.warning_count,
                "field_error_count": field_log.error_count,
                "field_fatal_error_count": field_log.fatal_error_count,
                "field_cpu_time_s": _float_or_none(field_log.cpu_time_s),
                "field_receiver_range_count": field_log.receiver_range_count,
                "field_receiver_depth_count": field_log.receiver_depth_count,
                "field_profile_range_count": field_log.profile_range_count,
                "field_receiver_range_offset_count": field_log.receiver_range_offset_count,
            }
        )
    return row


def compare_tl_grids(
    at_tl: np.ndarray,
    native_tl: np.ndarray,
    *,
    valid_min_db: float = 0.0,
    valid_max_db: float = 119.0,
) -> KrakenTLParityMetrics:
    """Compare native Kraken and AT Field TL grids on finite active samples."""

    at_tl = np.asarray(at_tl, dtype=np.float64)
    native_tl = np.asarray(native_tl, dtype=np.float64)
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
        return KrakenTLParityMetrics(0, nan, nan, nan, nan, nan, nan)

    abs_diff = np.abs(signed)
    return KrakenTLParityMetrics(
        sample_count=int(abs_diff.size),
        mae_db=float(np.mean(abs_diff)),
        median_abs_db=float(np.median(abs_diff)),
        p90_abs_db=float(np.percentile(abs_diff, 90.0)),
        max_abs_db=float(np.max(abs_diff)),
        bias_db=float(np.mean(signed)),
        rmse_db=float(np.sqrt(np.mean(signed * signed))),
    )


def _min_max(values: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    return float(np.min(values)), float(np.max(values))


def _attenuation_min_max(values: np.ndarray) -> tuple[float | None, float | None]:
    return _min_max(np.abs(np.asarray(values, dtype=np.float64)))


def summarize_native_modes(result: dict) -> KrakenModeSummary:
    """Summarize native Kraken modal diagnostics for AT ``.prt`` comparison."""

    wavenumbers = np.asarray(result.get("wavenumbers", []), dtype=np.complex128)
    mode_count = int(wavenumbers.size)
    wavenumber_min, wavenumber_max = _min_max(np.real(wavenumbers))

    attenuation = result.get("modal_attenuation")
    if attenuation is None:
        attenuation = np.imag(wavenumbers)
    attenuation_min, attenuation_max = _attenuation_min_max(np.asarray(attenuation, dtype=np.float64))

    phase_speeds = result.get("phase_speeds")
    phase_speed_min, phase_speed_max = _min_max(
        np.asarray(phase_speeds, dtype=np.float64) if phase_speeds is not None else np.array([])
    )

    return KrakenModeSummary(
        mode_count=mode_count,
        wavenumber_min_1pm=wavenumber_min,
        wavenumber_max_1pm=wavenumber_max,
        attenuation_min_1pm=attenuation_min,
        attenuation_max_1pm=attenuation_max,
        phase_speed_min_mps=phase_speed_min,
        phase_speed_max_mps=phase_speed_max,
    )


def summarize_prt_modes(log: ATPRTLog) -> KrakenModeSummary:
    """Summarize parsed AT Kraken/KRAKENC mode metadata from ``ATPRTLog``."""

    attenuation_min, attenuation_max = _attenuation_min_max(
        np.asarray(
            [
                value
                for value in (log.mode_attenuation_min_1pm, log.mode_attenuation_max_1pm)
                if value is not None
            ],
            dtype=np.float64,
        )
    )
    return KrakenModeSummary(
        mode_count=log.mode_count,
        wavenumber_min_1pm=log.mode_wavenumber_min_1pm,
        wavenumber_max_1pm=log.mode_wavenumber_max_1pm,
        attenuation_min_1pm=attenuation_min,
        attenuation_max_1pm=attenuation_max,
        phase_speed_min_mps=log.mode_phase_speed_min_mps,
        phase_speed_max_mps=log.mode_phase_speed_max_mps,
    )


def _diff(native_value: float | None, at_value: float | None) -> float | None:
    if native_value is None or at_value is None:
        return None
    return float(native_value - at_value)


def compare_mode_summaries(
    native_summary: KrakenModeSummary,
    at_summary: KrakenModeSummary,
) -> KrakenModeParityMetrics:
    """Compare native and AT mode summary ranges."""

    mode_count_delta = None
    if native_summary.mode_count is not None and at_summary.mode_count is not None:
        mode_count_delta = int(native_summary.mode_count - at_summary.mode_count)

    return KrakenModeParityMetrics(
        native_mode_count=native_summary.mode_count,
        at_mode_count=at_summary.mode_count,
        mode_count_delta=mode_count_delta,
        wavenumber_min_error_1pm=_diff(native_summary.wavenumber_min_1pm, at_summary.wavenumber_min_1pm),
        wavenumber_max_error_1pm=_diff(native_summary.wavenumber_max_1pm, at_summary.wavenumber_max_1pm),
        attenuation_min_error_1pm=_diff(native_summary.attenuation_min_1pm, at_summary.attenuation_min_1pm),
        attenuation_max_error_1pm=_diff(native_summary.attenuation_max_1pm, at_summary.attenuation_max_1pm),
        phase_speed_min_error_mps=_diff(native_summary.phase_speed_min_mps, at_summary.phase_speed_min_mps),
        phase_speed_max_error_mps=_diff(native_summary.phase_speed_max_mps, at_summary.phase_speed_max_mps),
    )


def compare_native_modes_to_prt(result: dict, log: ATPRTLog) -> KrakenModeParityMetrics:
    """Compare native Kraken modal diagnostics against parsed AT ``.prt`` metadata."""

    return compare_mode_summaries(summarize_native_modes(result), summarize_prt_modes(log))


def run_kraken_tl_parity(
    config: SimulationConfig,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    at_bin_path: str | Path | None = None,
    case_name: str = "kraken-tl-parity",
) -> KrakenTLParityReport:
    """Run native Kraken and AT Kraken/Field, returning metrics plus report row data."""

    native_result = compute_native_result(config, r_grid, z_grid)
    native_tl = _resample_tl_grid(
        native_result["tl_grid"],
        native_result["r_bins"],
        native_result["z_bins"],
        r_grid,
        z_grid,
    )
    at_result = run_at_tl_grid(config, r_grid, z_grid, at_bin_path=at_bin_path)
    tl_metrics = compare_tl_grids(at_result.tl_grid, native_tl)

    at_log = at_result.kraken_log or at_result.field_log
    mode_metrics = compare_native_modes_to_prt(native_result, at_log) if at_log is not None else None
    row = kraken_benchmark_row(case_name, tl_metrics, mode_metrics, at_log=at_log, field_log=at_result.field_log)
    return KrakenTLParityReport(
        tl_metrics=tl_metrics,
        mode_metrics=mode_metrics,
        at_result=at_result,
        benchmark_row=row,
    )


def run_kraken_benchmark_matrix(
    cases: list[KrakenBenchmarkCase] | tuple[KrakenBenchmarkCase, ...],
    *,
    at_bin_path: str | Path | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> KrakenBenchmarkMatrix:
    """Run multiple Kraken/Field parity cases and collect report rows."""

    reports: list[KrakenTLParityReport] = []
    rows: list[dict[str, int | float | str | bool | None]] = []
    for case in cases:
        r_grid = (
            np.asarray(case.r_grid, dtype=np.float64)
            if case.r_grid is not None
            else np.asarray(case.config.geometry.receivers.ranges, dtype=np.float64)
        )
        z_grid = (
            np.asarray(case.z_grid, dtype=np.float64)
            if case.z_grid is not None
            else np.asarray(case.config.geometry.receivers.depths, dtype=np.float64)
        )
        report = run_kraken_tl_parity(
            case.config,
            r_grid,
            z_grid,
            at_bin_path=at_bin_path if at_bin_path is not None else case.at_bin_path,
            case_name=case.name,
        )
        reports.append(report)
        rows.append(report.benchmark_row)
    matrix_rows = evaluate_benchmark_rows(rows, thresholds) if thresholds is not None else tuple(rows)
    return KrakenBenchmarkMatrix(reports=tuple(reports), rows=matrix_rows)


__all__ = [
    "BenchmarkThresholds",
    "KrakenATTLResult",
    "KrakenBenchmarkCase",
    "KrakenBenchmarkMatrix",
    "KrakenModeParityMetrics",
    "KrakenModeSummary",
    "KrakenTLParityMetrics",
    "KrakenTLParityReport",
    "compare_mode_summaries",
    "compare_native_modes_to_prt",
    "compare_tl_grids",
    "compute_native_result",
    "compute_native_tl_grid",
    "kraken_benchmark_row",
    "run_at_tl_grid",
    "run_kraken_benchmark_matrix",
    "run_kraken_tl_parity",
    "summarize_native_modes",
    "summarize_prt_modes",
]
