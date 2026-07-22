"""Utilities for comparing native Bellhop arrivals against AT ``.arr`` output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from pyacoustics.postprocessing.delay_and_sum import ArrivalData, ArrivalRecord
from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.external_io import generate_at_bellhop_env, read_at_arr
from pyacoustics.solvers.bellhop.arrivals import compute_arrival_grid, write_arrivals_ascii


@dataclass(frozen=True)
class ArrivalParityMetrics:
    """Summary metrics for an AT/native arrivals comparison."""

    at_count: int
    native_count: int
    matched_count: int
    unmatched_at_count: int
    unmatched_native_count: int
    match_fraction: float
    delay_mae: float
    delay_p90_abs: float
    delay_max_abs: float
    amplitude_median_relative_error: float
    amplitude_p90_relative_error: float
    phase_mae_rad: float
    phase_max_abs_rad: float
    launch_angle_mae: float
    receiver_angle_mae: float
    bounce_match_count: int
    bounce_match_fraction: float
    top_bounce_match_count: int
    top_bounce_match_fraction: float
    bottom_bounce_match_count: int
    bottom_bounce_match_fraction: float


@dataclass(frozen=True)
class _ArrivalMatch:
    receiver_index: tuple[int, ...]
    at_index: int
    native_index: int
    at_delay: float
    native_delay: float
    delay_error: float
    at_amplitude: float
    native_amplitude: float
    amplitude_relative_error: float
    at_phase_rad: float
    native_phase_rad: float
    phase_error_rad: float
    at_launch_angle: float
    native_launch_angle: float
    launch_angle_error: float
    at_receiver_angle: float
    native_receiver_angle: float
    receiver_angle_error: float
    at_top_bounces: int
    native_top_bounces: int
    top_bounce_match: bool
    at_bottom_bounces: int
    native_bottom_bounces: int
    bottom_bounce_match: bool


def _float_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def arrival_benchmark_row(
    case_name: str,
    metrics: ArrivalParityMetrics,
    *,
    method: str | None = None,
    delay_tolerance: float | None = None,
) -> dict[str, int | float | str | None]:
    """Flatten arrival parity metrics into one report/CSV-friendly row."""

    row: dict[str, int | float | str | None] = {
        "case": case_name,
        "method": method,
        "delay_tolerance_s": _float_or_none(delay_tolerance),
        "at_count": metrics.at_count,
        "native_count": metrics.native_count,
        "matched_count": metrics.matched_count,
        "unmatched_at_count": metrics.unmatched_at_count,
        "unmatched_native_count": metrics.unmatched_native_count,
        "match_fraction": _float_or_none(metrics.match_fraction),
        "delay_mae_s": _float_or_none(metrics.delay_mae),
        "delay_p90_abs_s": _float_or_none(metrics.delay_p90_abs),
        "delay_max_abs_s": _float_or_none(metrics.delay_max_abs),
        "amplitude_median_relative_error": _float_or_none(metrics.amplitude_median_relative_error),
        "amplitude_p90_relative_error": _float_or_none(metrics.amplitude_p90_relative_error),
        "phase_mae_rad": _float_or_none(metrics.phase_mae_rad),
        "phase_max_abs_rad": _float_or_none(metrics.phase_max_abs_rad),
        "launch_angle_mae_deg": _float_or_none(metrics.launch_angle_mae),
        "receiver_angle_mae_deg": _float_or_none(metrics.receiver_angle_mae),
        "bounce_match_count": metrics.bounce_match_count,
        "bounce_match_fraction": _float_or_none(metrics.bounce_match_fraction),
        "top_bounce_match_count": metrics.top_bounce_match_count,
        "top_bounce_match_fraction": _float_or_none(metrics.top_bounce_match_fraction),
        "bottom_bounce_match_count": metrics.bottom_bounce_match_count,
        "bottom_bounce_match_fraction": _float_or_none(metrics.bottom_bounce_match_fraction),
    }
    return row


def run_at_arrival_data(
    config: SimulationConfig,
    *,
    at_bin_path: str | Path | None = None,
    run_type: str = "A",
) -> ArrivalData:
    """Run AT Bellhop in arrivals mode and parse its ASCII ``.arr`` output."""

    external = ExternalSolver(config, bin_path=str(at_bin_path) if at_bin_path else None)
    if not external.is_available("bellhop"):
        raise RuntimeError(f"Legacy bellhop executable not found in {external.bin_path}.")

    exe_path = external.bin_path / "bellhop"
    if not exe_path.exists():
        exe_path = external.bin_path / "bellhop.exe"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        env_path = tmp_path / "arrival_parity.env"
        generate_at_bellhop_env(config, env_path, run_type=run_type)
        external.run_command([str(exe_path), "arrival_parity"], cwd=str(tmp_path))
        return read_at_arr(tmp_path / "arrival_parity.arr")


def compute_native_arrival_data(
    config: SimulationConfig,
    *,
    max_arrivals: int = 200,
    beam_shape: str = "hat",
    method: str = "beam",
    eigenray_tolerance_m: float = 0.25,
    eigenray_angle_samples: int | None = None,
) -> ArrivalData:
    """Compute native arrivals and parse the AT-compatible ASCII representation."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        arr_path = Path(tmp_dir) / "native.arr"
        grid = compute_arrival_grid(
            config,
            max_arrivals=max_arrivals,
            beam_shape=beam_shape,
            method=method,
            eigenray_tolerance_m=eigenray_tolerance_m,
            eigenray_angle_samples=eigenray_angle_samples,
        )
        source_type = "X" if getattr(config.solver, "source_type", "point") == "line" else "P"
        write_arrivals_ascii(arr_path, grid, source_type=source_type)
        return read_at_arr(arr_path)


def _match_record_by_delay(
    at_record: ArrivalRecord,
    native_record: ArrivalRecord,
    *,
    delay_tolerance: float,
    receiver_index: tuple[int, ...] = (),
) -> list[_ArrivalMatch]:
    matches = []
    used_native: set[int] = set()

    for at_idx, at_delay in enumerate(np.real(at_record.delays)):
        if native_record.count == 0:
            break

        native_delays = np.real(native_record.delays)
        candidates = []
        for candidate in range(native_record.count):
            candidate = int(candidate)
            if candidate in used_native:
                continue
            if native_record.num_top_bounces[candidate] != at_record.num_top_bounces[at_idx]:
                continue
            if native_record.num_bottom_bounces[candidate] != at_record.num_bottom_bounces[at_idx]:
                continue
            delay_abs = float(abs(native_delays[candidate] - at_delay))
            if delay_abs > delay_tolerance:
                continue
            angle_penalty = abs(float(native_record.src_decl_angles[candidate] - at_record.src_decl_angles[at_idx]))
            candidates.append((delay_abs, angle_penalty, candidate))
        if not candidates:
            continue
        _, _, native_idx = min(candidates)

        delay_error = float(native_delays[native_idx] - at_delay)

        used_native.add(native_idx)

        at_amp = abs(at_record.amplitudes[at_idx])
        native_amp = abs(native_record.amplitudes[native_idx])
        amp_rel_error = float("nan")
        if at_amp > np.finfo(float).eps:
            amp_rel_error = float(abs(native_amp - at_amp) / at_amp)
        phase_error = float(np.angle(native_record.amplitudes[native_idx] * np.conj(at_record.amplitudes[at_idx])))
        launch_angle_error = float(native_record.src_decl_angles[native_idx] - at_record.src_decl_angles[at_idx])
        receiver_angle_error = float(native_record.rcvr_decl_angles[native_idx] - at_record.rcvr_decl_angles[at_idx])
        top_bounce_match = bool(native_record.num_top_bounces[native_idx] == at_record.num_top_bounces[at_idx])
        bottom_bounce_match = bool(
            native_record.num_bottom_bounces[native_idx] == at_record.num_bottom_bounces[at_idx]
        )
        matches.append(
            _ArrivalMatch(
                receiver_index=receiver_index,
                at_index=int(at_idx),
                native_index=int(native_idx),
                at_delay=float(at_delay),
                native_delay=float(native_delays[native_idx]),
                delay_error=delay_error,
                at_amplitude=float(at_amp),
                native_amplitude=float(native_amp),
                amplitude_relative_error=amp_rel_error,
                at_phase_rad=float(np.angle(at_record.amplitudes[at_idx])),
                native_phase_rad=float(np.angle(native_record.amplitudes[native_idx])),
                phase_error_rad=phase_error,
                at_launch_angle=float(at_record.src_decl_angles[at_idx]),
                native_launch_angle=float(native_record.src_decl_angles[native_idx]),
                launch_angle_error=launch_angle_error,
                at_receiver_angle=float(at_record.rcvr_decl_angles[at_idx]),
                native_receiver_angle=float(native_record.rcvr_decl_angles[native_idx]),
                receiver_angle_error=receiver_angle_error,
                at_top_bounces=int(at_record.num_top_bounces[at_idx]),
                native_top_bounces=int(native_record.num_top_bounces[native_idx]),
                top_bounce_match=top_bounce_match,
                at_bottom_bounces=int(at_record.num_bottom_bounces[at_idx]),
                native_bottom_bounces=int(native_record.num_bottom_bounces[native_idx]),
                bottom_bounce_match=bottom_bounce_match,
            )
        )

    return matches


def compare_arrival_data(
    at_data: ArrivalData,
    native_data: ArrivalData,
    *,
    delay_tolerance: float = 0.02,
) -> ArrivalParityMetrics:
    """Compare AT/native arrivals using greedy nearest-delay matching."""

    if delay_tolerance < 0.0:
        raise ValueError("delay_tolerance must be non-negative")
    if at_data.is_3d != native_data.is_3d:
        raise ValueError("AT and native arrivals must have the same dimensionality")
    if at_data.arrivals.shape != native_data.arrivals.shape:
        raise ValueError("AT and native arrivals must share the same receiver/source grid")

    at_count = 0
    native_count = 0
    delay_errors: list[float] = []
    amp_rel_errors: list[float] = []
    phase_errors: list[float] = []
    launch_angle_errors: list[float] = []
    receiver_angle_errors: list[float] = []
    bounce_match_count = 0
    top_bounce_match_count = 0
    bottom_bounce_match_count = 0

    for receiver_index, at_record in np.ndenumerate(at_data.arrivals):
        native_record = native_data.arrivals[receiver_index]
        at_count += at_record.count
        native_count += native_record.count
        matches = _match_record_by_delay(
            at_record,
            native_record,
            delay_tolerance=delay_tolerance,
            receiver_index=receiver_index,
        )
        delay_errors.extend(match.delay_error for match in matches)
        launch_angle_errors.extend(match.launch_angle_error for match in matches)
        receiver_angle_errors.extend(match.receiver_angle_error for match in matches)
        phase_errors.extend(match.phase_error_rad for match in matches)
        top_bounce_match_count += sum(1 for match in matches if match.top_bounce_match)
        bottom_bounce_match_count += sum(1 for match in matches if match.bottom_bounce_match)
        bounce_match_count += sum(1 for match in matches if match.top_bounce_match and match.bottom_bounce_match)
        amp_rel_errors.extend(
            match.amplitude_relative_error
            for match in matches
            if np.isfinite(match.amplitude_relative_error)
        )

    if delay_errors:
        delay_abs = np.abs(np.array(delay_errors, dtype=np.float64))
        delay_mae = float(np.mean(delay_abs))
        delay_p90_abs = float(np.percentile(delay_abs, 90.0))
        delay_max_abs = float(np.max(delay_abs))
    else:
        delay_mae = float("nan")
        delay_p90_abs = float("nan")
        delay_max_abs = float("nan")

    if amp_rel_errors:
        amp_rel_array = np.array(amp_rel_errors, dtype=np.float64)
        amp_median = float(np.median(amp_rel_array))
        amp_p90 = float(np.percentile(amp_rel_array, 90.0))
    else:
        amp_median = float("nan")
        amp_p90 = float("nan")
    if phase_errors:
        phase_abs = np.abs(np.array(phase_errors, dtype=np.float64))
        phase_mae = float(np.mean(phase_abs))
        phase_max_abs = float(np.max(phase_abs))
    else:
        phase_mae = float("nan")
        phase_max_abs = float("nan")
    launch_angle_mae = (
        float(np.mean(np.abs(np.array(launch_angle_errors, dtype=np.float64))))
        if launch_angle_errors
        else float("nan")
    )
    receiver_angle_mae = (
        float(np.mean(np.abs(np.array(receiver_angle_errors, dtype=np.float64))))
        if receiver_angle_errors
        else float("nan")
    )
    matched_count = len(delay_errors)
    unmatched_at_count = max(at_count - matched_count, 0)
    unmatched_native_count = max(native_count - matched_count, 0)
    match_fraction = matched_count / at_count if at_count else float("nan")
    bounce_match_fraction = bounce_match_count / matched_count if matched_count else float("nan")
    top_bounce_match_fraction = top_bounce_match_count / matched_count if matched_count else float("nan")
    bottom_bounce_match_fraction = bottom_bounce_match_count / matched_count if matched_count else float("nan")

    return ArrivalParityMetrics(
        at_count=at_count,
        native_count=native_count,
        matched_count=matched_count,
        unmatched_at_count=unmatched_at_count,
        unmatched_native_count=unmatched_native_count,
        match_fraction=match_fraction,
        delay_mae=delay_mae,
        delay_p90_abs=delay_p90_abs,
        delay_max_abs=delay_max_abs,
        amplitude_median_relative_error=amp_median,
        amplitude_p90_relative_error=amp_p90,
        phase_mae_rad=phase_mae,
        phase_max_abs_rad=phase_max_abs,
        launch_angle_mae=launch_angle_mae,
        receiver_angle_mae=receiver_angle_mae,
        bounce_match_count=bounce_match_count,
        bounce_match_fraction=bounce_match_fraction,
        top_bounce_match_count=top_bounce_match_count,
        top_bounce_match_fraction=top_bounce_match_fraction,
        bottom_bounce_match_count=bottom_bounce_match_count,
        bottom_bounce_match_fraction=bottom_bounce_match_fraction,
    )


def plot_arrival_parity(
    at_data: ArrivalData,
    native_data: ArrivalData,
    output_path: str | Path,
    *,
    metrics: ArrivalParityMetrics | None = None,
    delay_tolerance: float = 0.02,
) -> Path:
    """Plot AT/native arrival delay and amplitude comparison."""

    import matplotlib.pyplot as plt

    if delay_tolerance < 0.0:
        raise ValueError("delay_tolerance must be non-negative")
    if metrics is None:
        metrics = compare_arrival_data(at_data, native_data, delay_tolerance=delay_tolerance)

    at_delays = []
    at_amps = []
    native_delays = []
    native_amps = []
    matches: list[_ArrivalMatch] = []

    for receiver_index, at_record in np.ndenumerate(at_data.arrivals):
        native_record = native_data.arrivals[receiver_index]
        at_delays.extend(float(v) for v in np.real(at_record.delays))
        at_amps.extend(float(abs(v)) for v in at_record.amplitudes)
        native_delays.extend(float(v) for v in np.real(native_record.delays))
        native_amps.extend(float(abs(v)) for v in native_record.amplitudes)

        matches.extend(
            _match_record_by_delay(
                at_record,
                native_record,
                delay_tolerance=delay_tolerance,
                receiver_index=receiver_index,
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

    ax = axes[0]
    ax.scatter(at_delays, at_amps, marker="o", label="AT Bellhop", color="#1f77b4")
    ax.scatter(native_delays, native_amps, marker="x", label="pybellhop native", color="#d62728")
    for match in matches:
        y0 = max(min(at_amps + native_amps), 1.0e-12) if at_amps or native_amps else 1.0e-12
        ax.plot([match.at_delay, match.native_delay], [y0, y0], color="#666666", linewidth=0.8, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Arrival amplitude")
    ax.set_title("AT Bellhop vs pybellhop native arrivals")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()

    ax = axes[1]
    if matches:
        errors_ms = np.array([match.delay_error for match in matches], dtype=np.float64) * 1000.0
        ax.bar(np.arange(len(errors_ms)), errors_ms, color="#2ca02c")
    else:
        errors_ms = np.array([], dtype=np.float64)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Matched arrival index")
    ax.set_ylabel("Native - AT delay (ms)")
    ax.set_title("Nearest-delay matched arrival errors")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)

    summary = (
        f"AT count: {metrics.at_count}\n"
        f"Native count: {metrics.native_count}\n"
        f"Matched: {metrics.matched_count} ({metrics.match_fraction:.3f})\n"
        f"Delay MAE: {metrics.delay_mae * 1000.0:.3f} ms\n"
        f"Delay P90: {metrics.delay_p90_abs * 1000.0:.3f} ms\n"
        f"Delay max: {metrics.delay_max_abs * 1000.0:.3f} ms\n"
        f"Amp median rel err: {metrics.amplitude_median_relative_error:.3f}\n"
        f"Amp P90 rel err: {metrics.amplitude_p90_relative_error:.3f}\n"
        f"Phase MAE: {metrics.phase_mae_rad:.3f} rad\n"
        f"Launch angle MAE: {metrics.launch_angle_mae:.3f} deg\n"
        f"Rcvr angle MAE: {metrics.receiver_angle_mae:.3f} deg\n"
        f"Bounce match: {metrics.bounce_match_count}/{metrics.matched_count}"
    )
    fig.text(
        0.985,
        0.5,
        summary,
        ha="right",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def arrival_match_table(
    at_data: ArrivalData,
    native_data: ArrivalData,
    *,
    delay_tolerance: float = 0.02,
) -> list[dict]:
    """Return per-arrival match diagnostics for AT/native arrival comparisons."""
    if delay_tolerance < 0.0:
        raise ValueError("delay_tolerance must be non-negative")
    if at_data.is_3d != native_data.is_3d:
        raise ValueError("AT and native arrivals must have the same dimensionality")
    if at_data.arrivals.shape != native_data.arrivals.shape:
        raise ValueError("AT and native arrivals must share the same receiver/source grid")

    rows = []
    for receiver_index, at_record in np.ndenumerate(at_data.arrivals):
        native_record = native_data.arrivals[receiver_index]
        matches = _match_record_by_delay(
            at_record,
            native_record,
            delay_tolerance=delay_tolerance,
            receiver_index=receiver_index,
        )
        for match in matches:
            rows.append(
                {
                    "receiver_index": match.receiver_index,
                    "at_index": match.at_index,
                    "native_index": match.native_index,
                    "at_delay": match.at_delay,
                    "native_delay": match.native_delay,
                    "delay_error": match.delay_error,
                    "at_amplitude": match.at_amplitude,
                    "native_amplitude": match.native_amplitude,
                    "amplitude_relative_error": match.amplitude_relative_error,
                    "at_phase_rad": match.at_phase_rad,
                    "native_phase_rad": match.native_phase_rad,
                    "phase_error_rad": match.phase_error_rad,
                    "at_launch_angle": match.at_launch_angle,
                    "native_launch_angle": match.native_launch_angle,
                    "launch_angle_error": match.launch_angle_error,
                    "at_receiver_angle": match.at_receiver_angle,
                    "native_receiver_angle": match.native_receiver_angle,
                    "receiver_angle_error": match.receiver_angle_error,
                    "at_top_bounces": match.at_top_bounces,
                    "native_top_bounces": match.native_top_bounces,
                    "top_bounce_match": match.top_bounce_match,
                    "at_bottom_bounces": match.at_bottom_bounces,
                    "native_bottom_bounces": match.native_bottom_bounces,
                    "bottom_bounce_match": match.bottom_bounce_match,
                }
            )
    return rows


__all__ = [
    "ArrivalParityMetrics",
    "arrival_benchmark_row",
    "arrival_match_table",
    "compare_arrival_data",
    "compute_native_arrival_data",
    "plot_arrival_parity",
    "run_at_arrival_data",
]
