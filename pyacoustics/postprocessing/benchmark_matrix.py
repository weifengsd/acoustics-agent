"""Canonical benchmark rows for post-processing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import hilbert

from pyacoustics.postprocessing.delay_and_sum import (
    ArrivalData,
    ArrivalRecord,
    DelayAndSumResult,
    _dopplerize,
    _modified_chirp_z_unit_circle,
    delay_and_sum,
    matched_filter_timeseries,
    save_matched_filter_matlab,
    save_matlab_timeseries,
)
from pyacoustics.postprocessing.external import (
    at_delayandsum_external_benchmark_row,
    at_matched_filter_external_benchmark_row,
    at_munkts_stack_external_benchmark_row,
    postprocessing_external_failure_row,
)
from pyacoustics.postprocessing.parity import (
    compare_at_rts_mat_payload,
    compare_matched_filter_mat_payload,
    compare_rts_mat_payload,
    compare_rts_timeseries_files,
    compare_timeseries,
    mat_payload_benchmark_row,
    read_rts_timeseries,
    timeseries_benchmark_row,
)
from pyacoustics.solvers.benchmark_thresholds import BenchmarkRow, BenchmarkThresholds, evaluate_benchmark_rows


@dataclass(frozen=True)
class PostprocessingBenchmarkMatrix:
    """Post-processing benchmark matrix rows."""

    rows: tuple[BenchmarkRow, ...]


def postprocessing_smoke_thresholds() -> BenchmarkThresholds:
    """Return smoke thresholds for canonical post-processing payload parity."""

    return BenchmarkThresholds(
        max_values={
            "timeseries_max_abs": 0.0,
            "timeseries_relative_l2_error": 0.0,
        },
        min_values={
            "sample_count": 1.0,
            "mat_payload_variable_count": 1.0,
        },
        ignore_missing=True,
    )


def _canonical_delay_result() -> DelayAndSumResult:
    timeseries = np.zeros((8, 2, 2), dtype=np.float64)
    timeseries[:, 0, 0] = [0.0, 1.0, 0.5, 0.0, 0.0, -0.25, 0.0, 0.0]
    timeseries[:, 1, 0] = [0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    timeseries[:, 0, 1] = [0.0, 0.0, 0.0, 1.5, 0.5, 0.0, 0.0, 0.0]
    timeseries[:, 1, 1] = [0.0, -0.25, 0.0, 0.0, 0.75, 0.25, 0.0, 0.0]
    time_windows = np.vstack(
        [
            np.arange(8, dtype=np.float64) / 1000.0,
            0.5 + np.arange(8, dtype=np.float64) / 1000.0,
        ]
    )
    return DelayAndSumResult(
        timeseries=timeseries,
        time_windows=time_windows,
        sample_rate=1000.0,
        receiver_depths=np.array([50.0, 100.0], dtype=np.float64),
        receiver_ranges=np.array([0.0, 500.0], dtype=np.float64),
        source_index=0,
        source_depths=np.array([25.0], dtype=np.float64),
    )


def _rts_payload_row(work_dir: Path) -> BenchmarkRow:
    result = _canonical_delay_result()
    reference_path = work_dir / "reference.rts.mat"
    candidate_path = work_dir / "candidate.rts.mat"
    save_matlab_timeseries(result, reference_path, plot_title="canonical_postprocessing_rts")
    save_matlab_timeseries(result, candidate_path, plot_title="canonical_postprocessing_rts")
    metrics = compare_rts_mat_payload(reference_path, candidate_path, exact_plot_title=True)
    return mat_payload_benchmark_row("postprocessing-rts-payload", metrics)


def _at_rts_payload_row(work_dir: Path) -> BenchmarkRow:
    from scipy.io import savemat

    reference_path = work_dir / "reference.at.rts.mat"
    candidate_path = work_dir / "candidate.at.rts.mat"
    payload = {
        "PlotTitle": "canonical_at_delayandsum_rts",
        "Pos": {
            "s": {"z": np.array([25.0], dtype=np.float64)},
            "r": {
                "r": np.array([500.0], dtype=np.float64),
                "z": np.array([50.0, 100.0], dtype=np.float64),
            },
            "theta": np.array([], dtype=np.float64),
        },
        "tout": np.arange(6, dtype=np.float64)[None, :] / 1000.0,
        "RTS": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.5],
                [0.25, 0.75],
                [0.0, 0.0],
                [-0.25, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }
    savemat(reference_path, payload)
    savemat(candidate_path, payload)
    metrics = compare_at_rts_mat_payload(reference_path, candidate_path, exact_plot_title=True)
    return mat_payload_benchmark_row("postprocessing-at-delayandsum-rts-payload", metrics)


def _at_rts_timeseries_row(work_dir: Path) -> BenchmarkRow:
    from scipy.io import savemat

    reference_path = work_dir / "reference.read_ts.rts.mat"
    candidate_path = work_dir / "candidate.read_ts.rts.mat"
    payload = {
        "PlotTitle": "canonical_at_read_ts_rts",
        "Pos": {
            "s": {"z": np.array([25.0], dtype=np.float64)},
            "r": {
                "r": np.array([500.0], dtype=np.float64),
                "z": np.array([50.0, 100.0], dtype=np.float64),
            },
            "theta": np.array([], dtype=np.float64),
        },
        "tout": np.arange(6, dtype=np.float64)[None, :] / 1000.0,
        "RTS": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.5],
                [0.25, 0.75],
                [0.0, 0.0],
                [-0.25, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }
    savemat(reference_path, payload)
    savemat(candidate_path, payload)
    metrics = compare_rts_timeseries_files(reference_path, candidate_path)
    return timeseries_benchmark_row(
        "postprocessing-at-read-ts-timeseries",
        metrics,
        kind="postprocessing_rts_timeseries",
    )


def _at_ascii_rts_timeseries_row(work_dir: Path) -> BenchmarkRow:
    reference_path = work_dir / "reference.read_ts.rts"
    candidate_path = work_dir / "candidate.read_ts.rts"
    payload = (
        "canonical_at_read_ts_ascii\n"
        "3\n"
        "50.0 100.0 150.0\n"
        "0.000 0.0 1.0 0.0\n"
        "0.001 0.5 0.0 -0.25\n"
        "0.002 0.0 0.25 0.75\n"
        "0.003 -0.125 0.0 0.0\n"
    )
    reference_path.write_text(payload, encoding="utf-8")
    candidate_path.write_text(payload, encoding="utf-8")
    metrics = compare_rts_timeseries_files(reference_path, candidate_path)
    return timeseries_benchmark_row(
        "postprocessing-at-read-ts-ascii-timeseries",
        metrics,
        kind="postprocessing_rts_ascii_timeseries",
    )


def _at_ascii_free_format_rts_timeseries_row(work_dir: Path) -> BenchmarkRow:
    reference_path = work_dir / "reference.read_ts.free_format.rts"
    candidate_path = work_dir / "candidate.read_ts.free_format.rts"
    payload = (
        "canonical_at_read_ts_ascii_free_format\n"
        "3, ! receiver depth count\n"
        "50.0D0,\n"
        "100.0D0,\n"
        "150.0D0 /\n"
        "0.000D0, 0.0D0, 1.0D0, 0.0D0,\n"
        "0.001D0, 0.5D0, 0.0D0, -0.25D0,\n"
        "0.002D0, 0.0D0, 0.25D0, 0.75D0,\n"
        "0.003D0, -0.125D0, 0.0D0, 0.0D0 /\n"
    )
    reference_path.write_text(payload, encoding="utf-8")
    candidate_path.write_text(payload, encoding="utf-8")
    parsed = read_rts_timeseries(reference_path)
    metrics = compare_rts_timeseries_files(reference_path, candidate_path)
    row = timeseries_benchmark_row(
        "postprocessing-at-read-ts-ascii-free-format-timeseries",
        metrics,
        kind="postprocessing_rts_ascii_timeseries",
    )
    row.update(
        {
            "postprocessing_time_sample_count": int(parsed.tout.size),
            "postprocessing_receiver_depth_count": int(parsed.rts.shape[1]) if parsed.rts.ndim >= 2 else 1,
            "postprocessing_ascii_free_format": True,
        }
    )
    return row


def _at_real_munkts_rts_timeseries_row(fixture_path: Path) -> BenchmarkRow:
    payload = read_rts_timeseries(fixture_path)
    row = timeseries_benchmark_row(
        "postprocessing-at-munkts-real-rts-timeseries",
        compare_rts_timeseries_files(fixture_path, fixture_path),
        kind="postprocessing_at_real_rts_timeseries",
    )
    row.update(
        {
            "postprocessing_fixture": str(fixture_path),
            "postprocessing_plot_title": payload.plot_title.rstrip(),
            "postprocessing_time_sample_count": int(payload.tout.size),
            "postprocessing_receiver_depth_count": int(payload.rts.shape[1]) if payload.rts.ndim >= 2 else 1,
        }
    )
    return row


def _delay_and_sum_grid_row() -> BenchmarkRow:
    ranges = np.array([0.0, 1000.0, 2000.0], dtype=np.float64)
    depths = np.array([40.0, 80.0, 120.0], dtype=np.float64)
    arrivals = np.empty((ranges.size, depths.size, 1), dtype=object)
    expected = np.zeros((30, depths.size, ranges.size), dtype=np.float64)
    sample_indices = np.array([2, 7, 11], dtype=np.int64)
    for ir, receiver_range in enumerate(ranges):
        base_time = receiver_range / 1000.0
        for iz, _depth in enumerate(depths):
            amplitudes = np.array(
                [
                    1.0 + 0.25 * ir,
                    0.5 + 0.1 * iz,
                    -0.25 * (ir + iz + 1),
                ],
                dtype=np.float64,
            )
            arrivals[ir, iz, 0] = ArrivalRecord(
                amplitudes=amplitudes.astype(np.complex128),
                delays=(base_time + sample_indices / 1000.0).astype(np.complex128),
                src_decl_angles=np.zeros(sample_indices.size),
                rcvr_decl_angles=np.zeros(sample_indices.size),
                num_top_bounces=np.array([0, 1, 0], dtype=np.int16),
                num_bottom_bounces=np.array([0, 0, 1], dtype=np.int16),
            )
            expected[sample_indices, iz, ir] = amplitudes
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([25.0], dtype=np.float64),
        receiver_depths=depths,
        receiver_ranges=ranges,
        arrivals=arrivals,
    )
    result = delay_and_sum(
        data,
        [1.0, 0.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.03,
        reduction_velocity=1000.0,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )
    row = timeseries_benchmark_row(
        "postprocessing-delayandsum-grid-multi-arrival",
        compare_timeseries(expected, result.timeseries),
        kind="postprocessing_delayandsum_grid",
    )
    row.update(
        {
            "postprocessing_receiver_range_count": int(ranges.size),
            "postprocessing_receiver_depth_count": int(depths.size),
            "postprocessing_arrival_count": int(sample_indices.size * ranges.size * depths.size),
        }
    )
    return row


def _matched_filter_payload_row(work_dir: Path) -> BenchmarkRow:
    result = _canonical_delay_result()
    matched = matched_filter_timeseries(result, [1.0, 0.5, -0.25], normalize_replica=False)
    reference_path = work_dir / "reference.matched_filter.mat"
    candidate_path = work_dir / "candidate.matched_filter.mat"
    save_matched_filter_matlab(matched, reference_path, plot_title="canonical_postprocessing_matched_filter")
    save_matched_filter_matlab(matched, candidate_path, plot_title="canonical_postprocessing_matched_filter")
    metrics = compare_matched_filter_mat_payload(reference_path, candidate_path, exact_plot_title=True)
    return mat_payload_benchmark_row("postprocessing-matched-filter-payload", metrics)


def _matched_filter_workflow_row() -> BenchmarkRow:
    ranges = np.array([0.0, 1000.0], dtype=np.float64)
    depths = np.array([50.0, 100.0], dtype=np.float64)
    arrivals = np.empty((ranges.size, depths.size, 1), dtype=object)
    expected_lags = np.zeros((depths.size, ranges.size), dtype=np.float64)
    expected_peaks = np.zeros((depths.size, ranges.size), dtype=np.float64)
    waveform = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    waveform_energy = float(np.dot(waveform, waveform))
    for ir, receiver_range in enumerate(ranges):
        base_time = receiver_range / 1000.0
        for iz, _depth in enumerate(depths):
            sample_index = 3 + ir + iz
            amplitude = 1.0 + 0.5 * ir + 0.25 * iz
            arrivals[ir, iz, 0] = ArrivalRecord(
                amplitudes=np.array([amplitude + 0.0j], dtype=np.complex128),
                delays=np.array([base_time + sample_index / 1000.0], dtype=np.complex128),
                src_decl_angles=np.zeros(1),
                rcvr_decl_angles=np.zeros(1),
                num_top_bounces=np.zeros(1, dtype=np.int16),
                num_bottom_bounces=np.zeros(1, dtype=np.int16),
            )
            expected_lags[iz, ir] = sample_index
            expected_peaks[iz, ir] = amplitude * waveform_energy
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([25.0], dtype=np.float64),
        receiver_depths=depths,
        receiver_ranges=ranges,
        arrivals=arrivals,
    )
    result = delay_and_sum(
        data,
        waveform,
        sample_rate=1000.0,
        window_duration=0.02,
        reduction_velocity=1000.0,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )
    matched = matched_filter_timeseries(result, waveform, normalize_replica=False)
    observed_lags = np.zeros_like(expected_lags)
    observed_peaks = np.zeros_like(expected_peaks)
    for iz in range(depths.size):
        for ir in range(ranges.size):
            trace = matched.timeseries[:, iz, ir]
            peak_index = int(np.argmax(trace))
            observed_lags[iz, ir] = matched.lags[peak_index]
            observed_peaks[iz, ir] = trace[peak_index]
    metrics = compare_timeseries(
        np.stack((expected_lags, expected_peaks)),
        np.stack((observed_lags, observed_peaks)),
    )
    row = timeseries_benchmark_row(
        "postprocessing-matched-filter-workflow",
        metrics,
        kind="postprocessing_matched_filter_workflow",
    )
    row.update(
        {
            "postprocessing_matched_filter_trace_count": int(ranges.size * depths.size),
            "postprocessing_matched_filter_peak_lag_max_abs": float(np.max(np.abs(observed_lags - expected_lags))),
            "postprocessing_matched_filter_peak_value_max_abs": float(np.max(np.abs(observed_peaks - expected_peaks))),
        }
    )
    return row


def _doppler_transform_row() -> BenchmarkRow:
    waveform = np.array([0.0, 1.0, -0.5, 0.25, 0.125, -0.25, 0.0, 0.0], dtype=np.float64)
    chirp_signal = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float64)
    chirp_reference = np.fft.fft(chirp_signal)
    chirp_candidate = _modified_chirp_z_unit_circle(
        chirp_signal,
        chirp_signal.size,
        2.0 * np.pi / chirp_signal.size,
    )
    alpha_one_reference = waveform
    alpha_one_candidate = _dopplerize(waveform, 1.0)
    chirp_error = np.concatenate(
        [
            np.real(chirp_candidate - chirp_reference),
            np.imag(chirp_candidate - chirp_reference),
        ]
    )
    alpha_lt_one = _dopplerize(waveform, 0.98)
    alpha_gt_one = _dopplerize(waveform, 1.02)
    row = timeseries_benchmark_row(
        "postprocessing-doppler-transform",
        compare_timeseries(alpha_one_reference, alpha_one_candidate),
        kind="postprocessing_doppler_transform",
    )
    row.update(
        {
            "doppler_alpha_lt_one_finite": bool(np.isfinite(alpha_lt_one).all()),
            "doppler_alpha_gt_one_finite": bool(np.isfinite(alpha_gt_one).all()),
            "doppler_alpha_lt_one_peak": float(np.max(np.abs(alpha_lt_one))),
            "doppler_alpha_gt_one_peak": float(np.max(np.abs(alpha_gt_one))),
            "doppler_reference_sample_count": int(alpha_one_reference.size),
            "doppler_chirp_fft_max_abs": float(np.max(np.abs(chirp_error))),
            "doppler_chirp_fft_rmse": float(np.sqrt(np.mean(chirp_error * chirp_error))),
        }
    )
    return row


def _doppler_delay_and_sum_workflow_row() -> BenchmarkRow:
    sample_rate = 1000.0
    waveform = np.array([0.0, 1.0, 0.5, -0.25, 0.0], dtype=np.float64)
    ranges = np.array([0.0, 1000.0], dtype=np.float64)
    depths = np.array([40.0, 80.0], dtype=np.float64)
    bearings = np.array([0.0, 180.0], dtype=np.float64)
    bins = np.array([-20.0, 0.0, 20.0], dtype=np.float64)
    receiver_velocity = np.array([20.0, 0.0, 0.0], dtype=np.float64)
    arrivals = np.empty((ranges.size, depths.size, bearings.size, 1), dtype=object)
    expected = np.zeros((30, depths.size, ranges.size, bearings.size), dtype=np.float64)
    doppler_bank = {
        float(bin_velocity): hilbert(_dopplerize(waveform, 1.0 - bin_velocity / 1500.0))
        for bin_velocity in bins
    }

    for ir, receiver_range in enumerate(ranges):
        base_time = receiver_range / 1000.0
        for iz, _depth in enumerate(depths):
            for itheta, bearing in enumerate(bearings):
                sample_index = 3 + ir + iz + itheta
                amplitude = 0.75 + 0.25 * ir + 0.10 * iz + 0.05 * itheta
                projection = 20.0 if bearing == 0.0 else -20.0
                source = doppler_bank[projection]
                arrivals[ir, iz, itheta, 0] = ArrivalRecord(
                    amplitudes=np.array([amplitude + 0.0j], dtype=np.complex128),
                    delays=np.array([base_time + sample_index / sample_rate], dtype=np.complex128),
                    src_decl_angles=np.zeros(1),
                    rcvr_decl_angles=np.zeros(1),
                    src_azim_angles=np.zeros(1),
                    rcvr_azim_angles=np.array([bearing], dtype=np.float64),
                    num_top_bounces=np.zeros(1, dtype=np.int16),
                    num_bottom_bounces=np.zeros(1, dtype=np.int16),
                )
                end = min(expected.shape[0], sample_index + source.size)
                expected[sample_index:end, iz, ir, itheta] = amplitude * np.real(source[: end - sample_index])

    data = ArrivalData(
        is_3d=True,
        frequency=100.0,
        source_depths=np.array([25.0], dtype=np.float64),
        receiver_depths=depths,
        receiver_ranges=ranges,
        receiver_bearings=bearings,
        source_x=np.array([0.0], dtype=np.float64),
        source_y=np.array([0.0], dtype=np.float64),
        arrivals=arrivals,
    )
    result = delay_and_sum(
        data,
        waveform,
        sample_rate=sample_rate,
        window_duration=0.03,
        reduction_velocity=1000.0,
        start_time_margin=0.0,
        receiver_velocity=receiver_velocity,
        doppler_reference_speed=1500.0,
        doppler_velocity_bins=bins,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )
    row = timeseries_benchmark_row(
        "postprocessing-doppler-delayandsum-workflow",
        compare_timeseries(expected, result.timeseries),
        kind="postprocessing_doppler_workflow",
    )
    row.update(
        {
            "postprocessing_doppler_trace_count": int(ranges.size * depths.size * bearings.size),
            "postprocessing_doppler_velocity_bin_count": int(bins.size),
            "postprocessing_receiver_range_count": int(ranges.size),
            "postprocessing_receiver_depth_count": int(depths.size),
            "postprocessing_receiver_bearing_count": int(bearings.size),
            "postprocessing_arrival_count": int(ranges.size * depths.size * bearings.size),
        }
    )
    return row


def _canonical_report_delay_result() -> DelayAndSumResult:
    base = _canonical_delay_result()
    receiver_ranges = np.array([500.0, 1000.0], dtype=np.float64)
    sample_offsets = np.arange(base.timeseries.shape[0], dtype=np.float64) / base.sample_rate
    time_windows = np.vstack([receiver_range / 1500.0 + sample_offsets for receiver_range in receiver_ranges])
    return DelayAndSumResult(
        timeseries=base.timeseries,
        time_windows=time_windows,
        sample_rate=base.sample_rate,
        receiver_depths=base.receiver_depths,
        receiver_ranges=receiver_ranges,
        source_index=base.source_index,
        receiver_bearings=base.receiver_bearings,
        source_depths=base.source_depths,
        source_x_index=base.source_x_index,
        source_y_index=base.source_y_index,
        source_x=base.source_x,
        source_y=base.source_y,
    )

def write_postprocessing_report_plots(output_dir: str | Path) -> tuple[Path, ...]:
    """Write canonical post-processing simulation plots for benchmark reports."""

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    import os
    mpl_config_dir = figures_dir / ".matplotlib"
    font_cache_dir = figures_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    font_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    os.environ["XDG_CACHE_HOME"] = str(font_cache_dir)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    delay_result = _canonical_report_delay_result()
    delay_time = np.arange(delay_result.timeseries.shape[0], dtype=np.float64) / delay_result.sample_rate
    delay_path = figures_dir / "postprocessing_delay_and_sum_timeseries.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for ir, receiver_range in enumerate(delay_result.receiver_ranges):
        for iz, receiver_depth in enumerate(delay_result.receiver_depths):
            label = f"r={receiver_range:g}m z={receiver_depth:g}m"
            ax.plot(delay_result.time_windows[ir], delay_result.timeseries[:, iz, ir], label=label)
    ax.set_title("Canonical delay-and-sum time series")
    ax.set_xlabel("Arrival time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(delay_path, dpi=150)
    plt.close(fig)

    matched = matched_filter_timeseries(delay_result, [1.0, 0.5, -0.25], normalize_replica=False)
    matched_path = figures_dir / "postprocessing_matched_filter_timeseries.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for ir, receiver_range in enumerate(matched.receiver_ranges):
        for iz, receiver_depth in enumerate(matched.receiver_depths):
            label = f"r={receiver_range:g}m z={receiver_depth:g}m"
            matched_time = delay_result.time_windows[ir, 0] + matched.lags / matched.sample_rate
            ax.plot(matched_time, matched.timeseries[:, iz, ir], label=label)
    ax.set_title("Canonical matched-filter output")
    ax.set_xlabel("Arrival time (s)")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(matched_path, dpi=150)
    plt.close(fig)

    import shutil
    shutil.rmtree(mpl_config_dir, ignore_errors=True)
    shutil.rmtree(font_cache_dir, ignore_errors=True)

    return (delay_path, matched_path)


def run_postprocessing_benchmark_matrix(
    *,
    thresholds: BenchmarkThresholds | None = None,
    at_munkts_rts_fixture: str | Path | None = None,
    at_munkts_source_dir: str | Path | None = None,
    at_delayandsum_reference_path: str | Path | None = None,
    at_delayandsum_work_dir: str | Path | None = None,
    at_matched_filter_reference_path: str | Path | None = None,
    at_matched_filter_work_dir: str | Path | None = None,
    at_matlab_dir: str | Path | None = None,
    at_bin_dir: str | Path | None = None,
    matlab_runtime: str | None = None,
    at_delayandsum_prelude_commands: tuple[str, ...] | list[str] = (),
    at_delayandsum_output_filename: str = "MunkB.rts.mat",
    at_delayandsum_timeout_s: float = 300.0,
    at_matched_filter_prelude_commands: tuple[str, ...] | list[str] = (),
    at_matched_filter_script_name: str = "matchedfilter",
    at_matched_filter_output_filename: str = "matched_filter.mat",
    at_matched_filter_timeout_s: float = 300.0,
) -> PostprocessingBenchmarkMatrix:
    """Run canonical post-processing payload parity rows."""

    with TemporaryDirectory(prefix="pyacoustics-postprocessing-benchmark-") as temp_dir:
        work_dir = Path(temp_dir)
        rows: list[BenchmarkRow] = [
            _rts_payload_row(work_dir),
            _at_rts_payload_row(work_dir),
            _at_rts_timeseries_row(work_dir),
            _at_ascii_rts_timeseries_row(work_dir),
            _at_ascii_free_format_rts_timeseries_row(work_dir),
            _delay_and_sum_grid_row(),
            _matched_filter_payload_row(work_dir),
            _matched_filter_workflow_row(),
            _doppler_transform_row(),
            _doppler_delay_and_sum_workflow_row(),
        ]
        fixture_path = Path(at_munkts_rts_fixture) if at_munkts_rts_fixture is not None else None
        if fixture_path is not None and fixture_path.exists():
            rows.append(_at_real_munkts_rts_timeseries_row(fixture_path))
        if at_munkts_source_dir is not None and at_delayandsum_reference_path is not None:
            raise ValueError("Use either at_munkts_source_dir or at_delayandsum_reference_path, not both")
        if at_munkts_source_dir is not None:
            if at_delayandsum_work_dir is None or at_matlab_dir is None:
                raise ValueError("AT MunkTS external parity requires source_dir, work_dir, and at_matlab_dir")
            commands = tuple(at_delayandsum_prelude_commands or ("scooter MunkS", "stack"))
            try:
                rows.append(
                    at_munkts_stack_external_benchmark_row(
                        source_dir=at_munkts_source_dir,
                        work_dir=at_delayandsum_work_dir,
                        at_matlab_dir=at_matlab_dir,
                        at_bin_dir=at_bin_dir,
                        runtime=matlab_runtime,
                        commands=commands,
                        output_filename="MunkS.rts.mat",
                        timeout_s=at_delayandsum_timeout_s,
                    )
                )
            except (FileNotFoundError, FileExistsError, RuntimeError, TimeoutError, OSError, ValueError, KeyError) as exc:
                rows.append(
                    postprocessing_external_failure_row(
                        "postprocessing-at-munkts-stack-external-rts-payload",
                        "postprocessing_at_stack_external",
                        exc,
                        runtime=matlab_runtime,
                        output_path=Path(at_delayandsum_work_dir) / "MunkS.rts.mat",
                        prelude_commands=commands,
                        fixture=Path(at_munkts_source_dir) / "MunkS.rts.mat",
                    )
                )
        if at_munkts_source_dir is None and (
            at_delayandsum_reference_path is not None
            or at_delayandsum_work_dir is not None
        ):
            if at_delayandsum_reference_path is None or at_delayandsum_work_dir is None or at_matlab_dir is None:
                raise ValueError(
                    "AT delayandsum external parity requires reference_path, work_dir, and at_matlab_dir"
                )
            prelude_commands = tuple(at_delayandsum_prelude_commands)
            try:
                rows.append(
                    at_delayandsum_external_benchmark_row(
                        reference_path=at_delayandsum_reference_path,
                        work_dir=at_delayandsum_work_dir,
                        at_matlab_dir=at_matlab_dir,
                        at_bin_dir=at_bin_dir,
                        runtime=matlab_runtime,
                        prelude_commands=prelude_commands,
                        output_filename=at_delayandsum_output_filename,
                        timeout_s=at_delayandsum_timeout_s,
                    )
                )
            except (FileNotFoundError, FileExistsError, RuntimeError, TimeoutError, OSError, ValueError, KeyError) as exc:
                rows.append(
                    postprocessing_external_failure_row(
                        "postprocessing-at-delayandsum-external-rts-payload",
                        "postprocessing_at_delayandsum_external",
                        exc,
                        runtime=matlab_runtime,
                        output_path=Path(at_delayandsum_work_dir) / at_delayandsum_output_filename,
                        prelude_commands=prelude_commands,
                        fixture=at_delayandsum_reference_path,
                    )
                )
        if at_matched_filter_reference_path is not None or at_matched_filter_work_dir is not None:
            if at_matched_filter_reference_path is None or at_matched_filter_work_dir is None or at_matlab_dir is None:
                raise ValueError(
                    "AT matched-filter external parity requires reference_path, work_dir, and at_matlab_dir"
                )
            matched_filter_prelude_commands = tuple(at_matched_filter_prelude_commands)
            try:
                rows.append(
                    at_matched_filter_external_benchmark_row(
                        reference_path=at_matched_filter_reference_path,
                        work_dir=at_matched_filter_work_dir,
                        at_matlab_dir=at_matlab_dir,
                        at_bin_dir=at_bin_dir,
                        runtime=matlab_runtime,
                        prelude_commands=matched_filter_prelude_commands,
                        script_name=at_matched_filter_script_name,
                        output_filename=at_matched_filter_output_filename,
                        timeout_s=at_matched_filter_timeout_s,
                    )
                )
            except (FileNotFoundError, FileExistsError, RuntimeError, TimeoutError, OSError, ValueError, KeyError) as exc:
                rows.append(
                    postprocessing_external_failure_row(
                        "postprocessing-at-matched-filter-external-payload",
                        "postprocessing_at_matched_filter_external",
                        exc,
                        runtime=matlab_runtime,
                        output_path=Path(at_matched_filter_work_dir) / at_matched_filter_output_filename,
                        prelude_commands=(*matched_filter_prelude_commands, at_matched_filter_script_name),
                        fixture=at_matched_filter_reference_path,
                    )
                )
    matrix_rows = evaluate_benchmark_rows(rows, thresholds) if thresholds is not None else rows
    return PostprocessingBenchmarkMatrix(rows=tuple(matrix_rows))


__all__ = [
    "PostprocessingBenchmarkMatrix",
    "postprocessing_smoke_thresholds",
    "run_postprocessing_benchmark_matrix",
    "write_postprocessing_report_plots",
]
