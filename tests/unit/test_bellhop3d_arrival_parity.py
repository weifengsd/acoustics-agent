from pathlib import Path

import numpy as np
import pytest

from pyacoustics.benchmarks.at_presets import canonical_bellhop3d_arrival_benchmark_cases
from pyacoustics.postprocessing.delay_and_sum import ArrivalData, ArrivalRecord
from pyacoustics.schema import BellhopConfig
from pyacoustics.simulation import Simulation
from pyacoustics.solvers.bellhop3d.arrival_parity import (
    Bellhop3DArrivalParityMetrics,
    bellhop3d_arrival_benchmark_row,
    compare_arrival_data_3d,
    run_bellhop3d_arrival_parity,
)


def _record(
    *,
    delay: float,
    amp: complex = 1.0 + 0.0j,
    src_decl: float = 0.0,
    src_azim: float = 0.0,
    rcvr_decl: float = 0.0,
    rcvr_azim: float = 0.0,
    top: int = 0,
    bottom: int = 0,
) -> ArrivalRecord:
    return ArrivalRecord(
        amplitudes=np.asarray([amp], dtype=np.complex128),
        delays=np.asarray([delay], dtype=np.complex128),
        src_decl_angles=np.asarray([src_decl], dtype=np.float64),
        src_azim_angles=np.asarray([src_azim], dtype=np.float64),
        rcvr_decl_angles=np.asarray([rcvr_decl], dtype=np.float64),
        rcvr_azim_angles=np.asarray([rcvr_azim], dtype=np.float64),
        phase_angles=np.asarray([np.angle(amp)], dtype=np.float64),
        num_top_bounces=np.asarray([top], dtype=np.int16),
        num_bottom_bounces=np.asarray([bottom], dtype=np.int16),
    )


def _arrival_data(record: ArrivalRecord) -> ArrivalData:
    arrivals = np.empty((1, 1, 1, 1), dtype=object)
    arrivals[0, 0, 0, 0] = record
    return _arrival_data_from_array(arrivals)


def _arrival_data_from_array(arrivals: np.ndarray) -> ArrivalData:
    return ArrivalData(
        is_3d=True,
        frequency=100.0,
        source_depths=np.asarray([100.0]),
        receiver_depths=np.asarray([100.0]),
        receiver_ranges=np.asarray([500.0]),
        receiver_bearings=np.asarray([0.0]),
        arrivals=arrivals,
    )


def _at_bin_path() -> Path:
    return Path(__file__).resolve().parents[3] / "at" / "bin"


def test_compare_arrival_data_3d_reports_azimuth_errors_and_row():
    at_data = _arrival_data(
        _record(
            delay=1.0,
            amp=np.exp(1j * 0.1),
            src_decl=2.0,
            src_azim=179.5,
            rcvr_decl=-1.0,
            rcvr_azim=-179.5,
            top=2,
            bottom=1,
        )
    )
    native_data = _arrival_data(
        _record(
            delay=1.001,
            amp=1.1 * np.exp(1j * 0.2),
            src_decl=3.0,
            src_azim=-179.5,
            rcvr_decl=-2.0,
            rcvr_azim=179.5,
            top=2,
            bottom=1,
        )
    )

    metrics = compare_arrival_data_3d(at_data, native_data, delay_tolerance_s=0.01)
    row = bellhop3d_arrival_benchmark_row("synthetic-3d-arrival", metrics, tolerance_m=20.0)

    assert metrics.matched_count == 1
    assert np.isclose(metrics.delay_mae_s, 0.001)
    assert np.isclose(metrics.amplitude_median_relative_error, 0.1)
    assert np.isclose(metrics.phase_mae_rad, 0.1)
    assert np.isclose(metrics.phase_max_abs_rad, 0.1)
    assert np.isclose(metrics.launch_declination_mae_deg, 1.0)
    assert np.isclose(metrics.launch_azimuth_mae_deg, 1.0)
    assert np.isclose(metrics.receiver_declination_mae_deg, 1.0)
    assert np.isclose(metrics.receiver_azimuth_mae_deg, 1.0)
    assert metrics.top_bounce_match_count == 1
    assert metrics.top_bounce_match_fraction == 1.0
    assert metrics.bottom_bounce_match_count == 1
    assert metrics.bottom_bounce_match_fraction == 1.0
    assert row["kind"] == "bellhop3d_arrival"
    assert row["tolerance_m"] == 20.0
    assert np.isclose(row["phase_mae_rad"], 0.1)
    assert np.isclose(row["phase_max_abs_rad"], 0.1)
    assert row["top_bounce_match_count"] == 1
    assert row["top_bounce_match_fraction"] == 1.0
    assert row["bottom_bounce_match_count"] == 1
    assert row["bottom_bounce_match_fraction"] == 1.0


def test_compare_arrival_data_3d_angle_tolerance_rejects_wrong_angle_match():
    at_data = _arrival_data(_record(delay=1.0, src_azim=0.0, rcvr_azim=0.0))
    arrivals = np.empty((1, 1, 1, 1), dtype=object)
    arrivals[0, 0, 0, 0] = ArrivalRecord(
        amplitudes=np.asarray([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
        delays=np.asarray([1.0001, 1.001], dtype=np.complex128),
        src_decl_angles=np.asarray([0.0, 0.0], dtype=np.float64),
        src_azim_angles=np.asarray([10.0, 0.0], dtype=np.float64),
        rcvr_decl_angles=np.asarray([0.0, 0.0], dtype=np.float64),
        rcvr_azim_angles=np.asarray([10.0, 0.0], dtype=np.float64),
        phase_angles=np.asarray([0.0, 0.0], dtype=np.float64),
        num_top_bounces=np.asarray([0, 0], dtype=np.int16),
        num_bottom_bounces=np.asarray([0, 0], dtype=np.int16),
    )
    native_data = _arrival_data_from_array(arrivals)

    loose = compare_arrival_data_3d(at_data, native_data, delay_tolerance_s=0.01)
    gated = compare_arrival_data_3d(at_data, native_data, delay_tolerance_s=0.01, angle_tolerance_deg=0.5)

    assert np.isclose(loose.delay_mae_s, 0.0001)
    assert np.isclose(loose.launch_azimuth_mae_deg, 10.0)
    assert np.isclose(gated.delay_mae_s, 0.001)
    assert gated.launch_azimuth_mae_deg == 0.0
    assert gated.unmatched_native_count == 1


def test_bellhop3d_iso_direct_arrival_parity_smoke():
    at_bin = _at_bin_path()
    if not ((at_bin / "bellhop3d").exists() or (at_bin / "bellhop3d.exe").exists()):
        pytest.skip("AT bellhop3d executable is not available")

    case = canonical_bellhop3d_arrival_benchmark_cases()[0]
    metrics = run_bellhop3d_arrival_parity(
        case.config,
        at_bin_path=at_bin,
        delay_tolerance_s=case.delay_tolerance_s,
        angle_tolerance_deg=case.angle_tolerance_deg,
        tolerance_m=case.tolerance_m,
    )

    assert metrics.at_count == 3
    assert metrics.native_count == 3
    assert metrics.matched_count == 3
    assert metrics.unmatched_at_count == 0
    assert metrics.unmatched_native_count == 0
    assert metrics.match_fraction == 1.0
    assert metrics.delay_mae_s < 1.0e-5
    assert metrics.amplitude_median_relative_error < 0.02
    assert metrics.launch_azimuth_mae_deg < 0.1
    assert metrics.bounce_match_fraction == 1.0


def test_simulation_forwards_bellhop3d_arrival_parity_options(monkeypatch):
    from pyacoustics.solvers.bellhop3d import arrival_parity as arrival_parity_module

    captured = {}

    def fake_run_bellhop3d_arrival_parity(config, **kwargs):
        captured["solver_type"] = config.solver.type
        captured.update(kwargs)
        return Bellhop3DArrivalParityMetrics(
            at_count=1,
            native_count=1,
            matched_count=1,
            unmatched_at_count=0,
            unmatched_native_count=0,
            match_fraction=1.0,
            delay_mae_s=0.0,
            delay_p90_abs_s=0.0,
            delay_max_abs_s=0.0,
            amplitude_median_relative_error=0.0,
            amplitude_p90_relative_error=0.0,
            phase_mae_rad=0.0,
            phase_max_abs_rad=0.0,
            launch_declination_mae_deg=0.0,
            launch_azimuth_mae_deg=0.0,
            receiver_declination_mae_deg=0.0,
            receiver_azimuth_mae_deg=0.0,
            bounce_match_count=1,
            bounce_match_fraction=1.0,
            top_bounce_match_count=1,
            top_bounce_match_fraction=1.0,
            bottom_bounce_match_count=1,
            bottom_bounce_match_fraction=1.0,
        )

    monkeypatch.setattr(arrival_parity_module, "run_bellhop3d_arrival_parity", fake_run_bellhop3d_arrival_parity)

    case = canonical_bellhop3d_arrival_benchmark_cases()[0]
    metrics = Simulation(case.config).compare_bellhop3d_arrivals_with_at(
        max_arrivals=7,
        delay_tolerance=0.01,
        angle_tolerance=0.1,
        tolerance_m=20.0,
        at_bin_path="/opt/at/bin",
        run_type="A",
    )

    assert metrics.matched_count == 1
    assert captured["solver_type"] == "bellhop3d"
    assert captured["max_arrivals"] == 7
    assert captured["delay_tolerance_s"] == 0.01
    assert captured["angle_tolerance_deg"] == 0.1
    assert captured["tolerance_m"] == 20.0
    assert captured["at_bin_path"] == "/opt/at/bin"
    assert captured["run_type"] == "A"


def test_simulation_bellhop3d_arrival_parity_rejects_non_bellhop3d():
    config = canonical_bellhop3d_arrival_benchmark_cases()[0].config
    config.solver = BellhopConfig(type="bellhop", angles=(-1.0, 1.0), num_beams=3)

    with pytest.raises(NotImplementedError, match="Bellhop3D arrival"):
        Simulation(config).compare_bellhop3d_arrivals_with_at()


def test_simulation_exposes_bellhop3d_arrival_parity_smoke():
    at_bin = _at_bin_path()
    if not ((at_bin / "bellhop3d").exists() or (at_bin / "bellhop3d.exe").exists()):
        pytest.skip("AT bellhop3d executable is not available")

    case = canonical_bellhop3d_arrival_benchmark_cases()[0]
    metrics = Simulation(case.config).compare_bellhop3d_arrivals_with_at(
        at_bin_path=at_bin,
        delay_tolerance=case.delay_tolerance_s,
        angle_tolerance=case.angle_tolerance_deg,
        tolerance_m=case.tolerance_m,
    )

    assert metrics.at_count == 3
    assert metrics.native_count == 3
    assert metrics.matched_count == 3
    assert metrics.delay_mae_s < 1.0e-5
