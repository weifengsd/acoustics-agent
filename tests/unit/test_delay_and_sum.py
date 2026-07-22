import importlib

import numpy as np
import pytest
from scipy.io import loadmat

delay_module = importlib.import_module("pyacoustics.postprocessing.delay_and_sum")
from pyacoustics.postprocessing.delay_and_sum import (
    ArrivalData,
    ArrivalRecord,
    DelayAndSumResult,
    delay_and_sum,
    matched_filter_timeseries,
    read_arrivals_ascii,
    save_matlab_timeseries,
    save_matched_filter_matlab,
)
from pyacoustics.schema import (
    BellhopConfig,
    BoundaryCondition,
    EnvironmentModel,
    GeometryModel,
    ReceiverModel,
    SimulationConfig,
    SourceModel,
    SSPLayer,
    SSPModel,
)
from pyacoustics.simulation import Simulation


def test_read_arrivals_ascii_2d(tmp_path):
    arr_file = tmp_path / "simple.arr"
    arr_file.write_text(
        "'2D'\n"
        "100.0\n"
        "1\n"
        "50.0 /\n"
        "1\n"
        "100.0 /\n"
        "1\n"
        "0.0 /\n"
        "1\n"
        "1\n"
        "2.0 90.0 0.003 0.0 -1.0 1.0 0 0\n"
    )

    data = read_arrivals_ascii(arr_file)

    assert not data.is_3d
    assert data.frequency == 100.0
    record = data.arrivals[0, 0, 0]
    assert record.count == 1
    assert np.isclose(record.amplitudes[0], 2.0j)
    assert np.isclose(record.delays[0], 0.003 + 0.0j)
    assert np.isclose(record.rcvr_decl_angles[0], 1.0)


def test_read_arrivals_ascii_2d_irregular_receiver_grid(tmp_path):
    arr_file = tmp_path / "irregular.arr"
    arr_file.write_text(
        "'2D'\n"
        "100.0\n"
        "1\n"
        "100.0 /\n"
        "2\n"
        "100.0 120.0 /\n"
        "2\n"
        "1000.0 1500.0 /\n"
        "2\n"
        "1\n"
        "1.0e-3 0.0 0.666666687 0.0 0.0 0.0 0 0\n"
        "2\n"
        "6.66606589e-4 0.0 1.00008821 0.0 0.763898134 0.763898134 0 0\n"
        "6.61917089e-4 0.0 1.00717354 0.0 6.84276915 -6.84276915 0 1\n"
    )

    data = read_arrivals_ascii(arr_file)

    assert not data.is_3d
    assert data.arrivals.shape == (2, 1, 1)
    np.testing.assert_allclose(data.receiver_depths, [100.0, 120.0])
    np.testing.assert_allclose(data.receiver_ranges, [1000.0, 1500.0])
    assert data.arrivals[0, 0, 0].count == 1
    assert data.arrivals[1, 0, 0].count == 2
    np.testing.assert_allclose(data.arrivals[1, 0, 0].num_bottom_bounces, [0, 1])


def test_delay_and_sum_places_arrival_at_sample_index():
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.002 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        arrivals=arrivals,
    )

    result = delay_and_sum(
        data,
        [1.0, 0.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.01,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )

    trace = result.timeseries[:, 0, 0]
    assert np.isclose(trace[2], 1.0)
    assert np.isclose(np.sum(np.abs(trace)) - abs(trace[2]), 0.0)


def test_delay_and_sum_applies_complex_phase_to_analytic_source():
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0j]),
        delays=np.array([0.0 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        arrivals=arrivals,
    )

    result = delay_and_sum(
        data,
        [1.0, 0.0, 0.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.004,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )

    # Multiplying by 1j should pull the Hilbert quadrature component into
    # the real received trace, so the output should not equal the input pulse.
    assert not np.allclose(result.timeseries[:, 0, 0], np.array([1.0, 0.0, 0.0, 0.0]))


def test_delay_and_sum_handles_multi_receiver_grid_and_multiple_arrivals():
    ranges = np.array([0.0, 1000.0])
    depths = np.array([50.0, 150.0])
    arrivals = np.empty((ranges.size, depths.size, 1), dtype=object)

    for ir, receiver_range in enumerate(ranges):
        base_time = receiver_range / 1000.0
        for iz, _ in enumerate(depths):
            delays = np.array([base_time + 0.002, base_time + 0.006], dtype=np.complex128)
            amplitudes = np.array([1.0 + 0.0j, (ir + iz + 2.0) + 0.0j], dtype=np.complex128)
            arrivals[ir, iz, 0] = ArrivalRecord(
                amplitudes=amplitudes,
                delays=delays,
                src_decl_angles=np.zeros(2),
                rcvr_decl_angles=np.zeros(2),
                num_top_bounces=np.zeros(2, dtype=np.int16),
                num_bottom_bounces=np.zeros(2, dtype=np.int16),
            )

    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([25.0]),
        receiver_depths=depths,
        receiver_ranges=ranges,
        arrivals=arrivals,
    )

    result = delay_and_sum(
        data,
        [1.0, 0.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.02,
        reduction_velocity=1000.0,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )

    assert result.timeseries.shape == (20, 2, 2)
    assert result.time_windows.shape == (2, 20)
    assert np.allclose(result.time_windows[0, :3], [0.0, 0.001, 0.002])
    assert np.allclose(result.time_windows[1, :3], [1.0, 1.001, 1.002])

    for ir in range(ranges.size):
        for iz in range(depths.size):
            trace = result.timeseries[:, iz, ir]
            assert np.isclose(trace[2], 1.0)
            assert np.isclose(trace[6], ir + iz + 2.0)


def test_delay_and_sum_doppler_accepts_single_velocity_bin():
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.002 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        arrivals=arrivals,
    )

    result = delay_and_sum(
        data,
        [1.0, 0.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.01,
        start_time_margin=0.0,
        receiver_velocity=(10.0, 0.0),
        doppler_velocity_bins=[0.0],
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )

    assert result.timeseries.shape == (10, 1, 1)
    assert np.isfinite(result.timeseries).all()
    assert np.max(np.abs(result.timeseries[:, 0, 0])) > 0.0


def test_delay_and_sum_modified_chirp_z_matches_fft_on_unit_circle():
    signal = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float64)

    transformed = delay_module._modified_chirp_z_unit_circle(signal, signal.size, 2.0 * np.pi / signal.size)

    np.testing.assert_allclose(transformed, np.fft.fft(signal), atol=1.0e-12)


def test_delay_and_sum_doppler_alpha_one_preserves_waveform():
    waveform = np.array([0.0, 1.0, -0.5, 0.25], dtype=np.float64)

    shifted = delay_module._dopplerize(waveform, 1.0)

    np.testing.assert_allclose(shifted, waveform)


def test_delay_and_sum_doppler_rejects_nonpositive_reference_speed():
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.0 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        arrivals=arrivals,
    )

    with pytest.raises(ValueError, match="doppler_reference_speed"):
        delay_and_sum(
            data,
            [1.0, 0.0],
            sample_rate=1000.0,
            receiver_velocity=(1.0, 0.0),
            doppler_reference_speed=0.0,
        )


def test_delay_and_sum_3d_doppler_uses_receiver_azimuth(monkeypatch):
    def fake_analytic_bank(waveform, velocity, doppler_reference_speed, doppler_velocity_bins):
        bank = np.zeros((len(waveform), 2), dtype=np.complex128)
        bank[0, 0] = 2.0
        bank[0, 1] = 5.0
        return bank, np.array([1.0, 0.0])

    monkeypatch.setattr(delay_module, "_analytic_bank", fake_analytic_bank)
    arrivals = np.empty((1, 1, 1, 1), dtype=object)
    arrivals[0, 0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.0 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        src_azim_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        rcvr_azim_angles=np.array([90.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=True,
        frequency=100.0,
        source_x=np.array([0.0]),
        source_y=np.array([0.0]),
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        receiver_bearings=np.array([90.0]),
        arrivals=arrivals,
    )

    result = delay_and_sum(
        data,
        [1.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.004,
        start_time_margin=0.0,
        receiver_velocity=(10.0, 0.0, 0.0),
        doppler_reference_speed=10.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )

    assert np.isclose(result.timeseries[0, 0, 0, 0], 2.0)


def test_delay_and_sum_rejects_3d_velocity_for_2d_arrivals():
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.0 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        arrivals=arrivals,
    )

    with pytest.raises(ValueError, match="3D arrivals"):
        delay_and_sum(
            data,
            [1.0, 0.0],
            sample_rate=1000.0,
            receiver_velocity=(1.0, 0.0, 0.0),
        )


def test_simulation_synthesizes_arrivals_file(tmp_path):
    arr_file = tmp_path / "simple.arr"
    arr_file.write_text(
        "'2D'\n"
        "100.0\n"
        "1\n"
        "50.0 /\n"
        "1\n"
        "100.0 /\n"
        "1\n"
        "0.0 /\n"
        "1\n"
        "1\n"
        "1.0 0.0 0.002 0.0 0.0 0.0 0 0\n"
    )
    config = SimulationConfig(
        project="delay and sum api",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(type="c-linear", data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)]),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[0.0], depths=[100.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-1.0, 1.0), num_beams=3, step_size=10.0),
    )
    sim = Simulation(config)
    mat_path = tmp_path / "simple.rts.mat"

    result = sim.synthesize_arrivals_file(
        arr_file,
        [1.0, 0.0, 0.0],
        sample_rate=1000.0,
        output_mat_path=mat_path,
        window_duration=0.01,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )

    assert result.timeseries.shape == (10, 1, 1)
    assert np.isclose(result.timeseries[2, 0, 0], 1.0)
    payload = loadmat(mat_path)
    assert payload["PlotTitle"].item() == "simple"
    assert payload["RTS"].shape == (10, 1)
    assert payload["tout"].shape == (1, 10)
    assert np.isclose(payload["RTS"][2, 0], -1.0)
    assert np.isclose(payload["sample_rate"].item(), 1000.0)
    assert payload["source_index"].item() == 0
    np.testing.assert_allclose(payload["source_depths"].ravel(), [50.0])
    assert np.isclose(payload["selected_source_depth"].item(), 50.0)
    assert payload["receiver_bearings"].size == 0


def test_matched_filter_timeseries_finds_replica_peak():
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.003 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        arrivals=arrivals,
    )
    waveform = [1.0, 2.0, 1.0]

    result = delay_and_sum(
        data,
        waveform,
        sample_rate=1000.0,
        window_duration=0.01,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )
    matched = matched_filter_timeseries(result, waveform, normalize_replica=False)

    trace = matched.timeseries[:, 0, 0]
    assert matched.timeseries.shape == result.timeseries.shape
    assert matched.lags[np.argmax(trace)] == 3
    assert np.isclose(np.max(trace), np.dot(waveform, waveform))


def test_delay_and_sum_3d_arrivals_matlab_payload(tmp_path):
    arrivals = np.empty((1, 1, 1, 1), dtype=object)
    arrivals[0, 0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.002 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        src_azim_angles=np.array([10.0]),
        rcvr_decl_angles=np.array([0.0]),
        rcvr_azim_angles=np.array([20.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=True,
        frequency=100.0,
        source_x=np.array([0.0]),
        source_y=np.array([0.0]),
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        receiver_bearings=np.array([30.0]),
        arrivals=arrivals,
    )
    sim = Simulation(
        SimulationConfig(
            project="3d delay and sum api",
            frequency=100.0,
            environment=EnvironmentModel(
                ssp=SSPModel(type="c-linear", data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)]),
                surface=BoundaryCondition(type="vacuum", depth=0.0),
                bottom=BoundaryCondition(type="rigid", depth=100.0),
            ),
            geometry=GeometryModel(
                source=SourceModel(depths=[50.0]),
                receivers=ReceiverModel(ranges=[0.0], depths=[100.0]),
            ),
            solver=BellhopConfig(type="bellhop", angles=(-1.0, 1.0), num_beams=3, step_size=10.0),
        )
    )
    mat_path = tmp_path / "three_d.rts.mat"

    result = sim.synthesize_arrivals(
        data,
        [1.0, 0.0, 0.0],
        sample_rate=1000.0,
        output_mat_path=mat_path,
        window_duration=0.01,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )
    payload = loadmat(mat_path)

    assert result.timeseries.shape == (10, 1, 1, 1)
    assert result.time_windows.shape == (1, 1, 10)
    assert np.isclose(result.timeseries[2, 0, 0, 0], 1.0)
    assert payload["RTS"].shape == (10, 1, 1)
    assert payload["tout"].shape == (1, 10)
    assert np.isclose(payload["RTS"][2, 0, 0], -1.0)
    assert payload["source_index"].item() == 0
    np.testing.assert_allclose(payload["source_depths"].ravel(), [50.0])
    assert np.isclose(payload["selected_source_depth"].item(), 50.0)
    assert payload["source_x_index"].item() == 0
    assert payload["source_y_index"].item() == 0
    np.testing.assert_allclose(payload["source_x"].ravel(), [0.0])
    np.testing.assert_allclose(payload["source_y"].ravel(), [0.0])
    assert np.isclose(payload["selected_source_x"].item(), 0.0)
    assert np.isclose(payload["selected_source_y"].item(), 0.0)
    np.testing.assert_allclose(payload["receiver_bearings"].ravel(), [30.0])


def test_delay_and_sum_3d_selects_source_xy_block(tmp_path):
    arrivals = np.empty((1, 1, 1, 1, 2, 2), dtype=object)
    arrivals.fill(None)
    arrivals[0, 0, 0, 0, 1, 1] = ArrivalRecord(
        amplitudes=np.array([3.0 + 0.0j]),
        delays=np.array([0.002 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        src_azim_angles=np.array([10.0]),
        rcvr_decl_angles=np.array([0.0]),
        rcvr_azim_angles=np.array([20.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=True,
        frequency=100.0,
        source_x=np.array([0.0, 250.0]),
        source_y=np.array([0.0, 100.0]),
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        receiver_bearings=np.array([30.0]),
        arrivals=arrivals,
    )

    default_result = delay_and_sum(
        data,
        [1.0, 0.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.01,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )
    selected_result = delay_and_sum(
        data,
        [1.0, 0.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.01,
        start_time_margin=0.0,
        source_x_index=1,
        source_y_index=1,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )

    assert np.allclose(default_result.timeseries, 0.0)
    assert np.isclose(selected_result.timeseries[2, 0, 0, 0], 3.0)
    assert selected_result.source_x_index == 1
    assert selected_result.source_y_index == 1
    np.testing.assert_allclose(selected_result.source_depths, [50.0])
    np.testing.assert_allclose(selected_result.source_x, [0.0, 250.0])
    np.testing.assert_allclose(selected_result.source_y, [0.0, 100.0])

    mat_path = tmp_path / "selected_source_xy.rts.mat"
    save_matlab_timeseries(selected_result, mat_path)
    payload = loadmat(mat_path)
    np.testing.assert_allclose(payload["source_x"].ravel(), [0.0, 250.0])
    np.testing.assert_allclose(payload["source_y"].ravel(), [0.0, 100.0])
    np.testing.assert_allclose(payload["source_depths"].ravel(), [50.0])
    assert np.isclose(payload["selected_source_depth"].item(), 50.0)
    assert payload["source_x_index"].item() == 1
    assert payload["source_y_index"].item() == 1
    assert np.isclose(payload["selected_source_x"].item(), 250.0)
    assert np.isclose(payload["selected_source_y"].item(), 100.0)


def test_delay_and_sum_rejects_source_xy_index_without_blocks():
    arrivals = np.empty((1, 1, 1, 1), dtype=object)
    arrivals[0, 0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.002 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        src_azim_angles=np.array([10.0]),
        rcvr_decl_angles=np.array([0.0]),
        rcvr_azim_angles=np.array([20.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=True,
        frequency=100.0,
        source_x=np.array([0.0]),
        source_y=np.array([0.0]),
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        receiver_bearings=np.array([30.0]),
        arrivals=arrivals,
    )

    with pytest.raises(IndexError, match="source_x_index"):
        delay_and_sum(
            data,
            [1.0, 0.0, 0.0],
            sample_rate=1000.0,
            source_x_index=1,
        )


def test_simulation_exposes_matched_filter_timeseries(tmp_path):
    sim = Simulation(
        SimulationConfig(
            project="matched filter api",
            frequency=100.0,
            environment=EnvironmentModel(
                ssp=SSPModel(type="c-linear", data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)]),
                surface=BoundaryCondition(type="vacuum", depth=0.0),
                bottom=BoundaryCondition(type="rigid", depth=100.0),
            ),
            geometry=GeometryModel(
                source=SourceModel(depths=[50.0]),
                receivers=ReceiverModel(ranges=[0.0], depths=[100.0]),
            ),
            solver=BellhopConfig(type="bellhop", angles=(-1.0, 1.0), num_beams=3, step_size=10.0),
        )
    )
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([0.001 + 0.0j]),
        src_decl_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    data = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        arrivals=arrivals,
    )

    result = sim.synthesize_arrivals(
        data,
        [1.0, 0.0],
        sample_rate=1000.0,
        window_duration=0.005,
        start_time_margin=0.0,
        normalize_source=False,
        invert_source_sign=False,
        shift_fraction=0.0,
    )
    mf_path = tmp_path / "sim_matched.mat"
    matched = sim.matched_filter_timeseries(result, [1.0, 0.0], output_mat_path=mf_path)
    payload = loadmat(mf_path)

    assert matched.timeseries.shape == result.timeseries.shape
    assert np.max(matched.timeseries[:, 0, 0]) > 0.0
    assert payload["PlotTitle"].item() == "matched filter api"
    assert payload["MF"].shape == (5, 1)
    assert payload["source_index"].item() == 0
    assert payload["receiver_bearings"].size == 0


def test_save_matched_filter_matlab_payload(tmp_path):
    result = DelayAndSumResult(
        timeseries=np.zeros((5, 1, 1), dtype=np.float64),
        time_windows=np.zeros((1, 5), dtype=np.float64),
        sample_rate=1000.0,
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        source_index=0,
    )
    result.timeseries[:, 0, 0] = [0.0, 1.0, 2.0, 1.0, 0.0]
    matched = matched_filter_timeseries(result, [1.0, 2.0, 1.0], normalize_replica=False)
    mf_path = tmp_path / "matched.mat"

    save_matched_filter_matlab(matched, mf_path, plot_title="matched")
    payload = loadmat(mf_path)

    assert payload["PlotTitle"].item() == "matched"
    assert payload["MF"].shape == (5, 1)
    np.testing.assert_allclose(payload["lags"].ravel(), matched.lags)
    np.testing.assert_allclose(payload["lag_seconds"].ravel(), matched.lags / matched.sample_rate)
    assert np.isclose(np.max(payload["MF"]), np.max(matched.timeseries[:, 0, 0]))
    assert payload["source_index"].item() == 0
    assert payload["receiver_bearings"].size == 0


def test_save_matched_filter_matlab_payload_preserves_3d_bearings(tmp_path):
    result = DelayAndSumResult(
        timeseries=np.zeros((5, 1, 1, 2), dtype=np.float64),
        time_windows=np.zeros((1, 2, 5), dtype=np.float64),
        sample_rate=1000.0,
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([0.0]),
        source_index=0,
        receiver_bearings=np.array([15.0, 45.0]),
        source_depths=np.array([50.0]),
        source_x_index=1,
        source_y_index=0,
        source_x=np.array([0.0, 250.0]),
        source_y=np.array([100.0]),
    )
    result.timeseries[:, 0, 0, 0] = [0.0, 1.0, 2.0, 1.0, 0.0]
    result.timeseries[:, 0, 0, 1] = [0.0, 0.5, 1.0, 0.5, 0.0]
    matched = matched_filter_timeseries(result, [1.0, 2.0, 1.0], normalize_replica=False)
    mf_path = tmp_path / "matched_3d.mat"

    save_matched_filter_matlab(matched, mf_path, plot_title="matched 3d")
    payload = loadmat(mf_path)

    assert payload["PlotTitle"].item() == "matched 3d"
    assert payload["MF"].shape == (5, 1, 2)
    assert payload["source_index"].item() == 0
    np.testing.assert_allclose(payload["source_depths"].ravel(), [50.0])
    assert np.isclose(payload["selected_source_depth"].item(), 50.0)
    assert payload["source_x_index"].item() == 1
    assert payload["source_y_index"].item() == 0
    np.testing.assert_allclose(payload["source_x"].ravel(), [0.0, 250.0])
    np.testing.assert_allclose(payload["source_y"].ravel(), [100.0])
    assert np.isclose(payload["selected_source_x"].item(), 250.0)
    assert np.isclose(payload["selected_source_y"].item(), 100.0)
    np.testing.assert_allclose(payload["receiver_bearings"].ravel(), [15.0, 45.0])
