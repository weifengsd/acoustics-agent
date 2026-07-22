from pathlib import Path

import numpy as np
import pytest

from pyacoustics.config import ConfigLoader
from pyacoustics.postprocessing.delay_and_sum import ArrivalData, ArrivalRecord
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
from pyacoustics.solvers.bellhop.arrival_parity import (
    ArrivalParityMetrics,
    arrival_benchmark_row,
    arrival_match_table,
    compare_arrival_data,
    compute_native_arrival_data,
    plot_arrival_parity,
    run_at_arrival_data,
)
from pyacoustics.solvers.bellhop.arrivals import (
    BellhopArrival,
    compute_arrival_grid,
    find_arrivals_at_receiver,
    _merge_arrivals_like_at,
    write_arrivals_ascii,
)
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.external_io import generate_at_bellhop_env
from pyacoustics.solvers.external_io import read_at_arr
from pyacoustics.solvers.external_io import write_at_arr


def _arrival_data(delays, amplitudes=None):
    delays = np.asarray(delays, dtype=np.float64)
    if amplitudes is None:
        amplitudes = np.ones(delays.size, dtype=np.complex128)
    else:
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=amplitudes,
        delays=delays.astype(np.complex128),
        src_decl_angles=np.zeros(delays.size),
        rcvr_decl_angles=np.zeros(delays.size),
        num_top_bounces=np.zeros(delays.size, dtype=np.int16),
        num_bottom_bounces=np.zeros(delays.size, dtype=np.int16),
    )
    return ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([100.0]),
        receiver_ranges=np.array([1000.0]),
        arrivals=arrivals,
    )


def _arrival_record_3d(amplitudes, delays, phase_angles=None):
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    delays = np.asarray(delays, dtype=np.complex128)
    phases = None if phase_angles is None else np.asarray(phase_angles, dtype=np.float64)
    n = delays.size
    return ArrivalRecord(
        amplitudes=amplitudes,
        delays=delays,
        src_decl_angles=np.linspace(-2.0, 2.0, n),
        src_azim_angles=np.linspace(10.0, 20.0, n),
        rcvr_decl_angles=np.linspace(3.0, 5.0, n),
        rcvr_azim_angles=np.linspace(30.0, 40.0, n),
        phase_angles=phases,
        num_top_bounces=np.arange(n, dtype=np.int16),
        num_bottom_bounces=np.arange(n, dtype=np.int16) + 1,
    )


def _iso_eigenray_config():
    return SimulationConfig(
        project="Eigenray arrivals",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=200.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="rigid", depth=200.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[100.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[100.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=41,
            step_size=10.0,
        ),
    )


def test_native_arrivals_write_at_ascii_roundtrip(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "Munk" / "munk.yaml"
    config = ConfigLoader.load_yaml(config_path)
    config.solver.num_beams = 21
    config.solver.step_size = 50.0
    config.geometry.receivers.ranges = [1000.0]
    config.geometry.receivers.depths = [1000.0]

    grid = compute_arrival_grid(config, max_arrivals=20)
    arr_path = tmp_path / "native.arr"
    write_arrivals_ascii(arr_path, grid)

    parsed = read_at_arr(arr_path)

    assert not parsed.is_3d
    assert np.isclose(parsed.frequency, config.frequency)
    assert np.allclose(parsed.source_depths, config.geometry.source.depths)
    assert np.allclose(parsed.receiver_ranges, config.geometry.receivers.ranges)
    assert np.allclose(parsed.receiver_depths, config.geometry.receivers.depths)

    native_arrivals = grid.arrivals[0, 0, 0]
    parsed_record = parsed.arrivals[0, 0, 0]
    assert parsed_record.count == len(native_arrivals)
    if native_arrivals:
        factor = 1.0 / np.sqrt(config.geometry.receivers.ranges[0])
        assert np.isclose(abs(parsed_record.amplitudes[0]), native_arrivals[0].amplitude * factor)
        assert np.isclose(np.angle(parsed_record.amplitudes[0]), native_arrivals[0].phase_rad, atol=1e-7)
        assert np.isclose(parsed_record.delays[0], native_arrivals[0].delay)
        assert np.isclose(parsed_record.src_decl_angles[0], native_arrivals[0].src_decl_angle)
        assert np.isclose(parsed_record.rcvr_decl_angles[0], native_arrivals[0].rcvr_decl_angle)
        assert parsed_record.num_top_bounces[0] == native_arrivals[0].num_top_bounces
        assert parsed_record.num_bottom_bounces[0] == native_arrivals[0].num_bottom_bounces


def test_2d_arrival_extraction_uses_at_raw_phase_path():
    arrivals = find_arrivals_at_receiver(
        5.0,
        100.0,
        np.array([0.0, 10.0]),
        np.array([100.0, 100.0]),
        np.array([1500.0, 1500.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([np.exp(1j * 0.2), 4.0 * np.exp(1j * 1.2)]),
        np.array([2.0 * np.pi + 0.2, 2.0 * np.pi + 1.2]),
        np.array([0, 0], dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
        2,
        100.0,
        0.0,
        200.0,
        1.0,
        0.0,
        1500.0,
        0,
    )

    assert len(arrivals) == 1
    _, amp, phase, *_ = arrivals[0]
    assert amp > 0.0
    assert np.isclose(phase, 2.0 * np.pi + 0.2)


def test_2d_arrival_extraction_includes_final_segment_endpoint():
    arrivals = find_arrivals_at_receiver(
        10.0,
        100.0,
        np.array([0.0, 10.0]),
        np.array([100.0, 100.0]),
        np.array([1500.0, 1500.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        np.array([0.0, 0.0]),
        np.array([0, 0], dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
        2,
        100.0,
        0.0,
        200.0,
        1.0,
        0.0,
        1500.0,
        0,
    )

    assert len(arrivals) == 1


def test_2d_arrival_extraction_accumulates_prior_caustics():
    arrivals = find_arrivals_at_receiver(
        17.5,
        100.0,
        np.array([0.0, 10.0, 20.0]),
        np.array([100.0, 100.0, 100.0]),
        np.array([1500.0, 1500.0, 1500.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, -2.0]),
        np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0, 0, 0], dtype=np.int64),
        np.array([0, 0, 0], dtype=np.int64),
        3,
        100.0,
        0.0,
        200.0,
        1.0,
        0.0,
        1500.0,
        0,
    )

    assert len(arrivals) == 1
    _, _, phase, *_ = arrivals[0]
    assert np.isclose(phase, np.pi / 2.0)


def test_2d_arrival_merging_preserves_distinct_launch_angles():
    arrivals = [
        BellhopArrival(
            amplitude=1.0,
            phase_rad=0.0,
            delay=complex(1.0, 0.0),
            src_decl_angle=-3.6,
            rcvr_decl_angle=2.4,
            num_top_bounces=0,
            num_bottom_bounces=0,
        ),
        BellhopArrival(
            amplitude=2.0,
            phase_rad=0.01,
            delay=complex(1.00001, 0.0),
            src_decl_angle=-2.8,
            rcvr_decl_angle=2.8,
            num_top_bounces=0,
            num_bottom_bounces=0,
        ),
    ]

    merged = _merge_arrivals_like_at(
        arrivals,
        omega=2.0 * np.pi * 50.0,
        launch_angle_tol_deg=0.25,
    )

    assert len(merged) == 2


def test_arrivals_ascii_parser_ignores_at_comments(tmp_path):
    arr_path = tmp_path / "commented.arr"
    arr_path.write_text(
        "'2D' ! coordinate system\n"
        "100.0 ! frequency\n"
        "1\n"
        "50.0 / ! source depths\n"
        "1\n"
        "100.0 / ! receiver depths\n"
        "1\n"
        "1000.0 / ! receiver ranges\n"
        "1 ! max arrivals\n"
        "1 ! arrivals at receiver\n"
        "2.0 90.0 0.75 0.01 -5.0 6.0 1 2 ! arrival record\n"
    )

    parsed = read_at_arr(arr_path)
    record = parsed.arrivals[0, 0, 0]

    assert not parsed.is_3d
    assert np.isclose(parsed.frequency, 100.0)
    assert record.count == 1
    assert np.isclose(record.amplitudes[0], 2.0j)
    assert np.isclose(record.delays[0], 0.75 + 0.01j)
    assert record.num_top_bounces[0] == 1
    assert record.num_bottom_bounces[0] == 2


def test_arrivals_ascii_parser_accepts_commas_and_d_exponents(tmp_path):
    arr_path = tmp_path / "fortran_tokens.arr"
    arr_path.write_text(
        "'2D'\n"
        "1.0D+02\n"
        "1\n"
        "5.0D+01 /\n"
        "1\n"
        "1.0D+02 /\n"
        "1\n"
        "1.0D+03 /\n"
        "1 /\n"
        "1 /\n"
        "2.0D+00, 9.0D+01, 7.5D-01, 1.0D-02, -5.0D+00, 6.0D+00, 1, 2 /\n"
    )

    parsed = read_at_arr(arr_path)
    record = parsed.arrivals[0, 0, 0]

    assert not parsed.is_3d
    assert np.isclose(parsed.frequency, 100.0)
    np.testing.assert_allclose(parsed.source_depths, [50.0])
    np.testing.assert_allclose(parsed.receiver_depths, [100.0])
    np.testing.assert_allclose(parsed.receiver_ranges, [1000.0])
    assert record.count == 1
    assert np.isclose(record.amplitudes[0], 2.0j)
    assert np.isclose(record.delays[0], 0.75 + 0.01j)
    assert np.isclose(record.src_decl_angles[0], -5.0)
    assert np.isclose(record.rcvr_decl_angles[0], 6.0)
    assert record.num_top_bounces[0] == 1
    assert record.num_bottom_bounces[0] == 2


def test_3d_arrivals_write_at_ascii_roundtrip(tmp_path):
    arrivals = np.empty((2, 1, 2, 1), dtype=object)
    arrivals[0, 0, 0, 0] = _arrival_record_3d(
        amplitudes=[1.0 + 0.0j, 0.0 + 2.0j],
        delays=[0.5 + 0.01j, 0.75 + 0.02j],
    )
    arrivals[1, 0, 0, 0] = ArrivalRecord(
        amplitudes=np.zeros(0, dtype=np.complex128),
        delays=np.zeros(0, dtype=np.complex128),
        src_decl_angles=np.zeros(0),
        src_azim_angles=np.zeros(0),
        rcvr_decl_angles=np.zeros(0),
        rcvr_azim_angles=np.zeros(0),
        num_top_bounces=np.zeros(0, dtype=np.int16),
        num_bottom_bounces=np.zeros(0, dtype=np.int16),
    )
    arrivals[0, 0, 1, 0] = _arrival_record_3d(
        amplitudes=[3.0 * np.exp(1j * np.deg2rad(-45.0))],
        delays=[1.25 + 0.0j],
    )
    arrivals[1, 0, 1, 0] = _arrival_record_3d(
        amplitudes=[0.5 * np.exp(1j * np.deg2rad(120.0))],
        delays=[1.5 + 0.03j],
        phase_angles=[np.deg2rad(240.0)],
    )
    data = ArrivalData(
        is_3d=True,
        frequency=50.0,
        source_x=np.array([0.0]),
        source_y=np.array([0.0]),
        source_depths=np.array([100.0]),
        receiver_depths=np.array([25.0]),
        receiver_ranges=np.array([1000.0, 2000.0]),
        receiver_bearings=np.array([0.0, 45.0]),
        arrivals=arrivals,
    )

    arr_path = tmp_path / "native_3d.arr"
    write_at_arr(arr_path, data)
    parsed = read_at_arr(arr_path)

    assert parsed.is_3d
    assert np.isclose(parsed.frequency, 50.0)
    assert np.allclose(parsed.source_x, [0.0])
    assert np.allclose(parsed.source_y, [0.0])
    assert np.allclose(parsed.receiver_bearings, [0.0, 45.0])
    assert parsed.arrivals.shape == (2, 1, 2, 1)

    first = parsed.arrivals[0, 0, 0, 0]
    assert first.count == 2
    assert np.allclose(np.abs(first.amplitudes), [1.0, 2.0])
    assert np.allclose(np.angle(first.amplitudes), [0.0, np.pi / 2.0])
    assert np.allclose(first.delays, [0.5 + 0.01j, 0.75 + 0.02j])
    assert np.allclose(first.src_decl_angles, [-2.0, 2.0])
    assert np.allclose(first.src_azim_angles, [10.0, 20.0])
    assert np.allclose(first.rcvr_decl_angles, [3.0, 5.0])
    assert np.allclose(first.rcvr_azim_angles, [30.0, 40.0])
    assert np.array_equal(first.num_top_bounces, [0, 1])
    assert np.array_equal(first.num_bottom_bounces, [1, 2])

    assert parsed.arrivals[1, 0, 0, 0].count == 0
    assert np.isclose(abs(parsed.arrivals[0, 0, 1, 0].amplitudes[0]), 3.0)
    wrapped_phase_record = parsed.arrivals[1, 0, 1, 0]
    assert np.isclose(np.angle(wrapped_phase_record.amplitudes[0]), np.deg2rad(-120.0))
    assert np.isclose(wrapped_phase_record.phase_angles[0], np.deg2rad(240.0))


def test_3d_arrivals_roundtrip_preserves_multiple_source_xy_blocks(tmp_path):
    arrivals = np.empty((1, 1, 1, 1, 2, 2), dtype=object)
    arrivals.fill(None)
    arrivals[0, 0, 0, 0, 0, 0] = _arrival_record_3d(
        amplitudes=[1.0 + 0.0j],
        delays=[0.1 + 0.0j],
    )
    arrivals[0, 0, 0, 0, 1, 0] = _arrival_record_3d(
        amplitudes=[2.0 + 0.0j],
        delays=[0.2 + 0.0j],
    )
    arrivals[0, 0, 0, 0, 1, 1] = _arrival_record_3d(
        amplitudes=[3.0 + 0.0j],
        delays=[0.3 + 0.0j],
    )
    data = ArrivalData(
        is_3d=True,
        frequency=50.0,
        source_x=np.array([0.0, 250.0]),
        source_y=np.array([0.0, 100.0]),
        source_depths=np.array([100.0]),
        receiver_depths=np.array([25.0]),
        receiver_ranges=np.array([1000.0]),
        receiver_bearings=np.array([0.0]),
        arrivals=arrivals,
    )

    arr_path = tmp_path / "multi_source_xy_3d.arr"
    write_at_arr(arr_path, data)
    parsed = read_at_arr(arr_path)

    assert parsed.arrivals.shape == (1, 1, 1, 1, 2, 2)
    np.testing.assert_allclose(parsed.source_x, [0.0, 250.0])
    np.testing.assert_allclose(parsed.source_y, [0.0, 100.0])
    assert np.isclose(abs(parsed.arrivals[0, 0, 0, 0, 0, 0].amplitudes[0]), 1.0)
    assert parsed.arrivals[0, 0, 0, 0, 0, 1].count == 0
    assert np.isclose(abs(parsed.arrivals[0, 0, 0, 0, 1, 0].amplitudes[0]), 2.0)
    assert np.isclose(abs(parsed.arrivals[0, 0, 0, 0, 1, 1].amplitudes[0]), 3.0)


def test_3d_arrivals_writer_rejects_multi_source_xy_with_legacy_shape(tmp_path):
    data = ArrivalData(
        is_3d=True,
        frequency=50.0,
        source_x=np.array([0.0, 250.0]),
        source_y=np.array([0.0, 100.0]),
        source_depths=np.array([100.0]),
        receiver_depths=np.array([25.0]),
        receiver_ranges=np.array([1000.0]),
        receiver_bearings=np.array([0.0]),
        arrivals=np.empty((1, 1, 1, 1), dtype=object),
    )

    with pytest.raises(ValueError, match="full 6D"):
        write_at_arr(tmp_path / "bad_multi_source_xy_3d.arr", data)


def test_3d_arrivals_writer_validates_shape(tmp_path):
    data = ArrivalData(
        is_3d=True,
        frequency=50.0,
        source_x=np.array([0.0]),
        source_y=np.array([0.0]),
        source_depths=np.array([100.0]),
        receiver_depths=np.array([25.0]),
        receiver_ranges=np.array([1000.0]),
        receiver_bearings=np.array([0.0]),
        arrivals=np.empty((1, 1, 1), dtype=object),
    )

    with pytest.raises(ValueError, match="shape"):
        write_at_arr(tmp_path / "bad_3d.arr", data)


def test_arrival_grid_validates_options():
    config_path = Path(__file__).resolve().parents[1] / "Munk" / "munk.yaml"
    config = ConfigLoader.load_yaml(config_path)
    config.solver.num_beams = 3
    config.solver.step_size = 100.0
    config.geometry.receivers.ranges = [1000.0]
    config.geometry.receivers.depths = [1000.0]

    with pytest.raises(ValueError, match="max_arrivals"):
        compute_arrival_grid(config, max_arrivals=0)

    with pytest.raises(ValueError, match="beam_shape"):
        compute_arrival_grid(config, beam_shape="triangle")

    with pytest.raises(ValueError, match="method"):
        compute_arrival_grid(config, method="unknown")


def test_arrivals_writer_creates_parent_directory(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "Munk" / "munk.yaml"
    config = ConfigLoader.load_yaml(config_path)
    config.solver.num_beams = 5
    config.solver.step_size = 100.0
    config.geometry.receivers.ranges = [1000.0]
    config.geometry.receivers.depths = [1000.0]

    grid = compute_arrival_grid(config, max_arrivals=5)
    arr_path = tmp_path / "nested" / "native.arr"
    write_arrivals_ascii(arr_path, grid)

    assert arr_path.exists()


def test_eigenray_arrival_grid_writes_at_ascii_roundtrip(tmp_path):
    config = _iso_eigenray_config()

    grid = compute_arrival_grid(
        config,
        method="eigenray",
        eigenray_angle_samples=41,
        eigenray_tolerance_m=0.05,
    )
    arrivals = grid.arrivals[0, 0, 0]
    assert len(arrivals) >= 1
    assert abs(arrivals[0].src_decl_angle) < 1.0e-6
    assert np.isclose(np.real(arrivals[0].delay), 1000.0 / 1500.0, atol=1.0e-3)

    arr_path = tmp_path / "eigenray.arr"
    write_arrivals_ascii(arr_path, grid)
    parsed = read_at_arr(arr_path)

    record = parsed.arrivals[0, 0, 0]
    assert record.count == len(arrivals)
    assert np.isclose(record.delays[0], arrivals[0].delay)


def test_eigenray_arrival_grid_uses_each_source_depth():
    config = _iso_eigenray_config()
    config.geometry.source.depths = [40.0, 100.0]
    config.geometry.receivers.depths = [100.0]
    config.solver.angles = (-5.0, 5.0)
    config.solver.num_beams = 101

    grid = compute_arrival_grid(
        config,
        method="eigenray",
        eigenray_angle_samples=101,
        eigenray_tolerance_m=0.05,
    )

    shallow_source_arrival = grid.arrivals[0, 0, 0][0]
    same_depth_source_arrival = grid.arrivals[0, 0, 1][0]
    assert abs(shallow_source_arrival.src_decl_angle) > 1.0
    assert abs(same_depth_source_arrival.src_decl_angle) < 1.0e-6
    assert np.isclose(np.real(same_depth_source_arrival.delay), 1000.0 / 1500.0, atol=1.0e-3)


def test_beam_arrival_grid_uses_each_source_depth():
    config = _iso_eigenray_config()
    config.geometry.source.depths = [40.0, 100.0]
    config.geometry.receivers.depths = [100.0]
    config.solver.angles = (-5.0, 5.0)
    config.solver.num_beams = 101
    config.solver.step_size = 5.0

    grid = compute_arrival_grid(config, method="beam", max_arrivals=20)

    shallow_source_arrival = min(grid.arrivals[0, 0, 0], key=lambda arrival: arrival.distance_to_receiver)
    same_depth_source_arrival = min(grid.arrivals[0, 0, 1], key=lambda arrival: arrival.distance_to_receiver)
    assert abs(shallow_source_arrival.src_decl_angle) > 1.0
    assert abs(same_depth_source_arrival.src_decl_angle) < 1.0e-6
    assert np.isclose(np.real(same_depth_source_arrival.delay), 1000.0 / 1500.0, atol=1.0e-3)


def test_simulation_arrival_grid_supports_eigenray_method():
    sim = Simulation(_iso_eigenray_config())

    grid = sim.compute_arrival_grid(
        method="eigenray",
        eigenray_angle_samples=41,
        eigenray_tolerance_m=0.05,
    )

    assert len(grid.arrivals[0, 0, 0]) >= 1


def test_simulation_arrival_grid_save_path_uses_line_source_type(tmp_path):
    config = _iso_eigenray_config()
    config.solver.source_type = "line"
    sim = Simulation(config)
    arr_path = tmp_path / "line_source.arr"

    grid = sim.compute_arrival_grid(
        save_path=arr_path,
        method="eigenray",
        eigenray_angle_samples=41,
        eigenray_tolerance_m=0.05,
    )

    native = grid.arrivals[0, 0, 0][0]
    parsed = read_at_arr(arr_path).arrivals[0, 0, 0]
    assert parsed.count >= 1
    assert np.isclose(abs(parsed.amplitudes[0]), 4.0 * np.sqrt(np.pi) * native.amplitude)


def test_native_arrival_parity_data_uses_line_source_type():
    config = _iso_eigenray_config()
    config.solver.source_type = "line"

    grid = compute_arrival_grid(
        config,
        method="eigenray",
        eigenray_angle_samples=41,
        eigenray_tolerance_m=0.05,
    )
    parsed = compute_native_arrival_data(
        config,
        method="eigenray",
        eigenray_angle_samples=41,
        eigenray_tolerance_m=0.05,
    )

    native = grid.arrivals[0, 0, 0][0]
    record = parsed.arrivals[0, 0, 0]
    assert record.count >= 1
    assert np.isclose(abs(record.amplitudes[0]), 4.0 * np.sqrt(np.pi) * native.amplitude)


def test_arrival_parity_matching_skips_used_nearest_candidate(tmp_path):
    at_data = _arrival_data([1.000, 1.011], amplitudes=[2.0 + 0.0j, 0.0 + 4.0j])
    native_data = _arrival_data([1.006, 1.012], amplitudes=[1.0j, -2.0 + 0.0j])

    metrics = compare_arrival_data(at_data, native_data, delay_tolerance=0.02)
    figure_path = plot_arrival_parity(
        at_data,
        native_data,
        tmp_path / "nested" / "parity.png",
        metrics=metrics,
        delay_tolerance=0.02,
    )

    assert metrics.at_count == 2
    assert metrics.native_count == 2
    assert metrics.matched_count == 2
    assert metrics.unmatched_at_count == 0
    assert metrics.unmatched_native_count == 0
    assert np.isclose(metrics.match_fraction, 1.0)
    assert np.isclose(metrics.delay_mae, 0.0035)
    assert np.isclose(metrics.delay_p90_abs, 0.0055)
    assert np.isclose(metrics.amplitude_median_relative_error, 0.5)
    assert np.isclose(metrics.amplitude_p90_relative_error, 0.5)
    assert np.isclose(metrics.phase_mae_rad, np.pi / 2.0)
    assert np.isclose(metrics.phase_max_abs_rad, np.pi / 2.0)
    assert metrics.bounce_match_count == 2
    assert np.isclose(metrics.bounce_match_fraction, 1.0)
    assert metrics.top_bounce_match_count == 2
    assert np.isclose(metrics.top_bounce_match_fraction, 1.0)
    assert metrics.bottom_bounce_match_count == 2
    assert np.isclose(metrics.bottom_bounce_match_fraction, 1.0)
    rows = arrival_match_table(at_data, native_data, delay_tolerance=0.02)
    assert len(rows) == 2
    assert rows[0]["receiver_index"] == (0, 0, 0)
    assert rows[0]["top_bounce_match"]
    assert rows[0]["bottom_bounce_match"]
    assert np.isclose(rows[0]["phase_error_rad"], np.pi / 2.0)
    assert np.isclose(rows[1]["phase_error_rad"], np.pi / 2.0)
    assert figure_path.exists()


def test_arrival_parity_matching_prefers_bounce_consistency():
    at_data = _arrival_data([1.000], amplitudes=[1.0 + 0.0j])
    native_data = _arrival_data([1.001, 1.002], amplitudes=[0.0 + 1.0j, 1.0 + 0.0j])
    at_record = at_data.arrivals[0, 0, 0]
    native_record = native_data.arrivals[0, 0, 0]
    at_record.num_top_bounces[:] = [1]
    at_record.num_bottom_bounces[:] = [0]
    native_record.num_top_bounces[:] = [0, 1]
    native_record.num_bottom_bounces[:] = [1, 0]

    rows = arrival_match_table(at_data, native_data, delay_tolerance=0.01)
    metrics = compare_arrival_data(at_data, native_data, delay_tolerance=0.01)

    assert rows[0]["native_index"] == 1
    assert rows[0]["top_bounce_match"]
    assert rows[0]["bottom_bounce_match"]
    assert np.isclose(rows[0]["phase_error_rad"], 0.0)
    assert metrics.bounce_match_fraction == 1.0
    assert metrics.unmatched_at_count == 0
    assert metrics.unmatched_native_count == 1
    assert metrics.match_fraction == 1.0


def test_arrival_parity_matching_requires_bounce_consistency():
    at_data = _arrival_data([1.000], amplitudes=[1.0 + 0.0j])
    native_data = _arrival_data([1.001], amplitudes=[1.0 + 0.0j])
    at_record = at_data.arrivals[0, 0, 0]
    native_record = native_data.arrivals[0, 0, 0]
    at_record.num_top_bounces[:] = [1]
    native_record.num_top_bounces[:] = [0]

    metrics = compare_arrival_data(at_data, native_data, delay_tolerance=0.01)
    rows = arrival_match_table(at_data, native_data, delay_tolerance=0.01)

    assert metrics.matched_count == 0
    assert metrics.unmatched_at_count == 1
    assert metrics.unmatched_native_count == 1
    assert metrics.match_fraction == 0.0
    assert rows == []


def test_arrival_benchmark_row_flattens_metrics_for_reports():
    metrics = ArrivalParityMetrics(
        at_count=3,
        native_count=2,
        matched_count=2,
        unmatched_at_count=1,
        unmatched_native_count=0,
        match_fraction=2.0 / 3.0,
        delay_mae=0.001,
        delay_p90_abs=0.002,
        delay_max_abs=0.003,
        amplitude_median_relative_error=0.1,
        amplitude_p90_relative_error=0.2,
        phase_mae_rad=0.3,
        phase_max_abs_rad=float("nan"),
        launch_angle_mae=1.0,
        receiver_angle_mae=2.0,
        bounce_match_count=1,
        bounce_match_fraction=0.5,
        top_bounce_match_count=2,
        top_bounce_match_fraction=1.0,
        bottom_bounce_match_count=1,
        bottom_bounce_match_fraction=0.5,
    )

    row = arrival_benchmark_row("munk-arrivals", metrics, method="eigenray", delay_tolerance=0.02)

    assert row["case"] == "munk-arrivals"
    assert row["method"] == "eigenray"
    assert row["delay_tolerance_s"] == 0.02
    assert row["at_count"] == 3
    assert row["matched_count"] == 2
    assert row["match_fraction"] == 2.0 / 3.0
    assert row["delay_p90_abs_s"] == 0.002
    assert row["phase_max_abs_rad"] is None
    assert row["top_bounce_match_fraction"] == 1.0
    assert row["bottom_bounce_match_fraction"] == 0.5


def test_arrival_parity_rejects_negative_delay_tolerance(tmp_path):
    at_data = _arrival_data([1.0])
    native_data = _arrival_data([1.0])

    with pytest.raises(ValueError, match="delay_tolerance"):
        compare_arrival_data(at_data, native_data, delay_tolerance=-0.001)

    with pytest.raises(ValueError, match="delay_tolerance"):
        plot_arrival_parity(at_data, native_data, tmp_path / "parity.png", delay_tolerance=-0.001)


def test_at_bellhop_env_arrivals_mode_uses_numeric_defaults(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "Munk" / "munk.yaml"
    config = ConfigLoader.load_yaml(config_path)

    env_path = tmp_path / "case.env"
    generate_at_bellhop_env(config, env_path, run_type="A")
    text = env_path.read_text()

    assert "'A'" in text
    assert "None" not in text


def test_native_arrivals_have_at_parity_smoke(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "Munk" / "munk.yaml"
    config = ConfigLoader.load_yaml(config_path)
    external = ExternalSolver(config)
    if not external.is_available("bellhop"):
        pytest.skip("AT bellhop executable is not available")

    config.solver.num_beams = 81
    config.solver.step_size = 25.0
    config.solver.angles = (-20.0, 20.0)
    config.geometry.receivers.ranges = [10000.0]
    config.geometry.receivers.depths = [1000.0]

    at_data = run_at_arrival_data(config)
    native_data = compute_native_arrival_data(config, max_arrivals=50)
    metrics = compare_arrival_data(at_data, native_data, delay_tolerance=0.02)
    figure_path = plot_arrival_parity(
        at_data,
        native_data,
        tmp_path / "bellhop_arrival_parity_smoke.png",
        metrics=metrics,
        delay_tolerance=0.02,
    )

    assert metrics.at_count > 0
    assert metrics.native_count > 0
    assert metrics.matched_count >= 1
    assert metrics.delay_mae < 0.02
    assert metrics.amplitude_median_relative_error < 0.25
    assert figure_path.exists()


def test_simulation_exposes_arrival_parity_metrics():
    config_path = Path(__file__).resolve().parents[1] / "Munk" / "munk.yaml"
    sim = Simulation(config_path)
    external = ExternalSolver(sim.config)
    if not external.is_available("bellhop"):
        pytest.skip("AT bellhop executable is not available")

    sim.config.solver.num_beams = 81
    sim.config.solver.step_size = 25.0
    sim.config.solver.angles = (-20.0, 20.0)
    sim.config.geometry.receivers.ranges = [10000.0]
    sim.config.geometry.receivers.depths = [1000.0]

    metrics = sim.compare_arrivals_with_at(max_arrivals=50, delay_tolerance=0.02)

    assert metrics.matched_count >= 1
    assert metrics.delay_mae < 0.02
    assert metrics.amplitude_median_relative_error < 0.25


def test_simulation_arrival_parity_forwards_eigenray_options(monkeypatch):
    from pyacoustics.solvers.bellhop import arrival_parity as arrival_parity_module

    captured = {}

    def fake_run_at_arrival_data(config, *, at_bin_path=None, run_type="A"):
        captured["at_bin_path"] = at_bin_path
        captured["run_type"] = run_type
        return _arrival_data([1.0])

    def fake_compute_native_arrival_data(config, **kwargs):
        captured.update(kwargs)
        return _arrival_data([1.0])

    monkeypatch.setattr(arrival_parity_module, "run_at_arrival_data", fake_run_at_arrival_data)
    monkeypatch.setattr(arrival_parity_module, "compute_native_arrival_data", fake_compute_native_arrival_data)

    sim = Simulation(_iso_eigenray_config())
    metrics = sim.compare_arrivals_with_at(
        max_arrivals=7,
        delay_tolerance=0.01,
        at_bin_path="/opt/at/bin",
        run_type="AB   3",
        beam_shape="gaussian",
        method="eigenray",
        eigenray_tolerance_m=0.05,
        eigenray_angle_samples=41,
    )

    assert metrics.matched_count == 1
    assert captured["at_bin_path"] == "/opt/at/bin"
    assert captured["run_type"] == "AB   3"
    assert captured["max_arrivals"] == 7
    assert captured["beam_shape"] == "gaussian"
    assert captured["method"] == "eigenray"
    assert captured["eigenray_tolerance_m"] == 0.05
    assert captured["eigenray_angle_samples"] == 41
