import numpy as np

from pyacoustics.postprocessing.delay_and_sum import read_arrivals_ascii
from pyacoustics.schema import (
    Bellhop3DConfig,
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
from pyacoustics.solvers.bellhop3d.arrivals import (
    Bellhop3DArrival,
    _append_arrival_at_style,
    _arrival_record,
    _interp_angle_rad,
    _nearest_segment_arrival,
    _prune_negligible_arrivals,
    compute_arrival_grid_3d,
    write_arrival_grid_3d,
)


def _direct_3d_arrival_config():
    return SimulationConfig(
        project="Direct 3D arrivals",
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
        solver=Bellhop3DConfig(
            elevation_angles=(0.0, 0.0),
            azimuth_angles=(0.0, 0.0),
            num_elevation_beams=1,
            num_azimuth_beams=1,
            step_size=50.0,
            box_x=1200.0,
            box_y=100.0,
            box_z=300.0,
        ),
    )


def test_compute_arrival_grid_3d_extracts_direct_path():
    data = compute_arrival_grid_3d(
        _direct_3d_arrival_config(),
        receiver_bearings=np.array([0.0]),
        tolerance_m=0.5,
    )

    record = data.arrivals[0, 0, 0, 0]

    assert data.is_3d
    assert data.arrivals.shape == (1, 1, 1, 1)
    assert np.allclose(data.receiver_bearings, [0.0])
    assert record.count == 1
    assert np.isclose(record.delays[0], 1000.0 / 1500.0)
    assert np.isclose(abs(record.amplitudes[0]), 1.0 / (2.0 * np.pi * 1000.0))
    assert np.allclose(record.src_decl_angles, [0.0])
    assert np.allclose(record.src_azim_angles, [0.0])
    assert np.allclose(record.rcvr_decl_angles, [0.0])
    assert np.allclose(record.rcvr_azim_angles, [0.0])
    assert np.array_equal(record.num_top_bounces, [0])
    assert np.array_equal(record.num_bottom_bounces, [0])


def test_simulation_writes_native_bellhop3d_arrivals_ascii(tmp_path):
    sim = Simulation(_direct_3d_arrival_config())

    arr_path = tmp_path / "native_3d.arr"
    data = sim.compute_arrival_grid(
        save_path=str(arr_path),
        receiver_bearings=np.array([0.0]),
        tolerance_m=0.5,
    )
    parsed = read_arrivals_ascii(arr_path)

    assert data.is_3d
    assert parsed.is_3d
    assert parsed.arrivals.shape == (1, 1, 1, 1)
    assert parsed.arrivals[0, 0, 0, 0].count == 1
    assert np.isclose(parsed.arrivals[0, 0, 0, 0].delays[0], 1000.0 / 1500.0)


def test_compute_arrival_grid_3d_traces_requested_source_depths():
    config = _direct_3d_arrival_config()
    data = compute_arrival_grid_3d(
        config,
        source_depths=np.array([80.0, 120.0]),
        receiver_depths=np.array([80.0, 120.0]),
        receiver_bearings=np.array([0.0]),
        tolerance_m=0.5,
    )

    assert data.arrivals.shape == (1, 2, 1, 2)
    assert data.arrivals[0, 0, 0, 0].count == 1
    assert data.arrivals[0, 1, 0, 1].count == 1
    assert data.arrivals[0, 1, 0, 0].count == 0
    assert data.arrivals[0, 0, 0, 1].count == 0
    assert np.isclose(data.arrivals[0, 0, 0, 0].delays[0], 1000.0 / 1500.0)
    assert np.isclose(data.arrivals[0, 1, 0, 1].delays[0], 1000.0 / 1500.0)


def test_compute_arrival_grid_3d_traces_source_xy_blocks(tmp_path):
    config = _direct_3d_arrival_config()
    data = compute_arrival_grid_3d(
        config,
        source_x=np.array([0.0, 100.0]),
        source_y=np.array([0.0]),
        receiver_bearings=np.array([0.0]),
        tolerance_m=0.5,
    )

    assert data.arrivals.shape == (1, 1, 1, 1, 2, 1)
    np.testing.assert_allclose(data.source_x, [0.0, 100.0])
    np.testing.assert_allclose(data.source_y, [0.0])
    assert data.arrivals[0, 0, 0, 0, 0, 0].count == 1
    assert data.arrivals[0, 0, 0, 0, 1, 0].count == 1
    assert np.isclose(data.arrivals[0, 0, 0, 0, 0, 0].delays[0], 1000.0 / 1500.0)
    assert np.isclose(data.arrivals[0, 0, 0, 0, 1, 0].delays[0], 1000.0 / 1500.0)

    arr_path = tmp_path / "native_3d_source_xy.arr"
    write_arrival_grid_3d(arr_path, data)
    parsed = read_arrivals_ascii(arr_path)

    assert parsed.arrivals.shape == data.arrivals.shape
    np.testing.assert_allclose(parsed.source_x, data.source_x)
    np.testing.assert_allclose(parsed.source_y, data.source_y)
    assert parsed.arrivals[0, 0, 0, 0, 0, 0].count == 1
    assert parsed.arrivals[0, 0, 0, 0, 1, 0].count == 1


def test_nearest_segment_arrival_uses_at_segment_endpoint_amplitude():
    arrival = _nearest_segment_arrival(
        receiver_x=5.0,
        receiver_y=0.0,
        receiver_z=100.0,
        x_path=np.array([0.0, 10.0]),
        y_path=np.array([0.0, 0.0]),
        z_path=np.array([100.0, 100.0]),
        q11_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        q12_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q21_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q22_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        amp_path=np.array([2.0 + 0.0j, 4.0 + 0.0j]),
        tau_path=np.array([0.0, 1.0]),
        kmah_path=np.array([0.0, 0.0]),
        c_path=np.array([1500.0, 1500.0]),
        phi_path=np.array([0.0, 0.0]),
        launch_elev_deg=0.0,
        launch_azim_deg=0.0,
        num_top_bounces=0,
        num_bottom_bounces=0,
        tolerance_m=0.1,
        c_src=1500.0,
        d_elev=0.1,
        d_azim=0.1,
    )

    assert arrival is not None
    assert np.isclose(arrival.delay.real, 0.5)
    assert np.isclose(abs(arrival.amplitude), 4.0 / (2.0 * np.pi))


def test_nearest_segment_arrival_uses_at_segment_start_raw_phase():
    arrival = _nearest_segment_arrival(
        receiver_x=5.0,
        receiver_y=0.0,
        receiver_z=100.0,
        x_path=np.array([0.0, 10.0]),
        y_path=np.array([0.0, 0.0]),
        z_path=np.array([100.0, 100.0]),
        q11_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        q12_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q21_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q22_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        amp_path=np.array([np.exp(1j * 0.2), 4.0 * np.exp(1j * 1.2)]),
        phase_path=np.array([2.0 * np.pi + 0.2, 2.0 * np.pi + 1.2]),
        tau_path=np.array([0.0, 1.0]),
        kmah_path=np.array([0.0, 0.0]),
        c_path=np.array([1500.0, 1500.0]),
        phi_path=np.array([0.0, 0.0]),
        launch_elev_deg=0.0,
        launch_azim_deg=0.0,
        num_top_bounces=0,
        num_bottom_bounces=0,
        tolerance_m=0.1,
        c_src=1500.0,
        d_elev=0.1,
        d_azim=0.1,
    )

    assert arrival is not None
    assert np.isclose(abs(arrival.amplitude), 4.0 / (2.0 * np.pi))
    assert np.isclose(arrival.phase_rad, 2.0 * np.pi + 0.2)
    assert np.isclose(np.angle(arrival.amplitude), 0.2)


def test_nearest_segment_arrival_interpolates_phi_for_gaussian_footprint():
    arrival = _nearest_segment_arrival(
        receiver_x=5.0,
        receiver_y=0.0,
        receiver_z=1.0,
        x_path=np.array([0.0, 10.0]),
        y_path=np.array([0.0, 0.0]),
        z_path=np.array([0.0, 0.0]),
        q11_path=np.array([2.0 + 0.0j, 2.0 + 0.0j]),
        q12_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q21_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q22_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        amp_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        tau_path=np.array([0.0, 1.0]),
        kmah_path=np.array([0.0, 0.0]),
        c_path=np.array([1500.0, 1500.0]),
        phi_path=np.array([0.0, np.pi / 2.0]),
        launch_elev_deg=0.0,
        launch_azim_deg=0.0,
        num_top_bounces=0,
        num_bottom_bounces=0,
        tolerance_m=2.0,
        c_src=1500.0,
        d_elev=1.0,
        d_azim=1.0,
    )

    expected_decay = np.exp(-0.3125)
    expected_amplitude = expected_decay / (2.0 * np.pi * np.sqrt(2.0))

    assert arrival is not None
    assert np.isclose(arrival.delay.real, 0.5)
    assert np.isclose(abs(arrival.amplitude), expected_amplitude)


def test_nearest_segment_arrival_accepts_beam_footprint_beyond_centerline_tolerance():
    arrival = _nearest_segment_arrival(
        receiver_x=50.0,
        receiver_y=3.0,
        receiver_z=0.0,
        x_path=np.array([0.0, 100.0]),
        y_path=np.array([0.0, 0.0]),
        z_path=np.array([0.0, 0.0]),
        q11_path=np.array([100.0 + 0.0j, 100.0 + 0.0j]),
        q12_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q21_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q22_path=np.array([100.0 + 0.0j, 100.0 + 0.0j]),
        amp_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        tau_path=np.array([0.0, 1.0]),
        kmah_path=np.array([0.0, 0.0]),
        c_path=np.array([1500.0, 1500.0]),
        phi_path=np.array([0.0, 0.0]),
        launch_elev_deg=0.0,
        launch_azim_deg=0.0,
        num_top_bounces=0,
        num_bottom_bounces=0,
        tolerance_m=0.5,
        c_src=1500.0,
        d_elev=0.01,
        d_azim=0.01,
    )

    assert arrival is not None
    assert arrival.distance_to_receiver > 0.5
    assert np.isclose(arrival.delay.real, 0.5)


def test_phi_interpolation_uses_shortest_angle_wrap():
    phi_mid = _interp_angle_rad(np.deg2rad(179.0), np.deg2rad(-179.0), 0.5)

    assert np.isclose(abs(phi_mid), np.pi)


def test_nearest_segment_arrival_uses_segment_kmah_state():
    arrival = _nearest_segment_arrival(
        receiver_x=7.5,
        receiver_y=0.0,
        receiver_z=100.0,
        x_path=np.array([0.0, 10.0]),
        y_path=np.array([0.0, 0.0]),
        z_path=np.array([100.0, 100.0]),
        q11_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        q12_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q21_path=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        q22_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        amp_path=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        tau_path=np.array([0.0, 1.0]),
        kmah_path=np.array([0.0, 1.0]),
        c_path=np.array([1500.0, 1500.0]),
        phi_path=np.array([0.0, 0.0]),
        launch_elev_deg=0.0,
        launch_azim_deg=0.0,
        num_top_bounces=0,
        num_bottom_bounces=0,
        tolerance_m=0.1,
        c_src=1500.0,
        d_elev=0.1,
        d_azim=0.1,
    )

    assert arrival is not None
    assert np.isclose(arrival.delay.real, 0.75)
    assert np.isclose(np.angle(arrival.amplitude), np.pi / 2.0)


def test_nearest_segment_arrival_accumulates_prior_and_in_segment_caustics():
    arrival = _nearest_segment_arrival(
        receiver_x=17.5,
        receiver_y=0.0,
        receiver_z=100.0,
        x_path=np.array([0.0, 10.0, 20.0]),
        y_path=np.array([0.0, 0.0, 0.0]),
        z_path=np.array([100.0, 100.0, 100.0]),
        q11_path=np.array([1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j]),
        q12_path=np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        q21_path=np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        q22_path=np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]),
        amp_path=np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]),
        tau_path=np.array([0.0, 1.0, 2.0]),
        kmah_path=np.array([0.0, 0.0, 0.0]),
        c_path=np.array([1500.0, 1500.0, 1500.0]),
        phi_path=np.array([0.0, 0.0, 0.0]),
        launch_elev_deg=0.0,
        launch_azim_deg=0.0,
        num_top_bounces=0,
        num_bottom_bounces=0,
        tolerance_m=0.1,
        c_src=1500.0,
        d_elev=0.1,
        d_azim=0.1,
    )

    assert arrival is not None
    assert np.isclose(arrival.delay.real, 1.75)
    assert np.isclose(abs(np.angle(arrival.amplitude)), np.pi)


def test_nearest_segment_arrival_uses_segment_endpoint_bounce_counts():
    arrival = _nearest_segment_arrival(
        receiver_x=15.0,
        receiver_y=0.0,
        receiver_z=100.0,
        x_path=np.array([0.0, 10.0, 20.0]),
        y_path=np.array([0.0, 0.0, 0.0]),
        z_path=np.array([100.0, 100.0, 100.0]),
        q11_path=np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]),
        q12_path=np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        q21_path=np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        q22_path=np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]),
        amp_path=np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]),
        tau_path=np.array([0.0, 1.0, 2.0]),
        kmah_path=np.array([0.0, 0.0, 0.0]),
        c_path=np.array([1500.0, 1500.0, 1500.0]),
        phi_path=np.array([0.0, 0.0, 0.0]),
        launch_elev_deg=0.0,
        launch_azim_deg=0.0,
        num_top_bounces=9,
        num_bottom_bounces=8,
        tolerance_m=0.1,
        c_src=1500.0,
        d_elev=0.1,
        d_azim=0.1,
        top_bounce_path=np.array([0, 1, 1]),
        bottom_bounce_path=np.array([0, 0, 2]),
    )

    assert arrival is not None
    assert arrival.num_top_bounces == 1
    assert arrival.num_bottom_bounces == 2


def _synthetic_arrival(
    amplitude: complex,
    delay: float,
    angle: float = 0.0,
    phase_rad: float | None = None,
) -> Bellhop3DArrival:
    return Bellhop3DArrival(
        amplitude=amplitude,
        delay=complex(delay, 0.0),
        src_decl_angle=angle,
        src_azim_angle=angle,
        rcvr_decl_angle=angle,
        rcvr_azim_angle=angle,
        num_top_bounces=0,
        num_bottom_bounces=0,
        distance_to_receiver=1.0,
        phase_rad=phase_rad,
    )


def test_add_arrival_3d_groups_adjacent_phase_matched_arrivals():
    arrivals = [_synthetic_arrival(1.0 + 0.0j, 1.0, angle=10.0)]

    _append_arrival_at_style(
        arrivals,
        _synthetic_arrival(2.0 + 0.0j, 1.00001, angle=20.0),
        omega=2.0 * np.pi * 100.0,
        max_arrivals=10,
    )

    assert len(arrivals) == 1
    assert np.isclose(abs(arrivals[0].amplitude), 3.0)
    assert np.isclose(arrivals[0].delay.real, (1.0 + 2.0 * 1.00001) / 3.0)
    assert np.isclose(arrivals[0].src_decl_angle, 50.0 / 3.0)


def test_add_arrival_3d_uses_unwrapped_at_phase_tolerance():
    arrivals = [_synthetic_arrival(1.0 + 0.0j, 1.0, phase_rad=0.0)]

    _append_arrival_at_style(
        arrivals,
        _synthetic_arrival(1.0 + 0.0j, 1.00001, phase_rad=2.0 * np.pi),
        omega=2.0 * np.pi * 100.0,
        max_arrivals=10,
    )

    assert len(arrivals) == 2


def test_add_arrival_3d_preserves_unwrapped_phase_when_grouping():
    arrivals = [
        _synthetic_arrival(
            np.exp(1j * np.deg2rad(240.0)),
            1.0,
            phase_rad=np.deg2rad(240.0),
        )
    ]

    _append_arrival_at_style(
        arrivals,
        _synthetic_arrival(
            2.0 * np.exp(1j * np.deg2rad(240.01)),
            1.00001,
            phase_rad=np.deg2rad(240.01),
        ),
        omega=2.0 * np.pi * 100.0,
        max_arrivals=10,
    )
    record = _arrival_record(arrivals)

    assert len(arrivals) == 1
    assert np.isclose(abs(arrivals[0].amplitude), 3.0)
    assert np.isclose(np.angle(arrivals[0].amplitude), np.deg2rad(-120.0))
    assert np.isclose(record.phase_angles[0], np.deg2rad(240.0))


def test_add_arrival_3d_ignores_weak_tail_that_would_break_adjacent_grouping():
    arrivals = [_synthetic_arrival(1.0 + 0.0j, 1.0, angle=10.0)]

    _append_arrival_at_style(
        arrivals,
        _synthetic_arrival(1.0e-12 + 0.0j, 1.1, angle=99.0),
        omega=2.0 * np.pi * 100.0,
        max_arrivals=10,
    )
    _append_arrival_at_style(
        arrivals,
        _synthetic_arrival(2.0 + 0.0j, 1.00001, angle=20.0),
        omega=2.0 * np.pi * 100.0,
        max_arrivals=10,
    )

    assert len(arrivals) == 1
    assert np.isclose(abs(arrivals[0].amplitude), 3.0)
    assert np.isclose(arrivals[0].src_decl_angle, 50.0 / 3.0)


def test_add_arrival_3d_replaces_weakest_when_capacity_is_full():
    arrivals = [
        _synthetic_arrival(1.0 + 0.0j, 1.0),
        _synthetic_arrival(2.0 + 0.0j, 2.0),
    ]

    _append_arrival_at_style(
        arrivals,
        _synthetic_arrival(3.0 + 0.0j, 3.0),
        omega=2.0 * np.pi * 100.0,
        max_arrivals=2,
    )

    assert sorted(abs(arrival.amplitude) for arrival in arrivals) == [2.0, 3.0]


def test_prune_negligible_arrivals_removes_numerical_weak_tail():
    arrivals = [
        _synthetic_arrival(1.0 + 0.0j, 1.0),
        _synthetic_arrival(1.0e-12 + 0.0j, 1.1),
        _synthetic_arrival(1.0e-8 + 0.0j, 1.2),
    ]

    pruned = _prune_negligible_arrivals(arrivals, relative_floor=1.0e-10)

    assert len(pruned) == 2
    assert [arrival.delay.real for arrival in pruned] == [1.0, 1.2]
