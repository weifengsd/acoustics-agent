import numpy as np
import pytest

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
from pyacoustics.solvers.bellhop.eigenrays import _default_angle_samples, compute_eigenrays, write_eigenrays_ray_ascii
from pyacoustics.solvers.bellhop.ray_parity import compute_native_eigenray_records
from pyacoustics.solvers.external_io import read_at_ray_records


def _iso_config():
    return SimulationConfig(
        project="Eigenray iso",
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


def test_compute_eigenrays_finds_direct_path_in_iso_waveguide():
    config = _iso_config()

    eigenrays = compute_eigenrays(
        config,
        1000.0,
        100.0,
        angle_samples=41,
        tolerance_m=0.05,
    )

    assert eigenrays
    direct = eigenrays[0]
    assert abs(direct.launch_angle) < 1.0e-6
    assert abs(direct.receiver_angle) < 1.0e-6
    assert direct.miss_distance <= 0.05
    assert direct.num_top_bounces == 0
    assert direct.num_bottom_bounces == 0
    assert np.isclose(direct.delay, 1000.0 / 1500.0, atol=1.0e-3)
    assert np.isclose(abs(direct.amplitude), 1.0 / np.sqrt(1000.0), atol=1.0e-6)
    assert np.isclose(direct.r_path[-1], 1000.0, atol=10.0)


def test_simulation_exposes_native_eigenray_search():
    sim = Simulation(_iso_config())

    eigenrays = sim.run_eigenrays(
        1000.0,
        100.0,
        angle_samples=41,
        tolerance_m=0.05,
    )

    assert len(eigenrays) >= 1
    assert eigenrays[0].miss_distance <= 0.05


def test_compute_eigenrays_validates_options():
    config = _iso_config()

    with pytest.raises(ValueError, match="range_m"):
        compute_eigenrays(config, -1.0, 100.0)
    with pytest.raises(ValueError, match="tolerance_m"):
        compute_eigenrays(config, 1000.0, 100.0, tolerance_m=0.0)
    with pytest.raises(ValueError, match="angle_samples"):
        compute_eigenrays(config, 1000.0, 100.0, angle_samples=1)


def test_default_eigenray_angle_samples_expand_with_fan_width():
    config = _iso_config()
    config.solver.angles = (-20.0, 20.0)
    config.solver.num_beams = 3

    assert _default_angle_samples(config) == 401

    config.solver.num_beams = 1201
    assert _default_angle_samples(config) == 1201

    config.solver.angles = (-200.0, 200.0)
    config.solver.num_beams = 3
    assert _default_angle_samples(config) == 2001


def test_compute_eigenrays_keeps_distinct_bounce_families():
    config = _iso_config()
    config.environment.bottom.depth = 100.0
    config.environment.ssp.data = [
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=100.0, c=1500.0),
    ]
    config.geometry.source.depths = [35.0]
    config.geometry.receivers.depths = [65.0]
    config.solver.angles = (-20.0, 20.0)
    config.solver.num_beams = 101
    config.solver.step_size = 2.5

    eigenrays = compute_eigenrays(
        config,
        1000.0,
        65.0,
        angle_samples=101,
        tolerance_m=0.1,
    )

    keys = {
        (
            ray.num_top_bounces,
            ray.num_bottom_bounces,
            np.sign(ray.launch_angle),
            np.sign(ray.receiver_angle),
        )
        for ray in eigenrays
    }
    assert len(eigenrays) == len(keys)
    assert len(eigenrays) >= 5
    assert max(ray.miss_distance for ray in eigenrays) <= 0.1


def test_write_eigenrays_ray_ascii_roundtrip(tmp_path):
    config = _iso_config()
    eigenrays = compute_eigenrays(
        config,
        1000.0,
        100.0,
        angle_samples=41,
        tolerance_m=0.05,
    )
    ray_path = tmp_path / "native_eigenray.ray"

    write_eigenrays_ray_ascii(ray_path, config, eigenrays)
    records = read_at_ray_records(ray_path)

    assert len(records) == len(eigenrays)
    assert records
    assert np.isclose(records[0].launch_angle_deg, eigenrays[0].launch_angle)
    assert records[0].num_top_bounces == eigenrays[0].num_top_bounces
    assert records[0].num_bottom_bounces == eigenrays[0].num_bottom_bounces
    np.testing.assert_allclose(records[0].ranges_m, eigenrays[0].r_path, atol=1.0e-6)
    np.testing.assert_allclose(records[0].depths_m, eigenrays[0].z_path, atol=1.0e-6)


def test_simulation_run_eigenrays_writes_at_ray_file(tmp_path):
    sim = Simulation(_iso_config())
    ray_path = tmp_path / "sim_eigenray.ray"

    eigenrays = sim.run_eigenrays(
        1000.0,
        100.0,
        angle_samples=41,
        tolerance_m=0.05,
        save_path=ray_path,
    )
    records = read_at_ray_records(ray_path)

    assert len(eigenrays) == len(records)
    assert records[0].ranges_m[-1] >= 1000.0 - sim.config.solver.step_size


def test_simulation_run_writes_native_2d_ray_file(tmp_path):
    sim = Simulation(_iso_config())
    sim.config.environment.bottom.depth = 100.0
    sim.config.environment.ssp.data = [
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=100.0, c=1500.0),
    ]
    sim.config.geometry.source.depths = [50.0]
    sim.config.geometry.receivers.ranges = [500.0]
    sim.config.solver.step_size = 5.0
    sim.config.solver.angles = (-20.0, 20.0)
    sim.config.solver.num_beams = 3
    ray_path = tmp_path / "native_ray.ray"

    rays = sim.run(save_ray_path=ray_path)
    records = read_at_ray_records(ray_path)

    assert len(records) == len(rays) == 3
    assert {record.coordinate_system for record in records} == {"rz"}
    np.testing.assert_allclose([record.launch_angle_deg for record in records], [-20.0, 0.0, 20.0])
    assert records[0].num_top_bounces > 0
    assert records[2].num_bottom_bounces > 0
    np.testing.assert_allclose(records[0].ranges_m, rays[0][0])
    np.testing.assert_allclose(records[0].depths_m, rays[0][1])


def test_native_at_style_eigenray_records_follow_beam_arrival_angles():
    config = _iso_config()
    config.environment.bottom.depth = 100.0
    config.environment.ssp.data = [
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=100.0, c=1500.0),
    ]
    config.geometry.source.depths = [35.0]
    config.geometry.receivers.depths = [65.0]
    config.solver.angles = (-20.0, 20.0)
    config.solver.num_beams = 101
    config.solver.step_size = 2.5

    records = compute_native_eigenray_records(config, max_arrivals=20)

    assert records
    assert all(record.ranges_m[-1] >= config.geometry.receivers.ranges[0] for record in records)
    assert {round(record.launch_angle_deg, 6) for record in records} <= {
        round(angle, 6) for angle in np.linspace(-20.0, 20.0, 101)
    }


def test_native_at_style_eigenray_records_preserve_raw_hits_before_arrival_merge():
    config = _iso_config()
    config.geometry.receivers.ranges = [500.0, 1000.0]
    config.geometry.receivers.depths = [100.0]
    config.solver.angles = (0.0, 0.0)
    config.solver.num_beams = 1
    config.solver.step_size = 10.0

    records = compute_native_eigenray_records(config, max_arrivals=10)

    assert len(records) == 2
    assert [round(record.launch_angle_deg, 6) for record in records] == [0.0, 0.0]
    assert records[0].ranges_m[-1] >= 500.0
    assert records[0].ranges_m[-1] < records[1].ranges_m[-1]
    assert records[1].ranges_m[-1] >= 1000.0
