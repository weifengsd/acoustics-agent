import sys
import os
import numpy as np
from pathlib import Path

# Insert package root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyacoustics.schema import Bathymetry3DModel
from pyacoustics.simulation import Simulation, _mask_bellhop3d_pressure_below_bathymetry
from pyacoustics.solvers.bellhop3d.beam import receiver_grid_segment_range
from pyacoustics.solvers.bellhop3d.core import PyBellhop3D
from pyacoustics.solvers.external_io import read_at_ray_records, read_at_shd_full

def test_3d_slanted_terrain_horizontal_refraction():
    """
    Kinematics Test:
    Verify that 3D rays experience active horizontal refraction (out-of-plane deviation)
    when propagating over a slanted three-dimensional bathymetry (bty) surface.
    """
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)

    # Configure elevation/azimuth sweeps
    sim.config.solver.num_elevation_beams = 5
    sim.config.solver.num_azimuth_beams = 5

    # Initialize the high-level PyBellhop3D solver using official API
    solver = PyBellhop3D(sim.config)
    rays_3d = solver.run_ray_tracing()

    assert len(rays_3d) > 0, "Ray tracer must yield active rays"

    # Find a ray launched at a non-zero azimuth angle to examine horizontal refraction.
    # Current rays_3d tuples include the full q-matrix and acoustic state paths.
    horizontal_refraction_detected = False

    for ray in rays_3d:
        x_path, y_path, z_path = ray[:3]

        # Check initial azimuth from launch angle logic
        # In a flat bottom 2D plane, y = x * tan(azim).
        # We check if there is an active bottom-driven out-of-plane deviation.
        if len(x_path) < 10:
            continue

        # Get launching bearing angle from initial trajectory step
        dx = x_path[1] - x_path[0]
        dy = y_path[1] - y_path[0]
        if abs(dx) < 1e-6:
            continue

        tan_azim = dy / dx
        expected_y_flat = x_path * tan_azim
        lateral_deviation = y_path - expected_y_flat

        max_deviation = np.max(np.abs(lateral_deviation))
        if max_deviation > 0.05:
            horizontal_refraction_detected = True
            print(f"Verified Horizontal Refraction: Out-of-plane deviation = {max_deviation:.4f} meters")
            break

    assert horizontal_refraction_detected, "3D slanted terrain must cause physical horizontal refraction away from initial azimuth plane"


def test_3d_ray_box_limits_path_length():
    """Bellhop3D box_x should stop a ray before the receiver range limit."""
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)
    sim.config.geometry.source.depths = [1000.0]
    sim.config.geometry.receivers.ranges = [0.0, 10000.0]
    sim.config.solver.num_elevation_beams = 1
    sim.config.solver.elevation_angles = [0.0, 0.0]
    sim.config.solver.num_azimuth_beams = 1
    sim.config.solver.azimuth_angles = [0.0, 0.0]
    sim.config.solver.box_x = 1200.0
    sim.config.solver.box_y = 10000.0
    sim.config.solver.box_z = 10000.0

    ray = PyBellhop3D(sim.config).run_ray_tracing()[0]
    x_path = ray[0]

    assert np.max(np.abs(x_path)) <= 1200.0 + 1e-6


def test_3d_coherent_tl_receiver_grid_mode_uses_at_layout():
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)
    sim.config.solver.num_elevation_beams = 1
    sim.config.solver.elevation_angles = [0.0, 0.0]
    sim.config.solver.num_azimuth_beams = 1
    sim.config.solver.azimuth_angles = [0.0, 0.0]
    sim.config.solver.step_size = 50.0
    sim.config.geometry.receivers.ranges = [1000.0]

    tl = sim.compute_coherent_tl_3d(num_x=4, num_z=5, grid_mode="receiver")

    assert tl.shape == (1, 5, 4)
    assert sim._coherent_tl_receiver_grid_3d[0].shape == tl.shape
    assert np.all(np.isfinite(tl))


def test_receiver_grid_range_bracket_is_source_relative():
    r0, r1 = receiver_grid_segment_range(100.0, 50.0, 200.0, 50.0, source_x=100.0, source_y=50.0)

    assert np.isclose(r0, 0.0)
    assert np.isclose(r1, 100.0)


def test_3d_receiver_grid_pressure_masks_receivers_below_bathymetry():
    pressure = np.ones((1, 1, 1, 1, 3, 2), dtype=np.complex128)
    bathymetry = Bathymetry3DModel(
        x=[0.0, 1000.0],
        y=[0.0, 1000.0],
        z=[
            [100.0, 100.0],
            [200.0, 200.0],
        ],
    )

    _mask_bellhop3d_pressure_below_bathymetry(
        pressure,
        source_x=np.array([0.0]),
        source_y=np.array([0.0]),
        theta_grid_deg=np.array([0.0]),
        r_grid=np.array([0.0, 1000.0]),
        z_grid=np.array([50.0, 100.0, 250.0]),
        bathymetry_3d=bathymetry,
    )

    assert pressure[0, 0, 0, 0, 0, 0] == 1.0
    assert pressure[0, 0, 0, 0, 1, 0] == 1.0
    assert pressure[0, 0, 0, 0, 2, 0] == 0.0
    assert pressure[0, 0, 0, 0, 0, 1] == 1.0
    assert pressure[0, 0, 0, 0, 1, 1] == 1.0
    assert pressure[0, 0, 0, 0, 2, 1] == 0.0


def test_3d_receiver_grid_pressure_keeps_receivers_on_bathymetry():
    pressure = np.ones((1, 1, 1, 1, 2, 1), dtype=np.complex128)
    bathymetry = Bathymetry3DModel(
        x=[0.0, 1000.0],
        y=[0.0, 1000.0],
        z=[
            [100.0, 100.0],
            [200.0, 200.0],
        ],
    )

    _mask_bellhop3d_pressure_below_bathymetry(
        pressure,
        source_x=np.array([0.0]),
        source_y=np.array([0.0]),
        theta_grid_deg=np.array([0.0]),
        r_grid=np.array([1000.0]),
        z_grid=np.array([200.0, 200.0001]),
        bathymetry_3d=bathymetry,
    )

    assert pressure[0, 0, 0, 0, 0, 0] == 1.0
    assert pressure[0, 0, 0, 0, 1, 0] == 0.0


def test_3d_coherent_tl_receiver_grid_writes_at_shd(tmp_path):
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)
    sim.config.solver.num_elevation_beams = 1
    sim.config.solver.elevation_angles = [0.0, 0.0]
    sim.config.solver.num_azimuth_beams = 1
    sim.config.solver.azimuth_angles = [0.0, 0.0]
    sim.config.solver.step_size = 50.0
    theta = np.array([0.0, 30.0], dtype=np.float64)
    ranges = np.array([0.0, 500.0, 1000.0], dtype=np.float64)
    depths = np.array([0.0, 500.0, 1000.0, 1500.0], dtype=np.float64)
    shd_path = tmp_path / "native_receiver_grid.shd"

    tl = sim.compute_coherent_tl_3d_receiver_grid(
        theta_grid_deg=theta,
        r_grid=ranges,
        z_grid=depths,
        save_shd_path=shd_path,
    )
    shade = read_at_shd_full(shd_path)
    pressure_abs = np.abs(shade.pressure[:, 0])
    tl_from_shd = -20.0 * np.log10(np.maximum(pressure_abs, 1e-20))
    tl_from_shd[pressure_abs < 1e-20] = 200.0
    pressure_grid, _, _, _ = sim._coherent_pressure_receiver_grid_3d

    assert shade.pressure.shape == (2, 1, 4, 3)
    np.testing.assert_allclose(shade.bearings_deg, theta)
    np.testing.assert_allclose(shade.source_depths_m, sim.config.geometry.source.depths)
    np.testing.assert_allclose(shade.receiver_depths_m, depths)
    np.testing.assert_allclose(shade.receiver_ranges_m, ranges)
    np.testing.assert_allclose(tl_from_shd, tl, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(shade.pressure[:, 0], pressure_grid)
    assert np.max(np.abs(np.imag(shade.pressure))) > 0.0


def test_3d_coherent_tl_receiver_grid_writes_multi_source_shd(tmp_path):
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)
    sim.config.geometry.source.depths = [900.0, 1100.0]
    sim.config.solver.num_elevation_beams = 1
    sim.config.solver.elevation_angles = [0.0, 0.0]
    sim.config.solver.num_azimuth_beams = 1
    sim.config.solver.azimuth_angles = [0.0, 0.0]
    sim.config.solver.step_size = 50.0
    theta = np.array([0.0, 30.0], dtype=np.float64)
    ranges = np.array([0.0, 500.0, 1000.0], dtype=np.float64)
    depths = np.array([900.0, 1000.0, 1100.0, 1200.0], dtype=np.float64)
    shd_path = tmp_path / "native_receiver_grid_multi_source.shd"

    tl = sim.compute_coherent_tl_3d_receiver_grid(
        theta_grid_deg=theta,
        r_grid=ranges,
        z_grid=depths,
        save_shd_path=shd_path,
    )
    shade = read_at_shd_full(shd_path)
    pressure_grid, _, _, _ = sim._coherent_pressure_receiver_grid_3d
    pressure_abs = np.abs(shade.pressure)
    tl_from_shd = -20.0 * np.log10(np.maximum(pressure_abs, 1e-20))
    tl_from_shd[pressure_abs < 1e-20] = 200.0

    assert tl.shape == (2, 2, 4, 3)
    assert shade.pressure.shape == (2, 2, 4, 3)
    np.testing.assert_allclose(shade.bearings_deg, theta)
    np.testing.assert_allclose(shade.source_depths_m, sim.config.geometry.source.depths)
    np.testing.assert_allclose(shade.receiver_depths_m, depths)
    np.testing.assert_allclose(shade.receiver_ranges_m, ranges)
    np.testing.assert_allclose(shade.pressure, pressure_grid)
    np.testing.assert_allclose(tl_from_shd, tl, rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(np.imag(shade.pressure))) > 0.0


def test_3d_coherent_tl_receiver_grid_writes_source_xy_shd(tmp_path):
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)
    sim.config.geometry.source.x = [0.0, 100.0]
    sim.config.geometry.source.y = [0.0]
    sim.config.solver.num_elevation_beams = 1
    sim.config.solver.elevation_angles = [0.0, 0.0]
    sim.config.solver.num_azimuth_beams = 1
    sim.config.solver.azimuth_angles = [0.0, 0.0]
    sim.config.solver.step_size = 50.0
    theta = np.array([0.0], dtype=np.float64)
    ranges = np.array([0.0, 500.0, 1000.0], dtype=np.float64)
    depths = np.array([900.0, 1000.0, 1100.0], dtype=np.float64)
    shd_path = tmp_path / "native_receiver_grid_source_xy.shd"

    tl = sim.compute_coherent_tl_3d_receiver_grid(
        theta_grid_deg=theta,
        r_grid=ranges,
        z_grid=depths,
        save_shd_path=shd_path,
    )
    shade = read_at_shd_full(shd_path)
    pressure_grid, _, _, _ = sim._coherent_pressure_receiver_grid_3d
    pressure_abs = np.abs(shade.pressure_full)
    tl_from_shd = -20.0 * np.log10(np.maximum(pressure_abs, 1e-20))
    tl_from_shd[pressure_abs < 1e-20] = 200.0

    assert tl.shape == (2, 1, 1, 1, 3, 3)
    assert shade.pressure_full.shape == pressure_grid.shape
    np.testing.assert_allclose(shade.source_x_m, [0.0, 100.0])
    np.testing.assert_allclose(shade.source_y_m, [0.0])
    np.testing.assert_allclose(shade.bearings_deg, theta)
    np.testing.assert_allclose(shade.source_depths_m, sim.config.geometry.source.depths)
    np.testing.assert_allclose(shade.receiver_depths_m, depths)
    np.testing.assert_allclose(shade.receiver_ranges_m, ranges)
    np.testing.assert_allclose(shade.pressure_full, pressure_grid)
    np.testing.assert_allclose(tl_from_shd, tl, rtol=1e-6, atol=1e-6)


def test_3d_run_writes_at_xyz_ray_file(tmp_path):
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)
    sim.config.solver.num_elevation_beams = 2
    sim.config.solver.elevation_angles = [-20.0, 20.0]
    sim.config.solver.num_azimuth_beams = 2
    sim.config.solver.azimuth_angles = [0.0, 30.0]
    sim.config.solver.step_size = 50.0
    sim.config.geometry.receivers.ranges = [10000.0]
    ray_path = tmp_path / "native3d.ray"

    rays = sim.run(save_ray_path=ray_path)
    records = read_at_ray_records(ray_path)

    assert len(records) == len(rays) == 4
    assert all(record.coordinate_system == "xyz" for record in records)
    assert {record.launch_azimuth_deg for record in records} == {0.0, 30.0}
    assert {record.launch_angle_deg for record in records} == {-20.0, 20.0}
    assert any(record.num_top_bounces > 0 for record in records)
    assert any(record.num_bottom_bounces > 0 for record in records)
    np.testing.assert_allclose(records[0].ranges_m, rays[0][0])
    np.testing.assert_allclose(records[0].y_m, rays[0][1])
    np.testing.assert_allclose(records[0].depths_m, rays[0][2])
