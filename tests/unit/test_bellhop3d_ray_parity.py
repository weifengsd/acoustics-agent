from pathlib import Path

import numpy as np
import pytest

from pyacoustics.benchmarks.at_presets import _munk_slanted_terrain_bellhop3d_config
from pyacoustics.schema import BellhopConfig
from pyacoustics.simulation import Simulation
from pyacoustics.solvers.bellhop3d.core import PyBellhop3D
from pyacoustics.solvers.external_io import ATRayRecord
from pyacoustics.solvers.bellhop3d.ray_parity import (
    Bellhop3DRayParityMetrics,
    bellhop3d_ray_benchmark_row,
    compare_ray_records_3d,
    run_bellhop3d_ray_parity,
)


def _at_bin_path() -> Path:
    return Path(__file__).resolve().parents[3] / "at" / "bin"


def test_bellhop3d_default_ray_box_uses_at_style_receiver_range_margin():
    config = _munk_slanted_terrain_bellhop3d_config()
    config.geometry.receivers.ranges = [0.0, 1000.0]
    config.solver.box_x = None
    config.solver.box_y = None

    solver = PyBellhop3D(config)

    assert np.isclose(solver.r_max, 1100.0)
    assert np.isclose(solver.box_x, 1100.0)
    assert np.isclose(solver.box_y, 1100.0)


def test_compare_ray_records_3d_limits_by_horizontal_range_from_source(monkeypatch):
    config = _munk_slanted_terrain_bellhop3d_config()
    config.solver.elevation_angles = (0.0, 0.0)
    config.solver.azimuth_angles = (60.0, 60.0)
    config.solver.num_elevation_beams = 1
    config.solver.num_azimuth_beams = 1

    ranges = np.asarray([0.0, 5.0, 10.0], dtype=np.float64)
    x = ranges * 0.5
    y = ranges * np.sin(np.deg2rad(60.0))
    z = 100.0 + ranges
    native_ray = (
        x,
        y,
        z,
        z,
        z,
        z,
        z,
        np.ones_like(ranges),
        ranges,
        np.zeros_like(ranges),
        np.full_like(ranges, 1500.0),
        0,
        0,
        0.0,
    )
    monkeypatch.setattr(
        "pyacoustics.solvers.bellhop3d.ray_parity._native_ray_lookup",
        lambda _config: {(0.0, 60.0): native_ray},
    )
    records = [
        ATRayRecord(
            launch_angle_deg=0.0,
            launch_azimuth_deg=60.0,
            coordinate_system="xyz",
            ranges_m=x,
            y_m=y,
            depths_m=z,
            amplitudes=np.ones_like(ranges, dtype=np.complex128),
            num_top_bounces=0,
            num_bottom_bounces=0,
        )
    ]

    metrics = compare_ray_records_3d(records, config, max_range_m=7.5)

    assert metrics.matched_ray_count == 1
    assert metrics.sample_count == 2
    assert metrics.depth_max_abs_m == 0.0
    assert metrics.y_max_abs_m == 0.0


def test_compare_ray_records_3d_selects_matching_source_depth(monkeypatch):
    config = _munk_slanted_terrain_bellhop3d_config()
    config.geometry.source.depths = [80.0, 120.0]
    config.solver.elevation_angles = (0.0, 0.0)
    config.solver.azimuth_angles = (0.0, 0.0)
    config.solver.num_elevation_beams = 1
    config.solver.num_azimuth_beams = 1

    ranges = np.asarray([0.0, 10.0, 20.0], dtype=np.float64)
    shallow_ray = (
        ranges,
        np.zeros_like(ranges),
        np.asarray([80.0, 80.0, 80.0], dtype=np.float64),
        ranges,
        ranges,
        ranges,
        ranges,
        np.ones_like(ranges),
        ranges,
        np.zeros_like(ranges),
        np.full_like(ranges, 1500.0),
        0,
        0,
        0.0,
    )
    deep_ray = (
        ranges,
        np.zeros_like(ranges),
        np.asarray([120.0, 121.0, 122.0], dtype=np.float64),
        ranges,
        ranges,
        ranges,
        ranges,
        np.ones_like(ranges),
        ranges,
        np.zeros_like(ranges),
        np.full_like(ranges, 1500.0),
        0,
        0,
        0.0,
    )
    monkeypatch.setattr(
        "pyacoustics.solvers.bellhop3d.ray_parity._native_ray_lookup",
        lambda _config: {(0.0, 0.0): [(0.0, 0.0, 80.0, shallow_ray), (0.0, 0.0, 120.0, deep_ray)]},
    )
    records = [
        ATRayRecord(
            launch_angle_deg=0.0,
            launch_azimuth_deg=0.0,
            coordinate_system="xyz",
            ranges_m=ranges,
            y_m=np.zeros_like(ranges),
            depths_m=np.asarray([120.0, 121.0, 122.0], dtype=np.float64),
            amplitudes=np.ones_like(ranges, dtype=np.complex128),
            num_top_bounces=0,
            num_bottom_bounces=0,
        )
    ]

    metrics = compare_ray_records_3d(records, config)

    assert metrics.matched_ray_count == 1
    assert metrics.depth_max_abs_m == 0.0


def test_bellhop3d_munk_slanted_terrain_ray_parity_smoke():
    at_bin = _at_bin_path()
    if not ((at_bin / "bellhop3d").exists() or (at_bin / "bellhop3d.exe").exists()):
        pytest.skip("AT bellhop3d executable is not available")

    metrics = run_bellhop3d_ray_parity(
        _munk_slanted_terrain_bellhop3d_config(),
        at_bin_path=at_bin,
        max_range_m=10000.0,
    )
    row = bellhop3d_ray_benchmark_row("bellhop3d-munk-slanted-terrain-rays", metrics)

    assert metrics.at_ray_count == 20
    assert metrics.matched_ray_count == 20
    assert metrics.unmatched_at_ray_count == 0
    assert metrics.sample_count > 20000
    assert metrics.depth_p90_abs_m < 2.5
    assert metrics.y_p90_abs_m < 1.0e-9
    assert metrics.bounce_match_fraction >= 0.90
    assert row["kind"] == "bellhop3d_ray"
    assert row["depth_p90_abs_m"] == metrics.depth_p90_abs_m


def test_simulation_forwards_bellhop3d_ray_parity_options(monkeypatch):
    from pyacoustics.solvers.bellhop3d import ray_parity as ray_parity_module

    captured = {}

    def fake_run_bellhop3d_ray_parity(config, **kwargs):
        captured["solver_type"] = config.solver.type
        captured.update(kwargs)
        return Bellhop3DRayParityMetrics(
            at_ray_count=1,
            native_ray_count=1,
            matched_ray_count=1,
            unmatched_at_ray_count=0,
            sample_count=2,
            depth_mae_m=0.0,
            depth_p90_abs_m=0.0,
            depth_max_abs_m=0.0,
            y_mae_m=0.0,
            y_p90_abs_m=0.0,
            y_max_abs_m=0.0,
            launch_angle_mae_deg=0.0,
            launch_azimuth_mae_deg=0.0,
            bounce_match_count=1,
            bounce_match_fraction=1.0,
            top_bounce_match_fraction=1.0,
            bottom_bounce_match_fraction=1.0,
        )

    monkeypatch.setattr(ray_parity_module, "run_bellhop3d_ray_parity", fake_run_bellhop3d_ray_parity)

    metrics = Simulation(_munk_slanted_terrain_bellhop3d_config()).compare_bellhop3d_rays_with_at(
        at_bin_path="/opt/at/bin",
        run_type="R",
        max_range_m=10000.0,
    )

    assert metrics.matched_ray_count == 1
    assert captured["solver_type"] == "bellhop3d"
    assert captured["at_bin_path"] == "/opt/at/bin"
    assert captured["run_type"] == "R"
    assert captured["max_range_m"] == 10000.0


def test_simulation_bellhop3d_ray_parity_rejects_non_bellhop3d():
    config = _munk_slanted_terrain_bellhop3d_config()
    config.solver = BellhopConfig(type="bellhop", angles=(-1.0, 1.0), num_beams=3)

    with pytest.raises(NotImplementedError, match="Bellhop3D ray"):
        Simulation(config).compare_bellhop3d_rays_with_at()


def test_simulation_exposes_bellhop3d_ray_parity_smoke():
    at_bin = _at_bin_path()
    if not ((at_bin / "bellhop3d").exists() or (at_bin / "bellhop3d.exe").exists()):
        pytest.skip("AT bellhop3d executable is not available")

    metrics = Simulation(_munk_slanted_terrain_bellhop3d_config()).compare_bellhop3d_rays_with_at(
        at_bin_path=at_bin,
        max_range_m=10000.0,
    )

    assert metrics.at_ray_count == 20
    assert metrics.matched_ray_count == 20
    assert metrics.unmatched_at_ray_count == 0
    assert metrics.depth_p90_abs_m < 2.5
