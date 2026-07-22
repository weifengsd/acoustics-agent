import os
from pathlib import Path

import numpy as np
import pytest

from pyacoustics.config import ConfigLoader
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
from pyacoustics.solvers.bellhop.arrival_parity import (
    arrival_match_table,
    compare_arrival_data,
    compute_native_arrival_data,
    run_at_arrival_data,
)
from pyacoustics.solvers.bellhop.ray_parity import (
    compare_ray_records,
    compute_native_eigenray_records,
    compute_native_ray_records,
    run_at_ray_records,
)
from pyacoustics.solvers.bellhop.tl_parity import (
    compare_tl_grids,
    compute_native_tl_grid,
    run_at_tl_grid,
)


def _at_bin_or_skip():
    bin_path = os.environ.get("AT_BIN_PATH")
    if not bin_path:
        pytest.skip("AT_BIN_PATH is not set")
    bin_path = Path(bin_path)
    bellhop = bin_path / "bellhop"
    if not bellhop.exists():
        bellhop = bin_path / "bellhop.exe"
    if not bellhop.exists():
        pytest.skip("AT bellhop executable is not available")
    return bin_path


def _iso_direct_config():
    return SimulationConfig(
        project="bellhop parity iso direct",
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
        solver=BellhopConfig(type="bellhop", angles=(-8.0, 8.0), num_beams=81, step_size=5.0),
    )


def _shallow_reflected_config():
    return SimulationConfig(
        project="bellhop parity shallow reflected",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=100.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[35.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[65.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-20.0, 20.0), num_beams=161, step_size=5.0),
    )


def _sloping_bathymetry_config():
    return SimulationConfig(
        project="bellhop parity sloping bathymetry",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=160.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=160.0,
                c_p=1700.0,
                density=1.5,
                attenuation_p=0.5,
                bathymetry=[(0.0, 100.0), (1000.0, 120.0), (2000.0, 140.0)],
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[35.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[65.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-20.0, 20.0), num_beams=201, step_size=2.5),
    )


def _sloping_altimetry_config():
    return SimulationConfig(
        project="bellhop parity sloping altimetry",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=120.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(
                type="vacuum",
                depth=0.0,
                altimetry=[(0.0, 0.0), (1000.0, 5.0), (2000.0, 10.0)],
            ),
            bottom=BoundaryCondition(type="rigid", depth=120.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[35.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[35.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-20.0, 20.0), num_beams=201, step_size=2.5),
    )


def _curvilinear_boundary_config():
    return SimulationConfig(
        project="bellhop parity curvilinear boundaries",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=170.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(
                type="vacuum",
                depth=0.0,
                altimetry=[
                    (0.0, 0.0),
                    (600.0, 5.0),
                    (1200.0, 0.0),
                    (2000.0, 4.0),
                ],
                altimetry_interpolation="C",
            ),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=170.0,
                c_p=1700.0,
                density=1.5,
                attenuation_p=0.5,
                bathymetry=[
                    (0.0, 110.0),
                    (600.0, 135.0),
                    (1200.0, 115.0),
                    (2000.0, 145.0),
                ],
                bathymetry_interpolation="C",
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[45.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[60.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-18.0, 18.0), num_beams=145, step_size=2.5),
    )


def _bottom_reflection_table_config():
    return SimulationConfig(
        project="bellhop parity bottom reflection table",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=100.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=100.0,
                c_p=1600.0,
                density=1.8,
                attenuation_p=0.0,
                reflection_table=[(0.0, 0.5, 0.0), (90.0, 0.5, 0.0)],
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[35.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[65.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-20.0, 20.0), num_beams=101, step_size=2.5),
    )


def _elastic_bottom_config():
    return SimulationConfig(
        project="bellhop parity elastic bottom",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=100.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=100.0,
                c_p=1700.0,
                c_s=700.0,
                density=1.8,
                attenuation_p=0.0,
                attenuation_s=0.0,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[35.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[65.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-20.0, 20.0), num_beams=101, step_size=2.5),
    )


def _long_format_bathymetry_config():
    return SimulationConfig(
        project="bellhop parity long format bathymetry",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=150.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=150.0,
                bathymetry=[
                    (0.0, 105.0),
                    (700.0, 125.0),
                    (1400.0, 115.0),
                    (2200.0, 135.0),
                ],
                bathymetry_interpolation="L",
                bathymetry_geoacoustics=[
                    (1650.0, 0.0, 1.6, 0.1, 0.0),
                    (1700.0, 650.0, 1.75, 0.15, 0.05),
                    (1750.0, 750.0, 1.85, 0.2, 0.05),
                    (1800.0, 800.0, 1.9, 0.25, 0.05),
                ],
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[40.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[60.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-18.0, 18.0), num_beams=121, step_size=2.5),
    )


def _munk_config():
    config = ConfigLoader.load_yaml(Path(__file__).resolve().parents[1] / "Munk" / "munk.yaml")
    config.solver.num_beams = 101
    config.solver.step_size = 25.0
    config.solver.angles = (-20.0, 20.0)
    config.geometry.receivers.ranges = [10000.0]
    config.geometry.receivers.depths = [1000.0]
    return config


@pytest.mark.parametrize(
    ("name", "factory", "delay_tolerance", "min_matches"),
    [
        ("iso_direct", _iso_direct_config, 0.01, 1),
        ("shallow_reflected", _shallow_reflected_config, 0.02, 1),
        ("sloping_bathymetry", _sloping_bathymetry_config, 0.02, 1),
        ("curvilinear_boundary", _curvilinear_boundary_config, 0.02, 5),
        ("elastic_bottom", _elastic_bottom_config, 0.02, 5),
        ("long_format_bathymetry", _long_format_bathymetry_config, 0.02, 5),
        ("munk", _munk_config, 0.02, 1),
    ],
)
def test_native_bellhop_arrival_parity_matrix(name, factory, delay_tolerance, min_matches):
    at_bin = _at_bin_or_skip()
    config = factory()

    at_data = run_at_arrival_data(config, at_bin_path=at_bin)
    native_data = compute_native_arrival_data(config, max_arrivals=80, beam_shape="hat")
    metrics = compare_arrival_data(at_data, native_data, delay_tolerance=delay_tolerance)
    matches = arrival_match_table(at_data, native_data, delay_tolerance=delay_tolerance)

    assert metrics.at_count > 0, name
    assert metrics.native_count > 0, name
    assert metrics.matched_count >= min_matches, name
    assert len(matches) == metrics.matched_count, name
    assert metrics.delay_max_abs <= delay_tolerance, name
    assert np.isfinite(metrics.launch_angle_mae), name
    assert np.isfinite(metrics.receiver_angle_mae), name
    assert any(row["top_bounce_match"] and row["bottom_bounce_match"] for row in matches), name


@pytest.mark.parametrize(
    ("name", "factory", "min_matches", "phase_mae_tolerance"),
    [
        ("iso_direct", _iso_direct_config, 1, 1.0e-6),
        ("shallow_reflected", _shallow_reflected_config, 5, 0.06),
        ("sloping_bathymetry", _sloping_bathymetry_config, 5, 0.03),
    ],
)
def test_native_bellhop_eigenray_arrival_delay_parity(name, factory, min_matches, phase_mae_tolerance):
    at_bin = _at_bin_or_skip()
    config = factory()

    at_data = run_at_arrival_data(config, at_bin_path=at_bin)
    native_data = compute_native_arrival_data(
        config,
        max_arrivals=80,
        method="eigenray",
        eigenray_tolerance_m=0.1,
        eigenray_angle_samples=101,
    )
    metrics = compare_arrival_data(at_data, native_data, delay_tolerance=0.02)

    assert metrics.at_count > 0, name
    assert metrics.native_count > 0, name
    assert metrics.matched_count >= min_matches, name
    assert metrics.delay_max_abs <= 0.02, name
    assert metrics.phase_mae_rad <= phase_mae_tolerance, name


def test_native_bellhop_eigenray_sloping_bathymetry_amplitude_and_bounce_parity():
    at_bin = _at_bin_or_skip()
    config = _sloping_bathymetry_config()
    config.solver.num_beams = 101
    config.solver.step_size = 2.5

    at_data = run_at_arrival_data(config, at_bin_path=at_bin)
    native_data = compute_native_arrival_data(
        config,
        max_arrivals=80,
        method="eigenray",
        eigenray_tolerance_m=0.2,
        eigenray_angle_samples=201,
    )
    metrics = compare_arrival_data(at_data, native_data, delay_tolerance=0.02)

    assert metrics.matched_count >= 5
    assert metrics.delay_mae < 0.001
    assert metrics.amplitude_median_relative_error < 0.1
    assert metrics.phase_mae_rad < 0.03
    assert metrics.bounce_match_fraction == 1.0


def test_native_bellhop_sloping_bathymetry_ray_path_parity():
    at_bin = _at_bin_or_skip()
    config = _sloping_bathymetry_config()

    at_records = run_at_ray_records(config, at_bin_path=at_bin)
    native_records = compute_native_ray_records(config)
    metrics = compare_ray_records(
        at_records,
        native_records,
        max_range_m=max(config.geometry.receivers.ranges),
        sample_count=101,
    )

    assert metrics.at_count == config.solver.num_beams
    assert metrics.native_count == config.solver.num_beams
    assert metrics.matched_count == config.solver.num_beams
    assert metrics.depth_mae < 0.005
    assert metrics.depth_max_abs < 0.01
    assert metrics.bounce_match_fraction > 0.85


def test_native_bellhop_munk_at_style_eigenray_ray_path_parity():
    at_bin = _at_bin_or_skip()
    config = _munk_config()

    at_records = run_at_ray_records(config, run_type="E", at_bin_path=at_bin)
    native_records = compute_native_eigenray_records(config)
    metrics = compare_ray_records(
        at_records,
        native_records,
        max_range_m=max(config.geometry.receivers.ranges),
        sample_count=101,
    )

    assert metrics.at_count == 5
    assert metrics.native_count == 5
    assert metrics.match_fraction == 1.0
    assert metrics.depth_mae < 5.0
    assert metrics.bounce_match_fraction == 1.0


def test_native_bellhop_sloping_altimetry_ray_path_parity():
    at_bin = _at_bin_or_skip()
    config = _sloping_altimetry_config()

    at_records = run_at_ray_records(config, at_bin_path=at_bin)
    native_records = compute_native_ray_records(config)
    metrics = compare_ray_records(
        at_records,
        native_records,
        max_range_m=max(config.geometry.receivers.ranges),
        sample_count=101,
    )

    assert metrics.at_count == config.solver.num_beams
    assert metrics.native_count == config.solver.num_beams
    assert metrics.matched_count == config.solver.num_beams
    assert metrics.depth_mae < 0.005
    assert metrics.depth_max_abs < 0.01
    assert metrics.bounce_match_fraction > 0.80


def test_native_bellhop_curvilinear_boundary_ray_path_parity():
    at_bin = _at_bin_or_skip()
    config = _curvilinear_boundary_config()

    at_records = run_at_ray_records(config, at_bin_path=at_bin)
    native_records = compute_native_ray_records(config)
    metrics = compare_ray_records(
        at_records,
        native_records,
        max_range_m=max(config.geometry.receivers.ranges),
        sample_count=101,
    )

    assert metrics.at_count == config.solver.num_beams
    assert metrics.native_count == config.solver.num_beams
    assert metrics.matched_count == config.solver.num_beams
    assert metrics.depth_mae < 0.01
    assert metrics.depth_max_abs < 0.3
    assert metrics.bounce_match_fraction > 0.75


def test_native_bellhop_elastic_bottom_ray_path_parity():
    at_bin = _at_bin_or_skip()
    config = _elastic_bottom_config()

    at_records = run_at_ray_records(config, at_bin_path=at_bin)
    native_records = compute_native_ray_records(config)
    metrics = compare_ray_records(
        at_records,
        native_records,
        max_range_m=max(config.geometry.receivers.ranges),
        sample_count=101,
    )

    assert metrics.at_count == config.solver.num_beams
    assert metrics.native_count == config.solver.num_beams
    assert metrics.matched_count == config.solver.num_beams
    assert metrics.depth_mae < 0.005
    assert metrics.depth_max_abs < 0.01
    assert metrics.bounce_match_fraction > 0.75


def test_native_bellhop_long_format_bathymetry_ray_path_parity():
    at_bin = _at_bin_or_skip()
    config = _long_format_bathymetry_config()

    at_records = run_at_ray_records(config, at_bin_path=at_bin)
    native_records = compute_native_ray_records(config)
    metrics = compare_ray_records(
        at_records,
        native_records,
        max_range_m=max(config.geometry.receivers.ranges),
        sample_count=101,
    )

    assert metrics.at_count == config.solver.num_beams
    assert metrics.native_count == config.solver.num_beams
    assert metrics.matched_count == config.solver.num_beams
    assert metrics.depth_mae < 0.005
    assert metrics.depth_max_abs < 0.01
    assert metrics.bounce_match_fraction > 0.85


@pytest.mark.parametrize(
    ("name", "factory", "run_type", "mae_tolerance", "p90_tolerance"),
    [
        ("iso_coherent", _iso_direct_config, "C", 1.5, 3.0),
        ("iso_incoherent", _iso_direct_config, "I", 4.5, 5.0),
        ("iso_semicoherent", _iso_direct_config, "S", 6.0, 8.0),
        ("shallow_incoherent", _shallow_reflected_config, "I", 2.5, 3.0),
        ("curvilinear_boundary_incoherent", _curvilinear_boundary_config, "I", 2.5, 3.0),
        ("elastic_bottom_incoherent", _elastic_bottom_config, "I", 1.0, 1.5),
        ("long_format_bathymetry_coherent", _long_format_bathymetry_config, "C", 1.0, 2.5),
        ("bottom_brc_coherent", _bottom_reflection_table_config, "C", 1.0, 1.5),
        ("bottom_brc_incoherent", _bottom_reflection_table_config, "I", 2.5, 3.0),
    ],
)
def test_native_bellhop_gaussian_tl_parity_matrix(name, factory, run_type, mae_tolerance, p90_tolerance):
    at_bin = _at_bin_or_skip()
    config = factory()
    config.solver.num_beams = 101
    config.solver.step_size = 2.5

    r_grid = np.linspace(100.0, 1000.0, 30)
    z_grid = np.linspace(10.0, config.environment.bottom.depth - 10.0, 30)

    at_tl = run_at_tl_grid(config, r_grid, z_grid, run_type=run_type, at_bin_path=at_bin)
    native_tl = compute_native_tl_grid(config, r_grid, z_grid, run_type=run_type)
    metrics = compare_tl_grids(at_tl, native_tl)

    assert metrics.sample_count > 0, name
    assert metrics.mae_db < mae_tolerance, name
    assert metrics.p90_abs_db < p90_tolerance, name
