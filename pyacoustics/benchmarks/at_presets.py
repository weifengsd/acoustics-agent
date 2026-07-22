"""Canonical AT/native parity benchmark presets.

The presets define cases only; they do not run Acoustics Toolbox executables.
"""

from __future__ import annotations

import copy

import numpy as np

from pyacoustics.schema import (
    Bathymetry3DModel,
    BellhopConfig,
    Bellhop3DConfig,
    BoundaryCondition,
    EnvironmentModel,
    GeometryModel,
    KrakenConfig,
    ReceiverModel,
    SimulationConfig,
    SourceModel,
    SSPLayer,
    SSPModel,
)
from pyacoustics.solvers.bellhop.benchmark_matrix import BellhopBenchmarkCase
from pyacoustics.solvers.bellhop3d.arrival_parity import Bellhop3DArrivalBenchmarkCase
from pyacoustics.solvers.bellhop3d.ray_parity import Bellhop3DRayBenchmarkCase
from pyacoustics.solvers.bellhop3d.tl_parity import Bellhop3DBenchmarkCase
from pyacoustics.solvers.benchmark_thresholds import BenchmarkThresholds
from pyacoustics.solvers.kraken.tl_parity import KrakenBenchmarkCase


def _iso_direct_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity iso direct",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=200.0, c=1500.0)],
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


def _iso_direct_bellhop3d_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop3d parity iso direct",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=200.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="rigid", depth=200.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[100.0]),
            receivers=ReceiverModel(ranges=[500.0, 1000.0], depths=[100.0]),
        ),
        solver=Bellhop3DConfig(
            elevation_angles=(-5.0, 5.0),
            azimuth_angles=(-1.0, 1.0),
            num_elevation_beams=21,
            num_azimuth_beams=3,
            step_size=10.0,
            box_x=1200.0,
            box_y=100.0,
            box_z=300.0,
        ),
    )


def _munk_slanted_terrain_bellhop3d_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop3d parity Munk slanted terrain",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1548.52),
                    SSPLayer(depth=1000.0, c=1500.0),
                    SSPLayer(depth=5000.0, c=1551.91),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(
                type="vacuum",
                depth=5000.0,
                bathymetry_3d=Bathymetry3DModel(
                    x=[-2000.0, 10000.0],
                    y=[-2000.0, 10000.0],
                    z=[
                        [2000.0, 3000.0],
                        [4000.0, 5000.0],
                    ],
                ),
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[1000.0]),
            receivers=ReceiverModel(ranges=[0.0, 10000.0], depths=[0.0, 5000.0]),
        ),
        solver=Bellhop3DConfig(
            elevation_angles=(-20.0, 20.0),
            azimuth_angles=(0.0, 90.0),
            num_elevation_beams=20,
            num_azimuth_beams=24,
            step_size=10.0,
        ),
    )


def _iso_direct_line_source_bellhop_config() -> SimulationConfig:
    config = _iso_direct_bellhop_config()
    config.project = "bellhop parity iso direct line source"
    config.solver.source_type = "line"
    return config


def _iso_direct_multi_source_depth_bellhop_config() -> SimulationConfig:
    config = _iso_direct_bellhop_config()
    config.project = "bellhop parity iso direct multi source depth"
    config.geometry.source.depths = [80.0, 120.0]
    config.geometry.receivers.depths = [100.0]
    config.solver.angles = (-5.0, 5.0)
    config.solver.num_beams = 101
    return config


def _iso_direct_irregular_receiver_bellhop_config() -> SimulationConfig:
    config = _iso_direct_bellhop_config()
    config.project = "bellhop parity iso direct irregular receiver grid"
    config.geometry.receivers.ranges = [1000.0, 1500.0]
    config.geometry.receivers.depths = [100.0, 120.0]
    config.solver.receiver_grid = "irregular"
    config.solver.num_beams = 101
    return config


def _iso_multipath_eigenray_bellhop_config() -> SimulationConfig:
    config = _iso_direct_bellhop_config()
    config.project = "bellhop parity iso multipath eigenray"
    config.solver.angles = (-35.0, 35.0)
    config.solver.num_beams = 281
    return config


def _munk_eigenray_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity Munk E ray",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1548.52),
                    SSPLayer(depth=200.0, c=1530.29),
                    SSPLayer(depth=250.0, c=1526.69),
                    SSPLayer(depth=400.0, c=1517.78),
                    SSPLayer(depth=600.0, c=1509.49),
                    SSPLayer(depth=800.0, c=1504.3),
                    SSPLayer(depth=1000.0, c=1501.38),
                    SSPLayer(depth=1200.0, c=1500.14),
                    SSPLayer(depth=1400.0, c=1500.12),
                    SSPLayer(depth=1600.0, c=1501.02),
                    SSPLayer(depth=1800.0, c=1502.57),
                    SSPLayer(depth=2000.0, c=1504.62),
                    SSPLayer(depth=2400.0, c=1509.69),
                    SSPLayer(depth=2800.0, c=1515.56),
                    SSPLayer(depth=3200.0, c=1521.85),
                    SSPLayer(depth=3600.0, c=1528.38),
                    SSPLayer(depth=4000.0, c=1535.04),
                    SSPLayer(depth=4400.0, c=1541.76),
                    SSPLayer(depth=4800.0, c=1548.52),
                    SSPLayer(depth=5000.0, c=1551.91),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="acousto-elastic", depth=5000.0, attenuation_p=0.5),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[1000.0]),
            receivers=ReceiverModel(ranges=[10000.0], depths=[1000.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-20.0, 20.0), num_beams=101, step_size=5.0),
    )


def _iso_direct_beam_type_bellhop_config(beam_type: str) -> SimulationConfig:
    config = _iso_direct_bellhop_config()
    config.project = f"bellhop parity iso direct {beam_type}"
    config.solver.beam_type = beam_type
    return config


def _shallow_reflected_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity shallow reflected",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
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


def _shallow_reflected_multi_source_depth_bellhop_config() -> SimulationConfig:
    config = _shallow_reflected_bellhop_config()
    config.project = "bellhop parity shallow reflected multi source depth"
    config.geometry.source.depths = [30.0, 40.0]
    config.solver.num_beams = 201
    return config


def _elastic_bottom_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity elastic bottom",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
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


def _elastic_bottom_multi_source_depth_bellhop_config() -> SimulationConfig:
    config = _elastic_bottom_bellhop_config()
    config.project = "bellhop parity elastic bottom multi source depth"
    config.geometry.source.depths = [30.0, 40.0]
    config.solver.num_beams = 151
    return config


def _sloping_bathymetry_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity sloping bathymetry",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=160.0, c=1500.0)],
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


def _sloping_altimetry_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity sloping altimetry",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=120.0, c=1500.0)],
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


def _long_format_bathymetry_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity long format bathymetry",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=150.0, c=1500.0)],
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


def _sloping_seongil_bellhop_config() -> SimulationConfig:
    config = _sloping_bathymetry_bellhop_config()
    config.project = "bellhop parity sloping bathymetry seongil shift"
    config.solver.beam_type = "geometric_hat"
    config.solver.beam_displacement = "seongil"
    return config


def _bottom_reflection_table_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity bottom reflection table",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=100.0,
                c_p=1600.0,
                density=1.8,
                attenuation_p=0.0,
                reflection_table=[(0.0, 0.5, 0.0), (45.0, 0.45, 45.0), (90.0, 0.5, 0.0)],
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[35.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[65.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-20.0, 20.0), num_beams=101, step_size=2.5),
    )


def _curvilinear_boundary_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity curvilinear boundaries",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=170.0, c=1500.0)],
            ),
            surface=BoundaryCondition(
                type="vacuum",
                depth=0.0,
                altimetry=[(0.0, 0.0), (600.0, 5.0), (1200.0, 0.0), (2000.0, 4.0)],
                altimetry_interpolation="C",
            ),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=170.0,
                c_p=1700.0,
                density=1.5,
                attenuation_p=0.5,
                bathymetry=[(0.0, 110.0), (600.0, 135.0), (1200.0, 115.0), (2000.0, 145.0)],
                bathymetry_interpolation="C",
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[45.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[60.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-18.0, 18.0), num_beams=145, step_size=2.5),
    )


def _range_dependent_ssp_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity range dependent SSP Q",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="quad-2d",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
                ranges=[0.0, 500.0, 1200.0],
                c_matrix=[
                    [1500.0, 1510.0, 1524.0],
                    [1500.0, 1510.0, 1524.0],
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-8.0, 8.0), num_beams=81, step_size=5.0),
    )


def _range_dependent_ssp_multi_source_depth_bellhop_config() -> SimulationConfig:
    config = _range_dependent_ssp_bellhop_config()
    config.project = "bellhop parity range dependent SSP Q multi source depth"
    config.geometry.source.depths = [45.0, 55.0]
    config.solver.num_beams = 121
    return config


def _range_dependent_ssp_reflection_table_bellhop_config() -> SimulationConfig:
    config = _range_dependent_ssp_bellhop_config()
    config.project = "bellhop parity range dependent SSP Q bottom reflection table"
    config.environment.bottom = BoundaryCondition(
        type="acousto-elastic",
        depth=100.0,
        c_p=1600.0,
        density=1.8,
        attenuation_p=0.0,
        reflection_table=[(0.0, 0.5, 0.0), (45.0, 0.45, 45.0), (90.0, 0.5, 0.0)],
    )
    config.solver.angles = (-10.0, 10.0)
    config.solver.num_beams = 101
    config.solver.step_size = 2.5
    return config


def _range_dependent_ssp_sloping_bathymetry_bellhop_config() -> SimulationConfig:
    return SimulationConfig(
        project="bellhop parity range dependent SSP Q sloping bathymetry",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="quad-2d",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=160.0, c=1500.0)],
                ranges=[0.0, 500.0, 1200.0],
                c_matrix=[
                    [1500.0, 1506.0, 1514.0],
                    [1498.0, 1504.0, 1512.0],
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
        solver=BellhopConfig(type="bellhop", angles=(-18.0, 18.0), num_beams=181, step_size=2.5),
    )


def _calib_like_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="native kraken at parity",
        frequency=250.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0, density=1.0),
                    SSPLayer(depth=100.0, c=1500.0, density=1.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=100.0,
                c_p=1590.0,
                density=1.2,
                attenuation_p=0.5,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 5000.0], depths=[0.0, 100.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1650.0), n_mesh_points=800, mesh_multiplier=[1, 2]),
    )


def _rigid_pekeris_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken rigid pekeris parity",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0, density=1.0),
                    SSPLayer(depth=100.0, c=1500.0, density=1.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0, 2500.0], depths=[25.0, 50.0, 75.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1700.0), n_mesh_points=400, mesh_multiplier=[1, 2]),
    )


def _rigid_surface_kraken_config() -> SimulationConfig:
    config = _rigid_pekeris_kraken_config()
    config.project = "kraken rigid surface parity"
    config.environment.surface = BoundaryCondition(type="rigid")
    return config


def _field_option_rc_kraken_config() -> SimulationConfig:
    config = _rigid_pekeris_kraken_config()
    config.project = "kraken rigid field rc parity"
    config.solver.field_option = "RC"
    return config


def _field_option_x_line_kraken_config() -> SimulationConfig:
    config = _rigid_pekeris_kraken_config()
    config.project = "kraken rigid field x line parity"
    config.solver.field_option = "XC"
    return config


def _field_option_s_scaled_kraken_config() -> SimulationConfig:
    config = _rigid_pekeris_kraken_config()
    config.project = "kraken rigid field s scaled parity"
    config.solver.field_option = "SC"
    return config


def _field_multi_source_depth_kraken_config() -> SimulationConfig:
    config = _rigid_pekeris_kraken_config()
    config.project = "kraken rigid field multi source depth parity"
    config.geometry.source.depths = [40.0, 60.0]
    config.solver.n_mesh_points = 600
    config.solver.mesh_multiplier = [1, 2]
    return config


def _field_offsets_max_modes_kraken_config() -> SimulationConfig:
    config = _rigid_pekeris_kraken_config()
    config.project = "kraken rigid field offsets max modes parity"
    config.solver.max_modes = 37
    config.solver.field_receiver_range_offsets_m = [-2.0, 0.0, 3.0]
    return config


def _field_truncated_max_modes_kraken_config() -> SimulationConfig:
    config = _rigid_pekeris_kraken_config()
    config.project = "kraken rigid field truncated max modes parity"
    config.solver.max_modes = 3
    return config


def _wide_grid_rigid_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken wide receiver grid parity",
        frequency=90.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1498.0, density=1.0),
                    SSPLayer(depth=140.0, c=1512.0, density=1.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=140.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[45.0]),
            receivers=ReceiverModel(
                ranges=[600.0, 1400.0, 2600.0, 4200.0],
                depths=[20.0, 55.0, 95.0, 130.0],
            ),
        ),
        solver=KrakenConfig(phase_speed_limits=(1430.0, 1800.0), n_mesh_points=450, mesh_multiplier=[1, 3]),
    )


def _deep_gradient_rigid_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken deep gradient receiver grid parity",
        frequency=75.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1488.0, density=1.0),
                    SSPLayer(depth=70.0, c=1492.0, density=1.0),
                    SSPLayer(depth=170.0, c=1510.0, density=1.0),
                    SSPLayer(depth=280.0, c=1530.0, density=1.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=280.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[80.0]),
            receivers=ReceiverModel(
                ranges=[800.0, 1800.0, 3600.0, 6000.0, 8500.0],
                depths=[40.0, 110.0, 190.0, 250.0],
            ),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1750.0), n_mesh_points=1000, mesh_multiplier=[1, 2]),
    )


def _reflection_table_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken bottom reflection table parity",
        frequency=150.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0, density=1.0),
                    SSPLayer(depth=120.0, c=1500.0, density=1.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=120.0,
                c_p=1650.0,
                density=1.6,
                attenuation_p=0.0,
                reflection_table=[(0.0, 0.8, 0.0), (45.0, 0.7, 45.0), (90.0, 0.5, 180.0)],
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[40.0]),
            receivers=ReceiverModel(ranges=[750.0, 1500.0], depths=[40.0, 80.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1750.0), n_mesh_points=800, mesh_multiplier=[1, 2]),
    )


def _surface_reflection_table_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken surface reflection table parity",
        frequency=150.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0, density=1.0),
                    SSPLayer(depth=120.0, c=1500.0, density=1.0),
                ],
            ),
            surface=BoundaryCondition(
                type="rigid",
                reflection_table=[(0.0, 0.9, 180.0), (45.0, 0.8, 135.0), (90.0, 0.6, 90.0)],
            ),
            bottom=BoundaryCondition(type="rigid", depth=120.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[40.0]),
            receivers=ReceiverModel(ranges=[750.0, 1500.0], depths=[40.0, 80.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1750.0), n_mesh_points=1000, mesh_multiplier=[1, 2, 4]),
    )


def _surface_bottom_reflection_table_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken surface bottom reflection table parity",
        frequency=150.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0, density=1.0),
                    SSPLayer(depth=120.0, c=1500.0, density=1.0),
                ],
            ),
            surface=BoundaryCondition(
                type="rigid",
                reflection_table=[(0.0, 0.9, 180.0), (45.0, 0.8, 135.0), (90.0, 0.6, 90.0)],
            ),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=120.0,
                c_p=1650.0,
                density=1.6,
                attenuation_p=0.0,
                reflection_table=[(0.0, 0.8, 0.0), (45.0, 0.7, 45.0), (90.0, 0.5, 180.0)],
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[40.0]),
            receivers=ReceiverModel(ranges=[750.0, 1500.0], depths=[40.0, 80.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1750.0), n_mesh_points=800, mesh_multiplier=[1, 2]),
    )


def _water_attenuation_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken water attenuation parity",
        frequency=150.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0, density=1.0, attenuation=0.15),
                    SSPLayer(depth=120.0, c=1500.0, density=1.0, attenuation=0.15),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=120.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[60.0]),
            receivers=ReceiverModel(ranges=[1000.0, 2500.0], depths=[30.0, 60.0, 90.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1700.0), n_mesh_points=1000, mesh_multiplier=[1, 2]),
    )


def _two_layer_acoustic_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken two layer acoustic parity",
        frequency=120.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1495.0, density=1.0),
                    SSPLayer(depth=70.0, c=1508.0, density=1.03),
                    SSPLayer(depth=160.0, c=1525.0, density=1.08),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=160.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0, 2200.0], depths=[40.0, 90.0, 140.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1750.0), n_mesh_points=600, mesh_multiplier=[1, 2]),
    )


def _nearfield_acoustic_halfspace_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken nearfield acoustic halfspace parity",
        frequency=120.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1490.0, density=1.0),
                    SSPLayer(depth=60.0, c=1498.0, density=1.0),
                    SSPLayer(depth=140.0, c=1520.0, density=1.03),
                    SSPLayer(depth=220.0, c=1538.0, density=1.06),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=220.0,
                c_p=1680.0,
                density=1.65,
                attenuation_p=0.0,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[70.0]),
            receivers=ReceiverModel(ranges=[100.0, 300.0, 600.0], depths=[120.0, 190.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1800.0), n_mesh_points=1000, mesh_multiplier=[1, 2]),
    )


def _lossy_elastic_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken lossy elastic bottom parity",
        frequency=200.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1490.0, density=1.0),
                    SSPLayer(depth=80.0, c=1505.0, density=1.0),
                    SSPLayer(depth=180.0, c=1520.0, density=1.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=180.0,
                c_p=1750.0,
                c_s=350.0,
                density=1.8,
                attenuation_p=0.35,
                attenuation_s=0.8,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[60.0]),
            receivers=ReceiverModel(ranges=[1000.0, 3000.0], depths=[50.0, 100.0, 150.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1900.0), n_mesh_points=1200, mesh_multiplier=[1, 2, 4, 8]),
    )


def _lossy_elastic_field_x_line_kraken_config() -> SimulationConfig:
    config = _lossy_elastic_kraken_config()
    config.project = "kraken lossy elastic bottom field X parity"
    config.solver.field_option = "XC"
    return config


def _layered_lossy_elastic_kraken_config() -> SimulationConfig:
    return SimulationConfig(
        project="kraken layered lossy elastic bottom parity",
        frequency=120.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1488.0, density=1.00, attenuation=0.02),
                    SSPLayer(depth=45.0, c=1496.0, density=1.01, attenuation=0.04),
                    SSPLayer(depth=110.0, c=1515.0, density=1.05, attenuation=0.06),
                    SSPLayer(depth=240.0, c=1540.0, density=1.08, attenuation=0.08),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=240.0,
                c_p=1820.0,
                c_s=520.0,
                density=1.95,
                attenuation_p=0.25,
                attenuation_s=0.35,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[75.0]),
            receivers=ReceiverModel(ranges=[800.0, 1800.0, 4200.0], depths=[60.0, 135.0, 210.0]),
        ),
        solver=KrakenConfig(phase_speed_limits=(1440.0, 1950.0), n_mesh_points=1600, mesh_multiplier=[1]),
    )


def _layered_lossy_elastic_field_s_scaled_kraken_config() -> SimulationConfig:
    config = _layered_lossy_elastic_kraken_config()
    config.project = "kraken layered lossy elastic bottom field S parity"
    config.solver.field_option = "SC"
    return config


def canonical_bellhop_benchmark_cases() -> tuple[BellhopBenchmarkCase, ...]:
    """Return a small canonical 2D Bellhop AT parity matrix."""

    iso = _iso_direct_bellhop_config()
    iso_line = _iso_direct_line_source_bellhop_config()
    iso_multi_source_depth = _iso_direct_multi_source_depth_bellhop_config()
    iso_irregular = _iso_direct_irregular_receiver_bellhop_config()
    iso_multipath = _iso_multipath_eigenray_bellhop_config()
    munk_eigenray = _munk_eigenray_bellhop_config()
    iso_geometric_gaussian = _iso_direct_beam_type_bellhop_config("geometric_gaussian")
    iso_simple_gaussian = _iso_direct_beam_type_bellhop_config("simple_gaussian")
    iso_ray_centered_hat = _iso_direct_beam_type_bellhop_config("geometric_hat_ray_centered")
    reflected = _shallow_reflected_bellhop_config()
    reflected_multi_source_depth = _shallow_reflected_multi_source_depth_bellhop_config()
    elastic_bottom = _elastic_bottom_bellhop_config()
    elastic_bottom_multi_source_depth = _elastic_bottom_multi_source_depth_bellhop_config()
    sloping = _sloping_bathymetry_bellhop_config()
    sloping_altimetry = _sloping_altimetry_bellhop_config()
    long_format_bathymetry = _long_format_bathymetry_bellhop_config()
    sloping_seongil = _sloping_seongil_bellhop_config()
    reflection_table = _bottom_reflection_table_bellhop_config()
    curvilinear = _curvilinear_boundary_bellhop_config()
    range_dependent_ssp = _range_dependent_ssp_bellhop_config()
    range_dependent_ssp_multi_source_depth = _range_dependent_ssp_multi_source_depth_bellhop_config()
    range_dependent_ssp_reflection_table = _range_dependent_ssp_reflection_table_bellhop_config()
    range_dependent_ssp_sloping_bathymetry = _range_dependent_ssp_sloping_bathymetry_bellhop_config()

    def ray_centered_hat(config: SimulationConfig) -> SimulationConfig:
        copied = copy.deepcopy(config)
        copied.solver.beam_type = "geometric_hat_ray_centered"
        return copied

    sloping_ray_centered_hat = ray_centered_hat(sloping)
    reflection_table_ray_centered_hat = ray_centered_hat(reflection_table)
    range_dependent_ssp_ray_centered_hat = ray_centered_hat(range_dependent_ssp)
    range_dependent_ssp_reflection_table_ray_centered_hat = ray_centered_hat(range_dependent_ssp_reflection_table)
    range_dependent_ssp_sloping_bathymetry_ray_centered_hat = ray_centered_hat(
        range_dependent_ssp_sloping_bathymetry
    )
    return (
        BellhopBenchmarkCase(
            "iso-direct-arrivals",
            "arrival",
            iso,
            options={"delay_tolerance": 0.01, "max_arrivals": 40, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "iso-direct-line-source-arrivals",
            "arrival",
            iso_line,
            options={"delay_tolerance": 0.01, "max_arrivals": 40, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "iso-direct-line-source-coherent-tl",
            "tl",
            iso_line,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "iso-direct-line-source-incoherent-tl",
            "tl",
            iso_line,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "iso-direct-multi-source-depth-arrivals",
            "arrival",
            iso_multi_source_depth,
            options={"delay_tolerance": 0.01, "max_arrivals": 40, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "iso-direct-multi-source-depth-coherent-tl",
            "tl",
            iso_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "iso-direct-multi-source-depth-incoherent-tl",
            "tl",
            iso_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "iso-direct-multi-source-depth-semicoherent-tl",
            "tl",
            iso_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "iso-direct-irregular-receiver-arrivals",
            "arrival",
            iso_irregular,
            options={"delay_tolerance": 0.02, "max_arrivals": 40, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "iso-multipath-eigenray-arrivals",
            "arrival",
            iso_multipath,
            options={
                "delay_tolerance": 0.02,
                "max_arrivals": 80,
                "method": "eigenray",
                "eigenray_tolerance_m": 0.1,
                "eigenray_angle_samples": 281,
            },
        ),
        BellhopBenchmarkCase(
            "iso-multipath-rays",
            "ray",
            iso_multipath,
            options={"max_range_m": 1000.0, "sample_count": 101},
        ),
        BellhopBenchmarkCase(
            "iso-direct-rays",
            "ray",
            iso,
            options={"max_range_m": 1000.0, "sample_count": 51},
        ),
        BellhopBenchmarkCase(
            "iso-direct-coherent-tl",
            "tl",
            iso,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "munk-beam-arrivals",
            "arrival",
            munk_eigenray,
            options={"delay_tolerance": 0.03, "max_arrivals": 80, "method": "beam"},
            thresholds=BenchmarkThresholds(
                max_values={
                    "delay_mae_s": 0.0001,
                    "amplitude_median_relative_error": 0.01,
                    "phase_mae_rad": 1.0e-6,
                    "launch_angle_mae_deg": 0.001,
                    "receiver_angle_mae_deg": 0.01,
                },
                min_values={"match_fraction": 1.0, "bounce_match_fraction": 1.0},
                ignore_missing=True,
            ),
        ),
        BellhopBenchmarkCase(
            "munk-gaussian-beam-arrivals",
            "arrival",
            munk_eigenray,
            options={
                "delay_tolerance": 0.03,
                "max_arrivals": 80,
                "method": "beam",
                "beam_shape": "gaussian",
                "run_type": "AB   3",
            },
            thresholds=BenchmarkThresholds(
                max_values={
                    "delay_mae_s": 0.0001,
                    "amplitude_median_relative_error": 0.002,
                    "phase_mae_rad": 1.0e-6,
                    "launch_angle_mae_deg": 0.01,
                    "receiver_angle_mae_deg": 0.01,
                },
                min_values={"match_fraction": 1.0, "bounce_match_fraction": 1.0},
                ignore_missing=True,
            ),
        ),
        BellhopBenchmarkCase(
            "munk-eigenray-rays",
            "ray",
            munk_eigenray,
            run_type="E",
            options={"max_range_m": 2000.0, "sample_count": 101, "max_arrivals": 40},
        ),
        BellhopBenchmarkCase(
            "iso-multipath-eigenray-rays",
            "ray",
            iso_multipath,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-eigenray-rays",
            "ray",
            reflected,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-eigenray-rays",
            "ray",
            elastic_bottom,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "bottom-reflection-table-eigenray-rays",
            "ray",
            reflection_table,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-eigenray-rays",
            "ray",
            sloping,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "sloping-altimetry-eigenray-rays",
            "ray",
            sloping_altimetry,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "long-format-bathymetry-eigenray-rays",
            "ray",
            long_format_bathymetry,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "curvilinear-boundary-eigenray-rays",
            "ray",
            curvilinear,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-eigenray-rays",
            "ray",
            range_dependent_ssp,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-bottom-reflection-table-eigenray-rays",
            "ray",
            range_dependent_ssp_reflection_table,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-sloping-bathymetry-eigenray-rays",
            "ray",
            range_dependent_ssp_sloping_bathymetry,
            run_type="E",
            options={"max_range_m": 1000.0, "sample_count": 101, "max_arrivals": 80},
        ),
        BellhopBenchmarkCase(
            "iso-direct-geometric-gaussian-tl",
            "tl",
            iso_geometric_gaussian,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "iso-direct-geometric-gaussian-incoherent-tl",
            "tl",
            iso_geometric_gaussian,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "iso-direct-geometric-gaussian-semicoherent-tl",
            "tl",
            iso_geometric_gaussian,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "iso-direct-simple-gaussian-tl",
            "tl",
            iso_simple_gaussian,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "iso-direct-ray-centered-hat-tl",
            "tl",
            iso_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([100.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "iso-direct-ray-centered-hat-incoherent-tl",
            "tl",
            iso_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([100.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-ray-centered-hat-tl",
            "tl",
            sloping_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-ray-centered-hat-semicoherent-tl",
            "tl",
            sloping_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "bottom-reflection-table-ray-centered-hat-semicoherent-tl",
            "tl",
            reflection_table_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-ray-centered-hat-tl",
            "tl",
            range_dependent_ssp_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([50.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-ray-centered-hat-incoherent-tl",
            "tl",
            range_dependent_ssp_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([50.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-ray-centered-hat-semicoherent-tl",
            "tl",
            range_dependent_ssp_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([50.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-bottom-reflection-table-ray-centered-hat-tl",
            "tl",
            range_dependent_ssp_reflection_table_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([50.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-bottom-reflection-table-ray-centered-hat-incoherent-tl",
            "tl",
            range_dependent_ssp_reflection_table_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([50.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-bottom-reflection-table-ray-centered-hat-semicoherent-tl",
            "tl",
            range_dependent_ssp_reflection_table_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([50.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-sloping-bathymetry-ray-centered-hat-tl",
            "tl",
            range_dependent_ssp_sloping_bathymetry_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-sloping-bathymetry-ray-centered-hat-semicoherent-tl",
            "tl",
            range_dependent_ssp_sloping_bathymetry_ray_centered_hat,
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "iso-direct-incoherent-tl",
            "tl",
            iso,
            r_grid=np.array([1000.0]),
            z_grid=np.array([100.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-arrivals",
            "arrival",
            reflected,
            options={"delay_tolerance": 0.02, "max_arrivals": 60, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-eigenray-arrivals",
            "arrival",
            reflected,
            options={
                "delay_tolerance": 0.02,
                "max_arrivals": 60,
                "method": "eigenray",
                "eigenray_tolerance_m": 0.1,
                "eigenray_angle_samples": 161,
            },
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-coherent-tl",
            "tl",
            reflected,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-incoherent-tl",
            "tl",
            reflected,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-semicoherent-tl",
            "tl",
            reflected,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-multi-source-depth-arrivals",
            "arrival",
            reflected_multi_source_depth,
            options={"delay_tolerance": 0.02, "max_arrivals": 80, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-multi-source-depth-coherent-tl",
            "tl",
            reflected_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-multi-source-depth-incoherent-tl",
            "tl",
            reflected_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "shallow-reflected-multi-source-depth-semicoherent-tl",
            "tl",
            reflected_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-rays",
            "ray",
            elastic_bottom,
            options={"max_range_m": 1000.0, "sample_count": 101},
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-arrivals",
            "arrival",
            elastic_bottom,
            options={"delay_tolerance": 0.03, "max_arrivals": 80, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-coherent-tl",
            "tl",
            elastic_bottom,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-incoherent-tl",
            "tl",
            elastic_bottom,
            r_grid=np.array([600.0, 1000.0, 1400.0]),
            z_grid=np.array([45.0, 65.0, 85.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-semicoherent-tl",
            "tl",
            elastic_bottom,
            r_grid=np.array([600.0, 1000.0, 1400.0]),
            z_grid=np.array([45.0, 65.0, 85.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-multi-source-depth-arrivals",
            "arrival",
            elastic_bottom_multi_source_depth,
            options={"delay_tolerance": 0.03, "max_arrivals": 100, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-multi-source-depth-coherent-tl",
            "tl",
            elastic_bottom_multi_source_depth,
            r_grid=np.array([600.0, 1000.0, 1400.0]),
            z_grid=np.array([45.0, 65.0, 85.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-multi-source-depth-incoherent-tl",
            "tl",
            elastic_bottom_multi_source_depth,
            r_grid=np.array([600.0, 1000.0, 1400.0]),
            z_grid=np.array([45.0, 65.0, 85.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "elastic-bottom-multi-source-depth-semicoherent-tl",
            "tl",
            elastic_bottom_multi_source_depth,
            r_grid=np.array([600.0, 1000.0, 1400.0]),
            z_grid=np.array([45.0, 65.0, 85.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-rays",
            "ray",
            sloping,
            options={"max_range_m": 1000.0, "sample_count": 51},
        ),
        BellhopBenchmarkCase(
            "sloping-altimetry-rays",
            "ray",
            sloping_altimetry,
            options={"max_range_m": 1000.0, "sample_count": 101},
        ),
        BellhopBenchmarkCase(
            "sloping-altimetry-coherent-tl",
            "tl",
            sloping_altimetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "sloping-altimetry-incoherent-tl",
            "tl",
            sloping_altimetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "sloping-altimetry-semicoherent-tl",
            "tl",
            sloping_altimetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "long-format-bathymetry-rays",
            "ray",
            long_format_bathymetry,
            options={"max_range_m": 1000.0, "sample_count": 101},
        ),
        BellhopBenchmarkCase(
            "long-format-bathymetry-coherent-tl",
            "tl",
            long_format_bathymetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([60.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "long-format-bathymetry-incoherent-tl",
            "tl",
            long_format_bathymetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([60.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "long-format-bathymetry-semicoherent-tl",
            "tl",
            long_format_bathymetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([60.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-coherent-tl",
            "tl",
            sloping,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-incoherent-tl",
            "tl",
            sloping,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-semicoherent-tl",
            "tl",
            sloping,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-seongil-shift-tl",
            "tl",
            sloping_seongil,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-seongil-shift-incoherent-tl",
            "tl",
            sloping_seongil,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "sloping-bathymetry-seongil-shift-semicoherent-tl",
            "tl",
            sloping_seongil,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "bottom-reflection-table-arrivals",
            "arrival",
            reflection_table,
            options={"delay_tolerance": 0.03, "max_arrivals": 80, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "bottom-reflection-table-rays",
            "ray",
            reflection_table,
            options={"max_range_m": 1000.0, "sample_count": 101},
        ),
        BellhopBenchmarkCase(
            "bottom-reflection-table-coherent-tl",
            "tl",
            reflection_table,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "bottom-reflection-table-incoherent-tl",
            "tl",
            reflection_table,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "bottom-reflection-table-semicoherent-tl",
            "tl",
            reflection_table,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "curvilinear-boundary-rays",
            "ray",
            curvilinear,
            options={"max_range_m": 1000.0, "sample_count": 51},
        ),
        BellhopBenchmarkCase(
            "curvilinear-boundary-coherent-tl",
            "tl",
            curvilinear,
            r_grid=np.array([600.0, 1000.0, 1400.0]),
            z_grid=np.array([45.0, 60.0, 85.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "curvilinear-boundary-incoherent-tl",
            "tl",
            curvilinear,
            r_grid=np.array([600.0, 1000.0, 1400.0]),
            z_grid=np.array([45.0, 60.0, 85.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "curvilinear-boundary-semicoherent-tl",
            "tl",
            curvilinear,
            r_grid=np.array([600.0, 1000.0, 1400.0]),
            z_grid=np.array([45.0, 60.0, 85.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-rays",
            "ray",
            range_dependent_ssp,
            options={"max_range_m": 1000.0, "sample_count": 51},
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-coherent-tl",
            "tl",
            range_dependent_ssp,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-incoherent-tl",
            "tl",
            range_dependent_ssp,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-semicoherent-tl",
            "tl",
            range_dependent_ssp,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-multi-source-depth-arrivals",
            "arrival",
            range_dependent_ssp_multi_source_depth,
            options={"delay_tolerance": 0.03, "max_arrivals": 80, "method": "beam"},
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-multi-source-depth-coherent-tl",
            "tl",
            range_dependent_ssp_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-multi-source-depth-incoherent-tl",
            "tl",
            range_dependent_ssp_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-multi-source-depth-semicoherent-tl",
            "tl",
            range_dependent_ssp_multi_source_depth,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-bottom-reflection-table-coherent-tl",
            "tl",
            range_dependent_ssp_reflection_table,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-bottom-reflection-table-incoherent-tl",
            "tl",
            range_dependent_ssp_reflection_table,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-bottom-reflection-table-semicoherent-tl",
            "tl",
            range_dependent_ssp_reflection_table,
            r_grid=np.array([1000.0]),
            z_grid=np.array([50.0]),
            run_type="S",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-sloping-bathymetry-coherent-tl",
            "tl",
            range_dependent_ssp_sloping_bathymetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="C",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-sloping-bathymetry-incoherent-tl",
            "tl",
            range_dependent_ssp_sloping_bathymetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="I",
        ),
        BellhopBenchmarkCase(
            "range-dependent-ssp-sloping-bathymetry-semicoherent-tl",
            "tl",
            range_dependent_ssp_sloping_bathymetry,
            r_grid=np.array([1000.0]),
            z_grid=np.array([65.0]),
            run_type="S",
        ),
    )


def canonical_bellhop3d_benchmark_cases() -> tuple[Bellhop3DBenchmarkCase, ...]:
    """Return a small canonical Bellhop3D receiver-grid TL parity matrix."""

    iso_direct = _iso_direct_bellhop3d_config()
    iso_multi_source_depth = _iso_direct_bellhop3d_config()
    iso_multi_source_depth.project = "bellhop3d parity iso multi-source-depth TL"
    iso_multi_source_depth.geometry.source.depths = [80.0, 100.0, 120.0]
    munk_slanted = _munk_slanted_terrain_bellhop3d_config()
    munk_oblique = copy.deepcopy(munk_slanted)
    munk_oblique.project = "bellhop3d parity Munk slanted terrain oblique"
    munk_steep_oblique = copy.deepcopy(munk_slanted)
    munk_steep_oblique.project = "bellhop3d parity Munk slanted terrain steep oblique"
    return (
        Bellhop3DBenchmarkCase(
            "bellhop3d-iso-direct-coherent-tl",
            iso_direct,
            theta_grid_deg=np.array([0.0]),
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([100.0]),
            run_type="C",
            active_tl_max_db=190.0,
        ),
        Bellhop3DBenchmarkCase(
            "bellhop3d-iso-multi-source-depth-coherent-tl",
            iso_multi_source_depth,
            theta_grid_deg=np.array([0.0]),
            r_grid=np.array([500.0, 1000.0]),
            z_grid=np.array([100.0]),
            run_type="C",
            active_tl_max_db=190.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "tl_mae_db": 1.0e-5,
                    "tl_p90_abs_db": 1.0e-5,
                    "tl_max_abs_db": 1.0e-5,
                    "at_only_active_count": 0.0,
                    "native_only_active_count": 0.0,
                    "native_weak_tail_count": 0.0,
                },
                min_values={"sample_count": 6.0, "active_overlap_fraction": 1.0},
                max_abs_values={"tl_bias_db": 1.0e-5},
                ignore_missing=True,
            ),
        ),
        Bellhop3DBenchmarkCase(
            "bellhop3d-munk-slanted-terrain-coherent-tl",
            munk_slanted,
            theta_grid_deg=np.array([0.0]),
            r_grid=np.linspace(0.0, 10000.0, 31),
            z_grid=np.linspace(0.0, 5000.0, 31),
            run_type="C",
            min_range_m=500.0,
            active_tl_max_db=110.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "tl_mae_db": 0.6,
                    "tl_p90_abs_db": 1.0,
                    "tl_max_abs_db": 12.0,
                    "at_only_active_count": 0.0,
                    "native_only_active_count": 10.0,
                    "native_weak_tail_count": 90.0,
                },
                min_values={"sample_count": 500.0, "active_overlap_fraction": 0.98},
                max_abs_values={"tl_bias_db": 0.3},
                ignore_missing=True,
            ),
        ),
        Bellhop3DBenchmarkCase(
            "bellhop3d-munk-slanted-terrain-oblique-tl",
            munk_oblique,
            theta_grid_deg=np.array([45.0]),
            r_grid=np.linspace(0.0, 10000.0, 31),
            z_grid=np.linspace(0.0, 5000.0, 31),
            run_type="C",
            min_range_m=500.0,
            active_tl_max_db=110.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "tl_mae_db": 0.25,
                    "tl_p90_abs_db": 0.6,
                    "tl_max_abs_db": 3.0,
                    "at_only_active_count": 2.0,
                    "native_only_active_count": 1.0,
                    "native_weak_tail_count": 60.0,
                },
                min_values={"sample_count": 560.0, "active_overlap_fraction": 0.995},
                max_abs_values={"tl_bias_db": 0.05},
                ignore_missing=True,
            ),
        ),
        Bellhop3DBenchmarkCase(
            "bellhop3d-munk-slanted-terrain-steep-oblique-tl",
            munk_steep_oblique,
            theta_grid_deg=np.array([60.0]),
            r_grid=np.linspace(0.0, 10000.0, 31),
            z_grid=np.linspace(0.0, 5000.0, 31),
            run_type="C",
            min_range_m=500.0,
            active_tl_max_db=110.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "tl_mae_db": 0.25,
                    "tl_p90_abs_db": 0.6,
                    "tl_max_abs_db": 3.5,
                    "at_only_active_count": 0.0,
                    "native_only_active_count": 0.0,
                    "native_weak_tail_count": 60.0,
                },
                min_values={"sample_count": 560.0, "active_overlap_fraction": 0.999},
                max_abs_values={"tl_bias_db": 0.05},
                ignore_missing=True,
            ),
        ),
    )


def canonical_bellhop3d_arrival_benchmark_cases() -> tuple[Bellhop3DArrivalBenchmarkCase, ...]:
    """Return canonical Bellhop3D arrivals parity residual trackers."""

    iso_direct = _iso_direct_bellhop3d_config()
    iso_direct.project = "bellhop3d parity iso direct arrivals"
    iso_direct.geometry.receivers.ranges = [500.0]
    iso_direct.geometry.receivers.depths = [100.0]
    iso_direct.solver.elevation_angles = (-2.0, 2.0)
    iso_direct.solver.azimuth_angles = (-2.0, 2.0)
    iso_direct.solver.num_elevation_beams = 21
    iso_direct.solver.num_azimuth_beams = 3
    iso_direct.solver.step_size = 10.0
    multi_depth = _iso_direct_bellhop3d_config()
    multi_depth.project = "bellhop3d parity iso multi-depth arrivals"
    multi_depth.geometry.receivers.ranges = [500.0]
    multi_depth.geometry.receivers.depths = [80.0, 100.0, 120.0]
    multi_depth.solver.elevation_angles = (-5.0, 5.0)
    multi_depth.solver.azimuth_angles = (-2.0, 2.0)
    multi_depth.solver.num_elevation_beams = 41
    multi_depth.solver.num_azimuth_beams = 3
    multi_depth.solver.step_size = 10.0
    off_axis = _iso_direct_bellhop3d_config()
    off_axis.project = "bellhop3d parity off-axis direct arrivals"
    off_axis.geometry.receivers.ranges = [500.0]
    off_axis.geometry.receivers.depths = [100.0]
    off_axis.solver.receiver_bearings = [2.0]
    off_axis.solver.elevation_angles = (-2.0, 2.0)
    off_axis.solver.azimuth_angles = (-2.0, 2.0)
    off_axis.solver.num_elevation_beams = 21
    off_axis.solver.num_azimuth_beams = 5
    off_axis.solver.step_size = 10.0
    wide_bearing = _iso_direct_bellhop3d_config()
    wide_bearing.project = "bellhop3d parity wide-bearing direct arrivals"
    wide_bearing.geometry.receivers.ranges = [500.0]
    wide_bearing.geometry.receivers.depths = [100.0]
    wide_bearing.solver.receiver_bearings = [30.0]
    wide_bearing.solver.elevation_angles = (-2.0, 2.0)
    wide_bearing.solver.azimuth_angles = (28.0, 32.0)
    wide_bearing.solver.num_elevation_beams = 21
    wide_bearing.solver.num_azimuth_beams = 5
    wide_bearing.solver.step_size = 10.0
    wide_bearing.solver.box_y = 600.0
    source_xy = _iso_direct_bellhop3d_config()
    source_xy.project = "bellhop3d parity source-xy direct arrivals"
    source_xy.geometry.source.x = [0.0, 20.0]
    source_xy.geometry.source.y = [0.0, 10.0]
    source_xy.geometry.receivers.ranges = [500.0]
    source_xy.geometry.receivers.depths = [100.0]
    source_xy.solver.elevation_angles = (-2.0, 2.0)
    source_xy.solver.azimuth_angles = (-2.0, 2.0)
    source_xy.solver.num_elevation_beams = 21
    source_xy.solver.num_azimuth_beams = 3
    source_xy.solver.step_size = 10.0
    source_xy.solver.box_x = 700.0
    source_xy.solver.box_y = 200.0
    source_depth_xy = _iso_direct_bellhop3d_config()
    source_depth_xy.project = "bellhop3d parity source-depth-xy direct arrivals"
    source_depth_xy.geometry.source.depths = [80.0, 120.0]
    source_depth_xy.geometry.source.x = [0.0, 20.0]
    source_depth_xy.geometry.source.y = [0.0, 10.0]
    source_depth_xy.geometry.receivers.ranges = [500.0]
    source_depth_xy.geometry.receivers.depths = [100.0]
    source_depth_xy.solver.elevation_angles = (-5.0, 5.0)
    source_depth_xy.solver.azimuth_angles = (-2.0, 2.0)
    source_depth_xy.solver.num_elevation_beams = 41
    source_depth_xy.solver.num_azimuth_beams = 3
    source_depth_xy.solver.step_size = 10.0
    source_depth_xy.solver.box_x = 700.0
    source_depth_xy.solver.box_y = 200.0
    munk_oblique = _munk_slanted_terrain_bellhop3d_config()
    munk_oblique.project = "bellhop3d parity Munk oblique arrivals"
    munk_oblique.geometry.receivers.ranges = [3000.0]
    munk_oblique.geometry.receivers.depths = [1000.0]
    munk_oblique.solver.receiver_bearings = [30.0]
    munk_oblique.solver.elevation_angles = (-20.0, 20.0)
    munk_oblique.solver.azimuth_angles = (25.0, 35.0)
    munk_oblique.solver.num_elevation_beams = 81
    munk_oblique.solver.num_azimuth_beams = 11
    munk_oblique.solver.step_size = 10.0
    return (
        Bellhop3DArrivalBenchmarkCase(
            "bellhop3d-iso-direct-arrivals",
            iso_direct,
            run_type="A",
            delay_tolerance_s=0.01,
            angle_tolerance_deg=0.1,
            tolerance_m=20.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_count": 0.0,
                    "unmatched_native_count": 0.0,
                    "delay_mae_s": 1.0e-5,
                    "amplitude_median_relative_error": 0.02,
                    "launch_declination_mae_deg": 0.1,
                    "launch_azimuth_mae_deg": 0.1,
                },
                min_values={
                    "matched_count": 3.0,
                    "match_fraction": 1.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DArrivalBenchmarkCase(
            "bellhop3d-iso-multi-depth-arrivals",
            multi_depth,
            run_type="A",
            delay_tolerance_s=0.01,
            angle_tolerance_deg=0.2,
            tolerance_m=20.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_count": 0.0,
                    "unmatched_native_count": 3.0,
                    "delay_mae_s": 1.0e-5,
                    "amplitude_median_relative_error": 0.02,
                    "amplitude_p90_relative_error": 0.02,
                    "launch_declination_mae_deg": 0.01,
                    "launch_azimuth_mae_deg": 0.01,
                    "receiver_declination_mae_deg": 0.01,
                    "receiver_azimuth_mae_deg": 0.01,
                },
                min_values={
                    "matched_count": 9.0,
                    "match_fraction": 1.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DArrivalBenchmarkCase(
            "bellhop3d-iso-direct-off-axis-arrivals",
            off_axis,
            run_type="A",
            delay_tolerance_s=0.01,
            angle_tolerance_deg=2.0,
            tolerance_m=20.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_count": 0.0,
                    "unmatched_native_count": 0.0,
                    "delay_mae_s": 1.0e-5,
                    "amplitude_median_relative_error": 0.01,
                    "launch_declination_mae_deg": 0.1,
                    "launch_azimuth_mae_deg": 0.01,
                    "receiver_declination_mae_deg": 0.1,
                    "receiver_azimuth_mae_deg": 0.01,
                },
                min_values={
                    "matched_count": 3.0,
                    "match_fraction": 1.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DArrivalBenchmarkCase(
            "bellhop3d-iso-direct-wide-bearing-arrivals",
            wide_bearing,
            run_type="A",
            delay_tolerance_s=0.01,
            angle_tolerance_deg=2.0,
            tolerance_m=20.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_count": 0.0,
                    "unmatched_native_count": 1.0,
                    "delay_mae_s": 1.0e-5,
                    "amplitude_median_relative_error": 0.15,
                    "amplitude_p90_relative_error": 0.3,
                    "launch_declination_mae_deg": 0.1,
                    "launch_azimuth_mae_deg": 0.2,
                    "receiver_declination_mae_deg": 0.1,
                    "receiver_azimuth_mae_deg": 0.2,
                },
                min_values={
                    "matched_count": 3.0,
                    "match_fraction": 1.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DArrivalBenchmarkCase(
            "bellhop3d-iso-source-xy-arrivals",
            source_xy,
            run_type="A",
            delay_tolerance_s=0.01,
            angle_tolerance_deg=0.5,
            tolerance_m=20.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_count": 0.0,
                    "unmatched_native_count": 0.0,
                    "delay_mae_s": 1.0e-5,
                    "amplitude_median_relative_error": 0.02,
                    "amplitude_p90_relative_error": 0.02,
                    "launch_declination_mae_deg": 0.01,
                    "launch_azimuth_mae_deg": 0.01,
                    "receiver_declination_mae_deg": 0.01,
                    "receiver_azimuth_mae_deg": 0.01,
                },
                min_values={
                    "matched_count": 12.0,
                    "match_fraction": 1.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DArrivalBenchmarkCase(
            "bellhop3d-iso-source-depth-xy-arrivals",
            source_depth_xy,
            run_type="A",
            delay_tolerance_s=0.01,
            angle_tolerance_deg=0.5,
            tolerance_m=20.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_count": 0.0,
                    "unmatched_native_count": 8.0,
                    "delay_mae_s": 1.0e-5,
                    "amplitude_median_relative_error": 0.02,
                    "amplitude_p90_relative_error": 0.02,
                    "launch_declination_mae_deg": 0.01,
                    "launch_azimuth_mae_deg": 0.01,
                    "receiver_declination_mae_deg": 0.01,
                    "receiver_azimuth_mae_deg": 0.01,
                },
                min_values={
                    "matched_count": 24.0,
                    "match_fraction": 1.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DArrivalBenchmarkCase(
            "bellhop3d-munk-slanted-terrain-oblique-arrivals",
            munk_oblique,
            run_type="A",
            delay_tolerance_s=0.03,
            angle_tolerance_deg=3.0,
            tolerance_m=50.0,
            max_arrivals=100,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_count": 2.0,
                    "unmatched_native_count": 3.0,
                    "delay_mae_s": 5.0e-4,
                    "amplitude_median_relative_error": 0.2,
                    "launch_declination_mae_deg": 0.6,
                    "launch_azimuth_mae_deg": 0.5,
                    "receiver_declination_mae_deg": 0.5,
                    "receiver_azimuth_mae_deg": 0.5,
                },
                min_values={
                    "matched_count": 50.0,
                    "match_fraction": 0.95,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
    )


def canonical_bellhop3d_ray_benchmark_cases() -> tuple[Bellhop3DRayBenchmarkCase, ...]:
    """Return canonical Bellhop3D ray path parity residual trackers."""

    iso_eigenray = _iso_direct_bellhop3d_config()
    iso_eigenray.project = "bellhop3d parity iso direct eigenrays"
    iso_eigenray.geometry.receivers.ranges = [500.0]
    iso_eigenray.geometry.receivers.depths = [100.0]
    iso_eigenray.solver.elevation_angles = (-2.0, 2.0)
    iso_eigenray.solver.azimuth_angles = (-2.0, 2.0)
    iso_eigenray.solver.num_elevation_beams = 21
    iso_eigenray.solver.num_azimuth_beams = 3
    iso_eigenray.solver.step_size = 10.0
    iso_off_axis_eigenray = _iso_direct_bellhop3d_config()
    iso_off_axis_eigenray.project = "bellhop3d parity off-axis direct eigenrays"
    iso_off_axis_eigenray.geometry.receivers.ranges = [500.0]
    iso_off_axis_eigenray.geometry.receivers.depths = [100.0]
    iso_off_axis_eigenray.solver.receiver_bearings = [2.0]
    iso_off_axis_eigenray.solver.elevation_angles = (-2.0, 2.0)
    iso_off_axis_eigenray.solver.azimuth_angles = (-2.0, 2.0)
    iso_off_axis_eigenray.solver.num_elevation_beams = 21
    iso_off_axis_eigenray.solver.num_azimuth_beams = 5
    iso_off_axis_eigenray.solver.step_size = 10.0
    iso_wide_eigenray = _iso_direct_bellhop3d_config()
    iso_wide_eigenray.project = "bellhop3d parity wide-bearing direct eigenrays"
    iso_wide_eigenray.geometry.receivers.ranges = [500.0]
    iso_wide_eigenray.geometry.receivers.depths = [100.0]
    iso_wide_eigenray.solver.receiver_bearings = [30.0]
    iso_wide_eigenray.solver.elevation_angles = (-2.0, 2.0)
    iso_wide_eigenray.solver.azimuth_angles = (28.0, 32.0)
    iso_wide_eigenray.solver.num_elevation_beams = 21
    iso_wide_eigenray.solver.num_azimuth_beams = 5
    iso_wide_eigenray.solver.step_size = 10.0
    iso_wide_eigenray.solver.box_y = 600.0
    iso_multi_depth_eigenray = _iso_direct_bellhop3d_config()
    iso_multi_depth_eigenray.project = "bellhop3d parity multi-depth direct eigenrays"
    iso_multi_depth_eigenray.geometry.receivers.ranges = [500.0]
    iso_multi_depth_eigenray.geometry.receivers.depths = [80.0, 100.0, 120.0]
    iso_multi_depth_eigenray.solver.elevation_angles = (-5.0, 5.0)
    iso_multi_depth_eigenray.solver.azimuth_angles = (-2.0, 2.0)
    iso_multi_depth_eigenray.solver.num_elevation_beams = 41
    iso_multi_depth_eigenray.solver.num_azimuth_beams = 3
    iso_multi_depth_eigenray.solver.step_size = 10.0
    iso_source_xy_eigenray = _iso_direct_bellhop3d_config()
    iso_source_xy_eigenray.project = "bellhop3d parity source-xy direct eigenrays"
    iso_source_xy_eigenray.geometry.source.x = [0.0, 20.0]
    iso_source_xy_eigenray.geometry.source.y = [0.0, 10.0]
    iso_source_xy_eigenray.geometry.receivers.ranges = [500.0]
    iso_source_xy_eigenray.geometry.receivers.depths = [100.0]
    iso_source_xy_eigenray.solver.elevation_angles = (-2.0, 2.0)
    iso_source_xy_eigenray.solver.azimuth_angles = (-2.0, 2.0)
    iso_source_xy_eigenray.solver.num_elevation_beams = 21
    iso_source_xy_eigenray.solver.num_azimuth_beams = 3
    iso_source_xy_eigenray.solver.step_size = 10.0
    iso_source_xy_eigenray.solver.box_x = 700.0
    iso_source_xy_eigenray.solver.box_y = 200.0
    iso_source_depth_xy_eigenray = _iso_direct_bellhop3d_config()
    iso_source_depth_xy_eigenray.project = "bellhop3d parity source-depth-xy direct eigenrays"
    iso_source_depth_xy_eigenray.geometry.source.depths = [80.0, 120.0]
    iso_source_depth_xy_eigenray.geometry.source.x = [0.0, 20.0]
    iso_source_depth_xy_eigenray.geometry.source.y = [0.0, 10.0]
    iso_source_depth_xy_eigenray.geometry.receivers.ranges = [500.0]
    iso_source_depth_xy_eigenray.geometry.receivers.depths = [100.0]
    iso_source_depth_xy_eigenray.solver.elevation_angles = (-5.0, 5.0)
    iso_source_depth_xy_eigenray.solver.azimuth_angles = (-2.0, 2.0)
    iso_source_depth_xy_eigenray.solver.num_elevation_beams = 41
    iso_source_depth_xy_eigenray.solver.num_azimuth_beams = 3
    iso_source_depth_xy_eigenray.solver.step_size = 10.0
    iso_source_depth_xy_eigenray.solver.box_x = 700.0
    iso_source_depth_xy_eigenray.solver.box_y = 200.0
    munk_slanted = _munk_slanted_terrain_bellhop3d_config()
    munk_oblique = copy.deepcopy(munk_slanted)
    munk_oblique.project = "bellhop3d parity Munk slanted terrain oblique rays"
    munk_oblique.solver.azimuth_angles = (30.0, 30.0)
    munk_oblique.solver.num_azimuth_beams = 1
    munk_steep_oblique = copy.deepcopy(munk_slanted)
    munk_steep_oblique.project = "bellhop3d parity Munk slanted terrain steep oblique rays"
    munk_steep_oblique.solver.azimuth_angles = (60.0, 60.0)
    munk_steep_oblique.solver.num_azimuth_beams = 1
    return (
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-iso-direct-eigenrays",
            iso_eigenray,
            run_type="E",
            max_range_m=1000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 1.0e-9,
                    "y_p90_abs_m": 1.0e-9,
                    "launch_azimuth_mae_deg": 1.0e-9,
                },
                min_values={
                    "sample_count": 800.0,
                    "matched_ray_count": 17.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-iso-off-axis-eigenrays",
            iso_off_axis_eigenray,
            run_type="E",
            max_range_m=1000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 1.0e-9,
                    "y_p90_abs_m": 1.0e-9,
                    "launch_azimuth_mae_deg": 1.0e-9,
                },
                min_values={
                    "sample_count": 800.0,
                    "matched_ray_count": 16.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-iso-wide-bearing-eigenrays",
            iso_wide_eigenray,
            run_type="E",
            max_range_m=1000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 1.0e-9,
                    "y_p90_abs_m": 1.0e-9,
                    "launch_azimuth_mae_deg": 1.0e-9,
                },
                min_values={
                    "sample_count": 1000.0,
                    "matched_ray_count": 21.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-iso-multi-depth-eigenrays",
            iso_multi_depth_eigenray,
            run_type="E",
            max_range_m=1000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 1.0e-9,
                    "y_p90_abs_m": 1.0e-9,
                    "launch_azimuth_mae_deg": 1.0e-9,
                },
                min_values={
                    "sample_count": 2000.0,
                    "matched_ray_count": 41.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-iso-source-xy-eigenrays",
            iso_source_xy_eigenray,
            run_type="E",
            max_range_m=1000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 1.0e-9,
                    "y_p90_abs_m": 1.0e-9,
                    "launch_azimuth_mae_deg": 1.0e-9,
                },
                min_values={
                    "sample_count": 1000.0,
                    "matched_ray_count": 21.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-iso-source-depth-xy-eigenrays",
            iso_source_depth_xy_eigenray,
            run_type="E",
            max_range_m=1000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 1.0e-9,
                    "y_p90_abs_m": 1.0e-9,
                    "launch_azimuth_mae_deg": 1.0e-9,
                },
                min_values={
                    "sample_count": 2000.0,
                    "matched_ray_count": 41.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-munk-slanted-terrain-rays",
            munk_slanted,
            run_type="R",
            max_range_m=10000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 2.5,
                    "y_p90_abs_m": 1.0e-9,
                },
                min_values={
                    "sample_count": 20000.0,
                    "matched_ray_count": 20.0,
                    "bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-munk-slanted-terrain-oblique-rays",
            munk_oblique,
            run_type="R",
            max_range_m=10000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 2.5,
                    "depth_max_abs_m": 3.5,
                    "y_p90_abs_m": 1.0e-8,
                    "launch_azimuth_mae_deg": 1.0e-9,
                },
                min_values={
                    "sample_count": 20000.0,
                    "matched_ray_count": 20.0,
                    "bounce_match_fraction": 1.0,
                    "top_bounce_match_fraction": 1.0,
                    "bottom_bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
        Bellhop3DRayBenchmarkCase(
            "bellhop3d-munk-slanted-terrain-steep-oblique-rays",
            munk_steep_oblique,
            run_type="R",
            max_range_m=10000.0,
            thresholds=BenchmarkThresholds(
                max_values={
                    "unmatched_at_ray_count": 0.0,
                    "depth_p90_abs_m": 2.5,
                    "depth_max_abs_m": 3.5,
                    "y_p90_abs_m": 1.0e-8,
                    "launch_azimuth_mae_deg": 1.0e-9,
                },
                min_values={
                    "sample_count": 20000.0,
                    "matched_ray_count": 20.0,
                    "bounce_match_fraction": 1.0,
                    "top_bounce_match_fraction": 1.0,
                    "bottom_bounce_match_fraction": 1.0,
                },
                ignore_missing=True,
            ),
        ),
    )


def canonical_kraken_benchmark_cases() -> tuple[KrakenBenchmarkCase, ...]:
    """Return a small canonical Kraken/Field AT parity matrix."""

    calib = _calib_like_kraken_config()
    rigid = _rigid_pekeris_kraken_config()
    rigid_surface = _rigid_surface_kraken_config()
    field_rc = _field_option_rc_kraken_config()
    field_x_line = _field_option_x_line_kraken_config()
    field_s_scaled = _field_option_s_scaled_kraken_config()
    field_multi_source_depth = _field_multi_source_depth_kraken_config()
    field_offsets = _field_offsets_max_modes_kraken_config()
    field_truncated_max_modes = _field_truncated_max_modes_kraken_config()
    wide_grid = _wide_grid_rigid_kraken_config()
    deep_gradient = _deep_gradient_rigid_kraken_config()
    reflection = _reflection_table_kraken_config()
    surface_reflection = _surface_reflection_table_kraken_config()
    surface_bottom_reflection = _surface_bottom_reflection_table_kraken_config()
    water_attenuation = _water_attenuation_kraken_config()
    two_layer = _two_layer_acoustic_kraken_config()
    nearfield_acoustic = _nearfield_acoustic_halfspace_kraken_config()
    lossy = _lossy_elastic_kraken_config()
    lossy_field_x = _lossy_elastic_field_x_line_kraken_config()
    layered_lossy = _layered_lossy_elastic_kraken_config()
    layered_lossy_field_s = _layered_lossy_elastic_field_s_scaled_kraken_config()
    return (
        KrakenBenchmarkCase(
            "kraken-calib-like-single-point",
            calib,
            r_grid=np.array([2500.0]),
            z_grid=np.array([50.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-pekeris-grid",
            rigid,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([25.0, 50.0, 75.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-surface-grid",
            rigid_surface,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([25.0, 50.0, 75.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-field-rc-grid",
            field_rc,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([25.0, 50.0, 75.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-field-x-line-grid",
            field_x_line,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([25.0, 50.0, 75.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-field-s-scaled-grid",
            field_s_scaled,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([25.0, 50.0, 75.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-field-multi-source-depth-grid",
            field_multi_source_depth,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([25.0, 50.0, 75.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-field-offsets-maxmodes-grid",
            field_offsets,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([25.0, 50.0, 75.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-field-truncated-maxmodes-grid",
            field_truncated_max_modes,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([25.0, 50.0, 75.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-wide-receiver-grid",
            wide_grid,
            r_grid=np.array([600.0, 1400.0, 2600.0, 4200.0]),
            z_grid=np.array([20.0, 55.0, 95.0, 130.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-rigid-deep-gradient-grid",
            deep_gradient,
            r_grid=np.array([800.0, 1800.0, 3600.0, 6000.0, 8500.0]),
            z_grid=np.array([40.0, 110.0, 190.0, 250.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-bottom-reflection-table-grid",
            reflection,
            r_grid=np.array([750.0, 1500.0]),
            z_grid=np.array([40.0, 80.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-surface-reflection-table-grid",
            surface_reflection,
            r_grid=np.array([750.0, 1500.0]),
            z_grid=np.array([40.0, 80.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-surface-bottom-reflection-table-grid",
            surface_bottom_reflection,
            r_grid=np.array([750.0, 1500.0]),
            z_grid=np.array([40.0, 80.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-water-attenuation-grid",
            water_attenuation,
            r_grid=np.array([1000.0, 2500.0]),
            z_grid=np.array([30.0, 60.0, 90.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-two-layer-acoustic-grid",
            two_layer,
            r_grid=np.array([1000.0, 2200.0]),
            z_grid=np.array([40.0, 90.0, 140.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-nearfield-acoustic-halfspace-grid",
            nearfield_acoustic,
            r_grid=np.array([100.0, 300.0, 600.0]),
            z_grid=np.array([120.0, 190.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-extreme-nearfield-acoustic-halfspace-grid",
            nearfield_acoustic,
            r_grid=np.array([50.0, 100.0, 300.0]),
            z_grid=np.array([120.0, 190.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-lossy-elastic-bottom-grid",
            lossy,
            r_grid=np.array([1000.0, 3000.0]),
            z_grid=np.array([50.0, 100.0, 150.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-lossy-elastic-field-x-line-grid",
            lossy_field_x,
            r_grid=np.array([1000.0, 3000.0]),
            z_grid=np.array([50.0, 100.0, 150.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-layered-lossy-elastic-grid",
            layered_lossy,
            r_grid=np.array([800.0, 1800.0, 4200.0]),
            z_grid=np.array([60.0, 135.0, 210.0]),
        ),
        KrakenBenchmarkCase(
            "kraken-layered-lossy-elastic-field-s-scaled-grid",
            layered_lossy_field_s,
            r_grid=np.array([800.0, 1800.0, 4200.0]),
            z_grid=np.array([60.0, 135.0, 210.0]),
        ),
    )


def bellhop_smoke_thresholds() -> BenchmarkThresholds:
    """Loose smoke thresholds for the canonical Bellhop matrix."""

    return BenchmarkThresholds(
        max_values={
            "tl_mae_db": 1.0,
            "tl_p90_abs_db": 3.0,
            "delay_mae_s": 0.02,
            "depth_mae_m": 2.0,
            "depth_p90_abs_m": 0.02,
            "amplitude_median_relative_error": 0.01,
            "phase_mae_rad": 0.01,
            "launch_angle_mae_deg": 0.01,
            "receiver_angle_mae_deg": 0.01,
        },
        min_values={"sample_count": 1.0, "match_fraction": 0.5, "bounce_match_fraction": 0.98},
        max_abs_values={"tl_bias_db": 3.0},
        ignore_missing=True,
    )


def bellhop3d_smoke_thresholds() -> BenchmarkThresholds:
    """Loose smoke thresholds for the canonical Bellhop3D matrix."""

    return BenchmarkThresholds(
        max_values={"tl_mae_db": 1.0, "tl_p90_abs_db": 3.0, "tl_max_abs_db": 5.0},
        min_values={"sample_count": 1.0},
        max_abs_values={"tl_bias_db": 3.0},
        ignore_missing=True,
    )


def kraken_smoke_thresholds() -> BenchmarkThresholds:
    """Loose smoke thresholds for the canonical Kraken matrix."""

    return BenchmarkThresholds(
        max_values={"tl_mae_db": 1.0, "tl_max_abs_db": 1.0},
        min_values={"sample_count": 1.0},
        max_abs_values={"tl_bias_db": 1.0, "mode_count_delta": 0.0},
        equals={
            "at_error_count": 0,
            "at_fatal_error_count": 0,
            "field_error_count": 0,
            "field_fatal_error_count": 0,
            "field_completed_without_errors": True,
        },
        ignore_missing=True,
    )


__all__ = [
    "bellhop3d_smoke_thresholds",
    "bellhop_smoke_thresholds",
    "canonical_bellhop3d_arrival_benchmark_cases",
    "canonical_bellhop3d_benchmark_cases",
    "canonical_bellhop3d_ray_benchmark_cases",
    "canonical_bellhop_benchmark_cases",
    "canonical_kraken_benchmark_cases",
    "kraken_smoke_thresholds",
]
