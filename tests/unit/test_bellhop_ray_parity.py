import numpy as np
import pytest

from pyacoustics.schema import (
    BellhopConfig,
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
from pyacoustics.simulation import Simulation
from pyacoustics.solvers.bellhop.ray_parity import compare_ray_records, compute_native_ray_records, ray_benchmark_row
from pyacoustics.solvers.external_io import ATRayRecord


def _ray(depth_offset=0.0, *, top=0, bottom=0):
    ranges = np.array([0.0, 500.0, 1000.0])
    depths = np.array([50.0, 60.0, 70.0]) + depth_offset
    return ATRayRecord(
        launch_angle_deg=0.0,
        ranges_m=ranges,
        depths_m=depths,
        amplitudes=np.ones_like(ranges, dtype=np.complex128),
        num_top_bounces=top,
        num_bottom_bounces=bottom,
    )


def test_ray_parity_reports_depth_percentile_and_unmatched_counts():
    at_records = [_ray(0.0, top=0, bottom=0), _ray(0.0, top=1, bottom=0), _ray(0.0)]
    native_records = [_ray(1.0, top=0, bottom=0), _ray(3.0, top=1, bottom=1)]

    metrics = compare_ray_records(at_records, native_records, sample_count=3)

    assert metrics.at_count == 3
    assert metrics.native_count == 2
    assert metrics.matched_count == 2
    assert metrics.unmatched_at_count == 1
    assert metrics.unmatched_native_count == 0
    assert np.isclose(metrics.match_fraction, 2.0 / 3.0)
    assert np.isclose(metrics.depth_mae, 2.0)
    assert np.isclose(metrics.depth_p90_abs, 3.0)
    assert np.isclose(metrics.depth_max_abs, 3.0)
    assert metrics.bounce_match_count == 1
    assert np.isclose(metrics.bounce_match_fraction, 0.5)


def test_ray_benchmark_row_flattens_metrics_for_reports():
    metrics = compare_ray_records([_ray(0.0)], [_ray(2.0)], sample_count=3)

    row = ray_benchmark_row("munk-rays", metrics, run_type="R")

    assert row["case"] == "munk-rays"
    assert row["run_type"] == "R"
    assert row["at_count"] == 1
    assert row["matched_count"] == 1
    assert row["depth_mae_m"] == 2.0
    assert row["depth_p90_abs_m"] == 2.0
    assert row["bounce_match_fraction"] == 1.0


def test_compute_native_ray_records_can_extend_trace_box_for_at_parity():
    config = SimulationConfig(
        project="bellhop native ray trace range",
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
        solver=BellhopConfig(type="bellhop", angles=(-8.0, -8.0), num_beams=1, step_size=5.0),
    )

    default_record = compute_native_ray_records(config)[0]
    extended_record = compute_native_ray_records(config, trace_range_m=1100.0)[0]

    assert default_record.ranges_m[-1] < 1100.0
    assert extended_record.ranges_m[-1] >= 1100.0
    assert config.geometry.receivers.ranges == [1000.0]


def test_simulation_compare_rays_with_at_rejects_non_bellhop_solver():
    config = SimulationConfig(
        project="kraken no ray parity",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0], depths=[50.0]),
        ),
        solver=KrakenConfig(n_mesh_points=100, mesh_multiplier=[1]),
    )

    with pytest.raises(NotImplementedError, match="Ray parity"):
        Simulation(config).compare_rays_with_at()
