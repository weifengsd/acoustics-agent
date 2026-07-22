import numpy as np

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
from pyacoustics.solvers.bellhop import benchmark_matrix as bellhop_matrix_module
from pyacoustics.solvers.bellhop.benchmark_matrix import (
    BenchmarkThresholds,
    BellhopBenchmarkCase,
    run_bellhop_benchmark_matrix,
)
from pyacoustics.solvers.external_io import ATRayRecord


def _bellhop_config():
    return SimulationConfig(
        project="bellhop matrix",
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
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 500.0], depths=[25.0, 75.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-5.0, 5.0), num_beams=11, step_size=10.0),
    )


def _arrival_data(delay):
    arrivals = np.empty((1, 1, 1), dtype=object)
    arrivals[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j]),
        delays=np.array([delay + 0.0j]),
        src_decl_angles=np.array([0.0]),
        rcvr_decl_angles=np.array([0.0]),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([0], dtype=np.int16),
    )
    return ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([50.0]),
        receiver_depths=np.array([75.0]),
        receiver_ranges=np.array([500.0]),
        arrivals=arrivals,
    )


def _ray_record(offset=0.0):
    return ATRayRecord(
        launch_angle_deg=0.0,
        ranges_m=np.array([0.0, 100.0, 200.0]),
        depths_m=np.array([50.0 + offset, 55.0 + offset, 60.0 + offset]),
        amplitudes=np.ones(3, dtype=np.complex128),
        num_top_bounces=0,
        num_bottom_bounces=0,
    )


def test_run_bellhop_benchmark_matrix_collects_tl_arrival_and_ray_rows(monkeypatch):
    calls = []

    def fake_run_at_tl_grid(config, r_grid, z_grid, *, run_type="C", at_bin_path=None):
        calls.append(("tl-at", np.asarray(r_grid).tolist(), np.asarray(z_grid).tolist(), run_type, at_bin_path))
        return np.array([[40.0, 50.0], [60.0, 70.0]])

    def fake_compute_native_tl_grid(config, r_grid, z_grid, *, run_type="C"):
        calls.append(("tl-native", np.asarray(r_grid).tolist(), np.asarray(z_grid).tolist(), run_type))
        return np.array([[41.0, 49.0], [61.0, 69.0]])

    def fake_run_at_arrival_data(config, *, at_bin_path=None, run_type="A"):
        calls.append(("arrival-at", at_bin_path, run_type))
        return _arrival_data(1.0)

    def fake_compute_native_arrival_data(config, **kwargs):
        calls.append(("arrival-native", kwargs))
        return _arrival_data(1.005)

    def fake_run_at_ray_records(config, *, run_type="R", at_bin_path=None):
        calls.append(("ray-at", run_type, at_bin_path))
        return [_ray_record()]

    def fake_compute_native_ray_records(config, **kwargs):
        calls.append(("ray-native", kwargs))
        return [_ray_record(offset=0.5)]

    def fake_compute_native_eigenray_records(config, **kwargs):
        calls.append(("e-ray-native", kwargs))
        return [_ray_record(offset=0.25)]

    monkeypatch.setattr(bellhop_matrix_module.tl_parity, "run_at_tl_grid", fake_run_at_tl_grid)
    monkeypatch.setattr(bellhop_matrix_module.tl_parity, "compute_native_tl_grid", fake_compute_native_tl_grid)
    monkeypatch.setattr(bellhop_matrix_module.arrival_parity, "run_at_arrival_data", fake_run_at_arrival_data)
    monkeypatch.setattr(
        bellhop_matrix_module.arrival_parity,
        "compute_native_arrival_data",
        fake_compute_native_arrival_data,
    )
    monkeypatch.setattr(bellhop_matrix_module.ray_parity, "run_at_ray_records", fake_run_at_ray_records)
    monkeypatch.setattr(bellhop_matrix_module.ray_parity, "compute_native_ray_records", fake_compute_native_ray_records)
    monkeypatch.setattr(
        bellhop_matrix_module.ray_parity,
        "compute_native_eigenray_records",
        fake_compute_native_eigenray_records,
    )

    config = _bellhop_config()
    matrix = run_bellhop_benchmark_matrix(
        (
            BellhopBenchmarkCase(
                "munk-coherent",
                "tl",
                config,
                r_grid=np.array([100.0, 300.0]),
                z_grid=np.array([20.0, 80.0]),
                run_type="s",
                at_bin_path="/case/bin",
            ),
            BellhopBenchmarkCase(
                "munk-arrivals",
                "arrival",
                config,
                options={
                    "method": "eigenray",
                    "delay_tolerance": 0.01,
                    "max_arrivals": 9,
                    "beam_shape": "gaussian",
                    "run_type": "AB   3",
                    "eigenray_tolerance_m": 0.05,
                    "eigenray_angle_samples": 41,
                },
                thresholds=BenchmarkThresholds(max_values={"delay_mae_s": 0.01}, ignore_missing=True),
            ),
            BellhopBenchmarkCase(
                "munk-rays",
                "ray",
                config,
                options={"max_range_m": 200.0, "sample_count": 5},
            ),
            BellhopBenchmarkCase(
                "munk-e-rays",
                "ray",
                config,
                run_type="E",
                options={"max_range_m": 200.0, "sample_count": 5, "max_arrivals": 7},
            ),
        ),
        at_bin_path="/global/bin",
            thresholds=BenchmarkThresholds(
                max_values={"tl_mae_db": 0.75, "depth_mae_m": 1.0, "delay_mae_s": 0.001},
                min_values={"match_fraction": 0.99},
                ignore_missing=True,
            ),
        )

    assert len(matrix.rows) == 4
    assert matrix.rows[0]["kind"] == "tl"
    assert matrix.rows[0]["case"] == "munk-coherent"
    assert matrix.rows[0]["run_type"] == "S"
    assert matrix.rows[0]["tl_mae_db"] == 1.0
    assert not matrix.rows[0]["passed"]
    assert "tl_mae_db=1 > 0.75" in matrix.rows[0]["failures"]
    assert matrix.rows[1]["kind"] == "arrival"
    assert matrix.rows[1]["case"] == "munk-arrivals"
    assert matrix.rows[1]["method"] == "eigenray"
    assert matrix.rows[1]["run_type"] == "AB   3"
    assert matrix.rows[1]["matched_count"] == 1
    assert matrix.rows[1]["passed"]
    assert matrix.rows[1]["delay_mae_s"] == 0.004999999999999893
    assert matrix.rows[2]["kind"] == "ray"
    assert matrix.rows[2]["case"] == "munk-rays"
    assert matrix.rows[2]["depth_mae_m"] == 0.5
    assert matrix.rows[2]["passed"]
    assert matrix.rows[3]["kind"] == "ray"
    assert matrix.rows[3]["case"] == "munk-e-rays"
    assert matrix.rows[3]["run_type"] == "E"
    assert matrix.rows[3]["depth_mae_m"] == 0.25
    assert matrix.rows[3]["passed"]

    assert calls[0] == ("tl-at", [100.0, 300.0], [20.0, 80.0], "S", "/global/bin")
    assert calls[1] == ("tl-native", [100.0, 300.0], [20.0, 80.0], "S")
    assert calls[2] == ("arrival-at", "/global/bin", "AB   3")
    assert calls[3][0] == "arrival-native"
    assert calls[3][1]["method"] == "eigenray"
    assert calls[3][1]["max_arrivals"] == 9
    assert calls[3][1]["eigenray_angle_samples"] == 41
    assert calls[4] == ("ray-at", "R", "/global/bin")
    assert calls[5] == ("ray-native", {"trace_range_m": 550.0})
    assert calls[6] == ("ray-at", "E", "/global/bin")
    assert calls[7] == ("e-ray-native", {"max_arrivals": 7, "beam_shape": "hat", "trace_range_m": 550.0})
