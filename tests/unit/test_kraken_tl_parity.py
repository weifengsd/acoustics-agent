from pathlib import Path

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
from pyacoustics.solvers.external_io import read_at_prt
from pyacoustics.solvers.kraken.tl_parity import (
    BenchmarkThresholds,
    KrakenATTLResult,
    KrakenBenchmarkCase,
    KrakenTLParityMetrics,
    KrakenTLParityReport,
    compare_mode_summaries,
    compare_native_modes_to_prt,
    compare_tl_grids,
    compute_native_tl_grid,
    kraken_benchmark_row,
    run_at_tl_grid,
    run_kraken_benchmark_matrix,
    run_kraken_tl_parity,
    summarize_native_modes,
    summarize_prt_modes,
)
from pyacoustics.solvers.kraken.field import compute_field


def _kraken_config():
    return SimulationConfig(
        project="native kraken parity grid",
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
            receivers=ReceiverModel(ranges=[100.0, 500.0, 1000.0], depths=[10.0, 50.0, 90.0]),
        ),
        solver=KrakenConfig(n_mesh_points=120, mesh_multiplier=[1], phase_speed_limits=(1450.0, 1700.0)),
    )


def test_compare_kraken_tl_grids_reports_matrix_metrics():
    at_tl = np.array([[40.0, 50.0, 120.0], [60.0, np.nan, 80.0]])
    native_tl = np.array([[41.0, 48.0, 121.0], [63.0, 70.0, 79.0]])

    metrics = compare_tl_grids(at_tl, native_tl)

    assert metrics.sample_count == 4
    assert np.isclose(metrics.mae_db, 1.75)
    assert np.isclose(metrics.median_abs_db, 1.5)
    assert np.isclose(metrics.p90_abs_db, 2.7)
    assert np.isclose(metrics.max_abs_db, 3.0)
    assert np.isclose(metrics.bias_db, 0.25)
    assert np.isclose(metrics.rmse_db, np.sqrt(15.0 / 4.0))


def test_compare_kraken_tl_grids_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        compare_tl_grids(np.ones((2, 2)), np.ones((2, 3)))


def test_compute_native_kraken_tl_grid_uses_explicit_receiver_grid():
    config = _kraken_config()

    tl_grid = compute_native_tl_grid(
        config,
        r_grid=np.array([100.0, 500.0, 1000.0]),
        z_grid=np.array([10.0, 50.0, 90.0]),
    )

    assert tl_grid.shape == (3, 3)
    assert np.isfinite(tl_grid).all()


def test_compute_native_kraken_tl_grid_keeps_multi_source_depth_axis():
    config = _kraken_config()
    config.geometry.source.depths = [40.0, 60.0]

    tl_grid = compute_native_tl_grid(
        config,
        r_grid=np.array([100.0, 500.0, 1000.0]),
        z_grid=np.array([10.0, 50.0, 90.0]),
    )

    assert tl_grid.shape == (2, 3, 3)
    assert np.isfinite(tl_grid).all()
    assert not np.allclose(tl_grid[0], tl_grid[1])


def test_compute_field_honors_at_field_range_options():
    modes = np.ones((1, 3), dtype=np.float64)
    roots = np.array([4.0 + 0.0j])
    z_arr = np.array([0.0, 50.0, 100.0])
    r_arr = np.array([100.0])
    z_r = np.array([50.0])

    point_tl = compute_field(modes, roots, z_arr, r_arr, 50.0, z_r, field_option="RC")
    scaled_tl = compute_field(modes, roots, z_arr, r_arr, 50.0, z_r, field_option="SC")
    line_tl = compute_field(modes, roots, z_arr, r_arr, 50.0, z_r, field_option="XC")

    assert np.isclose(scaled_tl[0, 0] - point_tl[0, 0], -20.0)
    assert np.isclose(line_tl[0, 0] - scaled_tl[0, 0], 3.010299956639812)


def test_run_kraken_tl_parity_returns_metrics_modes_log_and_row(monkeypatch, tmp_path):
    from pyacoustics.solvers.kraken import tl_parity as kraken_tl_parity_module

    prt_path = tmp_path / "kraken.prt"
    prt_path.write_text(
        " KRAKENC parity case\n"
        " Kraken completed successfully\n"
        "    I       k             alpha     Phase Speed     Group Speed\n"
        "          (1/m)           (1/m)        (m/s)           (m/s)\n"
        "    1   1.000000000      0.10E-05   1500.000000       1499.41\n"
        "    2   1.100000000      0.20E-05   1600.000000       1497.64\n"
    )
    log = read_at_prt(prt_path)
    field_prt_path = tmp_path / "field.prt"
    field_prt_path.write_text(
        " Running FIELD\n"
        " Sums modes, producing pressure\n"
        "    Number of Profile ranges, RProf =            1\n"
        "    Number of Receiver r-coordinates, Rr =            2\n"
        "    Number of Receiver z-coordinates, Rz =            2\n"
        "    Number of receiver range offsets (array tilt), Rro =            2\n"
        " Field completed successfully\n"
    )
    field_log = read_at_prt(field_prt_path)
    native_result = {
        "tl_grid": np.array([[40.0, 50.0], [60.0, 70.0]]),
        "r_bins": np.array([100.0, 200.0]),
        "z_bins": np.array([10.0, 20.0]),
        "wavenumbers": np.array([1.01 + 0.000001j, 1.12 + 0.000002j]),
        "phase_speeds": np.array([1501.0, 1602.0]),
        "modal_attenuation": np.array([0.000001, 0.000002]),
    }
    at_result = KrakenATTLResult(
        tl_grid=np.array([[41.0, 49.0], [61.0, 69.0]]),
        receiver_ranges_m=np.array([100.0, 200.0]),
        receiver_depths_m=np.array([10.0, 20.0]),
        kraken_log=log,
        field_log=field_log,
    )

    monkeypatch.setattr(
        kraken_tl_parity_module,
        "compute_native_result",
        lambda config, r_grid, z_grid: native_result,
    )
    monkeypatch.setattr(
        kraken_tl_parity_module,
        "run_at_tl_grid",
        lambda config, r_grid, z_grid, *, at_bin_path=None: at_result,
    )

    report = run_kraken_tl_parity(
        _kraken_config(),
        np.array([100.0, 200.0]),
        np.array([10.0, 20.0]),
        at_bin_path="/opt/at/bin",
        case_name="pekeris-kraken",
    )

    assert report.tl_metrics.sample_count == 4
    assert np.isclose(report.tl_metrics.mae_db, 1.0)
    assert report.mode_metrics.mode_count_delta == 0
    assert np.isclose(report.mode_metrics.wavenumber_min_error_1pm, 0.01)
    assert report.benchmark_row["case"] == "pekeris-kraken"
    assert report.benchmark_row["kind"] == "kraken_tl"
    assert report.benchmark_row["tl_mae_db"] == 1.0
    assert report.benchmark_row["at_completed_without_errors"]
    assert report.benchmark_row["field_completed_without_errors"]
    assert report.benchmark_row["field_receiver_range_count"] == 2
    assert report.benchmark_row["field_receiver_depth_count"] == 2
    assert report.benchmark_row["field_receiver_range_offset_count"] == 2
    assert report.at_result is at_result


def test_run_at_tl_grid_preserves_separate_kraken_and_field_prt_logs(monkeypatch, tmp_path):
    from pyacoustics.solvers.kraken import tl_parity as kraken_tl_parity_module

    class FakeExternalSolver:
        def __init__(self, config, bin_path=None):
            self.bin_path = tmp_path
            self.calls = []

        def is_available(self, exe_name):
            return exe_name in {"kraken", "krakenc", "field"}

        def run_command(self, command, *, cwd=None, stdin_input=None):
            cwd_path = Path(cwd)
            exe_name = Path(command[0]).stem
            if exe_name in {"kraken", "krakenc"}:
                (cwd_path / "kraken_tl_parity.prt").write_text(
                    " KRAKEN parity case\n"
                    " Number of modes =            2\n"
                    "    I       k             alpha     Phase Speed     Group Speed\n"
                    "          (1/m)           (1/m)        (m/s)           (m/s)\n"
                    "    1   1.000000000      0.10E-05   1500.000000       1499.41\n"
                    "    2   1.100000000      0.20E-05   1600.000000       1497.64\n"
                )
            else:
                (cwd_path / "field.prt").write_text(
                    " Running FIELD\n"
                    " Sums modes, producing pressure\n"
                    "    Number of Profile ranges, RProf =            1\n"
                    "    Number of Receiver r-coordinates, Rr =            2\n"
                    "    Number of Receiver z-coordinates, Rz =            2\n"
                    "    Number of receiver range offsets (array tilt), Rro =            2\n"
                    " Field completed successfully\n"
                )
                (cwd_path / "kraken_tl_parity.shd").touch()

    monkeypatch.setattr(kraken_tl_parity_module, "ExternalSolver", FakeExternalSolver)
    monkeypatch.setattr(kraken_tl_parity_module, "generate_at_kraken_env", lambda config, path: Path(path).write_text(""))
    monkeypatch.setattr(kraken_tl_parity_module, "generate_at_field_flp", lambda config, path: Path(path).write_text(""))

    class FakeShade:
        pressure = np.array([[[1.0 + 0.0j, 0.5 + 0.0j], [0.25 + 0.0j, 0.125 + 0.0j]]])
        receiver_depths_m = np.array([10.0, 20.0])
        receiver_ranges_m = np.array([100.0, 200.0])

    monkeypatch.setattr(
        kraken_tl_parity_module,
        "read_at_shd_full",
        lambda path: FakeShade(),
    )

    result = run_at_tl_grid(_kraken_config(), np.array([100.0, 200.0]), np.array([10.0, 20.0]))

    assert result.kraken_log.program == "KRAKEN parity case"
    assert result.kraken_log.mode_count == 2
    assert result.field_log.program == "Running FIELD"
    assert result.field_log.completed_without_errors
    assert result.field_log.receiver_range_count == 2


def test_run_kraken_benchmark_matrix_collects_reports_and_rows(monkeypatch):
    from pyacoustics.solvers.kraken import tl_parity as kraken_tl_parity_module

    calls = []

    def fake_run_kraken_tl_parity(config, r_grid, z_grid, *, at_bin_path=None, case_name="kraken-tl-parity"):
        calls.append(
            {
                "r_grid": np.asarray(r_grid, dtype=np.float64),
                "z_grid": np.asarray(z_grid, dtype=np.float64),
                "at_bin_path": at_bin_path,
                "case_name": case_name,
            }
        )
        return KrakenTLParityReport(
            tl_metrics=KrakenTLParityMetrics(
                sample_count=1,
                mae_db=0.1,
                median_abs_db=0.1,
                p90_abs_db=0.1,
                max_abs_db=0.1,
                bias_db=0.0,
                rmse_db=0.1,
            ),
            mode_metrics=None,
            at_result=KrakenATTLResult(
                tl_grid=np.array([[40.0]]),
                receiver_ranges_m=np.asarray(r_grid, dtype=np.float64),
                receiver_depths_m=np.asarray(z_grid, dtype=np.float64),
                kraken_log=None,
                field_log=None,
            ),
            benchmark_row={"case": case_name, "tl_mae_db": 0.1},
        )

    monkeypatch.setattr(kraken_tl_parity_module, "run_kraken_tl_parity", fake_run_kraken_tl_parity)

    config = _kraken_config()
    matrix = run_kraken_benchmark_matrix(
        (
            KrakenBenchmarkCase("default-grid", config, at_bin_path="/case/bin"),
            KrakenBenchmarkCase(
                "explicit-grid",
                config,
                r_grid=np.array([250.0, 750.0]),
                z_grid=np.array([25.0]),
                at_bin_path="/ignored/bin",
            ),
        ),
        at_bin_path="/global/bin",
        thresholds=BenchmarkThresholds(max_values={"tl_mae_db": 0.05}),
    )

    assert len(matrix.reports) == 2
    assert matrix.rows == (
        {
            "case": "default-grid",
            "tl_mae_db": 0.1,
            "passed": False,
            "failure_count": 1,
            "failures": "tl_mae_db=0.1 > 0.05",
        },
        {
            "case": "explicit-grid",
            "tl_mae_db": 0.1,
            "passed": False,
            "failure_count": 1,
            "failures": "tl_mae_db=0.1 > 0.05",
        },
    )
    assert calls[0]["case_name"] == "default-grid"
    assert calls[0]["at_bin_path"] == "/global/bin"
    assert np.allclose(calls[0]["r_grid"], [100.0, 500.0, 1000.0])
    assert np.allclose(calls[0]["z_grid"], [10.0, 50.0, 90.0])
    assert calls[1]["case_name"] == "explicit-grid"
    assert calls[1]["at_bin_path"] == "/global/bin"
    assert np.allclose(calls[1]["r_grid"], [250.0, 750.0])
    assert np.allclose(calls[1]["z_grid"], [25.0])


def test_simulation_compare_kraken_tl_with_at_rejects_non_kraken_solver():
    config = SimulationConfig(
        project="bellhop no kraken parity",
        frequency=100.0,
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
        solver=BellhopConfig(num_beams=3, angles=(-5.0, 5.0), step_size=10.0),
    )

    with pytest.raises(NotImplementedError, match="Kraken TL parity"):
        Simulation(config).compare_kraken_tl_with_at()


def test_simulation_compare_kraken_tl_with_at_can_return_full_report(monkeypatch):
    from pyacoustics.solvers.kraken import tl_parity as kraken_tl_parity_module

    calls = []
    at_result = KrakenATTLResult(
        tl_grid=np.array([[40.0]]),
        receiver_ranges_m=np.array([100.0]),
        receiver_depths_m=np.array([50.0]),
        kraken_log=None,
        field_log=None,
    )
    report = KrakenTLParityReport(
        tl_metrics=KrakenTLParityMetrics(
            sample_count=1,
            mae_db=0.1,
            median_abs_db=0.1,
            p90_abs_db=0.1,
            max_abs_db=0.1,
            bias_db=0.1,
            rmse_db=0.1,
        ),
        mode_metrics=None,
        at_result=at_result,
        benchmark_row={"case": "kraken-api"},
    )

    def fake_run_kraken_tl_parity(config, r_grid, z_grid, *, at_bin_path=None, case_name="kraken-tl-parity"):
        calls.append(
            {
                "r_grid": np.asarray(r_grid, dtype=np.float64),
                "z_grid": np.asarray(z_grid, dtype=np.float64),
                "at_bin_path": at_bin_path,
                "case_name": case_name,
            }
        )
        return report

    monkeypatch.setattr(kraken_tl_parity_module, "run_kraken_tl_parity", fake_run_kraken_tl_parity)

    sim = Simulation(_kraken_config())

    metrics = sim.compare_kraken_tl_with_at(at_bin_path="/opt/at/bin", case_name="kraken-api")
    full_report = sim.compare_kraken_tl_with_at(return_report=True, case_name="kraken-api")

    assert metrics is report.tl_metrics
    assert full_report is report
    assert calls[0]["case_name"] == "kraken-api"
    assert calls[0]["at_bin_path"] == "/opt/at/bin"
    assert calls[1]["case_name"] == "kraken-api"
    assert calls[1]["at_bin_path"] is None
    assert np.allclose(calls[0]["r_grid"], [100.0, 500.0, 1000.0])
    assert np.allclose(calls[0]["z_grid"], [10.0, 50.0, 90.0])


def test_kraken_mode_summary_compares_native_and_prt(tmp_path):
    prt_path = tmp_path / "kraken.prt"
    prt_path.write_text(
        " KRAKENC parity case\n"
        " Nominal Frequency =   250.0    Hz\n"
        "    I       k             alpha     Phase Speed     Group Speed\n"
        "          (1/m)           (1/m)        (m/s)           (m/s)\n"
        "    1   1.046756917     -0.21E-05   1500.631428       1499.41\n"
        "    2   1.045433990     -0.86E-05   1502.530377       1497.64\n"
        "    3   1.043225728     -0.19E-04   1505.710878       1494.69\n"
    )
    log = read_at_prt(prt_path)
    native_result = {
        "wavenumbers": np.array([1.0430 + 0.000010j, 1.0450 + 0.000011j, 1.0470 + 0.000012j]),
        "phase_speeds": np.array([1506.0, 1503.0, 1500.0]),
        "modal_attenuation": np.array([0.000010, 0.000011, 0.000012]),
    }

    native_summary = summarize_native_modes(native_result)
    at_summary = summarize_prt_modes(log)
    metrics = compare_mode_summaries(native_summary, at_summary)
    direct_metrics = compare_native_modes_to_prt(native_result, log)

    assert native_summary.mode_count == 3
    assert at_summary.mode_count == 3
    assert metrics == direct_metrics
    assert metrics.mode_count_delta == 0
    assert np.isclose(metrics.wavenumber_min_error_1pm, 1.0430 - 1.043225728)
    assert np.isclose(metrics.wavenumber_max_error_1pm, 1.0470 - 1.046756917)
    assert np.isclose(metrics.attenuation_min_error_1pm, 0.000010 - 0.21e-5)
    assert np.isclose(metrics.attenuation_max_error_1pm, 0.000012 - 0.19e-4)
    assert np.isclose(metrics.phase_speed_min_error_mps, 1500.0 - 1500.631428)
    assert np.isclose(metrics.phase_speed_max_error_mps, 1506.0 - 1505.710878)


def test_kraken_benchmark_row_flattens_metrics_and_prt_status(tmp_path):
    prt_path = tmp_path / "field.prt"
    prt_path.write_text(
        " Running FIELD\n"
        " Field completed successfully\n"
        " CPU Time =       0.25    s\n"
        " WARNING: benign diagnostic\n"
    )
    log = read_at_prt(prt_path)
    tl_metrics = KrakenTLParityMetrics(
        sample_count=9,
        mae_db=0.1,
        median_abs_db=0.05,
        p90_abs_db=0.2,
        max_abs_db=0.3,
        bias_db=-0.01,
        rmse_db=float("nan"),
    )
    mode_metrics = compare_mode_summaries(
        summarize_native_modes(
            {
                "wavenumbers": np.array([1.0 + 0.0j]),
                "phase_speeds": np.array([1500.0]),
                "modal_attenuation": np.array([0.0]),
            }
        ),
        summarize_prt_modes(log),
    )

    row = kraken_benchmark_row("field-smoke", tl_metrics, mode_metrics, at_log=log)

    assert row["case"] == "field-smoke"
    assert row["sample_count"] == 9
    assert row["tl_mae_db"] == 0.1
    assert row["tl_rmse_db"] is None
    assert row["native_mode_count"] == 1
    assert row["at_mode_count"] is None
    assert row["mode_count_delta"] is None
    assert row["at_program"] == "Running FIELD"
    assert row["at_completed_without_errors"]
    assert row["at_warning_count"] == 1
    assert row["at_error_count"] == 0
    assert row["at_cpu_time_s"] == 0.25
