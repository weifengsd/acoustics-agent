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
from pyacoustics.solvers.bellhop.tl_parity import (
    compare_tl_grids,
    compute_native_tl_grid,
    run_at_tl_grid,
    tl_benchmark_row,
)
from pyacoustics.solvers.external_io import read_at_shd_full


def _iso_config():
    return SimulationConfig(
        project="TL modes iso",
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
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=41,
            step_size=5.0,
        ),
    )


def test_simulation_exposes_incoherent_and_semicoherent_tl_modes():
    sim = Simulation(_iso_config())
    r_grid = np.array([100.0, 500.0, 1000.0])
    z_grid = np.array([25.0, 50.0, 75.0])

    incoherent = sim.compute_incoherent_tl(r_grid=r_grid, z_grid=z_grid)
    semicoherent = sim.compute_semicoherent_tl(r_grid=r_grid, z_grid=z_grid)

    assert incoherent.shape == (3, 3)
    assert semicoherent.shape == (3, 3)
    assert np.all(np.isfinite(incoherent))
    assert np.all(np.isfinite(semicoherent))
    assert hasattr(sim, "_incoherent_tl_cache")
    assert hasattr(sim, "_semicoherent_tl_cache")
    assert not np.allclose(incoherent, semicoherent)


def test_semicoherent_tl_applies_at_lloyd_mirror_source_factor():
    config = _iso_config()
    config.geometry.source.depths = [0.0]
    sim = Simulation(config)
    r_grid = np.array([500.0, 1000.0])
    z_grid = np.array([25.0, 50.0, 75.0])

    semicoherent = sim.compute_semicoherent_tl(r_grid=r_grid, z_grid=z_grid)

    assert np.all(semicoherent == 120.0)


def test_tl_parity_native_semicoherent_uses_s_run_type_kernel():
    config = _iso_config()
    config.geometry.source.depths = [0.0]
    r_grid = np.array([500.0, 1000.0])
    z_grid = np.array([25.0, 50.0, 75.0])

    incoherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="I")
    semicoherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="S")

    assert np.any(incoherent < 120.0)
    assert np.all(semicoherent == 120.0)


def test_tl_parity_native_semicoherent_accepts_single_receiver_grid():
    config = _iso_config()

    semicoherent = compute_native_tl_grid(
        config,
        r_grid=np.array([1000.0]),
        z_grid=np.array([50.0]),
        run_type="S",
    )

    assert semicoherent.shape == (1, 1)
    assert np.all(np.isfinite(semicoherent))


def test_tl_parity_native_keeps_multi_source_depth_axis():
    config = _iso_config()
    config.geometry.source.depths = [40.0, 60.0]
    r_grid = np.array([100.0, 500.0, 1000.0])
    z_grid = np.array([25.0, 50.0, 75.0])

    coherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="C")
    incoherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="I")
    semicoherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="S")

    assert coherent.shape == (2, 3, 3)
    assert incoherent.shape == (2, 3, 3)
    assert semicoherent.shape == (2, 3, 3)
    assert not np.allclose(coherent[0], coherent[1])
    assert not np.allclose(incoherent[0], incoherent[1])
    assert not np.allclose(semicoherent[0], semicoherent[1])


def test_tl_parity_native_simple_gaussian_coherent_uses_sgb_kernel(monkeypatch):
    from pyacoustics.solvers.bellhop import tl_parity as tl_parity_module

    config = _iso_config()
    config.solver.beam_type = "simple_gaussian"
    r_grid = np.array([1000.0])
    z_grid = np.array([50.0])
    sentinel = np.array([[42.0]])
    called = {}

    def fake_sgb(*args, **kwargs):
        called["sgb"] = True
        return sentinel

    monkeypatch.setattr(tl_parity_module, "compute_tl_field_simple_gaussian_coherent", fake_sgb)

    tl = compute_native_tl_grid(config, r_grid, z_grid, run_type="C")

    assert called["sgb"]
    assert tl is sentinel


def test_tl_parity_native_line_source_uses_at_pressure_scaling():
    config = _iso_config()
    r_grid = np.array([1000.0])
    z_grid = np.array([50.0])

    point_coherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="C")
    point_incoherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="I")
    config.solver.source_type = "line"
    line_coherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="C")
    line_incoherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="I")

    expected_drop = 20.0 * np.log10(4.0 * np.sqrt(np.pi) * np.sqrt(r_grid[0]))
    np.testing.assert_allclose(point_coherent - line_coherent, [[expected_drop]], atol=1.0e-9)
    np.testing.assert_allclose(point_incoherent - line_incoherent, [[expected_drop]], atol=2.0e-2)


def test_simulation_line_source_tl_uses_at_pressure_scaling():
    config = _iso_config()
    r_grid = np.array([1000.0])
    z_grid = np.array([50.0])

    point = Simulation(config).compute_coherent_tl(r_grid=r_grid, z_grid=z_grid)
    config.solver.source_type = "line"
    line = Simulation(config).compute_coherent_tl(r_grid=r_grid, z_grid=z_grid)

    expected_drop = 20.0 * np.log10(4.0 * np.sqrt(np.pi) * np.sqrt(r_grid[0]))
    np.testing.assert_allclose(point - line, [[expected_drop]], atol=1.0e-9)


def test_reflection_table_intensity_tl_uses_default_hat_beam_kernel():
    from pyacoustics.benchmarks.at_presets import canonical_bellhop_benchmark_cases

    config = next(
        case.config
        for case in canonical_bellhop_benchmark_cases()
        if case.name == "bottom-reflection-table-coherent-tl"
    )
    r_grid = np.array([1000.0])
    z_grid = np.array([65.0])

    incoherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="I")
    semicoherent = compute_native_tl_grid(config, r_grid, z_grid, run_type="S")

    np.testing.assert_allclose(incoherent, [[55.28706908]], atol=1.0e-6)
    np.testing.assert_allclose(semicoherent, [[54.67101278]], atol=1.0e-6)


def test_bellhop_tl_parity_reports_bias_rmse_and_benchmark_row():
    at_tl = np.array([[40.0, 50.0, 120.0], [60.0, np.nan, 80.0]])
    native_tl = np.array([[41.0, 48.0, 121.0], [63.0, 70.0, 79.0]])

    metrics = compare_tl_grids(at_tl, native_tl)
    row = tl_benchmark_row("iso-coherent", metrics, run_type="c")

    assert metrics.sample_count == 4
    assert np.isclose(metrics.mae_db, 1.75)
    assert np.isclose(metrics.bias_db, 0.25)
    assert np.isclose(metrics.rmse_db, np.sqrt(15.0 / 4.0))
    assert row["case"] == "iso-coherent"
    assert row["run_type"] == "C"
    assert row["tl_p90_abs_db"] == metrics.p90_abs_db
    assert row["tl_rmse_db"] == metrics.rmse_db


def test_run_at_tl_grid_reports_bellhop_prt_fatal_errors(monkeypatch, tmp_path):
    from pyacoustics.solvers.bellhop import tl_parity as tl_parity_module

    class FakeExternalSolver:
        def __init__(self, config, bin_path=None):
            self.bin_path = tmp_path

        def is_available(self, exe_name):
            return exe_name == "bellhop"

        def run_command(self, command, *, cwd=None, stdin_input=None):
            cwd_path = tmp_path if cwd is None else Path(cwd)
            (cwd_path / "tl_parity.prt").write_text(
                " BELLHOP\n"
                " *** FATAL ERROR ***\n"
                " Generated by program or subroutine: BELLHOP\n"
                " Number of ranges must be > 1\n"
            )
            (cwd_path / "tl_parity.shd").write_bytes(b"bad")

    monkeypatch.setattr(tl_parity_module, "ExternalSolver", FakeExternalSolver)

    with pytest.raises(RuntimeError, match="AT Bellhop TL run failed.*Number of ranges must be > 1"):
        run_at_tl_grid(_iso_config(), np.array([1000.0]), np.array([50.0]), at_bin_path=tmp_path)


def test_simulation_compare_tl_with_at_rejects_non_bellhop_solver():
    config = SimulationConfig(
        project="kraken no bellhop tl parity",
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

    with pytest.raises(NotImplementedError, match="Bellhop TL parity"):
        Simulation(config).compare_tl_with_at()


def test_compute_coherent_tl_writes_at_shd(tmp_path):
    sim = Simulation(_iso_config())
    r_grid = np.array([100.0, 500.0, 1000.0])
    z_grid = np.array([25.0, 50.0, 75.0])
    shd_path = tmp_path / "native_2d.shd"

    tl = sim.compute_coherent_tl(r_grid=r_grid, z_grid=z_grid, save_shd_path=shd_path)
    shade = read_at_shd_full(shd_path)
    pressure_abs = np.abs(shade.pressure[0, 0]).T
    tl_from_shd = -20.0 * np.log10(np.maximum(pressure_abs, 1e-20))
    tl_from_shd[pressure_abs < 1e-15] = 120.0
    pressure_grid, _, _ = sim._coherent_pressure_cache

    assert shade.pressure.shape == (1, 1, 3, 3)
    np.testing.assert_allclose(shade.source_depths_m, sim.config.geometry.source.depths)
    np.testing.assert_allclose(shade.receiver_depths_m, z_grid)
    np.testing.assert_allclose(shade.receiver_ranges_m, r_grid)
    np.testing.assert_allclose(tl_from_shd, tl, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(shade.pressure[0, 0].T, pressure_grid)
    assert np.max(np.abs(np.imag(shade.pressure))) > 0.0


def test_compute_coherent_tl_writes_multi_source_at_shd(tmp_path):
    config = _iso_config()
    config.geometry.source.depths = [40.0, 60.0]
    sim = Simulation(config)
    r_grid = np.array([100.0, 500.0, 1000.0])
    z_grid = np.array([25.0, 50.0, 75.0])
    shd_path = tmp_path / "native_2d_multi_source.shd"

    tl = sim.compute_coherent_tl(r_grid=r_grid, z_grid=z_grid, save_shd_path=shd_path)
    shade = read_at_shd_full(shd_path)
    pressure_abs = np.abs(shade.pressure[0])
    tl_from_shd = -20.0 * np.log10(np.maximum(pressure_abs, 1e-20))
    tl_from_shd[pressure_abs < 1e-15] = 120.0
    tl_from_shd = tl_from_shd.transpose(0, 2, 1)
    pressure_grid, _, _ = sim._coherent_pressure_cache

    assert tl.shape == (2, 3, 3)
    assert shade.pressure.shape == (1, 2, 3, 3)
    np.testing.assert_allclose(shade.source_depths_m, config.geometry.source.depths)
    np.testing.assert_allclose(shade.receiver_depths_m, z_grid)
    np.testing.assert_allclose(shade.receiver_ranges_m, r_grid)
    np.testing.assert_allclose(tl_from_shd, tl, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(shade.pressure[0].transpose(0, 2, 1), pressure_grid)
    assert np.max(np.abs(np.imag(shade.pressure))) > 0.0
    assert not np.allclose(tl[0], tl[1])


def test_intensity_tl_keeps_multi_source_depth_axis():
    config = _iso_config()
    config.geometry.source.depths = [40.0, 60.0]
    sim = Simulation(config)
    r_grid = np.array([100.0, 500.0, 1000.0])
    z_grid = np.array([25.0, 50.0, 75.0])

    incoherent = sim.compute_incoherent_tl(r_grid=r_grid, z_grid=z_grid)
    semicoherent = sim.compute_semicoherent_tl(r_grid=r_grid, z_grid=z_grid)

    assert incoherent.shape == (2, 3, 3)
    assert semicoherent.shape == (2, 3, 3)
    assert not np.allclose(incoherent[0], incoherent[1])
    assert not np.allclose(semicoherent[0], semicoherent[1])
