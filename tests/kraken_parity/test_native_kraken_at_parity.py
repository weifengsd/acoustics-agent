import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from pyacoustics.schema import (
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
from pyacoustics.solvers.external_io import generate_at_field_flp, read_at_prt, read_at_shd
from pyacoustics.solvers.kraken import PyKraken
from pyacoustics.solvers.kraken.tl_parity import (
    compare_mode_summaries,
    compare_tl_grids,
    compute_native_tl_grid,
    run_at_tl_grid,
    summarize_native_modes,
    summarize_prt_modes,
)


def _at_bin_or_skip():
    bin_path = os.environ.get("AT_BIN_PATH")
    if not bin_path:
        pytest.skip("AT_BIN_PATH is not set")
    bin_path = Path(bin_path)
    kraken = bin_path / "kraken"
    if not kraken.exists():
        kraken = bin_path / "kraken.exe"
    field = bin_path / "field"
    if not field.exists():
        field = bin_path / "field.exe"
    if not kraken.exists() or not field.exists():
        pytest.skip("AT kraken/field executables are not available")
    return bin_path


def _at_field_or_skip():
    bin_path = os.environ.get("AT_BIN_PATH")
    if not bin_path:
        pytest.skip("AT_BIN_PATH is not set")
    bin_path = Path(bin_path)
    field = bin_path / "field"
    if not field.exists():
        field = bin_path / "field.exe"
    if not field.exists():
        pytest.skip("AT field executable is not available")
    return field


def _calib_like_config():
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


def _two_layer_acoustic_config():
    return SimulationConfig(
        project="native kraken two layer field parity",
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


def test_native_kraken_matches_at_calib_like_tl_smoke():
    at_bin = _at_bin_or_skip()
    config = _calib_like_config()
    r_grid = np.array([2500.0])
    z_grid = np.array([50.0])

    native_tl = compute_native_tl_grid(config, r_grid, z_grid)
    at_result = run_at_tl_grid(config, r_grid, z_grid, at_bin_path=at_bin)
    metrics = compare_tl_grids(at_result.tl_grid, native_tl)

    assert np.isfinite(native_tl).all()
    assert np.isfinite(at_result.tl_grid).all()
    assert metrics.sample_count == 1
    assert metrics.mae_db < 1.0
    assert metrics.max_abs_db < 1.0
    assert abs(metrics.bias_db) < 1.0


@pytest.mark.parametrize(
    ("config_factory", "case_name"),
    [
        (_calib_like_config, "native_mod_field"),
        (_two_layer_acoustic_config, "native_two_layer_mod_field"),
    ],
)
def test_at_field_reads_native_kraken_mod_smoke(tmp_path, config_factory, case_name):
    field_exe = _at_field_or_skip()
    config = config_factory()

    result = PyKraken(config).run_normal_modes()
    PyKraken.write_modes(result, tmp_path / f"{case_name}.mod", title="KRAKENC native Kraken")
    generate_at_field_flp(config, tmp_path / f"{case_name}.flp")

    completed = subprocess.run(
        [str(field_exe), case_name],
        cwd=tmp_path,
        input=f"{case_name}\n",
        text=True,
        capture_output=True,
        check=False,
    )

    prt_path = tmp_path / f"{case_name}.prt"
    prt_tail = prt_path.read_text(errors="ignore")[-2000:] if prt_path.exists() else ""
    assert completed.returncode == 0, completed.stdout + completed.stderr + prt_tail
    assert prt_path.exists(), completed.stdout + completed.stderr

    shd_path = tmp_path / f"{case_name}.shd"
    assert shd_path.exists(), completed.stdout + completed.stderr + prt_tail

    log = read_at_prt(prt_path)
    mode_metrics = compare_mode_summaries(summarize_native_modes(result), summarize_prt_modes(log))
    assert log.completed_without_errors
    assert log.error_count == 0
    assert mode_metrics.native_mode_count == mode_metrics.at_mode_count
    assert mode_metrics.mode_count_delta == 0

    pressure, depths, ranges = read_at_shd(shd_path)
    assert pressure.size > 0
    assert depths.size > 0
    assert ranges.size > 0
    assert np.isfinite(np.abs(pressure)).any()
