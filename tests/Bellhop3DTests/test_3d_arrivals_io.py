import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pyacoustics.postprocessing.delay_and_sum import read_arrivals_ascii
from pyacoustics.schema import (
    Bellhop3DConfig,
    BoundaryCondition,
    EnvironmentModel,
    GeometryModel,
    ReceiverModel,
    SimulationConfig,
    SourceModel,
    SSPLayer,
    SSPModel,
)
from pyacoustics.solvers.bellhop3d.arrivals import compute_arrival_grid_3d


def _bellhop3d_executable() -> Path | None:
    candidates: list[Path] = []
    env_bin = os.environ.get("AT_BIN_PATH")
    if env_bin:
        candidates.extend([Path(env_bin) / "bellhop3d", Path(env_bin) / "bellhop3d.exe"])

    default_bin = Path(__file__).resolve().parents[3] / "at" / "bin"
    candidates.extend([default_bin / "bellhop3d", default_bin / "bellhop3d.exe"])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    found = shutil.which("bellhop3d") or shutil.which("bellhop3d.exe")
    return Path(found) if found else None


def _small_arrivals_env() -> str:
    scratch_dir = Path(__file__).resolve().parent / "scratch"
    env_text = (scratch_dir / "temp_plot.env").read_text()
    replacements = {
        "'CB   3'": "'A    3'",
        "100                           ! NRD (receiver depths)": "5                             ! NRD (receiver depths)",
        "100                           ! NR (receiver ranges)": "5                             ! NR (receiver ranges)",
        "20                            ! Nalpha": "5                             ! Nalpha",
        "24                            ! Nbeta": "5                             ! Nbeta",
    }
    for old, new in replacements.items():
        assert old in env_text
        env_text = env_text.replace(old, new)
    return env_text


def _direct_arrivals_env() -> str:
    scratch_dir = Path(__file__).resolve().parent / "scratch"
    env_text = (scratch_dir / "temp_plot.env").read_text()
    replacements = {
        "'Munk 3D Slanted Terrain Coherent TL'": "'Direct 3D Arrivals'",
        "100                           ! NRD (receiver depths)": "1                             ! NRD (receiver depths)",
        "0.0 5000.0 /                  ! RD (m)": "1000.0 /                      ! RD (m)",
        "100                           ! NR (receiver ranges)": "1                             ! NR (receiver ranges)",
        "0.0 10.0 /                    ! Range (km)": "1.0 /                         ! Range (km)",
        "'CB   3'": "'A    3'",
        "20                            ! Nalpha": "1                             ! Nalpha",
        "-20.0 20.0 /                  ! alpha1, alpha2 (degrees)": "0.0 0.0 /                     ! alpha1, alpha2 (degrees)",
        "24                            ! Nbeta": "1                             ! Nbeta",
        "0.0 90.0 /                    ! beta1, beta2 (degrees)": "0.0 0.0 /                     ! beta1, beta2 (degrees)",
    }
    for old, new in replacements.items():
        assert old in env_text
        env_text = env_text.replace(old, new)
    return env_text


def _native_direct_arrivals_config() -> SimulationConfig:
    return SimulationConfig(
        project="Direct 3D Arrivals",
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
            bottom=BoundaryCondition(type="vacuum", depth=5000.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[1000.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[1000.0]),
        ),
        solver=Bellhop3DConfig(
            elevation_angles=(0.0, 0.0),
            azimuth_angles=(0.0, 0.0),
            num_elevation_beams=1,
            num_azimuth_beams=1,
            step_size=10.0,
            box_x=1500.0,
            box_y=100.0,
            box_z=6000.0,
        ),
    )


def test_read_real_bellhop3d_arrivals_ascii(tmp_path):
    exe = _bellhop3d_executable()
    if exe is None:
        pytest.skip("AT bellhop3d executable is not available")

    scratch_dir = Path(__file__).resolve().parent / "scratch"
    (tmp_path / "small_arr.env").write_text(_small_arrivals_env())
    shutil.copyfile(scratch_dir / "temp_plot.bty", tmp_path / "small_arr.bty")

    result = subprocess.run(
        [str(exe), "small_arr"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    arr_path = tmp_path / "small_arr.arr"
    assert arr_path.exists()

    parsed = read_arrivals_ascii(arr_path)
    counts = np.array([record.count for record in parsed.arrivals.flat])

    assert parsed.is_3d
    assert np.isclose(parsed.frequency, 50.0)
    assert np.allclose(parsed.source_x, [0.0])
    assert np.allclose(parsed.source_y, [0.0])
    assert np.allclose(parsed.source_depths, [1000.0])
    assert np.allclose(parsed.receiver_depths, [0.0, 1250.0, 2500.0, 3750.0, 5000.0])
    assert np.allclose(parsed.receiver_ranges, [0.0, 2500.0, 5000.0, 7500.0, 10000.0])
    assert np.allclose(parsed.receiver_bearings, [0.0])
    assert parsed.arrivals.shape == (5, 5, 1, 1)
    assert counts.sum() > 0
    assert counts.max() > 0

    nonempty = next(record for record in parsed.arrivals.flat if record.count)
    assert nonempty.src_azim_angles is not None
    assert nonempty.rcvr_azim_angles is not None
    assert nonempty.delays.shape == nonempty.amplitudes.shape
    assert nonempty.src_azim_angles.shape == nonempty.amplitudes.shape
    assert nonempty.rcvr_azim_angles.shape == nonempty.amplitudes.shape
    assert np.isfinite(nonempty.delays.real).all()
    assert np.isfinite(nonempty.delays.imag).all()
    assert np.isfinite(nonempty.amplitudes).any()


def test_native_bellhop3d_direct_arrival_matches_at_delay_and_angles(tmp_path):
    exe = _bellhop3d_executable()
    if exe is None:
        pytest.skip("AT bellhop3d executable is not available")

    scratch_dir = Path(__file__).resolve().parent / "scratch"
    (tmp_path / "direct.env").write_text(_direct_arrivals_env())
    shutil.copyfile(scratch_dir / "temp_plot.bty", tmp_path / "direct.bty")

    result = subprocess.run(
        [str(exe), "direct"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    at_data = read_arrivals_ascii(tmp_path / "direct.arr")
    native_data = compute_arrival_grid_3d(
        _native_direct_arrivals_config(),
        receiver_bearings=np.array([0.0]),
        tolerance_m=0.1,
    )
    at_record = at_data.arrivals[0, 0, 0, 0]
    native_record = native_data.arrivals[0, 0, 0, 0]

    assert at_record.count == 1
    assert native_record.count == 1
    assert np.isclose(native_record.delays[0].real, at_record.delays[0].real, atol=1.0e-5)
    assert np.isclose(native_record.delays[0].imag, at_record.delays[0].imag, atol=1.0e-12)
    assert np.isclose(native_record.src_decl_angles[0], at_record.src_decl_angles[0], atol=1.0e-6)
    assert np.isclose(native_record.src_azim_angles[0], at_record.src_azim_angles[0], atol=1.0e-6)
    assert np.isclose(native_record.rcvr_decl_angles[0], at_record.rcvr_decl_angles[0], atol=1.0e-2)
    assert np.isclose(native_record.rcvr_azim_angles[0], at_record.rcvr_azim_angles[0], atol=1.0e-6)
    assert native_record.num_top_bounces[0] == at_record.num_top_bounces[0]
    assert native_record.num_bottom_bounces[0] == at_record.num_bottom_bounces[0]
