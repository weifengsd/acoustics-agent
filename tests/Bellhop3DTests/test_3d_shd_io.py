import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pyacoustics.solvers.external_io import read_at_shd_full


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


def _small_multi_bearing_tl_env() -> str:
    scratch_dir = Path(__file__).resolve().parent / "scratch"
    env_text = (scratch_dir / "temp_plot.env").read_text()
    replacements = {
        "1                             ! N_source_x": "2                             ! N_source_x",
        "0.0 /                         ! Sx (km)": "0.0 0.25 /                    ! Sx (km)",
        "1                             ! N_source_y": "2                             ! N_source_y",
        "0.0 /                         ! Sy (km)": "0.0 0.1 /                     ! Sy (km)",
        "100                           ! NRD (receiver depths)": "5                             ! NRD (receiver depths)",
        "100                           ! NR (receiver ranges)": "5                             ! NR (receiver ranges)",
        "1                             ! Ntheta (bearings)": "3                             ! Ntheta (bearings)",
        "0.0 /                         ! Bearing angles (degrees)": "0.0 30.0 60.0 /               ! Bearing angles (degrees)",
        "20                            ! Nalpha": "5                             ! Nalpha",
        "24                            ! Nbeta": "5                             ! Nbeta",
    }
    for old, new in replacements.items():
        assert old in env_text
        env_text = env_text.replace(old, new)
    return env_text


def test_read_real_bellhop3d_multi_source_xy_multi_bearing_shd(tmp_path):
    exe = _bellhop3d_executable()
    if exe is None:
        pytest.skip("AT bellhop3d executable is not available")

    scratch_dir = Path(__file__).resolve().parent / "scratch"
    (tmp_path / "multi_bearing.env").write_text(_small_multi_bearing_tl_env())
    shutil.copyfile(scratch_dir / "temp_plot.bty", tmp_path / "multi_bearing.bty")

    result = subprocess.run(
        [str(exe), "multi_bearing"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    shade = read_at_shd_full(tmp_path / "multi_bearing.shd")
    finite_pressure = np.abs(shade.pressure[np.isfinite(shade.pressure)])

    assert shade.title == "BELLHOP3D- Munk 3D Slanted Terrain Coherent TL"
    assert np.isclose(shade.frequency_hz, 50.0)
    assert shade.pressure.shape == (3, 1, 5, 5)
    assert shade.pressure_full.shape == (2, 2, 3, 1, 5, 5)
    assert shade.pressure.dtype == np.complex64
    assert shade.pressure_full.dtype == np.complex64
    np.testing.assert_allclose(shade.source_x_m, [0.0, 250.0])
    np.testing.assert_allclose(shade.source_y_m, [0.0, 100.0])
    np.testing.assert_allclose(shade.bearings_deg, [0.0, 30.0, 60.0])
    np.testing.assert_allclose(shade.source_depths_m, [1000.0])
    np.testing.assert_allclose(shade.receiver_depths_m, [0.0, 1250.0, 2500.0, 3750.0, 5000.0])
    np.testing.assert_allclose(shade.receiver_ranges_m, [0.0, 2500.0, 5000.0, 7500.0, 10000.0])
    assert finite_pressure.size > 0
    assert np.max(finite_pressure) < 1.0e10
