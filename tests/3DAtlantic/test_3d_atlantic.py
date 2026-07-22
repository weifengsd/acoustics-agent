import sys
import os
import subprocess
import pytest
from pathlib import Path

# Add package root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyacoustics.solvers.external import BellhopExternal

def resolve_at_bin():
    """
    Helper to resolve the directory containing the legacy binaries.
    Uses BellhopExternal's built-in path resolution logic.
    """
    dummy_solver = BellhopExternal(None)
    assert dummy_solver.bin_path is not None, "Acoustics Toolbox bin path cannot be resolved."
    return dummy_solver.bin_path

@pytest.fixture(scope="module", autouse=True)
def cleanup_temp_files():
    """Fixture to clean up temporary build/simulation files before and after tests."""
    target_dir = Path(__file__).resolve().parent
    extensions = ["*.mod", "*.prt", "*.shd", "*.grn", "*.asc", "SHDFIL"]

    def do_cleanup():
        for ext in extensions:
            for f in target_dir.glob(ext):
                try:
                    f.unlink()
                except OSError:
                    pass

    # Pre-test cleanup
    do_cleanup()
    yield
    # Post-test cleanup
    do_cleanup()

def test_3d_atlantic_kraken_and_field3d_integration():
    """
    3D Atlantic Ocean Waveguide Integration Test:
    1. Runs native Kraken (arm64) on 49 distinct environmental grid points.
    2. Runs native Field3D to synthesize the full 3D interpolated normal mode field.
    3. Verifies that output sound files (lant.shd, lant_90degrees.shd) are correctly compiled.
    """
    bin_path = resolve_at_bin()
    target_dir = Path(__file__).resolve().parent

    # 1. Resolve executable paths
    kraken_exe = bin_path / "kraken"
    if not kraken_exe.exists():
        kraken_exe = bin_path / "kraken.exe"

    field3d_exe = bin_path / "field3d"
    if not field3d_exe.exists():
        field3d_exe = bin_path / "field3d.exe"

    assert kraken_exe.exists(), "Kraken native executable not found."
    assert field3d_exe.exists(), "Field3D native executable not found."

    # 2. Define N. Atlantic spatial grid environment files matching krakenall csh script
    depths_a = [36, 38, 40, 42, 44, 46, 48, 50, 52]
    depths_b = [36, 38, 40, 42, 44, 46, 48]
    depths_e = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52]

    env_files = []
    for d in depths_a:
        env_files.append(f"lanta{d}")
    for d in depths_b:
        env_files.append(f"lantb{d}")
        env_files.append(f"lantc{d}")
        env_files.append(f"lantd{d}")
    for d in depths_e:
        env_files.append(f"lante{d:02d}")

    # 3. Parallel/Sequential Run of Kraken solver on all environment grids
    print(f"\n[AT 3DAtlantic] Executing Kraken solver on {len(env_files)} environments...")
    for idx, env in enumerate(env_files):
        env_path = target_dir / f"{env}.env"
        assert env_path.exists(), f"Environmental configuration file {env}.env is missing."

        # Run Kraken binary
        res = subprocess.run(
            [str(kraken_exe), env],
            cwd=str(target_dir),
            capture_output=True,
            text=True
        )
        assert res.returncode == 0, f"Kraken failed on {env}: {res.stderr}"

        # Double check that .mod file was produced
        mod_file = target_dir / f"{env}.mod"
        assert mod_file.exists(), f"Kraken run failed to generate normal mode (.mod) file for {env}"

    print("[AT 3DAtlantic] All 49 ocean grid normal modes compiled successfully.")

    # 4. Run Field3D to compute the 3D interpolated omni-directional sound field
    print("[AT 3DAtlantic] Synthesizing full 3D ocean sound field (lant)...")
    res = subprocess.run(
        [str(field3d_exe), "lant"],
        cwd=str(target_dir),
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Field3D 'lant' execution failed: {res.stderr}"

    shd_file = target_dir / "lant.shd"
    assert shd_file.exists(), "Field3D failed to generate full 3D sound field file (lant.shd)"
    assert shd_file.stat().st_size > 1000, "Full 3D sound field file lant.shd is empty or truncated."

    # 5. Run Field3D to compute single bearing (90 degrees radial) sound field
    print("[AT 3DAtlantic] Synthesizing single bearing (90 degrees) sound field...")
    res = subprocess.run(
        [str(field3d_exe), "lant_90degrees"],
        cwd=str(target_dir),
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Field3D 'lant_90degrees' execution failed: {res.stderr}"

    shd_file_90 = target_dir / "lant_90degrees.shd"
    assert shd_file_90.exists(), "Field3D failed to generate single bearing sound field file (lant_90degrees.shd)"
    assert shd_file_90.stat().st_size > 1000, "Single bearing sound field file lant_90degrees.shd is empty."

    print("🏆 [AT 3DAtlantic] 3D normal modes ocean grid test and 3D synthesis completed successfully!")
