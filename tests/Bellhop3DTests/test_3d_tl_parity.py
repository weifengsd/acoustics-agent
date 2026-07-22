import sys
import os
import numpy as np
import subprocess
import pytest
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# Insert package root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyacoustics.simulation import Simulation
from pyacoustics.solvers.external_io import read_at_shd

def run_legacy_shd(scratch_dir):
    env_content = """'Munk 3D Slanted Terrain Coherent TL' ! TITLE
50.0                          ! FREQ (Hz)
1                             ! NMEDIA
'CVW'                         ! SSPOPT (C-linear, Vacuum surface, BTY bottom)
3  0.0  5000.0                ! SSP pts, top_depth, bottom_flat_depth
    0.0  1548.52  /
 1000.0  1500.00  /
 5000.0  1551.91  /
'V~' 0.0                      ! Bottom option (Vacuum with ~ bathymetry), roughness
1                             ! N_source_x
0.0 /                         ! Sx (km)
1                             ! N_source_y
0.0 /                         ! Sy (km)
1                             ! NSD
1000.0 /                      ! SD (m)
100                           ! NRD (receiver depths)
0.0 5000.0 /                  ! RD (m)
100                           ! NR (receiver ranges)
0.0 10.0 /                    ! Range (km)
1                             ! Ntheta (bearings)
0.0 /                         ! Bearing angles (degrees)
'CB   3'                       ! 'C' (Coherent TL) + 'B' (Gaussian Beams) + '3' (Full 3D)
20                            ! Nalpha (Elevation angles)
-20.0 20.0 /                  ! alpha1, alpha2 (degrees)
24                            ! Nbeta (bearing/azimuth angles)
0.0 90.0 /                    ! beta1, beta2 (degrees)
10.0 10.0 10.0 20500.0         ! STEP (m), Box%x (km), Box%y (km), Box%z (m)
"""
    env_path = scratch_dir / "temp_test_tl.env"
    with open(env_path, "w") as f:
        f.write(env_content)

    bty_content = """'RS'
2
-2.0 20.0
2
-2.0 20.0
 2000.0 5666.667
 3833.333 7500.0
"""
    bty_path = scratch_dir / "temp_test_tl.bty"
    with open(bty_path, "w") as f:
        f.write(bty_content)

    bin_path = os.environ.get("AT_BIN_PATH")
    if not bin_path:
        pytest.skip("AT_BIN_PATH is not set")
    exe_path = Path(bin_path) / "bellhop3d"
    if not exe_path.exists():
        exe_path = Path(bin_path) / "bellhop3d.exe"
    if not exe_path.is_file():
        pytest.skip("Bellhop3D executable is not available under AT_BIN_PATH")
    cmd = [exe_path, "temp_test_tl"]
    subprocess.run(cmd, cwd=str(scratch_dir), capture_output=True, text=True, check=True)

    shd_file = scratch_dir / "temp_test_tl.shd"
    P_at, rz, rr = read_at_shd(shd_file)
    return P_at, rz, rr

def test_3d_coherent_tl_field_parity_check():
    """
    Integration Parity Test:
    Run both Native and Legacy 3D Coherent TL solvers. Interpolate and compare the fields.
    Assert that the Mean Absolute Error (MAE) is less than 10.0 dB, and that no
    non-physical near-field singularity regressions occur (Min TL > 40.0 dB).
    """
    scratch_dir = Path(__file__).resolve().parent.parent / "Bellhop3DTests" / "scratch"
    os.makedirs(scratch_dir, exist_ok=True)

    # 1. Run Legacy
    P_at, rz, rr = run_legacy_shd(scratch_dir)
    assert P_at is not None, "Failed to load Legacy SHD field"

    p_mag_at = np.abs(P_at)
    legacy_tl = np.zeros_like(p_mag_at)
    legacy_tl[p_mag_at < 1e-20] = 200.0
    legacy_tl[p_mag_at >= 1e-20] = -20.0 * np.log10(p_mag_at[p_mag_at >= 1e-20])

    # 2. Run Native PyBellhop3D
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)
    sim.config.solver.num_elevation_beams = 20
    sim.config.solver.num_azimuth_beams = 24

    sim.compute_coherent_tl_3d(num_x=121, num_y=9, num_z=121)
    TL_native, x_grid, y_grid, z_grid = sim._coherent_tl_cache_3d

    # 3. Interpolation onto Legacy grid
    interp = RegularGridInterpolator((x_grid, y_grid, z_grid), TL_native, bounds_error=False, fill_value=200.0)
    R_mesh, Z_mesh = np.meshgrid(rr, rz)
    query_pts = np.stack([R_mesh.ravel(), np.zeros_like(R_mesh).ravel(), Z_mesh.ravel()], axis=-1)
    native_tl_interp = interp(query_pts).reshape(R_mesh.shape)

    # 4. Filter water column (excluding boundary cutoffs and極近场 source region R < 500m)
    mask = (legacy_tl < 110.0) & (native_tl_interp < 190.0) & (R_mesh > 500.0)
    valid_native_tl = native_tl_interp[mask]
    valid_legacy_tl = legacy_tl[mask]

    assert len(valid_native_tl) > 0, "Water column grid matching should contain active points"

    # 5. Core Alignment Assertions
    # Assertion A: Bounded Near-Field / Space Minimum TL (Prevent singularity regression!)
    min_tl_native = np.min(valid_native_tl)
    print(f"Space Minimum Transmission Loss in Native: {min_tl_native:.4f} dB")
    assert min_tl_native > 40.0, f"REGRESSION: Near-field singularity detected! Min TL is {min_tl_native:.2f} dB (< 40 dB)"

    # Assertion B: Mean Absolute Error (MAE) must be < 10.0 dB
    tl_diff = np.abs(valid_native_tl - valid_legacy_tl)
    mae = np.mean(tl_diff)
    print(f"Transmission Loss Parity MAE: {mae:.4f} dB")
    assert mae < 10.0, f"REGRESSION: Parity check failed! 3D Coherent TL MAE is {mae:.2f} dB (>= 10.0 dB)"

    # 6. Direct AT-style receiver-grid influence check. This avoids the
    # intermediate Cartesian voxel interpolation used above.
    TL_receiver = sim.compute_coherent_tl_3d_receiver_grid(
        theta_grid_deg=np.array([0.0], dtype=np.float64),
        r_grid=rr,
        z_grid=rz,
    )[0]
    receiver_mask = (legacy_tl < 110.0) & (TL_receiver < 190.0) & (R_mesh > 500.0)
    receiver_diff = TL_receiver[receiver_mask] - legacy_tl[receiver_mask]
    receiver_abs_diff = np.abs(receiver_diff)
    receiver_mae = np.mean(receiver_abs_diff)
    receiver_bias = np.mean(receiver_diff)
    receiver_p90 = np.percentile(receiver_abs_diff, 90.0)
    print(f"Receiver-grid TL Parity MAE: {receiver_mae:.4f} dB")
    print(f"Receiver-grid TL Parity Bias: {receiver_bias:.4f} dB")
    print(f"Receiver-grid TL Parity P90: {receiver_p90:.4f} dB")
    assert receiver_mae < 1.0, (
        f"REGRESSION: Receiver-grid parity check failed! 3D Coherent TL MAE is "
        f"{receiver_mae:.2f} dB (>= 1.0 dB)"
    )
    assert abs(receiver_bias) < 0.2, "Receiver-grid TL has excessive systematic bias"
    assert receiver_p90 < 1.0, "Receiver-grid TL has excessive high-percentile error"

    # Clean up temp test files in test scratch
    for file in scratch_dir.iterdir():
        if file.is_file() and file.name.startswith("temp_test_tl"):
            try:
                file.unlink()
            except Exception:
                pass
    try:
        scratch_dir.rmdir()
    except Exception:
        pass
