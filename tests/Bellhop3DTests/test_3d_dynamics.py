import sys
import os
import numpy as np
from pathlib import Path

# Insert package root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyacoustics.simulation import Simulation
from pyacoustics.solvers.bellhop3d.core import PyBellhop3D

def test_3d_gaussian_beam_dynamics_and_caustic_regularization():
    """
    Dynamics Test:
    Verify that the complex determinant det(Q) behaves smoothly and waist regularization
    successfully prevents singular infinite amplitudes at caustic focal points.
    """
    config_path = Path(__file__).resolve().parent.parent / "Munk" / "munk_3d_terrain.yaml"
    sim = Simulation(config_path)

    # Configure elevation/azimuth sweeps
    sim.config.solver.num_elevation_beams = 5
    sim.config.solver.num_azimuth_beams = 5

    # Initialize the high-level PyBellhop3D solver using official API
    solver = PyBellhop3D(sim.config)
    rays_3d = solver.run_ray_tracing()

    assert len(rays_3d) > 0, "Ray tracer must yield active rays"

    # Verify dynamic properties for a few propagating rays
    tested_rays = 0
    for ray in rays_3d:
        (
            x_path, y_path, z_path,
            q11_path, q12_path, q21_path, q22_path,
            amp_path, tau_path, kmah_path, c_path,
            n_top, n_bot, elev
        ) = ray

        # Reconstruct detQ_path for the test
        detQ_path = q11_path * q22_path - q12_path * q21_path

        if len(x_path) < 10:
            continue

        # 1. Point source initial boundary check
        assert abs(detQ_path[0]) < 1e-10, "Point source boundary condition: Q(0) = 0"

        # 2. Check for smooth, non-infinite, non-nan determinants
        for q in detQ_path:
            q_mag = abs(q)
            assert not np.isnan(q_mag), "Determinant must not contain NaN"
            assert not np.isinf(q_mag), "Determinant must not contain Inf"

        # 3. Regularization protection: amplitude must not explode to infinity at caustics
        for amp_val in amp_path:
            amp_mag = abs(amp_val)
            assert amp_mag < 1e12, f"Waist regularization failed: Infinite amplitude detected! ({amp_mag:.2e})"
            assert amp_mag >= 0.0, "Amplitude magnitude should be non-negative"

        tested_rays += 1
        if tested_rays >= 3:
            break

    assert tested_rays > 0, "At least one active ray must be verified for Gaussian Beam DRT dynamics"
