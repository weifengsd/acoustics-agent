import pytest
import numpy as np
from pyacoustics.solvers.bellhop.step import initialize_ray, step2d_kinematic

def test_isovelocity_ray_tracing():
    # Setup isovelocity environment (c = 1500)
    z_arr = np.array([0.0, 1000.0])
    c_arr = np.array([1500.0, 1500.0])
    ssp_type = 0 # c-linear
    
    c_0 = 1500.0
    r_0, z_0 = 0.0, 500.0
    alpha_deg = 30.0 # Launch down at 30 degrees
    
    # Initialize
    r, z, xi, zeta, tau = initialize_ray(r_0, z_0, alpha_deg, c_0)
    
    # Check initial slowness
    assert pytest.approx(xi) == np.cos(np.radians(30)) / c_0
    assert pytest.approx(zeta) == np.sin(np.radians(30)) / c_0
    
    # Take a 10m step
    h = 10.0
    dummy_coeffs = np.zeros((4, 1)) # dummy 2D array for numba
    r_new, z_new, xi_new, zeta_new, tau_new = step2d_kinematic(
        h, r, z, xi, zeta, tau, ssp_type, z_arr, c_arr, dummy_coeffs
    )
    
    # For isovelocity, slowness shouldn't change
    assert pytest.approx(xi_new) == xi
    assert pytest.approx(zeta_new) == zeta
    
    # Distance traveled = c * slowness * h
    # r_new - r_0 should be h * cos(30)
    assert pytest.approx(r_new) == h * np.cos(np.radians(30))
    # z_new - z_0 should be h * sin(30)
    assert pytest.approx(z_new) == 500.0 + h * np.sin(np.radians(30))
    # travel time should be h / c
    assert pytest.approx(tau_new) == h / c_0
