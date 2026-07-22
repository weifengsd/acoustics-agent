import pytest
import numpy as np
from pyacoustics.solvers.bellhop.step import initialize_ray, step2d_kinematic, step2d_kinematic_dynamic

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
    empty_range = np.zeros(0, dtype=np.float64)
    r_new, z_new, xi_new, zeta_new, tau_new = step2d_kinematic(
        h, r, z, xi, zeta, tau, ssp_type, z_arr, c_arr, dummy_coeffs, empty_range
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


def test_quad_2d_step_uses_range_gradient():
    z_arr = np.array([0.0, 100.0], dtype=np.float64)
    r_arr = np.array([0.0, 100.0], dtype=np.float64)
    c_matrix = np.array(
        [
            [1500.0, 1510.0],
            [1500.0, 1510.0],
        ],
        dtype=np.float64,
    )
    c_0 = 1500.0
    r, z, xi, zeta, tau = initialize_ray(0.0, 50.0, 0.0, c_0)

    r_new, z_new, xi_new, zeta_new, tau_new = step2d_kinematic(
        10.0,
        r,
        z,
        xi,
        zeta,
        tau,
        3,
        z_arr,
        np.array([1500.0, 1500.0], dtype=np.float64),
        c_matrix,
        r_arr,
    )

    assert r_new > 0.0
    assert np.isclose(z_new, z, atol=1.0e-8)
    assert xi_new < xi
    assert np.isclose(zeta_new, zeta)
    assert tau_new > 0.0


def test_quad_2d_dynamic_step_uses_mixed_curvature():
    z_arr = np.array([0.0, 100.0], dtype=np.float64)
    r_arr = np.array([0.0, 100.0], dtype=np.float64)
    c_arr = np.array([1500.0, 1500.0], dtype=np.float64)
    no_mixed = np.array(
        [
            [1500.0, 1510.0],
            [1500.0, 1510.0],
        ],
        dtype=np.float64,
    )
    mixed = np.array(
        [
            [1500.0, 1500.0],
            [1500.0, 1600.0],
        ],
        dtype=np.float64,
    )
    r, z, xi, zeta, tau = initialize_ray(10.0, 10.0, 45.0, 1500.0)

    no_mixed_step = step2d_kinematic_dynamic(
        2.0,
        r,
        z,
        xi,
        zeta,
        tau,
        1.0,
        1.0,
        3,
        z_arr,
        c_arr,
        no_mixed,
        r_arr,
    )
    mixed_step = step2d_kinematic_dynamic(
        2.0,
        r,
        z,
        xi,
        zeta,
        tau,
        1.0,
        1.0,
        3,
        z_arr,
        c_arr,
        mixed,
        r_arr,
    )

    assert not np.isclose(mixed_step[5], no_mixed_step[5])
