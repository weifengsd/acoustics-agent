import numpy as np
import pytest
from pyacoustics.solvers.kraken.core import build_acoustic_matrix, build_acoustic_matrix_complex, top_impedance, bottom_impedance
from pyacoustics.solvers.kraken.roots import bisection_search_roots
from pyacoustics.solvers.kraken.modes import compute_mode_shapes_complex, compute_modes

def test_compute_modes_ideal_waveguide():
    D = 100.0
    c = 1500.0
    freq = 15.0
    omega = 2 * np.pi * freq
    
    N = 1000
    h = D / N
    c_arr = np.full(N + 1, c)
    density_arr = np.ones(N + 1)
    
    B1 = build_acoustic_matrix(h, omega, c_arr)
    f_top, g_top = top_impedance(0) # vacuum (phi=0)
    f_bot, g_bot = bottom_impedance(0.0, omega, 1, 0.0, 1.0) # rigid (phi'=0)
    
    k2_min = 0.0
    k2_max = (omega / c)**2
    
    roots = bisection_search_roots(k2_min, k2_max, h, B1, omega, 1, 0.0, 1.0, 0)
    assert len(roots) == 2

    modes, complex_roots = compute_modes(roots, h, B1, omega, 1, 0.0, 1.0, 0, density_arr)

    assert modes.shape == (2, N + 1)
    # Mode 1 Analytical: phi_1(z) = A * sin(k_z1 * z)
    # where k_z1 = 0.5 * pi / D
    # Normalization: int_0^D A^2 sin^2(k_z1 z) dz = 1
    # A^2 * D / 2 = 1 => A = sqrt(2/D) = sqrt(2/100) = sqrt(0.02)
    A = np.sqrt(2.0 / D)
    k_z1 = 0.5 * np.pi / D
    z_arr = np.linspace(0, D, N + 1)
    analytical_mode_1 = A * np.sin(k_z1 * z_arr)
    
    # Modes might be inverted depending on sign standardization
    # Our code standardizes to make the maximum absolute value positive.
    # sin(z) is positive in [0, D], so it should match.
    
    # Compare inner points (avoiding exact boundary edge effects if any)
    np.testing.assert_allclose(modes[0, 10:-10], analytical_mode_1[10:-10], rtol=1e-2, atol=1e-3)
    
    # Boundary checks
    assert abs(modes[0, 0]) < 1e-10 # Top vacuum
    
    # Bottom rigid: phi' = 0 -> phi[-1] should be close to phi[-2]
    # For mode 1, sin reaches its maximum at D, so derivative is 0.
    assert abs(modes[0, -1] - modes[0, -2]) < 1e-4


def test_compute_modes_applies_water_column_attenuation():
    D = 100.0
    c = 1500.0
    freq = 15.0
    omega = 2 * np.pi * freq

    N = 300
    h = D / N
    c_arr = np.full(N + 1, c)
    density_arr = np.ones(N + 1)
    attenuation_arr = np.full(N + 1, 0.5)

    B1 = build_acoustic_matrix(h, omega, c_arr)
    roots = bisection_search_roots(0.0, (omega / c) ** 2, h, B1, omega, 1, 0.0, 1.0, 0)

    _, complex_roots_lossless = compute_modes(roots, h, B1, omega, 1, 0.0, 1.0, 0, density_arr)
    _, complex_roots_lossy = compute_modes(
        roots,
        h,
        B1,
        omega,
        1,
        0.0,
        1.0,
        0,
        density_arr,
        attenuation_arr=attenuation_arr,
    )

    assert np.allclose(np.imag(complex_roots_lossless), 0.0)
    assert np.all(np.imag(complex_roots_lossy) > 0.0)


def test_compute_modes_uses_media_density_for_normalization():
    D = 100.0
    c = 1500.0
    freq = 50.0
    omega = 2 * np.pi * freq

    N = 400
    h = D / N
    c_arr = np.full(N + 1, c)
    B1 = build_acoustic_matrix(h, omega, c_arr)
    roots = bisection_search_roots(0.0, (omega / c) ** 2, h, B1, omega, 1, 0.0, 1.0, 0)

    media_start = np.array([0, N // 2], dtype=np.int64)
    media_end = np.array([N // 2, N], dtype=np.int64)
    media_rho = np.array([1.0, 1.5], dtype=np.float64)
    modes, _ = compute_modes(
        roots,
        h,
        B1,
        omega,
        1,
        0.0,
        1.0,
        0,
        np.ones(N + 1),
        media_start=media_start,
        media_end=media_end,
        media_rho=media_rho,
    )

    density = np.ones(N + 1)
    density[N // 2 :] = 1.5
    integrand = modes[0] ** 2 / density
    integral = h * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])

    np.testing.assert_allclose(integral, 1.0, rtol=1e-5)


def test_complex_mode_shapes_match_lossless_real_modes():
    D = 100.0
    c = 1500.0
    freq = 15.0
    omega = 2 * np.pi * freq

    N = 400
    h = D / N
    c_arr = np.full(N + 1, c)
    B1 = build_acoustic_matrix(h, omega, c_arr)
    roots = bisection_search_roots(0.0, (omega / c) ** 2, h, B1, omega, 1, 0.0, 1.0, 0)

    real_modes, _ = compute_modes(roots, h, B1, omega, 1, 0.0, 1.0, 0, np.ones(N + 1))
    complex_modes = compute_mode_shapes_complex(
        roots.astype(np.complex128),
        h,
        build_acoustic_matrix_complex(h, omega, c_arr),
        omega,
        1,
        0.0,
        1.0,
        0,
        np.array([0], dtype=np.int64),
        np.array([N], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
    )

    np.testing.assert_allclose(np.real(complex_modes), real_modes, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.imag(complex_modes), 0.0, atol=1e-10)


def test_complex_mode_shapes_include_lossy_phase():
    D = 100.0
    c = 1500.0
    freq = 50.0
    omega = 2 * np.pi * freq

    N = 300
    h = D / N
    c_arr = np.full(N + 1, c)
    attenuation_arr = np.full(N + 1, 0.5)
    B1 = build_acoustic_matrix(h, omega, c_arr)
    roots = bisection_search_roots(0.0, (omega / c) ** 2, h, B1, omega, 1, 0.0, 1.0, 0)
    complex_roots = roots.astype(np.complex128) + 1j * roots * 1e-3

    complex_modes = compute_mode_shapes_complex(
        complex_roots,
        h,
        build_acoustic_matrix_complex(h, omega, c_arr, attenuation_arr),
        omega,
        1,
        0.0,
        1.0,
        0,
        np.array([0], dtype=np.int64),
        np.array([N], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
    )

    assert np.iscomplexobj(complex_modes)
    assert np.max(np.abs(np.imag(complex_modes))) > 0.0
