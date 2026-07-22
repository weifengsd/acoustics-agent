import numpy as np
from pyacoustics.solvers.kraken.core import (
    build_acoustic_matrix,
    build_acoustic_matrix_complex,
    evaluate_dispersion_media_complex,
    reflection_table_impedance,
    top_impedance,
    bottom_impedance,
)
from pyacoustics.solvers.kraken.roots import (
    bisection_search_roots,
    bisection_search_roots_media,
    neville_extrapolation,
    secant_search_roots_media_complex,
)

def test_bisection_roots():
    D = 100.0
    c = 1500.0
    freq = 15.0
    omega = 2 * np.pi * freq
    
    N = 1000
    h = D / N
    c_arr = np.full(N + 1, c)
    
    B1 = build_acoustic_matrix(h, omega, c_arr)
    f_top, g_top = top_impedance(0) # vacuum
    f_bot, g_bot = bottom_impedance(0.0, omega, 1, 0.0, 1.0) # rigid
    
    k2_min = 0.0
    k2_max = (omega / c)**2
    
    roots = bisection_search_roots(k2_min, k2_max, h, B1, omega, 1, 0.0, 1.0, 0)
    
    # Analytical: k_z = (m - 0.5) * pi / D, k^2 = (omega/c)^2 - k_z^2
    kz_1 = (1 - 0.5) * np.pi / D
    k2_1 = (omega / c)**2 - kz_1**2
    
    kz_2 = (2 - 0.5) * np.pi / D
    k2_2 = (omega / c)**2 - kz_2**2
    
    assert len(roots) == 2
    # Roots should be ordered ascending in bisection by default, actually we need to check how bisection built it.
    # In bisection we did: m=1 to M, target_count=max+m.
    # Mode 1 has count=max+1. It corresponds to the highest k^2.
    # Wait, my bisection algorithm searches for the range [k2_min, k2_max].
    # count_mid >= target: low = mid. This means we move to higher k2 if we have too many modes.
    # So roots[0] is the highest k2 (mode 1).
    np.testing.assert_allclose(roots[0], k2_1, rtol=2e-3)
    np.testing.assert_allclose(roots[1], k2_2, rtol=2e-3)


def test_medium_density_affects_halfspace_roots():
    D = 100.0
    c = 1500.0
    freq = 50.0
    omega = 2 * np.pi * freq

    N = 600
    h = D / N
    c_arr = np.full(N + 1, c)
    B1 = build_acoustic_matrix(h, omega, c_arr)

    k2_min = (omega / 1700.0) ** 2
    k2_max = (omega / 1450.0) ** 2

    roots_rho1 = bisection_search_roots(
        k2_min, k2_max, h, B1, omega, 2, 1600.0, 1.8, 0, medium_rho=1.0
    )
    roots_rho12 = bisection_search_roots(
        k2_min, k2_max, h, B1, omega, 2, 1600.0, 1.8, 0, medium_rho=1.2
    )

    assert len(roots_rho1) > 0
    assert len(roots_rho1) == len(roots_rho12)
    assert not np.allclose(roots_rho1, roots_rho12)


def test_single_acoustic_medium_roots_match_legacy_path():
    D = 100.0
    c = 1500.0
    freq = 15.0
    omega = 2 * np.pi * freq

    N = 500
    h = D / N
    c_arr = np.full(N + 1, c)
    B1 = build_acoustic_matrix(h, omega, c_arr)
    k2_min = 0.0
    k2_max = (omega / c) ** 2

    roots_single = bisection_search_roots(k2_min, k2_max, h, B1, omega, 1, 0.0, 1.0, 0)
    roots_media = bisection_search_roots_media(
        k2_min,
        k2_max,
        h,
        B1,
        omega,
        1,
        0.0,
        1.0,
        0,
        np.array([0], dtype=np.int64),
        np.array([N], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
    )

    np.testing.assert_allclose(roots_media, roots_single)


def test_density_interface_changes_acoustic_medium_roots():
    D = 100.0
    c = 1500.0
    freq = 50.0
    omega = 2 * np.pi * freq

    N = 600
    h = D / N
    c_arr = np.full(N + 1, c)
    B1 = build_acoustic_matrix(h, omega, c_arr)
    k2_min = 0.0
    k2_max = (omega / c) ** 2

    roots_homogeneous = bisection_search_roots_media(
        k2_min,
        k2_max,
        h,
        B1,
        omega,
        1,
        0.0,
        1.0,
        0,
        np.array([0], dtype=np.int64),
        np.array([N], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
    )
    roots_interface = bisection_search_roots_media(
        k2_min,
        k2_max,
        h,
        B1,
        omega,
        1,
        0.0,
        1.0,
        0,
        np.array([0, N // 2], dtype=np.int64),
        np.array([N // 2, N], dtype=np.int64),
        np.array([1.0, 1.5], dtype=np.float64),
    )

    assert len(roots_homogeneous) == len(roots_interface)
    assert not np.allclose(roots_homogeneous, roots_interface)


def test_complex_secant_preserves_lossless_rigid_roots():
    D = 100.0
    c = 1500.0
    freq = 15.0
    omega = 2 * np.pi * freq

    N = 500
    h = D / N
    c_arr = np.full(N + 1, c)
    B1 = build_acoustic_matrix(h, omega, c_arr)
    roots = bisection_search_roots(0.0, (omega / c) ** 2, h, B1, omega, 1, 0.0, 1.0, 0)
    B1_complex = build_acoustic_matrix_complex(h, omega, c_arr)

    complex_roots = secant_search_roots_media_complex(
        roots,
        h,
        B1_complex,
        omega,
        1,
        0.0,
        1.0,
        0,
        np.array([0], dtype=np.int64),
        np.array([N], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
    )

    np.testing.assert_allclose(np.real(complex_roots), roots, rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(np.imag(complex_roots), 0.0, atol=1e-10)


def test_complex_secant_resolves_water_column_attenuation():
    D = 100.0
    c = 1500.0
    freq = 50.0
    omega = 2 * np.pi * freq

    N = 600
    h = D / N
    c_arr = np.full(N + 1, c)
    attenuation_arr = np.full(N + 1, 0.5)
    B1 = build_acoustic_matrix(h, omega, c_arr)
    roots = bisection_search_roots(0.0, (omega / c) ** 2, h, B1, omega, 1, 0.0, 1.0, 0)
    B1_complex = build_acoustic_matrix_complex(h, omega, c_arr, attenuation_arr)

    complex_roots = secant_search_roots_media_complex(
        roots,
        h,
        B1_complex,
        omega,
        1,
        0.0,
        1.0,
        0,
        np.array([0], dtype=np.int64),
        np.array([N], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
    )
    wavenumbers = np.sqrt(complex_roots)

    assert len(complex_roots) == len(roots)
    assert np.all(np.imag(wavenumbers) > 0.0)


def test_reflection_table_impedance_converts_to_robin_boundary():
    freq = 50.0
    omega = 2 * np.pi * freq
    c = 1500.0
    k2 = (omega / 1550.0) ** 2
    table = np.array([[0.0, 1.0, 180.0], [90.0, 1.0, 180.0]], dtype=np.float64)

    f, g = reflection_table_impedance(k2, omega, c, 1.0, table)

    np.testing.assert_allclose(f, 1.0 + 0.0j)
    np.testing.assert_allclose(g, 0.0 + 0.0j)


def test_reflection_table_impedance_uses_at_reciprocal_amplitude_convention():
    freq = 50.0
    omega = 2 * np.pi * freq
    c = 1500.0
    k2 = (omega / 1550.0) ** 2
    rho = 1.0
    table = np.array([[0.0, 0.5, 30.0], [90.0, 0.5, 30.0]], dtype=np.float64)

    f, g = reflection_table_impedance(k2, omega, c, rho, table)

    kz = np.sqrt((omega / c) ** 2 - k2)
    coefficient = 2.0 * np.exp(1j * np.deg2rad(30.0))
    expected = 1j * kz * (1.0 - coefficient) / (rho * (1.0 + coefficient))
    np.testing.assert_allclose(f, expected)
    np.testing.assert_allclose(g, 1.0 + 0.0j)


def test_reflection_table_changes_complex_dispersion():
    D = 100.0
    c = 1500.0
    freq = 50.0
    omega = 2 * np.pi * freq

    N = 300
    h = D / N
    c_arr = np.full(N + 1, c)
    B1_complex = build_acoustic_matrix_complex(h, omega, c_arr)
    k2 = (omega / 1520.0) ** 2
    media_start = np.array([0], dtype=np.int64)
    media_end = np.array([N], dtype=np.int64)
    media_rho = np.array([1.0], dtype=np.float64)
    pressure_release_table = np.array([[0.0, 1.0, 180.0], [90.0, 1.0, 180.0]], dtype=np.float64)

    delta_rigid = evaluate_dispersion_media_complex(
        k2,
        h,
        B1_complex,
        omega,
        1,
        1600.0,
        1.8,
        0,
        media_start,
        media_end,
        media_rho,
    )
    delta_table = evaluate_dispersion_media_complex(
        k2,
        h,
        B1_complex,
        omega,
        4,
        1600.0,
        1.8,
        0,
        media_start,
        media_end,
        media_rho,
        c_bot_inside=c,
        rho_bot_inside=1.0,
        bot_reflection_table=pressure_release_table,
    )

    assert not np.isclose(delta_table, delta_rigid)

def test_extrapolation():
    # True root is 10.0
    # h^2 error: f(h) = 10.0 + c * h^2
    # Suppose c = 2.0
    # multiplier 1 -> h=1 -> f=12.0
    # multiplier 2 -> h=0.5 -> f=10.5
    # multiplier 4 -> h=0.25 -> f=10.125
    
    seqs = [
        np.array([12.0]),
        np.array([10.5]),
        np.array([10.125])
    ]
    mults = [1, 2, 4]
    
    extrap = neville_extrapolation(seqs, mults)
    
    np.testing.assert_allclose(extrap[0], 10.0, atol=1e-10)
