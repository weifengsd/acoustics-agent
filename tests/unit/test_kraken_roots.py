import numpy as np
from pyacoustics.solvers.kraken.core import build_acoustic_matrix, top_impedance, bottom_impedance
from pyacoustics.solvers.kraken.roots import bisection_search_roots, neville_extrapolation

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
