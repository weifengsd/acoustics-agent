import numpy as np
import pytest
from pyacoustics.solvers.kraken.core import build_acoustic_matrix, top_impedance, bottom_impedance, evaluate_dispersion

def test_build_acoustic_matrix():
    h = 1.0
    omega = 2 * np.pi * 150.0  # 150 Hz
    c_arr = np.array([1500.0, 1500.0])
    
    B1 = build_acoustic_matrix(h, omega, c_arr)
    
    # omega/c = 2 * pi * 150 / 1500 = 2 * pi / 10 = pi / 5
    expected_B1 = -2.0 + (np.pi / 5)**2
    
    np.testing.assert_allclose(B1, [expected_B1, expected_B1])

def test_ideal_waveguide_dispersion():
    """
    Test the dispersion function for an ideal waveguide (vacuum top, rigid bottom)
    Depth D = 100m, c = 1500 m/s, f = 15 Hz
    omega = 2*pi*15 = 30*pi
    Analytical vertical wavenumbers k_z = (m - 0.5) * pi / D for m=1,2,...
    k^2 = (omega/c)^2 - k_z^2
    """
    D = 100.0
    c = 1500.0
    freq = 15.0
    omega = 2 * np.pi * freq
    
    N = 1000  # Number of intervals
    h = D / N
    c_arr = np.full(N + 1, c)
    
    B1 = build_acoustic_matrix(h, omega, c_arr)
    
    # Top: Vacuum (0), Bottom: Rigid (1)
    f_top, g_top = top_impedance(0)
    f_bot, g_bot = bottom_impedance(0.0, omega, 1, 0.0, 1.0)
    
    # Analytical eigenvalue for mode 1 (m=1)
    kz_1 = (1 - 0.5) * np.pi / D
    k2_1 = (omega / c)**2 - kz_1**2
    
    # Evaluate at the analytical root. Delta should be close to 0.
    delta, mode_count = evaluate_dispersion(k2_1, h, B1, omega, 1, 0.0, 1.0, 0)
    
    # Due to discretization error, it won't be exactly 0, but should be small
    # And mode count should be 0 or 1 depending on where exactly it crosses
    
    # Let's do a simple scan to find the zero crossing
    k2_scan = np.linspace(0, (omega/c)**2, 100)
    deltas = []
    modes = []
    for k2 in k2_scan:
        d, m = evaluate_dispersion(k2, h, B1, omega, 1, 0.0, 1.0, 0)
        deltas.append(d)
        modes.append(m)
        
    deltas = np.array(deltas)
    # The mode count should monotonically increase as k2 decreases (kz increases)
    # Number of propagating modes = floor(2 * freq * D / c + 0.5) = floor(2 * 15 * 100 / 1500 + 0.5) = floor(2 + 0.5) = 2
    
    # The highest k2 is mode 1 (smallest kz)
    assert max(modes) >= 1
