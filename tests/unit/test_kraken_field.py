import numpy as np
from pyacoustics.solvers.kraken.field import compute_field

def test_compute_field_shapes_and_values():
    M = 2
    N = 100
    R = 50
    Z = 20
    
    modes = np.random.rand(M, N)
    roots = np.array([0.1, 0.05])
    z_arr = np.linspace(0, 100, N)
    r_arr = np.linspace(10, 1000, R)
    z_s = 50.0
    z_r_arr = np.linspace(10, 90, Z)
    
    TL = compute_field(modes, roots, z_arr, r_arr, z_s, z_r_arr)
    
    assert TL.shape == (Z, R)
    # TL should generally be positive for standard scaling
    assert np.all(TL > -50.0) 
    
    # At r=0, we forced a small value, but let's test r=0 behavior
    r_arr_zero = np.array([0.0, 10.0])
    TL_zero = compute_field(modes, roots, z_arr, r_arr_zero, z_s, z_r_arr)
    assert not np.any(np.isnan(TL_zero))
    assert not np.any(np.isinf(TL_zero))
