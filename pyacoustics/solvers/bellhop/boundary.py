import numpy as np
from numba import njit

@njit(fastmath=True)
def check_boundary(z_old: float, z_new: float, z_surface: float, z_bottom: float) -> int:
    """
    Checks if a ray step crossed a boundary.
    Returns:
        0: no crossing
        1: surface crossing (z <= z_surface)
       -1: bottom crossing  (z >= z_bottom)
    """
    if z_new <= z_surface:
        return 1
    if z_new >= z_bottom:
        return -1
    return 0

@njit(fastmath=True)
def compute_boundary_intersection(
    r_old: float, z_old: float,
    r_new: float, z_new: float,
    z_boundary: float
) -> tuple[float, float]:
    """
    Linearly interpolates to find the (r, z) where the ray crossed z_boundary.
    Returns: (r_hit, fraction) where fraction is how far along the step the hit occurred.
    """
    dz = z_new - z_old
    if abs(dz) < 1e-30:
        return r_old, 0.0

    fraction = (z_boundary - z_old) / dz
    r_hit = r_old + fraction * (r_new - r_old)
    return r_hit, fraction

@njit(fastmath=True)
def reflect_ray(
    zeta: float,
    boundary_type: int
) -> float:
    return -zeta

@njit(fastmath=True)
def reflect_ray_amp(
    zeta: float, xi: float,
    boundary_type: int,
    rho1: float, c1: float,
    rho2: float, c2: float, alpha_dB_lambda: float
) -> tuple[float, complex]:
    """
    Reflects the ray and computes the Rayleigh reflection coefficient.
    Returns: new_zeta, reflection_coefficient (complex)
    """
    new_zeta = -zeta
    
    if boundary_type == 1:
        # Surface (vacuum/pressure release)
        return new_zeta, -1.0 + 0.0j
        
    if boundary_type == -1:
        # Bottom
        if c2 <= 0.0:
            return new_zeta, 1.0 + 0.0j # Rigid
            
        kz1_over_w = abs(zeta)
        
        alpha_nep = alpha_dB_lambda / 8.6858896
        alpha_factor = alpha_nep / (2.0 * np.pi)
        c2_complex = c2 / (1.0 + 1j * alpha_factor)
        
        radicand = (1.0 / c2_complex)**2 - xi**2
        kz2_over_w = np.sqrt(radicand)
            
        num = rho2 * kz1_over_w - rho1 * kz2_over_w
        den = rho2 * kz1_over_w + rho1 * kz2_over_w
        
        if abs(den) < 1e-30:
            R = 1.0 + 0.0j
        else:
            R = num / den
            
        return new_zeta, R
        
    return new_zeta, 1.0 + 0.0j
