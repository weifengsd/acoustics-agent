import numpy as np
from numba import njit
from pyacoustics.environment import evaluate_linear_ssp, evaluate_spline_ssp

@njit(fastmath=True)
def initialize_ray(r_0: float, z_0: float, alpha_deg: float, c_0: float) -> tuple[float, float, float, float, float]:
    """
    Initializes the ray state variables.
    alpha_deg: launch angle in degrees (positive is pointing downwards)
    Returns: r, z, xi, zeta, tau
    """
    alpha_rad = np.radians(alpha_deg)
    
    r = r_0
    z = z_0
    tau = 0.0
    
    # Slowness vector = unit_direction / c
    xi = np.cos(alpha_rad) / c_0
    zeta = np.sin(alpha_rad) / c_0
    
    return r, z, xi, zeta, tau

@njit(fastmath=True)
def step2d_kinematic(
    h: float, 
    r: float, z: float, xi: float, zeta: float, tau: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray
) -> tuple[float, float, float, float, float]:
    """
    Performs one step of kinematic ray tracing using the "polygon" 
    (modified midpoint) method.
    
    ssp_type: 0 for c-linear, 1 for spline
    Returns new state: r_new, z_new, xi_new, zeta_new, tau_new
    """
    
    # 1. Evaluate SSP at current position
    if ssp_type == 0:
        c1, dc_dz1, _ = evaluate_linear_ssp(z, z_arr, c_arr)
    else:
        c1, dc_dz1, _ = evaluate_spline_ssp(z, z_arr, c_coeffs)
        
    dc_dr1 = 0.0 # Range independent for now
    
    # 2. Half-step Predictor
    h2 = h / 2.0
    c1_inv2 = 1.0 / (c1 * c1)
    
    xi_half = xi - h2 * c1_inv2 * dc_dr1
    zeta_half = zeta - h2 * c1_inv2 * dc_dz1
    
    r_half = r + h2 * c1 * xi
    z_half = z + h2 * c1 * zeta
    
    # 3. Evaluate SSP at half-step
    if ssp_type == 0:
        c_half, dc_dz_half, _ = evaluate_linear_ssp(z_half, z_arr, c_arr)
    else:
        c_half, dc_dz_half, _ = evaluate_spline_ssp(z_half, z_arr, c_coeffs)
        
    dc_dr_half = 0.0 # Range independent
    
    # 4. Full-step Corrector
    c_half_inv2 = 1.0 / (c_half * c_half)
    
    xi_new = xi - h * c_half_inv2 * dc_dr_half
    zeta_new = zeta - h * c_half_inv2 * dc_dz_half
    
    r_new = r + h * c_half * xi_half
    z_new = z + h * c_half * zeta_half
    
    tau_new = tau + h / c_half
    
    return r_new, z_new, xi_new, zeta_new, tau_new


@njit(fastmath=True)
def step2d_kinematic_dynamic(
    h: float, 
    r: float, z: float, xi: float, zeta: float, tau: float,
    p: float, q: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Performs one step of kinematic AND dynamic ray tracing using the "polygon"
    (modified midpoint) method (second-order RK).
    
    ssp_type: 0 for c-linear, 1 for spline
    Returns: r_new, z_new, xi_new, zeta_new, tau_new, p_new, q_new, c_cur
    """
    # 1. Evaluate SSP at current position
    if ssp_type == 0:
        c1, dc_dz1, d2c_dz2_1 = evaluate_linear_ssp(z, z_arr, c_arr)
    else:
        c1, dc_dz1, d2c_dz2_1 = evaluate_spline_ssp(z, z_arr, c_coeffs)
        
    dc_dr1 = 0.0 # Range independent for now
    
    # Curvature at start point
    cos_theta1 = c1 * xi
    cos2_1 = cos_theta1 * cos_theta1
    c_nn1 = d2c_dz2_1 * cos2_1
    
    # 2. Half-step Predictor
    h2 = h / 2.0
    c1_inv2 = 1.0 / (c1 * c1)
    
    xi_half = xi - h2 * c1_inv2 * dc_dr1
    zeta_half = zeta - h2 * c1_inv2 * dc_dz1
    
    r_half = r + h2 * c1 * xi
    z_half = z + h2 * c1 * zeta
    
    # Predict p and q at half-step
    q_half = q + h2 * c1 * p
    p_half = p - h2 * (c_nn1 / (c1 * c1)) * q
    
    # 3. Evaluate SSP at half-step
    if ssp_type == 0:
        c_half, dc_dz_half, d2c_dz2_half = evaluate_linear_ssp(z_half, z_arr, c_arr)
    else:
        c_half, dc_dz_half, d2c_dz2_half = evaluate_spline_ssp(z_half, z_arr, c_coeffs)
        
    dc_dr_half = 0.0 # Range independent
    
    # Curvature at half-step position
    cos_theta_half = c_half * xi_half
    cos2_half = cos_theta_half * cos_theta_half
    c_nn_half = d2c_dz2_half * cos2_half
    
    # 4. Full-step Corrector
    c_half_inv2 = 1.0 / (c_half * c_half)
    
    xi_new = xi - h * c_half_inv2 * dc_dr_half
    zeta_new = zeta - h * c_half_inv2 * dc_dz_half
    
    r_new = r + h * c_half * xi_half
    z_new = z + h * c_half * zeta_half
    
    tau_new = tau + h / c_half
    
    # Correct p and q at full-step
    q_new = q + h * c_half * p_half
    p_new = p - h * (c_nn_half / (c_half * c_half)) * q_half
    
    return r_new, z_new, xi_new, zeta_new, tau_new, p_new, q_new, c1
