import numpy as np
from numba import njit
from pyacoustics.environment import evaluate_linear_ssp, evaluate_spline_ssp, evaluate_n2linear_ssp, evaluate_quad2d_ssp

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
def evaluate_step_ssp(
    ssp_type: int,
    r_q: float,
    z_q: float,
    z_arr: np.ndarray,
    c_arr: np.ndarray,
    c_coeffs: np.ndarray,
    ssp_r_arr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Evaluate the active SSP model for Step2D; type 3 uses c_coeffs as the Q c-matrix."""
    if ssp_type == 0:
        c, dc_dz, d2c_dz2 = evaluate_linear_ssp(z_q, z_arr, c_arr)
        return c, 0.0, dc_dz, d2c_dz2
    if ssp_type == 1:
        c, dc_dz, d2c_dz2 = evaluate_spline_ssp(z_q, z_arr, c_coeffs)
        return c, 0.0, dc_dz, d2c_dz2
    if ssp_type == 3:
        return evaluate_quad2d_ssp(r_q, z_q, ssp_r_arr, z_arr, c_coeffs)
    c, dc_dz, d2c_dz2 = evaluate_n2linear_ssp(z_q, z_arr, c_arr)
    return c, 0.0, dc_dz, d2c_dz2


@njit(fastmath=True)
def evaluate_quad2d_mixed_derivative(
    r_q: float,
    z_q: float,
    r_arr: np.ndarray,
    z_arr: np.ndarray,
    c_matrix: np.ndarray,
) -> float:
    """Return d2c/drdz for the active bilinear Q cell."""
    nr = len(r_arr)
    nz = len(z_arr)

    ir = np.searchsorted(r_arr, r_q) - 1
    iz = np.searchsorted(z_arr, z_q) - 1
    if ir < 0:
        ir = 0
    elif ir > nr - 2:
        ir = nr - 2
    if iz < 0:
        iz = 0
    elif iz > nz - 2:
        iz = nz - 2

    dr = r_arr[ir + 1] - r_arr[ir]
    dz = z_arr[iz + 1] - z_arr[iz]
    if dr == 0.0 or dz == 0.0:
        return 0.0

    c00 = c_matrix[iz, ir]
    c01 = c_matrix[iz, ir + 1]
    c10 = c_matrix[iz + 1, ir]
    c11 = c_matrix[iz + 1, ir + 1]
    return (c11 - c10 - c01 + c00) / (dr * dz)


@njit(fastmath=True)
def evaluate_step_normal_curvature(
    ssp_type: int,
    r_q: float,
    z_q: float,
    c_q: float,
    xi_q: float,
    zeta_q: float,
    d2c_dz2_q: float,
    z_arr: np.ndarray,
    c_coeffs: np.ndarray,
    ssp_r_arr: np.ndarray,
) -> float:
    """Evaluate c_nn along the ray-normal direction for Step2D dynamic rays."""
    cos_theta = c_q * xi_q
    if ssp_type == 3:
        sin_theta = c_q * zeta_q
        mixed = evaluate_quad2d_mixed_derivative(r_q, z_q, ssp_r_arr, z_arr, c_coeffs)
        return -2.0 * sin_theta * cos_theta * mixed
    return d2c_dz2_q * cos_theta * cos_theta


@njit(fastmath=True)
def step2d_kinematic(
    h: float, 
    r: float, z: float, xi: float, zeta: float, tau: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray,
    ssp_r_arr: np.ndarray
) -> tuple[float, float, float, float, float]:
    """
    Performs one step of kinematic ray tracing using the "polygon" 
    (modified midpoint) method.
    
    ssp_type: 0 for c-linear, 1 for spline
    Returns new state: r_new, z_new, xi_new, zeta_new, tau_new
    """
    
    # 1. Evaluate SSP at current position
    c1, dc_dr1, dc_dz1, _ = evaluate_step_ssp(
        ssp_type, r, z, z_arr, c_arr, c_coeffs, ssp_r_arr
    )
    
    # 2. Half-step Predictor
    h2 = h / 2.0
    c1_inv2 = 1.0 / (c1 * c1)
    
    xi_half = xi - h2 * c1_inv2 * dc_dr1
    zeta_half = zeta - h2 * c1_inv2 * dc_dz1
    
    r_half = r + h2 * c1 * xi
    z_half = z + h2 * c1 * zeta
    
    # 3. Evaluate SSP at half-step
    c_half, dc_dr_half, dc_dz_half, _ = evaluate_step_ssp(
        ssp_type, r_half, z_half, z_arr, c_arr, c_coeffs, ssp_r_arr
    )
    
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
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray,
    ssp_r_arr: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Performs one step of kinematic AND dynamic ray tracing using the "polygon"
    (modified midpoint) method (second-order RK).
    
    ssp_type: 0 for c-linear, 1 for spline
    Returns: r_new, z_new, xi_new, zeta_new, tau_new, p_new, q_new, c_cur
    """
    # 1. Evaluate SSP at current position
    c1, dc_dr1, dc_dz1, d2c_dz2_1 = evaluate_step_ssp(
        ssp_type, r, z, z_arr, c_arr, c_coeffs, ssp_r_arr
    )
    
    # Curvature at start point
    c_nn1 = evaluate_step_normal_curvature(
        ssp_type, r, z, c1, xi, zeta, d2c_dz2_1, z_arr, c_coeffs, ssp_r_arr
    )
    
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
    c_half, dc_dr_half, dc_dz_half, d2c_dz2_half = evaluate_step_ssp(
        ssp_type, r_half, z_half, z_arr, c_arr, c_coeffs, ssp_r_arr
    )
    
    # Curvature at half-step position
    c_nn_half = evaluate_step_normal_curvature(
        ssp_type,
        r_half,
        z_half,
        c_half,
        xi_half,
        zeta_half,
        d2c_dz2_half,
        z_arr,
        c_coeffs,
        ssp_r_arr,
    )
    
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
