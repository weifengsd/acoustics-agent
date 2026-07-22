import numpy as np
from numba import njit
from pyacoustics.environment import evaluate_hexahedral_ssp_3d, evaluate_linear_ssp, evaluate_spline_ssp, evaluate_n2linear_ssp

@njit(fastmath=True)
def initialize_ray_3d(x_0: float, y_0: float, z_0: float, alpha_elev_deg: float, alpha_azim_deg: float, c_0: float):
    """Initializes a 3D ray trajectory in Cartesian coordinates."""
    elev_rad = np.radians(alpha_elev_deg)
    azim_rad = np.radians(alpha_azim_deg)

    # 3D Direction cosines (xi, eta, zeta)
    xi = np.cos(elev_rad) * np.cos(azim_rad) / c_0
    eta = np.cos(elev_rad) * np.sin(azim_rad) / c_0
    zeta = np.sin(elev_rad) / c_0

    return x_0, y_0, z_0, xi, eta, zeta, 0.0

@njit(fastmath=True)
def step3d_kinematic(
    h: float, x: float, y: float, z: float, xi: float, eta: float, zeta: float, tau: float, phi: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray,
    ssp3d_x: np.ndarray, ssp3d_y: np.ndarray, ssp3d_z: np.ndarray, ssp3d_c: np.ndarray
):
    """
    Integrates the 3D Eikonal equations using weighted RK2 (Heun's method).
    Matches AT Step3DMod.f90 Phase 1 & 2 logic.
    """
    halfh = 0.5 * h

    # Stage 1 (Euler predictor - half step)
    if ssp_type == 0:
        c0, dc_dz0, _ = evaluate_linear_ssp(z, z_arr, c_arr)
        dc_dx0, dc_dy0 = 0.0, 0.0
    elif ssp_type == 1:
        c0, dc_dz0, _ = evaluate_spline_ssp(z, z_arr, c_coeffs)
        dc_dx0, dc_dy0 = 0.0, 0.0
    elif ssp_type == 2:
        c0, dc_dz0, _ = evaluate_n2linear_ssp(z, z_arr, c_arr)
        dc_dx0, dc_dy0 = 0.0, 0.0
    else:
        c0, dc_dx0, dc_dy0, dc_dz0, _ = evaluate_hexahedral_ssp_3d(
            x, y, z, ssp3d_x, ssp3d_y, ssp3d_z, ssp3d_c
        )

    csq0 = c0 * c0

    # Half-step Euler
    x1 = x + halfh * (c0 * xi)
    y1 = y + halfh * (c0 * eta)
    z1 = z + halfh * (c0 * zeta)
    xi1 = xi - halfh * dc_dx0 / csq0
    eta1 = eta - halfh * dc_dy0 / csq0
    zeta1 = zeta - halfh * dc_dz0 / csq0
    denom0 = xi * xi + eta * eta
    if denom0 > 1.0e-20:
        dphi0 = (1.0 / c0) * zeta * (eta * dc_dx0 - xi * dc_dy0) / denom0
    else:
        dphi0 = 0.0

    # Stage 2 (Corrector - full step evaluation at mid-point)
    if ssp_type == 0:
        c1, dc_dz1, _ = evaluate_linear_ssp(z1, z_arr, c_arr)
        dc_dx1, dc_dy1 = 0.0, 0.0
    elif ssp_type == 1:
        c1, dc_dz1, _ = evaluate_spline_ssp(z1, z_arr, c_coeffs)
        dc_dx1, dc_dy1 = 0.0, 0.0
    elif ssp_type == 2:
        c1, dc_dz1, _ = evaluate_n2linear_ssp(z1, z_arr, c_arr)
        dc_dx1, dc_dy1 = 0.0, 0.0
    else:
        c1, dc_dx1, dc_dy1, dc_dz1, _ = evaluate_hexahedral_ssp_3d(
            x1, y1, z1, ssp3d_x, ssp3d_y, ssp3d_z, ssp3d_c
        )

    csq1 = c1 * c1

    # AT Step3D uses hw0/hw1 blending. With a fixed accepted step this reduces
    # to equal half-step weights and avoids the more dissipative pure midpoint
    # update previously used here.
    hw0 = 0.5 * h
    hw1 = 0.5 * h
    x_new = x + hw0 * (c0 * xi) + hw1 * (c1 * xi1)
    y_new = y + hw0 * (c0 * eta) + hw1 * (c1 * eta1)
    z_new = z + hw0 * (c0 * zeta) + hw1 * (c1 * zeta1)
    xi_new = xi - hw0 * dc_dx0 / csq0 - hw1 * dc_dx1 / csq1
    eta_new = eta - hw0 * dc_dy0 / csq0 - hw1 * dc_dy1 / csq1
    zeta_new = zeta - hw0 * dc_dz0 / csq0 - hw1 * dc_dz1 / csq1
    tau_new = tau + hw0 / c0 + hw1 / c1
    denom1 = xi1 * xi1 + eta1 * eta1
    if denom1 > 1.0e-20:
        dphi1 = (1.0 / c1) * zeta1 * (eta1 * dc_dx1 - xi1 * dc_dy1) / denom1
    else:
        dphi1 = 0.0
    phi_new = phi + hw0 * dphi0 + hw1 * dphi1

    return x_new, y_new, z_new, xi_new, eta_new, zeta_new, tau_new, phi_new

@njit(fastmath=True)
def step3d_dynamic(
    h: float, c: float,
    q11: complex, q12: complex, q21: complex, q22: complex,
    p11: complex, p12: complex, p21: complex, p22: complex,
    c_nn11: float, c_nn12: float, c_nn21: float, c_nn22: float
):
    """
    Integrates the 3D Dynamic Ray Tracing (DRT) matrix equations using RK2.
    Matches AT Step3DMod.f90 phase 1 & 2 logic for p_tilde, q_tilde etc.
    dQ/ds = c * P
    dP/ds = - (1/c^2) * C_nn * Q
    """
    halfh = 0.5 * h
    csq = c * c
    coef = -1.0 / csq

    # Stage 1: Half-step Euler
    dq11_1 = halfh * c * p11
    dq12_1 = halfh * c * p12
    dq21_1 = halfh * c * p21
    dq22_1 = halfh * c * p22

    dp11_1 = halfh * coef * (c_nn11 * q11 + c_nn12 * q21)
    dp12_1 = halfh * coef * (c_nn11 * q12 + c_nn12 * q22)
    dp21_1 = halfh * coef * (c_nn21 * q11 + c_nn22 * q21)
    dp22_1 = halfh * coef * (c_nn21 * q12 + c_nn22 * q22)

    q11_mid = q11 + dq11_1
    q12_mid = q12 + dq12_1
    q21_mid = q21 + dq21_1
    q22_mid = q22 + dq22_1
    p11_mid = p11 + dp11_1
    p12_mid = p12 + dp12_1
    p21_mid = p21 + dp21_1
    p22_mid = p22 + dp22_1

    hw0 = 0.5 * h
    hw1 = 0.5 * h
    q11_new = q11 + hw0 * c * p11 + hw1 * c * p11_mid
    q12_new = q12 + hw0 * c * p12 + hw1 * c * p12_mid
    q21_new = q21 + hw0 * c * p21 + hw1 * c * p21_mid
    q22_new = q22 + hw0 * c * p22 + hw1 * c * p22_mid

    p11_new = p11 + hw0 * coef * (c_nn11 * q11 + c_nn12 * q21) + hw1 * coef * (c_nn11 * q11_mid + c_nn12 * q21_mid)
    p12_new = p12 + hw0 * coef * (c_nn11 * q12 + c_nn12 * q22) + hw1 * coef * (c_nn11 * q12_mid + c_nn12 * q22_mid)
    p21_new = p21 + hw0 * coef * (c_nn21 * q11 + c_nn22 * q21) + hw1 * coef * (c_nn21 * q11_mid + c_nn22 * q21_mid)
    p22_new = p22 + hw0 * coef * (c_nn21 * q12 + c_nn22 * q22) + hw1 * coef * (c_nn21 * q12_mid + c_nn22 * q22_mid)

    return q11_new, q12_new, q21_new, q22_new, p11_new, p12_new, p21_new, p22_new
