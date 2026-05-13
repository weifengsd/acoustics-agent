import numpy as np
from numba import njit
from pyacoustics.solvers.bellhop.step import initialize_ray, step2d_kinematic, step2d_kinematic_dynamic
from pyacoustics.solvers.bellhop.boundary import check_boundary, compute_boundary_intersection, reflect_ray, reflect_ray_amp
from pyacoustics.environment import evaluate_linear_ssp, evaluate_spline_ssp, evaluate_n2linear_ssp

# Maximum number of steps per ray (safety limit)
MAX_STEPS = 100000

@njit(fastmath=True)
def trace_single_ray(
    r_0: float, z_0: float, alpha_deg: float,
    h: float,
    z_surface: float, z_bottom: float,
    bot_rho: float, bot_c: float, bot_alpha: float,
    r_max: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray,
    max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Traces a single ray through the environment with boundary reflections.

    Returns:
        r_path: array of range coordinates
        z_path: array of depth coordinates
        amp_path: complex amplitude tracking boundary losses
        n_points: number of valid points in the path
        n_bounces: total number of boundary reflections
    """
    # Pre-allocate output arrays
    r_path = np.zeros(max_steps + 1)
    z_path = np.zeros(max_steps + 1)
    amp_path = np.zeros(max_steps + 1, dtype=np.complex128)

    # Get initial sound speed
    if ssp_type == 0:
        c_0, _, _ = evaluate_linear_ssp(z_0, z_arr, c_arr)
    elif ssp_type == 1:
        c_0, _, _ = evaluate_spline_ssp(z_0, z_arr, c_coeffs)
    else:
        c_0, _, _ = evaluate_n2linear_ssp(z_0, z_arr, c_arr)

    # Initialize ray state
    r, z, xi, zeta, tau = initialize_ray(r_0, z_0, alpha_deg, c_0)
    amp = 1.0 + 0.0j
    
    r_path[0] = r
    z_path[0] = z
    amp_path[0] = amp

    n_points = 1
    n_bounces = 0

    for step_i in range(max_steps):
        # --- 1. Take a full step ---
        r_new, z_new, xi_new, zeta_new, tau_new = step2d_kinematic(
            h, r, z, xi, zeta, tau, ssp_type, z_arr, c_arr, c_coeffs
        )

        # --- 2. Check for boundary crossing ---
        hit = check_boundary(z, z_new, z_surface, z_bottom)

        if hit != 0:
            # Determine which boundary was crossed
            z_bnd = z_surface if hit == 1 else z_bottom

            # Find exact intersection point
            r_hit, frac = compute_boundary_intersection(r, z, r_new, z_new, z_bnd)

            # Record the hit point
            r_path[n_points] = r_hit
            z_path[n_points] = z_bnd
            amp_path[n_points] = amp
            n_points += 1

            # Interpolate c at the boundary to compute reflection coeff accurately
            if ssp_type == 0:
                c_bnd, _, _ = evaluate_linear_ssp(z_bnd, z_arr, c_arr)
            elif ssp_type == 1:
                c_bnd, _, _ = evaluate_spline_ssp(z_bnd, z_arr, c_coeffs)
            else:
                c_bnd, _, _ = evaluate_n2linear_ssp(z_bnd, z_arr, c_arr)

            # Reflect the ray and update amplitude
            zeta_new, R = reflect_ray_amp(
                zeta_new, xi_new, hit,
                1.0, c_bnd, bot_rho, bot_c, bot_alpha
            )
            amp = amp * R

            # Update position to the boundary point, keep reflected slowness
            r = r_hit
            z = z_bnd
            xi = xi_new
            zeta = zeta_new
            tau = tau_new
            n_bounces += 1
        else:
            # No boundary hit, accept the step
            r_path[n_points] = r_new
            z_path[n_points] = z_new
            amp_path[n_points] = amp
            n_points += 1

            r = r_new
            z = z_new
            xi = xi_new
            zeta = zeta_new
            tau = tau_new

        # --- 3. Termination condition ---
        if r > r_max:
            break
        if n_points >= max_steps:
            break

    return r_path, z_path, amp_path, n_points, n_bounces


@njit(fastmath=True)
def trace_beam_ray(
    r_0: float, z_0: float, alpha_deg: float,
    h: float,
    z_surface: float, z_bottom: float,
    bot_rho: float, bot_c: float, bot_alpha: float,
    r_max: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray,
    max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Traces a single ray AND tracks the dynamic ray parameters (p, q)
    needed for Gaussian Beam summation.

    The dynamic ray equations track how a beam tube evolves:
      dq/ds = c * p
      dp/ds = -(1/c^2) * d2c/dn2 * q  (where n is normal to the ray)

    For a 2D range-independent medium the curvature term simplifies to:
      d2c/dn2 ≈ d2c/dz2 * cos^2(theta) - (1/c)*(dc/dz)^2 * sin^2(theta)

    Returns:
        r_path, z_path: ray coordinates
        c_path: sound speed along the ray
        tau_path: travel time
        p_path, q_path: dynamic ray parameters (beam width evolution)
        amp_path: complex amplitude (reflection losses)
        n_points: number of valid points
        n_bounces: total reflections
    """
    # Pre-allocate output arrays
    r_path = np.zeros(max_steps + 1)
    z_path = np.zeros(max_steps + 1)
    c_path = np.zeros(max_steps + 1)
    p_path = np.zeros(max_steps + 1)
    q_path = np.zeros(max_steps + 1)
    tau_path = np.zeros(max_steps + 1)
    amp_path = np.zeros(max_steps + 1, dtype=np.complex128)

    # Get initial sound speed and derivatives
    if ssp_type == 0:
        c_0, dc_dz_0, d2c_dz2_0 = evaluate_linear_ssp(z_0, z_arr, c_arr)
    elif ssp_type == 1:
        c_0, dc_dz_0, d2c_dz2_0 = evaluate_spline_ssp(z_0, z_arr, c_coeffs)
    else:
        c_0, dc_dz_0, d2c_dz2_0 = evaluate_n2linear_ssp(z_0, z_arr, c_arr)

    # Initialize ray state
    alpha_rad = np.radians(alpha_deg)
    r = r_0
    z = z_0
    xi = np.cos(alpha_rad) / c_0
    zeta = np.sin(alpha_rad) / c_0
    tau = 0.0

    # Initialize dynamic ray parameters
    # p = d(theta)/d(alpha), q = d(n)/d(alpha)
    # At the source: q_0 = 0 (zero beam width), p_0 = 1 (unit spreading rate)
    p = 1.0
    q = 0.0
    amp = 1.0 + 0.0j

    r_path[0] = r
    z_path[0] = z
    c_path[0] = c_0
    p_path[0] = p
    q_path[0] = q
    tau_path[0] = tau
    amp_path[0] = amp

    n_points = 1
    n_bounces = 0

    for step_i in range(max_steps):
        # --- 1. Kinematic & Dynamic Step (Second-order RK) ---
        r_new, z_new, xi_new, zeta_new, tau_new, p_new, q_new, c_cur = step2d_kinematic_dynamic(
            h, r, z, xi, zeta, tau, p, q, ssp_type, z_arr, c_arr, c_coeffs
        )

        # --- 4. Check for boundary crossing ---
        hit = check_boundary(z, z_new, z_surface, z_bottom)

        if hit != 0:
            z_bnd = z_surface if hit == 1 else z_bottom
            r_hit, frac = compute_boundary_intersection(r, z, r_new, z_new, z_bnd)

            r_path[n_points] = r_hit
            z_path[n_points] = z_bnd

            # Interpolate c at the boundary
            if ssp_type == 0:
                c_bnd, _, _ = evaluate_linear_ssp(z_bnd, z_arr, c_arr)
            elif ssp_type == 1:
                c_bnd, _, _ = evaluate_spline_ssp(z_bnd, z_arr, c_coeffs)
            else:
                c_bnd, _, _ = evaluate_n2linear_ssp(z_bnd, z_arr, c_arr)

            c_path[n_points] = c_bnd

            # Interpolate dynamic params
            p_path[n_points] = p + frac * (p_new - p)
            q_path[n_points] = q + frac * (q_new - q)
            tau_path[n_points] = tau + frac * (tau_new - tau)
            amp_path[n_points] = amp
            n_points += 1

            # Reflect and update amplitude
            zeta_new, R = reflect_ray_amp(
                zeta_new, xi_new, hit,
                1.0, c_bnd, bot_rho, bot_c, bot_alpha
            )
            amp = amp * R
            
            # At a reflection, p remains continuous for flat boundaries (RN = 0)
            # matching AT Bellhop's ReflectMod L205-206.
            p_new = p_new

            r = r_hit
            z = z_bnd
            xi = xi_new
            zeta = zeta_new
            tau = tau_new
            p = p + frac * (p_new - p)
            q = q + frac * (q_new - q)
            n_bounces += 1
        else:
            r_path[n_points] = r_new
            z_path[n_points] = z_new

            if ssp_type == 0:
                c_new, _, _ = evaluate_linear_ssp(z_new, z_arr, c_arr)
            elif ssp_type == 1:
                c_new, _, _ = evaluate_spline_ssp(z_new, z_arr, c_coeffs)
            else:
                c_new, _, _ = evaluate_n2linear_ssp(z_new, z_arr, c_arr)

            c_path[n_points] = c_new
            p_path[n_points] = p_new
            q_path[n_points] = q_new
            tau_path[n_points] = tau_new
            amp_path[n_points] = amp
            n_points += 1

            r = r_new
            z = z_new
            xi = xi_new
            zeta = zeta_new
            tau = tau_new
            p = p_new
            q = q_new

        # --- 5. Termination ---
        if r > r_max:
            break
        if n_points >= max_steps:
            break

    return r_path, z_path, c_path, tau_path, p_path, q_path, amp_path, n_points, n_bounces
