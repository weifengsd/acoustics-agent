import numpy as np
from numba import njit
from pyacoustics.solvers.bellhop3d.step import initialize_ray_3d, step3d_kinematic, step3d_dynamic
from pyacoustics.solvers.bellhop3d.boundary import (
    apply_reflection_curvature_correction,
    apply_ssp_interface_curvature_correction_3d,
    boundary_crossing_fraction_3d,
    check_boundary_3d,
    crossed_interior_z_interface,
    reduce_step_fraction_3d,
    reflect_ray_3d,
)
from pyacoustics.environment import evaluate_hexahedral_ssp_3d, evaluate_linear_ssp, evaluate_spline_ssp, evaluate_n2linear_ssp

MAX_STEPS = 100000

@njit(fastmath=True)
def _media_index_3d(r: float, media_r: np.ndarray, n_media: int) -> int:
    """Return the AT-style profile segment index for range-dependent boundary media."""
    if n_media <= 1 or len(media_r) == 0:
        return 0
    idx = int(np.searchsorted(media_r, r, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= n_media:
        return n_media - 1
    return idx

@njit(fastmath=True)
def trace_single_ray_3d(
    x_0: float, y_0: float, z_0: float,
    alpha_elev_deg: float, alpha_azim_deg: float,
    h: float,
    z_surface: float, bty_x: np.ndarray, bty_y: np.ndarray, bty_z: np.ndarray, z_bottom_flat: float,
    surf_rho: float, surf_c: float, surf_c_s: float, surf_alpha: float, surf_alpha_s: float,
    bot_rho: float, bot_c: float, bot_c_s: float, bot_alpha: float, bot_alpha_s: float,
    ati_media_r: np.ndarray, ati_geo_cp: np.ndarray, ati_geo_cs: np.ndarray,
    ati_geo_rho: np.ndarray, ati_geo_alpha: np.ndarray, ati_geo_alpha_s: np.ndarray,
    bty_media_r: np.ndarray, bty_geo_cp: np.ndarray, bty_geo_cs: np.ndarray,
    bty_geo_rho: np.ndarray, bty_geo_alpha: np.ndarray, bty_geo_alpha_s: np.ndarray,
    surface_refl_angles: np.ndarray, surface_refl_amp: np.ndarray, surface_refl_phase: np.ndarray,
    bottom_refl_angles: np.ndarray, bottom_refl_amp: np.ndarray, bottom_refl_phase: np.ndarray,
    r_max: float,
    box_x: float, box_y: float, box_z: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray,
    ssp3d_x: np.ndarray, ssp3d_y: np.ndarray, ssp3d_z: np.ndarray, ssp3d_c: np.ndarray,
    max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:

    x_path = np.zeros(max_steps + 1)
    y_path = np.zeros(max_steps + 1)
    z_path = np.zeros(max_steps + 1)
    q11_path = np.zeros(max_steps + 1, dtype=np.complex128)
    q12_path = np.zeros(max_steps + 1, dtype=np.complex128)
    q21_path = np.zeros(max_steps + 1, dtype=np.complex128)
    q22_path = np.zeros(max_steps + 1, dtype=np.complex128)
    amp_path = np.zeros(max_steps + 1, dtype=np.complex128)
    phase_path = np.zeros(max_steps + 1)
    tau_path = np.zeros(max_steps + 1)
    kmah_path = np.zeros(max_steps + 1)
    c_path = np.zeros(max_steps + 1) # Added to store local sound speed for Amp compensation
    phi_path = np.zeros(max_steps + 1)
    top_bounce_path = np.zeros(max_steps + 1, dtype=np.int64)
    bottom_bounce_path = np.zeros(max_steps + 1, dtype=np.int64)

    if ssp_type == 0:
        c_0, _, _ = evaluate_linear_ssp(z_0, z_arr, c_arr)
    elif ssp_type == 1:
        c_0, _, _ = evaluate_spline_ssp(z_0, z_arr, c_coeffs)
    elif ssp_type == 2:
        c_0, _, _ = evaluate_n2linear_ssp(z_0, z_arr, c_arr)
    else:
        c_0, _, _, _, _ = evaluate_hexahedral_ssp_3d(x_0, y_0, z_0, ssp3d_x, ssp3d_y, ssp3d_z, ssp3d_c)

    x, y, z, xi, eta, zeta, tau = initialize_ray_3d(x_0, y_0, z_0, alpha_elev_deg, alpha_azim_deg, c_0)
    phi = 0.0

    # Initialize 3D point source Dynamic Ray Tracing matrices
    # Our normalization: slowness-based P => q grows like s (meters). p11=p22=1/c_0.
    q11, q12, q21, q22 = 0j, 0j, 0j, 0j
    p11 = 1.0 / c_0 + 0j
    p22 = 1.0 / c_0 + 0j   # symmetric: remove cos(elev) coupling
    p12, p21 = 0j, 0j

    x_path[0] = x
    y_path[0] = y
    z_path[0] = z
    q11_path[0] = q11
    q12_path[0] = q12
    q21_path[0] = q21
    q22_path[0] = q22
    amp_path[0] = 1.0
    phase_path[0] = 0.0
    tau_path[0] = 0.0
    kmah_path[0] = 0.0
    c_path[0] = c_0
    phi_path[0] = phi
    top_bounce_path[0] = 0
    bottom_bounce_path[0] = 0

    n_points = 1
    n_top_bounces = 0
    n_bottom_bounces = 0

    for step_i in range(max_steps):
        # Determine if we need to reduce step to hit boundary
        h_eff = h

        x_trial = x + h * c_path[n_points-1] * xi
        y_trial = y + h * c_path[n_points-1] * eta
        z_trial = z + h * c_path[n_points-1] * zeta
        hit, z_b_local, dz_dx, dz_dy = check_boundary_3d(
            x, y, z, x_trial, y_trial, z_trial,
            z_surface, bty_x, bty_y, bty_z, z_bottom_flat
        )

        if hit != 0:
            frac = boundary_crossing_fraction_3d(
                x, y, z, x_trial, y_trial, z_trial, hit,
                z_surface, bty_x, bty_y, bty_z, z_bottom_flat
            )
            h_eff = max(0.0, min(h, h * frac))

        ux = c_path[n_points - 1] * xi
        uy = c_path[n_points - 1] * eta
        uz = c_path[n_points - 1] * zeta
        h_box = h
        if abs(x_trial) > box_x and abs(ux) > 1.0e-12:
            hx = (box_x - abs(x)) / abs(ux)
            if hx >= 0.0 and hx < h_box:
                h_box = hx
        if abs(y_trial) > box_y and abs(uy) > 1.0e-12:
            hy = (box_y - abs(y)) / abs(uy)
            if hy >= 0.0 and hy < h_box:
                h_box = hy
        if abs(z_trial) > box_z and abs(uz) > 1.0e-12:
            hz = (box_z - abs(z)) / abs(uz)
            if hz >= 0.0 and hz < h_box:
                h_box = hz
        if h_box < h_eff:
            h_eff = max(0.0, h_box)

        if ssp_type == 3:
            seg_frac = reduce_step_fraction_3d(
                x, y, z, ux, uy, uz, h,
                bty_x, bty_y, ssp3d_x, ssp3d_y, ssp3d_z,
                True
            )
        else:
            seg_frac = reduce_step_fraction_3d(
                x, y, z, ux, uy, uz, h,
                bty_x, bty_y, ssp3d_x, ssp3d_y, z_arr,
                False
            )
        if h * seg_frac < h_eff:
            h_eff = max(0.0, h * seg_frac)

        x_new, y_new, z_new, xi_new, eta_new, zeta_new, tau_new, phi_new = step3d_kinematic(
            h_eff, x, y, z, xi, eta, zeta, tau, phi, ssp_type, z_arr, c_arr, c_coeffs,
            ssp3d_x, ssp3d_y, ssp3d_z, ssp3d_c
        )

        # Advance DRT matrices
        if ssp_type == 0:
            c_current, dc_dz, d2c_dz2 = evaluate_linear_ssp(z, z_arr, c_arr)
            dc_dx, dc_dy = 0.0, 0.0
        elif ssp_type == 1:
            c_current, dc_dz, d2c_dz2 = evaluate_spline_ssp(z, z_arr, c_coeffs)
            dc_dx, dc_dy = 0.0, 0.0
        elif ssp_type == 2:
            c_current, dc_dz, d2c_dz2 = evaluate_n2linear_ssp(z, z_arr, c_arr)
            dc_dx, dc_dy = 0.0, 0.0
        else:
            c_current, dc_dx, dc_dy, dc_dz, d2c_dz2 = evaluate_hexahedral_ssp_3d(
                x, y, z, ssp3d_x, ssp3d_y, ssp3d_z, ssp3d_c
            )

        # Unit tangent and normals for Ray-Centered coordinates
        tx, ty, tz = c_current * xi, c_current * eta, c_current * zeta
        rl = np.sqrt(tx**2 + ty**2)
        if rl > 1e-10:
            e1_z = -rl * np.cos(phi)
            e2_z = -rl * np.sin(phi)
        else:
            e1_z = 0.0
            e2_z = 0.0

        # Curvature along normals: Cnn = d2c/dz2 * e_z^2
        # (Assuming range-independent czz is the only non-zero component of Hessian)
        cnn11 = d2c_dz2 * (e1_z**2)
        cnn12 = d2c_dz2 * (e1_z * e2_z)
        cnn21 = cnn12
        cnn22 = d2c_dz2 * (e2_z**2)

        q11, q12, q21, q22, p11, p12, p21, p22 = step3d_dynamic(
            h_eff, c_current,
            q11, q12, q21, q22, p11, p12, p21, p22,
            cnn11, cnn12, cnn21, cnn22
        )

        if (ssp_type == 0 or ssp_type == 2) and crossed_interior_z_interface(z, z_new, z_arr):
            if zeta_new > 0.0:
                z_after = z_new + 1.0e-6
            else:
                z_after = z_new - 1.0e-6

            if ssp_type == 0:
                c_interface, _, _ = evaluate_linear_ssp(z_new, z_arr, c_arr)
                _, dc_dz_after, _ = evaluate_linear_ssp(z_after, z_arr, c_arr)
            else:
                c_interface, _, _ = evaluate_n2linear_ssp(z_new, z_arr, c_arr)
                _, dc_dz_after, _ = evaluate_n2linear_ssp(z_after, z_arr, c_arr)

            grad_jump_z = dc_dz_after - dc_dz
            n_z = -1.0
            if zeta_new < 0.0:
                n_z = 1.0
            p11, p12, p21, p22 = apply_ssp_interface_curvature_correction_3d(
                p11, p12, p21, p22,
                q11, q12, q21, q22,
                xi_new, eta_new, zeta_new, c_interface,
                0.0, 0.0, n_z,
                0.0, 0.0, grad_jump_z,
            )

        # Recalculate hit with effective step
        hit, z_b_local, dz_dx, dz_dy = check_boundary_3d(
            x, y, z, x_new, y_new, z_new, z_surface, bty_x, bty_y, bty_z, z_bottom_flat
        )

        if hit != 0:
            if hit == 1:
                n_top_bounces += 1
            else:
                n_bottom_bounces += 1
            x_path[n_points] = x_new
            y_path[n_points] = y_new
            z_path[n_points] = z_new
            q11_path[n_points] = q11
            q12_path[n_points] = q12
            q21_path[n_points] = q21
            q22_path[n_points] = q22
            tau_path[n_points] = tau_new
            kmah_path[n_points] = kmah_path[n_points-1]
            phase_path[n_points] = phase_path[n_points - 1]
            phi_path[n_points] = phi_new
            top_bounce_path[n_points] = n_top_bounces
            bottom_bounce_path[n_points] = n_bottom_bounces

            if ssp_type == 0:
                c_bnd, dc_dz_bnd, _ = evaluate_linear_ssp(z_new, z_arr, c_arr)
                dc_dx_bnd, dc_dy_bnd = 0.0, 0.0
            elif ssp_type == 1:
                c_bnd, dc_dz_bnd, _ = evaluate_spline_ssp(z_new, z_arr, c_coeffs)
                dc_dx_bnd, dc_dy_bnd = 0.0, 0.0
            elif ssp_type == 2:
                c_bnd, dc_dz_bnd, _ = evaluate_n2linear_ssp(z_new, z_arr, c_arr)
                dc_dx_bnd, dc_dy_bnd = 0.0, 0.0
            else:
                c_bnd, dc_dx_bnd, dc_dy_bnd, dc_dz_bnd, _ = evaluate_hexahedral_ssp_3d(
                    x_new, y_new, z_new, ssp3d_x, ssp3d_y, ssp3d_z, ssp3d_c
                )

            c_path[n_points] = c_bnd

            refl_rho = bot_rho
            refl_c = bot_c
            refl_c_s = bot_c_s
            refl_alpha = bot_alpha
            refl_alpha_s = bot_alpha_s
            if hit == 1:
                refl_rho = surf_rho
                refl_c = surf_c
                refl_c_s = surf_c_s
                refl_alpha = surf_alpha
                refl_alpha_s = surf_alpha_s
                if len(ati_geo_cp) > 0:
                    media_idx = _media_index_3d(np.sqrt(x_new * x_new + y_new * y_new), ati_media_r, len(ati_geo_cp))
                    refl_c = ati_geo_cp[media_idx]
                    refl_c_s = ati_geo_cs[media_idx]
                    refl_rho = ati_geo_rho[media_idx]
                    refl_alpha = ati_geo_alpha[media_idx]
                    refl_alpha_s = ati_geo_alpha_s[media_idx]
            elif len(bty_geo_cp) > 0:
                media_idx = _media_index_3d(np.sqrt(x_new * x_new + y_new * y_new), bty_media_r, len(bty_geo_cp))
                refl_c = bty_geo_cp[media_idx]
                refl_c_s = bty_geo_cs[media_idx]
                refl_rho = bty_geo_rho[media_idx]
                refl_alpha = bty_geo_alpha[media_idx]
                refl_alpha_s = bty_geo_alpha_s[media_idx]

            xi_in, eta_in, zeta_in = xi_new, eta_new, zeta_new
            xi_new, eta_new, zeta_new, R = reflect_ray_3d(
                xi_new, eta_new, zeta_new,
                hit, dz_dx, dz_dy,
                refl_rho, refl_c, refl_c_s, refl_alpha, refl_alpha_s,
                c_bnd,
                surface_refl_angles, surface_refl_amp, surface_refl_phase,
                bottom_refl_angles, bottom_refl_amp, bottom_refl_phase,
                refl_rho, refl_c, refl_c_s, refl_alpha, refl_alpha_s,
            )
            p11, p12, p21, p22 = apply_reflection_curvature_correction(
                p11, p12, p21, p22,
                q11, q12, q21, q22,
                xi_in, eta_in, zeta_in,
                xi_new, eta_new, zeta_new,
                hit, dz_dx, dz_dy, c_bnd,
                dc_dx_bnd, dc_dy_bnd, dc_dz_bnd
            )

            amp_path[n_points] = amp_path[n_points - 1] * R
            phase_path[n_points] = phase_path[n_points] + np.angle(R)
            n_points += 1

            x, y, z = x_new, y_new, z_new
            xi, eta, zeta = xi_new, eta_new, zeta_new
            tau = tau_new
            phi = phi_new
        else:
            x_path[n_points] = x_new
            y_path[n_points] = y_new
            z_path[n_points] = z_new
            q11_path[n_points] = q11
            q12_path[n_points] = q12
            q21_path[n_points] = q21
            q22_path[n_points] = q22
            amp_path[n_points] = amp_path[n_points - 1]
            phase_path[n_points] = phase_path[n_points - 1]
            tau_path[n_points] = tau_new
            kmah_path[n_points] = kmah_path[n_points-1]
            phi_path[n_points] = phi_new
            top_bounce_path[n_points] = n_top_bounces
            bottom_bounce_path[n_points] = n_bottom_bounces

            if ssp_type == 0:
                c_now, _, _ = evaluate_linear_ssp(z_new, z_arr, c_arr)
            elif ssp_type == 1:
                c_now, _, _ = evaluate_spline_ssp(z_new, z_arr, c_coeffs)
            elif ssp_type == 2:
                c_now, _, _ = evaluate_n2linear_ssp(z_new, z_arr, c_arr)
            else:
                c_now, _, _, _, _ = evaluate_hexahedral_ssp_3d(
                    x_new, y_new, z_new, ssp3d_x, ssp3d_y, ssp3d_z, ssp3d_c
                )
            c_path[n_points] = c_now

            n_points += 1
            x, y, z = x_new, y_new, z_new
            xi, eta, zeta = xi_new, eta_new, zeta_new
            tau = tau_new
            phi = phi_new

        r_cyl = np.sqrt(x**2 + y**2)
        if r_cyl > r_max:
            break
        if abs(x) >= box_x or abs(y) >= box_y or abs(z) >= box_z:
            break
        if n_points >= max_steps:
            break

    return (
        x_path, y_path, z_path,
        q11_path, q12_path, q21_path, q22_path,
        amp_path, phase_path, tau_path, kmah_path, c_path, phi_path,
        top_bounce_path, bottom_bounce_path,
        n_points, n_top_bounces, n_bottom_bounces,
    )
