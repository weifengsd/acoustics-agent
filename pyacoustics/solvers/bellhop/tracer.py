import numpy as np
from numba import njit
from pyacoustics.solvers.bellhop.step import (
    evaluate_step_ssp,
    initialize_ray,
    step2d_kinematic,
    step2d_kinematic_dynamic,
)
from pyacoustics.solvers.bellhop.boundary import (
    apply_seongil_beam_displacement_2d,
    apply_reflection_curvature_correction_2d,
    apply_ssp_interface_curvature_correction_2d,
    apply_ssp_range_interface_curvature_correction_2d,
    check_boundary_profiles,
    compute_profile_intersection,
    crossed_profile_range_interface_2d,
    crossed_interior_z_interface_2d,
    get_profile_curvature,
    get_profile_reflection_vectors,
    reflect_ray_amp_vectors,
)

# Maximum number of steps per ray (safety limit)
MAX_STEPS = 100000
EVENT_TOL = 1.0e-10

@njit(fastmath=True)
def trace_single_ray(
    r_0: float, z_0: float, alpha_deg: float,
    h: float,
    ati_r: np.ndarray, ati_z: np.ndarray,
    ati_curvilinear: int,
    bty_r: np.ndarray, bty_z: np.ndarray,
    bty_curvilinear: int,
    bty_geo_cp: np.ndarray, bty_geo_cs: np.ndarray, bty_geo_rho: np.ndarray,
    bty_geo_alpha: np.ndarray, bty_geo_alpha_s: np.ndarray,
    ati_geo_cp: np.ndarray, ati_geo_cs: np.ndarray, ati_geo_rho: np.ndarray,
    ati_geo_alpha: np.ndarray, ati_geo_alpha_s: np.ndarray,
    surface_refl_angles: np.ndarray, surface_refl_amp: np.ndarray, surface_refl_phase: np.ndarray,
    bottom_refl_angles: np.ndarray, bottom_refl_amp: np.ndarray, bottom_refl_phase: np.ndarray,
    surf_rho: float, surf_c: float, surf_c_s: float, surf_alpha: float, surf_alpha_s: float,
    bot_rho: float, bot_c: float, bot_c_s: float, bot_alpha: float, bot_alpha_s: float,
    r_max: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray,
    ssp_r_arr: np.ndarray,
    max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    # Pre-allocate output arrays
    r_path = np.zeros(max_steps + 1)
    z_path = np.zeros(max_steps + 1)
    amp_path = np.zeros(max_steps + 1, dtype=np.complex128)

    # Get initial sound speed
    c_0, _, _, _ = evaluate_step_ssp(ssp_type, r_0, z_0, z_arr, c_arr, c_coeffs, ssp_r_arr)

    # Initialize ray state
    r, z, xi, zeta, tau = initialize_ray(r_0, z_0, alpha_deg, c_0)
    amp = 1.0 + 0.0j
    
    r_path[0] = r
    z_path[0] = z
    amp_path[0] = amp

    n_points = 1
    n_bounces = 0
    ati_idx = 0
    bty_idx = 0

    for step_i in range(max_steps):
        r_new, z_new, xi_new, zeta_new, tau_new = step2d_kinematic(
            h, r, z, xi, zeta, tau, ssp_type, z_arr, c_arr, c_coeffs, ssp_r_arr
        )

        crossed_interface, interface_frac = crossed_interior_z_interface_2d(z, z_new, z_arr)
        crossed_ati_range, ati_range_frac = crossed_profile_range_interface_2d(r, r_new, ati_r)
        crossed_bty_range, bty_range_frac = crossed_profile_range_interface_2d(r, r_new, bty_r)
        crossed_ssp_range, ssp_range_frac = crossed_profile_range_interface_2d(r, r_new, ssp_r_arr)
        crossed_range = 0
        range_frac = 2.0
        range_event_is_ssp = 0
        if crossed_ati_range == 1:
            crossed_range = 1
            range_frac = ati_range_frac
            range_event_is_ssp = 0
        if crossed_bty_range == 1 and bty_range_frac < range_frac:
            crossed_range = 1
            range_frac = bty_range_frac
            range_event_is_ssp = 0
        if crossed_ssp_range == 1 and ssp_range_frac < range_frac:
            crossed_range = 1
            range_frac = ssp_range_frac
            range_event_is_ssp = 1

        hit, next_ati_idx, next_bty_idx = check_boundary_profiles(
            r, z, r_new, z_new,
            ati_r, ati_z, ati_idx,
            bty_r, bty_z, bty_idx,
        )

        hit_frac = 2.0
        if hit != 0:
            if hit == 1:
                r_hit, z_hit, hit_frac = compute_profile_intersection(r, z, r_new, z_new, ati_r, ati_z, next_ati_idx)
            else:
                r_hit, z_hit, hit_frac = compute_profile_intersection(r, z, r_new, z_new, bty_r, bty_z, next_bty_idx)

        if (
            (ssp_type == 0 or ssp_type == 2 or ssp_type == 3)
            and crossed_interface == 1
            and interface_frac <= hit_frac + EVENT_TOL
            and interface_frac <= range_frac + EVENT_TOL
        ):
            r_int = r + interface_frac * (r_new - r)
            z_int = z + interface_frac * (z_new - z)
            xi_int = xi + interface_frac * (xi_new - xi)
            zeta_int = zeta + interface_frac * (zeta_new - zeta)
            tau_int = tau + interface_frac * (tau_new - tau)

            r_path[n_points] = r_int
            z_path[n_points] = z_int
            amp_path[n_points] = amp
            n_points += 1

            r = r_int
            z = z_int
            xi = xi_int
            zeta = zeta_int
            tau = tau_int
        elif crossed_range == 1 and range_frac <= hit_frac + EVENT_TOL:
            r_int = r + range_frac * (r_new - r)
            z_int = z + range_frac * (z_new - z)
            xi_int = xi + range_frac * (xi_new - xi)
            zeta_int = zeta + range_frac * (zeta_new - zeta)
            tau_int = tau + range_frac * (tau_new - tau)

            r_path[n_points] = r_int
            z_path[n_points] = z_int
            amp_path[n_points] = amp
            n_points += 1

            r = r_int
            z = z_int
            xi = xi_int
            zeta = zeta_int
            tau = tau_int
        elif hit != 0:
            ati_idx = next_ati_idx
            bty_idx = next_bty_idx
            xi_hit = xi + hit_frac * (xi_new - xi)
            zeta_hit = zeta + hit_frac * (zeta_new - zeta)
            tau_hit = tau + hit_frac * (tau_new - tau)

            r_path[n_points] = r_hit
            z_path[n_points] = z_hit
            amp_path[n_points] = amp
            n_points += 1

            c_bnd, _, _, _ = evaluate_step_ssp(
                ssp_type, r_hit, z_hit, z_arr, c_arr, c_coeffs, ssp_r_arr
            )

            if hit == -1:
                t_r, t_z, n_r, n_z, bty_slope = get_profile_reflection_vectors(
                    bty_r, bty_z, bty_idx, r_hit, z_hit, hit, bty_curvilinear
                )
                refl_c = bot_c
                refl_c_s = bot_c_s
                refl_rho = bot_rho
                refl_alpha = bot_alpha
                refl_alpha_s = bot_alpha_s
                if len(bty_geo_cp) > 0:
                    media_idx = bty_idx
                    if media_idx >= len(bty_geo_cp):
                        media_idx = len(bty_geo_cp) - 1
                    refl_c = bty_geo_cp[media_idx]
                    refl_c_s = bty_geo_cs[media_idx]
                    refl_rho = bty_geo_rho[media_idx]
                    refl_alpha = bty_geo_alpha[media_idx]
                    refl_alpha_s = bty_geo_alpha_s[media_idx]
            else:
                t_r, t_z, n_r, n_z, bty_slope = get_profile_reflection_vectors(
                    ati_r, ati_z, ati_idx, r_hit, z_hit, hit, ati_curvilinear
                )
                refl_c = surf_c
                refl_c_s = surf_c_s
                refl_rho = surf_rho
                refl_alpha = surf_alpha
                refl_alpha_s = surf_alpha_s
                if len(ati_geo_cp) > 0:
                    media_idx = ati_idx
                    if media_idx >= len(ati_geo_cp):
                        media_idx = len(ati_geo_cp) - 1
                    refl_c = ati_geo_cp[media_idx]
                    refl_c_s = ati_geo_cs[media_idx]
                    refl_rho = ati_geo_rho[media_idx]
                    refl_alpha = ati_geo_alpha[media_idx]
                    refl_alpha_s = ati_geo_alpha_s[media_idx]

            xi_new, zeta_new, R = reflect_ray_amp_vectors(
                xi_hit, zeta_hit, hit, t_r, t_z, n_r, n_z,
                1.0, c_bnd, refl_rho, refl_c, refl_c_s, refl_alpha, refl_alpha_s,
                surface_refl_angles, surface_refl_amp, surface_refl_phase,
                bottom_refl_angles, bottom_refl_amp, bottom_refl_phase,
            )
            amp = amp * R

            r = r_hit
            z = z_hit
            xi = xi_new
            zeta = zeta_new
            tau = tau_hit
            n_bounces += 1
        elif (ssp_type == 0 or ssp_type == 2 or ssp_type == 3):
            ati_idx = next_ati_idx
            bty_idx = next_bty_idx
            r_path[n_points] = r_new
            z_path[n_points] = z_new
            amp_path[n_points] = amp
            n_points += 1

            r = r_new
            z = z_new
            xi = xi_new
            zeta = zeta_new
            tau = tau_new
        else:
            ati_idx = next_ati_idx
            bty_idx = next_bty_idx
            r_path[n_points] = r_new
            z_path[n_points] = z_new
            amp_path[n_points] = amp
            n_points += 1

            r = r_new
            z = z_new
            xi = xi_new
            zeta = zeta_new
            tau = tau_new

        if r > r_max:
            break
        if n_points >= max_steps:
            break

    return r_path, z_path, amp_path, n_points, n_bounces

@njit(fastmath=True)
def trace_beam_ray(
    r_0: float, z_0: float, alpha_deg: float,
    h: float,
    ati_r: np.ndarray, ati_z: np.ndarray,
    ati_curvilinear: int,
    bty_r: np.ndarray, bty_z: np.ndarray,
    bty_curvilinear: int,
    bty_geo_cp: np.ndarray, bty_geo_cs: np.ndarray, bty_geo_rho: np.ndarray,
    bty_geo_alpha: np.ndarray, bty_geo_alpha_s: np.ndarray,
    ati_geo_cp: np.ndarray, ati_geo_cs: np.ndarray, ati_geo_rho: np.ndarray,
    ati_geo_alpha: np.ndarray, ati_geo_alpha_s: np.ndarray,
    surface_refl_angles: np.ndarray, surface_refl_amp: np.ndarray, surface_refl_phase: np.ndarray,
    bottom_refl_angles: np.ndarray, bottom_refl_amp: np.ndarray, bottom_refl_phase: np.ndarray,
    surf_rho: float, surf_c: float, surf_c_s: float, surf_alpha: float, surf_alpha_s: float,
    bot_rho: float, bot_c: float, bot_c_s: float, bot_alpha: float, bot_alpha_s: float,
    frequency: float,
    beam_displacement: int,
    r_max: float,
    ssp_type: int, z_arr: np.ndarray, c_arr: np.ndarray, c_coeffs: np.ndarray,
    ssp_r_arr: np.ndarray,
    max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    r_path = np.zeros(max_steps + 1)
    z_path = np.zeros(max_steps + 1)
    c_path = np.zeros(max_steps + 1)
    p_path = np.zeros(max_steps + 1)
    q_path = np.zeros(max_steps + 1)
    tau_path = np.zeros(max_steps + 1)
    amp_path = np.zeros(max_steps + 1, dtype=np.complex128)
    phase_path = np.zeros(max_steps + 1)
    top_bnc_path = np.zeros(max_steps + 1, dtype=np.int64)
    bot_bnc_path = np.zeros(max_steps + 1, dtype=np.int64)

    c_0, _, dc_dz_0, d2c_dz2_0 = evaluate_step_ssp(
        ssp_type, r_0, z_0, z_arr, c_arr, c_coeffs, ssp_r_arr
    )

    alpha_rad = np.radians(alpha_deg)
    r = r_0
    z = z_0
    xi = np.cos(alpha_rad) / c_0
    zeta = np.sin(alpha_rad) / c_0
    tau = 0.0

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
    phase_path[0] = 0.0
    top_bnc_path[0] = 0
    bot_bnc_path[0] = 0

    n_points = 1
    n_bounces = 0
    n_top_bounces = 0
    n_bottom_bounces = 0
    ati_idx = 0
    bty_idx = 0

    for step_i in range(max_steps):
        r_new, z_new, xi_new, zeta_new, tau_new, p_new, q_new, c_cur = step2d_kinematic_dynamic(
            h, r, z, xi, zeta, tau, p, q, ssp_type, z_arr, c_arr, c_coeffs, ssp_r_arr
        )

        crossed_interface, interface_frac = crossed_interior_z_interface_2d(z, z_new, z_arr)
        crossed_ati_range, ati_range_frac = crossed_profile_range_interface_2d(r, r_new, ati_r)
        crossed_bty_range, bty_range_frac = crossed_profile_range_interface_2d(r, r_new, bty_r)
        crossed_ssp_range, ssp_range_frac = crossed_profile_range_interface_2d(r, r_new, ssp_r_arr)
        crossed_range = 0
        range_frac = 2.0
        range_event_is_ssp = 0
        if crossed_ati_range == 1:
            crossed_range = 1
            range_frac = ati_range_frac
            range_event_is_ssp = 0
        if crossed_bty_range == 1 and bty_range_frac < range_frac:
            crossed_range = 1
            range_frac = bty_range_frac
            range_event_is_ssp = 0
        if crossed_ssp_range == 1 and ssp_range_frac < range_frac:
            crossed_range = 1
            range_frac = ssp_range_frac
            range_event_is_ssp = 1

        hit, next_ati_idx, next_bty_idx = check_boundary_profiles(
            r, z, r_new, z_new,
            ati_r, ati_z, ati_idx,
            bty_r, bty_z, bty_idx,
        )

        hit_frac = 2.0
        if hit != 0:
            if hit == 1:
                r_hit, z_hit, hit_frac = compute_profile_intersection(r, z, r_new, z_new, ati_r, ati_z, next_ati_idx)
            else:
                r_hit, z_hit, hit_frac = compute_profile_intersection(r, z, r_new, z_new, bty_r, bty_z, next_bty_idx)

        if (
            (ssp_type == 0 or ssp_type == 2 or ssp_type == 3)
            and crossed_interface == 1
            and interface_frac <= hit_frac + EVENT_TOL
            and interface_frac <= range_frac + EVENT_TOL
        ):
            r_int = r + interface_frac * (r_new - r)
            z_int = z + interface_frac * (z_new - z)
            xi_int = xi + interface_frac * (xi_new - xi)
            zeta_int = zeta + interface_frac * (zeta_new - zeta)
            tau_int = tau + interface_frac * (tau_new - tau)
            p_int = p + interface_frac * (p_new - p)
            q_int = q + interface_frac * (q_new - q)

            c_int, _, _, _ = evaluate_step_ssp(
                ssp_type, r_int, z_int, z_arr, c_arr, c_coeffs, ssp_r_arr
            )
            _, _, dc_dz_old, _ = evaluate_step_ssp(
                ssp_type, r, z, z_arr, c_arr, c_coeffs, ssp_r_arr
            )
            _, _, dc_dz_new, _ = evaluate_step_ssp(
                ssp_type, r_new, z_new, z_arr, c_arr, c_coeffs, ssp_r_arr
            )
            p_int = apply_ssp_interface_curvature_correction_2d(
                p_int,
                q_int,
                xi_int,
                zeta_int,
                c_int,
                dc_dz_old,
                dc_dz_new,
            )

            r_path[n_points] = r_int
            z_path[n_points] = z_int
            c_path[n_points] = c_int
            p_path[n_points] = p_int
            q_path[n_points] = q_int
            tau_path[n_points] = tau_int
            amp_path[n_points] = amp
            phase_path[n_points] = phase_path[n_points - 1]
            top_bnc_path[n_points] = n_top_bounces
            bot_bnc_path[n_points] = n_bottom_bounces
            n_points += 1

            r = r_int
            z = z_int
            xi = xi_int
            zeta = zeta_int
            tau = tau_int
            p = p_int
            q = q_int
        elif crossed_range == 1 and range_frac <= hit_frac + EVENT_TOL:
            r_int = r + range_frac * (r_new - r)
            z_int = z + range_frac * (z_new - z)
            xi_int = xi + range_frac * (xi_new - xi)
            zeta_int = zeta + range_frac * (zeta_new - zeta)
            tau_int = tau + range_frac * (tau_new - tau)
            p_int = p + range_frac * (p_new - p)
            q_int = q + range_frac * (q_new - q)

            c_int, _, _, _ = evaluate_step_ssp(
                ssp_type, r_int, z_int, z_arr, c_arr, c_coeffs, ssp_r_arr
            )
            if ssp_type == 3 and range_event_is_ssp == 1:
                _, dc_dr_old, _, _ = evaluate_step_ssp(
                    ssp_type, r, z, z_arr, c_arr, c_coeffs, ssp_r_arr
                )
                _, dc_dr_new, _, _ = evaluate_step_ssp(
                    ssp_type, r_new, z_new, z_arr, c_arr, c_coeffs, ssp_r_arr
                )
                p_int = apply_ssp_range_interface_curvature_correction_2d(
                    p_int,
                    q_int,
                    xi_int,
                    zeta_int,
                    c_int,
                    dc_dr_old,
                    dc_dr_new,
                )

            r_path[n_points] = r_int
            z_path[n_points] = z_int
            c_path[n_points] = c_int
            p_path[n_points] = p_int
            q_path[n_points] = q_int
            tau_path[n_points] = tau_int
            amp_path[n_points] = amp
            phase_path[n_points] = phase_path[n_points - 1]
            top_bnc_path[n_points] = n_top_bounces
            bot_bnc_path[n_points] = n_bottom_bounces
            n_points += 1

            r = r_int
            z = z_int
            xi = xi_int
            zeta = zeta_int
            tau = tau_int
            p = p_int
            q = q_int
        elif hit != 0:
            ati_idx = next_ati_idx
            bty_idx = next_bty_idx
            xi_hit = xi + hit_frac * (xi_new - xi)
            zeta_hit = zeta + hit_frac * (zeta_new - zeta)
            tau_hit = tau + hit_frac * (tau_new - tau)
            p_hit = p + hit_frac * (p_new - p)
            q_hit = q + hit_frac * (q_new - q)

            r_path[n_points] = r_hit
            z_path[n_points] = z_hit

            c_bnd, _, dc_dz_bnd, _ = evaluate_step_ssp(
                ssp_type, r_hit, z_hit, z_arr, c_arr, c_coeffs, ssp_r_arr
            )

            c_path[n_points] = c_bnd

            p_path[n_points] = p_hit
            q_path[n_points] = q_hit
            tau_path[n_points] = tau_hit
            amp_path[n_points] = amp
            phase_path[n_points] = phase_path[n_points - 1]
            if hit == 1:
                top_bnc_path[n_points] = n_top_bounces + 1
                bot_bnc_path[n_points] = n_bottom_bounces
            else:
                top_bnc_path[n_points] = n_top_bounces
                bot_bnc_path[n_points] = n_bottom_bounces + 1
            n_points += 1

            if hit == -1:
                t_r, t_z, n_r, n_z, bty_slope = get_profile_reflection_vectors(
                    bty_r, bty_z, bty_idx, r_hit, z_hit, hit, bty_curvilinear
                )
                kappa = get_profile_curvature(bty_r, bty_z, bty_idx, bty_curvilinear)
                refl_c = bot_c
                refl_c_s = bot_c_s
                refl_rho = bot_rho
                refl_alpha = bot_alpha
                refl_alpha_s = bot_alpha_s
                if len(bty_geo_cp) > 0:
                    media_idx = bty_idx
                    if media_idx >= len(bty_geo_cp):
                        media_idx = len(bty_geo_cp) - 1
                    refl_c = bty_geo_cp[media_idx]
                    refl_c_s = bty_geo_cs[media_idx]
                    refl_rho = bty_geo_rho[media_idx]
                    refl_alpha = bty_geo_alpha[media_idx]
                    refl_alpha_s = bty_geo_alpha_s[media_idx]
            else:
                t_r, t_z, n_r, n_z, bty_slope = get_profile_reflection_vectors(
                    ati_r, ati_z, ati_idx, r_hit, z_hit, hit, ati_curvilinear
                )
                kappa = get_profile_curvature(ati_r, ati_z, ati_idx, ati_curvilinear)
                refl_c = surf_c
                refl_c_s = surf_c_s
                refl_rho = surf_rho
                refl_alpha = surf_alpha
                refl_alpha_s = surf_alpha_s
                if len(ati_geo_cp) > 0:
                    media_idx = ati_idx
                    if media_idx >= len(ati_geo_cp):
                        media_idx = len(ati_geo_cp) - 1
                    refl_c = ati_geo_cp[media_idx]
                    refl_c_s = ati_geo_cs[media_idx]
                    refl_rho = ati_geo_rho[media_idx]
                    refl_alpha = ati_geo_alpha[media_idx]
                    refl_alpha_s = ati_geo_alpha_s[media_idx]

            xi_new, zeta_new, R = reflect_ray_amp_vectors(
                xi_hit, zeta_hit, hit, t_r, t_z, n_r, n_z,
                1.0, c_bnd, refl_rho, refl_c, refl_c_s, refl_alpha, refl_alpha_s,
                surface_refl_angles, surface_refl_amp, surface_refl_phase,
                bottom_refl_angles, bottom_refl_amp, bottom_refl_phase,
            )
            p_reflected = apply_reflection_curvature_correction_2d(
                p_hit,
                q_hit,
                xi_hit,
                zeta_hit,
                xi_new,
                zeta_new,
                c_bnd,
                dc_dz_bnd,
                hit,
                t_r,
                t_z,
                n_r,
                n_z,
                kappa,
            )
            q_reflected = q_hit
            r_reflected = r_hit
            z_reflected = z_hit
            tau_reflected = tau_hit
            if (
                hit == -1
                and beam_displacement == 1
                and len(bottom_refl_angles) == 0
            ):
                r_reflected, z_reflected, tau_reflected, q_reflected = apply_seongil_beam_displacement_2d(
                    r_hit,
                    z_hit,
                    tau_hit,
                    q_hit,
                    p_reflected,
                    xi_hit,
                    zeta_hit,
                    c_bnd,
                    frequency,
                    t_r,
                    t_z,
                    refl_rho,
                    refl_c,
                    refl_alpha,
                )
                r_path[n_points - 1] = r_reflected
                z_path[n_points - 1] = z_reflected
                tau_path[n_points - 1] = tau_reflected
                q_path[n_points - 1] = q_reflected
            amp = amp * R
            phase_path[n_points - 1] = phase_path[n_points - 1] + np.angle(R)
            if hit == 1:
                n_top_bounces += 1
            else:
                n_bottom_bounces += 1

            r = r_reflected
            z = z_reflected
            xi = xi_new
            zeta = zeta_new
            tau = tau_reflected
            p = p_reflected
            q = q_reflected
            n_bounces += 1
        elif (ssp_type == 0 or ssp_type == 2 or ssp_type == 3):
            ati_idx = next_ati_idx
            bty_idx = next_bty_idx
            r_path[n_points] = r_new
            z_path[n_points] = z_new

            c_new, _, _, _ = evaluate_step_ssp(
                ssp_type, r_new, z_new, z_arr, c_arr, c_coeffs, ssp_r_arr
            )

            c_path[n_points] = c_new
            p_path[n_points] = p_new
            q_path[n_points] = q_new
            tau_path[n_points] = tau_new
            amp_path[n_points] = amp
            phase_path[n_points] = phase_path[n_points - 1]
            top_bnc_path[n_points] = n_top_bounces
            bot_bnc_path[n_points] = n_bottom_bounces
            n_points += 1

            r = r_new
            z = z_new
            xi = xi_new
            zeta = zeta_new
            tau = tau_new
            p = p_new
            q = q_new
        else:
            ati_idx = next_ati_idx
            bty_idx = next_bty_idx
            r_path[n_points] = r_new
            z_path[n_points] = z_new

            c_new, _, _, _ = evaluate_step_ssp(
                ssp_type, r_new, z_new, z_arr, c_arr, c_coeffs, ssp_r_arr
            )

            c_path[n_points] = c_new
            p_path[n_points] = p_new
            q_path[n_points] = q_new
            tau_path[n_points] = tau_new
            amp_path[n_points] = amp
            phase_path[n_points] = phase_path[n_points - 1]
            top_bnc_path[n_points] = n_top_bounces
            bot_bnc_path[n_points] = n_bottom_bounces
            n_points += 1

            r = r_new
            z = z_new
            xi = xi_new
            zeta = zeta_new
            tau = tau_new
            p = p_new
            q = q_new

        if r > r_max:
            break
        if n_points >= max_steps:
            break

    return (
        r_path, z_path, c_path, tau_path, p_path, q_path,
        amp_path, phase_path, top_bnc_path, bot_bnc_path,
        n_points, n_bounces,
    )
