import numpy as np
from numba import njit

@njit(fastmath=True)
def check_boundary(
    r_old: float, z_old: float,
    r_new: float, z_new: float,
    z_surface: float,
    bty_r: np.ndarray, bty_z: np.ndarray, bty_idx: int
) -> tuple[int, int]:
    """
    Checks if a ray step crossed a boundary.
    Returns:
        hit_type: 0 no hit, 1 surface, -1 bottom
        new_bty_idx: updated bathymetry index
    """
    if z_new <= z_surface:
        return 1, bty_idx

    # Update bty_idx to match r_new
    n_bty = len(bty_r)
    while bty_idx < n_bty - 2 and r_new >= bty_r[bty_idx + 1]:
        bty_idx += 1

    if r_new <= bty_r[-1]:
        dr = bty_r[bty_idx+1] - bty_r[bty_idx]
        if dr > 1e-12:
            slope = (bty_z[bty_idx+1] - bty_z[bty_idx]) / dr
            z_bottom_local = bty_z[bty_idx] + slope * (r_new - bty_r[bty_idx])
        else:
            z_bottom_local = bty_z[bty_idx]
    else:
        z_bottom_local = bty_z[-1]

    if z_new >= z_bottom_local:
        return -1, bty_idx

    return 0, bty_idx


@njit(fastmath=True)
def _advance_profile_index(profile_r: np.ndarray, idx: int, r_value: float) -> int:
    n_profile = len(profile_r)
    while idx < n_profile - 2 and r_value >= profile_r[idx + 1]:
        idx += 1
    return idx


@njit(fastmath=True)
def _profile_value_at(profile_r: np.ndarray, profile_z: np.ndarray, idx: int, r_value: float) -> float:
    if r_value <= profile_r[-1]:
        dr = profile_r[idx + 1] - profile_r[idx]
        if dr > 1.0e-12:
            slope = (profile_z[idx + 1] - profile_z[idx]) / dr
            return profile_z[idx] + slope * (r_value - profile_r[idx])
        return profile_z[idx]
    return profile_z[-1]


@njit(fastmath=True)
def check_boundary_profiles(
    r_old: float, z_old: float,
    r_new: float, z_new: float,
    ati_r: np.ndarray, ati_z: np.ndarray, ati_idx: int,
    bty_r: np.ndarray, bty_z: np.ndarray, bty_idx: int
) -> tuple[int, int, int]:
    """
    Checks if a ray step crossed range-dependent surface or bottom profiles.
    Returns hit_type, updated ati_idx, updated bty_idx.
    """
    ati_idx = _advance_profile_index(ati_r, ati_idx, r_new)
    z_surface_local = _profile_value_at(ati_r, ati_z, ati_idx, r_new)
    if z_new <= z_surface_local:
        return 1, ati_idx, bty_idx

    bty_idx = _advance_profile_index(bty_r, bty_idx, r_new)
    z_bottom_local = _profile_value_at(bty_r, bty_z, bty_idx, r_new)
    if z_new >= z_bottom_local:
        return -1, ati_idx, bty_idx

    return 0, ati_idx, bty_idx


@njit(fastmath=True)
def crossed_interior_z_interface_2d(z_old: float, z_new: float, z_grid: np.ndarray) -> tuple[int, float]:
    """Return whether a 2D step crossed an interior SSP depth interface."""
    if len(z_grid) <= 2:
        return 0, 0.0
    z_min = min(z_old, z_new)
    z_max = max(z_old, z_new)
    for iz in range(1, len(z_grid) - 1):
        z_int = z_grid[iz]
        if z_int > z_min + 1.0e-10 and z_int < z_max - 1.0e-10:
            dz = z_new - z_old
            if abs(dz) <= 1.0e-30:
                return 0, 0.0
            frac = (z_int - z_old) / dz
            if frac > 1.0e-10 and frac < 1.0 - 1.0e-10:
                return 1, frac
    return 0, 0.0


@njit(fastmath=True)
def crossed_profile_range_interface_2d(r_old: float, r_new: float, r_grid: np.ndarray) -> tuple[int, float]:
    """Return whether a 2D step crossed an interior range-dependent profile node."""
    if len(r_grid) <= 2:
        return 0, 0.0
    r_min = min(r_old, r_new)
    r_max = max(r_old, r_new)
    for ir in range(1, len(r_grid) - 1):
        r_int = r_grid[ir]
        if r_int > r_min + 1.0e-10 and r_int < r_max - 1.0e-10:
            dr = r_new - r_old
            if abs(dr) <= 1.0e-30:
                return 0, 0.0
            frac = (r_int - r_old) / dr
            if frac > 1.0e-10 and frac < 1.0 - 1.0e-10:
                return 1, frac
    return 0, 0.0


@njit(fastmath=True)
def apply_ssp_interface_curvature_correction_2d(
    p: float,
    q: float,
    xi: float,
    zeta: float,
    c_interface: float,
    dc_dz_old: float,
    dc_dz_new: float,
) -> float:
    """Apply AT Step2D gradient-jump correction at a z SSP interface."""
    if abs(zeta) <= 1.0e-12 or abs(c_interface) <= 1.0e-30:
        return p
    grad_jump_z = dc_dz_new - dc_dz_old
    cnjump = grad_jump_z * xi
    csjump = grad_jump_z * zeta
    rm = xi / zeta
    rn = rm * (2.0 * cnjump - rm * csjump) / c_interface
    return p - q * rn


@njit(fastmath=True)
def apply_ssp_range_interface_curvature_correction_2d(
    p: float,
    q: float,
    xi: float,
    zeta: float,
    c_interface: float,
    dc_dr_old: float,
    dc_dr_new: float,
) -> float:
    """Apply the Step2D gradient-jump correction at a range SSP interface."""
    if abs(xi) <= 1.0e-12 or abs(c_interface) <= 1.0e-30:
        return p
    grad_jump_r = dc_dr_new - dc_dr_old
    cnjump = grad_jump_r * zeta
    csjump = grad_jump_r * xi
    rm = zeta / xi
    rn = rm * (2.0 * cnjump - rm * csjump) / c_interface
    return p - q * rn


@njit(fastmath=True)
def compute_boundary_intersection(
    r_old: float, z_old: float,
    r_new: float, z_new: float,
    z_surface: float, hit_type: int,
    bty_r: np.ndarray, bty_z: np.ndarray, bty_idx: int
) -> tuple[float, float, float]:
    """
    Returns: (r_hit, z_hit, fraction)
    """
    if hit_type == 1:
        dz = z_new - z_old
        if abs(dz) < 1e-30:
            return r_old, z_surface, 0.0
        fraction = (z_surface - z_old) / dz
        r_hit = r_old + fraction * (r_new - r_old)
        return r_hit, z_surface, fraction
    else:
        dr_ray = r_new - r_old
        dz_ray = z_new - z_old

        if bty_idx >= len(bty_r) - 1:
            z_bnd = bty_z[-1]
            if abs(dz_ray) < 1e-30:
                return r_old, z_bnd, 0.0
            fraction = (z_bnd - z_old) / dz_ray
            r_hit = r_old + fraction * dr_ray
            return r_hit, z_bnd, fraction

        dr_bty = bty_r[bty_idx+1] - bty_r[bty_idx]
        dz_bty = bty_z[bty_idx+1] - bty_z[bty_idx]

        den = dr_ray * dz_bty - dz_ray * dr_bty
        if abs(den) < 1e-30:
            return r_old, bty_z[bty_idx], 0.0

        t1 = ((r_old - bty_r[bty_idx]) * dz_bty - (z_old - bty_z[bty_idx]) * dr_bty) / den
        r_hit = r_old + t1 * dr_ray
        z_hit = z_old + t1 * dz_ray
        return r_hit, z_hit, t1


@njit(fastmath=True)
def compute_profile_intersection(
    r_old: float, z_old: float,
    r_new: float, z_new: float,
    profile_r: np.ndarray, profile_z: np.ndarray, profile_idx: int
) -> tuple[float, float, float]:
    """Return ray intersection with a range-dependent boundary profile."""
    dr_ray = r_new - r_old
    dz_ray = z_new - z_old

    if profile_idx >= len(profile_r) - 1:
        z_bnd = profile_z[-1]
        if abs(dz_ray) < 1.0e-30:
            return r_old, z_bnd, 0.0
        fraction = (z_bnd - z_old) / dz_ray
        r_hit = r_old + fraction * dr_ray
        return r_hit, z_bnd, fraction

    dr_profile = profile_r[profile_idx + 1] - profile_r[profile_idx]
    dz_profile = profile_z[profile_idx + 1] - profile_z[profile_idx]

    den = dr_ray * dz_profile - dz_ray * dr_profile
    if abs(den) < 1.0e-30:
        return r_old, profile_z[profile_idx], 0.0

    t1 = (
        (profile_r[profile_idx] - r_old) * dz_profile
        - (profile_z[profile_idx] - z_old) * dr_profile
    ) / den
    r_hit = r_old + t1 * dr_ray
    z_hit = z_old + t1 * dz_ray
    return r_hit, z_hit, t1

@njit(fastmath=True)
def get_bottom_slope(bty_r: np.ndarray, bty_z: np.ndarray, bty_idx: int) -> float:
    """Returns local bottom slope angle (radians)."""
    if bty_idx >= len(bty_r) - 1:
        return 0.0
    dr = bty_r[bty_idx+1] - bty_r[bty_idx]
    dz = bty_z[bty_idx+1] - bty_z[bty_idx]
    if abs(dr) < 1e-12:
        return 0.0
    return np.arctan2(dz, dr)


@njit(fastmath=True)
def get_profile_slope(profile_r: np.ndarray, profile_z: np.ndarray, profile_idx: int) -> float:
    """Returns local boundary slope angle (radians)."""
    if profile_idx >= len(profile_r) - 1:
        return 0.0
    dr = profile_r[profile_idx + 1] - profile_r[profile_idx]
    dz = profile_z[profile_idx + 1] - profile_z[profile_idx]
    if abs(dr) < 1.0e-12:
        return 0.0
    return np.arctan2(dz, dr)


@njit(fastmath=True)
def _profile_segment_tangent(profile_r: np.ndarray, profile_z: np.ndarray, segment_idx: int) -> tuple[float, float]:
    if segment_idx < 0 or segment_idx >= len(profile_r) - 1:
        return 1.0, 0.0
    dr = profile_r[segment_idx + 1] - profile_r[segment_idx]
    dz = profile_z[segment_idx + 1] - profile_z[segment_idx]
    seg_len = np.sqrt(dr * dr + dz * dz)
    if seg_len <= 1.0e-30:
        return 1.0, 0.0
    return dr / seg_len, dz / seg_len


@njit(fastmath=True)
def _profile_node_tangent(profile_r: np.ndarray, profile_z: np.ndarray, node_idx: int) -> tuple[float, float]:
    if node_idx <= 0:
        t_prev_r, t_prev_z = 1.0, 0.0
        t_next_r, t_next_z = _profile_segment_tangent(profile_r, profile_z, 0)
    elif node_idx >= len(profile_r) - 1:
        t_prev_r, t_prev_z = _profile_segment_tangent(profile_r, profile_z, len(profile_r) - 2)
        t_next_r, t_next_z = 1.0, 0.0
    else:
        t_prev_r, t_prev_z = _profile_segment_tangent(profile_r, profile_z, node_idx - 1)
        t_next_r, t_next_z = _profile_segment_tangent(profile_r, profile_z, node_idx)
    return 0.5 * (t_prev_r + t_next_r), 0.5 * (t_prev_z + t_next_z)


@njit(fastmath=True)
def get_profile_reflection_slope(
    profile_r: np.ndarray,
    profile_z: np.ndarray,
    profile_idx: int,
    r_eval: float,
    z_eval: float,
    curvilinear: int,
) -> float:
    """Return AT-style reflection tangent angle for L or C boundary profiles."""
    if curvilinear == 0 or profile_idx >= len(profile_r) - 1:
        return get_profile_slope(profile_r, profile_z, profile_idx)

    dr = profile_r[profile_idx + 1] - profile_r[profile_idx]
    dz = profile_z[profile_idx + 1] - profile_z[profile_idx]
    seg_len2 = dr * dr + dz * dz
    if seg_len2 <= 1.0e-30:
        s = 0.0
    else:
        s = ((r_eval - profile_r[profile_idx]) * dr + (z_eval - profile_z[profile_idx]) * dz) / seg_len2

    t_left_r, t_left_z = _profile_node_tangent(profile_r, profile_z, profile_idx)
    t_right_r, t_right_z = _profile_node_tangent(profile_r, profile_z, profile_idx + 1)
    t_r = (1.0 - s) * t_left_r + s * t_right_r
    t_z = (1.0 - s) * t_left_z + s * t_right_z
    if abs(t_r) < 1.0e-30 and abs(t_z) < 1.0e-30:
        return get_profile_slope(profile_r, profile_z, profile_idx)
    return np.arctan2(t_z, t_r)


@njit(fastmath=True)
def get_profile_reflection_vectors(
    profile_r: np.ndarray,
    profile_z: np.ndarray,
    profile_idx: int,
    r_eval: float,
    z_eval: float,
    hit_type: int,
    curvilinear: int,
) -> tuple[float, float, float, float, float]:
    """Return AT-style boundary tangent, normal, and tangent angle."""
    if curvilinear == 0 or profile_idx >= len(profile_r) - 1:
        t_r, t_z = _profile_segment_tangent(profile_r, profile_z, profile_idx)
    else:
        dr = profile_r[profile_idx + 1] - profile_r[profile_idx]
        dz = profile_z[profile_idx + 1] - profile_z[profile_idx]
        seg_len2 = dr * dr + dz * dz
        if seg_len2 <= 1.0e-30:
            s = 0.0
        else:
            s = ((r_eval - profile_r[profile_idx]) * dr + (z_eval - profile_z[profile_idx]) * dz) / seg_len2

        t_left_r, t_left_z = _profile_node_tangent(profile_r, profile_z, profile_idx)
        t_right_r, t_right_z = _profile_node_tangent(profile_r, profile_z, profile_idx + 1)
        t_r = (1.0 - s) * t_left_r + s * t_right_r
        t_z = (1.0 - s) * t_left_z + s * t_right_z
        if abs(t_r) < 1.0e-30 and abs(t_z) < 1.0e-30:
            t_r, t_z = _profile_segment_tangent(profile_r, profile_z, profile_idx)

    if hit_type == 1:
        n_r = t_z
        n_z = -t_r
    else:
        n_r = -t_z
        n_z = t_r
    return t_r, t_z, n_r, n_z, np.arctan2(t_z, t_r)


@njit(fastmath=True)
def get_profile_curvature(
    profile_r: np.ndarray,
    profile_z: np.ndarray,
    profile_idx: int,
    curvilinear: int,
) -> float:
    """Return AT's curvilinear Dss curvature approximation for a profile segment."""
    if curvilinear == 0 or profile_idx >= len(profile_r) - 1:
        return 0.0

    dr = profile_r[profile_idx + 1] - profile_r[profile_idx]
    dz = profile_z[profile_idx + 1] - profile_z[profile_idx]
    if abs(dr) < 1.0e-12:
        return 0.0

    dx_cur = dz / dr
    if profile_idx + 1 < len(profile_r) - 1:
        dr_next = profile_r[profile_idx + 2] - profile_r[profile_idx + 1]
        dz_next = profile_z[profile_idx + 2] - profile_z[profile_idx + 1]
        if abs(dr_next) < 1.0e-12:
            dx_next = 0.0
        else:
            dx_next = dz_next / dr_next
    else:
        dx_next = 0.0

    dxx = (dx_next - dx_cur) / dr
    seg_len = np.sqrt(dr * dr + dz * dz)
    if seg_len <= 1.0e-30:
        return 0.0
    t_r = dr / seg_len
    return dxx * t_r * t_r * t_r

@njit(fastmath=True)
def reflect_ray(
    xi: float, zeta: float,
    hit_type: int, bty_slope: float
) -> tuple[float, float]:
    """
    Specular reflection off boundary.
    Returns: (new_xi, new_zeta)
    """
    if hit_type == 1:
        n_r = np.sin(bty_slope)
        n_z = -np.cos(bty_slope)
    else:
        # Bottom reflection with slope beta
        n_r = -np.sin(bty_slope)
        n_z = np.cos(bty_slope)

    dot_prod = xi * n_r + zeta * n_z
    new_xi = xi - 2.0 * dot_prod * n_r
    new_zeta = zeta - 2.0 * dot_prod * n_z
    return new_xi, new_zeta

@njit(fastmath=True)
def reflect_ray_amp(
    xi: float, zeta: float,
    hit_type: int, bty_slope: float,
    rho1: float, c1: float,
    rho2: float, c2: float, alpha_dB_lambda: float,
    surface_refl_angles: np.ndarray, surface_refl_amp: np.ndarray, surface_refl_phase: np.ndarray,
    bottom_refl_angles: np.ndarray, bottom_refl_amp: np.ndarray, bottom_refl_phase: np.ndarray
) -> tuple[float, float, complex]:
    """
    Reflects ray and computes reflection coefficient.
    Returns: new_xi, new_zeta, reflection_coeff
    """
    new_xi, new_zeta = reflect_ray(xi, zeta, hit_type, bty_slope)

    ray_angle = np.arctan2(zeta, xi)
    grazing_angle = abs(ray_angle - bty_slope)
    grazing_deg = abs(np.degrees(grazing_angle))
    if grazing_deg > 90.0:
        grazing_deg = 180.0 - grazing_deg

    if hit_type == 1 and len(surface_refl_angles) > 0:
        amp = np.interp(grazing_deg, surface_refl_angles, surface_refl_amp)
        phase = np.interp(grazing_deg, surface_refl_angles, surface_refl_phase)
        return new_xi, new_zeta, amp * np.exp(1j * np.radians(phase))

    if hit_type == -1 and len(bottom_refl_angles) > 0:
        amp = np.interp(grazing_deg, bottom_refl_angles, bottom_refl_amp)
        phase = np.interp(grazing_deg, bottom_refl_angles, bottom_refl_phase)
        return new_xi, new_zeta, amp * np.exp(1j * np.radians(phase))

    if hit_type == 1:
        return new_xi, new_zeta, -1.0 + 0.0j

    if hit_type == -1:
        if c2 < 0.0:
            return new_xi, new_zeta, 1.0 + 0.0j
        if c2 <= 0.0:
            return new_xi, new_zeta, -1.0 + 0.0j

        v_mag = np.sqrt(xi**2 + zeta**2)
        kz1_over_w = np.sin(grazing_angle) * v_mag

        c2_complex = c2 + 1j * c2 * alpha_dB_lambda / (8.6858896 * 2.0 * np.pi)

        radicand = (1.0 / c2_complex)**2 - (v_mag * np.cos(grazing_angle))**2
        kz2_over_w = np.sqrt(radicand)

        num = rho2 * kz1_over_w - rho1 * kz2_over_w
        den = rho2 * kz1_over_w + rho1 * kz2_over_w

        if abs(den) < 1e-30:
            R = 1.0 + 0.0j
        else:
            R = num / den

        return new_xi, new_zeta, R

    return new_xi, new_zeta, 1.0 + 0.0j


@njit(fastmath=True)
def reflect_ray_amp_vectors(
    xi: float, zeta: float,
    hit_type: int,
    t_r: float, t_z: float,
    n_r: float, n_z: float,
    rho1: float, c1: float,
    rho2: float, c2: float, c_s2: float, alpha_dB_lambda: float, alpha_s_dB_lambda: float,
    surface_refl_angles: np.ndarray, surface_refl_amp: np.ndarray, surface_refl_phase: np.ndarray,
    bottom_refl_angles: np.ndarray, bottom_refl_amp: np.ndarray, bottom_refl_phase: np.ndarray
) -> tuple[float, float, complex]:
    """Reflect using AT's tangent/normal vectors, including unnormalized C nodes."""
    tg = xi * t_r + zeta * t_z
    th = xi * n_r + zeta * n_z
    new_xi = xi - 2.0 * th * n_r
    new_zeta = zeta - 2.0 * th * n_z

    grazing_deg = abs(np.degrees(np.arctan2(th, tg)))
    if grazing_deg > 90.0:
        grazing_deg = 180.0 - grazing_deg

    if hit_type == 1 and len(surface_refl_angles) > 0:
        amp = np.interp(grazing_deg, surface_refl_angles, surface_refl_amp)
        phase = np.interp(grazing_deg, surface_refl_angles, surface_refl_phase)
        return new_xi, new_zeta, amp * np.exp(1j * np.radians(phase))

    if hit_type == -1 and len(bottom_refl_angles) > 0:
        amp = np.interp(grazing_deg, bottom_refl_angles, bottom_refl_amp)
        phase = np.interp(grazing_deg, bottom_refl_angles, bottom_refl_phase)
        return new_xi, new_zeta, amp * np.exp(1j * np.radians(phase))

    if hit_type == -1:
        if c2 < 0.0:
            return new_xi, new_zeta, 1.0 + 0.0j
        if c2 <= 0.0:
            return new_xi, new_zeta, -1.0 + 0.0j
    elif hit_type == 1:
        if c2 < 0.0:
            return new_xi, new_zeta, 1.0 + 0.0j
        if c2 <= 0.0:
            return new_xi, new_zeta, -1.0 + 0.0j
    else:
        return new_xi, new_zeta, 1.0 + 0.0j

    omega = 1.0
    i_unit = 1j
    kx = omega * tg
    kz = omega * th

    c2_complex = c2 + i_unit * c2 * alpha_dB_lambda / (8.6858896 * 2.0 * np.pi)

    if c_s2 > 0.0:
        cs_complex = c_s2 + i_unit * c_s2 * alpha_s_dB_lambda / (8.6858896 * 2.0 * np.pi)

        kz_s2 = kx * kx - (omega / cs_complex) ** 2
        kz_p2 = kx * kx - (omega / c2_complex) ** 2
        kz_s = np.sqrt(kz_s2)
        kz_p = np.sqrt(kz_p2)
        if np.real(kz_s) == 0.0 and np.imag(kz_s) < 0.0:
            kz_s = -kz_s
        if np.real(kz_p) == 0.0 and np.imag(kz_p) < 0.0:
            kz_p = -kz_p
        mu = rho2 * cs_complex * cs_complex
        f = omega * omega * kz_p * (kx * kx - kz_s2)
        g = ((kz_s2 + kx * kx) ** 2 - 4.0 * kz_s * kz_p * kx * kx) * mu
    else:
        kz_p = np.sqrt(kx * kx - (omega / c2_complex) ** 2)
        if np.real(kz_p) == 0.0 and np.imag(kz_p) < 0.0:
            kz_p = -kz_p
        f = kz_p
        g = rho2

    den = rho1 * f + i_unit * kz * g
    if abs(den) < 1.0e-30:
        return new_xi, new_zeta, 1.0 + 0.0j
    refl = -((rho1 * f - i_unit * kz * g) / den)
    return new_xi, new_zeta, refl

    return new_xi, new_zeta, 1.0 + 0.0j


@njit(fastmath=True)
def apply_reflection_curvature_correction_2d(
    p: float,
    q: float,
    xi_in: float,
    zeta_in: float,
    xi_out: float,
    zeta_out: float,
    c: float,
    dc_dz: float,
    hit_type: int,
    t_bdry_r: float,
    t_bdry_z: float,
    n_bdry_r: float,
    n_bdry_z: float,
    kappa: float,
) -> float:
    """
    Apply AT Bellhop Reflect2D CurvatureCorrection2 for the scalar q(1)/p(1)
    geometric-beam dynamic variables.
    """
    if abs(q) < 1.0e-30 or c <= 0.0:
        return p

    tg = xi_in * t_bdry_r + zeta_in * t_bdry_z
    th = xi_in * n_bdry_r + zeta_in * n_bdry_z
    if abs(th) < 1.0e-30:
        return p

    urayt_in_r = c * xi_in
    urayt_in_z = c * zeta_in
    urayn_in_r = -urayt_in_z
    urayn_in_z = urayt_in_r

    urayt_out_r = c * xi_out
    urayt_out_z = c * zeta_out
    urayn_out_r = urayt_out_z
    urayn_out_z = -urayt_out_r

    cnjump = -dc_dz * (urayn_out_z - urayn_in_z)
    csjump = -dc_dz * (urayt_out_z - urayt_in_z)

    if hit_type == 1:
        cnjump = -cnjump
        kappa = -kappa

    rm = tg / th
    rn = 2.0 * kappa / (c * c) / th
    rn += rm * (2.0 * cnjump - rm * csjump) / (c * c)
    return p + q * rn


@njit(fastmath=True)
def apply_seongil_beam_displacement_2d(
    r: float,
    z: float,
    tau: float,
    q: float,
    p: float,
    xi_in: float,
    zeta_in: float,
    c: float,
    frequency: float,
    t_bdry_r: float,
    t_bdry_z: float,
    rho2: float,
    c2: float,
    alpha_dB_lambda: float,
) -> tuple[float, float, float, float]:
    """
    Apply AT Bellhop's optional Beam%Type(4:4) == 'S' bottom beam shift.

    This is Seongil's version in ReflectMod.f90: it displaces the reflected
    ray along the boundary tangent, adds a small phase travel-time correction,
    and adjusts scalar q to account for reflection-induced beam-width change.
    """
    if c <= 0.0 or c2 <= 0.0 or frequency <= 0.0:
        return r, z, tau, q

    co = xi_in * c
    si = zeta_in * c
    if abs(si) < 1.0e-30 or abs(co) < 1.0e-30:
        return r, z, tau, q

    c2_complex = c2 + 1j * c2 * alpha_dB_lambda / (8.6858896 * 2.0 * np.pi)
    ch = c / np.conj(c2_complex)
    omega = 2.0 * np.pi * frequency
    ck = omega / c

    a = 2.0 * rho2 * (1.0 - ch * ch)
    b = co * co - ch * ch
    d = rho2 * rho2 * si * si + b
    sb = np.sqrt(b)
    if abs(sb) < 1.0e-30 or abs(d) < 1.0e-30:
        return r, z, tau, q

    delta = a * co / si / (ck * sb * d)
    delta_real = np.real(delta)
    pdelta = delta_real / (c / co)

    cco = co * co
    ssi = si * si
    if abs(ssi) < 1.0e-30 or abs(b) < 1.0e-30:
        return r + delta_real * t_bdry_r, z + delta_real * t_bdry_z, tau + pdelta, q

    ddelta = -a / (ck * sb * d)
    ddelta -= a * cco / ssi / (ck * sb * d)
    ddelta += a * cco / (ck * b * sb * d)
    ddelta -= (
        a
        * co
        / si
        / (ck * sb * d * d)
        * (2.0 * rho2 * rho2 * si * co - 2.0 * co * si)
    )
    rddelta = -np.real(ddelta)
    if abs(rddelta) < 1.0e-30:
        q_shift = 0.0
    else:
        sddelta = rddelta / abs(rddelta)
        q_shift = sddelta * rddelta * si * c * p

    return (
        r + delta_real * t_bdry_r,
        z + delta_real * t_bdry_z,
        tau + pdelta,
        q + q_shift,
    )
