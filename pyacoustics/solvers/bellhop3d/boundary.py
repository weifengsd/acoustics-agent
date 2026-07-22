import numpy as np
from numba import njit

@njit(fastmath=True)
def get_reflection_coef(
    grazing_angle: float,
    rho1: float, c1: float,
    rho2: float, c2: float, alpha_dB_lambda: float
) -> complex:
    if c2 < 0.0:
        return 1.0 + 0.0j
    if c2 <= 0.0:
        return -1.0 + 0.0j

    v_mag = 1.0 / c1
    kz1_over_w = np.sin(grazing_angle) * v_mag

    alpha_nep = alpha_dB_lambda / 8.6858896
    alpha_factor = alpha_nep / (2.0 * np.pi)
    c2_complex = c2 / (1.0 + 1j * alpha_factor)

    radicand = (1.0 / c2_complex)**2 - (v_mag * np.cos(grazing_angle))**2
    kz2_over_w = np.sqrt(radicand + 0j)

    num = rho2 * kz1_over_w - rho1 * kz2_over_w
    den = rho2 * kz1_over_w + rho1 * kz2_over_w

    if abs(den) < 1e-30:
        return 1.0 + 0.0j
    return num / den


@njit(fastmath=True)
def get_reflection_coef_at(
    th: float,
    tg: float,
    rho1: float,
    c1: float,
    rho2: float,
    c2: float,
    c_s2: float,
    alpha_dB_lambda: float,
    alpha_s_dB_lambda: float,
) -> complex:
    """AT Bellhop-style fluid/elastic half-space reflection coefficient."""
    if c2 < 0.0:
        return 1.0 + 0.0j
    if c2 <= 0.0:
        return -1.0 + 0.0j

    omega = 1.0
    i_unit = 1j
    kx = omega * tg
    kz = omega * th

    alpha_p_nep = alpha_dB_lambda / 8.6858896
    alpha_p_factor = alpha_p_nep / (2.0 * np.pi)
    c2_complex = c2 / (1.0 + i_unit * alpha_p_factor)

    if c_s2 > 0.0:
        alpha_s_nep = alpha_s_dB_lambda / 8.6858896
        alpha_s_factor = alpha_s_nep / (2.0 * np.pi)
        cs_complex = c_s2 / (1.0 + i_unit * alpha_s_factor)
        kz_s2 = kx * kx - (omega / cs_complex) ** 2
        kz_p2 = kx * kx - (omega / c2_complex) ** 2
        kz_s = np.sqrt(kz_s2)
        kz_p = np.sqrt(kz_p2)
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
        return 1.0 + 0.0j
    return -((rho1 * f - i_unit * kz * g) / den)

@njit(fastmath=True)
def get_bathymetry_3d_z(x: float, y: float, bty_x: np.ndarray, bty_y: np.ndarray, bty_z: np.ndarray):
    """
    AT-compatible planar interpolation for 3D bathymetry grid.

    Bellhop3D does not use a smooth bilinear surface for reflection. It splits
    each rectangular bathymetry cell into two triangular facets and uses the
    active facet normal. Returning the facet slopes here makes specular
    reflection match bdry3DMod.f90's n1/n2 selection.
    Returns: (z_interpolated, dz_dx, dz_dy)
    """
    if len(bty_x) == 0 or len(bty_y) == 0:
        return 1e9, 0.0, 0.0  # Safe fallback

    nx = len(bty_x)
    ny = len(bty_y)

    # Clip to grid boundaries to prevent out-of-bounds evaluation
    x_c = min(max(x, bty_x[0]), bty_x[-1])
    y_c = min(max(y, bty_y[0]), bty_y[-1])

    idx_x = np.searchsorted(bty_x, x_c) - 1
    idx_y = np.searchsorted(bty_y, y_c) - 1

    idx_x = max(0, min(idx_x, nx - 2))
    idx_y = max(0, min(idx_y, ny - 2))

    x1, x2 = bty_x[idx_x], bty_x[idx_x+1]
    y1, y2 = bty_y[idx_y], bty_y[idx_y+1]

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 or dy == 0:
        return bty_z[idx_x, idx_y], 0.0, 0.0

    # AT gridded Bellhop3D .bty files store one row per y value, with x
    # varying across columns. Keep the native schema aligned with that
    # (len(y), len(x)) layout used by write_at_bathymetry_3d.
    z00 = bty_z[idx_y, idx_x]
    z10 = bty_z[idx_y, idx_x+1]
    z01 = bty_z[idx_y+1, idx_x]
    z11 = bty_z[idx_y+1, idx_x+1]

    # AT chooses between triangles p1-p2-p3 and p1-p3-p4 using the
    # diagonal p1->p3. This is equivalent to dot((x,y)-p1, [-dy, dx]) < 0.
    tri_side = (x_c - x1) * (-dy) + (y_c - y1) * dx
    if tri_side < 0.0:
        # Triangle 1: p1=(x1,y1,z00), p2=(x2,y1,z10), p3=(x2,y2,z11)
        dz_dx = (z10 - z00) / dx
        dz_dy = (z11 - z10) / dy
    else:
        # Triangle 2: p1=(x1,y1,z00), p3=(x2,y2,z11), p4=(x1,y2,z01)
        dz_dx = (z11 - z01) / dx
        dz_dy = (z01 - z00) / dy

    z_interp = z00 + dz_dx * (x_c - x1) + dz_dy * (y_c - y1)

    return z_interp, dz_dx, dz_dy

@njit(fastmath=True)
def boundary_crossing_fraction_3d(
    x_old: float, y_old: float, z_old: float,
    x_trial: float, y_trial: float, z_trial: float,
    hit: int,
    z_surface: float,
    bty_x: np.ndarray, bty_y: np.ndarray, bty_z: np.ndarray,
    z_bottom_flat: float
):
    """Return the fraction of a trial segment that lands on the active boundary."""
    if hit == 1:
        denom = z_trial - z_old
        if abs(denom) < 1e-14:
            return 1.0
        frac = (z_surface - z_old) / denom
        return min(1.0, max(0.0, frac))

    lo = 0.0
    hi = 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        xm = x_old + mid * (x_trial - x_old)
        ym = y_old + mid * (y_trial - y_old)
        zm = z_old + mid * (z_trial - z_old)
        if len(bty_x) > 0:
            zb, _, _ = get_bathymetry_3d_z(xm, ym, bty_x, bty_y, bty_z)
        else:
            zb = z_bottom_flat
        if zm >= zb:
            hi = mid
        else:
            lo = mid
    return hi

@njit(fastmath=True)
def _grid_segment_crossing_h(
    pos: float, trial: float, vel: float, grid: np.ndarray
):
    if len(grid) < 2 or abs(vel) < 1.0e-12:
        return 1.0e30

    p = min(max(pos, grid[0]), grid[-1])
    idx = np.searchsorted(grid, p) - 1
    idx = max(0, min(idx, len(grid) - 2))

    low = grid[idx]
    high = grid[idx + 1]
    if trial < low:
        h_cross = (low - pos) / vel
    elif trial > high:
        h_cross = (high - pos) / vel
    else:
        return 1.0e30

    if h_cross > 1.0e-9:
        return h_cross
    return 1.0e30

@njit(fastmath=True)
def _depth_segment_crossing_h(
    z: float, z_trial: float, vel_z: float, z_grid: np.ndarray
):
    if len(z_grid) < 2 or abs(vel_z) < 1.0e-12:
        return 1.0e30

    zc = min(max(z, z_grid[0]), z_grid[-1])
    idx = np.searchsorted(z_grid, zc) - 1
    idx = max(0, min(idx, len(z_grid) - 2))

    # Match AT: do not reduce at the top and bottom extrapolation layers.
    if z_trial < z_grid[idx] and idx > 0:
        h_cross = (z_grid[idx] - z) / vel_z
    elif z_trial > z_grid[idx + 1] and idx + 1 < len(z_grid) - 1:
        h_cross = (z_grid[idx + 1] - z) / vel_z
    else:
        return 1.0e30

    if h_cross > 1.0e-9:
        return h_cross
    return 1.0e30

@njit(fastmath=True)
def _bty_diagonal_crossing_h(
    x: float, y: float, ux: float, uy: float, h: float,
    bty_x: np.ndarray, bty_y: np.ndarray
):
    if len(bty_x) < 2 or len(bty_y) < 2:
        return 1.0e30

    x_c = min(max(x, bty_x[0]), bty_x[-1])
    y_c = min(max(y, bty_y[0]), bty_y[-1])

    idx_x = np.searchsorted(bty_x, x_c) - 1
    idx_y = np.searchsorted(bty_y, y_c) - 1
    idx_x = max(0, min(idx_x, len(bty_x) - 2))
    idx_y = max(0, min(idx_y, len(bty_y) - 2))

    x1, x2 = bty_x[idx_x], bty_x[idx_x + 1]
    y1, y2 = bty_y[idx_y], bty_y[idx_y + 1]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 or dy == 0.0:
        return 1.0e30

    side0 = (x - x1) * (-dy) + (y - y1) * dx
    side_vel = ux * (-dy) + uy * dx
    if abs(side_vel) < 1.0e-12:
        return 1.0e30

    h_cross = -side0 / side_vel
    if h_cross > 1.0e-9 and h_cross < h:
        return h_cross
    return 1.0e30

@njit(fastmath=True)
def reduce_step_fraction_3d(
    x: float, y: float, z: float,
    ux: float, uy: float, uz: float,
    h: float,
    bty_x: np.ndarray, bty_y: np.ndarray,
    ssp_x: np.ndarray, ssp_y: np.ndarray, ssp_z: np.ndarray,
    use_ssp3d: bool
):
    """
    AT ReduceStep3D-compatible grid event reduction.

    This handles non-reflection events that change interpolation segments:
    x/y bathymetry cells, active triangular facet diagonal, SSP z interfaces,
    and H/3D-SSP x/y cells. Top/bottom reflection and ray boxes are handled
    separately by the tracer.
    """
    x_trial = x + h * ux
    y_trial = y + h * uy
    z_trial = z + h * uz

    h_min = h

    hx = _grid_segment_crossing_h(x, x_trial, ux, bty_x)
    if hx < h_min:
        h_min = hx
    hy = _grid_segment_crossing_h(y, y_trial, uy, bty_y)
    if hy < h_min:
        h_min = hy
    hd = _bty_diagonal_crossing_h(x, y, ux, uy, h, bty_x, bty_y)
    if hd < h_min:
        h_min = hd

    hz = _depth_segment_crossing_h(z, z_trial, uz, ssp_z)
    if hz < h_min:
        h_min = hz

    if use_ssp3d:
        hsx = _grid_segment_crossing_h(x, x_trial, ux, ssp_x)
        if hsx < h_min:
            h_min = hsx
        hsy = _grid_segment_crossing_h(y, y_trial, uy, ssp_y)
        if hsy < h_min:
            h_min = hsy

    if h_min < h:
        return max(0.0, h_min / h)
    return 1.0

@njit(fastmath=True)
def check_boundary_3d(x_old: float, y_old: float, z_old: float, x_new: float, y_new: float, z_new: float, z_surface: float, bty_x: np.ndarray, bty_y: np.ndarray, bty_z: np.ndarray, z_bottom_flat: float):
    """
    Checks if a 3D ray segment intersected the surface or 3D bathymetry.
    Returns: hit_type (-1 for bottom, 1 for surface, 0 for None), local_bottom_depth, dz_dx, dz_dy
    """
    hit = 0
    dz_dx = 0.0
    dz_dy = 0.0

    if len(bty_x) > 0:
        z_b_new, dz_dx, dz_dy = get_bathymetry_3d_z(x_new, y_new, bty_x, bty_y, bty_z)
        z_b_old, _, _ = get_bathymetry_3d_z(x_old, y_old, bty_x, bty_y, bty_z)
    else:
        z_b_new = z_bottom_flat
        z_b_old = z_bottom_flat

    # Check bottom intersection (went from above bottom to below bottom)
    if z_new >= z_b_new and z_old <= z_b_old:
        hit = -1
    # Check surface intersection
    elif z_new <= z_surface and z_old >= z_surface:
        hit = 1

    return hit, z_b_new, dz_dx, dz_dy

@njit(fastmath=True)
def reflect_ray_3d(
    xi: float, eta: float, zeta: float,
    hit: int, dz_dx: float, dz_dy: float,
    bot_rho: float, bot_c: float, bot_c_s: float, bot_alpha: float, bot_alpha_s: float,
    c_bnd: float,
    surface_refl_angles: np.ndarray, surface_refl_amp: np.ndarray, surface_refl_phase: np.ndarray,
    bottom_refl_angles: np.ndarray, bottom_refl_amp: np.ndarray, bottom_refl_phase: np.ndarray,
    surf_rho: float = 1.0, surf_c: float = 0.0, surf_c_s: float = 0.0,
    surf_alpha: float = 0.0, surf_alpha_s: float = 0.0,
):
    """
    Performs 3D specular reflection based on incident slowness vector and local terrain normal.
    Returns: (xi_out, eta_out, zeta_out, Reflection_Coefficient)
    """
    if hit == 1:
        # Surface: Perfect reflection, normal is straight UP [0, 0, -1]
        n_x, n_y, n_z = 0.0, 0.0, -1.0
    else:
        # Bottom: Surface defined by f(x,y,z) = z - z_b(x,y) = 0
        # Normal Grad(f) = [-dz_dx, -dz_dy, 1], points UP into the fluid medium
        norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
        n_x = -dz_dx / norm
        n_y = -dz_dy / norm
        n_z = 1.0 / norm

    # Reflection: v_out = v_in - 2*(v_in dot n)*n
    v_dot_n = xi * n_x + eta * n_y + zeta * n_z

    xi_out = xi - 2 * v_dot_n * n_x
    eta_out = eta - 2 * v_dot_n * n_y
    zeta_out = zeta - 2 * v_dot_n * n_z

    # Calculate True Grazing Angle relative to the local inclined 3D plane
    # sin(grazing) = |v_dot_n| / |v|
    v_mag = np.sqrt(xi**2 + eta**2 + zeta**2)
    sin_grazing = abs(v_dot_n) / v_mag
    sin_grazing = min(1.0, max(0.0, sin_grazing)) # Safety clamp
    theta_grazing = np.arcsin(sin_grazing)
    grazing_deg = abs(np.degrees(theta_grazing))

    if hit == 1:
        if len(surface_refl_angles) > 0:
            amp = np.interp(grazing_deg, surface_refl_angles, surface_refl_amp)
            phase = np.interp(grazing_deg, surface_refl_angles, surface_refl_phase)
            R = amp * np.exp(1j * np.radians(phase))
        elif surf_c < 0.0:
            R = 1.0 + 0.0j
        elif surf_c <= 0.0:
            R = -1.0 + 0.0j
        else:
            tg = np.sqrt(max(0.0, v_mag * v_mag - v_dot_n * v_dot_n))
            R = get_reflection_coef_at(
                v_dot_n, tg,
                1.0, c_bnd,
                surf_rho, surf_c, surf_c_s, surf_alpha, surf_alpha_s,
            )

    # Acquire impedance-based bottom loss
    if hit == -1:
        if len(bottom_refl_angles) > 0:
            amp = np.interp(grazing_deg, bottom_refl_angles, bottom_refl_amp)
            phase = np.interp(grazing_deg, bottom_refl_angles, bottom_refl_phase)
            R = amp * np.exp(1j * np.radians(phase))
        else:
            tg = np.sqrt(max(0.0, v_mag * v_mag - v_dot_n * v_dot_n))
            R = get_reflection_coef_at(
                v_dot_n, tg,
                1.0, c_bnd,
                bot_rho, bot_c, bot_c_s, bot_alpha, bot_alpha_s,
            )

    return xi_out, eta_out, zeta_out, R

@njit(fastmath=True)
def apply_reflection_curvature_correction(
    p11: complex, p12: complex, p21: complex, p22: complex,
    q11: complex, q12: complex, q21: complex, q22: complex,
    xi_in: float, eta_in: float, zeta_in: float,
    xi_out: float, eta_out: float, zeta_out: float,
    hit: int, dz_dx: float, dz_dy: float, c_bnd: float,
    dc_dx: float, dc_dy: float, dc_dz: float
):
    """
    Lightweight AT-compatible reflection correction for the dynamic ray matrix.

    Full Reflect3DMod.f90 also includes boundary curvature terms. The native
    bathymetry API only carries first derivatives, so this applies the part
    that is always available and critical for rotating the p/q bases across a
    specular reflection. For locally planar facets the curvature terms are zero.
    """
    if hit == 1:
        n_x, n_y, n_z = 0.0, 0.0, -1.0
    else:
        norm = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
        n_x = -dz_dx / norm
        n_y = -dz_dy / norm
        n_z = 1.0 / norm

    tx = c_bnd * xi_in
    ty = c_bnd * eta_in
    tz = c_bnd * zeta_in
    tx_out = c_bnd * xi_out
    ty_out = c_bnd * eta_out
    tz_out = c_bnd * zeta_out
    # Incident reflection-plane normal rayn2 = -cross(rayt, nBdry).
    rn2x = -(ty * n_z - tz * n_y)
    rn2y = -(tz * n_x - tx * n_z)
    rn2z = -(tx * n_y - ty * n_x)
    rn2_norm = np.sqrt(rn2x * rn2x + rn2y * rn2y + rn2z * rn2z)
    if rn2_norm < 1e-12:
        return p11, p12, p21, p22
    rn2x /= rn2_norm
    rn2y /= rn2_norm
    rn2z /= rn2_norm
    rn1x = -(ty * rn2z - tz * rn2y)
    rn1y = -(tz * rn2x - tx * rn2z)
    rn1z = -(tx * rn2y - ty * rn2x)

    rn2ox = -(ty_out * n_z - tz_out * n_y)
    rn2oy = -(tz_out * n_x - tx_out * n_z)
    rn2oz = -(tx_out * n_y - ty_out * n_x)
    rn2o_norm = np.sqrt(rn2ox * rn2ox + rn2oy * rn2oy + rn2oz * rn2oz)
    if rn2o_norm < 1e-12:
        return p11, p12, p21, p22
    rn2ox /= rn2o_norm
    rn2oy /= rn2o_norm
    rn2oz /= rn2o_norm
    rn1ox = -(ty_out * rn2oz - tz_out * rn2oy)
    rn1oy = -(tz_out * rn2ox - tx_out * rn2oz)
    rn1oz = -(tx_out * rn2oy - ty_out * rn2ox)

    # Local ray-centered basis for phi=0, matching RayNormal_unit use elsewhere.
    rl = np.sqrt(tx * tx + ty * ty)
    if rl > 1e-12:
        e1x = tx * tz / rl
        e1y = ty * tz / rl
        e1z = -rl
        e2x = -ty / rl
        e2y = tx / rl
        e2z = 0.0
    else:
        e1x, e1y, e1z = 1.0, 0.0, 0.0
        e2x, e2y, e2z = 0.0, 1.0, 0.0

    r11 = rn1x * e1x + rn1y * e1y + rn1z * e1z
    r12 = rn1x * e2x + rn1y * e2y + rn1z * e2z
    r21 = -r12
    r22 = rn2x * e2x + rn2y * e2y + rn2z * e2z

    cn1jump = dc_dx * (-rn1ox - rn1x) + dc_dy * (-rn1oy - rn1y) + dc_dz * (-rn1oz - rn1z)
    cn2jump = dc_dx * (-rn2ox - rn2x) + dc_dy * (-rn2oy - rn2y) + dc_dz * (-rn2oz - rn2z)
    csjump = -(dc_dx * (tx_out - tx) + dc_dy * (ty_out - ty) + dc_dz * (tz_out - tz))

    th = tx * n_x + ty * n_y + tz * n_z
    tbx = tx - th * n_x
    tby = ty - th * n_y
    tbz = tz - th * n_z
    tg = np.sqrt(tbx * tbx + tby * tby + tbz * tbz)
    if abs(th) > 1e-12:
        rm = tg / th
        r1 = rm * (2.0 * cn1jump - rm * csjump) / (c_bnd * c_bnd)
        r2 = rm * cn2jump / (c_bnd * c_bnd)
    else:
        r1 = 0.0
        r2 = 0.0
    r3 = 0.0

    # Rotate p/q into the reflection-plane normal basis, apply the planar
    # portion of AT CurvatureCorrection2, and rotate p back. For triangular
    # bathymetry facets the boundary curvature matrix itself is zero.
    ptilde1 = r11 * p11 + r12 * p21
    ptilde2 = r11 * p12 + r12 * p22
    phat1 = r21 * p11 + r22 * p21
    phat2 = r21 * p12 + r22 * p22

    qtilde1 = r11 * q11 + r12 * q21
    qtilde2 = r11 * q12 + r12 * q22
    qhat1 = r21 * q11 + r22 * q21
    qhat2 = r21 * q12 + r22 * q22

    ptilde1 = ptilde1 + qtilde1 * r1 - qhat1 * r2
    ptilde2 = ptilde2 + qtilde2 * r1 - qhat2 * r2
    phat1 = phat1 + qtilde1 * r2 + qhat1 * r3
    phat2 = phat2 + qtilde2 * r2 + qhat2 * r3

    p11_out = r11 * ptilde1 + r21 * phat1
    p12_out = r11 * ptilde2 + r21 * phat2
    p21_out = r12 * ptilde1 + r22 * phat1
    p22_out = r12 * ptilde2 + r22 * phat2

    return p11_out, p12_out, p21_out, p22_out

@njit(fastmath=True)
def apply_ssp_interface_curvature_correction_3d(
    p11: complex, p12: complex, p21: complex, p22: complex,
    q11: complex, q12: complex, q21: complex, q22: complex,
    xi: float, eta: float, zeta: float, c_bnd: float,
    n_x: float, n_y: float, n_z: float,
    dc_dx_jump: float, dc_dy_jump: float, dc_dz_jump: float,
):
    """
    AT Step3DMod CurvatureCorrection for sound-speed gradient jumps.

    This correction is applied when ReduceStep3D lands a ray exactly on an SSP
    segment interface. It updates P while Q remains continuous.
    """
    th = xi * n_x + eta * n_y + zeta * n_z
    if abs(th) < 1.0e-12:
        return p11, p12, p21, p22

    tbx = xi - th * n_x
    tby = eta - th * n_y
    tbz = zeta - th * n_z
    tb_norm = np.sqrt(tbx * tbx + tby * tby + tbz * tbz)
    if tb_norm < 1.0e-12:
        return p11, p12, p21, p22
    tbx /= tb_norm
    tby /= tb_norm
    tbz /= tb_norm
    tg = xi * tbx + eta * tby + zeta * tbz

    ux = c_bnd * xi
    uy = c_bnd * eta
    uz = c_bnd * zeta

    rn2x = uy * n_z - uz * n_y
    rn2y = uz * n_x - ux * n_z
    rn2z = ux * n_y - uy * n_x
    rn2_norm = np.sqrt(rn2x * rn2x + rn2y * rn2y + rn2z * rn2z)
    if rn2_norm < 1.0e-12:
        return p11, p12, p21, p22
    rn2x /= rn2_norm
    rn2y /= rn2_norm
    rn2z /= rn2_norm

    rn1x = -(uy * rn2z - uz * rn2y)
    rn1y = -(uz * rn2x - ux * rn2z)
    rn1z = -(ux * rn2y - uy * rn2x)

    cn1jump = dc_dx_jump * rn1x + dc_dy_jump * rn1y + dc_dz_jump * rn1z
    cn2jump = dc_dx_jump * rn2x + dc_dy_jump * rn2y + dc_dz_jump * rn2z
    csjump = dc_dx_jump * ux + dc_dy_jump * uy + dc_dz_jump * uz

    rm = tg / th
    r1 = rm * (2.0 * cn1jump - rm * csjump) / (c_bnd * c_bnd)
    r2 = rm * cn2jump / (c_bnd * c_bnd)

    rl = np.sqrt(ux * ux + uy * uy)
    if rl > 1.0e-12:
        e1x = ux * uz / rl
        e1y = uy * uz / rl
        e1z = -rl
        e2x = -uy / rl
        e2y = ux / rl
        e2z = 0.0
    else:
        e1x, e1y, e1z = 1.0, 0.0, 0.0
        e2x, e2y, e2z = 0.0, 1.0, 0.0

    rot11 = rn1x * e1x + rn1y * e1y + rn1z * e1z
    rot12 = rn1x * e2x + rn1y * e2y + rn1z * e2z
    rot21 = -rot12
    rot22 = rn2x * e2x + rn2y * e2y + rn2z * e2z

    ptilde1 = rot11 * p11 + rot12 * p21
    ptilde2 = rot11 * p12 + rot12 * p22
    phat1 = rot21 * p11 + rot22 * p21
    phat2 = rot21 * p12 + rot22 * p22

    qtilde1 = rot11 * q11 + rot12 * q21
    qtilde2 = rot11 * q12 + rot12 * q22
    qhat1 = rot21 * q11 + rot22 * q21
    qhat2 = rot21 * q12 + rot22 * q22

    ptilde1 = ptilde1 - qtilde1 * r1 - qhat1 * r2
    ptilde2 = ptilde2 - qtilde2 * r1 - qhat2 * r2
    phat1 = phat1 - qtilde1 * r2
    phat2 = phat2 - qtilde2 * r2

    p11_out = rot11 * ptilde1 + rot21 * phat1
    p12_out = rot11 * ptilde2 + rot21 * phat2
    p21_out = rot12 * ptilde1 + rot22 * phat1
    p22_out = rot12 * ptilde2 + rot22 * phat2

    return p11_out, p12_out, p21_out, p22_out

@njit(fastmath=True)
def crossed_interior_z_interface(z_old: float, z_new: float, z_grid: np.ndarray):
    if len(z_grid) < 3:
        return False
    if z_new > z_old:
        for i in range(1, len(z_grid) - 1):
            if z_old < z_grid[i] and z_new >= z_grid[i]:
                return True
    elif z_new < z_old:
        for i in range(1, len(z_grid) - 1):
            if z_old > z_grid[i] and z_new <= z_grid[i]:
                return True
    return False
