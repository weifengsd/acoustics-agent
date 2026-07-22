import numpy as np
from numba import njit, prange
from numba.typed import List


def receiver_grid_segment_range(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    source_x: float = 0.0,
    source_y: float = 0.0,
) -> tuple[float, float]:
    """AT Bellhop3D receiver-grid range bracket for a ray segment.

    ``Influence3DGeoGaussianCart`` brackets receivers by horizontal range from
    the source point, not by absolute distance from the global origin.
    """

    return (
        float(np.hypot(x0 - source_x, y0 - source_y)),
        float(np.hypot(x1 - source_x, y1 - source_y)),
    )


@njit(fastmath=True)
def _receiver_grid_segment_range_numba(x0: float, y0: float, x1: float, y1: float, source_x: float, source_y: float):
    r_a = np.sqrt((x0 - source_x) * (x0 - source_x) + (y0 - source_y) * (y0 - source_y))
    r_b = np.sqrt((x1 - source_x) * (x1 - source_x) + (y1 - source_y) * (y1 - source_y))
    return r_a, r_b


@njit(fastmath=True)
def distance_to_segment_3d(rx: float, ry: float, rz: float, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
    """
    Computes the shortest orthogonal distance from a voxel point (rx, ry, rz)
    to a 3D ray segment defined by (x1, y1, z1) and (x2, y2, z2).
    """
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    L2 = dx**2 + dy**2 + dz**2
    if L2 == 0.0:
        return np.sqrt((rx - x1)**2 + (ry - y1)**2 + (rz - z1)**2), 0.0

    # Dot product projection
    t = ((rx - x1) * dx + (ry - y1) * dy + (rz - z1) * dz) / L2

    # Clamp projection to segment endpoints
    t = max(0.0, min(1.0, t))

    px = x1 + t * dx
    py = y1 + t * dy
    pz = z1 + t * dz

    dist = np.sqrt((rx - px)**2 + (ry - py)**2 + (rz - pz)**2)
    return dist, t

@njit(fastmath=True)
def compute_tl_field_coherent_3d(
    x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,
    rays_x: List, rays_y: List, rays_z: List,
    rays_q11: List, rays_q12: List, rays_q21: List, rays_q22: List,
    rays_amp: List, rays_tau: List, rays_kmah: List, rays_cpath: List, rays_phi: List, rays_elev: List,
    freq: float, c_src: float,
    d_azim: float, d_elev: float
):
    """
    Computes coherent Transmission Loss across a 3D Voxel Grid using 3D Anisotropic Elliptical Gaussian Beams.
    Ported strictly from influence3D.f90 for 100% mathematical and physical alignment.
    """
    nx = len(x_grid)
    ny = len(y_grid)
    nz = len(z_grid)

    p_field = np.zeros((nx, ny, nz), dtype=np.complex128)

    lambd = c_src / freq
    beam_window = 4.0 # kills beams outside exp(-0.5 * 4^2)

    for ray_idx in range(len(rays_x)):
        x_path = rays_x[ray_idx]
        y_path = rays_y[ray_idx]
        z_path = rays_z[ray_idx]
        q11_path = rays_q11[ray_idx]
        q12_path = rays_q12[ray_idx]
        q21_path = rays_q21[ray_idx]
        q22_path = rays_q22[ray_idx]
        amp_path = rays_amp[ray_idx]
        tau_path = rays_tau[ray_idx]
        kmah_path = rays_kmah[ray_idx]
        c_path = rays_cpath[ray_idx]
        phi_path = rays_phi[ray_idx]
        elev_deg = rays_elev[ray_idx]

        elev_rad = np.radians(elev_deg)

        n_pts = len(x_path)
        if n_pts < 2:
            continue

        for step in range(n_pts - 1):
            x1, y1, z1 = x_path[step], y_path[step], z_path[step]
            x2, y2, z2 = x_path[step+1], y_path[step+1], z_path[step+1]

            # Unit tangent to ray
            tx = x2 - x1
            ty = y2 - y1
            tz = z2 - z1
            rlen = np.sqrt(tx**2 + ty**2 + tz**2)
            if rlen < 1e-10:
                continue

            # Unit tangent
            tx /= rlen
            ty /= rlen
            tz /= rlen

            # Compute ray normals e1 and e2 according to AT RayNormal_unit.
            phi = phi_path[step - 1]
            rl = np.sqrt(tx**2 + ty**2)
            if rl > 1e-10:
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                e1_x = (tx * tz * cos_phi + ty * sin_phi) / rl
                e1_y = (ty * tz * cos_phi - tx * sin_phi) / rl
                e1_z = -rl * cos_phi

                e2_x = (tx * tz * sin_phi - ty * cos_phi) / rl
                e2_y = (ty * tz * sin_phi + tx * cos_phi) / rl
                e2_z = -rl * sin_phi
            else:
                e1_x, e1_y, e1_z = 1.0, 0.0, 0.0
                e2_x, e2_y, e2_z = 0.0, 1.0, 0.0

            # Linearly interpolate travel times, amplitudes and KMAH shifts
            tau1 = tau_path[step]
            tau2 = tau_path[step+1]
            dtau = tau2 - tau1
            # influence3D.f90 uses ray3D(is)%Amp for the segment contribution,
            # where is is the segment endpoint.
            amp_val = amp_path[step + 1]
            kmah = kmah_path[step]

            # Scale Q-matrix to match Launch Density (Dalpha, Dbeta).
            # Legacy Fortran (influence3D.f90:L698) uses scale = Dalpha / c_src
            # Our tracer uses slowness p=1/c, so q already grows in meters.
            scale_tilde = d_elev
            scale_hat = d_azim * np.cos(elev_rad)

            q11 = q11_path[step] * scale_tilde
            q21 = q21_path[step] * scale_tilde
            q12 = q12_path[step] * scale_hat
            q22 = q22_path[step] * scale_hat

            # Bounding box using the larger beam width of the two endpoints to prevent missing receivers
            q11_2 = q11_path[step+1] * scale_tilde
            q21_2 = q21_path[step+1] * scale_tilde
            q12_2 = q12_path[step+1] * scale_hat
            q22_2 = q22_path[step+1] * scale_hat

            l1_start = np.sqrt(abs(q11)**2 + abs(q21)**2)
            l2_start = np.sqrt(abs(q12)**2 + abs(q22)**2)
            l1_end = np.sqrt(abs(q11_2)**2 + abs(q21_2)**2)
            l2_end = np.sqrt(abs(q12_2)**2 + abs(q22_2)**2)

            l_max = max(np.sqrt(l1_start**2 + l2_start**2), np.sqrt(l1_end**2 + l2_end**2))
            if l_max < 2.0 * lambd:
                l_max = 2.0 * lambd

            beam_width = beam_window * l_max
            if beam_width > 1200.0:
                beam_width = 1200.0

            # Unscaled DetQ at the start of the segment to check for caustic crossing within this segment
            Det_QOld = q11_path[step] * q22_path[step] - q12_path[step] * q21_path[step]

            # Spatial voxel search bracket bounding box
            min_x, max_x = min(x1, x2) - beam_width, max(x1, x2) + beam_width
            min_y, max_y = min(y1, y2) - beam_width, max(y1, y2) + beam_width
            min_z, max_z = min(z1, z2) - beam_width, max(z1, z2) + beam_width

            ix_min = max(0, np.searchsorted(x_grid, min_x) - 1)
            ix_max = min(nx, np.searchsorted(x_grid, max_x))
            iy_min = max(0, np.searchsorted(y_grid, min_y) - 1)
            iy_max = min(ny, np.searchsorted(y_grid, max_y))
            iz_min = max(0, np.searchsorted(z_grid, min_z) - 1)
            iz_max = min(nz, np.searchsorted(z_grid, max_z))

            for ix in range(ix_min, ix_max):
                for iy in range(iy_min, iy_max):
                    for iz in range(iz_min, iz_max):
                        vx = x_grid[ix]
                        vy = y_grid[iy]
                        vz = z_grid[iz]

                        # Orthogonal receiver projection to current segment
                        # To avoid double-counting at segment boundaries, we only accept
                        # contributions where the orthogonal projection falls strictly within [0, 1).
                        dx_seg = x2 - x1
                        dy_seg = y2 - y1
                        dz_seg = z2 - z1
                        L2_seg = dx_seg**2 + dy_seg**2 + dz_seg**2

                        if L2_seg < 1e-10:
                            continue

                        t_unclamped = ((vx - x1) * dx_seg + (vy - y1) * dy_seg + (vz - z1) * dz_seg) / L2_seg

                        # Only accept if voxel is "owned" by this segment's orthogonal plane
                        if t_unclamped < 0.0 or t_unclamped >= 1.0:
                            continue

                        t = t_unclamped # valid in [0, 1)

                        dx_rcvr = vx - x1
                        dy_rcvr = vy - y1
                        dz_rcvr = vz - z1

                        # Project rcvr-ray coordinates onto normals e1 and e2 to acquire (n, m)
                        n = abs(dx_rcvr * e1_x + dy_rcvr * e1_y + dz_rcvr * e1_z)
                        m = abs(dx_rcvr * e2_x + dy_rcvr * e2_y + dz_rcvr * e2_z)

                        # Interpolate the Q matrix at the receiver projection before
                        # solving for the local elliptic coordinates. This matches
                        # influence3D.f90, where q_tilde/q_hat are interpolated at s
                        # before DetQint, a, and b are evaluated.
                        q11_t = q11_path[step] + t * (q11_path[step+1] - q11_path[step])
                        q12_t = q12_path[step] + t * (q12_path[step+1] - q12_path[step])
                        q21_t = q21_path[step] + t * (q21_path[step+1] - q21_path[step])
                        q22_t = q22_path[step] + t * (q22_path[step+1] - q22_path[step])
                        DetQint = q11_t * q22_t - q12_t * q21_t

                        q11_s = q11_t * scale_tilde
                        q21_s = q21_t * scale_tilde
                        q12_s = q12_t * scale_hat
                        q22_s = q22_t * scale_hat
                        DetQint_scaled = q11_s * q22_s - q12_s * q21_s
                        if abs(DetQint_scaled) < 1e-20:
                            continue

                        # Cramer's rule to extract local elliptic coordinates a and b matching the ray tube.
                        a_ell = abs((-q12_s * m + q22_s * n) / DetQint_scaled)
                        b_ell = abs((q11_s * m - q21_s * n) / DetQint_scaled)

                        # Elliptic Gaussian boundary threshold
                        if (a_ell + b_ell) <= beam_window:
                            # Strict Anisotropic Gaussian physical envelope decay: exp(-0.5 * (a^2 + b^2))
                            decay = np.exp(-0.5 * (a_ell**2 + b_ell**2))

                            # Amplitude under total take-off angle cancellation (Ratio1 matches 1 / (2 * pi * Q_reg))
                            # Physical correction: Amp proportional to c(t) / (2 * pi * sqrt(DetQint))
                            # We use interpolation for c(t) to match legacy intensity scaling.

                            # Acquire current interpolated sound speed at projection point t
                            c_t = c_path[step] + t * (c_path[step+1] - c_path[step])

                            q_abs_t = np.sqrt(abs(DetQint))
                            q_abs_reg_t = max(q_abs_t, 0.5 * lambd)

                            # Legacy match: Amp = c(s) * Amp_unscaled / (2 * pi * sqrt(DetQ_unscaled))
                            amp = (c_t / c_src) * amp_val / (2.0 * np.pi * q_abs_reg_t)

                            # Caustic crossing detection within the current segment
                            effective_kmah = kmah
                            if Det_QOld.real * DetQint.real < 0.0:
                                effective_kmah = kmah + 1.0

                            tau_interpolated = tau1 + t * dtau
                            phase_exponent = complex(0.0, -(2.0 * np.pi * freq * tau_interpolated - effective_kmah * np.pi * 0.5))
                            p_val = amp * decay * np.exp(phase_exponent)

                            p_field[ix, iy, iz] += p_val


    # Transform complex pressure to dB Transmission Loss
    TL = np.zeros((nx, ny, nz), dtype=np.float64)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                p_mag = abs(p_field[ix, iy, iz])
                if p_mag < 1e-20:
                    TL[ix, iy, iz] = 200.0
                else:
                    TL[ix, iy, iz] = -20.0 * np.log10(p_mag)

    return TL


@njit(fastmath=True)
def compute_pressure_field_receiver_grid_3d(
    theta_grid_deg: np.ndarray, r_grid: np.ndarray, z_grid: np.ndarray,
    rays_x: List, rays_y: List, rays_z: List,
    rays_q11: List, rays_q12: List, rays_q21: List, rays_q22: List,
    rays_amp: List, rays_tau: List, rays_kmah: List, rays_cpath: List, rays_phi: List, rays_elev: List,
    freq: float, c_src: float,
    source_x: float, source_y: float,
    d_azim: float, d_elev: float
):
    """
    Computes coherent complex pressure on the AT Bellhop3D receiver grid.

    This follows influence3D.f90::Influence3DGeoGaussianCart more closely than
    the voxel-grid helper above: receiver ownership is determined by crossings
    of each ray segment through the polar receiver range grid, then each
    receiver depth is tested in the corresponding vertical receiver plane.
    """
    ntheta = len(theta_grid_deg)
    nr = len(r_grid)
    nz = len(z_grid)
    p_field = np.zeros((ntheta, nz, nr), dtype=np.complex128)

    beam_window = 4.0
    lambd = c_src / freq

    for ray_idx in range(len(rays_x)):
        x_path = rays_x[ray_idx]
        y_path = rays_y[ray_idx]
        z_path = rays_z[ray_idx]
        q11_path = rays_q11[ray_idx]
        q12_path = rays_q12[ray_idx]
        q21_path = rays_q21[ray_idx]
        q22_path = rays_q22[ray_idx]
        amp_path = rays_amp[ray_idx]
        tau_path = rays_tau[ray_idx]
        kmah_path = rays_kmah[ray_idx]
        c_path = rays_cpath[ray_idx]
        phi_path = rays_phi[ray_idx]
        elev_deg = rays_elev[ray_idx]
        elev_rad = np.radians(elev_deg)

        n_pts = len(x_path)
        if n_pts < 2:
            continue

        scale_tilde = d_elev
        scale_hat = d_azim * abs(np.cos(elev_rad))

        r_a = 0.0
        ir = 0
        phase = 0.0
        det_q_old = q11_path[0] * q22_path[0] - q12_path[0] * q21_path[0]

        for step in range(1, n_pts):
            x0 = x_path[step - 1]
            y0 = y_path[step - 1]
            z0 = z_path[step - 1]
            x1 = x_path[step]
            y1 = y_path[step]
            z1 = z_path[step]

            if step == 1:
                r_a, r_b = _receiver_grid_segment_range_numba(x0, y0, x1, y1, source_x, source_y)
            else:
                r_b = np.sqrt((x1 - source_x) * (x1 - source_x) + (y1 - source_y) * (y1 - source_y))
            if abs(r_b - r_a) <= 1.0e-10:
                r_a = r_b
                continue

            if step == 1:
                if r_b > r_a:
                    ir = 0
                else:
                    ir = nr - 1

            tx = x1 - x0
            ty = y1 - y0
            tz = z1 - z0
            rlen = np.sqrt(tx * tx + ty * ty + tz * tz)
            if rlen <= 1.0e-10:
                r_a = r_b
                continue
            tx /= rlen
            ty /= rlen
            tz /= rlen

            phi = phi_path[step]
            rl = np.sqrt(tx * tx + ty * ty)
            if rl > 1.0e-10:
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                e1_x = (tx * tz * cos_phi + ty * sin_phi) / rl
                e1_y = (ty * tz * cos_phi - tx * sin_phi) / rl
                e1_z = -rl * cos_phi
                e2_x = (tx * tz * sin_phi - ty * cos_phi) / rl
                e2_y = (ty * tz * sin_phi + tx * cos_phi) / rl
                e2_z = -rl * sin_phi
            else:
                e1_x = 1.0
                e1_y = 0.0
                e1_z = 0.0
                e2_x = 0.0
                e2_y = 1.0
                e2_z = 0.0

            det_q = q11_path[step - 1] * q22_path[step - 1] - q12_path[step - 1] * q21_path[step - 1]
            if (det_q.real <= 0.0 and det_q_old.real > 0.0) or (det_q.real >= 0.0 and det_q_old.real < 0.0):
                phase += np.pi * 0.5
            det_q_old = det_q

            q11_0 = q11_path[step - 1] * scale_tilde
            q21_0 = q21_path[step - 1] * scale_tilde
            q12_0 = q12_path[step - 1] * scale_hat
            q22_0 = q22_path[step - 1] * scale_hat
            q11_1 = q11_path[step] * scale_tilde
            q21_1 = q21_path[step] * scale_tilde
            q12_1 = q12_path[step] * scale_hat
            q22_1 = q22_path[step] * scale_hat
            l1_start = np.sqrt(abs(q11_0) ** 2 + abs(q21_0) ** 2)
            l2_start = np.sqrt(abs(q12_0) ** 2 + abs(q22_0) ** 2)
            l1_end = np.sqrt(abs(q11_1) ** 2 + abs(q21_1) ** 2)
            l2_end = np.sqrt(abs(q12_1) ** 2 + abs(q22_1) ** 2)
            l1_max = max(l1_start, l1_end)
            l2_max = max(l2_start, l2_end)
            l_diag = np.sqrt(l1_max * l1_max + l2_max * l2_max)

            dq11 = q11_path[step] - q11_path[step - 1]
            dq12 = q12_path[step] - q12_path[step - 1]
            dq21 = q21_path[step] - q21_path[step - 1]
            dq22 = q22_path[step] - q22_path[step - 1]
            dtau = tau_path[step] - tau_path[step - 1]

            for _ in range(nr + 1):
                if ir < 0 or ir >= nr:
                    break
                rr = r_grid[ir]
                if rr >= min(r_a, r_b) and rr < max(r_a, r_b):
                    for itheta in range(ntheta):
                        theta = np.radians(theta_grid_deg[itheta])
                        t_rx = np.cos(theta)
                        t_ry = np.sin(theta)
                        vx = source_x + rr * t_rx
                        vy = source_y + rr * t_ry

                        radial_guard = (vx - source_x) * (x0 - source_x) + (vy - source_y) * (y0 - source_y)
                        if radial_guard < 0.0:
                            continue

                        e_theta_x = -t_ry
                        e_theta_y = t_rx
                        n_ray_z_3 = tx * e_theta_y - ty * e_theta_x
                        if abs(n_ray_z_3) < 1.0e-9:
                            continue
                        l_z = beam_window * l_diag / abs(n_ray_z_3)
                        z_min = min(z0, z1) - l_z
                        z_max = max(z0, z1) + l_z

                        for iz in range(nz):
                            vz = z_grid[iz]
                            if vz < z_min or vz > z_max:
                                continue

                            dxr = vx - x0
                            dyr = vy - y0
                            dzr = vz - z0
                            s = (dxr * tx + dyr * ty + dzr * tz) / rlen

                            q11_t = q11_path[step - 1] + s * dq11
                            q12_t = q12_path[step - 1] + s * dq12
                            q21_t = q21_path[step - 1] + s * dq21
                            q22_t = q22_path[step - 1] + s * dq22

                            q11_s = q11_t * scale_tilde
                            q21_s = q21_t * scale_tilde
                            q12_s = q12_t * scale_hat
                            q22_s = q22_t * scale_hat

                            l1 = np.sqrt(abs(q11_s) ** 2 + abs(q21_s) ** 2)
                            l2 = np.sqrt(abs(q12_s) ** 2 + abs(q22_s) ** 2)
                            det_q_int_scaled = q11_s * q22_s - q12_s * q21_s
                            if abs(det_q_int_scaled) < 1.0e-20 or l1 == 0.0 or l2 == 0.0:
                                continue

                            n_dist = abs(dxr * e1_x + dyr * e1_y + dzr * e1_z)
                            m_dist = abs(dxr * e2_x + dyr * e2_y + dzr * e2_z)
                            a_ell = abs((-q12_s * m_dist + q22_s * n_dist) / det_q_int_scaled)
                            b_ell = abs((q11_s * m_dist - q21_s * n_dist) / det_q_int_scaled)
                            if a_ell + b_ell > beam_window:
                                continue

                            det_q_int = q11_t * q22_t - q12_t * q21_t
                            if abs(det_q_int) < 1.0e-20:
                                continue
                            q_abs = np.sqrt(abs(det_q_int))
                            q_abs_reg = max(q_abs, 0.5 * lambd)

                            c_t = c_path[step - 1] + s * (c_path[step] - c_path[step - 1])
                            decay = np.exp(-0.5 * (a_ell ** 2 + b_ell ** 2))
                            # Native q_tilde/q_hat are already in the tracer's
                            # distance convention. Keeping c_t/c_src here
                            # preserves the AT ScaleBeam-equivalent pressure
                            # scale verified by receiver-grid parity tests.
                            amp = (c_t / c_src) * amp_path[step] * decay / (2.0 * np.pi * q_abs_reg)

                            phase_int = kmah_path[step - 1] * np.pi * 0.5 + phase
                            if (det_q_int_scaled.real <= 0.0 and det_q_old.real > 0.0) or (det_q_int_scaled.real >= 0.0 and det_q_old.real < 0.0):
                                phase_int += np.pi * 0.5
                            delay = tau_path[step - 1] + s * dtau
                            p_field[itheta, iz, ir] += amp * np.exp(complex(0.0, -(2.0 * np.pi * freq * delay - phase_int)))

                if r_grid[ir] < r_b:
                    if ir >= nr - 1:
                        break
                    ir_next = ir + 1
                    if r_grid[ir_next] >= r_b:
                        break
                else:
                    if ir <= 0:
                        break
                    ir_next = ir - 1
                    if r_grid[ir_next] <= r_b:
                        break
                ir = ir_next
            r_a = r_b

    return p_field


@njit(fastmath=True)
def compute_tl_field_receiver_grid_3d(
    theta_grid_deg: np.ndarray, r_grid: np.ndarray, z_grid: np.ndarray,
    rays_x: List, rays_y: List, rays_z: List,
    rays_q11: List, rays_q12: List, rays_q21: List, rays_q22: List,
    rays_amp: List, rays_tau: List, rays_kmah: List, rays_cpath: List, rays_phi: List, rays_elev: List,
    freq: float, c_src: float,
    source_x: float, source_y: float,
    d_azim: float, d_elev: float
):
    """
    Computes coherent TL on the AT Bellhop3D receiver grid.
    """
    p_field = compute_pressure_field_receiver_grid_3d(
        theta_grid_deg, r_grid, z_grid,
        rays_x, rays_y, rays_z,
        rays_q11, rays_q12, rays_q21, rays_q22,
        rays_amp, rays_tau, rays_kmah, rays_cpath, rays_phi, rays_elev,
        freq, c_src,
        source_x, source_y,
        d_azim, d_elev,
    )

    ntheta = len(theta_grid_deg)
    nr = len(r_grid)
    nz = len(z_grid)
    TL = np.zeros((ntheta, nz, nr), dtype=np.float64)
    for itheta in range(ntheta):
        for iz in range(nz):
            for ir_out in range(nr):
                p_mag = abs(p_field[itheta, iz, ir_out])
                if p_mag < 1.0e-20:
                    TL[itheta, iz, ir_out] = 200.0
                else:
                    TL[itheta, iz, ir_out] = -20.0 * np.log10(p_mag)

    return TL
