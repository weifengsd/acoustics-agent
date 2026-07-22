"""
Gaussian Beam Summation (Geometric Gaussian Beams) for coherent TL.
Scatter approach matching AT Bellhop's InfluenceGeoGaussianCart (influence.f90 L510-635).

Key AT formulas (for point source, RunType 'R'):
  Ratio1 = sqrt(|cos(alpha)|) / sqrt(2*pi)
  q0 = c_src / Dalpha
  sigma = |q / q0|
  sigma1 = max(sigma, min(0.2*freq*tau, pi*lambda))
  const = Ratio1 * sqrt(c / (q0 * sigma1)) * Amp
  W = sqrt(sigma / sigma1) * exp(-0.5 * (n/sigma1)^2)
  Amp_field = const * W
  
Post-scaling (ScalePressure): U *= -1.0 / sqrt(r)  for Geometric beams
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def compute_tl_field_coherent(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    use_thorp: bool = False,
    line_source: bool = False,
) -> np.ndarray:
    field_real, field_imag, field_incoh = compute_gaussian_fields(
        r_grid, z_grid,
        ray_r, ray_z,
        ray_c, ray_tau,
        ray_q,
        ray_amp,
        ray_npts,
        ray_alphas,
        freq, c_src,
        d_alpha,
        True,
        use_thorp,
        False,
        0.0,
        line_source,
    )
    return gaussian_fields_to_tl(r_grid, field_real, field_imag, field_incoh, True, line_source)


@njit(fastmath=True)
def compute_pressure_field_coherent(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    use_thorp: bool = False,
    line_source: bool = False,
) -> np.ndarray:
    field_real, field_imag, _ = compute_gaussian_fields(
        r_grid, z_grid,
        ray_r, ray_z,
        ray_c, ray_tau,
        ray_q,
        ray_amp,
        ray_npts,
        ray_alphas,
        freq, c_src,
        d_alpha,
        True,
        use_thorp,
        False,
        0.0,
        line_source,
    )
    return field_real + 1j * field_imag


@njit(fastmath=True)
def compute_tl_field_hat_coherent(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    use_thorp: bool = False,
    line_source: bool = False,
) -> np.ndarray:
    pressure = compute_hat_pressure_field_coherent(
        r_grid, z_grid,
        ray_r, ray_z,
        ray_c, ray_tau,
        ray_q,
        ray_amp,
        ray_npts,
        ray_alphas,
        freq, c_src,
        d_alpha,
        use_thorp,
        line_source,
    )
    p_abs = np.abs(pressure)
    TL = np.zeros((len(r_grid), len(z_grid)))
    for ir in range(len(r_grid)):
        for iz in range(len(z_grid)):
            if p_abs[ir, iz] < 1.0e-15:
                TL[ir, iz] = 120.0
            else:
                TL[ir, iz] = -20.0 * np.log10(p_abs[ir, iz])
    return TL


@njit(fastmath=True)
def compute_hat_pressure_field_coherent(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    use_thorp: bool = False,
    line_source: bool = False,
) -> np.ndarray:
    """Compute AT ``InfluenceGeoHatCart`` style coherent pressure."""

    n_r = len(r_grid)
    n_z = len(z_grid)
    n_rays = len(ray_npts)
    omega = 2.0 * np.pi * freq
    freq_khz = freq / 1000.0
    pi = np.pi

    pressure = np.zeros((n_r, n_z), dtype=np.complex128)
    q0 = c_src / d_alpha

    alpha_thorp = 0.0
    if use_thorp:
        f2 = freq_khz * freq_khz
        alpha_thorp = 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 0.000275 * f2 + 0.003

    for iray in range(n_rays):
        npts = ray_npts[iray]
        if npts < 2:
            continue

        alpha_rad = ray_alphas[iray]
        ratio1 = np.sqrt(abs(np.cos(alpha_rad))) if not line_source else 1.0
        phase_caustic = 0.0
        q_old = ray_q[iray, 0]
        r_a = ray_r[iray, 0]

        for i in range(1, npts):
            r_b = ray_r[iray, i]
            r1 = ray_r[iray, i - 1]
            z1 = ray_z[iray, i - 1]
            r2 = ray_r[iray, i]
            z2 = ray_z[iray, i]

            dr_seg = r2 - r1
            dz_seg = z2 - z1
            rlen_sq = dr_seg * dr_seg + dz_seg * dz_seg
            if rlen_sq < 1.0e-30:
                continue
            rlen = np.sqrt(rlen_sq)
            rayt_r = dr_seg / rlen
            rayt_z = dz_seg / rlen
            rayn_r = -rayt_z
            rayn_z = rayt_r

            dqds = ray_q[iray, i] - ray_q[iray, i - 1]
            dtau = ray_tau[iray, i] - ray_tau[iray, i - 1]
            q = ray_q[iray, i - 1]
            if (q <= 0.0 and q_old > 0.0) or (q >= 0.0 and q_old < 0.0):
                phase_caustic += pi / 2.0
            q_old = q

            if abs(rayt_r) < 1.0e-12:
                continue

            radius_max = max(abs(ray_q[iray, i - 1]), abs(ray_q[iray, i])) / q0 / abs(rayt_r)
            if radius_max <= 0.0:
                continue

            if abs(rayt_r) > 0.5:
                z_lo = min(z1, z2) - radius_max
                z_hi = max(z1, z2) + radius_max
            else:
                z_lo = z_grid[0]
                z_hi = z_grid[-1]

            r_seg_min = min(r_a, r_b)
            r_seg_max = max(r_a, r_b)

            for ir_g in range(n_r):
                r_rcv = r_grid[ir_g]
                if r_rcv < r_seg_min or r_rcv >= r_seg_max:
                    continue

                for iz_g in range(n_z):
                    z_rcv = z_grid[iz_g]
                    if z_rcv < z_lo or z_rcv > z_hi:
                        continue

                    dx_r = r_rcv - r1
                    dx_z = z_rcv - z1
                    s = (dx_r * rayt_r + dx_z * rayt_z) / rlen
                    n_dist = abs(dx_r * rayn_r + dx_z * rayn_z)
                    q_interp = ray_q[iray, i - 1] + s * dqds
                    radius = abs(q_interp / q0)
                    if radius <= 0.0 or n_dist >= radius:
                        continue

                    tau_interp = ray_tau[iray, i - 1] + s * dtau
                    const = ratio1 * np.sqrt(ray_c[iray, i] / abs(q_interp))
                    w_hat = (radius - n_dist) / radius
                    amp_r = ray_amp[iray, i].real
                    amp_i = ray_amp[iray, i].imag

                    beam_amp = const * w_hat
                    if use_thorp:
                        absorption_gain = 10.0 ** (-alpha_thorp * r_rcv / 20000.0)
                        beam_amp *= absorption_gain

                    phase_int = phase_caustic
                    if (q_interp <= 0.0 and q_old > 0.0) or (q_interp >= 0.0 and q_old < 0.0):
                        phase_int += pi / 2.0
                    phase_total = omega * tau_interp - phase_int

                    p_real = beam_amp * np.cos(phase_total)
                    p_imag = -beam_amp * np.sin(phase_total)
                    pr = p_real * amp_r - p_imag * amp_i
                    pi_val = p_real * amp_i + p_imag * amp_r

                    if line_source:
                        scale = 4.0 * np.sqrt(pi)
                    elif abs(r_rcv) > 1.0:
                        scale = 1.0 / np.sqrt(abs(r_rcv))
                    else:
                        scale = 0.0
                    pressure[ir_g, iz_g] += (pr + 1j * pi_val) * scale

            r_a = r_b

    return pressure


@njit(fastmath=True)
def compute_tl_field_incoherent(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    use_thorp: bool = False,
    line_source: bool = False,
) -> np.ndarray:
    field_real, field_imag, field_incoh = compute_gaussian_fields(
        r_grid, z_grid,
        ray_r, ray_z,
        ray_c, ray_tau,
        ray_q,
        ray_amp,
        ray_npts,
        ray_alphas,
        freq, c_src,
        d_alpha,
        False,
        use_thorp,
        False,
        0.0,
        line_source,
    )
    return gaussian_fields_to_tl(r_grid, field_real, field_imag, field_incoh, False, line_source)


@njit(fastmath=True)
def compute_tl_field_semicoherent(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    source_depth: float,
    use_thorp: bool = False,
    line_source: bool = False,
) -> np.ndarray:
    field_real, field_imag, field_incoh = compute_gaussian_fields(
        r_grid, z_grid,
        ray_r, ray_z,
        ray_c, ray_tau,
        ray_q,
        ray_amp,
        ray_npts,
        ray_alphas,
        freq, c_src,
        d_alpha,
        False,
        use_thorp,
        True,
        source_depth,
        line_source,
    )
    return gaussian_fields_to_tl(r_grid, field_real, field_imag, field_incoh, False, line_source)


@njit(fastmath=True)
def compute_tl_field_simple_gaussian_coherent(
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    ray_r: np.ndarray,
    ray_z: np.ndarray,
    ray_tau: np.ndarray,
    ray_amp: np.ndarray,
    ray_phase: np.ndarray,
    ray_q: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float,
    c_src: float,
    d_alpha: float,
    step_size: float,
    line_source: bool = False,
) -> np.ndarray:
    """Compute AT ``InfluenceSGB`` style coherent simple Gaussian beams."""

    n_r = len(r_grid)
    n_z = len(z_grid)
    n_rays = len(ray_npts)
    omega = 2.0 * np.pi * freq
    beta = 0.98
    a_beam = -4.0 * np.log(beta) / (d_alpha * d_alpha)
    cn = d_alpha * np.sqrt(a_beam / np.pi)
    pressure = np.zeros((n_r, n_z), dtype=np.complex128)

    for iray in range(n_rays):
        npts = ray_npts[iray]
        if npts < 2:
            continue

        alpha = ray_alphas[iray]
        ratio1 = 1.0 if line_source else np.sqrt(np.cos(alpha))
        phase = 0.0
        q_old = 1.0
        r_a = ray_r[iray, 0]

        for i in range(1, npts):
            r_b = ray_r[iray, i]
            q_prev = ray_q[iray, i - 1]
            if (q_prev < 0.0 and q_old >= 0.0) or (q_prev > 0.0 and q_old <= 0.0):
                phase += np.pi / 2.0
            q_old = q_prev

            if abs(r_b - r_a) <= 1000.0 * np.spacing(r_a):
                r_a = r_b
                continue

            r_seg_min = min(r_a, r_b)
            r_seg_max = max(r_a, r_b)
            for ir_g in range(n_r):
                r_rcv = r_grid[ir_g]
                if r_rcv < r_seg_min or r_rcv > r_seg_max or r_b <= r_rcv:
                    continue

                w = (r_rcv - r_a) / (r_b - r_a)
                x_z = ray_z[iray, i - 1] + w * (ray_z[iray, i] - ray_z[iray, i - 1])
                q_interp = ray_q[iray, i - 1] + w * (ray_q[iray, i] - ray_q[iray, i - 1])
                tau_interp = ray_tau[iray, i - 1] + w * (ray_tau[iray, i] - ray_tau[iray, i - 1])
                sint = i * step_size + w * step_size

                if (q_interp < 0.0 and q_old >= 0.0) or (q_interp > 0.0 and q_old <= 0.0):
                    phase += np.pi / 2.0

                dr_seg = ray_r[iray, i] - ray_r[iray, i - 1]
                dz_seg = ray_z[iray, i] - ray_z[iray, i - 1]
                seg_len = np.sqrt(dr_seg * dr_seg + dz_seg * dz_seg)
                if seg_len <= 0.0:
                    continue
                rayt_z = dz_seg / seg_len

                for iz_g in range(n_z):
                    delta_z = z_grid[iz_g] - x_z
                    cpa = abs(delta_z * (r_b - r_a)) / seg_len
                    ds_sq = delta_z * delta_z - cpa * cpa
                    ds = np.sqrt(max(0.0, ds_sq))
                    sx1 = sint + ds
                    if sx1 <= 0.0:
                        continue

                    theta = np.arctan(cpa / sx1)
                    delay = tau_interp + rayt_z * delta_z / c_src
                    phase_total = omega * delay - ray_phase[iray, i] - phase
                    contrib = ratio1 * cn * ray_amp[iray, i]
                    contrib *= np.exp(-a_beam * theta * theta - 1j * phase_total) / np.sqrt(sx1)
                    pressure[ir_g, iz_g] += contrib

                q_old = q_interp
            r_a = r_b

    tl = np.zeros((n_r, n_z))
    for ir_g in range(n_r):
        r_rcv = abs(r_grid[ir_g])
        if line_source:
            scale = 4.0 * np.sqrt(np.pi)
        elif r_rcv > 0.0:
            scale = -1.0 / np.sqrt(r_rcv)
        else:
            scale = 0.0
        for iz_g in range(n_z):
            p_abs = abs(scale * pressure[ir_g, iz_g])
            if p_abs < 1.0e-15:
                tl[ir_g, iz_g] = 120.0
            else:
                tl[ir_g, iz_g] = -20.0 * np.log10(p_abs)
    return tl


@njit(fastmath=True)
def compute_tl_field_hat_incoherent(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    use_thorp: bool = False,
    line_source: bool = False,
) -> np.ndarray:
    field_incoh = compute_hat_intensity_field(
        r_grid, z_grid,
        ray_r, ray_z,
        ray_c, ray_tau,
        ray_q,
        ray_amp,
        ray_npts,
        ray_alphas,
        freq, c_src,
        d_alpha,
        use_thorp,
        False,
        0.0,
        line_source,
    )
    return intensity_field_to_tl(r_grid, field_incoh, line_source)


@njit(fastmath=True)
def compute_tl_field_hat_semicoherent(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    source_depth: float,
    use_thorp: bool = False,
    line_source: bool = False,
) -> np.ndarray:
    field_incoh = compute_hat_intensity_field(
        r_grid, z_grid,
        ray_r, ray_z,
        ray_c, ray_tau,
        ray_q,
        ray_amp,
        ray_npts,
        ray_alphas,
        freq, c_src,
        d_alpha,
        use_thorp,
        True,
        source_depth,
        line_source,
    )
    return intensity_field_to_tl(r_grid, field_incoh, line_source)


@njit(fastmath=True)
def compute_hat_intensity_field(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    use_thorp: bool = False,
    semicoherent: bool = False,
    source_depth: float = 0.0,
    line_source: bool = False,
) -> np.ndarray:
    n_r = len(r_grid)
    n_z = len(z_grid)
    n_rays = len(ray_npts)
    omega = 2.0 * np.pi * freq
    freq_khz = freq / 1000.0

    field_incoh = np.zeros((n_r, n_z))
    q0 = c_src / d_alpha

    alpha_thorp = 0.0
    if use_thorp:
        f2 = freq_khz * freq_khz
        alpha_thorp = 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 0.000275 * f2 + 0.003

    for iray in range(n_rays):
        npts = ray_npts[iray]
        if npts < 2:
            continue

        alpha_rad = ray_alphas[iray]
        ratio1 = 1.0 if line_source else np.sqrt(abs(np.cos(alpha_rad)))
        source_factor = 1.0
        if semicoherent:
            source_factor = np.sqrt(2.0) * abs(np.sin(omega * source_depth * np.sin(alpha_rad) / c_src))

        q_old = ray_q[iray, 0]
        r_a = ray_r[iray, 0]

        for i in range(1, npts):
            r_b = ray_r[iray, i]
            r1 = ray_r[iray, i - 1]
            z1 = ray_z[iray, i - 1]
            r2 = ray_r[iray, i]
            z2 = ray_z[iray, i]

            dr_seg = r2 - r1
            dz_seg = z2 - z1
            rlen_sq = dr_seg * dr_seg + dz_seg * dz_seg
            if rlen_sq < 1e-30:
                continue
            rlen = np.sqrt(rlen_sq)
            rayt_r = dr_seg / rlen
            rayt_z = dz_seg / rlen
            rayn_r = -rayt_z
            rayn_z = rayt_r

            dqds = ray_q[iray, i] - ray_q[iray, i - 1]
            q = ray_q[iray, i - 1]
            q_old = q

            if abs(rayt_r) < 1e-12:
                continue

            radius_max = max(abs(ray_q[iray, i - 1]), abs(ray_q[iray, i])) / q0 / abs(rayt_r)
            if radius_max <= 0.0:
                continue

            if abs(rayt_r) > 0.5:
                z_lo = min(z1, z2) - radius_max
                z_hi = max(z1, z2) + radius_max
            else:
                z_lo = z_grid[0]
                z_hi = z_grid[-1]

            r_seg_min = min(r_a, r_b)
            r_seg_max = max(r_a, r_b)

            for ir_g in range(n_r):
                r_rcv = r_grid[ir_g]
                if r_rcv < r_seg_min or r_rcv >= r_seg_max:
                    continue

                for iz_g in range(n_z):
                    z_rcv = z_grid[iz_g]
                    if z_rcv < z_lo or z_rcv > z_hi:
                        continue

                    dx_r = r_rcv - r1
                    dx_z = z_rcv - z1
                    s = (dx_r * rayt_r + dx_z * rayt_z) / rlen
                    n_dist = abs(dx_r * rayn_r + dx_z * rayn_z)
                    q_interp = ray_q[iray, i - 1] + s * dqds
                    radius = abs(q_interp / q0)
                    if radius <= 0.0 or n_dist >= radius:
                        continue

                    amp_abs_sq = ray_amp[iray, i].real * ray_amp[iray, i].real
                    amp_abs_sq += ray_amp[iray, i].imag * ray_amp[iray, i].imag
                    amp_abs_sq *= source_factor * source_factor
                    const_sq = ratio1 * ratio1 * ray_c[iray, i] / abs(q_interp) * amp_abs_sq
                    w_hat = (radius - n_dist) / radius

                    if use_thorp:
                        absorption_gain = 10.0 ** (-alpha_thorp * r_rcv / 20000.0)
                        const_sq *= absorption_gain * absorption_gain

                    field_incoh[ir_g, iz_g] += const_sq * w_hat

            r_a = r_b

    return field_incoh


@njit(fastmath=True)
def compute_gaussian_fields(
    r_grid: np.ndarray, z_grid: np.ndarray,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_npts: np.ndarray,
    ray_alphas: np.ndarray,
    freq: float, c_src: float,
    d_alpha: float,
    coherent: bool,
    use_thorp: bool = False,
    semicoherent: bool = False,
    source_depth: float = 0.0,
    line_source: bool = False,
) -> np.ndarray:
    n_r = len(r_grid)
    n_z = len(z_grid)
    n_rays = len(ray_npts)
    omega = 2.0 * np.pi * freq
    freq_khz = freq / 1000.0
    pi = np.pi
    beam_window = 4  # AT uses BeamWindow = 4

    dr = (r_grid[-1] - r_grid[0]) / max(n_r - 1, 1)
    dz = (z_grid[-1] - z_grid[0]) / max(n_z - 1, 1)
    r_min = r_grid[0]
    z_min = z_grid[0]

    field_real = np.zeros((n_r, n_z))
    field_imag = np.zeros((n_r, n_z))
    field_incoh = np.zeros((n_r, n_z))

    q0 = c_src / d_alpha

    # Pre-calculate Thorp alpha if needed
    alpha_thorp = 0.0
    if use_thorp:
        f2 = freq_khz * freq_khz
        alpha_thorp = 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 0.000275 * f2 + 0.003

    for iray in range(n_rays):
        npts = ray_npts[iray]
        if npts < 2:
            continue

        alpha_rad = ray_alphas[iray]
        ratio1 = np.sqrt(abs(np.cos(alpha_rad))) / np.sqrt(2.0 * pi)
        source_factor = 1.0
        if semicoherent:
            source_factor = np.sqrt(2.0) * abs(np.sin(omega * source_depth * np.sin(alpha_rad) / c_src))

        phase_caustic = 0.0
        q_old = ray_q[iray, 0]

        for i in range(1, npts):
            r1 = ray_r[iray, i - 1]
            z1 = ray_z[iray, i - 1]
            r2 = ray_r[iray, i]
            z2 = ray_z[iray, i]

            q = ray_q[iray, i - 1]
            if (q <= 0.0 and q_old > 0.0) or (q >= 0.0 and q_old < 0.0):
                phase_caustic += pi / 2.0
            q_old = q

            # Ray segment vector
            dr_seg = r2 - r1
            dz_seg = z2 - z1
            rlen_sq = dr_seg * dr_seg + dz_seg * dz_seg
            if rlen_sq < 1e-30:
                continue
            rlen = np.sqrt(rlen_sq)
            
            # Unit tangent and normal (matching AT L550-551)
            rayt_r = dr_seg / rlen
            rayt_z = dz_seg / rlen
            rayn_r = -rayt_z
            rayn_z = rayt_r

            # Segment quantities for interpolation (matching AT L554-555)
            dqds = ray_q[iray, i] - ray_q[iray, i - 1]
            dtau_r = ray_tau[iray, i] - ray_tau[iray, i - 1]
            dtau_i = 0.0  # real tau

            # Beam width at segment level (matching AT L562-565)
            c_seg = ray_c[iray, i - 1]
            lam = c_seg / freq
            sigma_seg = max(abs(ray_q[iray, i - 1]), abs(ray_q[iray, i])) / q0
            if abs(rayt_r) > 1e-10:
                sigma_seg = sigma_seg / abs(rayt_r)  # project onto vertical
            tau_seg = ray_tau[iray, i]
            if tau_seg < 0:
                tau_seg = 0.0
            sigma_seg = max(sigma_seg, min(0.2 * freq * tau_seg, pi * lam))
            radius_max = beam_window * sigma_seg

            # Depth limits (matching AT L568-574)
            if abs(rayt_r) > 0.5:
                z_lo = min(z1, z2) - radius_max
                z_hi = max(z1, z2) + radius_max
            else:
                z_lo = z_grid[0]
                z_hi = z_grid[-1]

            if n_z == 1:
                iz_lo = 0
                iz_hi = 1
            else:
                iz_lo = int(max(0, (z_lo - z_min) / dz))
                iz_hi = int(min(n_z - 1, (z_hi - z_min) / dz)) + 1

            # Range bracket: which grid points fall in [min(r1,r2), max(r1,r2))?
            r_seg_min = min(r1, r2)
            r_seg_max = max(r1, r2)
            if n_r == 1:
                ir_lo = 0
                ir_hi = 1
            else:
                ir_lo = int(max(0, (r_seg_min - r_min) / dr))
                ir_hi = int(min(n_r - 1, (r_seg_max - r_min) / dr)) + 1

            for ir_g in range(ir_lo, ir_hi):
                r_rcv = r_grid[ir_g]
                if r_rcv < r_seg_min or r_rcv > r_seg_max:
                    continue
                if r_rcv == r_seg_max and i < npts - 1:
                    continue

                for iz_g in range(iz_lo, iz_hi):
                    z_rcv = z_grid[iz_g]

                    # AT L589-590: project receiver onto ray segment
                    dx_r = r_rcv - r1
                    dx_z = z_rcv - z1
                    s = (dx_r * rayt_r + dx_z * rayt_z) / rlen  # proportional dist
                    n_dist = abs(dx_r * rayn_r + dx_z * rayn_z)  # normal dist

                    # AT L591-593: interpolate q, compute sigma
                    q_interp = ray_q[iray, i - 1] + s * dqds
                    sigma = abs(q_interp / q0)
                    sigma1 = max(sigma, min(0.2 * freq * tau_seg, pi * lam))
                    if sigma1 < 1e-12:
                        sigma1 = pi * lam

                    # AT L598: beam window check
                    if n_dist >= beam_window * sigma1:
                        continue

                    # AT L599: interpolated delay
                    tau_interp = ray_tau[iray, i - 1] + s * dtau_r

                    # AT L604: amplitude constant
                    const = ratio1 * np.sqrt(ray_c[iray, i] / (q0 * sigma1))

                    amp_r = ray_amp[iray, i].real
                    amp_i = ray_amp[iray, i].imag

                    # AT L605: Gaussian decay with sigma ratio
                    W = np.sqrt(sigma / sigma1) if sigma > 0 else 0.0
                    W *= np.exp(-0.5 * (n_dist / sigma1) ** 2)

                    beam_amp = const * W
                    
                    # Apply Thorp Absorption gain
                    if use_thorp:
                        # alpha is dB/km, convert to linear gain for pressure
                        # gain = 10 ^ (-alpha * r_km / 20)
                        absorption_gain = 10.0 ** (-alpha_thorp * r_rcv / 20000.0)
                        beam_amp *= absorption_gain

                    # AT L610: phase
                    phase_int = phase_caustic
                    # Check for additional caustic at interpolated point
                    if (q_interp <= 0.0 and q_old > 0.0) or (q_interp >= 0.0 and q_old < 0.0):
                        phase_int += pi / 2.0

                    if coherent:
                        # AT L651: coherent contribution
                        phase_total = omega * tau_interp - phase_int
                        p_real = beam_amp * np.cos(phase_total)
                        p_imag = -beam_amp * np.sin(phase_total)

                        # Apply reflection amplitude (complex multiply)
                        pr = p_real * amp_r - p_imag * amp_i
                        pi_val = p_real * amp_i + p_imag * amp_r

                        # AT ScalePressure uses 1/sqrt(r) for point sources and
                        # 4*sqrt(pi) for line sources.
                        if line_source:
                            cyl = 4.0 * np.sqrt(pi)
                        elif r_rcv > 1.0:
                            cyl = 1.0 / np.sqrt(r_rcv)
                        else:
                            cyl = 1.0

                        field_real[ir_g, iz_g] += pr * cyl
                        field_imag[ir_g, iz_g] += pi_val * cyl
                    else:
                        amp_abs_sq = amp_r * amp_r + amp_i * amp_i
                        amp_abs_sq *= source_factor * source_factor
                        field_incoh[ir_g, iz_g] += np.sqrt(2.0 * pi) * const * const * amp_abs_sq * W

    return field_real, field_imag, field_incoh


@njit(fastmath=True)
def gaussian_fields_to_tl(
    r_grid: np.ndarray,
    field_real: np.ndarray,
    field_imag: np.ndarray,
    field_incoh: np.ndarray,
    coherent: bool,
    line_source: bool = False,
) -> np.ndarray:
    n_r, n_z = field_real.shape
    TL = np.zeros((n_r, n_z))
    for ir in range(n_r):
        for iz in range(n_z):
            if coherent:
                intensity = field_real[ir, iz] ** 2 + field_imag[ir, iz] ** 2
            else:
                r_rcv = abs(r_grid[ir])
                if line_source:
                    line_scale = 4.0 * np.sqrt(np.pi)
                    intensity = field_incoh[ir, iz] * line_scale * line_scale
                elif r_rcv > 1.0:
                    intensity = field_incoh[ir, iz] / r_rcv
                else:
                    intensity = 0.0
            if intensity < 1e-30:
                TL[ir, iz] = 120.0
            else:
                TL[ir, iz] = -10.0 * np.log10(intensity)

    return TL


@njit(fastmath=True)
def intensity_field_to_tl(
    r_grid: np.ndarray,
    field_incoh: np.ndarray,
    line_source: bool = False,
) -> np.ndarray:
    n_r, n_z = field_incoh.shape
    TL = np.zeros((n_r, n_z))
    for ir in range(n_r):
        r_rcv = abs(r_grid[ir])
        for iz in range(n_z):
            if line_source:
                line_scale = 4.0 * np.sqrt(np.pi)
                intensity = field_incoh[ir, iz] * line_scale * line_scale
            elif r_rcv > 1.0:
                intensity = field_incoh[ir, iz] / r_rcv
            else:
                intensity = 0.0
            if intensity < 1e-30:
                TL[ir, iz] = 120.0
            else:
                TL[ir, iz] = -10.0 * np.log10(intensity)
    return TL
