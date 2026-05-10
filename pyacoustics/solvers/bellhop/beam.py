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
    use_thorp: bool = False
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

        phase_caustic = 0.0
        q_old = ray_q[iray, 0]

        for i in range(1, npts - 1):  # P2: skip source segment (i=0) where q=0, matching AT L542
            r1 = ray_r[iray, i]
            z1 = ray_z[iray, i]
            r2 = ray_r[iray, i + 1]
            z2 = ray_z[iray, i + 1]

            q = ray_q[iray, i]
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
            dqds = ray_q[iray, i + 1] - ray_q[iray, i]
            dtau_r = ray_tau[iray, i + 1] - ray_tau[iray, i]
            dtau_i = 0.0  # real tau

            # Beam width at segment level (matching AT L562-565)
            c_seg = ray_c[iray, i]
            lam = c_seg / freq
            sigma_seg = max(abs(ray_q[iray, i]), abs(ray_q[iray, i + 1])) / q0
            if abs(rayt_r) > 1e-10:
                sigma_seg = sigma_seg / abs(rayt_r)  # project onto vertical
            tau_seg = ray_tau[iray, i + 1]
            if tau_seg < 0:
                tau_seg = 0.0
            sigma_seg = max(sigma_seg, min(0.2 * freq * tau_seg, pi * lam))
            radius_max = beam_window * sigma_seg

            # Depth limits (matching AT L568-574)
            z_lo = min(z1, z2) - radius_max
            z_hi = max(z1, z2) + radius_max

            iz_lo = int(max(0, (z_lo - z_min) / dz))
            iz_hi = int(min(n_z - 1, (z_hi - z_min) / dz)) + 1

            # Range bracket: which grid points fall in [min(r1,r2), max(r1,r2))?
            r_seg_min = min(r1, r2)
            r_seg_max = max(r1, r2)
            ir_lo = int(max(0, (r_seg_min - r_min) / dr))
            ir_hi = int(min(n_r - 1, (r_seg_max - r_min) / dr)) + 1

            for ir_g in range(ir_lo, ir_hi):
                r_rcv = r_grid[ir_g]
                if r_rcv < r_seg_min or r_rcv >= r_seg_max:
                    continue

                for iz_g in range(iz_lo, iz_hi):
                    z_rcv = z_grid[iz_g]

                    # AT L589-590: project receiver onto ray segment
                    dx_r = r_rcv - r1
                    dx_z = z_rcv - z1
                    s = (dx_r * rayt_r + dx_z * rayt_z) / rlen  # proportional dist
                    n_dist = abs(dx_r * rayn_r + dx_z * rayn_z)  # normal dist

                    # AT L591-593: interpolate q, compute sigma
                    q_interp = ray_q[iray, i] + s * dqds
                    sigma = abs(q_interp / q0)
                    # P1: use segment-level sigma_seg as floor for sigma1
                    sigma1 = max(sigma, sigma_seg)
                    sigma1 = max(sigma1, min(0.2 * freq * tau_seg, pi * lam))
                    if sigma1 < 1e-12:
                        sigma1 = pi * lam

                    # AT L598: beam window check
                    if n_dist >= beam_window * sigma1:
                        continue

                    # AT L599: interpolated delay
                    tau_interp = ray_tau[iray, i] + s * dtau_r

                    # AT L604: amplitude constant
                    c_interp = ray_c[iray, i] + s * (ray_c[iray, i + 1] - ray_c[iray, i])
                    const = ratio1 * np.sqrt(c_interp / (q0 * sigma1))

                    # Interpolate reflection amplitude
                    amp_r = ray_amp[iray, i].real + s * (ray_amp[iray, i + 1].real - ray_amp[iray, i].real)
                    amp_i = ray_amp[iray, i].imag + s * (ray_amp[iray, i + 1].imag - ray_amp[iray, i].imag)

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

                    # AT L651: coherent contribution
                    phase_total = omega * tau_interp - phase_int
                    p_real = beam_amp * np.cos(phase_total)
                    p_imag = -beam_amp * np.sin(phase_total)

                    # Apply reflection amplitude (complex multiply)
                    pr = p_real * amp_r - p_imag * amp_i
                    pi_val = p_real * amp_i + p_imag * amp_r

                    # AT ScalePressure L797: const/sqrt(r) where const=-1
                    if r_rcv > 1.0:
                        cyl = 1.0 / np.sqrt(r_rcv)
                    else:
                        cyl = 1.0

                    field_real[ir_g, iz_g] += pr * cyl
                    field_imag[ir_g, iz_g] += pi_val * cyl

    # Convert to TL
    TL = np.zeros((n_r, n_z))
    for ir in range(n_r):
        for iz in range(n_z):
            intensity = field_real[ir, iz] ** 2 + field_imag[ir, iz] ** 2
            if intensity < 1e-30:
                TL[ir, iz] = 120.0
            else:
                TL[ir, iz] = -10.0 * np.log10(intensity)

    return TL
