import numpy as np
import scipy.linalg


def _default_media(n_points: int, medium_rho: float):
    return (
        np.array([0], dtype=np.int64),
        np.array([n_points - 1], dtype=np.int64),
        np.array([medium_rho], dtype=np.float64),
    )


def _media_density(n_points: int, media_start, media_end, media_rho):
    density = np.ones(n_points, dtype=np.float64)
    for start, end, rho in zip(media_start, media_end, media_rho):
        density[int(start) : int(end) + 1] = float(rho)
    return density


def _build_inverse_iteration_matrix(
    k2,
    h,
    B1,
    omega,
    bot_bc_type,
    bot_c_p,
    bot_rho,
    top_bc_type,
    media_start,
    media_end,
    media_rho,
    bot_c_s,
):
    from .core import bottom_impedance, top_impedance

    n_points = len(B1)
    f_bot, g_bot = bottom_impedance(k2, omega, bot_bc_type, bot_c_p, bot_rho, bot_c_s)
    f_top, g_top = top_impedance(top_bc_type)

    diag_sum = np.zeros(n_points, dtype=np.float64)
    diag_count = np.zeros(n_points, dtype=np.float64)
    upper = np.zeros(n_points - 1, dtype=np.float64)
    lower = np.zeros(n_points - 1, dtype=np.float64)
    h2k2 = h**2 * k2

    for start, end, rho in zip(media_start, media_end, media_rho):
        start = int(start)
        end = int(end)
        h_rho = h * float(rho)
        diag_sum[start : end + 1] += (B1[start : end + 1] - h2k2) / h_rho
        diag_count[start : end + 1] += 1.0

        if end > start:
            coeff = 1.0 / h_rho
            upper[start:end] = coeff
            lower[start:end] = coeff

    diag_count = np.where(diag_count == 0.0, 1.0, diag_count)
    diag = diag_sum / diag_count

    if g_top == 0.0:
        diag[0] = 1.0
        if n_points > 1:
            upper[0] = 0.0
            lower[0] = 0.0
    else:
        diag[0] = 0.5 * diag[0] + f_top / g_top

    if g_bot == 0.0:
        diag[-1] = 1.0
        if n_points > 1:
            upper[-1] = 0.0
            lower[-1] = 0.0
    else:
        diag[-1] = 0.5 * diag[-1] - f_bot / g_bot

    ab = np.zeros((3, n_points), dtype=np.float64)
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower
    return ab, g_top, g_bot


def _build_inverse_iteration_matrix_complex(
    k2,
    h,
    B1,
    omega,
    bot_bc_type,
    bot_c_p,
    bot_rho,
    top_bc_type,
    media_start,
    media_end,
    media_rho,
    bot_c_s,
    bot_attenuation_p,
    bot_attenuation_s,
    c_top_inside=None,
    rho_top_inside=None,
    c_bot_inside=None,
    rho_bot_inside=None,
    top_reflection_table=None,
    bot_reflection_table=None,
):
    from .core import bottom_impedance_complex_with_table, top_impedance_complex

    n_points = len(B1)
    if c_top_inside is None:
        c_top_inside = omega * h / np.sqrt(2.0 + B1[0])
    if rho_top_inside is None:
        rho_top_inside = float(media_rho[0])
    if c_bot_inside is None:
        c_bot_inside = omega * h / np.sqrt(2.0 + B1[-1])
    if rho_bot_inside is None:
        rho_bot_inside = float(media_rho[-1])

    f_bot, g_bot = bottom_impedance_complex_with_table(
        k2,
        omega,
        bot_bc_type,
        bot_c_p,
        bot_rho,
        bot_c_s,
        bot_attenuation_p,
        bot_attenuation_s,
        c_bot_inside,
        rho_bot_inside,
        bot_reflection_table,
    )
    f_top, g_top = top_impedance_complex(
        k2, omega, top_bc_type, c_top_inside, rho_top_inside, top_reflection_table
    )

    diag_sum = np.zeros(n_points, dtype=np.complex128)
    diag_count = np.zeros(n_points, dtype=np.float64)
    upper = np.zeros(n_points - 1, dtype=np.complex128)
    lower = np.zeros(n_points - 1, dtype=np.complex128)
    h2k2 = h**2 * k2

    for start, end, rho in zip(media_start, media_end, media_rho):
        start = int(start)
        end = int(end)
        h_rho = h * float(rho)
        diag_sum[start : end + 1] += (B1[start : end + 1] - h2k2) / h_rho
        diag_count[start : end + 1] += 1.0

        if end > start:
            coeff = 1.0 / h_rho
            upper[start:end] = coeff
            lower[start:end] = coeff

    diag_count = np.where(diag_count == 0.0, 1.0, diag_count)
    diag = diag_sum / diag_count

    if g_top == 0.0:
        diag[0] = 1.0
        if n_points > 1:
            upper[0] = 0.0
            lower[0] = 0.0
    else:
        diag[0] = 0.5 * diag[0] + f_top / g_top

    if g_bot == 0.0:
        diag[-1] = 1.0
        if n_points > 1:
            upper[-1] = 0.0
            lower[-1] = 0.0
    else:
        diag[-1] = 0.5 * diag[-1] - f_bot / g_bot

    ab = np.zeros((3, n_points), dtype=np.complex128)
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower
    return ab, g_top, g_bot


def _normalize_mode(phi, h, media_density):
    integrand = phi**2 / media_density
    integral = h * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])
    if abs(integral) > 0.0:
        phi = phi / np.sqrt(integral)

    idx_max = np.argmax(np.abs(phi))
    if np.abs(phi[idx_max]) > 0.0:
        phi = phi * np.exp(-1j * np.angle(phi[idx_max]))
        if np.real(phi[idx_max]) < 0.0:
            phi = -phi
    return phi


def _normalize_complex_mode_with_boundary(
    phi,
    h,
    media_density,
    k2,
    omega,
    bot_bc_type,
    bot_c_p,
    bot_rho,
    top_bc_type,
    bot_c_s,
    bot_attenuation_p,
    bot_attenuation_s,
    c_top_inside,
    rho_top_inside,
    c_bot_inside,
    rho_bot_inside,
    top_reflection_table,
    bot_reflection_table,
):
    from .core import bottom_impedance_complex_with_table, top_impedance_complex

    integrand = phi**2 / media_density
    integral = h * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])
    delta = 1.0e-7 * k2 if abs(k2) > 0.0 else 1.0e-7
    k2_1 = k2 - delta
    k2_2 = k2 + delta

    drho_dx = 0.0 + 0.0j
    f_top1, g_top1 = top_impedance_complex(
        k2_1, omega, top_bc_type, c_top_inside, rho_top_inside, top_reflection_table
    )
    f_top2, g_top2 = top_impedance_complex(
        k2_2, omega, top_bc_type, c_top_inside, rho_top_inside, top_reflection_table
    )
    if abs(g_top1) > 0.0 and abs(g_top2) > 0.0:
        drho_dx = (f_top2 / g_top2 - f_top1 / g_top1) / (k2_2 - k2_1)

    deta_dx = 0.0 + 0.0j
    f_bot1, g_bot1 = bottom_impedance_complex_with_table(
        k2_1,
        omega,
        bot_bc_type,
        bot_c_p,
        bot_rho,
        bot_c_s,
        bot_attenuation_p,
        bot_attenuation_s,
        c_bot_inside,
        rho_bot_inside,
        bot_reflection_table,
    )
    f_bot2, g_bot2 = bottom_impedance_complex_with_table(
        k2_2,
        omega,
        bot_bc_type,
        bot_c_p,
        bot_rho,
        bot_c_s,
        bot_attenuation_p,
        bot_attenuation_s,
        c_bot_inside,
        rho_bot_inside,
        bot_reflection_table,
    )
    if abs(g_bot1) > 0.0 and abs(g_bot2) > 0.0:
        deta_dx = (f_bot2 / g_bot2 - f_bot1 / g_bot1) / (k2_2 - k2_1)

    rn = integral - drho_dx * phi[0] ** 2 + deta_dx * phi[-1] ** 2
    if abs(rn) > 0.0:
        phi = phi / np.sqrt(rn)

    idx_max = np.argmax(np.abs(phi))
    if np.abs(phi[idx_max]) > 0.0:
        phi = phi * np.exp(-1j * np.angle(phi[idx_max]))
        if np.real(phi[idx_max]) < 0.0:
            phi = -phi
    return phi


def compute_mode_shapes_complex(
    roots: np.ndarray,
    h: float,
    B1: np.ndarray,
    omega: float,
    bot_bc_type: int,
    bot_c_p: float,
    bot_rho: float,
    top_bc_type: int,
    media_start: np.ndarray,
    media_end: np.ndarray,
    media_rho: np.ndarray,
    bot_c_s: float = 0.0,
    bot_attenuation_p: float = 0.0,
    bot_attenuation_s: float = 0.0,
    c_top_inside: float | None = None,
    rho_top_inside: float | None = None,
    c_bot_inside: float | None = None,
    rho_bot_inside: float | None = None,
    top_reflection_table: np.ndarray | None = None,
    bot_reflection_table: np.ndarray | None = None,
    max_iter: int = 5,
) -> np.ndarray:
    """Compute KRAKENC-style complex mode vectors for complex eigenvalues."""
    roots = np.asarray(roots, dtype=np.complex128)
    n_modes = len(roots)
    n_points = len(B1)
    modes = np.zeros((n_modes, n_points), dtype=np.complex128)

    if n_modes == 0:
        return modes

    media_density = _media_density(n_points, media_start, media_end, media_rho)

    for mode_index, k2 in enumerate(roots):
        ab, g_top, g_bot = _build_inverse_iteration_matrix_complex(
            k2,
            h,
            B1,
            omega,
            bot_bc_type,
            bot_c_p,
            bot_rho,
            top_bc_type,
            media_start,
            media_end,
            media_rho,
            bot_c_s,
            bot_attenuation_p,
            bot_attenuation_s,
            c_top_inside,
            rho_top_inside,
            c_bot_inside,
            rho_bot_inside,
            top_reflection_table,
            bot_reflection_table,
        )

        phi = np.ones(n_points, dtype=np.complex128)
        if g_top == 0.0:
            phi[0] = 0.0
        if g_bot == 0.0:
            phi[-1] = 0.0

        for _ in range(max_iter):
            try:
                phi_new = scipy.linalg.solve_banded((1, 1), ab, phi)
            except scipy.linalg.LinAlgError:
                ab[1, :] += 1e-12
                phi_new = scipy.linalg.solve_banded((1, 1), ab, phi)

            norm = np.max(np.abs(phi_new))
            if norm <= 0.0:
                break
            phi = phi_new / norm

        if g_top == 0.0:
            phi[0] = 0.0
        if g_bot == 0.0:
            phi[-1] = 0.0

        modes[mode_index, :] = _normalize_complex_mode_with_boundary(
            phi,
            h,
            media_density,
            k2,
            omega,
            bot_bc_type,
            bot_c_p,
            bot_rho,
            top_bc_type,
            bot_c_s,
            bot_attenuation_p,
            bot_attenuation_s,
            c_top_inside if c_top_inside is not None else omega * h / np.sqrt(2.0 + B1[0]),
            rho_top_inside if rho_top_inside is not None else float(media_rho[0]),
            c_bot_inside if c_bot_inside is not None else omega * h / np.sqrt(2.0 + B1[-1]),
            rho_bot_inside if rho_bot_inside is not None else float(media_rho[-1]),
            top_reflection_table,
            bot_reflection_table,
        )

    return modes

def compute_modes(
    roots: np.ndarray, h: float, B1: np.ndarray, 
    omega: float, bot_bc_type: int, bot_c_p: float, bot_rho: float, top_bc_type: int,
    density_arr: np.ndarray = None, max_iter: int = 5, tol: float = 1e-6,
    bot_attenuation_p: float = 0.0, bot_c_s: float = 0.0, bot_attenuation_s: float = 0.0,
    attenuation_arr: np.ndarray = None, medium_rho: float = 1.0,
    media_start: np.ndarray = None, media_end: np.ndarray = None, media_rho: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the normalized mode shapes (eigenvectors) for the given roots (eigenvalues k^2).
    Uses Inverse Iteration on the tridiagonal finite difference matrix.
    Also computes the complex eigenvalues (including modal attenuation via perturbation theory).
    """
    M = len(roots)
    N = len(B1)
    
    modes = np.zeros((M, N), dtype=np.float64)
    complex_roots = np.zeros(M, dtype=np.complex128)
    
    if M == 0:
        return modes, complex_roots
        
    if density_arr is None:
        density_arr = np.ones(N, dtype=np.float64)
    if attenuation_arr is None:
        attenuation_arr = np.zeros(N, dtype=np.float64)
    if media_start is None or media_end is None or media_rho is None:
        media_start, media_end, media_rho = _default_media(N, medium_rho)
    media_density = _media_density(N, media_start, media_end, media_rho)
        
    for m in range(M):
        k2 = roots[m]

        from .core import bottom_impedance, top_impedance
        f_bot, g_bot = bottom_impedance(k2, omega, bot_bc_type, bot_c_p, bot_rho, bot_c_s)
        f_top, g_top = top_impedance(top_bc_type)

        ab, g_top, g_bot = _build_inverse_iteration_matrix(
            k2,
            h,
            B1,
            omega,
            bot_bc_type,
            bot_c_p,
            bot_rho,
            top_bc_type,
            media_start,
            media_end,
            media_rho,
            bot_c_s,
        )
            
        # Inverse Iteration
        phi = np.ones(N, dtype=np.float64)
        if g_top == 0.0: phi[0] = 0.0
        if g_bot == 0.0: phi[-1] = 0.0
        
        for _ in range(max_iter):
            try:
                phi_new = scipy.linalg.solve_banded((1, 1), ab, phi)
            except scipy.linalg.LinAlgError:
                # Singular matrix (exact eigenvalue). Add small perturbation.
                ab[1, :] += 1e-12
                phi_new = scipy.linalg.solve_banded((1, 1), ab, phi)
                
            norm = np.max(np.abs(phi_new))
            if norm > 0:
                phi = phi_new / norm
            else:
                break
                
        # Ensure exact zeros at boundaries if Dirichlet
        if g_top == 0.0: phi[0] = 0.0
        if g_bot == 0.0: phi[-1] = 0.0
            
        # Normalization: int |phi|^2 / rho dz = 1
        # Using Trapezoidal rule
        integrand = phi**2 / media_density
        integral = h * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])
        
        # Calculate admittance derivatives using finite difference (like AT Kraken)
        k2_1 = 0.9999999 * k2
        k2_2 = 1.0000001 * k2
        
        drho_dx = 0.0
        if g_top != 0.0:
            f_top1, g_top1 = top_impedance(top_bc_type)
            f_top2, g_top2 = top_impedance(top_bc_type)
            if g_top1 != 0.0 and g_top2 != 0.0:
                drho_dx = (f_top2/g_top2 - f_top1/g_top1) / (k2_2 - k2_1)
                
        deta_dx = 0.0
        if g_bot != 0.0:
            f_bot1, g_bot1 = bottom_impedance(k2_1, omega, bot_bc_type, bot_c_p, bot_rho, bot_c_s)
            f_bot2, g_bot2 = bottom_impedance(k2_2, omega, bot_bc_type, bot_c_p, bot_rho, bot_c_s)
            if g_bot1 != 0.0 and g_bot2 != 0.0:
                deta_dx = (f_bot2/g_bot2 - f_bot1/g_bot1) / (k2_2 - k2_1)
                
        # RN = SqNorm - DrhoDx * Phi( 1 ) ** 2 + DetaDx * Phi( NTotal1 ) ** 2
        rn = integral - drho_dx * phi[0]**2 + deta_dx * phi[-1]**2
        
        if rn > 0:
            phi /= np.sqrt(rn)
            
        # Standardize sign (make the first extremum positive)
        idx_max = np.argmax(np.abs(phi))
        if phi[idx_max] < 0:
            phi = -phi
            
        modes[m, :] = phi
        
        # --- Perturbation Theory for Modal Attenuation ---
        perturb_k2 = 0.0 + 0.0j

        # 0. Acoustic volume attenuation perturbation.
        if np.any(attenuation_arr > 0.0):
            beta_arr = attenuation_arr / (40.0 * np.pi * np.log10(np.e))
            omega_over_c_sq = (B1 + 2.0) / (h**2)
            b1c_arr = omega_over_c_sq * ((1.0 + 1j * beta_arr) ** 2).imag
            integrand_loss = b1c_arr * phi**2 / media_density
            volume_loss = h * (
                0.5 * integrand_loss[0] + np.sum(integrand_loss[1:-1]) + 0.5 * integrand_loss[-1]
            )
            perturb_k2 += 1j * volume_loss
        
        # 1. Bottom acoustic halfspace radiation/attenuation perturbation.
        # AT Kraken's real solver computes leaky-mode loss from the difference
        # between complex and real boundary admittance during Normalize().
        if bot_bc_type == 2:
            from .core import bottom_impedance_complex

            f_real, g_real = bottom_impedance(k2, omega, bot_bc_type, bot_c_p, bot_rho, bot_c_s)
            f_complex, g_complex = bottom_impedance_complex(
                k2,
                omega,
                bot_bc_type,
                bot_c_p,
                bot_rho,
                bot_c_s,
                bot_attenuation_p,
                bot_attenuation_s,
            )
            if g_real != 0.0 and g_complex != 0.0:
                delta_admittance = f_complex / g_complex - f_real / g_real
                perturb_k2 -= delta_admittance * phi[-1] ** 2

        # Write to complex roots array
        complex_root = k2 + perturb_k2
        if np.imag(np.sqrt(complex_root)) < 0.0:
            complex_root = np.conj(complex_root)
        complex_roots[m] = complex_root
        
    return modes, complex_roots
