import numpy as np
import math
from numba import njit


def attenuation_db_per_wavelength_to_beta(attenuation: float) -> float:
    """Convert dB/wavelength attenuation to the complex sound-speed beta."""
    return float(attenuation) / (40.0 * np.pi * np.log10(np.e))


def complex_sound_speed(c: float, attenuation: float = 0.0) -> complex:
    """Return AT-style complex sound speed for dB/wavelength attenuation."""
    beta = attenuation_db_per_wavelength_to_beta(attenuation)
    return complex(c) / (1.0 + 1j * beta)


def pekeris_root(z: complex) -> complex:
    """Complex square-root branch used for acoustic half-space impedances."""
    root = np.sqrt(complex(z))
    if root.real < 0.0:
        root = -root
    return root


def bottom_impedance_complex(
    k2: complex,
    omega: float,
    bc_type: int,
    c_p: float,
    density: float,
    c_s: float = 0.0,
    attenuation_p: float = 0.0,
    attenuation_s: float = 0.0,
) -> tuple[complex, complex]:
    """
    Complex boundary impedance for KRAKENC-style root refinement.

    The acoustic half-space returns an equivalent (f, g) pair with the same
    admittance ratio as AT's BCImpedancec. Scaling both terms does not affect
    the zero of the dispersion relation.
    """
    if bc_type == 0:
        return 1.0 + 0.0j, 0.0 + 0.0j
    if bc_type == 1:
        return 0.0 + 0.0j, 1.0 + 0.0j
    if bc_type == 2 or c_s <= 0.0:
        c_p_complex = complex_sound_speed(c_p, attenuation_p)
        gamma_p = pekeris_root(k2 - (omega / c_p_complex) ** 2)
        return gamma_p / density, 1.0 + 0.0j

    c_p_complex = complex_sound_speed(c_p, attenuation_p)
    c_s_complex = complex_sound_speed(c_s, attenuation_s)
    gamma_s2 = k2 - (omega / c_s_complex) ** 2
    gamma_p2 = k2 - (omega / c_p_complex) ** 2
    gamma_s = pekeris_root(gamma_s2)
    gamma_p = pekeris_root(gamma_p2)
    mu = density * c_s_complex**2

    f = omega**2 * gamma_p * (k2 - gamma_s2)
    g = ((gamma_s2 + k2) ** 2 - 4.0 * gamma_s * gamma_p * k2) * mu
    return f, g


def interpolate_reflection_coefficient(theta_deg: float, table: np.ndarray | None) -> complex | None:
    """Interpolate an AT-style angle/amplitude/phase reflection table."""
    if table is None:
        return None
    table = np.asarray(table, dtype=np.float64)
    if table.size == 0:
        return None
    if table.ndim != 2 or table.shape[1] != 3:
        raise ValueError("reflection table must have angle/amplitude/phase columns")

    angles = table[:, 0]
    amplitudes = table[:, 1]
    phases = table[:, 2]
    order = np.argsort(angles)
    angles = angles[order]
    amplitudes = amplitudes[order]
    phases = phases[order]

    theta = float(np.clip(theta_deg, angles[0], angles[-1]))
    amplitude = np.interp(theta, angles, amplitudes)
    phase = np.interp(theta, angles, phases)
    return amplitude * np.exp(1j * np.deg2rad(phase))


def _reflection_coefficient_for_robin(coefficient: complex) -> complex:
    """Convert AT's tabulated pressure coefficient to the Robin convention."""

    amplitude = abs(coefficient)
    if amplitude < 1e-12:
        return complex(coefficient)
    return coefficient / (amplitude * amplitude)


def reflection_table_impedance(
    k2: complex,
    omega: float,
    c_inside: float,
    rho_inside: float,
    table: np.ndarray | None,
) -> tuple[complex, complex] | None:
    """Convert a tabulated reflection coefficient into Robin impedance."""
    coefficient = interpolate_reflection_coefficient(
        np.rad2deg(np.arctan2(np.real(pekeris_root((omega / c_inside) ** 2 - k2)), np.real(np.sqrt(k2)))),
        table,
    )
    if coefficient is None:
        return None
    coefficient = _reflection_coefficient_for_robin(coefficient)

    kz = pekeris_root((omega / c_inside) ** 2 - k2)
    denom = rho_inside * (1.0 + coefficient)
    if abs(denom) < 1e-12:
        return 1.0 + 0.0j, 0.0 + 0.0j
    return 1j * kz * (1.0 - coefficient) / denom, 1.0 + 0.0j


def top_impedance_complex(
    k2: complex,
    omega: float,
    bc_type: int,
    c_inside: float,
    rho_inside: float,
    reflection_table: np.ndarray | None = None,
) -> tuple[complex, complex]:
    if bc_type == 0:
        return 1.0 + 0.0j, 0.0 + 0.0j
    if bc_type == 4:
        impedance = reflection_table_impedance(k2, omega, c_inside, rho_inside, reflection_table)
        if impedance is not None:
            f, g = impedance
            return f, -g
    if bc_type == 1 or bc_type == 4:
        return 0.0 + 0.0j, -1.0 + 0.0j
    return 1.0 + 0.0j, 0.0 + 0.0j


def bottom_impedance_complex_with_table(
    k2: complex,
    omega: float,
    bc_type: int,
    c_p: float,
    density: float,
    c_s: float,
    attenuation_p: float,
    attenuation_s: float,
    c_inside: float,
    rho_inside: float,
    reflection_table: np.ndarray | None = None,
) -> tuple[complex, complex]:
    if bc_type == 4:
        impedance = reflection_table_impedance(k2, omega, c_inside, rho_inside, reflection_table)
        if impedance is not None:
            return impedance
        return 0.0 + 0.0j, 1.0 + 0.0j
    return bottom_impedance_complex(k2, omega, bc_type, c_p, density, c_s, attenuation_p, attenuation_s)


def build_acoustic_matrix_complex(
    h: float, omega: float, c_arr: np.ndarray, attenuation_arr: np.ndarray | None = None
) -> np.ndarray:
    """Build the complex KRAKENC acoustic B1 diagonal."""
    if attenuation_arr is None:
        attenuation_arr = np.zeros_like(c_arr, dtype=np.float64)
    c_complex = np.asarray(
        [complex_sound_speed(c, attenuation) for c, attenuation in zip(c_arr, attenuation_arr)],
        dtype=np.complex128,
    )
    return -2.0 + h**2 * (omega / c_complex) ** 2


def evaluate_dispersion_media_complex(
    k2: complex,
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
) -> complex:
    """Evaluate the complex multi-medium dispersion relation."""
    if c_top_inside is None:
        c_top_inside = omega * h / np.sqrt(2.0 + B1[0])
    if rho_top_inside is None:
        rho_top_inside = float(media_rho[0])
    if c_bot_inside is None:
        c_bot_inside = omega * h / np.sqrt(2.0 + B1[-1])
    if rho_bot_inside is None:
        rho_bot_inside = float(media_rho[-1])

    f, g = bottom_impedance_complex_with_table(
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

    h2k2 = h**2 * k2
    for im in range(len(media_start) - 1, -1, -1):
        start = int(media_start[im])
        end = int(media_end[im])
        rho_medium = float(media_rho[im])

        if end <= start:
            continue

        p1 = -2.0 * g
        p2 = (B1[end] - h2k2) * g - 2.0 * h * f * rho_medium
        p0 = p1

        for j in range(end - 1, start - 1, -1):
            p0 = p1
            p1 = p2
            p2 = (h2k2 - B1[j]) * p1 - p0

            if abs(p2) > 1e50:
                p0 *= 1e-50
                p1 *= 1e-50
                p2 *= 1e-50

        f = -(p2 - p0) / (2.0 * h) / rho_medium
        g = -p1

    return f * g_top - g * f_top

@njit(fastmath=True)
def build_acoustic_matrix(h: float, omega: float, c_arr: np.ndarray) -> np.ndarray:
    """
    Builds the diagonal of the finite difference matrix (B1) for an acoustic layer.
    B1[j] = -2.0 + h**2 * (omega / c_arr[j])**2
    """
    N = len(c_arr)
    B1 = np.empty(N, dtype=np.float64)
    for j in range(N):
        B1[j] = -2.0 + h**2 * (omega / c_arr[j])**2
    return B1

@njit(fastmath=True)
def bottom_impedance(
    k2: float, omega: float, bc_type: int, c_p: float, density: float, c_s: float = 0.0
) -> tuple[float, float]:
    """
    Calculates the bottom impedance coefficients f, g for the boundary condition f * phi + g * phi' = 0.
    bc_type: 0 for vacuum, 1 for rigid, 2 for acoustic halfspace, 3 for elastic halfspace
    """
    if bc_type == 0:  # Vacuum (pressure release)
        return 1.0, 0.0
    elif bc_type == 1:  # Rigid
        return 0.0, 1.0
    elif bc_type == 2:  # Halfspace (Acoustic)
        gamma2 = k2 - (omega / c_p)**2
        if gamma2 >= 0:
            gamma = math.sqrt(gamma2)
            # f = rho_water/rho_bottom * gamma. Here we assume density is the relative density.
            return gamma / density, 1.0
        else:
            # Propagating into the bottom, fallback to rigid for basic real-axis root finding
            return 0.0, 1.0
    elif bc_type == 3:  # Halfspace (Elastic), matching AT BCImpedance.f90 for real KRAKEN
        if c_s <= 0.0:
            gamma2 = k2 - (omega / c_p)**2
            if gamma2 >= 0:
                gamma = math.sqrt(gamma2)
                return gamma / density, 1.0
            return 0.0, 1.0

        gamma_s2 = k2 - (omega / c_s)**2
        gamma_p2 = k2 - (omega / c_p)**2
        gamma_s = math.sqrt(gamma_s2) if gamma_s2 > 0.0 else 0.0
        gamma_p = math.sqrt(gamma_p2) if gamma_p2 > 0.0 else 0.0
        mu = density * c_s**2

        f = omega**2 * gamma_p * (k2 - gamma_s2)
        g = ((gamma_s2 + k2)**2 - 4.0 * gamma_s * gamma_p * k2) * mu
        return f, g
    return 0.0, 1.0

@njit(fastmath=True)
def top_impedance(bc_type: int) -> tuple[float, float]:
    """
    Calculates the top impedance coefficients f, g for the boundary condition f * phi + g * phi' = 0.
    bc_type: 0 for vacuum, 1 for rigid
    """
    if bc_type == 0:  # Vacuum (pressure release)
        return 1.0, 0.0
    elif bc_type == 1:  # Rigid
        return 0.0, 1.0
    return 1.0, 0.0

@njit(fastmath=True)
def evaluate_dispersion(
    k2: float, h: float, B1: np.ndarray, 
    omega: float, bot_bc_type: int, bot_c_p: float, bot_rho: float, top_bc_type: int,
    bot_c_s: float = 0.0, medium_rho: float = 1.0
) -> tuple[float, int]:
    """
    Evaluates the dispersion function Delta(k^2) using recurrent shooting.
    Returns the determinant Delta and the Sturm mode count.
    Uses O(h^2) central difference boundary conditions matching AT Kraken.
    """
    N = len(B1)
    
    # Calculate boundary impedances at this specific k2
    f_bot, g_bot = bottom_impedance(k2, omega, bot_bc_type, bot_c_p, bot_rho, bot_c_s)
    f_top, g_top = top_impedance(top_bc_type)
    
    # Initialize from bottom using central difference ghost node
    # Boundary condition: f_bot * phi_N + g_bot * phi'_N / rho_bot = 0
    rho_medium = medium_rho
    
    # p1 corresponds to phi_N, p2 corresponds to phi_{N-1}
    # To match AT: p1 = -2g, p2 = (B1 - h^2 k^2)g + 2*h*f*rho
    h2k2 = h**2 * k2
    
    if g_bot == 0.0: # Rigid or Vacuum where g=0 (Vacuum is f=1, g=0)
        # If g=0 (vacuum), phi_N = 0.
        p1 = 0.0
        p0 = -1.0 # This matches AT's Floor scaling fallback but simpler: phi_{N-1} can be arbitrary non-zero
    else:
        p1 = -2.0 * g_bot
        p2 = (B1[-1] - h2k2) * g_bot - 2.0 * h * f_bot * rho_medium
        p0 = p1
        p1 = p2
        
    mode_count = 0
    if bot_bc_type == 3 and g_bot > 0.0:
        mode_count += 1
    
    # Shoot upwards from N-1 down to 1
    # B1 represents the grid from z[0] to z[N-1]
    # Since p1 is already at N-1 (if g!=0), the loop starts at N-2
    
    if g_bot == 0.0:
        # If vacuum, start at N-1
        start_idx = N - 1
    else:
        start_idx = N - 2
        # Check if the initial step crossed zero
        if p0 * p1 <= 0.0 and p0 != 0.0:
            mode_count += 1
            
    for j in range(start_idx, 0, -1):
        p_next = (h2k2 - B1[j]) * p1 - p0
        
        # Sturm sequence count (number of zero crossings)
        if p1 * p_next <= 0.0 and p1 != 0.0:
            mode_count += 1
            
        # Rescale to prevent overflow
        if abs(p_next) > 1e10:
            p_next *= 1e-10
            p1 *= 1e-10
            
        p0 = p1
        p1 = p_next
        
    # Top BC
    # p1 is phi_0 (wait, loop ended at 1, so p1 is phi_1, p0 is phi_2)
    # The next step would calculate phi_0:
    p_next = (h2k2 - B1[0]) * p1 - p0
    phi_0 = p_next
    phi_1 = p1
    
    # f_top * phi_0 + g_top * phi'_0 / rho = 0
    # phi'_0 = (phi_1 - phi_{-1}) / 2h
    # phi_1 = (h^2 k^2 - B1_0) phi_0 - phi_{-1}  => phi_{-1} = (h^2 k^2 - B1_0) phi_0 - phi_1
    # phi'_0 = (2 phi_1 - (h^2 k^2 - B1_0) phi_0) / 2h
    # delta = f_top * phi_0 + g_top * (2 phi_1 - (h^2 k^2 - B1_0) phi_0) / (2h * rho_medium)
    
    if g_top == 0.0:
        delta = phi_0
    else:
        delta = f_top * phi_0 + g_top * (2.0 * phi_1 - (h2k2 - B1[0]) * phi_0) / (2.0 * h * rho_medium)
    
    # Check zero crossing at the top boundary
    if g_top != 0.0: 
        if delta * phi_0 <= 0.0 and phi_0 != 0.0:
             mode_count += 1
             
    return delta, mode_count


@njit(fastmath=True)
def evaluate_dispersion_media(
    k2: float, h: float, B1: np.ndarray,
    omega: float, bot_bc_type: int, bot_c_p: float, bot_rho: float, top_bc_type: int,
    media_start: np.ndarray, media_end: np.ndarray, media_rho: np.ndarray,
    bot_c_s: float = 0.0
) -> tuple[float, int]:
    """
    Evaluate the dispersion function through multiple acoustic media.

    Each medium uses the AT Kraken acoustic-layer recurrence with a constant
    density for that medium and passes the boundary admittance pair (f, g)
    upward across interfaces.
    """
    f_bot, g_bot = bottom_impedance(k2, omega, bot_bc_type, bot_c_p, bot_rho, bot_c_s)
    f_top, g_top = top_impedance(top_bc_type)

    f = f_bot
    g = g_bot
    h2k2 = h**2 * k2
    mode_count = 0
    if bot_bc_type == 3 and g > 0.0:
        mode_count += 1

    for im in range(len(media_start) - 1, -1, -1):
        start = media_start[im]
        end = media_end[im]
        rho_medium = media_rho[im]

        if end <= start:
            continue

        p1 = -2.0 * g
        p2 = (B1[end] - h2k2) * g - 2.0 * h * f * rho_medium
        p0 = p1

        for j in range(end - 1, start - 1, -1):
            p0 = p1
            p1 = p2
            p2 = (h2k2 - B1[j]) * p1 - p0

            if p0 * p1 <= 0.0 and p0 != 0.0:
                mode_count += 1

            if abs(p2) > 1e10:
                p0 *= 1e-10
                p1 *= 1e-10
                p2 *= 1e-10

        f = -(p2 - p0) / (2.0 * h) / rho_medium
        g = -p1

    delta = f * g_top - g * f_top
    top_crossing = g * delta
    if (top_bc_type == 1 and top_crossing < 0.0) or (top_bc_type != 1 and top_crossing > 0.0):
        mode_count += 1

    return delta, mode_count
