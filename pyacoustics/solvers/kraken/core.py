import numpy as np
import math
from numba import njit

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
def bottom_impedance(k2: float, omega: float, bc_type: int, c_p: float, density: float) -> tuple[float, float]:
    """
    Calculates the bottom impedance coefficients f, g for the boundary condition f * phi + g * phi' = 0.
    bc_type: 0 for vacuum, 1 for rigid, 2 for halfspace
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
    omega: float, bot_bc_type: int, bot_c_p: float, bot_rho: float, top_bc_type: int
) -> tuple[float, int]:
    """
    Evaluates the dispersion function Delta(k^2) using recurrent shooting.
    Returns the determinant Delta and the Sturm mode count.
    Uses O(h^2) central difference boundary conditions matching AT Kraken.
    """
    N = len(B1)
    
    # Calculate boundary impedances at this specific k2
    f_bot, g_bot = bottom_impedance(k2, omega, bot_bc_type, bot_c_p, bot_rho)
    f_top, g_top = top_impedance(top_bc_type)
    
    # Initialize from bottom using central difference ghost node
    # Boundary condition: f_bot * phi_N + g_bot * phi'_N / rho_bot = 0
    # For now we assume rho=1.0 inside the water column.
    rho_medium = 1.0
    
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
