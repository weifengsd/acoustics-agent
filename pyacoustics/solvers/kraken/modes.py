import numpy as np
import scipy.linalg
from numba import njit

def compute_modes(
    roots: np.ndarray, h: float, B1: np.ndarray, 
    omega: float, bot_bc_type: int, bot_c_p: float, bot_rho: float, top_bc_type: int,
    density_arr: np.ndarray = None, max_iter: int = 5, tol: float = 1e-6,
    bot_attenuation_p: float = 0.0
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
        
    rho_medium = 1.0 # Assuming constant density in water column for now
    
    for m in range(M):
        k2 = roots[m]
        
        from .core import bottom_impedance, top_impedance
        f_bot, g_bot = bottom_impedance(k2, omega, bot_bc_type, bot_c_p, bot_rho)
        f_top, g_top = top_impedance(top_bc_type)
        
        # Build banded matrix for scipy.linalg.solve_banded
        # ab shape (3, N): top row is upper diag, mid is diag, bottom is lower diag.
        ab = np.zeros((3, N), dtype=np.float64)
        
        # Off-diagonals are -1
        ab[0, 1:] = -1.0 # Upper diagonal
        ab[2, :-1] = -1.0 # Lower diagonal
        
        # Main diagonal
        for j in range(N):
            ab[1, j] = h**2 * k2 - B1[j]
            
        # Top Boundary Condition (j=0)
        if g_top == 0.0:
            ab[1, 0] = 1e20
            ab[0, 1] = 0.0
            ab[2, 0] = 0.0
        else:
            ab[1, 0] = 0.5 * (ab[1, 0] - 2.0 * h * rho_medium * f_top / g_top)
            
        # Bottom Boundary Condition (j=N-1)
        if g_bot == 0.0:
            ab[1, -1] = 1e20
            ab[0, -1] = 0.0
            ab[2, -2] = 0.0
        else:
            ab[1, -1] = 0.5 * (ab[1, -1] + 2.0 * h * rho_medium * f_bot / g_bot)
            
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
        integrand = phi**2 / density_arr
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
            f_bot1, g_bot1 = bottom_impedance(k2_1, omega, bot_bc_type, bot_c_p, bot_rho)
            f_bot2, g_bot2 = bottom_impedance(k2_2, omega, bot_bc_type, bot_c_p, bot_rho)
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
        
        # 1. Bottom Halfspace Perturbation
        if bot_attenuation_p > 0.0 and bot_bc_type == 2:
            # Convert dB/lambda to imaginary sound speed component beta
            beta = bot_attenuation_p / (40.0 * np.pi * np.log10(np.e))
            c_complex = bot_c_p / (1.0 + 1j * beta)
            
            # Complex admittance
            gamma2_complex = k2 - (omega / c_complex)**2
            gamma_complex = np.sqrt(gamma2_complex + 0j)
            admit_complex = gamma_complex / bot_rho
            
            # Real admittance
            gamma2_real = k2 - (omega / bot_c_p)**2
            # Using same logic as bottom_impedance
            if gamma2_real >= 0:
                gamma_real = np.sqrt(gamma2_real)
            else:
                gamma_real = 1j * np.sqrt(-gamma2_real)
            admit_real = gamma_real / bot_rho
            
            # Del = f2/g2 (complex) - f1/g1 (real)
            Del = admit_complex - admit_real
            
            # Perturbation_k = - Del * Phi[-1]**2
            perturb_k2 -= Del * phi[-1]**2
            
        # Write to complex roots array
        complex_roots[m] = k2 + perturb_k2
        
    return modes, complex_roots