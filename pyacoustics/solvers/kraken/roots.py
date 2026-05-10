import numpy as np
from numba import njit
from .core import evaluate_dispersion

@njit(fastmath=True)
def bisection_search_roots(
    k2_min: float, k2_max: float, 
    h: float, B1: np.ndarray, 
    omega: float, bot_bc_type: int, bot_c_p: float, bot_rho: float, top_bc_type: int,
    tol: float = 1e-8, max_iter: int = 100
) -> np.ndarray:
    """
    Finds roots (eigenvalues k^2) using Sturm sequence and bisection.
    This guarantees finding all propagating modes between k2_min and k2_max.
    """
    _, modes_min = evaluate_dispersion(k2_min, h, B1, omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type)
    _, modes_max = evaluate_dispersion(k2_max, h, B1, omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type)
    
    M = modes_min - modes_max
    if M <= 0:
        return np.zeros(0, dtype=np.float64)
        
    roots = np.empty(M, dtype=np.float64)
    
    for m in range(1, M + 1):
        target_mode_count_lower = modes_max + m
        
        low = k2_min
        high = k2_max
        
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            if (high - low) < tol:
                break
                
            _, count_mid = evaluate_dispersion(mid, h, B1, omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type)
            
            if count_mid >= target_mode_count_lower:
                low = mid
            else:
                high = mid
                
        roots[m - 1] = 0.5 * (low + high)
        
    return roots

@njit(fastmath=True)
def secant_search_roots(
    initial_guesses: np.ndarray,
    h: float, B1: np.ndarray,
    omega: float, bot_bc_type: int, bot_c_p: float, bot_rho: float, top_bc_type: int,
    tol: float = 1e-10, max_iter: int = 20
) -> np.ndarray:
    """
    Refines roots using the Secant method. Requires good initial guesses.
    """
    M = len(initial_guesses)
    roots = np.copy(initial_guesses)
    
    for m in range(M):
        k2_0 = roots[m]
        d0, _ = evaluate_dispersion(k2_0, h, B1, omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type)
        
        k2_1 = k2_0 * 1.0001 if k2_0 != 0 else 1e-6
        d1, _ = evaluate_dispersion(k2_1, h, B1, omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type)
        
        for _ in range(max_iter):
            if abs(d1 - d0) < 1e-14:
                break
                
            # Secant step
            k2_2 = k2_1 - d1 * (k2_1 - k2_0) / (d1 - d0)
            
            if abs(k2_2 - k2_1) < tol:
                k2_1 = k2_2
                break
                
            d2, _ = evaluate_dispersion(k2_2, h, B1, omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type)
            
            k2_0 = k2_1
            d0 = d1
            k2_1 = k2_2
            d1 = d2
            
        roots[m] = k2_1
        
    return roots

def neville_extrapolation(k2_sequences: list[np.ndarray], mesh_multipliers: list[int]) -> np.ndarray:
    """
    Applies Neville's algorithm for Richardson Extrapolation on a sequence of roots 
    found on grids with sizes proportional to 1/mesh_multipliers.
    Assuming error is O(h^2).
    """
    N_sets = len(k2_sequences)
    if N_sets == 0:
        return np.array([])
    if N_sets == 1:
        return k2_sequences[0]
        
    M = len(k2_sequences[0])
    
    # Check that all sets found the same number of modes.
    # If a finer grid found more/fewer modes, we have a problem.
    # For robust production code, we'd handle mismatch by matching modes.
    # Here we assume M is constant.
    for seq in k2_sequences:
        if len(seq) != M:
            # Fallback to the finest grid if mode mismatch occurs
            return k2_sequences[-1]
            
    T = np.zeros((N_sets, N_sets, M))
    for i in range(N_sets):
        T[i, 0, :] = k2_sequences[i]
        
    for j in range(1, N_sets):
        for i in range(N_sets - j):
            h_i = 1.0 / mesh_multipliers[i]
            h_ij = 1.0 / mesh_multipliers[i+j]
            ratio = (h_i / h_ij)**2
            
            T[i, j, :] = T[i+1, j-1, :] + (T[i+1, j-1, :] - T[i, j-1, :]) / (ratio - 1.0)
            
    return T[0, N_sets - 1, :]
