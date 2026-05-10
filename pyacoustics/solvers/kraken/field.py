import numpy as np

def compute_field(
    modes: np.ndarray, roots: np.ndarray, 
    z_arr: np.ndarray, r_arr: np.ndarray, 
    z_s: float, z_r_arr: np.ndarray, 
    rho_s: float = 1.0
) -> np.ndarray:
    """
    Computes the Transmission Loss (TL) field using modal expansion.
    
    modes: shape (M, N) where N is len(z_arr)
    roots: shape (M,) eigenvalues k^2
    z_arr: shape (N,) depth grid
    r_arr: shape (R,) range grid
    z_s: source depth
    z_r_arr: shape (Z,) receiver depth grid
    rho_s: density at source depth
    
    Returns: TL field shape (Z, R) in dB
    """
    M, N = modes.shape
    R = len(r_arr)
    Z = len(z_r_arr)
    
    if M == 0:
        return np.full((Z, R), 100.0) # High TL if no modes
        
    k_m = np.sqrt(roots.astype(np.complex128))
    
    # Interpolate modes to source and receiver depths
    # We use linear interpolation for simplicity
    phi_s = np.zeros(M, dtype=np.float64)
    phi_r = np.zeros((M, Z), dtype=np.float64)
    
    for m in range(M):
        phi_s[m] = np.interp(z_s, z_arr, modes[m, :])
        phi_r[m, :] = np.interp(z_r_arr, z_arr, modes[m, :])
        
    # Pre-calculate constant terms
    # Standard Green's function formulation is 1 / (rho_s * sqrt(8 * pi))
    # However, to match AT (Acoustics Toolbox) and Bellhop normalization where the 
    # pressure of a point source at 1 meter is 1.0 (i.e. p = e^{ikr}/r instead of e^{ikr}/(4*pi*r)),
    # we must multiply by 4*pi. 
    # 4*pi / sqrt(8*pi) = sqrt(2*pi).
    
    pressure = np.zeros((Z, R), dtype=np.complex128)
    
    const_factor = np.sqrt(2.0 * np.pi) / rho_s
    
    # Avoid division by zero at r=0
    r_safe = np.where(r_arr == 0, 1e-10, r_arr)
    cylindrical_spreading = 1.0 / np.sqrt(r_safe)
    
    # Mode sum
    for m in range(M):
        # shape (1, R)
        range_phase = np.exp(1j * k_m[m] * r_arr) / np.sqrt(k_m[m])
        # shape (Z, 1)
        depth_amp = (phi_s[m] * phi_r[m, :]).reshape(-1, 1)
        
        # Outer product sum
        pressure += depth_amp @ range_phase.reshape(1, -1)
        
    # Apply constants and spreading
    pressure *= const_factor * cylindrical_spreading.reshape(1, -1)
    
    # Calculate TL
    # If pressure is 0, clip to a small number to avoid log10(0)
    p_abs = np.abs(pressure)
    p_abs = np.where(p_abs < 1e-20, 1e-20, p_abs)
    
    TL = -20.0 * np.log10(p_abs)
    
    return TL
