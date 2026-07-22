import numpy as np

def compute_field(
    modes: np.ndarray, roots: np.ndarray, 
    z_arr: np.ndarray, r_arr: np.ndarray, 
    z_s: float, z_r_arr: np.ndarray, 
    rho_s: float = 1.0,
    field_option: str = "RA",
    receiver_range_offsets_m: np.ndarray | None = None,
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
    field_option: AT Field option string. The first character controls the
        range normalization: ``R`` point source, ``X`` line source, or ``S``
        scaled cylindrical point source.
    receiver_range_offsets_m: optional range offset per receiver depth for
        AT Field array tilt semantics.
    
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
    mode_dtype = np.result_type(modes.dtype, np.complex128)
    phi_s = np.zeros(M, dtype=mode_dtype)
    phi_r = np.zeros((M, Z), dtype=mode_dtype)
    
    for m in range(M):
        phi_s[m] = np.interp(z_s, z_arr, modes[m, :])
        phi_r[m, :] = np.interp(z_r_arr, z_arr, modes[m, :])
        
    option0 = (field_option or "RA").strip().upper()[:1] or "R"
    if option0 not in {"R", "X", "S"}:
        raise ValueError("Kraken field_option must start with 'R', 'X', or 'S'")

    const_factor = np.sqrt(2.0 * np.pi)
    
    pressure = np.zeros((Z, R), dtype=np.complex128)

    if receiver_range_offsets_m is None:
        range_offsets = np.zeros(Z, dtype=np.float64)
    else:
        range_offsets = np.asarray(receiver_range_offsets_m, dtype=np.float64)
        if range_offsets.size == 1:
            range_offsets = np.full(Z, float(range_offsets[0]), dtype=np.float64)
        elif range_offsets.size != Z:
            raise ValueError("receiver_range_offsets_m must have length 1 or match receiver depths")
    
    # Mode sum
    for m in range(M):
        # shape (1, R)
        if option0 == "X":
            modal_norm = k_m[m]
        else:
            modal_norm = np.sqrt(k_m[m])
        range_phase = np.exp(1j * k_m[m] * (r_arr.reshape(1, -1) + range_offsets.reshape(-1, 1))) / modal_norm
        # shape (Z, 1)
        depth_amp = (phi_s[m] * phi_r[m, :]).reshape(-1, 1)
        
        # Outer product sum
        pressure += depth_amp * range_phase
        
    # Apply constants and spreading
    if option0 == "R":
        r_eff = r_arr.reshape(1, -1) + range_offsets.reshape(-1, 1)
        r_safe = np.where(np.abs(r_eff) <= 1.0e-10, 1.0e-10, np.abs(r_eff))
        pressure *= const_factor / np.sqrt(r_safe)
    else:
        pressure *= const_factor
    
    # Calculate TL
    # If pressure is 0, clip to a small number to avoid log10(0)
    p_abs = np.abs(pressure)
    p_abs = np.where(p_abs < 1e-20, 1e-20, p_abs)
    
    TL = -20.0 * np.log10(p_abs)
    
    return TL
