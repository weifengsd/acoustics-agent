import numpy as np
from scipy.interpolate import CubicSpline
from numba import njit
from pyacoustics.schema import SSPModel

# ==========================================
# Numba JIT Core Functions (Fast Path)
# ==========================================

@njit(fastmath=True)
def evaluate_linear_ssp(z_q: float, z_arr: np.ndarray, c_arr: np.ndarray) -> tuple[float, float]:
    """
    Evaluates sound speed and gradient using piecewise linear interpolation.
    Returns: (c, dc_dz)
    """
    n = len(z_arr)
    # Handle out of bounds by clamping to endpoints
    if z_q <= z_arr[0]:
        c = c_arr[0]
        dc_dz = (c_arr[1] - c_arr[0]) / (z_arr[1] - z_arr[0])
        return c, dc_dz, 0.0
    if z_q >= z_arr[-1]:
        c = c_arr[-1]
        dc_dz = (c_arr[-1] - c_arr[-2]) / (z_arr[-1] - z_arr[-2])
        return c, dc_dz, 0.0
        
    # Binary search for the correct interval
    idx = np.searchsorted(z_arr, z_q) - 1
    
    dz = z_arr[idx+1] - z_arr[idx]
    dc = c_arr[idx+1] - c_arr[idx]
    
    dc_dz = dc / dz
    c = c_arr[idx] + dc_dz * (z_q - z_arr[idx])
    
    return c, dc_dz, 0.0

@njit(fastmath=True)
def evaluate_spline_ssp(z_q: float, z_arr: np.ndarray, c_coeffs: np.ndarray) -> tuple[float, float]:
    """
    Evaluates sound speed and gradient using cubic spline coefficients.
    c_coeffs is expected to be shape (4, N-1) from scipy CubicSpline.
    Returns: (c, dc_dz)
    """
    # Handle out of bounds by linear extrapolation from ends
    if z_q <= z_arr[0]:
        dx = z_q - z_arr[0]
        c = c_coeffs[3, 0] + c_coeffs[2, 0] * dx # a + b*dx
        dc_dz = c_coeffs[2, 0]
        return c, dc_dz, 0.0
    if z_q >= z_arr[-1]:
        idx = len(z_arr) - 2
        dx = z_q - z_arr[idx]
        c = c_coeffs[3, idx] + c_coeffs[2, idx] * dx + c_coeffs[1, idx] * dx**2 + c_coeffs[0, idx] * dx**3
        dc_dz = c_coeffs[2, idx] + 2 * c_coeffs[1, idx] * dx + 3 * c_coeffs[0, idx] * dx**2
        d2c_dz2 = 2 * c_coeffs[1, idx] + 6 * c_coeffs[0, idx] * dx
        return c, dc_dz, d2c_dz2

    # Binary search
    idx = np.searchsorted(z_arr, z_q) - 1
    dx = z_q - z_arr[idx]
    
    # Cubic polynomial evaluation
    c = c_coeffs[3, idx] + c_coeffs[2, idx] * dx + c_coeffs[1, idx] * dx**2 + c_coeffs[0, idx] * dx**3
    dc_dz = c_coeffs[2, idx] + 2 * c_coeffs[1, idx] * dx + 3 * c_coeffs[0, idx] * dx**2
    d2c_dz2 = 2 * c_coeffs[1, idx] + 6 * c_coeffs[0, idx] * dx
    
    return c, dc_dz, d2c_dz2

# ==========================================
# Python Object Oriented Wrapper
# ==========================================

class SSPInterpolator:
    """
    Provides sound speed profiling interpolations.
    Extracts pure numpy arrays suitable to be passed into Numba @njit functions.
    """
    def __init__(self, ssp_model: SSPModel):
        self.type = ssp_model.type
        self.z_arr = np.array([layer.depth for layer in ssp_model.data], dtype=np.float64)
        self.c_arr = np.array([layer.c for layer in ssp_model.data], dtype=np.float64)
        
        # Precompute spline coefficients if necessary
        self.c_coeffs = None
        if self.type == "spline":
            # bc_type='natural' is a common default for acoustic profiles if no derivatives are given
            cs = CubicSpline(self.z_arr, self.c_arr, bc_type='natural')
            self.c_coeffs = cs.c  # Shape: (4, N-1)

    def evaluate(self, z: float) -> tuple[float, float]:
        """
        Python-level evaluation (useful for testing and plotting).
        For hot-loops, do not call this. Pass `z_arr`, `c_arr`, and `c_coeffs` 
        directly into the numba compiled functions.
        """
        if self.type == "c-linear":
            return evaluate_linear_ssp(z, self.z_arr, self.c_arr)
        elif self.type == "spline":
            return evaluate_spline_ssp(z, self.z_arr, self.c_coeffs)
        else:
            raise NotImplementedError(f"SSP type {self.type} not implemented yet.")
