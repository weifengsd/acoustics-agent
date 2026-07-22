import numpy as np
from scipy.interpolate import CubicSpline
from numba import njit
from pyacoustics.schema import SSPModel

# ==========================================
# Numba JIT Core Functions (Fast Path)
# ==========================================

@njit(fastmath=True)
def evaluate_linear_ssp(z_q: float, z_arr: np.ndarray, c_arr: np.ndarray) -> tuple[float, float, float]:
    """
    Evaluates sound speed and gradient using piecewise linear interpolation.
    Returns: (c, dc_dz, d2c_dz2)
    """
    n = len(z_arr)
    # Handle out of bounds by linear extrapolation from ends
    if z_q <= z_arr[0]:
        dc_dz = (c_arr[1] - c_arr[0]) / (z_arr[1] - z_arr[0])
        c = c_arr[0] + dc_dz * (z_q - z_arr[0])
        return c, dc_dz, 0.0
    if z_q >= z_arr[-1]:
        dc_dz = (c_arr[-1] - c_arr[-2]) / (z_arr[-1] - z_arr[-2])
        c = c_arr[-1] + dc_dz * (z_q - z_arr[-1])
        return c, dc_dz, 0.0
        
    # Binary search for the correct interval
    idx = np.searchsorted(z_arr, z_q) - 1
    
    dz = z_arr[idx+1] - z_arr[idx]
    dc = c_arr[idx+1] - c_arr[idx]
    
    dc_dz = dc / dz
    c = c_arr[idx] + dc_dz * (z_q - z_arr[idx])
    
    return c, dc_dz, 0.0

@njit(fastmath=True)
def evaluate_spline_ssp(z_q: float, z_arr: np.ndarray, c_coeffs: np.ndarray) -> tuple[float, float, float]:
    """
    Evaluates sound speed and gradient using cubic spline coefficients.
    c_coeffs is expected to be shape (4, N-1) from scipy CubicSpline.
    Returns: (c, dc_dz, d2c_dz2)
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
        # Linear extrapolation for spline consistency at boundaries
        c_end = c_coeffs[3, idx] + c_coeffs[2, idx] * (z_arr[-1] - z_arr[idx]) + c_coeffs[1, idx] * (z_arr[-1] - z_arr[idx])**2 + c_coeffs[0, idx] * (z_arr[-1] - z_arr[idx])**3
        dc_dz_end = c_coeffs[2, idx] + 2 * c_coeffs[1, idx] * (z_arr[-1] - z_arr[idx]) + 3 * c_coeffs[0, idx] * (z_arr[-1] - z_arr[idx])**2
        c = c_end + dc_dz_end * (z_q - z_arr[-1])
        return c, dc_dz_end, 0.0

    # Binary search
    idx = np.searchsorted(z_arr, z_q) - 1
    dx = z_q - z_arr[idx]
    
    # Cubic polynomial evaluation
    c = c_coeffs[3, idx] + c_coeffs[2, idx] * dx + c_coeffs[1, idx] * dx**2 + c_coeffs[0, idx] * dx**3
    dc_dz = c_coeffs[2, idx] + 2 * c_coeffs[1, idx] * dx + 3 * c_coeffs[0, idx] * dx**2
    d2c_dz2 = 2 * c_coeffs[1, idx] + 6 * c_coeffs[0, idx] * dx
    
    return c, dc_dz, d2c_dz2

@njit(fastmath=True)
def evaluate_n2linear_ssp(z_q: float, z_arr: np.ndarray, c_arr: np.ndarray) -> tuple[float, float, float]:
    """
    Evaluates sound speed and gradient using n^2 linear interpolation.
    1/c^2 is linear in z.
    Returns: (c, dc_dz, d2c_dz2)
    """
    n = len(z_arr)
    if z_q <= z_arr[0]:
        n1sq = 1.0 / (c_arr[0] * c_arr[0])
        n2sq = 1.0 / (c_arr[1] * c_arr[1])
        dn2_dz = (n2sq - n1sq) / (z_arr[1] - z_arr[0])
        nsq = n1sq + dn2_dz * (z_q - z_arr[0])
        c = 1.0 / np.sqrt(nsq)
        dc_dz = -0.5 * (c**3) * dn2_dz
        d2c_dz2 = 0.75 * (c**5) * (dn2_dz**2)
        return c, dc_dz, d2c_dz2
        
    if z_q >= z_arr[-1]:
        n1sq = 1.0 / (c_arr[-2] * c_arr[-2])
        n2sq = 1.0 / (c_arr[-1] * c_arr[-1])
        dn2_dz = (n2sq - n1sq) / (z_arr[-1] - z_arr[-2])
        nsq = n2sq + dn2_dz * (z_q - z_arr[-1])
        c = 1.0 / np.sqrt(nsq)
        dc_dz = -0.5 * (c**3) * dn2_dz
        d2c_dz2 = 0.75 * (c**5) * (dn2_dz**2)
        return c, dc_dz, d2c_dz2
        
    idx = np.searchsorted(z_arr, z_q) - 1
    z1, z2 = z_arr[idx], z_arr[idx+1]
    c1, c2 = c_arr[idx], c_arr[idx+1]
    
    n1sq = 1.0 / (c1 * c1)
    n2sq = 1.0 / (c2 * c2)
    
    dn2_dz = (n2sq - n1sq) / (z2 - z1)
    nsq = n1sq + dn2_dz * (z_q - z1)
    
    c = 1.0 / np.sqrt(nsq)
    dc_dz = -0.5 * (c**3) * dn2_dz
    d2c_dz2 = 0.75 * (c**5) * (dn2_dz**2)
    
    return c, dc_dz, d2c_dz2

@njit(fastmath=True)
def evaluate_quad2d_ssp(
    r_q: float, z_q: float, r_arr: np.ndarray, z_arr: np.ndarray, c_matrix: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Evaluate a Bellhop Q/range-dependent SSP with bilinear interpolation.
    Returns: (c, dc_dr, dc_dz, d2c_dz2). The Q branch is piecewise linear
    in range and depth, so d2c/dz2 is zero inside a cell.
    """
    nr = len(r_arr)
    nz = len(z_arr)

    ir = np.searchsorted(r_arr, r_q) - 1
    iz = np.searchsorted(z_arr, z_q) - 1
    if ir < 0:
        ir = 0
    elif ir > nr - 2:
        ir = nr - 2
    if iz < 0:
        iz = 0
    elif iz > nz - 2:
        iz = nz - 2

    dr = r_arr[ir + 1] - r_arr[ir]
    dz = z_arr[iz + 1] - z_arr[iz]
    if dr == 0.0:
        dr = 1.0
    if dz == 0.0:
        dz = 1.0

    sr = (r_q - r_arr[ir]) / dr
    sz = (z_q - z_arr[iz]) / dz
    if sr < 0.0:
        sr = 0.0
    elif sr > 1.0:
        sr = 1.0
    if sz < 0.0:
        sz = 0.0
    elif sz > 1.0:
        sz = 1.0

    c00 = c_matrix[iz, ir]
    c01 = c_matrix[iz, ir + 1]
    c10 = c_matrix[iz + 1, ir]
    c11 = c_matrix[iz + 1, ir + 1]

    c_top = c00 + sr * (c01 - c00)
    c_bottom = c10 + sr * (c11 - c10)
    c = c_top + sz * (c_bottom - c_top)

    dc_dr_top = (c01 - c00) / dr
    dc_dr_bottom = (c11 - c10) / dr
    dc_dr = dc_dr_top + sz * (dc_dr_bottom - dc_dr_top)
    dc_dz = (c_bottom - c_top) / dz

    return c, dc_dr, dc_dz, 0.0

@njit(fastmath=True)
def evaluate_hexahedral_ssp_3d(
    x_q: float, y_q: float, z_q: float,
    x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, c_grid: np.ndarray
) -> tuple[float, float, float, float, float]:
    """
    Evaluates a Bellhop3D H/hexahedral SSP with AT-compatible interpolation.
    Returns: (c, dc_dx, dc_dy, dc_dz, d2c_dz2). The H branch is piecewise
    linear in each cell, so second derivatives are zero inside a cell.
    """
    nx = len(x_arr)
    ny = len(y_arr)
    nz = len(z_arr)

    ix = np.searchsorted(x_arr, x_q) - 1
    iy = np.searchsorted(y_arr, y_q) - 1
    iz = np.searchsorted(z_arr, z_q) - 1

    if ix < 0:
        ix = 0
    elif ix > nx - 2:
        ix = nx - 2

    if iy < 0:
        iy = 0
    elif iy > ny - 2:
        iy = ny - 2

    if iz < 0:
        iz = 0
    elif iz > nz - 2:
        iz = nz - 2

    dx = x_arr[ix + 1] - x_arr[ix]
    dy = y_arr[iy + 1] - y_arr[iy]
    dz = z_arr[iz + 1] - z_arr[iz]

    s1 = (x_q - x_arr[ix]) / dx
    if s1 < 0.0:
        s1 = 0.0
    elif s1 > 1.0:
        s1 = 1.0

    s2 = (y_q - y_arr[iy]) / dy
    if s2 < 0.0:
        s2 = 0.0
    elif s2 > 1.0:
        s2 = 1.0

    s3_m = z_q - z_arr[iz]

    cz11 = (c_grid[ix, iy, iz + 1] - c_grid[ix, iy, iz]) / dz
    cz12 = (c_grid[ix + 1, iy, iz + 1] - c_grid[ix + 1, iy, iz]) / dz
    cz21 = (c_grid[ix, iy + 1, iz + 1] - c_grid[ix, iy + 1, iz]) / dz
    cz22 = (c_grid[ix + 1, iy + 1, iz + 1] - c_grid[ix + 1, iy + 1, iz]) / dz

    c11 = c_grid[ix, iy, iz] + s3_m * cz11
    c21 = c_grid[ix + 1, iy, iz] + s3_m * cz12
    c12 = c_grid[ix, iy + 1, iz] + s3_m * cz21
    c22 = c_grid[ix + 1, iy + 1, iz] + s3_m * cz22

    c1_y = c11 + s1 * (c21 - c11)
    c2_y = c12 + s1 * (c22 - c12)
    dc_dy = (c2_y - c1_y) / dy

    c1_x = c11 + s2 * (c12 - c11)
    c2_x = c21 + s2 * (c22 - c21)
    c = c1_x + s1 * (c2_x - c1_x)
    dc_dx = (c2_x - c1_x) / dx

    cz1 = cz11 + s2 * (cz21 - cz11)
    cz2 = cz12 + s2 * (cz22 - cz12)
    dc_dz = cz1 + s1 * (cz2 - cz1)

    return c, dc_dx, dc_dy, dc_dz, 0.0

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
        self.density_arr = np.array([layer.density for layer in ssp_model.data], dtype=np.float64)
        self.attenuation_arr = np.array([layer.attenuation for layer in ssp_model.data], dtype=np.float64)
        self.r_arr = np.array(ssp_model.ranges or [], dtype=np.float64)
        self.c_matrix = (
            None if ssp_model.c_matrix is None else np.asarray(ssp_model.c_matrix, dtype=np.float64)
        )
        self.x_arr = None if ssp_model.x is None else np.asarray(ssp_model.x, dtype=np.float64)
        self.y_arr = None if ssp_model.y is None else np.asarray(ssp_model.y, dtype=np.float64)
        self.ssp3d_z_arr = None if ssp_model.z is None else np.asarray(ssp_model.z, dtype=np.float64)
        self.c_grid = None if ssp_model.c is None else np.asarray(ssp_model.c, dtype=np.float64)
        
        # Precompute spline coefficients if necessary
        self.c_coeffs = None
        if self.type == "spline":
            # bc_type='natural' is a common default for acoustic profiles if no derivatives are given
            cs = CubicSpline(self.z_arr, self.c_arr, bc_type='natural')
            self.c_coeffs = cs.c  # Shape: (4, N-1)

    def evaluate(self, z: float) -> tuple[float, float, float]:
        """
        Python-level evaluation (useful for testing and plotting).
        For hot-loops, do not call this. Pass `z_arr`, `c_arr`, and `c_coeffs` 
        directly into the numba compiled functions.
        """
        if self.type == "c-linear":
            return evaluate_linear_ssp(z, self.z_arr, self.c_arr)
        elif self.type == "spline":
            return evaluate_spline_ssp(z, self.z_arr, self.c_coeffs)
        elif self.type == "n2-linear":
            return evaluate_n2linear_ssp(z, self.z_arr, self.c_arr)
        elif self.type == "quad-2d":
            c, _, dc_dz, d2c_dz2 = self.evaluate_range_depth(0.0, z)
            return c, dc_dz, d2c_dz2
        else:
            raise NotImplementedError(f"SSP type {self.type} not implemented yet.")

    def evaluate_range_depth(self, r: float, z: float) -> tuple[float, float, float, float]:
        """Evaluate a range-dependent SSP at ``(range_m, depth_m)``."""
        if self.type != "quad-2d":
            c, dc_dz, d2c_dz2 = self.evaluate(z)
            return c, 0.0, dc_dz, d2c_dz2
        if self.c_matrix is None or self.r_arr.size == 0:
            raise ValueError("quad-2d SSP requires ranges and c_matrix")
        if self.c_matrix.shape != (self.z_arr.size, self.r_arr.size):
            raise ValueError("quad-2d c_matrix shape must be (depth_count, range_count)")
        return evaluate_quad2d_ssp(float(r), float(z), self.r_arr, self.z_arr, self.c_matrix)

    def evaluate_3d(self, x: float, y: float, z: float) -> tuple[float, float, float, float, float]:
        """Evaluate a Bellhop3D hexahedral SSP at ``(x_m, y_m, depth_m)``."""
        if self.type != "hexahedral-3d":
            c, dc_dz, d2c_dz2 = self.evaluate(z)
            return c, 0.0, 0.0, dc_dz, d2c_dz2
        if self.x_arr is None or self.y_arr is None or self.ssp3d_z_arr is None or self.c_grid is None:
            raise ValueError("hexahedral-3d SSP requires x, y, z, and c grids")
        expected_shape = (self.x_arr.size, self.y_arr.size, self.ssp3d_z_arr.size)
        if self.c_grid.shape != expected_shape:
            raise ValueError("hexahedral-3d c grid shape must be (x_count, y_count, z_count)")
        return evaluate_hexahedral_ssp_3d(
            float(x),
            float(y),
            float(z),
            self.x_arr,
            self.y_arr,
            self.ssp3d_z_arr,
            self.c_grid,
        )

    def evaluate_density(self, z: float) -> float:
        """Linearly interpolate density on the SSP depth grid."""
        return float(np.interp(z, self.z_arr, self.density_arr))
