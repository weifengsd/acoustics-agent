import numpy as np
from pyacoustics.schema import SimulationConfig
from pyacoustics.environment import SSPInterpolator
from .core import build_acoustic_matrix, top_impedance, bottom_impedance
from .roots import bisection_search_roots, secant_search_roots, neville_extrapolation
from .modes import compute_modes
from .field import compute_field

class PyKraken:
    """
    Main interface for the PyKraken Normal Modes solver.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.ssp = SSPInterpolator(config.environment.ssp)
        self.omega = 2.0 * np.pi * config.frequency
        
    def run_normal_modes(self):
        """
        Executes the normal modes simulation and returns the computed TL field.
        """
        depth_max = self.config.environment.bottom.depth
        if depth_max is None:
            depth_max = self.ssp.z_arr[-1]
            
        # Parse phase speed limits
        c_min, c_max = self.config.solver.phase_speed_limits
        k2_max = (self.omega / max(c_min, 1.0))**2 if c_min > 0 else (self.omega / 1400.0)**2
        k2_min = (self.omega / c_max)**2
        
        # We need a robust maximum c for k2_max if c_min is 0. 
        # Typically c_min is around 1400 in water. Let's find min sound speed in profile.
        if c_min <= 0:
            c_min = np.min(self.ssp.c_arr)
            k2_max = (self.omega / max(c_min - 100, 1000.0))**2
            
        N = self.config.solver.n_mesh_points
        h = depth_max / N
        z_arr = np.linspace(0, depth_max, N + 1)
        
        # Build environment arrays on the fine grid
        c_arr = np.zeros(N + 1)
        for i, z in enumerate(z_arr):
            c_arr[i], _, _ = self.ssp.evaluate(z)
            
        # For this version, assume density is 1.0 in water column
        density_arr = np.ones(N + 1)
        
        B1 = build_acoustic_matrix(h, self.omega, c_arr)
        
        # Parse boundaries
        top_bc = self.config.environment.surface.type
        top_bc_type = 0 if top_bc == "vacuum" else 1
        f_top, g_top = top_impedance(top_bc_type)
        
        bot_bc = self.config.environment.bottom.type
        bot_c_p = self.config.environment.bottom.c_p or 1600.0
        bot_rho = self.config.environment.bottom.density or 1.8 
        if bot_rho > 10.0:
            bot_rho /= 1000.0 
            
        if bot_bc == "vacuum":
            bot_bc_type = 0
        elif bot_bc == "rigid":
            bot_bc_type = 1
        else: # acousto-elastic or halfspace
            bot_bc_type = 2

        # 1. Bisection search on the base grid
        roots_base = bisection_search_roots(k2_min, k2_max, h, B1, self.omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type)
        
        if len(roots_base) == 0:
            # Return empty field if no modes
            return np.full((100, 200), 100.0)
            
        # 2. Richardson Extrapolation
        mults = self.config.solver.mesh_multiplier
        if len(mults) > 1:
            seqs = []
            for m in mults:
                # We need to refine the root around roots_base using secant on finer grids
                h_m = h / m
                N_m = N * m
                z_arr_m = np.linspace(0, depth_max, N_m + 1)
                c_arr_m = np.zeros(N_m + 1)
                for i, z in enumerate(z_arr_m):
                    c_arr_m[i], _, _ = self.ssp.evaluate(z)
                B1_m = build_acoustic_matrix(h_m, self.omega, c_arr_m)
                
                # Secant requires initial guesses. We use roots_base.
                roots_m = secant_search_roots(roots_base, h_m, B1_m, self.omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type)
                seqs.append(roots_m)
                
            final_roots = neville_extrapolation(seqs, mults)
        else:
            final_roots = roots_base
            
        # 3. Modes computation and modal attenuation perturbation
        bot_attenuation_p = self.config.environment.bottom.attenuation_p or 0.0
        modes, complex_roots = compute_modes(
            final_roots, h, B1, self.omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type, density_arr,
            bot_attenuation_p=bot_attenuation_p
        )
        
        # 4. Field synthesis
        ranges = self.config.geometry.receivers.ranges
        if len(ranges) > 2:
            r_arr = np.array(ranges)
            # Avoid exactly 0 range for TL log
            if r_arr[0] <= 0:
                r_arr[0] = 1.0
        else:
            r_max = max(ranges)
            r_arr = np.linspace(100.0, r_max, 200)
            
        depths = self.config.geometry.receivers.depths
        if len(depths) > 2:
            z_r_arr = np.array(depths)
        else:
            z_r_arr = np.linspace(0, depth_max, 100)
            
        z_s = self.config.geometry.source.depths[0]
        TL = compute_field(modes, complex_roots, z_arr, r_arr, z_s, z_r_arr)
        
        return {
            'tl_grid': TL,
            'r_bins': r_arr,
            'z_bins': z_r_arr,
            'modes': modes,
            'roots': final_roots
        }
