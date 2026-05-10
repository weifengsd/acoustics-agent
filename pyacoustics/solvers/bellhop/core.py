import numpy as np
from typing import List, Tuple, Optional
from pyacoustics.schema import SimulationConfig
from pyacoustics.environment import SSPInterpolator
from pyacoustics.solvers.bellhop.tracer import trace_single_ray, trace_beam_ray, MAX_STEPS

class PyBellhop:
    """Core orchestrator for the Bellhop ray tracing solver."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.ssp_interp = SSPInterpolator(config.environment.ssp)
        
        # Determine boundaries
        self.z_surface = config.environment.surface.depth if config.environment.surface.depth is not None else 0.0
        self.z_bottom = config.environment.bottom.depth
        
        # Bottom properties for reflection coefficients
        self.bot_rho = config.environment.bottom.density if hasattr(config.environment.bottom, 'density') and config.environment.bottom.density else 1.8
        self.bot_c = config.environment.bottom.c_p if hasattr(config.environment.bottom, 'c_p') and config.environment.bottom.c_p else 1600.0
        self.bot_alpha = config.environment.bottom.attenuation_p if hasattr(config.environment.bottom, 'attenuation_p') and config.environment.bottom.attenuation_p else 0.0
        
        # Max trace range is the max receiver range
        self.r_max = max(config.geometry.receivers.ranges)

        # Determine step size
        self.h = config.solver.step_size
        if self.h <= 0.0:
            # Adaptive step size: Try to use a step size that resolves the path
            # within a safe margin of MAX_STEPS (100,000). We aim for max 50,000 steps.
            self.h = max(self.r_max / 50000.0, 1.0)

    def _prepare_ssp_arrays(self):
        """Prepare SSP arrays for Numba-compiled functions."""
        ssp_type_int = 0 if self.ssp_interp.type == "c-linear" else 1
        z_arr = self.ssp_interp.z_arr
        c_arr = self.ssp_interp.c_arr
        c_coeffs = self.ssp_interp.c_coeffs if self.ssp_interp.c_coeffs is not None else np.zeros((4, 1))
        return ssp_type_int, z_arr, c_arr, c_coeffs

    def _generate_angles(self):
        """Generate the fan of launch angles."""
        alpha_start, alpha_end = self.config.solver.angles
        num_beams = self.config.solver.num_beams
        
        if num_beams == 0:
            # Auto-calculate optimal number of beams (Bellhop logic)
            z_src = self.config.geometry.source.depths[0]
            ssp_type_int, z_arr, c_arr, c_coeffs = self._prepare_ssp_arrays()
            from pyacoustics.environment import evaluate_linear_ssp, evaluate_spline_ssp
            if ssp_type_int == 0:
                c_src, _, _ = evaluate_linear_ssp(z_src, z_arr, c_arr)
            else:
                c_src, _, _ = evaluate_spline_ssp(z_src, z_arr, c_coeffs)
            
            freq = self.config.frequency
            r_max = self.r_max if self.r_max > 0 else 1000.0
            
            # Optimal angular spacing in radians
            d_alpha_opt = np.sqrt(c_src / (6.0 * freq * r_max))
            d_alpha_opt_deg = np.degrees(d_alpha_opt)
            
            span_deg = abs(alpha_end - alpha_start)
            num_beams = 2 + int(span_deg / d_alpha_opt_deg)
            self.config.solver.num_beams = num_beams

        if num_beams <= 1:
            return np.array([alpha_start])
        return np.linspace(alpha_start, alpha_end, num_beams)

    def run_ray_tracing(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Executes ray tracing for a fan of rays.
        Returns a list of (r_path, z_path, amp_path) arrays for each traced ray.
        """
        results = []
        ssp_type_int, z_arr, c_arr, c_coeffs = self._prepare_ssp_arrays()
        angles = self._generate_angles()

        # Loop over sources and angles
        for z_src in self.config.geometry.source.depths:
            for alpha_deg in angles:
                r_path, z_path, amp_path, n_pts, _ = trace_single_ray(
                    r_0=0.0,
                    z_0=z_src,
                    alpha_deg=alpha_deg,
                    h=self.h,
                    z_surface=self.z_surface,
                    z_bottom=self.z_bottom,
                    bot_rho=self.bot_rho,
                    bot_c=self.bot_c,
                    bot_alpha=self.bot_alpha,
                    r_max=self.r_max,
                    ssp_type=ssp_type_int,
                    z_arr=z_arr,
                    c_arr=c_arr,
                    c_coeffs=c_coeffs,
                    max_steps=MAX_STEPS
                )
                
                # Slice to valid points
                results.append((r_path[:n_pts], z_path[:n_pts], amp_path[:n_pts]))
                
        return results

    def run_beam_tracing(self) -> dict:
        """
        Executes beam tracing for a fan of rays.
        Returns structured beam data needed for Gaussian Beam Summation.
        """
        ssp_type_int, z_arr, c_arr, c_coeffs = self._prepare_ssp_arrays()
        angles = self._generate_angles()

        z_src = self.config.geometry.source.depths[0]

        # Get source sound speed
        from pyacoustics.environment import evaluate_linear_ssp, evaluate_spline_ssp
        if ssp_type_int == 0:
            c_src, _, _ = evaluate_linear_ssp(z_src, z_arr, c_arr)
        else:
            c_src, _, _ = evaluate_spline_ssp(z_src, z_arr, c_coeffs)

        num_beams = len(angles)

        # Determine max points per ray to pre-allocate
        max_pts = min(MAX_STEPS, int(self.r_max / self.h) + 1000)

        # Pre-allocate 2D arrays for all rays
        all_r = np.zeros((num_beams, max_pts))
        all_z = np.zeros((num_beams, max_pts))
        all_c = np.zeros((num_beams, max_pts))
        all_tau = np.zeros((num_beams, max_pts))
        all_p = np.zeros((num_beams, max_pts))
        all_q = np.zeros((num_beams, max_pts))
        all_amp = np.zeros((num_beams, max_pts), dtype=np.complex128)
        all_npts = np.zeros(num_beams, dtype=np.int64)
        all_alphas = np.zeros(num_beams)

        for i, alpha_deg in enumerate(angles):
            r_path, z_path, c_path, tau_path, p_path, q_path, amp_path, n_pts, _ = trace_beam_ray(
                r_0=0.0,
                z_0=z_src,
                alpha_deg=alpha_deg,
                h=self.h,
                z_surface=self.z_surface,
                z_bottom=self.z_bottom,
                bot_rho=self.bot_rho,
                bot_c=self.bot_c,
                bot_alpha=self.bot_alpha,
                r_max=self.r_max,
                ssp_type=ssp_type_int,
                z_arr=z_arr,
                c_arr=c_arr,
                c_coeffs=c_coeffs,
                max_steps=max_pts - 1
            )
            n = min(n_pts, max_pts)
            all_r[i, :n] = r_path[:n]
            all_z[i, :n] = z_path[:n]
            all_c[i, :n] = c_path[:n]
            all_tau[i, :n] = tau_path[:n]
            all_p[i, :n] = p_path[:n]
            all_q[i, :n] = q_path[:n]
            all_amp[i, :n] = amp_path[:n]
            all_npts[i] = n
            all_alphas[i] = np.radians(alpha_deg)

        return {
            'ray_r': all_r,
            'ray_z': all_z,
            'ray_c': all_c,
            'ray_tau': all_tau,
            'ray_p': all_p,
            'ray_q': all_q,
            'ray_amp': all_amp,
            'ray_npts': all_npts,
            'ray_alphas': all_alphas,
            'c_src': float(c_src),
            'num_beams': num_beams,
        }
