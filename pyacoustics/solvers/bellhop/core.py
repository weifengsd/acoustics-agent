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

        # Parse surface altimetry. Values are z coordinates in meters
        # with positive depth downward, matching the internal ray state.
        if config.environment.surface.altimetry:
            ati_data = np.array(config.environment.surface.altimetry, dtype=np.float64)
            self.ati_r = ati_data[:, 0]
            self.ati_z = ati_data[:, 1]
        else:
            self.ati_r = np.array([0.0, 1e9], dtype=np.float64)
            self.ati_z = np.array([self.z_surface, self.z_surface], dtype=np.float64)
        self.ati_curvilinear = 1 if getattr(config.environment.surface, "altimetry_interpolation", None) == "C" else 0

        if config.environment.surface.altimetry_geoacoustics:
            ati_geo = np.array(config.environment.surface.altimetry_geoacoustics, dtype=np.float64)
            self.ati_geo_cp = ati_geo[:, 0]
            self.ati_geo_cs = ati_geo[:, 1]
            self.ati_geo_rho = ati_geo[:, 2]
            self.ati_geo_alpha = ati_geo[:, 3]
            self.ati_geo_alpha_s = ati_geo[:, 4]
        else:
            self.ati_geo_cp = np.zeros(0, dtype=np.float64)
            self.ati_geo_cs = np.zeros(0, dtype=np.float64)
            self.ati_geo_rho = np.zeros(0, dtype=np.float64)
            self.ati_geo_alpha = np.zeros(0, dtype=np.float64)
            self.ati_geo_alpha_s = np.zeros(0, dtype=np.float64)
        
        # Parse Bathymetry
        if config.environment.bottom.bathymetry:
            bty_data = np.array(config.environment.bottom.bathymetry, dtype=np.float64)
            self.bty_r = bty_data[:, 0]
            self.bty_z = bty_data[:, 1]
        else:
            self.bty_r = np.array([0.0, 1e9], dtype=np.float64)
            self.bty_z = np.array([self.z_bottom, self.z_bottom], dtype=np.float64)
        self.bty_curvilinear = 1 if getattr(config.environment.bottom, "bathymetry_interpolation", None) == "C" else 0

        if config.environment.bottom.bathymetry_geoacoustics:
            bty_geo = np.array(config.environment.bottom.bathymetry_geoacoustics, dtype=np.float64)
            self.bty_geo_cp = bty_geo[:, 0]
            self.bty_geo_cs = bty_geo[:, 1]
            self.bty_geo_rho = bty_geo[:, 2]
            self.bty_geo_alpha = bty_geo[:, 3]
            self.bty_geo_alpha_s = bty_geo[:, 4]
        else:
            self.bty_geo_cp = np.zeros(0, dtype=np.float64)
            self.bty_geo_cs = np.zeros(0, dtype=np.float64)
            self.bty_geo_rho = np.zeros(0, dtype=np.float64)
            self.bty_geo_alpha = np.zeros(0, dtype=np.float64)
            self.bty_geo_alpha_s = np.zeros(0, dtype=np.float64)

        self.surface_refl_angles, self.surface_refl_amp, self.surface_refl_phase = self._reflection_arrays(
            config.environment.surface.reflection_table
        )
        self.bottom_refl_angles, self.bottom_refl_amp, self.bottom_refl_phase = self._reflection_arrays(
            config.environment.bottom.reflection_table
        )

        surface_type = getattr(config.environment.surface, "type", "vacuum")
        self.surf_rho = config.environment.surface.density if getattr(config.environment.surface, "density", None) else 1.0
        self.surf_c_s = config.environment.surface.c_s if getattr(config.environment.surface, "c_s", None) else 0.0
        self.surf_alpha = config.environment.surface.attenuation_p if getattr(config.environment.surface, "attenuation_p", None) else 0.0
        self.surf_alpha_s = config.environment.surface.attenuation_s if getattr(config.environment.surface, "attenuation_s", None) else 0.0
        if surface_type == "rigid":
            self.surf_c = -1.0
        elif surface_type == "acousto-elastic":
            self.surf_c = config.environment.surface.c_p if getattr(config.environment.surface, "c_p", None) else 1600.0
        else:
            self.surf_c = 0.0

        # Bottom properties for reflection coefficients
        bottom_type = getattr(config.environment.bottom, "type", "acousto-elastic")
        self.bot_rho = config.environment.bottom.density if hasattr(config.environment.bottom, 'density') and config.environment.bottom.density else 1.8
        self.bot_c_s = config.environment.bottom.c_s if hasattr(config.environment.bottom, 'c_s') and config.environment.bottom.c_s else 0.0
        self.bot_alpha = config.environment.bottom.attenuation_p if hasattr(config.environment.bottom, 'attenuation_p') and config.environment.bottom.attenuation_p else 0.0
        self.bot_alpha_s = config.environment.bottom.attenuation_s if hasattr(config.environment.bottom, 'attenuation_s') and config.environment.bottom.attenuation_s else 0.0
        if bottom_type in ("vacuum", "rigid"):
            self.bot_c = 0.0 if bottom_type == "vacuum" else -1.0
            self.bot_c_s = 0.0
            self.bot_alpha = 0.0
            self.bot_alpha_s = 0.0
        else:
            self.bot_c = config.environment.bottom.c_p if hasattr(config.environment.bottom, 'c_p') and config.environment.bottom.c_p else 1600.0
        
        # Max trace range is the max receiver range
        self.r_max = max(config.geometry.receivers.ranges)

        # Determine step size
        self.h = config.solver.step_size
        if self.h <= 0.0:
            # Adaptive step size: Try to use a step size that resolves the path
            # within a safe margin of MAX_STEPS (100,000). We aim for max 50,000 steps.
            self.h = max(self.r_max / 50000.0, 1.0)
        self.ssp_r_arr = self.ssp_interp.r_arr

    @staticmethod
    def _reflection_arrays(table):
        if not table:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty
        arr = np.array(table, dtype=np.float64)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        return arr[:, 0], arr[:, 1], arr[:, 2]

    def _prepare_ssp_arrays(self):
        """Prepare SSP arrays for Numba-compiled functions."""
        if self.ssp_interp.type == "c-linear":
            ssp_type_int = 0
        elif self.ssp_interp.type == "spline":
            ssp_type_int = 1
        elif self.ssp_interp.type == "n2-linear":
            ssp_type_int = 2
        elif self.ssp_interp.type == "quad-2d":
            ssp_type_int = 3
        else:
            raise NotImplementedError(f"SSP type {self.ssp_interp.type} not implemented yet.")
            
        z_arr = self.ssp_interp.z_arr
        c_arr = self.ssp_interp.c_arr
        if ssp_type_int == 3:
            if self.ssp_interp.c_matrix is None:
                raise ValueError("quad-2d SSP requires c_matrix")
            c_coeffs = self.ssp_interp.c_matrix
        else:
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
            from pyacoustics.environment import evaluate_linear_ssp, evaluate_spline_ssp, evaluate_n2linear_ssp
            if ssp_type_int == 3:
                c_src, _, _, _ = self.ssp_interp.evaluate_range_depth(0.0, z_src)
            elif ssp_type_int == 0:
                c_src, _, _ = evaluate_linear_ssp(z_src, z_arr, c_arr)
            elif ssp_type_int == 1:
                c_src, _, _ = evaluate_spline_ssp(z_src, z_arr, c_coeffs)
            else:
                c_src, _, _ = evaluate_n2linear_ssp(z_src, z_arr, c_arr)
            
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
                    ati_r=self.ati_r,
                    ati_z=self.ati_z,
                    ati_curvilinear=self.ati_curvilinear,
                    bty_r=self.bty_r,
                    bty_z=self.bty_z,
                    bty_curvilinear=self.bty_curvilinear,
                    bty_geo_cp=self.bty_geo_cp,
                    bty_geo_cs=self.bty_geo_cs,
                    bty_geo_rho=self.bty_geo_rho,
                    bty_geo_alpha=self.bty_geo_alpha,
                    bty_geo_alpha_s=self.bty_geo_alpha_s,
                    ati_geo_cp=self.ati_geo_cp,
                    ati_geo_cs=self.ati_geo_cs,
                    ati_geo_rho=self.ati_geo_rho,
                    ati_geo_alpha=self.ati_geo_alpha,
                    ati_geo_alpha_s=self.ati_geo_alpha_s,
                    surface_refl_angles=self.surface_refl_angles,
                    surface_refl_amp=self.surface_refl_amp,
                    surface_refl_phase=self.surface_refl_phase,
                    bottom_refl_angles=self.bottom_refl_angles,
                    bottom_refl_amp=self.bottom_refl_amp,
                    bottom_refl_phase=self.bottom_refl_phase,
                    surf_rho=self.surf_rho,
                    surf_c=self.surf_c,
                    surf_c_s=self.surf_c_s,
                    surf_alpha=self.surf_alpha,
                    surf_alpha_s=self.surf_alpha_s,
                    bot_rho=self.bot_rho,
                    bot_c=self.bot_c,
                    bot_c_s=self.bot_c_s,
                    bot_alpha=self.bot_alpha,
                    bot_alpha_s=self.bot_alpha_s,
                    r_max=self.r_max,
                    ssp_type=ssp_type_int,
                    z_arr=z_arr,
                    c_arr=c_arr,
                    c_coeffs=c_coeffs,
                    ssp_r_arr=self.ssp_r_arr,
                    max_steps=MAX_STEPS
                )
                
                # Slice to valid points
                results.append((r_path[:n_pts], z_path[:n_pts], amp_path[:n_pts]))
                
        return results

    def run_beam_tracing(self, source_depth: float | None = None) -> dict:
        """
        Executes beam tracing for a fan of rays.
        Returns structured beam data needed for Gaussian Beam Summation.
        """
        ssp_type_int, z_arr, c_arr, c_coeffs = self._prepare_ssp_arrays()
        angles = self._generate_angles()

        z_src = self.config.geometry.source.depths[0] if source_depth is None else float(source_depth)

        # Get source sound speed
        from pyacoustics.environment import evaluate_linear_ssp, evaluate_spline_ssp, evaluate_n2linear_ssp
        if ssp_type_int == 3:
            c_src, _, _, _ = self.ssp_interp.evaluate_range_depth(0.0, z_src)
        elif ssp_type_int == 0:
            c_src, _, _ = evaluate_linear_ssp(z_src, z_arr, c_arr)
        elif ssp_type_int == 1:
            c_src, _, _ = evaluate_spline_ssp(z_src, z_arr, c_coeffs)
        else:
            c_src, _, _ = evaluate_n2linear_ssp(z_src, z_arr, c_arr)

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
        all_phase = np.zeros((num_beams, max_pts))
        all_top_bnc = np.zeros((num_beams, max_pts), dtype=np.int64)
        all_bot_bnc = np.zeros((num_beams, max_pts), dtype=np.int64)
        all_npts = np.zeros(num_beams, dtype=np.int64)
        all_alphas = np.zeros(num_beams)

        for i, alpha_deg in enumerate(angles):
            displacement_mode = 1 if getattr(self.config.solver, "beam_displacement", "none") == "seongil" else 0
            (
                r_path, z_path, c_path, tau_path, p_path, q_path,
                amp_path, phase_path, top_bnc_path, bot_bnc_path,
                n_pts, _,
            ) = trace_beam_ray(
                r_0=0.0,
                z_0=z_src,
                alpha_deg=alpha_deg,
                h=self.h,
                ati_r=self.ati_r,
                ati_z=self.ati_z,
                ati_curvilinear=self.ati_curvilinear,
                bty_r=self.bty_r,
                bty_z=self.bty_z,
                bty_curvilinear=self.bty_curvilinear,
                bty_geo_cp=self.bty_geo_cp,
                bty_geo_cs=self.bty_geo_cs,
                bty_geo_rho=self.bty_geo_rho,
                bty_geo_alpha=self.bty_geo_alpha,
                bty_geo_alpha_s=self.bty_geo_alpha_s,
                ati_geo_cp=self.ati_geo_cp,
                ati_geo_cs=self.ati_geo_cs,
                ati_geo_rho=self.ati_geo_rho,
                ati_geo_alpha=self.ati_geo_alpha,
                ati_geo_alpha_s=self.ati_geo_alpha_s,
                surface_refl_angles=self.surface_refl_angles,
                surface_refl_amp=self.surface_refl_amp,
                surface_refl_phase=self.surface_refl_phase,
                bottom_refl_angles=self.bottom_refl_angles,
                bottom_refl_amp=self.bottom_refl_amp,
                bottom_refl_phase=self.bottom_refl_phase,
                surf_rho=self.surf_rho,
                surf_c=self.surf_c,
                surf_c_s=self.surf_c_s,
                surf_alpha=self.surf_alpha,
                surf_alpha_s=self.surf_alpha_s,
                bot_rho=self.bot_rho,
                bot_c=self.bot_c,
                bot_c_s=self.bot_c_s,
                bot_alpha=self.bot_alpha,
                bot_alpha_s=self.bot_alpha_s,
                frequency=self.config.frequency,
                beam_displacement=displacement_mode,
                r_max=self.r_max,
                ssp_type=ssp_type_int,
                z_arr=z_arr,
                c_arr=c_arr,
                c_coeffs=c_coeffs,
                ssp_r_arr=self.ssp_r_arr,
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
            all_phase[i, :n] = phase_path[:n]
            all_top_bnc[i, :n] = top_bnc_path[:n]
            all_bot_bnc[i, :n] = bot_bnc_path[:n]
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
            'ray_phase': all_phase,
            'ray_top_bnc': all_top_bnc,
            'ray_bot_bnc': all_bot_bnc,
            'ray_npts': all_npts,
            'ray_alphas': all_alphas,
            'c_src': float(c_src),
            'num_beams': num_beams,
        }
