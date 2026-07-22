import numpy as np
from pyacoustics.schema import SimulationConfig
from pyacoustics.environment import SSPInterpolator
from .core import build_acoustic_matrix, build_acoustic_matrix_complex, top_impedance, bottom_impedance, complex_sound_speed
from .roots import (
    bisection_search_roots_media,
    neville_extrapolation,
    secant_search_roots_media,
    secant_search_roots_media_complex,
)
from .modes import compute_mode_shapes_complex, compute_modes
from .field import compute_field


def _field_axis_values(values, *, start_floor=None, interpolated_count=1001):
    axis = np.asarray(values, dtype=np.float64)
    if axis.size == 0:
        raise ValueError("Kraken Field axis requires at least one value")
    if axis.size == 1:
        return axis.copy()
    if axis.size == 2:
        start = float(axis[0])
        stop = float(axis[1])
        if start_floor is not None and start <= 0.0:
            start = float(start_floor)
        return np.linspace(start, stop, interpolated_count)
    out = axis.copy()
    if start_floor is not None and out[0] <= 0.0:
        out[0] = float(start_floor)
    return out


class PyKraken:
    """
    Main interface for the PyKraken Normal Modes solver.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.ssp = SSPInterpolator(config.environment.ssp)
        self.omega = 2.0 * np.pi * config.frequency

    @staticmethod
    def write_modes(result, filename, title="PyAcoustics native Kraken"):
        from pyacoustics.solvers.external_io import write_at_mod

        write_at_mod(
            filename,
            result["modes"],
            result["z_bins_mod"],
            title=title,
            wavenumbers=result.get("wavenumbers"),
            metadata=result.get("mode_metadata"),
        )

    def _mode_diagnostics(self, roots):
        roots = np.asarray(roots, dtype=np.complex128)
        wavenumbers = np.sqrt(roots)
        phase_speeds = np.full(roots.shape, np.nan, dtype=np.float64)
        propagating = np.real(wavenumbers) > 0.0
        phase_speeds[propagating] = self.omega / np.real(wavenumbers[propagating])
        return {
            "complex_roots": roots,
            "wavenumbers": wavenumbers,
            "phase_speeds": phase_speeds,
            "modal_attenuation": np.imag(wavenumbers),
        }

    def _acoustic_media(self, z_arr):
        depths = self.ssp.z_arr
        densities = self.ssp.density_arr
        if getattr(self.config.solver, "acoustic_media_model", "auto") == "single":
            return (
                np.array([0], dtype=np.int64),
                np.array([len(z_arr) - 1], dtype=np.int64),
                np.array([float(densities[0]) if len(densities) else 1.0], dtype=np.float64),
            )
        starts = []
        ends = []
        rhos = []

        for i in range(len(depths) - 1):
            start = int(np.searchsorted(z_arr, depths[i], side="left"))
            end = int(np.searchsorted(z_arr, depths[i + 1], side="left"))
            if i == len(depths) - 2:
                end = len(z_arr) - 1
            start = max(0, min(start, len(z_arr) - 1))
            end = max(start + 1, min(end, len(z_arr) - 1))
            rho = float(densities[i])

            if starts and start <= ends[-1] and np.isclose(rhos[-1], rho):
                ends[-1] = end
            else:
                starts.append(start)
                ends.append(end)
                rhos.append(rho)

        if not starts:
            starts = [0]
            ends = [len(z_arr) - 1]
            rhos = [float(densities[0]) if len(densities) else 1.0]

        return (
            np.array(starts, dtype=np.int64),
            np.array(ends, dtype=np.int64),
            np.array(rhos, dtype=np.float64),
        )

    def _mode_metadata(
        self,
        z_arr,
        media_start,
        media_end,
        media_rho,
        bot_bc_type,
        bot_c_p,
        bot_c_s,
        bot_rho,
        bot_attenuation_p=0.0,
        bot_attenuation_s=0.0,
    ):
        media = []
        for start, end, rho in zip(media_start, media_end, media_rho):
            media.append(
                {
                    "n": int(end - start),
                    "material": "ACOUSTIC",
                    "depth": float(z_arr[int(start)]),
                    "rho": float(rho),
                }
            )

        top_bc = "F" if self._reflection_table(self.config.environment.surface) is not None else (
            "V" if self.config.environment.surface.type == "vacuum" else "R"
        )
        if bot_bc_type == 4:
            bottom_bc = "F"
        elif bot_bc_type == 0:
            bottom_bc = "V"
        elif bot_bc_type == 1:
            bottom_bc = "R"
        else:
            bottom_bc = "A"

        return {
            "frequency": float(self.config.frequency),
            "freq_vec": [float(self.config.frequency)],
            "media": media,
            "top": {
                "bc": top_bc,
                "c_p": 0.0,
                "c_s": 0.0,
                "rho": 0.0,
                "depth": float(z_arr[0]) if len(z_arr) else 0.0,
            },
            "bottom": {
                "bc": bottom_bc,
                "c_p": complex_sound_speed(bot_c_p, bot_attenuation_p),
                "c_s": complex_sound_speed(bot_c_s, bot_attenuation_s) if bot_c_s else 0.0 + 0.0j,
                "rho": float(bot_rho),
                "depth": float(self.config.environment.bottom.depth or (z_arr[-1] if len(z_arr) else 0.0)),
            },
        }

    def _needs_complex_root_refinement(self, attenuation_arr, bot_attenuation_p, bot_attenuation_s, bot_bc_type):
        if np.any(np.asarray(attenuation_arr) > 0.0):
            return True
        if bot_attenuation_p > 0.0 or bot_attenuation_s > 0.0:
            return True
        return bot_bc_type in {3, 4}

    @staticmethod
    def _reflection_table(boundary):
        table = getattr(boundary, "reflection_table", None)
        if table is None:
            return None
        return np.asarray(table, dtype=np.float64)

    def _refine_complex_roots(
        self,
        initial_roots,
        h,
        c_arr,
        attenuation_arr,
        bot_bc_type,
        bot_c_p,
        bot_rho,
        top_bc_type,
        media_start,
        media_end,
        media_rho,
        bot_c_s,
        bot_attenuation_p,
        bot_attenuation_s,
        top_reflection_table=None,
        bot_reflection_table=None,
    ):
        if len(initial_roots) == 0:
            return np.asarray(initial_roots, dtype=np.complex128)

        B1_complex = build_acoustic_matrix_complex(h, self.omega, c_arr, attenuation_arr)
        roots = secant_search_roots_media_complex(
            initial_roots,
            h,
            B1_complex,
            self.omega,
            bot_bc_type,
            bot_c_p,
            bot_rho,
            top_bc_type,
            media_start,
            media_end,
            media_rho,
            bot_c_s,
            bot_attenuation_p,
            bot_attenuation_s,
            float(c_arr[0]),
            float(media_rho[0]),
            float(c_arr[-1]),
            float(media_rho[-1]),
            top_reflection_table,
            bot_reflection_table,
        )

        wavenumbers = np.sqrt(roots)
        growth = np.imag(wavenumbers) < 0.0
        roots[growth] = np.conj(roots[growth])
        return roots

    def _compute_complex_modes(
        self,
        complex_roots,
        h,
        c_arr,
        attenuation_arr,
        bot_bc_type,
        bot_c_p,
        bot_rho,
        top_bc_type,
        media_start,
        media_end,
        media_rho,
        bot_c_s,
        bot_attenuation_p,
        bot_attenuation_s,
        top_reflection_table=None,
        bot_reflection_table=None,
    ):
        B1_complex = build_acoustic_matrix_complex(h, self.omega, c_arr, attenuation_arr)
        return compute_mode_shapes_complex(
            complex_roots,
            h,
            B1_complex,
            self.omega,
            bot_bc_type,
            bot_c_p,
            bot_rho,
            top_bc_type,
            media_start,
            media_end,
            media_rho,
            bot_c_s=bot_c_s,
            bot_attenuation_p=bot_attenuation_p,
            bot_attenuation_s=bot_attenuation_s,
            c_top_inside=float(c_arr[0]),
            rho_top_inside=float(media_rho[0]),
            c_bot_inside=float(c_arr[-1]),
            rho_bot_inside=float(media_rho[-1]),
            top_reflection_table=top_reflection_table,
            bot_reflection_table=bot_reflection_table,
        )
        
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
            
        density_arr = np.interp(z_arr, self.ssp.z_arr, self.ssp.density_arr)
        attenuation_arr = np.interp(z_arr, self.ssp.z_arr, self.ssp.attenuation_arr)
        medium_rho = float(density_arr[0])
        media_start, media_end, media_rho = self._acoustic_media(z_arr)
        
        B1 = build_acoustic_matrix(h, self.omega, c_arr)
        
        # Parse boundaries
        top_bc = self.config.environment.surface.type
        top_reflection_table = self._reflection_table(self.config.environment.surface)
        bot_reflection_table = self._reflection_table(self.config.environment.bottom)
        top_bc_type = 4 if top_reflection_table is not None else (0 if top_bc == "vacuum" else 1)
        f_top, g_top = top_impedance(top_bc_type)
        
        bot_bc = self.config.environment.bottom.type
        bot_c_p = self.config.environment.bottom.c_p or 1600.0
        bot_c_s = self.config.environment.bottom.c_s or 0.0
        bot_rho = self.config.environment.bottom.density or 1.8 
        if bot_rho > 10.0:
            bot_rho /= 1000.0 
            
        if bot_reflection_table is not None:
            bot_bc_type = 4
        elif bot_bc == "vacuum":
            bot_bc_type = 0
        elif bot_bc == "rigid":
            bot_bc_type = 1
        elif bot_c_s > 0.0:
            bot_bc_type = 3
        else: # acousto-elastic acoustic halfspace
            bot_bc_type = 2

        # 1. Bisection search on the base grid
        roots_base = bisection_search_roots_media(
            k2_min,
            k2_max,
            h,
            B1,
            self.omega,
            bot_bc_type,
            bot_c_p,
            bot_rho,
            top_bc_type,
            media_start,
            media_end,
            media_rho,
            bot_c_s,
        )
        
        if len(roots_base) == 0:
            # Return empty results if no modes
            r_max = max(self.config.geometry.receivers.ranges)
            r_arr = np.linspace(100.0, r_max, 200)
            z_r_arr = np.linspace(0, depth_max, 100)
            return {
                'tl_grid': np.full((100, 200), 120.0),
                'r_bins': r_arr,
                'z_bins': z_r_arr,
                'modes': np.array([]),
                'roots': np.array([]),
                'mode_metadata': self._mode_metadata(
                    z_arr,
                    media_start,
                    media_end,
                    media_rho,
                    bot_bc_type,
                    bot_c_p,
                    bot_c_s,
                    bot_rho,
                    self.config.environment.bottom.attenuation_p or 0.0,
                    self.config.environment.bottom.attenuation_s or 0.0,
                ),
                **self._mode_diagnostics(np.array([], dtype=np.complex128)),
            }
            
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
                media_start_m, media_end_m, media_rho_m = self._acoustic_media(z_arr_m)
                roots_m = secant_search_roots_media(
                    roots_base,
                    h_m,
                    B1_m,
                    self.omega,
                    bot_bc_type,
                    bot_c_p,
                    bot_rho,
                    top_bc_type,
                    media_start_m,
                    media_end_m,
                    media_rho_m,
                    bot_c_s,
                )
                seqs.append(roots_m)
                
            final_roots = neville_extrapolation(seqs, mults)
            mode_mesh_multiplier = max(int(m) for m in mults)
            if mode_mesh_multiplier > 1:
                N = self.config.solver.n_mesh_points * mode_mesh_multiplier
                h = depth_max / N
                z_arr = np.linspace(0, depth_max, N + 1)
                c_arr = np.zeros(N + 1)
                for i, z in enumerate(z_arr):
                    c_arr[i], _, _ = self.ssp.evaluate(z)
                density_arr = np.interp(z_arr, self.ssp.z_arr, self.ssp.density_arr)
                attenuation_arr = np.interp(z_arr, self.ssp.z_arr, self.ssp.attenuation_arr)
                medium_rho = float(density_arr[0])
                media_start, media_end, media_rho = self._acoustic_media(z_arr)
                B1 = build_acoustic_matrix(h, self.omega, c_arr)
        else:
            final_roots = roots_base
            
        # 3. Modes computation and modal attenuation perturbation
        bot_attenuation_p = self.config.environment.bottom.attenuation_p or 0.0
        bot_attenuation_s = self.config.environment.bottom.attenuation_s or 0.0
        modes, complex_roots = compute_modes(
            final_roots, h, B1, self.omega, bot_bc_type, bot_c_p, bot_rho, top_bc_type, density_arr,
            bot_attenuation_p=bot_attenuation_p,
            bot_c_s=bot_c_s,
            bot_attenuation_s=bot_attenuation_s,
            attenuation_arr=attenuation_arr,
            medium_rho=medium_rho,
            media_start=media_start,
            media_end=media_end,
            media_rho=media_rho,
        )

        if (
            self._needs_complex_root_refinement(attenuation_arr, bot_attenuation_p, bot_attenuation_s, bot_bc_type)
            or top_reflection_table is not None
        ):
            complex_roots = self._refine_complex_roots(
                complex_roots,
                h,
                c_arr,
                attenuation_arr,
                bot_bc_type,
                bot_c_p,
                bot_rho,
                top_bc_type,
                media_start,
                media_end,
                media_rho,
                bot_c_s,
                bot_attenuation_p,
                bot_attenuation_s,
                top_reflection_table,
                bot_reflection_table,
            )
            modes = self._compute_complex_modes(
                complex_roots,
                h,
                c_arr,
                attenuation_arr,
                bot_bc_type,
                bot_c_p,
                bot_rho,
                top_bc_type,
                media_start,
                media_end,
                media_rho,
                bot_c_s,
                bot_attenuation_p,
                bot_attenuation_s,
                top_reflection_table,
                bot_reflection_table,
            )
        
        # 4. Field synthesis
        ranges = self.config.geometry.receivers.ranges
        r_arr = _field_axis_values(ranges, start_floor=1.0)

        depths = self.config.geometry.receivers.depths
        z_r_arr = _field_axis_values(depths)
            
        field_max_modes = max(0, int(getattr(self.config.solver, "max_modes", 9999)))
        field_mode_count = min(field_max_modes, modes.shape[0])
        field_modes = modes[:field_mode_count, :]
        field_roots = complex_roots[:field_mode_count]
        tl_by_source = []
        for z_s in self.config.geometry.source.depths:
            rho_s = self.ssp.evaluate_density(z_s)
            tl_by_source.append(
                compute_field(
                    field_modes,
                    field_roots,
                    z_arr,
                    r_arr,
                    float(z_s),
                    z_r_arr,
                    rho_s=rho_s,
                    field_option=getattr(self.config.solver, "field_option", "RA"),
                    receiver_range_offsets_m=getattr(self.config.solver, "field_receiver_range_offsets_m", None),
                )
            )
        TL = tl_by_source[0] if len(tl_by_source) == 1 else np.stack(tl_by_source, axis=0)
        
        return {
            'tl_grid': TL,
            'r_bins': r_arr,
            'z_bins': z_r_arr,
            'z_bins_mod': z_arr,
            'modes': modes,
            'roots': final_roots,
            'mode_metadata': self._mode_metadata(
                z_arr,
                media_start,
                media_end,
                media_rho,
                bot_bc_type,
                bot_c_p,
                bot_c_s,
                bot_rho,
                bot_attenuation_p,
                bot_attenuation_s,
            ),
            **self._mode_diagnostics(complex_roots),
        }
