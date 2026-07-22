import numpy as np
from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop3d.tracer import trace_single_ray_3d
from pyacoustics.environment import (
    SSPInterpolator,
    evaluate_hexahedral_ssp_3d,
    evaluate_linear_ssp,
    evaluate_n2linear_ssp,
    evaluate_spline_ssp,
)

class PyBellhop3D:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.z_surface = config.environment.surface.depth or 0.0
        self.z_bottom = config.environment.bottom.depth or 5000.0
        source_x = np.asarray(config.geometry.source.x or [0.0], dtype=np.float64)
        source_y = np.asarray(config.geometry.source.y or [0.0], dtype=np.float64)
        source_offset = float(np.max(np.hypot(source_x, source_y))) if source_x.size and source_y.size else 0.0
        receiver_range_max = max(config.geometry.receivers.ranges)
        self.r_max = source_offset + 1.1 * receiver_range_max

        if not hasattr(self.config.solver, "azimuth_angles"):
            raise ValueError("Solver config must be Bellhop3DConfig with azimuth_angles specified.")

        self.h = self.config.solver.step_size or 10.0
        self.box_x = self.config.solver.box_x if self.config.solver.box_x is not None else self.r_max
        self.box_y = self.config.solver.box_y if self.config.solver.box_y is not None else self.r_max
        self.box_z = self.config.solver.box_z if self.config.solver.box_z is not None else 1.0e30

        # 2D Beam fan array: elevation x azimuth
        self.elevations = np.linspace(
            config.solver.elevation_angles[0],
            config.solver.elevation_angles[1],
            max(config.solver.num_elevation_beams, 1)
        )
        self.azimuths = np.linspace(
            config.solver.azimuth_angles[0],
            config.solver.azimuth_angles[1],
            max(config.solver.num_azimuth_beams, 1)
        )

    def _prepare_ssp_arrays(self):
        """Prepare 1D SSP arrays for Numba-compiled functions."""
        ssp = self.config.environment.ssp
        ssp_type_map = {"c-linear": 0, "spline": 1, "n2-linear": 2}
        ssp_type_int = ssp_type_map.get(ssp.type, 0)

        interp = SSPInterpolator(ssp)
        c_coeffs = interp.c_coeffs if interp.c_coeffs is not None else np.zeros((4, 1), dtype=np.float64)
        return ssp_type_int, interp.z_arr, interp.c_arr, c_coeffs

    def run_ray_tracing(self, include_phi: bool = False):
        """Executes 3D ray tracing loop across the azimuth-elevation mesh."""
        ssp = self.config.environment.ssp

        if ssp.type == "hexahedral-3d":
            if ssp.x is None or ssp.y is None or ssp.z is None or ssp.c is None:
                raise ValueError("hexahedral-3d SSP requires x, y, z, and c grids")
            ssp_type_int = 3
            z_arr = np.zeros(2, dtype=np.float64)
            c_arr = np.zeros(2, dtype=np.float64)
            c_coeffs = np.zeros((4, 1), dtype=np.float64)
            ssp3d_x = np.array(ssp.x, dtype=np.float64)
            ssp3d_y = np.array(ssp.y, dtype=np.float64)
            ssp3d_z = np.array(ssp.z, dtype=np.float64)
            ssp3d_c = np.array(ssp.c, dtype=np.float64)
        else:
            ssp_type_int, z_arr, c_arr, c_coeffs = self._prepare_ssp_arrays()
            ssp3d_x = np.zeros(2, dtype=np.float64)
            ssp3d_y = np.zeros(2, dtype=np.float64)
            ssp3d_z = np.zeros(2, dtype=np.float64)
            ssp3d_c = np.zeros((2, 2, 2), dtype=np.float64)

        bot = self.config.environment.bottom
        bot_rho = bot.density if bot.density is not None else 1.0
        bot_c_s = bot.c_s if bot.c_s is not None else 0.0
        bot_alpha = bot.attenuation_p if bot.attenuation_p is not None else 0.0
        bot_alpha_s = bot.attenuation_s if bot.attenuation_s is not None else 0.0
        if bot.type in ("vacuum", "rigid"):
            bot_c = 0.0 if bot.type == "vacuum" else -1.0
            bot_c_s = 0.0
            bot_alpha = 0.0
            bot_alpha_s = 0.0
        else:
            bot_c = bot.c_p if bot.c_p is not None else 1500.0
        if bot.bathymetry and bot.bathymetry_geoacoustics:
            bty_media_r = np.array([point[0] for point in bot.bathymetry], dtype=np.float64)
            bty_geo = np.array(bot.bathymetry_geoacoustics, dtype=np.float64)
            bty_geo_cp = bty_geo[:, 0]
            bty_geo_cs = bty_geo[:, 1]
            bty_geo_rho = bty_geo[:, 2]
            bty_geo_alpha = bty_geo[:, 3]
            bty_geo_alpha_s = bty_geo[:, 4]
        else:
            bty_media_r = np.zeros(0, dtype=np.float64)
            bty_geo_cp = np.zeros(0, dtype=np.float64)
            bty_geo_cs = np.zeros(0, dtype=np.float64)
            bty_geo_rho = np.zeros(0, dtype=np.float64)
            bty_geo_alpha = np.zeros(0, dtype=np.float64)
            bty_geo_alpha_s = np.zeros(0, dtype=np.float64)

        surface = self.config.environment.surface
        surf_rho = surface.density if surface.density is not None else 1.0
        surf_c_s = surface.c_s if surface.c_s is not None else 0.0
        surf_alpha = surface.attenuation_p if surface.attenuation_p is not None else 0.0
        surf_alpha_s = surface.attenuation_s if surface.attenuation_s is not None else 0.0
        if surface.type == "rigid":
            surf_c = -1.0
        elif surface.type == "acousto-elastic":
            surf_c = surface.c_p if surface.c_p is not None else 1600.0
        else:
            surf_c = 0.0
        if surface.altimetry and surface.altimetry_geoacoustics:
            ati_media_r = np.array([point[0] for point in surface.altimetry], dtype=np.float64)
            ati_geo = np.array(surface.altimetry_geoacoustics, dtype=np.float64)
            ati_geo_cp = ati_geo[:, 0]
            ati_geo_cs = ati_geo[:, 1]
            ati_geo_rho = ati_geo[:, 2]
            ati_geo_alpha = ati_geo[:, 3]
            ati_geo_alpha_s = ati_geo[:, 4]
        else:
            ati_media_r = np.zeros(0, dtype=np.float64)
            ati_geo_cp = np.zeros(0, dtype=np.float64)
            ati_geo_cs = np.zeros(0, dtype=np.float64)
            ati_geo_rho = np.zeros(0, dtype=np.float64)
            ati_geo_alpha = np.zeros(0, dtype=np.float64)
            ati_geo_alpha_s = np.zeros(0, dtype=np.float64)

        surface_refl_angles, surface_refl_amp, surface_refl_phase = self._reflection_arrays(
            self.config.environment.surface.reflection_table
        )
        bottom_refl_angles, bottom_refl_amp, bottom_refl_phase = self._reflection_arrays(
            self.config.environment.bottom.reflection_table
        )

        if bot.bathymetry_3d:
            bty3d_x = np.array(bot.bathymetry_3d.x, dtype=np.float64)
            bty3d_y = np.array(bot.bathymetry_3d.y, dtype=np.float64)
            bty3d_z = np.array(bot.bathymetry_3d.z, dtype=np.float64)
        else:
            bty3d_x = np.zeros(0, dtype=np.float64)
            bty3d_y = np.zeros(0, dtype=np.float64)
            bty3d_z = np.zeros((0,0), dtype=np.float64)

        rays_3d = []
        source_x = self.config.geometry.source.x or [0.0]
        source_y = self.config.geometry.source.y or [0.0]
        for z_src in self.config.geometry.source.depths:
            for x_src in source_x:
                for y_src in source_y:
                    for azim in self.azimuths:
                        for elev in self.elevations:
                            (
                                x_p, y_p, z_p,
                                q11_p, q12_p, q21_p, q22_p,
                                amp_p, phase_p, tau_p, kmah_p, c_p, phi_p,
                                top_bounce_p, bottom_bounce_p,
                                n_pts, n_top_bounces, n_bottom_bounces,
                            ) = trace_single_ray_3d(
                                x_0=float(x_src), y_0=float(y_src), z_0=z_src,
                                alpha_elev_deg=elev, alpha_azim_deg=azim,
                                h=self.h,
                                z_surface=self.z_surface,
                                bty_x=bty3d_x, bty_y=bty3d_y, bty_z=bty3d_z, z_bottom_flat=self.z_bottom,
                                surf_rho=surf_rho, surf_c=surf_c, surf_c_s=surf_c_s,
                                surf_alpha=surf_alpha, surf_alpha_s=surf_alpha_s,
                                bot_rho=bot_rho, bot_c=bot_c, bot_c_s=bot_c_s,
                                bot_alpha=bot_alpha, bot_alpha_s=bot_alpha_s,
                                ati_media_r=ati_media_r,
                                ati_geo_cp=ati_geo_cp,
                                ati_geo_cs=ati_geo_cs,
                                ati_geo_rho=ati_geo_rho,
                                ati_geo_alpha=ati_geo_alpha,
                                ati_geo_alpha_s=ati_geo_alpha_s,
                                bty_media_r=bty_media_r,
                                bty_geo_cp=bty_geo_cp,
                                bty_geo_cs=bty_geo_cs,
                                bty_geo_rho=bty_geo_rho,
                                bty_geo_alpha=bty_geo_alpha,
                                bty_geo_alpha_s=bty_geo_alpha_s,
                                surface_refl_angles=surface_refl_angles,
                                surface_refl_amp=surface_refl_amp,
                                surface_refl_phase=surface_refl_phase,
                                bottom_refl_angles=bottom_refl_angles,
                                bottom_refl_amp=bottom_refl_amp,
                                bottom_refl_phase=bottom_refl_phase,
                                r_max=self.r_max,
                                box_x=self.box_x, box_y=self.box_y, box_z=self.box_z,
                                ssp_type=ssp_type_int,
                                z_arr=z_arr, c_arr=c_arr, c_coeffs=c_coeffs,
                                ssp3d_x=ssp3d_x, ssp3d_y=ssp3d_y, ssp3d_z=ssp3d_z, ssp3d_c=ssp3d_c,
                                max_steps=100000
                            )
                            if include_phi:
                                rays_3d.append((
                                    x_p[:n_pts], y_p[:n_pts], z_p[:n_pts],
                                    q11_p[:n_pts], q12_p[:n_pts], q21_p[:n_pts], q22_p[:n_pts],
                                    amp_p[:n_pts], phase_p[:n_pts], tau_p[:n_pts], kmah_p[:n_pts], c_p[:n_pts], phi_p[:n_pts],
                                    top_bounce_p[:n_pts], bottom_bounce_p[:n_pts],
                                    int(n_top_bounces), int(n_bottom_bounces), elev
                                ))
                            else:
                                rays_3d.append((
                                    x_p[:n_pts], y_p[:n_pts], z_p[:n_pts],
                                    q11_p[:n_pts], q12_p[:n_pts], q21_p[:n_pts], q22_p[:n_pts],
                                    amp_p[:n_pts], tau_p[:n_pts], kmah_p[:n_pts], c_p[:n_pts],
                                    int(n_top_bounces), int(n_bottom_bounces), elev
                                ))
        return rays_3d

    @staticmethod
    def _reflection_arrays(table):
        if not table:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty
        arr = np.array(table, dtype=np.float64)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        return arr[:, 0], arr[:, 1], arr[:, 2]

    def run_beam_tracing(self):
        """Runs the 3D rays and packages them into Numba typed lists for fast Voxel Grid processing."""
        rays_3d = self.run_ray_tracing(include_phi=True)

        from numba.typed import List
        rx = List()
        ry = List()
        rz = List()
        rq11 = List()
        rq12 = List()
        rq21 = List()
        rq22 = List()
        ramp = List()
        rtau = List()
        rkmah = List()
        rcpath = List()
        rphi = List()
        relev = List()

        for (
            x, y, z, q11, q12, q21, q22, amp, _phase, tau, kmah, cpath, phi,
            _top_path, _bottom_path, _top_bounces, _bottom_bounces, elev,
        ) in rays_3d:
            rx.append(x)
            ry.append(y)
            rz.append(z)
            rq11.append(q11)
            rq12.append(q12)
            rq21.append(q21)
            rq22.append(q22)
            ramp.append(amp)
            rtau.append(tau)
            rkmah.append(kmah)
            rcpath.append(cpath)
            rphi.append(phi)
            relev.append(elev)

        ssp = self.config.environment.ssp
        if ssp.type == "hexahedral-3d":
            if ssp.x is None or ssp.y is None or ssp.z is None or ssp.c is None:
                raise ValueError("hexahedral-3d SSP requires x, y, z, and c grids")
            source_x0 = (self.config.geometry.source.x or [0.0])[0]
            source_y0 = (self.config.geometry.source.y or [0.0])[0]
            c_src, _, _, _, _ = evaluate_hexahedral_ssp_3d(
                float(source_x0), float(source_y0), self.config.geometry.source.depths[0],
                np.array(ssp.x, dtype=np.float64),
                np.array(ssp.y, dtype=np.float64),
                np.array(ssp.z, dtype=np.float64),
                np.array(ssp.c, dtype=np.float64),
            )
        else:
            z_src = self.config.geometry.source.depths[0]
            ssp_type_int, z_arr, c_arr, c_coeffs = self._prepare_ssp_arrays()
            if ssp_type_int == 1:
                c_src, _, _ = evaluate_spline_ssp(z_src, z_arr, c_coeffs)
            elif ssp_type_int == 2:
                c_src, _, _ = evaluate_n2linear_ssp(z_src, z_arr, c_arr)
            else:
                c_src, _, _ = evaluate_linear_ssp(z_src, z_arr, c_arr)

        num_azim = max(self.config.solver.num_azimuth_beams, 1)
        num_elev = max(self.config.solver.num_elevation_beams, 1)

        # Correct for N-1 intervals in linspace to match physical angular density
        div_azim = (num_azim - 1) if num_azim > 1 else 1
        div_elev = (num_elev - 1) if num_elev > 1 else 1

        d_azim = np.radians(abs(self.config.solver.azimuth_angles[1] - self.config.solver.azimuth_angles[0])) / div_azim
        d_elev = np.radians(abs(self.config.solver.elevation_angles[1] - self.config.solver.elevation_angles[0])) / div_elev

        if d_azim == 0: d_azim = 0.001 # Small default for single ray
        if d_elev == 0: d_elev = 0.001

        return {
            "rays_x": rx,
            "rays_y": ry,
            "rays_z": rz,
            "rays_q11": rq11,
            "rays_q12": rq12,
            "rays_q21": rq21,
            "rays_q22": rq22,
            "rays_amp": ramp,
            "rays_tau": rtau,
            "rays_kmah": rkmah,
            "rays_cpath": rcpath,
            "rays_phi": rphi,
            "rays_elev": relev,
            "freq": self.config.frequency,
            "c_src": c_src,
            "d_azim": d_azim,
            "d_elev": d_elev,
        }
