import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Union, Optional, Literal
from pathlib import Path
from pyacoustics.config import ConfigLoader
from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop.core import PyBellhop
from pyacoustics.solvers.external import ExternalSolver, BellhopExternal, KrakenExternal
from pyacoustics.plot import plot_rays


def _mask_bellhop3d_pressure_below_bathymetry(
    pressure_by_source: np.ndarray,
    *,
    source_x: np.ndarray,
    source_y: np.ndarray,
    theta_grid_deg: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    bathymetry_3d,
) -> None:
    """Zero Bellhop3D receiver-grid pressure at receivers below the local bottom."""

    if bathymetry_3d is None:
        return

    from pyacoustics.solvers.bellhop3d.boundary import get_bathymetry_3d_z

    bty_x = np.asarray(bathymetry_3d.x, dtype=np.float64)
    bty_y = np.asarray(bathymetry_3d.y, dtype=np.float64)
    bty_z = np.asarray(bathymetry_3d.z, dtype=np.float64)
    if bty_x.size < 2 or bty_y.size < 2 or bty_z.size == 0:
        return

    for isx, source_x_value in enumerate(source_x):
        for isy, source_y_value in enumerate(source_y):
            for itheta, theta_deg in enumerate(theta_grid_deg):
                theta_rad = np.radians(theta_deg)
                cos_theta = np.cos(theta_rad)
                sin_theta = np.sin(theta_rad)
                for ir, receiver_range in enumerate(r_grid):
                    receiver_x = float(source_x_value + receiver_range * cos_theta)
                    receiver_y = float(source_y_value + receiver_range * sin_theta)
                    bottom_z, _, _ = get_bathymetry_3d_z(receiver_x, receiver_y, bty_x, bty_y, bty_z)
                    below_bottom = z_grid > float(bottom_z) + 1.0e-9
                    if np.any(below_bottom):
                        pressure_by_source[isx, isy, itheta, :, below_bottom, ir] = 0.0


class Simulation:
    """Top level interface for the pyacoustics engine."""

    def __init__(self, config: Union[str, Path, dict, SimulationConfig],
                 external_bin_path: Optional[str] = None,
                 mode: Literal["native", "legacy"] = "native"):
        self.mode = mode
        if isinstance(config, (str, Path)):
            from pyacoustics.config import ConfigLoader
            self.config = ConfigLoader.load_yaml(str(config))
        elif isinstance(config, dict):
            from pyacoustics.config import ConfigLoader
            self.config = ConfigLoader.from_dict(config)

        else:
            self.config = config

        from pyacoustics.solvers.external import ExternalSolver
        self.external_solver = ExternalSolver(self.config, bin_path=external_bin_path)
        self.ray_paths = None
        self._tl_cache = None
        self._coherent_tl_cache = None

    def run(self, mode: Optional[Literal["native", "legacy"]] = None, save_ray_path: str = None):
        """
        Runs the simulation based on the loaded configuration.
        """
        run_mode = mode if mode is not None else self.mode
        if run_mode == "legacy":
            if self.config.solver.type == "bellhop":
                solver = BellhopExternal(self.config, bin_path=self.external_solver.bin_path)
                return solver.run()
            elif self.config.solver.type == "normal_modes":
                solver = KrakenExternal(self.config, bin_path=self.external_solver.bin_path)
                res = solver.run()
                self._tl_cache = (res['tl_grid'], res['r_bins'], res['z_bins'])
                self._coherent_tl_cache = self._tl_cache
                return res
            else:
                raise NotImplementedError(f"Legacy solver for {self.config.solver.type} not yet implemented.")

        if self.config.solver.type == "bellhop":
            solver = PyBellhop(self.config)
            self.ray_paths = solver.run_ray_tracing()
            if save_ray_path is not None:
                self._write_2d_ray_file(save_ray_path, self.ray_paths)
            return self.ray_paths
        elif self.config.solver.type == "bellhop3d":
            from pyacoustics.solvers.bellhop3d.core import PyBellhop3D
            solver = PyBellhop3D(self.config)
            self.ray_paths_3d = solver.run_ray_tracing()
            if save_ray_path is not None:
                self._write_3d_ray_file(save_ray_path, self.ray_paths_3d)
            return self.ray_paths_3d
        elif self.config.solver.type == "normal_modes":
            from pyacoustics.solvers.kraken import PyKraken
            solver = PyKraken(self.config)
            res = solver.run_normal_modes()
            self._coherent_tl_cache = (res['tl_grid'], res['r_bins'], res['z_bins'])
            # Since normal modes directly computes TL, we can alias it to the standard tl cache
            self._tl_cache = self._coherent_tl_cache
            return res
        else:
            raise NotImplementedError(f"Solver type {self.config.solver.type} not implemented.")

    def _write_2d_ray_file(self, save_ray_path: str, ray_paths) -> None:
        from pyacoustics.solvers.external_io import ATRayRecord, write_at_ray_records

        beam = PyBellhop(self.config).run_beam_tracing()
        angles = np.linspace(
            self.config.solver.angles[0],
            self.config.solver.angles[1],
            max(self.config.solver.num_beams, 1),
        )
        records = []
        for idx, (ranges, depths, amplitudes) in enumerate(ray_paths):
            launch_angle = angles[idx % len(angles)] if len(angles) else 0.0
            beam_idx = idx % beam["num_beams"]
            npts = max(int(beam["ray_npts"][beam_idx]), 1)
            records.append(
                ATRayRecord(
                    launch_angle_deg=float(launch_angle),
                    ranges_m=np.asarray(ranges, dtype=np.float64),
                    depths_m=np.asarray(depths, dtype=np.float64),
                    amplitudes=np.asarray(amplitudes, dtype=np.complex128),
                    num_top_bounces=int(beam["ray_top_bnc"][beam_idx, npts - 1]),
                    num_bottom_bounces=int(beam["ray_bot_bnc"][beam_idx, npts - 1]),
                )
            )
        write_at_ray_records(
            save_ray_path,
            records,
            title=self.config.project,
            frequency=self.config.frequency,
            top_depth=self.config.environment.surface.depth or 0.0,
            bottom_depth=self.config.environment.bottom.depth or 0.0,
        )

    def _write_3d_ray_file(self, save_ray_path: str, ray_paths_3d) -> None:
        from pyacoustics.solvers.external_io import ATRayRecord, write_at_ray_records

        azimuths = np.linspace(
            self.config.solver.azimuth_angles[0],
            self.config.solver.azimuth_angles[1],
            max(self.config.solver.num_azimuth_beams, 1),
        )
        num_elev = max(self.config.solver.num_elevation_beams, 1)
        records = []
        for idx, ray in enumerate(ray_paths_3d):
            x_path, y_path, z_path = ray[0], ray[1], ray[2]
            launch_angle = float(ray[-1])
            azimuth_idx = (idx // num_elev) % len(azimuths)
            records.append(
                ATRayRecord(
                    launch_angle_deg=launch_angle,
                    launch_azimuth_deg=float(azimuths[azimuth_idx]),
                    coordinate_system="xyz",
                    ranges_m=np.asarray(x_path, dtype=np.float64),
                    y_m=np.asarray(y_path, dtype=np.float64),
                    depths_m=np.asarray(z_path, dtype=np.float64),
                    amplitudes=np.ones(len(x_path), dtype=np.complex128),
                    num_top_bounces=int(ray[-3]),
                    num_bottom_bounces=int(ray[-2]),
                )
            )
        write_at_ray_records(
            save_ray_path,
            records,
            title=self.config.project,
            frequency=self.config.frequency,
            top_depth=self.config.environment.surface.depth or 0.0,
            bottom_depth=self.config.environment.bottom.depth or 0.0,
        )

    def plot_rays(self, save_path: str = None):
        """Generates and optionally saves the ray plot."""
        if not hasattr(self, 'ray_paths'):
            raise RuntimeError("Simulation has not been run yet. Call .run() first.")
        from pyacoustics.plot import plot_rays
        return plot_rays(self.config, self.ray_paths, save_path)

    def plot_tl(self, save_path: str = None, **kwargs):
        """Generates and optionally saves the Transmission Loss heatmap."""
        if self.config.solver.type == "normal_modes":
            # For normal modes, _tl_cache is already populated during run()
            if not hasattr(self, '_tl_cache'):
                raise RuntimeError("Simulation has not been run yet. Call .run() first.")
            import matplotlib.pyplot as plt
            tl_grid, r_bins, z_bins = self._tl_cache
            fig, ax = plt.subplots(figsize=(12, 6))
            from pyacoustics.plot import get_auto_clim
            auto_vmin, auto_vmax = get_auto_clim(tl_grid)
            vmin = kwargs.get('vmin', auto_vmin)
            vmax = kwargs.get('vmax', auto_vmax)
            im = ax.imshow(
                tl_grid, extent=[r_bins[0] / 1000.0, r_bins[-1] / 1000.0, z_bins[-1], z_bins[0]],
                aspect='auto', cmap='jet_r',
                vmin=vmin, vmax=vmax
            )
            plt.colorbar(im, label='Transmission Loss (dB)')
            ax.set_xlabel('Range (km)')
            ax.set_ylabel('Depth (m)')
            ax.set_title(f'Transmission Loss (Normal Modes) - {self.config.project}')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig
        else:
            if not hasattr(self, 'ray_paths'):
                raise RuntimeError("Simulation has not been run yet. Call .run() first.")
            from pyacoustics.plot import plot_tl
            return plot_tl(self.config, self.ray_paths, save_path, **kwargs)

    def get_tl(self, range_m: float, depth_m: float) -> float:
        """
        Returns the approximate Transmission Loss (dB) at a specific range and depth.
        Uses nearest-neighbor lookup on the computed TL grid.
        """
        if not hasattr(self, '_tl_cache'):
            if not hasattr(self, 'ray_paths'):
                raise RuntimeError("Simulation has not been run yet. Call .run() first.")
            from pyacoustics.plot import compute_tl_field
            self._tl_cache = compute_tl_field(self.config, self.ray_paths)

        tl_grid, r_bins, z_bins = self._tl_cache

        # Find nearest indices
        import numpy as np
        r_idx = np.abs(r_bins - range_m).argmin()
        z_idx = np.abs(z_bins - depth_m).argmin()

        nr = len(r_bins)
        nz = len(z_bins)
        if tl_grid.shape == (nz, nr):
            # (depth, range) layout, e.g. normal modes
            z_idx = min(z_idx, nz - 1)
            r_idx = min(r_idx, nr - 1)
            return float(tl_grid[z_idx, r_idx])
        else:
            # (range, depth) layout, e.g. ray/beam solvers
            r_idx = min(r_idx, nr - 1)
            z_idx = min(z_idx, nz - 1)
            return float(tl_grid[r_idx, z_idx])

    def compute_coherent_tl(
        self,
        num_r: int = 200,
        num_z: int = 100,
        save_path: str = None,
        save_shd_path: str = None,
        r_grid: np.ndarray = None,
        z_grid: np.ndarray = None,
    ):
        """
        Computes the coherent Transmission Loss field using Gaussian Beam Summation.
        """
        r_max = max(self.config.geometry.receivers.ranges)
        z_max = self.config.environment.bottom.depth
        if r_grid is None:
            r_grid = np.linspace(100.0, r_max, num_r)
        if z_grid is None:
            z_grid = np.linspace(0.0, z_max, num_z)

        if self.mode == "legacy":
            if self.config.solver.type != "bellhop":
                raise NotImplementedError("Legacy coherent TL only implemented for Bellhop.")

            import tempfile
            import subprocess
            from pyacoustics.solvers.external_io import generate_at_bellhop_env, read_at_shd

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                env_file = tmp_path / "legacy_run.env"

                orig_rr = self.config.geometry.receivers.ranges
                orig_rd = self.config.geometry.receivers.depths

                try:
                    self.config.geometry.receivers.ranges = r_grid.tolist()
                    self.config.geometry.receivers.depths = z_grid.tolist()

                    generate_at_bellhop_env(self.config, env_file, run_type="C")

                    exe_path = self.external_solver.bin_path / "bellhop"
                    if not exe_path.exists():
                        exe_path = self.external_solver.bin_path / "bellhop.exe"

                    subprocess.run([str(exe_path), "legacy_run"], cwd=str(tmp_path), capture_output=True, check=True)

                    shd_file = tmp_path / "legacy_run.shd"
                    if not shd_file.exists():
                        raise RuntimeError("Legacy bellhop failed to produce .shd file.")

                    P, rz_shd, rr_shd = read_at_shd(shd_file)
                    TL = -20 * np.log10(np.abs(P) + 1e-20)
                    TL = TL.T # To match (r, z)

                    self._coherent_tl_cache = (TL, r_grid, z_grid)
                    if save_shd_path is not None:
                        self._write_tl_shd(save_shd_path, TL, r_grid, z_grid)
                    return TL
                finally:
                    self.config.geometry.receivers.ranges = orig_rr
                    self.config.geometry.receivers.depths = orig_rd


        r_max = max(self.config.geometry.receivers.ranges)
        z_max = self.config.environment.bottom.depth
        freq = self.config.frequency

        if r_grid is None:
            r_grid = np.linspace(100.0, r_max, num_r)  # Avoid r=0 singularity
        if z_grid is None:
            z_grid = np.linspace(0.0, z_max, num_z)

        from pyacoustics.solvers.bellhop.beam import (
            compute_hat_pressure_field_coherent,
            compute_pressure_field_coherent,
        )

        use_thorp = (self.config.environment.absorption_model == "thorp")
        line_source = getattr(self.config.solver, "source_type", "point") == "line"
        beam_type = getattr(self.config.solver, "beam_type", "default")
        pressure_kernel = (
            compute_hat_pressure_field_coherent
            if beam_type in {"default", "geometric_hat"}
            else compute_pressure_field_coherent
        )
        source_depths = np.asarray(self.config.geometry.source.depths, dtype=np.float32)
        pressure_by_source = []
        for source_depth in source_depths:
            source_config = deepcopy(self.config)
            source_config.geometry.source.depths = [float(source_depth)]
            solver = PyBellhop(source_config)
            beam_data = solver.run_beam_tracing()

            angles = source_config.solver.angles
            d_alpha = np.radians(angles[1] - angles[0]) / max(source_config.solver.num_beams - 1, 1)
            pressure_by_source.append(
                pressure_kernel(
                    r_grid, z_grid,
                    beam_data['ray_r'], beam_data['ray_z'],
                    beam_data['ray_c'], beam_data['ray_tau'],
                    beam_data['ray_q'], beam_data['ray_amp'],
                    beam_data['ray_npts'],
                    beam_data['ray_alphas'],
                    freq, beam_data['c_src'],
                    d_alpha,
                    use_thorp=use_thorp,
                    line_source=line_source,
                )
            )

        pressure_stack = np.stack(pressure_by_source, axis=0)
        pressure_abs = np.abs(pressure_stack)
        tl_stack = -20.0 * np.log10(np.maximum(pressure_abs, 1.0e-15))
        tl_stack[pressure_abs < 1.0e-15] = 120.0
        TL = tl_stack[0] if source_depths.size == 1 else tl_stack
        pressure_grid = pressure_stack[0] if source_depths.size == 1 else pressure_stack

        self._coherent_tl_cache = (TL, r_grid, z_grid)
        self._coherent_pressure_cache = (pressure_grid, r_grid, z_grid)
        if save_shd_path is not None:
            from pyacoustics.solvers.external_io import write_at_shd

            write_at_shd(
                save_shd_path,
                pressure_stack.transpose(0, 2, 1)[np.newaxis, :, :, :],
                z_grid,
                r_grid,
                title=self.config.project,
                frequency=self.config.frequency,
                source_depths_m=source_depths,
            )

        # Plot if requested
        if save_path:
            fig, ax = plt.subplots(figsize=(12, 6))
            from pyacoustics.plot import get_auto_clim
            plot_tl = TL if TL.ndim == 2 else TL[0]
            vmin, vmax = get_auto_clim(plot_tl)
            im = ax.imshow(
                plot_tl.T, extent=[r_grid[0] / 1000.0, r_grid[-1] / 1000.0, z_max, 0],
                aspect='auto', cmap='jet_r',
                vmin=vmin, vmax=vmax
            )
            plt.colorbar(im, label='Transmission Loss (dB)')
            ax.set_xlabel('Range (km)')
            ax.set_ylabel('Depth (m)')
            ax.set_title(f'Coherent TL (Gaussian Beam) - {self.config.project}')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        return TL

    def _write_tl_shd(self, save_shd_path: str, TL: np.ndarray, r_grid: np.ndarray, z_grid: np.ndarray) -> None:
        """Write a 2D TL grid as an AT-compatible ``.shd`` pressure-magnitude file."""
        from pyacoustics.solvers.external_io import write_at_shd

        source_depths = np.asarray(self.config.geometry.source.depths, dtype=np.float32)
        tl_array = np.asarray(TL, dtype=np.float64)
        if tl_array.ndim == 2:
            if source_depths.size != 1:
                raise ValueError("2D TL output requires one TL grid per source depth")
            pressure = np.power(10.0, -tl_array.T / 20.0).astype(np.complex128)
        elif tl_array.ndim == 3:
            if tl_array.shape[0] != source_depths.size:
                raise ValueError("TL source-depth axis must match configured source depths")
            pressure = np.power(10.0, -tl_array.transpose(0, 2, 1) / 20.0).astype(np.complex128)
            pressure = pressure[np.newaxis, :, :, :]
        else:
            raise ValueError("TL must be shaped (range, depth) or (source_depth, range, depth)")
        write_at_shd(
            save_shd_path,
            pressure,
            z_grid,
            r_grid,
            title=self.config.project,
            frequency=self.config.frequency,
            source_depths_m=source_depths,
        )

    def get_coherent_tl(self, range_m: float, depth_m: float) -> float:
        """
        Returns the coherent Transmission Loss (dB) at a specific range and depth.
        Must call compute_coherent_tl() first.
        """
        if not hasattr(self, '_coherent_tl_cache'):
            raise RuntimeError("Coherent TL has not been computed. Call .compute_coherent_tl() first.")

        import numpy as np
        tl_grid, r_grid, z_grid = self._coherent_tl_cache

        r_idx = np.abs(r_grid - range_m).argmin()
        z_idx = np.abs(z_grid - depth_m).argmin()

        r_idx = min(r_idx, tl_grid.shape[0] - 1)
        z_idx = min(z_idx, tl_grid.shape[1] - 1)

        return float(tl_grid[r_idx, z_idx])

    def compute_incoherent_tl(
        self,
        num_r: int = 200,
        num_z: int = 100,
        r_grid: np.ndarray = None,
        z_grid: np.ndarray = None,
    ):
        """Computes native Bellhop incoherent TL (AT RunType ``I``)."""
        return self._compute_intensity_tl(
            num_r=num_r,
            num_z=num_z,
            r_grid=r_grid,
            z_grid=z_grid,
            cache_name="_incoherent_tl_cache",
            semicoherent=False,
        )

    def compute_semicoherent_tl(
        self,
        num_r: int = 200,
        num_z: int = 100,
        r_grid: np.ndarray = None,
        z_grid: np.ndarray = None,
    ):
        """Computes native Bellhop semi-coherent TL (AT RunType ``S``)."""
        return self._compute_intensity_tl(
            num_r=num_r,
            num_z=num_z,
            r_grid=r_grid,
            z_grid=z_grid,
            cache_name="_semicoherent_tl_cache",
            semicoherent=True,
        )

    def _compute_intensity_tl(
        self,
        *,
        num_r: int,
        num_z: int,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
        cache_name: str,
        semicoherent: bool,
    ):
        if self.config.solver.type != "bellhop":
            raise NotImplementedError("Incoherent and semi-coherent TL are currently only supported for Bellhop.")

        r_max = max(self.config.geometry.receivers.ranges)
        z_max = self.config.environment.bottom.depth
        if r_grid is None:
            r_grid = np.linspace(100.0, r_max, num_r)
        if z_grid is None:
            z_grid = np.linspace(0.0, z_max, num_z)

        from pyacoustics.solvers.bellhop.beam import (
            compute_tl_field_hat_incoherent,
            compute_tl_field_hat_semicoherent,
            compute_tl_field_incoherent,
            compute_tl_field_semicoherent,
        )

        use_thorp = self.config.environment.absorption_model == "thorp"
        beam_type = getattr(self.config.solver, "beam_type", "default")
        line_source = getattr(self.config.solver, "source_type", "point") == "line"
        use_hat_intensity = beam_type in {"default", "geometric_hat"}
        source_depths = np.asarray(self.config.geometry.source.depths, dtype=np.float64)
        solver = PyBellhop(self.config)
        tl_by_source = []
        for source_depth in source_depths:
            beam_data = solver.run_beam_tracing(source_depth=float(source_depth))
            angles = self.config.solver.angles
            d_alpha = np.radians(angles[1] - angles[0]) / max(self.config.solver.num_beams - 1, 1)
            if semicoherent:
                semicoherent_kernel = (
                    compute_tl_field_hat_semicoherent if use_hat_intensity else compute_tl_field_semicoherent
                )
                tl_by_source.append(
                    semicoherent_kernel(
                        r_grid,
                        z_grid,
                        beam_data["ray_r"],
                        beam_data["ray_z"],
                        beam_data["ray_c"],
                        beam_data["ray_tau"],
                        beam_data["ray_q"],
                        beam_data["ray_amp"],
                        beam_data["ray_npts"],
                        beam_data["ray_alphas"],
                        self.config.frequency,
                        beam_data["c_src"],
                        d_alpha,
                        float(source_depth),
                        use_thorp=use_thorp,
                        line_source=line_source,
                    )
                )
            else:
                incoherent_kernel = compute_tl_field_hat_incoherent if use_hat_intensity else compute_tl_field_incoherent
                tl_by_source.append(
                    incoherent_kernel(
                        r_grid,
                        z_grid,
                        beam_data["ray_r"],
                        beam_data["ray_z"],
                        beam_data["ray_c"],
                        beam_data["ray_tau"],
                        beam_data["ray_q"],
                        beam_data["ray_amp"],
                        beam_data["ray_npts"],
                        beam_data["ray_alphas"],
                        self.config.frequency,
                        beam_data["c_src"],
                        d_alpha,
                        use_thorp=use_thorp,
                        line_source=line_source,
                    )
                )
        tl_stack = np.stack(tl_by_source, axis=0)
        TL = tl_stack[0] if source_depths.size == 1 else tl_stack
        setattr(self, cache_name, (TL, r_grid, z_grid))
        return TL

    def run_arrivals(self, range_m: float, depth_m: float):
        """
        Extracts multi-path arrival information at a specific receiver location.
        """
        if self.mode == "legacy":
            if self.config.solver.type != "bellhop":
                raise NotImplementedError("Legacy arrivals analysis is currently only supported for Bellhop.")

            import tempfile
            import subprocess
            from pyacoustics.solvers.external_io import generate_at_bellhop_env, read_at_arr

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                env_file = tmp_path / "legacy_run.env"

                # 1. Update config geometry to match the single receiver requested
                # (Bellhop .arr output depends on the .env receiver grid)
                orig_rd = self.config.geometry.receivers.depths
                orig_rr = self.config.geometry.receivers.ranges
                self.config.geometry.receivers.depths = [depth_m]
                self.config.geometry.receivers.ranges = [range_m]

                try:
                    generate_at_bellhop_env(self.config, env_file, run_type="A")

                    # 2. Run Bellhop
                    exe_path = self.external_solver.bin_path / "bellhop"
                    if not exe_path.exists():
                        exe_path = self.external_solver.bin_path / "bellhop.exe"

                    subprocess.run([str(exe_path), "legacy_run"], cwd=str(tmp_path), capture_output=True, check=True)

                    # 3. Read .arr
                    arr_file = tmp_path / "legacy_run.arr"
                    if not arr_file.exists():
                        raise RuntimeError("Legacy bellhop failed to produce .arr file.")

                    return read_at_arr(arr_file)
                finally:
                    # Restore original config
                    self.config.geometry.receivers.depths = orig_rd
                    self.config.geometry.receivers.ranges = orig_rr

        if self.config.solver.type != "bellhop":
            raise NotImplementedError("Arrivals analysis is currently only supported for Bellhop.")

        from pyacoustics.solvers.bellhop.arrivals import compute_arrivals
        return compute_arrivals(self.config, range_m, depth_m)

    def synthesize_arrivals(
        self,
        arrivals,
        source_waveform,
        *,
        sample_rate: float,
        output_mat_path: str = None,
        **kwargs,
    ):
        """Run delay-and-sum post-processing for an ``ArrivalData`` object."""
        from pyacoustics.postprocessing.delay_and_sum import delay_and_sum, save_matlab_timeseries

        result = delay_and_sum(arrivals, source_waveform, sample_rate=sample_rate, **kwargs)
        if output_mat_path is not None:
            save_matlab_timeseries(result, output_mat_path, plot_title=self.config.project)
        return result

    def synthesize_arrivals_file(
        self,
        arrivals_path: str,
        source_waveform,
        *,
        sample_rate: float,
        output_mat_path: str = None,
        **kwargs,
    ):
        """Run delay-and-sum post-processing for an AT ASCII ``.arr`` file."""
        from pyacoustics.postprocessing.delay_and_sum import synthesize_from_arrivals_file

        return synthesize_from_arrivals_file(
            arrivals_path,
            source_waveform,
            sample_rate=sample_rate,
            output_mat_path=output_mat_path,
            **kwargs,
        )

    def matched_filter_timeseries(self, result, replica_waveform, output_mat_path: str = None, **kwargs):
        """Apply matched filtering to a delay-and-sum result."""
        from pyacoustics.postprocessing.delay_and_sum import matched_filter_timeseries, save_matched_filter_matlab

        matched = matched_filter_timeseries(result, replica_waveform, **kwargs)
        if output_mat_path is not None:
            save_matched_filter_matlab(matched, output_mat_path, plot_title=self.config.project)
        return matched

    def run_eigenrays(
        self,
        range_m: float,
        depth_m: float,
        *,
        angle_samples: int = None,
        tolerance_m: float = 0.25,
        max_iterations: int = 40,
        save_path: str = None,
    ):
        """
        Find native 2D geometric eigenrays to a receiver location.

        If save_path is provided, writes an AT-compatible ASCII .ray file.
        """
        if self.config.solver.type != "bellhop":
            raise NotImplementedError("Eigenray search is currently only supported for Bellhop.")

        from pyacoustics.solvers.bellhop.eigenrays import compute_eigenrays, write_eigenrays_ray_ascii

        eigenrays = compute_eigenrays(
            self.config,
            range_m,
            depth_m,
            angle_samples=angle_samples,
            tolerance_m=tolerance_m,
            max_iterations=max_iterations,
        )
        if save_path is not None:
            write_eigenrays_ray_ascii(save_path, self.config, eigenrays)
        return eigenrays

    def compute_arrival_grid(
        self,
        save_path: str = None,
        max_arrivals: int = 200,
        beam_shape: Literal["hat", "gaussian"] = "hat",
        method: Literal["beam", "eigenray"] = "beam",
        eigenray_tolerance_m: float = 0.25,
        eigenray_angle_samples: int = None,
        receiver_bearings: np.ndarray = None,
        tolerance_m: float = None,
    ):
        """
        Computes native Bellhop/Bellhop3D arrivals on the configured receiver grid.

        If save_path is provided, writes an AT-compatible ASCII .arr file.
        """
        if self.config.solver.type == "bellhop3d":
            from pyacoustics.solvers.bellhop3d.arrivals import compute_arrival_grid_3d, write_arrival_grid_3d

            grid = compute_arrival_grid_3d(
                self.config,
                max_arrivals=max_arrivals,
                receiver_bearings=receiver_bearings,
                tolerance_m=tolerance_m,
            )
            if save_path is not None:
                write_arrival_grid_3d(save_path, grid)
            return grid

        if self.config.solver.type != "bellhop":
            raise NotImplementedError("Arrival grids are currently only supported for Bellhop and Bellhop3D.")

        from pyacoustics.solvers.bellhop.arrivals import compute_arrival_grid, write_arrivals_ascii

        grid = compute_arrival_grid(
            self.config,
            max_arrivals=max_arrivals,
            beam_shape=beam_shape,
            method=method,
            eigenray_tolerance_m=eigenray_tolerance_m,
            eigenray_angle_samples=eigenray_angle_samples,
        )
        if save_path is not None:
            source_type = "X" if getattr(self.config.solver, "source_type", "point") == "line" else "P"
            write_arrivals_ascii(save_path, grid, source_type=source_type)
        return grid

    def compare_arrivals_with_at(
        self,
        *,
        max_arrivals: int = 200,
        delay_tolerance: float = 0.02,
        at_bin_path: str = None,
        run_type: str = "A",
        beam_shape: Literal["hat", "gaussian"] = "hat",
        method: Literal["beam", "eigenray"] = "beam",
        eigenray_tolerance_m: float = 0.25,
        eigenray_angle_samples: int = None,
    ):
        """
        Compare native Bellhop arrivals against AT Bellhop .arr output.
        """
        if self.config.solver.type != "bellhop":
            raise NotImplementedError("Arrival parity comparison is currently only supported for Bellhop.")

        from pyacoustics.solvers.bellhop.arrival_parity import (
            compare_arrival_data,
            compute_native_arrival_data,
            run_at_arrival_data,
        )

        at_data = run_at_arrival_data(self.config, at_bin_path=at_bin_path, run_type=run_type)
        native_data = compute_native_arrival_data(
            self.config,
            max_arrivals=max_arrivals,
            beam_shape=beam_shape,
            method=method,
            eigenray_tolerance_m=eigenray_tolerance_m,
            eigenray_angle_samples=eigenray_angle_samples,
        )
        return compare_arrival_data(at_data, native_data, delay_tolerance=delay_tolerance)

    def compare_bellhop3d_arrivals_with_at(
        self,
        *,
        max_arrivals: int = 200,
        delay_tolerance: float = 0.02,
        angle_tolerance: float = None,
        tolerance_m: float = None,
        at_bin_path: str = None,
        run_type: str = "A",
    ):
        """
        Compare native Bellhop3D arrivals against AT Bellhop3D .arr output.
        """
        if self.config.solver.type != "bellhop3d":
            raise NotImplementedError("Bellhop3D arrival parity comparison is only supported for Bellhop3D.")

        from pyacoustics.solvers.bellhop3d.arrival_parity import run_bellhop3d_arrival_parity

        return run_bellhop3d_arrival_parity(
            self.config,
            at_bin_path=at_bin_path,
            run_type=run_type,
            delay_tolerance_s=delay_tolerance,
            angle_tolerance_deg=angle_tolerance,
            tolerance_m=tolerance_m,
            max_arrivals=max_arrivals,
        )

    def compare_rays_with_at(
        self,
        *,
        max_range_m: float = None,
        sample_count: int = 101,
        at_bin_path: str = None,
    ):
        """
        Compare native Bellhop ray paths against AT Bellhop .ray output.
        """
        if self.config.solver.type != "bellhop":
            raise NotImplementedError("Ray parity comparison is currently only supported for Bellhop.")

        from pyacoustics.solvers.bellhop.ray_parity import (
            compare_ray_records,
            compute_native_ray_records,
            run_at_ray_records,
        )

        at_records = run_at_ray_records(self.config, at_bin_path=at_bin_path)
        native_records = compute_native_ray_records(self.config)
        return compare_ray_records(
            at_records,
            native_records,
            max_range_m=max_range_m,
            sample_count=sample_count,
        )

    def compare_bellhop3d_rays_with_at(
        self,
        *,
        max_range_m: float = None,
        at_bin_path: str = None,
        run_type: str = "R",
    ):
        """
        Compare native Bellhop3D ray paths against AT Bellhop3D .ray output.
        """
        if self.config.solver.type != "bellhop3d":
            raise NotImplementedError("Bellhop3D ray parity comparison is only supported for Bellhop3D.")

        from pyacoustics.solvers.bellhop3d.ray_parity import run_bellhop3d_ray_parity

        return run_bellhop3d_ray_parity(
            self.config,
            at_bin_path=at_bin_path,
            run_type=run_type,
            max_range_m=max_range_m,
        )

    def compare_tl_with_at(
        self,
        *,
        r_grid: np.ndarray = None,
        z_grid: np.ndarray = None,
        run_type: Literal["C", "I", "S"] = "C",
        at_bin_path: str = None,
    ):
        """
        Compare native Bellhop TL against AT Bellhop .shd output.
        """
        if self.config.solver.type != "bellhop":
            raise NotImplementedError("Bellhop TL parity comparison is currently only supported for Bellhop.")

        from pyacoustics.solvers.bellhop.tl_parity import (
            compare_tl_grids,
            compute_native_tl_grid,
            run_at_tl_grid,
        )

        if r_grid is None:
            r_grid = np.asarray(self.config.geometry.receivers.ranges, dtype=np.float64)
        if z_grid is None:
            z_grid = np.asarray(self.config.geometry.receivers.depths, dtype=np.float64)

        at_tl = run_at_tl_grid(self.config, r_grid, z_grid, run_type=run_type, at_bin_path=at_bin_path)
        native_tl = compute_native_tl_grid(self.config, r_grid, z_grid, run_type=run_type)
        return compare_tl_grids(at_tl, native_tl)

    def compare_kraken_tl_with_at(
        self,
        *,
        r_grid: np.ndarray = None,
        z_grid: np.ndarray = None,
        at_bin_path: str = None,
        return_report: bool = False,
        case_name: str = "kraken-tl-parity",
    ):
        """
        Compare native Kraken TL against AT Kraken/KRAKENC + Field output.
        """
        if self.config.solver.type != "normal_modes":
            raise NotImplementedError("Kraken TL parity comparison is currently only supported for normal_modes.")

        from pyacoustics.solvers.kraken.tl_parity import (
            run_kraken_tl_parity,
        )

        if r_grid is None:
            r_grid = np.asarray(self.config.geometry.receivers.ranges, dtype=np.float64)
        if z_grid is None:
            z_grid = np.asarray(self.config.geometry.receivers.depths, dtype=np.float64)

        report = run_kraken_tl_parity(
            self.config,
            r_grid,
            z_grid,
            at_bin_path=at_bin_path,
            case_name=case_name,
        )
        return report if return_report else report.tl_metrics

    def compare_bellhop3d_tl_with_at(
        self,
        *,
        theta_grid_deg: np.ndarray = None,
        r_grid: np.ndarray = None,
        z_grid: np.ndarray = None,
        at_bin_path: str = None,
        run_type: Literal["C"] = "C",
        min_range_m: float = None,
        active_tl_max_db: float = 190.0,
    ):
        """
        Compare native Bellhop3D receiver-grid TL against AT Bellhop3D .shd output.
        """
        if self.config.solver.type != "bellhop3d":
            raise NotImplementedError("Bellhop3D TL parity comparison is only supported for Bellhop3D.")

        from pyacoustics.solvers.bellhop3d.tl_parity import (
            compare_tl_grids_3d,
            compute_native_tl_grid_3d,
            run_at_tl_grid_3d,
        )

        if theta_grid_deg is None:
            theta_grid_deg = np.asarray(getattr(self.config.solver, "receiver_bearings", [0.0]), dtype=np.float64)
        if r_grid is None:
            r_grid = np.asarray(self.config.geometry.receivers.ranges, dtype=np.float64)
        if z_grid is None:
            z_grid = np.asarray(self.config.geometry.receivers.depths, dtype=np.float64)

        at_tl = run_at_tl_grid_3d(
            self.config,
            theta_grid_deg,
            r_grid,
            z_grid,
            at_bin_path=at_bin_path,
            run_type=run_type,
        )
        native_tl = compute_native_tl_grid_3d(self.config, theta_grid_deg, r_grid, z_grid)
        return compare_tl_grids_3d(
            at_tl,
            native_tl,
            r_grid=r_grid,
            min_range_m=min_range_m,
            active_tl_max_db=active_tl_max_db,
        )

    def compute_coherent_tl_3d(
        self,
        num_x: int = 50,
        num_y: int = 50,
        num_z: int = 50,
        save_path: str = None,
        save_shd_path: str = None,
        grid_mode: Literal["voxel", "receiver"] = "voxel",
    ):
        """
        Computes coherent 3D Transmission Loss using 3D Gaussian beams.

        ``grid_mode="voxel"`` returns a Cartesian visualization grid with
        shape ``(x, y, z)``. ``grid_mode="receiver"`` returns the AT-style
        receiver grid with shape ``(bearing, depth, range)`` and stores it in
        ``_coherent_tl_receiver_grid_3d``.
        """
        if self.config.solver.type != "bellhop3d":
            raise RuntimeError("Must use Bellhop3D solver for 3D TL computation.")

        grid_mode_normalized = grid_mode.lower()
        if grid_mode_normalized not in {"voxel", "receiver"}:
            raise ValueError("grid_mode must be either 'voxel' or 'receiver'")
        if grid_mode_normalized == "receiver":
            theta_grid_deg = np.array([0.0], dtype=np.float64)
            r_grid = np.linspace(0.0, max(self.config.geometry.receivers.ranges), num_x)
            z_grid = np.linspace(0.0, self.config.environment.bottom.depth or 5000.0, num_z)
            return self.compute_coherent_tl_3d_receiver_grid(
                theta_grid_deg=theta_grid_deg,
                r_grid=r_grid,
                z_grid=z_grid,
                save_shd_path=save_shd_path,
            )

        from pyacoustics.solvers.bellhop3d.core import PyBellhop3D
        solver = PyBellhop3D(self.config)

        beam_data = solver.run_beam_tracing()

        r_max = max(self.config.geometry.receivers.ranges)
        z_max = self.config.environment.bottom.depth or 5000.0

        # Symmetric 3D spatial voxel grid centered at the source
        x_grid = np.linspace(-r_max, r_max, num_x)
        y_grid = np.linspace(-r_max, r_max, num_y)
        z_grid = np.linspace(0.0, z_max, num_z)

        from pyacoustics.solvers.bellhop3d.beam import compute_tl_field_coherent_3d

        TL = compute_tl_field_coherent_3d(
            x_grid, y_grid, z_grid,
            beam_data["rays_x"], beam_data["rays_y"], beam_data["rays_z"],
            beam_data["rays_q11"], beam_data["rays_q12"], beam_data["rays_q21"], beam_data["rays_q22"],
            beam_data["rays_amp"], beam_data["rays_tau"], beam_data["rays_kmah"], beam_data["rays_cpath"], beam_data["rays_phi"], beam_data["rays_elev"],
            beam_data["freq"], beam_data["c_src"],
            beam_data["d_azim"], beam_data["d_elev"]
        )

        self._coherent_tl_cache_3d = (TL, x_grid, y_grid, z_grid)

        # Output slice plots if requested
        if save_path:
            import matplotlib.pyplot as plt
            # Create a 2x2 plot showing different slices
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            from pyacoustics.plot import get_auto_clim
            vmin, vmax = get_auto_clim(TL)

            # Slice 1: XY plane at mid depth
            mid_z_idx = num_z // 2
            im0 = axs[0, 0].imshow(
                TL[:, :, mid_z_idx].T,
                extent=[x_grid[0]/1000, x_grid[-1]/1000, y_grid[0]/1000, y_grid[-1]/1000],
                aspect='auto', cmap='jet_r', vmin=vmin, vmax=vmax, origin='lower'
            )
            axs[0, 0].set_title(f"XY Plane (Depth={z_grid[mid_z_idx]:.0f}m)")
            axs[0, 0].set_xlabel("X (km)"); axs[0, 0].set_ylabel("Y (km)")

            # Slice 2: XZ plane at Y=0
            zero_y_idx = np.argmin(np.abs(y_grid))
            axs[0, 1].imshow(
                TL[:, zero_y_idx, :].T,
                extent=[x_grid[0]/1000, x_grid[-1]/1000, z_max, 0],
                aspect='auto', cmap='jet_r', vmin=vmin, vmax=vmax
            )
            axs[0, 1].set_title(f"XZ Plane (Y={y_grid[zero_y_idx]:.0f}m)")
            axs[0, 1].set_xlabel("X (km)"); axs[0, 1].set_ylabel("Depth (m)")

            # Slice 3: YZ plane at X=0
            zero_x_idx = np.argmin(np.abs(x_grid))
            axs[1, 0].imshow(
                TL[zero_x_idx, :, :].T,
                extent=[y_grid[0]/1000, y_grid[-1]/1000, z_max, 0],
                aspect='auto', cmap='jet_r', vmin=vmin, vmax=vmax
            )
            axs[1, 0].set_title(f"YZ Plane (X={x_grid[zero_x_idx]:.0f}m)")
            axs[1, 0].set_xlabel("Y (km)"); axs[1, 0].set_ylabel("Depth (m)")

            axs[1, 1].axis('off')
            plt.colorbar(im0, ax=axs[1,1], fraction=0.5, label='Transmission Loss (dB)')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        return TL

    def compute_coherent_tl_3d_receiver_grid(
        self,
        theta_grid_deg: np.ndarray = None,
        r_grid: np.ndarray = None,
        z_grid: np.ndarray = None,
        save_shd_path: str = None,
        mask_below_bathymetry: bool = False,
    ):
        """
        Computes coherent 3D TL directly on the Bellhop3D receiver grid.

        The returned TL has shape (theta, depth, range), matching AT .shd
        ordering before singleton bearing dimensions are squeezed by readers.
        """
        if self.config.solver.type != "bellhop3d":
            raise RuntimeError("Must use Bellhop3D solver for 3D TL computation.")

        if theta_grid_deg is None:
            theta_grid_deg = np.array([0.0], dtype=np.float64)
        else:
            theta_grid_deg = np.asarray(theta_grid_deg, dtype=np.float64)

        if r_grid is None:
            r_max = max(self.config.geometry.receivers.ranges)
            r_grid = np.linspace(0.0, r_max, 100)
        else:
            r_grid = np.asarray(r_grid, dtype=np.float64)

        if z_grid is None:
            z_max = self.config.environment.bottom.depth or 5000.0
            z_grid = np.linspace(0.0, z_max, 100)
        else:
            z_grid = np.asarray(z_grid, dtype=np.float64)

        from pyacoustics.solvers.bellhop3d.core import PyBellhop3D
        from pyacoustics.solvers.bellhop3d.beam import compute_pressure_field_receiver_grid_3d

        source_depths = np.asarray(self.config.geometry.source.depths, dtype=np.float32)
        source_x = np.asarray(self.config.geometry.source.x or [0.0], dtype=np.float64)
        source_y = np.asarray(self.config.geometry.source.y or [0.0], dtype=np.float64)
        pressure_by_source = np.empty(
            (source_x.size, source_y.size, theta_grid_deg.size, source_depths.size, z_grid.size, r_grid.size),
            dtype=np.complex128,
        )
        for isx, source_x_value in enumerate(source_x):
            for isy, source_y_value in enumerate(source_y):
                for isd, source_depth in enumerate(source_depths):
                    source_config = deepcopy(self.config)
                    source_config.geometry.source.depths = [float(source_depth)]
                    source_config.geometry.source.x = [float(source_x_value)]
                    source_config.geometry.source.y = [float(source_y_value)]
                    solver = PyBellhop3D(source_config)
                    beam_data = solver.run_beam_tracing()

                    pressure_by_source[isx, isy, :, isd, :, :] = compute_pressure_field_receiver_grid_3d(
                        theta_grid_deg, r_grid, z_grid,
                        beam_data["rays_x"], beam_data["rays_y"], beam_data["rays_z"],
                        beam_data["rays_q11"], beam_data["rays_q12"], beam_data["rays_q21"], beam_data["rays_q22"],
                        beam_data["rays_amp"], beam_data["rays_tau"], beam_data["rays_kmah"], beam_data["rays_cpath"], beam_data["rays_phi"], beam_data["rays_elev"],
                        beam_data["freq"], beam_data["c_src"],
                        float(source_x_value), float(source_y_value),
                        beam_data["d_azim"], beam_data["d_elev"],
                    )

        if mask_below_bathymetry:
            _mask_bellhop3d_pressure_below_bathymetry(
                pressure_by_source,
                source_x=source_x,
                source_y=source_y,
                theta_grid_deg=theta_grid_deg,
                r_grid=r_grid,
                z_grid=z_grid,
                bathymetry_3d=getattr(self.config.environment.bottom, "bathymetry_3d", None),
            )

        pressure_4d = pressure_by_source[0, 0]
        pressure_grid = pressure_4d[:, 0] if source_depths.size == 1 else pressure_4d
        if source_x.size * source_y.size > 1:
            pressure_grid = pressure_by_source
        pressure_abs = np.abs(pressure_grid)
        TL = -20.0 * np.log10(np.maximum(pressure_abs, 1.0e-20))
        TL[pressure_abs < 1.0e-20] = 200.0
        self._coherent_tl_receiver_grid_3d = (TL, theta_grid_deg, r_grid, z_grid)
        self._coherent_pressure_receiver_grid_3d = (pressure_grid, theta_grid_deg, r_grid, z_grid)
        if save_shd_path is not None:
            from pyacoustics.solvers.external_io import write_at_shd

            pressure_for_shd = pressure_by_source if source_x.size * source_y.size > 1 else pressure_4d
            write_at_shd(
                save_shd_path,
                pressure_for_shd,
                z_grid,
                r_grid,
                title=self.config.project,
                frequency=self.config.frequency,
                source_depths_m=source_depths,
                source_x_m=source_x,
                source_y_m=source_y,
                bearings_deg=theta_grid_deg,
            )
        return TL
