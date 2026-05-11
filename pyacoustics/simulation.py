import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional, Literal
from pathlib import Path
from pyacoustics.config import ConfigLoader
from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop.core import PyBellhop
from pyacoustics.solvers.external import ExternalSolver, BellhopExternal
from pyacoustics.plot import plot_rays

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

    def run(self, mode: Optional[Literal["native", "legacy"]] = None):
        """
        Runs the simulation based on the loaded configuration.
        """
        run_mode = mode if mode is not None else self.mode
        if run_mode == "legacy":
            if self.config.solver.type == "bellhop":
                solver = BellhopExternal(self.config, bin_path=self.external_solver.bin_path)
                return solver.run()
            else:
                raise NotImplementedError(f"Legacy solver for {self.config.solver.type} not yet implemented.")

        if self.config.solver.type == "bellhop":
            solver = PyBellhop(self.config)
            self.ray_paths = solver.run_ray_tracing()
            return self.ray_paths
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
                tl_grid.T, extent=[r_bins[0] / 1000.0, r_bins[-1] / 1000.0, z_bins[-1], 0],
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
            else:
                plt.show()
            plt.close(fig)
            return tl_grid
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
        
        # Clip to avoid boundary errors
        r_idx = min(r_idx, tl_grid.shape[0] - 1)
        z_idx = min(z_idx, tl_grid.shape[1] - 1)
        
        return float(tl_grid[r_idx, z_idx])

    def compute_coherent_tl(self, num_r: int = 200, num_z: int = 100, save_path: str = None, r_grid: np.ndarray = None, z_grid: np.ndarray = None):
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
                    
                    generate_at_bellhop_env(self.config, env_file)
                    with open(env_file, 'r') as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.strip() in ["'R'", "'A'", "'C'"]:
                            if i > 10: 
                                 lines[i] = "'C'\n"
                                 break
                    with open(env_file, 'w') as f:
                        f.writelines(lines)

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
                    return TL
                finally:
                    self.config.geometry.receivers.ranges = orig_rr
                    self.config.geometry.receivers.depths = orig_rd


        solver = PyBellhop(self.config)

        beam_data = solver.run_beam_tracing()


        r_max = max(self.config.geometry.receivers.ranges)
        z_max = self.config.environment.bottom.depth
        freq = self.config.frequency

        if r_grid is None:
            r_grid = np.linspace(100.0, r_max, num_r)  # Avoid r=0 singularity
        if z_grid is None:
            z_grid = np.linspace(0.0, z_max, num_z)

        # Angular spacing
        angles = self.config.solver.angles
        d_alpha = np.radians(angles[1] - angles[0]) / max(self.config.solver.num_beams - 1, 1)

        # Beam width heuristic: wavelength-based
        wavelength = beam_data['c_src'] / freq
        beam_width = max(wavelength, 10.0)

        from pyacoustics.solvers.bellhop.beam import compute_tl_field_coherent

        use_thorp = (self.config.environment.absorption_model == "thorp")

        TL = compute_tl_field_coherent(
            r_grid, z_grid,
            beam_data['ray_r'], beam_data['ray_z'],
            beam_data['ray_c'], beam_data['ray_tau'],
            beam_data['ray_q'], beam_data['ray_amp'],
            beam_data['ray_npts'],
            beam_data['ray_alphas'],
            freq, beam_data['c_src'],
            d_alpha,
            use_thorp=use_thorp
        )

        self._coherent_tl_cache = (TL, r_grid, z_grid)

        # Plot if requested
        if save_path:
            fig, ax = plt.subplots(figsize=(12, 6))
            from pyacoustics.plot import get_auto_clim
            vmin, vmax = get_auto_clim(TL)
            im = ax.imshow(
                TL.T, extent=[r_grid[0] / 1000.0, r_grid[-1] / 1000.0, z_max, 0],
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
                    generate_at_bellhop_env(self.config, env_file)
                    # Force Arrivals mode 'A'
                    with open(env_file, 'r') as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.strip() in ["'R'", "'C'", "'A'"]:
                            if i > 10:
                                lines[i] = "'A'\n"
                                break
                    with open(env_file, 'w') as f:
                        f.writelines(lines)

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

