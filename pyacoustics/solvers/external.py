import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from pyacoustics.schema import SimulationConfig

class ExternalSolver:
    """
    Base class for wrapping legacy Fortran-based acoustics solvers (AT).
    """
    def __init__(self, config: SimulationConfig, bin_path: Optional[str] = None):
        self.config = config
        self.bin_path = self._resolve_bin_path(bin_path)
        
    def _resolve_bin_path(self, provided_path: Optional[str]) -> Path:
        """
        Determines the directory containing the legacy binaries.
        Priority: 
        1. provided_path argument
        2. AT_BIN_PATH environment variable
        3. Local 'at/bin' directory relative to the package
        """
        # 1. Check provided path
        if provided_path:
            p = Path(provided_path)
            if p.exists() and p.is_dir():
                return p
        
        # 2. Check environment variable
        env_path = os.environ.get("AT_BIN_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists() and p.is_dir():
                return p
                
        # 3. Check default relative path (relative to pyacoustics package)
        # Assuming structure: /some/path/pyacoustics/solvers/external.py
        # Default 'at/bin' might be in the root of the project: /some/path/at/bin
        pkg_root = Path(__file__).parent.parent.parent.parent # Coding/acoustics-agent/
        default_path = pkg_root / "at" / "bin"
        if default_path.exists() and default_path.is_dir():
            return default_path
            
        return None

    def is_available(self, exe_name: str) -> bool:
        """Checks if a specific executable exists in the bin_path."""
        if not self.bin_path:
            return False
        
        # Check for both with and without .exe extension
        exe_path = self.bin_path / exe_name
        if exe_path.exists():
            return True
            
        exe_path_with_ext = self.bin_path / f"{exe_name}.exe"
        if exe_path_with_ext.exists():
            return True
            
        return shutil.who(exe_name, path=str(self.bin_path)) is not None

    def run_command(self, cmd_args: list[str], cwd: str, stdin_input: str = None):
        """Runs the external command and handles errors."""
        try:
            result = subprocess.run(
                cmd_args, 
                cwd=cwd,
                input=stdin_input,
                capture_output=True, 
                text=True, 
                check=False  # We check manually to capture output on failure
            )
            if result.returncode != 0:
                error_msg = f"External solver failed (rc={result.returncode}): {result.stderr}\nOutput: {result.stdout}"
                raise RuntimeError(error_msg)
            return result.stdout
        except FileNotFoundError as e:
            raise RuntimeError(f"Executable not found: {cmd_args[0]}") from e

class BellhopExternal(ExternalSolver):
    """Wrapper for the legacy Bellhop ray tracing solver."""
    
    def run(self):
        """
        Executes legacy Bellhop binary. 
        Note: This is a hybrid implementation that handles both Ray Tracing (R) 
        and Coherent TL (C) based on common patterns.
        """
        if not self.is_available("bellhop"):
            raise RuntimeError(f"Legacy bellhop executable not found in {self.bin_path}. Please set AT_BIN_PATH.")
            
        from pyacoustics.solvers.external_io import generate_at_bellhop_env, read_at_shd, read_at_ray
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env_file = tmp_path / "legacy_run.env"
            
            # 1. Generate Input File
            generate_at_bellhop_env(self.config, env_file)
            
            # 2. Run Bellhop (Ray mode first to get paths)
            exe_path = self.bin_path / "bellhop"
            if not exe_path.exists():
                exe_path = self.bin_path / "bellhop.exe"
                
            self.run_command([str(exe_path), "legacy_run"], cwd=str(tmp_path))
            
            # 3. Parse Ray Paths
            ray_file = tmp_path / "legacy_run.ray"
            ray_paths = []
            if ray_file.exists():
                ray_paths = read_at_ray(ray_file)
            
            # 4. If we need Coherent TL, run it again in 'C' mode
            # (In legacy Bellhop, the .env 'R/C/I' option determines the output)
            # To be efficient, if the user requested TL, we should have used 'C' in generate_at_bellhop_env.
            # Here we just return the ray paths for now as standard run() behavior.
            return ray_paths

class KrakenExternal(ExternalSolver):
    """Wrapper for the legacy Kraken normal mode solver."""
    
    def run(self):
        """
        Executes legacy Kraken binary and then field binary to get TL.
        """
        if not self.is_available("kraken"):
            raise RuntimeError(f"Legacy kraken executable not found in {self.bin_path}.")
            
        from pyacoustics.solvers.external_io import generate_at_kraken_env, generate_at_field_flp, read_at_shd
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            case_name = "legacy_run"
            env_file = tmp_path / f"{case_name}.env"
            flp_file = tmp_path / f"{case_name}.flp"
            
            # 1. Generate Input Files
            generate_at_kraken_env(self.config, env_file)
            generate_at_field_flp(self.config, flp_file)
            import shutil
            debug_dir = Path("/Users/fengwei/VibeWorking/Coding/acoustics-agent-app/debug_output")
            debug_dir.mkdir(exist_ok=True)
            shutil.copy(env_file, debug_dir / "app_last.env")
            shutil.copy(flp_file, debug_dir / "app_last.flp")
            

            
            # 2. Run Kraken (or Krakenc for complex)
            # If there's attenuation or complex BC, we should use krakenc
            is_complex = False
            if self.config.environment.bottom.attenuation_p > 0:
                is_complex = True
            
            exe_name = "krakenc" if is_complex else "kraken"
            exe_path = self.bin_path / exe_name
            if not exe_path.exists():
                exe_path = self.bin_path / f"{exe_name}.exe"
                
            # Fallback to kraken if krakenc not found
            if not exe_path.exists() and is_complex:
                exe_path = self.bin_path / "kraken"
                if not exe_path.exists():
                    exe_path = self.bin_path / "kraken.exe"
            
            # Pass case_name both as argv AND via stdin to maximize compatibility
            self.run_command(
                [str(exe_path), case_name], 
                cwd=str(tmp_path),
                stdin_input=f"{case_name}\n"
            )
            mod_file = tmp_path / f"{case_name}.mod"
            if mod_file.exists():
                shutil.copy(mod_file, debug_dir / "app_last.mod")
            

            
            # 3. Run Field
            # field reads its input from the .flp file identified by the case name.
            # On many AT builds, field reads the case name from stdin (interactive mode).
            field_exe = self.bin_path / "field"
            if not field_exe.exists():
                field_exe = self.bin_path / "field.exe"
            # Pass case_name both as argv AND via stdin to maximize compatibility
            field_output = self.run_command(
                [str(field_exe), case_name],
                cwd=str(tmp_path),
                stdin_input=f"{case_name}\n"  # some AT builds expect stdin
            )
            
            # 4. Parse SHD
            shd_file = tmp_path / f"{case_name}.shd"
            if not shd_file.exists():
                # Read PRT file for detailed diagnostics
                prt_file = tmp_path / f"{case_name}.prt"
                prt_content = ""
                if prt_file.exists():
                    try:
                        prt_content = f"\nPRT Log Tail:\n{prt_file.read_text()[-2000:]}"
                    except: pass
                
                mod_file = tmp_path / f"{case_name}.mod"
                if not mod_file.exists():
                    raise RuntimeError(f"Legacy kraken run failed to generate .mod file. {prt_content}")
                raise RuntimeError(f"Legacy field run failed to generate .shd file. Check if .flp and .mod files are compatible.\nField Output: {field_output}{prt_content}")
                 
            from pyacoustics.solvers.external_io import generate_at_kraken_env, generate_at_field_flp, read_at_shd, read_at_mod
            
            # ... (after shd_file.exists check)
            P, rz, rr = read_at_shd(shd_file)
            
            # Also parse MOD file for mode shapes
            mod_file = tmp_path / f"{case_name}.mod"
            modes_data, z_bins_mod = read_at_mod(mod_file)
            
            # Standard output format: TL (dB) as a 2D array
            TL = -20 * np.log10(np.abs(P) + 1e-20)
            
            return {
                "tl_grid": TL,
                "r_bins": rr,
                "z_bins": rz,
                "modes": modes_data,
                "z_bins_mod": z_bins_mod
            }

import numpy as np # Needed for log10
