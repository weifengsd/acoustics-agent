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

    def run_command(self, cmd_args: list[str], cwd: str):
        """Runs the external command and handles errors."""
        try:
            result = subprocess.run(
                cmd_args, 
                cwd=cwd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"External solver failed: {e.stderr}") from e

class BellhopExternal(ExternalSolver):
    """Wrapper for the legacy Bellhop ray tracing solver."""
    
    def run(self):
        if not self.is_available("bellhop"):
            raise RuntimeError("Legacy bellhop executable not found. Please set AT_BIN_PATH.")
            
        # Implementation for converting SimulationConfig to .env and running bellhop
        # This will be completed as needed.
        pass
