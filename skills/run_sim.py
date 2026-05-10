import time
from pathlib import Path
from pyacoustics.simulation import Simulation

def run_sim(yaml_path: str, mode: str = "native") -> dict:
    """
    Execute the acoustic ray tracing simulation based on the given YAML configuration.
    
    Args:
        yaml_path: Path to the YAML configuration file.
        mode: "native" for the pure-Python implementation, "legacy" for the external AT binaries.
        
    Returns:
        A dictionary containing simulation statistics (execution time, rays traced, etc.).
    """
    if not Path(yaml_path).exists():
        return {"error": f"Configuration file {yaml_path} not found."}
        
    try:
        sim = Simulation(yaml_path)
        
        start_time = time.time()
        # For legacy mode, returns might differ, but we aim for structured stats
        results = sim.run(mode=mode)
        exec_time = time.time() - start_time
        
        if mode == "native":
            rays = results
            total_rays = len(rays)
            points_per_ray = [len(r[0]) for r in rays]
            avg_points = sum(points_per_ray) / total_rays if total_rays > 0 else 0
            
            return {
                "status": "success",
                "execution_time_seconds": round(exec_time, 4),
                "total_rays_traced": total_rays,
                "average_steps_per_ray": round(avg_points, 1),
                "max_steps_in_a_ray": max(points_per_ray) if total_rays > 0 else 0,
                "solver": sim.config.solver.type,
                "mode": mode
            }
        else:
            return {
                "status": "success",
                "execution_time_seconds": round(exec_time, 4),
                "solver": sim.config.solver.type,
                "mode": mode,
                "message": "Legacy simulation completed successfully."
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
