import yaml
from typing import List, Tuple, Optional, Literal

def make_env(
    yaml_path: str,
    depth: float,
    source_depth: float,
    angles: Tuple[float, float] = (-20.0, 20.0),
    num_beams: int = 21,
    ssp_type: Literal["c-linear", "spline", "n2-linear"] = "c-linear",
    ssp_data: Optional[List[Tuple[float, float]]] = None,
    frequency: float = 100.0,
    max_range: float = 50000.0,
    step_size: float = 0.0
) -> str:
    """
    Generate an acoustic simulation environment YAML configuration file.
    
    Args:
        yaml_path: Path to save the generated YAML file (e.g., 'workspace.yaml').
        depth: Bottom depth of the waveguide in meters.
        source_depth: Depth of the acoustic source in meters.
        angles: Min and max launch angles in degrees, e.g., (-20.0, 20.0).
        num_beams: Number of rays to launch.
        ssp_type: Interpolation type for the sound speed profile.
        ssp_data: List of (depth, sound_speed) points. If None, an isovelocity profile (c=1500) is used.
        frequency: Source frequency in Hz.
        max_range: Maximum range for the receivers/simulation in meters.
        
    Returns:
        The path to the generated YAML file.
    """
    if ssp_data is None:
        ssp_data = [(0.0, 1500.0), (depth, 1500.0)]
        
    # Convert tuples to schema expected format
    ssp_formatted = [{"depth": float(z), "c": float(c)} for z, c in ssp_data]

    config_dict = {
        "project": f"Sim_{yaml_path.split('.')[0]}",
        "frequency": float(frequency),
        "environment": {
            "ssp": {
                "type": ssp_type,
                "data": ssp_formatted
            },
            "surface": {"type": "vacuum"},
            "bottom": {
                "type": "acousto-elastic", 
                "depth": float(depth),
                "c_p": 1600.0,
                "density": 1.8
            }
        },
        "geometry": {
            "source": {"depths": [float(source_depth)]},
            "receivers": {
                "ranges": [0.0, float(max_range)],
                "depths": [0.0, float(depth)]
            }
        },
        "solver": {
            "type": "bellhop",
            "angles": [float(angles[0]), float(angles[1])],
            "num_beams": int(num_beams),
            "step_size": float(step_size)
        }
    }

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, sort_keys=False)

    return f"Configuration successfully written to {yaml_path}"
