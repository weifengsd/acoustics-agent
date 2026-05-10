from pathlib import Path
import os
from pyacoustics.simulation import Simulation

def plot_sim(yaml_path: str, output_img: str) -> str:
    """
    Run the acoustic simulation and generate a visualization plot containing
    the Sound Speed Profile and Ray Trajectories.
    
    Args:
        yaml_path: Path to the YAML configuration file.
        output_img: The name of the output image file (e.g., 'results.png').
        
    Returns:
        The absolute path to the generated image artifact.
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        return f"Error: Configuration file {yaml_path} not found."
        
    # By default, in an Agent environment, we might want to save plots 
    # to the current working directory, or a specific artifacts folder.
    # Here we save it directly to the requested output_img path.
    out_path = Path(output_img).absolute()
        
    try:
        sim = Simulation(yaml_path)
        # We need to run it before we can plot it
        sim.run()
        
        # Plot and save
        sim.plot_rays(str(out_path))
        
        return f"Plot successfully generated and saved to: {out_path}"
    except Exception as e:
        return f"Error during plotting: {str(e)}"
