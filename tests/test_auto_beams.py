
import yaml
from pyacoustics.simulation import Simulation
import numpy as np

from pathlib import Path

def test_auto_beams():
    # 1. Load Munk config
    config_path = Path(__file__).parent / 'Munk' / 'munk.yaml'
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 2. Ensure num_beams is 0 (to trigger auto-calculation)
    config_dict['solver']['num_beams'] = 0
    
    print(f"Initial num_beams in config: {config_dict['solver']['num_beams']}")
    
    # 3. Initialize simulation
    sim = Simulation(config_dict)
    
    # 4. Running the simulation will trigger _generate_angles in PyBellhop
    # which calculates the optimal number of beams.
    sim.run()
    
    # Check the calculated num_beams
    auto_num_beams = sim.config.solver.num_beams
    print(f"Automatically calculated num_beams: {auto_num_beams}")
    
    # 5. Compute coherent TL and save plot
    save_path = 'coherent_tl_munk_auto_beams.png'
    tl = sim.compute_coherent_tl(num_r=300, num_z=150, save_path=save_path)
    
    print(f"Coherent TL field computed. Plot saved to {save_path}")
    print(f"TL field shape: {tl.shape}")

if __name__ == "__main__":
    test_auto_beams()
