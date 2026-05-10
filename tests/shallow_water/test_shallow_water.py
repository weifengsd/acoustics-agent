from pathlib import Path
from pyacoustics.simulation import Simulation

def test_shallow_water_runs_successfully():
    """Verify that the shallow water isovelocity simulation runs without error."""
    config_path = Path(__file__).parent / "shallow_water.yaml"
    
    sim = Simulation(config_path)
    rays = sim.run()
    
    # Check that it generated some rays
    assert len(rays) > 0
    
    # Check that rays propagated successfully
    r_path, z_path, _ = rays[0]
    assert len(r_path) > 10
