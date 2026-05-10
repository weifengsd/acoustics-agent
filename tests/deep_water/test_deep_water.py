from pathlib import Path
from pyacoustics.simulation import Simulation

def test_deep_water_runs_successfully():
    """Verify that the deep water SOFAR channel simulation runs without error."""
    config_path = Path(__file__).parent / "deep_water.yaml"
    
    sim = Simulation(config_path)
    rays = sim.run()
    
    # Check that it generated some rays
    assert len(rays) > 0
    
    # Deep water simulation is very long (500km), verify rays didn't stop immediately
    r_path, z_path, _ = rays[0]
    assert len(r_path) > 100
