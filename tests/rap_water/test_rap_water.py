from pathlib import Path
from pyacoustics.simulation import Simulation

def test_rap_water_runs_successfully():
    """Verify that the Reliable Acoustic Path (RAP) simulation runs without error."""
    config_path = Path(__file__).parent / "rap_water.yaml"
    
    sim = Simulation(config_path)
    rays = sim.run()
    
    # Check that it generated some rays
    assert len(rays) > 0
    
    # Verify rays propagated
    r_path, z_path, _ = rays[0]
    assert len(r_path) > 50
