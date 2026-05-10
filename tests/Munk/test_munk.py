import os
from pathlib import Path
from pyacoustics.simulation import Simulation

def test_munk_profile_runs_successfully():
    """Verify that the classic Munk profile simulation runs without error."""
    config_path = Path(__file__).parent / "munk.yaml"
    
    sim = Simulation(config_path)
    rays = sim.run()
    
    # Check that it generated some rays
    assert len(rays) > 0
    
    # Check that the first ray actually propagated somewhere
    r_path, z_path, _ = rays[0]
    assert len(r_path) > 100
    assert r_path[-1] > 0
