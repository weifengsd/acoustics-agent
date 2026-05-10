import os
from pathlib import Path
from pyacoustics.simulation import Simulation

def test_isovelocity_runs_successfully():
    """Verify that the isovelocity waveguide simulation runs without error."""
    config_path = Path(__file__).parent / "iso.yaml"
    
    sim = Simulation(config_path)
    rays = sim.run()
    
    # Check that it generated rays
    assert len(rays) > 0
    
    # In an isovelocity waveguide, a horizontal ray (angle 0) should remain horizontal
    # Let's find the horizontal ray
    horizontal_ray_idx = len(rays) // 2 # Since angles are -20 to 20, 0 is in the middle
    r_path, z_path, _ = rays[horizontal_ray_idx]
    
    # Verify the horizontal (or near-horizontal) ray stayed near the source depth (50m)
    # Allow a tolerance for small angles and floating point accumulation
    for z in z_path:
        assert abs(z - 50.0) < 1.0
