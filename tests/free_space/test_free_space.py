from pathlib import Path
from pyacoustics.simulation import Simulation

def test_free_space_runs_successfully():
    """Verify that the free space simulation runs without error."""
    config_path = Path(__file__).parent / "free_space.yaml"
    sim = Simulation(config_path)
    rays = sim.run()
    assert len(rays) > 0
    assert all(len(r) > 10 for r, z, _ in rays)
