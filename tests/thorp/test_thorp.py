from pathlib import Path
from pyacoustics.simulation import Simulation

def test_thorp_runs_successfully():
    """Verify that the thorp absorption simulation runs without error."""
    config_path = Path(__file__).parent / "thorp.yaml"
    sim = Simulation(config_path)
    rays = sim.run()
    assert len(rays) > 0
    assert all(len(r) > 10 for r, z, _ in rays)
