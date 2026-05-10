from pathlib import Path
from pyacoustics.simulation import Simulation

def test_arctic_runs_successfully():
    """Verify that the arctic deep water simulation runs without error."""
    config_path = Path(__file__).parent / "arctic.yaml"
    sim = Simulation(config_path)
    rays = sim.run()
    assert len(rays) > 0
    assert all(len(r) > 10 for r, z, _ in rays)
