from pathlib import Path
from pyacoustics.simulation import Simulation

def test_lloyd_runs_successfully():
    """Verify that the Lloyd Mirror simulation runs without error."""
    config_path = Path(__file__).parent / "lloyd.yaml"
    sim = Simulation(config_path)
    rays = sim.run()
    assert len(rays) > 0
    assert all(len(r) > 5 for r, z, _ in rays)
