from pathlib import Path
from pyacoustics.simulation import Simulation

def test_shaded_source_runs_successfully():
    """Verify that the shaded source simulation runs without error."""
    config_path = Path(__file__).parent / "shaded_source.yaml"
    sim = Simulation(config_path)
    rays = sim.run()
    assert len(rays) > 0
    assert all(len(r) > 10 for r, z, _ in rays)
