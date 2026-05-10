from pathlib import Path
from pyacoustics.simulation import Simulation

def test_head_wave_runs_successfully():
    """Verify that the head wave simulation runs without error."""
    config_path = Path(__file__).parent / "head_wave.yaml"
    sim = Simulation(config_path)
    rays = sim.run()
    assert len(rays) > 0
    assert all(len(r) > 5 for r, z, _ in rays)
