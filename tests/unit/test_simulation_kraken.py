import pytest
import numpy as np
from pyacoustics.schema import SimulationConfig, EnvironmentModel, SSPModel, SSPLayer, BoundaryCondition, GeometryModel, SourceModel, ReceiverModel, KrakenConfig
from pyacoustics.simulation import Simulation

def test_kraken_simulation_integration():
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=100.0, c=1500.0)
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(type="rigid", depth=100.0)
    )
    geom = GeometryModel(
        source=SourceModel(depths=[50.0]),
        receivers=ReceiverModel(ranges=[100.0, 500.0, 1000.0], depths=[10.0, 50.0, 90.0])
    )
    solver_config = KrakenConfig(n_mesh_points=500, mesh_multiplier=[1])
    
    config = SimulationConfig(
        project="Test Kraken Integration",
        frequency=50.0,
        environment=env,
        geometry=geom,
        solver=solver_config
    )
    
    sim = Simulation(config)
    result = sim.run()
    
    assert 'tl_grid' in result
    assert 'modes' in result
    assert 'roots' in result
    
    # Check that TL cache was set correctly
    assert hasattr(sim, '_tl_cache')
    assert hasattr(sim, '_coherent_tl_cache')
    
    tl = sim.get_tl(500.0, 50.0)
    assert isinstance(tl, float)
    assert not np.isnan(tl)
    assert tl > 0.0 # TL should be a positive value
