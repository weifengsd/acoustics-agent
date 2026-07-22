import pytest
import numpy as np
from pyacoustics.schema import SimulationConfig, EnvironmentModel, SSPModel, SSPLayer, BoundaryCondition, GeometryModel, SourceModel, ReceiverModel, KrakenConfig
from pyacoustics.simulation import Simulation
from pyacoustics.solvers.external import KrakenExternal
from pyacoustics.solvers.external_io import read_at_mod, read_at_mod_metadata, read_at_mod_wavenumbers
from pyacoustics.solvers.kraken import PyKraken
from pyacoustics.solvers.kraken.core import complex_sound_speed


def _kraken_external_config(ssp_layers=None, bottom=None):
    ssp = SSPModel(type="c-linear", data=ssp_layers or [
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=100.0, c=1500.0),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=bottom or BoundaryCondition(type="rigid", depth=100.0),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[50.0]),
        receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
    )
    return SimulationConfig(
        project="Test Kraken External Solver Selection",
        frequency=50.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(n_mesh_points=300, mesh_multiplier=[1]),
    )


def test_kraken_external_uses_real_solver_for_lossless_case():
    config = _kraken_external_config()

    assert not KrakenExternal._needs_krakenc(config)


def test_kraken_external_uses_real_solver_for_lossless_acoustic_halfspace():
    config = _kraken_external_config(bottom=BoundaryCondition(
        type="acousto-elastic",
        depth=100.0,
        c_p=1600.0,
        density=1.8,
        attenuation_p=0.0,
    ))

    assert not KrakenExternal._needs_krakenc(config)


@pytest.mark.parametrize(
    "config",
    [
        _kraken_external_config(ssp_layers=[
            SSPLayer(depth=0.0, c=1500.0, attenuation=0.2),
            SSPLayer(depth=100.0, c=1500.0, attenuation=0.2),
        ]),
        _kraken_external_config(bottom=BoundaryCondition(
            type="acousto-elastic",
            depth=100.0,
            c_p=1600.0,
            density=1.8,
            attenuation_p=0.5,
        )),
        _kraken_external_config(bottom=BoundaryCondition(
            type="acousto-elastic",
            depth=100.0,
            c_p=1600.0,
            c_s=500.0,
            density=1.8,
            attenuation_s=0.2,
        )),
        _kraken_external_config(bottom=BoundaryCondition(
            type="acousto-elastic",
            depth=100.0,
            c_p=1600.0,
            density=1.8,
            reflection_table=[(0.0, 0.8, 0.0), (90.0, 0.8, 0.0)],
        )),
    ],
)
def test_kraken_external_uses_complex_solver_for_lossy_or_tabulated_cases(config):
    assert KrakenExternal._needs_krakenc(config)


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
    assert 'complex_roots' in result
    assert 'wavenumbers' in result
    assert 'phase_speeds' in result
    assert 'modal_attenuation' in result
    assert 'mode_metadata' in result
    assert result['wavenumbers'].shape == result['complex_roots'].shape
    assert result['phase_speeds'].shape == result['complex_roots'].shape
    assert result['modal_attenuation'].shape == result['complex_roots'].shape
    
    # Check that TL cache was set correctly
    assert hasattr(sim, '_tl_cache')
    assert hasattr(sim, '_coherent_tl_cache')
    
    tl = sim.get_tl(500.0, 50.0)
    assert isinstance(tl, float)
    assert not np.isnan(tl)
    assert tl > 0.0 # TL should be a positive value


def test_kraken_modal_diagnostics_reflect_attenuation():
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1500.0, attenuation=0.2),
        SSPLayer(depth=100.0, c=1500.0, attenuation=0.2),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(type="rigid", depth=100.0),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[50.0]),
        receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
    )
    config = SimulationConfig(
        project="Test Kraken Modal Diagnostics",
        frequency=50.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(n_mesh_points=300, mesh_multiplier=[1]),
    )

    result = PyKraken(config).run_normal_modes()

    assert np.all(np.real(result["wavenumbers"]) > 0.0)
    assert np.all(result["phase_speeds"] > 0.0)
    assert np.any(result["modal_attenuation"] > 0.0)


def test_kraken_lossless_acoustic_halfspace_adds_leaky_mode_attenuation():
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1490.0, density=1.0),
        SSPLayer(depth=60.0, c=1498.0, density=1.0),
        SSPLayer(depth=140.0, c=1520.0, density=1.03),
        SSPLayer(depth=220.0, c=1538.0, density=1.06),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(
            type="acousto-elastic",
            depth=220.0,
            c_p=1680.0,
            density=1.65,
            attenuation_p=0.0,
        ),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[70.0]),
        receivers=ReceiverModel(ranges=[800.0, 1800.0, 3200.0], depths=[50.0, 120.0, 190.0]),
    )
    config = SimulationConfig(
        project="Test Kraken Lossless Acoustic Halfspace Leakage",
        frequency=100.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1800.0), n_mesh_points=300, mesh_multiplier=[1]),
    )

    result = PyKraken(config).run_normal_modes()

    assert np.all(result["modal_attenuation"] >= 0.0)
    assert np.max(result["modal_attenuation"]) > 1.0e-3


def test_kraken_complex_root_refinement_handles_bottom_attenuation():
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=100.0, c=1500.0),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(
            type="acousto-elastic",
            depth=100.0,
            c_p=1600.0,
            density=1.8,
            attenuation_p=0.5,
        ),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[50.0]),
        receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
    )
    config = SimulationConfig(
        project="Test Kraken KRAKENC Complex Roots",
        frequency=50.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1650.0), n_mesh_points=300, mesh_multiplier=[1]),
    )

    result = PyKraken(config).run_normal_modes()

    assert len(result["complex_roots"]) == len(result["roots"])
    assert np.all(np.imag(result["wavenumbers"]) >= 0.0)
    assert np.any(result["modal_attenuation"] > 0.0)


def test_kraken_reflection_table_bottom_boundary_drives_complex_roots():
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=100.0, c=1500.0),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(
            type="acousto-elastic",
            depth=100.0,
            c_p=1600.0,
            density=1.8,
            reflection_table=[(0.0, 0.8, 0.0), (90.0, 0.8, 0.0)],
        ),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[50.0]),
        receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
    )
    config = SimulationConfig(
        project="Test Kraken Reflection Table",
        frequency=50.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1650.0), n_mesh_points=300, mesh_multiplier=[1]),
    )

    result = PyKraken(config).run_normal_modes()

    assert result["mode_metadata"]["bottom"]["bc"] == "F"
    assert len(result["complex_roots"]) == len(result["roots"])
    assert np.any(result["modal_attenuation"] > 0.0)


def test_kraken_mesh_multiplier_uses_finest_grid_for_modes():
    config = _kraken_external_config()
    config.solver.n_mesh_points = 120
    config.solver.mesh_multiplier = [1, 2]

    result = PyKraken(config).run_normal_modes()

    assert len(result["z_bins_mod"]) == 241
    assert result["modes"].shape[1] == 241


def test_kraken_write_modes_roundtrip(tmp_path):
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=100.0, c=1500.0),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(type="rigid", depth=100.0),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[50.0]),
        receivers=ReceiverModel(ranges=[100.0, 500.0, 1000.0], depths=[10.0, 50.0, 90.0]),
    )
    config = SimulationConfig(
        project="Test Kraken MOD writer",
        frequency=50.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(n_mesh_points=300, mesh_multiplier=[1]),
    )
    solver = PyKraken(config)
    result = solver.run_normal_modes()
    mod_path = tmp_path / "native.mod"

    solver.write_modes(result, mod_path)
    modes, z_bins = read_at_mod(mod_path)
    wavenumbers = read_at_mod_wavenumbers(mod_path)
    metadata = read_at_mod_metadata(mod_path)

    np.testing.assert_allclose(z_bins, result["z_bins_mod"].astype(np.float32))
    np.testing.assert_allclose(modes, result["modes"].astype(np.complex64), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(wavenumbers, result["wavenumbers"].astype(np.complex64), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(metadata["freq_vec"], [config.frequency])
    assert metadata["nmedia"] == len(result["mode_metadata"]["media"])


def test_kraken_write_modes_preserves_multi_medium_metadata(tmp_path):
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1495.0, density=1.0),
        SSPLayer(depth=70.0, c=1508.0, density=1.03),
        SSPLayer(depth=160.0, c=1525.0, density=1.08),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(type="rigid", depth=160.0),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[50.0]),
        receivers=ReceiverModel(ranges=[1000.0, 2200.0], depths=[40.0, 90.0, 140.0]),
    )
    config = SimulationConfig(
        project="Test Kraken multi-medium MOD writer",
        frequency=120.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1750.0), n_mesh_points=320, mesh_multiplier=[1]),
    )
    solver = PyKraken(config)
    result = solver.run_normal_modes()
    mod_path = tmp_path / "native_multi_medium.mod"

    solver.write_modes(result, mod_path)
    metadata = read_at_mod_metadata(mod_path)

    assert metadata["nmedia"] == 2
    assert metadata["nmedia"] == len(result["mode_metadata"]["media"])
    assert [medium["material"] for medium in metadata["media"]] == ["ACOUSTIC", "ACOUSTIC"]
    np.testing.assert_allclose([medium["depth"] for medium in metadata["media"]], [0.0, 70.0], atol=1.0e-6)
    np.testing.assert_allclose([medium["rho"] for medium in metadata["media"]], [1.0, 1.03], rtol=1.0e-6)
    assert metadata["bottom"]["bc"] == "R"
    np.testing.assert_allclose(metadata["bottom"]["depth"], 160.0)


def test_kraken_write_modes_preserves_lossy_elastic_bottom_metadata(tmp_path):
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1500.0, density=1.0),
        SSPLayer(depth=100.0, c=1510.0, density=1.0),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(
            type="acousto-elastic",
            depth=100.0,
            c_p=1700.0,
            c_s=600.0,
            density=1.8,
            attenuation_p=0.3,
            attenuation_s=0.05,
        ),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[40.0]),
        receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
    )
    config = SimulationConfig(
        project="Test Kraken lossy elastic MOD metadata",
        frequency=80.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(phase_speed_limits=(1450.0, 1800.0), n_mesh_points=260, mesh_multiplier=[1]),
    )
    solver = PyKraken(config)
    result = solver.run_normal_modes()
    mod_path = tmp_path / "native_lossy_elastic.mod"

    solver.write_modes(result, mod_path)
    metadata = read_at_mod_metadata(mod_path)

    assert metadata["bottom"]["bc"] == "A"
    np.testing.assert_allclose(metadata["bottom"]["c_p"], complex_sound_speed(1700.0, 0.3), rtol=1.0e-6)
    np.testing.assert_allclose(metadata["bottom"]["c_s"], complex_sound_speed(600.0, 0.05), rtol=1.0e-6)
    np.testing.assert_allclose(metadata["bottom"]["rho"], 1.8)
    np.testing.assert_allclose(metadata["bottom"]["depth"], 100.0)


def test_kraken_single_acoustic_media_model_matches_at_single_medium_env():
    ssp = SSPModel(type="c-linear", data=[
        SSPLayer(depth=0.0, c=1495.0, density=1.0),
        SSPLayer(depth=70.0, c=1508.0, density=1.03),
        SSPLayer(depth=160.0, c=1525.0, density=1.08),
    ])
    env = EnvironmentModel(
        ssp=ssp,
        surface=BoundaryCondition(type="vacuum"),
        bottom=BoundaryCondition(type="rigid", depth=160.0),
    )
    geom = GeometryModel(
        source=SourceModel(depths=[50.0]),
        receivers=ReceiverModel(ranges=[1000.0], depths=[90.0]),
    )
    config = SimulationConfig(
        project="Test Kraken single acoustic medium",
        frequency=120.0,
        environment=env,
        geometry=geom,
        solver=KrakenConfig(
            phase_speed_limits=(1450.0, 1750.0),
            n_mesh_points=320,
            mesh_multiplier=[1],
            acoustic_media_model="single",
        ),
    )

    result = PyKraken(config).run_normal_modes()

    assert result["mode_metadata"]["media"] == [
        {"n": 320, "material": "ACOUSTIC", "depth": 0.0, "rho": 1.0}
    ]
