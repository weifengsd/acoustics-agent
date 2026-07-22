import pytest
import numpy as np
from pyacoustics.config import ConfigLoader
from pyacoustics.schema import (
    BellhopConfig,
    BoundaryCondition,
    EnvironmentModel,
    GeometryModel,
    ReceiverModel,
    SimulationConfig,
    SourceModel,
    SSPModel,
    SSPLayer,
)
from pyacoustics.environment import SSPInterpolator
from pyacoustics.solvers.bellhop.core import PyBellhop
from pyacoustics.solvers.external_io import generate_at_bellhop_env

def test_linear_ssp():
    data = [SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1520.0)]
    model = SSPModel(type="c-linear", data=data)
    interp = SSPInterpolator(model)
    
    # Test at bounds
    c, dc, d2c = interp.evaluate(0.0)
    assert c == 1500.0
    assert dc == 0.2
    
    # Test midpoint
    c, dc, d2c = interp.evaluate(50.0)
    assert c == 1510.0
    assert dc == 0.2
    
    # Test out of bounds (implemented as linear extrapolation from ends)
    c, dc, d2c = interp.evaluate(-10.0)
    assert c == 1498.0


def test_linear_ssp_density():
    data = [
        SSPLayer(depth=0.0, c=1500.0, density=1.0),
        SSPLayer(depth=100.0, c=1520.0, density=1.5),
    ]
    model = SSPModel(type="c-linear", data=data)
    interp = SSPInterpolator(model)

    assert np.isclose(interp.evaluate_density(50.0), 1.25)
    
def test_spline_ssp():
    # 3 points for a simple curve
    data = [
        SSPLayer(depth=0.0, c=1500.0),
        SSPLayer(depth=50.0, c=1490.0),
        SSPLayer(depth=100.0, c=1510.0)
    ]
    model = SSPModel(type="spline", data=data)
    interp = SSPInterpolator(model)
    
    # At exact points, it should match
    c0, _, _ = interp.evaluate(0.0)
    assert pytest.approx(c0) == 1500.0
    
    c50, _, _ = interp.evaluate(50.0)
    assert pytest.approx(c50) == 1490.0


def test_quad_2d_ssp_evaluates_range_depth_bilinear_cell():
    model = SSPModel(
        type="quad-2d",
        data=[
            SSPLayer(depth=0.0, c=1500.0),
            SSPLayer(depth=100.0, c=1490.0),
            SSPLayer(depth=200.0, c=1510.0),
        ],
        ranges=[0.0, 1000.0, 3000.0],
        c_matrix=[
            [1500.0, 1502.0, 1506.0],
            [1490.0, 1496.0, 1504.0],
            [1510.0, 1518.0, 1530.0],
        ],
    )
    interp = SSPInterpolator(model)

    c, dc_dr, dc_dz, d2c_dz2 = interp.evaluate_range_depth(500.0, 50.0)

    assert np.isclose(c, 1497.0)
    assert np.isclose(dc_dr, 0.004)
    assert np.isclose(dc_dz, -0.08)
    assert d2c_dz2 == 0.0


def test_hexahedral_3d_ssp_interpolator_evaluates_xyz_gradient():
    x = [0.0, 10.0]
    y = [0.0, 20.0]
    z = [0.0, 5.0]
    c_grid = np.zeros((2, 2, 2), dtype=np.float64)
    for ix, xv in enumerate(x):
        for iy, yv in enumerate(y):
            for iz, zv in enumerate(z):
                c_grid[ix, iy, iz] = 1500.0 + 2.0 * xv + 3.0 * yv + 4.0 * zv
    model = SSPModel(
        type="hexahedral-3d",
        data=[],
        x=x,
        y=y,
        z=z,
        c=c_grid.tolist(),
    )
    interp = SSPInterpolator(model)

    speed, dc_dx, dc_dy, dc_dz, d2c_dz2 = interp.evaluate_3d(2.5, 5.0, 1.0)

    assert np.isclose(speed, 1524.0)
    assert np.isclose(dc_dx, 2.0)
    assert np.isclose(dc_dy, 3.0)
    assert np.isclose(dc_dz, 4.0)
    assert d2c_dz2 == 0.0


def test_hexahedral_3d_ssp_interpolator_rejects_missing_or_bad_grid():
    missing_grid = SSPModel(
        type="hexahedral-3d",
        data=[],
        x=[0.0, 10.0],
        y=[0.0, 20.0],
        z=[0.0, 5.0],
    )
    with pytest.raises(ValueError, match="x, y, z, and c grids"):
        SSPInterpolator(missing_grid).evaluate_3d(0.0, 0.0, 0.0)

    bad_shape = SSPModel(
        type="hexahedral-3d",
        data=[],
        x=[0.0, 10.0],
        y=[0.0, 20.0],
        z=[0.0, 5.0],
        c=np.zeros((2, 2, 1), dtype=np.float64).tolist(),
    )
    with pytest.raises(ValueError, match="shape must be"):
        SSPInterpolator(bad_shape).evaluate_3d(0.0, 0.0, 0.0)


def test_native_bellhop_traces_quad_2d_ssp_range_node():
    config = SimulationConfig(
        project="quad native tracing",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="quad-2d",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1510.0)],
                ranges=[0.0, 500.0, 1000.0],
                c_matrix=[[1500.0, 1510.0, 1520.0], [1500.0, 1510.0, 1520.0]],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(0.0, 0.0), num_beams=1, step_size=200.0),
    )

    traces = PyBellhop(config).run_ray_tracing()

    assert len(traces) == 1
    r_path, z_path, amp_path = traces[0]
    assert np.all(np.isfinite(r_path))
    assert np.all(np.isfinite(z_path))
    assert np.any(np.isclose(r_path, 500.0, atol=1.0e-8))


def test_config_loader_resolves_at_boundary_files(tmp_path):
    (tmp_path / "terrain.bty").write_text(
        "'LL'\n"
        "3\n"
        "0.0 100.0 1600.0 0.0 1.6 0.1 0.0\n"
        "1.0 120.0 1650.0 0.0 1.7 0.2 0.0\n"
        "2.5 90.0 1700.0 0.0 1.8 0.3 0.0\n"
    )
    (tmp_path / "surface.ati").write_text(
        "'L'\n"
        "2\n"
        "0.0 0.0\n"
        "2.5 1.0\n"
    )
    (tmp_path / "bottom.brc").write_text(
        "2\n"
        "0.0 0.5 0.0\n"
        "90.0 0.75 180.0\n"
    )
    (tmp_path / "surface.irc").write_text(
        "0.0 1.0 180.0\n"
        "90.0 1.0 180.0\n"
    )
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        "project: Boundary files\n"
        "frequency: 100.0\n"
        "environment:\n"
        "  ssp:\n"
        "    type: c-linear\n"
        "    data:\n"
        "      - {depth: 0.0, c: 1500.0}\n"
        "      - {depth: 200.0, c: 1500.0}\n"
        "  bottom:\n"
        "    type: acousto-elastic\n"
        "    depth: 120.0\n"
        "    rms_roughness_m: 0.25\n"
        "    bathymetry_file: terrain.bty\n"
        "    reflection_file: bottom.brc\n"
        "  surface:\n"
        "    type: vacuum\n"
        "    depth: 0.0\n"
        "    altimetry_file: surface.ati\n"
        "    reflection_file: surface.irc\n"
        "geometry:\n"
        "  source:\n"
        "    depths: [50.0]\n"
        "  receivers:\n"
        "    ranges: [1000.0]\n"
        "    depths: [50.0]\n"
        "solver:\n"
        "  type: bellhop\n"
        "  angles: [-10.0, 10.0]\n"
        "  num_beams: 3\n"
        "  step_size: 10.0\n"
    )

    config = ConfigLoader.load_yaml(config_path)

    assert config.environment.bottom.bathymetry_file == "terrain.bty"
    assert np.isclose(config.environment.bottom.rms_roughness_m, 0.25)
    assert config.environment.bottom.bathymetry_interpolation == "L"
    assert config.environment.bottom.bathymetry == [(0.0, 100.0), (1000.0, 120.0), (2500.0, 90.0)]
    assert config.environment.bottom.bathymetry_geoacoustics == [
        (1600.0, 0.0, 1.6, 0.1, 0.0),
        (1650.0, 0.0, 1.7, 0.2, 0.0),
        (1700.0, 0.0, 1.8, 0.3, 0.0),
    ]
    assert config.environment.bottom.reflection_file == "bottom.brc"
    assert config.environment.bottom.reflection_table == [(0.0, 0.5, 0.0), (90.0, 0.75, 180.0)]
    assert config.environment.surface.altimetry_file == "surface.ati"
    assert config.environment.surface.altimetry_interpolation == "L"
    assert config.environment.surface.altimetry == [(0.0, 0.0), (2500.0, 1.0)]
    assert config.environment.surface.reflection_file == "surface.irc"
    assert config.environment.surface.reflection_table == [(0.0, 1.0, 180.0), (90.0, 1.0, 180.0)]

    env_path = tmp_path / "case.env"
    generate_at_bellhop_env(config, env_path, run_type="R")
    assert "'F~' 0.25" in env_path.read_text().splitlines()
