import numpy as np

from pyacoustics.schema import (
    Bellhop3DConfig,
    BoundaryCondition,
    EnvironmentModel,
    GeometryModel,
    ReceiverModel,
    SimulationConfig,
    SourceModel,
    SSPLayer,
    SSPModel,
)
from pyacoustics.solvers.bellhop3d.core import PyBellhop3D
from pyacoustics.solvers.bellhop3d.boundary import (
    apply_reflection_curvature_correction,
    get_bathymetry_3d_z,
    get_reflection_coef,
    reduce_step_fraction_3d,
    reflect_ray_3d,
)


def _minimal_bellhop3d_config(bottom_type: str) -> SimulationConfig:
    return SimulationConfig(
        project="3d bottom type mapping",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(
                type=bottom_type,
                depth=100.0,
                c_p=1700.0,
                c_s=600.0,
                attenuation_p=0.5,
                attenuation_s=0.1,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0], depths=[50.0]),
        ),
        solver=Bellhop3DConfig(
            elevation_angles=(0.0, 0.0),
            azimuth_angles=(0.0, 0.0),
            num_elevation_beams=1,
            num_azimuth_beams=1,
            step_size=1.0,
        ),
    )


def test_pybellhop3d_bottom_type_overrides_default_halfspace(monkeypatch):
    captured = []

    def fake_trace_single_ray_3d(*args, **kwargs):
        captured.append(
            (
                kwargs["bot_c"],
                kwargs["bot_c_s"],
                kwargs["bot_alpha"],
                kwargs["bot_alpha_s"],
            )
        )
        n = 1
        zeros = np.zeros(n, dtype=np.float64)
        zeros_c = np.zeros(n, dtype=np.complex128)
        zeros_i = np.zeros(n, dtype=np.int64)
        return (
            zeros,
            zeros,
            zeros,
            zeros_c,
            zeros_c,
            zeros_c,
            zeros_c,
            zeros_c,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros_i,
            zeros_i,
            n,
            0,
            0,
        )

    import pyacoustics.solvers.bellhop3d.core as core_mod

    monkeypatch.setattr(core_mod, "trace_single_ray_3d", fake_trace_single_ray_3d)

    PyBellhop3D(_minimal_bellhop3d_config("vacuum")).run_ray_tracing()
    PyBellhop3D(_minimal_bellhop3d_config("rigid")).run_ray_tracing()
    PyBellhop3D(_minimal_bellhop3d_config("acousto-elastic")).run_ray_tracing()

    assert captured == [
        (0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0),
        (1700.0, 600.0, 0.5, 0.1),
    ]


def test_bellhop3d_bathymetry_geoacoustics_controls_bottom_reflection():
    def make_config(use_geo: bool) -> SimulationConfig:
        bottom_kwargs = {}
        if use_geo:
            bottom_kwargs = {
                "bathymetry": [(0.0, 100.0), (1000.0, 100.0)],
                "bathymetry_geoacoustics": [
                    (2500.0, 900.0, 3.0, 0.6, 0.2),
                    (2500.0, 900.0, 3.0, 0.6, 0.2),
                ],
            }
        return SimulationConfig(
            project="3d bty geo reflection",
            frequency=100.0,
            environment=EnvironmentModel(
                ssp=SSPModel(
                    type="c-linear",
                    data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
                ),
                surface=BoundaryCondition(type="vacuum", depth=0.0),
                bottom=BoundaryCondition(
                    type="acousto-elastic",
                    depth=100.0,
                    c_p=1600.0,
                    c_s=0.0,
                    density=1.2,
                    attenuation_p=0.0,
                    attenuation_s=0.0,
                    **bottom_kwargs,
                ),
            ),
            geometry=GeometryModel(
                source=SourceModel(depths=[50.0]),
                receivers=ReceiverModel(ranges=[120.0], depths=[50.0]),
            ),
            solver=Bellhop3DConfig(
                elevation_angles=(45.0, 45.0),
                azimuth_angles=(0.0, 0.0),
                num_elevation_beams=1,
                num_azimuth_beams=1,
                step_size=5.0,
            ),
        )

    no_geo_ray = PyBellhop3D(make_config(False)).run_ray_tracing(include_phi=True)[0]
    geo_ray = PyBellhop3D(make_config(True)).run_ray_tracing(include_phi=True)[0]

    no_geo_amp = no_geo_ray[7]
    geo_amp = geo_ray[7]
    no_geo_first_bottom = np.flatnonzero(np.abs(no_geo_amp - 1.0) > 1.0e-10)[0]
    geo_first_bottom = np.flatnonzero(np.abs(geo_amp - 1.0) > 1.0e-10)[0]

    assert no_geo_first_bottom == geo_first_bottom
    assert not np.isclose(abs(no_geo_amp[no_geo_first_bottom]), abs(geo_amp[geo_first_bottom]))


def test_bellhop3d_bathymetry_uses_at_triangular_facets():
    bty_x = np.array([0.0, 1.0], dtype=np.float64)
    bty_y = np.array([0.0, 1.0], dtype=np.float64)
    bty_z = np.array(
        [
            [0.0, 10.0],
            [100.0, 1000.0],
        ],
        dtype=np.float64,
    )

    z_tri1, dz_dx_tri1, dz_dy_tri1 = get_bathymetry_3d_z(0.75, 0.25, bty_x, bty_y, bty_z)
    z_tri2, dz_dx_tri2, dz_dy_tri2 = get_bathymetry_3d_z(0.25, 0.75, bty_x, bty_y, bty_z)

    assert np.isclose(z_tri1, 255.0)
    assert np.isclose(dz_dx_tri1, 10.0)
    assert np.isclose(dz_dy_tri1, 990.0)
    assert np.isclose(z_tri2, 300.0)
    assert np.isclose(dz_dx_tri2, 900.0)
    assert np.isclose(dz_dy_tri2, 100.0)


def test_bellhop3d_reduce_step_stops_on_bathymetry_diagonal():
    bty_x = np.array([0.0, 1.0], dtype=np.float64)
    bty_y = np.array([0.0, 1.0], dtype=np.float64)
    empty = np.zeros(0, dtype=np.float64)

    frac = reduce_step_fraction_3d(
        0.25,
        0.75,
        100.0,
        1.0,
        -1.0,
        0.0,
        1.0,
        bty_x,
        bty_y,
        empty,
        empty,
        empty,
        False,
    )

    assert np.isclose(frac, 0.25)


def test_bellhop3d_reflection_curvature_rotates_dynamic_ray_on_sloped_facet():
    empty = np.zeros(0, dtype=np.float64)
    xi = 0.20 / 1500.0
    eta = 0.10 / 1500.0
    zeta = 0.97 / 1500.0
    xi_out, eta_out, zeta_out, _ = reflect_ray_3d(
        xi,
        eta,
        zeta,
        -1,
        0.2,
        -0.1,
        1.8,
        1700.0,
        0.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )

    p_in = (
        1.0 / 1500.0 + 0.0j,
        0.1 / 1500.0 + 0.0j,
        0.05 / 1500.0 + 0.0j,
        1.2 / 1500.0 + 0.0j,
    )
    q_in = (2.0 + 0.0j, 0.3 + 0.0j, -0.2 + 0.0j, 1.5 + 0.0j)

    p_out = apply_reflection_curvature_correction(
        *p_in,
        *q_in,
        xi,
        eta,
        zeta,
        xi_out,
        eta_out,
        zeta_out,
        -1,
        0.2,
        -0.1,
        1500.0,
        0.0,
        0.0,
        0.0,
    )

    assert np.all(np.isfinite(np.asarray(p_out, dtype=np.complex128)))
    assert not np.allclose(np.asarray(p_out), np.asarray(p_in))


def test_bellhop3d_reflection_curvature_applies_sound_speed_gradient_jump():
    empty = np.zeros(0, dtype=np.float64)
    xi = 0.20 / 1500.0
    eta = 0.10 / 1500.0
    zeta = 0.97 / 1500.0
    xi_out, eta_out, zeta_out, _ = reflect_ray_3d(
        xi,
        eta,
        zeta,
        -1,
        0.2,
        -0.1,
        1.8,
        1700.0,
        0.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )
    args = (
        1.0 / 1500.0 + 0.0j,
        0.1 / 1500.0 + 0.0j,
        0.05 / 1500.0 + 0.0j,
        1.2 / 1500.0 + 0.0j,
        2.0 + 0.0j,
        0.3 + 0.0j,
        -0.2 + 0.0j,
        1.5 + 0.0j,
        xi,
        eta,
        zeta,
        xi_out,
        eta_out,
        zeta_out,
        -1,
        0.2,
        -0.1,
        1500.0,
    )

    without_gradient = apply_reflection_curvature_correction(*args, 0.0, 0.0, 0.0)
    with_gradient = apply_reflection_curvature_correction(*args, 0.0, 0.0, 0.017)

    assert np.all(np.isfinite(np.asarray(with_gradient, dtype=np.complex128)))
    assert not np.allclose(np.asarray(with_gradient), np.asarray(without_gradient), atol=1.0e-12, rtol=0.0)


def test_bellhop3d_bottom_reflection_table_overrides_halfspace():
    empty = np.zeros(0, dtype=np.float64)
    table_angles = np.array([0.0, 90.0], dtype=np.float64)
    table_amp = np.array([0.5, 0.5], dtype=np.float64)
    table_phase = np.array([0.0, 0.0], dtype=np.float64)

    _, _, _, coeff = reflect_ray_3d(
        0.0,
        0.5 / 1500.0,
        0.5 / 1500.0,
        -1,
        0.0,
        0.0,
        1.8,
        1700.0,
        700.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        table_angles,
        table_amp,
        table_phase,
    )

    assert np.isclose(coeff, 0.5 + 0.0j)


def test_bellhop3d_rigid_surface_uses_positive_reflection():
    empty = np.zeros(0, dtype=np.float64)

    _, _, _, coeff = reflect_ray_3d(
        0.0,
        0.5 / 1500.0,
        -0.5 / 1500.0,
        1,
        0.0,
        0.0,
        1.8,
        1700.0,
        0.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
    )

    assert np.isclose(coeff, 1.0 + 0.0j)


def test_bellhop3d_acoustic_surface_uses_halfspace_coefficient():
    empty = np.zeros(0, dtype=np.float64)

    _, _, _, coeff = reflect_ray_3d(
        0.0,
        0.5 / 1500.0,
        -0.5 / 1500.0,
        1,
        0.0,
        0.0,
        1.8,
        1700.0,
        0.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        1.2,
        1300.0,
        0.0,
        0.0,
        0.0,
    )

    assert not np.isclose(coeff, -1.0 + 0.0j)
    assert not np.isclose(coeff, 1.0 + 0.0j)


def test_bellhop3d_bottom_vacuum_and_rigid_have_at_reflection_signs():
    empty = np.zeros(0, dtype=np.float64)
    args = (
        0.0,
        0.5 / 1500.0,
        0.5 / 1500.0,
        -1,
        0.0,
        0.0,
        1.0,
    )

    _, _, _, vacuum_coeff = reflect_ray_3d(
        *args,
        0.0,
        0.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )
    _, _, _, rigid_coeff = reflect_ray_3d(
        *args,
        -1.0,
        0.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )

    assert np.isclose(vacuum_coeff, -1.0 + 0.0j)
    assert np.isclose(rigid_coeff, 1.0 + 0.0j)


def test_bellhop3d_legacy_reflection_coef_uses_at_vacuum_rigid_signs():
    grazing = np.radians(30.0)

    vacuum_coeff = get_reflection_coef(grazing, 1.0, 1500.0, 1.0, 0.0, 0.0)
    rigid_coeff = get_reflection_coef(grazing, 1.0, 1500.0, 1.0, -1.0, 0.0)

    assert np.isclose(vacuum_coeff, -1.0 + 0.0j)
    assert np.isclose(rigid_coeff, 1.0 + 0.0j)


def test_bellhop3d_elastic_bottom_changes_halfspace_coefficient():
    empty = np.zeros(0, dtype=np.float64)
    args = (
        0.0,
        0.5 / 1500.0,
        0.5 / 1500.0,
        -1,
        0.0,
        0.0,
        1.8,
        1700.0,
    )

    _, _, _, fluid_coeff = reflect_ray_3d(
        *args,
        0.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )
    _, _, _, elastic_coeff = reflect_ray_3d(
        *args,
        700.0,
        0.0,
        0.0,
        1500.0,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )

    assert not np.isclose(fluid_coeff, elastic_coeff)
