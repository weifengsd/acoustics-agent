import pytest
import numpy as np
from pyacoustics.schema import (
    BellhopConfig,
    BoundaryCondition,
    EnvironmentModel,
    GeometryModel,
    ReceiverModel,
    SimulationConfig,
    SourceModel,
    SSPLayer,
    SSPModel,
)
from pyacoustics.solvers.bellhop.boundary import reflect_ray_amp, reflect_ray_amp_vectors
from pyacoustics.solvers.bellhop.core import PyBellhop
from pyacoustics.solvers.bellhop.step import step2d_kinematic_dynamic
from pyacoustics.solvers.bellhop.tracer import trace_beam_ray, trace_single_ray

class TestBoundaryReflection:
    """Tests for the ray tracer with boundary reflections in an isovelocity waveguide."""

    def setup_method(self):
        """Common setup: isovelocity waveguide, c=1500 m/s, depth=100m."""
        self.z_arr = np.array([0.0, 100.0])
        self.c_arr = np.array([1500.0, 1500.0])
        self.c_coeffs = np.zeros((4, 1))  # dummy for linear mode
        self.ssp_type = 0
        self.z_surface = 0.0
        self.z_bottom = 100.0
        self.ati_r = np.array([0.0, 1e9], dtype=np.float64)
        self.ati_z = np.array([0.0, 0.0], dtype=np.float64)
        self.bty_r = np.array([0.0, 1e9], dtype=np.float64)
        self.bty_z = np.array([100.0, 100.0], dtype=np.float64)
        self.empty_refl = np.zeros(0, dtype=np.float64)
        self.empty_geo = np.zeros(0, dtype=np.float64)
        self.empty_range = np.zeros(0, dtype=np.float64)
        self.surface_props = (1.0, 0.0, 0.0, 0.0, 0.0)
        self.h = 1.0       # step size 1m
        self.r_max = 500.0  # trace out to 500m range
        self.max_steps = 10000

    def test_pybellhop_bottom_type_overrides_default_halfspace(self):
        """Native Bellhop should honor schema bottom vacuum/rigid options."""
        def make_solver(bottom_type):
            config = SimulationConfig(
                project="bottom type mapping",
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
                solver=BellhopConfig(angles=(-5.0, 5.0), num_beams=3, step_size=1.0),
            )
            return PyBellhop(config)

        vacuum = make_solver("vacuum")
        rigid = make_solver("rigid")
        elastic = make_solver("acousto-elastic")

        assert vacuum.bot_c == 0.0
        assert rigid.bot_c == -1.0
        assert vacuum.bot_c_s == 0.0
        assert rigid.bot_c_s == 0.0
        assert elastic.bot_c == 1700.0
        assert elastic.bot_c_s == 600.0

    def test_horizontal_ray_no_bounce(self):
        """A horizontal ray at mid-depth should never hit a boundary."""
        r_path, z_path, _, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, 0.0,  # r0, z0, angle=0 (horizontal)
            self.h, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0, # bot properties
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps
        )
        assert n_bounces == 0
        # All z values should be 50m
        assert np.allclose(z_path[:n_pts], 50.0, atol=1e-6)

    def test_downward_ray_hits_bottom(self):
        """A downward ray should hit the bottom and bounce back up."""
        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, 30.0,  # 30 deg downward
            self.h, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0, # bot properties
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps
        )
        assert n_bounces >= 1
        # All z values should remain within [0, 100]
        valid_z = z_path[:n_pts]
        assert np.all(valid_z >= self.z_surface - 0.01)
        assert np.all(valid_z <= self.z_bottom + 0.01)

    def test_upward_ray_hits_surface(self):
        """An upward ray should hit the surface and bounce back down."""
        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, -30.0,  # -30 deg upward
            self.h, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0, # bot properties
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps
        )
        assert n_bounces >= 1
        valid_z = z_path[:n_pts]
        assert np.all(valid_z >= self.z_surface - 0.01)
        assert np.all(valid_z <= self.z_bottom + 0.01)

    def test_upward_ray_hits_range_dependent_surface(self):
        """An upward ray should hit a sloped surface profile instead of z=0."""
        ati_r = np.array([0.0, 1000.0], dtype=np.float64)
        ati_z = np.array([10.0, 10.0], dtype=np.float64)

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, -30.0,
            self.h, ati_r, ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        assert n_bounces > 0
        assert np.isclose(np.min(z_path[:n_pts]), 10.0, atol=0.25)

    def test_steep_ray_multiple_bounces(self):
        """A steep ray should bounce many times within 500m range."""
        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, 60.0,  # 60 deg steep downward
            self.h, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0, # bot properties
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps
        )
        assert n_bounces >= 3  # Should bounce many times
        valid_z = z_path[:n_pts]
        assert np.all(valid_z >= self.z_surface - 0.01)
        assert np.all(valid_z <= self.z_bottom + 0.01)

    def test_bottom_reflection_table_controls_amplitude(self):
        """A bottom .brc-style table should override half-space reflection amplitude."""
        refl_angles = np.array([0.0, 90.0], dtype=np.float64)
        refl_amp = np.array([0.5, 0.5], dtype=np.float64)
        refl_phase = np.array([0.0, 0.0], dtype=np.float64)

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, 30.0,
            self.h, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            refl_angles, refl_amp, refl_phase,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        assert n_bounces >= 1
        nonzero_amp = np.abs(amp_path[:n_pts][np.abs(amp_path[:n_pts]) > 0])
        reflected_amp = nonzero_amp[nonzero_amp < 0.999]
        assert reflected_amp.size > 0
        assert np.isclose(reflected_amp[0], 0.5)

    def test_surface_reflection_table_controls_complex_coefficient(self):
        """A surface .irc-style table should override pressure-release reflection."""
        refl_angles = np.array([0.0, 90.0], dtype=np.float64)
        refl_amp = np.array([0.25, 0.25], dtype=np.float64)
        refl_phase = np.array([180.0, 180.0], dtype=np.float64)

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, -30.0,
            self.h, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            refl_angles, refl_amp, refl_phase,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        assert n_bounces >= 1
        nonzero_amp = amp_path[:n_pts][np.abs(amp_path[:n_pts]) > 0]
        reflected_amp = nonzero_amp[np.abs(nonzero_amp) < 0.999]
        assert reflected_amp.size > 0
        assert np.isclose(reflected_amp[0], -0.25 + 0.0j)

    def test_rigid_surface_returns_positive_reflection(self):
        """A non-table rigid surface should reflect with +1 instead of vacuum -1."""
        xi = np.cos(np.radians(-30.0)) / 1500.0
        zeta = np.sin(np.radians(-30.0)) / 1500.0

        _, _, rigid_r = reflect_ray_amp_vectors(
            xi, zeta, 1,
            1.0, 0.0, 0.0, -1.0,
            1.0, 1500.0, 1.0, -1.0, 0.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )

        assert np.isclose(rigid_r, 1.0 + 0.0j)

    def test_bottom_vacuum_and_rigid_have_at_reflection_signs(self):
        """AT vacuum bottom is pressure-release (-1), while rigid bottom is +1."""
        xi = np.cos(np.radians(30.0)) / 1500.0
        zeta = np.sin(np.radians(30.0)) / 1500.0

        _, _, vacuum_r = reflect_ray_amp_vectors(
            xi, zeta, -1,
            1.0, 0.0, 0.0, 1.0,
            1.0, 1500.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )
        _, _, rigid_r = reflect_ray_amp_vectors(
            xi, zeta, -1,
            1.0, 0.0, 0.0, 1.0,
            1.0, 1500.0, 1.0, -1.0, 0.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )

        assert np.isclose(vacuum_r, -1.0 + 0.0j)
        assert np.isclose(rigid_r, 1.0 + 0.0j)

    def test_legacy_bottom_reflection_helper_uses_at_vacuum_rigid_signs(self):
        """The older slope-based helper should match AT bottom V/R signs too."""
        xi = np.cos(np.radians(30.0)) / 1500.0
        zeta = np.sin(np.radians(30.0)) / 1500.0

        _, _, vacuum_r = reflect_ray_amp(
            xi, zeta, -1, 0.0,
            1.0, 1500.0, 1.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )
        _, _, rigid_r = reflect_ray_amp(
            xi, zeta, -1, 0.0,
            1.0, 1500.0, 1.0, -1.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )

        assert np.isclose(vacuum_r, -1.0 + 0.0j)
        assert np.isclose(rigid_r, 1.0 + 0.0j)

    def test_acoustic_surface_uses_halfspace_reflection(self):
        """A positive surface c_p should use the AT half-space branch."""
        xi = np.cos(np.radians(-30.0)) / 1500.0
        zeta = np.sin(np.radians(-30.0)) / 1500.0

        _, _, surface_r = reflect_ray_amp_vectors(
            xi, zeta, 1,
            1.0, 0.0, 0.0, -1.0,
            1.0, 1500.0, 1.2, 1300.0, 0.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )

        assert not np.isclose(surface_r, -1.0 + 0.0j)
        assert not np.isclose(surface_r, 1.0 + 0.0j)

    def test_surface_altimetry_geoacoustics_controls_halfspace_reflection(self):
        """AT long-format .ati geoacoustics should override default surface media."""
        ati_geo_cp = np.array([1300.0, 1300.0], dtype=np.float64)
        ati_geo_cs = np.array([0.0, 0.0], dtype=np.float64)
        ati_geo_rho = np.array([1.2, 1.2], dtype=np.float64)
        ati_geo_alpha = np.array([0.0, 0.0], dtype=np.float64)
        ati_geo_alpha_s = np.array([0.0, 0.0], dtype=np.float64)

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, -30.0,
            self.h, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            ati_geo_cp, ati_geo_cs, ati_geo_rho, ati_geo_alpha, ati_geo_alpha_s,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            120.0,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        assert n_bounces >= 1
        reflected = amp_path[:n_pts][np.abs(amp_path[:n_pts] - 1.0) > 1.0e-6]
        assert reflected.size > 0

        xi = np.cos(np.radians(-30.0)) / 1500.0
        zeta = np.sin(np.radians(-30.0)) / 1500.0
        _, _, expected_r = reflect_ray_amp_vectors(
            xi, zeta, 1,
            1.0, 0.0, 0.0, -1.0,
            1.0, 1500.0, 1.2, 1300.0, 0.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )
        assert np.isclose(reflected[0], expected_r)
        assert not np.isclose(reflected[0], -1.0 + 0.0j)

    def test_elastic_bottom_shear_speed_changes_reflection_coefficient(self):
        """A nonzero shear speed should use the AT elastic half-space branch."""
        xi = np.cos(np.radians(30.0)) / 1500.0
        zeta = np.sin(np.radians(30.0)) / 1500.0

        _, _, fluid_r = reflect_ray_amp_vectors(
            xi, zeta, -1,
            1.0, 0.0, 0.0, 1.0,
            1.0, 1500.0, 1.8, 1700.0, 0.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )
        _, _, elastic_r = reflect_ray_amp_vectors(
            xi, zeta, -1,
            1.0, 0.0, 0.0, 1.0,
            1.0, 1500.0, 1.8, 1700.0, 700.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )

        assert not np.isclose(fluid_r, elastic_r)

    def test_elastic_bottom_uses_at_shear_wavenumber_branch(self):
        """Elastic reflection should use the decaying shear branch for evanescent waves."""
        xi = np.cos(np.radians(16.699205)) / 1500.0
        zeta = np.sin(np.radians(16.699205)) / 1500.0

        _, _, elastic_r = reflect_ray_amp_vectors(
            xi, zeta, -1,
            1.0, 0.0, 0.0, 1.0,
            1.0, 1500.0, 1.8, 1700.0, 700.0, 0.0, 0.0,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
        )

        assert np.isclose(abs(elastic_r), 0.7584835101342591)
        assert abs(elastic_r) < 1.0

    def test_seongil_beam_displacement_is_explicit_opt_in(self):
        """AT Beam%Type(4:4)='S' shift should only run when requested."""
        args = (
            0.0, 50.0, 30.0,
            self.h, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.5, 0.0,
            100.0,
        )

        plain = trace_beam_ray(
            *args,
            0,
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )
        shifted = trace_beam_ray(
            *args,
            1,
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        plain_r, plain_z, _, plain_tau, _, plain_q, _, _, _, plain_bot, plain_n, _ = plain
        shifted_r, shifted_z, _, shifted_tau, _, shifted_q, _, _, _, shifted_bot, shifted_n, _ = shifted
        plain_bottom_idx = int(np.flatnonzero(plain_bot[:plain_n] > 0)[0])
        shifted_bottom_idx = int(np.flatnonzero(shifted_bot[:shifted_n] > 0)[0])

        assert np.isclose(plain_z[plain_bottom_idx], self.z_bottom)
        assert not np.isclose(shifted_r[shifted_bottom_idx], plain_r[plain_bottom_idx])
        assert np.isclose(shifted_z[shifted_bottom_idx], plain_z[plain_bottom_idx])
        assert not np.isclose(shifted_tau[shifted_bottom_idx], plain_tau[plain_bottom_idx])
        assert not np.isclose(shifted_q[shifted_bottom_idx], plain_q[plain_bottom_idx])

    def test_ray_lands_on_interior_ssp_interface(self):
        """AT ReduceStep2D lands rays on SSP z interfaces before continuing."""
        z_arr = np.array([0.0, 40.0, 200.0], dtype=np.float64)
        c_arr = np.array([1500.0, 1510.0, 1530.0], dtype=np.float64)

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 20.0, 30.0,
            80.0, self.ati_r, self.ati_z, 0, np.array([0.0, 1e9]), np.array([200.0, 200.0]), 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            300.0,
            0, z_arr, c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        assert n_bounces == 0
        assert np.any(np.isclose(z_path[:n_pts], 40.0, atol=1.0e-8))

    def test_ray_lands_on_earlier_ssp_interface_before_later_boundary(self):
        """AT ReduceStep2D processes the first event inside a large step."""
        z_arr = np.array([0.0, 40.0, 60.0], dtype=np.float64)
        c_arr = np.array([1500.0, 1510.0, 1520.0], dtype=np.float64)
        bottom_r = np.array([0.0, 1e9], dtype=np.float64)
        bottom_z = np.array([60.0, 60.0], dtype=np.float64)

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 20.0, 30.0,
            100.0, self.ati_r, self.ati_z, 0, bottom_r, bottom_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            300.0,
            0, z_arr, c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        assert np.isclose(z_path[1], 40.0, atol=1.0e-8)
        assert n_bounces >= 1
        assert np.any(np.isclose(z_path[:n_pts], 60.0, atol=1.0e-8))

    def test_ray_lands_on_quad_2d_depth_interface(self):
        """Q SSP depth-grid interfaces are ReduceStep2D events."""
        z_arr = np.array([0.0, 40.0, 200.0], dtype=np.float64)
        c_arr = np.array([1500.0, 1510.0, 1530.0], dtype=np.float64)
        ssp_r = np.array([0.0, 1000.0], dtype=np.float64)
        c_matrix = np.array(
            [
                [1500.0, 1500.0],
                [1510.0, 1510.0],
                [1530.0, 1530.0],
            ],
            dtype=np.float64,
        )

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 20.0, 30.0,
            80.0, self.ati_r, self.ati_z, 0, np.array([0.0, 1e9]), np.array([200.0, 200.0]), 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            300.0,
            3, z_arr, c_arr, c_matrix,
            ssp_r,
            self.max_steps,
        )

        assert n_bounces == 0
        assert np.any(np.isclose(z_path[:n_pts], 40.0, atol=1.0e-8))

    def test_ray_lands_on_range_dependent_boundary_node_before_later_reflection(self):
        """AT ReduceStep2D lands on range-dependent boundary nodes before reflection."""
        bottom_r = np.array([0.0, 50.0, 1e9], dtype=np.float64)
        bottom_z = np.array([100.0, 60.0, 60.0], dtype=np.float64)

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 20.0, 30.0,
            150.0, self.ati_r, self.ati_z, 0, bottom_r, bottom_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            300.0,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        assert np.isclose(r_path[1], 50.0, atol=1.0e-8)
        assert z_path[1] < 60.0
        assert n_bounces >= 1
        assert np.any(np.isclose(z_path[:n_pts], 60.0, atol=1.0e-8))

    def test_ray_lands_on_range_dependent_ssp_node(self):
        """AT ReduceStep2D lands on SSP range-profile nodes for Q environments."""
        ssp_r = np.array([0.0, 50.0, 1e9], dtype=np.float64)

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 20.0, 0.0,
            150.0, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            300.0,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            ssp_r,
            self.max_steps,
        )

        assert n_bounces == 0
        assert np.isclose(r_path[1], 50.0, atol=1.0e-8)
        assert np.isclose(z_path[1], 20.0, atol=1.0e-8)

    def test_ray_lands_on_profile_node_before_same_point_reflection(self):
        """AT ReduceStep2D tie-break keeps boundary nodes before same-point reflection."""
        bottom_r = np.array([0.0, 50.0, 1e9], dtype=np.float64)
        bottom_z = np.array([100.0, 50.0, 50.0], dtype=np.float64)
        alpha = float(np.degrees(np.arctan2(30.0, 50.0)))

        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 20.0, alpha,
            100.0, self.ati_r, self.ati_z, 0, bottom_r, bottom_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            300.0,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )

        assert np.allclose(r_path[1:3], [50.0, 50.0], atol=1.0e-8)
        assert np.allclose(z_path[1:3], [50.0, 50.0], atol=1.0e-8)
        assert n_bounces >= 1

    def test_beam_ray_applies_ssp_interface_gradient_jump_to_p(self):
        """C-linear SSP interfaces apply the AT Step2D dynamic-ray p jump."""
        z_arr = np.array([0.0, 40.0, 200.0], dtype=np.float64)
        c_arr = np.array([1500.0, 1510.0, 1530.0], dtype=np.float64)
        h = 80.0
        z0 = 20.0
        alpha = 30.0
        c0 = 1505.0
        xi0 = np.cos(np.radians(alpha)) / c0
        zeta0 = np.sin(np.radians(alpha)) / c0

        full = step2d_kinematic_dynamic(
            h, 0.0, z0, xi0, zeta0, 0.0, 1.0, 0.0,
            0, z_arr, c_arr, self.c_coeffs,
            self.empty_range,
        )
        r_new, z_new, xi_new, zeta_new, tau_new, p_new, q_new, _ = full
        frac = (40.0 - z0) / (z_new - z0)
        xi_int = xi0 + frac * (xi_new - xi0)
        zeta_int = zeta0 + frac * (zeta_new - zeta0)
        p_uncorrected = 1.0 + frac * (p_new - 1.0)
        q_int = frac * q_new
        c_int = 1510.0
        dc_old = (1510.0 - 1500.0) / 40.0
        dc_new = (1530.0 - 1510.0) / 160.0
        grad_jump_z = dc_new - dc_old
        cnjump = grad_jump_z * xi_int
        csjump = grad_jump_z * zeta_int
        rn = (xi_int / zeta_int) * (2.0 * cnjump - (xi_int / zeta_int) * csjump) / c_int
        expected_p = p_uncorrected - q_int * rn

        traced = trace_beam_ray(
            0.0, z0, alpha,
            h, self.ati_r, self.ati_z, 0, np.array([0.0, 1e9]), np.array([200.0, 200.0]), 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            50.0,
            0,
            300.0,
            0, z_arr, c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )
        _, z_path, _, _, p_path, q_path, _, _, _, _, n_pts, _ = traced
        interface_idx = int(np.flatnonzero(np.isclose(z_path[:n_pts], 40.0, atol=1.0e-8))[0])

        assert np.isclose(p_path[interface_idx], expected_p)
        assert not np.isclose(p_path[interface_idx], p_uncorrected)
        assert np.isclose(q_path[interface_idx], q_int)

    def test_beam_ray_applies_quad_2d_depth_interface_gradient_jump_to_p(self):
        """Q SSP depth-grid interfaces apply the Step2D dynamic-ray p jump."""
        z_arr = np.array([0.0, 40.0, 200.0], dtype=np.float64)
        c_arr = np.array([1500.0, 1510.0, 1530.0], dtype=np.float64)
        ssp_r = np.array([0.0, 1000.0], dtype=np.float64)
        c_matrix = np.array(
            [
                [1500.0, 1500.0],
                [1510.0, 1510.0],
                [1530.0, 1530.0],
            ],
            dtype=np.float64,
        )
        h = 80.0
        z0 = 20.0
        alpha = 30.0
        c0 = 1505.0
        xi0 = np.cos(np.radians(alpha)) / c0
        zeta0 = np.sin(np.radians(alpha)) / c0

        full = step2d_kinematic_dynamic(
            h, 0.0, z0, xi0, zeta0, 0.0, 1.0, 0.0,
            3, z_arr, c_arr, c_matrix,
            ssp_r,
        )
        r_new, z_new, xi_new, zeta_new, tau_new, p_new, q_new, _ = full
        frac = (40.0 - z0) / (z_new - z0)
        xi_int = xi0 + frac * (xi_new - xi0)
        zeta_int = zeta0 + frac * (zeta_new - zeta0)
        p_uncorrected = 1.0 + frac * (p_new - 1.0)
        q_int = frac * q_new
        c_int = 1510.0
        dc_old = (1510.0 - 1500.0) / 40.0
        dc_new = (1530.0 - 1510.0) / 160.0
        grad_jump_z = dc_new - dc_old
        cnjump = grad_jump_z * xi_int
        csjump = grad_jump_z * zeta_int
        rn = (xi_int / zeta_int) * (2.0 * cnjump - (xi_int / zeta_int) * csjump) / c_int
        expected_p = p_uncorrected - q_int * rn

        traced = trace_beam_ray(
            0.0, z0, alpha,
            h, self.ati_r, self.ati_z, 0, np.array([0.0, 1e9]), np.array([200.0, 200.0]), 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            50.0,
            0,
            300.0,
            3, z_arr, c_arr, c_matrix,
            ssp_r,
            self.max_steps,
        )
        _, z_path, _, _, p_path, q_path, _, _, _, _, n_pts, _ = traced
        interface_idx = int(np.flatnonzero(np.isclose(z_path[:n_pts], 40.0, atol=1.0e-8))[0])

        assert np.isclose(p_path[interface_idx], expected_p)
        assert not np.isclose(p_path[interface_idx], p_uncorrected)
        assert np.isclose(q_path[interface_idx], q_int)

    def test_beam_ray_lands_on_earlier_ssp_interface_before_later_boundary(self):
        """Dynamic rays keep the SSP interface event before a later reflection."""
        z_arr = np.array([0.0, 40.0, 60.0], dtype=np.float64)
        c_arr = np.array([1500.0, 1510.0, 1520.0], dtype=np.float64)
        bottom_r = np.array([0.0, 1e9], dtype=np.float64)
        bottom_z = np.array([60.0, 60.0], dtype=np.float64)

        traced = trace_beam_ray(
            0.0, 20.0, 30.0,
            100.0, self.ati_r, self.ati_z, 0, bottom_r, bottom_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            50.0,
            0,
            300.0,
            0, z_arr, c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )
        _, z_path, _, _, _, _, _, _, _, bot_bnc, n_pts, n_bounces = traced

        assert np.isclose(z_path[1], 40.0, atol=1.0e-8)
        assert n_bounces >= 1
        assert np.any(bot_bnc[:n_pts] > 0)

    def test_beam_ray_lands_on_range_dependent_boundary_node_before_later_reflection(self):
        """Dynamic rays keep range-dependent boundary node events before reflection."""
        bottom_r = np.array([0.0, 50.0, 1e9], dtype=np.float64)
        bottom_z = np.array([100.0, 60.0, 60.0], dtype=np.float64)

        traced = trace_beam_ray(
            0.0, 20.0, 30.0,
            150.0, self.ati_r, self.ati_z, 0, bottom_r, bottom_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            50.0,
            0,
            300.0,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )
        r_path, z_path, _, _, _, _, _, _, _, bot_bnc, n_pts, n_bounces = traced

        assert np.isclose(r_path[1], 50.0, atol=1.0e-8)
        assert z_path[1] < 60.0
        assert n_bounces >= 1
        assert np.any(bot_bnc[:n_pts] > 0)

    def test_beam_ray_lands_on_range_dependent_ssp_node(self):
        """Dynamic rays keep SSP range-profile node events before continuing."""
        ssp_r = np.array([0.0, 50.0, 1e9], dtype=np.float64)

        traced = trace_beam_ray(
            0.0, 20.0, 0.0,
            150.0, self.ati_r, self.ati_z, 0, self.bty_r, self.bty_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            50.0,
            0,
            300.0,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            ssp_r,
            self.max_steps,
        )
        r_path, z_path, _, _, _, _, _, _, top_bnc, bot_bnc, n_pts, n_bounces = traced

        assert n_bounces == 0
        assert np.isclose(r_path[1], 50.0, atol=1.0e-8)
        assert np.isclose(z_path[1], 20.0, atol=1.0e-8)
        assert top_bnc[1] == 0
        assert bot_bnc[1] == 0

    def test_beam_ray_applies_quad_2d_range_interface_gradient_jump_to_p(self):
        """Q SSP range interfaces apply the Step2D dynamic-ray p jump."""
        z_arr = np.array([0.0, 200.0], dtype=np.float64)
        c_arr = np.array([1500.0, 1500.0], dtype=np.float64)
        ssp_r = np.array([0.0, 50.0, 100.0], dtype=np.float64)
        c_matrix = np.array(
            [
                [1500.0, 1500.0, 1510.0],
                [1500.0, 1500.0, 1510.0],
            ],
            dtype=np.float64,
        )
        h = 100.0
        z0 = 20.0
        alpha = 30.0
        c0 = 1500.0
        xi0 = np.cos(np.radians(alpha)) / c0
        zeta0 = np.sin(np.radians(alpha)) / c0

        full = step2d_kinematic_dynamic(
            h, 0.0, z0, xi0, zeta0, 0.0, 1.0, 0.0,
            3, z_arr, c_arr, c_matrix,
            ssp_r,
        )
        r_new, z_new, xi_new, zeta_new, tau_new, p_new, q_new, _ = full
        frac = 50.0 / r_new
        xi_int = xi0 + frac * (xi_new - xi0)
        zeta_int = zeta0 + frac * (zeta_new - zeta0)
        p_uncorrected = 1.0 + frac * (p_new - 1.0)
        q_int = frac * q_new
        c_int = 1500.0
        dc_old = 0.0
        dc_new = (1510.0 - 1500.0) / 50.0
        grad_jump_r = dc_new - dc_old
        cnjump = grad_jump_r * zeta_int
        csjump = grad_jump_r * xi_int
        rm = zeta_int / xi_int
        rn = rm * (2.0 * cnjump - rm * csjump) / c_int
        expected_p = p_uncorrected - q_int * rn

        traced = trace_beam_ray(
            0.0, z0, alpha,
            h, self.ati_r, self.ati_z, 0, np.array([0.0, 1e9]), np.array([200.0, 200.0]), 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            50.0,
            0,
            300.0,
            3, z_arr, c_arr, c_matrix,
            ssp_r,
            self.max_steps,
        )
        r_path, _, _, _, p_path, q_path, _, _, _, _, n_pts, _ = traced
        interface_idx = int(np.flatnonzero(np.isclose(r_path[:n_pts], 50.0, atol=1.0e-8))[0])

        assert np.isclose(p_path[interface_idx], expected_p)
        assert not np.isclose(p_path[interface_idx], p_uncorrected)
        assert np.isclose(q_path[interface_idx], q_int)

    def test_beam_ray_lands_on_profile_node_before_same_point_reflection(self):
        """Dynamic rays tie-break profile nodes before same-point reflection."""
        bottom_r = np.array([0.0, 50.0, 1e9], dtype=np.float64)
        bottom_z = np.array([100.0, 50.0, 50.0], dtype=np.float64)
        alpha = float(np.degrees(np.arctan2(30.0, 50.0)))

        traced = trace_beam_ray(
            0.0, 20.0, alpha,
            100.0, self.ati_r, self.ati_z, 0, bottom_r, bottom_z, 0,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo, self.empty_geo,
            self.empty_refl, self.empty_refl, self.empty_refl,
            self.empty_refl, self.empty_refl, self.empty_refl,
            *self.surface_props,
            1.8, 1600.0, 0.0, 0.0, 0.0,
            50.0,
            0,
            300.0,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.empty_range,
            self.max_steps,
        )
        r_path, z_path, _, _, _, _, _, _, _, bot_bnc, n_pts, n_bounces = traced

        assert np.allclose(r_path[1:3], [50.0, 50.0], atol=1.0e-8)
        assert np.allclose(z_path[1:3], [50.0, 50.0], atol=1.0e-8)
        assert bot_bnc[1] == 0
        assert bot_bnc[2] == 1
        assert n_bounces >= 1
