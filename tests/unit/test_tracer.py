import pytest
import numpy as np
from pyacoustics.solvers.bellhop.tracer import trace_single_ray

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
        self.h = 1.0       # step size 1m
        self.r_max = 500.0  # trace out to 500m range
        self.max_steps = 10000

    def test_horizontal_ray_no_bounce(self):
        """A horizontal ray at mid-depth should never hit a boundary."""
        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, 0.0,  # r0, z0, angle=0 (horizontal)
            self.h, self.z_surface, self.z_bottom, 
            1.8, 1600.0, 0.0, # bot properties
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.max_steps
        )
        assert n_bounces == 0
        # All z values should be 50m
        assert np.allclose(z_path[:n_pts], 50.0, atol=1e-6)

    def test_downward_ray_hits_bottom(self):
        """A downward ray should hit the bottom and bounce back up."""
        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, 30.0,  # 30 deg downward
            self.h, self.z_surface, self.z_bottom,
            1.8, 1600.0, 0.0, # bot properties
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
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
            self.h, self.z_surface, self.z_bottom,
            1.8, 1600.0, 0.0, # bot properties
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.max_steps
        )
        assert n_bounces >= 1
        valid_z = z_path[:n_pts]
        assert np.all(valid_z >= self.z_surface - 0.01)
        assert np.all(valid_z <= self.z_bottom + 0.01)

    def test_steep_ray_multiple_bounces(self):
        """A steep ray should bounce many times within 500m range."""
        r_path, z_path, amp_path, n_pts, n_bounces = trace_single_ray(
            0.0, 50.0, 60.0,  # 60 deg steep downward
            self.h, self.z_surface, self.z_bottom,
            1.8, 1600.0, 0.0, # bot properties
            self.r_max,
            self.ssp_type, self.z_arr, self.c_arr, self.c_coeffs,
            self.max_steps
        )
        assert n_bounces >= 3  # Should bounce many times
        valid_z = z_path[:n_pts]
        assert np.all(valid_z >= self.z_surface - 0.01)
        assert np.all(valid_z <= self.z_bottom + 0.01)
