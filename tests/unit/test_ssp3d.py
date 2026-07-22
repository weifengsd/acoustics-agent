import numpy as np
import pytest

from pyacoustics.environment import evaluate_hexahedral_ssp_3d
from pyacoustics.solvers.bellhop3d.boundary import reduce_step_fraction_3d
from pyacoustics.solvers.external_io import read_at_ssp2d, read_at_ssp3d


def test_read_at_ssp3d_taiwan_shape_and_units():
    x, y, z, c = read_at_ssp3d("tests/Bellhop3DTests/Taiwan/Taiwan.ssp")

    assert x.shape == (63,)
    assert y.shape == (68,)
    assert z.shape == (33,)
    assert c.shape == (63, 68, 33)
    assert x[1] > 8000.0
    assert y[1] > 8000.0
    assert z[-1] == 5500.0
    assert np.isclose(c[0, 0, 0], 1524.692139)


def test_read_at_ssp3d_accepts_comments_slashes_and_d_exponents(tmp_path):
    ssp_path = tmp_path / "mini.ssp"
    ssp_path.write_text(
        "2 /              ! nx\n"
        "0.0D+00 1.0D+00 / ! x km\n"
        "2 /              ! ny\n"
        "0.0D+00 2.0D+00 / ! y km\n"
        "2 /              ! nz\n"
        "0.0D+00 1.0D+02 / ! z m\n"
        "1500.0D+00 1501.0D+00 / ! z0 y0 x sweep\n"
        "1502.0D+00 1503.0D+00 /\n"
        "1510.0D+00 1511.0D+00 /\n"
        "1512.0D+00 1513.0D+00 /\n"
    )

    x, y, z, c = read_at_ssp3d(ssp_path)

    np.testing.assert_allclose(x, [0.0, 1000.0])
    np.testing.assert_allclose(y, [0.0, 2000.0])
    np.testing.assert_allclose(z, [0.0, 100.0])
    assert c.shape == (2, 2, 2)
    np.testing.assert_allclose(c[:, 0, 0], [1500.0, 1501.0])
    np.testing.assert_allclose(c[:, 1, 0], [1502.0, 1503.0])
    np.testing.assert_allclose(c[:, 0, 1], [1510.0, 1511.0])
    np.testing.assert_allclose(c[:, 1, 1], [1512.0, 1513.0])


def test_read_at_ssp2d_range_dependent_sidecar(tmp_path):
    ssp_path = tmp_path / "range_dependent.ssp"
    ssp_path.write_text(
        "3\n"
        "0.0 12.5 25.0 /\n"
        "1536 1536 1536\n"
        "1506 1508.75 1511.5\n"
        "1545 1545 1545\n",
        encoding="utf-8",
    )

    ssp = read_at_ssp2d(ssp_path)

    assert ssp.ranges_m.shape == (3,)
    assert ssp.c_matrix.shape == (3, 3)
    np.testing.assert_allclose(ssp.ranges_m, [0.0, 12500.0, 25000.0])
    assert np.isclose(ssp.c_matrix[0, 0], 1536.0)
    assert np.isclose(ssp.c_matrix[-1, -1], 1545.0)


def test_read_at_ssp2d_rejects_inconsistent_rows(tmp_path):
    ssp_path = tmp_path / "bad.ssp"
    ssp_path.write_text(
        "3\n"
        "0.0 1.0 2.0 /\n"
        "1500.0 1501.0 /\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected 3 sound-speed values"):
        read_at_ssp2d(ssp_path)


def test_evaluate_hexahedral_ssp_3d_trilinear_gradient():
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 20.0])
    z = np.array([0.0, 5.0])
    c = np.zeros((2, 2, 2))
    for ix, xv in enumerate(x):
        for iy, yv in enumerate(y):
            for iz, zv in enumerate(z):
                c[ix, iy, iz] = 1500.0 + 2.0 * xv + 3.0 * yv + 4.0 * zv

    speed, dc_dx, dc_dy, dc_dz, d2c_dz2 = evaluate_hexahedral_ssp_3d(2.5, 5.0, 1.0, x, y, z, c)

    assert np.isclose(speed, 1524.0)
    assert np.isclose(dc_dx, 2.0)
    assert np.isclose(dc_dy, 3.0)
    assert np.isclose(dc_dz, 4.0)
    assert d2c_dz2 == 0.0


def test_reduce_step_fraction_hits_grid_segment():
    grid = np.array([0.0, 10.0, 20.0])
    z_grid = np.array([0.0, 100.0, 200.0])
    frac = reduce_step_fraction_3d(
        6.0, 5.0, 50.0,
        1.0, 0.0, 0.0,
        10.0,
        grid, grid,
        grid, grid, z_grid,
        False,
    )

    assert np.isclose(frac, 0.4)


def test_reduce_step_fraction_hits_bty_diagonal():
    grid = np.array([0.0, 10.0])
    z_grid = np.array([0.0, 100.0])
    frac = reduce_step_fraction_3d(
        2.0, 1.0, 50.0,
        0.0, 1.0, 0.0,
        10.0,
        grid, grid,
        grid, grid, z_grid,
        False,
    )

    assert np.isclose(frac, 0.1)
