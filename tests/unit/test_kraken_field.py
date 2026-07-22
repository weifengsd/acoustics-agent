from copy import deepcopy

import numpy as np

from pyacoustics.benchmarks.at_presets import _rigid_pekeris_kraken_config
from pyacoustics.solvers.kraken import PyKraken, _field_axis_values
from pyacoustics.solvers.kraken.field import compute_field

def test_compute_field_shapes_and_values():
    M = 2
    N = 100
    R = 50
    Z = 20
    
    modes = np.random.rand(M, N)
    roots = np.array([0.1, 0.05])
    z_arr = np.linspace(0, 100, N)
    r_arr = np.linspace(10, 1000, R)
    z_s = 50.0
    z_r_arr = np.linspace(10, 90, Z)
    
    TL = compute_field(modes, roots, z_arr, r_arr, z_s, z_r_arr)
    
    assert TL.shape == (Z, R)
    # TL should generally be positive for standard scaling
    assert np.all(TL > -50.0) 
    
    # At r=0, we forced a small value, but let's test r=0 behavior
    r_arr_zero = np.array([0.0, 10.0])
    TL_zero = compute_field(modes, roots, z_arr, r_arr_zero, z_s, z_r_arr)
    assert not np.any(np.isnan(TL_zero))
    assert not np.any(np.isinf(TL_zero))


def test_compute_field_ignores_source_density_for_at_evaluate_mod_scaling():
    modes = np.ones((1, 8), dtype=np.float64)
    roots = np.array([0.1])
    z_arr = np.linspace(0, 70, 8)
    r_arr = np.array([100.0])
    z_r_arr = np.array([30.0])

    tl_rho1 = compute_field(modes, roots, z_arr, r_arr, 20.0, z_r_arr, rho_s=1.0)
    tl_rho2 = compute_field(modes, roots, z_arr, r_arr, 20.0, z_r_arr, rho_s=2.0)

    np.testing.assert_allclose(tl_rho2, tl_rho1)


def test_field_axis_values_match_at_endpoint_grid_semantics():
    np.testing.assert_allclose(_field_axis_values([42.0]), [42.0])

    expanded = _field_axis_values([10.0, 20.0], interpolated_count=5)
    np.testing.assert_allclose(expanded, [10.0, 12.5, 15.0, 17.5, 20.0])

    explicit = _field_axis_values([10.0, 20.0, 40.0])
    np.testing.assert_allclose(explicit, [10.0, 20.0, 40.0])

    range_axis = _field_axis_values([0.0, 10.0], start_floor=1.0, interpolated_count=4)
    np.testing.assert_allclose(range_axis, [1.0, 4.0, 7.0, 10.0])


def test_pykraken_max_modes_limits_field_synthesis_without_truncating_diagnostics():
    base_config = _rigid_pekeris_kraken_config()
    full_result = PyKraken(deepcopy(base_config)).run_normal_modes()

    truncated_config = deepcopy(base_config)
    truncated_config.solver.max_modes = 3
    truncated_result = PyKraken(truncated_config).run_normal_modes()

    assert len(full_result["roots"]) == len(truncated_result["roots"]) == 6
    assert full_result["tl_grid"].shape == truncated_result["tl_grid"].shape
    assert np.max(np.abs(full_result["tl_grid"] - truncated_result["tl_grid"])) > 1.0


def test_pykraken_field_synthesis_keeps_multi_source_depth_axis():
    config = _rigid_pekeris_kraken_config()
    config.geometry.source.depths = [40.0, 60.0]

    result = PyKraken(config).run_normal_modes()

    assert result["tl_grid"].shape == (2, len(result["z_bins"]), len(result["r_bins"]))
    assert np.isfinite(result["tl_grid"]).all()
    assert not np.allclose(result["tl_grid"][0], result["tl_grid"][1])
