import numpy as np

from pyacoustics.solvers.bellhop3d.tl_parity import (
    bellhop3d_tl_benchmark_row,
    compare_tl_grids_3d,
)


def test_compare_bellhop3d_tl_grids_reports_active_grid_metrics():
    at_tl = np.array([[[60.0, 62.0], [200.0, 64.0]]])
    native_tl = np.array([[[61.0, 61.5], [80.0, 66.0]]])

    metrics = compare_tl_grids_3d(at_tl, native_tl)
    row = bellhop3d_tl_benchmark_row(
        "bellhop3d-smoke",
        metrics,
        min_range_m=500.0,
        active_tl_max_db=110.0,
    )

    assert metrics.sample_count == 3
    assert metrics.at_active_count == 3
    assert metrics.native_active_count == 4
    assert metrics.at_only_active_count == 0
    assert metrics.native_only_active_count == 1
    assert metrics.native_weak_tail_count == 0
    assert np.isclose(metrics.active_overlap_fraction, 3.0 / 4.0)
    assert np.isclose(metrics.mae_db, (1.0 + 0.5 + 2.0) / 3.0)
    assert np.isclose(metrics.bias_db, (1.0 - 0.5 + 2.0) / 3.0)
    assert row["case"] == "bellhop3d-smoke"
    assert row["kind"] == "bellhop3d_tl"
    assert row["sample_count"] == 3
    assert row["at_active_count"] == 3
    assert row["native_active_count"] == 4
    assert row["at_only_active_count"] == 0
    assert row["native_only_active_count"] == 1
    assert row["native_weak_tail_count"] == 0
    assert np.isclose(row["active_overlap_fraction"], 3.0 / 4.0)
    assert row["min_range_m"] == 500.0
    assert row["active_tl_max_db"] == 110.0


def test_compare_bellhop3d_tl_grids_supports_range_and_shadow_mask():
    at_tl = np.array([[[55.0, 58.0, 120.0], [60.0, 62.0, 64.0]]])
    native_tl = np.array([[[90.0, 57.0, 121.0], [59.0, 61.0, 65.0]]])
    ranges = np.array([0.0, 1000.0, 2000.0])

    metrics = compare_tl_grids_3d(
        at_tl,
        native_tl,
        r_grid=ranges,
        min_range_m=500.0,
        active_tl_max_db=110.0,
    )

    assert metrics.sample_count == 3
    assert metrics.at_active_count == 3
    assert metrics.native_active_count == 3
    assert metrics.at_only_active_count == 0
    assert metrics.native_only_active_count == 0
    assert metrics.native_weak_tail_count == 1
    assert np.isclose(metrics.active_overlap_fraction, 1.0)
    assert np.isclose(metrics.mae_db, 1.0)
    assert np.isclose(metrics.max_abs_db, 1.0)
