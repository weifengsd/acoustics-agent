from pyacoustics.benchmarks.file_formats import (
    at_file_format_smoke_thresholds,
    run_at_file_format_benchmark_matrix,
)


LOCAL_FORMAT_KINDS = {
    "at_arr_roundtrip",
    "at_bathymetry_3d_roundtrip",
    "at_boundary_profile_roundtrip",
    "at_env_summary_parse",
    "at_field_flp_roundtrip",
    "at_mod_roundtrip",
    "at_prt_parse",
    "at_ray_roundtrip",
    "at_reflection_table_roundtrip",
    "at_shd_roundtrip",
    "at_source_beam_pattern_roundtrip",
    "at_ssp3d_read",
}


def test_local_file_format_matrix_is_self_contained_and_green():
    matrix = run_at_file_format_benchmark_matrix(
        thresholds=at_file_format_smoke_thresholds(),
    )

    assert len(matrix.rows) >= 40
    assert {row["kind"] for row in matrix.rows} == LOCAL_FORMAT_KINDS
    assert all(row["passed"] for row in matrix.rows)
    assert all(row["file_format_roundtrip_ok"] for row in matrix.rows)


def test_relative_sample_root_matches_default_checkout():
    thresholds = at_file_format_smoke_thresholds()
    default_matrix = run_at_file_format_benchmark_matrix(thresholds=thresholds)
    relative_matrix = run_at_file_format_benchmark_matrix(
        sample_root=".",
        thresholds=thresholds,
    )

    assert [row["case"] for row in relative_matrix.rows] == [
        row["case"] for row in default_matrix.rows
    ]


def test_missing_generated_solver_outputs_are_not_required_fixtures():
    matrix = run_at_file_format_benchmark_matrix(
        thresholds=at_file_format_smoke_thresholds(),
    )
    cases = {row["case"] for row in matrix.rows}

    assert "shd-roundtrip-real-tests-Munk-munk_at" not in cases
    assert "prt-parse-real-tests-Munk-munk_at" not in cases
    assert "bathymetry-3d-roundtrip-real-tests-Bellhop3DTests-scratch-temp_plot" not in cases
