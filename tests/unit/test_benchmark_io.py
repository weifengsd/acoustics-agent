import csv
import json

from pyacoustics.solvers.benchmark_io import (
    benchmark_fieldnames,
    write_benchmark_rows_csv,
    write_benchmark_rows_json,
)


def test_benchmark_fieldnames_preserve_common_columns_first():
    rows = (
        {"case": "bellhop-arr", "kind": "arrival", "match_fraction": 1.0, "delay_mae_s": 0.001},
        {"case": "kraken-tl", "tl_mae_db": 0.2, "mode_count_delta": 0, "custom_metric": 7},
    )

    fields = benchmark_fieldnames(rows)

    assert fields[:3] == ("case", "kind", "tl_mae_db")
    assert "match_fraction" in fields
    assert "mode_count_delta" in fields
    assert fields[-1] == "custom_metric"


def test_benchmark_fieldnames_include_timeseries_metrics_before_custom_fields():
    rows = (
        {
            "case": "delay-and-sum",
            "kind": "postprocessing_timeseries",
            "timeseries_mae": 0.1,
            "timeseries_p90_abs": 0.2,
            "timeseries_relative_l2_error": 0.01,
            "postprocessing_time_sample_count": 4,
            "postprocessing_receiver_range_count": 3,
            "postprocessing_receiver_depth_count": 3,
            "postprocessing_arrival_count": 27,
            "postprocessing_ascii_free_format": True,
            "postprocessing_matched_filter_trace_count": 4,
            "postprocessing_matched_filter_peak_lag_max_abs": 0.0,
            "postprocessing_matched_filter_peak_value_max_abs": 0.0,
            "postprocessing_doppler_trace_count": 8,
            "postprocessing_doppler_velocity_bin_count": 3,
            "postprocessing_external_ok": False,
            "postprocessing_external_runtime": "/usr/bin/octave",
            "postprocessing_external_returncode": 0,
            "postprocessing_external_output_path": "MunkB.rts.mat",
            "postprocessing_external_prelude_commands": "scooter MunkS; stack",
            "postprocessing_external_prepared_file_count": 4,
            "postprocessing_external_error": "runtime missing",
            "custom_metric": 7,
        },
    )

    fields = benchmark_fieldnames(rows)

    assert fields[:2] == ("case", "kind")
    assert fields.index("timeseries_mae") < fields.index("timeseries_p90_abs")
    assert fields.index("timeseries_p90_abs") < fields.index("timeseries_relative_l2_error")
    assert fields.index("timeseries_relative_l2_error") < fields.index("postprocessing_time_sample_count")
    assert fields.index("postprocessing_time_sample_count") < fields.index("postprocessing_receiver_range_count")
    assert fields.index("postprocessing_receiver_range_count") < fields.index("postprocessing_receiver_depth_count")
    assert fields.index("postprocessing_receiver_depth_count") < fields.index("postprocessing_arrival_count")
    assert fields.index("postprocessing_arrival_count") < fields.index("postprocessing_ascii_free_format")
    assert fields.index("postprocessing_ascii_free_format") < fields.index("postprocessing_matched_filter_trace_count")
    assert fields.index("postprocessing_matched_filter_trace_count") < fields.index("postprocessing_matched_filter_peak_lag_max_abs")
    assert fields.index("postprocessing_matched_filter_peak_lag_max_abs") < fields.index("postprocessing_matched_filter_peak_value_max_abs")
    assert fields.index("postprocessing_matched_filter_peak_value_max_abs") < fields.index("postprocessing_doppler_trace_count")
    assert fields.index("postprocessing_doppler_trace_count") < fields.index("postprocessing_doppler_velocity_bin_count")
    assert fields.index("postprocessing_doppler_velocity_bin_count") < fields.index("postprocessing_external_ok")
    assert fields.index("postprocessing_external_ok") < fields.index("postprocessing_external_runtime")
    assert fields.index("postprocessing_external_runtime") < fields.index("postprocessing_external_returncode")
    assert fields.index("postprocessing_external_returncode") < fields.index("postprocessing_external_output_path")
    assert fields.index("postprocessing_external_output_path") < fields.index("postprocessing_external_prelude_commands")
    assert fields.index("postprocessing_external_prelude_commands") < fields.index("postprocessing_external_prepared_file_count")
    assert fields.index("postprocessing_external_prepared_file_count") < fields.index("postprocessing_external_error")
    assert fields.index("postprocessing_external_error") < fields.index("custom_metric")


def test_benchmark_fieldnames_include_mat_payload_metrics_before_custom_fields():
    rows = (
        {
            "case": "rts-payload",
            "kind": "postprocessing_rts_mat",
            "timeseries_mae": 0.0,
            "mat_payload_variable_count": 12,
            "mat_payload_variables": "RTS,tout",
            "custom_metric": 7,
        },
    )

    fields = benchmark_fieldnames(rows)

    assert fields.index("timeseries_mae") < fields.index("mat_payload_variable_count")
    assert fields.index("mat_payload_variable_count") < fields.index("mat_payload_variables")
    assert fields.index("mat_payload_variables") < fields.index("custom_metric")


def test_benchmark_fieldnames_include_bellhop3d_active_metrics_before_tl_metrics():
    rows = (
        {
            "case": "bellhop3d",
            "kind": "bellhop3d_tl",
            "sample_count": 10,
            "active_overlap_fraction": 0.8,
            "native_only_active_count": 2,
            "native_weak_tail_count": 3,
            "tl_mae_db": 1.0,
            "custom_metric": 7,
        },
    )

    fields = benchmark_fieldnames(rows)

    assert fields.index("sample_count") < fields.index("active_overlap_fraction")
    assert fields.index("active_overlap_fraction") < fields.index("tl_mae_db")
    assert fields.index("native_only_active_count") < fields.index("tl_mae_db")
    assert fields.index("native_weak_tail_count") < fields.index("tl_mae_db")


def test_benchmark_fieldnames_include_bellhop3d_ray_metrics_before_custom_fields():
    rows = (
        {
            "case": "bellhop3d-ray",
            "kind": "bellhop3d_ray",
            "sample_count": 100,
            "matched_ray_count": 20,
            "depth_p90_abs_m": 2.0,
            "y_p90_abs_m": 0.0,
            "bounce_match_fraction": 0.9,
            "custom_metric": 7,
        },
    )

    fields = benchmark_fieldnames(rows)

    assert fields.index("sample_count") < fields.index("depth_p90_abs_m")
    assert fields.index("depth_p90_abs_m") < fields.index("y_p90_abs_m")
    assert fields.index("y_p90_abs_m") < fields.index("matched_ray_count")
    assert fields.index("matched_ray_count") < fields.index("bounce_match_fraction")
    assert fields.index("bounce_match_fraction") < fields.index("custom_metric")


def test_benchmark_fieldnames_include_bellhop3d_arrival_angle_metrics_before_custom_fields():
    rows = (
        {
            "case": "bellhop3d-arrival",
            "kind": "bellhop3d_arrival",
            "match_fraction": 1.0,
            "launch_declination_mae_deg": 0.01,
            "launch_azimuth_mae_deg": 0.02,
            "receiver_declination_mae_deg": 0.03,
            "receiver_azimuth_mae_deg": 0.04,
            "custom_metric": 7,
        },
    )

    fields = benchmark_fieldnames(rows)

    assert fields.index("match_fraction") < fields.index("launch_declination_mae_deg")
    assert fields.index("launch_declination_mae_deg") < fields.index("launch_azimuth_mae_deg")
    assert fields.index("launch_azimuth_mae_deg") < fields.index("receiver_declination_mae_deg")
    assert fields.index("receiver_declination_mae_deg") < fields.index("receiver_azimuth_mae_deg")
    assert fields.index("receiver_azimuth_mae_deg") < fields.index("custom_metric")


def test_benchmark_fieldnames_include_field3d_runtime_metrics_before_file_formats():
    rows = (
        {
            "case": "field3d-runtime",
            "kind": "field3d_runtime",
            "sample_count": 0,
            "field3d_runtime_ok": False,
            "field3d_case": "lantgbt",
            "field3d_elapsed_s": 30.0,
            "field3d_environment_count": 56,
            "field3d_mod_count": 56,
            "field3d_flp_normalized": False,
            "field3d_shd_exists": False,
            "field3d_shd_size_bytes": 0,
            "field3d_frequency_hz": None,
            "field3d_pressure_sample_count": 0,
            "field3d_pressure_finite_count": 0,
            "field3d_pressure_max_abs": None,
            "field3d_receiver_depth_count": None,
            "field3d_receiver_range_count": None,
            "field3d_receiver_theta_count": None,
            "field3d_source_depth_count": None,
            "field3d_source_x_count": None,
            "field3d_source_y_count": None,
            "field3d_prt_program": None,
            "field3d_prt_completed_without_errors": False,
            "field3d_prt_warning_count": None,
            "field3d_prt_error_count": None,
            "field3d_prt_fatal_error_count": None,
            "field3d_error": "field3d timed out",
            "file_format_roundtrip_ok": True,
            "custom_metric": 7,
        },
    )

    fields = benchmark_fieldnames(rows)

    assert fields.index("field3d_runtime_ok") < fields.index("field3d_case")
    assert fields.index("field3d_case") < fields.index("field3d_elapsed_s")
    assert fields.index("field3d_elapsed_s") < fields.index("field3d_environment_count")
    assert fields.index("field3d_environment_count") < fields.index("field3d_mod_count")
    assert fields.index("field3d_mod_count") < fields.index("field3d_flp_normalized")
    assert fields.index("field3d_flp_normalized") < fields.index("field3d_shd_exists")
    assert fields.index("field3d_shd_exists") < fields.index("field3d_shd_size_bytes")
    assert fields.index("field3d_shd_size_bytes") < fields.index("field3d_frequency_hz")
    assert fields.index("field3d_frequency_hz") < fields.index("field3d_pressure_sample_count")
    assert fields.index("field3d_pressure_sample_count") < fields.index("field3d_pressure_finite_count")
    assert fields.index("field3d_pressure_finite_count") < fields.index("field3d_pressure_max_abs")
    assert fields.index("field3d_pressure_max_abs") < fields.index("field3d_receiver_depth_count")
    assert fields.index("field3d_receiver_depth_count") < fields.index("field3d_receiver_range_count")
    assert fields.index("field3d_receiver_range_count") < fields.index("field3d_receiver_theta_count")
    assert fields.index("field3d_receiver_theta_count") < fields.index("field3d_source_depth_count")
    assert fields.index("field3d_source_depth_count") < fields.index("field3d_source_x_count")
    assert fields.index("field3d_source_x_count") < fields.index("field3d_source_y_count")
    assert fields.index("field3d_source_y_count") < fields.index("field3d_prt_program")
    assert fields.index("field3d_prt_program") < fields.index("field3d_prt_completed_without_errors")
    assert fields.index("field3d_prt_completed_without_errors") < fields.index("field3d_prt_warning_count")
    assert fields.index("field3d_prt_warning_count") < fields.index("field3d_prt_error_count")
    assert fields.index("field3d_prt_error_count") < fields.index("field3d_prt_fatal_error_count")
    assert fields.index("field3d_prt_fatal_error_count") < fields.index("field3d_error")
    assert fields.index("field3d_error") < fields.index("file_format_roundtrip_ok")
    assert fields.index("file_format_roundtrip_ok") < fields.index("custom_metric")


def test_benchmark_fieldnames_include_file_format_metrics_before_custom_fields():
    rows = (
        {
            "case": "field-flp",
            "kind": "at_field_flp_roundtrip",
            "sample_count": 32,
            "file_format_roundtrip_ok": True,
            "file_format_read_ok": True,
            "file_format_path": "tests/3DAtlantic/lant.flp",
            "file_format_scalar_match_count": 21,
            "file_format_array_match_count": 11,
            "env_model_family": "bellhop3d",
            "env_title": "3D Munk profile",
            "env_frequency_hz": 50.0,
            "env_nmedia": 1,
            "env_ssp_option": "C",
            "env_top_option": "V",
            "env_bottom_option": "A",
            "env_run_type": "CG   3",
            "env_source_depth_count": 1,
            "env_receiver_depth_count": 1,
            "env_receiver_range_count": 1001,
            "env_source_x_count": 1,
            "env_source_y_count": 1,
            "env_receiver_bearing_count": 37,
            "env_beam_count": 200,
            "env_beam_bearing_count": 144,
            "env_box_range_km": 99.1,
            "env_box_y_km": 99.1,
            "env_box_z_m": 20500.0,
            "ram_title": "RAM Wedge problem",
            "ram_frequency_hz": 25.0,
            "ram_profile_count": 1,
            "ram_bathymetry_point_count": 2,
            "ram_ssp_point_count": 2,
            "matlab_script_line_count": 12,
            "matlab_script_code_line_count": 8,
            "matlab_script_at_command_count": 3,
            "matlab_script_at_commands": "bellhop:1,plotshd:1,read_shd:1",
            "matlab_script_external_command_count": 2,
            "matlab_script_external_commands": "field:1,kraken:1",
            "fortran_source_family": "Bellhop",
            "fortran_source_line_count": 100,
            "fortran_source_code_line_count": 80,
            "fortran_source_program_count": 1,
            "fortran_source_program_names": "BELLHOP",
            "fortran_source_module_count": 0,
            "fortran_source_subroutine_count": 4,
            "fortran_source_function_count": 1,
            "fortran_source_used_module_count": 7,
            "field_flp_is_3d": True,
            "field_flp_node_count": 397,
            "mod_mode_count": 2,
            "shd_plot_type": "TL",
            "shd_frequency_count": 2,
            "boundary_profile_suffix": ".bty",
            "boundary_point_count": 4,
            "bathymetry_3d_z_count": 4,
            "ssp2d_range_count": 9,
            "ssp2d_depth_row_count": 10,
            "ssp2d_sample_count": 90,
            "ssp2d_finite_count": 90,
            "ssp2d_c_min_mps": 1497.0,
            "ssp2d_c_max_mps": 1545.0,
            "ssp3d_x_count": 63,
            "ssp3d_y_count": 68,
            "ssp3d_z_count": 33,
            "ssp3d_sample_count": 141372,
            "ssp3d_finite_count": 141372,
            "ssp3d_c_min_mps": 1480.0,
            "ssp3d_c_max_mps": 1550.0,
            "reflection_table_suffix": ".brc",
            "reflection_angle_count": 5,
            "source_beam_pattern_sample_count": 37,
            "source_beam_pattern_angle_min_deg": -180.0,
            "source_beam_pattern_angle_max_deg": 180.0,
            "source_beam_pattern_level_min_db": -50.0,
            "source_beam_pattern_level_max_db": 10.0,
            "ray_record_count": 2,
            "ray_coordinate_system": "rz",
            "arr_is_3d": True,
            "arr_arrival_count": 3,
            "prt_program": "BELLHOP",
            "prt_title": "BELLHOP",
            "prt_case_title": "Munk profile",
            "prt_completed_successfully": True,
            "prt_completed_without_errors": True,
            "prt_warning_count": 0,
            "prt_error_count": 0,
            "prt_fatal_error_count": 0,
            "prt_issue_count": 0,
            "prt_parameter_count": 8,
            "prt_cpu_time_s": 0.545,
            "prt_beam_shift_description": "No beam shift in effect",
            "prt_max_arrivals": 10,
            "custom_metric": 7,
        },
    )

    fields = benchmark_fieldnames(rows)

    assert fields.index("file_format_roundtrip_ok") < fields.index("file_format_path")
    assert fields.index("file_format_roundtrip_ok") < fields.index("file_format_read_ok")
    assert fields.index("file_format_read_ok") < fields.index("file_format_path")
    assert fields.index("file_format_path") < fields.index("file_format_scalar_match_count")
    assert fields.index("file_format_array_match_count") < fields.index("env_model_family")
    assert fields.index("env_model_family") < fields.index("env_title")
    assert fields.index("env_title") < fields.index("env_frequency_hz")
    assert fields.index("env_frequency_hz") < fields.index("env_nmedia")
    assert fields.index("env_nmedia") < fields.index("env_ssp_option")
    assert fields.index("env_ssp_option") < fields.index("env_top_option")
    assert fields.index("env_top_option") < fields.index("env_bottom_option")
    assert fields.index("env_bottom_option") < fields.index("env_run_type")
    assert fields.index("env_run_type") < fields.index("env_source_depth_count")
    assert fields.index("env_source_depth_count") < fields.index("env_receiver_depth_count")
    assert fields.index("env_receiver_depth_count") < fields.index("env_receiver_range_count")
    assert fields.index("env_receiver_range_count") < fields.index("env_source_x_count")
    assert fields.index("env_source_x_count") < fields.index("env_source_y_count")
    assert fields.index("env_source_y_count") < fields.index("env_receiver_bearing_count")
    assert fields.index("env_receiver_bearing_count") < fields.index("env_beam_count")
    assert fields.index("env_beam_count") < fields.index("env_beam_bearing_count")
    assert fields.index("env_beam_bearing_count") < fields.index("env_box_range_km")
    assert fields.index("env_box_range_km") < fields.index("env_box_y_km")
    assert fields.index("env_box_y_km") < fields.index("env_box_z_m")
    assert fields.index("env_box_z_m") < fields.index("ram_title")
    assert fields.index("ram_title") < fields.index("ram_frequency_hz")
    assert fields.index("ram_frequency_hz") < fields.index("ram_profile_count")
    assert fields.index("ram_profile_count") < fields.index("ram_bathymetry_point_count")
    assert fields.index("ram_bathymetry_point_count") < fields.index("ram_ssp_point_count")
    assert fields.index("ram_ssp_point_count") < fields.index("matlab_script_line_count")
    assert fields.index("matlab_script_line_count") < fields.index("matlab_script_code_line_count")
    assert fields.index("matlab_script_code_line_count") < fields.index("matlab_script_at_command_count")
    assert fields.index("matlab_script_at_command_count") < fields.index("matlab_script_at_commands")
    assert fields.index("matlab_script_at_commands") < fields.index("matlab_script_external_command_count")
    assert fields.index("matlab_script_external_command_count") < fields.index("matlab_script_external_commands")
    assert fields.index("matlab_script_external_commands") < fields.index("fortran_source_family")
    assert fields.index("fortran_source_family") < fields.index("fortran_source_line_count")
    assert fields.index("fortran_source_line_count") < fields.index("fortran_source_program_count")
    assert fields.index("fortran_source_program_count") < fields.index("fortran_source_program_names")
    assert fields.index("fortran_source_used_module_count") < fields.index("field_flp_is_3d")
    assert fields.index("field_flp_is_3d") < fields.index("field_flp_node_count")
    assert fields.index("field_flp_node_count") < fields.index("mod_mode_count")
    assert fields.index("mod_mode_count") < fields.index("shd_plot_type")
    assert fields.index("shd_plot_type") < fields.index("shd_frequency_count")
    assert fields.index("shd_frequency_count") < fields.index("boundary_profile_suffix")
    assert fields.index("boundary_profile_suffix") < fields.index("boundary_point_count")
    assert fields.index("boundary_point_count") < fields.index("bathymetry_3d_z_count")
    assert fields.index("bathymetry_3d_z_count") < fields.index("ssp2d_range_count")
    assert fields.index("ssp2d_range_count") < fields.index("ssp2d_depth_row_count")
    assert fields.index("ssp2d_depth_row_count") < fields.index("ssp2d_sample_count")
    assert fields.index("ssp2d_sample_count") < fields.index("ssp2d_finite_count")
    assert fields.index("ssp2d_finite_count") < fields.index("ssp2d_c_min_mps")
    assert fields.index("ssp2d_c_min_mps") < fields.index("ssp2d_c_max_mps")
    assert fields.index("ssp2d_c_max_mps") < fields.index("ssp3d_x_count")
    assert fields.index("ssp3d_x_count") < fields.index("ssp3d_y_count")
    assert fields.index("ssp3d_y_count") < fields.index("ssp3d_z_count")
    assert fields.index("ssp3d_z_count") < fields.index("ssp3d_sample_count")
    assert fields.index("ssp3d_sample_count") < fields.index("ssp3d_finite_count")
    assert fields.index("ssp3d_finite_count") < fields.index("ssp3d_c_min_mps")
    assert fields.index("ssp3d_c_min_mps") < fields.index("ssp3d_c_max_mps")
    assert fields.index("ssp3d_c_max_mps") < fields.index("reflection_table_suffix")
    assert fields.index("reflection_table_suffix") < fields.index("reflection_angle_count")
    assert fields.index("reflection_angle_count") < fields.index("source_beam_pattern_sample_count")
    assert fields.index("source_beam_pattern_sample_count") < fields.index("source_beam_pattern_angle_min_deg")
    assert fields.index("source_beam_pattern_angle_min_deg") < fields.index("source_beam_pattern_angle_max_deg")
    assert fields.index("source_beam_pattern_angle_max_deg") < fields.index("source_beam_pattern_level_min_db")
    assert fields.index("source_beam_pattern_level_min_db") < fields.index("source_beam_pattern_level_max_db")
    assert fields.index("source_beam_pattern_level_max_db") < fields.index("ray_record_count")
    assert fields.index("ray_record_count") < fields.index("ray_coordinate_system")
    assert fields.index("ray_coordinate_system") < fields.index("arr_is_3d")
    assert fields.index("arr_is_3d") < fields.index("arr_arrival_count")
    assert fields.index("arr_arrival_count") < fields.index("prt_program")
    assert fields.index("prt_program") < fields.index("prt_title")
    assert fields.index("prt_title") < fields.index("prt_case_title")
    assert fields.index("prt_case_title") < fields.index("prt_completed_successfully")
    assert fields.index("prt_completed_successfully") < fields.index("prt_completed_without_errors")
    assert fields.index("prt_completed_without_errors") < fields.index("prt_warning_count")
    assert fields.index("prt_warning_count") < fields.index("prt_error_count")
    assert fields.index("prt_error_count") < fields.index("prt_fatal_error_count")
    assert fields.index("prt_fatal_error_count") < fields.index("prt_issue_count")
    assert fields.index("prt_issue_count") < fields.index("prt_parameter_count")
    assert fields.index("prt_parameter_count") < fields.index("prt_cpu_time_s")
    assert fields.index("prt_cpu_time_s") < fields.index("prt_beam_shift_description")
    assert fields.index("prt_beam_shift_description") < fields.index("prt_max_arrivals")
    assert fields.index("prt_max_arrivals") < fields.index("custom_metric")


def test_write_benchmark_rows_csv_and_json_roundtrip(tmp_path):
    rows = (
        {
            "case": "bellhop-arr",
            "kind": "arrival",
            "passed": True,
            "failure_count": 0,
            "failures": None,
            "match_fraction": 1.0,
        },
        {
            "case": "kraken-tl",
            "passed": False,
            "failure_count": 1,
            "failures": "tl_mae_db=2 > 1",
            "tl_mae_db": 2.0,
        },
    )

    csv_path = write_benchmark_rows_csv(tmp_path / "reports" / "matrix.csv", rows)
    json_path = write_benchmark_rows_json(tmp_path / "reports" / "matrix.json", rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = json.loads(json_path.read_text(encoding="utf-8"))

    assert csv_rows[0]["case"] == "bellhop-arr"
    assert csv_rows[0]["passed"] == "True"
    assert csv_rows[0]["failures"] == ""
    assert csv_rows[1]["tl_mae_db"] == "2.0"
    assert json_rows == list(rows)


def test_write_benchmark_rows_csv_allows_explicit_fieldnames(tmp_path):
    rows = ({"case": "only-case", "ignored": 1},)

    csv_path = write_benchmark_rows_csv(tmp_path / "matrix.csv", rows, fieldnames=("case",))

    assert csv_path.read_text(encoding="utf-8").splitlines() == ["case", "only-case"]
