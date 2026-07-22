import numpy as np

from pyacoustics.benchmarks import (
    bellhop3d_smoke_thresholds,
    bellhop_smoke_thresholds,
    canonical_bellhop3d_arrival_benchmark_cases,
    canonical_bellhop3d_benchmark_cases,
    canonical_bellhop3d_ray_benchmark_cases,
    canonical_bellhop_benchmark_cases,
    canonical_kraken_benchmark_cases,
    kraken_smoke_thresholds,
)


def test_canonical_bellhop_benchmark_cases_cover_core_parity_kinds():
    cases = canonical_bellhop_benchmark_cases()
    names = {case.name for case in cases}
    kinds = {case.kind for case in cases}

    assert len(cases) == 101
    assert kinds == {"tl", "arrival", "ray"}
    assert "iso-direct-arrivals" in names
    assert "iso-direct-line-source-arrivals" in names
    assert "iso-direct-line-source-coherent-tl" in names
    assert "iso-direct-line-source-incoherent-tl" in names
    assert "iso-direct-multi-source-depth-arrivals" in names
    assert "iso-direct-multi-source-depth-coherent-tl" in names
    assert "iso-direct-multi-source-depth-incoherent-tl" in names
    assert "iso-direct-multi-source-depth-semicoherent-tl" in names
    assert "iso-direct-irregular-receiver-arrivals" in names
    assert "iso-multipath-eigenray-arrivals" in names
    assert "iso-multipath-rays" in names
    assert "iso-direct-geometric-gaussian-tl" in names
    assert "iso-direct-geometric-gaussian-incoherent-tl" in names
    assert "iso-direct-geometric-gaussian-semicoherent-tl" in names
    assert "iso-direct-simple-gaussian-tl" in names
    assert "iso-direct-ray-centered-hat-tl" in names
    assert "iso-direct-ray-centered-hat-incoherent-tl" in names
    assert "sloping-bathymetry-ray-centered-hat-tl" in names
    assert "sloping-bathymetry-ray-centered-hat-semicoherent-tl" in names
    assert "bottom-reflection-table-ray-centered-hat-semicoherent-tl" in names
    assert "range-dependent-ssp-ray-centered-hat-tl" in names
    assert "range-dependent-ssp-ray-centered-hat-incoherent-tl" in names
    assert "range-dependent-ssp-ray-centered-hat-semicoherent-tl" in names
    assert "range-dependent-ssp-bottom-reflection-table-ray-centered-hat-tl" in names
    assert "range-dependent-ssp-bottom-reflection-table-ray-centered-hat-incoherent-tl" in names
    assert "range-dependent-ssp-bottom-reflection-table-ray-centered-hat-semicoherent-tl" in names
    assert "range-dependent-ssp-sloping-bathymetry-ray-centered-hat-tl" in names
    assert "range-dependent-ssp-sloping-bathymetry-ray-centered-hat-semicoherent-tl" in names
    assert "iso-direct-coherent-tl" in names
    assert "iso-direct-incoherent-tl" in names
    assert "munk-beam-arrivals" in names
    assert "munk-gaussian-beam-arrivals" in names
    assert "munk-eigenray-rays" in names
    assert "iso-multipath-eigenray-rays" in names
    assert "shallow-reflected-eigenray-rays" in names
    assert "elastic-bottom-eigenray-rays" in names
    assert "bottom-reflection-table-eigenray-rays" in names
    assert "sloping-bathymetry-eigenray-rays" in names
    assert "sloping-altimetry-eigenray-rays" in names
    assert "long-format-bathymetry-eigenray-rays" in names
    assert "curvilinear-boundary-eigenray-rays" in names
    assert "range-dependent-ssp-eigenray-rays" in names
    assert "range-dependent-ssp-bottom-reflection-table-eigenray-rays" in names
    assert "range-dependent-ssp-sloping-bathymetry-eigenray-rays" in names
    assert "shallow-reflected-eigenray-arrivals" in names
    assert "shallow-reflected-coherent-tl" in names
    assert "shallow-reflected-incoherent-tl" in names
    assert "shallow-reflected-semicoherent-tl" in names
    assert "shallow-reflected-multi-source-depth-arrivals" in names
    assert "shallow-reflected-multi-source-depth-coherent-tl" in names
    assert "shallow-reflected-multi-source-depth-incoherent-tl" in names
    assert "shallow-reflected-multi-source-depth-semicoherent-tl" in names
    assert "elastic-bottom-rays" in names
    assert "elastic-bottom-arrivals" in names
    assert "elastic-bottom-coherent-tl" in names
    assert "elastic-bottom-incoherent-tl" in names
    assert "elastic-bottom-semicoherent-tl" in names
    assert "elastic-bottom-multi-source-depth-arrivals" in names
    assert "elastic-bottom-multi-source-depth-coherent-tl" in names
    assert "elastic-bottom-multi-source-depth-incoherent-tl" in names
    assert "elastic-bottom-multi-source-depth-semicoherent-tl" in names
    assert "sloping-altimetry-rays" in names
    assert "sloping-altimetry-coherent-tl" in names
    assert "sloping-altimetry-incoherent-tl" in names
    assert "sloping-altimetry-semicoherent-tl" in names
    assert "long-format-bathymetry-rays" in names
    assert "long-format-bathymetry-coherent-tl" in names
    assert "long-format-bathymetry-incoherent-tl" in names
    assert "long-format-bathymetry-semicoherent-tl" in names
    assert "sloping-bathymetry-coherent-tl" in names
    assert "sloping-bathymetry-incoherent-tl" in names
    assert "sloping-bathymetry-semicoherent-tl" in names
    assert "sloping-bathymetry-seongil-shift-tl" in names
    assert "sloping-bathymetry-seongil-shift-incoherent-tl" in names
    assert "sloping-bathymetry-seongil-shift-semicoherent-tl" in names
    assert "bottom-reflection-table-arrivals" in names
    assert "bottom-reflection-table-rays" in names
    assert "bottom-reflection-table-coherent-tl" in names
    assert "bottom-reflection-table-incoherent-tl" in names
    assert "bottom-reflection-table-semicoherent-tl" in names
    assert "curvilinear-boundary-rays" in names
    assert "range-dependent-ssp-rays" in names
    assert "range-dependent-ssp-coherent-tl" in names
    assert "range-dependent-ssp-incoherent-tl" in names
    assert "range-dependent-ssp-semicoherent-tl" in names
    assert "range-dependent-ssp-multi-source-depth-arrivals" in names
    assert "range-dependent-ssp-multi-source-depth-coherent-tl" in names
    assert "range-dependent-ssp-multi-source-depth-incoherent-tl" in names
    assert "range-dependent-ssp-multi-source-depth-semicoherent-tl" in names
    assert "range-dependent-ssp-bottom-reflection-table-coherent-tl" in names
    assert "range-dependent-ssp-bottom-reflection-table-incoherent-tl" in names
    assert "range-dependent-ssp-bottom-reflection-table-semicoherent-tl" in names
    assert "range-dependent-ssp-sloping-bathymetry-coherent-tl" in names
    assert "range-dependent-ssp-sloping-bathymetry-incoherent-tl" in names
    assert "range-dependent-ssp-sloping-bathymetry-semicoherent-tl" in names

    eigenray_case = next(case for case in cases if case.name == "shallow-reflected-eigenray-arrivals")
    assert eigenray_case.options["method"] == "eigenray"
    assert eigenray_case.options["eigenray_angle_samples"] == 161

    line_source_case = next(case for case in cases if case.name == "iso-direct-line-source-arrivals")
    assert line_source_case.config.solver.source_type == "line"
    assert line_source_case.options["method"] == "beam"

    line_source_tl_cases = [
        case for case in cases if case.name.startswith("iso-direct-line-source-") and case.kind == "tl"
    ]
    assert {case.run_type for case in line_source_tl_cases} == {"C", "I"}
    for line_tl_case in line_source_tl_cases:
        assert line_tl_case.config.solver.source_type == "line"
        assert np.allclose(line_tl_case.r_grid, [1000.0])
        assert np.allclose(line_tl_case.z_grid, [100.0])

    multi_source_case = next(case for case in cases if case.name == "iso-direct-multi-source-depth-arrivals")
    assert multi_source_case.options["method"] == "beam"
    assert multi_source_case.config.geometry.source.depths == [80.0, 120.0]
    assert multi_source_case.config.geometry.receivers.depths == [100.0]
    multi_source_tl_cases = [
        case for case in cases if case.name.startswith("iso-direct-multi-source-depth-") and case.kind == "tl"
    ]
    assert {case.run_type for case in multi_source_tl_cases} == {"C", "I", "S"}
    for multi_source_tl_case in multi_source_tl_cases:
        assert multi_source_tl_case.config.geometry.source.depths == [80.0, 120.0]
        assert np.allclose(multi_source_tl_case.r_grid, [1000.0])
        assert np.allclose(multi_source_tl_case.z_grid, [100.0])

    irregular_case = next(case for case in cases if case.name == "iso-direct-irregular-receiver-arrivals")
    assert irregular_case.config.solver.receiver_grid == "irregular"
    assert irregular_case.config.geometry.receivers.ranges == [1000.0, 1500.0]
    assert irregular_case.config.geometry.receivers.depths == [100.0, 120.0]

    multipath_case = next(case for case in cases if case.name == "iso-multipath-eigenray-arrivals")
    assert multipath_case.options["method"] == "eigenray"
    assert multipath_case.options["eigenray_angle_samples"] == 281
    assert multipath_case.options["max_arrivals"] == 80
    assert multipath_case.config.solver.angles == (-35.0, 35.0)
    assert multipath_case.config.solver.num_beams == 281

    multipath_ray_case = next(case for case in cases if case.name == "iso-multipath-rays")
    assert multipath_ray_case.kind == "ray"
    assert multipath_ray_case.options["sample_count"] == 101
    assert multipath_ray_case.config.solver.angles == (-35.0, 35.0)
    assert multipath_ray_case.config.solver.num_beams == 281

    munk_arrival_case = next(case for case in cases if case.name == "munk-beam-arrivals")
    assert munk_arrival_case.options["method"] == "beam"
    assert munk_arrival_case.options["delay_tolerance"] == 0.03
    assert munk_arrival_case.thresholds is not None
    assert munk_arrival_case.config.solver.step_size == 5.0
    assert munk_arrival_case.thresholds.max_values["delay_mae_s"] == 0.0001
    assert munk_arrival_case.thresholds.max_values["amplitude_median_relative_error"] == 0.01
    assert munk_arrival_case.thresholds.max_values["phase_mae_rad"] == 1.0e-6
    assert munk_arrival_case.thresholds.max_values["receiver_angle_mae_deg"] == 0.01
    assert munk_arrival_case.thresholds.min_values["match_fraction"] == 1.0

    munk_gaussian_case = next(case for case in cases if case.name == "munk-gaussian-beam-arrivals")
    assert munk_gaussian_case.options["method"] == "beam"
    assert munk_gaussian_case.options["beam_shape"] == "gaussian"
    assert munk_gaussian_case.options["run_type"] == "AB   3"
    assert munk_gaussian_case.config.solver.step_size == 5.0
    assert munk_gaussian_case.thresholds is not None
    assert munk_gaussian_case.thresholds.max_values["amplitude_median_relative_error"] == 0.002
    assert munk_gaussian_case.thresholds.max_values["launch_angle_mae_deg"] == 0.01
    assert munk_gaussian_case.thresholds.max_values["receiver_angle_mae_deg"] == 0.01
    assert munk_gaussian_case.thresholds.min_values["match_fraction"] == 1.0

    munk_e_case = next(case for case in cases if case.name == "munk-eigenray-rays")
    assert munk_e_case.run_type == "E"
    assert munk_e_case.options["max_range_m"] == 2000.0
    assert munk_e_case.config.solver.num_beams == 101

    complex_e_ray_cases = [
        case
        for case in cases
        if case.name
        in {
            "iso-multipath-eigenray-rays",
            "shallow-reflected-eigenray-rays",
            "elastic-bottom-eigenray-rays",
            "bottom-reflection-table-eigenray-rays",
            "sloping-bathymetry-eigenray-rays",
            "sloping-altimetry-eigenray-rays",
            "long-format-bathymetry-eigenray-rays",
            "curvilinear-boundary-eigenray-rays",
            "range-dependent-ssp-eigenray-rays",
            "range-dependent-ssp-bottom-reflection-table-eigenray-rays",
            "range-dependent-ssp-sloping-bathymetry-eigenray-rays",
        }
    ]
    assert len(complex_e_ray_cases) == 11
    for complex_e_ray_case in complex_e_ray_cases:
        assert complex_e_ray_case.kind == "ray"
        assert complex_e_ray_case.run_type == "E"
        assert complex_e_ray_case.options["max_range_m"] == 1000.0
        assert complex_e_ray_case.options["sample_count"] == 101

    gaussian_case = next(case for case in cases if case.name == "iso-direct-geometric-gaussian-tl")
    assert gaussian_case.config.solver.beam_type == "geometric_gaussian"
    assert gaussian_case.run_type == "C"
    gaussian_intensity_cases = [
        case for case in cases if case.name.startswith("iso-direct-geometric-gaussian-") and case.run_type in {"I", "S"}
    ]
    assert {case.run_type for case in gaussian_intensity_cases} == {"I", "S"}
    for gaussian_intensity_case in gaussian_intensity_cases:
        assert gaussian_intensity_case.config.solver.beam_type == "geometric_gaussian"
        assert np.allclose(gaussian_intensity_case.r_grid, [1000.0])
        assert np.allclose(gaussian_intensity_case.z_grid, [100.0])

    sgb_case = next(case for case in cases if case.name == "iso-direct-simple-gaussian-tl")
    assert sgb_case.config.solver.beam_type == "simple_gaussian"
    assert sgb_case.run_type == "C"

    ray_centered_case = next(case for case in cases if case.name == "iso-direct-ray-centered-hat-tl")
    assert ray_centered_case.config.solver.beam_type == "geometric_hat_ray_centered"
    assert ray_centered_case.run_type == "C"
    assert np.allclose(ray_centered_case.r_grid, [500.0, 1000.0])
    ray_centered_i_case = next(case for case in cases if case.name == "iso-direct-ray-centered-hat-incoherent-tl")
    assert ray_centered_i_case.config.solver.beam_type == "geometric_hat_ray_centered"
    assert ray_centered_i_case.run_type == "I"
    assert np.allclose(ray_centered_i_case.r_grid, [500.0, 1000.0])
    assert np.allclose(ray_centered_i_case.z_grid, [100.0])
    complex_ray_centered_cases = [
        case
        for case in cases
        if "ray-centered-hat" in case.name and not case.name.startswith("iso-direct")
    ]
    assert len(complex_ray_centered_cases) == 11
    for complex_ray_centered_case in complex_ray_centered_cases:
        assert complex_ray_centered_case.kind == "tl"
        assert complex_ray_centered_case.config.solver.beam_type == "geometric_hat_ray_centered"
        assert np.asarray(complex_ray_centered_case.r_grid).size >= 2

    iso_direct_tl_cases = [
        case for case in cases if case.name in {"iso-direct-coherent-tl", "iso-direct-incoherent-tl"}
    ]
    assert {case.run_type for case in iso_direct_tl_cases} == {"C", "I"}
    for iso_direct_tl_case in iso_direct_tl_cases:
        assert np.allclose(iso_direct_tl_case.r_grid, [1000.0])
        assert np.allclose(iso_direct_tl_case.z_grid, [100.0])

    shallow_tl_cases = [case for case in cases if case.name.startswith("shallow-reflected-") and case.kind == "tl"]
    assert {case.run_type for case in shallow_tl_cases} == {"C", "I", "S"}
    for shallow_tl_case in shallow_tl_cases:
        assert np.allclose(shallow_tl_case.r_grid, [1000.0])
        assert np.allclose(shallow_tl_case.z_grid, [65.0])

    shallow_multi_source_case = next(
        case for case in cases if case.name == "shallow-reflected-multi-source-depth-arrivals"
    )
    assert shallow_multi_source_case.options["method"] == "beam"
    assert shallow_multi_source_case.options["max_arrivals"] == 80
    assert shallow_multi_source_case.config.geometry.source.depths == [30.0, 40.0]
    assert shallow_multi_source_case.config.solver.num_beams == 201
    shallow_multi_source_tl_cases = [
        case
        for case in cases
        if case.name.startswith("shallow-reflected-multi-source-depth-") and case.kind == "tl"
    ]
    assert {case.run_type for case in shallow_multi_source_tl_cases} == {"C", "I", "S"}
    for shallow_multi_source_tl_case in shallow_multi_source_tl_cases:
        assert shallow_multi_source_tl_case.config.geometry.source.depths == [30.0, 40.0]
        assert np.allclose(shallow_multi_source_tl_case.r_grid, [1000.0])
        assert np.allclose(shallow_multi_source_tl_case.z_grid, [65.0])

    tl_cases = {
        case.name: case
        for case in cases
        if case.name
        in {
            "sloping-bathymetry-coherent-tl",
            "sloping-bathymetry-incoherent-tl",
            "sloping-bathymetry-semicoherent-tl",
        }
    }
    assert {case.run_type for case in tl_cases.values()} == {"C", "I", "S"}
    for tl_case in tl_cases.values():
        assert np.allclose(tl_case.r_grid, [1000.0])
        assert np.allclose(tl_case.z_grid, [65.0])
        assert tl_case.config.solver.type == "bellhop"

    seongil_case = next(case for case in cases if case.name == "sloping-bathymetry-seongil-shift-tl")
    assert seongil_case.config.solver.beam_type == "geometric_hat"
    assert seongil_case.config.solver.beam_displacement == "seongil"
    assert seongil_case.run_type == "C"
    seongil_intensity_cases = [
        case
        for case in cases
        if case.name.startswith("sloping-bathymetry-seongil-shift-") and case.run_type in {"I", "S"}
    ]
    assert {case.run_type for case in seongil_intensity_cases} == {"I", "S"}
    for seongil_intensity_case in seongil_intensity_cases:
        assert seongil_intensity_case.config.solver.beam_type == "geometric_hat"
        assert seongil_intensity_case.config.solver.beam_displacement == "seongil"
        assert np.allclose(seongil_intensity_case.r_grid, [1000.0])
        assert np.allclose(seongil_intensity_case.z_grid, [65.0])

    reflection_case = next(case for case in cases if case.name == "bottom-reflection-table-arrivals")
    assert reflection_case.config.environment.bottom.reflection_table is not None
    assert len(reflection_case.config.environment.bottom.reflection_table) == 3
    assert reflection_case.options["max_arrivals"] == 80
    assert reflection_case.options["method"] == "beam"

    reflection_ray_case = next(case for case in cases if case.name == "bottom-reflection-table-rays")
    assert reflection_ray_case.config.environment.bottom.reflection_table is not None
    assert reflection_ray_case.kind == "ray"
    assert reflection_ray_case.options["sample_count"] == 101

    elastic_case = next(case for case in cases if case.name == "elastic-bottom-rays")
    assert elastic_case.config.environment.bottom.type == "acousto-elastic"
    assert elastic_case.config.environment.bottom.c_p == 1700.0
    assert elastic_case.config.environment.bottom.c_s == 700.0
    assert elastic_case.config.environment.bottom.density == 1.8
    assert elastic_case.kind == "ray"
    assert elastic_case.options["max_range_m"] == 1000.0
    assert elastic_case.options["sample_count"] == 101
    elastic_arrival_case = next(case for case in cases if case.name == "elastic-bottom-arrivals")
    assert elastic_arrival_case.kind == "arrival"
    assert elastic_arrival_case.options["method"] == "beam"
    assert elastic_arrival_case.options["delay_tolerance"] == 0.03
    elastic_tl_case = next(case for case in cases if case.name == "elastic-bottom-coherent-tl")
    assert elastic_tl_case.run_type == "C"
    assert np.allclose(elastic_tl_case.r_grid, [1000.0])
    assert np.allclose(elastic_tl_case.z_grid, [65.0])
    elastic_intensity_cases = [
        case for case in cases if case.name.startswith("elastic-bottom-") and case.run_type in {"I", "S"}
    ]
    assert {case.run_type for case in elastic_intensity_cases} == {"I", "S"}
    for elastic_intensity_case in elastic_intensity_cases:
        assert np.allclose(elastic_intensity_case.r_grid, [600.0, 1000.0, 1400.0])
        assert np.allclose(elastic_intensity_case.z_grid, [45.0, 65.0, 85.0])

    elastic_multi_source_case = next(case for case in cases if case.name == "elastic-bottom-multi-source-depth-arrivals")
    assert elastic_multi_source_case.options["method"] == "beam"
    assert elastic_multi_source_case.options["max_arrivals"] == 100
    assert elastic_multi_source_case.config.geometry.source.depths == [30.0, 40.0]
    assert elastic_multi_source_case.config.environment.bottom.type == "acousto-elastic"
    elastic_multi_source_tl_cases = [
        case for case in cases if case.name.startswith("elastic-bottom-multi-source-depth-") and case.kind == "tl"
    ]
    assert {case.run_type for case in elastic_multi_source_tl_cases} == {"C", "I", "S"}
    for elastic_multi_source_tl_case in elastic_multi_source_tl_cases:
        assert elastic_multi_source_tl_case.config.geometry.source.depths == [30.0, 40.0]
        assert np.allclose(elastic_multi_source_tl_case.r_grid, [600.0, 1000.0, 1400.0])
        assert np.allclose(elastic_multi_source_tl_case.z_grid, [45.0, 65.0, 85.0])

    sloping_altimetry_case = next(case for case in cases if case.name == "sloping-altimetry-rays")
    assert sloping_altimetry_case.config.environment.surface.altimetry == [
        (0.0, 0.0),
        (1000.0, 5.0),
        (2000.0, 10.0),
    ]
    assert sloping_altimetry_case.config.environment.bottom.type == "rigid"
    assert sloping_altimetry_case.options["sample_count"] == 101
    sloping_altimetry_tl_cases = [
        case for case in cases if case.name.startswith("sloping-altimetry-") and case.kind == "tl"
    ]
    assert {case.run_type for case in sloping_altimetry_tl_cases} == {"C", "I", "S"}
    for sloping_altimetry_tl_case in sloping_altimetry_tl_cases:
        assert np.allclose(sloping_altimetry_tl_case.r_grid, [1000.0])
        assert np.allclose(sloping_altimetry_tl_case.z_grid, [65.0])

    long_format_case = next(case for case in cases if case.name == "long-format-bathymetry-rays")
    assert long_format_case.config.environment.bottom.bathymetry_interpolation == "L"
    assert len(long_format_case.config.environment.bottom.bathymetry_geoacoustics) == 4
    assert long_format_case.config.environment.bottom.bathymetry_geoacoustics[1][1] == 650.0
    assert long_format_case.config.environment.bottom.bathymetry_geoacoustics[1][4] == 0.05
    assert long_format_case.kind == "ray"
    assert long_format_case.options["sample_count"] == 101

    long_format_tl_case = next(case for case in cases if case.name == "long-format-bathymetry-coherent-tl")
    assert long_format_tl_case.run_type == "C"
    assert np.allclose(long_format_tl_case.r_grid, [1000.0])
    assert np.allclose(long_format_tl_case.z_grid, [60.0])
    assert long_format_tl_case.config.environment.bottom.bathymetry_geoacoustics[3][0] == 1800.0
    long_format_intensity_cases = [
        case for case in cases if case.name.startswith("long-format-bathymetry-") and case.run_type in {"I", "S"}
    ]
    assert {case.run_type for case in long_format_intensity_cases} == {"I", "S"}
    for long_format_intensity_case in long_format_intensity_cases:
        assert np.allclose(long_format_intensity_case.r_grid, [1000.0])
        assert np.allclose(long_format_intensity_case.z_grid, [60.0])

    reflection_tl_case = next(case for case in cases if case.name == "bottom-reflection-table-coherent-tl")
    assert reflection_tl_case.config.environment.bottom.reflection_table is not None
    assert reflection_tl_case.run_type == "C"
    assert np.allclose(reflection_tl_case.r_grid, [1000.0])
    assert np.allclose(reflection_tl_case.z_grid, [65.0])
    reflection_i_case = next(case for case in cases if case.name == "bottom-reflection-table-incoherent-tl")
    reflection_s_case = next(case for case in cases if case.name == "bottom-reflection-table-semicoherent-tl")
    assert reflection_i_case.run_type == "I"
    assert reflection_s_case.run_type == "S"

    curvilinear_case = next(case for case in cases if case.name == "curvilinear-boundary-rays")
    assert curvilinear_case.config.environment.surface.altimetry_interpolation == "C"
    assert curvilinear_case.config.environment.bottom.bathymetry_interpolation == "C"
    assert curvilinear_case.options["sample_count"] == 51
    curvilinear_tl_case = next(case for case in cases if case.name == "curvilinear-boundary-coherent-tl")
    assert curvilinear_tl_case.run_type == "C"
    assert np.allclose(curvilinear_tl_case.r_grid, [600.0, 1000.0, 1400.0])
    assert np.allclose(curvilinear_tl_case.z_grid, [45.0, 60.0, 85.0])
    curvilinear_intensity_cases = [
        case for case in cases if case.name.startswith("curvilinear-boundary-") and case.run_type in {"I", "S"}
    ]
    assert {case.run_type for case in curvilinear_intensity_cases} == {"I", "S"}
    for curvilinear_intensity_case in curvilinear_intensity_cases:
        assert np.allclose(curvilinear_intensity_case.r_grid, [600.0, 1000.0, 1400.0])
        assert np.allclose(curvilinear_intensity_case.z_grid, [45.0, 60.0, 85.0])

    q_ray_case = next(case for case in cases if case.name == "range-dependent-ssp-rays")
    assert q_ray_case.config.environment.ssp.type == "quad-2d"
    assert q_ray_case.config.environment.ssp.ranges == [0.0, 500.0, 1200.0]
    assert q_ray_case.options["sample_count"] == 51
    q_tl_cases = [
        case
        for case in cases
        if case.name
        in {
            "range-dependent-ssp-coherent-tl",
            "range-dependent-ssp-incoherent-tl",
            "range-dependent-ssp-semicoherent-tl",
        }
    ]
    assert {case.run_type for case in q_tl_cases} == {"C", "I", "S"}
    for q_tl_case in q_tl_cases:
        assert q_tl_case.config.environment.ssp.type == "quad-2d"
        assert np.allclose(q_tl_case.r_grid, [1000.0])
        assert np.allclose(q_tl_case.z_grid, [50.0])

    q_multi_source_case = next(
        case for case in cases if case.name == "range-dependent-ssp-multi-source-depth-arrivals"
    )
    assert q_multi_source_case.options["method"] == "beam"
    assert q_multi_source_case.options["max_arrivals"] == 80
    assert q_multi_source_case.config.environment.ssp.type == "quad-2d"
    assert q_multi_source_case.config.geometry.source.depths == [45.0, 55.0]
    assert q_multi_source_case.config.solver.num_beams == 121
    q_multi_source_tl_cases = [
        case
        for case in cases
        if case.name.startswith("range-dependent-ssp-multi-source-depth-") and case.kind == "tl"
    ]
    assert {case.run_type for case in q_multi_source_tl_cases} == {"C", "I", "S"}
    for q_multi_source_tl_case in q_multi_source_tl_cases:
        assert q_multi_source_tl_case.config.geometry.source.depths == [45.0, 55.0]
        assert q_multi_source_tl_case.config.environment.ssp.type == "quad-2d"
        assert np.allclose(q_multi_source_tl_case.r_grid, [1000.0])
        assert np.allclose(q_multi_source_tl_case.z_grid, [50.0])

    q_reflection_tl_cases = [
        case
        for case in cases
        if case.name
        in {
            "range-dependent-ssp-bottom-reflection-table-coherent-tl",
            "range-dependent-ssp-bottom-reflection-table-incoherent-tl",
            "range-dependent-ssp-bottom-reflection-table-semicoherent-tl",
        }
    ]
    assert {case.run_type for case in q_reflection_tl_cases} == {"C", "I", "S"}
    for q_reflection_tl_case in q_reflection_tl_cases:
        assert q_reflection_tl_case.config.environment.ssp.type == "quad-2d"
        assert q_reflection_tl_case.config.environment.bottom.reflection_table is not None
        assert len(q_reflection_tl_case.config.environment.bottom.reflection_table) == 3
        assert q_reflection_tl_case.config.solver.num_beams == 101
        assert np.allclose(q_reflection_tl_case.r_grid, [1000.0])
        assert np.allclose(q_reflection_tl_case.z_grid, [50.0])

    q_sloping_tl_cases = [
        case
        for case in cases
        if case.name
        in {
            "range-dependent-ssp-sloping-bathymetry-coherent-tl",
            "range-dependent-ssp-sloping-bathymetry-incoherent-tl",
            "range-dependent-ssp-sloping-bathymetry-semicoherent-tl",
        }
    ]
    assert {case.run_type for case in q_sloping_tl_cases} == {"C", "I", "S"}
    for q_sloping_tl_case in q_sloping_tl_cases:
        assert q_sloping_tl_case.config.environment.ssp.type == "quad-2d"
        assert q_sloping_tl_case.config.environment.ssp.ranges == [0.0, 500.0, 1200.0]
        assert q_sloping_tl_case.config.environment.bottom.bathymetry == [
            (0.0, 100.0),
            (1000.0, 120.0),
            (2000.0, 140.0),
        ]
        assert q_sloping_tl_case.config.solver.num_beams == 181
        assert np.allclose(q_sloping_tl_case.r_grid, [1000.0])
        assert np.allclose(q_sloping_tl_case.z_grid, [65.0])


def test_canonical_kraken_benchmark_cases_cover_calib_like_tl_case():
    cases = canonical_kraken_benchmark_cases()
    names = {case.name for case in cases}

    assert len(cases) == 22
    assert names == {
        "kraken-calib-like-single-point",
        "kraken-rigid-pekeris-grid",
        "kraken-rigid-surface-grid",
        "kraken-rigid-field-rc-grid",
        "kraken-rigid-field-x-line-grid",
        "kraken-rigid-field-s-scaled-grid",
        "kraken-rigid-field-multi-source-depth-grid",
        "kraken-rigid-field-offsets-maxmodes-grid",
        "kraken-rigid-field-truncated-maxmodes-grid",
        "kraken-rigid-wide-receiver-grid",
        "kraken-rigid-deep-gradient-grid",
        "kraken-bottom-reflection-table-grid",
        "kraken-surface-reflection-table-grid",
        "kraken-surface-bottom-reflection-table-grid",
        "kraken-water-attenuation-grid",
        "kraken-two-layer-acoustic-grid",
        "kraken-nearfield-acoustic-halfspace-grid",
        "kraken-extreme-nearfield-acoustic-halfspace-grid",
        "kraken-lossy-elastic-bottom-grid",
        "kraken-lossy-elastic-field-x-line-grid",
        "kraken-layered-lossy-elastic-grid",
        "kraken-layered-lossy-elastic-field-s-scaled-grid",
    }

    calib = next(case for case in cases if case.name == "kraken-calib-like-single-point")
    assert calib.config.solver.type == "normal_modes"
    assert np.allclose(calib.r_grid, [2500.0])
    assert np.allclose(calib.z_grid, [50.0])

    wide_grid = next(case for case in cases if case.name == "kraken-rigid-wide-receiver-grid")
    assert wide_grid.config.solver.phase_speed_limits == (1430.0, 1800.0)
    assert wide_grid.config.solver.mesh_multiplier == [1, 3]
    assert np.allclose(wide_grid.r_grid, [600.0, 1400.0, 2600.0, 4200.0])
    assert np.allclose(wide_grid.z_grid, [20.0, 55.0, 95.0, 130.0])
    assert calib.config.solver.phase_speed_limits == (1450.0, 1650.0)

    deep_gradient = next(case for case in cases if case.name == "kraken-rigid-deep-gradient-grid")
    assert deep_gradient.config.frequency == 75.0
    assert deep_gradient.config.environment.bottom.depth == 280.0
    assert deep_gradient.config.solver.n_mesh_points == 1000
    assert deep_gradient.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(deep_gradient.r_grid, [800.0, 1800.0, 3600.0, 6000.0, 8500.0])
    assert np.allclose(deep_gradient.z_grid, [40.0, 110.0, 190.0, 250.0])

    rigid = next(case for case in cases if case.name == "kraken-rigid-pekeris-grid")
    assert rigid.config.environment.bottom.type == "rigid"
    assert rigid.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(rigid.r_grid, [1000.0, 2500.0])
    assert np.allclose(rigid.z_grid, [25.0, 50.0, 75.0])

    rigid_surface = next(case for case in cases if case.name == "kraken-rigid-surface-grid")
    assert rigid_surface.config.environment.surface.type == "rigid"
    assert rigid_surface.config.environment.bottom.type == "rigid"
    assert rigid_surface.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(rigid_surface.r_grid, [1000.0, 2500.0])
    assert np.allclose(rigid_surface.z_grid, [25.0, 50.0, 75.0])

    field_rc = next(case for case in cases if case.name == "kraken-rigid-field-rc-grid")
    assert field_rc.config.solver.field_option == "RC"
    assert field_rc.config.environment.bottom.type == "rigid"
    assert field_rc.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(field_rc.r_grid, [1000.0, 2500.0])
    assert np.allclose(field_rc.z_grid, [25.0, 50.0, 75.0])

    field_x = next(case for case in cases if case.name == "kraken-rigid-field-x-line-grid")
    assert field_x.config.solver.field_option == "XC"
    assert field_x.config.environment.bottom.type == "rigid"
    assert field_x.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(field_x.r_grid, [1000.0, 2500.0])
    assert np.allclose(field_x.z_grid, [25.0, 50.0, 75.0])

    field_s = next(case for case in cases if case.name == "kraken-rigid-field-s-scaled-grid")
    assert field_s.config.solver.field_option == "SC"
    assert field_s.config.environment.bottom.type == "rigid"
    assert field_s.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(field_s.r_grid, [1000.0, 2500.0])
    assert np.allclose(field_s.z_grid, [25.0, 50.0, 75.0])

    field_multi_source = next(case for case in cases if case.name == "kraken-rigid-field-multi-source-depth-grid")
    assert field_multi_source.config.geometry.source.depths == [40.0, 60.0]
    assert field_multi_source.config.solver.n_mesh_points == 600
    assert field_multi_source.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(field_multi_source.r_grid, [1000.0, 2500.0])
    assert np.allclose(field_multi_source.z_grid, [25.0, 50.0, 75.0])

    field_offsets = next(case for case in cases if case.name == "kraken-rigid-field-offsets-maxmodes-grid")
    assert field_offsets.config.solver.max_modes == 37
    assert field_offsets.config.solver.field_receiver_range_offsets_m == [-2.0, 0.0, 3.0]
    assert field_offsets.config.environment.bottom.type == "rigid"
    assert field_offsets.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(field_offsets.r_grid, [1000.0, 2500.0])
    assert np.allclose(field_offsets.z_grid, [25.0, 50.0, 75.0])

    field_truncated = next(case for case in cases if case.name == "kraken-rigid-field-truncated-maxmodes-grid")
    assert field_truncated.config.solver.max_modes == 3
    assert field_truncated.config.environment.bottom.type == "rigid"
    assert field_truncated.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(field_truncated.r_grid, [1000.0, 2500.0])
    assert np.allclose(field_truncated.z_grid, [25.0, 50.0, 75.0])

    reflection = next(case for case in cases if case.name == "kraken-bottom-reflection-table-grid")
    assert reflection.config.environment.bottom.reflection_table is not None
    assert len(reflection.config.environment.bottom.reflection_table) == 3
    assert reflection.config.solver.n_mesh_points == 800
    assert reflection.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(reflection.r_grid, [750.0, 1500.0])

    surface_reflection = next(case for case in cases if case.name == "kraken-surface-reflection-table-grid")
    assert surface_reflection.config.environment.surface.reflection_table is not None
    assert len(surface_reflection.config.environment.surface.reflection_table) == 3
    assert surface_reflection.config.environment.bottom.type == "rigid"
    assert surface_reflection.config.solver.n_mesh_points == 1000
    assert surface_reflection.config.solver.mesh_multiplier == [1, 2, 4]
    assert np.allclose(surface_reflection.r_grid, [750.0, 1500.0])

    surface_bottom_reflection = next(
        case for case in cases if case.name == "kraken-surface-bottom-reflection-table-grid"
    )
    assert surface_bottom_reflection.config.environment.surface.reflection_table is not None
    assert surface_bottom_reflection.config.environment.bottom.reflection_table is not None
    assert len(surface_bottom_reflection.config.environment.surface.reflection_table) == 3
    assert len(surface_bottom_reflection.config.environment.bottom.reflection_table) == 3
    assert surface_bottom_reflection.config.solver.n_mesh_points == 800
    assert surface_bottom_reflection.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(surface_bottom_reflection.r_grid, [750.0, 1500.0])

    water_attenuation = next(case for case in cases if case.name == "kraken-water-attenuation-grid")
    assert water_attenuation.config.solver.n_mesh_points == 1000
    assert water_attenuation.config.solver.mesh_multiplier == [1, 2]
    assert all(layer.attenuation == 0.15 for layer in water_attenuation.config.environment.ssp.data)
    assert np.allclose(water_attenuation.r_grid, [1000.0, 2500.0])
    assert np.allclose(water_attenuation.z_grid, [30.0, 60.0, 90.0])

    two_layer = next(case for case in cases if case.name == "kraken-two-layer-acoustic-grid")
    assert [layer.density for layer in two_layer.config.environment.ssp.data] == [1.0, 1.03, 1.08]
    assert np.allclose(two_layer.r_grid, [1000.0, 2200.0])
    assert np.allclose(two_layer.z_grid, [40.0, 90.0, 140.0])

    nearfield = next(case for case in cases if case.name == "kraken-nearfield-acoustic-halfspace-grid")
    assert nearfield.config.environment.bottom.type == "acousto-elastic"
    assert nearfield.config.environment.bottom.c_p == 1680.0
    assert nearfield.config.environment.bottom.c_s is None
    assert nearfield.config.solver.phase_speed_limits == (1450.0, 1800.0)
    assert nearfield.config.solver.n_mesh_points == 1000
    assert nearfield.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(nearfield.r_grid, [100.0, 300.0, 600.0])
    assert np.allclose(nearfield.z_grid, [120.0, 190.0])

    extreme_nearfield = next(
        case for case in cases if case.name == "kraken-extreme-nearfield-acoustic-halfspace-grid"
    )
    assert extreme_nearfield.config.environment.bottom.type == "acousto-elastic"
    assert extreme_nearfield.config.environment.bottom.c_p == 1680.0
    assert extreme_nearfield.config.environment.bottom.c_s is None
    assert extreme_nearfield.config.solver.phase_speed_limits == (1450.0, 1800.0)
    assert extreme_nearfield.config.solver.n_mesh_points == 1000
    assert extreme_nearfield.config.solver.mesh_multiplier == [1, 2]
    assert np.allclose(extreme_nearfield.r_grid, [50.0, 100.0, 300.0])
    assert np.allclose(extreme_nearfield.z_grid, [120.0, 190.0])

    lossy = next(case for case in cases if case.name == "kraken-lossy-elastic-bottom-grid")
    assert lossy.config.environment.bottom.type == "acousto-elastic"
    assert lossy.config.environment.bottom.c_s == 350.0
    assert lossy.config.environment.bottom.attenuation_p == 0.35
    assert lossy.config.environment.bottom.attenuation_s == 0.8
    assert lossy.config.solver.n_mesh_points == 1200
    assert lossy.config.solver.mesh_multiplier == [1, 2, 4, 8]

    lossy_field_x = next(case for case in cases if case.name == "kraken-lossy-elastic-field-x-line-grid")
    assert lossy_field_x.config.solver.field_option == "XC"
    assert lossy_field_x.config.environment.bottom.type == "acousto-elastic"
    assert lossy_field_x.config.environment.bottom.c_s == 350.0
    assert np.allclose(lossy_field_x.r_grid, [1000.0, 3000.0])
    assert np.allclose(lossy_field_x.z_grid, [50.0, 100.0, 150.0])

    layered_lossy = next(case for case in cases if case.name == "kraken-layered-lossy-elastic-grid")
    assert layered_lossy.config.environment.bottom.type == "acousto-elastic"
    assert layered_lossy.config.environment.bottom.c_s == 520.0
    assert layered_lossy.config.environment.bottom.attenuation_p == 0.25
    assert layered_lossy.config.environment.bottom.attenuation_s == 0.35
    assert [layer.density for layer in layered_lossy.config.environment.ssp.data] == [1.0, 1.01, 1.05, 1.08]
    assert [layer.attenuation for layer in layered_lossy.config.environment.ssp.data] == [0.02, 0.04, 0.06, 0.08]
    assert layered_lossy.config.solver.phase_speed_limits == (1440.0, 1950.0)
    assert layered_lossy.config.solver.n_mesh_points == 1600
    assert layered_lossy.config.solver.mesh_multiplier == [1]
    assert np.allclose(layered_lossy.r_grid, [800.0, 1800.0, 4200.0])
    assert np.allclose(layered_lossy.z_grid, [60.0, 135.0, 210.0])

    layered_lossy_field_s = next(
        case for case in cases if case.name == "kraken-layered-lossy-elastic-field-s-scaled-grid"
    )
    assert layered_lossy_field_s.config.solver.field_option == "SC"
    assert layered_lossy_field_s.config.environment.bottom.type == "acousto-elastic"
    assert layered_lossy_field_s.config.environment.bottom.c_s == 520.0
    assert np.allclose(layered_lossy_field_s.r_grid, [800.0, 1800.0, 4200.0])
    assert np.allclose(layered_lossy_field_s.z_grid, [60.0, 135.0, 210.0])


def test_canonical_bellhop3d_benchmark_cases_cover_receiver_grid_smoke():
    cases = canonical_bellhop3d_benchmark_cases()
    names = {case.name for case in cases}

    assert len(cases) == 5
    assert names == {
        "bellhop3d-iso-direct-coherent-tl",
        "bellhop3d-iso-multi-source-depth-coherent-tl",
        "bellhop3d-munk-slanted-terrain-coherent-tl",
        "bellhop3d-munk-slanted-terrain-oblique-tl",
        "bellhop3d-munk-slanted-terrain-steep-oblique-tl",
    }
    case = next(case for case in cases if case.name == "bellhop3d-iso-direct-coherent-tl")
    assert case.config.solver.type == "bellhop3d"
    assert case.config.solver.elevation_angles == (-5.0, 5.0)
    assert case.config.solver.azimuth_angles == (-1.0, 1.0)
    assert case.config.solver.num_elevation_beams == 21
    assert case.config.solver.num_azimuth_beams == 3
    assert case.run_type == "C"
    assert np.allclose(case.theta_grid_deg, [0.0])
    assert np.allclose(case.r_grid, [500.0, 1000.0])
    assert np.allclose(case.z_grid, [100.0])

    multi_source = next(case for case in cases if case.name == "bellhop3d-iso-multi-source-depth-coherent-tl")
    assert multi_source.config.geometry.source.depths == [80.0, 100.0, 120.0]
    assert multi_source.config.solver.type == "bellhop3d"
    assert np.allclose(multi_source.theta_grid_deg, [0.0])
    assert np.allclose(multi_source.r_grid, [500.0, 1000.0])
    assert np.allclose(multi_source.z_grid, [100.0])
    assert multi_source.active_tl_max_db == 190.0
    assert multi_source.thresholds is not None
    assert multi_source.thresholds.max_values["tl_mae_db"] == 1.0e-5
    assert multi_source.thresholds.max_values["at_only_active_count"] == 0.0
    assert multi_source.thresholds.max_values["native_only_active_count"] == 0.0
    assert multi_source.thresholds.min_values["sample_count"] == 6.0
    assert multi_source.thresholds.min_values["active_overlap_fraction"] == 1.0

    munk_case = next(case for case in cases if case.name == "bellhop3d-munk-slanted-terrain-coherent-tl")
    assert munk_case.config.environment.bottom.bathymetry_3d is not None
    assert munk_case.config.solver.elevation_angles == (-20.0, 20.0)
    assert munk_case.config.solver.azimuth_angles == (0.0, 90.0)
    assert munk_case.config.solver.num_elevation_beams == 20
    assert munk_case.config.solver.num_azimuth_beams == 24
    assert munk_case.min_range_m == 500.0
    assert munk_case.active_tl_max_db == 110.0
    assert munk_case.thresholds is not None
    assert munk_case.thresholds.max_values["tl_mae_db"] == 0.6
    assert munk_case.thresholds.max_values["at_only_active_count"] == 0.0
    assert munk_case.thresholds.max_values["native_only_active_count"] == 10.0
    assert munk_case.thresholds.max_values["native_weak_tail_count"] == 90.0
    assert munk_case.thresholds.min_values["active_overlap_fraction"] == 0.98
    assert len(munk_case.r_grid) == 31
    assert len(munk_case.z_grid) == 31

    oblique_case = next(case for case in cases if case.name == "bellhop3d-munk-slanted-terrain-oblique-tl")
    assert oblique_case.config.environment.bottom.bathymetry_3d is not None
    assert oblique_case.config.solver.elevation_angles == (-20.0, 20.0)
    assert oblique_case.config.solver.azimuth_angles == (0.0, 90.0)
    assert oblique_case.min_range_m == 500.0
    assert oblique_case.active_tl_max_db == 110.0
    assert np.allclose(oblique_case.theta_grid_deg, [45.0])
    assert len(oblique_case.r_grid) == 31
    assert len(oblique_case.z_grid) == 31
    assert oblique_case.thresholds is not None
    assert oblique_case.thresholds.max_values["tl_mae_db"] == 0.25
    assert oblique_case.thresholds.max_values["tl_p90_abs_db"] == 0.6
    assert oblique_case.thresholds.max_values["tl_max_abs_db"] == 3.0
    assert oblique_case.thresholds.max_values["at_only_active_count"] == 2.0
    assert oblique_case.thresholds.max_values["native_only_active_count"] == 1.0
    assert oblique_case.thresholds.max_values["native_weak_tail_count"] == 60.0
    assert oblique_case.thresholds.min_values["sample_count"] == 560.0
    assert oblique_case.thresholds.min_values["active_overlap_fraction"] == 0.995
    assert oblique_case.thresholds.max_abs_values["tl_bias_db"] == 0.05

    steep_oblique_case = next(
        case for case in cases if case.name == "bellhop3d-munk-slanted-terrain-steep-oblique-tl"
    )
    assert steep_oblique_case.config.environment.bottom.bathymetry_3d is not None
    assert steep_oblique_case.config.solver.elevation_angles == (-20.0, 20.0)
    assert steep_oblique_case.config.solver.azimuth_angles == (0.0, 90.0)
    assert np.allclose(steep_oblique_case.theta_grid_deg, [60.0])
    assert steep_oblique_case.min_range_m == 500.0
    assert steep_oblique_case.active_tl_max_db == 110.0
    assert len(steep_oblique_case.r_grid) == 31
    assert len(steep_oblique_case.z_grid) == 31
    assert steep_oblique_case.thresholds is not None
    assert steep_oblique_case.thresholds.max_values["tl_mae_db"] == 0.25
    assert steep_oblique_case.thresholds.max_values["tl_p90_abs_db"] == 0.6
    assert steep_oblique_case.thresholds.max_values["tl_max_abs_db"] == 3.5
    assert steep_oblique_case.thresholds.max_values["at_only_active_count"] == 0.0
    assert steep_oblique_case.thresholds.max_values["native_only_active_count"] == 0.0
    assert steep_oblique_case.thresholds.max_values["native_weak_tail_count"] == 60.0
    assert steep_oblique_case.thresholds.min_values["sample_count"] == 560.0
    assert steep_oblique_case.thresholds.min_values["active_overlap_fraction"] == 0.999
    assert steep_oblique_case.thresholds.max_abs_values["tl_bias_db"] == 0.05


def test_canonical_bellhop3d_ray_benchmark_cases_cover_ray_path_parity():
    cases = canonical_bellhop3d_ray_benchmark_cases()
    names = {case.name for case in cases}

    assert len(cases) == 9
    assert names == {
        "bellhop3d-iso-direct-eigenrays",
        "bellhop3d-iso-off-axis-eigenrays",
        "bellhop3d-iso-wide-bearing-eigenrays",
        "bellhop3d-iso-multi-depth-eigenrays",
        "bellhop3d-iso-source-xy-eigenrays",
        "bellhop3d-iso-source-depth-xy-eigenrays",
        "bellhop3d-munk-slanted-terrain-rays",
        "bellhop3d-munk-slanted-terrain-oblique-rays",
        "bellhop3d-munk-slanted-terrain-steep-oblique-rays",
    }
    eigenray = next(case for case in cases if case.name == "bellhop3d-iso-direct-eigenrays")
    assert eigenray.config.solver.type == "bellhop3d"
    assert eigenray.run_type == "E"
    assert eigenray.max_range_m == 1000.0
    assert eigenray.config.geometry.receivers.ranges == [500.0]
    assert eigenray.config.solver.elevation_angles == (-2.0, 2.0)
    assert eigenray.thresholds is not None
    assert eigenray.thresholds.max_values["unmatched_at_ray_count"] == 0.0
    assert eigenray.thresholds.max_values["depth_p90_abs_m"] == 1.0e-9
    assert eigenray.thresholds.max_values["y_p90_abs_m"] == 1.0e-9
    assert eigenray.thresholds.min_values["matched_ray_count"] == 17.0
    assert eigenray.thresholds.min_values["bounce_match_fraction"] == 1.0

    off_axis_eigenray = next(case for case in cases if case.name == "bellhop3d-iso-off-axis-eigenrays")
    assert off_axis_eigenray.run_type == "E"
    assert off_axis_eigenray.config.solver.receiver_bearings == [2.0]
    assert off_axis_eigenray.config.solver.num_azimuth_beams == 5
    assert off_axis_eigenray.thresholds is not None
    assert off_axis_eigenray.thresholds.min_values["matched_ray_count"] == 16.0

    wide_eigenray = next(case for case in cases if case.name == "bellhop3d-iso-wide-bearing-eigenrays")
    assert wide_eigenray.run_type == "E"
    assert wide_eigenray.config.solver.receiver_bearings == [30.0]
    assert wide_eigenray.config.solver.azimuth_angles == (28.0, 32.0)
    assert wide_eigenray.config.solver.box_y == 600.0
    assert wide_eigenray.thresholds is not None
    assert wide_eigenray.thresholds.min_values["matched_ray_count"] == 21.0

    multi_depth_eigenray = next(case for case in cases if case.name == "bellhop3d-iso-multi-depth-eigenrays")
    assert multi_depth_eigenray.run_type == "E"
    assert multi_depth_eigenray.config.geometry.receivers.depths == [80.0, 100.0, 120.0]
    assert multi_depth_eigenray.config.solver.num_elevation_beams == 41
    assert multi_depth_eigenray.thresholds is not None
    assert multi_depth_eigenray.thresholds.min_values["matched_ray_count"] == 41.0

    source_xy_eigenray = next(case for case in cases if case.name == "bellhop3d-iso-source-xy-eigenrays")
    assert source_xy_eigenray.run_type == "E"
    assert source_xy_eigenray.config.geometry.source.x == [0.0, 20.0]
    assert source_xy_eigenray.config.geometry.source.y == [0.0, 10.0]
    assert source_xy_eigenray.config.solver.box_x == 700.0
    assert source_xy_eigenray.config.solver.box_y == 200.0
    assert source_xy_eigenray.thresholds is not None
    assert source_xy_eigenray.thresholds.min_values["matched_ray_count"] == 21.0

    source_depth_xy_eigenray = next(case for case in cases if case.name == "bellhop3d-iso-source-depth-xy-eigenrays")
    assert source_depth_xy_eigenray.run_type == "E"
    assert source_depth_xy_eigenray.config.geometry.source.depths == [80.0, 120.0]
    assert source_depth_xy_eigenray.config.geometry.source.x == [0.0, 20.0]
    assert source_depth_xy_eigenray.config.geometry.source.y == [0.0, 10.0]
    assert source_depth_xy_eigenray.config.solver.num_elevation_beams == 41
    assert source_depth_xy_eigenray.thresholds is not None
    assert source_depth_xy_eigenray.thresholds.min_values["matched_ray_count"] == 41.0

    case = next(case for case in cases if case.name == "bellhop3d-munk-slanted-terrain-rays")
    assert case.name == "bellhop3d-munk-slanted-terrain-rays"
    assert case.config.solver.type == "bellhop3d"
    assert case.run_type == "R"
    assert case.max_range_m == 10000.0
    assert case.thresholds is not None
    assert case.thresholds.max_values["unmatched_at_ray_count"] == 0.0
    assert case.thresholds.max_values["depth_p90_abs_m"] == 2.5
    assert case.thresholds.max_values["y_p90_abs_m"] == 1.0e-9
    assert case.thresholds.min_values["sample_count"] == 20000.0
    assert case.thresholds.min_values["matched_ray_count"] == 20.0
    assert case.thresholds.min_values["bounce_match_fraction"] == 1.0

    oblique = next(case for case in cases if case.name == "bellhop3d-munk-slanted-terrain-oblique-rays")
    assert oblique.config.solver.type == "bellhop3d"
    assert oblique.config.solver.azimuth_angles == (30.0, 30.0)
    assert oblique.config.solver.num_azimuth_beams == 1
    assert oblique.run_type == "R"
    assert oblique.max_range_m == 10000.0
    assert oblique.thresholds is not None
    assert oblique.thresholds.max_values["unmatched_at_ray_count"] == 0.0
    assert oblique.thresholds.max_values["depth_p90_abs_m"] == 2.5
    assert oblique.thresholds.max_values["depth_max_abs_m"] == 3.5
    assert oblique.thresholds.max_values["y_p90_abs_m"] == 1.0e-8
    assert oblique.thresholds.max_values["launch_azimuth_mae_deg"] == 1.0e-9
    assert oblique.thresholds.min_values["sample_count"] == 20000.0
    assert oblique.thresholds.min_values["matched_ray_count"] == 20.0
    assert oblique.thresholds.min_values["bounce_match_fraction"] == 1.0
    assert oblique.thresholds.min_values["top_bounce_match_fraction"] == 1.0
    assert oblique.thresholds.min_values["bottom_bounce_match_fraction"] == 1.0

    steep_oblique = next(case for case in cases if case.name == "bellhop3d-munk-slanted-terrain-steep-oblique-rays")
    assert steep_oblique.config.solver.type == "bellhop3d"
    assert steep_oblique.config.solver.azimuth_angles == (60.0, 60.0)
    assert steep_oblique.config.solver.num_azimuth_beams == 1
    assert steep_oblique.run_type == "R"
    assert steep_oblique.max_range_m == 10000.0
    assert steep_oblique.thresholds is not None
    assert steep_oblique.thresholds.max_values["unmatched_at_ray_count"] == 0.0
    assert steep_oblique.thresholds.max_values["depth_p90_abs_m"] == 2.5
    assert steep_oblique.thresholds.max_values["depth_max_abs_m"] == 3.5
    assert steep_oblique.thresholds.max_values["y_p90_abs_m"] == 1.0e-8
    assert steep_oblique.thresholds.max_values["launch_azimuth_mae_deg"] == 1.0e-9
    assert steep_oblique.thresholds.min_values["sample_count"] == 20000.0
    assert steep_oblique.thresholds.min_values["matched_ray_count"] == 20.0
    assert steep_oblique.thresholds.min_values["bounce_match_fraction"] == 1.0
    assert steep_oblique.thresholds.min_values["top_bounce_match_fraction"] == 1.0
    assert steep_oblique.thresholds.min_values["bottom_bounce_match_fraction"] == 1.0


def test_canonical_bellhop3d_arrival_benchmark_cases_cover_arrival_parity():
    cases = canonical_bellhop3d_arrival_benchmark_cases()
    names = {case.name for case in cases}

    assert len(cases) == 7
    assert names == {
        "bellhop3d-iso-direct-arrivals",
        "bellhop3d-iso-multi-depth-arrivals",
        "bellhop3d-iso-direct-off-axis-arrivals",
        "bellhop3d-iso-direct-wide-bearing-arrivals",
        "bellhop3d-iso-source-xy-arrivals",
        "bellhop3d-iso-source-depth-xy-arrivals",
        "bellhop3d-munk-slanted-terrain-oblique-arrivals",
    }
    case = next(case for case in cases if case.name == "bellhop3d-iso-direct-arrivals")
    assert case.name == "bellhop3d-iso-direct-arrivals"
    assert case.config.solver.type == "bellhop3d"
    assert case.run_type == "A"
    assert case.delay_tolerance_s == 0.01
    assert case.angle_tolerance_deg == 0.1
    assert case.tolerance_m == 20.0
    assert case.config.geometry.receivers.ranges == [500.0]
    assert case.config.solver.elevation_angles == (-2.0, 2.0)
    assert case.config.solver.azimuth_angles == (-2.0, 2.0)
    assert case.config.solver.num_elevation_beams == 21
    assert case.config.solver.num_azimuth_beams == 3
    assert case.thresholds is not None
    assert case.thresholds.max_values["unmatched_at_count"] == 0.0
    assert case.thresholds.max_values["unmatched_native_count"] == 0.0
    assert case.thresholds.max_values["delay_mae_s"] == 1.0e-5
    assert case.thresholds.max_values["amplitude_median_relative_error"] == 0.02
    assert case.thresholds.min_values["matched_count"] == 3.0
    assert case.thresholds.min_values["match_fraction"] == 1.0

    multi_depth = next(case for case in cases if case.name == "bellhop3d-iso-multi-depth-arrivals")
    assert multi_depth.config.geometry.receivers.ranges == [500.0]
    assert multi_depth.config.geometry.receivers.depths == [80.0, 100.0, 120.0]
    assert multi_depth.config.solver.elevation_angles == (-5.0, 5.0)
    assert multi_depth.config.solver.num_elevation_beams == 41
    assert multi_depth.angle_tolerance_deg == 0.2
    assert multi_depth.thresholds is not None
    assert multi_depth.thresholds.max_values["unmatched_at_count"] == 0.0
    assert multi_depth.thresholds.max_values["unmatched_native_count"] == 3.0
    assert multi_depth.thresholds.max_values["amplitude_p90_relative_error"] == 0.02
    assert multi_depth.thresholds.min_values["matched_count"] == 9.0
    assert multi_depth.thresholds.min_values["match_fraction"] == 1.0

    off_axis = next(case for case in cases if case.name == "bellhop3d-iso-direct-off-axis-arrivals")
    assert off_axis.config.solver.receiver_bearings == [2.0]
    assert off_axis.config.solver.num_azimuth_beams == 5
    assert off_axis.angle_tolerance_deg == 2.0
    assert off_axis.thresholds is not None
    assert off_axis.thresholds.max_values["delay_mae_s"] == 1.0e-5
    assert off_axis.thresholds.max_values["amplitude_median_relative_error"] == 0.01
    assert off_axis.thresholds.max_values["launch_azimuth_mae_deg"] == 0.01

    wide_bearing = next(case for case in cases if case.name == "bellhop3d-iso-direct-wide-bearing-arrivals")
    assert wide_bearing.config.geometry.receivers.ranges == [500.0]
    assert wide_bearing.config.geometry.receivers.depths == [100.0]
    assert wide_bearing.config.solver.receiver_bearings == [30.0]
    assert wide_bearing.config.solver.azimuth_angles == (28.0, 32.0)
    assert wide_bearing.config.solver.num_elevation_beams == 21
    assert wide_bearing.config.solver.num_azimuth_beams == 5
    assert wide_bearing.config.solver.box_y == 600.0
    assert wide_bearing.delay_tolerance_s == 0.01
    assert wide_bearing.angle_tolerance_deg == 2.0
    assert wide_bearing.tolerance_m == 20.0
    assert wide_bearing.thresholds is not None
    assert wide_bearing.thresholds.max_values["unmatched_at_count"] == 0.0
    assert wide_bearing.thresholds.max_values["unmatched_native_count"] == 1.0
    assert wide_bearing.thresholds.max_values["delay_mae_s"] == 1.0e-5
    assert wide_bearing.thresholds.max_values["amplitude_median_relative_error"] == 0.15
    assert wide_bearing.thresholds.max_values["amplitude_p90_relative_error"] == 0.3
    assert wide_bearing.thresholds.max_values["launch_azimuth_mae_deg"] == 0.2
    assert wide_bearing.thresholds.min_values["matched_count"] == 3.0
    assert wide_bearing.thresholds.min_values["match_fraction"] == 1.0
    assert wide_bearing.thresholds.min_values["bounce_match_fraction"] == 1.0

    source_xy = next(case for case in cases if case.name == "bellhop3d-iso-source-xy-arrivals")
    assert source_xy.config.geometry.source.x == [0.0, 20.0]
    assert source_xy.config.geometry.source.y == [0.0, 10.0]
    assert source_xy.config.solver.box_x == 700.0
    assert source_xy.config.solver.box_y == 200.0
    assert source_xy.angle_tolerance_deg == 0.5
    assert source_xy.thresholds is not None
    assert source_xy.thresholds.max_values["unmatched_native_count"] == 0.0
    assert source_xy.thresholds.min_values["matched_count"] == 12.0
    assert source_xy.thresholds.min_values["match_fraction"] == 1.0

    source_depth_xy = next(case for case in cases if case.name == "bellhop3d-iso-source-depth-xy-arrivals")
    assert source_depth_xy.config.geometry.source.depths == [80.0, 120.0]
    assert source_depth_xy.config.geometry.source.x == [0.0, 20.0]
    assert source_depth_xy.config.geometry.source.y == [0.0, 10.0]
    assert source_depth_xy.config.solver.elevation_angles == (-5.0, 5.0)
    assert source_depth_xy.config.solver.num_elevation_beams == 41
    assert source_depth_xy.thresholds is not None
    assert source_depth_xy.thresholds.max_values["unmatched_native_count"] == 8.0
    assert source_depth_xy.thresholds.min_values["matched_count"] == 24.0
    assert source_depth_xy.thresholds.min_values["match_fraction"] == 1.0

    munk_oblique = next(
        case for case in cases if case.name == "bellhop3d-munk-slanted-terrain-oblique-arrivals"
    )
    assert munk_oblique.config.environment.bottom.bathymetry_3d is not None
    assert munk_oblique.config.geometry.receivers.ranges == [3000.0]
    assert munk_oblique.config.geometry.receivers.depths == [1000.0]
    assert munk_oblique.config.solver.receiver_bearings == [30.0]
    assert munk_oblique.config.solver.elevation_angles == (-20.0, 20.0)
    assert munk_oblique.config.solver.azimuth_angles == (25.0, 35.0)
    assert munk_oblique.config.solver.num_elevation_beams == 81
    assert munk_oblique.config.solver.num_azimuth_beams == 11
    assert munk_oblique.delay_tolerance_s == 0.03
    assert munk_oblique.angle_tolerance_deg == 3.0
    assert munk_oblique.tolerance_m == 50.0
    assert munk_oblique.max_arrivals == 100
    assert munk_oblique.thresholds is not None
    assert munk_oblique.thresholds.max_values["unmatched_at_count"] == 2.0
    assert munk_oblique.thresholds.max_values["unmatched_native_count"] == 3.0
    assert munk_oblique.thresholds.max_values["delay_mae_s"] == 5.0e-4
    assert munk_oblique.thresholds.max_values["amplitude_median_relative_error"] == 0.2
    assert munk_oblique.thresholds.max_values["launch_declination_mae_deg"] == 0.6
    assert munk_oblique.thresholds.max_values["launch_azimuth_mae_deg"] == 0.5
    assert munk_oblique.thresholds.min_values["matched_count"] == 50.0
    assert munk_oblique.thresholds.min_values["match_fraction"] == 0.95
    assert munk_oblique.thresholds.min_values["bounce_match_fraction"] == 1.0


def test_canonical_benchmark_thresholds_are_matrix_friendly():
    bellhop3d_thresholds = bellhop3d_smoke_thresholds()
    bellhop_thresholds = bellhop_smoke_thresholds()
    kraken_thresholds = kraken_smoke_thresholds()

    assert bellhop3d_thresholds.ignore_missing
    assert bellhop3d_thresholds.max_values["tl_mae_db"] == 1.0
    assert bellhop3d_thresholds.min_values["sample_count"] == 1.0
    assert bellhop_thresholds.ignore_missing
    assert bellhop_thresholds.max_values["tl_mae_db"] == 1.0
    assert bellhop_thresholds.max_values["delay_mae_s"] == 0.02
    assert bellhop_thresholds.max_values["depth_p90_abs_m"] == 0.02
    assert bellhop_thresholds.max_values["amplitude_median_relative_error"] == 0.01
    assert bellhop_thresholds.min_values["sample_count"] == 1.0
    assert bellhop_thresholds.min_values["match_fraction"] == 0.5
    assert bellhop_thresholds.min_values["bounce_match_fraction"] == 0.98
    assert kraken_thresholds.ignore_missing
    assert kraken_thresholds.min_values["sample_count"] == 1.0
    assert kraken_thresholds.equals["at_error_count"] == 0
    assert kraken_thresholds.equals["at_fatal_error_count"] == 0
    assert kraken_thresholds.equals["field_error_count"] == 0
    assert kraken_thresholds.equals["field_fatal_error_count"] == 0
    assert kraken_thresholds.equals["field_completed_without_errors"] is True
    assert kraken_thresholds.max_values["tl_mae_db"] == 1.0
