import numpy as np
import pytest
import struct
from pathlib import Path

from pyacoustics.solvers.external_io import (
    ATBathymetry3D,
    ATBoundaryProfile,
    ATRayRecord,
    ATReflectionTable,
    ATSourceBeamPattern,
    generate_at_bellhop_env,
    generate_at_bellhop3d_env,
    generate_at_kraken_env,
    read_at_bathymetry_3d,
    read_at_bathymetry_3d_provinces,
    read_at_ray_records,
    read_at_shd,
    write_at_ray_records,
    read_at_ati,
    read_at_brc,
    read_at_bty,
    read_at_irc,
    read_at_mod,
    read_at_mod_metadata,
    read_at_mod_wavenumbers,
    read_at_field_flp,
    read_at_env_summary,
    read_at_fortran_source_summary,
    read_at_matlab_script_summary,
    read_at_prt,
    read_at_ram_in_summary,
    read_at_ray,
    read_at_shd_full,
    read_at_reflection_table,
    read_at_source_beam_pattern,
    read_at_trc,
    write_at_boundary_profile,
    write_at_bathymetry_3d,
    write_at_bathymetry_3d_provinces,
    write_at_field_flp,
    write_at_mod,
    write_at_shd,
    write_at_reflection_table,
    write_at_source_beam_pattern,
)
from pyacoustics.schema import (
    Bathymetry3DModel,
    BoundaryCondition,
    BellhopConfig,
    Bellhop3DConfig,
    EnvironmentModel,
    GeometryModel,
    KrakenConfig,
    ReceiverModel,
    SimulationConfig,
    SourceModel,
    SSPLayer,
    SSPModel,
)


def test_read_at_prt_extracts_warnings_and_errors(tmp_path):
    prt_path = tmp_path / "case.prt"
    prt_path.write_text(
        "BELLHOP test run\n"
        "Case title from env\n"
        " frequency =   50.00     Hz\n"
        " Dummy parameter NMedia =            1\n"
        " Depth =    5000.00 m\n"
        "                                 ( RMS roughness =   0.25     )\n"
        "    Number of Source   z-coordinates, Sz =            1\n"
        "    Number of Receiver z-coordinates, Rz =          501\n"
        "    Receiver z-coordinates, Rz (m)\n"
        "       0.0  10.0\n"
        "       ...\n"
        "       5000.0\n"
        "    Number of Receiver r-coordinates, Rr =         1001\n"
        "    Receiver r-coordinates, Rr (km)\n"
        "       0.0  0.1\n"
        "       ...\n"
        "       100.0\n"
        " Coherent TL calculation\n"
        " Geometric hat beams in Cartesian coordinates\n"
        " Point source (cylindrical coordinates)\n"
        " Rectilinear receiver grid: Receivers at Rr( : ) x Rz( : )\n"
        "    Number of beams in elevation   =         1000\n"
        "    Beam take-off angles (degrees)\n"
        "      -20.0 -19.96\n"
        "       ...\n"
        "       20.0\n"
        " Step length,       deltas =   50.00     m\n"
        "   Box%z  =   5500.     m\n"
        "   Box%r  =   101.0    km\n"
        "( Maximum # of arrivals = 10 )\n"
        " CPU Time =       0.545    s\n"
        "Normal progress line\n"
        "*** WARNING *** step size was reduced\n"
        "FATAL ERROR: ray escaped the box\n"
        "0 errors detected after cleanup\n"
    )

    log = read_at_prt(prt_path)

    assert log.title == "BELLHOP test run"
    assert log.program == "BELLHOP test run"
    assert log.case_title == "Case title from env"
    assert np.isclose(log.frequency_hz, 50.0)
    assert log.nmedia == 1
    assert np.isclose(log.bottom_depth_m, 5000.0)
    assert np.isclose(log.rms_roughness_m, 0.25)
    assert log.source_depth_count == 1
    assert log.receiver_depth_count == 501
    assert log.receiver_range_count == 1001
    assert np.isclose(log.receiver_depth_min_m, 0.0)
    assert np.isclose(log.receiver_depth_max_m, 5000.0)
    assert np.isclose(log.receiver_range_min_km, 0.0)
    assert np.isclose(log.receiver_range_max_km, 100.0)
    assert log.beam_count == 1000
    assert np.isclose(log.beam_angle_min_deg, -20.0)
    assert np.isclose(log.beam_angle_max_deg, 20.0)
    assert np.isclose(log.step_length_m, 50.0)
    assert np.isclose(log.box_depth_m, 5500.0)
    assert np.isclose(log.box_range_km, 101.0)
    assert log.max_arrivals == 10
    assert np.isclose(log.cpu_time_s, 0.545)
    assert log.run_type_description == "Coherent TL calculation"
    assert log.beam_description == "Geometric hat beams in Cartesian coordinates"
    assert log.source_description == "Point source (cylindrical coordinates)"
    assert log.receiver_grid_description == "Rectilinear receiver grid: Receivers at Rr( : ) x Rz( : )"
    assert log.parameters["frequency"] == 50.0
    assert log.parameters["Step length, deltas"] == 50.0
    assert log.warnings == ["*** WARNING *** step size was reduced"]
    assert log.errors == ["FATAL ERROR: ray escaped the box"]
    assert log.warning_count == 1
    assert log.error_count == 1
    assert log.fatal_error_count == 1
    assert not log.completed_without_errors
    assert "Normal progress line" in log.text


def test_read_at_env_summary_detects_bellhop_kraken_and_bellhop3d(tmp_path):
    bellhop_env = tmp_path / "bellhop.env"
    bellhop_env.write_text(
        "'Bellhop summary'\n"
        "250.0\n"
        "1\n"
        "'CVW'\n"
        "2 0.0 100.0\n"
        "0.0 1500.0 /\n"
        "100.0 1500.0 /\n"
        "'A' 0.0\n"
        "100.0 1590.0 0.0 1.2 0.5 0.0 /\n"
        "1\n"
        "50.0 /\n"
        "101\n"
        "0.0 100.0 /\n"
        "501\n"
        "0.0 5.0 /\n"
        "'C'\n"
        "1789\n"
        "-89 89 /\n"
        "0.0 101.0 5.1\n"
    )
    bellhop = read_at_env_summary(bellhop_env)
    assert bellhop.model_family == "bellhop"
    assert bellhop.title == "Bellhop summary"
    assert bellhop.frequency_hz == 250.0
    assert bellhop.nmedia == 1
    assert bellhop.ssp_option == "C"
    assert bellhop.top_option == "V"
    assert bellhop.bottom_option == "A"
    assert bellhop.run_type == "C"
    assert bellhop.source_depth_count == 1
    assert bellhop.receiver_depth_count == 101
    assert bellhop.receiver_range_count == 501
    assert bellhop.beam_count == 1789
    assert np.isclose(bellhop.box_z_m, 101.0)
    assert np.isclose(bellhop.box_range_km, 5.1)

    kraken_env = tmp_path / "kraken.env"
    kraken_env.write_text(
        "'Kraken summary'\n"
        "250.0\n"
        "1\n"
        "'CVW'\n"
        "1000 0.0 100.0\n"
        "0.0 1500.0 /\n"
        "100.0 1500.0 /\n"
        "'A' 0.0\n"
        "100.0 1590.0 0.0 1.2 0.5 /\n"
        "1400.0 20000.0\n"
        "1000.0\n"
        "1\n"
        "50.0 /\n"
        "101\n"
        "0.0 100.0 /\n"
    )
    kraken = read_at_env_summary(kraken_env)
    assert kraken.model_family == "kraken"
    assert kraken.run_type is None
    assert kraken.source_depth_count == 1
    assert kraken.receiver_depth_count == 101
    assert np.isclose(kraken.box_range_km, 1000.0)

    bellhop3d_env = tmp_path / "bellhop3d.env"
    bellhop3d_env.write_text(
        "'Bellhop3D summary'\n"
        "50.0\n"
        "1\n"
        "'CVW'\n"
        "2 0.0 20000.0\n"
        "0.0 1548.52 /\n"
        "20000.0 1551.91 /\n"
        "'A~' 0.0\n"
        "20000.0 1600.0 0.0 1.8 0.8 /\n"
        "1\n"
        "0.0 /\n"
        "1\n"
        "0.0 /\n"
        "1\n"
        "1000.0 /\n"
        "1\n"
        "1000.0 5000.0 /\n"
        "1001\n"
        "0.0 100.0 /\n"
        "37\n"
        "0.0 360.0 /\n"
        "'CG   3'\n"
        "200\n"
        "-14.66 20.0 /\n"
        "144\n"
        "0.0 360.0 /\n"
        "0.0 99.1 99.1 20500.0\n"
    )
    bellhop3d = read_at_env_summary(bellhop3d_env)
    assert bellhop3d.model_family == "bellhop3d"
    assert bellhop3d.run_type == "CG   3"
    assert bellhop3d.source_x_count == 1
    assert bellhop3d.source_y_count == 1
    assert bellhop3d.source_depth_count == 1
    assert bellhop3d.receiver_depth_count == 1
    assert bellhop3d.receiver_range_count == 1001
    assert bellhop3d.receiver_bearing_count == 37
    assert bellhop3d.beam_count == 200
    assert bellhop3d.beam_bearing_count == 144
    assert np.isclose(bellhop3d.box_range_km, 99.1)
    assert np.isclose(bellhop3d.box_y_km, 99.1)
    assert np.isclose(bellhop3d.box_z_m, 20500.0)


def test_read_at_env_summary_handles_compact_nx2d_and_legacy_range_dependent_kraken(tmp_path):
    nx2d_env = tmp_path / "munknx2d.env"
    nx2d_env.write_text(
        "'Three-dimensional Munk profile'\n"
        "'NMNR'\n"
        "50.0\n"
        "-14.66, 14.66, 44\n"
        "0.0, 360.0, 37\n"
        "2\n"
        "800, 200.0\n"
        "0.3\n"
        "1000.0, 800.0\n"
        "0.0, 360.0, 37\n"
        "0.0, 100.0, 201\n"
        "'NX2D'\n"
    )
    nx2d = read_at_env_summary(nx2d_env)
    assert nx2d.model_family == "bellhop3d"
    assert nx2d.title == "Three-dimensional Munk profile"
    assert nx2d.run_type == "NX2D"
    assert nx2d.frequency_hz == 50.0
    assert nx2d.nmedia == 2
    assert nx2d.beam_count == 44
    assert nx2d.beam_bearing_count == 37
    assert nx2d.receiver_bearing_count == 37
    assert nx2d.receiver_range_count == 201
    assert nx2d.source_depth_count == 1
    assert nx2d.receiver_depth_count == 1
    assert np.isclose(nx2d.box_range_km, 100.0)

    kraken_env = tmp_path / "munkr.env"
    kraken_env.write_text(
        "Range-dependent Munk profile (100 km)\n"
        "50.0\n"
        "2\n"
        "NKF\n"
        "500, 0.0, 26\n"
        "0.0, 1555.10, 0.0, 1.0\n"
        "5000.0, 1558.95, 0.0, 1.0\n"
        "1, 0.0, 1\n"
        "5000.0, 1558.95, 0.0, 1.0, 0.0\n"
        "1450.0, 1551.91\n"
        "0\n"
        "2048, 1\n"
    )
    kraken = read_at_env_summary(kraken_env)
    assert kraken.model_family == "kraken"
    assert kraken.title == "Range-dependent Munk profile (100 km)"
    assert kraken.frequency_hz == 50.0
    assert kraken.nmedia == 2
    assert kraken.ssp_option == "N"
    assert kraken.top_option == "K"
    assert kraken.bottom_option is None
    assert kraken.run_type is None


def test_read_at_ram_in_summary_extracts_range_dependent_blocks(tmp_path):
    ram_path = tmp_path / "ram.in"
    ram_path.write_text(
        "RAM Wedge problem\n"
        "25.0 100.0 30.0\n"
        "4000.0 5.0 1\n"
        "1000.0 1 2 1000.0\n"
        "1500.0 2 1 0.0\n"
        "0.0 200.0\n"
        "4000.0 0.0\n"
        "-1 -1\n"
        "0 1500\n"
        "3000 1500\n"
        "-1 -1\n"
        "0.0 1700\n"
        "-1 -1\n"
        "0.0 1.5\n"
        "-1 -1\n"
        "0.0 0.5\n"
        "900.0 0.5\n"
        "1000.0 10.0\n"
        "-1 -1\n"
        "25000.0\n"
        "0 1500\n"
        "3000 1500\n"
        "-1 -1\n"
        "0.0 1700\n"
        "-1 -1\n"
        "0.0 1.5\n"
        "-1 -1\n"
        "0.0 0.5\n"
        "900.0 0.5\n"
        "1000.0 10.0\n"
        "-1 -1\n",
        encoding="utf-8",
    )

    summary = read_at_ram_in_summary(ram_path)

    assert summary.title == "RAM Wedge problem"
    assert summary.frequency_hz == 25.0
    assert summary.source_depth_m == 100.0
    assert summary.receiver_depth_m == 30.0
    assert summary.range_dependent_profile_count == 2
    assert summary.sediment_layer_count == 1
    assert summary.bathymetry_point_count == 2
    assert summary.profile_range_count == 1
    assert summary.ssp_point_count == 4
    assert summary.bottom_sound_speed_point_count == 2
    assert summary.density_point_count == 2
    assert summary.attenuation_point_count == 6
    assert np.isclose(summary.profile_range_min_m, 25000.0)
    assert np.isclose(summary.sound_speed_min_mps, 1500.0)
    assert np.isclose(summary.bottom_sound_speed_max_mps, 1700.0)
    assert np.isclose(summary.density_max_gpcm3, 1.5)
    assert np.isclose(summary.attenuation_max_db_per_wavelength, 10.0)


def test_read_at_matlab_script_summary_extracts_at_command_inventory(tmp_path):
    script_path = tmp_path / "runtests.m"
    script_path.write_text(
        "% AT test driver\n"
        "function runtests\n"
        "bellhop('MunkB')\n"
        "plotshd('MunkB.shd')\n"
        "name = 'not % a comment'; % trailing comment\n"
        "kraken MunkK\n"
        "read_shd('MunkK.shd');\n"
        "delayandsum('MunkS.rts.mat')\n",
        encoding="utf-8",
    )

    summary = read_at_matlab_script_summary(script_path)

    assert summary.line_count == 8
    assert summary.code_line_count == 7
    assert summary.comment_line_count == 1
    assert summary.function_names == ("runtests",)
    assert summary.at_command_counts["bellhop"] == 1
    assert summary.at_command_counts["plotshd"] == 1
    assert summary.at_command_counts["kraken"] == 1
    assert summary.at_command_counts["read_shd"] == 1
    assert summary.at_command_counts["delayandsum"] == 1
    assert summary.external_command_counts == {}
    assert summary.referenced_extensions == (".mat", ".shd")


def test_read_at_matlab_script_summary_extracts_external_executable_inventory(tmp_path):
    script_path = tmp_path / "runtests.m"
    script_path.write_text(
        "runkraken = which('kraken.exe');\n"
        "runfield = which('field.exe');\n"
        "eval( [ '! \"' runkraken '\" gulf_rd' ] );\n"
        "eval( [ '! \"' runkraken '\" gulf_cm' ] );\n"
        "eval( [ '! \"' runfield '\" gulf_ri' ] );\n"
        "!field3d.exe < lant.flp\n"
        "system('krakenc.exe < foo.env')\n"
        "% eval( [ '! \"' runkraken '\" commented' ] );\n",
        encoding="utf-8",
    )

    summary = read_at_matlab_script_summary(script_path)

    assert summary.external_command_counts == {
        "field": 1,
        "field3d": 1,
        "kraken": 2,
        "krakenc": 1,
    }


def test_read_at_fortran_source_summary_extracts_static_inventory(tmp_path):
    source_path = tmp_path / "solver.f90"
    source_path.write_text(
        "! AT solver source\n"
        "program Bellhop\n"
        "  use MathConstants\n"
        "  use RWSHDFile, only: WriteHeader\n"
        "  call Log('not ! a comment') ! trailing comment\n"
        "end program Bellhop\n"
        "module TraceMod\n"
        "contains\n"
        "  subroutine TraceRay(alpha)\n"
        "  end subroutine TraceRay\n"
        "  real function TravelTime(r)\n"
        "  end function TravelTime\n"
        "end module TraceMod\n",
        encoding="utf-8",
    )

    summary = read_at_fortran_source_summary(source_path)

    assert summary.line_count == 13
    assert summary.comment_line_count == 1
    assert summary.program_names == ("Bellhop",)
    assert summary.module_names == ("TraceMod",)
    assert summary.subroutine_names == ("TraceRay",)
    assert summary.function_names == ("TravelTime",)
    assert summary.used_modules == ("MathConstants", "RWSHDFile")


def test_read_at_boundary_profile_accepts_count_then_interpolation_header(tmp_path):
    path = tmp_path / "parabot39.bty"
    path.write_text(
        "3 'C'\n"
        "0.0 500.0\n"
        "0.5 550.0\n"
        "1.0 600.0\n"
    )

    profile = read_at_bty(path)

    assert profile.interpolation == "C"
    assert profile.ranges_m.tolist() == [0.0, 500.0, 1000.0]
    assert profile.values_m.tolist() == [500.0, 550.0, 600.0]
    assert profile.geoacoustics is None


def test_read_write_at_bathymetry_3d_provinces_roundtrip(tmp_path):
    source_path = tmp_path / "provinces.bty"
    source_path.write_text(
        "'RL'\n"
        "3\n"
        "-1.0 0.0 1.0\n"
        "2\n"
        "0.0 2.0\n"
        "100.0 110.0 120.0\n"
        "130.0 140.0 150.0\n"
        "1 2 2\n"
        "3 3 1\n"
        "3\n"
        "1500.0 0.0 1.8 0.8 0.0\n"
        "1550.0 0.0 1.9 0.6 0.0\n"
        "1600.0 700.0 2.0 0.4 0.1\n"
    )

    parsed = read_at_bathymetry_3d_provinces(source_path)

    assert parsed.interpolation == "RL"
    assert parsed.province_property_kind == "geoacoustic"
    assert parsed.x_m.tolist() == [-1000.0, 0.0, 1000.0]
    assert parsed.y_m.tolist() == [0.0, 2000.0]
    assert parsed.z_m.shape == (2, 3)
    assert parsed.province_indices.tolist() == [[1, 2, 2], [3, 3, 1]]
    assert parsed.province_properties.shape == (3, 5)

    roundtrip_path = tmp_path / "roundtrip.bty"
    write_at_bathymetry_3d_provinces(roundtrip_path, parsed)
    reparsed = read_at_bathymetry_3d_provinces(roundtrip_path)

    assert reparsed.interpolation == parsed.interpolation
    assert reparsed.province_property_kind == parsed.province_property_kind
    np.testing.assert_allclose(reparsed.x_m, parsed.x_m)
    np.testing.assert_allclose(reparsed.y_m, parsed.y_m)
    np.testing.assert_allclose(reparsed.z_m, parsed.z_m)
    np.testing.assert_array_equal(reparsed.province_indices, parsed.province_indices)
    np.testing.assert_allclose(reparsed.province_properties, parsed.province_properties)


def test_read_at_prt_extracts_krakenc_metadata(tmp_path):
    prt_path = tmp_path / "krakenc.prt"
    prt_path.write_text(
        " _________________________________________________\n"
        "\n"
        " KRAKENC-native kraken at parity\n"
        " Nominal Frequency =   250.0    Hz\n"
        " NMedia    =   1\n"
        "\n"
        "     N2-Linear approximation to SSP\n"
        "     Attenuation units: dB/wavelength\n"
        "     VACUUM\n"
        "       ( # mesh points =     0  RMS roughness =   0.00     m )\n"
        " Sound speed profile:\n"
        "      0.00      1500.00      0.00     1.00       0.0000    0.0000\n"
        "    100.00      1500.00      0.00     1.00       0.0000    0.0000\n"
        "       ( # mesh points =   333  automatically calculated     )\n"
        "     ACOUSTO-ELASTIC half-space\n"
        "    100.00      1590.00      0.00     1.20       0.5000    0.0000\n"
        " cLow =   1450.0     m/s      cHigh =   1650.0     m/s\n"
        " RMax =   0.0000     km\n"
        "    Number of Source   z-coordinates, Sz =            1\n"
        "    Number of Receiver z-coordinates, Rz =         1001\n"
        " Mesh multiplier   CPU seconds\n"
        " Max. number of restarts in root finder, MaxTries =            1\n"
        "        1            0.545E-03s\n"
        "    I       k             alpha     Phase Speed     Group Speed\n"
        "          (1/m)           (1/m)        (m/s)           (m/s)\n"
        "    1   1.046756917     -0.21E-05   1500.631428       1499.41\n"
        "    2   1.045433990     -0.86E-05   1502.530377       1497.64\n"
        "    3   1.043225728     -0.19E-04   1505.710878       1494.69\n"
        " _________________________________________________\n"
        " End of environmental file\n"
    )

    log = read_at_prt(prt_path)

    assert log.title == "KRAKENC-native kraken at parity"
    assert log.program == "KRAKENC-native kraken at parity"
    assert log.case_title is None
    assert np.isclose(log.frequency_hz, 250.0)
    assert log.nmedia == 1
    assert log.source_depth_count == 1
    assert log.receiver_depth_count == 1001
    assert log.attenuation_units == "dB/wavelength"
    assert np.isclose(log.c_low_mps, 1450.0)
    assert np.isclose(log.c_high_mps, 1650.0)
    assert np.isclose(log.rmax_km, 0.0)
    assert log.mesh_point_count == 333
    assert log.max_root_finder_restarts == 1
    assert log.mesh_multiplier_cpu_s == {1: 0.000545}
    assert log.mode_count == 3
    assert np.isclose(log.mode_wavenumber_min_1pm, 1.043225728)
    assert np.isclose(log.mode_wavenumber_max_1pm, 1.046756917)
    assert np.isclose(log.mode_attenuation_min_1pm, -0.19e-4)
    assert np.isclose(log.mode_attenuation_max_1pm, -0.21e-5)
    assert np.isclose(log.mode_phase_speed_min_mps, 1500.631428)
    assert np.isclose(log.mode_phase_speed_max_mps, 1505.710878)
    assert np.isclose(log.mode_group_speed_min_mps, 1494.69)
    assert np.isclose(log.mode_group_speed_max_mps, 1499.41)
    assert np.isclose(log.cpu_time_s, 0.000545)
    assert log.completed_successfully
    assert log.completed_without_errors
    assert log.parameters["Nominal Frequency"] == 250.0
    assert log.parameters["Max. number of restarts in root finder, MaxTries"] == 1.0


def test_read_at_prt_extracts_field_metadata(tmp_path):
    prt_path = tmp_path / "field.prt"
    prt_path.write_text(
        "\n"
        " __________________________________________________________________________\n"
        " Running FIELD\n"
        " Sums modes, producing pressure\n"
        "\n"
        " Point source\n"
        " Omnidirectional source\n"
        " Coherent mode addition\n"
        "\n"
        "    Number of Profile ranges, RProf =            1\n"
        "    Profile ranges, RProf (km)\n"
        "                 0.00000\n"
        "\n"
        " Range-independent calculation\n"
        "\n"
        "    Number of Receiver r-coordinates, Rr =         1001\n"
        "    Receiver r-coordinates, Rr (km)\n"
        "                0.100000      0.104900      0.109800      0.114700\n"
        "          ...\n"
        "   5.00000\n"
        "    Number of Source   z-coordinates, Sz =            1\n"
        "    Source   z-coordinates, Sz (m)\n"
        "                 50.0000\n"
        " __________________________________________________________________________\n"
        "    Number of Receiver z-coordinates, Rz =         1001\n"
        "    Receiver z-coordinates, Rz (m)\n"
        "                 0.00000      0.100000      0.200000      0.300000\n"
        "          ...\n"
        "   100.000\n"
        "    Number of receiver range offsets (array tilt), Rro =         1001\n"
        "    receiver range offsets (array tilt), Rro (m)\n"
        "                 0.00000       0.00000       0.00000       0.00000\n"
        "          ...\n"
        "   0.00000\n"
        "\n"
        " Field completed successfully\n"
    )

    log = read_at_prt(prt_path)

    assert log.title == "Running FIELD"
    assert log.program == "Running FIELD"
    assert log.case_title == "Sums modes, producing pressure"
    assert log.source_description == "Point source"
    assert log.source_beam_description == "Omnidirectional source"
    assert log.mode_addition_description == "Coherent mode addition"
    assert log.calculation_description == "Range-independent calculation"
    assert log.profile_range_count == 1
    assert np.isclose(log.profile_range_min_km, 0.0)
    assert np.isclose(log.profile_range_max_km, 0.0)
    assert log.source_depth_count == 1
    assert np.isclose(log.source_depth_min_m, 50.0)
    assert np.isclose(log.source_depth_max_m, 50.0)
    assert log.receiver_depth_count == 1001
    assert log.receiver_range_count == 1001
    assert np.isclose(log.receiver_depth_min_m, 0.0)
    assert np.isclose(log.receiver_depth_max_m, 100.0)
    assert np.isclose(log.receiver_range_min_km, 0.1)
    assert np.isclose(log.receiver_range_max_km, 5.0)
    assert log.receiver_range_offset_count == 1001
    assert np.isclose(log.receiver_range_offset_min_m, 0.0)
    assert np.isclose(log.receiver_range_offset_max_m, 0.0)
    assert log.completed_successfully
    assert log.completed_without_errors
    assert log.parameters["Number of Profile ranges, RProf"] == 1.0
    assert log.parameters["Number of receiver range offsets (array tilt), Rro"] == 1001.0


def test_read_at_prt_extracts_field3d_metadata(tmp_path):
    prt_path = tmp_path / "field3d.prt"
    prt_path.write_text(
        "\n"
        " __________________________________________________________________________\n"
        " Running FIELD3D\n"
        " Sums modes, producing pressure\n"
        "\n"
        " Option = STDFM\n"
        " Alpha scan =     0.5000       360.000          720\n"
        " Step =       250.000 m    NSTEPS =          1100\n"
        " Number of modes =           60\n"
        "    Number of Source   x-coordinates, Sx =            1\n"
        "    Source   x-coordinates, Sx (km)\n"
        "                 333.000\n"
        "    Number of Source   y-coordinates, Sy =            1\n"
        "    Source   y-coordinates, Sy (km)\n"
        "                 315.000\n"
        "    Number of Source   z-coordinates, Sz =            1\n"
        "    Source   z-coordinates, Sz (m)\n"
        "                 400.000\n"
        " __________________________________________________________________________\n"
        "    Number of Receiver z-coordinates, Rz =            1\n"
        "    Receiver z-coordinates, Rz (m)\n"
        "                 400.000\n"
        "    Number of Receiver r-coordinates, Rr =         1001\n"
        "    Receiver r-coordinates, Rr (km)\n"
        "                 0.00000      0.275000      0.550000      0.825000\n"
        "          ...\n"
        "   275.000\n"
        "    Number of Receiver theta-coordinates, theta =          181\n"
        "    Receiver theta-coordinates, theta (degrees)\n"
        "                 0.00000       2.00000       4.00000       6.00000\n"
        "          ...\n"
        "   360.000\n"
        " NNodes =          397\n"
        " NElts  =          727\n"
        "CPU time =      0.167    s\n"
        " Field3D completed successfully\n"
    )

    log = read_at_prt(prt_path)

    assert log.title == "Running FIELD3D"
    assert log.program == "Running FIELD3D"
    assert log.case_title == "Sums modes, producing pressure"
    assert log.field_option == "STDFM"
    assert np.isclose(log.field_alpha_min_deg, 0.5)
    assert np.isclose(log.field_alpha_max_deg, 360.0)
    assert log.field_alpha_count == 720
    assert np.isclose(log.field_step_m, 250.0)
    assert log.field_step_count == 1100
    assert log.mode_count == 60
    assert log.source_x_count == 1
    assert log.source_y_count == 1
    assert np.isclose(log.source_x_min_km, 333.0)
    assert np.isclose(log.source_x_max_km, 333.0)
    assert np.isclose(log.source_y_min_km, 315.0)
    assert np.isclose(log.source_y_max_km, 315.0)
    assert log.source_depth_count == 1
    assert np.isclose(log.source_depth_min_m, 400.0)
    assert np.isclose(log.source_depth_max_m, 400.0)
    assert log.receiver_depth_count == 1
    assert log.receiver_range_count == 1001
    assert log.receiver_theta_count == 181
    assert np.isclose(log.receiver_depth_min_m, 400.0)
    assert np.isclose(log.receiver_depth_max_m, 400.0)
    assert np.isclose(log.receiver_range_min_km, 0.0)
    assert np.isclose(log.receiver_range_max_km, 275.0)
    assert np.isclose(log.receiver_theta_min_deg, 0.0)
    assert np.isclose(log.receiver_theta_max_deg, 360.0)
    assert log.node_count == 397
    assert log.element_count == 727
    assert np.isclose(log.cpu_time_s, 0.167)
    assert log.completed_successfully
    assert log.completed_without_errors
    assert log.source_description is None
    assert log.parameters["Option"] == "STDFM"
    assert log.parameters["Number of modes"] == 60.0
    assert log.parameters["NNodes"] == 397.0


def test_read_at_prt_extracts_bellhop3d_metadata(tmp_path):
    prt_path = tmp_path / "bellhop3d.prt"
    prt_path.write_text(
        " BELLHOP/BELLHOP3D\n"
        "\n"
        " BELLHOP3D- Munk 3D Slanted Terrain Coherent TL\n"
        " frequency =   50.00     Hz\n"
        " Dummy parameter NMedia =            1\n"
        "     C-linear approximation to SSP\n"
        "     Attenuation units: dB/wavelength\n"
        "     VACUUM\n"
        " Depth =    5000.00 m\n"
        "                                 ( RMS roughness =   0.00     )\n"
        "     Bathymetry file selected\n"
        "     VACUUM\n"
        " __________________________________________________________________________\n"
        "    Number of Source   x-coordinates, Sx =            1\n"
        "    Source   x-coordinates, Sx (km)\n"
        "                 0.00000\n"
        "    Number of Source   y-coordinates, Sy =            1\n"
        "    Source   y-coordinates, Sy (km)\n"
        "                 0.00000\n"
        "    Number of Source   z-coordinates, Sz =            1\n"
        "    Source   z-coordinates, Sz (m)\n"
        "                 1000.00\n"
        " __________________________________________________________________________\n"
        "    Number of Receiver z-coordinates, Rz =          100\n"
        "    Number of Receiver r-coordinates, Rr =          100\n"
        "    Number of Receiver theta-coordinates, theta =            1\n"
        " Coherent TL calculation\n"
        " Geometric gaussian beams in Cartesian coordinates\n"
        " Point source (cylindrical coordinates)\n"
        " Rectilinear receiver grid: Receivers at Rr( : ) x Rz( : )\n"
        " 3D calculation\n"
        " __________________________________________________________________________\n"
        "    Number of beams in elevation   =           20\n"
        "    Beam take-off angles (degrees)\n"
        "  -30.0000      -20.0000       0.00000       20.0000       30.0000\n"
        "    Number of beams in bearing   =           24\n"
        "    Beam take-off angles (degrees)\n"
        "  -70.0000      -51.6667      -33.3333      -15.0000       3.33333\n"
        "   21.6667       40.0000\n"
        " Step length,        deltas =   10.00     m\n"
        " Maximum ray range (m):\n"
        "   Box%x =  0.1000E+05\n"
        "   Box%y =  0.1000E+05\n"
        "   Box%z =  0.2050E+05\n"
        " No beam shift in effect\n"
        " Using bottom-bathymetry file\n"
        " Regular grid for a 3D run\n"
        " Short format (bathymetry only)\n"
        " Number of bathymetry points in x           2\n"
        "  -120.000       120.000\n"
        " Number of bathymetry points in y           2\n"
        "  -60.000        60.000\n"
        " Source coordinate (m)\n"
        " xs =    0.00         0.00        0.100E+04\n"
        " Geometric beam, Gaussian-shaped, Cart. coord.\n"
        " HalfWidth1  =    0.0000000000000000\n"
        " HalfWidth2  =    0.0000000000000000\n"
        " EpsMult     =    1.0000000000000000\n"
        " CPU Time =       0.662E-01s\n"
    )

    log = read_at_prt(prt_path)

    assert log.program == "BELLHOP/BELLHOP3D"
    assert log.case_title == "BELLHOP3D- Munk 3D Slanted Terrain Coherent TL"
    assert np.isclose(log.frequency_hz, 50.0)
    assert log.nmedia == 1
    assert np.isclose(log.bottom_depth_m, 5000.0)
    assert log.attenuation_units == "dB/wavelength"
    assert log.source_x_count == 1
    assert log.source_y_count == 1
    assert log.source_depth_count == 1
    assert np.isclose(log.source_coordinate_x_m, 0.0)
    assert np.isclose(log.source_coordinate_y_m, 0.0)
    assert np.isclose(log.source_coordinate_z_m, 1000.0)
    assert log.receiver_depth_count == 100
    assert log.receiver_range_count == 100
    assert log.receiver_theta_count == 1
    assert log.run_type_description == "Coherent TL calculation"
    assert log.beam_description == "Geometric gaussian beams in Cartesian coordinates"
    assert log.source_description == "Point source (cylindrical coordinates)"
    assert log.receiver_grid_description == "Rectilinear receiver grid: Receivers at Rr( : ) x Rz( : )"
    assert log.calculation_description == "3D calculation"
    assert log.beam_count == 20
    assert log.beam_bearing_count == 24
    assert np.isclose(log.beam_angle_min_deg, -30.0)
    assert np.isclose(log.beam_angle_max_deg, 30.0)
    assert np.isclose(log.beam_bearing_angle_min_deg, -70.0)
    assert np.isclose(log.beam_bearing_angle_max_deg, 40.0)
    assert np.isclose(log.beam_halfwidth1, 0.0)
    assert np.isclose(log.beam_halfwidth2, 0.0)
    assert np.isclose(log.eps_multiplier, 1.0)
    assert np.isclose(log.step_length_m, 10.0)
    assert np.isclose(log.box_x_m, 10000.0)
    assert np.isclose(log.box_y_m, 10000.0)
    assert np.isclose(log.box_depth_m, 20500.0)
    assert log.beam_shift_description == "No beam shift in effect"
    assert log.bathymetry_x_count == 2
    assert log.bathymetry_y_count == 2
    assert np.isclose(log.bathymetry_x_min_km, -120.0)
    assert np.isclose(log.bathymetry_x_max_km, 120.0)
    assert np.isclose(log.bathymetry_y_min_km, -60.0)
    assert np.isclose(log.bathymetry_y_max_km, 60.0)
    assert np.isclose(log.cpu_time_s, 0.0662)
    assert log.parameters["Number of beams in bearing"] == 24.0
    assert log.parameters["Box%x"] == 10000.0


def test_generate_at_kraken_env_preserves_density_and_shear_fields(tmp_path):
    config = SimulationConfig(
        project="kraken elastic env",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0, density=1.0, attenuation=0.1),
                    SSPLayer(depth=100.0, c=1510.0, density=1.2, attenuation=0.2),
                ],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=100.0,
                c_p=3000.0,
                c_s=1200.0,
                density=2.0,
                attenuation_p=0.3,
                attenuation_s=0.4,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[20.0, 80.0]),
        ),
        solver=KrakenConfig(),
    )
    env_path = tmp_path / "case.env"

    generate_at_kraken_env(config, env_path)
    text = env_path.read_text()

    assert "0.0 1500.0 0.0 1.0 0.1 0.0 /" in text
    assert "100.0 1510.0 0.0 1.2 0.2 0.0 /" in text
    assert "100.0 3000.0 1200.0 2.0 0.3 0.4 /" in text


@pytest.mark.parametrize(
    ("ssp_type", "expected_option"),
    [
        ("c-linear", "'CVW'"),
        ("n2-linear", "'NVW'"),
        ("spline", "'SVW'"),
    ],
)
def test_generate_at_kraken_env_maps_ssp_interpolation_option(tmp_path, ssp_type, expected_option):
    config = SimulationConfig(
        project="kraken ssp interpolation option",
        frequency=75.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type=ssp_type,
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1510.0)],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=KrakenConfig(),
    )
    env_path = tmp_path / f"{ssp_type}.env"

    generate_at_kraken_env(config, env_path)

    assert env_path.read_text().splitlines()[3] == expected_option


def test_generate_at_kraken_env_writes_mesh_count_and_rmax(tmp_path):
    config = SimulationConfig(
        project="kraken mesh and rmax env",
        frequency=75.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[500.0, 2500.0], depths=[20.0, 80.0]),
        ),
        solver=KrakenConfig(n_mesh_points=777, phase_speed_limits=(1400.0, 1700.0)),
    )
    env_path = tmp_path / "mesh_rmax.env"

    generate_at_kraken_env(config, env_path)

    lines = env_path.read_text().splitlines()
    assert lines[4] == "777 0.0 100.0"
    assert lines[9] == "1400.0 1700.0"
    assert lines[10] == "2.5"


def test_generate_at_kraken_env_preserves_explicit_receiver_depths(tmp_path):
    config = SimulationConfig(
        project="kraken explicit receiver depths env",
        frequency=75.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[500.0, 2500.0], depths=[10.0, 40.0, 90.0]),
        ),
        solver=KrakenConfig(),
    )
    env_path = tmp_path / "explicit_depths.env"

    generate_at_kraken_env(config, env_path)

    lines = env_path.read_text().splitlines()
    assert lines[-2] == "3"
    assert lines[-1] == "10 40 90 /"


def test_generate_at_kraken_env_writes_bottom_reflection_table(tmp_path):
    config = SimulationConfig(
        project="kraken reflection env",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=100.0,
                c_p=1600.0,
                density=1.8,
                reflection_table=[(0.0, 0.8, 0.0), (90.0, 0.5, 180.0)],
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
        ),
        solver=KrakenConfig(),
    )
    env_path = tmp_path / "case.env"

    generate_at_kraken_env(config, env_path)

    lines = env_path.read_text().splitlines()
    bottom_idx = lines.index("'F' 0.0")
    assert lines[bottom_idx + 1] == "1450.0 1000000.0"
    table = read_at_brc(tmp_path / "case.brc")
    np.testing.assert_allclose(table.angles_deg, [0.0, 90.0])
    np.testing.assert_allclose(table.amplitudes, [0.8, 0.5])
    np.testing.assert_allclose(table.phases_deg, [0.0, 180.0])


def test_generate_at_kraken_env_omits_halfspace_for_vacuum_and_rigid_bottom(tmp_path):
    def make_config(bottom_type: str) -> SimulationConfig:
        return SimulationConfig(
            project=f"kraken {bottom_type} bottom env",
            frequency=50.0,
            environment=EnvironmentModel(
                ssp=SSPModel(
                    type="c-linear",
                    data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
                ),
                surface=BoundaryCondition(type="vacuum"),
                bottom=BoundaryCondition(
                    type=bottom_type,
                    depth=100.0,
                    c_p=3200.0,
                    c_s=1200.0,
                    density=2.2,
                    attenuation_p=0.3,
                    attenuation_s=0.4,
                ),
            ),
            geometry=GeometryModel(
                source=SourceModel(depths=[50.0]),
                receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
            ),
            solver=KrakenConfig(),
        )

    for bottom_type, option in (("vacuum", "'V' 0.0"), ("rigid", "'R' 0.0")):
        env_path = tmp_path / f"{bottom_type}.env"
        generate_at_kraken_env(make_config(bottom_type), env_path)

        lines = env_path.read_text().splitlines()
        bottom_idx = lines.index(option)
        assert lines[bottom_idx + 1] == "1450.0 1000000.0"
        assert "100.0 3200.0 1200.0 2.2 0.3 0.4 /" not in env_path.read_text()


def test_generate_at_kraken_env_writes_rigid_surface_option(tmp_path):
    config = SimulationConfig(
        project="kraken rigid surface env",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="rigid", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
        ),
        solver=KrakenConfig(),
    )
    env_path = tmp_path / "rigid_top.env"

    generate_at_kraken_env(config, env_path)

    assert env_path.read_text().splitlines()[3] == "'CRW'"


def test_generate_at_kraken_env_writes_surface_reflection_table(tmp_path):
    config = SimulationConfig(
        project="kraken surface reflection env",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(
                type="vacuum",
                depth=0.0,
                reflection_table=[(0.0, 1.0, 180.0), (90.0, 0.75, 90.0)],
            ),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
        ),
        solver=KrakenConfig(),
    )
    env_path = tmp_path / "surface_refl.env"

    generate_at_kraken_env(config, env_path)

    assert env_path.read_text().splitlines()[3] == "'CFW'"
    table = read_at_trc(tmp_path / "surface_refl.trc")
    np.testing.assert_allclose(table.angles_deg, [0.0, 90.0])
    np.testing.assert_allclose(table.amplitudes, [1.0, 0.75])
    np.testing.assert_allclose(table.phases_deg, [180.0, 90.0])


def test_generate_at_kraken_env_writes_bottom_roughness(tmp_path):
    config = SimulationConfig(
        project="kraken rough bottom env",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=100.0,
                c_p=1700.0,
                density=1.8,
                rms_roughness_m=0.25,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 1000.0], depths=[50.0]),
        ),
        solver=KrakenConfig(),
    )
    env_path = tmp_path / "rough_kraken.env"

    generate_at_kraken_env(config, env_path)

    assert "'A' 0.25" in env_path.read_text().splitlines()


def test_generate_at_field_flp_starts_profile_at_zero(tmp_path):
    from pyacoustics.solvers.external_io import generate_at_field_flp

    config = SimulationConfig(
        project="field flp",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(type="c-linear", data=[SSPLayer(0.0, 1500.0), SSPLayer(100.0, 1500.0)]),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 5000.0], depths=[0.0, 100.0]),
        ),
        solver=KrakenConfig(),
    )
    flp_path = tmp_path / "case.flp"

    generate_at_field_flp(config, flp_path)
    lines = flp_path.read_text().splitlines()

    assert lines[4] == "0.0 /"
    assert lines[6] == "0.1 5 /"


def test_generate_at_field_flp_uses_configured_max_modes(tmp_path):
    from pyacoustics.solvers.external_io import generate_at_field_flp

    config = SimulationConfig(
        project="field max modes",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(type="c-linear", data=[SSPLayer(0.0, 1500.0), SSPLayer(100.0, 1500.0)]),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 5000.0], depths=[0.0, 100.0]),
        ),
        solver=KrakenConfig(max_modes=37),
    )
    flp_path = tmp_path / "max_modes.flp"

    generate_at_field_flp(config, flp_path)

    assert flp_path.read_text().splitlines()[2] == "37"
    assert read_at_field_flp(flp_path).max_modes == 37


def test_generate_at_field_flp_uses_configured_field_option(tmp_path):
    from pyacoustics.solvers.external_io import generate_at_field_flp

    config = SimulationConfig(
        project="field option",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(type="c-linear", data=[SSPLayer(0.0, 1500.0), SSPLayer(100.0, 1500.0)]),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 5000.0], depths=[0.0, 100.0]),
        ),
        solver=KrakenConfig(field_option="RC"),
    )
    flp_path = tmp_path / "field_option.flp"

    generate_at_field_flp(config, flp_path)

    assert flp_path.read_text().splitlines()[1] == "'RC'"
    assert read_at_field_flp(flp_path).option == "RC"


def test_generate_at_field_flp_uses_configured_receiver_range_offsets(tmp_path):
    from pyacoustics.solvers.external_io import generate_at_field_flp

    config = SimulationConfig(
        project="field receiver offsets",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(type="c-linear", data=[SSPLayer(0.0, 1500.0), SSPLayer(100.0, 1500.0)]),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 5000.0], depths=[0.0, 50.0, 100.0]),
        ),
        solver=KrakenConfig(field_receiver_range_offsets_m=[-2.0, 0.0, 3.0]),
    )
    flp_path = tmp_path / "receiver_offsets.flp"

    generate_at_field_flp(config, flp_path)

    lines = flp_path.read_text().splitlines()
    assert lines[11] == "3"
    assert lines[12] == "-2 0 3 /"
    np.testing.assert_allclose(read_at_field_flp(flp_path).receiver_range_offsets_m, [-2.0, 0.0, 3.0])


def test_generate_at_field_flp_uses_configured_profile_ranges(tmp_path):
    from pyacoustics.solvers.external_io import generate_at_field_flp

    config = SimulationConfig(
        project="field profile ranges",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(type="c-linear", data=[SSPLayer(0.0, 1500.0), SSPLayer(100.0, 1500.0)]),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[100.0, 5000.0], depths=[0.0, 100.0]),
        ),
        solver=KrakenConfig(field_profile_ranges_km=[0.0, 2.5, 5.0]),
    )
    flp_path = tmp_path / "profile_ranges.flp"

    generate_at_field_flp(config, flp_path)

    lines = flp_path.read_text().splitlines()
    assert lines[3] == "3"
    assert lines[4] == "0 2.5 5 /"
    flp = read_at_field_flp(flp_path)
    assert flp.profile_range_count == 3
    np.testing.assert_allclose(flp.profile_ranges_km, [0.0, 2.5, 5.0])


def test_read_at_field_flp_expands_2d_endpoint_grids():
    flp_path = Path(__file__).resolve().parents[1] / "Munk" / "munk_at.flp"

    flp = read_at_field_flp(flp_path)

    assert flp.title == "Munk Profile FLP"
    assert flp.option == "RA"
    assert flp.max_modes == 9999
    assert flp.is_3d is False
    assert flp.profile_range_count == 1
    np.testing.assert_allclose(flp.profile_ranges_km, [0.0])
    assert flp.receiver_range_count == 200
    np.testing.assert_allclose(flp.receiver_ranges_km[[0, -1]], [0.1, 100.0])
    assert flp.source_depth_count == 1
    np.testing.assert_allclose(flp.source_depths_m, [1000.0])
    assert flp.receiver_depth_count == 100
    np.testing.assert_allclose(flp.receiver_depths_m[[0, -1]], [0.0, 5000.0])
    assert flp.receiver_range_offset_count == 100
    np.testing.assert_allclose(flp.receiver_range_offsets_m, np.zeros(100))


def test_generate_at_field_flp_preserves_explicit_receiver_grids(tmp_path):
    from pyacoustics.solvers.external_io import generate_at_field_flp

    config = SimulationConfig(
        project="field explicit grids",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(type="c-linear", data=[SSPLayer(0.0, 1500.0), SSPLayer(100.0, 1500.0)]),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[25.0, 50.0]),
            receivers=ReceiverModel(ranges=[100.0, 300.0, 900.0], depths=[10.0, 40.0, 90.0]),
        ),
        solver=KrakenConfig(),
    )
    flp_path = tmp_path / "explicit.flp"

    generate_at_field_flp(config, flp_path)
    lines = flp_path.read_text().splitlines()

    assert lines[5] == "3"
    assert lines[6] == "0.1 0.3 0.9 /"
    assert lines[7] == "2"
    assert lines[8] == "25 50 /"
    assert lines[9] == "3"
    assert lines[10] == "10 40 90 /"
    assert lines[11] == "3"
    assert lines[12] == "0 0 0 /"

    flp = read_at_field_flp(flp_path)
    assert flp.receiver_range_count == 3
    np.testing.assert_allclose(flp.receiver_ranges_km, [0.1, 0.3, 0.9])
    assert flp.receiver_depth_count == 3
    np.testing.assert_allclose(flp.receiver_depths_m, [10.0, 40.0, 90.0])
    assert flp.receiver_range_offset_count == 3
    np.testing.assert_allclose(flp.receiver_range_offsets_m, [0.0, 0.0, 0.0])


def test_generate_at_field_flp_writes_field3d_config(tmp_path):
    from pyacoustics.solvers.external_io import generate_at_field_flp

    config = SimulationConfig(
        project="field3d config",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(type="c-linear", data=[SSPLayer(0.0, 1500.0), SSPLayer(100.0, 1500.0)]),
            surface=BoundaryCondition(type="vacuum"),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0], x=[1000.0], y=[2000.0]),
            receivers=ReceiverModel(ranges=[0.0, 5000.0], depths=[0.0, 100.0]),
        ),
        solver=KrakenConfig(
            field_option="STDFM",
            max_modes=42,
            field_receiver_thetas_deg=[0.0, 90.0, 180.0],
            field_node_x_km=[0.0, 1.0, 0.0],
            field_node_y_km=[0.0, 0.0, 1.0],
            field_node_env_files=["env001", "env002", "env003"],
            field_elements=[(1, 2, 3)],
            field_alpha_scan=(0.5, 360.0, 720),
            field_step=(250.0, 1100),
            field_epmult=1.25,
        ),
    )
    flp_path = tmp_path / "field3d_config.flp"

    generate_at_field_flp(config, flp_path)
    flp = read_at_field_flp(flp_path)

    assert flp.is_3d
    assert flp.option == "STDFM"
    assert flp.max_modes == 42
    np.testing.assert_allclose(flp.source_x_km, [1.0])
    np.testing.assert_allclose(flp.source_y_km, [2.0])
    np.testing.assert_allclose(flp.receiver_ranges_km[[0, -1]], [0.0, 5.0])
    np.testing.assert_allclose(flp.receiver_depths_m[[0, -1]], [0.0, 100.0])
    np.testing.assert_allclose(flp.receiver_thetas_deg, [0.0, 90.0, 180.0])
    assert flp.node_env_files == ["env001", "env002", "env003"]
    np.testing.assert_array_equal(flp.elements, np.array([[1, 2, 3]]))
    assert np.isclose(flp.alpha_min_deg, 0.5)
    assert np.isclose(flp.alpha_max_deg, 360.0)
    assert flp.alpha_count == 720
    assert np.isclose(flp.step_m, 250.0)
    assert flp.step_count == 1100
    assert np.isclose(flp.epmult, 1.25)


def test_read_at_field_flp_extracts_field3d_grids_and_nodes():
    flp_path = Path(__file__).resolve().parents[1] / "3DAtlantic" / "lant_90degrees.flp"

    flp = read_at_field_flp(flp_path)

    assert flp.title == ""
    assert flp.option == "STDFM"
    assert flp.max_modes == 60
    assert flp.is_3d is True
    assert flp.source_x_count == 1
    np.testing.assert_allclose(flp.source_x_km, [333.0])
    assert flp.source_y_count == 1
    np.testing.assert_allclose(flp.source_y_km, [315.0])
    assert flp.source_depth_count == 1
    np.testing.assert_allclose(flp.source_depths_m, [100.0])
    assert flp.receiver_depth_count == 201
    np.testing.assert_allclose(flp.receiver_depths_m[[0, -1]], [0.0, 5000.0])
    assert flp.receiver_range_count == 1001
    np.testing.assert_allclose(flp.receiver_ranges_km[[0, -1]], [0.0, 275.0])
    assert flp.receiver_theta_count == 1
    np.testing.assert_allclose(flp.receiver_thetas_deg, [90.0])
    assert flp.node_count == 397
    np.testing.assert_allclose(flp.node_x_km[:2], [0.0, 15.54])
    np.testing.assert_allclose(flp.node_y_km[:2], [648.63, 667.16])
    assert flp.node_env_files[:2] == ["lante02", "lante02"]
    assert flp.element_count == 727
    assert flp.elements.shape == (727, 3)
    np.testing.assert_array_equal(flp.elements[0], [1, 2, 3])
    np.testing.assert_array_equal(flp.elements[-1], [392, 393, 394])
    assert np.isclose(flp.alpha_min_deg, 0.5)
    assert np.isclose(flp.alpha_max_deg, 360.0)
    assert flp.alpha_count == 720
    assert np.isclose(flp.step_m, 250.0)
    assert flp.step_count == 1100
    assert np.isclose(flp.epmult, 1.0)


def test_read_at_field_flp_supports_compact_2d_range_only_layout(tmp_path):
    flp_path = tmp_path / "slice2d.flp"
    flp_path.write_text(
        "'RBO'  ! 'R/X (coord), Pos/Neg/Both'\n"
        "1001\n"
        "0.0 10.0 /\n",
        encoding="utf-8",
    )

    flp = read_at_field_flp(flp_path)
    roundtrip_path = tmp_path / "slice2d_roundtrip.flp"
    write_at_field_flp(flp, roundtrip_path)
    reparsed = read_at_field_flp(roundtrip_path)

    assert flp.title == ""
    assert flp.option == "RBO"
    assert flp.max_modes == 0
    assert flp.is_3d is False
    assert flp.receiver_range_count == 1001
    np.testing.assert_allclose(flp.receiver_ranges_km[[0, -1]], [0.0, 10.0])
    assert reparsed.option == flp.option
    assert reparsed.receiver_range_count == flp.receiver_range_count
    np.testing.assert_allclose(reparsed.receiver_ranges_km, flp.receiver_ranges_km)


def test_read_at_field_flp_supports_two_line_compact_range_grid(tmp_path):
    flp_path = tmp_path / "hs_s.flp"
    flp_path.write_text(
        "'HBO' ! 'R/X (coord), Pos/Neg/Both'\n"
        "0.0 1.0 1000 ! RMIN, RMAX, NR\n",
        encoding="utf-8",
    )

    flp = read_at_field_flp(flp_path)
    roundtrip_path = tmp_path / "hs_s_roundtrip.flp"
    write_at_field_flp(flp, roundtrip_path)
    reparsed = read_at_field_flp(roundtrip_path)

    assert flp.title == ""
    assert flp.option == "HBO"
    assert flp.max_modes == 0
    assert flp.is_3d is False
    assert flp.receiver_range_count == 1000
    assert flp.receiver_ranges_km.size == 1000
    np.testing.assert_allclose(flp.receiver_ranges_km[[0, -1]], [0.0, 1.0])
    assert roundtrip_path.read_text(encoding="utf-8").splitlines() == ["'HBO'", "0 1 1000"]
    assert reparsed.option == flp.option
    assert reparsed.receiver_range_count == flp.receiver_range_count
    np.testing.assert_allclose(reparsed.receiver_ranges_km, flp.receiver_ranges_km)


def test_read_at_field_flp_extracts_field3d_omni_and_volume_variants():
    omni = read_at_field_flp(Path(__file__).resolve().parents[1] / "3DAtlantic" / "lant.flp")
    volume = read_at_field_flp(Path(__file__).resolve().parents[1] / "3DAtlantic" / "lantvol.flp")
    pacific = read_at_field_flp(Path(__file__).resolve().parents[1] / "3DAtlantic" / "pacific.flp")
    gbt = read_at_field_flp(Path(__file__).resolve().parents[1] / "3DAtlantic" / "lantgbt.flp")

    assert omni.option == "STDFM"
    assert omni.receiver_theta_count == 181
    np.testing.assert_allclose(omni.receiver_thetas_deg[[0, -1]], [0.0, 360.0])
    assert omni.node_count == 397
    assert omni.element_count == 727

    assert volume.option == "STDFM"
    assert volume.source_x_count == 1
    assert volume.source_y_count == 1
    np.testing.assert_allclose(volume.source_x_km, [333.0])
    np.testing.assert_allclose(volume.source_y_km, [315.0])
    assert volume.receiver_depth_count == 26
    np.testing.assert_allclose(volume.receiver_depths_m[[0, -1]], [0.0, 5000.0])
    assert volume.receiver_range_count == 251
    np.testing.assert_allclose(volume.receiver_ranges_km[[0, -1]], [0.0, 275.0])
    assert volume.receiver_theta_count == 37
    np.testing.assert_allclose(volume.receiver_thetas_deg[[0, -1]], [0.0, 360.0])
    assert volume.node_count == 397
    assert volume.element_count == 727
    assert volume.alpha_count == 720
    assert volume.step_count == 1100

    assert pacific.title == ""
    assert pacific.option == "STDRM"
    assert pacific.source_x_count == 1
    assert pacific.source_y_count == 1
    np.testing.assert_allclose(pacific.source_x_km, [0.001])
    np.testing.assert_allclose(pacific.source_y_km, [0.001])
    assert pacific.receiver_range_count == 501
    np.testing.assert_allclose(pacific.receiver_ranges_km[[0, -1]], [0.0, 400.0])
    assert pacific.receiver_theta_count == 37
    np.testing.assert_allclose(pacific.receiver_thetas_deg[[0, -1]], [-135.0, 45.0])
    assert pacific.node_count == 291
    assert pacific.element_count == 525
    assert pacific.alpha_count == 90
    assert pacific.step_count == 600

    assert gbt.option == "GBTFM"
    assert gbt.is_3d is True
    assert gbt.receiver_theta_count == 181
    assert gbt.node_count == 397
    assert gbt.element_count == 727


def test_read_at_field_flp_handles_multiline_endpoint_grids():
    flp = read_at_field_flp(Path(__file__).resolve().parents[1] / "Bellhop3DTests" / "KoreanSeas" / "KoreanSea_3D.flp")

    assert flp.title == "Korean Sea 3D (3D run)"
    assert flp.option == "STDFM"
    assert flp.source_x_count == 2
    assert flp.source_y_count == 2
    np.testing.assert_allclose(flp.source_x_km, [950000.0, 1050000.0])
    np.testing.assert_allclose(flp.source_y_km, [750000.0, 900000.0])
    assert flp.receiver_depth_count == 251
    np.testing.assert_allclose(flp.receiver_depths_m[[0, -1]], [0.0, 2500.0])
    assert flp.receiver_range_count == 501
    np.testing.assert_allclose(flp.receiver_ranges_km[[0, -1]], [0.0, 50.0])
    assert flp.receiver_theta_count == 37
    np.testing.assert_allclose(flp.receiver_thetas_deg[[0, -1]], [0.0, 360.0])
    assert flp.node_count == 4891
    assert flp.element_count == 9504
    np.testing.assert_array_equal(flp.elements[0], [1, 2, 74])
    np.testing.assert_array_equal(flp.elements[-1], [4818, 4890, 4891])


def test_write_at_field_flp_roundtrips_2d_and_field3d(tmp_path):
    source_2d = Path(__file__).resolve().parents[1] / "Munk" / "munk_at.flp"
    source_3d = Path(__file__).resolve().parents[1] / "3DAtlantic" / "lant_90degrees.flp"

    flp_2d = read_at_field_flp(source_2d)
    flp_2d_path = tmp_path / "munk_roundtrip.flp"
    write_at_field_flp(flp_2d, flp_2d_path)
    roundtrip_2d = read_at_field_flp(flp_2d_path)

    assert roundtrip_2d.title == flp_2d.title
    assert roundtrip_2d.option == flp_2d.option
    assert roundtrip_2d.max_modes == flp_2d.max_modes
    assert roundtrip_2d.profile_range_count == flp_2d.profile_range_count
    assert roundtrip_2d.receiver_range_count == flp_2d.receiver_range_count
    np.testing.assert_allclose(roundtrip_2d.receiver_ranges_km, flp_2d.receiver_ranges_km)
    np.testing.assert_allclose(roundtrip_2d.receiver_depths_m, flp_2d.receiver_depths_m)
    np.testing.assert_allclose(roundtrip_2d.receiver_range_offsets_m, flp_2d.receiver_range_offsets_m)

    flp_3d = read_at_field_flp(source_3d)
    flp_3d_path = tmp_path / "lant90_roundtrip.flp"
    write_at_field_flp(flp_3d, flp_3d_path)
    roundtrip_3d = read_at_field_flp(flp_3d_path)

    assert roundtrip_3d.title == flp_3d.title
    assert roundtrip_3d.option == flp_3d.option
    assert roundtrip_3d.max_modes == flp_3d.max_modes
    assert roundtrip_3d.source_x_count == flp_3d.source_x_count
    assert roundtrip_3d.source_y_count == flp_3d.source_y_count
    assert roundtrip_3d.receiver_theta_count == flp_3d.receiver_theta_count
    assert roundtrip_3d.node_count == flp_3d.node_count
    assert roundtrip_3d.element_count == flp_3d.element_count
    assert roundtrip_3d.alpha_count == flp_3d.alpha_count
    assert roundtrip_3d.step_count == flp_3d.step_count
    np.testing.assert_allclose(roundtrip_3d.source_x_km, flp_3d.source_x_km)
    np.testing.assert_allclose(roundtrip_3d.source_y_km, flp_3d.source_y_km)
    np.testing.assert_allclose(roundtrip_3d.receiver_thetas_deg, flp_3d.receiver_thetas_deg)
    np.testing.assert_allclose(roundtrip_3d.node_x_km, flp_3d.node_x_km)
    np.testing.assert_allclose(roundtrip_3d.node_y_km, flp_3d.node_y_km)
    assert roundtrip_3d.node_env_files == flp_3d.node_env_files
    np.testing.assert_array_equal(roundtrip_3d.elements, flp_3d.elements)
    assert np.isclose(roundtrip_3d.epmult, flp_3d.epmult)


def test_write_at_field_flp_roundtrips_all_repository_samples(tmp_path):
    scalar_fields = [
        "title",
        "option",
        "max_modes",
        "is_3d",
        "profile_range_count",
        "receiver_range_count",
        "source_depth_count",
        "receiver_depth_count",
        "receiver_range_offset_count",
        "source_x_count",
        "source_y_count",
        "receiver_theta_count",
        "node_count",
        "node_env_files",
        "element_count",
        "alpha_min_deg",
        "alpha_max_deg",
        "alpha_count",
        "step_m",
        "step_count",
        "epmult",
    ]
    array_fields = [
        "profile_ranges_km",
        "receiver_ranges_km",
        "source_depths_m",
        "receiver_depths_m",
        "receiver_range_offsets_m",
        "source_x_km",
        "source_y_km",
        "receiver_thetas_deg",
        "node_x_km",
        "node_y_km",
        "elements",
    ]
    sample_paths = sorted(Path(__file__).resolve().parents[1].rglob("*.flp"))

    assert sample_paths
    for source_path in sample_paths:
        parsed = read_at_field_flp(source_path)
        roundtrip_path = tmp_path / source_path.relative_to(Path(__file__).resolve().parents[1])
        roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
        write_at_field_flp(parsed, roundtrip_path)
        reparsed = read_at_field_flp(roundtrip_path)

        for field_name in scalar_fields:
            assert getattr(reparsed, field_name) == getattr(parsed, field_name), field_name
        for field_name in array_fields:
            expected = getattr(parsed, field_name)
            actual = getattr(reparsed, field_name)
            if expected is None:
                assert actual is None, field_name
            else:
                np.testing.assert_allclose(actual, expected, err_msg=field_name)


def test_write_at_mod_roundtrip(tmp_path):
    z_bins = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    modes = np.array(
        [
            [1.0 + 0.0j, 0.5 + 0.1j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.2 - 0.1j, 1.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    mod_path = tmp_path / "native.mod"

    write_at_mod(mod_path, modes, z_bins)
    parsed_modes, parsed_z = read_at_mod(mod_path)

    np.testing.assert_allclose(parsed_z, z_bins)
    np.testing.assert_allclose(parsed_modes, modes)


def test_write_at_mod_wavenumbers_roundtrip(tmp_path):
    z_bins = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    modes = np.ones((2, 3), dtype=np.complex64)
    wavenumbers = np.array([0.1 + 0.001j, 0.08 + 0.002j], dtype=np.complex64)
    mod_path = tmp_path / "native_with_k.mod"

    write_at_mod(mod_path, modes, z_bins, wavenumbers=wavenumbers)

    np.testing.assert_allclose(read_at_mod_wavenumbers(mod_path), wavenumbers)


def test_write_at_mod_metadata_roundtrip(tmp_path):
    z_bins = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    modes = np.ones((2, 4), dtype=np.complex64)
    metadata = {
        "freq_vec": [50.0],
        "media": [
            {"n": 2, "material": "ACOUSTIC", "depth": 0.0, "rho": 1.0},
            {"n": 1, "material": "ACOUSTIC", "depth": 20.0, "rho": 1.5},
        ],
        "top": {"bc": "V", "c_p": 0.0, "c_s": 0.0, "rho": 0.0, "depth": 0.0},
        "bottom": {"bc": "A", "c_p": 1600.0, "c_s": 0.0, "rho": 1.8, "depth": 30.0},
    }
    mod_path = tmp_path / "native_metadata.mod"

    write_at_mod(mod_path, modes, z_bins, wavenumbers=np.array([0.1, 0.2], dtype=np.complex64), metadata=metadata)
    parsed = read_at_mod_metadata(mod_path)

    assert parsed["nfreq"] == 1
    assert parsed["nmedia"] == 2
    assert parsed["ntot"] == 4
    assert parsed["nmat"] == 4
    assert parsed["mode_count"] == 2
    np.testing.assert_allclose(parsed["freq_vec"], [50.0])
    np.testing.assert_allclose(parsed["z_bins"], z_bins)
    assert parsed["media"][0]["n"] == 2
    assert parsed["media"][0]["material"] == "ACOUSTIC"
    np.testing.assert_allclose(parsed["media"][1]["depth"], 20.0)
    np.testing.assert_allclose(parsed["media"][1]["rho"], 1.5)
    assert parsed["top"]["bc"] == "V"
    np.testing.assert_allclose(parsed["top"]["c_p"], 0.0 + 0.0j)
    np.testing.assert_allclose(parsed["top"]["c_s"], 0.0 + 0.0j)
    np.testing.assert_allclose(parsed["top"]["rho"], 0.0)
    np.testing.assert_allclose(parsed["top"]["depth"], 0.0)
    assert parsed["bottom"]["bc"] == "A"
    np.testing.assert_allclose(parsed["bottom"]["c_p"], 1600.0 + 0.0j)
    np.testing.assert_allclose(parsed["bottom"]["c_s"], 0.0 + 0.0j)
    np.testing.assert_allclose(parsed["bottom"]["rho"], 1.8)
    np.testing.assert_allclose(parsed["bottom"]["depth"], 30.0)


def test_write_at_mod_metadata_preserves_complex_halfspaces_and_multifrequency(tmp_path):
    z_bins = np.array([0.0, 20.0, 40.0, 60.0], dtype=np.float32)
    modes = np.ones((3, 4), dtype=np.complex64)
    metadata = {
        "freq_vec": [25.0, 50.0],
        "media": [
            {"n": 2, "material": "ACOUSTIC", "depth": 0.0, "rho": 1.0},
            {"n": 1, "material": "ELASTIC", "depth": 40.0, "rho": 2.1},
        ],
        "top": {
            "bc": "A",
            "c_p": 1480.0 + 0.25j,
            "c_s": 0.0 + 0.0j,
            "rho": 1.0,
            "depth": 0.0,
        },
        "bottom": {
            "bc": "A",
            "c_p": 1700.0 + 0.75j,
            "c_s": 600.0 + 0.125j,
            "rho": 2.2,
            "depth": 60.0,
        },
    }
    wavenumbers = np.array([0.15 + 0.001j, 0.12 + 0.002j, 0.1 + 0.003j], dtype=np.complex64)
    mod_path = tmp_path / "native_complex_metadata.mod"

    write_at_mod(mod_path, modes, z_bins, wavenumbers=wavenumbers, metadata=metadata)
    parsed = read_at_mod_metadata(mod_path)

    assert parsed["nfreq"] == 2
    assert parsed["nmedia"] == 2
    assert parsed["mode_count"] == 3
    np.testing.assert_allclose(parsed["freq_vec"], [25.0, 50.0])
    np.testing.assert_allclose(parsed["z_bins"], z_bins)
    assert parsed["media"][1]["material"] == "ELASTIC"
    np.testing.assert_allclose(parsed["media"][1]["depth"], 40.0)
    np.testing.assert_allclose(parsed["media"][1]["rho"], 2.1)
    assert parsed["top"]["bc"] == "A"
    np.testing.assert_allclose(parsed["top"]["c_p"], 1480.0 + 0.25j)
    np.testing.assert_allclose(parsed["top"]["c_s"], 0.0 + 0.0j)
    np.testing.assert_allclose(parsed["bottom"]["c_p"], 1700.0 + 0.75j)
    np.testing.assert_allclose(parsed["bottom"]["c_s"], 600.0 + 0.125j)
    np.testing.assert_allclose(parsed["bottom"]["rho"], 2.2)
    np.testing.assert_allclose(parsed["bottom"]["depth"], 60.0)
    np.testing.assert_allclose(read_at_mod_wavenumbers(mod_path), wavenumbers)


def test_write_at_shd_roundtrip(tmp_path):
    depths = np.array([10.0, 50.0, 90.0], dtype=np.float32)
    ranges = np.array([100.0, 500.0], dtype=np.float64)
    pressure = np.array(
        [
            [1.0 + 0.5j, 0.25 - 0.1j],
            [0.3 + 0.2j, -0.4 + 0.05j],
            [0.0 - 0.1j, 0.8 + 0.0j],
        ],
        dtype=np.complex128,
    )
    shd_path = tmp_path / "native.shd"

    write_at_shd(shd_path, pressure, depths, ranges, title="native shd", frequency=100.0)
    parsed_pressure, parsed_depths, parsed_ranges = read_at_shd(shd_path)

    np.testing.assert_allclose(parsed_depths, depths)
    np.testing.assert_allclose(parsed_ranges, ranges)
    np.testing.assert_allclose(parsed_pressure, pressure)


def test_write_at_shd_uses_at_binary_layout(tmp_path):
    depths = np.array([10.0], dtype=np.float32)
    ranges = np.array([100.0, 200.0], dtype=np.float64)
    pressure = np.array([[1.0 + 0.5j, 0.25 - 0.125j]], dtype=np.complex128)
    shd_path = tmp_path / "native_at_layout.shd"

    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="native AT layout",
        frequency=100.0,
        bearings_deg=np.array([30.0]),
        source_x_m=np.array([250.0]),
        source_y_m=np.array([100.0]),
        attenuation=0.75,
    )

    with shd_path.open("rb") as f:
        lrecl = struct.unpack("<i", f.read(4))[0]
        rec_size = 4 * lrecl

        def read_rec(n):
            f.seek((n - 1) * rec_size)
            return f.read(rec_size)

        assert lrecl >= 41
        assert read_rec(2)[:10] == b"rectilin  "
        dims_and_header = struct.unpack("<7idd", read_rec(3)[:44])
        assert dims_and_header[:7] == (1, 1, 1, 1, 1, 1, 2)
        assert dims_and_header[7:] == (100.0, 0.75)
        assert struct.unpack("<d", read_rec(4)[:8])[0] == 100.0
        assert struct.unpack("<d", read_rec(5)[:8])[0] == 30.0
        assert struct.unpack("<d", read_rec(6)[:8])[0] == 250.0
        assert struct.unpack("<d", read_rec(7)[:8])[0] == 100.0
        pressure_pairs = np.frombuffer(read_rec(11)[:16], dtype="<f4")
        np.testing.assert_allclose(pressure_pairs, [1.0, 0.5, 0.25, -0.125])

    shade = read_at_shd_full(shd_path)
    assert shade.pressure.dtype == np.complex64
    np.testing.assert_allclose(shade.pressure[0, 0], pressure.astype(np.complex64))


def test_read_at_shd_full_handles_field_blank_plot_type_float_frequency(tmp_path):
    shd_path = tmp_path / "field_blank_plot_type.shd"
    rec_size = 164
    lrecl = rec_size // 4
    records = [bytearray(rec_size) for _ in range(11)]
    records[0][:4] = struct.pack("<i", lrecl)
    records[0][4:84] = b"FIELD blank PlotType".ljust(80, b" ")
    records[1][:10] = b"          "
    records[2][:28] = struct.pack("<7i", 1, 1, 1, 1, 1, 1, 2)
    records[3][:4] = struct.pack("<f", 250.0)
    records[4][:8] = struct.pack("<d", 0.0)
    records[5][:8] = struct.pack("<d", 0.0)
    records[6][:8] = struct.pack("<d", 0.0)
    records[7][:4] = struct.pack("<f", 50.0)
    records[8][:4] = struct.pack("<f", 25.0)
    records[9][:16] = np.array([100.0, 500.0], dtype="<f8").tobytes()
    records[10][:16] = np.array([1.0, 0.25, 0.5, -0.125], dtype="<f4").tobytes()
    with shd_path.open("wb") as f:
        for record in records:
            f.write(record)

    shade = read_at_shd_full(shd_path)
    pressure, depths, ranges = read_at_shd(shd_path)

    assert shade.plot_type is None
    assert np.isclose(shade.frequency_hz, 250.0)
    np.testing.assert_allclose(shade.source_depths_m, [50.0])
    np.testing.assert_allclose(shade.receiver_depths_m, [25.0])
    np.testing.assert_allclose(shade.receiver_ranges_m, [100.0, 500.0])
    np.testing.assert_allclose(shade.pressure[0, 0, 0], [1.0 + 0.25j, 0.5 - 0.125j])
    np.testing.assert_allclose(pressure, [[1.0 + 0.25j, 0.5 - 0.125j]])
    np.testing.assert_allclose(depths, [25.0])
    np.testing.assert_allclose(ranges, [100.0, 500.0])


def test_write_at_shd_full_4d_roundtrip(tmp_path):
    depths = np.array([10.0, 50.0], dtype=np.float32)
    ranges = np.array([100.0, 500.0, 900.0], dtype=np.float64)
    source_depths = np.array([20.0, 40.0], dtype=np.float32)
    bearings = np.array([0.0, 45.0, 90.0], dtype=np.float32)
    pressure = np.zeros((3, 2, 2, 3), dtype=np.complex128)
    for itheta in range(3):
        for isd in range(2):
            for iz in range(2):
                pressure[itheta, isd, iz] = (
                    (itheta + 1) * 10.0
                    + (isd + 1)
                    + 0.1 * iz
                    + 1j * np.array([0.0, 0.5, 1.0])
                )
    shd_path = tmp_path / "native3d.shd"

    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="native 3d shd",
        frequency=250.0,
        source_depths_m=source_depths,
        bearings_deg=bearings,
    )
    shade = read_at_shd_full(shd_path)
    parsed_pressure, parsed_depths, parsed_ranges = read_at_shd(shd_path)

    assert shade.title == "native 3d shd"
    assert np.isclose(shade.frequency_hz, 250.0)
    np.testing.assert_allclose(shade.bearings_deg, bearings)
    np.testing.assert_allclose(shade.source_x_m, [0.0])
    np.testing.assert_allclose(shade.source_y_m, [0.0])
    np.testing.assert_allclose(shade.source_depths_m, source_depths)
    np.testing.assert_allclose(shade.receiver_depths_m, depths)
    np.testing.assert_allclose(shade.receiver_ranges_m, ranges)
    np.testing.assert_allclose(shade.pressure, pressure)
    np.testing.assert_allclose(shade.pressure_full, pressure[np.newaxis, np.newaxis])
    np.testing.assert_allclose(parsed_pressure, pressure[0, 0])
    np.testing.assert_allclose(parsed_depths, depths)
    np.testing.assert_allclose(parsed_ranges, ranges)


def test_write_at_shd_full_6d_roundtrip(tmp_path):
    depths = np.array([10.0, 50.0], dtype=np.float32)
    ranges = np.array([100.0, 500.0], dtype=np.float64)
    source_x = np.array([0.0, 250.0], dtype=np.float32)
    source_y = np.array([0.0, 100.0], dtype=np.float32)
    source_depths = np.array([20.0], dtype=np.float32)
    bearings = np.array([0.0, 45.0], dtype=np.float32)
    pressure = np.zeros((2, 2, 2, 1, 2, 2), dtype=np.complex128)
    for isx in range(2):
        for isy in range(2):
            for itheta in range(2):
                pressure[isx, isy, itheta, 0] = (
                    100.0 * isx
                    + 10.0 * isy
                    + itheta
                    + np.array([[1.0, 2.0], [3.0, 4.0]])
                    + 1j * np.array([[0.1, 0.2], [0.3, 0.4]])
                )
    shd_path = tmp_path / "native3d_source_xy.shd"

    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="native source xy shd",
        frequency=125.0,
        source_x_m=source_x,
        source_y_m=source_y,
        source_depths_m=source_depths,
        bearings_deg=bearings,
    )
    shade = read_at_shd_full(shd_path)

    assert shade.title == "native source xy shd"
    assert np.isclose(shade.frequency_hz, 125.0)
    np.testing.assert_allclose(shade.source_x_m, source_x)
    np.testing.assert_allclose(shade.source_y_m, source_y)
    np.testing.assert_allclose(shade.bearings_deg, bearings)
    np.testing.assert_allclose(shade.source_depths_m, source_depths)
    np.testing.assert_allclose(shade.receiver_depths_m, depths)
    np.testing.assert_allclose(shade.receiver_ranges_m, ranges)
    np.testing.assert_allclose(shade.pressure, pressure[0, 0])
    np.testing.assert_allclose(shade.pressure_full, pressure)


def test_write_at_shd_full_7d_multifrequency_roundtrip(tmp_path):
    depths = np.array([10.0, 50.0], dtype=np.float32)
    ranges = np.array([100.0, 500.0], dtype=np.float64)
    source_x = np.array([0.0], dtype=np.float64)
    source_y = np.array([0.0], dtype=np.float64)
    source_depths = np.array([20.0], dtype=np.float32)
    bearings = np.array([0.0, 45.0], dtype=np.float64)
    frequencies = np.array([100.0, 200.0], dtype=np.float64)
    pressure = np.zeros((2, 1, 1, 2, 1, 2, 2), dtype=np.complex128)
    for ifreq in range(2):
        for itheta in range(2):
            pressure[ifreq, 0, 0, itheta, 0] = (
                1000.0 * ifreq
                + 10.0 * itheta
                + np.array([[1.0, 2.0], [3.0, 4.0]])
                + 1j * np.array([[0.25, 0.5], [0.75, 1.0]])
            )
    shd_path = tmp_path / "native_broadband.shd"

    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="native broadband shd",
        frequencies_hz=frequencies,
        source_x_m=source_x,
        source_y_m=source_y,
        source_depths_m=source_depths,
        bearings_deg=bearings,
        attenuation=0.125,
    )
    shade = read_at_shd_full(shd_path)

    assert shade.title == "native broadband shd"
    assert shade.plot_type == "rectilin"
    assert np.isclose(shade.frequency_hz, frequencies[0])
    assert np.isclose(shade.attenuation, 0.125)
    np.testing.assert_allclose(shade.frequency_vector_hz, frequencies)
    np.testing.assert_allclose(shade.pressure, pressure[0, 0, 0].astype(np.complex64))
    np.testing.assert_allclose(shade.pressure_full, pressure[0].astype(np.complex64))
    np.testing.assert_allclose(shade.pressure_frequency_full, pressure.astype(np.complex64))

    with shd_path.open("rb") as f:
        lrecl = struct.unpack("<i", f.read(4))[0]
        rec_size = 4 * lrecl

        def read_rec(n):
            f.seek((n - 1) * rec_size)
            return f.read(rec_size)

        assert struct.unpack("<7idd", read_rec(3)[:44])[:7] == (2, 2, 1, 1, 1, 2, 2)
        np.testing.assert_allclose(np.frombuffer(read_rec(4)[:16], dtype="<f8"), frequencies)
        first_freq_first_bearing = np.frombuffer(read_rec(11)[:16], dtype="<f4")
        second_freq_first_bearing = np.frombuffer(read_rec(15)[:16], dtype="<f4")
        np.testing.assert_allclose(first_freq_first_bearing, [1.0, 0.25, 2.0, 0.5])
        np.testing.assert_allclose(second_freq_first_bearing, [1001.0, 0.25, 1002.0, 0.5])


def test_read_at_shd_full_selects_nearest_frequency_and_source_xy(tmp_path):
    depths = np.array([10.0], dtype=np.float32)
    ranges = np.array([100.0, 200.0], dtype=np.float64)
    source_x = np.array([0.0, 250.0], dtype=np.float64)
    source_y = np.array([0.0, 100.0], dtype=np.float64)
    source_depths = np.array([20.0], dtype=np.float32)
    bearings = np.array([0.0], dtype=np.float64)
    frequencies = np.array([100.0, 200.0], dtype=np.float64)
    pressure = np.zeros((2, 2, 2, 1, 1, 1, 2), dtype=np.complex128)
    for ifreq in range(2):
        for isx in range(2):
            for isy in range(2):
                pressure[ifreq, isx, isy, 0, 0, 0] = (
                    1000.0 * ifreq
                    + 100.0 * isx
                    + 10.0 * isy
                    + np.array([1.0, 2.0])
                    + 1j * np.array([0.25, 0.5])
                )
    shd_path = tmp_path / "native_broadband_grid.shd"

    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="native broadband grid shd",
        frequencies_hz=frequencies,
        source_x_m=source_x,
        source_y_m=source_y,
        source_depths_m=source_depths,
        bearings_deg=bearings,
    )
    shade = read_at_shd_full(
        shd_path,
        frequency_hz=190.0,
        source_x_m=240.0,
        source_y_m=90.0,
    )

    assert shade.selected_frequency_index == 1
    assert shade.selected_source_x_index == 1
    assert shade.selected_source_y_index == 1
    assert np.isclose(shade.frequency_hz, 200.0)
    np.testing.assert_allclose(shade.pressure, pressure[1, 1, 1].astype(np.complex64))
    np.testing.assert_allclose(shade.pressure_full, pressure[1].astype(np.complex64))
    np.testing.assert_allclose(shade.pressure_frequency_full, pressure.astype(np.complex64))


def test_write_at_shd_irregular_roundtrip(tmp_path):
    depths = np.array([10.0, 20.0, 40.0], dtype=np.float32)
    ranges = np.array([100.0, 250.0, 500.0], dtype=np.float64)
    pressure = np.array([[1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j]], dtype=np.complex128)
    shd_path = tmp_path / "native_irregular.shd"

    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="native irregular shd",
        frequency=75.0,
        plot_type="irregular",
    )
    shade = read_at_shd_full(shd_path)
    parsed_pressure, parsed_depths, parsed_ranges = read_at_shd(shd_path)

    assert shade.title == "native irregular shd"
    assert shade.plot_type == "irregular"
    assert shade.pressure.shape == (1, 1, 1, 3)
    np.testing.assert_allclose(shade.receiver_depths_m, depths)
    np.testing.assert_allclose(shade.receiver_ranges_m, ranges)
    np.testing.assert_allclose(shade.pressure[0, 0], pressure.astype(np.complex64))
    np.testing.assert_allclose(parsed_pressure, pressure.astype(np.complex64))
    np.testing.assert_allclose(parsed_depths, depths)
    np.testing.assert_allclose(parsed_ranges, ranges)

    with shd_path.open("rb") as f:
        lrecl = struct.unpack("<i", f.read(4))[0]
        rec_size = 4 * lrecl

        def read_rec(n):
            f.seek((n - 1) * rec_size)
            return f.read(rec_size)

        assert read_rec(2)[:10] == b"irregular "
        assert struct.unpack("<7idd", read_rec(3)[:44])[:7] == (1, 1, 1, 1, 1, 3, 3)
        pressure_pairs = np.frombuffer(read_rec(11)[:24], dtype="<f4")
        np.testing.assert_allclose(pressure_pairs, [1.0, 0.1, 2.0, 0.2, 3.0, 0.3])


def test_write_at_shd_irregular_requires_depth_per_range(tmp_path):
    with pytest.raises(ValueError, match="one receiver depth per receiver range"):
        write_at_shd(
            tmp_path / "bad_irregular.shd",
            np.ones((1, 3), dtype=np.complex128),
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([100.0, 250.0, 500.0], dtype=np.float64),
            plot_type="irregular",
        )


def test_write_at_shd_tl_plot_type_uses_compressed_source_xy_header(tmp_path):
    depths = np.array([10.0], dtype=np.float32)
    ranges = np.array([100.0, 250.0], dtype=np.float64)
    source_x = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    source_y = np.array([0.0, 50.0, 100.0, 150.0], dtype=np.float64)
    pressure = np.zeros((3, 4, 1, 1, 1, 2), dtype=np.complex128)
    for isx in range(3):
        for isy in range(4):
            pressure[isx, isy, 0, 0, 0] = (
                10.0 * isx + isy + np.array([1.0, 2.0])
                + 1j * np.array([0.1, 0.2])
            )
    shd_path = tmp_path / "native_tl_compressed.shd"

    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="native TL compressed shd",
        frequency=50.0,
        plot_type="TL",
        source_x_m=source_x,
        source_y_m=source_y,
    )
    shade = read_at_shd_full(shd_path)

    assert shade.plot_type == "TL"
    np.testing.assert_allclose(shade.source_x_m, source_x)
    np.testing.assert_allclose(shade.source_y_m, source_y)
    np.testing.assert_allclose(shade.pressure_full, pressure.astype(np.complex64))

    with shd_path.open("rb") as f:
        lrecl = struct.unpack("<i", f.read(4))[0]
        rec_size = 4 * lrecl

        def read_rec(n):
            f.seek((n - 1) * rec_size)
            return f.read(rec_size)

        assert read_rec(2)[:10] == b"TL        "
        assert struct.unpack("<7idd", read_rec(3)[:44])[:7] == (1, 1, 3, 4, 1, 1, 2)
        np.testing.assert_allclose(np.frombuffer(read_rec(6)[:16], dtype="<f8"), [100.0, 300.0])
        np.testing.assert_allclose(np.frombuffer(read_rec(7)[:16], dtype="<f8"), [0.0, 150.0])


def test_write_at_shd_rejects_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="pressure shape"):
        write_at_shd(
            tmp_path / "bad.shd",
            np.ones((2, 2), dtype=np.complex128),
            np.array([10.0]),
            np.array([100.0, 200.0]),
        )


def test_read_and_write_at_boundary_profile_roundtrip(tmp_path):
    profile = ATBoundaryProfile(
        interpolation="L",
        ranges_m=np.array([0.0, 1000.0, 2500.0]),
        values_m=np.array([0.0, 1.5, -0.5]),
    )

    ati_path = tmp_path / "nested" / "surface.ati"
    write_at_boundary_profile(ati_path, profile)
    parsed_ati = read_at_ati(ati_path)

    assert parsed_ati.interpolation == "L"
    assert np.allclose(parsed_ati.ranges_m, profile.ranges_m)
    assert np.allclose(parsed_ati.values_m, profile.values_m)


def test_read_and_write_at_boundary_profile_long_format_roundtrip(tmp_path):
    profile = ATBoundaryProfile(
        interpolation="C",
        ranges_m=np.array([0.0, 1000.0]),
        values_m=np.array([100.0, 125.0]),
        geoacoustics=np.array(
            [
                [1600.0, 0.0, 1.6, 0.1, 0.0],
                [1700.0, 0.0, 1.8, 0.2, 0.0],
            ]
        ),
    )

    bty_path = tmp_path / "terrain.bty"
    write_at_boundary_profile(bty_path, profile)
    parsed = read_at_bty(bty_path)

    assert bty_path.read_text().splitlines()[0] == "'CL'"
    assert parsed.interpolation == "C"
    np.testing.assert_allclose(parsed.ranges_m, profile.ranges_m)
    np.testing.assert_allclose(parsed.values_m, profile.values_m)
    np.testing.assert_allclose(parsed.geoacoustics, profile.geoacoustics)


def test_read_at_ray_records_preserves_angle_and_bounce_metadata(tmp_path):
    ray_path = tmp_path / "case.ray"
    ray_path.write_text(
        "'BELLHOP- test'\n"
        "100.0\n"
        "1 1 1\n"
        "1 1\n"
        "0.0\n"
        "100.0\n"
        "'rz'\n"
        "5.0\n"
        "3 0 1\n"
        "0.0 10.0\n"
        "50.0 20.0\n"
        "100.0 30.0\n"
    )

    records = read_at_ray_records(ray_path)

    assert len(records) == 1
    assert records[0].launch_angle_deg == 5.0
    assert records[0].num_top_bounces == 0
    assert records[0].num_bottom_bounces == 1
    np.testing.assert_allclose(records[0].ranges_m, [0.0, 50.0, 100.0])


def test_read_at_ray_records_preserves_3d_xyz_metadata(tmp_path):
    ray_path = tmp_path / "case3d.ray"
    ray_path.write_text(
        "'BELLHOP3D test'\n"
        "100.0\n"
        "1 1 1\n"
        "1 1\n"
        "0.0\n"
        "100.0\n"
        "'xyz'\n"
        "-5.0 30.0\n"
        "3 1 2\n"
        "0.0 0.0 10.0\n"
        "50.0 3.0 20.0\n"
        "100.0 8.0 30.0\n"
    )

    records = read_at_ray_records(ray_path)
    legacy = read_at_ray(ray_path)

    assert len(records) == 1
    assert records[0].coordinate_system == "xyz"
    assert np.isclose(records[0].launch_angle_deg, -5.0)
    assert np.isclose(records[0].launch_azimuth_deg, 30.0)
    assert records[0].num_top_bounces == 1
    assert records[0].num_bottom_bounces == 2
    np.testing.assert_allclose(records[0].ranges_m, [0.0, 50.0, 100.0])
    np.testing.assert_allclose(records[0].y_m, [0.0, 3.0, 8.0])
    np.testing.assert_allclose(records[0].depths_m, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(records[0].amplitudes, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(legacy[0][0], records[0].ranges_m)
    np.testing.assert_allclose(legacy[0][1], records[0].y_m)
    np.testing.assert_allclose(legacy[0][2], records[0].depths_m)


def test_read_at_ray_records_supports_bellhop3d_alpha_radian_blocks(tmp_path):
    ray_path = tmp_path / "case3d_at.ray"
    ray_path.write_text(
        "'BELLHOP3D AT ray'\n"
        "100.0\n"
        "1 1 1\n"
        "2 3\n"
        "0.0\n"
        "200.0\n"
        "'xyz'\n"
        "-8.7266462599716474E-002\n"
        "3 0 0\n"
        "0.0 0.0 100.0\n"
        "9.960429728 -0.173859948 99.128442573\n"
        "19.920859456 -0.347719895 98.256885145\n"
    )

    records = read_at_ray_records(ray_path)

    assert len(records) == 1
    assert records[0].coordinate_system == "xyz"
    assert np.isclose(records[0].launch_angle_deg, -5.0)
    assert np.isclose(records[0].launch_azimuth_deg, -1.0)
    assert records[0].num_top_bounces == 0
    assert records[0].num_bottom_bounces == 0
    np.testing.assert_allclose(records[0].ranges_m, [0.0, 9.960429728, 19.920859456])
    np.testing.assert_allclose(records[0].y_m, [0.0, -0.173859948, -0.347719895])
    np.testing.assert_allclose(records[0].depths_m, [100.0, 99.128442573, 98.256885145])


def test_read_at_ray_records_accepts_at_comments_slashes_and_d_exponents(tmp_path):
    ray_path = tmp_path / "case3d_comments.ray"
    ray_path.write_text(
        "'BELLHOP3D commented ray'\n"
        "1.0D+02       ! frequency\n"
        "1 1 1         ! NSx NSy NSz\n"
        "1 1 /         ! NRays\n"
        "0.0           ! top\n"
        "1.0D+02       ! bottom\n"
        "'xyz' /        ! coordinate system\n"
        "-5.0D+00 3.0D+01 / ! alpha beta\n"
        "3 1 2 /       ! npts top bottom\n"
        "0.0D+00 0.0D+00 1.0D+01 / ! x y z\n"
        "5.0D+01 3.0D+00 2.0D+01 /\n"
        "1.0D+02 8.0D+00 3.0D+01 /\n"
    )

    records = read_at_ray_records(ray_path)

    assert len(records) == 1
    assert records[0].coordinate_system == "xyz"
    assert np.isclose(records[0].launch_angle_deg, -5.0)
    assert np.isclose(records[0].launch_azimuth_deg, 30.0)
    assert records[0].num_top_bounces == 1
    assert records[0].num_bottom_bounces == 2
    np.testing.assert_allclose(records[0].ranges_m, [0.0, 50.0, 100.0])
    np.testing.assert_allclose(records[0].y_m, [0.0, 3.0, 8.0])
    np.testing.assert_allclose(records[0].depths_m, [10.0, 20.0, 30.0])


def test_write_at_ray_records_roundtrip(tmp_path):
    ray_path = tmp_path / "roundtrip.ray"
    source_records = [
        ATRayRecord(
            launch_angle_deg=-3.5,
            ranges_m=np.array([0.0, 100.0, 250.0]),
            depths_m=np.array([20.0, 35.0, 50.0]),
            amplitudes=np.ones(3),
            num_top_bounces=1,
            num_bottom_bounces=0,
        ),
        ATRayRecord(
            launch_angle_deg=4.0,
            ranges_m=np.array([0.0, 125.0]),
            depths_m=np.array([20.0, 10.0]),
            amplitudes=np.ones(2),
            num_top_bounces=0,
            num_bottom_bounces=1,
        ),
    ]

    write_at_ray_records(
        ray_path,
        source_records,
        title="roundtrip",
        frequency=100.0,
        top_depth=0.0,
        bottom_depth=100.0,
    )
    records = read_at_ray_records(ray_path)

    assert len(records) == len(source_records)
    for got, expected in zip(records, source_records):
        assert np.isclose(got.launch_angle_deg, expected.launch_angle_deg)
        assert got.num_top_bounces == expected.num_top_bounces
        assert got.num_bottom_bounces == expected.num_bottom_bounces
        np.testing.assert_allclose(got.ranges_m, expected.ranges_m)
        np.testing.assert_allclose(got.depths_m, expected.depths_m)
    np.testing.assert_allclose(records[0].amplitudes, [1.0, 1.0, 1.0])


def test_write_at_ray_records_3d_xyz_roundtrip(tmp_path):
    ray_path = tmp_path / "roundtrip3d.ray"
    source_records = [
        ATRayRecord(
            launch_angle_deg=-3.5,
            launch_azimuth_deg=25.0,
            coordinate_system="xyz",
            ranges_m=np.array([0.0, 100.0, 250.0]),
            y_m=np.array([0.0, 5.0, 15.0]),
            depths_m=np.array([20.0, 35.0, 50.0]),
            amplitudes=np.ones(3),
            num_top_bounces=1,
            num_bottom_bounces=0,
        ),
    ]

    write_at_ray_records(
        ray_path,
        source_records,
        title="roundtrip3d",
        frequency=100.0,
        top_depth=0.0,
        bottom_depth=100.0,
    )
    records = read_at_ray_records(ray_path)

    assert ray_path.read_text().splitlines()[6] == "'xyz'"
    assert len(records) == 1
    assert records[0].coordinate_system == "xyz"
    assert np.isclose(records[0].launch_azimuth_deg, 25.0)
    np.testing.assert_allclose(records[0].ranges_m, source_records[0].ranges_m)
    np.testing.assert_allclose(records[0].y_m, source_records[0].y_m)
    np.testing.assert_allclose(records[0].depths_m, source_records[0].depths_m)


def test_generate_at_bellhop_env_writes_range_dependent_boundary_files(tmp_path):
    config = SimulationConfig(
        project="Bellhop RD boundary export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=250.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(
                type="vacuum",
                depth=0.0,
                altimetry=[(0.0, 0.0), (1000.0, 1.0)],
                altimetry_interpolation="C",
            ),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=250.0,
                bathymetry=[(0.0, 120.0), (1000.0, 150.0), (2000.0, 180.0)],
                bathymetry_interpolation="C",
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[2000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "rd.env"
    generate_at_bellhop_env(config, env_path, run_type="A")

    env_text = env_path.read_text()
    assert "'CVW *'" in env_text
    assert "'A~' 0.0" in env_text

    ati = read_at_ati(tmp_path / "rd.ati")
    bty = read_at_bty(tmp_path / "rd.bty")
    assert ati.interpolation == "C"
    assert bty.interpolation == "C"
    np.testing.assert_allclose(ati.ranges_m, [0.0, 1000.0])
    np.testing.assert_allclose(ati.values_m, [0.0, 1.0])
    np.testing.assert_allclose(bty.ranges_m, [0.0, 1000.0, 2000.0])
    np.testing.assert_allclose(bty.values_m, [120.0, 150.0, 180.0])


def test_generate_at_bellhop_env_writes_quad_2d_ssp_file(tmp_path):
    config = SimulationConfig(
        project="Bellhop quad SSP export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="quad-2d",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=100.0, c=1495.0),
                    SSPLayer(depth=200.0, c=1510.0),
                ],
                ranges=[0.0, 1000.0, 2500.0],
                c_matrix=[
                    [1500.0, 1501.0, 1503.0],
                    [1495.0, 1498.0, 1502.0],
                    [1510.0, 1512.0, 1515.0],
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="rigid", depth=200.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[2500.0], depths=[75.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "quad.env"
    generate_at_bellhop_env(config, env_path, run_type="R")

    assert "'QVW'" in env_path.read_text()
    assert (tmp_path / "quad.ssp").read_text().splitlines() == [
        "3",
        "0 1 2.5 /",
        "1500 1501 1503 /",
        "1495 1498 1502 /",
        "1510 1512 1515 /",
    ]


def test_generate_at_bellhop_env_writes_rigid_surface_option(tmp_path):
    config = SimulationConfig(
        project="Bellhop rigid surface export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="rigid", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "rigid.env"
    generate_at_bellhop_env(config, env_path, run_type="R")

    lines = env_path.read_text().splitlines()
    assert lines[3] == "'CRW'"
    assert lines[4] == "2 0.0 100.0"


def test_generate_at_bellhop_env_writes_seongil_beam_shift_run_type(tmp_path):
    config = SimulationConfig(
        project="Bellhop Seongil beam shift export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=21,
            step_size=5.0,
            beam_displacement="seongil",
        ),
    )

    env_path = tmp_path / "seongil.env"
    generate_at_bellhop_env(config, env_path, run_type="C")

    lines = env_path.read_text().splitlines()
    assert "'C     S'" in lines


def test_generate_at_bellhop_env_writes_line_source_run_type(tmp_path):
    config = SimulationConfig(
        project="Bellhop line source export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=21,
            step_size=5.0,
            source_type="line",
        ),
    )

    env_path = tmp_path / "line_source.env"
    generate_at_bellhop_env(config, env_path, run_type="C")

    assert "'C  X   '" in env_path.read_text().splitlines()


def test_generate_at_bellhop_env_combines_line_source_and_seongil_run_type(tmp_path):
    config = SimulationConfig(
        project="Bellhop line source Seongil export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=21,
            step_size=5.0,
            beam_displacement="seongil",
            source_type="line",
        ),
    )

    env_path = tmp_path / "line_source_seongil.env"
    generate_at_bellhop_env(config, env_path, run_type="C")

    assert "'C  X  S'" in env_path.read_text().splitlines()


def test_generate_at_bellhop_env_writes_irregular_receiver_grid_run_type(tmp_path):
    config = SimulationConfig(
        project="Bellhop irregular receiver export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0, 1500.0], depths=[25.0, 75.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=21,
            step_size=5.0,
            receiver_grid="irregular",
        ),
    )

    env_path = tmp_path / "irregular.env"
    generate_at_bellhop_env(config, env_path, run_type="C")

    assert "'C   I  '" in env_path.read_text().splitlines()


@pytest.mark.parametrize(
    ("beam_type", "run_type_line"),
    [
        ("geometric_hat", "'CG     '"),
        ("geometric_gaussian", "'CB     '"),
        ("simple_gaussian", "'CS     '"),
        ("geometric_hat_ray_centered", "'Cg     '"),
        ("geometric_gaussian_ray_centered", "'Cb     '"),
    ],
)
def test_generate_at_bellhop_env_writes_configured_beam_type(tmp_path, beam_type, run_type_line):
    config = SimulationConfig(
        project="Bellhop beam type export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=21,
            step_size=5.0,
            beam_type=beam_type,
        ),
    )

    env_path = tmp_path / f"{beam_type}.env"
    generate_at_bellhop_env(config, env_path, run_type="C")

    assert run_type_line in env_path.read_text().splitlines()


def test_generate_at_bellhop_env_combines_run_type_flags(tmp_path):
    config = SimulationConfig(
        project="Bellhop combined run type export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0, 1500.0], depths=[25.0, 75.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=21,
            step_size=5.0,
            beam_type="geometric_gaussian",
            beam_displacement="seongil",
            source_type="line",
            receiver_grid="irregular",
        ),
    )

    env_path = tmp_path / "combined_run_type.env"
    generate_at_bellhop_env(config, env_path, run_type="C")

    assert "'CB XI S'" in env_path.read_text().splitlines()


def test_generate_at_bellhop_env_rejects_irregular_grid_shape_mismatch(tmp_path):
    config = SimulationConfig(
        project="Bellhop bad irregular receiver export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0, 1500.0], depths=[25.0]),
        ),
        solver=BellhopConfig(
            type="bellhop",
            angles=(-10.0, 10.0),
            num_beams=21,
            step_size=5.0,
            receiver_grid="irregular",
        ),
    )

    with pytest.raises(ValueError, match="irregular receiver grid"):
        generate_at_bellhop_env(config, tmp_path / "bad_irregular.env", run_type="C")


def test_generate_at_bellhop_env_writes_acoustic_surface_halfspace_before_ssp(tmp_path):
    config = SimulationConfig(
        project="Bellhop acoustic surface export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(
                type="acousto-elastic",
                depth=0.0,
                c_p=1300.0,
                c_s=20.0,
                density=1.2,
                attenuation_p=0.1,
                attenuation_s=0.02,
            ),
            bottom=BoundaryCondition(type="acousto-elastic", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "acoustic.env"
    generate_at_bellhop_env(config, env_path, run_type="R")

    lines = env_path.read_text().splitlines()
    assert lines[3] == "'CAW'"
    assert lines[4] == "0.0 1300.0 20.0 1.2 0.1 0.02 /"
    assert lines[5] == "2 0.0 100.0"


def test_generate_at_bellhop_env_writes_elastic_bottom_shear_fields(tmp_path):
    config = SimulationConfig(
        project="Bellhop elastic bottom export",
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
                c_p=1700.0,
                c_s=600.0,
                density=1.9,
                attenuation_p=0.25,
                attenuation_s=0.05,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "elastic_bottom.env"
    generate_at_bellhop_env(config, env_path, run_type="R")

    lines = env_path.read_text().splitlines()
    bottom_idx = lines.index("'A' 0.0")
    assert lines[bottom_idx + 1] == "100.0 1700.0 600.0 1.9 0.25 0.05 /"


def test_generate_at_bellhop_env_writes_bottom_roughness(tmp_path):
    config = SimulationConfig(
        project="Bellhop rough bottom export",
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
                c_p=1700.0,
                density=1.9,
                rms_roughness_m=0.25,
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "rough_bottom.env"
    generate_at_bellhop_env(config, env_path, run_type="R")

    assert "'A' 0.25" in env_path.read_text().splitlines()


def test_generate_at_bellhop_env_writes_rigid_bottom_option_without_halfspace(tmp_path):
    config = SimulationConfig(
        project="Bellhop rigid bottom export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="rigid", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "rigid_bottom.env"
    generate_at_bellhop_env(config, env_path, run_type="R")

    lines = env_path.read_text().splitlines()
    bottom_idx = lines.index("'R' 0.0")
    assert lines[bottom_idx + 1] == "1"
    assert "100.0 1600.0 0.0 1.8 0.5 0.0 /" not in lines


def test_generate_at_bellhop_env_writes_vacuum_bottom_option_without_halfspace(tmp_path):
    config = SimulationConfig(
        project="Bellhop vacuum bottom export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[SSPLayer(depth=0.0, c=1500.0), SSPLayer(depth=100.0, c=1500.0)],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="vacuum", depth=100.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "vacuum_bottom.env"
    generate_at_bellhop_env(config, env_path, run_type="R")

    lines = env_path.read_text().splitlines()
    bottom_idx = lines.index("'V' 0.0")
    assert lines[bottom_idx + 1] == "1"
    assert "100.0 1600.0 0.0 1.8 0.5 0.0 /" not in lines


def test_generate_at_bellhop_env_writes_reflection_tables(tmp_path):
    config = SimulationConfig(
        project="Bellhop reflection table export",
        frequency=100.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=100.0, c=1500.0),
                ],
            ),
            surface=BoundaryCondition(
                type="vacuum",
                depth=0.0,
                reflection_table=[(0.0, 0.9, 180.0), (90.0, 0.9, 180.0)],
            ),
            bottom=BoundaryCondition(
                type="acousto-elastic",
                depth=100.0,
                reflection_table=[(0.0, 0.5, 0.0), (90.0, 0.5, 0.0)],
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[50.0]),
            receivers=ReceiverModel(ranges=[1000.0], depths=[50.0]),
        ),
        solver=BellhopConfig(type="bellhop", angles=(-10.0, 10.0), num_beams=21, step_size=5.0),
    )

    env_path = tmp_path / "refl.env"
    generate_at_bellhop_env(config, env_path, run_type="I")

    env_text = env_path.read_text()
    assert "'CFW'" in env_text
    assert "'F' 0.0" in env_text
    assert "100.0 1600.0 0.0" not in env_text

    top_table = read_at_irc(tmp_path / "refl.irc")
    bottom_table = read_at_brc(tmp_path / "refl.brc")
    np.testing.assert_allclose(top_table.amplitudes, [0.9, 0.9])
    np.testing.assert_allclose(top_table.phases_deg, [180.0, 180.0])
    np.testing.assert_allclose(bottom_table.amplitudes, [0.5, 0.5])
    np.testing.assert_allclose(bottom_table.phases_deg, [0.0, 0.0])


def test_generate_at_bellhop3d_env_writes_source_xy_and_bearings(tmp_path):
    config = SimulationConfig(
        project="bellhop3d source xy",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1500.0),
                    SSPLayer(depth=1000.0, c=1490.0),
                    SSPLayer(depth=5000.0, c=1520.0),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(type="vacuum", depth=5000.0),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[100.0, 200.0], x=[0.0, 250.0], y=[0.0, 100.0]),
            receivers=ReceiverModel(ranges=[0.0, 10000.0], depths=[50.0, 500.0]),
        ),
        solver=Bellhop3DConfig(
            elevation_angles=(-20.0, 20.0),
            azimuth_angles=(0.0, 90.0),
            num_elevation_beams=20,
            num_azimuth_beams=24,
            step_size=10.0,
            box_x=10.0,
            box_y=10.0,
            box_z=5500.0,
        ),
    )
    config.solver.receiver_bearings = [0.0, 45.0, 90.0]

    env_path = tmp_path / "bellhop3d.env"
    generate_at_bellhop3d_env(config, env_path, run_type="C")

    lines = env_path.read_text().splitlines()
    assert lines[9] == "2"
    assert lines[10] == "0.0 0.25 /"
    assert lines[11] == "2"
    assert lines[12] == "0.0 0.1 /"
    assert lines[13] == "2"
    assert lines[14] == "100.0 200.0 /"
    assert lines[19] == "3"
    assert lines[20] == "0.0 45.0 90.0 /"
    assert lines[21] == "'CB   3'"
    assert lines[22] == "20"
    assert lines[24] == "24"
    assert lines[26] == "10.0 10.0 10.0 5500.0"


def test_generate_at_bellhop3d_env_writes_bathymetry_3d_grid(tmp_path):
    config = SimulationConfig(
        project="bellhop3d 3d bathymetry",
        frequency=50.0,
        environment=EnvironmentModel(
            ssp=SSPModel(
                type="c-linear",
                data=[
                    SSPLayer(depth=0.0, c=1548.52),
                    SSPLayer(depth=1000.0, c=1500.0),
                    SSPLayer(depth=5000.0, c=1551.91),
                ],
            ),
            surface=BoundaryCondition(type="vacuum", depth=0.0),
            bottom=BoundaryCondition(
                type="vacuum",
                depth=5000.0,
                bathymetry_3d=Bathymetry3DModel(
                    x=[-2000.0, 10000.0],
                    y=[-2000.0, 10000.0],
                    z=[[2000.0, 3000.0], [4000.0, 5000.0]],
                ),
            ),
        ),
        geometry=GeometryModel(
            source=SourceModel(depths=[1000.0]),
            receivers=ReceiverModel(ranges=[0.0, 10000.0], depths=[0.0, 5000.0]),
        ),
        solver=Bellhop3DConfig(
            elevation_angles=(-20.0, 20.0),
            azimuth_angles=(0.0, 90.0),
            num_elevation_beams=20,
            num_azimuth_beams=24,
            step_size=10.0,
            box_x=10.0,
            box_y=10.0,
            box_z=20500.0,
        ),
    )

    env_path = tmp_path / "bellhop3d_bty.env"
    generate_at_bellhop3d_env(config, env_path, run_type="C")

    env_text = env_path.read_text()
    assert "'V~' 0.0" in env_text

    bty_lines = (tmp_path / "bellhop3d_bty.bty").read_text().splitlines()
    assert bty_lines == [
        "'R'",
        "2",
        "-2 10 /",
        "2",
        "-2 10 /",
        "2000 3000",
        "4000 5000",
    ]


def test_read_at_bty_accepts_comments_slash_and_unquoted_interpolation(tmp_path):
    bty_path = tmp_path / "case.bty"
    bty_path.write_text(
        "C ! cubic interpolation\n"
        "3 /\n"
        "0.0 100.0\n"
        "1.5 120.0 /\n"
        "2.0 90.0\n"
    )

    profile = read_at_bty(bty_path)

    assert profile.interpolation == "C"
    assert np.allclose(profile.ranges_m, [0.0, 1500.0, 2000.0])
    assert np.allclose(profile.values_m, [100.0, 120.0, 90.0])


def test_read_at_bty_long_format_accepts_commas_and_d_exponents(tmp_path):
    bty_path = tmp_path / "case_long.bty"
    bty_path.write_text(
        "'CL' ! cubic long-format profile\n"
        "2 /\n"
        "0.0D+00, 1.0D+02, 1.6D+03, 0.0D+00, 1.5D+00, 1.0D-01, 0.0D+00 /\n"
        "1.5D+00, 1.2D+02, 1.7D+03, 5.0D+02, 1.8D+00, 2.0D-01, 5.0D-02 /\n"
    )

    profile = read_at_bty(bty_path)

    assert profile.interpolation == "C"
    np.testing.assert_allclose(profile.ranges_m, [0.0, 1500.0])
    np.testing.assert_allclose(profile.values_m, [100.0, 120.0])
    np.testing.assert_allclose(
        profile.geoacoustics,
        [
            [1600.0, 0.0, 1.5, 0.1, 0.0],
            [1700.0, 500.0, 1.8, 0.2, 0.05],
        ],
    )


def test_read_and_write_at_bathymetry_3d_roundtrip(tmp_path):
    bty_path = tmp_path / "terrain_3d.bty"
    bty_path.write_text(
        "'RS'\n"
        "2\n"
        "-2.0 20.0 /\n"
        "3\n"
        "-1.0 0.0 1.0 /\n"
        "2000.0 3000.0\n"
        "3500.0 4500.0\n"
        "5000.0 6500.0\n",
        encoding="utf-8",
    )

    parsed = read_at_bathymetry_3d(bty_path)

    assert isinstance(parsed, ATBathymetry3D)
    assert parsed.interpolation == "RS"
    np.testing.assert_allclose(parsed.x_m, [-2000.0, 20000.0])
    np.testing.assert_allclose(parsed.y_m, [-1000.0, 0.0, 1000.0])
    np.testing.assert_allclose(parsed.z_m, [[2000.0, 3000.0], [3500.0, 4500.0], [5000.0, 6500.0]])

    roundtrip_path = tmp_path / "roundtrip.bty"
    write_at_bathymetry_3d(roundtrip_path, parsed)
    reparsed = read_at_bathymetry_3d(roundtrip_path)

    assert reparsed.interpolation == "RS"
    np.testing.assert_allclose(reparsed.x_m, parsed.x_m)
    np.testing.assert_allclose(reparsed.y_m, parsed.y_m)
    np.testing.assert_allclose(reparsed.z_m, parsed.z_m)


def test_read_at_bathymetry_3d_expands_at_endpoint_coordinate_rows(tmp_path):
    bty_path = tmp_path / "terrain_3d_endpoint_coordinates.bty"
    bty_path.write_text(
        "'R'\n"
        "4\n"
        "0.0 3.0 /\n"
        "3\n"
        "-1.0 1.0 /\n"
        "10.0 11.0 12.0 13.0\n"
        "20.0 21.0 22.0 23.0\n"
        "30.0 31.0 32.0 33.0\n",
        encoding="utf-8",
    )

    parsed = read_at_bathymetry_3d(bty_path)

    np.testing.assert_allclose(parsed.x_m, [0.0, 1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(parsed.y_m, [-1000.0, 0.0, 1000.0])
    np.testing.assert_allclose(parsed.z_m, [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]])


def test_read_at_boundary_profile_rejects_truncated_file(tmp_path):
    bty_path = tmp_path / "bad.bty"
    bty_path.write_text("'L'\n2\n0.0 100.0\n")

    with pytest.raises(ValueError, match="Expected 2"):
        read_at_bty(bty_path)


def test_read_and_write_at_source_beam_pattern_roundtrip(tmp_path):
    pattern = ATSourceBeamPattern(
        angles_deg=np.array([-20.0, 0.0, 20.0]),
        levels_db=np.array([-3.0, 0.0, -3.0]),
    )

    sbp_path = tmp_path / "source.sbp"
    write_at_source_beam_pattern(sbp_path, pattern)
    parsed = read_at_source_beam_pattern(sbp_path)

    np.testing.assert_allclose(parsed.angles_deg, pattern.angles_deg)
    np.testing.assert_allclose(parsed.levels_db, pattern.levels_db)


def test_read_and_write_reflection_table_roundtrip(tmp_path):
    table = ATReflectionTable(
        angles_deg=np.array([0.0, 30.0, 60.0]),
        amplitudes=np.array([1.0, 0.8, 0.2]),
        phases_deg=np.array([180.0, 90.0, 0.0]),
    )

    brc_path = tmp_path / "nested" / "bottom.brc"
    write_at_reflection_table(brc_path, table)
    parsed_brc = read_at_brc(brc_path)

    assert np.allclose(parsed_brc.angles_deg, table.angles_deg)
    assert np.allclose(parsed_brc.amplitudes, table.amplitudes)
    assert np.allclose(parsed_brc.phases_deg, table.phases_deg)
    assert np.allclose(parsed_brc.coefficients, table.coefficients)

    trc_path = tmp_path / "nested" / "top.trc"
    write_at_reflection_table(trc_path, table)
    parsed_trc = read_at_trc(trc_path)

    assert np.allclose(parsed_trc.angles_deg, table.angles_deg)
    assert np.allclose(parsed_trc.amplitudes, table.amplitudes)
    assert np.allclose(parsed_trc.phases_deg, table.phases_deg)
    assert np.allclose(parsed_trc.coefficients, table.coefficients)


def test_read_reflection_table_without_explicit_count(tmp_path):
    irc_path = tmp_path / "surface.irc"
    irc_path.write_text(
        "0.0 1.0 180.0\n"
        "45.0 0.5 -90.0\n"
    )

    table = read_at_irc(irc_path)

    assert np.allclose(table.angles_deg, [0.0, 45.0])
    assert np.allclose(table.amplitudes, [1.0, 0.5])
    assert np.allclose(table.phases_deg, [180.0, -90.0])


def test_read_reflection_table_accepts_comments_commas_slashes_and_d_exponents(tmp_path):
    irc_path = tmp_path / "surface.irc"
    irc_path.write_text(
        "2 / ! count\n"
        "0.0D+00, 1.0D+00, 1.8D+02 /\n"
        "9.0D+01, 5.0D-01, -9.0D+01 / ! grazing amp phase\n"
    )

    table = read_at_irc(irc_path)

    np.testing.assert_allclose(table.angles_deg, [0.0, 90.0])
    np.testing.assert_allclose(table.amplitudes, [1.0, 0.5])
    np.testing.assert_allclose(table.phases_deg, [180.0, -90.0])


def test_read_reflection_table_accepts_bounce_six_column_irc(tmp_path):
    irc_path = tmp_path / "bounce.irc"
    irc_path.write_text(
        "'BOUNCE- compact fixture' 250.0\n"
        "2\n"
        "0.0 -2.0 3.0 4.0 1.0 0\n"
        "1.57079632679 1.0 0.0 -2.0 0.5 0\n",
        encoding="utf-8",
    )

    table = read_at_irc(irc_path)

    np.testing.assert_allclose(table.angles_deg, [0.0, 90.0], atol=1.0e-9)
    np.testing.assert_allclose(table.amplitudes, [5.0, 2.0])
    np.testing.assert_allclose(table.phases_deg, [53.13010235415598, -90.0])


def test_read_reflection_table_rejects_non_triples(tmp_path):
    table_path = tmp_path / "bad.brc"
    table_path.write_text("0.0 1.0\n")

    with pytest.raises(ValueError, match="triples"):
        read_at_reflection_table(table_path)
