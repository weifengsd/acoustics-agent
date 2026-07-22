import csv
import json

import pytest

from pyacoustics.benchmarks import runner as runner_module
from pyacoustics.benchmarks.runner import run_canonical_at_benchmark_matrix, write_benchmark_report_plots


class _Matrix:
    def __init__(self, rows):
        self.rows = tuple(rows)


def test_write_benchmark_report_plots_creates_status_and_coverage_figures(tmp_path):
    rows = (
        {"case": "ok-a", "kind": "bellhop_tl", "passed": True},
        {"case": "ok-b", "kind": "bellhop_tl", "passed": True},
        {"case": "bad", "kind": "kraken_tl", "passed": False, "failures": "mae"},
        {"case": "skip", "kind": "field3d_runtime", "passed": False, "skipped": True},
    )

    status_path, coverage_path = write_benchmark_report_plots(tmp_path, rows, suite="unit")

    assert status_path == tmp_path / "figures" / "unit_benchmark_status.png"
    assert coverage_path == tmp_path / "figures" / "unit_benchmark_kind_coverage.png"
    assert status_path.stat().st_size > 0
    assert coverage_path.stat().st_size > 0


def test_run_canonical_at_benchmark_matrix_writes_combined_outputs(monkeypatch, tmp_path):
    calls = []

    def fake_run_bellhop_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "bellhop-a", "kind": "arrival", "passed": True},))

    def fake_run_kraken_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("kraken", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "kraken-a", "tl_mae_db": 0.2, "passed": True},))

    def fake_run_bellhop3d_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "bellhop3d-a", "tl_mae_db": 0.1, "passed": True},))

    def fake_run_bellhop3d_arrival_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d-arrival", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "bellhop3d-arrival-a", "kind": "bellhop3d_arrival", "passed": True},))

    def fake_run_bellhop3d_ray_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d-ray", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "bellhop3d-ray-a", "kind": "bellhop3d_ray", "passed": True},))

    def fake_canonical_field3d_runtime_benchmark_cases(*, sample_root=None):
        calls.append(("field3d-cases", sample_root))
        return ("field3d-case",)

    def fake_run_field3d_runtime_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("field3d", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "field3d-a", "kind": "field3d_runtime", "passed": True},))

    def fake_canonical_field3d_runtime_benchmark_cases(*, sample_root=None):
        calls.append(("field3d-cases", sample_root))
        return ("field3d-case",)

    def fake_canonical_field3d_runtime_benchmark_cases(*, sample_root=None):
        calls.append(("field3d-cases", sample_root))
        return ("field3d-case",)

    def fake_run_postprocessing_benchmark_matrix(*, thresholds=None):
        calls.append(("postprocessing", thresholds is not None))
        return _Matrix(({"case": "postprocessing-a", "kind": "postprocessing_rts_mat", "passed": True},))

    def fake_run_at_file_format_benchmark_matrix(*, sample_root=None, thresholds=None):
        calls.append(("formats", sample_root, thresholds is not None))
        return _Matrix(({"case": "format-a", "kind": "at_field_flp_roundtrip", "passed": True},))

    monkeypatch.setattr(runner_module, "run_bellhop_benchmark_matrix", fake_run_bellhop_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_kraken_benchmark_matrix", fake_run_kraken_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_bellhop3d_benchmark_matrix", fake_run_bellhop3d_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_bellhop3d_arrival_benchmark_matrix",
        fake_run_bellhop3d_arrival_benchmark_matrix,
    )
    monkeypatch.setattr(runner_module, "run_bellhop3d_ray_benchmark_matrix", fake_run_bellhop3d_ray_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_field3d_runtime_benchmark_matrix",
        fake_run_field3d_runtime_benchmark_matrix,
    )
    monkeypatch.setattr(
        runner_module,
        "canonical_field3d_runtime_benchmark_cases",
        fake_canonical_field3d_runtime_benchmark_cases,
    )
    monkeypatch.setattr(
        runner_module,
        "canonical_field3d_runtime_benchmark_cases",
        fake_canonical_field3d_runtime_benchmark_cases,
    )
    monkeypatch.setattr(
        runner_module,
        "canonical_field3d_runtime_benchmark_cases",
        fake_canonical_field3d_runtime_benchmark_cases,
    )
    monkeypatch.setattr(runner_module, "run_postprocessing_benchmark_matrix", fake_run_postprocessing_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_at_file_format_benchmark_matrix",
        fake_run_at_file_format_benchmark_matrix,
    )

    result = run_canonical_at_benchmark_matrix(
        at_bin_path="/opt/at/bin",
        output_dir=tmp_path,
        skip_if_missing_at=False,
    )

    assert calls == [("bellhop", 101, "/opt/at/bin", True), ("kraken", 22, "/opt/at/bin", True)]
    assert [row["case"] for row in result.rows] == ["bellhop-a", "kraken-a"]
    assert result.csv_path.name == "canonical_all_at_benchmark_matrix.csv"
    assert result.json_path.name == "canonical_all_at_benchmark_matrix.json"

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert [row["case"] for row in csv_rows] == ["bellhop-a", "kraken-a"]
    assert json_rows == list(result.rows)


def test_run_canonical_at_benchmark_matrix_writes_full_outputs(monkeypatch, tmp_path):
    calls = []

    def fake_run_bellhop_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "bellhop-a", "kind": "arrival", "passed": True},))

    def fake_run_kraken_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("kraken", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "kraken-a", "kind": "kraken_tl", "passed": True},))

    def fake_run_bellhop3d_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "bellhop3d-a", "kind": "bellhop3d_tl", "passed": True},))

    def fake_run_bellhop3d_arrival_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d-arrival", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "bellhop3d-arrival-a", "kind": "bellhop3d_arrival", "passed": True},))

    def fake_run_bellhop3d_ray_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d-ray", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "bellhop3d-ray-a", "kind": "bellhop3d_ray", "passed": True},))

    def fake_run_field3d_runtime_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("field3d", len(cases), at_bin_path, thresholds is not None))
        return _Matrix(({"case": "field3d-a", "kind": "field3d_runtime", "passed": True},))

    def fake_canonical_field3d_runtime_benchmark_cases(*, sample_root=None):
        calls.append(("field3d-cases", sample_root))
        return ("field3d-case",)

    def fake_run_postprocessing_benchmark_matrix(*, thresholds=None):
        calls.append(("postprocessing", thresholds is not None))
        return _Matrix(({"case": "postprocessing-a", "kind": "postprocessing_rts_mat", "passed": True},))

    def fake_run_at_file_format_benchmark_matrix(*, sample_root=None, thresholds=None):
        calls.append(("formats", sample_root, thresholds is not None))
        return _Matrix(({"case": "format-a", "kind": "at_field_flp_roundtrip", "passed": True},))

    monkeypatch.setattr(runner_module, "run_bellhop_benchmark_matrix", fake_run_bellhop_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_kraken_benchmark_matrix", fake_run_kraken_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_bellhop3d_benchmark_matrix", fake_run_bellhop3d_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_bellhop3d_arrival_benchmark_matrix",
        fake_run_bellhop3d_arrival_benchmark_matrix,
    )
    monkeypatch.setattr(runner_module, "run_bellhop3d_ray_benchmark_matrix", fake_run_bellhop3d_ray_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_field3d_runtime_benchmark_matrix",
        fake_run_field3d_runtime_benchmark_matrix,
    )
    monkeypatch.setattr(
        runner_module,
        "canonical_field3d_runtime_benchmark_cases",
        fake_canonical_field3d_runtime_benchmark_cases,
    )
    monkeypatch.setattr(runner_module, "run_postprocessing_benchmark_matrix", fake_run_postprocessing_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_at_file_format_benchmark_matrix",
        fake_run_at_file_format_benchmark_matrix,
    )

    result = run_canonical_at_benchmark_matrix(
        suite="full",
        at_bin_path="/opt/at/bin",
        sample_root=tmp_path / "fixtures",
        output_dir=tmp_path,
        skip_if_missing_at=False,
    )

    assert calls == [
        ("bellhop", 101, "/opt/at/bin", True),
        ("kraken", 22, "/opt/at/bin", True),
        ("bellhop3d", 5, "/opt/at/bin", True),
        ("bellhop3d-arrival", 7, "/opt/at/bin", False),
        ("bellhop3d-ray", 9, "/opt/at/bin", False),
        ("field3d-cases", tmp_path / "fixtures"),
        ("field3d", 1, "/opt/at/bin", True),
        ("postprocessing", True),
        ("formats", tmp_path / "fixtures", True),
    ]
    assert [row["case"] for row in result.rows] == [
        "bellhop-a",
        "kraken-a",
        "bellhop3d-a",
        "bellhop3d-arrival-a",
        "bellhop3d-ray-a",
        "field3d-a",
        "postprocessing-a",
        "format-a",
    ]
    assert result.csv_path.name == "canonical_full_at_benchmark_matrix.csv"
    assert result.json_path.name == "canonical_full_at_benchmark_matrix.json"


def test_run_canonical_at_benchmark_matrix_can_run_one_suite_without_outputs(monkeypatch):
    calls = []

    def fake_run_bellhop_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop", len(cases)))
        return _Matrix(({"case": "bellhop-only"},))

    def fake_run_kraken_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("kraken", len(cases)))
        return _Matrix(({"case": "kraken-only"},))

    def fake_run_bellhop3d_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d", len(cases), thresholds is not None))
        return _Matrix(({"case": "bellhop3d-only"},))

    def fake_run_bellhop3d_arrival_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d-arrival", len(cases), thresholds is not None))
        return _Matrix(({"case": "bellhop3d-arrival-only"},))

    def fake_run_bellhop3d_ray_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("bellhop3d-ray", len(cases), thresholds is not None))
        return _Matrix(({"case": "bellhop3d-ray-only"},))

    def fake_run_field3d_runtime_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append(("field3d", len(cases), thresholds is not None))
        return _Matrix(({"case": "field3d-only"},))

    def fake_run_postprocessing_benchmark_matrix(*, thresholds=None):
        calls.append(("postprocessing", thresholds is not None))
        return _Matrix(({"case": "postprocessing-only"},))

    def fake_run_at_file_format_benchmark_matrix(*, sample_root=None, thresholds=None):
        calls.append(("formats", sample_root, thresholds is not None))
        return _Matrix(({"case": "formats-only"},))

    monkeypatch.setattr(runner_module, "run_bellhop_benchmark_matrix", fake_run_bellhop_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_kraken_benchmark_matrix", fake_run_kraken_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_bellhop3d_benchmark_matrix", fake_run_bellhop3d_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_bellhop3d_arrival_benchmark_matrix",
        fake_run_bellhop3d_arrival_benchmark_matrix,
    )
    monkeypatch.setattr(runner_module, "run_bellhop3d_ray_benchmark_matrix", fake_run_bellhop3d_ray_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_field3d_runtime_benchmark_matrix",
        fake_run_field3d_runtime_benchmark_matrix,
    )
    monkeypatch.setattr(runner_module, "run_postprocessing_benchmark_matrix", fake_run_postprocessing_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_at_file_format_benchmark_matrix",
        fake_run_at_file_format_benchmark_matrix,
    )

    result = run_canonical_at_benchmark_matrix(suite="bellhop", skip_if_missing_at=False)

    assert calls == [("bellhop", 101)]
    assert result.rows == ({"case": "bellhop-only"},)
    assert result.csv_path is None
    assert result.json_path is None

    calls.clear()
    result = run_canonical_at_benchmark_matrix(suite="field3d", skip_if_missing_at=False)

    assert calls == [("field3d", 1, True)]
    assert result.rows == ({"case": "field3d-only"},)
    assert result.csv_path is None
    assert result.json_path is None

    calls.clear()
    result = run_canonical_at_benchmark_matrix(suite="bellhop3d", skip_if_missing_at=False)

    assert calls == [("bellhop3d", 5, True), ("bellhop3d-arrival", 7, False), ("bellhop3d-ray", 9, False)]
    assert result.rows == (
        {"case": "bellhop3d-only"},
        {"case": "bellhop3d-arrival-only"},
        {"case": "bellhop3d-ray-only"},
    )
    assert result.csv_path is None
    assert result.json_path is None

    calls.clear()
    result = run_canonical_at_benchmark_matrix(suite="postprocessing", skip_if_missing_at=False)

    assert calls == [("postprocessing", True)]
    assert result.rows == ({"case": "postprocessing-only"},)
    assert result.csv_path is None
    assert result.json_path is None

    calls.clear()
    result = run_canonical_at_benchmark_matrix(suite="formats", skip_if_missing_at=False)

    assert calls == [("formats", None, True)]
    assert result.rows == ({"case": "formats-only"},)
    assert result.csv_path is None
    assert result.json_path is None


def test_run_canonical_at_benchmark_matrix_forwards_external_postprocessing_args(monkeypatch, tmp_path):
    calls = []

    def fake_run_postprocessing_benchmark_matrix(**kwargs):
        calls.append(kwargs)
        return _Matrix(({"case": "postprocessing-external"},))

    monkeypatch.setattr(runner_module, "run_postprocessing_benchmark_matrix", fake_run_postprocessing_benchmark_matrix)

    source_dir = tmp_path / "MunkTS"
    work_dir = tmp_path / "work"
    at_matlab_dir = tmp_path / "Matlab"
    at_bin_dir = tmp_path / "bin"
    result = run_canonical_at_benchmark_matrix(
        suite="postprocessing",
        skip_if_missing_at=False,
        at_munkts_source_dir=source_dir,
        at_delayandsum_work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        matlab_runtime="octave",
        at_delayandsum_prelude_commands=("scooter MunkS", "stack"),
        at_delayandsum_output_filename="case.rts.mat",
        at_delayandsum_timeout_s=12.5,
    )

    assert result.rows == ({"case": "postprocessing-external"},)
    assert calls[0]["thresholds"] is not None
    assert calls[0]["at_munkts_source_dir"] == source_dir
    assert calls[0]["at_delayandsum_reference_path"] is None
    assert calls[0]["at_delayandsum_work_dir"] == work_dir
    assert calls[0]["at_matlab_dir"] == at_matlab_dir
    assert calls[0]["at_bin_dir"] == at_bin_dir
    assert calls[0]["matlab_runtime"] == "octave"
    assert calls[0]["at_delayandsum_prelude_commands"] == ("scooter MunkS", "stack")
    assert calls[0]["at_delayandsum_output_filename"] == "case.rts.mat"
    assert calls[0]["at_delayandsum_timeout_s"] == 12.5


def test_run_canonical_at_benchmark_matrix_infers_postprocessing_matlab_dir_from_at_bin(monkeypatch, tmp_path):
    calls = []

    def fake_run_postprocessing_benchmark_matrix(**kwargs):
        calls.append(kwargs)
        return _Matrix(({"case": "postprocessing-external"},))

    monkeypatch.setattr(runner_module, "run_postprocessing_benchmark_matrix", fake_run_postprocessing_benchmark_matrix)

    at_root = tmp_path / "at"
    at_bin_path = at_root / "bin"
    at_matlab_dir = at_root / "Matlab"
    at_bin_path.mkdir(parents=True)
    at_matlab_dir.mkdir()
    result = run_canonical_at_benchmark_matrix(
        suite="postprocessing",
        skip_if_missing_at=False,
        at_bin_path=at_bin_path,
        at_munkts_source_dir=tmp_path / "MunkTS",
        at_delayandsum_work_dir=tmp_path / "work",
        matlab_runtime="octave",
    )

    assert result.rows == ({"case": "postprocessing-external"},)
    assert calls[0]["at_matlab_dir"] == at_matlab_dir
    assert calls[0]["at_bin_dir"] == at_bin_path


def test_run_canonical_at_benchmark_matrix_rejects_unknown_suite():
    with pytest.raises(ValueError, match="suite"):
        run_canonical_at_benchmark_matrix(suite="bad")


def test_run_canonical_at_benchmark_matrix_skips_missing_at(monkeypatch, tmp_path):
    calls = []

    class MissingExternalSolver:
        bin_path = tmp_path / "missing-at-bin"

        def __init__(self, config, bin_path=None):
            pass

        def is_available(self, exe_name):
            return False

    def fake_run_bellhop_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append("bellhop")
        return _Matrix(())

    def fake_run_kraken_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append("kraken")
        return _Matrix(())

    def fake_run_bellhop3d_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append("bellhop3d")
        return _Matrix(())

    def fake_run_bellhop3d_arrival_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append("bellhop3d-arrival")
        return _Matrix(())

    def fake_run_bellhop3d_ray_benchmark_matrix(cases, *, at_bin_path=None, thresholds=None):
        calls.append("bellhop3d-ray")
        return _Matrix(())

    def fake_run_postprocessing_benchmark_matrix(*, thresholds=None):
        calls.append("postprocessing")
        return _Matrix(
                (
                    {"case": "postprocessing-rts-payload", "kind": "postprocessing_rts_mat", "passed": True},
                    {"case": "postprocessing-at-delayandsum-rts-payload", "kind": "postprocessing_at_rts_mat", "passed": True},
                    {"case": "postprocessing-at-read-ts-timeseries", "kind": "postprocessing_rts_timeseries", "passed": True},
                    {
                        "case": "postprocessing-at-read-ts-ascii-timeseries",
                        "kind": "postprocessing_rts_ascii_timeseries",
                        "passed": True,
                    },
                    {
                        "case": "postprocessing-at-read-ts-ascii-free-format-timeseries",
                        "kind": "postprocessing_rts_ascii_timeseries",
                        "passed": True,
                    },
                    {
                        "case": "postprocessing-delayandsum-grid-multi-arrival",
                        "kind": "postprocessing_delayandsum_grid",
                        "passed": True,
                    },
                    {"case": "postprocessing-matched-filter-payload", "kind": "postprocessing_matched_filter_mat", "passed": True},
                    {
                        "case": "postprocessing-matched-filter-workflow",
                        "kind": "postprocessing_matched_filter_workflow",
                        "passed": True,
                    },
                    {"case": "postprocessing-doppler-transform", "kind": "postprocessing_doppler_transform", "passed": True},
                    {
                        "case": "postprocessing-doppler-delayandsum-workflow",
                        "kind": "postprocessing_doppler_workflow",
                        "passed": True,
                    },
                    {
                        "case": "postprocessing-at-munkts-real-rts-timeseries",
                        "kind": "postprocessing_at_real_rts_timeseries",
                        "passed": True,
                    },
                )
            )

    def fake_run_at_file_format_benchmark_matrix(*, sample_root=None, thresholds=None):
        calls.append("formats")
        return _Matrix(
                (
                    *(
                        {
                            "case": f"field-flp-roundtrip-{index}",
                            "kind": "at_field_flp_roundtrip",
                        "passed": True,
                    }
                    for index in range(103)
                    ),
                    {"case": "mod-roundtrip-real", "kind": "at_mod_roundtrip", "passed": True},
                    {"case": "mod-roundtrip-real-calib", "kind": "at_mod_roundtrip", "passed": True},
                    {"case": "mod-roundtrip-a", "kind": "at_mod_roundtrip", "passed": True},
                    {"case": "mod-roundtrip-b", "kind": "at_mod_roundtrip", "passed": True},
                {"case": "shd-roundtrip-real", "kind": "at_shd_roundtrip", "passed": True},
                {"case": "shd-roundtrip-real-field", "kind": "at_shd_roundtrip", "passed": True},
                {"case": "shd-roundtrip-real-calib", "kind": "at_shd_roundtrip", "passed": True},
                {"case": "shd-roundtrip-a", "kind": "at_shd_roundtrip", "passed": True},
                {"case": "shd-roundtrip-b", "kind": "at_shd_roundtrip", "passed": True},
                {"case": "boundary-roundtrip-a", "kind": "at_boundary_profile_roundtrip", "passed": True},
                {"case": "boundary-roundtrip-b", "kind": "at_boundary_profile_roundtrip", "passed": True},
                {"case": "boundary-roundtrip-real-ellipse", "kind": "at_boundary_profile_roundtrip", "passed": True},
                {"case": "boundary-roundtrip-real-parabot", "kind": "at_boundary_profile_roundtrip", "passed": True},
                {"case": "boundary-roundtrip-real-block", "kind": "at_boundary_profile_roundtrip", "passed": True},
                {"case": "bathymetry-3d-roundtrip-real", "kind": "at_bathymetry_3d_roundtrip", "passed": True},
                {"case": "reflection-roundtrip-a", "kind": "at_reflection_table_roundtrip", "passed": True},
                {"case": "reflection-roundtrip-b", "kind": "at_reflection_table_roundtrip", "passed": True},
                {"case": "reflection-roundtrip-real-calibbounce", "kind": "at_reflection_table_roundtrip", "passed": True},
                {"case": "reflection-roundtrip-real-calibbounce-irc", "kind": "at_reflection_table_roundtrip", "passed": True},
                {"case": "ray-roundtrip-a", "kind": "at_ray_roundtrip", "passed": True},
                {"case": "ray-roundtrip-b", "kind": "at_ray_roundtrip", "passed": True},
                {"case": "ray-roundtrip-real-perfect-wedge", "kind": "at_ray_roundtrip", "passed": True},
                {"case": "arr-roundtrip-a", "kind": "at_arr_roundtrip", "passed": True},
                {"case": "arr-roundtrip-b", "kind": "at_arr_roundtrip", "passed": True},
                *(
                    {
                        "case": f"bathymetry-3d-roundtrip-real-{index}",
                        "kind": "at_bathymetry_3d_roundtrip",
                        "passed": True,
                    }
                    for index in range(31)
                ),
                *(
                    {
                        "case": f"bathymetry-3d-province-roundtrip-real-{index}",
                        "kind": "at_bathymetry_3d_province_roundtrip",
                        "passed": True,
                    }
                    for index in range(4)
                ),
                {"case": "prt-parse-a", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-b", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-c", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-real-bellhop", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-real-field", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-real-kraken", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-real-bellhop3d", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-real-pekeris", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-real-calib-kraken", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-real-calib-field", "kind": "at_prt_parse", "passed": True},
                {"case": "prt-parse-real-perfect-wedge", "kind": "at_prt_parse", "passed": True},
            )
        )

    monkeypatch.setattr(runner_module, "run_bellhop_benchmark_matrix", fake_run_bellhop_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_kraken_benchmark_matrix", fake_run_kraken_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_bellhop3d_benchmark_matrix", fake_run_bellhop3d_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_bellhop3d_arrival_benchmark_matrix",
        fake_run_bellhop3d_arrival_benchmark_matrix,
    )
    monkeypatch.setattr(runner_module, "run_bellhop3d_ray_benchmark_matrix", fake_run_bellhop3d_ray_benchmark_matrix)
    monkeypatch.setattr(runner_module, "run_postprocessing_benchmark_matrix", fake_run_postprocessing_benchmark_matrix)
    monkeypatch.setattr(
        runner_module,
        "run_at_file_format_benchmark_matrix",
        fake_run_at_file_format_benchmark_matrix,
    )
    monkeypatch.setattr(runner_module, "ExternalSolver", MissingExternalSolver)

    result = run_canonical_at_benchmark_matrix(
        suite="all",
        output_dir=tmp_path,
    )

    assert calls == []
    assert len(result.rows) == 123
    assert {row["kind"] for row in result.rows} == {"arrival", "ray", "tl", "kraken_tl"}
    assert all(row["skipped"] for row in result.rows)
    assert all(not row["passed"] for row in result.rows)
    assert result.csv_path.exists()
    assert result.json_path.exists()

    result = run_canonical_at_benchmark_matrix(suite="bellhop3d", output_dir=tmp_path)

    assert calls == []
    assert len(result.rows) == 21
    assert result.rows[0]["kind"] == "bellhop3d_tl"
    assert result.rows[0]["skipped"] is True
    assert [row["kind"] for row in result.rows[5:12]] == [
        "bellhop3d_arrival",
        "bellhop3d_arrival",
        "bellhop3d_arrival",
        "bellhop3d_arrival",
        "bellhop3d_arrival",
        "bellhop3d_arrival",
        "bellhop3d_arrival",
    ]
    assert [row["kind"] for row in result.rows[-9:]] == [
        "bellhop3d_ray",
        "bellhop3d_ray",
        "bellhop3d_ray",
        "bellhop3d_ray",
        "bellhop3d_ray",
        "bellhop3d_ray",
        "bellhop3d_ray",
        "bellhop3d_ray",
        "bellhop3d_ray",
    ]

    result = run_canonical_at_benchmark_matrix(suite="full", output_dir=tmp_path)

    assert len(result.rows) == 329
    assert {row["kind"] for row in result.rows} == {
        "arrival",
        "ray",
        "tl",
        "kraken_tl",
        "bellhop3d_tl",
        "bellhop3d_arrival",
        "bellhop3d_ray",
        "field3d_runtime",
        "postprocessing_rts_mat",
        "postprocessing_at_rts_mat",
        "postprocessing_rts_timeseries",
        "postprocessing_rts_ascii_timeseries",
        "postprocessing_delayandsum_grid",
        "postprocessing_matched_filter_mat",
        "postprocessing_matched_filter_workflow",
        "postprocessing_doppler_transform",
        "postprocessing_doppler_workflow",
        "postprocessing_at_real_rts_timeseries",
        "at_field_flp_roundtrip",
        "at_mod_roundtrip",
        "at_shd_roundtrip",
        "at_boundary_profile_roundtrip",
        "at_bathymetry_3d_roundtrip",
        "at_bathymetry_3d_province_roundtrip",
        "at_reflection_table_roundtrip",
        "at_ray_roundtrip",
        "at_arr_roundtrip",
        "at_prt_parse",
    }
    assert calls == ["postprocessing", "formats"]
    assert all(row["skipped"] for row in result.rows[:-184])
    assert all(row["passed"] for row in result.rows[-184:])


def test_benchmark_runner_cli_forwards_options(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run_canonical_at_benchmark_matrix(**kwargs):
        calls.append(kwargs)
        return runner_module.BenchmarkMatrixRun(
            rows=(
                {"case": "a", "passed": True},
                {"case": "b", "passed": False, "skipped": True},
            ),
            csv_path=tmp_path / "matrix.csv",
            json_path=None,
        )

    monkeypatch.setattr(runner_module, "run_canonical_at_benchmark_matrix", fake_run_canonical_at_benchmark_matrix)

    code = runner_module.main(
        [
            "--suite",
            "postprocessing",
            "--at-bin-path",
            str(tmp_path / "at" / "bin"),
            "--output-dir",
            str(tmp_path / "reports"),
            "--sample-root",
            str(tmp_path / "fixtures"),
            "--no-json",
            "--strict-at",
            "--at-munkts-source-dir",
            str(tmp_path / "MunkTS"),
            "--at-delayandsum-work-dir",
            str(tmp_path / "work"),
            "--at-matlab-dir",
            str(tmp_path / "Matlab"),
            "--at-bin-dir",
            str(tmp_path / "bin"),
            "--matlab-runtime",
            "octave",
            "--at-delayandsum-prelude-command",
            "scooter MunkS",
            "--at-delayandsum-prelude-command",
            "stack",
            "--at-delayandsum-output-filename",
            "MunkS.rts.mat",
            "--at-delayandsum-timeout-s",
            "12.5",
        ]
    )

    assert code == 1
    assert calls == [
        {
            "suite": "postprocessing",
            "at_bin_path": tmp_path / "at" / "bin",
            "output_dir": tmp_path / "reports",
            "sample_root": tmp_path / "fixtures",
            "write_csv": True,
            "write_json": False,
            "skip_if_missing_at": False,
            "at_munkts_source_dir": tmp_path / "MunkTS",
            "at_delayandsum_reference_path": None,
            "at_delayandsum_work_dir": tmp_path / "work",
            "at_matched_filter_reference_path": None,
            "at_matched_filter_work_dir": None,
            "at_matlab_dir": tmp_path / "Matlab",
            "at_bin_dir": tmp_path / "bin",
            "matlab_runtime": "octave",
            "at_delayandsum_prelude_commands": ("scooter MunkS", "stack"),
            "at_delayandsum_output_filename": "MunkS.rts.mat",
            "at_delayandsum_timeout_s": 12.5,
            "at_matched_filter_prelude_commands": (),
            "at_matched_filter_script_name": "matchedfilter",
            "at_matched_filter_output_filename": "matched_filter.mat",
            "at_matched_filter_timeout_s": 300.0,
        }
    ]
    output = capsys.readouterr().out
    assert "postprocessing: 1/2 passed, 0 failed, 1 skipped" in output
    assert "Failures:" not in output
    assert f"CSV: {tmp_path / 'matrix.csv'}" in output

    code = runner_module.main(
        [
            "--suite",
            "postprocessing",
            "--output-dir",
            str(tmp_path / "reports"),
            "--allow-failures",
        ]
    )

    assert code == 0


def test_benchmark_runner_cli_prints_failure_summary(monkeypatch, tmp_path, capsys):
    def fake_run_canonical_at_benchmark_matrix(**_kwargs):
        return runner_module.BenchmarkMatrixRun(
            rows=(
                {"case": "ok", "passed": True},
                {"case": "bad", "passed": False, "failures": "tl_mae_db=2 > 1"},
            ),
            csv_path=None,
            json_path=None,
        )

    monkeypatch.setattr(runner_module, "run_canonical_at_benchmark_matrix", fake_run_canonical_at_benchmark_matrix)

    code = runner_module.main(["--suite", "bellhop", "--no-csv", "--no-json"])

    assert code == 1
    output = capsys.readouterr().out
    assert "bellhop: 1/2 passed, 1 failed, 0 skipped" in output
    assert "Failures:" in output
    assert "- bad: tl_mae_db=2 > 1" in output
