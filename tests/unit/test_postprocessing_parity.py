import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.io import savemat

from pyacoustics.postprocessing.benchmark_matrix import (
    postprocessing_smoke_thresholds,
    run_postprocessing_benchmark_matrix,
    write_postprocessing_report_plots,
)
from pyacoustics.postprocessing.external import (
    at_delayandsum_external_benchmark_row,
    at_matched_filter_external_benchmark_row,
    at_munkts_stack_external_benchmark_row,
    find_matlab_runtime,
    postprocessing_external_failure_row,
    prepare_at_munkts_work_dir,
    run_at_delayandsum_script,
)
from pyacoustics.postprocessing.parity import (
    compare_at_rts_mat_payload,
    compare_matched_filter_mat_payload,
    compare_mat_timeseries,
    compare_rts_mat_payload,
    compare_rts_timeseries_files,
    compare_timeseries,
    mat_payload_benchmark_row,
    read_rts_timeseries,
    timeseries_benchmark_row,
)


def _matched_filter_payload():
    return {
        "PlotTitle": "matched",
        "MF": np.array([[0.0], [1.0], [0.0]]),
        "lags": np.array([-1, 0, 1]),
        "lag_seconds": np.array([-0.001, 0.0, 0.001]),
        "sample_rate": np.array([[1000.0]]),
        "receiver_depths": np.array([100.0]),
        "receiver_ranges": np.array([500.0]),
        "source_index": np.array([[0]]),
        "source_depths": np.array([50.0]),
        "selected_source_depth": np.array([50.0]),
        "source_x_index": np.array([[0]]),
        "source_y_index": np.array([[0]]),
        "source_x": np.array([]),
        "source_y": np.array([]),
        "selected_source_x": np.array([]),
        "selected_source_y": np.array([]),
        "receiver_bearings": np.array([]),
    }

def test_compare_timeseries_reports_error_metrics():
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    candidate = np.array([[1.5, 1.5], [3.0, 5.0]])

    metrics = compare_timeseries(reference, candidate)
    row = timeseries_benchmark_row("delayandsum-reference", metrics)

    assert metrics.sample_count == 4
    assert np.isclose(metrics.mae, 0.5)
    assert np.isclose(metrics.max_abs, 1.0)
    assert row["case"] == "delayandsum-reference"
    assert row["kind"] == "postprocessing_timeseries"
    assert np.isclose(row["timeseries_mae"], 0.5)


def test_postprocessing_benchmark_matrix_includes_doppler_transform_row():
    matrix = run_postprocessing_benchmark_matrix(thresholds=postprocessing_smoke_thresholds())

    assert len(matrix.rows) == 10
    assert all(row["passed"] for row in matrix.rows)
    ascii_row = next(
        row
        for row in matrix.rows
        if row["case"] == "postprocessing-at-read-ts-ascii-timeseries"
    )
    assert ascii_row["sample_count"] == 12
    assert ascii_row["timeseries_max_abs"] == 0.0
    free_format_ascii_row = next(
        row
        for row in matrix.rows
        if row["case"] == "postprocessing-at-read-ts-ascii-free-format-timeseries"
    )
    assert free_format_ascii_row["sample_count"] == 12
    assert free_format_ascii_row["timeseries_max_abs"] == 0.0
    assert free_format_ascii_row["postprocessing_time_sample_count"] == 4
    assert free_format_ascii_row["postprocessing_receiver_depth_count"] == 3
    assert free_format_ascii_row["postprocessing_ascii_free_format"] is True
    grid_row = next(row for row in matrix.rows if row["kind"] == "postprocessing_delayandsum_grid")
    assert grid_row["sample_count"] == 270
    assert grid_row["timeseries_max_abs"] == 0.0
    assert grid_row["postprocessing_receiver_range_count"] == 3
    assert grid_row["postprocessing_receiver_depth_count"] == 3
    assert grid_row["postprocessing_arrival_count"] == 27
    matched_workflow_row = next(row for row in matrix.rows if row["kind"] == "postprocessing_matched_filter_workflow")
    assert matched_workflow_row["sample_count"] == 8
    assert matched_workflow_row["timeseries_max_abs"] == 0.0
    assert matched_workflow_row["postprocessing_matched_filter_trace_count"] == 4
    assert matched_workflow_row["postprocessing_matched_filter_peak_lag_max_abs"] == 0.0
    assert matched_workflow_row["postprocessing_matched_filter_peak_value_max_abs"] == 0.0
    doppler_row = next(row for row in matrix.rows if row["kind"] == "postprocessing_doppler_transform")
    assert doppler_row["timeseries_max_abs"] == 0.0
    assert doppler_row["timeseries_relative_l2_error"] == 0.0
    assert doppler_row["doppler_alpha_lt_one_finite"] is True
    assert doppler_row["doppler_alpha_gt_one_finite"] is True
    assert doppler_row["doppler_alpha_lt_one_peak"] > 0.0
    assert doppler_row["doppler_alpha_gt_one_peak"] > 0.0
    assert doppler_row["doppler_chirp_fft_max_abs"] < 1.0e-12
    assert doppler_row["doppler_chirp_fft_rmse"] < 1.0e-12
    doppler_workflow_row = next(row for row in matrix.rows if row["kind"] == "postprocessing_doppler_workflow")
    assert doppler_workflow_row["sample_count"] == 240
    assert doppler_workflow_row["timeseries_max_abs"] == 0.0
    assert doppler_workflow_row["postprocessing_doppler_trace_count"] == 8
    assert doppler_workflow_row["postprocessing_doppler_velocity_bin_count"] == 3
    assert doppler_workflow_row["postprocessing_receiver_bearing_count"] == 2


def test_write_postprocessing_report_plots_outputs_png_files(tmp_path):
    paths = write_postprocessing_report_plots(tmp_path)

    assert {path.name for path in paths} == {
        "postprocessing_delay_and_sum_timeseries.png",
        "postprocessing_matched_filter_timeseries.png",
    }
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)

def test_compare_timeseries_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        compare_timeseries(np.zeros((2, 2)), np.zeros((2, 3)))


def test_compare_mat_timeseries_reads_rts_payload(tmp_path):
    reference_path = tmp_path / "reference.rts.mat"
    candidate_path = tmp_path / "candidate.rts.mat"
    savemat(reference_path, {"RTS": np.array([[1.0, 2.0]])})
    savemat(candidate_path, {"RTS": np.array([[1.0, 2.25]])})

    metrics = compare_mat_timeseries(reference_path, candidate_path)

    assert metrics.sample_count == 2
    assert np.isclose(metrics.mae, 0.125)


def test_compare_rts_mat_payload_checks_data_axis_and_metadata(tmp_path):
    reference_path = tmp_path / "reference.rts.mat"
    candidate_path = tmp_path / "candidate.rts.mat"
    common = {
        "PlotTitle": "case",
        "tout": np.array([[0.0, 0.001, 0.002]]),
        "sample_rate": np.array([[1000.0]]),
        "receiver_depths": np.array([50.0, 100.0]),
        "receiver_ranges": np.array([1000.0]),
        "source_index": np.array([[0]]),
        "source_depths": np.array([25.0]),
        "selected_source_depth": np.array([25.0]),
        "source_x_index": np.array([[0]]),
        "source_y_index": np.array([[0]]),
        "source_x": np.array([]),
        "source_y": np.array([]),
        "selected_source_x": np.array([]),
        "selected_source_y": np.array([]),
        "receiver_bearings": np.array([]),
    }
    savemat(reference_path, {**common, "RTS": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])})
    savemat(candidate_path, {**common, "RTS": np.array([[1.0, 2.25], [3.0, 4.0], [5.5, 6.0]])})

    metrics = compare_rts_mat_payload(reference_path, candidate_path, exact_plot_title=True)
    row = mat_payload_benchmark_row("rts-workflow", metrics)

    assert metrics.payload_kind == "postprocessing_rts_mat"
    assert metrics.variable_metrics["RTS"].sample_count == 6
    assert metrics.variable_metrics["tout"].max_abs == 0.0
    assert metrics.sample_count > metrics.variable_metrics["RTS"].sample_count
    assert np.isclose(metrics.max_abs, 0.5)
    assert row["kind"] == "postprocessing_rts_mat"
    assert row["mat_payload_variable_count"] == len(metrics.compared_variables)
    assert "RTS" in row["mat_payload_variables"]


def test_compare_matched_filter_mat_payload_checks_lags(tmp_path):
    reference_path = tmp_path / "reference.mf.mat"
    candidate_path = tmp_path / "candidate.mf.mat"
    common = {
        "PlotTitle": "matched",
        "MF": np.array([[0.0], [1.0], [0.0]]),
        "sample_rate": np.array([[1000.0]]),
        "receiver_depths": np.array([100.0]),
        "receiver_ranges": np.array([500.0]),
        "source_index": np.array([[0]]),
        "source_depths": np.array([50.0]),
        "selected_source_depth": np.array([50.0]),
        "source_x_index": np.array([[0]]),
        "source_y_index": np.array([[0]]),
        "source_x": np.array([]),
        "source_y": np.array([]),
        "selected_source_x": np.array([]),
        "selected_source_y": np.array([]),
        "receiver_bearings": np.array([]),
    }
    savemat(reference_path, {**common, "lags": np.array([-1, 0, 1]), "lag_seconds": np.array([-0.001, 0.0, 0.001])})
    savemat(candidate_path, {**common, "lags": np.array([-1, 0, 2]), "lag_seconds": np.array([-0.001, 0.0, 0.002])})

    metrics = compare_matched_filter_mat_payload(reference_path, candidate_path, exact_plot_title=True)

    assert metrics.payload_kind == "postprocessing_matched_filter_mat"
    assert np.isclose(metrics.variable_metrics["lags"].max_abs, 1.0)
    assert np.isclose(metrics.variable_metrics["lag_seconds"].max_abs, 0.001)


def test_compare_at_rts_mat_payload_supports_compact_delayandsum_schema(tmp_path):
    reference_path = tmp_path / "reference.at.rts.mat"
    candidate_path = tmp_path / "candidate.at.rts.mat"
    common = {
        "PlotTitle": "MunkB",
        "Pos": {
            "s": {"z": np.array([1000.0])},
            "r": {"z": np.array([100.0, 200.0]), "r": np.array([5000.0])},
        },
        "tout": np.array([[6.0, 6.001, 6.002]]),
    }
    savemat(reference_path, {**common, "RTS": np.array([[0.0, 1.0], [0.5, 0.0], [0.0, -0.25]])})
    savemat(candidate_path, {**common, "RTS": np.array([[0.0, 1.0], [0.75, 0.0], [0.0, -0.25]])})

    metrics = compare_at_rts_mat_payload(reference_path, candidate_path, exact_plot_title=True)

    assert metrics.payload_kind == "postprocessing_at_rts_mat"
    assert metrics.compared_variables == ("RTS", "tout")
    assert np.isclose(metrics.variable_metrics["RTS"].max_abs, 0.25)
    assert metrics.variable_metrics["tout"].max_abs == 0.0


def test_compare_at_rts_mat_payload_requires_pos_schema(tmp_path):
    reference_path = tmp_path / "reference.at.rts.mat"
    candidate_path = tmp_path / "candidate.at.rts.mat"
    savemat(reference_path, {"PlotTitle": "case", "RTS": np.array([[1.0]]), "tout": np.array([[0.0]])})
    savemat(candidate_path, {"PlotTitle": "case", "RTS": np.array([[1.0]]), "tout": np.array([[0.0]])})

    with pytest.raises(KeyError, match="Pos"):
        compare_at_rts_mat_payload(reference_path, candidate_path)


def test_find_matlab_runtime_prefers_matlab_then_octave(monkeypatch):
    calls = []

    def fake_which(name):
        calls.append(name)
        return "/usr/local/bin/octave" if name == "octave" else None

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", fake_which)

    assert find_matlab_runtime() == "/usr/local/bin/octave"
    assert calls == ["matlab", "octave"]


def test_prepare_at_munkts_work_dir_copies_inputs_and_reference(tmp_path):
    source_dir = tmp_path / "MunkTS"
    source_dir.mkdir()
    for name in ("MunkS.env", "MunkB.env", "MunkS.flp", "MunkS.rts.mat"):
        (source_dir / name).write_text(name, encoding="utf-8")

    prepared = prepare_at_munkts_work_dir(
        source_dir=source_dir,
        work_dir=tmp_path / "work",
    )

    assert prepared.work_dir == tmp_path / "work"
    assert prepared.reference_path == tmp_path / "work" / "MunkS.rts.mat"
    assert {path.name for path in prepared.copied_paths} == {"MunkS.env", "MunkB.env", "MunkS.flp", "MunkS.rts.mat"}
    assert prepared.reference_path.read_text(encoding="utf-8") == "MunkS.rts.mat"


def test_prepare_at_munkts_work_dir_rejects_existing_files(tmp_path):
    source_dir = tmp_path / "MunkTS"
    source_dir.mkdir()
    for name in ("MunkS.env", "MunkB.env", "MunkS.flp", "MunkS.rts.mat"):
        (source_dir / name).write_text(name, encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "MunkS.env").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_at_munkts_work_dir(source_dir=source_dir, work_dir=work_dir)


def test_prepare_at_munkts_work_dir_requires_fixture_files(tmp_path):
    source_dir = tmp_path / "MunkTS"
    source_dir.mkdir()
    (source_dir / "MunkS.env").write_text("env", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="MunkB.env"):
        prepare_at_munkts_work_dir(source_dir=source_dir, work_dir=tmp_path / "work")


def test_run_at_delayandsum_script_uses_octave_command_and_checks_output(monkeypatch, tmp_path):
    at_matlab_dir = tmp_path / "AT Matlab"
    at_matlab_dir.mkdir()
    at_bin_dir = tmp_path / "bin"
    at_bin_dir.mkdir()

    def fake_run(command, **kwargs):
        assert command[0] == "/usr/bin/octave"
        assert command[1:3] == ("--quiet", "--eval")
        assert "addpath(genpath(" in command[3]
        assert f"addpath('{at_bin_dir}')" in command[3]
        assert "scooter MunkS;" in command[3]
        assert "stack;" in command[3]
        assert command[3].strip().endswith("delayandsum;")
        assert kwargs["cwd"] == tmp_path
        (tmp_path / "MunkB.rts.mat").write_bytes(b"mat")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda name: "/usr/bin/octave")
    monkeypatch.setattr("pyacoustics.postprocessing.external.subprocess.run", fake_run)

    run = run_at_delayandsum_script(
        work_dir=tmp_path,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        runtime="octave",
        prelude_commands=("scooter MunkS", "stack"),
    )

    assert run.returncode == 0
    assert run.output_path == tmp_path / "MunkB.rts.mat"
    assert run.stdout == "ok"


def test_run_at_delayandsum_script_discovers_sibling_at_bin_dir(monkeypatch, tmp_path):
    at_root = tmp_path / "at"
    at_matlab_dir = at_root / "Matlab"
    at_bin_dir = at_root / "bin"
    at_matlab_dir.mkdir(parents=True)
    at_bin_dir.mkdir()

    def fake_run(command, **kwargs):
        assert f"addpath('{at_bin_dir}')" in command[3]
        (tmp_path / "MunkB.rts.mat").write_bytes(b"mat")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda name: "/usr/bin/octave")
    monkeypatch.setattr("pyacoustics.postprocessing.external.subprocess.run", fake_run)

    run = run_at_delayandsum_script(
        work_dir=tmp_path,
        at_matlab_dir=at_matlab_dir,
        runtime="octave",
    )

    assert run.output_path == tmp_path / "MunkB.rts.mat"


def test_at_delayandsum_external_benchmark_row_compares_generated_payload(monkeypatch, tmp_path):
    at_matlab_dir = tmp_path / "Matlab"
    at_matlab_dir.mkdir()
    reference_path = tmp_path / "reference.rts.mat"
    payload = {
        "PlotTitle": "MunkB",
        "Pos": {
            "s": {"z": np.array([1000.0])},
            "r": {"z": np.array([100.0, 200.0]), "r": np.array([5000.0])},
        },
        "tout": np.array([[6.0, 6.001, 6.002]]),
        "RTS": np.array([[0.0, 1.0], [0.5, 0.0], [0.0, -0.25]]),
    }
    savemat(reference_path, payload)

    def fake_run(command, **_kwargs):
        savemat(tmp_path / "MunkB.rts.mat", payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda name: "/Applications/MATLAB/bin/matlab")
    monkeypatch.setattr("pyacoustics.postprocessing.external.subprocess.run", fake_run)

    row = at_delayandsum_external_benchmark_row(
        reference_path=reference_path,
        work_dir=tmp_path,
        at_matlab_dir=at_matlab_dir,
        runtime="matlab",
        prelude_commands=("scooter MunkS", "stack"),
    )

    assert row["kind"] == "postprocessing_at_delayandsum_external"
    assert row["mat_payload_variable_count"] == 2
    assert row["timeseries_max_abs"] == 0.0
    assert row["postprocessing_external_returncode"] == 0
    assert row["postprocessing_external_runtime"].endswith("matlab")
    assert row["postprocessing_external_prelude_commands"] == "scooter MunkS; stack"


def test_at_matched_filter_external_benchmark_row_compares_generated_payload(monkeypatch, tmp_path):
    at_matlab_dir = tmp_path / "Matlab"
    at_matlab_dir.mkdir()
    reference_path = tmp_path / "reference.matched_filter.mat"
    payload = _matched_filter_payload()
    savemat(reference_path, payload)

    def fake_run(command, **_kwargs):
        assert command[0] == "/Applications/MATLAB/bin/matlab"
        assert "prepare_matched_filter;" in command[2]
        assert command[2].strip().endswith("matchedfilter;")
        savemat(tmp_path / "matched_filter.mat", payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda name: "/Applications/MATLAB/bin/matlab")
    monkeypatch.setattr("pyacoustics.postprocessing.external.subprocess.run", fake_run)

    row = at_matched_filter_external_benchmark_row(
        reference_path=reference_path,
        work_dir=tmp_path,
        at_matlab_dir=at_matlab_dir,
        runtime="matlab",
        prelude_commands=("prepare_matched_filter",),
    )

    assert row["kind"] == "postprocessing_at_matched_filter_external"
    assert row["mat_payload_variable_count"] == 11
    assert row["timeseries_max_abs"] == 0.0
    assert row["postprocessing_external_returncode"] == 0
    assert row["postprocessing_external_runtime"].endswith("matlab")
    assert row["postprocessing_external_prelude_commands"] == "prepare_matched_filter; matchedfilter"

def test_at_munkts_stack_external_benchmark_row_prepares_fixture_and_compares(monkeypatch, tmp_path):
    source_dir = tmp_path / "MunkTS"
    source_dir.mkdir()
    for name in ("MunkS.env", "MunkB.env", "MunkS.flp"):
        (source_dir / name).write_text(name, encoding="utf-8")
    payload = {
        "PlotTitle": "MunkB",
        "Pos": {
            "s": {"z": np.array([1000.0])},
            "r": {"z": np.array([100.0, 200.0]), "r": np.array([5000.0])},
        },
        "tout": np.array([[6.0, 6.001, 6.002]]),
        "RTS": np.array([[0.0, 1.0], [0.5, 0.0], [0.0, -0.25]]),
    }
    savemat(source_dir / "MunkS.rts.mat", payload)
    at_matlab_dir = tmp_path / "Matlab"
    at_matlab_dir.mkdir()
    at_bin_dir = tmp_path / "bin"
    at_bin_dir.mkdir()
    work_dir = tmp_path / "work"

    def fake_run(command, **_kwargs):
        assert "scooter MunkS;" in command[3]
        assert "stack;" in command[3]
        savemat(work_dir / "MunkS.rts.mat", payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda name: "/usr/bin/octave")
    monkeypatch.setattr("pyacoustics.postprocessing.external.subprocess.run", fake_run)

    row = at_munkts_stack_external_benchmark_row(
        source_dir=source_dir,
        work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        runtime="octave",
    )

    assert row["case"] == "postprocessing-at-munkts-stack-external-rts-payload"
    assert row["kind"] == "postprocessing_at_stack_external"
    assert row["timeseries_max_abs"] == 0.0
    assert row["postprocessing_external_prepared_file_count"] == 3
    assert row["postprocessing_fixture"] == str(source_dir / "MunkS.rts.mat")
    assert (work_dir / "MunkS.env").exists()
    assert (work_dir / "MunkB.env").exists()
    assert (work_dir / "MunkS.flp").exists()


def test_postprocessing_benchmark_matrix_can_include_external_delayandsum_row(monkeypatch, tmp_path):
    at_matlab_dir = tmp_path / "Matlab"
    at_matlab_dir.mkdir()
    reference_path = tmp_path / "reference.rts.mat"
    payload = {
        "PlotTitle": "MunkB",
        "Pos": {
            "s": {"z": np.array([1000.0])},
            "r": {"z": np.array([100.0, 200.0]), "r": np.array([5000.0])},
        },
        "tout": np.array([[6.0, 6.001, 6.002]]),
        "RTS": np.array([[0.0, 1.0], [0.5, 0.0], [0.0, -0.25]]),
    }
    savemat(reference_path, payload)

    def fake_run(command, **_kwargs):
        savemat(tmp_path / "MunkB.rts.mat", payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda name: "/usr/bin/octave")
    monkeypatch.setattr("pyacoustics.postprocessing.external.subprocess.run", fake_run)

    matrix = run_postprocessing_benchmark_matrix(
        thresholds=postprocessing_smoke_thresholds(),
        at_delayandsum_reference_path=reference_path,
        at_delayandsum_work_dir=tmp_path,
        at_matlab_dir=at_matlab_dir,
        matlab_runtime="octave",
        at_delayandsum_prelude_commands=("scooter MunkS", "stack"),
    )

    external_row = next(row for row in matrix.rows if row["kind"] == "postprocessing_at_delayandsum_external")
    assert external_row["passed"]
    assert external_row["mat_payload_variable_count"] == 2
    assert external_row["timeseries_max_abs"] == 0.0
    assert external_row["postprocessing_external_runtime"].endswith("octave")
    assert external_row["postprocessing_external_prelude_commands"] == "scooter MunkS; stack"


def test_postprocessing_benchmark_matrix_can_include_munkts_external_delayandsum_row(monkeypatch, tmp_path):
    source_dir = tmp_path / "MunkTS"
    source_dir.mkdir()
    for name in ("MunkS.env", "MunkB.env", "MunkS.flp"):
        (source_dir / name).write_text(name, encoding="utf-8")
    payload = {
        "PlotTitle": "MunkB",
        "Pos": {
            "s": {"z": np.array([1000.0])},
            "r": {"z": np.array([100.0, 200.0]), "r": np.array([5000.0])},
        },
        "tout": np.array([[6.0, 6.001, 6.002]]),
        "RTS": np.array([[0.0, 1.0], [0.5, 0.0], [0.0, -0.25]]),
    }
    savemat(source_dir / "MunkS.rts.mat", payload)
    at_matlab_dir = tmp_path / "Matlab"
    at_matlab_dir.mkdir()
    at_bin_dir = tmp_path / "bin"
    at_bin_dir.mkdir()
    work_dir = tmp_path / "work"

    def fake_run(command, **_kwargs):
        savemat(work_dir / "MunkS.rts.mat", payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda name: "/usr/bin/octave")
    monkeypatch.setattr("pyacoustics.postprocessing.external.subprocess.run", fake_run)

    matrix = run_postprocessing_benchmark_matrix(
        thresholds=postprocessing_smoke_thresholds(),
        at_munkts_source_dir=source_dir,
        at_delayandsum_work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        matlab_runtime="octave",
    )

    external_row = next(row for row in matrix.rows if row["case"] == "postprocessing-at-munkts-stack-external-rts-payload")
    assert external_row["passed"]
    assert external_row["postprocessing_external_prepared_file_count"] == 3
    assert external_row["timeseries_max_abs"] == 0.0


def test_postprocessing_benchmark_matrix_can_include_external_matched_filter_row(monkeypatch, tmp_path):
    at_matlab_dir = tmp_path / "Matlab"
    at_matlab_dir.mkdir()
    reference_path = tmp_path / "reference.matched_filter.mat"
    payload = _matched_filter_payload()
    savemat(reference_path, payload)

    def fake_run(command, **_kwargs):
        savemat(tmp_path / "matched_filter.mat", payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda name: "/usr/bin/octave")
    monkeypatch.setattr("pyacoustics.postprocessing.external.subprocess.run", fake_run)

    matrix = run_postprocessing_benchmark_matrix(
        thresholds=postprocessing_smoke_thresholds(),
        at_matched_filter_reference_path=reference_path,
        at_matched_filter_work_dir=tmp_path,
        at_matlab_dir=at_matlab_dir,
        matlab_runtime="octave",
        at_matched_filter_prelude_commands=("prepare_matched_filter",),
    )

    external_row = next(row for row in matrix.rows if row["kind"] == "postprocessing_at_matched_filter_external")
    assert external_row["passed"]
    assert external_row["mat_payload_variable_count"] == 11
    assert external_row["timeseries_max_abs"] == 0.0
    assert external_row["postprocessing_external_runtime"].endswith("octave")
    assert external_row["postprocessing_external_prelude_commands"] == "prepare_matched_filter; matchedfilter"

def test_postprocessing_benchmark_matrix_records_external_delayandsum_failure(monkeypatch, tmp_path):
    at_matlab_dir = tmp_path / "Matlab"
    at_matlab_dir.mkdir()
    reference_path = tmp_path / "reference.rts.mat"
    savemat(
        reference_path,
        {
            "PlotTitle": "MunkB",
            "Pos": {"s": {"z": np.array([1000.0])}, "r": {"z": np.array([100.0]), "r": np.array([5000.0])}},
            "tout": np.array([[6.0]]),
            "RTS": np.array([[1.0]]),
        },
    )
    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda _name: None)

    matrix = run_postprocessing_benchmark_matrix(
        thresholds=postprocessing_smoke_thresholds(),
        at_delayandsum_reference_path=reference_path,
        at_delayandsum_work_dir=tmp_path,
        at_matlab_dir=at_matlab_dir,
        matlab_runtime="missing-octave",
    )

    row = next(row for row in matrix.rows if row["case"] == "postprocessing-at-delayandsum-external-rts-payload")
    assert row["passed"] is False
    assert row["sample_count"] == 0
    assert row["mat_payload_variable_count"] == 0
    assert row["postprocessing_external_ok"] is False
    assert row["postprocessing_external_runtime"] == "missing-octave"
    assert row["postprocessing_external_output_path"] == str(tmp_path / "MunkB.rts.mat")
    assert row["postprocessing_fixture"] == str(reference_path)
    assert "Matlab-compatible runtime not found" in row["postprocessing_external_error"]
    assert "sample_count=0 < 1" in row["failures"]


def test_postprocessing_external_failure_row_carries_failure_summary(tmp_path):
    row = postprocessing_external_failure_row(
        "postprocessing-external-failure",
        "postprocessing_at_delayandsum_external",
        RuntimeError("runtime failed"),
        runtime="octave",
        output_path=tmp_path / "MunkB.rts.mat",
        prelude_commands=("scooter MunkS", "stack"),
        prepared_file_count=3,
        fixture=tmp_path / "reference.rts.mat",
    )

    assert row["passed"] is False
    assert row["failure_count"] == 1
    assert row["failures"] == "runtime failed"
    assert row["postprocessing_external_error"] == "runtime failed"
    assert row["postprocessing_external_output_path"] == str(tmp_path / "MunkB.rts.mat")
    assert row["postprocessing_external_prelude_commands"] == "scooter MunkS; stack"
    assert row["postprocessing_external_prepared_file_count"] == 3
    assert row["postprocessing_fixture"] == str(tmp_path / "reference.rts.mat")


def test_postprocessing_benchmark_matrix_records_munkts_external_failure(monkeypatch, tmp_path):
    source_dir = tmp_path / "MunkTS"
    source_dir.mkdir()
    for name in ("MunkS.env", "MunkB.env", "MunkS.flp"):
        (source_dir / name).write_text(name, encoding="utf-8")
    savemat(
        source_dir / "MunkS.rts.mat",
        {
            "PlotTitle": "MunkB",
            "Pos": {"s": {"z": np.array([1000.0])}, "r": {"z": np.array([100.0]), "r": np.array([5000.0])}},
            "tout": np.array([[6.0]]),
            "RTS": np.array([[1.0]]),
        },
    )
    at_matlab_dir = tmp_path / "Matlab"
    at_matlab_dir.mkdir()
    work_dir = tmp_path / "work"
    monkeypatch.setattr("pyacoustics.postprocessing.external.shutil.which", lambda _name: None)

    matrix = run_postprocessing_benchmark_matrix(
        thresholds=postprocessing_smoke_thresholds(),
        at_munkts_source_dir=source_dir,
        at_delayandsum_work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        matlab_runtime="missing-octave",
    )

    row = next(row for row in matrix.rows if row["case"] == "postprocessing-at-munkts-stack-external-rts-payload")
    assert row["passed"] is False
    assert row["postprocessing_external_ok"] is False
    assert row["postprocessing_external_runtime"] == "missing-octave"
    assert row["postprocessing_external_output_path"] == str(work_dir / "MunkS.rts.mat")
    assert row["postprocessing_fixture"] == str(source_dir / "MunkS.rts.mat")
    assert "scooter MunkS; stack" == row["postprocessing_external_prelude_commands"]
    assert "Matlab-compatible runtime not found" in row["postprocessing_external_error"]


def test_postprocessing_benchmark_matrix_rejects_mixed_munkts_and_generic_external_args(tmp_path):
    with pytest.raises(ValueError, match="Use either at_munkts_source_dir"):
        run_postprocessing_benchmark_matrix(
            at_munkts_source_dir=tmp_path / "MunkTS",
            at_delayandsum_reference_path=tmp_path / "reference.rts.mat",
            at_delayandsum_work_dir=tmp_path / "work",
            at_matlab_dir=tmp_path / "Matlab",
        )


def test_postprocessing_benchmark_matrix_requires_complete_external_delayandsum_args(tmp_path):
    with pytest.raises(ValueError, match="reference_path, work_dir, and at_matlab_dir"):
        run_postprocessing_benchmark_matrix(
            at_delayandsum_reference_path=tmp_path / "reference.rts.mat",
        )


def test_read_rts_timeseries_normalizes_at_mat_payload_to_time_major():
    fixture_value = os.environ.get("AT_MUNKTS_RTS_FIXTURE")
    if not fixture_value:
        pytest.skip("AT_MUNKTS_RTS_FIXTURE is not set")
    payload_path = Path(fixture_value)
    if not payload_path.is_file():
        pytest.skip("AT_MUNKTS_RTS_FIXTURE does not point to a file")

    payload = read_rts_timeseries(payload_path)

    assert "Munk" in payload.plot_title
    assert payload.tout.ndim == 1
    assert payload.rts.shape[0] == payload.tout.size
    assert payload.rts.shape[1] == 10
    assert payload.pos is not None
    assert np.isfinite(payload.rts).any()


def test_read_rts_timeseries_reads_ascii_layout(tmp_path):
    ts_path = tmp_path / "simple.rts"
    ts_path.write_text(
        "simple\n"
        "2\n"
        "50.0 100.0\n"
        "0.0 1.0 2.0\n"
        "0.1 3.0 4.0\n",
        encoding="utf-8",
    )

    payload = read_rts_timeseries(ts_path)

    assert payload.plot_title == "simple"
    np.testing.assert_allclose(payload.tout, [0.0, 0.1])
    np.testing.assert_allclose(payload.rts, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(payload.pos["r"]["z"], [50.0, 100.0])


def test_read_rts_timeseries_accepts_at_style_ascii_numeric_variants(tmp_path):
    ts_path = tmp_path / "at_style.rts"
    ts_path.write_text(
        "at-style\n"
        "2, ! receiver depth count\n"
        "5.0D1, 1.0D2 /\n"
        "0.0D0, 1.0D0, 2.0D0\n"
        "1.0D-1 3.0D0 4.0D0 # sample row\n",
        encoding="utf-8",
    )

    payload = read_rts_timeseries(ts_path)

    assert payload.plot_title == "at-style"
    np.testing.assert_allclose(payload.tout, [0.0, 0.1])
    np.testing.assert_allclose(payload.rts, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(payload.pos["r"]["z"], [50.0, 100.0])


def test_compare_rts_timeseries_files_uses_read_ts_normalization(tmp_path):
    reference_path = tmp_path / "reference.rts.mat"
    candidate_path = tmp_path / "candidate.rts.mat"
    common = {
        "PlotTitle": "read-ts",
        "Pos": {"r": {"z": np.array([50.0, 100.0])}},
        "tout": np.array([[0.0, 0.1, 0.2]]),
    }
    savemat(reference_path, {**common, "RTS": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])})
    savemat(candidate_path, {**common, "RTS": np.array([[1.0, 2.0], [3.25, 4.0], [5.0, 6.5]])})

    metrics = compare_rts_timeseries_files(reference_path, candidate_path)

    assert metrics.sample_count == 6
    assert np.isclose(metrics.max_abs, 0.5)
    assert np.isclose(metrics.mae, 0.125)


def test_compare_rts_mat_payload_requires_workflow_fields(tmp_path):
    reference_path = tmp_path / "reference.rts.mat"
    candidate_path = tmp_path / "candidate.rts.mat"
    savemat(reference_path, {"RTS": np.array([[1.0]]), "tout": np.array([[0.0]])})
    savemat(candidate_path, {"RTS": np.array([[1.0]]), "tout": np.array([[0.0]])})

    with pytest.raises(KeyError, match="sample_rate"):
        compare_rts_mat_payload(reference_path, candidate_path)


def test_compare_rts_mat_payload_reports_variable_shape_mismatch(tmp_path):
    full_payload = {
        "PlotTitle": "case",
        "RTS": np.array([[1.0]]),
        "tout": np.array([[0.0]]),
        "sample_rate": np.array([[1000.0]]),
        "receiver_depths": np.array([100.0]),
        "receiver_ranges": np.array([0.0]),
        "source_index": np.array([[0]]),
        "source_depths": np.array([50.0]),
        "selected_source_depth": np.array([50.0]),
        "source_x_index": np.array([[0]]),
        "source_y_index": np.array([[0]]),
        "source_x": np.array([]),
        "source_y": np.array([]),
        "selected_source_x": np.array([]),
        "selected_source_y": np.array([]),
        "receiver_bearings": np.array([]),
    }
    reference_path = tmp_path / "reference.rts.mat"
    candidate_path = tmp_path / "candidate.rts.mat"
    savemat(reference_path, full_payload)
    savemat(candidate_path, {**full_payload, "RTS": np.array([[1.0, 2.0]])})

    with pytest.raises(ValueError, match="RTS.*shape mismatch"):
        compare_rts_mat_payload(reference_path, candidate_path)
