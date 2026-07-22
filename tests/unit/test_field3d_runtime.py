from pathlib import Path
import subprocess

import pytest

from pyacoustics.benchmarks import field3d_runtime as field3d_module
from pyacoustics.benchmarks.field3d_runtime import (
    Field3DRuntimeBenchmarkCase,
    field3d_runtime_smoke_thresholds,
    run_field3d_runtime_benchmark_matrix,
)


def _write_runtime_fixture(tmp_path: Path, *, case_names: tuple[str, ...] = ("slow",)) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "kraken").write_text("", encoding="utf-8")
    (bin_dir / "field3d").write_text("", encoding="utf-8")

    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    (sample_dir / "lanta36.env").write_text("'runtime env'\n", encoding="utf-8")
    for case_name in case_names:
        (sample_dir / f"{case_name}.flp").write_text("'runtime flp'\n", encoding="utf-8")
    return bin_dir, sample_dir


def test_field3d_runtime_records_field3d_execution_failure(monkeypatch, tmp_path):
    bin_dir, sample_dir = _write_runtime_fixture(tmp_path)
    calls = []

    def fake_run_external(exe_path, case_name, work_dir, *, timeout_s):
        calls.append((Path(exe_path).name, case_name, timeout_s))
        if case_name == "lanta36":
            (work_dir / "lanta36.mod").write_text("mode", encoding="utf-8")
            return
        raise RuntimeError("field3d timed out for slow after 0.01 seconds")

    monkeypatch.setattr(field3d_module, "_atlantic_environment_names", lambda: ("lanta36",))
    monkeypatch.setattr(field3d_module, "_run_external", fake_run_external)

    case = Field3DRuntimeBenchmarkCase(
        name="field3d-timeout",
        sample_dir=sample_dir,
        field3d_cases=("slow",),
        normalized_flp_cases=(),
        command_timeout_s=0.01,
    )
    matrix = run_field3d_runtime_benchmark_matrix(
        (case,),
        at_bin_path=bin_dir,
        thresholds=field3d_runtime_smoke_thresholds(),
    )

    assert calls == [("kraken", "lanta36", 0.01), ("field3d", "slow", 0.01)]
    row = matrix.rows[0]
    assert row["case"] == "field3d-timeout-slow"
    assert row["passed"] is False
    assert row["field3d_runtime_ok"] is False
    assert row["field3d_environment_count"] == 1
    assert row["field3d_mod_count"] == 1
    assert "field3d timed out" in row["field3d_error"]
    assert "field3d_runtime_ok is not true" in row["failures"]


def test_field3d_failure_row_carries_failure_summary_without_thresholds(tmp_path):
    case = Field3DRuntimeBenchmarkCase(
        name="field3d-failure",
        sample_dir=tmp_path,
        field3d_cases=("slow",),
        normalized_flp_cases=(),
    )

    row = field3d_module._field3d_failure_row(
        case,
        "slow",
        error=RuntimeError("field3d failed"),
        elapsed_s=1.25,
        environment_count=2,
        mod_count=1,
    )

    assert row["passed"] is False
    assert row["failure_count"] == 1
    assert row["failures"] == "field3d failed"
    assert row["field3d_error"] == "field3d failed"
    assert row["field3d_elapsed_s"] == 1.25


def test_field3d_runtime_records_kraken_setup_failure_for_all_cases(monkeypatch, tmp_path):
    bin_dir, sample_dir = _write_runtime_fixture(tmp_path, case_names=("a", "b"))

    def fake_run_external(exe_path, case_name, work_dir, *, timeout_s):
        raise RuntimeError(f"{Path(exe_path).name} failed for {case_name}")

    monkeypatch.setattr(field3d_module, "_atlantic_environment_names", lambda: ("lanta36",))
    monkeypatch.setattr(field3d_module, "_run_external", fake_run_external)

    case = Field3DRuntimeBenchmarkCase(
        name="field3d-setup-failure",
        sample_dir=sample_dir,
        field3d_cases=("a", "b"),
        normalized_flp_cases=(),
    )
    matrix = run_field3d_runtime_benchmark_matrix(
        (case,),
        at_bin_path=bin_dir,
        thresholds=field3d_runtime_smoke_thresholds(),
    )

    assert [row["case"] for row in matrix.rows] == ["field3d-setup-failure-a", "field3d-setup-failure-b"]
    assert all(row["passed"] is False for row in matrix.rows)
    assert all(row["field3d_environment_count"] == 1 for row in matrix.rows)
    assert all(row["field3d_mod_count"] == 0 for row in matrix.rows)
    assert all("kraken failed for lanta36" in row["field3d_error"] for row in matrix.rows)


def test_run_external_reports_timeout(monkeypatch, tmp_path):
    def fake_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["field3d", "slow"], timeout=0.5)

    monkeypatch.setattr(field3d_module.subprocess, "run", fake_subprocess_run)

    with pytest.raises(RuntimeError, match="field3d timed out for slow after 0.5 seconds"):
        field3d_module._run_external(tmp_path / "field3d", "slow", tmp_path, timeout_s=0.5)
