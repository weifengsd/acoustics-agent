from pyacoustics.solvers.benchmark_thresholds import BenchmarkThresholds, evaluate_benchmark_row


def test_evaluate_benchmark_row_reports_all_threshold_failures():
    row = {
        "case": "threshold-smoke",
        "tl_mae_db": 1.25,
        "match_fraction": 0.8,
        "mode_count_delta": -2,
        "at_completed_without_errors": False,
    }
    thresholds = BenchmarkThresholds(
        max_values={"tl_mae_db": 1.0},
        min_values={"match_fraction": 0.9},
        max_abs_values={"mode_count_delta": 1},
        equals={"case": "threshold-smoke"},
        required_true=("at_completed_without_errors",),
    )

    evaluated = evaluate_benchmark_row(row, thresholds)

    assert not evaluated["passed"]
    assert evaluated["failure_count"] == 4
    assert "tl_mae_db=1.25 > 1" in evaluated["failures"]
    assert "match_fraction=0.8 < 0.9" in evaluated["failures"]
    assert "|mode_count_delta|=2 > 1" in evaluated["failures"]
    assert "at_completed_without_errors is not true" in evaluated["failures"]


def test_evaluate_benchmark_row_marks_passing_rows():
    row = {"case": "pass", "tl_mae_db": 0.25, "match_fraction": 1.0}
    thresholds = BenchmarkThresholds(max_values={"tl_mae_db": 0.5}, min_values={"match_fraction": 0.95})

    evaluated = evaluate_benchmark_row(row, thresholds)

    assert evaluated["passed"]
    assert evaluated["failure_count"] == 0
    assert evaluated["failures"] is None


def test_evaluate_benchmark_row_can_ignore_missing_metrics_for_mixed_rows():
    row = {"case": "arrival-only", "match_fraction": 1.0}
    strict = evaluate_benchmark_row(row, BenchmarkThresholds(max_values={"tl_mae_db": 1.0}))
    mixed = evaluate_benchmark_row(row, BenchmarkThresholds(max_values={"tl_mae_db": 1.0}, ignore_missing=True))

    assert not strict["passed"]
    assert strict["failures"] == "tl_mae_db missing"
    assert mixed["passed"]


def test_evaluate_benchmark_row_can_ignore_missing_equalities_for_mixed_rows():
    row = {"case": "flp-row", "file_format_roundtrip_ok": True}
    strict = evaluate_benchmark_row(row, BenchmarkThresholds(equals={"prt_issue_count": 0}))
    mixed = evaluate_benchmark_row(row, BenchmarkThresholds(equals={"prt_issue_count": 0}, ignore_missing=True))
    present_mismatch = evaluate_benchmark_row(
        {"case": "prt-row", "prt_issue_count": 1},
        BenchmarkThresholds(equals={"prt_issue_count": 0}, ignore_missing=True),
    )

    assert not strict["passed"]
    assert strict["failures"] == "prt_issue_count=None != 0"
    assert mixed["passed"]
    assert not present_mismatch["passed"]
    assert present_mismatch["failures"] == "prt_issue_count=1 != 0"


def test_evaluate_benchmark_row_rejects_zero_sample_rows():
    row = {"case": "empty-tl", "sample_count": 0, "tl_mae_db": None}
    thresholds = BenchmarkThresholds(min_values={"sample_count": 1.0}, ignore_missing=True)

    evaluated = evaluate_benchmark_row(row, thresholds)

    assert not evaluated["passed"]
    assert evaluated["failures"] == "sample_count=0 < 1"
