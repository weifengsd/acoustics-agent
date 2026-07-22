"""Convenience runner for canonical AT/native benchmark matrices."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pyacoustics.benchmarks.at_presets import (
    bellhop3d_smoke_thresholds,
    bellhop_smoke_thresholds,
    canonical_bellhop3d_arrival_benchmark_cases,
    canonical_bellhop3d_benchmark_cases,
    canonical_bellhop3d_ray_benchmark_cases,
    canonical_bellhop_benchmark_cases,
    canonical_kraken_benchmark_cases,
    kraken_smoke_thresholds,
)
from pyacoustics.benchmarks.file_formats import (
    at_file_format_smoke_thresholds,
    run_at_file_format_benchmark_matrix,
)
from pyacoustics.benchmarks.field3d_runtime import (
    canonical_field3d_runtime_benchmark_cases,
    field3d_runtime_smoke_thresholds,
    run_field3d_runtime_benchmark_matrix,
)
from pyacoustics.solvers.bellhop.benchmark_matrix import run_bellhop_benchmark_matrix
from pyacoustics.solvers.bellhop3d.arrival_parity import run_bellhop3d_arrival_benchmark_matrix
from pyacoustics.solvers.bellhop3d.ray_parity import run_bellhop3d_ray_benchmark_matrix
from pyacoustics.solvers.bellhop3d.tl_parity import run_bellhop3d_benchmark_matrix
from pyacoustics.solvers.benchmark_io import write_benchmark_rows_csv, write_benchmark_rows_json
from pyacoustics.solvers.benchmark_thresholds import BenchmarkRow
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.kraken.tl_parity import run_kraken_benchmark_matrix
from pyacoustics.postprocessing.benchmark_matrix import (
    postprocessing_smoke_thresholds,
    run_postprocessing_benchmark_matrix,
    write_postprocessing_report_plots,
)


BenchmarkSuite = Literal["bellhop", "kraken", "bellhop3d", "field3d", "postprocessing", "formats", "all", "full"]
BENCHMARK_SUITES: tuple[str, ...] = ("bellhop", "kraken", "bellhop3d", "field3d", "postprocessing", "formats", "all", "full")


@dataclass(frozen=True)
class BenchmarkMatrixRun:
    """Canonical benchmark matrix rows and exported artifact paths."""

    rows: tuple[BenchmarkRow, ...]
    csv_path: Path | None
    json_path: Path | None


def _write_outputs(
    output_dir: Path | None,
    stem: str,
    rows: tuple[BenchmarkRow, ...],
    *,
    write_csv: bool,
    write_json: bool,
) -> tuple[Path | None, Path | None]:
    if output_dir is None:
        return None, None
    csv_path = write_benchmark_rows_csv(output_dir / f"{stem}.csv", rows) if write_csv else None
    json_path = write_benchmark_rows_json(output_dir / f"{stem}.json", rows) if write_json else None
    return csv_path, json_path


def _skipped_row(case_name: str, kind: str, reason: str) -> BenchmarkRow:
    return {
        "case": case_name,
        "kind": kind,
        "passed": False,
        "skipped": True,
        "failure_count": 1,
        "failures": reason,
    }


def _failed_rows(rows: tuple[BenchmarkRow, ...] | list[BenchmarkRow]) -> tuple[BenchmarkRow, ...]:
    return tuple(row for row in rows if row.get("passed") is not True and not row.get("skipped"))


def _failure_summary_lines(rows: tuple[BenchmarkRow, ...] | list[BenchmarkRow], *, limit: int = 5) -> tuple[str, ...]:
    failed_rows = _failed_rows(rows)
    lines: list[str] = []
    for row in failed_rows[:limit]:
        case = row.get("case") or "<unknown>"
        reason = row.get("failures") or row.get("field3d_error") or row.get("postprocessing_external_error") or "failed"
        lines.append(f"- {case}: {reason}")
    remaining = len(failed_rows) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} more failed row(s)")
    return tuple(lines)


def write_benchmark_report_plots(output_dir: str | Path, rows: Sequence[BenchmarkRow], *, suite: str) -> tuple[Path, Path]:
    """Write compact benchmark summary figures for HTML reports."""

    output = Path(output_dir)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = figures_dir / ".matplotlib"
    font_cache_dir = figures_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    font_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    os.environ["XDG_CACHE_HOME"] = str(font_cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_count = len(rows)
    passed = sum(1 for row in rows if row.get("passed") is True)
    skipped = sum(1 for row in rows if row.get("skipped"))
    failed = row_count - passed - skipped

    status_path = figures_dir / f"{suite}_benchmark_status.png"
    coverage_path = figures_dir / f"{suite}_benchmark_kind_coverage.png"

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    labels = ("passed", "failed", "skipped")
    values = (passed, failed, skipped)
    colors = ("#087443", "#b42318", "#b26a00")
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title(f"{suite} benchmark status")
    ax.set_ylabel("row count")
    ax.set_ylim(0, max(values + (1,)) * 1.18)
    ax.grid(axis="y", color="#dbe2ea", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(status_path)
    plt.close(fig)

    kind_counts = Counter(str(row.get("kind") or "unknown") for row in rows)
    ordered = sorted(kind_counts.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        ordered = [("none", 0)]
    labels = [name for name, _count in ordered]
    values = [count for _name, count in ordered]
    height = max(4.2, 0.32 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, height), dpi=150)
    y_positions = range(len(labels))
    ax.barh(y_positions, values, color="#1769aa", height=0.58)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("row count")
    ax.set_title(f"{suite} benchmark coverage by row kind")
    ax.grid(axis="x", color="#dbe2ea", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for y, value in zip(y_positions, values):
        ax.text(value, y, f" {value}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(coverage_path)
    plt.close(fig)

    shutil.rmtree(mpl_config_dir, ignore_errors=True)
    shutil.rmtree(font_cache_dir, ignore_errors=True)
    return status_path, coverage_path


def _infer_at_matlab_dir_from_bin(at_bin_path: str | Path | None) -> Path | None:
    if at_bin_path is None:
        return None
    candidate = Path(at_bin_path).parent / "Matlab"
    return candidate if candidate.is_dir() else None


def run_canonical_at_benchmark_matrix(
    *,
    suite: BenchmarkSuite = "all",
    at_bin_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_root: str | Path | None = None,
    write_csv: bool = True,
    write_json: bool = True,
    skip_if_missing_at: bool = True,
    at_munkts_source_dir: str | Path | None = None,
    at_delayandsum_reference_path: str | Path | None = None,
    at_delayandsum_work_dir: str | Path | None = None,
    at_matched_filter_reference_path: str | Path | None = None,
    at_matched_filter_work_dir: str | Path | None = None,
    at_matlab_dir: str | Path | None = None,
    at_bin_dir: str | Path | None = None,
    matlab_runtime: str | None = None,
    at_delayandsum_prelude_commands: tuple[str, ...] | list[str] = (),
    at_delayandsum_output_filename: str = "MunkB.rts.mat",
    at_delayandsum_timeout_s: float = 300.0,
    at_matched_filter_prelude_commands: tuple[str, ...] | list[str] = (),
    at_matched_filter_script_name: str = "matchedfilter",
    at_matched_filter_output_filename: str = "matched_filter.mat",
    at_matched_filter_timeout_s: float = 300.0,
) -> BenchmarkMatrixRun:
    """Run canonical AT parity matrices and optionally export rows."""

    output = Path(output_dir) if output_dir is not None else None
    rows: list[BenchmarkRow] = []

    if suite in {"bellhop", "all", "full"}:
        bellhop_cases = canonical_bellhop_benchmark_cases()
        external = ExternalSolver(bellhop_cases[0].config, bin_path=str(at_bin_path) if at_bin_path else None)
        if skip_if_missing_at and not external.is_available("bellhop"):
            rows.extend(
                _skipped_row(case.name, case.kind, f"AT bellhop executable not found in {external.bin_path}")
                for case in bellhop_cases
            )
        else:
            bellhop_matrix = run_bellhop_benchmark_matrix(
                bellhop_cases,
                at_bin_path=at_bin_path,
                thresholds=bellhop_smoke_thresholds(),
            )
            rows.extend(bellhop_matrix.rows)

    if suite in {"kraken", "all", "full"}:
        kraken_cases = canonical_kraken_benchmark_cases()
        external = ExternalSolver(kraken_cases[0].config, bin_path=str(at_bin_path) if at_bin_path else None)
        has_kraken = external.is_available("kraken") or external.is_available("krakenc")
        has_field = external.is_available("field")
        if skip_if_missing_at and not (has_kraken and has_field):
            rows.extend(
                _skipped_row(case.name, "kraken_tl", f"AT kraken/field executables not found in {external.bin_path}")
                for case in kraken_cases
            )
        else:
            kraken_matrix = run_kraken_benchmark_matrix(
                kraken_cases,
                at_bin_path=at_bin_path,
                thresholds=kraken_smoke_thresholds(),
            )
            rows.extend(kraken_matrix.rows)

    if suite in {"bellhop3d", "full"}:
        bellhop3d_cases = canonical_bellhop3d_benchmark_cases()
        bellhop3d_arrival_cases = canonical_bellhop3d_arrival_benchmark_cases()
        bellhop3d_ray_cases = canonical_bellhop3d_ray_benchmark_cases()
        external = ExternalSolver(bellhop3d_cases[0].config, bin_path=str(at_bin_path) if at_bin_path else None)
        if skip_if_missing_at and not external.is_available("bellhop3d"):
            rows.extend(
                _skipped_row(case.name, "bellhop3d_tl", f"AT bellhop3d executable not found in {external.bin_path}")
                for case in bellhop3d_cases
            )
            rows.extend(
                _skipped_row(case.name, "bellhop3d_arrival", f"AT bellhop3d executable not found in {external.bin_path}")
                for case in bellhop3d_arrival_cases
            )
            rows.extend(
                _skipped_row(case.name, "bellhop3d_ray", f"AT bellhop3d executable not found in {external.bin_path}")
                for case in bellhop3d_ray_cases
            )
        else:
            bellhop3d_matrix = run_bellhop3d_benchmark_matrix(
                bellhop3d_cases,
                at_bin_path=at_bin_path,
                thresholds=bellhop3d_smoke_thresholds(),
            )
            rows.extend(bellhop3d_matrix.rows)
            bellhop3d_arrival_matrix = run_bellhop3d_arrival_benchmark_matrix(
                bellhop3d_arrival_cases,
                at_bin_path=at_bin_path,
            )
            rows.extend(bellhop3d_arrival_matrix.rows)
            bellhop3d_ray_matrix = run_bellhop3d_ray_benchmark_matrix(
                bellhop3d_ray_cases,
                at_bin_path=at_bin_path,
            )
            rows.extend(bellhop3d_ray_matrix.rows)

    if suite in {"field3d", "full"}:
        field3d_cases = canonical_field3d_runtime_benchmark_cases(sample_root=sample_root)
        external = ExternalSolver(None, bin_path=str(at_bin_path) if at_bin_path else None)
        has_kraken = external.is_available("kraken")
        has_field3d = external.is_available("field3d")
        if skip_if_missing_at and not (has_kraken and has_field3d):
            rows.extend(
                _skipped_row(
                    case.name,
                    "field3d_runtime",
                    f"AT kraken/field3d executables not found in {external.bin_path}",
                )
                for case in field3d_cases
            )
        else:
            field3d_matrix = run_field3d_runtime_benchmark_matrix(
                field3d_cases,
                at_bin_path=at_bin_path,
                thresholds=field3d_runtime_smoke_thresholds(),
            )
            rows.extend(field3d_matrix.rows)

    if suite in {"postprocessing", "full"}:
        postprocessing_kwargs = {}
        inferred_at_matlab_dir = at_matlab_dir or _infer_at_matlab_dir_from_bin(at_bin_path)
        inferred_at_bin_dir = at_bin_dir or at_bin_path
        wants_external_postprocessing = (
            at_delayandsum_reference_path is not None
            or at_munkts_source_dir is not None
            or at_delayandsum_work_dir is not None
            or at_matched_filter_reference_path is not None
            or at_matched_filter_work_dir is not None
            or at_matlab_dir is not None
            or at_bin_dir is not None
            or matlab_runtime is not None
        )
        if wants_external_postprocessing:
            postprocessing_kwargs.update(
                {
                    "at_munkts_source_dir": at_munkts_source_dir,
                    "at_delayandsum_reference_path": at_delayandsum_reference_path,
                    "at_delayandsum_work_dir": at_delayandsum_work_dir,
                    "at_matched_filter_reference_path": at_matched_filter_reference_path,
                    "at_matched_filter_work_dir": at_matched_filter_work_dir,
                    "at_matlab_dir": inferred_at_matlab_dir,
                    "at_bin_dir": inferred_at_bin_dir,
                    "matlab_runtime": matlab_runtime,
                    "at_delayandsum_prelude_commands": at_delayandsum_prelude_commands,
                    "at_delayandsum_output_filename": at_delayandsum_output_filename,
                    "at_delayandsum_timeout_s": at_delayandsum_timeout_s,
                    "at_matched_filter_prelude_commands": at_matched_filter_prelude_commands,
                    "at_matched_filter_script_name": at_matched_filter_script_name,
                    "at_matched_filter_output_filename": at_matched_filter_output_filename,
                    "at_matched_filter_timeout_s": at_matched_filter_timeout_s,
                }
            )
        postprocessing_matrix = run_postprocessing_benchmark_matrix(
            thresholds=postprocessing_smoke_thresholds(),
            **postprocessing_kwargs,
        )
        rows.extend(postprocessing_matrix.rows)
        if output is not None:
            write_postprocessing_report_plots(output)

    if suite in {"formats", "full"}:
        file_format_matrix = run_at_file_format_benchmark_matrix(
            sample_root=sample_root,
            thresholds=at_file_format_smoke_thresholds(),
        )
        rows.extend(file_format_matrix.rows)

    if suite not in {"bellhop", "kraken", "bellhop3d", "field3d", "postprocessing", "formats", "all", "full"}:
        raise ValueError(
            "suite must be one of 'bellhop', 'kraken', 'bellhop3d', "
            "'field3d', 'postprocessing', 'formats', 'all', or 'full'"
        )

    matrix_rows = tuple(rows)
    if output is not None:
        write_benchmark_report_plots(output, matrix_rows, suite=suite)
    csv_path, json_path = _write_outputs(
        output,
        f"canonical_{suite}_at_benchmark_matrix",
        matrix_rows,
        write_csv=write_csv,
        write_json=write_json,
    )
    return BenchmarkMatrixRun(rows=matrix_rows, csv_path=csv_path, json_path=json_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acoustics-at-benchmark",
        description="Run canonical Acoustics Toolbox/native benchmark matrices.",
    )
    parser.add_argument("--suite", choices=BENCHMARK_SUITES, default="all")
    parser.add_argument("--at-bin-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/at_benchmark"))
    parser.add_argument(
        "--sample-root",
        type=Path,
        help="Repository or fixture root containing tests/... samples for formats and Field3D runtime suites.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Do not write the CSV matrix artifact.")
    parser.add_argument("--no-json", action="store_true", help="Do not write the JSON matrix artifact.")
    parser.add_argument(
        "--strict-at",
        action="store_true",
        help="Fail instead of emitting skipped rows when AT executables are missing.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code 0 even when benchmark rows fail or are skipped.",
    )
    parser.add_argument("--at-munkts-source-dir", type=Path)
    parser.add_argument("--at-delayandsum-reference-path", type=Path)
    parser.add_argument("--at-delayandsum-work-dir", type=Path)
    parser.add_argument("--at-matched-filter-reference-path", type=Path)
    parser.add_argument("--at-matched-filter-work-dir", type=Path)
    parser.add_argument("--at-matlab-dir", type=Path)
    parser.add_argument("--at-bin-dir", type=Path)
    parser.add_argument("--matlab-runtime")
    parser.add_argument(
        "--at-delayandsum-prelude-command",
        action="append",
        default=[],
        help="Matlab/Octave command to run before delayandsum.m; may be repeated.",
    )
    parser.add_argument("--at-delayandsum-output-filename", default="MunkB.rts.mat")
    parser.add_argument("--at-delayandsum-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--at-matched-filter-prelude-command",
        action="append",
        default=[],
        help="Matlab/Octave command to run before the matched-filter script; may be repeated.",
    )
    parser.add_argument("--at-matched-filter-script-name", default="matchedfilter")
    parser.add_argument("--at-matched-filter-output-filename", default="matched_filter.mat")
    parser.add_argument("--at-matched-filter-timeout-s", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for canonical AT benchmark matrices."""

    args = _build_arg_parser().parse_args(argv)
    result = run_canonical_at_benchmark_matrix(
        suite=args.suite,
        at_bin_path=args.at_bin_path,
        output_dir=args.output_dir,
        sample_root=args.sample_root,
        write_csv=not args.no_csv,
        write_json=not args.no_json,
        skip_if_missing_at=not args.strict_at,
        at_munkts_source_dir=args.at_munkts_source_dir,
        at_delayandsum_reference_path=args.at_delayandsum_reference_path,
        at_delayandsum_work_dir=args.at_delayandsum_work_dir,
        at_matched_filter_reference_path=args.at_matched_filter_reference_path,
        at_matched_filter_work_dir=args.at_matched_filter_work_dir,
        at_matlab_dir=args.at_matlab_dir,
        at_bin_dir=args.at_bin_dir,
        matlab_runtime=args.matlab_runtime,
        at_delayandsum_prelude_commands=tuple(args.at_delayandsum_prelude_command),
        at_delayandsum_output_filename=args.at_delayandsum_output_filename,
        at_delayandsum_timeout_s=args.at_delayandsum_timeout_s,
        at_matched_filter_prelude_commands=tuple(args.at_matched_filter_prelude_command),
        at_matched_filter_script_name=args.at_matched_filter_script_name,
        at_matched_filter_output_filename=args.at_matched_filter_output_filename,
        at_matched_filter_timeout_s=args.at_matched_filter_timeout_s,
    )
    passed = sum(1 for row in result.rows if row.get("passed"))
    skipped = sum(1 for row in result.rows if row.get("skipped"))
    failed = len(_failed_rows(result.rows))
    print(f"{args.suite}: {passed}/{len(result.rows)} passed, {failed} failed, {skipped} skipped")
    if failed:
        print("Failures:")
        for line in _failure_summary_lines(result.rows):
            print(line)
    if result.csv_path is not None:
        print(f"CSV: {result.csv_path}")
    if result.json_path is not None:
        print(f"JSON: {result.json_path}")
    return 0 if args.allow_failures or passed == len(result.rows) else 1


__all__ = [
    "BENCHMARK_SUITES",
    "BenchmarkMatrixRun",
    "BenchmarkSuite",
    "main",
    "run_canonical_at_benchmark_matrix",
    "write_benchmark_report_plots",
]


if __name__ == "__main__":
    raise SystemExit(main())
