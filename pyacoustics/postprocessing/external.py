"""External Matlab/Octave runners for AT post-processing parity."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pyacoustics.postprocessing.parity import (
    compare_at_rts_mat_payload,
    compare_matched_filter_mat_payload,
    mat_payload_benchmark_row,
)
from pyacoustics.solvers.benchmark_thresholds import BenchmarkRow


@dataclass(frozen=True)
class ATDelayAndSumExternalRun:
    """Result of running AT ``delayandsum.m`` through Matlab or Octave."""

    runtime: str
    command: tuple[str, ...]
    work_dir: Path
    output_path: Path
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ATMunkTSWorkDir:
    """Prepared AT MunkTS working directory for external parity runs."""

    work_dir: Path
    reference_path: Path
    copied_paths: tuple[Path, ...]


def postprocessing_external_failure_row(
    case_name: str,
    kind: str,
    error: Exception,
    *,
    runtime: str | None = None,
    output_path: str | Path | None = None,
    prelude_commands: tuple[str, ...] | list[str] = (),
    prepared_file_count: int | None = None,
    fixture: str | Path | None = None,
) -> BenchmarkRow:
    """Return a benchmark row for a failed optional external AT workflow."""

    row: BenchmarkRow = {
        "case": case_name,
        "kind": kind,
        "passed": False,
        "failure_count": 1,
        "failures": str(error),
        "sample_count": 0,
        "mat_payload_variable_count": 0,
        "postprocessing_external_ok": False,
        "postprocessing_external_runtime": runtime,
        "postprocessing_external_returncode": None,
        "postprocessing_external_output_path": str(output_path) if output_path is not None else None,
        "postprocessing_external_prelude_commands": "; ".join(prelude_commands),
        "postprocessing_external_error": str(error),
    }
    if prepared_file_count is not None:
        row["postprocessing_external_prepared_file_count"] = prepared_file_count
    if fixture is not None:
        row["postprocessing_fixture"] = str(fixture)
    return row


def prepare_at_munkts_work_dir(
    *,
    source_dir: str | Path,
    work_dir: str | Path,
    include_reference: bool = True,
    overwrite: bool = False,
) -> ATMunkTSWorkDir:
    """Copy AT MunkTS inputs into a scratch directory for external parity.

    The copied files are the minimal checked-in AT MunkTS inputs needed before
    a Matlab/Octave prelude such as ``scooter MunkS; stack`` can generate the
    runtime files consumed by ``delayandsum.m``. The reference ``MunkS.rts.mat``
    is copied by default so it can be used directly as the parity reference.
    """

    source_path = Path(source_dir)
    work_path = Path(work_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"AT MunkTS source directory not found: {source_path}")
    work_path.mkdir(parents=True, exist_ok=True)
    filenames = ["MunkS.env", "MunkB.env", "MunkS.flp"]
    if include_reference:
        filenames.append("MunkS.rts.mat")
    copied: list[Path] = []
    for filename in filenames:
        src = source_path / filename
        if not src.exists():
            raise FileNotFoundError(f"AT MunkTS fixture missing {src}")
        dst = work_path / filename
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
        shutil.copy2(src, dst)
        copied.append(dst)
    return ATMunkTSWorkDir(
        work_dir=work_path,
        reference_path=work_path / "MunkS.rts.mat",
        copied_paths=tuple(copied),
    )


def find_matlab_runtime(preferred: str | None = None) -> str:
    """Return a Matlab-compatible runtime executable path.

    ``preferred`` may be an explicit executable path/name. Without it, Matlab
    is preferred over Octave because AT scripts are authored for Matlab, while
    Octave remains a useful open-source fallback.
    """

    candidates = (preferred,) if preferred else ("matlab", "octave")
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    requested = preferred if preferred else "matlab or octave"
    raise FileNotFoundError(f"Matlab-compatible runtime not found: {requested}")


def _matlab_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _runtime_command(runtime: str, code: str) -> tuple[str, ...]:
    executable = Path(runtime).name.lower()
    if "octave" in executable:
        return (runtime, "--quiet", "--eval", code)
    return (runtime, "-batch", code)


def _resolved_at_bin_dir(at_matlab_dir: Path, at_bin_dir: str | Path | None) -> Path | None:
    if at_bin_dir is not None:
        return Path(at_bin_dir)
    sibling_bin = at_matlab_dir.parent / "bin"
    return sibling_bin if sibling_bin.is_dir() else None


def _run_matlab_commands(
    *,
    work_dir: str | Path,
    at_matlab_dir: str | Path,
    at_bin_dir: str | Path | None = None,
    commands: tuple[str, ...] | list[str],
    output_filename: str,
    runtime: str | None = None,
    timeout_s: float = 300.0,
) -> ATDelayAndSumExternalRun:
    work_path = Path(work_dir)
    matlab_path = Path(at_matlab_dir)
    runtime_path = find_matlab_runtime(runtime)
    output_path = work_path / output_filename
    command_text = " ".join(f"{command.strip()};" for command in commands if command.strip())
    path_setup = f"addpath(genpath({_matlab_string(matlab_path)})); "
    resolved_bin_path = _resolved_at_bin_dir(matlab_path, at_bin_dir)
    if resolved_bin_path is not None:
        path_setup += f"addpath({_matlab_string(resolved_bin_path)}); "
    code = path_setup + f"cd({_matlab_string(work_path)}); " + command_text
    command = _runtime_command(runtime_path, code)
    completed = subprocess.run(
        command,
        cwd=work_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    run = ATDelayAndSumExternalRun(
        runtime=runtime_path,
        command=command,
        work_dir=work_path,
        output_path=output_path,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "AT Matlab/Octave external run failed "
            f"with return code {completed.returncode}: {completed.stderr[-1000:]}"
        )
    if not output_path.exists():
        raise FileNotFoundError(f"AT Matlab/Octave run did not create {output_path}")
    return run


def run_at_delayandsum_script(
    *,
    work_dir: str | Path,
    at_matlab_dir: str | Path,
    at_bin_dir: str | Path | None = None,
    runtime: str | None = None,
    prelude_commands: tuple[str, ...] | list[str] = (),
    output_filename: str = "MunkB.rts.mat",
    timeout_s: float = 300.0,
) -> ATDelayAndSumExternalRun:
    """Run AT ``delayandsum.m`` in ``work_dir`` and return the generated payload.

    The working directory is expected to contain the arrivals file referenced by
    the AT script, normally ``MunkB.arr``. ``prelude_commands`` can run AT
    setup commands such as ``scooter MunkS`` before ``delayandsum``.
    ``at_matlab_dir`` should point at the AT ``Matlab`` directory so
    dependencies such as ``read_arrivals_asc`` and ``cans`` are on the Matlab
    path.
    """

    return _run_matlab_commands(
        work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        commands=(*tuple(prelude_commands), "delayandsum"),
        output_filename=output_filename,
        runtime=runtime,
        timeout_s=timeout_s,
    )


def run_at_matched_filter_script(
    *,
    work_dir: str | Path,
    at_matlab_dir: str | Path,
    at_bin_dir: str | Path | None = None,
    runtime: str | None = None,
    prelude_commands: tuple[str, ...] | list[str] = (),
    script_name: str = "matchedfilter",
    output_filename: str = "matched_filter.mat",
    timeout_s: float = 300.0,
) -> ATDelayAndSumExternalRun:
    """Run a Matlab/Octave matched-filter workflow and return its payload."""

    return _run_matlab_commands(
        work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        commands=(*tuple(prelude_commands), script_name),
        output_filename=output_filename,
        runtime=runtime,
        timeout_s=timeout_s,
    )

def at_delayandsum_external_benchmark_row(
    *,
    reference_path: str | Path,
    work_dir: str | Path,
    at_matlab_dir: str | Path,
    at_bin_dir: str | Path | None = None,
    runtime: str | None = None,
    prelude_commands: tuple[str, ...] | list[str] = (),
    output_filename: str = "MunkB.rts.mat",
    timeout_s: float = 300.0,
) -> BenchmarkRow:
    """Run AT ``delayandsum.m`` and compare its ``.rts.mat`` payload."""

    run = run_at_delayandsum_script(
        work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        runtime=runtime,
        prelude_commands=prelude_commands,
        output_filename=output_filename,
        timeout_s=timeout_s,
    )
    metrics = compare_at_rts_mat_payload(reference_path, run.output_path)
    row = mat_payload_benchmark_row("postprocessing-at-delayandsum-external-rts-payload", metrics)
    row.update(
        {
            "kind": "postprocessing_at_delayandsum_external",
            "postprocessing_external_ok": True,
            "postprocessing_external_runtime": run.runtime,
            "postprocessing_external_returncode": run.returncode,
            "postprocessing_external_output_path": str(run.output_path),
            "postprocessing_external_prelude_commands": "; ".join(prelude_commands),
        }
    )
    return row


def at_matched_filter_external_benchmark_row(
    *,
    reference_path: str | Path,
    work_dir: str | Path,
    at_matlab_dir: str | Path,
    at_bin_dir: str | Path | None = None,
    runtime: str | None = None,
    prelude_commands: tuple[str, ...] | list[str] = (),
    script_name: str = "matchedfilter",
    output_filename: str = "matched_filter.mat",
    timeout_s: float = 300.0,
) -> BenchmarkRow:
    """Run a Matlab matched-filter workflow and compare its ``.mat`` payload."""

    run = run_at_matched_filter_script(
        work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        runtime=runtime,
        prelude_commands=prelude_commands,
        script_name=script_name,
        output_filename=output_filename,
        timeout_s=timeout_s,
    )
    metrics = compare_matched_filter_mat_payload(reference_path, run.output_path)
    row = mat_payload_benchmark_row("postprocessing-at-matched-filter-external-payload", metrics)
    row.update(
        {
            "kind": "postprocessing_at_matched_filter_external",
            "postprocessing_external_ok": True,
            "postprocessing_external_runtime": run.runtime,
            "postprocessing_external_returncode": run.returncode,
            "postprocessing_external_output_path": str(run.output_path),
            "postprocessing_external_prelude_commands": "; ".join((*tuple(prelude_commands), script_name)),
        }
    )
    return row

def at_munkts_stack_external_benchmark_row(
    *,
    source_dir: str | Path,
    work_dir: str | Path,
    at_matlab_dir: str | Path,
    at_bin_dir: str | Path | None = None,
    runtime: str | None = None,
    commands: tuple[str, ...] | list[str] = ("scooter MunkS", "stack"),
    output_filename: str = "MunkS.rts.mat",
    timeout_s: float = 300.0,
    overwrite: bool = False,
) -> BenchmarkRow:
    """Prepare and run the AT MunkTS ``scooter``/``stack`` external parity row."""

    prepared = prepare_at_munkts_work_dir(
        source_dir=source_dir,
        work_dir=work_dir,
        include_reference=False,
        overwrite=overwrite,
    )
    source_path = Path(source_dir)
    reference_path = source_path / "MunkS.rts.mat"
    if not reference_path.exists():
        raise FileNotFoundError(f"AT MunkTS reference missing {reference_path}")
    run = _run_matlab_commands(
        work_dir=work_dir,
        at_matlab_dir=at_matlab_dir,
        at_bin_dir=at_bin_dir,
        commands=commands,
        output_filename=output_filename,
        runtime=runtime,
        timeout_s=timeout_s,
    )
    metrics = compare_at_rts_mat_payload(reference_path, run.output_path)
    row = mat_payload_benchmark_row("postprocessing-at-munkts-stack-external-rts-payload", metrics)
    row.update(
        {
            "kind": "postprocessing_at_stack_external",
            "postprocessing_fixture": str(reference_path),
            "postprocessing_external_ok": True,
            "postprocessing_external_runtime": run.runtime,
            "postprocessing_external_returncode": run.returncode,
            "postprocessing_external_output_path": str(run.output_path),
            "postprocessing_external_prelude_commands": "; ".join(commands),
            "postprocessing_external_prepared_file_count": len(prepared.copied_paths),
        }
    )
    return row


__all__ = [
    "ATDelayAndSumExternalRun",
    "ATMunkTSWorkDir",
    "at_delayandsum_external_benchmark_row",
    "at_matched_filter_external_benchmark_row",
    "at_munkts_stack_external_benchmark_row",
    "find_matlab_runtime",
    "postprocessing_external_failure_row",
    "prepare_at_munkts_work_dir",
    "run_at_matched_filter_script",
    "run_at_delayandsum_script",
]
