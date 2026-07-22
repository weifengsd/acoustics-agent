"""Runtime benchmark rows for AT Field3D scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from pyacoustics.solvers.benchmark_thresholds import (
    BenchmarkRow,
    BenchmarkThresholds,
    evaluate_benchmark_rows,
)
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.external_io import read_at_field_flp, read_at_prt, read_at_shd_full, write_at_field_flp


@dataclass(frozen=True)
class Field3DRuntimeBenchmarkCase:
    """One AT Field3D runtime benchmark scenario."""

    name: str
    sample_dir: Path
    field3d_cases: tuple[str, ...] = ("lant_90degrees", "lant", "lantvol")
    normalized_flp_cases: tuple[str, ...] = ("lantvol",)
    command_timeout_s: float = 30.0


@dataclass(frozen=True)
class Field3DRuntimeBenchmarkMatrix:
    """Field3D runtime benchmark rows."""

    rows: tuple[BenchmarkRow, ...]


def field3d_runtime_smoke_thresholds() -> BenchmarkThresholds:
    """Return smoke thresholds for canonical Field3D runtime checks."""

    return BenchmarkThresholds(
        min_values={
            "sample_count": 1,
            "field3d_environment_count": 49,
            "field3d_mod_count": 49,
            "field3d_shd_size_bytes": 1000,
            "field3d_pressure_sample_count": 1,
            "field3d_pressure_finite_count": 1,
            "field3d_pressure_max_abs": 1.0e-12,
        },
        required_true=(
            "field3d_runtime_ok",
            "field3d_shd_exists",
            "field3d_prt_completed_without_errors",
        ),
    )


def canonical_field3d_runtime_benchmark_cases(
    sample_root: str | Path | None = None,
) -> tuple[Field3DRuntimeBenchmarkCase, ...]:
    """Return canonical AT Field3D runtime cases."""

    project_root = Path(sample_root).resolve() if sample_root is not None else Path(__file__).resolve().parents[2]
    return (
        Field3DRuntimeBenchmarkCase(
            name="field3d-3datlantic-lant-90degrees-runtime",
            sample_dir=project_root / "tests" / "3DAtlantic",
        ),
    )


def _resolve_executable(external: ExternalSolver, name: str) -> Path:
    exe_path = external.bin_path / name
    if exe_path.exists():
        return exe_path
    exe_path = external.bin_path / f"{name}.exe"
    if exe_path.exists():
        return exe_path
    raise RuntimeError(f"Legacy {name} executable not found in {external.bin_path}.")


def _atlantic_environment_names() -> tuple[str, ...]:
    depths_a = (36, 38, 40, 42, 44, 46, 48, 50, 52)
    depths_b = (36, 38, 40, 42, 44, 46, 48)
    depths_e = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52)

    names: list[str] = [f"lanta{depth}" for depth in depths_a]
    for depth in depths_b:
        names.extend((f"lantb{depth}", f"lantc{depth}", f"lantd{depth}"))
    names.extend(f"lante{depth:02d}" for depth in depths_e)
    return tuple(names)


def _copy_field3d_inputs(sample_dir: Path, work_dir: Path) -> None:
    for source_path in sample_dir.iterdir():
        if source_path.suffix in {".env", ".flp"}:
            shutil.copy2(source_path, work_dir / source_path.name)


def _normalize_field3d_flp(case_name: str, work_dir: Path) -> None:
    flp_path = work_dir / f"{case_name}.flp"
    write_at_field_flp(read_at_field_flp(flp_path), flp_path)


def _run_external(exe_path: Path, case_name: str, work_dir: Path, *, timeout_s: float) -> None:
    try:
        result = subprocess.run(
            [str(exe_path), case_name],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        raise RuntimeError(
            f"{exe_path.name} timed out for {case_name} after {timeout_s:g} seconds.\n"
            f"stdout:\n{stdout[-2000:]}\n"
            f"stderr:\n{stderr[-2000:]}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"{exe_path.name} failed for {case_name} (rc={result.returncode}).\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )


def _field3d_failure_row(
    case: Field3DRuntimeBenchmarkCase,
    field3d_case: str,
    *,
    error: Exception,
    elapsed_s: float,
    environment_count: int = 0,
    mod_count: int = 0,
) -> BenchmarkRow:
    return {
        "case": f"{case.name}-{field3d_case}",
        "kind": "field3d_runtime",
        "passed": False,
        "failure_count": 1,
        "failures": str(error),
        "sample_count": 0,
        "field3d_runtime_ok": False,
        "field3d_environment_count": environment_count,
        "field3d_mod_count": mod_count,
        "field3d_case": field3d_case,
        "field3d_flp_normalized": field3d_case in set(case.normalized_flp_cases),
        "field3d_shd_exists": False,
        "field3d_shd_size_bytes": 0,
        "field3d_pressure_sample_count": 0,
        "field3d_pressure_finite_count": 0,
        "field3d_pressure_max_abs": None,
        "field3d_prt_completed_without_errors": False,
        "field3d_elapsed_s": float(elapsed_s),
        "field3d_error": str(error),
    }


def _field3d_runtime_case_rows(
    case: Field3DRuntimeBenchmarkCase,
    *,
    at_bin_path: str | Path | None = None,
) -> tuple[BenchmarkRow, ...]:
    external = ExternalSolver(None, bin_path=str(at_bin_path) if at_bin_path else None)
    kraken_exe = _resolve_executable(external, "kraken")
    field3d_exe = _resolve_executable(external, "field3d")
    env_names = _atlantic_environment_names()
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="pyacoustics-field3d-runtime-") as temp_dir:
        work_dir = Path(temp_dir)
        _copy_field3d_inputs(case.sample_dir, work_dir)
        normalized_cases = set(case.normalized_flp_cases)
        try:
            for field3d_case in normalized_cases:
                _normalize_field3d_flp(field3d_case, work_dir)

            for env_name in env_names:
                if not (work_dir / f"{env_name}.env").exists():
                    raise RuntimeError(f"Field3D runtime sample missing {env_name}.env")
                _run_external(kraken_exe, env_name, work_dir, timeout_s=case.command_timeout_s)
        except (RuntimeError, OSError, ValueError) as exc:
            mod_count = sum(1 for env_name in env_names if (work_dir / f"{env_name}.mod").exists())
            return tuple(
                _field3d_failure_row(
                    case,
                    field3d_case,
                    error=exc,
                    elapsed_s=time.perf_counter() - start,
                    environment_count=len(env_names),
                    mod_count=mod_count,
                )
                for field3d_case in case.field3d_cases
            )

        case_rows: list[BenchmarkRow] = []
        for field3d_case in case.field3d_cases:
            shd_path = work_dir / f"{field3d_case}.shd"
            prt_path = work_dir / f"{field3d_case}.prt"
            try:
                _run_external(field3d_exe, field3d_case, work_dir, timeout_s=case.command_timeout_s)
                shade = read_at_shd_full(shd_path)
                pressure_abs = np.abs(np.asarray(shade.pressure, dtype=np.complex128))
                finite_pressure = pressure_abs[np.isfinite(pressure_abs)]
                prt_log = read_at_prt(prt_path) if prt_path.exists() else None
            except (RuntimeError, OSError, ValueError) as exc:
                case_rows.append(
                    _field3d_failure_row(
                        case,
                        field3d_case,
                        error=exc,
                        elapsed_s=time.perf_counter() - start,
                        environment_count=len(env_names),
                        mod_count=sum(1 for env_name in env_names if (work_dir / f"{env_name}.mod").exists()),
                    )
                )
                continue
            case_rows.append(
                {
                    "case": f"{case.name}-{field3d_case}",
                    "kind": "field3d_runtime",
                    "sample_count": int(finite_pressure.size),
                    "field3d_runtime_ok": True,
                    "field3d_environment_count": len(env_names),
                    "field3d_mod_count": sum(1 for env_name in env_names if (work_dir / f"{env_name}.mod").exists()),
                    "field3d_case": field3d_case,
                    "field3d_flp_normalized": field3d_case in normalized_cases,
                    "field3d_shd_exists": shd_path.exists(),
                    "field3d_shd_size_bytes": shd_path.stat().st_size if shd_path.exists() else 0,
                    "field3d_pressure_sample_count": int(pressure_abs.size),
                    "field3d_pressure_finite_count": int(finite_pressure.size),
                    "field3d_pressure_max_abs": float(finite_pressure.max()) if finite_pressure.size else None,
                    "field3d_frequency_hz": shade.frequency_hz,
                    "field3d_receiver_depth_count": int(shade.receiver_depths_m.size),
                    "field3d_receiver_range_count": int(shade.receiver_ranges_m.size),
                    "field3d_receiver_theta_count": int(shade.bearings_deg.size),
                    "field3d_source_depth_count": int(shade.source_depths_m.size),
                    "field3d_source_x_count": int(shade.source_x_m.size),
                    "field3d_source_y_count": int(shade.source_y_m.size),
                    "field3d_prt_program": prt_log.program if prt_log is not None else None,
                    "field3d_prt_completed_without_errors": (
                        prt_log.completed_without_errors if prt_log is not None else False
                    ),
                    "field3d_prt_warning_count": prt_log.warning_count if prt_log is not None else None,
                    "field3d_prt_error_count": prt_log.error_count if prt_log is not None else None,
                    "field3d_prt_fatal_error_count": prt_log.fatal_error_count if prt_log is not None else None,
                    "field3d_elapsed_s": float(time.perf_counter() - start),
                }
            )

    return tuple(case_rows)


def run_field3d_runtime_benchmark_matrix(
    cases: tuple[Field3DRuntimeBenchmarkCase, ...] | list[Field3DRuntimeBenchmarkCase],
    *,
    at_bin_path: str | Path | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> Field3DRuntimeBenchmarkMatrix:
    """Run Field3D runtime checks and collect flattened rows."""

    rows = tuple(
        row
        for case in cases
        for row in _field3d_runtime_case_rows(case, at_bin_path=at_bin_path)
    )
    if thresholds is not None:
        rows = evaluate_benchmark_rows(rows, thresholds)
    return Field3DRuntimeBenchmarkMatrix(rows=rows)


__all__ = [
    "Field3DRuntimeBenchmarkCase",
    "Field3DRuntimeBenchmarkMatrix",
    "canonical_field3d_runtime_benchmark_cases",
    "field3d_runtime_smoke_thresholds",
    "run_field3d_runtime_benchmark_matrix",
]
