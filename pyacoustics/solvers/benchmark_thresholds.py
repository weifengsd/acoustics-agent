"""Threshold helpers for AT/native benchmark matrix rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


BenchmarkRow = dict[str, int | float | str | bool | None]


@dataclass(frozen=True)
class BenchmarkThresholds:
    """Declarative pass/fail thresholds for flattened benchmark rows."""

    max_values: Mapping[str, float] = field(default_factory=dict)
    min_values: Mapping[str, float] = field(default_factory=dict)
    max_abs_values: Mapping[str, float] = field(default_factory=dict)
    equals: Mapping[str, int | float | str | bool | None] = field(default_factory=dict)
    required_true: tuple[str, ...] = ()
    ignore_missing: bool = False


def _finite_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def evaluate_benchmark_row(row: BenchmarkRow, thresholds: BenchmarkThresholds) -> BenchmarkRow:
    """Return a copy of ``row`` with benchmark pass/fail status fields."""

    failures: list[str] = []
    for key, limit in thresholds.max_values.items():
        value = _finite_float(row.get(key))
        if value is None:
            if not thresholds.ignore_missing:
                failures.append(f"{key} missing")
        elif value > float(limit):
            failures.append(f"{key}={value:g} > {float(limit):g}")

    for key, limit in thresholds.min_values.items():
        value = _finite_float(row.get(key))
        if value is None:
            if not thresholds.ignore_missing:
                failures.append(f"{key} missing")
        elif value < float(limit):
            failures.append(f"{key}={value:g} < {float(limit):g}")

    for key, limit in thresholds.max_abs_values.items():
        value = _finite_float(row.get(key))
        if value is None:
            if not thresholds.ignore_missing:
                failures.append(f"{key} missing")
        elif abs(value) > float(limit):
            failures.append(f"|{key}|={abs(value):g} > {float(limit):g}")

    for key, expected in thresholds.equals.items():
        if key not in row and thresholds.ignore_missing:
            continue
        value = row.get(key)
        if value != expected:
            failures.append(f"{key}={value!r} != {expected!r}")

    for key in thresholds.required_true:
        if row.get(key) is not True:
            failures.append(f"{key} is not true")

    evaluated = dict(row)
    evaluated["passed"] = not failures
    evaluated["failure_count"] = len(failures)
    evaluated["failures"] = "; ".join(failures) if failures else None
    return evaluated


def evaluate_benchmark_rows(
    rows: list[BenchmarkRow] | tuple[BenchmarkRow, ...],
    thresholds: BenchmarkThresholds,
) -> tuple[BenchmarkRow, ...]:
    """Apply benchmark thresholds to a sequence of flattened rows."""

    return tuple(evaluate_benchmark_row(row, thresholds) for row in rows)


__all__ = [
    "BenchmarkRow",
    "BenchmarkThresholds",
    "evaluate_benchmark_row",
    "evaluate_benchmark_rows",
]
