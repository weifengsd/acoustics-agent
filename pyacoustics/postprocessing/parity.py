"""Parity metrics for post-processing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class TimeseriesParityMetrics:
    """Summary metrics for synthesized time-series parity comparisons."""

    sample_count: int
    mae: float
    median_abs: float
    p90_abs: float
    max_abs: float
    bias: float
    rmse: float
    relative_l2_error: float


@dataclass(frozen=True)
class MatPayloadParityMetrics:
    """Aggregate parity metrics for Matlab workflow payloads."""

    payload_kind: str
    compared_variables: tuple[str, ...]
    sample_count: int
    mae: float
    median_abs: float
    p90_abs: float
    max_abs: float
    bias: float
    rmse: float
    relative_l2_error: float
    variable_metrics: dict[str, TimeseriesParityMetrics]


@dataclass(frozen=True)
class RtsTimeseriesPayload:
    """Normalized AT/Python received time-series payload."""

    plot_title: str
    tout: np.ndarray
    rts: np.ndarray
    pos: object | None = None


_FLOAT_TOKEN_RE = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?"
)


def compare_timeseries(reference: np.ndarray, candidate: np.ndarray) -> TimeseriesParityMetrics:
    """Compare two real-valued time-series arrays with identical shape."""

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate time-series arrays must have identical shapes")
    mask = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(mask):
        raise ValueError("No finite time-series samples to compare")
    diff = cand[mask] - ref[mask]
    abs_diff = np.abs(diff)
    ref_norm = float(np.linalg.norm(ref[mask]))
    return TimeseriesParityMetrics(
        sample_count=int(abs_diff.size),
        mae=float(np.mean(abs_diff)),
        median_abs=float(np.median(abs_diff)),
        p90_abs=float(np.percentile(abs_diff, 90.0)),
        max_abs=float(np.max(abs_diff)),
        bias=float(np.mean(diff)),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        relative_l2_error=float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0.0 else float(np.linalg.norm(diff)),
    )


def _mat_array(path: str | Path, key: str) -> np.ndarray:
    payload = loadmat(path)
    if key not in payload:
        raise KeyError(f"{Path(path).name} does not contain Matlab variable {key!r}")
    return np.asarray(payload[key], dtype=np.float64)


def _mat_payload(path: str | Path) -> dict:
    return {key: value for key, value in loadmat(path).items() if not key.startswith("__")}


def _plot_title(value) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    item = array.flat[0]
    return str(item.item() if hasattr(item, "item") else item)


def _time_major_rts(rts: np.ndarray, tout: np.ndarray) -> np.ndarray:
    values = np.asarray(rts, dtype=np.float64)
    time_count = int(np.asarray(tout).size)
    if values.ndim < 2 or time_count == 0:
        return values
    if values.shape[0] == time_count:
        return values
    if values.shape[1] == time_count:
        axes = list(range(values.ndim))
        axes[0], axes[1] = axes[1], axes[0]
        return np.transpose(values, axes=axes)
    return values


def _ascii_numeric_values(lines: list[str]) -> np.ndarray:
    values: list[float] = []
    for line in lines:
        clean = line.split("!", 1)[0].split("#", 1)[0].replace(",", " ").replace("/", " ")
        values.extend(float(match.group(0).replace("D", "E").replace("d", "e")) for match in _FLOAT_TOKEN_RE.finditer(clean))
    return np.asarray(values, dtype=np.float64)


def read_rts_timeseries(path: str | Path) -> RtsTimeseriesPayload:
    """Read AT/Python received time-series files with time-major ``RTS``.

    Mirrors the useful part of AT Matlab ``read_ts.m``: ``.mat`` files must
    contain ``PlotTitle``, ``Pos``, ``tout`` and ``RTS``; ASCII files contain a
    title line, receiver-depth count/depths, then rows with time followed by
    receiver-depth samples.
    """

    input_path = Path(path)
    if input_path.suffix == ".mat":
        payload = _mat_payload(input_path)
        _require_variables(payload, input_path, ("PlotTitle", "Pos", "tout", "RTS"))
        tout = np.asarray(payload["tout"], dtype=np.float64).reshape(-1)
        rts = _time_major_rts(np.asarray(payload["RTS"], dtype=np.float64), tout)
        return RtsTimeseriesPayload(
            plot_title=_plot_title(payload["PlotTitle"]),
            pos=payload["Pos"],
            tout=tout,
            rts=rts,
        )

    lines = input_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty time-series file: {input_path}")
    plot_title = lines[0]
    numeric = _ascii_numeric_values(lines[1:])
    if numeric.size < 1:
        raise ValueError(f"Time-series file {input_path.name} does not contain receiver-depth metadata")
    receiver_depth_count = int(round(float(numeric[0])))
    offset = 1 + receiver_depth_count
    if numeric.size <= offset:
        raise ValueError(f"Time-series file {input_path.name} does not contain RTS samples")
    remaining = numeric[offset:]
    row_width = receiver_depth_count + 1
    if remaining.size % row_width != 0:
        raise ValueError(f"Time-series file {input_path.name} sample table has inconsistent width")
    table = remaining.reshape((-1, row_width))
    return RtsTimeseriesPayload(
        plot_title=plot_title,
        pos={"r": {"z": numeric[1:offset]}},
        tout=table[:, 0],
        rts=table[:, 1:],
    )


def _require_variables(payload: dict, path: str | Path, variables: tuple[str, ...]) -> None:
    missing = [variable for variable in variables if variable not in payload]
    if missing:
        missing_list = ", ".join(repr(variable) for variable in missing)
        raise KeyError(f"{Path(path).name} does not contain Matlab variable(s) {missing_list}")


def _numeric_array(payload: dict, path: str | Path, variable: str) -> np.ndarray:
    try:
        return np.asarray(payload[variable], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{Path(path).name} Matlab variable {variable!r} is not numeric") from exc


def _compare_exact_variable(reference: dict, candidate: dict, variable: str) -> None:
    ref = np.asarray(reference[variable])
    cand = np.asarray(candidate[variable])
    if ref.shape != cand.shape or not np.array_equal(ref, cand):
        raise ValueError(f"Matlab variable {variable!r} does not match exactly")


def _compare_numeric_variable(
    reference_payload: dict,
    candidate_payload: dict,
    reference_path: str | Path,
    candidate_path: str | Path,
    variable: str,
) -> TimeseriesParityMetrics | None:
    ref = _numeric_array(reference_payload, reference_path, variable)
    cand = _numeric_array(candidate_payload, candidate_path, variable)
    if ref.shape != cand.shape:
        raise ValueError(f"Matlab variable {variable!r} shape mismatch: {ref.shape} != {cand.shape}")
    if ref.size == 0 and cand.size == 0:
        return None
    return compare_timeseries(ref, cand)


def compare_mat_payload(
    reference_mat_path: str | Path,
    candidate_mat_path: str | Path,
    *,
    numeric_variables: tuple[str, ...],
    exact_variables: tuple[str, ...] = (),
    required_variables: tuple[str, ...] = (),
    payload_kind: str = "mat_payload",
) -> MatPayloadParityMetrics:
    """Compare a Matlab workflow payload using required numeric/exact fields."""

    reference_payload = _mat_payload(reference_mat_path)
    candidate_payload = _mat_payload(candidate_mat_path)
    required = numeric_variables + exact_variables + required_variables
    _require_variables(reference_payload, reference_mat_path, required)
    _require_variables(candidate_payload, candidate_mat_path, required)

    for variable in exact_variables:
        _compare_exact_variable(reference_payload, candidate_payload, variable)

    variable_metrics: dict[str, TimeseriesParityMetrics] = {}
    diffs: list[np.ndarray] = []
    references: list[np.ndarray] = []
    for variable in numeric_variables:
        metrics = _compare_numeric_variable(
            reference_payload,
            candidate_payload,
            reference_mat_path,
            candidate_mat_path,
            variable,
        )
        if metrics is None:
            continue
        variable_metrics[variable] = metrics
        ref = _numeric_array(reference_payload, reference_mat_path, variable)
        cand = _numeric_array(candidate_payload, candidate_mat_path, variable)
        mask = np.isfinite(ref) & np.isfinite(cand)
        diffs.append((cand[mask] - ref[mask]).ravel())
        references.append(ref[mask].ravel())

    if not diffs:
        raise ValueError("No finite numeric Matlab payload samples to compare")
    diff = np.concatenate(diffs)
    ref_values = np.concatenate(references)
    abs_diff = np.abs(diff)
    ref_norm = float(np.linalg.norm(ref_values))
    return MatPayloadParityMetrics(
        payload_kind=payload_kind,
        compared_variables=tuple(variable_metrics),
        sample_count=int(abs_diff.size),
        mae=float(np.mean(abs_diff)),
        median_abs=float(np.median(abs_diff)),
        p90_abs=float(np.percentile(abs_diff, 90.0)),
        max_abs=float(np.max(abs_diff)),
        bias=float(np.mean(diff)),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        relative_l2_error=float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0.0 else float(np.linalg.norm(diff)),
        variable_metrics=variable_metrics,
    )


def compare_mat_timeseries(
    reference_mat_path: str | Path,
    candidate_mat_path: str | Path,
    *,
    key: str = "RTS",
) -> TimeseriesParityMetrics:
    """Compare one numeric timeseries variable from two Matlab ``.mat`` files."""

    return compare_timeseries(_mat_array(reference_mat_path, key), _mat_array(candidate_mat_path, key))


def compare_rts_timeseries_files(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    compare_tout: bool = True,
) -> TimeseriesParityMetrics:
    """Compare AT/Python RTS files after ``read_ts.m`` style normalization.

    Both files are read through :func:`read_rts_timeseries`, so Matlab compact
    ``.rts.mat`` payloads and ASCII time-series files are compared in the same
    time-major ``RTS`` layout.
    """

    reference = read_rts_timeseries(reference_path)
    candidate = read_rts_timeseries(candidate_path)
    if compare_tout:
        compare_timeseries(reference.tout, candidate.tout)
    return compare_timeseries(reference.rts, candidate.rts)


_RTS_NUMERIC_VARIABLES = (
    "RTS",
    "tout",
    "sample_rate",
    "receiver_depths",
    "receiver_ranges",
    "source_index",
    "source_depths",
    "selected_source_depth",
    "source_x_index",
    "source_y_index",
    "source_x",
    "source_y",
    "selected_source_x",
    "selected_source_y",
    "receiver_bearings",
)


_MATCHED_FILTER_NUMERIC_VARIABLES = (
    "MF",
    "lags",
    "lag_seconds",
    "sample_rate",
    "receiver_depths",
    "receiver_ranges",
    "source_index",
    "source_depths",
    "selected_source_depth",
    "source_x_index",
    "source_y_index",
    "source_x",
    "source_y",
    "selected_source_x",
    "selected_source_y",
    "receiver_bearings",
)


def compare_rts_mat_payload(
    reference_mat_path: str | Path,
    candidate_mat_path: str | Path,
    *,
    exact_plot_title: bool = False,
) -> MatPayloadParityMetrics:
    """Compare Matlab ``.rts.mat`` delay-and-sum payloads end to end."""

    return compare_mat_payload(
        reference_mat_path,
        candidate_mat_path,
        numeric_variables=_RTS_NUMERIC_VARIABLES,
        exact_variables=("PlotTitle",) if exact_plot_title else (),
        payload_kind="postprocessing_rts_mat",
    )


def compare_at_rts_mat_payload(
    reference_mat_path: str | Path,
    candidate_mat_path: str | Path,
    *,
    exact_plot_title: bool = False,
) -> MatPayloadParityMetrics:
    """Compare AT Matlab ``delayandsum.m`` style ``.rts.mat`` payloads.

    AT's Matlab writer stores a compact payload with ``PlotTitle``, ``Pos``,
    ``tout`` and ``RTS``. ``Pos`` is required for schema compatibility but is
    intentionally not folded into numeric error metrics.
    """

    return compare_mat_payload(
        reference_mat_path,
        candidate_mat_path,
        numeric_variables=("RTS", "tout"),
        exact_variables=("PlotTitle",) if exact_plot_title else (),
        required_variables=("Pos",),
        payload_kind="postprocessing_at_rts_mat",
    )


def compare_matched_filter_mat_payload(
    reference_mat_path: str | Path,
    candidate_mat_path: str | Path,
    *,
    exact_plot_title: bool = False,
) -> MatPayloadParityMetrics:
    """Compare Matlab matched-filter payloads end to end."""

    return compare_mat_payload(
        reference_mat_path,
        candidate_mat_path,
        numeric_variables=_MATCHED_FILTER_NUMERIC_VARIABLES,
        exact_variables=("PlotTitle",) if exact_plot_title else (),
        payload_kind="postprocessing_matched_filter_mat",
    )


def timeseries_benchmark_row(
    case_name: str,
    metrics: TimeseriesParityMetrics,
    *,
    kind: str = "postprocessing_timeseries",
) -> dict[str, int | float | str]:
    """Flatten post-processing parity metrics into one report/CSV-friendly row."""

    return {
        "case": case_name,
        "kind": kind,
        "sample_count": metrics.sample_count,
        "timeseries_mae": metrics.mae,
        "timeseries_median_abs": metrics.median_abs,
        "timeseries_p90_abs": metrics.p90_abs,
        "timeseries_max_abs": metrics.max_abs,
        "timeseries_bias": metrics.bias,
        "timeseries_rmse": metrics.rmse,
        "timeseries_relative_l2_error": metrics.relative_l2_error,
    }


def mat_payload_benchmark_row(
    case_name: str,
    metrics: MatPayloadParityMetrics,
    *,
    kind: str | None = None,
) -> dict[str, int | float | str]:
    """Flatten Matlab payload parity metrics into one report/CSV-friendly row."""

    return {
        "case": case_name,
        "kind": kind or metrics.payload_kind,
        "sample_count": metrics.sample_count,
        "timeseries_mae": metrics.mae,
        "timeseries_median_abs": metrics.median_abs,
        "timeseries_p90_abs": metrics.p90_abs,
        "timeseries_max_abs": metrics.max_abs,
        "timeseries_bias": metrics.bias,
        "timeseries_rmse": metrics.rmse,
        "timeseries_relative_l2_error": metrics.relative_l2_error,
        "mat_payload_variable_count": len(metrics.compared_variables),
        "mat_payload_variables": ",".join(metrics.compared_variables),
    }


__all__ = [
    "MatPayloadParityMetrics",
    "RtsTimeseriesPayload",
    "TimeseriesParityMetrics",
    "compare_at_rts_mat_payload",
    "compare_mat_payload",
    "compare_matched_filter_mat_payload",
    "compare_mat_timeseries",
    "compare_rts_mat_payload",
    "compare_rts_timeseries_files",
    "compare_timeseries",
    "mat_payload_benchmark_row",
    "read_rts_timeseries",
    "timeseries_benchmark_row",
]
