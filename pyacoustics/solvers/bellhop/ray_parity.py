"""Utilities for comparing native Bellhop ray paths against AT ``.ray`` output."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import tempfile

import numpy as np

from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop.core import PyBellhop
from pyacoustics.solvers.external import ExternalSolver
from pyacoustics.solvers.external_io import ATRayRecord, generate_at_bellhop_env, read_at_ray_records


@dataclass(frozen=True)
class RayPathParityMetrics:
    """Summary metrics for an AT/native ray fan comparison."""

    at_count: int
    native_count: int
    matched_count: int
    unmatched_at_count: int
    unmatched_native_count: int
    match_fraction: float
    depth_mae: float
    depth_p90_abs: float
    depth_max_abs: float
    launch_angle_mae_deg: float
    launch_angle_max_abs_deg: float
    bounce_match_count: int
    bounce_match_fraction: float


def _float_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def ray_benchmark_row(
    case_name: str,
    metrics: RayPathParityMetrics,
    *,
    run_type: str | None = None,
) -> dict[str, int | float | str | None]:
    """Flatten ray path parity metrics into one report/CSV-friendly row."""

    return {
        "case": case_name,
        "run_type": run_type,
        "at_count": metrics.at_count,
        "native_count": metrics.native_count,
        "matched_count": metrics.matched_count,
        "unmatched_at_count": metrics.unmatched_at_count,
        "unmatched_native_count": metrics.unmatched_native_count,
        "match_fraction": _float_or_none(metrics.match_fraction),
        "depth_mae_m": _float_or_none(metrics.depth_mae),
        "depth_p90_abs_m": _float_or_none(metrics.depth_p90_abs),
        "depth_max_abs_m": _float_or_none(metrics.depth_max_abs),
        "launch_angle_mae_deg": _float_or_none(metrics.launch_angle_mae_deg),
        "launch_angle_max_abs_deg": _float_or_none(metrics.launch_angle_max_abs_deg),
        "bounce_match_count": metrics.bounce_match_count,
        "bounce_match_fraction": _float_or_none(metrics.bounce_match_fraction),
    }


def run_at_ray_records(
    config: SimulationConfig,
    *,
    run_type: str = "R",
    at_bin_path: str | Path | None = None,
) -> list[ATRayRecord]:
    """Run AT Bellhop in ray/eigenray mode and parse its ``.ray`` records."""

    external = ExternalSolver(config, bin_path=str(at_bin_path) if at_bin_path else None)
    if not external.is_available("bellhop"):
        raise RuntimeError(f"Legacy bellhop executable not found in {external.bin_path}.")

    exe_path = external.bin_path / "bellhop"
    if not exe_path.exists():
        exe_path = external.bin_path / "bellhop.exe"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        env_path = tmp_path / "ray_parity.env"
        generate_at_bellhop_env(config, env_path, run_type=run_type)
        external.run_command([str(exe_path), "ray_parity"], cwd=str(tmp_path))
        return read_at_ray_records(tmp_path / "ray_parity.ray")


def _beam_records(
    config: SimulationConfig,
    *,
    trace_range_m: float | None = None,
) -> list[ATRayRecord]:
    native_config = config
    if trace_range_m is not None:
        receiver_ranges = list(config.geometry.receivers.ranges)
        if receiver_ranges and float(trace_range_m) > max(receiver_ranges):
            native_config = copy.deepcopy(config)
            native_config.geometry.receivers.ranges = [*receiver_ranges, float(trace_range_m)]

    beam = PyBellhop(native_config).run_beam_tracing()
    records: list[ATRayRecord] = []
    for ray_idx in range(int(beam["num_beams"])):
        npts = int(beam["ray_npts"][ray_idx])
        if npts <= 0:
            continue
        records.append(
            ATRayRecord(
                launch_angle_deg=float(np.degrees(beam["ray_alphas"][ray_idx])),
                ranges_m=np.array(beam["ray_r"][ray_idx, :npts], copy=True),
                depths_m=np.array(beam["ray_z"][ray_idx, :npts], copy=True),
                amplitudes=np.array(beam["ray_amp"][ray_idx, :npts], copy=True),
                num_top_bounces=int(beam["ray_top_bnc"][ray_idx, npts - 1]),
                num_bottom_bounces=int(beam["ray_bot_bnc"][ray_idx, npts - 1]),
            )
        )
    return records


def compute_native_ray_records(
    config: SimulationConfig,
    *,
    trace_range_m: float | None = None,
) -> list[ATRayRecord]:
    """Compute native ray-fan records using the beam tracer metadata path."""

    return _beam_records(config, trace_range_m=trace_range_m)


def _raw_eigenray_hit_segment_indices(
    *,
    receiver_range: float,
    receiver_depth: float,
    ray_r: np.ndarray,
    ray_z: np.ndarray,
    ray_c: np.ndarray,
    ray_tau: np.ndarray,
    ray_q: np.ndarray,
    ray_npts: int,
    frequency: float,
    d_alpha: float,
    c_src: float,
    beam_shape: str,
) -> list[int]:
    beam_window = 4.0
    q0 = c_src / d_alpha
    hit_indices: list[int] = []

    for i in range(max(int(ray_npts) - 1, 0)):
        r1 = float(ray_r[i])
        z1 = float(ray_z[i])
        r2 = float(ray_r[i + 1])
        z2 = float(ray_z[i + 1])
        dr = r2 - r1
        if abs(dr) < 1.0e-10:
            continue

        dz = z2 - z1
        segment_length = float(np.hypot(dr, dz))
        if segment_length < 1.0e-10:
            continue
        rayt_r = dr / segment_length
        rayt_z = dz / segment_length
        if abs(rayt_r) < 1.0e-10:
            continue

        if receiver_range < min(r1, r2) or receiver_range > max(r1, r2):
            continue

        q_start = float(ray_q[i])
        q_end = float(ray_q[i + 1])
        sigma_seg = max(abs(q_start), abs(q_end)) / q0 / abs(rayt_r)
        if beam_shape == "gaussian":
            local_lambda = float(ray_c[i]) / float(frequency)
            sigma_floor = min(0.2 * float(frequency) * float(ray_tau[i + 1]), np.pi * local_lambda)
            radius_max = beam_window * max(sigma_seg, sigma_floor)
        else:
            radius_max = sigma_seg
        if abs(rayt_r) > 0.5:
            z_low = min(z1, z2) - radius_max
            z_high = max(z1, z2) + radius_max
            if receiver_depth < z_low or receiver_depth > z_high:
                continue

        frac = (receiver_range - r1) / dr
        if frac < 0.0 or frac > 1.0:
            continue
        if frac == 1.0 and i < int(ray_npts) - 2:
            continue

        dx_r = receiver_range - r1
        dz_r = receiver_depth - z1
        s = (dx_r * rayt_r + dz_r * rayt_z) / segment_length
        n_dist = abs(dx_r * (-rayt_z) + dz_r * rayt_r)
        q_int = q_start + s * (q_end - q_start)
        sigma = abs(q_int / q0)
        if beam_shape == "gaussian":
            local_lambda = float(ray_c[i]) / float(frequency)
            sigma_floor = min(0.2 * float(frequency) * float(ray_tau[i + 1]), np.pi * local_lambda)
            sigma1 = max(sigma, sigma_floor)
            in_beam = n_dist < beam_window * sigma1 and sigma1 > 0.0
        else:
            sigma1 = sigma
            in_beam = n_dist < sigma1 and sigma1 > 0.0
        if in_beam:
            hit_indices.append(i + 1)

    return hit_indices


def compute_native_eigenray_records(
    config: SimulationConfig,
    *,
    max_arrivals: int = 200,
    beam_shape: str = "hat",
    trace_range_m: float | None = None,
) -> list[ATRayRecord]:
    """Compute native AT-style ``E`` ray records selected from raw beam hits.

    AT Bellhop ``E`` writes a ``.ray`` file containing the launch rays that
    contributed at the receiver influence stage. The write happens before
    ``AddArr`` merges nearby arrivals, so this intentionally preserves raw
    beam hits instead of deduplicating final arrival angles.
    """

    beam_shape_normalized = beam_shape.lower()
    if beam_shape_normalized not in {"hat", "gaussian"}:
        raise ValueError("beam_shape must be either 'hat' or 'gaussian'")

    native_config = config
    if trace_range_m is not None:
        receiver_ranges = list(config.geometry.receivers.ranges)
        if receiver_ranges and float(trace_range_m) > max(receiver_ranges):
            native_config = copy.deepcopy(config)
            native_config.geometry.receivers.ranges = [*receiver_ranges, float(trace_range_m)]

    beam = PyBellhop(native_config).run_beam_tracing()
    ranges = np.asarray(config.geometry.receivers.ranges, dtype=np.float64)
    depths = np.asarray(config.geometry.receivers.depths, dtype=np.float64)
    irregular_receiver_grid = getattr(config.solver, "receiver_grid", "rectilinear") == "irregular"
    if irregular_receiver_grid and len(ranges) != len(depths):
        raise ValueError("Bellhop irregular receiver grid requires matching receiver depth and range counts")

    alphas = np.asarray(beam["ray_alphas"], dtype=np.float64)
    if len(alphas) > 1:
        d_alpha = float(np.abs(np.mean(np.diff(alphas))))
    else:
        d_alpha = 0.01
    if d_alpha <= np.finfo(float).eps:
        d_alpha = 0.01

    records: list[ATRayRecord] = []
    for ray_idx in range(int(beam["num_beams"])):
        npts = int(beam["ray_npts"][ray_idx])
        if npts <= 1:
            continue
        depth_iter = enumerate(depths if not irregular_receiver_grid else [0.0])
        for iz, depth_value in depth_iter:
            for ir, receiver_range in enumerate(ranges):
                receiver_depth = float(depths[ir] if irregular_receiver_grid else depth_value)
                hit_indices = _raw_eigenray_hit_segment_indices(
                    receiver_range=float(receiver_range),
                    receiver_depth=receiver_depth,
                    ray_r=beam["ray_r"][ray_idx],
                    ray_z=beam["ray_z"][ray_idx],
                    ray_c=beam["ray_c"][ray_idx],
                    ray_tau=beam["ray_tau"][ray_idx],
                    ray_q=beam["ray_q"][ray_idx],
                    ray_npts=npts,
                    frequency=float(config.frequency),
                    d_alpha=d_alpha,
                    c_src=float(beam["c_src"]),
                    beam_shape=beam_shape_normalized,
                )
                for hit_idx in hit_indices:
                    stop = min(int(hit_idx) + 1, npts)
                    records.append(
                        ATRayRecord(
                            launch_angle_deg=float(np.degrees(beam["ray_alphas"][ray_idx])),
                            ranges_m=np.array(beam["ray_r"][ray_idx, :stop], copy=True),
                            depths_m=np.array(beam["ray_z"][ray_idx, :stop], copy=True),
                            amplitudes=np.array(beam["ray_amp"][ray_idx, :stop], copy=True),
                            num_top_bounces=int(beam["ray_top_bnc"][ray_idx, stop - 1]),
                            num_bottom_bounces=int(beam["ray_bot_bnc"][ray_idx, stop - 1]),
                        )
                    )
                    if len(records) >= max_arrivals:
                        return records
    return records


def compare_ray_records(
    at_records: list[ATRayRecord],
    native_records: list[ATRayRecord],
    *,
    max_range_m: float | None = None,
    sample_count: int = 101,
) -> RayPathParityMetrics:
    """Compare corresponding rays by interpolating depth onto shared range samples."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    matched_count = min(len(at_records), len(native_records))
    depth_errors: list[float] = []
    launch_angle_errors: list[float] = []
    bounce_match_count = 0

    for at_ray, native_ray in zip(at_records[:matched_count], native_records[:matched_count]):
        launch_angle_errors.append(abs(float(native_ray.launch_angle_deg) - float(at_ray.launch_angle_deg)))
        r_stop = min(float(at_ray.ranges_m[-1]), float(native_ray.ranges_m[-1]))
        if max_range_m is not None:
            r_stop = min(r_stop, float(max_range_m))
        if r_stop <= 0.0:
            continue

        sample_ranges = np.linspace(0.0, r_stop, sample_count)
        at_depths = np.interp(sample_ranges, at_ray.ranges_m, at_ray.depths_m)
        native_depths = np.interp(sample_ranges, native_ray.ranges_m, native_ray.depths_m)
        depth_errors.extend(np.abs(native_depths - at_depths).tolist())

        if (
            at_ray.num_top_bounces == native_ray.num_top_bounces
            and at_ray.num_bottom_bounces == native_ray.num_bottom_bounces
        ):
            bounce_match_count += 1

    if depth_errors:
        errors = np.array(depth_errors, dtype=np.float64)
        depth_mae = float(np.mean(errors))
        depth_p90_abs = float(np.percentile(errors, 90.0))
        depth_max_abs = float(np.max(errors))
    else:
        depth_mae = float("nan")
        depth_p90_abs = float("nan")
        depth_max_abs = float("nan")
    if launch_angle_errors:
        angle_errors = np.array(launch_angle_errors, dtype=np.float64)
        launch_angle_mae = float(np.mean(angle_errors))
        launch_angle_max_abs = float(np.max(angle_errors))
    else:
        launch_angle_mae = float("nan")
        launch_angle_max_abs = float("nan")
    unmatched_at_count = max(len(at_records) - matched_count, 0)
    unmatched_native_count = max(len(native_records) - matched_count, 0)
    match_fraction = matched_count / len(at_records) if at_records else float("nan")
    bounce_match_fraction = bounce_match_count / matched_count if matched_count else float("nan")

    return RayPathParityMetrics(
        at_count=len(at_records),
        native_count=len(native_records),
        matched_count=matched_count,
        unmatched_at_count=unmatched_at_count,
        unmatched_native_count=unmatched_native_count,
        match_fraction=match_fraction,
        depth_mae=depth_mae,
        depth_p90_abs=depth_p90_abs,
        depth_max_abs=depth_max_abs,
        launch_angle_mae_deg=launch_angle_mae,
        launch_angle_max_abs_deg=launch_angle_max_abs,
        bounce_match_count=bounce_match_count,
        bounce_match_fraction=bounce_match_fraction,
    )
