from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop.tracer import MAX_STEPS, trace_beam_ray

DEFAULT_EIGENRAY_MIN_SAMPLES = 101
DEFAULT_EIGENRAY_SAMPLES_PER_DEGREE = 10.0
DEFAULT_EIGENRAY_MAX_SAMPLES = 2001


@dataclass
class BellhopEigenray:
    """A 2D eigenray connecting the source to one receiver."""

    launch_angle: float
    receiver_angle: float
    delay: float
    amplitude: complex
    phase_rad: float
    num_top_bounces: int
    num_bottom_bounces: int
    miss_distance: float
    r_path: np.ndarray
    z_path: np.ndarray

    def to_legacy_dict(self) -> dict:
        return {
            "launch_angle": self.launch_angle,
            "arrival_angle": self.receiver_angle,
            "tau": self.delay,
            "amplitude": self.amplitude,
            "phase_rad": self.phase_rad,
            "num_top_bounces": self.num_top_bounces,
            "num_bottom_bounces": self.num_bottom_bounces,
            "miss_distance": self.miss_distance,
            "r_path": self.r_path,
            "z_path": self.z_path,
        }


def _trace_angle(solver, angle_deg: float, source_depth: float | None = None):
    ssp_type_int, z_arr, c_arr, c_coeffs = solver._prepare_ssp_arrays()
    z_src = solver.config.geometry.source.depths[0] if source_depth is None else float(source_depth)
    max_pts = min(MAX_STEPS, int(solver.r_max / solver.h) + 1000)
    return trace_beam_ray(
        r_0=0.0,
        z_0=z_src,
        alpha_deg=float(angle_deg),
        h=solver.h,
        ati_r=solver.ati_r,
        ati_z=solver.ati_z,
        ati_curvilinear=solver.ati_curvilinear,
        bty_r=solver.bty_r,
        bty_z=solver.bty_z,
        bty_curvilinear=solver.bty_curvilinear,
        bty_geo_cp=solver.bty_geo_cp,
        bty_geo_cs=solver.bty_geo_cs,
        bty_geo_rho=solver.bty_geo_rho,
        bty_geo_alpha=solver.bty_geo_alpha,
        bty_geo_alpha_s=solver.bty_geo_alpha_s,
        ati_geo_cp=solver.ati_geo_cp,
        ati_geo_cs=solver.ati_geo_cs,
        ati_geo_rho=solver.ati_geo_rho,
        ati_geo_alpha=solver.ati_geo_alpha,
        ati_geo_alpha_s=solver.ati_geo_alpha_s,
        surface_refl_angles=solver.surface_refl_angles,
        surface_refl_amp=solver.surface_refl_amp,
        surface_refl_phase=solver.surface_refl_phase,
        bottom_refl_angles=solver.bottom_refl_angles,
        bottom_refl_amp=solver.bottom_refl_amp,
        bottom_refl_phase=solver.bottom_refl_phase,
        surf_rho=solver.surf_rho,
        surf_c=solver.surf_c,
        surf_c_s=solver.surf_c_s,
        surf_alpha=solver.surf_alpha,
        surf_alpha_s=solver.surf_alpha_s,
        bot_rho=solver.bot_rho,
        bot_c=solver.bot_c,
        bot_c_s=solver.bot_c_s,
        bot_alpha=solver.bot_alpha,
        bot_alpha_s=solver.bot_alpha_s,
        frequency=solver.config.frequency,
        beam_displacement=1 if getattr(solver.config.solver, "beam_displacement", "none") == "seongil" else 0,
        r_max=solver.r_max,
        ssp_type=ssp_type_int,
        z_arr=z_arr,
        c_arr=c_arr,
        c_coeffs=c_coeffs,
        ssp_r_arr=solver.ssp_r_arr,
        max_steps=max_pts - 1,
    )


def _interpolate_receiver_crossing(
    trace_result,
    receiver_range: float,
    receiver_depth: float,
    launch_angle_deg: float,
):
    (
        r_path,
        z_path,
        c_path,
        tau_path,
        p_path,
        q_path,
        amp_path,
        phase_path,
        top_bnc_path,
        bot_bnc_path,
        n_pts,
        _,
    ) = trace_result

    if n_pts < 2:
        return None

    phase = 0.0
    q_old = float(q_path[0])

    for i in range(int(n_pts) - 1):
        r1 = float(r_path[i])
        r2 = float(r_path[i + 1])
        q_start = float(q_path[i])
        if (q_start <= 0.0 and q_old > 0.0) or (q_start >= 0.0 and q_old < 0.0):
            phase += np.pi * 0.5
        q_old = q_start

        if r1 == r2:
            continue
        if receiver_range < min(r1, r2) or receiver_range > max(r1, r2):
            continue

        frac = (receiver_range - r1) / (r2 - r1)
        if frac < 0.0 or frac > 1.0:
            continue

        z = float(z_path[i] + frac * (z_path[i + 1] - z_path[i]))
        tau = float(tau_path[i] + frac * (tau_path[i + 1] - tau_path[i]))
        amp_reflection = abs(amp_path[i + 1])
        c_int = float(c_path[i] + frac * (c_path[i + 1] - c_path[i]))
        q_int = float(q_path[i] + frac * (q_path[i + 1] - q_path[i]))
        phase_int = phase
        if (q_int <= 0.0 and q_old > 0.0) or (q_int >= 0.0 and q_old < 0.0):
            phase_int += np.pi * 0.5
        if abs(q_int) > np.finfo(float).eps and c_int > 0.0:
            ratio1 = np.sqrt(abs(np.cos(np.radians(launch_angle_deg))))
            raw_phase = float(phase_path[i] + phase_int)
            amp = ratio1 * np.sqrt(c_int / abs(q_int)) * amp_reflection * np.exp(1j * raw_phase)
        else:
            amp = 0.0 + 0.0j
            raw_phase = 0.0
        dz = float(z_path[i + 1] - z_path[i])
        dr = float(r_path[i + 1] - r_path[i])
        receiver_angle = float(np.degrees(np.arctan2(dz, dr)))
        top_bnc = int(top_bnc_path[i + 1])
        bot_bnc = int(bot_bnc_path[i + 1])
        return {
            "z": z,
            "miss": z - receiver_depth,
            "tau": tau,
            "amp": amp,
            "phase_rad": raw_phase,
            "receiver_angle": receiver_angle,
            "top_bnc": top_bnc,
            "bot_bnc": bot_bnc,
            "r_path": r_path[: int(n_pts)].copy(),
            "z_path": z_path[: int(n_pts)].copy(),
        }

    return None


def _candidate_at_angle(
    solver,
    angle_deg: float,
    receiver_range: float,
    receiver_depth: float,
    source_depth: float | None = None,
):
    trace_result = _trace_angle(solver, angle_deg, source_depth=source_depth)
    crossing = _interpolate_receiver_crossing(trace_result, receiver_range, receiver_depth, angle_deg)
    if crossing is None:
        return None
    return angle_deg, crossing


def _make_eigenray(angle_deg: float, crossing) -> BellhopEigenray:
    return BellhopEigenray(
        launch_angle=float(angle_deg),
        receiver_angle=float(crossing["receiver_angle"]),
        delay=float(crossing["tau"]),
        amplitude=complex(crossing["amp"]),
        phase_rad=float(crossing["phase_rad"]),
        num_top_bounces=int(crossing["top_bnc"]),
        num_bottom_bounces=int(crossing["bot_bnc"]),
        miss_distance=float(abs(crossing["miss"])),
        r_path=crossing["r_path"],
        z_path=crossing["z_path"],
    )


def _eigenray_group_key(angle_deg: float, crossing) -> tuple[int, int, int, int]:
    launch_sign = 0
    if angle_deg > 1.0e-8:
        launch_sign = 1
    elif angle_deg < -1.0e-8:
        launch_sign = -1

    receiver_angle = float(crossing["receiver_angle"])
    receiver_sign = 0
    if receiver_angle > 1.0e-8:
        receiver_sign = 1
    elif receiver_angle < -1.0e-8:
        receiver_sign = -1

    return (
        int(crossing["top_bnc"]),
        int(crossing["bot_bnc"]),
        launch_sign,
        receiver_sign,
    )


def _default_angle_samples(config: SimulationConfig) -> int:
    alpha_start, alpha_end = config.solver.angles
    span_deg = abs(float(alpha_end) - float(alpha_start))
    span_samples = int(np.ceil(span_deg * DEFAULT_EIGENRAY_SAMPLES_PER_DEGREE)) + 1
    requested = max(
        int(getattr(config.solver, "num_beams", 0) or 0),
        DEFAULT_EIGENRAY_MIN_SAMPLES,
        span_samples,
    )
    return min(requested, DEFAULT_EIGENRAY_MAX_SAMPLES)


def compute_eigenrays(
    config: SimulationConfig,
    range_m: float,
    depth_m: float,
    *,
    source_depth: float | None = None,
    angle_samples: int | None = None,
    tolerance_m: float = 0.25,
    max_iterations: int = 40,
) -> list[BellhopEigenray]:
    """Find 2D geometric eigenrays by launch-angle bracketing.

    This implements the first native ``E``-style search layer: rays are traced
    over a launch-angle fan, adjacent rays whose receiver-depth miss changes
    sign bracket an eigenray, and bisection refines the connecting angle.
    """
    if range_m < 0.0:
        raise ValueError("range_m must be non-negative")
    if tolerance_m <= 0.0:
        raise ValueError("tolerance_m must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    from pyacoustics.solvers.bellhop.core import PyBellhop

    solver = PyBellhop(config)
    alpha_start, alpha_end = config.solver.angles
    if angle_samples is None:
        angle_samples = _default_angle_samples(config)
    if angle_samples < 2:
        raise ValueError("angle_samples must be at least 2")

    angles = np.linspace(float(alpha_start), float(alpha_end), int(angle_samples))
    candidates = [
        _candidate_at_angle(solver, float(angle), range_m, depth_m, source_depth=source_depth)
        for angle in angles
    ]

    eigenray_candidates: list[tuple[tuple[int, int, int, int], float, dict]] = []
    angle_spacing = abs(float(alpha_end) - float(alpha_start)) / float(int(angle_samples) - 1)
    duplicate_angle_tol_deg = max(1.0e-3, min(5.0e-2, 0.25 * angle_spacing))
    duplicate_delay_tol_s = 1.0e-5

    def add_if_new(angle, crossing):
        if abs(crossing["miss"]) > tolerance_m:
            return
        key = _eigenray_group_key(float(angle), crossing)
        for idx, (existing_key, existing_angle, existing_crossing) in enumerate(eigenray_candidates):
            if existing_key != key:
                continue
            same_angle = abs(float(angle) - existing_angle) <= duplicate_angle_tol_deg
            same_delay = abs(float(crossing["tau"]) - float(existing_crossing["tau"])) <= duplicate_delay_tol_s
            if not (same_angle or same_delay):
                continue
            if abs(existing_crossing["miss"]) <= abs(crossing["miss"]):
                return
            eigenray_candidates[idx] = (key, float(angle), crossing)
            return
        eigenray_candidates.append((key, float(angle), crossing))

    for candidate in candidates:
        if candidate is None:
            continue
        angle, crossing = candidate
        add_if_new(angle, crossing)

    for left, right in zip(candidates[:-1], candidates[1:]):
        if left is None or right is None:
            continue
        left_angle, left_crossing = left
        right_angle, right_crossing = right
        left_miss = float(left_crossing["miss"])
        right_miss = float(right_crossing["miss"])
        if left_miss == 0.0 or right_miss == 0.0:
            continue
        if left_miss * right_miss > 0.0:
            continue

        lo_angle = float(left_angle)
        hi_angle = float(right_angle)
        lo_crossing = left_crossing
        hi_crossing = right_crossing
        best_angle = lo_angle
        best_crossing = lo_crossing

        for _ in range(max_iterations):
            mid_angle = 0.5 * (lo_angle + hi_angle)
            mid = _candidate_at_angle(solver, mid_angle, range_m, depth_m, source_depth=source_depth)
            if mid is None:
                break
            _, mid_crossing = mid
            if abs(mid_crossing["miss"]) < abs(best_crossing["miss"]):
                best_angle = mid_angle
                best_crossing = mid_crossing
            if abs(mid_crossing["miss"]) <= tolerance_m:
                break
            if float(lo_crossing["miss"]) * float(mid_crossing["miss"]) <= 0.0:
                hi_angle = mid_angle
                hi_crossing = mid_crossing
            else:
                lo_angle = mid_angle
                lo_crossing = mid_crossing

        add_if_new(best_angle, best_crossing)

    for left, middle, right in zip(candidates[:-2], candidates[1:-1], candidates[2:]):
        if left is None or middle is None or right is None:
            continue
        left_angle, left_crossing = left
        mid_angle, mid_crossing = middle
        right_angle, right_crossing = right
        left_abs = abs(float(left_crossing["miss"]))
        mid_abs = abs(float(mid_crossing["miss"]))
        right_abs = abs(float(right_crossing["miss"]))
        if mid_abs > left_abs or mid_abs > right_abs:
            continue

        lo_angle = float(left_angle)
        hi_angle = float(right_angle)
        best_angle = float(mid_angle)
        best_crossing = mid_crossing

        for _ in range(max_iterations):
            span = hi_angle - lo_angle
            if abs(span) < 1e-8:
                break
            m1 = lo_angle + span / 3.0
            m2 = hi_angle - span / 3.0
            cand1 = _candidate_at_angle(solver, m1, range_m, depth_m, source_depth=source_depth)
            cand2 = _candidate_at_angle(solver, m2, range_m, depth_m, source_depth=source_depth)
            if cand1 is None or cand2 is None:
                break
            _, crossing1 = cand1
            _, crossing2 = cand2
            miss1 = abs(float(crossing1["miss"]))
            miss2 = abs(float(crossing2["miss"]))
            if miss1 < abs(float(best_crossing["miss"])):
                best_angle = m1
                best_crossing = crossing1
            if miss2 < abs(float(best_crossing["miss"])):
                best_angle = m2
                best_crossing = crossing2
            if miss1 < miss2:
                hi_angle = m2
            else:
                lo_angle = m1
            if abs(float(best_crossing["miss"])) <= tolerance_m:
                break

        add_if_new(best_angle, best_crossing)

    eigenrays = [
        _make_eigenray(angle, crossing)
        for _, angle, crossing in eigenray_candidates
    ]
    eigenrays.sort(key=lambda ray: (ray.delay, ray.num_top_bounces, ray.num_bottom_bounces, ray.launch_angle))
    return eigenrays


def compute_eigenrays_legacy(config: SimulationConfig, range_m: float, depth_m: float, **kwargs):
    return [ray.to_legacy_dict() for ray in compute_eigenrays(config, range_m, depth_m, **kwargs)]


def write_eigenrays_ray_ascii(
    path: str | Path,
    config: SimulationConfig,
    eigenrays: list[BellhopEigenray],
) -> None:
    """Write native 2D eigenrays as an AT-compatible ASCII ``.ray`` file.

    AT Bellhop writes both ``R`` ray traces and ``E`` eigenray traces through
    ``WriteRay2D``. The header below follows the 2D layout consumed by AT's
    ``plotray.m`` and by :func:`pyacoustics.solvers.external_io.read_at_ray_records`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    title = config.project.replace("'", "")
    z_top = config.environment.surface.depth
    if z_top is None:
        z_top = 0.0
    z_bottom = config.environment.bottom.depth
    if z_bottom is None:
        z_bottom = max(point.depth for point in config.environment.ssp.data)

    with path.open("w") as f:
        f.write(f"'{title}'\n")
        f.write(f"{config.frequency:.9g}\n")
        f.write("1 1 1\n")
        f.write(f"{len(eigenrays)} 1\n")
        f.write(f"{float(z_top):.9g}\n")
        f.write(f"{float(z_bottom):.9g}\n")
        f.write("'rz'\n")
        for ray in eigenrays:
            npts = len(ray.r_path)
            f.write(f"{ray.launch_angle:.9g}\n")
            f.write(
                f"{npts:d} {ray.num_top_bounces:d} {ray.num_bottom_bounces:d}\n"
            )
            for r, z in zip(ray.r_path, ray.z_path):
                f.write(f"{float(r):.9g} {float(z):.9g}\n")


__all__ = [
    "BellhopEigenray",
    "compute_eigenrays",
    "compute_eigenrays_legacy",
    "write_eigenrays_ray_ascii",
]
