from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from pyacoustics.postprocessing.delay_and_sum import ArrivalData, ArrivalRecord, write_arrivals_ascii
from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop3d.core import PyBellhop3D


@dataclass
class Bellhop3DArrival:
    amplitude: complex
    delay: complex
    src_decl_angle: float
    src_azim_angle: float
    rcvr_decl_angle: float
    rcvr_azim_angle: float
    num_top_bounces: int
    num_bottom_bounces: int
    distance_to_receiver: float
    phase_rad: float | None = None


def _empty_record() -> ArrivalRecord:
    empty_float = np.zeros(0, dtype=np.float64)
    empty_complex = np.zeros(0, dtype=np.complex128)
    return ArrivalRecord(
        amplitudes=empty_complex,
        delays=empty_complex,
        src_decl_angles=empty_float,
        src_azim_angles=empty_float,
        rcvr_decl_angles=empty_float,
        rcvr_azim_angles=empty_float,
        phase_angles=empty_float,
        num_top_bounces=np.zeros(0, dtype=np.int16),
        num_bottom_bounces=np.zeros(0, dtype=np.int16),
    )


def _arrival_record(arrivals: list[Bellhop3DArrival]) -> ArrivalRecord:
    if not arrivals:
        return _empty_record()
    return ArrivalRecord(
        amplitudes=np.asarray([arrival.amplitude for arrival in arrivals], dtype=np.complex128),
        delays=np.asarray([arrival.delay for arrival in arrivals], dtype=np.complex128),
        src_decl_angles=np.asarray([arrival.src_decl_angle for arrival in arrivals], dtype=np.float64),
        src_azim_angles=np.asarray([arrival.src_azim_angle for arrival in arrivals], dtype=np.float64),
        rcvr_decl_angles=np.asarray([arrival.rcvr_decl_angle for arrival in arrivals], dtype=np.float64),
        rcvr_azim_angles=np.asarray([arrival.rcvr_azim_angle for arrival in arrivals], dtype=np.float64),
        phase_angles=np.asarray(
            [
                np.angle(arrival.amplitude) if arrival.phase_rad is None else arrival.phase_rad
                for arrival in arrivals
            ],
            dtype=np.float64,
        ),
        num_top_bounces=np.asarray([arrival.num_top_bounces for arrival in arrivals], dtype=np.int16),
        num_bottom_bounces=np.asarray([arrival.num_bottom_bounces for arrival in arrivals], dtype=np.int16),
    )


def _interp_angle_rad(a0: float, a1: float, frac: float) -> float:
    delta = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
    return float(a0 + frac * delta)


def _nearest_segment_arrival(
    *,
    receiver_x: float,
    receiver_y: float,
    receiver_z: float,
    x_path: np.ndarray,
    y_path: np.ndarray,
    z_path: np.ndarray,
    q11_path: np.ndarray,
    q12_path: np.ndarray,
    q21_path: np.ndarray,
    q22_path: np.ndarray,
    amp_path: np.ndarray,
    tau_path: np.ndarray,
    kmah_path: np.ndarray,
    c_path: np.ndarray,
    phi_path: np.ndarray,
    launch_elev_deg: float,
    launch_azim_deg: float,
    num_top_bounces: int,
    num_bottom_bounces: int,
    tolerance_m: float,
    c_src: float,
    d_elev: float,
    d_azim: float,
    beam_window: float = 4.0,
    phase_path: np.ndarray | None = None,
    top_bounce_path: np.ndarray | None = None,
    bottom_bounce_path: np.ndarray | None = None,
) -> Bellhop3DArrival | None:
    best: Bellhop3DArrival | None = None

    receiver = np.array([receiver_x, receiver_y, receiver_z], dtype=np.float64)
    for idx in range(len(x_path) - 1):
        p0 = np.array([x_path[idx], y_path[idx], z_path[idx]], dtype=np.float64)
        p1 = np.array([x_path[idx + 1], y_path[idx + 1], z_path[idx + 1]], dtype=np.float64)
        delta = p1 - p0
        seg_len2 = float(np.dot(delta, delta))
        if seg_len2 <= np.finfo(float).eps:
            continue

        frac = float(np.dot(receiver - p0, delta) / seg_len2)
        if frac < 0.0 or frac >= 1.0:
            continue

        closest = p0 + frac * delta
        miss = float(np.linalg.norm(receiver - closest))

        tangent = delta / np.sqrt(seg_len2)
        horizontal = float(np.hypot(tangent[0], tangent[1]))
        phi = _interp_angle_rad(float(phi_path[idx]), float(phi_path[idx + 1]), frac)
        if horizontal > 1.0e-10:
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            e1 = np.array(
                [
                    (tangent[0] * tangent[2] * cos_phi + tangent[1] * sin_phi) / horizontal,
                    (tangent[1] * tangent[2] * cos_phi - tangent[0] * sin_phi) / horizontal,
                    -horizontal * cos_phi,
                ]
            )
            e2 = np.array(
                [
                    (tangent[0] * tangent[2] * sin_phi - tangent[1] * cos_phi) / horizontal,
                    (tangent[1] * tangent[2] * sin_phi + tangent[0] * cos_phi) / horizontal,
                    -horizontal * sin_phi,
                ]
            )
        else:
            e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.array([0.0, 1.0, 0.0])

        q11 = q11_path[idx] + frac * (q11_path[idx + 1] - q11_path[idx])
        q12 = q12_path[idx] + frac * (q12_path[idx + 1] - q12_path[idx])
        q21 = q21_path[idx] + frac * (q21_path[idx + 1] - q21_path[idx])
        q22 = q22_path[idx] + frac * (q22_path[idx + 1] - q22_path[idx])
        scale_tilde = d_elev
        scale_hat = d_azim * abs(np.cos(np.radians(launch_elev_deg)))
        q11_scaled = q11 * scale_tilde
        q21_scaled = q21 * scale_tilde
        q12_scaled = q12 * scale_hat
        q22_scaled = q22 * scale_hat
        det_q_scaled = q11_scaled * q22_scaled - q12_scaled * q21_scaled
        det_q = q11 * q22 - q12 * q21
        if abs(det_q_scaled) < 1.0e-20 or abs(det_q) < 1.0e-20:
            continue

        offset = receiver - p0
        n_dist = abs(float(np.dot(offset, e1)))
        m_dist = abs(float(np.dot(offset, e2)))
        a_ell = abs((-q12_scaled * m_dist + q22_scaled * n_dist) / det_q_scaled)
        b_ell = abs((q11_scaled * m_dist - q21_scaled * n_dist) / det_q_scaled)
        if miss > tolerance_m and a_ell + b_ell >= beam_window:
            continue

        c_t = c_path[idx] + frac * (c_path[idx + 1] - c_path[idx])
        amp_endpoint = amp_path[idx + 1]
        decay = np.exp(-0.5 * (a_ell**2 + b_ell**2))
        base_amplitude = (c_t / c_src) * abs(amp_endpoint) * decay / (2.0 * np.pi * np.sqrt(abs(det_q)))
        tau = tau_path[idx] + frac * (tau_path[idx + 1] - tau_path[idx])

        caustic_phase = 0.0
        det_old = q11_path[0] * q22_path[0] - q12_path[0] * q21_path[0]
        for step in range(1, idx + 1):
            det_step = q11_path[step] * q22_path[step] - q12_path[step] * q21_path[step]
            if (det_step.real <= 0.0 and det_old.real > 0.0) or (
                det_step.real >= 0.0 and det_old.real < 0.0
            ):
                caustic_phase += np.pi * 0.5
            det_old = det_step
        kmah_idx = idx if frac < 0.5 else idx + 1
        caustic_phase += float(kmah_path[kmah_idx]) * np.pi * 0.5
        if (det_q.real <= 0.0 and det_old.real > 0.0) or (
            det_q.real >= 0.0 and det_old.real < 0.0
        ):
            caustic_phase += np.pi * 0.5
        path_phase = float(np.angle(amp_path[idx]) if phase_path is None else phase_path[idx])
        phase_rad = path_phase + caustic_phase
        amplitude = complex(abs(base_amplitude) * np.exp(1j * phase_rad))

        rcvr_decl = float(np.degrees(np.arctan2(delta[2], np.hypot(delta[0], delta[1]))))
        rcvr_azim = float(np.degrees(np.arctan2(delta[1], delta[0])))
        top_bounces = int(num_top_bounces if top_bounce_path is None else top_bounce_path[idx + 1])
        bottom_bounces = int(num_bottom_bounces if bottom_bounce_path is None else bottom_bounce_path[idx + 1])
        arrival = Bellhop3DArrival(
            amplitude=amplitude,
            delay=complex(tau, 0.0),
            src_decl_angle=float(launch_elev_deg),
            src_azim_angle=float(launch_azim_deg),
            rcvr_decl_angle=rcvr_decl,
            rcvr_azim_angle=rcvr_azim,
            num_top_bounces=top_bounces,
            num_bottom_bounces=bottom_bounces,
            distance_to_receiver=miss,
            phase_rad=phase_rad,
        )
        if best is None or arrival.distance_to_receiver < best.distance_to_receiver:
            best = arrival

    return best


def _append_arrival_at_style(
    arrivals: list[Bellhop3DArrival],
    arrival: Bellhop3DArrival,
    *,
    omega: float,
    max_arrivals: int,
    phase_tolerance: float = 0.05,
    relative_floor: float = 1.0e-10,
) -> None:
    """Apply AT ``AddArr3D`` grouping and strongest-arrival retention."""

    if arrivals:
        previous = arrivals[-1]
        delay_close = omega * abs(arrival.delay - previous.delay) < phase_tolerance
        previous_phase = np.angle(previous.amplitude) if previous.phase_rad is None else previous.phase_rad
        arrival_phase = np.angle(arrival.amplitude) if arrival.phase_rad is None else arrival.phase_rad
        phase_delta = arrival_phase - previous_phase
        if delay_close and abs(phase_delta) < phase_tolerance:
            amp_previous = abs(previous.amplitude)
            amp_new = abs(arrival.amplitude)
            amp_total = amp_previous + amp_new
            if amp_total > np.finfo(float).eps:
                weight_previous = amp_previous / amp_total
                weight_new = amp_new / amp_total
                previous.delay = weight_previous * previous.delay + weight_new * arrival.delay
                previous.src_decl_angle = (
                    weight_previous * previous.src_decl_angle + weight_new * arrival.src_decl_angle
                )
                previous.src_azim_angle = (
                    weight_previous * previous.src_azim_angle + weight_new * arrival.src_azim_angle
                )
                previous.rcvr_decl_angle = (
                    weight_previous * previous.rcvr_decl_angle + weight_new * arrival.rcvr_decl_angle
                )
                previous.rcvr_azim_angle = (
                    weight_previous * previous.rcvr_azim_angle + weight_new * arrival.rcvr_azim_angle
                )
                previous.distance_to_receiver = min(
                    previous.distance_to_receiver,
                    arrival.distance_to_receiver,
                )
                previous.phase_rad = previous_phase
                previous.amplitude = amp_total * np.exp(1j * previous_phase)
            return

        strongest = max(abs(existing.amplitude) for existing in arrivals)
        if strongest > np.finfo(float).eps and abs(arrival.amplitude) < strongest * relative_floor:
            return

    if len(arrivals) < max_arrivals:
        arrivals.append(arrival)
        return

    weakest_idx = int(np.argmin([abs(existing.amplitude) for existing in arrivals]))
    if abs(arrival.amplitude) > abs(arrivals[weakest_idx].amplitude):
        arrivals[weakest_idx] = arrival


def _prune_negligible_arrivals(
    arrivals: list[Bellhop3DArrival],
    *,
    relative_floor: float = 1.0e-10,
) -> list[Bellhop3DArrival]:
    if not arrivals:
        return arrivals
    max_amplitude = max(abs(arrival.amplitude) for arrival in arrivals)
    if max_amplitude <= np.finfo(float).eps:
        return []
    amplitude_floor = max_amplitude * relative_floor
    return [arrival for arrival in arrivals if abs(arrival.amplitude) >= amplitude_floor]


def compute_arrival_grid_3d(
    config: SimulationConfig,
    *,
    receiver_ranges: np.ndarray | None = None,
    receiver_depths: np.ndarray | None = None,
    receiver_bearings: np.ndarray | None = None,
    source_depths: np.ndarray | None = None,
    source_x: np.ndarray | None = None,
    source_y: np.ndarray | None = None,
    max_arrivals: int = 200,
    tolerance_m: float | None = None,
) -> ArrivalData:
    """Compute a native Bellhop3D arrivals grid from traced 3D ray paths.

    This is a geometric extraction layer over the existing 3D ray fan: each
    receiver is matched to nearby ray segments and the arrival fields are
    linearly interpolated along the best segment for each ray. It establishes a
    native 3D ``ArrivalData`` path and AT-compatible ``.arr`` output support;
    full AT-style 3D eigenray search/grouping remains a separate refinement.
    """

    if config.solver.type != "bellhop3d":
        raise ValueError("compute_arrival_grid_3d requires a Bellhop3D solver config")
    if max_arrivals <= 0:
        raise ValueError("max_arrivals must be positive")

    ranges = (
        np.asarray(receiver_ranges, dtype=np.float64)
        if receiver_ranges is not None
        else np.asarray(config.geometry.receivers.ranges, dtype=np.float64)
    )
    depths = (
        np.asarray(receiver_depths, dtype=np.float64)
        if receiver_depths is not None
        else np.asarray(config.geometry.receivers.depths, dtype=np.float64)
    )
    bearings = (
        np.asarray(receiver_bearings, dtype=np.float64)
        if receiver_bearings is not None
        else np.array([0.0], dtype=np.float64)
    )
    sources = (
        np.asarray(source_depths, dtype=np.float64)
        if source_depths is not None
        else np.asarray(config.geometry.source.depths, dtype=np.float64)
    )
    source_x_values = (
        np.asarray(source_x, dtype=np.float64)
        if source_x is not None
        else np.asarray(config.geometry.source.x or [0.0], dtype=np.float64)
    )
    source_y_values = (
        np.asarray(source_y, dtype=np.float64)
        if source_y is not None
        else np.asarray(config.geometry.source.y or [0.0], dtype=np.float64)
    )
    if tolerance_m is None:
        step = float(config.solver.step_size or 10.0)
        tolerance_m = max(1.0, 0.5 * step)
    if tolerance_m <= 0.0:
        raise ValueError("tolerance_m must be positive")
    if source_x_values.size == 0 or source_y_values.size == 0:
        raise ValueError("source_x and source_y grids must not be empty")

    num_elev = max(config.solver.num_elevation_beams, 1)
    num_azim = max(config.solver.num_azimuth_beams, 1)
    azimuths = np.linspace(config.solver.azimuth_angles[0], config.solver.azimuth_angles[1], num_azim)
    elevations = np.linspace(config.solver.elevation_angles[0], config.solver.elevation_angles[1], num_elev)
    d_azim = np.radians(abs(config.solver.azimuth_angles[1] - config.solver.azimuth_angles[0])) / max(
        num_azim - 1, 1
    )
    d_elev = np.radians(abs(config.solver.elevation_angles[1] - config.solver.elevation_angles[0])) / max(
        num_elev - 1, 1
    )
    if d_azim == 0.0:
        d_azim = 0.001
    if d_elev == 0.0:
        d_elev = 0.001
    rays_by_source = {}
    c_src_by_source = {}
    for source_depth in sources:
        for source_x_value in source_x_values:
            for source_y_value in source_y_values:
                source_config = deepcopy(config)
                source_config.geometry.source.depths = [float(source_depth)]
                source_config.geometry.source.x = [float(source_x_value)]
                source_config.geometry.source.y = [float(source_y_value)]
                source_solver = PyBellhop3D(source_config)
                source_rays = source_solver.run_ray_tracing(include_phi=True)
                key = (float(source_depth), float(source_x_value), float(source_y_value))
                rays_by_source[key] = source_rays
                c_src_by_source[key] = float(source_rays[0][11][0])

    has_source_xy_blocks = source_x_values.size * source_y_values.size > 1
    if has_source_xy_blocks:
        arrivals_grid = np.empty(
            (len(ranges), len(depths), len(bearings), len(sources), len(source_x_values), len(source_y_values)),
            dtype=object,
        )
    else:
        arrivals_grid = np.empty((len(ranges), len(depths), len(bearings), len(sources)), dtype=object)
    for isd in range(len(sources)):
        for isx, source_x_value in enumerate(source_x_values):
            for isy, source_y_value in enumerate(source_y_values):
                key = (float(sources[isd]), float(source_x_value), float(source_y_value))
                source_rays = rays_by_source[key]
                for itheta, bearing_deg in enumerate(bearings):
                    theta = np.deg2rad(bearing_deg)
                    cos_theta = float(np.cos(theta))
                    sin_theta = float(np.sin(theta))
                    for iz, receiver_depth in enumerate(depths):
                        for ir, receiver_range in enumerate(ranges):
                            receiver_x = float(source_x_value + receiver_range * cos_theta)
                            receiver_y = float(source_y_value + receiver_range * sin_theta)
                            raw: list[Bellhop3DArrival] = []
                            for ray_idx, ray in enumerate(source_rays):
                                azimuth_idx = (ray_idx // num_elev) % num_azim
                                elevation_idx = ray_idx % num_elev
                                has_phase_path = len(ray) >= 18
                                bounce_offset = 13 if has_phase_path else 12
                                has_bounce_paths = len(ray) >= bounce_offset + 2
                                arrival = _nearest_segment_arrival(
                                    receiver_x=receiver_x,
                                    receiver_y=receiver_y,
                                    receiver_z=float(receiver_depth),
                                    x_path=np.asarray(ray[0], dtype=np.float64),
                                    y_path=np.asarray(ray[1], dtype=np.float64),
                                    z_path=np.asarray(ray[2], dtype=np.float64),
                                    q11_path=np.asarray(ray[3], dtype=np.complex128),
                                    q12_path=np.asarray(ray[4], dtype=np.complex128),
                                    q21_path=np.asarray(ray[5], dtype=np.complex128),
                                    q22_path=np.asarray(ray[6], dtype=np.complex128),
                                    amp_path=np.asarray(ray[7], dtype=np.complex128),
                                    phase_path=np.asarray(ray[8], dtype=np.float64) if has_phase_path else None,
                                    tau_path=np.asarray(ray[9 if has_phase_path else 8], dtype=np.float64),
                                    kmah_path=np.asarray(ray[10 if has_phase_path else 9], dtype=np.float64),
                                    c_path=np.asarray(ray[11 if has_phase_path else 10], dtype=np.float64),
                                    phi_path=np.asarray(ray[12 if has_phase_path else 11], dtype=np.float64),
                                    launch_elev_deg=float(elevations[elevation_idx]),
                                    launch_azim_deg=float(azimuths[azimuth_idx]),
                                    num_top_bounces=int(ray[-3]),
                                    num_bottom_bounces=int(ray[-2]),
                                    top_bounce_path=np.asarray(ray[bounce_offset], dtype=np.int64) if has_bounce_paths else None,
                                    bottom_bounce_path=np.asarray(ray[bounce_offset + 1], dtype=np.int64) if has_bounce_paths else None,
                                    tolerance_m=float(tolerance_m),
                                    c_src=float(c_src_by_source[key]),
                                    d_elev=float(d_elev),
                                    d_azim=float(d_azim),
                                )
                                if arrival is not None:
                                    _append_arrival_at_style(
                                        raw,
                                        arrival,
                                        omega=2.0 * np.pi * float(config.frequency),
                                        max_arrivals=max_arrivals,
                                    )
                            record = _arrival_record(_prune_negligible_arrivals(raw))
                            if has_source_xy_blocks:
                                arrivals_grid[ir, iz, itheta, isd, isx, isy] = record
                            else:
                                arrivals_grid[ir, iz, itheta, isd] = record

    return ArrivalData(
        is_3d=True,
        frequency=float(config.frequency),
        source_x=source_x_values,
        source_y=source_y_values,
        source_depths=sources,
        receiver_depths=depths,
        receiver_ranges=ranges,
        receiver_bearings=bearings,
        arrivals=arrivals_grid,
    )


def write_arrival_grid_3d(path, arrivals: ArrivalData) -> None:
    write_arrivals_ascii(path, arrivals)


__all__ = [
    "Bellhop3DArrival",
    "compute_arrival_grid_3d",
    "write_arrival_grid_3d",
]
