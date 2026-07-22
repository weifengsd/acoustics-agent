from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from pyacoustics.schema import SimulationConfig


@dataclass
class BellhopArrival:
    amplitude: float
    phase_rad: float
    delay: complex
    src_decl_angle: float
    rcvr_decl_angle: float
    num_top_bounces: int
    num_bottom_bounces: int
    distance_to_receiver: float = 0.0

    @property
    def complex_amplitude(self) -> complex:
        return self.amplitude * np.exp(1j * self.phase_rad)

    def to_legacy_dict(self) -> dict:
        return {
            "tau": float(np.real(self.delay)),
            "amplitude": self.complex_amplitude,
            "arrival_angle": self.rcvr_decl_angle,
            "launch_angle": self.src_decl_angle,
            "dist": self.distance_to_receiver,
            "num_top_bounces": self.num_top_bounces,
            "num_bottom_bounces": self.num_bottom_bounces,
        }


@dataclass
class BellhopArrivalGrid:
    frequency: float
    source_depths: np.ndarray
    receiver_depths: np.ndarray
    receiver_ranges: np.ndarray
    arrivals: np.ndarray


@njit(fastmath=True)
def find_arrivals_at_receiver(
    r_rcv: float,
    z_rcv: float,
    ray_r: np.ndarray,
    ray_z: np.ndarray,
    ray_c: np.ndarray,
    ray_tau: np.ndarray,
    ray_p: np.ndarray,
    ray_q: np.ndarray,
    ray_amp: np.ndarray,
    ray_phase: np.ndarray,
    ray_top_bnc: np.ndarray,
    ray_bot_bnc: np.ndarray,
    n_pts: int,
    freq: float,
    z_min: float,
    z_max: float,
    d_alpha: float,
    alpha_rad: float,
    c_src: float,
    beam_shape: int,
):
    arrivals = []

    beam_window = 4.0
    q0 = c_src / d_alpha
    if beam_shape == 0:
        ratio1 = np.sqrt(abs(np.cos(alpha_rad)))
    else:
        ratio1 = np.sqrt(abs(np.cos(alpha_rad))) / np.sqrt(2.0 * np.pi)
    phase = 0.0
    q_old = ray_q[0]

    for i in range(n_pts - 1):
        r1, z1 = ray_r[i], ray_z[i]
        r2, z2 = ray_r[i + 1], ray_z[i + 1]

        dr = r2 - r1
        if abs(dr) < 1e-10:
            continue

        dz = z2 - z1
        rlen = np.sqrt(dr * dr + dz * dz)
        if rlen < 1.0e-10:
            continue
        rayt_r = dr / rlen
        rayt_z = dz / rlen
        if abs(rayt_r) < 1.0e-10:
            continue

        q_start = ray_q[i]
        q_end = ray_q[i + 1]
        if (q_start <= 0.0 and q_old > 0.0) or (q_start >= 0.0 and q_old < 0.0):
            phase += np.pi * 0.5
        q_old = q_start

        r_min_seg, r_max_seg = min(r1, r2), max(r1, r2)
        if r_rcv < r_min_seg or r_rcv > r_max_seg:
            continue

        local_lambda = ray_c[i] / freq
        sigma_floor = min(0.2 * freq * ray_tau[i + 1], np.pi * local_lambda)
        sigma_seg = max(abs(q_start), abs(q_end)) / q0 / abs(rayt_r)
        if beam_shape == 1:
            sigma_seg = max(sigma_seg, sigma_floor)
            radius_max = beam_window * sigma_seg
        else:
            radius_max = sigma_seg
        if abs(rayt_r) > 0.5:
            z_low = min(z1, z2) - radius_max
            z_high = max(z1, z2) + radius_max
            if z_rcv < z_low or z_rcv > z_high:
                continue

        frac = (r_rcv - r1) / dr
        if frac < 0.0 or frac > 1.0:
            continue
        if frac == 1.0 and i < n_pts - 2:
            continue

        dx_r = r_rcv - r1
        dz_r = z_rcv - z1
        s = (dx_r * rayt_r + dz_r * rayt_z) / rlen

        n_dist = abs(dx_r * (-rayt_z) + dz_r * rayt_r)
        q_int = q_start + s * (q_end - q_start)
        sigma = abs(q_int / q0)
        if beam_shape == 1:
            sigma1 = max(sigma, sigma_floor)
            in_beam = n_dist < beam_window * sigma1 and sigma1 > 0.0
        else:
            sigma1 = sigma
            in_beam = n_dist < sigma1 and sigma1 > 0.0

        if in_beam:
            tau = ray_tau[i] + s * (ray_tau[i + 1] - ray_tau[i])
            if beam_shape == 1:
                const = ratio1 * np.sqrt(ray_c[i + 1] / (q0 * sigma1)) * abs(ray_amp[i + 1])
                if sigma > 0.0:
                    weight = np.sqrt(sigma / sigma1) * np.exp(-0.5 * (n_dist / sigma1) ** 2)
                else:
                    weight = 0.0
            else:
                const = ratio1 * np.sqrt(ray_c[i + 1] / abs(q_int)) * abs(ray_amp[i + 1])
                weight = (sigma1 - n_dist) / sigma1
            phase_int = phase
            if (q_int <= 0.0 and q_old > 0.0) or (q_int >= 0.0 and q_old < 0.0):
                phase_int += np.pi * 0.5
            amp = abs(const * weight)
            phase_out = ray_phase[i] + phase_int
            angle_rad = np.arctan2(dz, dr)
            top_bnc = int(ray_top_bnc[i + 1])
            bot_bnc = int(ray_bot_bnc[i + 1])

            arrivals.append((tau, amp, phase_out, np.degrees(angle_rad), n_dist, top_bnc, bot_bnc))

    return arrivals


def _merge_arrivals_like_at(
    arrivals: list[BellhopArrival],
    *,
    omega: float,
    phase_tol: float = 0.05,
    launch_angle_tol_deg: float | None = None,
    max_arrivals: int = 200,
) -> list[BellhopArrival]:
    if not arrivals:
        return []

    merged: list[BellhopArrival] = []

    for arrival in arrivals:
        if merged:
            last = merged[-1]
            same_time = omega * abs(arrival.delay - last.delay) < phase_tol
            same_phase = abs(arrival.phase_rad - last.phase_rad) < phase_tol
            same_launch_angle = (
                True
                if launch_angle_tol_deg is None
                else abs(arrival.src_decl_angle - last.src_decl_angle) <= launch_angle_tol_deg
            )
            if same_time and same_phase and same_launch_angle:
                amp_total = last.amplitude + arrival.amplitude
                if abs(amp_total) > np.finfo(float).eps:
                    w1 = last.amplitude / amp_total
                    w2 = arrival.amplitude / amp_total
                    merged[-1] = BellhopArrival(
                        amplitude=amp_total,
                        phase_rad=last.phase_rad,
                        delay=w1 * last.delay + w2 * arrival.delay,
                        src_decl_angle=w1 * last.src_decl_angle + w2 * arrival.src_decl_angle,
                        rcvr_decl_angle=w1 * last.rcvr_decl_angle + w2 * arrival.rcvr_decl_angle,
                        num_top_bounces=last.num_top_bounces,
                        num_bottom_bounces=last.num_bottom_bounces,
                        distance_to_receiver=min(last.distance_to_receiver, arrival.distance_to_receiver),
                    )
                continue

        if len(merged) < max_arrivals:
            merged.append(arrival)
        else:
            weakest = min(range(len(merged)), key=lambda i: merged[i].amplitude)
            if arrival.amplitude > merged[weakest].amplitude:
                merged[weakest] = arrival

    return merged


def compute_arrival_grid(
    config: SimulationConfig,
    *,
    receiver_ranges: np.ndarray | None = None,
    receiver_depths: np.ndarray | None = None,
    source_depths: np.ndarray | None = None,
    max_arrivals: int = 200,
    beam_shape: str = "hat",
    method: str = "beam",
    eigenray_tolerance_m: float = 0.25,
    eigenray_angle_samples: int | None = None,
) -> BellhopArrivalGrid:
    from pyacoustics.solvers.bellhop.core import PyBellhop

    if max_arrivals <= 0:
        raise ValueError("max_arrivals must be positive")
    beam_shape_normalized = beam_shape.lower()
    if beam_shape_normalized not in {"hat", "gaussian"}:
        raise ValueError("beam_shape must be either 'hat' or 'gaussian'")
    method_normalized = method.lower()
    if method_normalized not in {"beam", "eigenray"}:
        raise ValueError("method must be either 'beam' or 'eigenray'")
    if eigenray_tolerance_m <= 0.0:
        raise ValueError("eigenray_tolerance_m must be positive")

    solver = PyBellhop(config)

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
    sources = (
        np.asarray(source_depths, dtype=np.float64)
        if source_depths is not None
        else np.asarray(config.geometry.source.depths, dtype=np.float64)
    )
    irregular_receiver_grid = getattr(config.solver, "receiver_grid", "rectilinear") == "irregular"
    if irregular_receiver_grid and len(ranges) != len(depths):
        raise ValueError("Bellhop irregular receiver grid requires matching receiver depth and range counts")

    # Keep AT-compatible source-depth slices in the native .arr grid.
    depth_count = 1 if irregular_receiver_grid else len(depths)
    arrivals_grid = np.empty((len(ranges), depth_count, len(sources)), dtype=object)
    omega = 2.0 * np.pi * config.frequency

    if method_normalized == "eigenray":
        from pyacoustics.solvers.bellhop.eigenrays import compute_eigenrays

        for isd, source_depth in enumerate(sources):
            depth_iter = enumerate(depths if not irregular_receiver_grid else [0.0])
            for iz, depth_value in depth_iter:
                for ir, receiver_range in enumerate(ranges):
                    depth = depths[ir] if irregular_receiver_grid else depth_value
                    eigenrays = compute_eigenrays(
                        config,
                        float(receiver_range),
                        float(depth),
                        source_depth=float(source_depth),
                        angle_samples=eigenray_angle_samples,
                        tolerance_m=eigenray_tolerance_m,
                    )
                    arrivals_grid[ir, iz, isd] = [
                        BellhopArrival(
                            amplitude=float(abs(ray.amplitude)),
                            phase_rad=float(ray.phase_rad),
                            delay=complex(ray.delay, 0.0),
                            src_decl_angle=float(ray.launch_angle),
                            rcvr_decl_angle=float(ray.receiver_angle),
                            num_top_bounces=int(ray.num_top_bounces),
                            num_bottom_bounces=int(ray.num_bottom_bounces),
                            distance_to_receiver=float(ray.miss_distance),
                        )
                        for ray in eigenrays[:max_arrivals]
                    ]
        return BellhopArrivalGrid(
            frequency=float(config.frequency),
            source_depths=sources,
            receiver_depths=depths,
            receiver_ranges=ranges,
            arrivals=arrivals_grid,
        )

    beam_shape_code = 1 if beam_shape_normalized == "gaussian" else 0

    for isd, source_depth in enumerate(sources):
        beam_data = solver.run_beam_tracing(source_depth=float(source_depth))
        alphas = beam_data["ray_alphas"]
        if len(alphas) > 1:
            d_alpha = float(np.abs(np.mean(np.diff(alphas))))
        else:
            d_alpha = 0.01
        if d_alpha <= np.finfo(float).eps:
            d_alpha = 0.01

        depth_iter = enumerate(depths if not irregular_receiver_grid else [0.0])
        for iz, depth_value in depth_iter:
            for ir, receiver_range in enumerate(ranges):
                depth = depths[ir] if irregular_receiver_grid else depth_value
                raw: list[BellhopArrival] = []
                for i in range(beam_data["num_beams"]):
                    ray_arrs = find_arrivals_at_receiver(
                        receiver_range,
                        depth,
                        beam_data["ray_r"][i],
                        beam_data["ray_z"][i],
                        beam_data["ray_c"][i],
                        beam_data["ray_tau"][i],
                        beam_data["ray_p"][i],
                        beam_data["ray_q"][i],
                        beam_data["ray_amp"][i],
                        beam_data["ray_phase"][i],
                        beam_data["ray_top_bnc"][i],
                        beam_data["ray_bot_bnc"][i],
                        int(beam_data["ray_npts"][i]),
                        config.frequency,
                        0.0,
                        config.environment.bottom.depth,
                        d_alpha,
                        float(alphas[i]),
                        float(beam_data["c_src"]),
                        beam_shape_code,
                    )
                    for tau, amp, phase, rcvr_angle, dist, top_bnc, bot_bnc in ray_arrs:
                        raw.append(
                            BellhopArrival(
                                amplitude=float(amp),
                                phase_rad=float(phase),
                                delay=complex(tau, 0.0),
                                src_decl_angle=float(np.degrees(alphas[i])),
                                rcvr_decl_angle=float(rcvr_angle),
                                num_top_bounces=int(top_bnc),
                                num_bottom_bounces=int(bot_bnc),
                                distance_to_receiver=float(dist),
                            )
                        )
                arrivals_grid[ir, iz, isd] = _merge_arrivals_like_at(
                    raw,
                    omega=omega,
                    launch_angle_tol_deg=(
                        0.25
                        if beam_shape_normalized == "hat" and len(config.environment.ssp.data) > 8
                        else None
                    ),
                    max_arrivals=max_arrivals,
                )

    return BellhopArrivalGrid(
        frequency=float(config.frequency),
        source_depths=sources,
        receiver_depths=depths,
        receiver_ranges=ranges,
        arrivals=arrivals_grid,
    )


def compute_arrivals(config: SimulationConfig, r_rcv: float, z_rcv: float):
    grid = compute_arrival_grid(
        config,
        receiver_ranges=np.array([r_rcv], dtype=np.float64),
        receiver_depths=np.array([z_rcv], dtype=np.float64),
    )
    return [arrival.to_legacy_dict() for arrival in grid.arrivals[0, 0, 0]]


def write_arrivals_ascii(
    path: str | Path,
    grid: BellhopArrivalGrid,
    *,
    source_type: str = "P",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_narr = 0
    for item in grid.arrivals.flat:
        max_narr = max(max_narr, len(item))

    with path.open("w") as f:
        f.write("'2D'\n")
        f.write(f"{grid.frequency:.9g}\n")
        f.write(f"{len(grid.source_depths)}\n")
        f.write(" ".join(f"{z:.9g}" for z in grid.source_depths) + " /\n")
        f.write(f"{len(grid.receiver_depths)}\n")
        f.write(" ".join(f"{z:.9g}" for z in grid.receiver_depths) + " /\n")
        f.write(f"{len(grid.receiver_ranges)}\n")
        f.write(" ".join(f"{r:.9g}" for r in grid.receiver_ranges) + " /\n")

        irregular_receiver_grid = (
            grid.arrivals.shape[1] == 1
            and len(grid.receiver_ranges) == len(grid.receiver_depths)
            and len(grid.receiver_depths) > 1
        )
        for isd in range(len(grid.source_depths)):
            f.write(f"{max_narr}\n")
            depth_iter = range(1) if irregular_receiver_grid else range(len(grid.receiver_depths))
            for iz in depth_iter:
                for ir, receiver_range in enumerate(grid.receiver_ranges):
                    arrivals = grid.arrivals[ir, iz, isd]
                    f.write(f"{len(arrivals)}\n")
                    if source_type.upper() == "X":
                        factor = 4.0 * np.sqrt(np.pi)
                    else:
                        factor = 1.0e5 if receiver_range == 0.0 else 1.0 / np.sqrt(receiver_range)
                    for arrival in arrivals:
                        f.write(
                            f"{factor * arrival.amplitude:.9g} "
                            f"{np.degrees(arrival.phase_rad):.9g} "
                            f"{np.real(arrival.delay):.9g} "
                            f"{np.imag(arrival.delay):.9g} "
                            f"{arrival.src_decl_angle:.9g} "
                            f"{arrival.rcvr_decl_angle:.9g} "
                            f"{arrival.num_top_bounces:d} "
                            f"{arrival.num_bottom_bounces:d}\n"
                        )


__all__ = [
    "BellhopArrival",
    "BellhopArrivalGrid",
    "compute_arrival_grid",
    "compute_arrivals",
    "find_arrivals_at_receiver",
    "write_arrivals_ascii",
]
