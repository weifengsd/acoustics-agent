"""Delay-and-sum synthesis for Bellhop arrival files.

This module is a framework-oriented Python refactor of AT Matlab
``delayandsum.m``. It converts Bellhop/Bellhop3D arrivals into receiver
time series by delaying and summing analytic copies of a source waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import savemat
from scipy.signal import correlate, correlation_lags, fftconvolve, hilbert, resample_poly


@dataclass(frozen=True)
class ArrivalRecord:
    """Arrivals for a single receiver/source combination."""

    amplitudes: np.ndarray
    delays: np.ndarray
    src_decl_angles: np.ndarray
    rcvr_decl_angles: np.ndarray
    num_top_bounces: np.ndarray
    num_bottom_bounces: np.ndarray
    src_azim_angles: np.ndarray | None = None
    rcvr_azim_angles: np.ndarray | None = None
    phase_angles: np.ndarray | None = None

    @property
    def count(self) -> int:
        return int(self.amplitudes.size)


@dataclass(frozen=True)
class ArrivalData:
    """Parsed AT ASCII arrivals and receiver/source grids."""

    is_3d: bool
    frequency: float
    source_depths: np.ndarray
    receiver_depths: np.ndarray
    receiver_ranges: np.ndarray
    arrivals: np.ndarray
    source_x: np.ndarray | None = None
    source_y: np.ndarray | None = None
    receiver_bearings: np.ndarray | None = None


@dataclass(frozen=True)
class DelayAndSumResult:
    """Synthesized receiver time-series cube."""

    timeseries: np.ndarray
    time_windows: np.ndarray
    sample_rate: float
    receiver_depths: np.ndarray
    receiver_ranges: np.ndarray
    source_index: int
    receiver_bearings: np.ndarray | None = None
    source_depths: np.ndarray | None = None
    source_x_index: int = 0
    source_y_index: int = 0
    source_x: np.ndarray | None = None
    source_y: np.ndarray | None = None


@dataclass(frozen=True)
class MatchedFilterResult:
    """Matched-filtered receiver time-series cube."""

    timeseries: np.ndarray
    lags: np.ndarray
    sample_rate: float
    receiver_depths: np.ndarray
    receiver_ranges: np.ndarray
    source_index: int
    receiver_bearings: np.ndarray | None = None
    source_depths: np.ndarray | None = None
    source_x_index: int = 0
    source_y_index: int = 0
    source_x: np.ndarray | None = None
    source_y: np.ndarray | None = None


def _empty_record(is_3d: bool) -> ArrivalRecord:
    empty_float = np.zeros(0, dtype=np.float64)
    empty_complex = np.zeros(0, dtype=np.complex128)
    return ArrivalRecord(
        amplitudes=empty_complex,
        delays=empty_complex,
        src_decl_angles=empty_float,
        rcvr_decl_angles=empty_float,
        src_azim_angles=empty_float if is_3d else None,
        rcvr_azim_angles=empty_float if is_3d else None,
        phase_angles=empty_float,
        num_top_bounces=np.zeros(0, dtype=np.int16),
        num_bottom_bounces=np.zeros(0, dtype=np.int16),
    )


def _tokenize_arrivals(path: str | Path) -> list[str]:
    tokens: list[str] = []
    for line in Path(path).read_text().splitlines():
        clean = line.split("!", 1)[0].replace(",", " ").replace("/", " ")
        for token in clean.split():
            if "'" in token or '"' in token:
                tokens.append(token)
            else:
                tokens.append(token.replace("D", "E").replace("d", "e"))
    return tokens


def _take(tokens: Sequence[str], idx: int, count: int) -> tuple[list[str], int]:
    end = idx + count
    if end > len(tokens):
        raise ValueError("Unexpected end of arrivals file")
    return list(tokens[idx:end]), end


def _can_parse_2d_arrival_records(tokens: Sequence[str], idx: int, n_records: int, n_sources: int) -> bool:
    probe = idx
    try:
        for _ in range(n_sources):
            probe += 1
            for _ in range(n_records):
                narr = int(tokens[probe])
                probe += 1 + 8 * narr
                if probe > len(tokens):
                    return False
    except (IndexError, ValueError):
        return False
    return probe == len(tokens)


def read_arrivals_ascii(path: str | Path) -> ArrivalData:
    """Read an AT ASCII ``.arr`` file.

    Supports both the ``'2D'`` and ``'3D'`` layouts used by
    ``read_arrivals_asc.m`` in Acoustics Toolbox.
    """

    tokens = _tokenize_arrivals(path)
    if not tokens:
        raise ValueError(f"Empty arrivals file: {path}")

    flag = tokens[0].strip("'\"")
    if flag not in {"2D", "3D"}:
        raise ValueError(f"Not an ASCII arrivals file: {path}")
    is_3d = flag == "3D"
    idx = 1

    frequency = float(tokens[idx])
    idx += 1

    source_x = None
    source_y = None
    receiver_bearings = None

    if is_3d:
        nsx = int(tokens[idx]); idx += 1
        raw, idx = _take(tokens, idx, nsx)
        source_x = np.array([float(v) for v in raw], dtype=np.float64)

        nsy = int(tokens[idx]); idx += 1
        raw, idx = _take(tokens, idx, nsy)
        source_y = np.array([float(v) for v in raw], dtype=np.float64)

    nsz = int(tokens[idx]); idx += 1
    raw, idx = _take(tokens, idx, nsz)
    source_depths = np.array([float(v) for v in raw], dtype=np.float64)

    nrz = int(tokens[idx]); idx += 1
    raw, idx = _take(tokens, idx, nrz)
    receiver_depths = np.array([float(v) for v in raw], dtype=np.float64)

    nrr = int(tokens[idx]); idx += 1
    raw, idx = _take(tokens, idx, nrr)
    receiver_ranges = np.array([float(v) for v in raw], dtype=np.float64)

    irregular_2d = False
    if is_3d:
        ntheta = int(tokens[idx]); idx += 1
        raw, idx = _take(tokens, idx, ntheta)
        receiver_bearings = np.array([float(v) for v in raw], dtype=np.float64)
        if source_x is not None and source_y is not None and source_x.size * source_y.size > 1:
            arrivals = np.empty((nrr, nrz, ntheta, nsz, source_x.size, source_y.size), dtype=object)
        else:
            arrivals = np.empty((nrr, nrz, ntheta, nsz), dtype=object)
    else:
        ntheta = 0
        rectangular_records = nrr * nrz
        if (
            nrr == nrz
            and not _can_parse_2d_arrival_records(tokens, idx, rectangular_records, nsz)
            and _can_parse_2d_arrival_records(tokens, idx, nrr, nsz)
        ):
            irregular_2d = True
            arrivals = np.empty((nrr, 1, nsz), dtype=object)
        else:
            arrivals = np.empty((nrr, nrz, nsz), dtype=object)

    for isd in range(nsz):
        source_x_iter = range(source_x.size) if is_3d and arrivals.ndim == 6 else range(1)
        source_y_iter = range(source_y.size) if is_3d and arrivals.ndim == 6 else range(1)
        for isx in source_x_iter:
            for isy in source_y_iter:
                idx += 1  # max arrivals for this source, retained only in AT for preallocation
                theta_iter = range(ntheta) if is_3d else range(1)
                for itheta in theta_iter:
                    depth_iter = range(1) if irregular_2d else range(nrz)
                    for iz in depth_iter:
                        range_iter = range(nrr)
                        for ir in range_iter:
                            narr = int(tokens[idx])
                            idx += 1
                            if narr == 0:
                                record = _empty_record(is_3d)
                            elif is_3d:
                                raw, idx = _take(tokens, idx, 10 * narr)
                                da = np.array([float(v) for v in raw], dtype=np.float64).reshape(narr, 10)
                                phases = np.deg2rad(da[:, 1])
                                record = ArrivalRecord(
                                    amplitudes=da[:, 0] * np.exp(1j * phases),
                                    delays=da[:, 2] + 1j * da[:, 3],
                                    src_decl_angles=da[:, 4],
                                    src_azim_angles=da[:, 5],
                                    rcvr_decl_angles=da[:, 6],
                                    rcvr_azim_angles=da[:, 7],
                                    phase_angles=phases,
                                    num_top_bounces=da[:, 8].astype(np.int16),
                                    num_bottom_bounces=da[:, 9].astype(np.int16),
                                )
                            else:
                                raw, idx = _take(tokens, idx, 8 * narr)
                                da = np.array([float(v) for v in raw], dtype=np.float64).reshape(narr, 8)
                                phases = np.deg2rad(da[:, 1])
                                record = ArrivalRecord(
                                    amplitudes=da[:, 0] * np.exp(1j * phases),
                                    delays=da[:, 2] + 1j * da[:, 3],
                                    src_decl_angles=da[:, 4],
                                    rcvr_decl_angles=da[:, 5],
                                    phase_angles=phases,
                                    num_top_bounces=da[:, 6].astype(np.int16),
                                    num_bottom_bounces=da[:, 7].astype(np.int16),
                                )
                            if is_3d and arrivals.ndim == 6:
                                arrivals[ir, iz, itheta, isd, isx, isy] = record
                            elif is_3d:
                                arrivals[ir, iz, itheta, isd] = record
                            elif irregular_2d:
                                arrivals[ir, 0, isd] = record
                            else:
                                arrivals[ir, iz, isd] = record

    return ArrivalData(
        is_3d=is_3d,
        frequency=frequency,
        source_depths=source_depths,
        receiver_depths=receiver_depths,
        receiver_ranges=receiver_ranges,
        receiver_bearings=receiver_bearings,
        source_x=source_x,
        source_y=source_y,
        arrivals=arrivals,
    )


def _point_source_arrival_factor(receiver_range: float) -> float:
    return 1.0e5 if receiver_range == 0.0 else 1.0 / np.sqrt(receiver_range)


def write_arrivals_ascii(
    path: str | Path,
    arrivals: ArrivalData,
    *,
    source_type: str = "P",
    scale_3d_2d_radials: bool = False,
) -> None:
    """Write an AT ASCII ``.arr`` file from an :class:`ArrivalData` object.

    The 2D layout matches AT ``WriteArrivalsASCII``. The 3D layout matches
    ``WriteArrivalsASCII3D`` for full Bellhop3D runs, whose arrival amplitudes
    are already geometrically scaled. Set ``scale_3d_2d_radials=True`` only for
    Bellhop3D runs that store copied 2D radial arrivals.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    source_depths = np.asarray(arrivals.source_depths, dtype=np.float64)
    receiver_depths = np.asarray(arrivals.receiver_depths, dtype=np.float64)
    receiver_ranges = np.asarray(arrivals.receiver_ranges, dtype=np.float64)

    if arrivals.is_3d:
        if arrivals.source_x is None or arrivals.source_y is None:
            raise ValueError("3D arrivals require source_x and source_y grids")
        if arrivals.receiver_bearings is None:
            raise ValueError("3D arrivals require receiver_bearings")
        source_x = np.asarray(arrivals.source_x, dtype=np.float64)
        source_y = np.asarray(arrivals.source_y, dtype=np.float64)
        receiver_bearings = np.asarray(arrivals.receiver_bearings, dtype=np.float64)
        expected_shape = (
            len(receiver_ranges),
            len(receiver_depths),
            len(receiver_bearings),
            len(source_depths),
        )
        expected_shape_full_3d = expected_shape + (len(source_x), len(source_y))
    else:
        source_x = source_y = receiver_bearings = None
        expected_shape = (len(receiver_ranges), len(receiver_depths), len(source_depths))
        expected_shape_full_3d = None

    if arrivals.arrivals.shape != expected_shape and arrivals.arrivals.shape != expected_shape_full_3d:
        raise ValueError(
            f"arrivals array shape {arrivals.arrivals.shape} does not match "
            f"{expected_shape}"
            + (f" or {expected_shape_full_3d}" if expected_shape_full_3d is not None else "")
        )
    if (
        arrivals.is_3d
        and arrivals.arrivals.shape == expected_shape
        and expected_shape_full_3d is not None
        and len(source_x) * len(source_y) > 1
    ):
        raise ValueError("3D arrivals with multiple source_x/source_y values require the full 6D shape")

    with path.open("w") as f:
        if arrivals.is_3d:
            f.write("'3D'\n")
            f.write(f"{arrivals.frequency:.9g}\n")
            f.write(f"{len(source_x)}\n")
            f.write(" ".join(f"{x:.9g}" for x in source_x) + " /\n")
            f.write(f"{len(source_y)}\n")
            f.write(" ".join(f"{y:.9g}" for y in source_y) + " /\n")
        else:
            f.write("'2D'\n")
            f.write(f"{arrivals.frequency:.9g}\n")

        f.write(f"{len(source_depths)}\n")
        f.write(" ".join(f"{z:.9g}" for z in source_depths) + " /\n")
        f.write(f"{len(receiver_depths)}\n")
        f.write(" ".join(f"{z:.9g}" for z in receiver_depths) + " /\n")
        f.write(f"{len(receiver_ranges)}\n")
        f.write(" ".join(f"{r:.9g}" for r in receiver_ranges) + " /\n")

        if arrivals.is_3d:
            f.write(f"{len(receiver_bearings)}\n")
            f.write(" ".join(f"{theta:.9g}" for theta in receiver_bearings) + " /\n")

        def write_record(record: ArrivalRecord | None, receiver_range: float) -> None:
            if record is None:
                record = _empty_record(arrivals.is_3d)
            f.write(f"{record.count}\n")

            if arrivals.is_3d and not scale_3d_2d_radials:
                factor = 1.0
            elif source_type.upper() == "X":
                factor = 4.0 * np.sqrt(np.pi)
            else:
                factor = _point_source_arrival_factor(float(receiver_range))

            for ia in range(record.count):
                phase_angle = (
                    float(record.phase_angles[ia])
                    if record.phase_angles is not None
                    else float(np.angle(record.amplitudes[ia]))
                )
                if arrivals.is_3d:
                    src_az = 0.0 if record.src_azim_angles is None else record.src_azim_angles[ia]
                    rcvr_az = 0.0 if record.rcvr_azim_angles is None else record.rcvr_azim_angles[ia]
                    f.write(
                        f"{factor * abs(record.amplitudes[ia]):.9g} "
                        f"{np.degrees(phase_angle):.9g} "
                        f"{np.real(record.delays[ia]):.9g} "
                        f"{np.imag(record.delays[ia]):.9g} "
                        f"{record.src_decl_angles[ia]:.9g} "
                        f"{src_az:.9g} "
                        f"{record.rcvr_decl_angles[ia]:.9g} "
                        f"{rcvr_az:.9g} "
                        f"{int(record.num_top_bounces[ia]):d} "
                        f"{int(record.num_bottom_bounces[ia]):d}\n"
                    )
                else:
                    f.write(
                        f"{factor * abs(record.amplitudes[ia]):.9g} "
                        f"{np.degrees(phase_angle):.9g} "
                        f"{np.real(record.delays[ia]):.9g} "
                        f"{np.imag(record.delays[ia]):.9g} "
                        f"{record.src_decl_angles[ia]:.9g} "
                        f"{record.rcvr_decl_angles[ia]:.9g} "
                        f"{int(record.num_top_bounces[ia]):d} "
                        f"{int(record.num_bottom_bounces[ia]):d}\n"
                    )

        for isd in range(len(source_depths)):
            source_x_iter = range(len(source_x)) if arrivals.is_3d and arrivals.arrivals.ndim == 6 else range(1)
            source_y_iter = range(len(source_y)) if arrivals.is_3d and arrivals.arrivals.ndim == 6 else range(1)
            for isx in source_x_iter:
                for isy in source_y_iter:
                    if arrivals.is_3d and arrivals.arrivals.ndim == 6:
                        source_records = arrivals.arrivals[:, :, :, isd, isx, isy].flat
                    elif arrivals.is_3d:
                        source_records = arrivals.arrivals[:, :, :, isd].flat
                    else:
                        source_records = arrivals.arrivals[:, :, isd].flat
                    max_narr = max((0 if item is None else item.count for item in source_records), default=0)
                    f.write(f"{max_narr}\n")
                    theta_iter = range(len(receiver_bearings)) if arrivals.is_3d else range(1)
                    for itheta in theta_iter:
                        for iz in range(len(receiver_depths)):
                            for ir, receiver_range in enumerate(receiver_ranges):
                                if arrivals.is_3d and arrivals.arrivals.ndim == 6:
                                    record = arrivals.arrivals[ir, iz, itheta, isd, isx, isy]
                                elif arrivals.is_3d:
                                    record = arrivals.arrivals[ir, iz, itheta, isd]
                                else:
                                    record = arrivals.arrivals[ir, iz, isd]
                                write_record(record, float(receiver_range))


def prepare_source_waveform(
    source: Sequence[float],
    *,
    normalize: bool = True,
    invert_sign: bool = True,
    shift_fraction: float = 0.1,
) -> np.ndarray:
    """Normalize and front-pad a source waveform as in ``delayandsum.m``."""

    waveform = np.asarray(source, dtype=np.float64).copy()
    if waveform.ndim != 1:
        raise ValueError("source waveform must be one-dimensional")
    if waveform.size == 0:
        raise ValueError("source waveform must not be empty")

    peak = np.max(np.abs(waveform))
    if normalize and peak > 0.0:
        waveform = waveform / peak
    if invert_sign:
        waveform = -waveform

    shift = int(round(shift_fraction * waveform.size))
    if shift > 0:
        waveform = np.roll(waveform, shift)
    return waveform


def _next_radix2(length: int) -> int:
    n = 1
    while n < length:
        n *= 2
    return n


def _modified_chirp_z_unit_circle(x: np.ndarray, m: int, delta_phi: float, phi_0: float = 0.0) -> np.ndarray:
    n = x.size
    delta = delta_phi / 2.0
    n_v = np.arange(-(n - 1), m, dtype=np.float64)
    v = np.exp(1j * delta * n_v * n_v)
    n_y = np.arange(n, dtype=np.float64)
    y = np.exp(-1j * phi_0) * x * np.exp(-1j * delta * n_y * n_y)
    g = fftconvolve(y, v, mode="full")
    n_x = np.arange(m, dtype=np.float64)
    return g[n - 1 : n + m - 1] * np.exp(-1j * delta * n_x * n_x)


def _dopplerize(waveform: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        raise ValueError("Doppler alpha must be positive")
    if alpha == 1.0:
        return np.asarray(waveform, dtype=np.float64).copy()

    nt = waveform.size
    nfft = _next_radix2(nt)
    padded = np.zeros(nfft, dtype=np.float64)
    padded[:nt] = waveform
    ups = 1
    if alpha < 1.0:
        padded = resample_poly(padded, 2, 1)
        ups = 2

    delta_phi = 2.0 * np.pi / (ups * nfft * alpha)
    spectrum = _modified_chirp_z_unit_circle(padded, nfft, delta_phi)
    spectrum[nfft // 2 + 1 :] = 0.0
    scaled = np.real((2.0 / ups) * (1.0 / alpha) * np.fft.ifft(spectrum, nfft))
    return np.asarray(scaled[:nt], dtype=np.float64)


def _analytic_bank(
    waveform: np.ndarray,
    velocity: np.ndarray,
    doppler_reference_speed: float,
    doppler_velocity_bins: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if np.linalg.norm(velocity) <= 0.0:
        return hilbert(waveform)[:, None], None
    if doppler_reference_speed <= 0.0:
        raise ValueError("doppler_reference_speed must be positive")

    if doppler_velocity_bins is None:
        vmax = max(1.0, np.linalg.norm(velocity))
        doppler_velocity_bins = np.linspace(-vmax, vmax, 51)
    alpha_vec = 1.0 - doppler_velocity_bins / doppler_reference_speed
    bank = np.zeros((waveform.size, alpha_vec.size), dtype=np.complex128)
    for i, alpha in enumerate(alpha_vec):
        bank[:, i] = hilbert(_dopplerize(waveform, float(alpha)))
    return bank, alpha_vec


def _receiver_tangent(record: ArrivalRecord, arrival_index: int, velocity_ndim: int) -> np.ndarray:
    decl = np.deg2rad(record.rcvr_decl_angles[arrival_index])
    if velocity_ndim == 3:
        azim = 0.0 if record.rcvr_azim_angles is None else np.deg2rad(record.rcvr_azim_angles[arrival_index])
        return np.array(
            [
                np.cos(decl) * np.cos(azim),
                np.cos(decl) * np.sin(azim),
                np.sin(decl),
            ],
            dtype=np.float64,
        )
    return np.array([np.cos(decl), np.sin(decl)], dtype=np.float64)


def _add_arrival(
    output: np.ndarray,
    analytic_source: np.ndarray,
    amplitude: complex,
    delay: complex,
    tstart: float,
    sample_interval: float,
    omega: float,
) -> None:
    arrival_time = float(np.real(delay)) - tstart
    volume_loss = float(np.exp(omega * np.imag(delay)))

    start = int(round(arrival_time / sample_interval))
    end = start + analytic_source.size
    src_start = 0
    src_end = analytic_source.size

    if start < 0:
        src_start = -start
        start = 0
    if end > output.size:
        src_end -= end - output.size
        end = output.size
    if start >= end or src_start >= src_end:
        return

    output[start:end] += volume_loss * np.real(amplitude * analytic_source[src_start:src_end])


def delay_and_sum(
    arrivals: ArrivalData,
    source_waveform: Sequence[float],
    *,
    sample_rate: float,
    window_duration: float = 10.0,
    reduction_velocity: float = 1520.0,
    start_time_margin: float = 0.1,
    source_index: int = 0,
    source_x_index: int = 0,
    source_y_index: int = 0,
    receiver_velocity: Sequence[float] = (0.0, 0.0),
    doppler_reference_speed: float = 1500.0,
    doppler_velocity_bins: Sequence[float] | None = None,
    normalize_source: bool = True,
    invert_source_sign: bool = True,
    shift_fraction: float = 0.1,
) -> DelayAndSumResult:
    """Synthesize receiver waveforms from arrivals.

    Returns a time-series array shaped ``(time, depth, range)`` for 2D
    arrivals and ``(time, depth, range, bearing)`` for 3D arrivals.
    """

    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if window_duration <= 0.0:
        raise ValueError("window_duration must be positive")
    if source_index < 0 or source_index >= arrivals.source_depths.size:
        raise IndexError("source_index out of range")
    if source_x_index < 0:
        raise IndexError("source_x_index out of range")
    if source_y_index < 0:
        raise IndexError("source_y_index out of range")
    has_source_xy_blocks = arrivals.is_3d and arrivals.arrivals.ndim == 6
    if has_source_xy_blocks:
        if arrivals.source_x is None or source_x_index >= arrivals.source_x.size:
            raise IndexError("source_x_index out of range")
        if arrivals.source_y is None or source_y_index >= arrivals.source_y.size:
            raise IndexError("source_y_index out of range")
    elif source_x_index != 0 or source_y_index != 0:
        raise IndexError("source_x_index/source_y_index require 3D arrivals with source coordinate blocks")

    waveform = prepare_source_waveform(
        source_waveform,
        normalize=normalize_source,
        invert_sign=invert_source_sign,
        shift_fraction=shift_fraction,
    )
    velocity = np.asarray(receiver_velocity, dtype=np.float64)
    if velocity.shape not in {(2,), (3,)}:
        raise ValueError("receiver_velocity must contain two components (vr, vz) or three components (vx, vy, vz)")
    if velocity.shape == (3,) and not arrivals.is_3d:
        raise ValueError("three-component receiver_velocity requires 3D arrivals")
    bins = None if doppler_velocity_bins is None else np.asarray(doppler_velocity_bins, dtype=np.float64)
    analytic_bank, alpha_vec = _analytic_bank(waveform, velocity, doppler_reference_speed, bins)

    sample_interval = 1.0 / sample_rate
    nt = int(round(window_duration / sample_interval))
    nrr = arrivals.receiver_ranges.size
    nrz = arrivals.receiver_depths.size
    ntheta = arrivals.receiver_bearings.size if arrivals.is_3d and arrivals.receiver_bearings is not None else 1

    if arrivals.is_3d:
        timeseries = np.zeros((nt, nrz, nrr, ntheta), dtype=np.float64)
        time_windows = np.zeros((nrr, ntheta, nt), dtype=np.float64)
    else:
        timeseries = np.zeros((nt, nrz, nrr), dtype=np.float64)
        time_windows = np.zeros((nrr, nt), dtype=np.float64)

    omega = 2.0 * np.pi * arrivals.frequency
    offsets = np.arange(nt, dtype=np.float64) * sample_interval

    for ir, receiver_range in enumerate(arrivals.receiver_ranges):
        tstart = receiver_range / reduction_velocity - start_time_margin
        if arrivals.is_3d:
            time_windows[ir, :, :] = tstart + offsets
        else:
            time_windows[ir, :] = tstart + offsets

        for iz in range(nrz):
            for itheta in range(ntheta):
                record = (
                    arrivals.arrivals[ir, iz, itheta, source_index, source_x_index, source_y_index]
                    if has_source_xy_blocks
                    else (
                        arrivals.arrivals[ir, iz, itheta, source_index]
                        if arrivals.is_3d
                        else arrivals.arrivals[ir, iz, source_index]
                    )
                )
                if record is None:
                    record = _empty_record(arrivals.is_3d)
                target = timeseries[:, iz, ir, itheta] if arrivals.is_3d else timeseries[:, iz, ir]
                for ia in range(record.count):
                    if alpha_vec is None:
                        source = analytic_bank[:, 0]
                    else:
                        tangent = _receiver_tangent(record, ia, velocity.size)
                        alpha = 1.0 - np.dot(velocity / doppler_reference_speed, tangent)
                        if alpha_vec.size == 1:
                            ialpha = 0
                        else:
                            ialpha = int(round((alpha - alpha_vec[0]) / (alpha_vec[-1] - alpha_vec[0]) * (alpha_vec.size - 1)))
                            ialpha = max(0, min(ialpha, alpha_vec.size - 1))
                        source = analytic_bank[:, ialpha]
                    _add_arrival(
                        target,
                        source,
                        record.amplitudes[ia],
                        record.delays[ia],
                        tstart,
                        sample_interval,
                        omega,
                    )

    return DelayAndSumResult(
        timeseries=timeseries,
        time_windows=time_windows,
        sample_rate=sample_rate,
        receiver_depths=arrivals.receiver_depths,
        receiver_ranges=arrivals.receiver_ranges,
        source_index=source_index,
        receiver_bearings=arrivals.receiver_bearings,
        source_depths=arrivals.source_depths,
        source_x_index=source_x_index,
        source_y_index=source_y_index,
        source_x=arrivals.source_x,
        source_y=arrivals.source_y,
    )


def save_matlab_timeseries(
    result: DelayAndSumResult,
    path: str | Path,
    *,
    plot_title: str = "delay_and_sum",
    receiver_range_index: int | None = None,
) -> None:
    """Save a Matlab-compatible ``.rts.mat`` payload."""

    if receiver_range_index is None:
        receiver_range_index = result.receiver_ranges.size - 1
    if receiver_range_index < 0 or receiver_range_index >= result.receiver_ranges.size:
        raise IndexError("receiver_range_index out of range")

    if result.timeseries.ndim == 4:
        rts = -result.timeseries[:, :, receiver_range_index, :]
        tout = result.time_windows[receiver_range_index, :, :]
    else:
        rts = -result.timeseries[:, :, receiver_range_index]
        tout = result.time_windows[receiver_range_index, :]

    savemat(
        path,
        {
            "PlotTitle": plot_title,
            "tout": tout,
            "RTS": rts,
            "sample_rate": result.sample_rate,
            "receiver_depths": result.receiver_depths,
            "receiver_ranges": result.receiver_ranges,
            "source_index": result.source_index,
            "source_depths": np.array([]) if result.source_depths is None else result.source_depths,
            "selected_source_depth": (
                np.array([])
                if result.source_depths is None
                else np.array([result.source_depths[result.source_index]], dtype=np.float64)
            ),
            "source_x_index": result.source_x_index,
            "source_y_index": result.source_y_index,
            "source_x": np.array([]) if result.source_x is None else result.source_x,
            "source_y": np.array([]) if result.source_y is None else result.source_y,
            "selected_source_x": (
                np.array([])
                if result.source_x is None
                else np.array([result.source_x[result.source_x_index]], dtype=np.float64)
            ),
            "selected_source_y": (
                np.array([])
                if result.source_y is None
                else np.array([result.source_y[result.source_y_index]], dtype=np.float64)
            ),
            "receiver_bearings": np.array([]) if result.receiver_bearings is None else result.receiver_bearings,
        },
    )


def matched_filter_timeseries(
    result: DelayAndSumResult,
    replica_waveform: Sequence[float],
    *,
    mode: str = "same",
    normalize_replica: bool = True,
) -> MatchedFilterResult:
    """Apply a matched filter to every receiver trace along the time axis."""

    replica = np.asarray(replica_waveform, dtype=np.float64)
    if replica.ndim != 1:
        raise ValueError("replica_waveform must be one-dimensional")
    if replica.size == 0:
        raise ValueError("replica_waveform must not be empty")
    if mode not in {"same", "full", "valid"}:
        raise ValueError("mode must be 'same', 'full', or 'valid'")

    if normalize_replica:
        norm = np.linalg.norm(replica)
        if norm > 0.0:
            replica = replica / norm

    filtered_shape = list(result.timeseries.shape)
    filtered_shape[0] = correlation_lags(result.timeseries.shape[0], replica.size, mode=mode).size
    filtered = np.zeros(filtered_shape, dtype=np.float64)

    for index in np.ndindex(result.timeseries.shape[1:]):
        filtered[(slice(None),) + index] = correlate(result.timeseries[(slice(None),) + index], replica, mode=mode)

    return MatchedFilterResult(
        timeseries=filtered,
        lags=correlation_lags(result.timeseries.shape[0], replica.size, mode=mode),
        sample_rate=result.sample_rate,
        receiver_depths=result.receiver_depths,
        receiver_ranges=result.receiver_ranges,
        source_index=result.source_index,
        receiver_bearings=result.receiver_bearings,
        source_depths=result.source_depths,
        source_x_index=result.source_x_index,
        source_y_index=result.source_y_index,
        source_x=result.source_x,
        source_y=result.source_y,
    )


def save_matched_filter_matlab(
    result: MatchedFilterResult,
    path: str | Path,
    *,
    plot_title: str = "matched_filter",
    receiver_range_index: int | None = None,
) -> None:
    """Save a Matlab-compatible matched-filter payload."""

    if receiver_range_index is None:
        receiver_range_index = result.receiver_ranges.size - 1
    if receiver_range_index < 0 or receiver_range_index >= result.receiver_ranges.size:
        raise IndexError("receiver_range_index out of range")

    if result.timeseries.ndim == 4:
        mf = result.timeseries[:, :, receiver_range_index, :]
    else:
        mf = result.timeseries[:, :, receiver_range_index]

    savemat(
        path,
        {
            "PlotTitle": plot_title,
            "MF": mf,
            "lags": result.lags,
            "lag_seconds": result.lags / result.sample_rate,
            "sample_rate": result.sample_rate,
            "receiver_depths": result.receiver_depths,
            "receiver_ranges": result.receiver_ranges,
            "source_index": result.source_index,
            "source_depths": np.array([]) if result.source_depths is None else result.source_depths,
            "selected_source_depth": (
                np.array([])
                if result.source_depths is None
                else np.array([result.source_depths[result.source_index]], dtype=np.float64)
            ),
            "source_x_index": result.source_x_index,
            "source_y_index": result.source_y_index,
            "source_x": np.array([]) if result.source_x is None else result.source_x,
            "source_y": np.array([]) if result.source_y is None else result.source_y,
            "selected_source_x": (
                np.array([])
                if result.source_x is None
                else np.array([result.source_x[result.source_x_index]], dtype=np.float64)
            ),
            "selected_source_y": (
                np.array([])
                if result.source_y is None
                else np.array([result.source_y[result.source_y_index]], dtype=np.float64)
            ),
            "receiver_bearings": np.array([]) if result.receiver_bearings is None else result.receiver_bearings,
        },
    )


def synthesize_from_arrivals_file(
    arrivals_path: str | Path,
    source_waveform: Sequence[float],
    *,
    sample_rate: float,
    output_mat_path: str | Path | None = None,
    **kwargs,
) -> DelayAndSumResult:
    """Convenience wrapper for file-based workflows."""

    arrival_data = read_arrivals_ascii(arrivals_path)
    result = delay_and_sum(arrival_data, source_waveform, sample_rate=sample_rate, **kwargs)
    if output_mat_path is not None:
        save_matlab_timeseries(result, output_mat_path, plot_title=Path(arrivals_path).stem)
    return result


__all__ = [
    "ArrivalData",
    "ArrivalRecord",
    "DelayAndSumResult",
    "MatchedFilterResult",
    "delay_and_sum",
    "matched_filter_timeseries",
    "prepare_source_waveform",
    "read_arrivals_ascii",
    "save_matched_filter_matlab",
    "save_matlab_timeseries",
    "synthesize_from_arrivals_file",
]
