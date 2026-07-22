#!/usr/bin/env python3
"""Command-line wrapper for Bellhop delay-and-sum post-processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pyacoustics.postprocessing.delay_and_sum import (
    save_matlab_timeseries,
    synthesize_from_arrivals_file,
)


def _load_source(path: Path) -> tuple[np.ndarray, float | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path), None
    if suffix == ".wav":
        sample_rate, data = wavfile.read(path)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim > 1:
            data = data[:, 0]
        peak = np.max(np.abs(data))
        if peak > 0.0:
            data = data / peak
        return data, float(sample_rate)
    raise ValueError("source file must be .npy or .wav")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrivals", type=Path, help="AT ASCII .arr file")
    parser.add_argument("source", type=Path, help="source waveform (.npy or .wav)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output .mat path")
    parser.add_argument("--sample-rate", type=float, default=None, help="source sample rate for .npy input")
    parser.add_argument("--window-duration", type=float, default=10.0)
    parser.add_argument("--reduction-velocity", type=float, default=1520.0)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--receiver-range-index", type=int, default=None)
    args = parser.parse_args()

    source, file_sample_rate = _load_source(args.source)
    sample_rate = args.sample_rate if args.sample_rate is not None else file_sample_rate
    if sample_rate is None:
        raise ValueError("--sample-rate is required for .npy source waveforms")

    output = args.output
    if output is None:
        output = args.arrivals.with_suffix(".rts.mat")

    result = synthesize_from_arrivals_file(
        args.arrivals,
        source,
        sample_rate=sample_rate,
        window_duration=args.window_duration,
        reduction_velocity=args.reduction_velocity,
        source_index=args.source_index,
    )
    save_matlab_timeseries(
        result,
        output,
        plot_title=args.arrivals.stem,
        receiver_range_index=args.receiver_range_index,
    )
    print(f"Saved delay-and-sum result to {output}")


if __name__ == "__main__":
    main()
