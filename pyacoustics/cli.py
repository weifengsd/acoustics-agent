"""Command-line interface for running pyacoustics simulations."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acoustics-agent",
        description="Run pyacoustics simulations from YAML configuration files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a simulation config.")
    run_parser.add_argument("config", type=Path, help="Path to a YAML simulation config.")
    run_parser.add_argument(
        "--mode",
        choices=("native", "legacy"),
        default="native",
        help="Use the native Python solver or legacy Acoustics Toolbox executables.",
    )
    run_parser.add_argument(
        "--at-bin-path",
        type=Path,
        help="Directory containing optional Acoustics Toolbox executables for legacy mode.",
    )
    run_parser.add_argument("--ray-output", type=Path, help="Write AT-compatible ray output.")
    run_parser.add_argument("--tl-output", type=Path, help="Write a transmission-loss plot.")
    run_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting even when the selected solver can produce a default plot.",
    )
    return parser


def _configure_cli_cache_dirs() -> None:
    cache_root = Path(tempfile.gettempdir()) / "acoustics-agent-cache"
    mpl_config_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def _run_simulation(args: argparse.Namespace) -> int:
    _configure_cli_cache_dirs()

    from pyacoustics.simulation import Simulation

    sim = Simulation(
        args.config,
        external_bin_path=str(args.at_bin_path) if args.at_bin_path else None,
        mode=args.mode,
    )
    result = sim.run(mode=args.mode, save_ray_path=str(args.ray_output) if args.ray_output else None)

    if args.tl_output is not None:
        args.tl_output.parent.mkdir(parents=True, exist_ok=True)
        sim.plot_tl(str(args.tl_output))
    elif not args.no_plot and args.ray_output is None and getattr(sim.config.solver, "type", None) == "bellhop":
        # Keep the CLI side-effect light by not opening an interactive window.
        pass

    solver_type = getattr(sim.config.solver, "type", "unknown")
    result_count = len(result) if hasattr(result, "__len__") else 1
    print(f"Completed {solver_type} simulation: {args.config} ({result_count} result item(s))")
    if args.ray_output is not None:
        print(f"Ray output: {args.ray_output}")
    if args.tl_output is not None:
        print(f"TL plot: {args.tl_output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _build_arg_parser().parse_args(argv)
    if args.command == "run":
        return _run_simulation(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
