# Usage Guide

This guide covers the stable public entry points. Solver-specific configuration fields and AT compatibility details are documented in [acoustics-agent_user_manual.md](acoustics-agent_user_manual.md).

## Configure a Scenario

Scenarios use YAML and map directly to `SimulationConfig`:

```yaml
project: Example waveguide
frequency: 100.0
environment:
  ssp:
    type: c-linear
    data:
      - {depth: 0.0, c: 1500.0}
      - {depth: 200.0, c: 1500.0}
  surface:
    type: vacuum
    depth: 0.0
  bottom:
    type: rigid
    depth: 200.0
geometry:
  source:
    depths: [100.0]
  receivers:
    ranges: [500.0, 1000.0]
    depths: [50.0, 100.0, 150.0]
solver:
  type: bellhop
  angles: [-10.0, 10.0]
  num_beams: 101
  step_size: 5.0
```

Use `configs/example.yaml` as the smallest runnable example.

## Command Line

```bash
acoustics-agent run configs/example.yaml --ray-output output/example.ray --no-plot
```

Useful options:

- `--mode native|legacy`: select the Python solver or local AT executable.
- `--at-bin-path PATH`: override the AT binary directory for a legacy run.
- `--ray-output PATH`: write an AT-compatible ray file.
- `--tl-output PATH`: compute and plot coherent TL.
- `--no-plot`: skip interactive plotting.

## Python API

```python
import numpy as np

from pyacoustics.simulation import Simulation

sim = Simulation("configs/example.yaml")
rays = sim.run(mode="native")

ranges = np.linspace(100.0, 1000.0, 40)
depths = np.linspace(0.0, 200.0, 30)
coherent = sim.compute_coherent_tl(r_grid=ranges, z_grid=depths)
incoherent = sim.compute_incoherent_tl(r_grid=ranges, z_grid=depths)
semicoherent = sim.compute_semicoherent_tl(r_grid=ranges, z_grid=depths)
```

Arrivals and eigenrays:

```python
arrivals = sim.compute_arrival_grid(method="beam")
eigenrays = sim.run_eigenrays(range_m=1000.0, depth_m=100.0)
```

## AT Parity

AT is optional for native simulation. To compare native results with AT:

```bash
export AT_BIN_PATH=/path/to/AcousticsToolbox/bin
acoustics-at-benchmark --suite all --at-bin-path "$AT_BIN_PATH" --strict-at
```

Benchmark outputs default to `reports/at_benchmark/`, which is ignored by Git. Use `--output-dir` to choose another location.

## Agent Skills

The `skills/` directory exposes small helpers for creating environments, running simulations, and plotting results. These helpers use the same schema and `Simulation` API as direct Python callers; they do not introduce a separate solver path.
