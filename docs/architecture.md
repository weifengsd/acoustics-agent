# Architecture

## Design Goals

`acoustics-agent` separates native numerical kernels, public orchestration APIs, external AT adapters, and parity tooling. Native simulation must remain usable without Acoustics Toolbox, Matlab, or Octave.

## Main Layers

1. `pyacoustics.schema` defines dataclass-based configuration models.
2. `pyacoustics.config` loads YAML or dictionaries and resolves referenced boundary files.
3. `pyacoustics.environment` evaluates 1D, range-dependent 2D, and hexahedral 3D sound-speed fields.
4. `pyacoustics.solvers` contains native Bellhop, Bellhop3D, and Kraken kernels plus external-process adapters.
5. `pyacoustics.simulation.Simulation` provides the high-level API and output caches.
6. `pyacoustics.postprocessing` consumes arrival data for time-series, Doppler, and matched-filter workflows.
7. `pyacoustics.benchmarks` compares native and external results and applies explicit numerical thresholds.

## Solver Boundaries

- Bellhop and Bellhop3D keep kinematic tracing, boundary interaction, beam dynamics, field synthesis, arrivals, and parity comparison in separate modules.
- Kraken separates roots, modes, field synthesis, orchestration, and parity comparison.
- `external_io.py` owns AT-compatible serialization. Native kernels do not parse text or binary AT files directly.
- `external.py` owns executable discovery and process execution. AT remains optional.

## Data Flow

```text
YAML/dict
  -> SimulationConfig
  -> Simulation
  -> native solver or external adapter
  -> rays / modes / pressure / arrivals
  -> plots, AT-compatible files, or postprocessing
```

Parity runs execute both the native path and an external AT path, normalize their outputs into typed records, compare only finite overlapping samples, and evaluate named thresholds. Generated matrices are build artifacts rather than source files.

## Extension Rules

- Add a schema field only when it is consumed by a solver, serializer, or documented compatibility path.
- Keep Numba kernels free of filesystem and subprocess operations.
- Preserve array-axis conventions in public APIs and document any conversion at I/O boundaries.
- Add focused unit tests for numerical helpers and one parity case for externally observable solver behavior.
- Do not add generated `.shd`, `.prt`, plots, matrices, or scratch work to the repository.
