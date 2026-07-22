# acoustics-agent

`acoustics-agent` is a Python-native underwater acoustic simulation framework. It provides Bellhop-style 2D ray tracing, Bellhop3D-style propagation, Kraken-style normal modes, acoustic postprocessing, Acoustics Toolbox (AT) file I/O, and reproducible AT/native parity benchmarks.

[Chinese README](README_ZH.md) | [User guide](docs/usage.md) | [Architecture](docs/architecture.md) | [Validation](docs/validation.md)

## Highlights

- Native Python solvers accelerated with NumPy, SciPy, and Numba.
- Bellhop 2D ray, eigenray, arrival, coherent, incoherent, and semi-coherent TL workflows.
- Bellhop3D ray, arrival, terrain, and receiver-grid TL workflows.
- Kraken/Field normal-mode and field synthesis workflows.
- AT-compatible readers and writers for `.env`, `.flp`, `.mod`, `.shd`, `.ati`, `.bty`, `.ssp`, `.irc`, `.trc`, `.brc`, `.sbp`, `.ray`, `.arr`, and `.prt` data.
- Optional execution of local AT binaries for parity validation.
- Delay-and-sum, Doppler, and matched-filter postprocessing helpers.
- YAML, Python, CLI, and agent-oriented workflow entry points.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

Install development dependencies with:

```bash
python -m pip install -e ".[dev]"
```

AT is optional. For parity runs, point `AT_BIN_PATH` at a local Acoustics Toolbox binary directory:

```bash
export AT_BIN_PATH=/path/to/AcousticsToolbox/bin
```

## Quick Start

Run the example scenario from the command line:

```bash
acoustics-agent run configs/example.yaml --ray-output output/example.ray --no-plot
```

Or use the Python API:

```python
from pyacoustics.simulation import Simulation

simulation = Simulation("configs/example.yaml")
rays = simulation.run(mode="native")
tl = simulation.compute_coherent_tl(num_r=200, num_z=100)
```

The CLI creates parent directories for requested output files. Generated simulation and benchmark artifacts should remain outside version control.

## Validation

Fast local checks do not require AT:

```bash
pytest tests/unit -q
```

When AT is available, run the main parity matrix:

```bash
acoustics-at-benchmark --suite all --at-bin-path "$AT_BIN_PATH" --strict-at
```

The most recent maintained baseline covers 101 Bellhop rows, 22 Kraken/Field rows, 21 Bellhop3D rows, 3 Field3D runtime rows, 11 postprocessing rows, and 1192 AT file-format rows. These numbers are reference evidence, not a substitute for rerunning the gate on a release candidate. See [docs/validation.md](docs/validation.md) for commands and scope.

## Repository Layout

- `pyacoustics/`: simulation engine, solvers, AT I/O, benchmarks, and postprocessing.
- `configs/`: small reusable YAML scenarios.
- `skills/`: agent-friendly environment, execution, and plotting helpers.
- `tests/`: unit tests, integration tests, and required numerical fixtures.
- `docs/`: user, architecture, validation, and release documentation.
- `tools/`: repository maintenance utilities.

Generated `reports/`, `scratch/`, build outputs, solver outputs, and test screenshots are intentionally ignored.

## Citation

**Research citation requirement:** publications, technical reports, benchmarks, datasets, and other research outputs that materially use `acoustics-agent` or results produced by it should cite this software repository. Use GitHub's **Cite this repository** action or the metadata in [CITATION.cff](CITATION.cff).

Redistributed source or binary copies must also retain the copyright and MIT license notice as described in [LICENSE](LICENSE). The research citation policy documents the project's attribution expectation and does not modify the MIT license terms.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Use GitHub issues for reproducible bugs and feature proposals. Security concerns should follow [SECURITY.md](SECURITY.md).

## License

The Python implementation is licensed under the [MIT License](LICENSE). Acoustics Toolbox is an optional external dependency and is not bundled. A small set of compatibility fixtures derived from AT is covered separately; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
