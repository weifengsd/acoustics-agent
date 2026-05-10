# acoustics-agent: AI-Native Underwater Acoustic Simulation Framework

`acoustics-agent` is a modern framework for underwater acoustics, designed for both high-performance physical simulation and seamless AI orchestration. At its core is **pyacoustics**, a high-performance, pure-Python modernization of the legacy Acoustics Toolbox.

## 🚀 Key Features
- **High Performance**: Numba-accelerated ray tracing (**pyacoustics** engine) reaching 1000+ rays/sec.
- **AI-Native**: Modular "Skills" library designed for seamless LLM orchestration.
- **Zero-Dependency**: Runs out-of-the-box on standard Anaconda/Miniconda environments.
- **Modern Workflow**: YAML-based configuration, eliminating proprietary legacy formats.
- **Legacy Compatibility**: Native support for calling legacy **Acoustics Toolbox (AT)** binaries for validation and comparison.
- **Comprehensive Testing**: 27+ integration benchmarks covering Munk, Pekeris, RAP, and more.

## 📁 Project Structure
- `pyacoustics/`: Core physics engine and solvers (MIT Licensed).
- `skills/`: AI-orchestration layer (make_env, run_sim, plot_sim).
- `tests/`: Extensive test suite and environmental benchmarks.
- `docs/`: Detailed design and usage documentation.

## 🛠 Installation
The project is designed to be zero-dependency on top of the standard scientific stack.
```bash
# Recommended: Standard Anaconda environment
conda install numpy scipy numba pyyaml matplotlib
```
Or via pip:
```bash
pip install -e .
```

### Optional External Prerequisite
For full feature parity with legacy Bellhop/Kraken, you may optionally configure the **Acoustics Toolbox (AT)**.
1. Download or compile AT binaries from the [official source](http://oalib.hlsresearch.com/AcousticsToolbox/).
2. Set the `AT_BIN_PATH` environment variable pointing to your `at/bin` directory.
   ```bash
   export AT_BIN_PATH=/path/to/at/bin
   ```

## 📖 Quick Start
You can run a simulation using the provided Skills or the high-level Simulation API:

```python
from pyacoustics.simulation import Simulation

# Run from a YAML config
sim = Simulation("tests/Munk/munk.yaml")
sim.run()

# Plot results
sim.plot_tl("output.png")
```

## 🧪 Testing
Run the full test suite (27+ tests):
```bash
pytest tests/
```
