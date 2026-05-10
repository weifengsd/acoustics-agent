# CLAUDE.md - Development Guidelines for pyacoustics

This document provides instructions for AI assistants and developers working on the `pyacoustics` codebase.

## 🛠 Build & Test Commands
- **Run All Tests**: `pytest tests/`
- **Run Specific Test**: `pytest tests/Munk/test_munk.py`
- **Benchmark Performance**: `python3 scripts/benchmark.py`
- **Install in Dev Mode**: `pip install -e .`

## 🎨 Code Style & Standards
- **Naming**: Use `snake_case` for functions/variables and `PascalCase` for classes.
- **Typing**: Use Python 3.10+ type hints.
- **Physics Core**: Most physics solvers in `pyacoustics/solvers/` use **Numba** (`@njit`).
  - Avoid creating objects (classes/dicts) inside jitted functions.
  - Use NumPy arrays for all numerical data.
  - Ensure all types are inferable by Numba (avoid list of mixed types).
- **Configuration**: Always use `dataclasses` in `pyacoustics/schema.py`. Avoid external dependencies like Pydantic for core logic.
- **Formatting**: Adhere to PEP 8 standards.

## 🧱 Architecture Overview
- **Engine**: `pyacoustics/simulation.py` is the main entry point.
- **Config**: `pyacoustics/config.py` handles YAML to dataclass mapping.
- **Skills**: AI-facing tools are in `skills/`. These should remain high-level and easy for an LLM to call.
- **Testing**: Every new environment or feature must have a corresponding test directory in `tests/`.

## 🤝 Contribution Workflow
1. Use `skills/make_env.py` to generate new test scenarios.
2. Verify performance using `scripts/benchmark.py`.
3. Ensure 100% test pass rate before pushing to `main`.
