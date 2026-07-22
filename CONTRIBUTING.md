# Contributing

Thank you for improving `acoustics-agent`. Contributions should keep native simulation usable without proprietary or external runtimes and should preserve reproducible numerical behavior.

## Development Setup

```bash
git clone https://github.com/weifengsd/acoustics-agent.git
cd acoustics-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Before Opening a Pull Request

1. Add focused tests for changed behavior.
2. Run `pytest tests/unit -q`.
3. Run relevant integration or parity cases for solver changes.
4. Update public documentation and `CHANGELOG.md` for user-visible changes.
5. Confirm that generated reports, solver outputs, plots, caches, and local paths are not staged.

AT-dependent checks require a separately installed Acoustics Toolbox:

```bash
export AT_BIN_PATH=/path/to/AcousticsToolbox/bin
acoustics-at-benchmark --suite all --at-bin-path "$AT_BIN_PATH" --strict-at
```

## Pull Requests

- Keep changes scoped and explain the acoustic or compatibility behavior being changed.
- Include the exact commands used for verification.
- Report numerical tolerances and sample counts when changing parity behavior.
- Do not weaken thresholds solely to make a regression pass.
- Do not commit Acoustics Toolbox binaries, Matlab installations, generated benchmark matrices, or scratch outputs.

## Bug Reports

Include the configuration, solver mode, Python version, platform, expected behavior, actual behavior, and the smallest reproducible input. Attach generated output only when it is needed to diagnose the issue.

## Licensing

By contributing original code, you agree that it may be distributed under the repository's MIT License. Do not copy third-party implementation code into the MIT-licensed package. Third-party fixtures must retain their original license and be documented in `THIRD_PARTY_NOTICES.md`.
