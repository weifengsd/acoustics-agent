# Release Checklist

## Package

- [ ] The version in `pyproject.toml` matches `pyacoustics.__version__`.
- [ ] `python -m pip wheel . --no-deps --no-build-isolation` succeeds.
- [ ] The wheel contains `pyacoustics.benchmarks`, `pyacoustics.postprocessing`, and all solver packages.
- [ ] `acoustics-agent --help` and `acoustics-at-benchmark --help` succeed after installation.

## Documentation

- [ ] README quick-start commands work from a clean checkout.
- [ ] Public API or configuration changes are documented.
- [ ] `CHANGELOG.md` contains the release date and user-visible changes.
- [ ] Known limitations and external runtime requirements are explicit.

## Validation

- [ ] `pytest tests/unit -q` passes.
- [ ] Representative integration tests pass.
- [ ] The `all` AT matrix is rerun when AT binaries are available.
- [ ] The `full` matrix is rerun for a release that claims full AT parity.
- [ ] Optional Matlab/Octave checks are either run or disclosed as unavailable.

## Repository Hygiene

- [ ] No `.shd`, `.prt`, generated plots, benchmark matrices, scratch files, caches, or local paths are staged.
- [ ] Required numerical fixtures are documented in `THIRD_PARTY_NOTICES.md`.
- [ ] Generated benchmark artifacts are attached to the release or CI run, not committed.
- [ ] `git status --short` is clean after building and testing.

## Publish

- [ ] Tag the reviewed commit with the package version.
- [ ] Publish release notes and generated parity artifacts.
- [ ] Verify installation from the published artifact in a clean environment.
