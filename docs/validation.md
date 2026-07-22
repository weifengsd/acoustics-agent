# Validation Guide

Validation is layered so ordinary development does not require Acoustics Toolbox or expensive end-to-end runs.

## Fast Gate

```bash
pytest tests/unit -q
```

This gate covers schemas, numerical helpers, native solver behavior, AT file readers/writers, benchmark row construction, and CLI behavior.

## Integration Gate

```bash
pytest tests/Munk tests/iso -q
```

Use this gate after solver, configuration, or orchestration changes.

## Main AT Gate

```bash
acoustics-at-benchmark \
  --suite all \
  --at-bin-path "$AT_BIN_PATH" \
  --strict-at
```

The maintained reference scope contains:

| Suite | Rows |
| --- | ---: |
| Bellhop 2D | 101 |
| Kraken / Field | 22 |
| Combined `all` | 123 |

## Full Release Gate

```bash
acoustics-at-benchmark \
  --suite full \
  --at-bin-path "$AT_BIN_PATH" \
  --sample-root /path/to/AcousticsToolbox \
  --strict-at
```

The full reference scope adds 21 Bellhop3D rows, 3 Field3D runtime rows, 11 postprocessing rows, and 1192 AT file-format rows, for 1350 rows in total. `--sample-root` must point to a source tree containing the AT sample corpus used by the format and Field3D suites.

Some external postprocessing checks additionally require Matlab or Octave. The runner reports unavailable external runtimes as skipped or failed according to its strictness flags.

## Artifacts

CSV, JSON, and figure outputs default to `reports/at_benchmark/`. The directory is intentionally ignored. For releases, regenerate the artifacts from the tagged commit and attach them to the GitHub Release or CI run instead of committing repeated result snapshots.

Historical pass counts describe the maintained baseline. A release is validated only by a fresh run against the release candidate.
