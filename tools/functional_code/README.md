# Functional Code

This directory stores standalone functional/reference scripts that support
acoustic-model workflows but are not part of the core `pyacoustics` package.

Typical examples include time-series post-processing scripts such as AT
Matlab `delayandsum.m`, conversion utilities, analysis prototypes, and
workflow helpers that may later be ported into Python modules.

## Delay-and-sum

`delay_and_sum.py` is a command-line wrapper around
`pyacoustics.postprocessing.delay_and_sum`, the Python refactor of AT Matlab
`delayandsum.m`.

Example:

```bash
PYTHONPATH=. python tools/functional_code/delay_and_sum.py MunkB.arr source.npy \
  --sample-rate 1000 --window-duration 10 --reduction-velocity 1520
```
