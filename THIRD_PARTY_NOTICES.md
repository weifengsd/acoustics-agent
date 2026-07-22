# Third-Party Notices

## Acoustics Toolbox

Acoustics Toolbox (AT) is a separate project distributed under the GNU General Public License version 3. It is not a runtime dependency of the native Python solvers and its binaries or implementation source are not bundled in this repository.

This repository contains a limited number of AT-compatible numerical fixtures under:

- `tests/3DAtlantic/`
- `tests/Bellhop3DTests/KoreanSeas/`
- `tests/Bellhop3DTests/Taiwan/`
- selected reference files under `tests/Munk/`

Files copied unchanged from the AT sample corpus remain covered by GPL-3.0 and are used only for compatibility and regression validation. The applicable license text is stored at `tests/third_party/acoustics-toolbox/LICENSE`.

Project code under `pyacoustics/` is independently implemented and covered by the repository's MIT License unless a file states otherwise.
