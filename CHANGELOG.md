# Changelog

This file tracks released tags and upcoming release work for `hrtfpykit`.

Historical baseline tags keep their original names. Future release tags should
use the `v0.0.x` format.

## [v0.0.5] - 2026-05-13

### Added

- Added the Sphinx and Furo documentation site structure with overview, quick
  start, SOFA, HRTF, plots, datasets, tutorials, and tests pages.
- Added quick start plot images and documentation assets.
- Added `CONTRIBUTING.md` with issue reporting, pull request, HRTF, SOFA,
  dataset, documentation, and license guidance.
- Added `collate_samples` documentation and PyTorch batching support for dataset
  samples.
- Added integration tests and updated plotting, dataset, HRTF, and SOFA test
  coverage.

### Changed

- Reworked the README into a public project front page with badges, project
  pitch, installation, quick start, citation, and license sections.
- Updated package metadata for PyPI, including static version metadata, SPDX
  license metadata, Python classifiers, audience classifiers, topic classifiers,
  and project keywords.
- Removed `wheel` from `build-system.requires`; modern setuptools is now the
  only build backend requirement.
- Moved version ownership to `pyproject.toml`; `hrtfpykit.__version__` now reads
  installed package metadata.
- Reorganized public modules so coordinate, DSP, metric, plane, and spherical
  harmonic helpers are available from stable package level modules.
- Reworked the docs configuration to read the package version from
  `pyproject.toml`.

### Fixed

- Fixed documentation builds so Sphinx can build with warnings treated as
  errors.
- Fixed package wheel builds with isolated build environments.
- Fixed plot comparison documentation and quick start image sizing.

## [baseline-v0.0.4] - 2026-05-07

### Added

- Added the datasets API with dataset configuration, resource loading,
  checksums, subject splits, resource summaries, specs, spec registry, spec
  workflow, dataset transforms, and value extraction.
- Added HUTUBS and SONICOM dataset integrations.
- Added dataset tests for HUTUBS and SONICOM.
- Added acoustic context handling for dataset samples.
- Added HRTF dataset specs for HRTF values, ITD, ILD, spherical harmonics,
  meshes, metadata, anthropometry, images, and videos.
- Added comparison plots for amplitude, magnitude, ITD, ILD, ITD difference,
  ILD difference, full-grid LSD, and LSD plane views.
- Added spherical harmonic utilities and spherical harmonic plotting support.
- Added HRTF helper utilities and expanded HRTF object methods.
- Added project logo and image assets.
- Added GPL license metadata and documentation updates for HRTF and SOFA APIs.

### Changed

- Expanded the HRTF API with more transforms, metrics, directivity handling,
  and Mesh2HRTF TF compatibility.
- Renamed the HRTF plotting module from `hrtf_plots.py` to `hrtf.py`.
- Expanded plots package exports and plotting labels, legends, and titles.
- Updated package metadata and documentation for the dataset focused baseline.
- Reworked dataset internals from the earlier placeholder module into a full
  package.

### Fixed

- Fixed ITD calculation behavior.
- Fixed LSD method behavior and documentation.
- Fixed non scalar `add_itd()` behavior.
- Fixed SONICOM checksum handling.
- Fixed dataset construction, cleanup, and test issues found while stabilizing
  HUTUBS and SONICOM workflows.

## [baseline-v0.0.3] - 2026-04-12

### Added

- Added the HRTF API with HRTF objects, IR and TF domain abstractions, source
  handling, transforms, metrics, directivity tools, and HRTF selection.
- Added IR to TF and TF to IR conversion workflows.
- Added DSP utilities for filtering, padding, normalization, resampling,
  convolution, DTF, CTF, ITD, and ILD workflows.
- Added coordinate and plane utilities for source position queries and
  horizontal, median, and frontal plane selection.
- Added the plots package with axes, labels, layouts, legends, figure helpers,
  polar helpers, plot titles, and HRTF plotting methods.
- Added plots for magnitude, amplitude, amplitude and magnitude, spectrum
  planes, source grids, ITD, ILD, and spatial planes.
- Added dummy SimpleFreeFieldHRIR and SimpleFreeFieldHRTF documentation
  notebooks.
- Added tests for coordinates, domains, planes, plots, SOFA data logic, and
  convention management.

### Changed

- Refactored legacy HRTF modules into the `hrtfpykit.hrtf` package.
- Refactored SOFA implementation from `sofa.core` into `sofa.sofa` and
  `sofa_helpers`.
- Removed legacy `source.py`, `time_domain.py`, `frequency_domain.py`,
  `transforms.py`, `visualization.py`, and older utility modules.
- Reworked package structure toward clearer SOFA, HRTF, and plots boundaries.
- Updated public imports and module organization for the plots baseline.

### Fixed

- Fixed HRTF selection behavior.
- Fixed SOFA clone behavior.
- Fixed multi-plot layout behavior.
- Fixed horizontal plane selection behavior.
- Fixed filter, padding, and IR duration handling.
- Fixed coordinate handling and named position queries used by plots.

## [baseline-v0.0.2] - 2026-03-19

### Added

- Added SOFA convention manager support.
- Added expanded SOFA convention specifications.
- Added SOFA security checks.
- Added SOFA CRUD tests, convention manager tests, and security tests.
- Added early HRTF abstraction work with HRTF, time domain, frequency domain,
  and source modules.
- Added SOFA API documentation for the stable SOFA baseline.

### Changed

- Made the SOFA API convention agnostic.
- Expanded SOFA specification handling beyond the first blueprint.
- Cleaned the codebase around the SOFA API.
- Updated SOFA data, check, wraps, and utility behavior.

### Fixed

- Fixed SOFA convention checks.
- Fixed SOFA file security validation behavior.
- Fixed SOFA data logic issues found during CRUD and convention tests.

## [baseline-v0.0.1] - 2026-03-11

### Added

- Added the first complete SOFA API package.
- Added SOFA data wrappers for dimensions, variables, global attributes, and
  variable attributes.
- Added SOFA convention definitions.
- Added SOFA file checking utilities.
- Added SOFA CRUD methods for dimensions, variables, attributes, and file level
  operations.
- Added tests for SOFA data logic and SOFA checks.
- Added initial project metadata in `pyproject.toml`.
- Added early documentation pages for development and project overview.

### Changed

- Reorganized the project into a package structure under `hrtfpykit`.
- Moved early standalone HRTF loading and visualization scripts into package
  modules.
- Stopped tracking generated cache files and large local SOFA files.

### Fixed

- Fixed early SOFA data logic around dimensions and variables.
- Fixed SOFA CRUD behavior during the first SOFA API stabilization.

## [baseline-v0.0.0] - 2026-02-25

### Added

- Added the initial project baseline.
- Added early HRTF loading and visualization scripts.
- Added the first HRTF class prototype.
- Added initial sample SOFA files for local development.
- Added the first repository structure and package idea.

### Changed

- Cleaned the initial repository structure and `.gitignore`.

### Fixed

- No fixes recorded for this baseline tag.
