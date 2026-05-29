# Changelog

This file tracks released tags and upcoming release work for `hrtfpykit`.

Historical baseline tags keep their original names. Future release tags should
use the `v0.x.y` format.

## [v0.1.1.dev0] - Unreleased

### Added

- 

### Changed

- 

### Fixed

- 

## [v0.1.0] - 2026-05-29

### Added

- Added the "Starting with hrtfpykit.plots" tutorial notebook covering direct
  comparison plots, figure display controls, amplitude and magnitude
  comparisons, ITD and ILD cue curves, spatial difference maps, LSD plots, and
  spherical harmonic reconstruction diagnostics.
- Added a spherical harmonic workflow section to the "Starting with hrtfpykit.hrtf"
  tutorial, covering `sht()`, `sht_inverse()`, `sht_error()`,
  SH coefficient shapes, basis shapes, and reconstruction from magnitude values.
- Added ARI HRTF SHA-256 checksums for the `hrtf b`, `hrtf c`, and `hrtf d`
  SOFA files, excluding the duplicate legacy files.
- Added the `ARI` dataset class with official HRTF SOFA resource downloads.
- Added ARI anthropometry and metadata CSV resource downloads backed by the
  `ari_anthropometry_and_metadata` repository.
- Added ARI anthropometry ear selection so `AnthropometrySpec(ear="left")` and
  `AnthropometrySpec(ear="right")` filter ear-specific fields by `L_` and `R_`
  prefixes while preserving shared `x*` measurements.
- Added ARI dataset tests for configuration, download planning, checksum
  failures, spec workflow immutability, CSV resource plans, and anthropometry
  ear selection.
- Added SONICOM subjects `P0401` through `P0405` with measured HRTF and
  available scanned mesh checksums.
- Added the "Starting with hrtfpykit.datasets" tutorial notebook covering
  SONICOM resource downloads, map-style dataset construction, sample metadata,
  deterministic splits, dataset-level HRTF preprocessing, compatible dataset
  concatenation, PyTorch batching with `collate_samples()`, and a small HRTF
  autoencoder workflow.
- Added the "Mastering hrtfpykit.datasets Specs" tutorial notebook covering
  HUTUBS spec workflows, acoustic specs, custom image resources, subject
  resource matching, HRTF transforms, `index_by` sample construction,
  `grouped_by` resource matching, context encodings, and PIL or torchvision
  image transforms.

### Changed

- Moved the importable package into a `src/hrtfpykit` layout and updated
  packaging, CI, type checking, and documentation import paths.
- Restructured the plotting tutorial into smaller runnable sections and code
  cells so each plot family can be inspected and modified independently.
- Reordered the plotting tutorial imports so shared plotting functions are
  imported once before the example that explains figure controls.
- Changed dataset resource summaries to report subject coverage first for every
  resource family, followed by physical file counts when available.
- Improved dataset spec path errors for anthropometry, metadata, mesh, image,
  and video resources.
- Improved missing anthropometry, metadata, image, and video resource errors so
  they show the selected path, dataset root, configured source, and download
  resource hint when available.
- Changed HRTF download checksum lookup to support flat checksum maps keyed by
  file name, in addition to the existing grouped type/version/sample rate maps.
- Changed dataset HRTF and mesh resource scanning and download planning to
  support subject specific path maps for datasets whose official filenames do
  not share one template.
- Reordered HUTUBS and SONICOM CI dataset checks so both files follow the same
  baseline test sequence before dataset specific resource checks.
- Updated the Quick Start examples so they download the SONICOM example HRTFs
  before loading SOFA, HRTF, and comparison-plot examples.
- Updated SONICOM dataset documentation with a dated implementation-status note
  that distinguishes the active upstream dataset from the subject range
  currently supported by hrtfpykit.

### Fixed

- Fixed anthropometry and metadata resource summaries so checked, available,
  and missing counts use the active dataset subject scope and treat discarded
  empty, NaN, or infinite rows as missing for those subjects.
- Fixed dataset download planning so one official resource missing from the
  checksum map, such as the HUTUBS `pp7_3DheadMesh.ply` mesh, is skipped
  instead of aborting the full download plan.
- Fixed the SOFA tutorial reload example so it explicitly disables convention
  checking when confirming the saved tutorial file, matching the surrounding
  explanation.

## [v0.0.8] - 2026-05-21

### Added

- Added a GitHub Actions CD workflow that publishes tagged GitHub Releases to
  PyPI through Trusted Publishing.

### Changed

- 

### Fixed

- 

## [v0.0.7] - 2026-05-21

### Added

- Added integration coverage for `collate_samples()` tensor dtype behavior,
  including floating acoustic values, numeric feature dictionaries, integer
  indices, and path resources.
- Added dataset download support for resource-specific base URLs so future
  dataset configs can fetch HRTF, mesh, anthropometry, or metadata resources
  from different servers while preserving the same resource-relative paths.
- Added `BaseDataset.name` as a public read-only accessor for the active dataset
  configuration name.
- Added `meta` to dataset samples with dataset name, subject ID, and active row
  context for mixed-dataset provenance.
- Added `azimuth_angles` and `elevation_angles` filters to `HRTF.select()` for
  source-grid selection by nearest available spherical angles.
- Added the `Starting with hrtfpykit.sofa` tutorial notebook covering SOFA file
  download, security and convention checks, inspection, editing, structured
  copies, save, and reload workflows.
- Added the `Starting with hrtfpykit.hrtf` tutorial notebook covering
  `load_hrtf()`, HRTF composed interfaces, source selection, transforms,
  metrics, plots, SOFA synchronization, save, and reload workflows.

### Changed

- Changed `HRTF.select()` IR cropping so the recomputed TF uses the cropped IR
  length as the FFT length. Higher frequency resolution can still be requested
  afterward with `HRTF.transform.modify_fft_length()`.
- Updated `collate_samples()` as the PyTorch batching path for hrtfpykit
  datasets: homogeneous numeric arrays, tensors, scalars, and numeric feature
  dictionaries are returned as tensors, floating values are converted to
  `torch.float32`, integer and boolean values keep their natural tensor dtypes,
  and paths, strings, ragged values, mixed `None` values, and heterogeneous
  resources remain Python lists.

### Fixed

- 

## [v0.0.6] - 2026-05-15

### Added

- Added a GitHub Actions CI workflow that runs linting, type checking, core
  SOFA/HRTF/plots/integration tests, and non-skipped HUTUBS and SONICOM
  configuration/download-plan tests.
- Added `tests/pp1_HRIRs_measured.sofa` as the small committed SOFA fixture for
  CI-friendly SOFA, HRTF, plotting, and integration tests.
- Added `tests/test_integration.py` coverage for the full SOFA-backed workflow:
  HRTF load/select/transform, metrics, plotting, SOFA save/reload, convention
  conversion, and dataset spec resolution across acoustic and non-acoustic spec
  families.
- Added a detailed Tests documentation page covering CI smoke tests, local deep
  dataset tests, SOFA fixture discovery, plot options, dataset download flags,
  pytest options, and troubleshooting.
- Added Furo sidebar project links with local GitHub and PyPI SVG assets.
- Added a `Changelog` project URL so the PyPI project page links back to
  `CHANGELOG.md`.
- Added `verify_checksum` to HUTUBS and SONICOM downloads. The default remains
  checksum verification enabled, while `verify_checksum=False` skips official
  SHA-256 checks and still keeps file existence, non-empty, and archive
  integrity checks.

### Changed

- Removed `SOFA.create_dummy()` and the unused SOFA helper functions that only
  supported dummy SOFA object construction.
- Added project metadata for `ruff` and `mypy` quality checks.
- Cleaned the package typing baseline so `mypy hrtfpykit` can run as a required
  CI check.
- Reworked SOFA, HRTF, and plot tests to auto-detect the committed test SOFA
  fixture while still allowing explicit `--sofa-path` and
  `--compare-sofa-paths` overrides.
- Updated HUTUBS and SONICOM dataset tests so `--subjects` controls the
  subject-scoped download checks and the opt-in download tests run before later
  dataset assertions in the same pytest run.
- Normalized missing DC handling for `SimpleFreeFieldHRTF` files so loaded TF
  data, `reset()`, `update_sofa()`, and `save()` share the same inserted 0 Hz
  bin.
- Preserved Mesh2HRTF compatible reconstruction options on `HRTF` objects across
  `clone()`, `reset()`, transforms, and directivity workflows.
- Reworked the documentation API reference around direct public entry points:
  `hrtfpykit.sofa`, `hrtfpykit.hrtf`, `hrtfpykit.plots`, and
  `hrtfpykit.datasets`, with separate pages for SOFA validation helpers, HRTF
  metrics, spherical harmonic utilities, plot functions, dataset specs, and
  `HRTFTransform`.
- Updated the Quick Start to describe the integrated SOFA, HRTF, plots, and
  datasets workflow, including flatter plot examples and dataset spec links.

### Fixed

- Fixed `SimpleFreeFieldHRTF` files without DC so `HRTF.TF` no longer exposes
  one fewer frequency bin than the reconstructed IR implies.
- Fixed `HRTF.reset()` for Mesh2HRTF compatible loads so it does not silently
  reload with the normal inverse reconstruction path.
- Fixed generated SOFA metadata paths to avoid relying on the removed
  `hrtfpykit.sofa.__version__` attribute.

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
