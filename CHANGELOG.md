# Changelog

This file tracks released tags and upcoming release work for `hrtfpykit`.

Historical baseline tags keep their original names. Future release tags should
use the `vx.y.z` format.

## [v0.2.0] - 2026-06-13

### Added

- Added configurable local resource path patterns for HRTF, mesh,
  anthropometry, and metadata resources so dataset scanners can discover files
  in canonical flat layouts, subject folders, and semantic folders such as
  `subject_id/hrtf/...`, `subject_id/mesh/...`, `metadata/...`,
  `metadata_and_readme/...`, `anthropometry/...`, and `anthro/...`.
- Added concrete server-specific dataset downloader classes for SOFAcoustics,
  Imperial, TU Berlin DepositOnce, and the SONICOM ecosystem, reusing shared
  download mechanics while keeping server-specific planning logic in the
  concrete downloader classes.
- Added `DownloadServerConfig` and per-dataset `download_servers` mappings so
  datasets can expose multiple official download sources with separate URLs,
  checksums, database endpoints, archives, supported filters, and server-level
  download exclusions.
- Added TU Berlin DepositOnce SHA-256 checksums for the official HUTUBS archive
  files.
- Added shared downloader validation for HRTF and mesh variant selectors so all
  download servers report unsupported types, sample rates, and versions with the
  same error format before server-specific planning begins.
- Added ARI NH HRTF variant selection. `dataset_hrtf_variant="NH"` scans the
  full configured NH collection, while `{"type": "NH", "version": "b"}`
  (or `c`/`d`) selects one ARI filename group. Downloads support the same NH
  selector plus `download_hrtf_variant="all"` for every configured ARI HRTF
  family.
- Added `coordinate_system`, `plane`, and `plane_angle` selection to
  `Sources.get_positions`, so users can request spherical, Cartesian, or
  lateral-polar positions and filter source grids by horizontal, median, or
  frontal planes directly at read time.
- Added `frequency_bands` to `hrtf_difference(metric="lsd")`, matching
  `HRTFSpec` frequency-band selection semantics and remaining mutually
  exclusive with explicit `frequencies` queries.
- Added `plot_etc` for position-based energy time curve plots from HRIR data.
- Added `plot_etc_plane` for plane-based energy time curve heatmaps over
  horizontal and median HRTF source planes.
- Added public HRTF metric documentation pages for RMS, ITD, ILD, ITD
  difference, ILD difference.
- Added `subject_ids` and `download_subject_ids` to dataset constructors so
  users can include a small subject scope directly, then apply
  `exclude_subject_ids` or `download_exclude_subject_ids` on top of that scope.
- Added dataset HRTF cache controls with `preload_hrtfs` and
  `clear_cache`, so selected subject HRTFs can be loaded eagerly for training
  workflows or released explicitly when cached SOFA-backed objects are no longer
  needed.
- Added dataset HRTF loading controls `check_sofa_against_conventions` and
  `sofa_open` to `BaseDataset`, `ARI`, `HUTUBS`, and `SONICOM`.
- Added `SOFA.is_open()`, `SOFA.close()`, and `SOFA.open()` for explicit
  management of the backing netCDF4 dataset handle.
- Added `load_hrtf(..., sofa_open=False)` for loading HRTF data into memory and
  closing the file-backed SOFA dataset handle after the HRTF object has been
  initialized.
- Added `Transform.add_ild()` and `Transform.delete_ild()` for TF-domain
  ILD editing, with matching dataset wrappers `HRTFTransform.add_ild()` and
  `HRTFTransform.delete_ild()`. `add_ild()` accepts scalar dB values, one value
  per source, or one value per source and frequency bin; `delete_ild()` removes
  frequency-dependent ILD with symmetric left/right gain correction.
- Added `Transform.apply_crop()` and `HRTFTransform.apply_crop()` for
  sample-indexed HRIR interval removal with trailing zero padding and optional
  end tapering before TF resynchronization. Omitting `start_sample` starts the
  crop at sample 0.
- Added `hrtf_difference()` for HRTF comparison with `rmse`, `mae`,
  `nrmse`, and `lsd` metrics. `rmse` and `mae` return linear HRIR
  amplitude error, `nrmse` normalizes by the reference HRIR energy and
  returns dB, and `lsd` compares TF magnitudes in dB.
- Added `compare_hrtf_difference()` for source-map visualization of
  `hrtf_difference()` metrics with scatter and heatmap renderers.
- Added `hrtfpykit.datasets.torch.hrtf_loss()` for PyTorch training losses
  based on RMSE, MAE, and LSD over tensor predictions while keeping gradients
  connected to the model output.
- Added `hrtfpykit.datasets.torch` documentation as the PyTorch integration
  module for dataset workflows, with separate docs pages for `collate_samples()`
  and `hrtf_loss()`.
- Added the "Mastering HRTF transformation and visualization" tutorial
  notebook for transform, metric, and plot workflows based on SONICOM HRTFs.
- Added ear-selective support to `Transform.apply_window()`,
  `Transform.apply_crop()`, `Transform.apply_padding()`,
  `Transform.apply_fir_filter()`, `Transform.apply_iir_filter()`, and
  `Transform.apply_gain()`, with matching `HRTFTransform` wrappers.
  `apply_padding()` also supports
  `preserve_length=True` for start padding that delays the selected ear without
  increasing the HRIR sample count.

### Changed

- Changed dataset downloads to check the same local resource path candidates
  used by the dataset scanners before downloading. Existing valid local files
  found through `local_path_patterns` are now verified and reused instead of
  downloaded again.
- Changed catalog-based download jobs to derive scanner-compatible local path
  candidates from the shared dataset resource config, so ecosystem downloads can
  reuse files already stored in semantic local layouts such as
  `subject_id/hrtf/measured/...`.
- Changed dataset constructors to separate `download_exclude_subject_ids` from
  `exclude_subject_ids`, so download filtering and dataset-building filtering
  can be controlled independently.
- Changed `ARI`, `HUTUBS`, and `SONICOM` to default `verbose=True`, so
  resource and dataset summaries are printed unless users disable verbose output.
- Changed dataset HRTF loading to use the internal `load_dataset_hrtf` wrapper
  and forward dataset-level SOFA loading options to the canonical
  `hrtfpykit.hrtf.load_hrtf` function.
- Changed standard dataset HRTF loading to default to
  `check_sofa_against_conventions=False` and `sofa_open=False`, avoiding SOFA
  convention warnings and open netCDF handles during normal dataset use while
  keeping loaded IR, TF, source positions, sample rates, and frequency bins
  available in memory.
- Changed built-in download exclusions to live on `DownloadServerConfig` as
  `download_exclude_subject_ids` instead of dataset construction config.
- Removed the legacy single-server `DownloadConfig` and `DatasetConfig.download`
  path; downloads now use `download_servers` exclusively.
- Removed collection-style `__getitem__`, `__iter__`, and `__len__` methods from
  the SOFA data wrappers; users should call `get`, `get_names`, `get_values`,
  `get_all`, and `summary` explicitly.
- Removed `modify_source_coordinate_system` from HRTF transform APIs because it
  changed only the in-memory read coordinate system and did not rewrite SOFA
  source-position values on save. Now a similar feature (get the source positions 
  in different coordinate system references like `spherical`, `cartesian` or 
  `lateral-polar`) can be accessed through :
  `hrtf.Sources.get_positions(coordinate_system="spherical")`.
- Renamed horizontal-plane cue plot parameters to `plane_angle` across ITD/ILD
  curve plots, absolute cue plots, comparison plots, and the shared polar curve
  helper.
- Renamed ITD/ILD/LSD plot APIs for explicit signed, absolute, broad-band, and
  frequency-dependent semantics: `plot_itd`,
  `plot_absolute_itd`, `plot_ild`, `plot_absolute_ild`,
  `plot_ild_fd`, `compare_absolute_itd`, `compare_absolute_ild`,
  `compare_itd`, `compare_ild`, `compare_itd_difference`,
  `compare_ild_difference`, and `compare_hrtf_difference` replace the older curve, plane, and
  comparison-plot names. Metric difference functions are now
  `itd_difference` and `ild_difference`.
- Changed plots that expose `azimuth_range_mode` to default to
  `azimuth_range_mode="0-360"`. The signed `"-180-180"` mode now reverses
  azimuth only when it is plotted on the x-axis, so listener-left appears on
  the left side of Cartesian figures while y-axis azimuth keeps normal numeric
  orientation. Plane plot docstrings now document horizontal-plane azimuth and
  median-plane lateral-polar polar angle conventions.
- Changed source-map comparison plots `compare_itd_difference`,
  `compare_ild_difference`, and `compare_hrtf_difference` to default to
  `plot_type="heatmap"`. Quickstart HRTF-difference comparison images were
  regenerated for the scatter and heatmap examples.
- Changed `itd_difference`, `ild_difference`, and
  `hrtf_difference(metric="lsd")` to compare
  `hrtf_reference` against one HRTF or a sequence of HRTFs. A single compared
  HRTF returns the natural metric shape, while several compared HRTF metric
  arrays keep a leading comparison axis.
- Changed `itd_difference`, `ild_difference`, and
  `hrtf_difference(metric="lsd")` reduction controls to
  use `reduction_axis` and `reduction_method`, with metric-specific axes such
  as `itds`, `ilds`, `differences`, `positions`, `ears`, and `global`.
  `source` and `sources` remain accepted as aliases for position reductions.
- Changed dataset spec axis naming to use `position` and `positions` as the
  canonical `index_by` and `grouped_by` spelling while accepting `source` and
  `sources` as aliases.
- Changed `ild` and `ild_difference` to return decibel values only.
- Changed `itd` into an HRTF-level metric exported from `hrtfpykit.hrtf`.
  It now reads `hrtf.IR.values` and `hrtf.IR.sample_rate` directly and supports
  `absolute=True` for unsigned ITD values.
- Changed ITD time outputs from `output="seconds"` to `output="time"`;
  `itd`, `itd_difference`, ITD plots, and ITD difference plots now report
  time values in microseconds. `itd` and `ITDSpec` now default to
  `output="time"`. `Transform.add_itd(unit="time")` and
  `HRTFTransform.add_itd(unit="time")` also interpret time values as
  microseconds.
- Changed HRIR duration helpers and time-domain plot axes to milliseconds:
  `signal_duration`, `hrtf.IR.ir_duration`, `plot_amplitude`,
  `compare_amplitude`, and ETC time axes use ms.
- Changed `ITDSpec` to expose the `absolute` ITD metric option for dataset
  feature extraction.
- Changed SOFAcoustics download server configuration to use direct per-dataset
  base URLs instead of a shared base URL plus `path_prefix`.
- Changed `hrtf_difference(metric="lsd")` reduction semantics to compute
  the natural frequency-reduced
  LSD per compared HRTF, position, and ear first, then reduce those metric values
  with `reduction_axis`.
- Changed `rms` into an HRTF-level metric exported from `hrtfpykit.hrtf`.
  It computes per-HRIR RMS over samples first, then applies
  `reduction_axis` and `reduction_method` over position and ear axes.
- Changed `sht` to require an `HRTF` object and reject standalone `TF` domain
  inputs, keeping spherical-harmonic decomposition tied to the active HRTF
  source grid and frequency metadata.
- Changed `ConventionsManager.available_conventions_specifications()` to return
  the formatted convention table and print only when `display=True`.
- Removed redundant `plot_amplitude_and_magnitude`; use `plot_amplitude` and
  `plot_magnitude` separately.
- Removed `plot_lsd_plane` .
- Removed `IR.get_itd()` and `IR.get_rms()` because ITD and RMS are exposed
  through the public `hrtfpykit.hrtf` metric API.
- Removed `IR.get_ild()` because ILD is exposed through the public
  `hrtfpykit.hrtf.ild` metric.
- Removed `hrtfpykit.utils.dsp.rms`; use `hrtfpykit.hrtf.rms` for HRTF RMS
  metrics.
- Removed the `output` parameter from `ild`, `ild_difference`, `ILDSpec`, and
  `compare_ild_difference`, and removed stale `fft_length` from `ILDSpec`; ILD
  values and ILD differences are always in dB.
- Changed single-HRTF plots from inherited `HRTF` methods to module-level
  `hrtfpykit.plots` functions such as `plot_magnitude(hrtf, ...)`,
  `plot_etc(hrtf, ...)`, and `plot_source_grid(hrtf, ...)`.
- Removed the legacy `compare_lsd()` plot API. Use
  `compare_hrtf_difference(metric="lsd")` for LSD source-map comparisons.
- Improved the `hrtf_difference()` docstring with direct descriptions of
  `IR.values`, `TF.values`, source and ear selection, reductions, and return
  units.
- Improved `IR`, `TF`, and `Sources` docstrings with task-focused examples for
  inspecting time-domain data, frequency-domain data, and source-grid queries.
- Changed checksum planning to use an explicit per-job `checksum_key` instead of
  fuzzy fallback lookup across full paths, filenames, and subject/file pairs.
  Checksum verification now targets downloader-managed resources only and remains
  independent from local scanner path layouts.
- Changed TU Berlin HUTUBS downloads to treat archives as transport files:
  archives are kept under `archives/`, usable HRTF SOFA files, mesh PLY files,
  and `AntrhopometricMeasures.csv` are normalized into the dataset root, and
  temporary extracted archive folders are removed.
- Changed TU Berlin HUTUBS archive planning to skip archive jobs when the
  normalized usable resource files already exist in the dataset root.
- Changed HUTUBS mesh configuration to use the official 58-subject mesh scope
  instead of assuming every HUTUBS subject has a mesh file.
- Changed download summaries to support archive-derived resource counts so TU
  Berlin archive downloads can report usable HRTF, mesh, and anthropometry file
  counts instead of only zip-file counts.
- Changed dataset download failures to emit warnings with the full download
  summary after processing all planned jobs, allowing successful files from the
  same request to remain available.
- Improved download failure messages with HTTP or URL error details, and added
  a warning when a download request produces no planned files because the
  selected resource or variant is not available from the configured source. The
  warning now reports only the variant selectors relevant to the requested
  resources.
- Improved ARI, HUTUBS, and SONICOM dataset docstrings with explicit download
  server options, download resource support, dataset/download variant choices,
  subject availability, and scanner-compatible local layouts for manually copied
  resources.
- Improved SOFA wrapper API documentation with consistent method ordering and
  practical examples for dimensions, variables, global attributes, and variable
  attributes.
- Changed SOFA-backed accessors and copy/save helpers to raise clear errors
  when the backing netCDF4 dataset is missing or closed.
- Removed SOFA load, save, and edit print side effects from library methods.
- Documented in `AnthropometrySpec` and `MetadataSpec` that subjects with
  missing, empty, NaN, or infinite table fields are removed during dataset
  construction, and that users can exclude incomplete fields with
  `exclude_column` or `exclude_row` depending on table orientation.

### Fixed

- Fixed `HRTF.update_sofa()` source-subset synchronization so selected
  single-source HRTFs are saved with `M=1` instead of broadcasting the
  selected source back to the original measurement count.
- Fixed SOFA synchronization to reject ear-selected HRTFs instead of saving
  squeezed single-ear arrays as ambiguous receiver data.
- Fixed `HRTFSpec(domain="frequency", signal="ir")` validation so frequency
  specs require explicit `tf_*` signal names.
- Fixed downloader/scanner inconsistency where files found by dataset scanning,
  such as `root/metadata.csv` for SONICOM metadata, could still be downloaded
  again because the downloader only checked the official configured destination.
- Fixed catalog downloaders so existing files in scanner-accepted local layouts
  are verified and reused before transfer instead of being downloaded again.
- Fixed SONICOM ecosystem synthetic HRTF checksum lookup to use the existing
  subject-specific checksum keys such as `P0001/HRIR_SONICOM_48000.sofa`.
- Fixed TU Berlin HUTUBS repeated-download behavior where existing normalized
  root resources were ignored and only `archives/*.zip` files were considered.
- Fixed verbose dataset constructors so download summaries print when
  `verbose=True` even if files were only verified or no new files were
  downloaded.
- Fixed dataset HRTF validation to use temporary per-subject caches and close
  cached SOFA handles after validation, avoiding leftover validation HRTFs in
  the runtime dataset cache.
- Fixed SOFA clone, save, and copy flows to preserve variable storage metadata,
  including compression filters, chunking, endian settings, checksums,
  quantization options, and fill values.
- Fixed HRTF SOFA handle lifecycle during clone, update, and save workflows so
  temporary and replaced netCDF handles are closed and closed SOFA datasets fail
  explicitly.

## [v0.1.2] - 2026-06-02

### Added

-

### Changed

- Lowered the minimum supported Python version from 3.13 to 3.12 to support
  current Google Colab runtimes.
- Added Python 3.12 to the CI test matrix.

### Fixed

-

## [v0.1.1] - 2026-06-01

### Added

- Added `HRTFSpec.frequencies` for sparse nearest-bin TF selection and
  `HRTFSpec.frequency_bands` for inclusive native-grid TF band selection after
  dataset-level and spec-level HRTF transforms are applied.

### Changed

- Changed `HRTF.select()` IR cropping arguments from `start` and `end` to
  `start_sample` and `end_sample`, and removed the redundant `start_seconds`
  and `end_seconds` arguments.

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
- Reordered tutorial navigation so the transformation and visualization tutorial
  appears next to the HRTF and plots tutorials in the sidebar.
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
