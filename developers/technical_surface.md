# Technical surface

`hrtfpykit` exposes four public technical layers. The top-level `src/hrtfpykit/__init__.py` exports only `__version__`; users and internal modules normally enter through one of the four subpackages.

```text
hrtfpykit
├── sofa       -> SOFA file abstraction and validation
├── hrtf       -> acoustic object, domains, source grid, transforms, metrics, SH
├── plots      -> Visualization layer built using Matplotlib
└── datasets   -> public dataset pipelines, specs, downloads, batching/loss utilities
```

## Public layer exports

### `hrtfpykit.sofa`

Defined by `src/hrtfpykit/sofa/__init__.py`:

- `hrtfpykit.sofa.sofa.SOFA`
- `hrtfpykit.sofa.sofa.load_sofa()`
- `hrtfpykit.sofa.check.check_sofa_against_conventions()`
- `hrtfpykit.sofa.check.check_sofa_security()`

The technical surface is file oriented. `hrtfpykit.sofa.sofa.load_sofa()` opens a `.sofa` file, creates a `hrtfpykit.sofa.sofa.SOFA` object, and delegates netCDF4 opening to `hrtfpykit.sofa.sofa_helpers.open_sofa()` in `src/hrtfpykit/sofa/sofa_helpers.py`. `hrtfpykit.sofa.sofa.SOFA` then exposes wrappers through `hrtfpykit.sofa.sofa.SOFA.Dimensions`, `hrtfpykit.sofa.sofa.SOFA.GlobalAttributes`, `hrtfpykit.sofa.sofa.SOFA.Variables`, and `hrtfpykit.sofa.sofa.SOFA.VariableAttributes`.

### `hrtfpykit.hrtf`

Defined by `src/hrtfpykit/hrtf/__init__.py`:

- object and loader: `hrtfpykit.hrtf.hrtf.HRTF`, `hrtfpykit.hrtf.hrtf.load_hrtf()`
- metrics: `hrtfpykit.utils.metrics.itd()`, `hrtfpykit.utils.metrics.ild()`, `hrtfpykit.utils.metrics.rms()`, `hrtfpykit.utils.metrics.itd_difference()`, `hrtfpykit.utils.metrics.ild_difference()`, `hrtfpykit.utils.metrics.hrtf_difference()`
- spherical harmonics: `hrtfpykit.utils.sh.SH`, `hrtfpykit.utils.sh.sht()`, `hrtfpykit.utils.sh.sht_inverse()`, `hrtfpykit.utils.sh.sht_error()`
- directivity helper: `hrtfpykit.utils.directivity.hrtf_from_dtf_and_ctf()`

The technical surface is acoustic-object oriented. `hrtfpykit.hrtf.hrtf.load_hrtf()` reads SOFA data through `hrtfpykit.sofa.sofa.load_sofa()`, validates `SOFAConventions`, constructs an `hrtfpykit.hrtf.hrtf.HRTF`, and populates the `hrtfpykit.hrtf.domain.IR`, `hrtfpykit.hrtf.domain.TF`, and `hrtfpykit.hrtf.sources.Sources` views.

### `hrtfpykit.plots`

Defined by `src/hrtfpykit/plots/__init__.py`:

- single-HRTF plots: `hrtfpykit.plots.hrtf.plot_magnitude()`, `hrtfpykit.plots.hrtf.plot_amplitude()`, `hrtfpykit.plots.hrtf.plot_etc()`, `hrtfpykit.plots.hrtf.plot_etc_plane()`, `hrtfpykit.plots.hrtf.plot_spectrum_plane()`, `hrtfpykit.plots.hrtf.plot_elevation_spectrum()`, `hrtfpykit.plots.hrtf.plot_itd()`, `hrtfpykit.plots.hrtf.plot_absolute_itd()`, `hrtfpykit.plots.hrtf.plot_ild_fd()`, `hrtfpykit.plots.hrtf.plot_ild()`, `hrtfpykit.plots.hrtf.plot_absolute_ild()`, `hrtfpykit.plots.hrtf.plot_source_grid()`, `hrtfpykit.plots.hrtf.plot_plane_grid()`
- comparison plots: `hrtfpykit.plots.compare.compare_magnitude()`, `hrtfpykit.plots.compare.compare_amplitude()`, `hrtfpykit.plots.compare.compare_absolute_itd()`, `hrtfpykit.plots.compare.compare_absolute_ild()`, `hrtfpykit.plots.compare.compare_itd()`, `hrtfpykit.plots.compare.compare_ild()`, `hrtfpykit.plots.compare.compare_itd_difference()`, `hrtfpykit.plots.compare.compare_ild_difference()`, `hrtfpykit.plots.compare.compare_hrtf_difference()`
- spherical-harmonic diagnostics: `hrtfpykit.plots.sh.sht_reconstruction_comparison()`, `hrtfpykit.plots.sh.sht_reconstruction_error()`

The technical surface is visualization oriented. High-level functions consume `hrtfpykit.hrtf.hrtf.HRTF` objects and metric outputs, then route rendering through `hrtfpykit.plots.figure.Figure`, `hrtfpykit.plots.layouts.Layout`, `hrtfpykit.plots.types.TwoDimension`, `hrtfpykit.plots.types.Heatmap`, `hrtfpykit.plots.types.ThreeDimension`, axis helpers, labels, titles, and legends.

### `hrtfpykit.datasets`

Defined by `src/hrtfpykit/datasets/__init__.py`:

- dataset classes: `hrtfpykit.datasets.ari.ARI`, `hrtfpykit.datasets.hutubs.HUTUBS`, `hrtfpykit.datasets.sonicom.SONICOM`
- specs: `hrtfpykit.datasets.specs.HRTFSpec`, `hrtfpykit.datasets.specs.ITDSpec`, `hrtfpykit.datasets.specs.ILDSpec`, `hrtfpykit.datasets.specs.SHSpec`, `hrtfpykit.datasets.specs.MeshSpec`, `hrtfpykit.datasets.specs.AnthropometrySpec`, `hrtfpykit.datasets.specs.MetadataSpec`, `hrtfpykit.datasets.specs.ImageSpec`, `hrtfpykit.datasets.specs.VideoSpec`
- transforms: `hrtfpykit.datasets.transforms.HRTFTransform`

The optional PyTorch integration lives under the dataset layer as `hrtfpykit.datasets.torch`, with public functions `hrtfpykit.datasets.torch.collate_samples()` and `hrtfpykit.datasets.torch.hrtf_loss()` in `src/hrtfpykit/datasets/torch.py`. PyTorch is intentionally not imported by the core dataset package initializer.

## Cross-layer integration surface

The normal integration path is:

```text
SOFA file
  ↓ hrtfpykit.sofa.sofa.load_sofa()
hrtfpykit.sofa.sofa.SOFA
  ↓ hrtfpykit.hrtf.hrtf.load_hrtf()
hrtfpykit.hrtf.hrtf.HRTF
  ├── hrtfpykit.hrtf.domain.IR / hrtfpykit.hrtf.domain.TF
  ├── hrtfpykit.hrtf.sources.Sources / hrtfpykit.hrtf.transforms.Transform
  ├── hrtfpykit.utils.metrics / hrtfpykit.utils.sh
  ├── hrtfpykit.plots
  └── hrtfpykit.datasets spec extraction
```

`hrtfpykit.datasets` can load `hrtfpykit.hrtf.hrtf.HRTF` objects internally through `hrtfpykit.datasets.load.load_dataset_hrtf()` and can call HRTF transforms through `hrtfpykit.datasets.transforms.HRTFTransform`. `hrtfpykit.plots` should remain a consumer of `hrtfpykit.hrtf.hrtf.HRTF` state. `hrtfpykit.hrtf` should not import plotting code.
