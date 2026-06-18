# Architecture

The architecture is layer-oriented, with four public package layers and several internal subsystems inside those layers.

```text
hrtfpykit.sofa
    ↓
hrtfpykit.hrtf
    ├── metrics / SH / transforms / source coordinates
    ├── hrtfpykit.plots
    └── hrtfpykit.datasets
```

This diagram is directional. `hrtfpykit.hrtf` uses `hrtfpykit.sofa`; `hrtfpykit.plots` and `hrtfpykit.datasets` consume `hrtfpykit.hrtf.hrtf.HRTF` objects. The core HRTF layer must not depend on the plotting layer or dataset layer.

## Layer responsibilities

### `hrtfpykit.sofa`

Owns SOFA file handling. The central class is `hrtfpykit.sofa.sofa.SOFA` in `src/hrtfpykit/sofa/sofa.py`.

`hrtfpykit.sofa.sofa.SOFA` composes an open or closed `netCDF4.Dataset` through the `hrtfpykit.sofa.sofa.SOFA.netCDF4_dataset` attribute and tracks the associated path through `hrtfpykit.sofa.sofa.SOFA.path`. It does not inherit from `netCDF4.Dataset`; it wraps it and exposes controlled access through wrapper properties:

- `hrtfpykit.sofa.sofa.SOFA.Dimensions` -> `hrtfpykit.sofa.data._Dimensions`
- `hrtfpykit.sofa.sofa.SOFA.GlobalAttributes` -> `hrtfpykit.sofa.data._GlobalAttributes`
- `hrtfpykit.sofa.sofa.SOFA.Variables` -> `hrtfpykit.sofa.data._Variables`
- `hrtfpykit.sofa.sofa.SOFA.VariableAttributes` -> `hrtfpykit.sofa.data._VariableAttributes`

The wrapper classes live in `src/hrtfpykit/sofa/data.py` and use value wrappers from `src/hrtfpykit/sofa/wraps.py`: `hrtfpykit.sofa.wraps.DimensionsWrap`, `hrtfpykit.sofa.wraps.AttributesWrap`, and `hrtfpykit.sofa.wraps.VariablesWrap`.

### `hrtfpykit.hrtf`

Owns the acoustic abstraction. The central class is `hrtfpykit.hrtf.hrtf.HRTF` in `src/hrtfpykit/hrtf/hrtf.py`.

`hrtfpykit.hrtf.hrtf.HRTF` composes:

- `hrtfpykit.hrtf.hrtf.HRTF.Sofa`: optional backing `hrtfpykit.sofa.sofa.SOFA` object;
- `hrtfpykit.hrtf.hrtf.HRTF.IR`: cached `hrtfpykit.hrtf.domain.IR` domain view from `src/hrtfpykit/hrtf/domain.py`;
- `hrtfpykit.hrtf.hrtf.HRTF.TF`: cached `hrtfpykit.hrtf.domain.TF` domain view from `src/hrtfpykit/hrtf/domain.py`;
- `hrtfpykit.hrtf.hrtf.HRTF.Sources`: cached `hrtfpykit.hrtf.sources.Sources` object from `src/hrtfpykit/hrtf/sources.py`;
- `hrtfpykit.hrtf.hrtf.HRTF.transform`: cached `hrtfpykit.hrtf.transforms.Transform` object from `src/hrtfpykit/hrtf/transforms.py`.

These are composition relationships, not inheritance relationships. `hrtfpykit.hrtf.hrtf.HRTF` does not inherit from `hrtfpykit.sofa.sofa.SOFA`, `hrtfpykit.hrtf.domain.IR`, `hrtfpykit.hrtf.domain.TF`, `hrtfpykit.hrtf.sources.Sources`, or `hrtfpykit.hrtf.transforms.Transform`. It coordinates them.

### `hrtfpykit.plots`

Owns visualization. The public functions in `src/hrtfpykit/plots/hrtf.py`, `src/hrtfpykit/plots/compare.py`, and `src/hrtfpykit/plots/sh.py` create Matplotlib figures from hrtfpykit.hrtf.hrtf.HRTF objects, comparison metrics, and spherical-harmonic diagnostics.

The plotting layer internally composes:

- `hrtfpykit.plots.figure.Figure` from `src/hrtfpykit/plots/figure.py`;
- frozen dataclass layout objects from `src/hrtfpykit/plots/layouts.py`, such as `hrtfpykit.plots.layouts.Layout_1`, `hrtfpykit.plots.layouts.Layout_2Vertical`, `hrtfpykit.plots.layouts.Layout_2Horizontal`, and `hrtfpykit.plots.layouts.Layout_3`;
- primitive renderers from `src/hrtfpykit/plots/types.py`: `hrtfpykit.plots.types.TwoDimension`, `hrtfpykit.plots.types.Heatmap`, and `hrtfpykit.plots.types.ThreeDimension`;
- axis, label, legend, and title helpers.

`hrtfpykit.plots.figure.Figure` does not own acoustic data. It owns Matplotlib figure construction and primitive dispatch.

### `hrtfpykit.datasets`

Owns public dataset pipelines. The central base class is `hrtfpykit.datasets.base.BaseDataset` in `src/hrtfpykit/datasets/base.py`.

Concrete datasets inherit from `hrtfpykit.datasets.base.BaseDataset`:

- `hrtfpykit.datasets.ari.ARI(hrtfpykit.datasets.base.BaseDataset)` in `src/hrtfpykit/datasets/ari.py`
- `hrtfpykit.datasets.hutubs.HUTUBS(hrtfpykit.datasets.base.BaseDataset)` in `src/hrtfpykit/datasets/hutubs.py`
- `hrtfpykit.datasets.sonicom.SONICOM(hrtfpykit.datasets.base.BaseDataset)` in `src/hrtfpykit/datasets/sonicom.py`

`hrtfpykit.datasets.base.BaseDataset` composes a `hrtfpykit.datasets.state.DatasetState` object in `hrtfpykit.datasets.base.BaseDataset._state`. `hrtfpykit.datasets.build.DatasetBuilder(self).build(...)` fills that state by coordinating spec normalization, resource scanning, split planning, acoustic context creation, and row construction.

## Important inheritance relationships

### Dataset inheritance

`hrtfpykit.datasets.ari.ARI`, `hrtfpykit.datasets.hutubs.HUTUBS`, and `hrtfpykit.datasets.sonicom.SONICOM` inherit `hrtfpykit.datasets.base.BaseDataset` because they share the same indexed dataset behavior:

- `hrtfpykit.datasets.base.BaseDataset.__len__()`
- `hrtfpykit.datasets.base.BaseDataset.__getitem__()`
- `hrtfpykit.datasets.base.BaseDataset.get_subject_hrtf()`
- `hrtfpykit.datasets.base.BaseDataset.preload_hrtfs()`
- `hrtfpykit.datasets.base.BaseDataset.clear_cache()`
- summary and metadata properties

The concrete classes mainly provide dataset-specific configuration and download behavior before calling `super().__init__(...)` with a concrete config such as `hrtfpykit.datasets.config.ARIConfig`, `hrtfpykit.datasets.config.HUTUBSConfig`, or `hrtfpykit.datasets.config.SONICOMConfig`.

### Config inheritance

Dataset config dataclasses inherit from `hrtfpykit.datasets.config.DatasetConfig`:

- `hrtfpykit.datasets.config.ARIConfig(hrtfpykit.datasets.config.DatasetConfig)`
- `hrtfpykit.datasets.config.HUTUBSConfig(hrtfpykit.datasets.config.DatasetConfig)`
- `hrtfpykit.datasets.config.SONICOMConfig(hrtfpykit.datasets.config.DatasetConfig)`

They compose resource configs such as `hrtfpykit.datasets.config.HRTFConfig`, `hrtfpykit.datasets.config.MeshConfig`, `hrtfpykit.datasets.config.AnthropometryConfig`, `hrtfpykit.datasets.config.MetadataConfig`, `hrtfpykit.datasets.config.ImageConfig`, `hrtfpykit.datasets.config.VideoConfig`, and `hrtfpykit.datasets.config.DownloadServerConfig`.

### SOFA data wrapper inheritance

The SOFA accessor implementation uses internal inheritance:

- `hrtfpykit.sofa.data._Data(ABC)` in `src/hrtfpykit/sofa/data.py`
- `hrtfpykit.sofa.data._Dimensions(hrtfpykit.sofa.data._Data)`
- `hrtfpykit.sofa.data._AttributesBase(hrtfpykit.sofa.data._Data)`
- `hrtfpykit.sofa.data._GlobalAttributes(hrtfpykit.sofa.data._AttributesBase)`
- `hrtfpykit.sofa.data._VariableAttributes(hrtfpykit.sofa.data._AttributesBase)`
- `hrtfpykit.sofa.data._Variables(hrtfpykit.sofa.data._Data)`

This inheritance is implementation detail for SOFA collection wrappers. User code should go through `hrtfpykit.sofa.sofa.SOFA.Dimensions`, `hrtfpykit.sofa.sofa.SOFA.GlobalAttributes`, `hrtfpykit.sofa.sofa.SOFA.Variables`, and `hrtfpykit.sofa.sofa.SOFA.VariableAttributes`.

## Important composition relationships

### `hrtfpykit.hrtf.hrtf.HRTF` composition

```text
hrtfpykit.hrtf.hrtf.HRTF
├── Sofa: hrtfpykit.sofa.sofa.SOFA | None
├── IR: hrtfpykit.hrtf.domain.IR
├── TF: hrtfpykit.hrtf.domain.TF
├── Sources: hrtfpykit.hrtf.sources.Sources
└── transform: hrtfpykit.hrtf.transforms.Transform
```

`hrtfpykit.hrtf.hrtf.HRTF.IR`, `hrtfpykit.hrtf.hrtf.HRTF.TF`, `hrtfpykit.hrtf.hrtf.HRTF.Sources`, and `hrtfpykit.hrtf.hrtf.HRTF.transform` are `cached_property` interfaces. They are created lazily and then reused.

`hrtfpykit.hrtf.domain.IR` owns `hrtfpykit.hrtf.domain.IR.values` and `hrtfpykit.hrtf.domain.IR.sample_rate`. `hrtfpykit.hrtf.domain.TF` owns `hrtfpykit.hrtf.domain.TF.values` and `hrtfpykit.hrtf.domain.TF.frequency_bins`. `hrtfpykit.hrtf.sources.Sources` owns an in-memory copy of `SourcePosition` plus coordinate metadata. `hrtfpykit.hrtf.transforms.Transform` owns a reference to the parent hrtfpykit.hrtf.hrtf.HRTF and returns cloned/transformed hrtfpykit.hrtf.hrtf.HRTF objects.

### Dataset state composition

```text
hrtfpykit.datasets.base.BaseDataset
└── _state: hrtfpykit.datasets.state.DatasetState
    ├── config
    ├── specs and spec plan
    ├── resource paths and indexes
    ├── split state
    ├── acoustic context
    ├── cache
    └── rows
```

`hrtfpykit.datasets.state.DatasetState` is the explicit shared memory of the dataset layer. `hrtfpykit.datasets.build.DatasetBuilder`, `hrtfpykit.datasets.resources.DatasetResources`, `hrtfpykit.datasets.split.DatasetSplitPlanner`, `hrtfpykit.datasets.acoustic_context.DatasetAcousticContext`, and `hrtfpykit.datasets.values.DatasetSampleValueSelector` cooperate by reading and writing this state.

### Plot composition

```text
hrtfpykit.plots.hrtf.plot_spectrum_plane(...)
    ↓
hrtfpykit.plots.figure.Figure(
    hrtfpykit.plots.layouts.Layout_1(...)
    or hrtfpykit.plots.layouts.Layout_2Horizontal(...)
)
    ↓
hrtfpykit.plots.figure.Figure.create_heatmap(...)
    ↓
hrtfpykit.plots.types.Heatmap.create(...)
    ↓
matplotlib.axes.Axes.pcolormesh(...) + matplotlib.figure.Figure.colorbar(...)
```

The same pattern applies to line plots through `hrtfpykit.plots.figure.Figure.create_two_dimension()` and 3D source-grid plots through `hrtfpykit.plots.figure.Figure.create_three_dimension()`.

## Design rule

New architecture documentation should always answer three questions with code names:

1. Which class or function owns this behavior?
2. Which object composes or calls which other object?
3. What invariant does this code preserve?
