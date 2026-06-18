# Dependency rules

These rules keep the four public layers stable and prevent architecture drift.

## Public layer order

```text
hrtfpykit.sofa
    ↓
hrtfpykit.hrtf
    ↓
hrtfpykit.plots
hrtfpykit.datasets
```

`hrtfpykit.plots` and `hrtfpykit.datasets` are consumers of `hrtfpykit.hrtf`. They should not become dependencies of the HRTF core.

## Allowed dependencies

### `hrtfpykit.sofa`

Allowed:

- `netCDF4`
- `numpy`
- `hrtfpykit.sofa.*`
- warning utilities such as `hrtfpykit.utils.warnings`

The SOFA layer owns file validation and netCDF handle management. It should not depend on `hrtfpykit.hrtf`, `hrtfpykit.plots`, or `hrtfpykit.datasets`.

### `hrtfpykit.hrtf`

Allowed:

- `hrtfpykit.sofa`
- `hrtfpykit.hrtf.domain.IR`
- `hrtfpykit.hrtf.domain.TF`
- `hrtfpykit.hrtf.sources.Sources`
- `hrtfpykit.hrtf.transforms.Transform`
- `hrtfpykit.utils.dsp`
- `hrtfpykit.utils.metrics`
- `hrtfpykit.utils.sh`
- coordinate and plane helpers from `hrtfpykit.utils.coordinates` and `hrtfpykit.utils.planes`

Forbidden:

- importing `hrtfpykit.plots` from HRTF core;
- importing `hrtfpykit.datasets` from HRTF core;
- putting dataset-specific resource logic inside `hrtfpykit.hrtf.hrtf.HRTF`, `hrtfpykit.hrtf.domain.IR`, `hrtfpykit.hrtf.domain.TF`, `hrtfpykit.hrtf.sources.Sources`, or `hrtfpykit.hrtf.transforms.Transform`.

### `hrtfpykit.plots`

Allowed:

- `hrtfpykit.hrtf.hrtf.HRTF`-like objects;
- public metrics such as `hrtfpykit.utils.metrics.itd()`, `hrtfpykit.utils.metrics.ild()`, `hrtfpykit.utils.metrics.hrtf_difference()`, and SH helpers;
- plotting internals such as `hrtfpykit.plots.figure.Figure`, `hrtfpykit.plots.layouts.Layout_*`, `hrtfpykit.plots.types.TwoDimension`, `hrtfpykit.plots.types.Heatmap`, `hrtfpykit.plots.types.ThreeDimension`, axes, labels, legends, and titles;
- Matplotlib and NumPy.

Forbidden:

- mutating `hrtfpykit.hrtf.domain.IR.values` or `hrtfpykit.hrtf.domain.TF.values`;
- making plots depend on dataset classes such as `hrtfpykit.datasets.ari.ARI`, `hrtfpykit.datasets.hutubs.HUTUBS`, or `hrtfpykit.datasets.sonicom.SONICOM`;
- embedding dataset download or resource discovery behavior in plotting functions.

### `hrtfpykit.datasets`

Allowed:

- `hrtfpykit.hrtf.hrtf.load_hrtf()` through dataset loading helpers;
- `hrtfpykit.hrtf.hrtf.HRTF` transforms through `hrtfpykit.datasets.transforms.HRTFTransform`;
- metrics and SH utilities indirectly through spec value selectors;
- local resource, download, and spec workflow modules in `hrtfpykit.datasets.*`.

Forbidden:

- importing `hrtfpykit.plots` in dataset construction or sample extraction;
- making core dataset classes require PyTorch;
- writing dataset-specific special cases into generic HRTF transforms.

PyTorch code must stay isolated in `hrtfpykit.datasets.torch`. `hrtfpykit.datasets.torch.collate_samples()` and `hrtfpykit.datasets.torch.hrtf_loss()` import PyTorch only when needed, so the dataset package can remain usable without torch installed.

## Invariant rules

### Acoustic shapes

Standard in-memory HRTF arrays use these shapes:

```text
hrtfpykit.hrtf.domain.IR.values -> (positions, ears, samples)
hrtfpykit.hrtf.domain.TF.values -> (positions, ears, frequency_bins)
hrtfpykit.hrtf.sources.Sources.positions -> (positions, 3)
```

Any transform or selection that changes one acoustic representation must keep the sibling representation synchronized.

### SOFA synchronization

`hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` is the boundary between in-memory acoustic state and SOFA persistence. Transform methods should not silently write to disk. `hrtfpykit.hrtf.hrtf.HRTF.save()` should go through `hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` and then `hrtfpykit.sofa.sofa.SOFA.save()`.

### Dataset resource rules

`hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.build()` must run before `hrtfpykit.datasets.resources.DatasetResources.build()` because specs decide which resource families are required. `hrtfpykit.datasets.resources.DatasetResources.build()` must run before `hrtfpykit.datasets.split.DatasetSplitPlanner.build()` because subjects missing required resources should be removed before split selection.

### Plotting rules

Plot functions should return Matplotlib `hrtfpykit.plots.figure.Figure` objects and should not hide acoustic mutation inside visualization calls. If a plot needs reduced values, compute them locally or through public metrics, not by changing the input `hrtfpykit.hrtf.hrtf.HRTF` object.

## Common mistakes to avoid

- Do not make `hrtfpykit.hrtf.hrtf.HRTF` inherit from `hrtfpykit.sofa.sofa.SOFA`; it should compose `hrtfpykit.sofa.sofa.SOFA` through `hrtfpykit.hrtf.hrtf.HRTF.Sofa`.
- Do not make dataset specs load files directly when the dataset state/resource scanner already owns resource paths.
- Do not put `torch` imports into `hrtfpykit.datasets.__init__` or `hrtfpykit.datasets.base.BaseDataset`.
- Do not put Matplotlib layout logic in `hrtfpykit.hrtf`.
- Do not add new public layer names unless they become real package-level public surfaces.
