# Data flow

This page describes runtime data flow across the four public layers.

## SOFA file to `hrtfpykit.hrtf.hrtf.HRTF`

```text
hrtfpykit.hrtf.hrtf.load_hrtf(path)
    ↓ calls
hrtfpykit.sofa.sofa.load_sofa(path)
    ↓ returns
hrtfpykit.sofa.sofa.SOFA
    ↓ wrapped by
hrtfpykit.hrtf.hrtf.HRTF(Sofa)
    ├── hrtfpykit.hrtf.domain.IR.values / hrtfpykit.hrtf.domain.IR.sample_rate
    ├── hrtfpykit.hrtf.domain.TF.values / hrtfpykit.hrtf.domain.TF.frequency_bins
    └── hrtfpykit.hrtf.sources.Sources.positions / SourcePosition metadata
```

`hrtfpykit.hrtf.hrtf.load_hrtf()` in `src/hrtfpykit/hrtf/hrtf.py` is the main acoustic loader. It always starts by calling `hrtfpykit.sofa.sofa.load_sofa()`. The resulting `hrtfpykit.sofa.sofa.SOFA` object is stored on `hrtfpykit.hrtf.hrtf.HRTF.Sofa`.

For `SimpleFreeFieldHRIR`:

1. `hrtfpykit.hrtf.hrtf.load_hrtf()` reads `Data.IR` and `Data.SamplingRate` from `hrtfpykit.sofa.sofa.SOFA.Variables`.
2. It calls `hrtfpykit.utils.dsp.tf_from_ir()` from `src/hrtfpykit/utils/dsp.py`.
3. It stores the time-domain array in `hrtfpykit.hrtf.domain.IR.values`.
4. It stores the derived complex spectrum in `hrtfpykit.hrtf.domain.TF.values`.
5. It initializes `hrtfpykit.hrtf.sources.Sources` so source positions are available in memory.

For `SimpleFreeFieldHRTF`:

1. `hrtfpykit.hrtf.hrtf.load_hrtf()` reads `Data.Real`, `Data.Imag`, and `N` from `hrtfpykit.sofa.sofa.SOFA.Variables`.
2. It combines `Data.Real + 1j * Data.Imag` into a complex TF.
3. It calls `hrtfpykit.utils.dsp.prepend_missing_dc()` when the DC bin is absent but inferable.
4. It calls `hrtfpykit.utils.dsp.ir_from_tf()` to reconstruct `hrtfpykit.hrtf.domain.IR.values` and sample rate.
5. It stores `hrtfpykit.hrtf.domain.TF.values`, `hrtfpykit.hrtf.domain.TF.frequency_bins`, `hrtfpykit.hrtf.domain.IR.values`, and `hrtfpykit.hrtf.domain.IR.sample_rate`.

## Transform flow

```text
hrtf.transform.apply_window(...)
    ↓
hrtfpykit.hrtf.transforms.Transform.apply_window(...)
    ↓ clones parent hrtfpykit.hrtf.hrtf.HRTF
hrtfpykit.hrtf.hrtf.HRTF.clone()
    ↓ modifies hrtfpykit.hrtf.domain.IR or hrtfpykit.hrtf.domain.TF
hrtfpykit.utils.dsp.* or hrtfpykit.utils.metrics.* helper
    ↓ synchronizes sibling domain
hrtfpykit.utils.dsp.tf_from_ir(...) or hrtfpykit.utils.dsp.ir_from_tf(...)
    ↓ returns
new hrtfpykit.hrtf.hrtf.HRTF
```

`hrtfpykit.hrtf.hrtf.HRTF.transform` returns a `hrtfpykit.hrtf.transforms.Transform` object from `src/hrtfpykit/hrtf/transforms.py`. Transform methods are immutable from the caller perspective: they operate on a cloned HRTF and return the new object.

Examples:

- `hrtfpykit.hrtf.transforms.Transform.apply_window()` uses `hrtfpykit.utils.dsp.window()` from `src/hrtfpykit/utils/dsp.py` and then refreshes TF through `hrtfpykit.utils.dsp.tf_from_ir()`.
- `hrtfpykit.hrtf.transforms.Transform.apply_padding()` uses `hrtfpykit.utils.dsp.padding()` and then refreshes TF.
- `hrtfpykit.hrtf.transforms.Transform.modify_magnitude()` modifies TF magnitude and then refreshes IR through `hrtfpykit.utils.dsp.ir_from_tf()`.
- `hrtfpykit.hrtf.transforms.Transform.add_itd()` and `hrtfpykit.hrtf.transforms.Transform.delete_itd()` use ITD logic from `hrtfpykit.utils.metrics.itd()` and domain conversion helpers.
- `hrtfpykit.hrtf.transforms.Transform.add_ild()` and `hrtfpykit.hrtf.transforms.Transform.delete_ild()` use TF-domain level edits and the public `hrtfpykit.utils.metrics.ild()` metric.

The invariant is that `hrtfpykit.hrtf.domain.IR.values`, `hrtfpykit.hrtf.domain.IR.sample_rate`, `hrtfpykit.hrtf.domain.TF.values`, and `hrtfpykit.hrtf.domain.TF.frequency_bins` should remain synchronized after a transform.

## Selection flow

```text
hrtfpykit.hrtf.hrtf.HRTF.select(...)
    ↓ resolves source/ear/sample/frequency selection
hrtfpykit.hrtf.sources.Sources.get_position_index(...)
    / hrtfpykit.hrtf.sources.Sources.get_positions(...)
    ↓ clones hrtfpykit.hrtf.hrtf.HRTF
selected hrtfpykit.hrtf.hrtf.HRTF
    ├── sliced hrtfpykit.hrtf.domain.IR.values
    ├── sliced hrtfpykit.hrtf.domain.TF.values
    └── hrtfpykit.hrtf.sources.Sources._selected_indices
```

`hrtfpykit.hrtf.hrtf.HRTF.select()` creates a selected HRTF view. `hrtfpykit.hrtf.sources.Sources` handles coordinate-aware source resolution through methods such as `hrtfpykit.hrtf.sources.Sources.get_position_index()`, `hrtfpykit.hrtf.sources.Sources.get_positions()`, `hrtfpykit.hrtf.sources.Sources.get_azimuth_angles()`, and `hrtfpykit.hrtf.sources.Sources.get_elevation_angles()`.

Source rows are always the first axis of standard HRTF arrays:

- `hrtfpykit.hrtf.domain.IR.values`: `(positions, ears, samples)`
- `hrtfpykit.hrtf.domain.TF.values`: `(positions, ears, frequency_bins)`

## Synchronizing back to SOFA

```text
transformed_or_selected_hrtf
    ↓
hrtfpykit.hrtf.hrtf.HRTF.update_sofa()
    ↓
hrtfpykit.sofa.sofa.SOFA.copy_with(...) or hrtfpykit.sofa.sofa.SOFA variable edits
    ↓
hrtfpykit.hrtf.hrtf.HRTF.save(path)
    ↓
hrtfpykit.sofa.sofa.SOFA.save(path)
```

`hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` synchronizes the in-memory `hrtfpykit.hrtf.hrtf.HRTF` state back to the backing `hrtfpykit.sofa.sofa.SOFA` object. It handles differences between `SimpleFreeFieldHRIR` and `SimpleFreeFieldHRTF` and validates shape compatibility before saving.

`hrtfpykit.hrtf.hrtf.HRTF.save()` calls `hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` and then `hrtfpykit.sofa.sofa.SOFA.save()`. `hrtfpykit.sofa.sofa.SOFA.save()` preserves dimensions, global attributes, variables, variable attributes, and variable storage metadata when writing a new file.

## Dataset construction flow

```text
hrtfpykit.datasets.ari.ARI(...)
    / hrtfpykit.datasets.hutubs.HUTUBS(...)
    / hrtfpykit.datasets.sonicom.SONICOM(...)
    ↓ inherit
hrtfpykit.datasets.base.BaseDataset.__init__()
    ↓ creates
hrtfpykit.datasets.state.DatasetState
    ↓ delegates to
hrtfpykit.datasets.build.DatasetBuilder.build()
    ├── hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.build()
    ├── hrtfpykit.datasets.resources.DatasetResources.build()
    ├── hrtfpykit.datasets.split.DatasetSplitPlanner.build()
    ├── hrtfpykit.datasets.acoustic_context.DatasetAcousticContext.build()
    └── hrtfpykit.datasets.build.DatasetBuilder._build_rows()
```

Concrete datasets provide config and download behavior. The shared dataset behavior is in `hrtfpykit.datasets.base.BaseDataset`.

The important ordering inside `hrtfpykit.datasets.build.DatasetBuilder.build()` is:

1. Normalize dataset variants such as `dataset_hrtf_variant` and `dataset_mesh_variant`.
2. Normalize `inputs` and `target` through `hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.build()`.
3. Scan local resources with `hrtfpykit.datasets.resources.DatasetResources.build()`.
4. Plan selected subjects with `hrtfpykit.datasets.split.DatasetSplitPlanner.build()`.
5. Derive acoustic axes through `hrtfpykit.datasets.acoustic_context.DatasetAcousticContext.build()`.
6. Build row dictionaries with `hrtfpykit.datasets.build.DatasetBuilder._build_rows()`.
7. Optionally preload HRTFs with `hrtfpykit.datasets.base.BaseDataset.preload_hrtfs()`.

## Dataset indexing flow

```text
dataset[index]
    ↓
hrtfpykit.datasets.base.BaseDataset.__getitem__()
    ↓ reads row from hrtfpykit.datasets.state.DatasetState.rows
    ↓ for each spec
hrtfpykit.datasets.values.DatasetSampleValueSelector.get_sample_value()
    ↓ dispatches by hrtfpykit.datasets.specs_registry.get_spec_descriptor(...)
hrtfpykit.datasets.values.DatasetSampleValueSelector.get_hrtf_spec_value(...)
    / hrtfpykit.datasets.values.DatasetSampleValueSelector.get_itd_spec_value(...)
    / hrtfpykit.datasets.values.DatasetSampleValueSelector.get_mesh_spec_value(...)
    / ...
    ↓ returns
{"inputs": ..., "target": ..., "meta": ...}
```

`hrtfpykit.datasets.base.BaseDataset.__getitem__()` always returns a dictionary with `inputs`, `target`, and `meta` keys. It also adds context encodings such as `position_one_hot`, `position_index`, `ear_one_hot`, `ear_index`, `frequency_one_hot`, `frequency_index`, `sample_one_hot`, or `sample_index` when requested by specs.

## Plotting flow

```text
hrtfpykit.plots.hrtf.plot_* or hrtfpykit.plots.compare.compare_*
    ↓ reads hrtfpykit.hrtf.hrtf.HRTF / hrtfpykit.utils.metrics values
hrtfpykit.plots.figure.Figure(layout)
    ↓ get axes
hrtfpykit.plots.figure.Figure.get_ax(...)
    ↓ render primitive
hrtfpykit.plots.figure.Figure.create_two_dimension()
hrtfpykit.plots.figure.Figure.create_heatmap()
hrtfpykit.plots.figure.Figure.create_three_dimension()
    ↓ apply hrtfpykit.plots.axis / labels / legends / titles helpers
matplotlib.figure.Figure
```

Plot functions do not own acoustic data. They consume `hrtfpykit.hrtf.hrtf.HRTF`, metrics, and source positions. They should not mutate `hrtfpykit.hrtf.hrtf.HRTF.IR`, `hrtfpykit.hrtf.hrtf.HRTF.TF`, or `hrtfpykit.hrtf.hrtf.HRTF.Sources`.
