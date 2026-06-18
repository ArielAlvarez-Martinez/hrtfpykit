# `hrtfpykit.datasets` layer

The datasets layer owns public dataset construction, resource discovery, download planning, spec normalization, sample indexing, and optional PyTorch integration.

## Public entry points

From `src/hrtfpykit/datasets/__init__.py`:

Dataset classes:

- `hrtfpykit.datasets.ari.ARI`
- `hrtfpykit.datasets.hutubs.HUTUBS`
- `hrtfpykit.datasets.sonicom.SONICOM`

Specs:

- `hrtfpykit.datasets.specs.HRTFSpec`
- `hrtfpykit.datasets.specs.ITDSpec`
- `hrtfpykit.datasets.specs.ILDSpec`
- `hrtfpykit.datasets.specs.SHSpec`
- `hrtfpykit.datasets.specs.MeshSpec`
- `hrtfpykit.datasets.specs.AnthropometrySpec`
- `hrtfpykit.datasets.specs.MetadataSpec`
- `hrtfpykit.datasets.specs.ImageSpec`
- `hrtfpykit.datasets.specs.VideoSpec`

Transforms:

- `hrtfpykit.datasets.transforms.HRTFTransform`

Optional PyTorch integration:

- `hrtfpykit.datasets.torch.collate_samples()` in `hrtfpykit.datasets.torch`
- `hrtfpykit.datasets.torch.hrtf_loss()` in `hrtfpykit.datasets.torch`

## Inheritance structure

Concrete public datasets inherit from `hrtfpykit.datasets.base.BaseDataset`:

```text
hrtfpykit.datasets.base.BaseDataset
├── hrtfpykit.datasets.ari.ARI
├── hrtfpykit.datasets.hutubs.HUTUBS
└── hrtfpykit.datasets.sonicom.SONICOM
```

Files:

- `hrtfpykit.datasets.base.BaseDataset`: `src/hrtfpykit/datasets/base.py`
- `hrtfpykit.datasets.ari.ARI`: `src/hrtfpykit/datasets/ari.py`
- `hrtfpykit.datasets.hutubs.HUTUBS`: `src/hrtfpykit/datasets/hutubs.py`
- `hrtfpykit.datasets.sonicom.SONICOM`: `src/hrtfpykit/datasets/sonicom.py`

The concrete classes handle dataset-specific defaults, download server selection, and config selection. Shared dataset behavior is inherited from `hrtfpykit.datasets.base.BaseDataset`.

## Config composition

Dataset config dataclasses live in `src/hrtfpykit/datasets/config.py`.

`hrtfpykit.datasets.config.DatasetConfig` is the base schema. Concrete config classes inherit it:

- `hrtfpykit.datasets.config.ARIConfig(hrtfpykit.datasets.config.DatasetConfig)`
- `hrtfpykit.datasets.config.HUTUBSConfig(hrtfpykit.datasets.config.DatasetConfig)`
- `hrtfpykit.datasets.config.SONICOMConfig(hrtfpykit.datasets.config.DatasetConfig)`

A dataset config composes resource configs:

- `hrtfpykit.datasets.config.HRTFConfig`
- `hrtfpykit.datasets.config.MeshConfig`
- `hrtfpykit.datasets.config.AnthropometryConfig`
- `hrtfpykit.datasets.config.MetadataConfig`
- `hrtfpykit.datasets.config.ImageConfig`
- `hrtfpykit.datasets.config.VideoConfig`
- `hrtfpykit.datasets.config.DownloadServerConfig`
- `hrtfpykit.datasets.config.ResourceTypeConfig`

This keeps dataset metadata declarative. `hrtfpykit.datasets.build.DatasetBuilder`, `hrtfpykit.datasets.resources.DatasetResources`, and downloaders consume the config instead of hardcoding filenames throughout the dataset layer.

## Construction workflow

`hrtfpykit.datasets.base.BaseDataset.__init__()` creates a `hrtfpykit.datasets.state.DatasetState` and delegates construction to `hrtfpykit.datasets.build.DatasetBuilder(self).build(...)`.

```text
hrtfpykit.datasets.base.BaseDataset.__init__()
    ↓
hrtfpykit.datasets.state.DatasetState()
    ↓
hrtfpykit.datasets.build.DatasetBuilder(self).build(...)
    ├── hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.build()
    ├── hrtfpykit.datasets.resources.DatasetResources.build()
    ├── hrtfpykit.datasets.split.DatasetSplitPlanner.build()
    ├── hrtfpykit.datasets.acoustic_context.DatasetAcousticContext.build()
    └── hrtfpykit.datasets.build.DatasetBuilder._build_rows()
```

`hrtfpykit.datasets.build.DatasetBuilder` is defined in `src/hrtfpykit/datasets/build.py`. It owns construction order and writes resolved values into `hrtfpykit.datasets.state.DatasetState`.

The construction order is important:

1. Store base config and root.
2. Normalize `dataset_hrtf_variant` and `dataset_mesh_variant`.
3. Normalize specs with `hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.build()`.
4. Scan resources with `hrtfpykit.datasets.resources.DatasetResources.build()`.
5. Select subjects with `hrtfpykit.datasets.split.DatasetSplitPlanner.build()`.
6. Build acoustic axes with `hrtfpykit.datasets.acoustic_context.DatasetAcousticContext.build()`.
7. Build row dictionaries with `hrtfpykit.datasets.build.DatasetBuilder._build_rows()`.
8. Clear cache and optionally preload HRTFs.

## State ownership

`hrtfpykit.datasets.state.DatasetState` in `src/hrtfpykit/datasets/state.py` owns the constructed runtime state:

- config and root;
- selected variants;
- normalized input and target specs;
- spec names and index axes;
- resource paths and indexes;
- subject lists and split settings;
- acoustic axes and selected indices;
- row dictionaries;
- cache;
- resource and dataset summaries.

`hrtfpykit.datasets.base.BaseDataset` properties expose selected values from `_state` without duplicating state across attributes.

## Spec workflow

Specs are defined in `src/hrtfpykit/datasets/specs.py`:

- `hrtfpykit.datasets.specs.HRTFSpec`
- `hrtfpykit.datasets.specs.ITDSpec`
- `hrtfpykit.datasets.specs.ILDSpec`
- `hrtfpykit.datasets.specs.SHSpec`
- `hrtfpykit.datasets.specs.MeshSpec`
- `hrtfpykit.datasets.specs.AnthropometrySpec`
- `hrtfpykit.datasets.specs.MetadataSpec`
- `hrtfpykit.datasets.specs.ImageSpec`
- `hrtfpykit.datasets.specs.VideoSpec`

`hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.build()` in `src/hrtfpykit/datasets/specs_workflow.py` normalizes and validates specs. It enforces shared `index_by` behavior across indexed specs and validates axis compatibility for context encodings such as `position_one_hot`, `ear_index`, `frequency_index`, and `sample_index`.

Specs do not own dataset resources. They describe what values should be extracted and how those values should be indexed or grouped.

## Resource workflow

Resource scanning is coordinated by classes in `src/hrtfpykit/datasets/resources.py`:

- `hrtfpykit.datasets.resources.DatasetResources`
- `hrtfpykit.datasets.resources.DatasetResourcesScanner`
- `hrtfpykit.datasets.resources.DatasetResourcesValidator`
- `hrtfpykit.datasets.resources.DatasetResourcesPlan`

The scanner resolves local HRTF, mesh, anthropometry, metadata, image, and video resources according to dataset config and active specs. Subjects missing required resource families are excluded before split planning.

Download logic lives in `src/hrtfpykit/datasets/download.py`:

- `hrtfpykit.datasets.download.BaseDownload(ABC)`
- `hrtfpykit.datasets.download.PathPatternDownload(hrtfpykit.datasets.download.BaseDownload)`
- `hrtfpykit.datasets.download.SOFAcousticsDownload(hrtfpykit.datasets.download.PathPatternDownload)`
- `hrtfpykit.datasets.download.ImperialDownload(hrtfpykit.datasets.download.PathPatternDownload)`
- `hrtfpykit.datasets.download.TUBerlinDownload(hrtfpykit.datasets.download.BaseDownload)`
- `hrtfpykit.datasets.download.SONICOMEcosystemDownload(hrtfpykit.datasets.download.BaseDownload)`

`hrtfpykit.datasets.download.BaseDownload` owns shared validation, checksum, URL, and file-transfer logic. Concrete downloaders encode server-specific planning.

## Sample extraction

`hrtfpykit.datasets.base.BaseDataset.__getitem__()` reads one row from `hrtfpykit.datasets.state.DatasetState.rows` and dispatches each spec through `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_sample_value()`.

`hrtfpykit.datasets.values.DatasetSampleValueSelector` is defined in `src/hrtfpykit/datasets/values.py`. It dispatches to methods such as:

- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_hrtf_spec_value()`
- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_itd_spec_value()`
- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_ild_spec_value()`
- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_sh_spec_value()`
- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_mesh_spec_value()`
- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_anthropometry_spec_value()`
- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_metadata_spec_value()`
- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_image_spec_value()`
- `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_video_spec_value()`

The selector is stateless. It reads `hrtfpykit.datasets.state.DatasetState`, loads or reuses HRTFs through dataset cache, applies spec transforms, slices row axes, and returns concrete sample values.

`hrtfpykit.datasets.base.BaseDataset.__getitem__()` returns:

```python
{
    "inputs": ...,
    "target": ...,
    "meta": ...,
}
```

Context encodings requested by specs are added under `inputs`.

## HRTF transforms in datasets

`hrtfpykit.datasets.transforms.HRTFTransform` is defined in `src/hrtfpykit/datasets/transforms.py`.

It is a factory namespace for callables that receive an `hrtfpykit.hrtf.hrtf.HRTF` object and return an `hrtfpykit.hrtf.hrtf.HRTF` object. Most methods delegate to `hrtfpykit.hrtf.hrtf.HRTF.transform.*` through `hrtfpykit.datasets.transforms.HRTFTransform.build()`. `hrtfpykit.datasets.transforms.HRTFTransform.select()` delegates to `hrtfpykit.hrtf.hrtf.HRTF.select()` directly.

This design lets dataset construction apply reproducible HRTF preprocessing without putting transform logic into dataset classes.

## Optional PyTorch integration

`src/hrtfpykit/datasets/torch.py` contains:

- `hrtfpykit.datasets.torch.collate_samples()`
- `hrtfpykit.datasets.torch.hrtf_loss()`

This module is intentionally under `hrtfpykit.datasets` because it integrates dataset samples with PyTorch training loops. Torch imports are localized to this module/function path and should not be moved into `hrtfpykit.datasets.base.BaseDataset` or `hrtfpykit.datasets.__init__`.

## Logic example: build and index a dataset sample

This example shows the dataset construction and runtime indexing workflow. The
dataset layer owns resource discovery, spec normalization, row construction, and
sample extraction. Acoustic loading still delegates to `hrtfpykit.hrtf`.

```python
from hrtfpykit.datasets import HRTFSpec, HUTUBS, SHSpec

dataset = HUTUBS(
    root="datasets/hutubs",
    inputs=HRTFSpec(
        domain="frequency",
        signal="tf_magnitude_db",
        ears="left",
        index_by=("subject",),
        name="magnitude",
    ),
    target=SHSpec(
        sh_order=9,
        ears="left",
        index_by=("subject",),
        name="sh",
    ),
    split="train",
)

sample = dataset[0]
magnitude = sample["inputs"]["magnitude"]
coefficients = sample["target"]["sh"]
```

Construction flow:

```text
hrtfpykit.datasets.hutubs.HUTUBS(...)
    -> hrtfpykit.datasets.hutubs.HUTUBS.__init__(...)
    -> hrtfpykit.datasets.config.HUTUBSConfig()
    -> hrtfpykit.datasets.download.SOFAcousticsDownload(...)
       or hrtfpykit.datasets.download.TUBerlinDownload(...) when download=True
    -> hrtfpykit.datasets.base.BaseDataset.__init__(config=config, ...)
    -> hrtfpykit.datasets.build.DatasetBuilder(self).build(...)
       -> hrtfpykit.datasets.state.DatasetState()
       -> hrtfpykit.datasets.split.DatasetSplitPlanner.map_subject_ids(...)
          for requested subject filters
       -> hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.build(
              config=config, inputs=inputs, target=target
          )
       -> hrtfpykit.datasets.resources.DatasetResources.build(
              dataset, subject_ids=subject_ids, exclude_subject_ids=exclude_subject_ids
          )
       -> hrtfpykit.datasets.split.DatasetSplitPlanner.build(
              dataset, split=split, split_ratio=split_ratio, split_seed=split_seed
          )
       -> hrtfpykit.datasets.acoustic_context.DatasetAcousticContext.build(dataset)
       -> hrtfpykit.datasets.build.DatasetBuilder._build_rows(...)
       -> hrtfpykit.datasets.base.BaseDataset.clear_cache(dataset)
       -> hrtfpykit.datasets.base.BaseDataset.preload_hrtfs(dataset)
          when preload_hrtfs=True
```

Runtime indexing flow:

```text
dataset[0]
    -> hrtfpykit.datasets.base.BaseDataset.__getitem__(dataset, index=0)
    -> hrtfpykit.datasets.state.DatasetState rows via dataset._state
    -> row = dataset._state.rows[0]
    -> hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.get_spec_name(spec)
    -> hrtfpykit.datasets.values.DatasetSampleValueSelector.get_sample_value(
           dataset, spec, subject_id, row
       )
       -> hrtfpykit.datasets.specs_registry.get_spec_descriptor(spec)
       -> dataset override method if present
       -> hrtfpykit.datasets.values.DatasetSampleValueSelector.get_hrtf_spec_value(...)
          for hrtfpykit.datasets.specs.HRTFSpec
          -> hrtfpykit.datasets.sanitize.sanitize_index_by(spec.index_by)
          -> hrtfpykit.datasets.sanitize.sanitize_ears(spec.ears)
          -> hrtfpykit.datasets.base.BaseDataset.get_subject_hrtf(dataset, subject_id)
             -> hrtfpykit.datasets.load.load_dataset_hrtf(dataset, subject_id)
                -> hrtfpykit.datasets.split.DatasetSplitPlanner.map_subject_ids(...)
                -> hrtfpykit.hrtf.hrtf.load_hrtf(...)
                -> dataset._state.dataset_hrtf_transform(...) when configured
          -> spec.transform(hrtf) when configured
          -> hrtfpykit.hrtf.domain.TF.values / hrtfpykit.hrtf.domain.IR.values
          -> hrtfpykit.utils.dsp.real(...)
             / hrtfpykit.utils.dsp.imag(...)
             / hrtfpykit.utils.dsp.magnitude(...)
             / hrtfpykit.utils.dsp.magnitude_db(...)
             / hrtfpykit.utils.dsp.phase(...) for frequency signals
          -> numpy.take(...) and numpy.squeeze(...) for row-axis slicing
       -> hrtfpykit.datasets.values.DatasetSampleValueSelector.get_sh_spec_value(...)
          for hrtfpykit.datasets.specs.SHSpec
          -> hrtfpykit.datasets.base.BaseDataset.get_subject_hrtf(dataset, subject_id)
          -> spec.transform(hrtf) when configured
          -> hrtfpykit.utils.sh.sht(
                 selected_hrtf, sh_order=spec.sh_order, ear=sh_ear, epsilon=spec.epsilon
             ).C
          -> dataset._state.cache[("sh", subject_id, id(spec))]
          -> row-axis slicing
    -> assemble {"inputs": inputs, "target": target_values, "meta": meta}
    -> return sample
```

What each construction step does:

1. `hrtfpykit.datasets.hutubs.HUTUBS.__init__()` supplies HUTUBS-specific defaults, download behavior, and
   `hrtfpykit.datasets.config.HUTUBSConfig` to the shared dataset pipeline.
2. `hrtfpykit.datasets.base.BaseDataset.__init__()` creates the public dataset object and delegates
   construction to `hrtfpykit.datasets.build.DatasetBuilder(self).build(...)`.
3. `hrtfpykit.datasets.build.DatasetBuilder.build()` creates a fresh `hrtfpykit.datasets.state.DatasetState`, stores root and
   variant selections, and controls construction order.
4. `hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow.build()` normalizes `inputs` and `target`, resolves
   spec names, validates `index_by` / `grouped_by`, and enforces shared indexing
   rules across specs.
5. `hrtfpykit.datasets.resources.DatasetResources.build()` scans local HRTF, mesh, anthropometry, metadata,
   image, and video resources required by the active specs.
6. `hrtfpykit.datasets.split.DatasetSplitPlanner.build()` chooses the active subjects for `split="train"`
   after subject exclusions and resource availability are known.
7. `hrtfpykit.datasets.acoustic_context.DatasetAcousticContext.build()` derives reusable acoustic axes such as
   positions, ears, samples, frequencies, and selected indices.
8. `hrtfpykit.datasets.build.DatasetBuilder._build_rows()` creates the row dictionaries consumed by
   `hrtfpykit.datasets.base.BaseDataset.__len__()` and `hrtfpykit.datasets.base.BaseDataset.__getitem__()`.

What each indexing step does:

1. `hrtfpykit.datasets.base.BaseDataset.__getitem__()` reads one row from `hrtfpykit.datasets.state.DatasetState.rows`.
2. For each input and target spec, it calls
   `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_sample_value(dataset, spec, subject_id, row)`.
3. The selector asks the spec registry which method should handle the spec, for
   example `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_hrtf_spec_value()` for `hrtfpykit.datasets.specs.HRTFSpec` or `hrtfpykit.datasets.values.DatasetSampleValueSelector.get_sh_spec_value()` for
   `hrtfpykit.datasets.specs.SHSpec`.
4. HRTF-derived specs load or reuse a cached subject HRTF through
   `hrtfpykit.datasets.base.BaseDataset.get_subject_hrtf()`. That path delegates actual SOFA/HRTF
   parsing to `hrtfpykit.datasets.load.load_dataset_hrtf()` and then to `hrtfpykit.hrtf.hrtf.load_hrtf()`.
5. Dataset-level transforms from `dataset_hrtf_transform` run before spec values
   are extracted. Spec-level transforms and row-level axis slicing are applied
   afterward.
6. The final sample dictionary is assembled with `inputs`, `target`, and `meta`.
   If a `torch.utils.data.DataLoader` is used later, `hrtfpykit.datasets.torch.collate_samples()` can
   batch those dictionaries without changing dataset construction or indexing.

## Invariants

- Concrete dataset classes should inherit `hrtfpykit.datasets.base.BaseDataset` and pass a concrete config into the shared builder.
- Specs decide required resources before scanning.
- Resource scanning decides available subjects before split planning.
- Row construction happens after selected subjects and selected axes are known.
- `hrtfpykit.datasets.base.BaseDataset.__getitem__()` is the only runtime path that turns rows and specs into sample dictionaries.
- PyTorch integration remains optional and isolated.

## Do not do this

- Do not put plotting logic in dataset classes.
- Do not make `hrtfpykit.datasets.base.BaseDataset` import torch.
- Do not make specs perform downloads.
- Do not make download options implicitly follow dataset construction options; `download_*` and `dataset_*` variants are intentionally separate.
- Do not bypass `hrtfpykit.datasets.state.DatasetState` by storing parallel state on concrete dataset classes unless there is a concrete reason.
