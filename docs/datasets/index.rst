hrtfpykit.datasets
==================

Description:
------------

``hrtfpykit.datasets`` is hrtfpykit's construction layer for public HRTF
datasets. It turns dataset-specific resource layouts into indexed dataset
objects whose samples are declared with ``inputs`` and ``target`` specs. These
objects follow the map style dataset pattern: they provide ``len(dataset)`` and
integer ``dataset[index]`` access, so constructed datasets can be passed directly
to PyTorch data loaders and similar batching pipelines. This layer is designed
for workflows where acoustic data, subject resources, and derived values need to
be selected, aligned, split, and reused reproducibly.

Each integer access returns a sample dictionary ready for batching.  The
``sample["inputs"]`` entry contains the values requested by the ``inputs``
specs, together with any requested context encodings such as position, ear,
frequency, or sample indices.  The ``sample["target"]`` entry contains the
value or values requested by the ``target`` specs.  This keeps the same dataset
object usable for direct inspection, preprocessing scripts, and PyTorch
``DataLoader`` workflows.

Concrete dataset classes such as :class:`~hrtfpykit.datasets.HUTUBS` and
:class:`~hrtfpykit.datasets.SONICOM` handle dataset-specific subject identifiers,
folder layouts, downloadable resource groups, resource variants, subject
exclusions, train/validation/test splits, and construction summaries.

Specs describe what a dataset sample should contain. They are small
configuration objects passed to dataset constructors through the ``inputs`` and
``target`` arguments. A spec does not load files by itself. Instead, the dataset
reads the specs during construction and uses them to decide which resource
families are required, which subjects can be included, how rows should be
indexed, and which values should be returned when a sample is requested.

Each spec defines one requested value. Acoustic specs such as
:class:`~hrtfpykit.datasets.HRTFSpec`, :class:`~hrtfpykit.datasets.ITDSpec`,
:class:`~hrtfpykit.datasets.ILDSpec`, and
:class:`~hrtfpykit.datasets.SHSpec` select HRTF-derived data, source subsets,
ears, frequency bins, samples, or derived cues. Resource specs such as
:class:`~hrtfpykit.datasets.MeshSpec`,
:class:`~hrtfpykit.datasets.AnthropometrySpec`,
:class:`~hrtfpykit.datasets.MetadataSpec`,
:class:`~hrtfpykit.datasets.ImageSpec`, and
:class:`~hrtfpykit.datasets.VideoSpec` request non-acoustic resources aligned
with the same subject and row context.

Specs make dataset construction explicit and reproducible. They separate the
dataset storage layout from the values a model, analysis script, or preprocessing
pipeline actually needs. The same local HUTUBS or SONICOM root can produce
different datasets by changing the spec list: one dataset can expose full HRIR
arrays, another can expose position-indexed ITD values, and another can combine
HRTF data with anthropometry, mesh, image, or metadata resources. During
indexing, the dataset uses the specs to populate ``sample["inputs"]`` and
``sample["target"]`` with consistent keys, selected axes, optional one-hot or
index encodings, and any per-value transforms requested by the spec.

The datasets layer builds on :doc:`hrtfpykit.hrtf <../hrtf/index>` for acoustic
resources.  Subject HRTF files are loaded as
:class:`~hrtfpykit.hrtf.HRTF` objects, optional dataset-level HRTF transforms can
be applied, and specs then extract arrays, binaural cues, or
spherical-harmonic values from the loaded object.  This keeps dataset samples
consistent with the same HRTF loading, selection, transformation, and
synchronization logic used elsewhere in hrtfpykit.

With these tools, users can choose a concrete dataset integration, select local and
download variants, define inputs and targets with specs, exclude subjects,
construct deterministic subject splits, inspect resource and dataset summaries,
load individual subject HRTFs, and retrieve indexed samples for analysis,
preprocessing, or model training with map-style data loading.

Content:
--------

.. toctree::
   :maxdepth: 1

   HRTFSpec
   ITDSpec
   ILDSpec
   SHSpec
   MeshSpec
   AnthropometrySpec
   MetadataSpec
   ImageSpec
   VideoSpec
   HRTFTransform
   collate_samples <collate_samples>
   HUTUBS <hutubs>
   SONICOM <sonicom>
