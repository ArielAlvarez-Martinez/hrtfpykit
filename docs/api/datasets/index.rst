datasets
========

Description:
------------

The datasets layer is hrtfpykit's spec driven construction layer for HRTF
datasets.  It turns dataset-specific resource layouts into indexed dataset
objects whose samples are defined explicitly by ``inputs`` and ``target`` specs.
These objects follow the map style dataset pattern: they provide ``len(dataset)``
and integer ``dataset[index]`` access, so constructed datasets can be passed
directly to PyTorch data loaders and similar batching pipelines.  This layer is
designed for workflows where acoustic data, subject resources, and derived
values need to be selected, aligned, split, and reused reproducibly.

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
exclusions, train/validation/test splits, and construction summaries.  Specs
describe what each sample should contain.  Acoustic specs such as
:class:`~hrtfpykit.datasets.HRTFSpec`, :class:`~hrtfpykit.datasets.ITDSpec`,
:class:`~hrtfpykit.datasets.ILDSpec`, and
:class:`~hrtfpykit.datasets.SHSpec` request values derived from subject HRTFs.
Resource specs such as :class:`~hrtfpykit.datasets.MeshSpec`,
:class:`~hrtfpykit.datasets.AnthropometrySpec`,
:class:`~hrtfpykit.datasets.MetadataSpec`,
:class:`~hrtfpykit.datasets.ImageSpec`, and
:class:`~hrtfpykit.datasets.VideoSpec` request non-acoustic resources aligned
with the same subject and row context.

The datasets layer builds on :doc:`hrtf <../hrtf/index>` for acoustic
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
   :maxdepth: 3

   Specs <specs>
   Dataset HRTF transforms <transforms>
   collate_samples <collate_samples>
   HUTUBS <hutubs>
   SONICOM <sonicom>
