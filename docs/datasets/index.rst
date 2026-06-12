hrtfpykit.datasets
==================

Description:
------------

``hrtfpykit.datasets`` builds map-style datasets from public HRTF datasets and
aligned custom resources. It turns dataset-specific file layouts into objects
with ``len(dataset)`` and ``dataset[index]`` access, so the same dataset can be
inspected directly, preprocessed in scripts, or passed to PyTorch-style data
loaders.

Dataset integrations such as :class:`~hrtfpykit.datasets.ARI`,
:class:`~hrtfpykit.datasets.HUTUBS`, and
:class:`~hrtfpykit.datasets.SONICOM` keep dataset-specific rules outside user
code. They handle subject identifiers, folder layouts, downloadable resource
groups, resource variants where the dataset defines them, excluded subjects,
deterministic splits, and resource summaries.

The built-in dataset pages distinguish dataset scope from implementation status.
Dataset scope describes the public dataset and the resource families hrtfpykit
can use. Implementation status describes the exact subject identifiers,
resource groups, server choices, and checksum-backed files configured in the
current hrtfpykit release. This matters most for actively evolving sources such
as SONICOM, where the Imperial transfer server and SONICOM ecosystem catalogs do
not necessarily expose the same subject/resource coverage at the same time.

Users do not have to use the built-in download step. If files were downloaded
previously or prepared by another workflow, they can be copied under the dataset
``root``. Dataset scanners look for the official layout and the semantic local
alternatives declared in the dataset configuration, such as
``subject_id/hrtf/...``, ``subject_id/mesh/...``, ``metadata/...``, and
``anthropometry/...`` folders where supported.

Samples are declared with specs passed through ``inputs`` and ``target``. Each
spec names one value to return and the context used to index it. Acoustic specs
such as :class:`~hrtfpykit.datasets.HRTFSpec`,
:class:`~hrtfpykit.datasets.ITDSpec`, :class:`~hrtfpykit.datasets.ILDSpec`, and
:class:`~hrtfpykit.datasets.SHSpec` request HRTF-derived arrays, cues, or
spherical-harmonic values. Resource specs such as
:class:`~hrtfpykit.datasets.MeshSpec`,
:class:`~hrtfpykit.datasets.AnthropometrySpec`,
:class:`~hrtfpykit.datasets.MetadataSpec`,
:class:`~hrtfpykit.datasets.ImageSpec`, and
:class:`~hrtfpykit.datasets.VideoSpec` align non-acoustic subject resources with
the same sample rows.

Resource specs can use official resources when a dataset provides them, such as
SONICOM meshes or metadata, but they are not limited to those files. They can
also point to custom resources prepared for an experiment. ``AnthropometrySpec``
and ``MetadataSpec`` can read custom tables aligned with the dataset subject
IDs. ``MeshSpec`` can use a custom mesh root while preserving the mesh naming
pattern declared by the dataset configuration. ``ImageSpec`` and ``VideoSpec``
scan explicit media roots organized by subject, which makes them useful for
visual pipelines such as ear images rendered from meshes or collected by another
acquisition process.

For image and video resources, the media root must contain one folder per
dataset subject. Subject folders can use the canonical dataset subject ID, or the
aliases ``subjectN`` and ``subject_N`` based on the dataset subject number. When
the resource is grouped by ``("subject", "ear")``, each subject folder must
contain ear folders such as ``left`` and ``right``:

.. code-block:: text

   ear_images/
      pp2/
         left/
            image_001.png
         right/
            image_001.png
      subject3/
         left/
            image_001.png
         right/
            image_001.png

Transforms on resource specs define how selected paths or table values become
the values returned by ``dataset[index]``. Without a transform, mesh, image, and
video specs return organized paths, while anthropometry and metadata specs
return the selected table values. With a transform, hrtfpykit still handles
subject alignment, resource inspection, split selection, and sample construction;
the transform defines how the selected resource is loaded and prepared.

For example, ``ImageSpec`` can locate custom ear render images for each subject,
then a transform can open those images with PIL, apply a torchvision ``Compose``
pipeline, resize, normalize, augment, and return the array or tensor expected by
a model. In an HRTF individualization workflow, those image values can be used as
inputs while ``HRTFSpec`` or ``SHSpec`` provides the acoustic target. The same
pattern applies to videos, meshes, anthropometry, and metadata: hrtfpykit
resolves the aligned resource, and the transform turns that resource into the
sample value.

When ``dataset[index]`` is called, the dataset returns a dictionary with
``sample["inputs"]``, ``sample["target"]``, and ``sample["meta"]`` entries.
``inputs`` and ``target`` contain the values requested by the selected specs.
``meta`` keeps row provenance separate from model values: it records the dataset
name, native subject ID, and the active row context such as position, ear,
frequency, or sample index when those axes are part of ``index_by``. This is
especially useful when datasets are combined with PyTorch ``ConcatDataset`` or
when one subject expands into many indexed rows. Acoustic values are resolved
through :doc:`hrtfpykit.hrtf <../hrtf/index>`: subject SOFA files are loaded as
:class:`~hrtfpykit.hrtf.HRTF` objects, optional dataset-level transforms are
applied, and specs extract the requested values from the loaded object. As a
result, dataset samples use the same HRTF loading and transformation path as the
rest of hrtfpykit.

This design is useful for HRTF individualization and related research tasks,
where acoustic targets often need to be paired with subject information. By
combining custom or official image, video, mesh, anthropometry, metadata, and
acoustic specs in one dataset definition, users can build reproducible
multimodal pipelines for deep learning experiments without maintaining separate
subject matching code for each resource family.

Content:
--------

.. toctree::
   :maxdepth: 1

   ARI <ari>
   HUTUBS <hutubs>
   SONICOM <sonicom>
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
   torch <torch/index>
