SONICOM
=======

``SONICOM`` is the public dataset interface for the SONICOM HRTF dataset. It
supports measured and synthetic HRTFs, scanned and synthetic meshes, and subject
metadata.

Use ``SONICOM`` when you need SONICOM HRTFs as indexed samples, optionally paired
with metadata or geometry resources.

What SONICOM provides
---------------------

HRTF variants
   SONICOM HRTFs are selected with a variant dictionary containing ``type``,
   ``sample_rate``, and ``version``.

Mesh variants
   SONICOM meshes are selected with a variant dictionary containing ``type`` and
   ``version``.

Metadata
   Metadata is exposed through ``MetadataSpec`` and is independent from
   anthropometry resources.

Download planning
   Download parameters are explicit and independent from dataset construction
   variants.

Construction example
--------------------

.. code-block:: python

   from hrtfpykit.datasets import SONICOM, HRTFSpec, MetadataSpec

   dataset = SONICOM(
       root="datasets/sonicom",
       dataset_hrtf_variant={
           "type": "measured",
           "sample_rate": 44100,
           "version": "FreeFieldComp",
       },
       inputs=[HRTFSpec(), MetadataSpec()],
   )

   sample = dataset[0]
   hrtf_value = sample["inputs"]["hrtf"]
   metadata = sample["inputs"]["metadata"]

Mesh example
------------

.. code-block:: python

   from hrtfpykit.datasets import SONICOM, MeshSpec

   dataset = SONICOM(
       root="datasets/sonicom",
       dataset_mesh_variant={"type": "scanned", "version": "watertight"},
       inputs=MeshSpec(),
   )

   mesh = dataset[0]["inputs"]["mesh"]

Download behavior
-----------------

Download parameters control the files that are downloaded:

- ``download_resources`` selects resource families.
- ``download_hrtf_variant`` selects HRTF download axes.
- ``download_mesh_variant`` selects mesh download axes.

Dataset construction parameters control what local files are scanned and used:

- ``dataset_hrtf_variant`` for HRTFs.
- ``dataset_mesh_variant`` for meshes.

Example:

.. code-block:: python

   dataset = SONICOM(
       root="datasets/sonicom",
       download=True,
       download_resources="hrtf",
       download_hrtf_variant={
           "type": "measured",
           "sample_rate": 44100,
           "version": "FreeFieldComp",
       },
       dataset_hrtf_variant={
           "type": "measured",
           "sample_rate": 44100,
           "version": "FreeFieldComp",
       },
       inputs=HRTFSpec(),
   )

API reference
-------------

.. autoclass:: hrtfpykit.datasets.SONICOM
   :members:
   :show-inheritance:
