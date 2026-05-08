Dataset specs
=============

Dataset specs are the public contract between a dataset and one sample returned
by ``dataset[index]``. A spec does not load data by itself. Instead, it tells the
dataset builder which resource family is needed, how subjects and axes should be
indexed, and which value should be extracted for ``sample["inputs"]`` or
``sample["target"]``.

The same spec classes are used by all concrete datasets. For example,
``HRTFSpec`` requests HRTF/HRIR values from HUTUBS or SONICOM, while
``MetadataSpec`` requests metadata rows from any dataset that defines a metadata
resource.

How specs affect a dataset
--------------------------

Specs participate in four parts of dataset construction.

Resource selection
   Each spec maps to one resource family. ``HRTFSpec``, ``ITDSpec``,
   ``ILDSpec``, and ``SHSpec`` require HRTF resources. ``MeshSpec`` requires mesh
   resources. ``AnthropometrySpec`` and ``MetadataSpec`` require table resources.
   ``ImageSpec`` and ``VideoSpec`` require media resources.

Subject intersection
   If multiple specs require different resources, the dataset keeps subjects that
   are available in all required resource families. Missing subjects are removed
   before rows are built.

Row indexing
   ``index_by`` and grouping options decide how many rows the dataset has. A
   subject-only spec creates one row per selected subject. A position-indexed
   acoustic spec creates one row per selected subject and position. Ear-,
   frequency-, or sample-indexed specs add their corresponding axes.

Value extraction
   When ``dataset[index]`` is called, the row context is passed to each spec and
   the dataset extracts the requested value into ``sample["inputs"]`` or
   ``sample["target"]``.

Acoustic specs
--------------

HRTFSpec
~~~~~~~~

``HRTFSpec`` requests HRTF or HRIR values from the selected HRTF resource. Use it
when the sample should contain measured, simulated, or synthetic HRTF data.
Depending on its configuration, the returned value can come from the time-domain
IR or frequency-domain TF representation.

Example:

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, HRTFSpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=HRTFSpec(
           index_by=("subject", "position"),
           positions=(0, 1, 2),
           domain="frequency",
       ),
   )

   sample = dataset[0]
   hrtf_value = sample["inputs"]["hrtf"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.HRTFSpec
   :members:

ITDSpec
~~~~~~~

``ITDSpec`` requests interaural time difference values derived from each loaded
subject HRTF. Use it when the dataset sample needs binaural timing information
instead of the full HRTF/HRIR signal.

Example:

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, ITDSpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=ITDSpec(index_by=("subject", "position")),
   )

   itd_value = dataset[0]["inputs"]["itd"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.ITDSpec
   :members:

ILDSpec
~~~~~~~

``ILDSpec`` requests interaural level difference values derived from each loaded
subject HRTF. Use it when samples need binaural level cues instead of raw HRTF
signals.

Example:

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, ILDSpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=ILDSpec(index_by=("subject", "position")),
   )

   ild_value = dataset[0]["inputs"]["ild"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.ILDSpec
   :members:

SHSpec
~~~~~~

``SHSpec`` requests spherical-harmonic representations derived from HRTF data.
Use it when a dataset should expose compact spatial representations rather than
raw source-grid values.

Example:

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, SHSpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=SHSpec(sh_order=4, index_by=("subject", "frequency")),
   )

   sh_value = dataset[0]["inputs"]["sh"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.SHSpec
   :members:

Resource specs
--------------

MeshSpec
~~~~~~~~

``MeshSpec`` requests mesh files associated with a subject. Use it when HRTF
samples need to be paired with scanned or synthetic geometry.

Example:

.. code-block:: python

   from hrtfpykit.datasets import SONICOM, MeshSpec

   dataset = SONICOM(
       root="datasets/sonicom",
       inputs=MeshSpec(),
       dataset_mesh_variant={"type": "scanned", "version": "watertight"},
   )

   mesh_value = dataset[0]["inputs"]["mesh"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.MeshSpec
   :members:

AnthropometrySpec
~~~~~~~~~~~~~~~~~

``AnthropometrySpec`` requests subject anthropometry values. Use it for physical
head, ear, or body measurements that should be paired with acoustic data.

Example:

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, AnthropometrySpec, HRTFSpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=[HRTFSpec(), AnthropometrySpec()],
   )

   anthropometry = dataset[0]["inputs"]["anthropometry"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.AnthropometrySpec
   :members:

MetadataSpec
~~~~~~~~~~~~

``MetadataSpec`` requests general subject metadata. Metadata is separate from
anthropometry so a dataset can expose both resources at the same time.

Example:

.. code-block:: python

   from hrtfpykit.datasets import SONICOM, MetadataSpec, HRTFSpec

   dataset = SONICOM(
       root="datasets/sonicom",
       inputs=[HRTFSpec(), MetadataSpec()],
   )

   metadata = dataset[0]["inputs"]["metadata"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.MetadataSpec
   :members:

ImageSpec
~~~~~~~~~

``ImageSpec`` requests image media associated with a subject. It can be grouped
by subject or by subject and ear, depending on how the media resource is laid
out.

Example:

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, ImageSpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=ImageSpec(path="datasets/hutubs-images"),
   )

   images = dataset[0]["inputs"]["image"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.ImageSpec
   :members:

VideoSpec
~~~~~~~~~

``VideoSpec`` requests video media associated with a subject. It follows the
same resource-selection model as ``ImageSpec`` but returns video resource values.

Example:

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, VideoSpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=VideoSpec(path="datasets/hutubs-videos"),
   )

   videos = dataset[0]["inputs"]["video"]

API reference
^^^^^^^^^^^^^

.. autoclass:: hrtfpykit.datasets.VideoSpec
   :members:
