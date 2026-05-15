Specs
=====

Specs describe what a dataset sample should contain. They are small
configuration objects passed to dataset constructors through the ``inputs`` and
``target`` arguments of classes such as :class:`~hrtfpykit.datasets.HUTUBS` and
:class:`~hrtfpykit.datasets.SONICOM`. A spec does not load files by itself.
Instead, the dataset reads the specs during construction and uses them to decide
which resource families are required, which subjects can be included, how rows
should be indexed, and which values should be returned when a sample is
requested.

Each spec defines one requested value. Acoustic specs such as
:class:`~hrtfpykit.datasets.HRTFSpec`, :class:`~hrtfpykit.datasets.ITDSpec`,
:class:`~hrtfpykit.datasets.ILDSpec`, and :class:`~hrtfpykit.datasets.SHSpec`
select HRTF-derived data, spatial subsets, ears, frequency bins, samples, or
derived cues. Resource specs such as :class:`~hrtfpykit.datasets.MeshSpec`,
:class:`~hrtfpykit.datasets.AnthropometrySpec`,
:class:`~hrtfpykit.datasets.MetadataSpec`, :class:`~hrtfpykit.datasets.ImageSpec`,
and :class:`~hrtfpykit.datasets.VideoSpec` request non-acoustic resources that
are aligned with the same subject and row context.

Specs are important because they make dataset construction explicit and
reproducible. They separate the dataset storage layout from the values a model,
analysis script, or preprocessing pipeline actually needs. The same local
HUTUBS or SONICOM root can therefore produce different datasets by changing the
spec list: one dataset can expose full HRIR arrays, another can expose
position-indexed ITD values, and another can combine HRTF data with
anthropometry, mesh, image, or metadata resources. During indexing, the dataset
uses the specs to populate ``sample["inputs"]`` and ``sample["target"]`` with
consistent keys, selected axes, optional one-hot or index encodings, and any
per-value transforms requested by the spec.

.. autoclass:: hrtfpykit.datasets.HRTFSpec
   :members:

.. autoclass:: hrtfpykit.datasets.ITDSpec
   :members:

.. autoclass:: hrtfpykit.datasets.ILDSpec
   :members:

.. autoclass:: hrtfpykit.datasets.SHSpec
   :members:

.. autoclass:: hrtfpykit.datasets.MeshSpec
   :members:

.. autoclass:: hrtfpykit.datasets.AnthropometrySpec
   :members:

.. autoclass:: hrtfpykit.datasets.MetadataSpec
   :members:

.. autoclass:: hrtfpykit.datasets.ImageSpec
   :members:

.. autoclass:: hrtfpykit.datasets.VideoSpec
   :members:
