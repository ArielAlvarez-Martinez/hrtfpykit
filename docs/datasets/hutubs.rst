HUTUBS
======

``HUTUBS`` is the public dataset interface for the HUTUBS HRTF dataset. It turns
the local HUTUBS resource layout into an indexed Python dataset with shared
``BaseDataset`` behavior: subject filtering, resource intersection, split
planning, HRTF loading, and spec-driven sample extraction.

Use ``HUTUBS`` when you want HUTUBS HRTFs together with optional resources such
as anthropometry, mesh, images, or videos.

What HUTUBS provides
--------------------

HRTF variants
   HUTUBS supports measured and simulated HRTF resources. The selected value is
   controlled by ``dataset_hrtf_variant``.

Subject handling
   Subject references can be passed as canonical IDs, numeric strings, integers,
   or ``subject1`` / ``subject_1`` style aliases. Exclusions are applied before
   resource scanning and split selection.

Spec-driven samples
   ``inputs`` and ``target`` define the sample contract. The returned sample is a
   dictionary with ``"inputs"`` and ``"target"`` keys.

Summaries
   ``resources_summary()`` reports scanned resources. ``dataset_summary()``
   reports the final dataset state.

Construction example
--------------------

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, HRTFSpec, AnthropometrySpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       dataset_hrtf_variant="measured",
       inputs=[
           HRTFSpec(index_by=("subject", "position")),
           AnthropometrySpec(),
       ],
       split="train",
       split_ratio=(0.8, 0.1, 0.1),
       split_seed=0,
   )

   sample = dataset[0]
   hrtf_value = sample["inputs"]["hrtf"]
   anthropometry = sample["inputs"]["anthropometry"]

Loading one subject HRTF
------------------------

``get_subject_hrtf`` uses the same subject mapping, resource lookup, cache, and
dataset-level HRTF transform used by indexed sample extraction.

.. code-block:: python

   subject_id = dataset.selected_subjects[0]
   hrtf = dataset.get_subject_hrtf(subject_id)

Download behavior
-----------------

If ``download=True``, HUTUBS downloads the requested official resources before
constructing the dataset. Download selection is independent from dataset
construction selection:

- ``download_resources`` selects which resource families are downloaded.
- ``download_hrtf_variant`` selects which HRTF variant is downloaded.
- ``dataset_hrtf_variant`` selects which local HRTF files are scanned and used
  after download.

Example:

.. code-block:: python

   dataset = HUTUBS(
       root="datasets/hutubs",
       download=True,
       download_resources="hrtf",
       download_hrtf_variant="measured",
       dataset_hrtf_variant="measured",
       inputs=HRTFSpec(),
   )

API reference
-------------

.. autoclass:: hrtfpykit.datasets.HUTUBS
   :members:
   :show-inheritance:
