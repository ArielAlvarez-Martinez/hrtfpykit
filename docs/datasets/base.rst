BaseDataset
===========

:class:`hrtfpykit.datasets.base.BaseDataset` is the shared dataset interface
used by concrete dataset classes such as :class:`hrtfpykit.datasets.HUTUBS` and
:class:`hrtfpykit.datasets.SONICOM`. It documents the common construction
state, resource summaries, selected subjects, acoustic context, split
information, input and target specs, and integer-indexed sample access.

User code normally instantiates a concrete dataset class. This page is the
reference for inherited dataset attributes and behavior that are common to all
dataset integrations.

.. autoclass:: hrtfpykit.datasets.base.BaseDataset
   :members:
   :special-members: __len__, __getitem__
   :show-inheritance:
