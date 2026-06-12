torch
=====

``hrtfpykit.datasets.torch`` is the PyTorch integration module for dataset
workflows. It keeps PyTorch-specific batching and training-loop APIs separate
from the framework-neutral dataset classes, while still giving PyTorch users a
stable import path inside ``hrtfpykit.datasets``.

PyTorch is required for this submodule. The core ``hrtfpykit`` runtime
dependency set leaves PyTorch outside the base installation, so environments
that import ``hrtfpykit.datasets.torch`` must provide PyTorch themselves.

Import PyTorch integration APIs from this submodule explicitly:

.. code-block:: python

   from hrtfpykit.datasets.torch import collate_samples, hrtf_loss

.. toctree::
   :maxdepth: 1

   collate_samples
   hrtf_loss
