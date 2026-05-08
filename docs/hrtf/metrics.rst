Metrics
=======

The HRTF package exposes metric functions for common HRTF comparisons and
binaural features.

Import
------

.. code-block:: python

   from hrtfpykit.hrtf import itd, ild, itd_difference, ild_difference, lsd

Examples
--------

.. code-block:: python

   from hrtfpykit.hrtf import load_hrtf, itd, ild

   hrtf = load_hrtf("subject_001.sofa")

   itd_values = itd(hrtf.IR, output="samples")
   ild_values = ild(hrtf.TF)

API reference
-------------

.. autofunction:: hrtfpykit.hrtf.itd

.. autofunction:: hrtfpykit.hrtf.ild

.. autofunction:: hrtfpykit.hrtf.itd_difference

.. autofunction:: hrtfpykit.hrtf.ild_difference

.. autofunction:: hrtfpykit.hrtf.lsd
