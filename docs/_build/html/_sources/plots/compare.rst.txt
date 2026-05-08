Compare HRTF plots
==================

Compare plots visualize differences or shared trends across multiple HRTF
objects. They are functions under ``hrtfpykit.plots`` rather than methods on one
``HRTF`` object.

Use compare plots when validating processing changes, comparing datasets, or
inspecting two processing variants of the same subject.

Example
-------

.. code-block:: python

   from hrtfpykit.hrtf import load_hrtf
   from hrtfpykit.plots import compare_magnitude, compare_lsd

   reference = load_hrtf("reference.sofa")
   candidate = load_hrtf("candidate.sofa")

   compare_magnitude([reference, candidate], positions="front", ear="left", show=False)
   compare_lsd(reference, candidate, ear="left", show=False)

Plot groups
-----------

Direct signal comparison
   ``compare_magnitude`` plots selected magnitude values from multiple HRTFs.

Binaural comparisons
   ``compare_absolute_itd``, ``compare_absolute_ild``, ``compare_itd_curve``,
   ``compare_ild_curve``, ``compare_itd_difference``, and
   ``compare_ild_difference`` compare timing and level cues.

Spectral distance comparisons
   ``compare_lsd`` and ``compare_lsd_plane`` compare log-spectral distance.

API reference
-------------

.. autofunction:: hrtfpykit.plots.compare_magnitude

.. autofunction:: hrtfpykit.plots.compare_absolute_itd

.. autofunction:: hrtfpykit.plots.compare_absolute_ild

.. autofunction:: hrtfpykit.plots.compare_itd_curve

.. autofunction:: hrtfpykit.plots.compare_ild_curve

.. autofunction:: hrtfpykit.plots.compare_itd_difference

.. autofunction:: hrtfpykit.plots.compare_ild_difference

.. autofunction:: hrtfpykit.plots.compare_lsd

.. autofunction:: hrtfpykit.plots.compare_lsd_plane
