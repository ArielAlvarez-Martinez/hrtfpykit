Single HRTF plots
=================

Single-HRTF plots are methods on ``HRTF`` objects. They visualize the current
state of the object, including any previous selection or transformation.

Use these methods when inspecting one HRTF at a time.

Example
-------

.. code-block:: python

   from hrtfpykit.hrtf import load_hrtf

   hrtf = load_hrtf("subject_001.sofa")

   hrtf.plot_magnitude(positions="front", ear="both", show=False)
   hrtf.plot_amplitude(positions="front", ear="left", show=False)
   hrtf.plot_source_grid(show=False)

Plot groups
-----------

Signal plots
   ``plot_magnitude``, ``plot_amplitude``, and
   ``plot_amplitude_and_magnitude`` inspect the signal at selected positions and
   ears.

Plane and spectrum plots
   ``plot_spectrum_plane`` and ``plot_elevation_spectrum`` show spectral changes
   across spatial subsets.

ITD and ILD plots
   ``plot_itd_curve``, ``plot_absolute_itd``, ``plot_ild_plane``,
   ``plot_ild_curve``, and ``plot_absolute_ild`` visualize binaural timing and
   level cues.

Source-grid plots
   ``plot_source_grid`` and ``plot_plane_grid`` visualize the spatial sampling
   of the current HRTF.

Showing figures
---------------

Most plotting methods accept ``show``. Use ``show=False`` in scripts, tests, or
batch workflows when you want to control Matplotlib display manually.

API reference
-------------

.. autoclass:: hrtfpykit.plots.hrtf.HRTFPlots
   :members:
