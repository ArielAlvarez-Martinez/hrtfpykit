hrtfpykit.plots
===============

Description:
------------

``hrtfpykit.plots`` is hrtfpykit's visualization layer for HRTF and HRIR analysis.
It turns :class:`~hrtfpykit.hrtf.HRTF` objects, comparisons between HRTF
objects, and spherical-harmonic reconstruction results into Matplotlib figures
configured with hrtfpykit layouts, axes, labels, legends, and titles.  Its role
is to make the acoustic state of the library visible: source positions, ear
channels, time samples, frequency bins, magnitude and amplitude responses,
interaural cues, spectral differences, and spherical-harmonic errors.

This layer is closely tied to :doc:`hrtfpykit.hrtf <../hrtf/index>`.
Single-object plotting methods are accessed from loaded
:class:`~hrtfpykit.hrtf.HRTF` objects through
:class:`~hrtfpykit.plots.hrtf.HRTFPlots` and are documented with the HRTF API.
Those methods read the active :class:`~hrtfpykit.hrtf.domain.IR`,
:class:`~hrtfpykit.hrtf.domain.TF`, and
:class:`~hrtfpykit.hrtf.sources.Sources` views, which keeps plots aligned with
the current HRTF state instead of a detached copy of the data.

Comparison plots extend that same idea to multiple HRTF objects.  They resolve
positions, ears, and frequency ranges against each object, then show magnitude,
amplitude, ITD, ILD, and LSD relationships in a shared figure.  The
spherical-harmonic plotting functions connect :doc:`sht <../hrtf/sht>`
workflows back to the
original HRTF object by visualizing reconstructed spectra and reconstruction
error.

With these tools, users can plot the state of one loaded
:class:`~hrtfpykit.hrtf.HRTF` object, compare several HRTFs, inspect source
grids, review magnitude, amplitude, ITD, ILD, and LSD relationships, visualize
spatial cue differences, and evaluate spherical-harmonic reconstructions with
the same coordinate, domain, and plotting conventions used by the rest of
hrtfpykit.

Content:
--------

.. toctree::
   :maxdepth: 3

   compare_magnitude
   compare_amplitude
   compare_absolute_itd
   compare_absolute_ild
   compare_itd_curve
   compare_ild_curve
   compare_itd_difference
   compare_ild_difference
   compare_lsd
   compare_lsd_plane
   sht_reconstruction_comparison
   sht_reconstruction_error
