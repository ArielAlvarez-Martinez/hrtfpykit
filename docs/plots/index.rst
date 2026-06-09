hrtfpykit.plots
===============

.. py:module:: hrtfpykit.plots

Description:
------------

``hrtfpykit.plots`` is hrtfpykit's visualization layer for HRTF and HRIR analysis.
It turns :class:`~hrtfpykit.hrtf.HRTF` objects, comparisons between HRTF
objects, and spherical-harmonic reconstruction results into Matplotlib figures
configured with hrtfpykit layouts, axes, labels, legends, and titles. Its role
is to make the acoustic state of the library visible: source positions, ear
channels, time samples, frequency bins, magnitude and amplitude responses,
energy-time curves, interaural cues, spectral differences, and spherical-
harmonic errors.

Single-HRTF plots are ordinary functions in this module. They receive a loaded
:class:`~hrtfpykit.hrtf.HRTF` object as their first argument and read the active
:class:`~hrtfpykit.hrtf.domain.IR`, :class:`~hrtfpykit.hrtf.domain.TF`, and
:class:`~hrtfpykit.hrtf.sources.Sources` views. That keeps figures aligned with
source selections, transforms, padding, gain changes, and domain conversions
already applied to the object without making plotting part of the core HRTF
class.

Comparison plots extend the same pattern to multiple HRTF objects. They resolve
positions, ears, and frequency ranges against each object, then show magnitude,
amplitude, ITD, ILD, and LSD relationships in a shared figure. The spherical-
harmonic plotting functions connect :doc:`sht <../hrtf/sht>` workflows back to
an HRTF object by visualizing reconstructed spectra and reconstruction error.

With these tools, users can plot one loaded HRTF, compare several HRTFs,
inspect source grids and spatial planes, review magnitude, amplitude, energy-
time, ITD, ILD, and LSD relationships, visualize spatial cue differences, and
evaluate spherical-harmonic reconstructions with the same coordinate, domain,
and plotting conventions used by the rest of hrtfpykit.

Content:
--------

.. toctree::
   :maxdepth: 1

   plot_magnitude
   plot_amplitude
   plot_etc
   plot_etc_plane
   plot_spectrum_plane
   plot_elevation_spectrum
   plot_signed_itd
   plot_abs_itd
   plot_signed_fd_ild
   plot_signed_bb_ild
   plot_abs_bb_ild
   plot_source_grid
   plot_plane_grid
   compare_magnitude
   compare_amplitude
   compare_abs_itd
   compare_abs_bb_ild
   compare_signed_itd
   compare_signed_bb_ild
   plot_abs_itd_diff
   plot_abs_bb_ild_diff
   plot_lsd
   sht_reconstruction_comparison
   sht_reconstruction_error
