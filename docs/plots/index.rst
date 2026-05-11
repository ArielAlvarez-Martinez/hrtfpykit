plots
=====

The plots API is hrtfpykit's visualization layer for HRTF and HRIR analysis.
It turns :class:`~hrtfpykit.hrtf.HRTF` objects, comparisons between HRTF
objects, and spherical-harmonic reconstruction results into Matplotlib figures
configured with hrtfpykit layouts, axes, labels, legends, and titles.  Its role
is to make the acoustic state of the library visible: source positions, ear
channels, time samples, frequency bins, magnitude and amplitude responses,
interaural cues, spectral differences, and spherical-harmonic errors.

The plots API is closely tied to the :doc:`hrtf API <../hrtf/index>`.
Single-object plotting methods are grouped in
:class:`~hrtfpykit.plots.hrtf.HRTFPlots` and inherited by
:class:`~hrtfpykit.hrtf.HRTF`, so figures are created from the same object
that was loaded, selected, transformed, or synchronized.  Those methods read the
active :class:`~hrtfpykit.hrtf.domain.IR`,
:class:`~hrtfpykit.hrtf.domain.TF`, and
:class:`~hrtfpykit.hrtf.sources.Sources` views, which keeps plots aligned with
the current HRTF state instead of a detached copy of the data.

Comparison plots extend that same idea to multiple HRTF objects.  They resolve
positions, ears, and frequency ranges against each object, then show magnitude,
amplitude, ITD, ILD, and LSD relationships in a shared figure.  The
spherical-harmonic plotting functions connect
:doc:`spherical-harmonic workflows <../hrtf/spherical_harmonics>` back to the
original HRTF object by visualizing reconstructed spectra and reconstruction
error.

With this API, users can plot the state of one loaded
:class:`~hrtfpykit.hrtf.HRTF` object, compare several HRTFs, inspect source
grids, review magnitude, amplitude, ITD, ILD, and LSD relationships, visualize
spatial cue differences, and evaluate spherical-harmonic reconstructions with
the same coordinate, domain, and plotting conventions used by the rest of
hrtfpykit.

.. toctree::
   :maxdepth: 3

   HRTF plots <hrtf_plots>
   HRTF comparison plots <compare_plots>
   Spherical-harmonic plots <sh_plots>
