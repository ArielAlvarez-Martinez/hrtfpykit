PLOTS-API
=========

The plotting API is user-facing and separate from core HRTF processing. Use it
for inspection, comparison, debugging, and visual validation of acoustic data.

Single-HRTF plots are methods on loaded ``HRTF`` objects. Compare plots are
functions under ``hrtfpykit.plots``.

.. raw:: html

   <div class="card-grid">
     <div class="doc-card"><strong><a href="single_hrtf.html">Single HRTF plots</a></strong><p>Magnitude, amplitude, spectrum, ITD, ILD, and source-grid plots for one HRTF.</p></div>
     <div class="doc-card"><strong><a href="compare.html">Compare HRTF plots</a></strong><p>Compare magnitudes, binaural cues, and spectral distance across HRTFs.</p></div>
     <div class="doc-card"><strong><a href="spherical_harmonics.html">Spherical-harmonic plots</a></strong><p>Inspect SH reconstruction quality and reconstruction error.</p></div>
   </div>

.. toctree::
   :maxdepth: 2

   single_hrtf
   compare
   spherical_harmonics
