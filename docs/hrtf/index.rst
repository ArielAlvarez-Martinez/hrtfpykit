HRTF-API
========

The HRTF layer is the acoustic object interface. It loads SOFA HRTF/HRIR files
as ``HRTF`` objects and keeps both time-domain impulse responses and
frequency-domain transfer functions available.

Use this layer when you want to work with acoustic data rather than raw SOFA
variables. Plotting is documented separately in :doc:`../plots/index`.

.. raw:: html

   <div class="card-grid">
     <div class="doc-card"><strong><a href="loading.html">Loading</a></strong><p>Load SimpleFreeFieldHRIR and SimpleFreeFieldHRTF files with <code>load_hrtf</code>.</p></div>
     <div class="doc-card"><strong><a href="class.html">HRTF Class</a></strong><p>Understand the SOFA-backed acoustic object, selection, saving, and synchronization model.</p></div>
     <div class="doc-card"><strong><a href="domains.html">IR and TF</a></strong><p>Use time-domain impulse responses and frequency-domain transfer functions.</p></div>
     <div class="doc-card"><strong><a href="sources.html">Sources</a></strong><p>Query source positions, angle grids, coordinate systems, and named positions.</p></div>
     <div class="doc-card"><strong><a href="transforms.html">Transforms</a></strong><p>Build non-destructive processing pipelines for HRTF and HRIR data.</p></div>
     <div class="doc-card"><strong><a href="metrics.html">Metrics</a></strong><p>Compute ITD, ILD, and log-spectral distance values.</p></div>
     <div class="doc-card"><strong><a href="spherical_harmonics.html">Spherical harmonics</a></strong><p>Represent and reconstruct HRTF magnitude data with SH utilities.</p></div>
   </div>

.. toctree::
   :maxdepth: 2

   loading
   class
   domains
   sources
   transforms
   metrics
   spherical_harmonics
