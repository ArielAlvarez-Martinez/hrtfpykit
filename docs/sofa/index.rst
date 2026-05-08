SOFA-API
========

The SOFA layer is the structured file interface for SOFA data. It loads SOFA
files into ``SOFA`` objects backed by ``netCDF4.Dataset`` and exposes dimensions,
global attributes, variable attributes, and variables through wrapper
collections.

Use this layer when you need direct access to the SOFA file structure rather
than the acoustic HRTF abstraction.

.. raw:: html

   <div class="card-grid">
     <div class="doc-card"><strong><a href="loading.html">Loading</a></strong><p>Open SOFA files with the public <code>load_sofa</code> entry point.</p></div>
     <div class="doc-card"><strong><a href="class.html">SOFA Class</a></strong><p>Inspect and edit dimensions, attributes, variables, clones, and saved files.</p></div>
     <div class="doc-card"><strong><a href="validation.html">Validation</a></strong><p>Check SOFA conventions and run file safety checks.</p></div>
     <div class="doc-card"><strong><a href="conventions.html">Conventions</a></strong><p>Manage local SimpleFreeFieldHRIR and SimpleFreeFieldHRTF specifications.</p></div>
   </div>

.. toctree::
   :maxdepth: 2

   loading
   class
   validation
   conventions
