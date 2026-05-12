Overview
========

.. image:: assets/images/hrtfpykit.png
   :alt: hrtfpykit overview
   :class: overview-hero-image
   :width: 100%


Core documentation
------------------

.. raw:: html

   <div class="card-grid">
     <div class="doc-card">
       <strong><a href="quick_start.html">Quick Start</a></strong>
       <p>Start with the shortest complete workflows for loading, transforming, plotting, and dataset samples.</p>
     </div>
     <div class="doc-card">
       <strong><a href="sofa/index.html">SOFA</a></strong>
       <p>Open, inspect, validate, edit, clone, and save .sofa files as structured SOFA objects.</p>
     </div>
     <div class="doc-card">
       <strong><a href="hrtf/index.html">HRTF</a></strong>
       <p>Load HRTF/HRIR SOFA files as HRTF objects with IR, TF, Sources, transforms, metrics, and spherical harmonics.</p>
     </div>
     <div class="doc-card">
       <strong><a href="plots/index.html">Plots</a></strong>
       <p>Create HRTF figures, comparison plots, source-grid views, cue plots, and spherical-harmonic diagnostics.</p>
     </div>
     <div class="doc-card">
       <strong><a href="datasets/index.html">Datasets</a></strong>
       <p>Build spec-driven, map-style datasets with explicit resources, variants, splits, inputs, targets, and batching.</p>
     </div>
     <div class="doc-card">
       <strong><a href="tests.html">Tests</a></strong>
       <p>Run real-file SOFA, HRTF, plotting, HUTUBS, and SONICOM integration checks.</p>
     </div>
   </div>

Typical workflow
----------------

.. raw:: html

   <div class="workflow-strip">
     <div class="workflow-step"><strong>1. Open</strong>Use <code>load_sofa</code> for file-level work or <code>load_hrtf</code> for HRTF/HRIR workflows.</div>
     <div class="workflow-step"><strong>2. Inspect</strong>Read SOFA dimensions, metadata, variables, source positions, IR values, and TF views.</div>
     <div class="workflow-step"><strong>3. Process</strong>Select positions, transform IR/TF data, compute metrics, or build spherical-harmonic representations.</div>
     <div class="workflow-step"><strong>4. Use</strong>Create plots, build spec-driven dataset samples, or batch map-style datasets with a DataLoader.</div>
   </div>

Core imports
------------

.. code-block:: python

   from hrtfpykit.sofa import load_sofa
   from hrtfpykit.hrtf import load_hrtf
   from hrtfpykit.datasets import HUTUBS, SONICOM, HRTFSpec, collate_samples

The root package currently exposes package metadata. Use the API package imports
above for user workflows.

.. toctree::
   :maxdepth: 2

   self
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: HRTFPYKIT API

   sofa/index
   hrtf/index
   plots/index
   datasets/index

.. toctree::
   :maxdepth: 1
   :caption: Guides
   :hidden:

   tests
   tutorials/index
