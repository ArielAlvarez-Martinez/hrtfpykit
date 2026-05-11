Overview
========

.. raw:: html

   <section class="hero">
     <h1>hrtfpykit</h1>
     <p>
       A Python toolkit for SOFA-backed HRTF and HRIR workflows. Load acoustic
       SOFA files, work with synchronized time and frequency-domain HRTFs,
       visualize spatial audio structure, and build indexed datasets from
       public HRTF resources.
     </p>
   </section>

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
       <p>Open, inspect, validate, edit, clone, and save SOFA files through the structured sofa API.</p>
     </div>
     <div class="doc-card">
       <strong><a href="hrtf/index.html">HRTF</a></strong>
       <p>Use SOFA-backed HRTF objects with IR/TF views, sources, transforms, metrics, and spherical harmonics.</p>
     </div>
     <div class="doc-card">
       <strong><a href="plots/index.html">Plots</a></strong>
       <p>Visualize one HRTF or compare multiple HRTFs with the plots API.</p>
     </div>
     <div class="doc-card">
       <strong><a href="datasets/index.html">Datasets</a></strong>
       <p>Build spec-driven HUTUBS and SONICOM datasets with explicit resource and variant selection.</p>
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
     <div class="workflow-step"><strong>1. Load</strong>Open a SOFA file with <code>load_sofa</code> or an acoustic object with <code>load_hrtf</code>.</div>
     <div class="workflow-step"><strong>2. Inspect</strong>Read dimensions, source positions, IR values, TF magnitudes, and metadata.</div>
     <div class="workflow-step"><strong>3. Transform</strong>Select positions, modify IR/TF data, or build dataset-level HRTF transforms.</div>
     <div class="workflow-step"><strong>4. Visualize or train</strong>Create plots or expose values through HUTUBS and SONICOM dataset specs.</div>
   </div>

Core imports
------------

.. code-block:: python

   from hrtfpykit.sofa import load_sofa
   from hrtfpykit.hrtf import load_hrtf
   from hrtfpykit.datasets import HUTUBS, SONICOM, HRTFSpec

The root package currently exposes package metadata only. Use the subpackage
imports above for user workflows.

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
