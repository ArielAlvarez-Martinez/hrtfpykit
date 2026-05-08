DATASETS-API
============

The dataset layer builds indexed datasets from local or downloadable HRTF
resources. Dataset specs define which resource values are exposed as sample
inputs and targets, while concrete datasets define resource layouts and download
behavior.

Use this layer when an HRTF resource should behave like an indexed Python
dataset for training, evaluation, analysis, or paired-resource extraction.

.. raw:: html

   <div class="card-grid">
     <div class="doc-card"><strong><a href="specs.html">Dataset specs</a></strong><p>Define sample values, resource requirements, and row indexing behavior.</p></div>
     <div class="doc-card"><strong><a href="transforms.html">Dataset HRTF transforms</a></strong><p>Apply reusable HRTF transforms before sample extraction.</p></div>
     <div class="doc-card"><strong><a href="hutubs.html">HUTUBS</a></strong><p>Build HUTUBS samples from measured or simulated HRTFs and optional resources.</p></div>
     <div class="doc-card"><strong><a href="sonicom.html">SONICOM</a></strong><p>Build SONICOM samples from measured/synthetic HRTFs, meshes, and metadata.</p></div>
   </div>

.. toctree::
   :maxdepth: 2

   specs
   transforms
   hutubs
   sonicom
