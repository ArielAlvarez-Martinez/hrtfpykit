hrtfpykit API
=============

Description:
------------

The hrtfpykit API is organized around the main stages of an HRTF workflow:
opening SOFA files, representing HRTF and HRIR data, inspecting acoustic
structure through plots, and building dataset pipelines for analysis or model
training. Each layer keeps the same file, source geometry, acoustic arrays, and
subject resources connected, so users can move from low level inspection to
complete experiments without changing conventions between tools.

- :doc:`sofa <sofa/index>` is the file layer. It opens, validates, edits,
  clones, summarizes, and saves SOFA files through structured Python objects.
- :doc:`hrtf <hrtf/index>` is the acoustic layer. It loads
  ``SimpleFreeFieldHRIR`` and ``SimpleFreeFieldHRTF`` files as HRTF objects with
  synchronized IR, TF, source position, transform, metric, and spherical harmonic
  workflows.
- :doc:`plots <plots/index>` is the visualization layer. It creates HRTF plots,
  comparison plots, source grid views, cue curves, spatial planes, and
  spherical harmonic reconstruction figures.
- :doc:`datasets <datasets/index>` is the dataset layer. It uses specs to turn
  public HRTF dataset resources into indexed samples, inputs, targets, splits,
  and PyTorch compatible batches.

Content:
--------

.. toctree::
   :maxdepth: 2

   sofa/index
   hrtf/index
   plots/index
   datasets/index
