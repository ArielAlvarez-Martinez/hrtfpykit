hrtfpykit
=========

``hrtfpykit`` is a Python toolkit for SOFA-backed HRTF and HRIR workflows. It
loads acoustic SOFA files, exposes HRTFs as processing objects, visualizes HRTF
structure, and builds indexed datasets from public HRTF resources.

Documentation map
-----------------

.. list-table::
   :widths: 24 76
   :header-rows: 0

   * - :doc:`SOFA <sofa/index>`
     - Load, inspect, validate, edit, clone, and save SOFA files.
   * - :doc:`HRTF <hrtf/index>`
     - Work with HRTF/HRIR objects, domain views, source grids, transforms,
       metrics, and spherical harmonics.
   * - :doc:`Plots <plots/index>`
     - Visualize one HRTF or compare multiple HRTFs.
   * - :doc:`Datasets <datasets/index>`
     - Build spec-driven HUTUBS and SONICOM datasets.
   * - :doc:`Testing <testing>`
     - Run the real-file and dataset integration tests.

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
   :caption: API documentation
   :hidden:

   sofa/index
   hrtf/index
   plots/index
   datasets/index
   testing

.. toctree::
   :maxdepth: 1
   :caption: Guides
   :hidden:

   quickstart
   tutorials/index
