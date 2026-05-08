Convention specifications
=========================

``ConventionsManager`` manages the local SOFA convention specification registry.
It can inspect, add, delete, import, and export convention specifications.

Import
------

.. code-block:: python

   from hrtfpykit.sofa import ConventionsManager

Examples
--------

List registered convention families and versions:

.. code-block:: python

   ConventionsManager.available_conventions_specifications()

Inspect one specification:

.. code-block:: python

   spec = ConventionsManager.inspect_sofa_specification(
       "SimpleFreeFieldHRIR",
       "1.2",
   )

Export a specification:

.. code-block:: python

   ConventionsManager.export_convention_specification_json(
       "SimpleFreeFieldHRIR",
       "1.2",
       "SimpleFreeFieldHRIR-1.2.json",
   )

.. autoclass:: hrtfpykit.sofa.ConventionsManager
   :members:
   :undoc-members:
