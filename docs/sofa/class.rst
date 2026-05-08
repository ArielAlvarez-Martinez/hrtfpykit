SOFA Class
==========

``SOFA`` is the structured file object used by ``hrtfpykit`` to represent SOFA
files. It wraps a ``netCDF4.Dataset`` and exposes the SOFA structure through
explicit object properties and editing methods.

Most users create ``SOFA`` objects with ``load_sofa``. Use ``SOFA`` directly
when creating an in-memory convention-backed object with ``SOFA.create_dummy``.

Structured access
-----------------

``Dimensions``
   Dimension wrappers with names, sizes, and unlimited-dimension status.

``GlobalAttributes``
   Global SOFA metadata.

``VariableAttributes``
   Per-variable attributes exposed by names such as ``"Data.IR:Units"``.

``Variables``
   SOFA variables with NumPy array values and attribute wrappers.

Example:

.. code-block:: python

   from hrtfpykit.sofa import load_sofa

   sofa = load_sofa("subject_001.sofa")

   dimensions = sofa.Dimensions.get_names()
   convention = sofa.GlobalAttributes.get("SOFAConventions").value
   ir = sofa.Variables.get("Data.IR").value

Editing model
-------------

The editing API is explicit. There are separate methods for dimensions, global
attributes, variable attributes, and variables. Each method validates that the
requested target exists or does not exist as appropriate.

Example:

.. code-block:: python

   clone = sofa.clone()
   clone.create_global_attribute("ProcessingNote", "checked with hrtfpykit")
   clone.modify_global_attribute("ProcessingNote", "validated")
   clone.save("subject_001_validated.sofa", overwrite=True)

Creating SOFA data
------------------

``SOFA.create_dummy`` creates an in-memory SOFA object backed by a registered
SOFA convention specification.

.. code-block:: python

   from hrtfpykit.sofa.sofa import SOFA

   sofa = SOFA.create_dummy(
       "SimpleFreeFieldHRIR",
       version="1.2",
       dim_sizes={"M": 100, "R": 2, "N": 256, "C": 3},
   )

Saving and cloning
------------------

``clone`` creates an independent in-memory copy. ``copy_with`` creates a cloned
object while applying dimension, variable, and global-attribute overrides.
``save`` writes the current object to disk.

.. autoclass:: hrtfpykit.sofa.sofa.SOFA
   :members:
   :show-inheritance:
