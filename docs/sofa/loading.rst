Loading SOFA files
==================

``load_sofa`` is the canonical entry point for opening a SOFA file.

Import
------

.. code-block:: python

   from hrtfpykit.sofa import load_sofa

Basic usage
-----------

.. code-block:: python

   sofa = load_sofa("subject_001.sofa")

   conventions = sofa.GlobalAttributes.get("SOFAConventions").value
   variables = sofa.Variables.get_names()

Parameters
----------

``path``
   Path to the ``.sofa`` file.

``mode``
   NetCDF open mode. The default is ``"r"`` for read-only access. Use ``"r+"``
   only when controlled in-place editing is required.

``parallel``
   Passed to the NetCDF backend for parallel loading support.

``check_sofa_against_conventions``
   If ``True``, validates the loaded file against the declared SOFA convention
   and emits SOFA convention warnings when the file differs from the registered
   convention specification.

Return value
------------

``load_sofa`` returns a :class:`hrtfpykit.sofa.sofa.SOFA` object.

API reference
-------------

.. autofunction:: hrtfpykit.sofa.load_sofa
