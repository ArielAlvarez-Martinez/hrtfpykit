Validation and security checks
==============================

SOFA convention validation
--------------------------

``check_sofa_against_conventions`` validates a SOFA path or open NetCDF dataset
against the local SOFA convention registry.

.. code-block:: python

   from hrtfpykit.sofa import check_sofa_against_conventions

   result = check_sofa_against_conventions("subject_001.sofa")
   print(result["convention"])

The function returns a dictionary with the resolved convention name and version.
When convention differences are found, it emits ``SOFAConventionWarning``
warnings.

Security checks
---------------

``check_sofa_security`` performs HDF5/SOFA-oriented safety checks. It can inspect
parsed SOFA attributes or, in paranoid mode, scan raw file bytes without parsing
the SOFA file.

.. code-block:: python

   from hrtfpykit.sofa import check_sofa_security

   report = check_sofa_security("subject_001.sofa", print_report=False)
   print(report["passed"])

API reference
-------------

.. autofunction:: hrtfpykit.sofa.check_sofa_against_conventions

.. autofunction:: hrtfpykit.sofa.check_sofa_security
