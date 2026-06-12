SOFA
====

.. autoclass:: hrtfpykit.sofa.SOFA
   :members:
   :show-inheritance:

Composed interfaces
-------------------

:class:`~hrtfpykit.sofa.SOFA` keeps the loaded SOFA file around one netCDF4
storage handle and exposes the main SOFA file surfaces through composed
interface properties.  In normal use, users do not create those interface
objects directly.  They load or receive a SOFA instance and access each surface
from that instance: ``sofa.Dimensions`` for dimension names, sizes, and wrapped
dimension objects; ``sofa.GlobalAttributes`` for file-level convention and
application metadata; ``sofa.Variables`` for SOFA arrays such as HRTF/HRIR data,
source positions, sampling rates, and frequency vectors; and
``sofa.VariableAttributes`` for per-variable metadata such as coordinate-system
types, units, and semantic labels.

All four access paths read the same parent SOFA state through the current
``sofa.netCDF4_dataset`` handle.  The handle must be open: if no dataset is
attached, or if the dataset was closed with :meth:`~hrtfpykit.sofa.SOFA.close`,
these properties raise ``ValueError``.  File-backed SOFA objects can be reopened
with :meth:`~hrtfpykit.sofa.SOFA.open`; in-memory clones should be saved or kept
open while they are being edited.



.. autoclass:: hrtfpykit.sofa.data._Dimensions

   .. automethod:: get
   .. automethod:: get_names
   .. automethod:: get_values
   .. automethod:: get_all
   .. automethod:: summary


.. autoclass:: hrtfpykit.sofa.data._GlobalAttributes

   .. automethod:: get
   .. automethod:: get_names
   .. automethod:: get_values
   .. automethod:: get_all
   .. automethod:: summary


.. autoclass:: hrtfpykit.sofa.data._Variables

   .. automethod:: get
   .. automethod:: get_names
   .. automethod:: get_values
   .. automethod:: get_all
   .. automethod:: summary


.. autoclass:: hrtfpykit.sofa.data._VariableAttributes

   .. automethod:: get
   .. automethod:: get_names
   .. automethod:: get_values
   .. automethod:: get_all
   .. automethod:: summary
