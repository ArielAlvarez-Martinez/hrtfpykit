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

All four access paths read the same parent SOFA state.  When no SOFA file is
loaded they return ``None``; when a file is loaded, they reflect the current
``sofa.netCDF4_dataset`` handle used by inspection, editing, summary, clone, and
save workflows.



.. autoclass:: hrtfpykit.sofa.data._Dimensions
   :members:
   :special-members: __len__, __iter__, __getitem__



.. autoclass:: hrtfpykit.sofa.data._GlobalAttributes
   :members:
   :inherited-members:
   :special-members: __len__, __iter__, __getitem__



.. autoclass:: hrtfpykit.sofa.data._Variables
   :members:
   :special-members: __len__, __iter__, __getitem__



.. autoclass:: hrtfpykit.sofa.data._VariableAttributes
   :members:
   :inherited-members:
   :special-members: __len__, __iter__, __getitem__
