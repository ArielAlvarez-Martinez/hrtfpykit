HRTF Class
==========

.. autoclass:: hrtfpykit.hrtf.HRTF
   :members:
   :show-inheritance:

Composed interfaces
-------------------

:class:`~hrtfpykit.hrtf.HRTF` keeps the loaded HRTF/HRIR state in one object
and exposes related data and operations through composed interface properties.
In normal use, users do not create those interface objects directly.  They load
or receive an HRTF instance and access each working surface from that instance:
``hrtf.IR`` for time-domain HRIR values, sample-rate metadata, duration, and ITD
helpers; ``hrtf.TF`` for frequency-domain HRTF values, frequency bins,
magnitude, phase, real, and imaginary views; ``hrtf.Sources`` for source
positions, coordinate conversion, named positions, and selection state; and
``hrtf.transform`` for immutable processing operations.

All four access paths operate on the same parent HRTF state.  A source
selection, transform, synchronized SOFA update, metric calculation, or plot is
therefore based on the current object rather than on detached copies of the IR,
TF, source-grid, or transform logic.



.. autoclass:: hrtfpykit.hrtf.domain.IR
   :members:
   :undoc-members:



.. autoclass:: hrtfpykit.hrtf.domain.TF
   :members:
   :undoc-members:



.. autoclass:: hrtfpykit.hrtf.sources.Sources
   :members:
   :undoc-members:



.. autoclass:: hrtfpykit.hrtf.transforms.Transform
   :members:
