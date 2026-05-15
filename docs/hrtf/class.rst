HRTF
====

.. autoclass:: hrtfpykit.hrtf.HRTF
   :members:
   :show-inheritance:

Composed interfaces
-------------------

:class:`~hrtfpykit.hrtf.HRTF` keeps the loaded HRTF/HRIR state in one object
and exposes related data and operations through composed interface properties.
In normal use, users do not create those interface objects directly.  They load
or receive an HRTF instance and access each working surface from that instance:
:attr:`~hrtfpykit.hrtf.HRTF.IR` for the :class:`~hrtfpykit.hrtf.domain.IR`
object with time-domain HRIR values, sample-rate metadata,
:attr:`~hrtfpykit.hrtf.domain.IR.ir_duration`, and
:meth:`~hrtfpykit.hrtf.domain.IR.get_itd`; :attr:`~hrtfpykit.hrtf.HRTF.TF` for
the :class:`~hrtfpykit.hrtf.domain.TF` object with frequency-domain HRTF values,
frequency bins, :attr:`~hrtfpykit.hrtf.domain.TF.magnitude`,
:attr:`~hrtfpykit.hrtf.domain.TF.phase`, :attr:`~hrtfpykit.hrtf.domain.TF.real`,
and :attr:`~hrtfpykit.hrtf.domain.TF.imag`; :attr:`~hrtfpykit.hrtf.HRTF.Sources`
for the :class:`~hrtfpykit.hrtf.sources.Sources` object with
:meth:`~hrtfpykit.hrtf.sources.Sources.get_positions`, coordinate conversion,
named position queries, and selection state; and
:attr:`~hrtfpykit.hrtf.HRTF.transform` for the
:class:`~hrtfpykit.hrtf.transforms.Transform` object with immutable processing
operations.  Plotting methods from :class:`~hrtfpykit.plots.hrtf.HRTFPlots`
are also available on the same :class:`~hrtfpykit.hrtf.HRTF` instance.  Users
call methods such as
:meth:`plot_magnitude() <hrtfpykit.plots.hrtf.HRTFPlots.plot_magnitude>`
or
:meth:`plot_source_grid() <hrtfpykit.plots.hrtf.HRTFPlots.plot_source_grid>`
from the loaded HRTF object; they do not create an ``HRTFPlots`` object directly.

These access paths and plotting methods operate on the same parent
:class:`~hrtfpykit.hrtf.HRTF` state. A :meth:`~hrtfpykit.hrtf.HRTF.select`
call, transformation, :meth:`~hrtfpykit.hrtf.HRTF.update_sofa` synchronization,
metric calculation, or plot is therefore based on the current object rather
than on detached copies of the IR, TF, source-grid, or transform logic.



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



.. autoclass:: hrtfpykit.plots.hrtf.HRTFPlots
   :members:
