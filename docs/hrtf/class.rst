HRTF
====

.. autoclass:: hrtfpykit.hrtf.HRTF
   :members:
   :show-inheritance:

Composed interfaces
-------------------

:class:`~hrtfpykit.hrtf.HRTF` keeps the loaded HRTF/HRIR state in one object
and exposes related data and operations through composed interface properties.
In normal use, users do not create those interface objects directly. They load
or receive an HRTF instance and access each working surface from that instance:
:attr:`~hrtfpykit.hrtf.HRTF.IR` for the :class:`~hrtfpykit.hrtf.domain.IR`
object with time-domain HRIR values, sample-rate metadata,
:attr:`~hrtfpykit.hrtf.domain.IR.ir_duration`,
:meth:`~hrtfpykit.hrtf.domain.IR.get_itd`,
:meth:`~hrtfpykit.hrtf.domain.IR.get_ild`, and
:meth:`~hrtfpykit.hrtf.domain.IR.get_rms`; :attr:`~hrtfpykit.hrtf.HRTF.TF` for
the :class:`~hrtfpykit.hrtf.domain.TF` object with frequency-domain HRTF values,
frequency bins, :attr:`~hrtfpykit.hrtf.domain.TF.magnitude`,
:attr:`~hrtfpykit.hrtf.domain.TF.phase`, :attr:`~hrtfpykit.hrtf.domain.TF.real`,
and :attr:`~hrtfpykit.hrtf.domain.TF.imag`; :attr:`~hrtfpykit.hrtf.HRTF.Sources`
for the :class:`~hrtfpykit.hrtf.sources.Sources` object with
:meth:`~hrtfpykit.hrtf.sources.Sources.get_positions`, coordinate conversion,
named position queries, and selection state; and :attr:`~hrtfpykit.hrtf.HRTF.transform`
for the :class:`~hrtfpykit.hrtf.transforms.Transform` object with immutable
processing operations.

Plotting is intentionally separate from the core HRTF class. Use
:mod:`hrtfpykit.plots` functions such as :func:`~hrtfpykit.plots.plot_magnitude`
or :func:`~hrtfpykit.plots.plot_source_grid` and pass the loaded HRTF object as
the first argument. This keeps visualization in the plotting layer while still
using the current IR, TF, and source-grid state of the object.



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

