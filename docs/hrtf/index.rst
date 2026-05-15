hrtfpykit.hrtf
==============

Description:
------------

``hrtfpykit.hrtf`` is the main HRTF/HRIR data layer in hrtfpykit.  It relies on
:doc:`hrtfpykit.sofa <../sofa/index>` to open, access, modify, and save SOFA files,
then represents files declared with the
``SimpleFreeFieldHRIR`` and ``SimpleFreeFieldHRTF`` conventions as
:class:`~hrtfpykit.hrtf.HRTF` objects. Each :class:`~hrtfpykit.hrtf.HRTF`
instance keeps its backing :class:`~hrtfpykit.sofa.SOFA` object available
through :attr:`~hrtfpykit.hrtf.HRTF.Sofa`, so dimensions, variables,
attributes, and metadata can still be reached from the loaded HRTF object.

HRTF and HRIR data describe the same binaural filtering behavior in different
domains: HRIR in the time domain and HRTF in the frequency domain.  This layer
keeps both representations available through coordinated
:class:`~hrtfpykit.hrtf.domain.IR` and :class:`~hrtfpykit.hrtf.domain.TF` views,
so workflows can move between impulse response operations, frequency domain
analysis, spatial selection, and file synchronization while preserving the
original source grid and metadata.

The :class:`~hrtfpykit.hrtf.HRTF` object is also the shared acoustic object
used by other hrtfpykit layers.  In workflows handled by
:doc:`hrtfpykit.plots <../plots/index>`, the current HRTF state drives the
generated figures: plots read the active :class:`~hrtfpykit.hrtf.domain.IR`,
:class:`~hrtfpykit.hrtf.domain.TF`, and
:class:`~hrtfpykit.hrtf.sources.Sources` views, so visualizations follow the
file that was loaded, the source positions that were selected, the ear channels
in use, and any transforms already applied.  In workflows handled by
:doc:`hrtfpykit.datasets <../datasets/index>`, concrete datasets load each subject
resource through hrtf before specs extract values.  That makes
:class:`~hrtfpykit.hrtf.HRTF` the object from which
:class:`~hrtfpykit.datasets.HRTFSpec`,
:class:`~hrtfpykit.datasets.ITDSpec`, :class:`~hrtfpykit.datasets.ILDSpec`, and
:class:`~hrtfpykit.datasets.SHSpec` obtain acoustic arrays, binaural cues, or
spherical-harmonic data.

With these tools, users can load compatible HRTF or HRIR SOFA files as
:class:`~hrtfpykit.hrtf.HRTF` objects, inspect source positions, select
subsets of measurements, apply processing operations, compute perceptual or
spectral metrics, build spherical-harmonic representations, synchronize modified
HRTFs with their backing SOFA objects, and pass processed HRTFs into plotting and
dataset pipelines.

Content:
--------

.. toctree::
   :maxdepth: 3

   load_hrtf <loading>
   HRTF <class>
   itd_difference
   ild_difference
   lsd
   SH
   sht
   sht_inverse
   sht_error
