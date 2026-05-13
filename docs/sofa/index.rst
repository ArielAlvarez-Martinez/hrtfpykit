sofa
====

The sofa API is hrtfpykit's direct interface for ``.sofa`` files.  SOFA
(``Spatially Oriented Format for Acoustics``) files organize spatial-acoustic
measurements together with declared conventions, named dimensions, data
variables, global attributes, variable attributes, source positions, receiver
information, and other metadata needed to interpret the file correctly.

The :class:`~hrtfpykit.sofa.SOFA` class is the abstraction used to inspect,
validate, edit, clone, copy, summarize, and save those files.  It gives direct
access to dimensions, global attributes, variables, and variable attributes
without making users work against a raw netCDF4 object.  The sofa API is not
limited to ``SimpleFreeFieldHRIR`` and ``SimpleFreeFieldHRTF`` files; it works
with the stable SOFA conventions registered in hrtfpykit, following the stable
convention families listed by the `SOFA project <https://www.sofaconventions.org/mediawiki/index.php/SOFA_conventions>`__.

The :doc:`hrtf API <../hrtf/index>` relies on the sofa API when it loads
``SimpleFreeFieldHRIR`` and ``SimpleFreeFieldHRTF`` files.  A loaded
:class:`~hrtfpykit.hrtf.HRTF` object keeps its backing
:class:`~hrtfpykit.sofa.SOFA` object available through
:attr:`~hrtfpykit.hrtf.HRTF.Sofa`, so HRTF workflows can still reach the
original file dimensions, variables, attributes, and convention metadata.

With this API, users can open ``.sofa`` files as
:class:`~hrtfpykit.sofa.SOFA` objects, inspect convention metadata, run
convention and safety checks, edit dimensions, variables, or attributes, create
clones or copies, build dummy SOFA objects, and save modified files.

.. toctree::
   :maxdepth: 3

   Loading SOFA files <loading>
   SOFA Class <class>
   SOFA files validation <validation>
