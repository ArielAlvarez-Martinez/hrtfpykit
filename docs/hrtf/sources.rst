Sources
=======

``hrtf.Sources`` exposes the source grid stored in the SOFA file. It converts
source positions between supported coordinate systems and resolves position
queries to real source-grid indices.

Supported coordinate systems
----------------------------

``spherical``
   ``(azimuth, elevation, radius)``.

``cartesian``
   ``(x, y, z)``.

``lateral-polar``
   ``(lateral, polar, radius)``.

Examples
--------

Read positions:

.. code-block:: python

   from hrtfpykit.hrtf import load_hrtf

   hrtf = load_hrtf("subject_001.sofa")
   positions = hrtf.Sources.get_positions()

Inspect available angles:

.. code-block:: python

   azimuths = hrtf.Sources.get_azimuth_angles()
   elevations = hrtf.Sources.get_elevation_angles()

Resolve a named or numeric position:

.. code-block:: python

   index, real_position = hrtf.Sources.get_position_index("front")
   index, real_position = hrtf.Sources.get_position_index([90.0, 0.0, 1.0])

API reference
-------------

.. autoclass:: hrtfpykit.hrtf.sources.Sources
   :members:
   :undoc-members:
