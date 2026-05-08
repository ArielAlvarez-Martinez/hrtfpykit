HRTF Class
==========

``HRTF`` is the central acoustic object in ``hrtfpykit``. It represents one
SOFA-backed HRTF/HRIR file as an object with synchronized time-domain and
frequency-domain views.

Most users create ``HRTF`` objects with ``load_hrtf`` rather than constructing
``HRTF`` directly.

Object model
------------

``Sofa``
   Backed ``SOFA`` object. This keeps the acoustic object connected to the
   original SOFA structure for metadata, convention state, and saving.

``IR``
   Time-domain impulse-response view.

``TF``
   Frequency-domain transfer-function view.

``Sources``
   Source-grid query and coordinate conversion interface.

``transform``
   Non-destructive transformation API. Transform methods return new ``HRTF``
   objects.

Typical object workflow
-----------------------

.. code-block:: python

   from hrtfpykit.hrtf import load_hrtf

   hrtf = load_hrtf("subject_001.sofa")
   selected = hrtf.select(positions=["front", "left"], ear="both")
   processed = selected.transform.apply_window("hann")

   processed.update_sofa(change_sofa_dimensions=True)
   processed.save("subject_001_processed.sofa", overwrite=True)

Selection
---------

``select`` returns a new ``HRTF`` object containing the selected source
positions, plane, ear, or IR time region.

Examples:

.. code-block:: python

   front = hrtf.select(positions="front")
   horizontal = hrtf.select(plane="horizontal", plane_angle=0.0)
   left_ear = hrtf.select(ear="left")
   cropped = hrtf.select(start_seconds=0.0, end_seconds=0.01)

SOFA synchronization
--------------------

Transforms and selections modify in-memory acoustic data. ``update_sofa`` writes
that in-memory state into the backed SOFA object. ``save`` calls
``update_sofa`` before writing to disk.

Use ``change_sofa_dimensions=True`` when selected or transformed data changes
SOFA dimension sizes, such as selecting a subset of positions or changing FFT
length.

.. autoclass:: hrtfpykit.hrtf.hrtf.HRTF
   :members:
   :show-inheritance:
