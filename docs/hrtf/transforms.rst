Transforms
==========

``hrtf.transform`` exposes non-destructive acoustic transforms. Every transform
returns a new ``HRTF`` object and leaves the source object unchanged.

This design makes it safe to branch processing pipelines and compare multiple
versions of the same original HRTF.

Example
-------

.. code-block:: python

   from hrtfpykit.hrtf import load_hrtf

   hrtf = load_hrtf("subject_001.sofa")

   front = hrtf.select(positions="front")
   windowed = front.transform.apply_window("hann")
   quieter = windowed.transform.apply_gain(-6.0, scale="db")

   quieter.save(
       "subject_001_front_quieter.sofa",
       overwrite=True,
       change_sofa_dimensions=True,
   )

Transform groups
----------------

Time-domain transforms
   Use these when changing or preprocessing HRIR values:
   ``apply_window``, ``apply_padding``, ``upsampling``, ``downsampling``,
   ``apply_fir_filter``, ``apply_iir_filter``, and ``minimum_phase``.

Frequency-domain transforms
   Use these when changing transfer-function values:
   ``modify_tf``, ``modify_magnitude``, ``modify_phase``, ``apply_gain``, and
   ``modify_fft_length``.

Directivity transforms
   ``to_ctf`` extracts a common transfer function. ``to_dtf`` extracts a
   directional transfer function.

Timing transforms
   ``add_itd`` applies a controlled interaural delay. ``delete_itd`` estimates
   and removes interaural delay from IR data.

Chaining
--------

Because every transform returns a new ``HRTF``, transforms can be chained by
assigning the result of each call.

.. code-block:: python

   processed = (
       hrtf
       .select(positions="front")
       .transform.apply_window("hann")
       .transform.modify_fft_length(1024)
   )

.. autoclass:: hrtfpykit.hrtf.transforms.Transform
   :members:
