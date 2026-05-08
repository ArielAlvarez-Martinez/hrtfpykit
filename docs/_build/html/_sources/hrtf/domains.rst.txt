IR and TF domain views
======================

A loaded ``HRTF`` object exposes two acoustic domain views.

IR
--

``hrtf.IR`` stores time-domain impulse responses.

Main attributes and methods:

``values``
   NumPy array containing IR values.

``sample_rate``
   Sampling rate in Hz.

``ir_length``
   Number of samples along the last IR axis.

``ir_duration``
   IR duration in seconds.

``get_itd(...)``
   Computes interaural time difference from the current IR.

Example:

.. code-block:: python

   hrtf = load_hrtf("subject_001.sofa")
   print(hrtf.IR.sample_rate)
   print(hrtf.IR.ir_duration)
   itd_values = hrtf.IR.get_itd(output="samples")

TF
--

``hrtf.TF`` stores frequency-domain transfer functions.

Main attributes and methods:

``values``
   Complex NumPy array containing TF values.

``frequency_bins``
   One-dimensional array of frequency bins in Hz.

``magnitude``
   Linear magnitude.

``get_magnitude_db(reference=1.0)``
   Magnitude converted to dB.

``phase``
   Phase in degrees.

``real`` and ``imag``
   Real and imaginary components.

Example:

.. code-block:: python

   hrtf = load_hrtf("subject_001.sofa")
   magnitude_db = hrtf.TF.get_magnitude_db()
   phase = hrtf.TF.phase

API reference
-------------

.. autoclass:: hrtfpykit.hrtf.domain.IR
   :members:
   :undoc-members:

.. autoclass:: hrtfpykit.hrtf.domain.TF
   :members:
   :undoc-members:
