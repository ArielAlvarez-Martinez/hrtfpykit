Loading HRTFs
=============

``load_hrtf`` loads a SOFA file as an acoustic ``HRTF`` object.

Import
------

.. code-block:: python

   from hrtfpykit.hrtf import load_hrtf

Basic usage
-----------

.. code-block:: python

   hrtf = load_hrtf("subject_001.sofa")

   print(hrtf.SOFAConventions)
   print(hrtf.IR.values.shape)
   print(hrtf.TF.values.shape)

Supported SOFA conventions
--------------------------

``load_hrtf`` supports:

``SimpleFreeFieldHRIR``
   Reads ``Data.IR`` and ``Data.SamplingRate`` and derives ``TF``.

``SimpleFreeFieldHRTF``
   Reads ``Data.Real``, ``Data.Imag``, and ``N`` and derives ``IR``.

Parameters
----------

``path``
   SOFA file path.

``mode``
   NetCDF open mode passed to the SOFA loader.

``parallel``
   Parallel NetCDF loading flag.

``check_sofa_against_conventions``
   Runs SOFA convention checks during loading.

``fft_length``
   Optional FFT length for HRIR-to-HRTF conversion.

``mesh2hrtf_compatible`` and ``mesh2hrtf_n_shift``
   Options for Mesh2HRTF-style frequency-domain reconstruction.

.. autofunction:: hrtfpykit.hrtf.load_hrtf
