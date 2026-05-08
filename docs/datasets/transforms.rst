Dataset HRTF transforms
=======================

``HRTFTransform`` creates reusable callables for dataset-level or spec-level HRTF
processing. The transform is applied to loaded HRTF objects before sample values
are extracted.

This is useful when every sample should be built from a consistent transformed
version of the source HRTF, for example a selected spatial subset, a windowed IR,
a DTF representation, or a modified FFT length.

How transform composition works
-------------------------------

``HRTFTransform.build`` composes transform callables in order. Each transform
receives the output of the previous transform and returns the next object in the
chain.

Example:

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, HRTFSpec, HRTFTransform

   transform = HRTFTransform.build(
       HRTFTransform.select(positions="front"),
       HRTFTransform.apply_window("hann"),
       HRTFTransform.modify_fft_length(512),
   )

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=HRTFSpec(),
       dataset_hrtf_transform=transform,
   )

Common transform families
-------------------------

Selection
   ``select`` restricts positions, planes, ears, or IR crop ranges.

Time-domain transforms
   ``apply_window``, ``apply_padding``, ``upsampling``, ``downsampling``,
   ``apply_fir_filter``, ``apply_iir_filter``, and ``minimum_phase``.

Frequency-domain transforms
   ``to_ctf``, ``to_dtf``, ``modify_tf``, ``modify_magnitude``, ``modify_phase``,
   ``apply_gain``, and ``modify_fft_length``.

ITD transforms
   ``add_itd`` and ``delete_itd``.

.. autoclass:: hrtfpykit.datasets.HRTFTransform
   :members:
