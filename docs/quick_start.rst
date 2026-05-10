Quick Start
===========

This page gives the shortest practical workflows for the main public entry
points. The detailed reference pages explain parameters, return values, and
larger examples.

Load a SOFA file
----------------

.. code-block:: python

   from hrtfpykit.sofa import load_sofa

   sofa = load_sofa("subject_001.sofa")
   print(sofa.GlobalAttributes.get("SOFAConventions").value)
   print(sofa.Variables.get_names())

Load and inspect an HRTF
------------------------

.. code-block:: python

   from hrtfpykit.hrtf import load_hrtf

   hrtf = load_hrtf("subject_001.sofa")
   print(hrtf.IR.values.shape)
   print(hrtf.TF.frequency_bins.shape)
   print(hrtf.Sources.get_positions().shape)

Select and transform
--------------------

.. code-block:: python

   selected = hrtf.select(positions="front", ear="both")
   processed = selected.transform.apply_window("hann")
   processed.save(
       "subject_001_front_windowed.sofa",
       overwrite=True,
       change_sofa_dimensions=True,
   )

Create plots
------------

.. code-block:: python

   hrtf.plot_magnitude(positions="front", ear="both", show=True)
   hrtf.plot_source_grid(show=True)

Build a dataset sample
----------------------

.. code-block:: python

   from hrtfpykit.datasets import HUTUBS, HRTFSpec, AnthropometrySpec

   dataset = HUTUBS(
       root="datasets/hutubs",
       inputs=[
           HRTFSpec(index_by=("subject", "position")),
           AnthropometrySpec(),
       ],
       split="train",
   )

   sample = dataset[0]
   print(sample["inputs"].keys())
