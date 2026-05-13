Overview
========

.. image:: assets/images/hrtfpykit.png
   :alt: hrtfpykit overview
   :class: overview-hero-image
   :width: 100%

What is hrtfpykit?
------------------

**hrtfpykit** is a **Python toolkit** for working with **Head Related Transfer Functions
(HRTFs)** stored in SOFA files. Around that core, it creates an ecosystem of
tools for **HRTF research** and **dataset pipelines** creation.


Why hrtfpykit?
--------------

**HRTF research** often requires more than reading one SOFA file. If you have
worked with HRTFs, you have probably met the usual ritual: searching for public
datasets, discovering that every measurement setup has its own personality,
reshaping HRIRs until the arrays finally agree, and jumping between scripts,
platforms, and tools that were never designed to speak to each other. Datasets
such as ARI, HUTUBS, and SONICOM made this work much more accessible, especially
compared with the pre SOFA days of CSV files, spreadsheets, and heroic column
name interpretation. Even today, the workflow can still become fragmented very
quickly.

**hrtfpykit** was created to keep those steps in the **same ecosystem**. It gives
**researchers** a way to work with HRTFs without losing the connection between
the file, the acoustic representation, and the experiment.


What does hrtfpykit enable?
---------------------------

**hrtfpykit** can enable complete HRTF workflows, from file inspection to dataset
construction. It is designed for users who need to understand, process,
visualize, compare, and reuse HRTF data across research and deep learning
tasks.

- **Open**, inspect, validate, edit, clone, and save SOFA files.
- **Load** HRTFs as acoustic objects with time domain and frequency domain views.
- **Select** source positions, ears, samples, and frequency bins.
- **Modify** HRTFs through transformations, domain conversions, and acoustic
  processing steps.
- **Generate** plots to inspect spectral cues, magnitude, amplitude, ITD, LSD, and
  differences between HRTFs with comparison plots, which is especially useful for
  HRTF individualization.
- **Combine** HRTFs with subject data such as anthropometry, metadata, meshes, and
  images.
- **Create** map-style dataset pipelines for training multimodal deep learning models.
- **Build** deep learning experiments for HRTF individualization and related tasks.

What is the hrtfpykit architecture?
-----------------------------------

**hrtfpykit** is organized around a modular workflow architecture built on four main APIs. The
SOFA layer handles file structure, the HRTF layer builds the acoustic object,
the plots layer visualizes HRTF data, and the datasets layer turns public HRTF
resources into training and analysis samples.

- :doc:`sofa API </sofa/index>`: open, inspect, validate, edit, clone, and save SOFA files as
  structured Python objects.
- :doc:`hrtf API </hrtf/index>`: load SOFA files as HRTF objects with IR data, TF data, source
  positions, transforms, metrics, and spherical harmonics.
- :doc:`plots API </plots/index>`: visualize HRTF objects through spectral cues, source grids,
  binaural cues, spherical harmonic reconstructions, and comparison plots.
- :doc:`datasets API </datasets/index>`: build public HRTF dataset pipelines with explicit inputs,
  targets, variants, splits, subject resources, and batching utilities.

Content:
--------

.. toctree::
   :maxdepth: 1

   self
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: HRTFPYKIT API

   sofa/index
   hrtf/index
   plots/index
   datasets/index

.. toctree::
   :maxdepth: 1
   :caption: Guides

   tutorials/index
   tests
