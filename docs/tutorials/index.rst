Tutorials
=========

Guided notebooks teach hrtfpykit through real workflows. They sit between the
Quick Start and the API reference: each notebook introduces one public surface,
prepares the data it needs, explains the purpose of each step, and shows the
code that performs the operation.

Run them interactively in JupyterLab, Jupyter Notebook, or VS Code with
Microsoft's official Jupyter extension. Cell-by-cell execution keeps objects,
intermediate values, and generated figures close to the code that produced
them.

The first workflows are introductory, but they are still HRTF workflows. It
helps to be comfortable running Python notebooks, reading NumPy array shapes,
and following core SOFA/HRTF terms such as SOFA files, HRIR, HRTF, source
positions, ear channels, samples, frequency bins, and sample rate. Later
material builds on that foundation with HRTF transforms, comparison metrics,
map-style dataset construction, PyTorch data loaders, batched tensors, model
architectures, training loops, validation loops, and HRTF individualization
experiments.

.. _tutorials-setup:

Set Up
------

Use a Python environment where ``hrtfpykit`` and the notebook tools are
installed. A normal Python environment is enough:

.. code-block:: bash

   pip install hrtfpykit jupyterlab ipykernel

We recommend using an isolated `conda environment
<https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html>`_.
Create the environment first, activate it, and install ``hrtfpykit`` and the
notebook tools inside it:

.. code-block:: bash

   conda create --name hrtfpykit python
   conda activate hrtfpykit
   pip install hrtfpykit jupyterlab ipykernel

After installation, open the notebooks with JupyterLab, Jupyter Notebook, or VS
Code. In VS Code, install Microsoft's official Jupyter extension and select the
environment where ``hrtfpykit`` is installed.

If notebooks are new to you, start with the official `Jupyter Notebook
installation documentation
<https://docs.jupyter.org/en/stable/install/notebook-classic.html>`_. For
details about IPython kernels and managing notebook kernels across Python
environments, see the official `IPython and IPykernel installation
documentation <https://ipython.readthedocs.io/en/stable/install/index.html>`_.

Content:
--------

.. toctree::
   :maxdepth: 1

   Starting with hrtfpykit.sofa <starting_with_hrtfpykit_sofa>
   Starting with hrtfpykit.hrtf <starting_with_hrtfpykit_hrtf>
   Starting with hrtfpykit.plots <starting_with_hrtfpykit_plots>
