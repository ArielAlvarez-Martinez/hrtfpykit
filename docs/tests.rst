Tests
=====

The test suite is organized around five files:

- ``tests/test_sofa.py``
- ``tests/test_hrtf.py``
- ``tests/test_plots.py``
- ``tests/test_hutubs.py``
- ``tests/test_sonicom.py``

SOFA, HRTF, and plot tests use real SOFA/HRTF files. Dataset tests use local
HUTUBS and SONICOM roots when available and skip when required resources are
missing.

Common commands
---------------

Run SOFA, HRTF, and plot tests with real files:

.. code-block:: bash

   pytest tests/test_sofa.py tests/test_hrtf.py tests/test_plots.py \
     --sofa-path /path/to/file.sofa \
     --compare-sofa-paths /path/to/a.sofa /path/to/b.sofa

Run HUTUBS tests:

.. code-block:: bash

   pytest tests/test_hutubs.py \
     --hutubs-root /path/to/hutubs \
     --subjects 5

Run SONICOM tests:

.. code-block:: bash

   pytest tests/test_sonicom.py \
     --sonicom-root /path/to/sonicom \
     --subjects 5

Run a collection-only check:

.. code-block:: bash

   pytest tests --collect-only -q

Full dataset matrices
---------------------

Use ``--full`` only when intentionally running the larger dataset matrix. Prefer
combining it with ``--subjects`` and ``-k`` to keep memory use controlled.
