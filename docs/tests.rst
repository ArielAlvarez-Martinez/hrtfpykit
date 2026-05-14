Tests
=====

The hrtfpykit tests are split into two layers:

- Fast, fixture-based tests that should be suitable for normal CI.
- Dataset tests for HUTUBS and SONICOM that may need large local resources or
  network downloads, so they are intended mainly for explicit local or deep CI
  runs.

The suite is intentionally behavioral. It does not try to benchmark runtime.
Instead, it checks whether the main SOFA, HRTF, plotting, and dataset APIs keep
working together with realistic SOFA-backed data.

Test Files
----------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - File
     - Scope
     - What it checks
   * - ``tests/test_sofa.py``
     - SOFA I/O and editing
     - Loads a real SOFA file, inspects dimensions, variables, summaries,
       convention/security checks, clones the file, creates/modifies/deletes
       dimensions, variables, global attributes and variable attributes, then
       saves and reloads the edited file.
   * - ``tests/test_hrtf.py``
     - Core ``HRTF`` behavior
     - Loads HRIR/HRTF SOFA files, validates IR/TF/source contracts, applies the
       transform API, checks immutable transform behavior, selection/cropping,
       metrics, SOFA synchronization, convention conversion, save/reload, and
       invalid argument errors.
   * - ``tests/test_plots.py``
     - Plot API smoke tests
     - Runs plotting methods with Matplotlib, checks that figures and axes are
       created, covers comparison plots, ITD/ILD/LSD plots, source grids, plane
       grids, spectrum plots, and SHT reconstruction plots.
   * - ``tests/test_integration.py``
     - Big-picture integration
     - Exercises the full workflow across loading, selecting, transforming,
       metrics, plotting, SOFA save/reload, and a temporary dataset pipeline
       that resolves all spec families.
   * - ``tests/test_hutubs.py``
     - HUTUBS dataset behavior
     - Checks HUTUBS config, download plans, checksum errors, subject limiting,
       dataset splits, spec workflows, HRTF/ITD/ILD/SH specs, media specs, mesh
       specs, anthropometry, summary output, and optional real downloads.
   * - ``tests/test_sonicom.py``
     - SONICOM dataset behavior
     - Checks SONICOM config, excluded subjects, download plans, checksum
       errors, subject limiting, dataset splits, HRTF/ITD/ILD/SH specs, mesh and
       metadata specs, summary output, and optional real downloads.

SOFA Fixtures
-------------

The fast SOFA-backed tests use a small committed fixture by default:
``tests/pp1_HRIRs_measured.sofa``. This keeps CI independent from a local
dataset checkout.

``tests/test_sofa.py``, ``tests/test_hrtf.py``, and ``tests/test_plots.py`` can
also be pointed at another SOFA file with ``--sofa-path``:

.. code-block:: bash

   python -m pytest tests/test_sofa.py tests/test_hrtf.py tests/test_plots.py \
     --sofa-path /path/to/file.sofa \
     -q -ra

``tests/test_plots.py`` needs two HRTFs for comparison plots. If
``--compare-sofa-paths`` is not provided, the same SOFA file is used twice for a
smoke test. Pass explicit files when you want to compare different HRTFs:

.. code-block:: bash

   python -m pytest tests/test_plots.py \
     --compare-sofa-paths /path/to/a.sofa /path/to/b.sofa

``tests/test_integration.py`` resolves its SOFA input in this order:

1. ``HRTFPYKIT_TEST_INTEGRATION_SOFA_PATH``
2. ``tests/fixtures/integration_hrtf.sofa``, if present
3. ``tests/pp1_HRIRs_measured.sofa``, if present
4. ``--sofa-path`` / ``HRTFPYKIT_TEST_SOFA_PATH``

Use ``HRTFPYKIT_TEST_INTEGRATION_SOFA_PATH`` when you need to force a specific
integration fixture while the default test fixture is present.

Recommended CI Commands
-----------------------

For normal CI, run the fast, no-network tests:

.. code-block:: bash

   python -m pytest tests/test_integration.py -q -ra

or run the broader SOFA-backed smoke group:

.. code-block:: bash

   python -m pytest \
     tests/test_sofa.py \
     tests/test_hrtf.py \
     tests/test_plots.py \
     tests/test_integration.py \
     -q -ra

These tests should not download HUTUBS or SONICOM resources. They only need the
small SOFA fixture committed under ``tests/``.

Running Individual Groups
-------------------------

Run one file:

.. code-block:: bash

   python -m pytest tests/test_hrtf.py -vv -ra

Run selected tests by name:

.. code-block:: bash

   python -m pytest tests/test_hrtf.py -k "select or save" -vv -ra

Run a collection-only check:

.. code-block:: bash

   python -m pytest tests --collect-only -q

Show skip reasons and warnings:

.. code-block:: bash

   python -m pytest tests/test_sonicom.py -vv -ra

Use ``-s`` only when you want to see captured stdout/stderr while tests run:

.. code-block:: bash

   python -m pytest tests/test_hutubs.py -vv -ra -s

Plot Tests
----------

Plot tests use Matplotlib's non-interactive ``Agg`` backend by default. This is
the correct mode for CI.

To display figures during local visual checks, pass ``--show`` or ``--visual``:

.. code-block:: bash

   python -m pytest tests/test_plots.py --show

Dataset Tests
-------------

HUTUBS and SONICOM tests are different from the fixture-based tests. They are
designed to validate the dataset APIs against real dataset layouts, so they
skip when the required local roots are missing. Skips are expected when you run
these files without dataset resources.

Run HUTUBS smoke tests with a local dataset root:

.. code-block:: bash

   python -m pytest tests/test_hutubs.py \
     --hutubs-root /path/to/hutubs \
     --subjects 3 \
     -q -ra

Run SONICOM smoke tests with a local dataset root:

.. code-block:: bash

   python -m pytest tests/test_sonicom.py \
     --sonicom-root /path/to/sonicom \
     --subjects 3 \
     -q -ra

By default, the dataset files use a safe subject limit of 3 if no explicit
``--subjects`` value or subject-limit environment variable is provided.
Passing ``--subjects`` is still recommended because it makes the run scope
obvious.

Use ``--full`` only for local deep checks or intentionally larger CI jobs:

.. code-block:: bash

   python -m pytest tests/test_hutubs.py \
     --hutubs-root /path/to/hutubs \
     --subjects 3 \
     --full \
     -q -ra

``--full`` expands the dataset matrix. For HUTUBS, it expands combinations,
splits, and HRTF variants. For SONICOM, it expands combinations, splits, and
HRTF variants. This can take much longer than the default smoke mode.

Download Tests
--------------

Dataset downloads are opt-in because they need network access, disk space, and
time. Use a dedicated root when testing downloads.

Run HUTUBS download validation:

.. code-block:: bash

   python -m pytest tests/test_hutubs.py \
     --hutubs-root /path/to/hutubs-download-root \
     --hutubs-download \
     --subjects 3 \
     -q -ra

Run SONICOM download validation:

.. code-block:: bash

   python -m pytest tests/test_sonicom.py \
     --sonicom-root /path/to/sonicom-download-root \
     --sonicom-download \
     --subjects 3 \
     -q -ra

When ``--hutubs-download`` or ``--sonicom-download`` is passed, pytest orders the
download test before the rest of that file. This lets the subsequent dataset
tests reuse the resources that were just downloaded in the same run.

The download tests cover every resource option at the current subject limit:

- HUTUBS: ``anthropometry``, ``hrtf``, ``mesh``, and ``all``.
- SONICOM: ``metadata``, ``hrtf``, ``mesh``, and ``all``.

``--subjects`` controls the subject-scoped download size. For example,
``--subjects 3`` downloads and validates three subject-scoped HRTF/mesh
resources plus the global resources required by the selected download mode.

Pytest Options
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Option
     - Environment variable
     - Purpose
   * - ``--sofa-path``
     - ``HRTFPYKIT_TEST_SOFA_PATH`` or ``HRTFPYKIT_SOFA_PATH``
     - SOFA file used by SOFA, HRTF, and plot tests when overriding the default
       fixture.
   * - ``--compare-sofa-paths``
     - ``HRTFPYKIT_TEST_COMPARE_SOFA_PATHS``
     - Two or more SOFA files for plot comparison tests. Environment values use
       ``os.pathsep`` between paths.
   * - ``--hutubs-root``
     - ``HUTUBS_TEST_HUTUBS_ROOT`` or ``HUTUBS_ROOT``
     - Local HUTUBS dataset root.
   * - ``--sonicom-root``
     - ``SONICOM_TEST_ROOT`` or ``SONICOM_ROOT``
     - Local SONICOM dataset root.
   * - ``--image-path``
     - ``HUTUBS_TEST_IMAGE_PATH``, ``HUTUBS_IMAGE_PATH``, or
       ``HUTUBS_IMAGE_ROOT``
     - Optional HUTUBS image/video root for media specs.
   * - ``--subjects``
     - ``HUTUBS_TEST_SUBJECT_LIMIT`` or ``SONICOM_TEST_SUBJECT_LIMIT``
     - Subject limit for HUTUBS and SONICOM tests, including download tests.
   * - ``--full``
     - ``HUTUBS_TEST_FULL`` and ``SONICOM_TEST_FULL``
     - Expands the dataset test matrices beyond smoke mode.
   * - ``--hutubs-download``
     - ``HUTUBS_TEST_DOWNLOAD``
     - Enables network download tests for HUTUBS.
   * - ``--sonicom-download``
     - ``SONICOM_TEST_DOWNLOAD``
     - Enables network download tests for SONICOM.
   * - ``--show`` / ``--visual``
     - ``HRTFPYKIT_TEST_SHOW_PLOTS``
     - Displays Matplotlib figures during plot tests instead of using only the
       headless test backend.

Troubleshooting
---------------

If a test file reports only skipped tests, run with ``-ra`` to see why:

.. code-block:: bash

   python -m pytest tests/test_hrtf.py -vv -ra

Common causes are:

- The SOFA fixture is missing from ``tests/``.
- The dataset root was not passed for HUTUBS or SONICOM.
- The dataset root exists but does not contain the requested subject/resources.
- A media spec test needs ``--image-path``.

SOFA convention and shape warnings may appear during tests. They are useful
diagnostics and do not fail the run unless pytest reports them as failures.
