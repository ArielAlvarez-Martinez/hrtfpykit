Tests
=====

The hrtfpykit test suite checks the main user workflows across SOFA files,
HRTF objects, plots, integration behavior, and public dataset classes. The fast
SOFA, HRTF, plotting, and integration tests use the committed SOFA fixture by
default, which makes them suitable for normal CI without a local dataset
checkout.

Dataset tests cover ARI, HUTUBS, and SONICOM. They validate configuration,
download planning, checksum handling, subject limiting, spec workflow behavior,
quiet construction, local resource handling, and optional network downloads.
Tests that need large dataset roots or network access are guarded by explicit
options so the default test run stays lightweight.

The suite is intentionally behavioral. It does not try to benchmark runtime.
Instead, it checks whether the SOFA, HRTF, plotting, and dataset workflows keep
working together with realistic SOFA backed data.

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
     - Full workflow integration
     - Exercises the full workflow across loading, selecting, transforming,
       metrics, plotting, SOFA save/reload, and a temporary dataset pipeline
       that resolves all spec families.
   * - ``tests/test_hutubs.py``
     - HUTUBS dataset behavior
     - Follows the shared dataset test order for configuration, download plan,
       checksum, subject limit, variant key, spec workflow, and quiet construction
       checks. It also covers HUTUBS specific anthropometry downloads, HRTF spec
       grids, real local dataset combinations, media specs, mesh specs, indexed
       samples, summary output, and optional downloads.
   * - ``tests/test_sonicom.py``
     - SONICOM dataset behavior
     - Follows the shared dataset test order for configuration, download plan,
       checksum, subject limit, variant key, spec workflow, and quiet construction
       checks. It also covers SONICOM specific metadata downloads, windowed
       checksum coverage, HRTF spec grids, real local dataset combinations, mesh
       and metadata specs, indexed samples, summary output, and optional downloads.
   * - ``tests/test_ari.py``
     - ARI dataset behavior
     - Follows the shared dataset test order for configuration, download plan,
       checksum, subject limit, variant key, spec workflow, and quiet construction
       checks. It also covers ARI HRTF path maps, anthropometry and metadata
       download plans, anthropometry ear selection, lightweight generated CSV
       resources, and optional downloads.

Dataset Test Structure
----------------------

``tests/test_ari.py``, ``tests/test_hutubs.py``, and ``tests/test_sonicom.py``
follow a shared dataset testing pattern. Each file starts with configuration,
download planning, checksum, subject limit, variant validation, and spec workflow
checks, then adds the resource checks that belong to that dataset.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Test group
     - Shared behavior
     - Dataset specific coverage
   * - Configuration checks
     - Validate dataset name, subject IDs, subject ordering, duplicate IDs, and
       configured exclusions.
     - ARI checks 263 configured subjects. HUTUBS checks 96 subjects. SONICOM
       checks 400 configured subjects and its excluded subject list.
   * - Download plan checks
     - Build no network plans, verify resources, checksums, URLs, destination
       paths, and the active subject limit.
     - HUTUBS and SONICOM check HRTF variant plans. ARI checks its HRTF path map
       and global anthropometry/metadata resources.
   * - Failure checks
     - Verify missing checksums, checksum mismatches, and unsupported variant
       keys raise clear errors.
     - HUTUBS and SONICOM cover dataset variant validation. ARI covers download
       variant validation.
   * - Spec workflow checks
     - Ensure spec normalization does not mutate user provided spec objects.
     - Each dataset uses specs that match its resource families: HUTUBS uses
       image and anthropometry specs, SONICOM uses metadata and mesh specs, and
       ARI uses anthropometry and metadata specs.
   * - Local resource checks
     - Exercise dataset construction, selected subjects, indexed samples, summary
       output, and quiet construction when the required local resources exist.
     - HUTUBS and SONICOM run larger local matrices. ARI uses generated
       lightweight CSV resources for anthropometry and metadata checks.
   * - Download execution checks
     - Run only when the dataset download flag is enabled. These tests download
       the requested resources for the selected subjects and verify the
       resulting files.
     - ARI supports ``anthropometry``, ``metadata``, ``hrtf``, and ``all``.
       HUTUBS supports ``anthropometry``, ``hrtf``, ``mesh``, and ``all``.
       SONICOM supports ``metadata``, ``hrtf``, ``mesh``, and ``all``.

SOFA Fixtures
-------------

The fast SOFA backed tests use a small committed fixture by default:
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

For normal CI, run the fast, no network tests:

.. code-block:: bash

   python -m pytest tests/test_integration.py -q -ra

or run the broader SOFA backed smoke group:

.. code-block:: bash

   python -m pytest \
     tests/test_sofa.py \
     tests/test_hrtf.py \
     tests/test_plots.py \
     tests/test_integration.py \
     -q -ra

These tests should not download ARI, HUTUBS, or SONICOM resources. They only
need the small SOFA fixture committed under ``tests/``.

The GitHub Actions CI also runs the shared no network dataset checks from
``tests/test_hutubs.py``, ``tests/test_sonicom.py``, and ``tests/test_ari.py`` in
that order. The selected dataset checks cover:

- configuration and exclusion metadata;
- HRTF download plan construction;
- subject limit handling for ``download_resources="all"``;
- missing checksums and checksum mismatches;
- unsupported variant key errors;
- spec workflow immutability;
- global resource download plans such as anthropometry or metadata;
- SONICOM windowed checksum coverage.

Those checks do not require local dataset roots and do not download network
resources.

Running Individual Groups
-------------------------

Run one file:

.. code-block:: bash

   python -m pytest tests/test_hrtf.py -vv -ra

Run selected tests by name:

.. code-block:: bash

   python -m pytest tests/test_hrtf.py -k "select or save" -vv -ra

Run a collection only check:

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

ARI, HUTUBS, and SONICOM use the same dataset test pattern for fast checks:
configuration validity, download plan construction, subject limit behavior,
checksum failures, invalid variant key errors, spec workflow immutability, and
quiet construction behavior. These checks are no network tests and are the
dataset checks used in normal CI.

HUTUBS and SONICOM include local resource matrix tests for real dataset
layouts. Those tests validate splits, variants, selected subjects, acoustic
specs, non acoustic specs, summaries, indexed samples, and subject length for
ear indexed rows. They skip when the required local roots or media folders are
missing.

ARI includes checks for its HRTF path map, anthropometry and metadata download
plans, anthropometry ear selection, and generated local CSV resources.

Run ARI checks:

.. code-block:: bash

   python -m pytest tests/test_ari.py \
     --subjects 3 \
     -q -ra

Pass ``--ari-root`` when you want ARI downloads or an existing local ARI root to
be used by a run.

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
``--subjects`` value or dataset specific subject limit environment variable is
provided.
Passing ``--subjects`` is still recommended because it makes the run scope
obvious.

Use ``--full`` only for local deep checks or intentionally larger CI jobs:

.. code-block:: bash

   python -m pytest tests/test_hutubs.py \
     --hutubs-root /path/to/hutubs \
     --subjects 3 \
     --full \
     -q -ra

``--full`` expands dataset matrices for files that define a full mode. HUTUBS
and SONICOM use it to expand combinations, splits, and HRTF variants. This can
take much longer than the default smoke mode.

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

Run ARI download validation:

.. code-block:: bash

   python -m pytest tests/test_ari.py \
     --ari-root /path/to/ari-download-root \
     --ari-download \
     --subjects 3 \
     -q -ra

When ``--hutubs-download``, ``--sonicom-download``, or ``--ari-download`` is
passed, the corresponding network download test is enabled. HUTUBS and SONICOM
run the download test before the rest of the file so local resource tests can
reuse the files downloaded in the same run.

The download tests cover every resource option for the selected subjects:

- ARI: ``anthropometry``, ``metadata``, ``hrtf``, and ``all``.
- HUTUBS: ``anthropometry``, ``hrtf``, ``mesh``, and ``all``.
- SONICOM: ``metadata``, ``hrtf``, ``mesh``, and ``all``.

``--subjects`` controls the subject scoped download size. For example,
``--subjects 3`` downloads and validates three subject scoped HRTF or mesh
resources plus the global resources required by the selected download mode. ARI
uses subject scoped HRTF jobs plus global anthropometry and metadata jobs.

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
   * - ``--ari-root``
     - ``ARI_TEST_ROOT`` or ``ARI_ROOT``
     - Local ARI dataset root.
   * - ``--image-path``
     - ``HUTUBS_TEST_IMAGE_PATH``, ``HUTUBS_IMAGE_PATH``, or
       ``HUTUBS_IMAGE_ROOT``
     - Optional HUTUBS image/video root for media specs.
   * - ``--subjects``
     - ``HUTUBS_TEST_SUBJECT_LIMIT``, ``SONICOM_TEST_SUBJECT_LIMIT``, or
       ``ARI_TEST_SUBJECT_LIMIT``
     - Subject limit for ARI, HUTUBS, and SONICOM tests, including download
       tests.
   * - ``--full``
     - ``HUTUBS_TEST_FULL``, ``SONICOM_TEST_FULL``, and ``ARI_TEST_FULL``
     - Enables dataset full mode flags. HUTUBS and SONICOM use this to expand
       matrix combinations.
   * - ``--hutubs-download``
     - ``HUTUBS_TEST_DOWNLOAD``
     - Enables network download tests for HUTUBS.
   * - ``--sonicom-download``
     - ``SONICOM_TEST_DOWNLOAD``
     - Enables network download tests for SONICOM.
   * - ``--ari-download``
     - ``ARI_TEST_DOWNLOAD``
     - Enables network download tests for ARI.
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
- The dataset root was not passed for a local resource or download dataset run.
- The dataset root exists but does not contain the requested subject/resources.
- A media spec test needs ``--image-path``.

SOFA convention and shape warnings may appear during tests. They are useful
diagnostics and do not fail the run unless pytest reports them as failures.
