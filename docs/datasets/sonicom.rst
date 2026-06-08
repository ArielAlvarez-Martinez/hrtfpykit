SONICOM
=======

SONICOM is a public head-related transfer function dataset created within the
SONICOM project for spatial audio and immersive audio research. The public
dataset page describes measured HRTFs together with related subject resources
such as 3D models and depth images. In hrtfpykit,
:class:`~hrtfpykit.datasets.SONICOM` maps the SONICOM folder layout into the
package's shared dataset interface.

The dataset paper was published by Isaac Engel, Rapolas Daugintis, Thibault
Vicente, Aidan O. T. Hogg, Johan Pauwels, Arnaud J. Tournier, and Lorenzo
Picinali in the Journal of the Audio Engineering Society. The official SONICOM
pages and publications should be read as release snapshots: the original public
dataset page describes HRTF data measured from 200 subjects, and the 2025
extended dataset announcement describes additional measured participants,
synthetic HRTFs generated from processed 3D scans, and continued work to expand
the dataset.

**Dataset scope.**

SONICOM is represented in hrtfpykit as a subject-indexed dataset family with
measured and synthetic HRTF SOFA files, scanned and synthetic mesh resources,
and the official metadata table. The dataset is actively evolving upstream, so
hrtfpykit documents the implemented subject/resource scope separately from the
broader SONICOM project and ecosystem pages.

The SONICOM resources used by hrtfpykit are:

- HRTF/HRIR SOFA files for acoustic data.
- 3D scan or synthetic mesh resources.
- The official metadata table.

SONICOM HRIRs are released at 96 kHz and 24 bits, with lower-rate 44.1 kHz and
48 kHz versions also available for measured HRTFs. hrtfpykit loads these SOFA
files through the same HRTF workflow used by the rest of the package.

**Implementation status.**

Last updated: 2026-06-05. The current hrtfpykit SONICOM configuration exposes
subject identifiers ``P0001`` through ``P0405``. The Imperial transfer server is
the most complete configured source for this implementation: it provides
``metadata``, ``hrtf``, and ``mesh`` resource groups, with six download-level
subject exclusions configured for resources that are not available from that
server: ``P0253``, ``P0258``, ``P0270``, ``P0272``, ``P0275``, and ``P0396``.

The ``sonicom-ecosystem`` server is configured separately and currently supports
``hrtf`` and ``mesh`` downloads from ecosystem database catalogs. It does not
provide the SONICOM metadata table through this downloader, and its public
catalog coverage can differ from the 405 subject identifiers configured for the
Imperial transfer layout. New SONICOM subjects or resources can appear upstream
after a hrtfpykit release; hrtfpykit must be updated with the corresponding
subject identifiers, resource paths, and checksums before those files become part
of the built-in verified configuration.

**Variants and layout.**

SONICOM provides measured HRTF variants with both sample-rate and processing
version selectors. hrtfpykit supports measured HRTF sample rates ``44100``,
``48000``, and ``96000`` with these versions:

- ``Raw``
- ``Raw_NoITD``
- ``Windowed``
- ``Windowed_NoITD``
- ``FreeFieldComp``
- ``FreeFieldComp_NoITD``
- ``FreeFieldCompMinPhase``
- ``FreeFieldCompMinPhase_NoITD``

Measured HRTF files are expected under
``{subject_id}/HRTF/HRTF/{sample_rate_label}/`` with names of the form
``{subject_id}_{version}_{sample_rate_label}.sofa``. The default HRTF
selection in hrtfpykit is ``type=measured, sample_rate=44100,
version=FreeFieldComp``.

Synthetic HRTFs use the ``synthetic`` type, the ``generic`` version, and sample
rates ``44100`` or ``48000``. They are expected under
``{subject_id}/SYNTHETIC_HRTF/`` as ``HRIR_SONICOM_{sample_rate}.sofa``.

SONICOM mesh resources are selected independently from HRTF resources. Scanned
meshes support ``raw``, ``point_cloud``, and ``watertight`` versions. Synthetic
meshes support ``preprocessed``, ``plugged``, ``graded_left``, and
``graded_right`` versions. The default mesh selection in hrtfpykit is
``type=scanned, version=watertight``.

**Local resource discovery.**

Users can download SONICOM files through hrtfpykit or copy previously downloaded
files under ``root``. Measured HRTFs are discovered from the official layout and
from semantic local alternatives. For example, the default measured HRTF for
``P0001`` can be discovered as:

.. code-block:: text

   P0001/HRTF/HRTF/44kHz/P0001_FreeFieldComp_44kHz.sofa
   P0001/P0001_FreeFieldComp_44kHz.sofa
   P0001/hrtf/measured/P0001_FreeFieldComp_44kHz.sofa
   P0001/hrtf/measured/44100/P0001_FreeFieldComp_44kHz.sofa
   P0001/hrtf/measured/44kHz/P0001_FreeFieldComp_44kHz.sofa
   P0001/hrtf/measured/FreeFieldComp/44100/P0001_FreeFieldComp_44kHz.sofa
   P0001/hrtf/measured/FreeFieldComp/44kHz/P0001_FreeFieldComp_44kHz.sofa

Synthetic HRTFs are discovered as
``P0001/SYNTHETIC_HRTF/HRIR_SONICOM_44100.sofa`` or from
``P0001/HRIR_SONICOM_44100.sofa``, ``P0001/hrtf/synthetic/HRIR_SONICOM_44100.sofa``,
and ``P0001/hrtf/synthetic/44100/HRIR_SONICOM_44100.sofa`` style layouts.

Scanned meshes are discovered from the official ``P0001/3DSCAN/...`` layout and
from ``P0001/mesh/scanned/...`` alternatives. Synthetic meshes are discovered
from ``P0001/SYNTHETIC_HRTF/...`` and ``P0001/mesh/synthetic/...`` alternatives.
Metadata is discovered as ``metadata_and_readme/metadata.csv``,
``metadata_and_readme/*.csv``, ``metadata/metadata.csv``, or ``metadata.csv``.

**Downloads.**

Set ``download=True`` to download selected resources before dataset construction,
and use ``download_resources`` to choose which resource groups to fetch.
``download_hrtf_variant`` and ``download_mesh_variant`` control which official
variants are downloaded. ``dataset_hrtf_variant`` and ``dataset_mesh_variant``
control which local variants are scanned and used for samples.

Download selection is separate from dataset construction. This is especially
important for SONICOM because the configured servers differ: ``imperial`` can
fetch metadata, HRTF, and mesh resources from direct paths, while
``sonicom-ecosystem`` fetches HRTF and mesh resources from public ecosystem
catalogs and does not provide metadata through this downloader. By default,
downloads verify SHA-256 checksums. ``verify_checksum=False`` skips checksum
verification when that behavior is explicitly required, but keeping checksum
verification enabled is recommended.

**References.**

- `Official SONICOM HRTF dataset page <https://www.sonicom.eu/tools-and-resources/hrtf-dataset/>`__
- `Extended SONICOM dataset announcement <https://www.sonicom.eu/the-new-and-extended-sonicom-hrtf-dataset/>`__
- `SONICOM transfer dataset root <https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/>`__

.. autoclass:: hrtfpykit.datasets.SONICOM
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__
