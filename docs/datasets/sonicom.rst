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

**Implementation status.**

Last updated: 2026-05-29. SONICOM is an actively developing dataset, and
new subjects or resources can appear after a hrtfpykit release. This
implementation supports the released resources indexed by subject identifiers
``P0001`` through ``P0405``. To use newer SONICOM releases, hrtfpykit must first
be updated with the corresponding subject identifiers, resource paths, and
checksums.

**Dataset scope.**

hrtfpykit is configured for SONICOM subject identifiers ``P0001`` through
``P0405``. The built-in configuration excludes ``P0253``, ``P0258``,
``P0270``, ``P0272``, ``P0275``, and ``P0396`` before resource scanning and
split planning. Actual subject availability depends on the resource groups and
variants present under the local dataset root.

The SONICOM resources used by hrtfpykit are:

- HRTF/HRIR SOFA files for acoustic data.
- 3D scan or synthetic mesh resources.
- The official metadata table.

SONICOM HRIRs are released at 96 kHz and 24 bits, with lower-rate 44.1 kHz and
48 kHz versions also available for measured HRTFs. hrtfpykit loads these SOFA
files through the same HRTF workflow used by the rest of the package.

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

The metadata table is expected at ``metadata_and_readme/metadata.csv``.

**Downloads.**

The built-in downloader uses the SONICOM transfer URL and supports the
``metadata``, ``hrtf``, and ``mesh`` resource groups. Set ``download=True`` to
download resources before dataset construction, and use ``download_resources``
to choose which resource groups to fetch.

``download_hrtf_variant`` and ``download_mesh_variant`` control which variants
are downloaded. ``dataset_hrtf_variant`` and ``dataset_mesh_variant`` control
which local variants are scanned and used for samples. Keeping download
selection separate from dataset construction makes the selected local resources
explicit.

**References.**

- `Official SONICOM HRTF dataset page <https://www.sonicom.eu/tools-and-resources/hrtf-dataset/>`__
- `Extended SONICOM dataset announcement <https://www.sonicom.eu/the-new-and-extended-sonicom-hrtf-dataset/>`__
- `SONICOM transfer dataset root <https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/>`__

.. autoclass:: hrtfpykit.datasets.SONICOM
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__
