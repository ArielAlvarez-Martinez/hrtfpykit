ARI
===

ARI is a public head-related transfer function database from the Acoustic
Research Institute of the Austrian Academy of Sciences. In hrtfpykit,
:class:`~hrtfpykit.datasets.ARI` maps the ARI SOFA files and aligned subject
resources into the package's shared dataset interface.

The ARI integration is designed for workflows that need acoustic HRTF data and
optional subject resources under the same map-style dataset object. Acoustic
specs load the subject SOFA file through :func:`~hrtfpykit.hrtf.load_hrtf`, and
resource specs can request the ARI anthropometry and metadata CSV files when
those resources are present or downloaded.

**Dataset scope.**

ARI is represented in hrtfpykit as the official NH HRTF collection plus aligned
CSV subject resources. The exposed subject IDs use the ``nh`` form, such as
``nh2``, ``nh720``, or ``nh1059``. Acoustic specs use the HRTF SOFA files, while
:class:`~hrtfpykit.datasets.AnthropometrySpec` and
:class:`~hrtfpykit.datasets.MetadataSpec` can use the ARI anthropometry and
metadata CSV tables when those files are available locally or downloaded.

The ARI resources used by hrtfpykit are:

- HRTF SOFA files for acoustic data.
- ``anthro.csv`` for numeric anthropometry measurements.
- ``metadata.csv`` for subject descriptors such as sex, age, weight, and
  recording dates.

The anthropometry and metadata CSV resources are derived from the public ARI
``anthro.mat`` MATLAB file and are stored as separate tables so they can be
requested independently.

**Implementation status.**

The current hrtfpykit ARI configuration contains 263 checksum-backed NH HRTF
files. These files are distributed in ``b``, ``c``, and ``d`` filename groups.
Use ``dataset_hrtf_variant="NH"`` for the full configured NH collection, or
pass a dictionary such as ``{"type": "NH", "version": "b"}`` with version
``"b"``, ``"c"``, or ``"d"`` to scan one filename group.

Two official download sources are configured:

- ``sofacoustics`` provides HRTF, anthropometry, and metadata resources.
- ``sonicom-ecosystem`` provides ARI HRTF resources from the configured
  ecosystem database catalogs.

The ARI ecosystem configuration is intentionally limited to the current NH
``b``, ``c``, and ``d`` catalogs. Other ecosystem ARI-related databases are not
mixed into this dataset until their subject IDs, variants, and checksums are
modeled explicitly.

**Local resource discovery.**

Users can download ARI files through hrtfpykit or copy previously downloaded
files under ``root``. The scanner accepts the official root-level HRTF filenames
and semantic subject folders. For example, subject ``nh2`` can be discovered as:

.. code-block:: text

   hrtf b_nh2.sofa
   nh2/hrtf b_nh2.sofa
   nh2/hrtf/hrtf b_nh2.sofa
   nh2/hrtf/nh/hrtf b_nh2.sofa
   nh2/hrtf/nh/b/hrtf b_nh2.sofa

Anthropometry is discovered as ``anthro.csv``,
``anthropometry/anthro.csv``, ``anthropometry/*.csv``,
``anthro/anthro.csv``, or ``anthro/*.csv``. Metadata is discovered as
``metadata.csv``, ``metadata/metadata.csv``, or ``metadata/*.csv``.

ARI anthropometry uses shared measurement columns such as ``x1`` and ear
measurement columns with ``L_`` and ``R_`` prefixes. When
:class:`~hrtfpykit.datasets.AnthropometrySpec` requests ``ear="left"`` or
``ear="right"``, hrtfpykit returns the shared fields plus the fields that
match the requested ear prefix.

**Downloads.**

Set ``download=True`` to download selected resources before dataset construction,
and use ``download_resources`` to choose which resource groups to fetch.
``download_hrtf_variant="all"`` or ``download_hrtf_variant="NH"`` downloads the
full configured NH collection. Passing a dictionary such as ``{"type": "NH",
"version": "b"}`` with version ``"b"``, ``"c"``, or ``"d"`` downloads one ARI
filename group.

Download selection is separate from dataset construction. ``download_resources``
and ``download_hrtf_variant`` decide which official files are fetched;
``dataset_hrtf_variant``, ``inputs``, and ``target`` decide which local files are
required for samples after the download step. By default, downloads verify
SHA-256 checksums. ``verify_checksum=False`` skips checksum verification when
that behavior is explicitly required, but keeping checksum verification enabled
is recommended.

**References.**

- `SOFA database ARI directory <https://sofacoustics.org/data/database/ari/>`__
- `ARI HRTF database page <https://www.oeaw.ac.at/en/ari/outreach/software/hrtf-database>`__
- `ARI anthropometry measurement description <https://www.oeaw.ac.at/fileadmin/Institute/ISF/IMG/software/readme.pdf>`__
- `ARI anthropometry and metadata CSV resources <https://github.com/ArielAlvarez-Martinez/ari_anthropometry_and_metadata>`__

.. autoclass:: hrtfpykit.datasets.ARI
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__
