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

hrtfpykit configures ARI subject identifiers from the official HRTF files
included in the checksum map. The exposed subject IDs use the ``nh`` form, such
as ``nh2``, ``nh720``, or ``nh1059``. The ARI configuration contains
263 subject HRTF files.

The ARI resources used by hrtfpykit are:

- HRTF SOFA files for acoustic data.
- ``anthro.csv`` for numeric anthropometry measurements.
- ``metadata.csv`` for subject descriptors such as sex, age, weight, and
  recording dates.

The anthropometry and metadata CSV resources are derived from the public ARI
``anthro.mat`` MATLAB file and are stored as separate tables so they can be
requested independently with :class:`~hrtfpykit.datasets.AnthropometrySpec` and
:class:`~hrtfpykit.datasets.MetadataSpec`.

**Layout and subject paths.**

ARI HRTF filenames are not generated from one common subject path template. The
configuration therefore stores a subject path map from each canonical subject ID
to its SOFA filename. Examples include ``hrtf b_nh2.sofa``,
``hrtf c_nh831.sofa``, and ``hrtf d_nh1059.sofa``.

The official ARI HRTF files are distributed in ``b``, ``c``, and ``d`` filename
groups. hrtfpykit treats the included files as one compatible ARI HRTF
collection because they share the same source grid, IR shape, and sample rate.
The dataset class does not expose a public group selector. If a workflow needs
only a subset of ARI subjects, pass ``exclude_subject_ids`` when constructing
the dataset.

ARI anthropometry uses shared measurement columns such as ``x1`` and ear
measurement columns with ``L_`` and ``R_`` prefixes. When
:class:`~hrtfpykit.datasets.AnthropometrySpec` requests ``ear="left"`` or
``ear="right"``, hrtfpykit returns the shared fields plus the fields that
match the requested ear prefix.

**Downloads.**

The built-in downloader supports the ``hrtf``, ``anthropometry``, and
``metadata`` resource groups. Set ``download=True`` to download resources before
dataset construction, and use ``download_resources`` to choose which resource
groups to fetch.

ARI does not use ``download_hrtf_variant`` or ``dataset_hrtf_variant``. The HRTF
resource family is represented by the configured subject path map.

By default, downloads verify SHA-256 checksums. ``verify_checksum=False`` skips
checksum verification when that behavior is explicitly required, but keeping
checksum verification enabled is recommended.

**References.**

- `SOFA database ARI directory <https://sofacoustics.org/data/database/ari/>`__
- `ARI HRTF database page <https://www.oeaw.ac.at/en/ari/outreach/software/hrtf-database>`__
- `ARI anthropometry measurement description <https://www.oeaw.ac.at/fileadmin/Institute/ISF/IMG/software/readme.pdf>`__
- `ARI anthropometry and metadata CSV resources <https://github.com/ArielAlvarez-Martinez/ari_anthropometry_and_metadata>`__

.. autoclass:: hrtfpykit.datasets.ARI
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__
