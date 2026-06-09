HUTUBS
======

HUTUBS is a public head-related transfer function database released in 2019.
The release contains HRTF-related data for 96 subject entries and is distributed
through TU Berlin DepositOnce and the SOFA database. In hrtfpykit,
:class:`~hrtfpykit.datasets.HUTUBS` maps the HUTUBS file layout into the
package's shared dataset interface.

The database was published by Fabian Brinkmann, Manoj Dinakaran, Robert
Pelzer, Jan Joschka Wohlgemuth, Fabian Seipel, Daniel Voss, Peter Grosche, and
Stefan Weinzierl. The release documentation credits the Audio Communication
Group at Technische Universitaet Berlin, with collaborators from Huawei
Technologies Munich Research Centre and Sennheiser.

**Dataset scope.**

HUTUBS is represented in hrtfpykit as a 96-entry dataset family with measured
and simulated HRIR SOFA files, optional 3D head meshes, and the official
anthropometry table. hrtfpykit exposes subjects as ``pp1`` through ``pp96``.
The release documentation notes that subjects 1 and 96 are repeated
measurements of the FABIAN head and torso simulator, and subjects 22 and 88 are
repeated measurements of one human subject.

The HUTUBS resources used by hrtfpykit are:

- HRIR SOFA files for acoustic data.
- 3D head meshes in PLY format.
- The official anthropometry table.

The release documentation reports 96 HRIR sets, 93 anthropometry sets, and 58
head meshes. Resource availability can therefore differ by subject when a
dataset requests meshes or anthropometry in addition to HRIR data.

**Implementation status.**

The current hrtfpykit HUTUBS configuration exposes the released ``pp1`` through
``pp96`` subject identifiers. HRTFs can be selected with
``dataset_hrtf_variant="measured"`` or ``dataset_hrtf_variant="simulated"``.
The measured HRIR grid has 440 source positions. The simulated HRIR grid has
1730 source positions. Both variants are loaded through the same SOFA-backed
HRTF workflow in hrtfpykit.

Two official download sources are configured:

- ``sofacoustics`` provides individually addressable HRTF, mesh, and
  anthropometry files and supports subject/resource/variant filtering.
- ``tu-berlin`` downloads complete DepositOnce archives for selected resource
  families, extracts them, normalizes usable files into ``root``, and keeps the
  original ZIP files under ``archives/``. It does not support subject or HRTF
  variant filtering.

**Local resource discovery.**

Users can download HUTUBS files through hrtfpykit or copy previously downloaded
files under ``root``. HRTFs are discovered from the official root-level names and
from subject folders. For example, measured HRIRs can be discovered as:

.. code-block:: text

   pp1_HRIRs_measured.sofa
   pp1/pp1_HRIRs_measured.sofa
   pp1/hrtf/measured/pp1_HRIRs_measured.sofa

Simulated HRIRs use the same layouts with ``pp1_HRIRs_simulated.sofa``. Meshes
are discovered as:

.. code-block:: text

   pp1_3DheadMesh.ply
   pp1/pp1_3DheadMesh.ply
   pp1/mesh/pp1_3DheadMesh.ply
   pp1/mesh/default/pp1_3DheadMesh.ply

The anthropometry table is discovered as ``AntrhopometricMeasures.csv``,
``anthropometry/AntrhopometricMeasures.csv``, ``anthropometry/*.csv``,
``anthro/AntrhopometricMeasures.csv``, or ``anthro/*.csv``. HUTUBS
anthropometry uses ``L_`` and ``R_`` prefixes for left- and right-ear fields,
and hrtfpykit uses those prefixes when selecting ear-specific anthropometry
values.

**Downloads.**

Set ``download=True`` to download selected resources before dataset construction,
and use ``download_resources`` to choose the resource groups to fetch.
``download_hrtf_variant`` controls which HRTF variant is downloaded when the
selected server supports HRTF variant filtering. ``dataset_hrtf_variant``
controls which local HRTF variant is scanned and used for samples.

Download selection is separate from dataset construction. This makes it possible
to fetch resources in one step and build datasets from local files in another
step, including files copied manually from an existing HUTUBS download. By
default, downloads verify SHA-256 checksums. ``verify_checksum=False`` skips
checksum verification when that behavior is explicitly required, but keeping
checksum verification enabled is recommended.

**References.**

- `Official HUTUBS release documentation <https://www.sofaconventions.org/data/database_sofa_0.6/hutubs/Documentation.pdf>`__
- `TU Berlin DepositOnce HUTUBS record <https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960>`__
- `SOFA database HUTUBS directory <https://sofacoustics.org/data/database/hutubs/>`__

.. autoclass:: hrtfpykit.datasets.HUTUBS
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__
