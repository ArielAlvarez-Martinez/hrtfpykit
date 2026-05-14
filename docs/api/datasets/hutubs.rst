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

The HUTUBS release contains subject identifiers 1 through 96. hrtfpykit exposes
these subjects as ``pp1`` through ``pp96``. The release documentation notes
that subjects 1 and 96 are repeated measurements of the FABIAN head and torso
simulator, and subjects 22 and 88 are repeated measurements of one human
subject.

The HUTUBS resources used by hrtfpykit are:

- HRIR SOFA files for acoustic data.
- 3D head meshes in PLY format.
- The official anthropometry table.

The release documentation reports 96 HRIR sets, 93 anthropometry sets, and 58
head meshes. Resource availability can therefore differ by subject when a
dataset requests meshes or anthropometry in addition to HRIR data.

**Variants and layout.**

HUTUBS provides two HRIR SOFA variants for each subject:

- ``measured``: files named ``{subject_id}_HRIRs_measured.sofa``.
- ``simulated``: files named ``{subject_id}_HRIRs_simulated.sofa``.

Use ``dataset_hrtf_variant`` to choose which local HRTF variant the dataset
loads during construction. The default is ``measured``.

The measured HRIR grid has 440 source positions. The simulated HRIR grid has
1730 source positions. Both variants are loaded through the same SOFA-backed
HRTF workflow in hrtfpykit.

The default mesh layout uses ``{subject_id}_3DheadMesh.ply``. The
anthropometry table is ``AntrhopometricMeasures.csv``. HUTUBS anthropometry
uses ``L_`` and ``R_`` prefixes for left- and right-ear fields, and hrtfpykit
uses those prefixes when selecting ear-specific anthropometry values.

**Downloads.**

The built-in downloader uses the HUTUBS SOFA database URL and supports the
``hrtf``, ``mesh``, and ``anthropometry`` resource groups. Set ``download=True``
to download resources before dataset construction, and use
``download_resources`` to choose the resource groups to fetch.

``download_hrtf_variant`` controls which HRTF variant is downloaded.
``dataset_hrtf_variant`` controls which local HRTF variant is scanned and used
for samples. Keeping these settings separate makes it possible to download and
construct datasets in explicit steps.

**References.**

- `Official HUTUBS release documentation <https://www.sofaconventions.org/data/database_sofa_0.6/hutubs/Documentation.pdf>`__
- `SOFA database HUTUBS directory <https://sofacoustics.org/data/database/hutubs/>`__

.. autoclass:: hrtfpykit.datasets.HUTUBS
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__
