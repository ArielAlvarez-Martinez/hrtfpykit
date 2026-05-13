<p align="center">
  <img src="docs/assets/images/hrtfpykit.png" alt="hrtfpykit" width="720">
</p>

<p align="center">
  <strong>SOFA files, HRTF objects, scientific plots, and dataset pipelines in one Python toolkit.</strong>
</p>

<p align="center">
  <img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white">
  <img alt="License GPL 3.0 only" src="https://img.shields.io/badge/license-GPL--3.0--only-blue">
  <img alt="SOFA HRTF HRIR" src="https://img.shields.io/badge/SOFA-HRTF%20%7C%20HRIR-0a9396">
  <img alt="Docs Sphinx Furo" src="https://img.shields.io/badge/docs-Sphinx%20%2B%20Furo-872ee0">
</p>

## What is hrtfpykit?

**hrtfpykit** is a **Python toolkit** for working with **Head Related Transfer
Functions (HRTFs)** stored in SOFA files. Around that core, it creates an
ecosystem of tools for **HRTF research** and **dataset pipelines** creation.

## Why hrtfpykit?

**HRTF research** often requires more than reading one SOFA file. If you have
worked with HRTFs, you have probably met the usual ritual: searching for public
datasets, discovering that every measurement setup has its own personality,
adapting HRIR arrays to different dataset layouts, and moving between scripts,
platforms, and tools with different assumptions. Datasets
such as ARI, HUTUBS, and SONICOM made this work much more accessible, especially
compared with the pre SOFA days of CSV files, spreadsheets, and heroic column
name interpretation. Even today, the workflow can still become fragmented very
quickly.

**hrtfpykit** was created to make those steps part of a clearer workflow. It
gives **researchers** a way to work with HRTFs without losing the connection
between the file, the acoustic representation, and the experiment.

## What you can do

- **Open**, inspect, validate, edit, clone, and save SOFA files.
- **Load** HRTFs as objects with synchronized IR and TF representations.
- **Select** source positions, ears, samples, and frequency bins.
- **Modify** HRTFs with transformations and acoustic processing steps.
- **Create** plots for amplitude, magnitude, spectral cues, ITD, ILD, LSD, source
  grids, spatial planes, and HRTF comparisons.
- **Build** map style dataset pipelines from public HRTF datasets.
- **Align** HRTFs with subject resources for analysis and deep learning workflows.

## Documentation

Once I've ready the public docs web site

## Installation

```bash
pip install hrtfpykit
```

For local development from the project root:

```bash
pip install -e ".[test,docs]"
```

`hrtfpykit` requires Python 3.13 or newer.

## Quick example

```python
from hrtfpykit.hrtf import load_hrtf
from hrtfpykit.datasets import HUTUBS, HRTFSpec

hrtf = load_hrtf("subject_001.sofa")

print(hrtf.IR.values.shape)
print(hrtf.TF.values.shape)
print(hrtf.Sources.get_positions().shape)

hrtf.plot_magnitude(
    positions="front",
    ear="both",
    reference="max",
)

dataset = HUTUBS(
    root="datasets/hutubs",
    inputs=HRTFSpec(
        domain="frequency",
        signal="tf_magnitude",
        index_by=("subject", "position"),
        name="magnitude",
    ),
)

sample = dataset[0]
print(sample["inputs"].keys())

```

## Citation

Forum acusticum article, arvix  or whatever

## License

`hrtfpykit` is distributed under the GPL 3.0 only license. See
[LICENSE](LICENSE) for details.
