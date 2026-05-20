<p align="center">
  <img src="https://raw.githubusercontent.com/ArielAlvarez-Martinez/hrtfpykit/main/docs/assets/images/hrtfpykit.png" alt="hrtfpykit" width="720">
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.13+" src="https://raw.githubusercontent.com/ArielAlvarez-Martinez/hrtfpykit/main/docs/assets/badges/python.svg"></a>
  <a href="https://pypi.org/project/hrtfpykit/"><img alt="PyPI package" src="https://raw.githubusercontent.com/ArielAlvarez-Martinez/hrtfpykit/main/docs/assets/badges/pypi.svg"></a>
  <a href="https://hrtfpykit.readthedocs.io/en/stable/"><img alt="Docs Sphinx Furo" src="https://raw.githubusercontent.com/ArielAlvarez-Martinez/hrtfpykit/main/docs/assets/badges/docs.svg"></a>
  <a href="https://github.com/ArielAlvarez-Martinez/hrtfpykit/actions/workflows/ci.yml"><img alt="CI/CD workflow" src="https://raw.githubusercontent.com/ArielAlvarez-Martinez/hrtfpykit/main/docs/assets/badges/ci-cd.svg"></a>
  <a href="https://www.sofaconventions.org/mediawiki/index.php/SOFA_conventions"><img alt="SOFA HRTF HRIR" src="https://raw.githubusercontent.com/ArielAlvarez-Martinez/hrtfpykit/main/docs/assets/badges/sofa.svg"></a>
  <a href="https://github.com/ArielAlvarez-Martinez/hrtfpykit/blob/main/LICENSE"><img alt="License GPL 3.0 only" src="https://raw.githubusercontent.com/ArielAlvarez-Martinez/hrtfpykit/main/docs/assets/badges/license.svg"></a>
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

## What does hrtfpykit enable?

**hrtfpykit** can enable complete HRTF workflows, from file inspection to dataset
construction. It is designed for users who need to understand, process,
visualize, compare, and reuse HRTF data across research and deep learning tasks.

- **Open**, inspect, validate, edit, clone, and save SOFA files.
- **Load** HRTFs as acoustic objects with time domain and frequency domain views.
- **Select** source positions, ears, samples, and frequency bins.
- **Modify** HRTFs through transformations, domain conversions, and acoustic
  processing steps.
- **Generate** plots to inspect spectral cues, magnitude, amplitude, ITD, LSD, and
  differences between HRTFs with comparison plots, which is especially useful for
  HRTF individualization.
- **Combine** HRTFs with subject data such as anthropometry, metadata, meshes, and
  images.
- **Create** map-style dataset pipelines for training multimodal deep learning models.
- **Build** deep learning experiments for HRTF individualization and related tasks.

## Installation

```bash
pip install hrtfpykit
```

For local installation from source:

```bash
git clone https://github.com/ArielAlvarez-Martinez/hrtfpykit.git
cd hrtfpykit
pip install .
```

For local development from source:

```bash
git clone https://github.com/ArielAlvarez-Martinez/hrtfpykit.git
cd hrtfpykit
pip install -e ".[test,docs]"
```

`hrtfpykit` requires Python 3.13 or newer.

## Quick start

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

Forum acusticum article, arXiv  or whatever

## License

`hrtfpykit` is distributed under the GPL 3.0 only license. See
[LICENSE](LICENSE) for details.
