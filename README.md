# hrtfpykit

A Python toolkit for handling HRTF data.

## Installation (editable)

```bash
pip install -e .
```

For development (tests):

```bash
pip install -e .[dev]
```

## Package layout

- `hrtfpykit/`: main package code
- `hrtfs/`: sample SOFA files (project data)
- `tests/`: unit tests
- `docs/`: project docs

## Quick start

```python
from hrtfpykit.hrtf_loader import load_hrtf

hrtf_mag, source_dirs, freqs, fs = load_hrtf("hrtf.sofa")
```
