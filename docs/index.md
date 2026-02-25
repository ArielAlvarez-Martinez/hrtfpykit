# hrtfpykit docs

`hrtfpykit` is a lightweight toolkit for loading and working with HRTF datasets.

## What is included

- SOFA file loading utilities
- HRTF magnitude extraction with FFT
- Basic visualization helpers
- HRTF transformation tools

## Quick start

```python
from hrtfpykit.hrtf_loader import load_hrtf

hrtf_mag, source_dirs, freqs, fs = load_hrtf("hrtf.sofa")
```

## Next pages

- `docs/development.md` for local setup and tests
