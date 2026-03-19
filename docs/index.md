# hrtfpykit docs

`hrtfpykit` is a lightweight toolkit for loading and working with HRTFs.

## What is included

- SOFA file loading and inspection
- SOFA convention validation and security checks
- HRTF transforms and visualization helpers

## Quick start

```python
from hrtfpykit.sofa import SOFA, check_sofa_against_conventions

sofa = SOFA.load("hrtf.sofa")
print(sofa.summary())
check_sofa_against_conventions(sofa)
```

## Next pages

- `docs/hrtfpykit-sofa.md` for the SOFA API reference
- `docs/development.md` for local setup and tests
