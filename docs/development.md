# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run tests

```bash
pytest -q
```

## Notes

- Runtime dependency for SOFA loading is the `sofa` package.
- If you change the package API, update `hrtfpykit/__init__.py` and docs examples.
