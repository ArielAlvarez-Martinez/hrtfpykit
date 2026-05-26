# Contributing to hrtfpykit

Contributions are welcome. The most useful contributions for `hrtfpykit` are
clear bug reports, focused fixes, documentation improvements, plotting examples,
SOFA convention support, HRTF processing improvements, and public dataset
pipeline integrations.

You can help by:

- Reporting an issue.
- Improving documentation or examples.
- Fixing bugs.
- Proposing a new feature.
- Improving SOFA, HRTF, plots, or datasets workflows.

## Reporting an Issue

Open an issue in the project repository:

<https://github.com/ArielAlvarez-Martinez/hrtfpykit/issues>

Please include the information that makes the problem reproducible:

- `hrtfpykit` version or commit.
- Python version and operating system.
- A small code example that shows the problem.
- The full error traceback, if there is one.
- Expected behavior and actual behavior.
- Whether the issue affects SOFA loading, HRTF processing, plots, metrics, or
  dataset construction.

Do not upload large private datasets or large SOFA files to an issue. If a file
is needed, use the smallest public or synthetic example that reproduces the
problem.

## Submitting a Pull Request

1. Create a feature branch from the current main branch.
2. Keep the change focused on one problem or one feature.
3. Follow the existing code style and public API naming patterns.
4. Update documentation when the public behavior changes.
5. Add or update examples only when they help users understand the change.
6. Run the relevant checks before opening the pull request.
7. Open the pull request with a clear description of what changed and why.

Avoid committing generated files or local data such as `docs/_build/`, `dist/`,
`build/`, local HRTF datasets, temporary wheels, or large SOFA files.

## Contribution Guidelines

Use these guidelines for code, documentation, examples, tests, and dataset
integrations:

- Keep changes focused and aligned with the public API.
- Keep HRTF, HRIR, and SOFA behavior explicit when a change touches acoustic
  data loading, conversion, validation, or saving.
- Preserve the relationship between metadata, source positions, ears, IR data,
  TF data, sample axes, and frequency bins.
- Keep time domain and frequency domain behavior consistent.
- Prefer clear errors when input data is invalid.
- Keep dataset samples explicit through specs.
- Keep inputs, targets, splits, variants, and subject resources reproducible.
- Do not assume local private paths or unavailable datasets.
- Keep documentation aligned with the current public API.
- Keep examples short and focused.


## Local Checks

Run the relevant checks before opening a pull request. For a full local pass:

```bash
python -m pytest
python -m ruff check src/hrtfpykit
python -m mypy src/hrtfpykit
```

Build the documentation:

```bash
pyhton -m sphinx -W -b html docs docs/_build/html
```

Build package distributions:

```bash
python -m build
```

Check package metadata before publishing:

```bash
python -m twine check dist/*
```

## License

By contributing, you agree that your contributions will be licensed under the
GPL 3.0 only license used by `hrtfpykit`. See [LICENSE](LICENSE) for details.
