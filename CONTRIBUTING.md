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
- The SOFA convention involved, when relevant.
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
4. Use the `hrtfpykit` SOFA API for SOFA file handling inside HRTF workflows.
5. Update documentation when the public behavior changes.
6. Add or update examples only when they help users understand the change.
7. Run the relevant checks before opening the pull request.
8. Open the pull request with a clear description of what changed and why.

Avoid committing generated files or local data such as `docs/_build/`, `dist/`,
`build/`, local HRTF datasets, temporary wheels, or large SOFA files.

## HRTF and SOFA Contributions

For changes involving HRTF or SOFA behavior:

- Keep `SimpleFreeFieldHRIR` and `SimpleFreeFieldHRTF` behavior explicit.
- Preserve the connection between SOFA metadata, source positions, IR data, and
  TF data.
- Prefer clear errors when input data is invalid.
- Keep time domain and frequency domain behavior consistent.
- Be careful with source coordinate systems, ears, sample axes, and frequency
  bins.

For dataset contributions:

- Keep dataset samples explicit through specs.
- Keep inputs, targets, splits, variants, and subject resources reproducible.
- Do not assume local private paths or unavailable datasets.
- Keep batching behavior compatible with `collate_samples` when possible.

## Documentation and Plot Examples

Documentation should match the current public API. When adding examples:

- Use import paths that users can run.
- Keep examples short and focused.
- Avoid private helper APIs.
- Do not invent behavior that is not implemented.
- Regenerate plot images only when the documented plotting example changes.

## Local Checks

Run tests:

```bash
pytest
```

Build the documentation:

```bash
sphinx -W -b html docs docs/_build/html
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
