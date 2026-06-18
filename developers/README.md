# Developer technical notes

This directory is the maintainer-facing technical map for `hrtfpykit`. The purpose is to keep the internal architecture, layer responsibilities, workflows, invariants, and dependency rules explicit for future development.

The public architecture is organized around four package layers (of course in the future this can change):

- `hrtfpykit.sofa`
- `hrtfpykit.hrtf`
- `hrtfpykit.plots`
- `hrtfpykit.datasets`

## Reading order

1. [`technical_surface.md`](technical_surface.md) — public technical surface and layer map.
2. [`architecture.md`](architecture.md) — internal architecture and class relationships.
3. [`data_flow.md`](data_flow.md) — runtime workflows across layers.
4. [`dependency_rules.md`](dependency_rules.md) — allowed and forbidden dependencies.
5. Layer pages:
   - [`layers/sofa.md`](layers/sofa.md)
   - [`layers/hrtf.md`](layers/hrtf.md)
   - [`layers/plots.md`](layers/plots.md)
   - [`layers/datasets.md`](layers/datasets.md)

## Naming convention

Use the public package layer names when describing architecture:

```text
hrtfpykit.sofa
hrtfpykit.hrtf
hrtfpykit.plots
hrtfpykit.datasets
```

Internal systems such as transforms, metrics, source coordinates, dataset specs, and PyTorch integration are documented inside the relevant layer page.
