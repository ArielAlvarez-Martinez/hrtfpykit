# `hrtfpykit.plots` layer

The plots layer owns visualization. It consumes `hrtfpykit.hrtf.hrtf.HRTF` objects, metrics, source grids, and spherical-harmonic results, then returns Matplotlib figures.

## Public entry points

From `src/hrtfpykit/plots/__init__.py`:

Single-HRTF plots:

- `hrtfpykit.plots.hrtf.plot_magnitude()`
- `hrtfpykit.plots.hrtf.plot_amplitude()`
- `hrtfpykit.plots.hrtf.plot_etc()`
- `hrtfpykit.plots.hrtf.plot_etc_plane()`
- `hrtfpykit.plots.hrtf.plot_spectrum_plane()`
- `hrtfpykit.plots.hrtf.plot_elevation_spectrum()`
- `hrtfpykit.plots.hrtf.plot_itd()`
- `hrtfpykit.plots.hrtf.plot_absolute_itd()`
- `hrtfpykit.plots.hrtf.plot_ild_fd()`
- `hrtfpykit.plots.hrtf.plot_ild()`
- `hrtfpykit.plots.hrtf.plot_absolute_ild()`
- `hrtfpykit.plots.hrtf.plot_source_grid()`
- `hrtfpykit.plots.hrtf.plot_plane_grid()`

Comparison plots:

- `hrtfpykit.plots.compare.compare_magnitude()`
- `hrtfpykit.plots.compare.compare_amplitude()`
- `hrtfpykit.plots.compare.compare_absolute_itd()`
- `hrtfpykit.plots.compare.compare_absolute_ild()`
- `hrtfpykit.plots.compare.compare_itd()`
- `hrtfpykit.plots.compare.compare_ild()`
- `hrtfpykit.plots.compare.compare_itd_difference()`
- `hrtfpykit.plots.compare.compare_ild_difference()`
- `hrtfpykit.plots.compare.compare_hrtf_difference()`

Spherical-harmonic diagnostics:

- `hrtfpykit.plots.sh.sht_reconstruction_comparison()`
- `hrtfpykit.plots.sh.sht_reconstruction_error()`

## Internal rendering architecture

The core plotting wrapper is `hrtfpykit.plots.figure.Figure` in `src/hrtfpykit/plots/figure.py`.

```text
hrtfpykit.plots.hrtf.* / hrtfpykit.plots.compare.* / hrtfpykit.plots.sh.*
    ↓
hrtfpykit.plots.figure.Figure(layout)
    ↓
hrtfpykit.plots.figure.Figure.get_ax(...)
    ↓
hrtfpykit.plots.figure.Figure.create_two_dimension()
hrtfpykit.plots.figure.Figure.create_heatmap()
hrtfpykit.plots.figure.Figure.create_three_dimension()
    ↓
Matplotlib artists
```

`hrtfpykit.plots.figure.Figure` composes a native Matplotlib figure and flattened axes array:

- `hrtfpykit.plots.figure.Figure.fig`
- `hrtfpykit.plots.figure.Figure.axes`
- `hrtfpykit.plots.figure.Figure.layout`
- `hrtfpykit.plots.figure.Figure.positions`
- `hrtfpykit.plots.figure.Figure.projection`

It calls `hrtfpykit.plots.figure.Figure.configure_rc()` before creating figures so hrtfpykit plots share default font and title sizes from `hrtfpykit.plots.default.RC` in `src/hrtfpykit/plots/default.py`.

## Layout objects

Layouts are frozen dataclasses in `src/hrtfpykit/plots/layouts.py`.

- `hrtfpykit.plots.layouts.Layout` is the base dataclass.
- `hrtfpykit.plots.layouts.Layout_1` is a one-panel layout with position `main`.
- `hrtfpykit.plots.layouts.Layout_2Vertical` is a stacked layout with positions `top` and `bottom`.
- `hrtfpykit.plots.layouts.Layout_2Horizontal` is a side-by-side layout with positions `left` and `right`.
- `hrtfpykit.plots.layouts.Layout_3` is a 2x2 layout with positions `top_left`, `top_right`, `bottom_left`, and `bottom_right`.

Layouts compose:

- `hrtfpykit.plots.default.FigureSize`
- `hrtfpykit.plots.default.Margins`
- axis sharing flags
- subplot names
- title offsets

High-level plot functions choose the layout based on the requested plot type, selected ears, selected positions, or comparison structure.

## Primitive renderers

Primitive renderers live in `src/hrtfpykit/plots/types.py`.

### `hrtfpykit.plots.types.TwoDimension`

`hrtfpykit.plots.types.TwoDimension.create()` validates that the axis is not 3D and delegates to `matplotlib.axes.Axes.plot()`. It is used for amplitude, magnitude, ITD, ILD, LSD, and other line-style plots.

### `hrtfpykit.plots.types.Heatmap`

`hrtfpykit.plots.types.Heatmap.create()` validates that the axis is not 3D, resolves a supported colormap, calls `matplotlib.axes.Axes.pcolormesh()`, and optionally creates a colorbar with `mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax).append_axes(...)` and `matplotlib.figure.Figure.colorbar(...)`.

Relevant class attributes:

- `hrtfpykit.plots.types.Heatmap.colormaps`
- `hrtfpykit.plots.types.Heatmap.colorbar_location`
- `hrtfpykit.plots.types.Heatmap.colorbar_fraction`
- `hrtfpykit.plots.types.Heatmap.colorbar_pad`

These control heatmap colormap names, colorbar side, width, and separation.

### `hrtfpykit.plots.types.ThreeDimension`

`hrtfpykit.plots.types.ThreeDimension.create()` validates that the axis is 3D and delegates to `matplotlib.axes.Axes.scatter()`. It is used by source-grid and plane-grid plots.

## Axis and label helpers

High-level plot modules use helpers from:

- `src/hrtfpykit/plots/axis.py`
- `src/hrtfpykit/plots/axis_helpers.py`
- `src/hrtfpykit/plots/labels.py`
- `src/hrtfpykit/plots/legends.py`
- `src/hrtfpykit/plots/titles.py`

These helpers centralize coordinate labels, tick formatting, title construction, direction markers, ear legends, and value labels. This prevents every high-level plot function from hardcoding axis conventions differently.

## High-level plot logic

`src/hrtfpykit/plots/hrtf.py` contains single-HRTF plots. These functions read data from `hrtfpykit.hrtf.hrtf.HRTF.IR`, `hrtfpykit.hrtf.hrtf.HRTF.TF`, `hrtfpykit.hrtf.hrtf.HRTF.Sources`, and public metrics.

Examples:

- `hrtfpykit.plots.hrtf.plot_etc_plane()` builds an energy-time heatmap for a selected source plane.
- `hrtfpykit.plots.hrtf.plot_spectrum_plane()` builds a frequency-by-angle heatmap from `hrtfpykit.hrtf.hrtf.HRTF.TF` magnitude values.
- `hrtfpykit.plots.hrtf.plot_absolute_itd()` uses ITD values and source positions to render a horizontal-plane cue map.
- `hrtfpykit.plots.hrtf.plot_source_grid()` and `hrtfpykit.plots.hrtf.plot_plane_grid()` use `hrtfpykit.hrtf.sources.Sources.get_positions()` and 3D rendering.

`src/hrtfpykit/plots/compare.py` contains comparison plots. These functions consume multiple HRTFs or metric differences and use the same `hrtfpykit.plots.figure.Figure`, layout, primitive, axis, and label system.

`src/hrtfpykit/plots/sh.py` contains SH-specific diagnostics and uses `hrtfpykit.utils.sh.sht_error()` from the HRTF/SH utilities.

## Logic example: render a spectrum-plane heatmap

This example shows the plotting workflow for a single-HRTF heatmap. The plots
layer consumes an already loaded `hrtfpykit.hrtf.hrtf.HRTF`; it does not load SOFA files and does
not mutate acoustic arrays.

```python
from hrtfpykit.hrtf import load_hrtf
from hrtfpykit.plots import plot_spectrum_plane

hrtf = load_hrtf("subject.sofa")
fig = plot_spectrum_plane(
    hrtf,
    ear="left",
    plane="median",
    freq_max=18000.0,
)
fig.savefig("spectrum_median_left.png", dpi=150)
```

Calling flow:

```text
hrtfpykit.plots.hrtf.plot_spectrum_plane(
    hrtf, ear="left", plane="median", freq_max=18000.0
)
    -> hrtfpykit.plots.axis.AzimuthAnglesAxis.get_range_mode(
           range_mode=azimuth_range_mode
       )
    -> hrtfpykit.plots.default.Margins()
    -> hrtfpykit.plots.layouts.Layout_1(
           figsize=hrtfpykit.plots.layouts.Layout_1().figsize,
           margins=resolved_margins,
       )
       or hrtfpykit.plots.layouts.Layout_2Horizontal(
           figsize=hrtfpykit.plots.layouts.Layout_2Horizontal().figsize,
           margins=resolved_margins,
       )
    -> hrtfpykit.plots.figure.Figure(resolved_layout)
       -> hrtfpykit.plots.figure.Figure.create(layout, projection=None)
          -> hrtfpykit.plots.figure.Figure.configure_rc()
          -> matplotlib.pyplot.subplots(...)
          -> matplotlib.figure.Figure.subplots_adjust(...)

    -> hrtfpykit.utils.planes.get_median_plane(
           hrtf=hrtf, plane_angle=plane_angle, angle_unit="degrees"
       )
       or hrtfpykit.utils.planes.get_horizontal_plane(...) when plane="horizontal"
    -> hrtfpykit.utils.coordinates.get_source_positions(
           sources=hrtf.Sources,
           coordinate_system="spherical",
           angle_unit="degrees",
       )
    -> hrtfpykit.utils.coordinates.spherical_to_lateral_polar(...)
       when plane="median"
    -> hrtfpykit.plots.axis.FrequencyLinearAxis.build(
           frequency_bins=hrtf.TF.frequency_bins,
           freq_min=freq_min,
           freq_max=freq_max,
           margin_ratio=0.0,
       )
       or hrtfpykit.plots.axis.FrequencyLogAxis.build(...) when x_axis="log"
    -> hrtfpykit.hrtf.domain.TF.magnitude
    -> hrtfpykit.utils.dsp.magnitude_to_db(...) when unit="db"

    -> hrtfpykit.plots.figure.Figure.get_ax(figure, "main")
    -> hrtfpykit.plots.figure.Figure.create_heatmap(figure, ...)
       -> hrtfpykit.plots.types.Heatmap.create(...)
          -> matplotlib.axes.Axes.pcolormesh(...)
          -> mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax)
          -> mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes(...)
          -> matplotlib.figure.Figure.colorbar(...)
    -> matplotlib.axes.Axes.margins(x=0.0, y=0.0)
    -> hrtfpykit.plots.axis.FrequencyLinearAxis.apply(...)
       or hrtfpykit.plots.axis.FrequencyLogAxis.apply(...)
    -> hrtfpykit.plots.axis.PolarAnglesAxis.apply(...) for median plane
       or hrtfpykit.plots.axis.AzimuthAnglesAxis.transform_values(...)
       -> hrtfpykit.plots.axis.AzimuthAnglesAxis.apply(...) for horizontal plane
    -> hrtfpykit.plots.titles.Titles.create_subplots_titles(...) when show_titles=True
    -> hrtfpykit.plots.titles.Titles.create_figure_title(
           ..., hrtfpykit.plots.titles.Titles.create_plane_title(...)
       ) when show_titles=True
    -> matplotlib.pyplot.show() when show=True
    -> return matplotlib.figure.Figure
```

What each step does:

1. `hrtfpykit.plots.hrtf.plot_spectrum_plane()` normalizes `azimuth_range_mode` through
   `hrtfpykit.plots.axis.AzimuthAnglesAxis.get_range_mode()` and creates `hrtfpykit.plots.default.Margins()`.
2. It chooses `hrtfpykit.plots.layouts.Layout_1` for a single selected ear or `hrtfpykit.plots.layouts.Layout_2Horizontal` for
   `ear="both"`, then constructs `hrtfpykit.plots.figure.Figure(resolved_layout)`. The `hrtfpykit.plots.figure.Figure`
   constructor calls `hrtfpykit.plots.figure.Figure.create()`, which calls `hrtfpykit.plots.figure.Figure.configure_rc()`,
   `matplotlib.pyplot.subplots(...)`, and `matplotlib.figure.Figure.subplots_adjust(...)`.
3. Plane selection is explicit: `hrtfpykit.utils.planes.get_median_plane(...)` for `plane="median"`
   or `hrtfpykit.utils.planes.get_horizontal_plane(...)` for `plane="horizontal"`. Source coordinates
   are read with `hrtfpykit.utils.coordinates.get_source_positions(sources=hrtf.Sources, ...)`; median-plane
   angles are derived with `hrtfpykit.utils.coordinates.spherical_to_lateral_polar(...)`.
4. Frequency limits such as `freq_max=18000.0` are resolved by
   `hrtfpykit.plots.axis.FrequencyLinearAxis.build(...)` or `hrtfpykit.plots.axis.FrequencyLogAxis.build(...)`, then used
   to create the frequency mask applied to `hrtfpykit.hrtf.domain.TF.frequency_bins`.
5. Spectrum values come from `hrtfpykit.hrtf.domain.TF.magnitude`. For `unit="db"`, values are
   converted with `hrtfpykit.utils.dsp.magnitude_to_db(...)`.
6. For each subplot, `hrtfpykit.plots.figure.Figure.get_ax(figure, subplot_position)` resolves the target axis.
   Horizontal-plane values are first normalized with
   `hrtfpykit.plots.axis.AzimuthAnglesAxis.transform_values(...)`; median-plane values are used as
   polar-angle values.
7. `hrtfpykit.plots.figure.Figure.create_heatmap(figure, ...)` delegates drawing to `hrtfpykit.plots.types.Heatmap.create(...)`,
   which calls `matplotlib.axes.Axes.pcolormesh(...)`, creates the colorbar axis with
   `mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax).append_axes(...)`, and attaches the colorbar with
   `matplotlib.figure.Figure.colorbar(...)`.
8. Axis and title calls are explicit: `hrtfpykit.plots.axis.FrequencyLinearAxis.apply(...)` or
   `hrtfpykit.plots.axis.FrequencyLogAxis.apply(...)`, `hrtfpykit.plots.axis.PolarAnglesAxis.apply(...)` or
   `hrtfpykit.plots.axis.AzimuthAnglesAxis.apply(...)`, `hrtfpykit.plots.titles.Titles.create_subplots_titles(...)`, and
   `hrtfpykit.plots.titles.Titles.create_figure_title(..., hrtfpykit.plots.titles.Titles.create_plane_title(...))`.

The same architecture is used by other heatmap-style functions such as
`hrtfpykit.plots.hrtf.plot_etc_plane()`, `hrtfpykit.plots.hrtf.plot_elevation_spectrum()`, and
`hrtfpykit.plots.compare.compare_hrtf_difference()`. Line plots use `hrtfpykit.plots.figure.Figure.create_two_dimension()` and
`hrtfpykit.plots.types.TwoDimension.create()`. Source-grid plots use `hrtfpykit.plots.figure.Figure.create_three_dimension()`
and `hrtfpykit.plots.types.ThreeDimension.create()`.

## Invariants

- Plot functions should return Matplotlib `matplotlib.figure.Figure` objects.
- Plot functions should not mutate `hrtfpykit.hrtf.domain.IR.values`, `hrtfpykit.hrtf.domain.TF.values`, or `hrtfpykit.hrtf.hrtf.HRTF.Sources`.
- Plot functions should use public HRTF/metric abstractions instead of dataset-specific resource assumptions.
- Coordinate conventions must be explicit in axis helpers and labels, not hidden in ad hoc plot code.

## Do not do this

- Do not import dataset classes into plot modules.
- Do not implement SOFA loading inside plot functions.
- Do not make plotting functions change acoustic data as a side effect.
- Do not bypass `hrtfpykit.plots.figure.Figure`, layout objects, and primitive renderers for new plot families unless there is a concrete technical reason.
