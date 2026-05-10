from __future__ import annotations

from dataclasses import dataclass, field

from .default import FigureSize, Margins


@dataclass(frozen=True)
class Layout:
    """Immutable subplot layout definition used by the plotting wrapper.

    :class:`~hrtfpykit.plots.layouts.Layout` is the declarative configuration
    consumed by :class:`~hrtfpykit.plots.figure.Figure` when it creates a
    Matplotlib figure. It stores the subplot grid shape, named axis positions,
    figure size, margins, axis-sharing flags, and title offsets required by the
    plotting layer.

    The layout object does not create Matplotlib artists directly.
    :class:`~hrtfpykit.plots.figure.Figure` reads these fields, calls
    matplotlib.pyplot.subplots, flattens the returned axes, and exposes those
    axes by integer index or by the names in ``positions``. HRTF plot methods use
    concrete subclasses to select the correct arrangement for single-position
    plots, stacked impulse-response and magnitude views, side-by-side ear plots,
    and multi-position comparison panels.

    Notes
    -----
    Layouts are frozen dataclasses. To customize margins, figure size, axis
    sharing, or title spacing, create a new layout instance with the desired
    field values and pass it to :class:`~hrtfpykit.plots.figure.Figure`.

    Attributes
    ----------
    code : int
        Internal layout identifier copied to :attr:`~hrtfpykit.plots.figure.Figure.layout`.
    rows, cols : int
        Number of subplot rows and columns passed to Matplotlib.
    positions : tuple[str, ...]
        Names assigned to the flattened axes array.
        :meth:`~hrtfpykit.plots.figure.Figure.get_ax` accepts these names as
        stable subplot keys.
    figsize : FigureSize or tuple[float, float]
        Figure dimensions. :class:`~hrtfpykit.plots.default.FigureSize` values
        are converted to (width, height) before figure creation.
    margins : Margins
        Subplot margin and spacing values passed to the Matplotlib figure.
    sharex, sharey : bool
        Axis-sharing flags forwarded to matplotlib.pyplot.subplots.
    figure_title_offset : float
        Vertical offset added to margins.top when
        :class:`~hrtfpykit.plots.figure.Figure` computes the figure-level title
        position.
    subplot_title_y : float
        Subplot-title y position used when a figure-level title is also drawn.

    """

    code: int
    rows: int
    cols: int
    positions: tuple[str, ...]
    figsize: FigureSize | tuple[float, float] = field(default_factory=FigureSize)
    margins: Margins = field(default_factory=Margins)
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.07
    subplot_title_y: float = 0.92


@dataclass(frozen=True)
class Layout_1(Layout):
    """Single-axis layout for one-panel HRTF plots.

    :class:`~hrtfpykit.plots.layouts.Layout_1` defines a 1x1 subplot grid with
    the position name ``main``. It is used when a plot renders all information on
    one axis, including single-position magnitude curves, source-grid views,
    polar ITD or ILD curves, spherical-harmonic diagnostics, and single heatmap
    panels.

    Notes
    -----
    The layout can be used with regular, polar, or 3D Matplotlib projections by
    passing the desired projection to :class:`~hrtfpykit.plots.figure.Figure`.
    Axis sharing is disabled because the layout contains only one axis.

    Attributes
    ----------
    code : int
        Layout identifier 1.
    rows, cols : int
        Single-row, single-column subplot grid.
    positions : tuple[str, ...]
        Contains only ``main``.
    figure_title_offset : float
        Figure-title offset tuned for a single-panel figure.
    subplot_title_y : float
        Subplot-title y position used when a figure-level title is present.

    """

    code: int = 1
    rows: int = 1
    cols: int = 1
    positions: tuple[str, ...] = ("main",)
    figsize: FigureSize | tuple[float, float] = field(default_factory=FigureSize)
    margins: Margins = field(default_factory=Margins)
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.08
    subplot_title_y: float = 0.90


@dataclass(frozen=True)
class Layout_2Vertical(Layout):
    """Stacked two-axis layout for vertically related plot panels.

    :class:`~hrtfpykit.plots.layouts.Layout_2Vertical` defines a 2x1 subplot
    grid with the position names ``top`` and ``bottom``. It is used when two
    related views should be aligned vertically, such as an impulse response
    above its magnitude response or two selected source positions that should
    share horizontal context.

    Notes
    -----
    sharex is enabled by default so stacked panels align on the horizontal
    axis. sharey remains disabled because amplitude, magnitude, and metric
    values often require independent y-axis scaling.

    Attributes
    ----------
    code : int
        Layout identifier 21.
    rows, cols : int
        Two-row, single-column subplot grid.
    positions : tuple[str, ...]
        Contains ``top`` and ``bottom`` in flattened-axis order.
    sharex : bool
        Enabled by default for vertically aligned panels.
    sharey : bool
        Disabled by default so each panel can scale independently.
    figure_title_offset : float
        Figure-title offset tuned for stacked panels.
    subplot_title_y : float
        Subplot-title y position used when a figure-level title is present.

    """

    code: int = 21
    rows: int = 2
    cols: int = 1
    positions: tuple[str, ...] = ("top", "bottom")
    figsize: FigureSize | tuple[float, float] = field(default_factory=FigureSize)
    margins: Margins = field(default_factory=Margins)
    sharex: bool = True
    sharey: bool = False
    figure_title_offset: float = 0.08
    subplot_title_y: float = 0.90


@dataclass(frozen=True)
class Layout_2Horizontal(Layout):
    """Side-by-side two-axis layout for ear or panel comparisons.

    :class:`~hrtfpykit.plots.layouts.Layout_2Horizontal` defines a 1x2 subplot
    grid with the position names ``left`` and ``right``. HRTF plotting functions
    use it when two panels should be compared side by side, most commonly when
    ``ear`` is ``both`` and the plot creates one subplot for the left ear and one
    subplot for the right ear.

    Notes
    -----
    The default figure is wider than the base size to preserve readable labels,
    legends, and titles across two columns. Axis sharing is disabled by default
    because side-by-side panels may represent different ears, planes, or value
    ranges.

    Attributes
    ----------
    code : int
        Layout identifier 22.
    rows, cols : int
        Single-row, two-column subplot grid.
    positions : tuple[str, ...]
        Contains ``left`` and ``right`` in flattened-axis order.
    figsize : FigureSize or tuple[float, float]
        Defaults to :class:`~hrtfpykit.plots.default.FigureSize` with width 12
        and height 6.
    sharex, sharey : bool
        Disabled by default for independent side-by-side panels.
    figure_title_offset : float
        Figure-title offset tuned for the horizontal layout.
    subplot_title_y : float
        Subplot-title y position used when a figure-level title is present.

    """

    code: int = 22
    rows: int = 1
    cols: int = 2
    positions: tuple[str, ...] = ("left", "right")
    figsize: FigureSize | tuple[float, float] = field(
        default_factory=lambda: FigureSize(width=12, height=6)
    )
    margins: Margins = field(default_factory=Margins)
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.08
    subplot_title_y: float = 0.98


@dataclass(frozen=True)
class Layout_3(Layout):
    """Four-axis layout for multi-position HRTF plot grids.

    :class:`~hrtfpykit.plots.layouts.Layout_3` defines a 2x2 subplot grid with
    positions named ``top_left``, ``top_right``, ``bottom_left``, and
    ``bottom_right``. Plot methods use it when up to four source positions or
    comparison panels should be shown in one figure.

    The layout intentionally provides four axes even when a caller only needs
    three panels. In that case, plot methods populate the first three flattened
    axes and hide the remaining axis through :meth:`~hrtfpykit.plots.figure.Figure.hide_unused_axes`.

    Notes
    -----
    The default figure is larger than the base size to keep subplot titles,
    ticks, legends, and colorbars readable across a 2x2 grid. Axis sharing is
    disabled so each source position or metric panel can use its own scale.

    Attributes
    ----------
    code : int
        Layout identifier 3.
    rows, cols : int
        Two-row, two-column subplot grid.
    positions : tuple[str, ...]
        Contains ``top_left``, ``top_right``, ``bottom_left``, and
        ``bottom_right`` in flattened-axis order.
    figsize : FigureSize or tuple[float, float]
        Defaults to :class:`~hrtfpykit.plots.default.FigureSize` with width 10
        and height 7.
    sharex, sharey : bool
        Disabled by default for independent multi-position panels.
    figure_title_offset : float
        Figure-title offset tuned for the 2x2 layout.
    subplot_title_y : float
        Subplot-title y position used when a figure-level title is present.

    """

    code: int = 3
    rows: int = 2
    cols: int = 2
    positions: tuple[str, ...] = ("top_left", "top_right", "bottom_left", "bottom_right")
    figsize: FigureSize | tuple[float, float] = field(
        default_factory=lambda: FigureSize(width=10, height=7)
    )
    margins: Margins = field(default_factory=Margins)
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.08
    subplot_title_y: float = 0.98
