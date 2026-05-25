from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, cast

from .default import FigureSize, RC
from .layouts import Layout
from .types import Heatmap, ThreeDimension, TwoDimension


class Figure:
    shared_x_visible: bool = True

    def __init__(self, layout: Layout, projection: str | None = None):
        """Create a Matplotlib figure from an hrtfpykit layout definition.

        :class:`~hrtfpykit.plots.figure.Figure` is the plotting-layer wrapper
        used by HRTF, HRIR, comparison, spherical-harmonic, and spatial-grid plot
        helpers. It converts a :class:`~hrtfpykit.plots.layouts.Layout` object
        into a Matplotlib figure, stores a flattened axes array, applies
        hrtfpykit default rcParams, and exposes small primitive-dispatch methods
        for line plots, heatmaps, and 3D scatter plots.

        The wrapper keeps subplot construction and primitive routing consistent
        across the plotting package. High-level plot functions choose a layout,
        create Figure(layout), resolve axes by index or by named position,
        draw through the wrapper methods, and then apply labels, titles, grids,
        and legends.

        Parameters
        ----------
        layout : Layout
            Layout definition containing subplot rows, columns, named positions,
            figure size, margins, axis-sharing flags, and title offsets.
        projection : str | None, default=None
            Optional Matplotlib projection passed to every subplot in the layout,
            such as ``polar`` for polar cue plots or ``3d`` for source-grid
            views.

        Returns
        -------
        None

        Notes
        -----
        The constructor calls :meth:`~hrtfpykit.plots.figure.Figure.create`,
        which updates Matplotlib rcParams through
        :meth:`~hrtfpykit.plots.figure.Figure.configure_rc`. The returned axes
        are always flattened into a one-dimensional object array so plot code
        can use a common access pattern for 1x1, stacked, side-by-side, and 2x2
        layouts.

        Attributes
        ----------
        layout : int
            Layout code copied from layout.code and used by plot methods for
            layout-aware behavior.
        positions : tuple[str, ...]
            Named subplot positions copied from layout.positions.
        projection : str | None
            Matplotlib projection used for subplot creation.
        figure_title_y : float
            Vertical position used by title helpers for figure-level titles.
        fig : matplotlib.figure.Figure
            Matplotlib figure created from the layout.
        axes : numpy.ndarray
            Flattened one-dimensional array of Matplotlib axes created for the
            layout.

        """
        self.layout = layout.code
        self.positions = layout.positions
        self.projection = projection
        self.figure_title_y = min(
            layout.margins.top + layout.figure_title_offset,
            0.98,
        )
        self.fig, self.axes = self.create(layout, projection=projection)

    @staticmethod
    def configure_rc() -> None:
        """Apply hrtfpykit's default Matplotlib text rcParams.

        The plotting layer calls this method before creating figures so font
        sizes for axes, tick labels, legends, and figure titles remain
        consistent across all generated HRTF documentation and analysis plots.
        Values are read from :class:`~hrtfpykit.plots.default.RC` and written to
        matplotlib.pyplot.rcParams.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        configure_rc updates global Matplotlib rcParams for the active Python
        process. It intentionally only changes text-size and title-weight
        settings used by hrtfpykit plots.

        """
        rc = RC()
        plt.rcParams.update(
            {
                "font.size": rc.default,
                "axes.titlesize": rc.axis_title,
                "axes.labelsize": rc.axis_labels,
                "xtick.labelsize": rc.ticks,
                "ytick.labelsize": rc.ticks,
                "legend.fontsize": rc.legend,
                "legend.title_fontsize": rc.legend_title,
                "figure.titlesize": rc.fig_title,
                "figure.titleweight": "bold",
            }
        )

    @staticmethod
    def create(
        layout: Layout,
        projection: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Create a Matplotlib figure and flattened axes from a layout object.

        The method resolves the figure size, applies default rcParams, creates
        subplots using the layout's rows, columns, sharing flags, and optional
        projection, then applies the layout margins. Matplotlib's nested axes
        output is converted into a one-dimensional object array for consistent
        indexing by higher-level plot functions.

        Each returned axis receives internal title-position attributes used by
        :class:`~hrtfpykit.plots.titles.Titles` when subplot titles and
        figure-level titles are combined.

        Parameters
        ----------
        layout : Layout
            Layout definition with grid size, figure size, margins, axis-sharing
            flags, and title offsets.
        projection : str | None, default=None
            Optional subplot projection passed as Matplotlib subplot_kw for
            every axis in the layout.

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            Matplotlib figure and flattened object array of subplot axes.

        Notes
        -----
        :class:`~hrtfpykit.plots.default.FigureSize` layout values are converted
        to a (width, height) tuple before figure creation. Tuple figure sizes are
        passed through unchanged.

        """
        Figure.configure_rc()
        if isinstance(layout.figsize, FigureSize):
            resolved_figsize = (layout.figsize.width, layout.figsize.height)
        else:
            resolved_figsize = layout.figsize
        subplot_kwargs: dict[str, object] = {}
        if projection is not None:
            subplot_kwargs["subplot_kw"] = {"projection": projection}
        fig, axes = cast(Any, plt.subplots)(
            layout.rows,
            layout.cols,
            figsize=resolved_figsize,
            sharex=layout.sharex,
            sharey=layout.sharey,
            squeeze=False,
            **subplot_kwargs,
        )
        fig.subplots_adjust(
            left=layout.margins.left,
            bottom=layout.margins.bottom,
            right=layout.margins.right,
            top=layout.margins.top,
            wspace=layout.margins.wspace,
            hspace=layout.margins.hspace,
        )
        reshaped_axes = np.asarray(axes, dtype=object).reshape(-1)
        for ax in reshaped_axes:
            setattr(ax, "hrtfpykit_subplot_title_y", 1.0)
            setattr(
                ax,
                "hrtfpykit_subplot_title_y_with_figure_title",
                layout.subplot_title_y,
            )
        return fig, reshaped_axes

    def get_ax(self, position: int | str = 0) -> plt.Axes:
        """Return one subplot axis by flattened index or layout position name.

        position can be an integer index into the flattened axes array or a
        named position from the layout, such as ``main``, ``top``,
        ``bottom``, ``left``, or ``right``. Named positions keep the plot
        code readable when a layout has semantic panels.

        Parameters
        ----------
        position : int | str, default=0
            Subplot index or position key defined by the layout.

        Returns
        -------
        plt.Axes
            Selected subplot axis.

        Raises
        ------
        ValueError
            If a string position is not defined by the layout, or if an integer
            index is outside the flattened axes array.

        """
        if isinstance(position, str):
            if position not in self.positions:
                raise ValueError(
                    f"position must be one of: {', '.join(self.positions)}"
                )
            axis_index = self.positions.index(position)
        else:
            axis_index = int(position)
            if axis_index < 0 or axis_index >= self.axes.size:
                raise ValueError(
                    f"position index must be between 0 and {self.axes.size - 1}"
                )
        return self.axes[axis_index]

    def hide_unused_axes(self, used_axes: int) -> None:
        """Hide unused trailing axes in a multi-panel layout.

        Some plot functions use a larger layout than the number of panels
        requested by the caller, for example a 2x2 layout for three selected
        source positions. This method leaves the first used_axes axes
        visible and hides every remaining axis in the flattened axes array.

        Parameters
        ----------
        used_axes : int
            Number of axes that remain visible from the beginning of the
            flattened axes array. Must be non-negative.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If used_axes is negative.

        """
        if used_axes < 0:
            raise ValueError("used_axes must be non-negative")
        for ax in self.axes[used_axes:]:
            ax.set_visible(False)

    def create_two_dimension(self, ax: plt.Axes, x, y, **kwargs):
        """Draw one or more two-dimensional line traces on an axis.

        This is the figure-level dispatcher for
        :class:`~hrtfpykit.plots.types.TwoDimension`. It forwards x, y,
        and Matplotlib line keyword arguments to the low-level primitive while
        keeping high-level HRTF plot code independent of the concrete primitive
        class.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis. The underlying primitive rejects 3D axes.
        x : array-like
            X-axis values passed to matplotlib.axes.Axes.plot.
        y : array-like
            Y-axis values passed to matplotlib.axes.Axes.plot.
        **kwargs
            Additional line style, marker, color, label, and other keyword
            arguments forwarded unchanged to Matplotlib.

        Returns
        -------
        list[matplotlib.lines.Line2D]
            Line artists returned by Matplotlib.

        Raises
        ------
        ValueError
            If ax is a 3D projection axis.

        """
        return TwoDimension.create(
            ax=ax,
            x=x,
            y=y,
            **kwargs,
        )

    def create_heatmap(
        self,
        ax: plt.Axes,
        x,
        y,
        values,
        label: str | None = None,
        colormap: str | None = None,
        colorbar: bool = True,
        colorbar_location: str | None = None,
        colorbar_fraction: float | None = None,
        colorbar_pad: float | None = None,
        colorbar_label: str | None = None,
        **kwargs,
    ):
        """Draw a heatmap on an axis and optionally attach a colorbar.

        This is the figure-level dispatcher for
        :class:`~hrtfpykit.plots.types.Heatmap`. It forwards coordinates, values,
        colorbar options, and Matplotlib pcolormesh keyword arguments to the
        low-level primitive while supplying the current Matplotlib figure for
        colorbar creation.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis. The underlying primitive rejects 3D axes.
        x : array-like
            X-axis coordinates passed to matplotlib.axes.Axes.pcolormesh.
        y : array-like
            Y-axis coordinates passed to matplotlib.axes.Axes.pcolormesh.
        values : array-like
            Heatmap matrix values passed to pcolormesh.
        label : str | None, default=None
            Default colorbar label used when colorbar_label is not supplied.
        colormap : str | None, default=None
            Supported hrtfpykit colormap name. None uses the primitive's
            default colormap.
        colorbar : bool, default=True
            Whether to render the heatmap colorbar.
        colorbar_location : str | None, default=None
            Colorbar side/location passed to the appended colorbar axis.
        colorbar_fraction : float | None, default=None
            Relative colorbar size used by the heatmap primitive.
        colorbar_pad : float | None, default=None
            Padding between the main axis and the appended colorbar axis.
        colorbar_label : str | None, default=None
            Colorbar label override. When provided, it takes precedence over
            label.
        **kwargs
            Additional keyword arguments forwarded unchanged to pcolormesh.

        Returns
        -------
        matplotlib.collections.QuadMesh
            Heatmap artist returned by Matplotlib.

        Raises
        ------
        ValueError
            If ax is a 3D projection axis or the selected colormap is not
            supported by the heatmap primitive.

        """
        return Heatmap.create(
            ax=ax,
            x=x,
            y=y,
            values=values,
            fig=self.fig,
            label=label,
            colormap=colormap,
            colorbar=colorbar,
            colorbar_location=colorbar_location,
            colorbar_fraction=colorbar_fraction,
            colorbar_pad=colorbar_pad,
            colorbar_label=colorbar_label,
            **kwargs,
        )

    def create_three_dimension(
        self,
        ax: plt.Axes,
        x,
        y,
        z,
        s: float = 28.0,
        color: str = "steelblue",
        edgecolors: str = "black",
        linewidths: float = 0.4,
        depthshade: bool = True,
        **kwargs,
    ):
        """Draw a three-dimensional scatter plot on a 3D axis.

        This is the figure-level dispatcher for
        :class:`~hrtfpykit.plots.types.ThreeDimension`. It forwards coordinates,
        marker styling, and additional Matplotlib scatter keyword arguments to
        the low-level primitive used by source-grid and plane-grid plots.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis. The underlying primitive requires a 3D
            projection axis.
        x : array-like
            X coordinates passed to Axes3D.scatter.
        y : array-like
            Y coordinates passed to Axes3D.scatter.
        z : array-like
            Z coordinates passed to Axes3D.scatter.
        s : float, default=28.0
            Marker size.
        color : str, default=``steelblue``
            Marker face color.
        edgecolors : str, default=``black``
            Marker edge color.
        linewidths : float, default=0.4
            Marker edge width.
        depthshade : bool, default=True
            Whether to apply depth shading.
        **kwargs
            Additional keyword arguments forwarded unchanged to Matplotlib.

        Returns
        -------
        mpl_toolkits.mplot3d.art3d.Path3DCollection
            3D scatter artist returned by Matplotlib.

        Raises
        ------
        ValueError
            If ax is not a 3D projection axis.

        """
        return ThreeDimension.create(
            ax=ax,
            x=x,
            y=y,
            z=z,
            s=s,
            color=color,
            edgecolors=edgecolors,
            linewidths=linewidths,
            depthshade=depthshade,
            **kwargs,
        )
