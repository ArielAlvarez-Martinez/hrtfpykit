from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Any, cast
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


class TwoDimension:
    """Primitive renderer for two-dimensional Matplotlib line plots.

    :class:`~hrtfpykit.plots.types.TwoDimension` is the low-level line-plot
    adapter used by :class:`~hrtfpykit.plots.figure.Figure`. It keeps the
    plotting layer's 2D rendering path explicit by validating that the target
    axis is not a 3D projection, forwarding all style arguments to
    matplotlib.axes.Axes.plot, and returning the native Matplotlib line artists
    without wrapping or copying them.

    Use this wrapper for spectra, impulse responses, metrics, or other
    one-dimensional traces that should be drawn on a normal Cartesian or polar
    Matplotlib axis. Use :class:`~hrtfpykit.plots.types.ThreeDimension` for 3D
    source-grid views.
    """

    @staticmethod
    def create(
        ax: Axes,
        x,
        y,
        **kwargs,
    ) -> list[Line2D]:
        """Draw one or more two-dimensional line series on a non-3D axis.

        The method is a small validation layer around Axes.plot. It rejects axes
        whose name is ``3d`` so that line plots do not silently render into the
        wrong projection, then forwards x, y, and all keyword arguments
        directly to Matplotlib. Matplotlib handles scalar, one-dimensional, and
        multi-series inputs according to its normal plot rules.

        Parameters
        ----------
        ax : Axes
            Target Matplotlib axis. The axis must not be a 3D projection.
        x : array-like
            X-axis values passed to Axes.plot.
        y : array-like
            Y-axis values passed to Axes.plot. Shape compatibility is delegated
            to Matplotlib.
        **kwargs
            Line style, marker, color, label, and other keyword arguments forwarded
            unchanged to Axes.plot.

        Returns
        -------
        list[Line2D]
            Line artists returned by Matplotlib. Multi-series inputs may produce more
            than one artist.

        Raises
        ------
        ValueError
            If ax is a 3D projection axis.

        """
        if getattr(ax, "name", "") == "3d":
            raise ValueError("TwoDimension does not accept 3d axes")
        return ax.plot(x, y, **kwargs)


class Heatmap:
    """Primitive renderer for Matplotlib heatmaps and attached colorbars.

    :class:`~hrtfpykit.plots.types.Heatmap` is the low-level heatmap adapter
    used by :class:`~hrtfpykit.plots.figure.Figure`. It validates that heatmaps
    are drawn on a non-3D axis, resolves hrtfpykit's supported colormap names,
    renders values with matplotlib.axes.Axes.pcolormesh, and optionally appends
    a colorbar axis using mpl_toolkits.axes_grid1.make_axes_locatable.

    Notes
    -----
    colorbar_fraction is interpreted as a fraction of the main axis size and is
    converted to the percentage string expected by append_axes. A figure is
    required only when colorbar=True because the colorbar is created through
    matplotlib.figure.Figure.colorbar.

    Attributes
    ----------
    colormaps : dict[str, str]
        Supported hrtfpykit colormap names and the Matplotlib colormap names they
        resolve to.
    colorbar_location : str
        Default side passed to append_axes when a colorbar is enabled.
    colorbar_fraction : float
        Default colorbar size expressed as a fraction of the main axis size.
    colorbar_pad : float
        Default padding between the heatmap axis and appended colorbar axis.
    """

    colormaps: dict[str, str] = {
        "viridis": "viridis",
        "magma": "magma",
        "cividis": "cividis",
        "jet": "jet",
    }
    colorbar_location: str = "right"
    colorbar_fraction: float = 0.03
    colorbar_pad: float = 0.2

    @staticmethod
    def create(
        ax: Axes,
        x,
        y,
        values,
        fig: plt.Figure | None = None,
        label: str | None = None,
        colormap: str | None = None,
        colorbar: bool = True,
        colorbar_location: str | None = None,
        colorbar_fraction: float | None = None,
        colorbar_pad: float | None = None,
        colorbar_label: str | None = None,
        **kwargs,
    ) -> QuadMesh:
        """Draw a pseudocolor heatmap and optionally attach a colorbar.

        The method validates the axis projection, resolves a supported colormap, and
        delegates grid rendering to Axes.pcolormesh. When colorbar is true, a
        new colorbar axis is appended beside ax and linked to the returned
        QuadMesh. The colorbar label is resolved from colorbar_label first and
        falls back to label when no override is provided.

        Parameters
        ----------
        ax : Axes
            Target Matplotlib axis. The axis must not be a 3D projection.
        x : array-like
            X-axis coordinates forwarded to Axes.pcolormesh. One-dimensional or
            two-dimensional coordinate inputs follow Matplotlib's pcolormesh
            rules.
        y : array-like
            Y-axis coordinates forwarded to Axes.pcolormesh.
        values : array-like
            Heatmap matrix values forwarded as the color array.
        fig : plt.Figure | None, default=None
            Figure used for colorbar creation. Required when colorbar=True and
            ignored when colorbar=False.
        label : str | None, default=None
            Default colorbar label used when colorbar_label is not supplied.
        colormap : str | None, default=None
            Supported hrtfpykit colormap name. None selects ``jet`` for
            backwards-compatible plotting defaults.
        colorbar : bool, default=True
            Whether to draw a colorbar.
        colorbar_location : str | None, default=None
            Colorbar side/location passed to append_axes. None uses
            :attr:`~hrtfpykit.plots.types.Heatmap.colorbar_location`.
        colorbar_fraction : float | None, default=None
            Relative colorbar size. None uses
            :attr:`~hrtfpykit.plots.types.Heatmap.colorbar_fraction`.
        colorbar_pad : float | None, default=None
            Padding between the main axis and colorbar axis. None uses
            :attr:`~hrtfpykit.plots.types.Heatmap.colorbar_pad`.
        colorbar_label : str | None, default=None
            Colorbar label override. When provided, this value takes precedence over
            label.
        **kwargs
            Additional keyword arguments forwarded unchanged to Axes.pcolormesh.

        Returns
        -------
        QuadMesh
            Heatmap artist returned by Axes.pcolormesh.

        Raises
        ------
        ValueError
            If ax is a 3D projection axis, if colormap is not one of the
            supported names in
            :attr:`~hrtfpykit.plots.types.Heatmap.colormaps`, or if fig is missing
            while colorbar=True.

        """
        if getattr(ax, "name", "") == "3d":
            raise ValueError("Heatmap does not accept 3d axes")
        resolved_colormap = "jet" if colormap is None else str(colormap)
        if resolved_colormap not in Heatmap.colormaps:
            raise ValueError(
                f"heatmap cmap accepts: {', '.join(Heatmap.colormaps)}"
            )
        mesh = ax.pcolormesh(
            x,
            y,
            values,
            cmap=Heatmap.colormaps[resolved_colormap],
            **kwargs,
        )
        if not bool(colorbar):
            return mesh
        if fig is None:
            raise ValueError("fig is required when colorbar is enabled")
        resolved_location = (
            Heatmap.colorbar_location
            if colorbar_location is None
            else colorbar_location
        )
        resolved_fraction = (
            Heatmap.colorbar_fraction
            if colorbar_fraction is None
            else colorbar_fraction
        )
        resolved_pad = (
            Heatmap.colorbar_pad
            if colorbar_pad is None
            else colorbar_pad
        )
        resolved_label = label if colorbar_label is None else colorbar_label
        divider = make_axes_locatable(ax)
        colorbar_size = f"{float(resolved_fraction) * 100.0:.1f}%"
        cax = divider.append_axes(
            resolved_location,
            size=colorbar_size,
            pad=resolved_pad,
        )
        fig.colorbar(mesh, cax=cax, label=resolved_label)
        return mesh


class ThreeDimension:
    """Primitive renderer for three-dimensional Matplotlib scatter plots.

    :class:`~hrtfpykit.plots.types.ThreeDimension` is the low-level 3D scatter
    adapter used by :class:`~hrtfpykit.plots.figure.Figure`. It validates that
    the target axis uses a 3D projection and delegates point rendering to
    Axes.scatter with hrtfpykit's default marker size, face color, edge color,
    edge width, and depth shading.

    Use this wrapper for source-position grids and other spatial point clouds. It is
    intentionally limited to scatter-style primitives; two-dimensional traces and
    heatmaps are handled by :class:`~hrtfpykit.plots.types.TwoDimension` and
    :class:`~hrtfpykit.plots.types.Heatmap`.
    """

    @staticmethod
    def create(
        ax: Axes,
        x,
        y,
        z,
        s: float = 28.0,
        color: str = "steelblue",
        edgecolors: str = "black",
        linewidths: float = 0.4,
        depthshade: bool = True,
        **kwargs,
    ) -> Path3DCollection:
        """Draw a three-dimensional scatter plot on a 3D Matplotlib axis.

        The method is a validation and defaults layer around Axes.scatter for 3D
        source-grid rendering. It requires an axis whose name is ``3d`` and
        forwards coordinates, marker defaults, and extra keyword arguments directly to
        Matplotlib.

        Parameters
        ----------
        ax : Axes
            Target Matplotlib axis with a ``3d`` projection.
        x : array-like
            X coordinates of the plotted points.
        y : array-like
            Y coordinates of the plotted points.
        z : array-like
            Z coordinates of the plotted points.
        s : float, default=28.0
            Marker size forwarded to Axes.scatter.
        color : str, default=``steelblue``
            Marker face color.
        edgecolors : str, default=``black``
            Marker edge color.
        linewidths : float, default=0.4
            Marker edge width.
        depthshade : bool, default=True
            Whether to apply depth shading.
        **kwargs
            Additional arguments forwarded to Axes.scatter.

        Returns
        -------
        Path3DCollection
            Scatter artist returned by Matplotlib.

        Raises
        ------
        ValueError
            If ax is not a 3D projection axis.

        """
        if getattr(ax, "name", "") != "3d":
            raise ValueError("ThreeDimension requires a 3d projection")
        scatter_kwargs: dict[str, object] = dict(kwargs)
        scatter_kwargs["s"] = s
        scatter_kwargs["color"] = color
        scatter_kwargs["edgecolors"] = edgecolors
        scatter_kwargs["linewidths"] = linewidths
        scatter_kwargs["depthshade"] = depthshade
        return cast(Any, ax).scatter(
            x,
            y,
            z,
            **scatter_kwargs,
        )
