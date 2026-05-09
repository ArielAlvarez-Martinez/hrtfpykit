from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


class TwoDimension:
    """Matplotlib primitive wrapper for two-dimensional line plots.

    The wrapper centralizes validation for line plots that must be drawn on a
    non-3D axis and returns the Matplotlib line artists created by
    ``Axes.plot``.
    """

    @staticmethod
    def create(
        ax: Axes,
        x,
        y,
        **kwargs,
    ) -> list[Line2D]:
        """Create a 2D line plot on a non-3D axis.

        Parameters
        ----------
        ax : Axes
            Target Matplotlib axis.
        x : array-like
            X-axis values.
        y : array-like
            Y-axis values.
        **kwargs
            Additional arguments forwarded to ``Axes.plot``.

        Returns
        -------
        list[Line2D]
            Line artists returned by Matplotlib.
        """
        if getattr(ax, "name", "") == "3d":
            raise ValueError("TwoDimension does not accept 3d axes")
        return ax.plot(x, y, **kwargs)


class Heatmap:
    """Heatmap primitive wrapper with optional colorbar support."""

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
        """Create a heatmap and optionally attach a colorbar.

        Parameters
        ----------
        ax : Axes
            Target Matplotlib axis.
        x : array-like
            X-axis coordinates.
        y : array-like
            Y-axis coordinates.
        values : array-like
            Heatmap matrix values.
        fig : plt.Figure | None, default=None
            Figure used for colorbar creation when enabled.
        label : str | None, default=None
            Default colorbar label.
        colormap : str | None, default=None
            Colormap name.
        colorbar : bool, default=True
            Whether to draw a colorbar.
        colorbar_location : str | None, default=None
            Colorbar side/location used by ``append_axes``.
        colorbar_fraction : float | None, default=None
            Relative colorbar size.
        colorbar_pad : float | None, default=None
            Padding between axis and colorbar.
        colorbar_label : str | None, default=None
            Colorbar label override.
        **kwargs
            Additional arguments forwarded to ``Axes.pcolormesh``.

        Returns
        -------
        QuadMesh
            Heatmap artist returned by Matplotlib.
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
    """Matplotlib primitive wrapper for three-dimensional source plots.

    The wrapper validates that the target axis uses a 3D projection and then
    delegates point rendering to ``Axes.scatter`` with the library defaults for
    marker size, color, edge color, and depth shading.
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
        """Create a 3D scatter plot on a 3D axis.

        Parameters
        ----------
        ax : Axes
            Target Matplotlib axis with ``3d`` projection.
        x : array-like
            X coordinates.
        y : array-like
            Y coordinates.
        z : array-like
            Z coordinates.
        s : float, default=28.0
            Marker size.
        color : str, default="steelblue"
            Marker face color.
        edgecolors : str, default="black"
            Marker edge color.
        linewidths : float, default=0.4
            Marker edge width.
        depthshade : bool, default=True
            Whether to apply depth shading.
        **kwargs
            Additional arguments forwarded to ``Axes.scatter``.

        Returns
        -------
        Path3DCollection
            Scatter artist returned by Matplotlib.
        """
        if getattr(ax, "name", "") != "3d":
            raise ValueError("ThreeDimension requires a 3d projection")
        return ax.scatter(
            x,
            y,
            z,
            s=s,
            color=color,
            edgecolors=edgecolors,
            linewidths=linewidths,
            depthshade=depthshade,
            **kwargs,
        )
