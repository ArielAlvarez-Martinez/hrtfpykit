from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .options import HeatmapOptions


class TwoDimension:
    @staticmethod
    def create(
        ax: Axes,
        x,
        y,
        **kwargs,
    ) -> list[Line2D]:
        if getattr(ax, "name", "") == "3d":
            raise ValueError("TwoDimension does not accept 3d axes")
        return ax.plot(x, y, **kwargs)


class Heatmap:
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
        options: HeatmapOptions | None = None,
        colormap: str | None = None,
        **kwargs,
    ) -> QuadMesh:
        if getattr(ax, "name", "") == "3d":
            raise ValueError("Heatmap does not accept 3d axes")
        heatmap_options = HeatmapOptions() if options is None else options
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
        colorbar_enabled = (
            True if heatmap_options.colorbar is None else heatmap_options.colorbar
        )
        if not colorbar_enabled:
            return mesh
        if fig is None:
            raise ValueError("fig is required when colorbar is enabled")
        resolved_location = (
            Heatmap.colorbar_location
            if heatmap_options.colorbar_location is None
            else heatmap_options.colorbar_location
        )
        resolved_fraction = (
            Heatmap.colorbar_fraction
            if heatmap_options.colorbar_fraction is None
            else heatmap_options.colorbar_fraction
        )
        resolved_pad = (
            Heatmap.colorbar_pad
            if heatmap_options.colorbar_pad is None
            else heatmap_options.colorbar_pad
        )
        resolved_label = (
            label if heatmap_options.colorbar_label is None else heatmap_options.colorbar_label
        )
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
