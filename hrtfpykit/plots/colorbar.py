from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .options import HeatmapOptions


class ColorBar:
    colormaps: dict[str, str] = {
        "viridis": "viridis",
        "magma": "magma",
        "cividis": "cividis",
        "jet": "jet",
    }
    location: str = "right"
    fraction: float = 0.03
    pad: float = 0.2

    @staticmethod
    def create(
        fig: plt.Figure,
        ax: plt.Axes,
        mesh,
        label: str,
        options: HeatmapOptions | None = None,
        colormap: str | None = None,
    ) -> None:
        heatmap_options = HeatmapOptions() if options is None else options
        colorbar_enabled = (
            True if heatmap_options.colorbar is None else heatmap_options.colorbar
        )
        if not colorbar_enabled:
            return
        resolved_colormap = "jet" if colormap is None else str(colormap)
        if resolved_colormap not in ColorBar.colormaps:
            raise ValueError(
                f"heatmap cmap accepts: {', '.join(ColorBar.colormaps)}"
            )
        if hasattr(mesh, "set_cmap"):
            mesh.set_cmap(ColorBar.colormaps[resolved_colormap])
        resolved_location = (
            ColorBar.location
            if heatmap_options.colorbar_location is None
            else heatmap_options.colorbar_location
        )
        resolved_fraction = (
            ColorBar.fraction
            if heatmap_options.colorbar_fraction is None
            else heatmap_options.colorbar_fraction
        )
        resolved_pad = (
            ColorBar.pad
            if heatmap_options.colorbar_pad is None
            else heatmap_options.colorbar_pad
        )
        resolved_label = (
            label
            if heatmap_options.colorbar_label is None
            else heatmap_options.colorbar_label
        )
        divider = make_axes_locatable(ax)
        colorbar_size = f"{float(resolved_fraction) * 100.0:.1f}%"
        cax = divider.append_axes(
            resolved_location,
            size=colorbar_size,
            pad=resolved_pad,
        )
        fig.colorbar(
            mesh,
            cax=cax,
            label=resolved_label,
        )
