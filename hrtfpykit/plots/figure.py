from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .default import FigureSize, RC
from .layouts import Layout
from .options import AxisOptions, PlotOptions
from .types import Heatmap, ThreeDimension, TwoDimension


class Figure:
    shared_x_visible: bool = True

    def __init__(self, layout: Layout, projection: str | None = None):
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
        Figure.configure_rc()
        if isinstance(layout.figsize, FigureSize):
            resolved_figsize = (layout.figsize.width, layout.figsize.height)
        else:
            resolved_figsize = layout.figsize
        subplot_kwargs: dict[str, object] = {}
        if projection is not None:
            subplot_kwargs["subplot_kw"] = {"projection": projection}
        fig, axes = plt.subplots(
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
        if used_axes < 0:
            raise ValueError("used_axes must be non-negative")
        for ax in self.axes[used_axes:]:
            ax.set_visible(False)

    def get_subplots_axis_options(
        self,
        plot_options: PlotOptions,
    ) -> dict[int, AxisOptions]:
        subplot_axis_options: dict[int, AxisOptions] = {}
        if plot_options.subplots is None:
            return subplot_axis_options
        for subplot, subplot_options in plot_options.subplots.items():
            if isinstance(subplot, str):
                if subplot not in self.positions:
                    raise ValueError(
                        f"subplot accepts: {', '.join(self.positions)}"
                    )
                subplot_index = self.positions.index(subplot)
            else:
                subplot_index = int(subplot)
                if subplot_index < 0 or subplot_index >= self.axes.size:
                    raise ValueError(
                        f"subplot index must be between 0 and {self.axes.size - 1}"
                    )
            if subplot_index in subplot_axis_options:
                raise ValueError(
                    f"subplot override for subplot {subplot_index} is duplicated"
                )
            subplot_axis_options[subplot_index] = subplot_options
        return subplot_axis_options

    def create_two_dimension(self, ax: plt.Axes, x, y, **kwargs):
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
        options=None,
        colormap: str | None = None,
        **kwargs,
    ):
        return Heatmap.create(
            ax=ax,
            x=x,
            y=y,
            values=values,
            fig=self.fig,
            label=label,
            options=options,
            colormap=colormap,
            **kwargs,
        )

    def create_three_dimension(self, ax: plt.Axes, x, y, z, **kwargs):
        return ThreeDimension.create(
            ax=ax,
            x=x,
            y=y,
            z=z,
            **kwargs,
        )
