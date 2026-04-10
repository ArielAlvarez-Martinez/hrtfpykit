from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .default import FigureSize, RC
from .legends import Ear
from .layouts import Layout
from .options import AxisOptions, PlotOptions
from .titles import Titles


class Figure:
    shared_x_visible: bool = True

    def __init__(self, layout: Layout):
        self.layout = layout.code
        self.positions = layout.positions
        self.fig, self.axes, self.figure_title_y = self.create(layout)

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
    def create(layout: Layout) -> tuple[plt.Figure, np.ndarray, float]:
        Figure.configure_rc()
        if isinstance(layout.figsize, FigureSize):
            resolved_figsize = (layout.figsize.width, layout.figsize.height)
        else:
            resolved_figsize = layout.figsize
        subplot_kwargs: dict[str, object] = {}
        if layout.projection is not None:
            subplot_kwargs["subplot_kw"] = {"projection": layout.projection}
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
        figure_title_y = min(layout.margins.top + layout.figure_title_offset, 0.98)
        reshaped_axes = np.asarray(axes, dtype=object).reshape(-1)
        for ax in reshaped_axes:
            setattr(ax, "hrtfpykit_subplot_title_y", 1.0)
            setattr(
                ax,
                "hrtfpykit_subplot_title_y_with_figure_title",
                layout.subplot_title_y,
            )
        return fig, reshaped_axes, figure_title_y

    def get_axis(self, position: int | str = 0) -> plt.Axes:
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

    def get_panel_axis_options(
        self,
        plot_options: PlotOptions,
    ) -> dict[int, AxisOptions]:
        panel_axis_options: dict[int, AxisOptions] = {}
        if plot_options.panels is None:
            return panel_axis_options
        for panel, panel_options in plot_options.panels.items():
            if isinstance(panel, str):
                if panel not in self.positions:
                    raise ValueError(
                        f"panel accepts: {', '.join(self.positions)}"
                    )
                panel_index = self.positions.index(panel)
            else:
                panel_index = int(panel)
                if panel_index < 0 or panel_index >= self.axes.size:
                    raise ValueError(
                        f"panel index must be between 0 and {self.axes.size - 1}"
                    )
            if panel_index in panel_axis_options:
                raise ValueError(
                    f"panel override for subplot {panel_index} is duplicated"
                )
            panel_axis_options[panel_index] = panel_options
        return panel_axis_options

    def apply_panel(
        self,
        ax: plt.Axes,
        selected_positions: np.ndarray,
        ear: str,
        options: AxisOptions | None = None,
        legend_location: str = "upper right",
    ) -> None:
        axis_options = AxisOptions() if options is None else options
        legend_options = axis_options.legend
        shared_x_visible = (
            Figure.shared_x_visible
            if axis_options.shared_x_visible is None
            else axis_options.shared_x_visible
        )
        default_title = Titles.create_position_title(
            selected_positions=selected_positions,
        )
        resolved_title = default_title if axis_options.title is None else axis_options.title
        Titles.create_subplots_titles(ax=ax, title=resolved_title)
        if shared_x_visible:
            ax.tick_params(axis="x", which="both", labelbottom=True)
        legend_enabled = True if legend_options is None or legend_options.enabled is None else legend_options.enabled
        if legend_enabled:
            resolved_legend_location = (
                legend_location
                if legend_options is None or legend_options.location is None
                else legend_options.location
            )
            legend_labels = None if legend_options is None else legend_options.labels
            Ear.apply(
                ax=ax,
                ear=ear,
                location=resolved_legend_location,
                labels=legend_labels,
            )
        grid_enabled = True if axis_options.grid is None else axis_options.grid
        if grid_enabled:
            ax.grid(True)
