from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .dsp import calculate_ild, calculate_itd, magnitude_to_db
from .spatial import Sources


if TYPE_CHECKING:
    from .hrtf import HRTF


@dataclass(frozen=True)
class Margins:
    top: float = 0.9
    bottom: float = 0.1
    left: float = 0.1
    right: float = 0.9
    wspace: float = 0.35
    hspace: float = 0.35


@dataclass(frozen=True)
class FigSizeDefault:
    width: float = 8
    height: float = 6


@dataclass(frozen=True)
class Rc:
    legend_title: float = 10
    legend: float = 9
    ticks: float = 7
    axis_labels: float = 9
    default: float = 10
    axis_title: float = 10
    fig_title: float = 12


@dataclass(frozen=True)
class Labels:
    frequency: str = "Frequency(kHz)"
    magnitude_db: str = "Magnitude (dB)"
    magnitude_linear: str = "Magnitude"
    time: str = "Time (s)"
    samples: str = "Samples"
    impulse_response: str = "Amplitude"
    itd_seconds: str = "Absolute ITD (s)"
    ild_db: str = "Absolute ILD (dB)"
    azimuth: str = "Azimuth (degrees)"
    elevation: str = "Elevation (degrees)"
    lateral: str = "Lateral (degrees)"
    polar: str = "Polar (degrees)"


@dataclass(frozen=True)
class Titles:
    spherical_alias: str = "{name} : [Azimuth= {az}°, Elevation= {el}°]"
    spherical_position: str = "Position : [Azimuth= {az}°, Elevation= {el}°]"
    cartesian_alias: str = "{name} : [x= {x}, y= {y}, z= {z}]"
    cartesian_position: str = "Position : [x= {x}, y= {y}, z= {z}]"
    lateral_polar_alias: str = "{name} : [Lateral= {lateral}°, Polar= {polar}°]"
    lateral_polar_position: str = "Position : [Lateral= {lateral}°, Polar= {polar}°]"
    horizontal_plane: str = "Horizontal Plane"
    median_plane: str = "Median Plane"
    elevation_spectrum: str = "Elevation Spectrum : [Azimuth= {angle}°]"


@dataclass(frozen=True)
class AcceptedParameters:
    units: tuple[str, str] = ("db", "linear")
    ears: tuple[str, str, str] = ("left", "right", "both")
    x_axes: tuple[str, str] = ("time", "samples")
    frequency_x_axes: tuple[str, str] = ("log", "linear")
    planes: tuple[str, str, str] = ("horizontal", "median", "frontal")


@dataclass(frozen=True)
class FigureOptions:
    figsize: tuple[float, float] | None = None
    margins: Margins | None = None
    title: str | None = None


@dataclass(frozen=True)
class LegendOptions:
    enabled: bool | None = None
    labels: tuple[str, ...] | list[str] | None = None
    location: str | None = None

    def merge(self, options: LegendOptions | None = None) -> LegendOptions:
        if options is None:
            return self
        return LegendOptions(
            enabled=self.enabled if options.enabled is None else options.enabled,
            labels=self.labels if options.labels is None else options.labels,
            location=self.location if options.location is None else options.location,
        )


@dataclass(frozen=True)
class FrequencyAxisOptions:
    freq_min: float | None = None
    freq_max: float | None = None
    ticks: tuple[float, ...] | list[float] | None = None
    labels: tuple[str, ...] | list[str] | None = None
    margin_ratio: float | None = None

    def merge(
        self,
        options: FrequencyAxisOptions | None = None,
    ) -> FrequencyAxisOptions:
        if options is None:
            return self
        return FrequencyAxisOptions(
            freq_min=self.freq_min if options.freq_min is None else options.freq_min,
            freq_max=self.freq_max if options.freq_max is None else options.freq_max,
            ticks=self.ticks if options.ticks is None else options.ticks,
            labels=self.labels if options.labels is None else options.labels,
            margin_ratio=(
                self.margin_ratio
                if options.margin_ratio is None
                else options.margin_ratio
            ),
        )


@dataclass(frozen=True)
class AzimuthAxisOptions:
    range_mode: str | None = None

    def merge(
        self,
        options: AzimuthAxisOptions | None = None,
    ) -> AzimuthAxisOptions:
        if options is None:
            return self
        return AzimuthAxisOptions(
            range_mode=self.range_mode if options.range_mode is None else options.range_mode,
        )


@dataclass(frozen=True)
class AxisOptions:
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    shared_x_visible: bool | None = None
    grid: bool | None = None
    legend: LegendOptions | None = None
    frequency_axis: FrequencyAxisOptions | None = None
    azimuth_axis: AzimuthAxisOptions | None = None

    def merge(self, options: AxisOptions | None = None) -> AxisOptions:
        if options is None:
            return self
        base_legend = LegendOptions() if self.legend is None else self.legend
        base_frequency_axis = (
            FrequencyAxisOptions()
            if self.frequency_axis is None
            else self.frequency_axis
        )
        base_azimuth_axis = (
            AzimuthAxisOptions()
            if self.azimuth_axis is None
            else self.azimuth_axis
        )
        return AxisOptions(
            xlabel=self.xlabel if options.xlabel is None else options.xlabel,
            ylabel=self.ylabel if options.ylabel is None else options.ylabel,
            title=self.title if options.title is None else options.title,
            shared_x_visible=(
                self.shared_x_visible
                if options.shared_x_visible is None
                else options.shared_x_visible
            ),
            grid=self.grid if options.grid is None else options.grid,
            legend=base_legend.merge(options.legend),
            frequency_axis=base_frequency_axis.merge(options.frequency_axis),
            azimuth_axis=base_azimuth_axis.merge(options.azimuth_axis),
        )


@dataclass(frozen=True)
class PlotOptions:
    figure: FigureOptions | None = None
    axis: AxisOptions | None = None
    heatmap: HeatmapOptions | None = None
    panels: dict[int | str, AxisOptions] | None = None
    show: bool = True


@dataclass(frozen=True)
class HeatmapOptions:
    cmap: str | None = None
    colorbar: bool | None = None
    colorbar_label: str | None = None
    colorbar_location: str | None = None
    colorbar_fraction: float | None = None
    colorbar_pad: float | None = None

    def merge(self, options: HeatmapOptions | None = None) -> HeatmapOptions:
        if options is None:
            return self
        return HeatmapOptions(
            cmap=self.cmap if options.cmap is None else options.cmap,
            colorbar=self.colorbar if options.colorbar is None else options.colorbar,
            colorbar_label=(
                self.colorbar_label
                if options.colorbar_label is None
                else options.colorbar_label
            ),
            colorbar_location=(
                self.colorbar_location
                if options.colorbar_location is None
                else options.colorbar_location
            ),
            colorbar_fraction=(
                self.colorbar_fraction
                if options.colorbar_fraction is None
                else options.colorbar_fraction
            ),
            colorbar_pad=(
                self.colorbar_pad
                if options.colorbar_pad is None
                else options.colorbar_pad
            ),
        )


class Heatmap:
    colormaps: dict[str, str] = {
        "default": "viridis",
        "magma": "magma",
        "cividis": "cividis",
        "jet": "jet",
    }
    colorbar_location: str = "right"
    colorbar_fraction: float = 0.03
    colorbar_pad: float = 0.2
    axis_margin_ratio: float = 0.0

    @staticmethod
    def create_colormap(
        options: HeatmapOptions | None = None,
    ) -> str:
        heatmap_options = HeatmapOptions() if options is None else options
        color_key = (
            "default" if heatmap_options.cmap is None else str(heatmap_options.cmap)
        )
        if color_key not in Heatmap.colormaps:
            raise ValueError(
                f"heatmap cmap accepts: {', '.join(Heatmap.colormaps)}"
            )
        return Heatmap.colormaps[color_key]

    @staticmethod
    def create_colorbar(
        fig: plt.Figure,
        ax: plt.Axes,
        mesh,
        label: str,
        options: HeatmapOptions | None = None,
    ) -> None:
        heatmap_options = HeatmapOptions() if options is None else options
        colorbar_enabled = (
            True if heatmap_options.colorbar is None else heatmap_options.colorbar
        )
        if not colorbar_enabled:
            return
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


class SourcePositionData:
    @staticmethod
    def create_positions(
        sources: Sources,
        coordinate_system: str,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        source_positions = np.asarray(
            sources.get_positions(angle_unit=angle_unit),
            dtype=float,
        )
        if (
            source_positions.ndim != 2
            or source_positions.shape[0] == 0
            or source_positions.shape[1] != 3
        ):
            raise ValueError("Source positions must have shape (M, 3)")

        source_system = str(sources.get_source_coordinate_system()).strip().lower()
        target_system = str(coordinate_system).strip().lower()
        if target_system == source_system:
            return source_positions
        if target_system == "cartesian":
            if source_system == "spherical":
                return sources.spherical_to_cartesian(
                    source_positions,
                    angle_unit=angle_unit,
                )
            if source_system == "lateral-polar":
                return sources.lateral_polar_to_cartesian(
                    source_positions,
                    angle_unit=angle_unit,
                )
        if target_system == "spherical":
            if source_system == "cartesian":
                return sources.cartesian_to_spherical(
                    source_positions,
                    angle_unit=angle_unit,
                )
            if source_system == "lateral-polar":
                return sources.lateral_polar_to_spherical(
                    source_positions,
                    angle_unit=angle_unit,
                )
        if target_system == "lateral-polar":
            if source_system == "cartesian":
                return sources.cartesian_to_lateral_polar(
                    source_positions,
                    angle_unit=angle_unit,
                )
            if source_system == "spherical":
                return sources.spherical_to_lateral_polar(
                    source_positions,
                    angle_unit=angle_unit,
                )
        raise ValueError(f"Unsupported source coordinate system conversion: {source_system!r} -> {target_system!r}")


class ThreeDimensional:
    view_elev: float = 22.0
    view_azim: float = -37.0
    xlabel: str = "X (m)"
    ylabel: str = "Y (m)"
    zlabel: str = "Z (m)"
    arrow_color: str = "#303030"
    arrow_linewidth: float = 2.8
    arrow_length_ratio: float = 0.32
    arrow_delta_ratio: float = 0.50
    arrow_label_offset_ratio: float = 0.10
    right_label_vertical_offset_ratio: float = 0.18
    label_box: dict[str, object] = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.88,
    }

    @staticmethod
    def create_layout(
        figsize: tuple[float, float] | None = None,
        margins: Margins | None = None,
    ) -> tuple[LayoutFigure, plt.Axes]:
        layout = Projection.create_layout(
            layout=1,
            projection="3d",
            figsize=figsize,
            margins=margins,
        )
        return layout, layout.get_axis("main")

    @staticmethod
    def configure_axis(
        ax: plt.Axes,
        cartesian_positions: np.ndarray,
    ) -> float:
        x_values = np.asarray(cartesian_positions[:, 0], dtype=float)
        y_values = np.asarray(cartesian_positions[:, 1], dtype=float)
        z_values = np.asarray(cartesian_positions[:, 2], dtype=float)

        ax.set_xlabel(ThreeDimensional.xlabel)
        ax.set_ylabel(ThreeDimensional.ylabel)
        ax.set_zlabel(ThreeDimensional.zlabel)
        ax.view_init(elev=ThreeDimensional.view_elev, azim=ThreeDimensional.view_azim)

        x_center = (float(np.min(x_values)) + float(np.max(x_values))) / 2.0
        y_center = (float(np.min(y_values)) + float(np.max(y_values))) / 2.0
        z_center = (float(np.min(z_values)) + float(np.max(z_values))) / 2.0
        axis_span = max(
            float(np.max(x_values) - np.min(x_values)),
            float(np.max(y_values) - np.min(y_values)),
            float(np.max(z_values) - np.min(z_values)),
            1.0,
        )
        axis_half_span = axis_span / 2.0
        ax.set_xlim(x_center - axis_half_span, x_center + axis_half_span)
        ax.set_ylim(y_center - axis_half_span, y_center + axis_half_span)
        ax.set_zlim(z_center - axis_half_span, z_center + axis_half_span)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        return axis_half_span

    @staticmethod
    def create_direction_markers(
        ax: plt.Axes,
        sources: Sources,
        axis_half_span: float,
    ) -> None:
        _, front_position = sources.get_position_index(
            np.array([0.0, 0.0], dtype=float),
            coordinate_system="spherical",
            angle_unit="degrees",
        )
        _, right_position = sources.get_position_index(
            np.array([270.0, 0.0], dtype=float),
            coordinate_system="spherical",
            angle_unit="degrees",
        )
        _, up_position = sources.get_position_index(
            np.array([0.0, 90.0], dtype=float),
            coordinate_system="spherical",
            angle_unit="degrees",
        )

        front_tail = sources.spherical_to_cartesian(front_position, angle_unit="degrees")
        right_tail = sources.spherical_to_cartesian(right_position, angle_unit="degrees")
        up_tail = sources.spherical_to_cartesian(up_position, angle_unit="degrees")
        front_direction = front_tail / max(float(np.linalg.norm(front_tail)), 1e-12)
        right_direction = right_tail / max(float(np.linalg.norm(right_tail)), 1e-12)
        up_direction = up_tail / max(float(np.linalg.norm(up_tail)), 1e-12)
        arrow_delta_radius = ThreeDimensional.arrow_delta_ratio * axis_half_span

        ax.quiver(
            *front_tail,
            *(front_direction * arrow_delta_radius),
            color=ThreeDimensional.arrow_color,
            linewidth=ThreeDimensional.arrow_linewidth,
            arrow_length_ratio=ThreeDimensional.arrow_length_ratio,
        )
        ax.text(
            *(
                front_tail
                + front_direction
                * (
                    arrow_delta_radius
                    + ThreeDimensional.arrow_label_offset_ratio * axis_half_span
                )
            ),
            "Front",
            color=ThreeDimensional.arrow_color,
            fontweight="bold",
            fontsize=11,
            ha="left",
            va="center",
            bbox=ThreeDimensional.label_box,
        )

        ax.quiver(
            *right_tail,
            *(right_direction * arrow_delta_radius),
            color=ThreeDimensional.arrow_color,
            linewidth=ThreeDimensional.arrow_linewidth,
            arrow_length_ratio=ThreeDimensional.arrow_length_ratio,
        )
        ax.text(
            *(
                right_tail
                + right_direction
                * (
                    arrow_delta_radius
                    + ThreeDimensional.arrow_label_offset_ratio * axis_half_span
                )
                + np.array(
                    [0.0, 0.0, ThreeDimensional.right_label_vertical_offset_ratio * axis_half_span]
                )
            ),
            "Right",
            color=ThreeDimensional.arrow_color,
            fontweight="bold",
            fontsize=11,
            ha="left",
            va="bottom",
            bbox=ThreeDimensional.label_box,
        )

        ax.quiver(
            *up_tail,
            *(up_direction * arrow_delta_radius),
            color=ThreeDimensional.arrow_color,
            linewidth=ThreeDimensional.arrow_linewidth,
            arrow_length_ratio=ThreeDimensional.arrow_length_ratio,
        )
        ax.text(
            *(
                up_tail
                + up_direction
                * (
                    arrow_delta_radius
                    + ThreeDimensional.arrow_label_offset_ratio * axis_half_span
                )
            ),
            "Up",
            color=ThreeDimensional.arrow_color,
            fontweight="bold",
            fontsize=11,
            ha="left",
            va="bottom",
            bbox=ThreeDimensional.label_box,
        )


class Legends:
    locations: dict[str, str] = {"default": "upper left"}

    @staticmethod
    def apply(
        ax: plt.Axes,
        labels: list[str],
        location: str | None = None,
    ) -> None:
        if len(labels) == 0:
            raise ValueError("labels must contain at least one entry")
        resolved_location = (
            Legends.locations["default"] if location is None else location
        )
        ax.legend(labels=labels, loc=resolved_location)

    @staticmethod
    def ears_legend(
        ax: plt.Axes,
        ear: str,
        location: str | None = None,
        labels: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        default_labels_by_ear = {
            "both": ["Left Ear", "Right Ear"],
            "left": ["Left Ear"],
            "right": ["Right Ear"],
        }
        if ear not in default_labels_by_ear:
            raise ValueError("ear accepts left, right, or both")
        default_labels = default_labels_by_ear[ear]
        expected_label_count = len(default_labels)
        resolved_labels = (
            default_labels if labels is None else [str(label) for label in labels]
        )
        if len(resolved_labels) != expected_label_count:
            raise ValueError(
                f"legend labels must contain {expected_label_count} entries for ear='{ear}'"
            )
        Legends.apply(ax=ax, labels=resolved_labels, location=location)


@dataclass
class LayoutFigure:
    fig: plt.Figure
    axes: np.ndarray
    layout: int
    positions: tuple[str, ...]
    figure_title_y: float

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

    def set_figure_title(self, title: str) -> None:
        visible_axes = [ax for ax in self.axes if ax.get_visible()]
        if len(visible_axes) == 0:
            figure_title_x = 0.5
        else:
            left = min(ax.get_position().x0 for ax in visible_axes)
            right = max(ax.get_position().x1 for ax in visible_axes)
            figure_title_x = (left + right) / 2.0
            for ax in visible_axes:
                subplot_title_y = getattr(
                    ax,
                    "hrtfpykit_subplot_title_y_with_figure_title",
                    None,
                )
                if subplot_title_y is not None:
                    ax.title.set_y(float(subplot_title_y))
        self.fig.suptitle(title, x=figure_title_x, y=self.figure_title_y)

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


def configure_rc() -> None:
    rc = Rc()
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


class Layout:
    layout: int = 1
    rows: int = 1
    cols: int = 1
    positions: tuple[str, ...] = ("main",)
    figsize: tuple[float, float] = (
        FigSizeDefault().width,
        FigSizeDefault().height,
    )
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.07
    subplot_title_y: float = 0.92

    @classmethod
    def create(
        cls,
        figsize: tuple[float, float] | None = None,
        margins: Margins | None = None,
    ) -> LayoutFigure:
        configure_rc()
        resolved_figsize = cls.figsize if figsize is None else figsize
        resolved_margins = Margins() if margins is None else margins
        fig, axes = plt.subplots(
            cls.rows,
            cls.cols,
            figsize=resolved_figsize,
            sharex=cls.sharex,
            sharey=cls.sharey,
            squeeze=False,
        )
        fig.subplots_adjust(
            left=resolved_margins.left,
            bottom=resolved_margins.bottom,
            right=resolved_margins.right,
            top=resolved_margins.top,
            wspace=resolved_margins.wspace,
            hspace=resolved_margins.hspace,
        )
        figure_title_y = min(resolved_margins.top + cls.figure_title_offset, 0.98)
        reshaped_axes = np.asarray(axes, dtype=object).reshape(-1)
        for ax in reshaped_axes:
            setattr(ax, "hrtfpykit_subplot_title_y", 1.0)
            setattr(
                ax,
                "hrtfpykit_subplot_title_y_with_figure_title",
                cls.subplot_title_y,
            )
        return LayoutFigure(
            fig=fig,
            axes=reshaped_axes,
            layout=cls.layout,
            positions=cls.positions,
            figure_title_y=figure_title_y,
        )


class Layout1(Layout):
    layout = 1
    rows = 1
    cols = 1
    positions = ("main",)
    figsize = (
        FigSizeDefault().width,
        FigSizeDefault().height,
    )
    figure_title_offset = 0.08
    subplot_title_y = 0.90


class Layout2Vertical(Layout):
    layout = 21
    rows = 2
    cols = 1
    positions = ("top", "bottom")
    figsize = (
        FigSizeDefault().width,
        FigSizeDefault().height,
    )
    sharex = True
    figure_title_offset = 0.08
    subplot_title_y = 0.90


class Layout2VerticalIndependent(Layout2Vertical):
    layout = 23
    sharex = False
    figsize = (8, 12)


class Layout3(Layout):
    layout = 3
    rows = 2
    cols = 2
    positions = ("top_left", "top_right", "bottom_left", "bottom_right")
    figsize = (
        FigSizeDefault().width + 2,
        FigSizeDefault().height + 1,
    )
    figure_title_offset = 0.08
    subplot_title_y = 0.98


class Layout2Horizontal(Layout):
    layout = 22
    rows = 1
    cols = 2
    positions = ("left", "right")
    figsize = (12, 6)
    sharex = False
    figure_title_offset = 0.08
    subplot_title_y = 0.98


class LayoutFactory:
    registry: dict[int, type[Layout]] = {
        1: Layout1,
        21: Layout2Vertical,
        22: Layout2Horizontal,
        23: Layout2VerticalIndependent,
        3: Layout3,
    }

    @classmethod
    def create(
        cls,
        layout: int,
        figsize: tuple[float, float] | None = None,
        margins: Margins | None = None,
    ) -> LayoutFigure:
        if layout not in cls.registry:
            raise ValueError(
                f"layout accepts: {', '.join(str(value) for value in cls.registry)}"
            )
        return cls.registry[layout].create(figsize=figsize, margins=margins)


class Projection:
    @staticmethod
    def create_layout(
        layout: int,
        projection: str,
        figsize: tuple[float, float] | None = None,
        margins: Margins | None = None,
    ) -> LayoutFigure:
        if layout not in LayoutFactory.registry:
            raise ValueError(
                f"layout accepts: {', '.join(str(value) for value in LayoutFactory.registry)}"
            )
        layout_class = LayoutFactory.registry[layout]
        configure_rc()
        resolved_figsize = layout_class.figsize if figsize is None else figsize
        resolved_margins = Margins() if margins is None else margins
        fig = plt.figure(figsize=resolved_figsize)
        fig.subplots_adjust(
            left=resolved_margins.left,
            bottom=resolved_margins.bottom,
            right=resolved_margins.right,
            top=resolved_margins.top,
            wspace=resolved_margins.wspace,
            hspace=resolved_margins.hspace,
        )
        axes: list[plt.Axes] = []
        shared_x_axis = None
        shared_y_axis = None
        for subplot_index in range(layout_class.rows * layout_class.cols):
            subplot_kwargs: dict[str, object] = {"projection": projection}
            if layout_class.sharex and shared_x_axis is not None:
                subplot_kwargs["sharex"] = shared_x_axis
            if layout_class.sharey and shared_y_axis is not None:
                subplot_kwargs["sharey"] = shared_y_axis
            ax = fig.add_subplot(
                layout_class.rows,
                layout_class.cols,
                subplot_index + 1,
                **subplot_kwargs,
            )
            if shared_x_axis is None:
                shared_x_axis = ax
            if shared_y_axis is None:
                shared_y_axis = ax
            setattr(ax, "hrtfpykit_subplot_title_y", 1.0)
            setattr(
                ax,
                "hrtfpykit_subplot_title_y_with_figure_title",
                layout_class.subplot_title_y,
            )
            axes.append(ax)
        return LayoutFigure(
            fig=fig,
            axes=np.asarray(axes, dtype=object),
            layout=layout_class.layout,
            positions=layout_class.positions,
            figure_title_y=min(
                resolved_margins.top + layout_class.figure_title_offset,
                0.98,
            ),
        )


class Polar:
    theta_tick_step: float = 30.0

    @staticmethod
    def create_horizontal_plane_curve(
        sources: Sources,
        planes,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices, _ = planes.get_horizontal_plane_indices(
            elevation=0.0,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError("Horizontal plane does not contain any source positions")

        spherical_positions = SourcePositionData.create_positions(
            sources=sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]
        azimuth_values = np.mod(np.asarray(spherical_positions[:, 0], dtype=float), 360.0)
        plane_values = np.asarray(values, dtype=float)[indices]
        if plane_values.ndim != 1:
            plane_values = np.asarray(plane_values, dtype=float).reshape(-1)

        sort_indices = np.argsort(azimuth_values)
        sorted_azimuth_values = azimuth_values[sort_indices]
        sorted_plane_values = plane_values[sort_indices]
        if sorted_azimuth_values.size > 1:
            theta_values = np.deg2rad(
                np.concatenate(
                    (
                        sorted_azimuth_values,
                        np.array([sorted_azimuth_values[0] + 360.0], dtype=float),
                    )
                )
            )
            radial_values = np.concatenate(
                (
                    sorted_plane_values,
                    np.array([sorted_plane_values[0]], dtype=float),
                )
            )
        else:
            theta_values = np.deg2rad(sorted_azimuth_values)
            radial_values = sorted_plane_values
        return theta_values, radial_values, sorted_plane_values


def create_layout(
    layout: int,
    figsize: tuple[float, float] | None = None,
    margins: Margins | None = None,
) -> LayoutFigure:
    return LayoutFactory.create(layout=layout, figsize=figsize, margins=margins)


class Axis:
    shared_x_visible: bool = True
    direction_tick_step: float = 20.0
    elevation_tick_step: float = 10.0
    azimuth_range_modes: tuple[str, str] = ("0-360", "-180-180")
    azimuth_limits_unsigned: tuple[float, float] = (0.0, 360.0)
    azimuth_limits_signed: tuple[float, float] = (-180.0, 180.0)
    lateral_limits: tuple[float, float] = (-90.0, 90.0)
    polar_limits: tuple[float, float] = (-90.0, 270.0)
    frequency_ticks_log: tuple[float, ...] = (
        250,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        20000,
    )
    frequency_tick_labels_log: tuple[str, ...] = (
        "0.25",
        "0.5",
        "1",
        "2",
        "4",
        "8",
        "16",
        "20",
    )
    frequency_ticks_linear: tuple[float, ...] = (
        2000,
        4000,
        6000,
        8000,
        10000,
        12000,
        14000,
        16000,
        18000,
        20000,
    )
    frequency_tick_labels_linear: tuple[str, ...] = (
        "2",
        "4",
        "6",
        "8",
        "10",
        "12",
        "14",
        "16",
        "18",
        "20",
    )
    frequency_margin_ratio: float = 0.03

    @staticmethod
    def create_label_axis(
        ax: plt.Axes,
        axis: str,
        default_label: str,
        options: AxisOptions | None = None,
    ) -> None:
        axis_options = AxisOptions() if options is None else options
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis accepts 'x', 'y', or 'z'")
        if axis == "x":
            set_label = ax.set_xlabel
            resolved_label = (
                default_label if axis_options.xlabel is None else axis_options.xlabel
            )
        elif axis == "y":
            set_label = ax.set_ylabel
            resolved_label = (
                default_label if axis_options.ylabel is None else axis_options.ylabel
            )
        else:
            set_label = getattr(ax, "set_zlabel", None)
            if set_label is None:
                raise ValueError("z-axis labeling requires a matplotlib 3D axis")
            resolved_label = default_label
        set_label(resolved_label)

    @staticmethod
    def create_direction_axis(
        ax: plt.Axes,
        axis: str,
        default_label: str,
        values: np.ndarray | None = None,
        tick_step: float | None = None,
        default_limits: tuple[float, float] | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        resolved_tick_step = (
            float(Axis.direction_tick_step) if tick_step is None else float(tick_step)
        )
        if resolved_tick_step <= 0.0:
            raise ValueError("tick_step must be positive")
        Axis.create_label_axis(
            ax=ax,
            axis=axis,
            default_label=default_label,
            options=options,
        )
        if axis == "x":
            set_limits = ax.set_xlim
            axis_object = ax.xaxis
        elif axis == "y":
            set_limits = ax.set_ylim
            axis_object = ax.yaxis
        elif axis == "z":
            set_limits = getattr(ax, "set_zlim", None)
            axis_object = getattr(ax, "zaxis", None)
            if set_limits is None or axis_object is None:
                raise ValueError("z-axis directional formatting requires a matplotlib 3D axis")
        else:
            raise ValueError("axis accepts 'x', 'y', or 'z'")

        if values is None:
            if default_limits is not None:
                set_limits(*default_limits)
                tick_start = np.floor(float(default_limits[0]) / resolved_tick_step) + 1.0
                tick_stop = np.ceil(float(default_limits[1]) / resolved_tick_step) - 1.0
                if tick_start <= tick_stop:
                    tick_values = tuple(
                        float(value)
                        for value in np.arange(
                            tick_start * resolved_tick_step,
                            (tick_stop + 1.0) * resolved_tick_step,
                            resolved_tick_step,
                        )
                    )
                else:
                    tick_values = ()
            else:
                tick_values = ()
            tick_labels = tuple(f"{int(np.rint(value))}" for value in tick_values)
            axis_object.set_major_locator(FixedLocator(tick_values))
            axis_object.set_major_formatter(FixedFormatter(tick_labels))
            return

        resolved_values = np.unique(np.asarray(values, dtype=float).reshape(-1))
        if resolved_values.size == 0:
            raise ValueError("direction axis values must contain at least one value")
        if not np.all(np.isfinite(resolved_values)):
            raise ValueError("direction axis values must be finite")

        if resolved_values.size == 1:
            axis_limits = (
                float(resolved_values[0]),
                float(resolved_values[0]),
            )
            tick_values = np.array([], dtype=float)
        else:
            axis_limits = (
                float(resolved_values[0]),
                float(resolved_values[-1]),
            )
            tick_start = np.floor(axis_limits[0] / resolved_tick_step) + 1.0
            tick_stop = np.ceil(axis_limits[1] / resolved_tick_step) - 1.0
            if tick_start <= tick_stop:
                tick_values = np.arange(
                    tick_start * resolved_tick_step,
                    (tick_stop + 1.0) * resolved_tick_step,
                    resolved_tick_step,
                    dtype=float,
                )
            else:
                tick_values = np.array([], dtype=float)

        if tick_values.size > 0:
            lower_label = int(np.rint(axis_limits[0]))
            upper_label = int(np.rint(axis_limits[1]))
            tick_labels_int = np.rint(tick_values).astype(int)
            keep_mask = (tick_labels_int != lower_label) & (tick_labels_int != upper_label)
            tick_values = tick_values[keep_mask]

        set_limits(*axis_limits)
        tick_positions = tuple(float(value) for value in tick_values)
        tick_labels = tuple(f"{int(np.rint(value))}" for value in tick_values)
        axis_object.set_major_locator(FixedLocator(tick_positions))
        axis_object.set_major_formatter(FixedFormatter(tick_labels))

    @staticmethod
    def create_panel_axis(
        ax: plt.Axes,
        axis: str,
        default_label: str,
        selected_positions: np.ndarray,
        position_coordinate_system: str,
        ear: str,
        options: AxisOptions | None = None,
        legend_location: str = "upper right",
    ) -> None:
        axis_options = AxisOptions() if options is None else options
        legend_options = LegendOptions() if axis_options.legend is None else axis_options.legend
        shared_x_visible = (
            Axis.shared_x_visible
            if axis_options.shared_x_visible is None
            else axis_options.shared_x_visible
        )
        default_title = Axis.create_position_title(
            selected_positions=selected_positions,
            position_coordinate_system=position_coordinate_system,
        )
        Axis.create_label_axis(
            ax=ax,
            axis=axis,
            default_label=default_label,
            options=axis_options,
        )
        resolved_title = default_title if axis_options.title is None else axis_options.title
        subplot_title_y = getattr(ax, "hrtfpykit_subplot_title_y", None)
        if subplot_title_y is None:
            ax.set_title(resolved_title)
        else:
            ax.set_title(resolved_title, y=float(subplot_title_y))
        if shared_x_visible:
            ax.tick_params(axis="x", which="both", labelbottom=True)
        legend_enabled = True if legend_options.enabled is None else legend_options.enabled
        if legend_enabled:
            resolved_legend_location = (
                legend_location
                if legend_options.location is None
                else legend_options.location
            )
            Legends.ears_legend(
                ax=ax,
                ear=ear,
                location=resolved_legend_location,
                labels=legend_options.labels,
            )
        grid_enabled = True if axis_options.grid is None else axis_options.grid
        if grid_enabled:
            ax.grid(True)

    @staticmethod
    def create_position_title(
        selected_positions: np.ndarray,
        position_coordinate_system: str,
    ) -> str:
        titles = Titles()
        position_alias = Sources.get_position_alias(
            selected_positions,
            coordinate_system=position_coordinate_system,
        )
        if position_coordinate_system == "spherical":
            if position_alias is None:
                default_title = titles.spherical_position.format(
                    az=float(selected_positions[0]),
                    el=float(selected_positions[1]),
                )
            else:
                default_title = titles.spherical_alias.format(
                    name=position_alias.capitalize(),
                    az=float(selected_positions[0]),
                    el=float(selected_positions[1]),
                )
        elif position_coordinate_system == "cartesian":
            if position_alias is None:
                default_title = titles.cartesian_position.format(
                    x=float(selected_positions[0]),
                    y=float(selected_positions[1]),
                    z=float(selected_positions[2]),
                )
            else:
                default_title = titles.cartesian_alias.format(
                    name=position_alias.capitalize(),
                    x=float(selected_positions[0]),
                    y=float(selected_positions[1]),
                    z=float(selected_positions[2]),
                )
        elif position_coordinate_system == "lateral-polar":
            if position_alias is None:
                return titles.lateral_polar_position.format(
                    lateral=float(selected_positions[0]),
                    polar=float(selected_positions[1]),
                )
            else:
                return titles.lateral_polar_alias.format(
                    name=position_alias.capitalize(),
                    lateral=float(selected_positions[0]),
                    polar=float(selected_positions[1]),
                )
        else:
            raise ValueError(
                "position_coordinate_system accepts spherical, cartesian, or lateral-polar"
            )
        return default_title

    @staticmethod
    def create_frequency_axis(
        ax: plt.Axes | None,
        axis: str,
        x_axis: str,
        frequency_bins: np.ndarray | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        label: str | None = None,
        options: FrequencyAxisOptions | None = None,
    ) -> FrequencyAxisOptions:
        frequency_axis_options = FrequencyAxisOptions() if options is None else options
        resolved_frequency_bins = None
        if frequency_bins is not None:
            resolved_frequency_bins = np.asarray(frequency_bins, dtype=float)
            if resolved_frequency_bins.ndim != 1 or resolved_frequency_bins.size == 0:
                raise ValueError("frequency_bins must be a non-empty 1D array")
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis accepts 'x', 'y', or 'z'")
        x_axis_key = str(x_axis).strip().lower()
        if x_axis_key not in AcceptedParameters().frequency_x_axes:
            raise ValueError("x_axis accepts log or linear")
        requested_freq_min = (
            frequency_axis_options.freq_min if freq_min is None else freq_min
        )
        if requested_freq_min is None:
            if resolved_frequency_bins is None:
                raise ValueError("freq_min is required when frequency_bins are not provided")
            if x_axis_key == "log":
                positive_frequency_bins = resolved_frequency_bins[
                    resolved_frequency_bins > 0.0
                ]
                if positive_frequency_bins.size == 0:
                    raise ValueError(
                        "frequency_bins must include a positive value for logarithmic frequency axis"
                    )
                resolved_freq_min = float(np.min(positive_frequency_bins))
            else:
                resolved_freq_min = float(np.min(resolved_frequency_bins))
        else:
            resolved_freq_min = float(requested_freq_min)
        requested_freq_max = (
            frequency_axis_options.freq_max if freq_max is None else freq_max
        )
        if requested_freq_max is None:
            if resolved_frequency_bins is None:
                raise ValueError("freq_max is required when frequency_bins are not provided")
            resolved_freq_max = float(np.max(resolved_frequency_bins))
        else:
            resolved_freq_max = float(requested_freq_max)
        resolved_margin_ratio = (
            float(Axis.frequency_margin_ratio)
            if frequency_axis_options.margin_ratio is None
            else float(frequency_axis_options.margin_ratio)
        )
        if not np.isfinite(resolved_freq_min) or not np.isfinite(resolved_freq_max):
            raise ValueError("freq_min and freq_max must be finite values")
        if resolved_freq_min >= resolved_freq_max:
            raise ValueError("freq_min must be smaller than freq_max")
        if x_axis_key == "log" and resolved_freq_min <= 0.0:
            raise ValueError("freq_min must be positive for logarithmic frequency axis")
        if resolved_margin_ratio < 0.0:
            raise ValueError("margin_ratio must be non-negative")

        default_ticks = (
            Axis.frequency_ticks_log
            if x_axis_key == "log"
            else Axis.frequency_ticks_linear
        )
        default_labels = (
            Axis.frequency_tick_labels_log
            if x_axis_key == "log"
            else Axis.frequency_tick_labels_linear
        )
        resolved_ticks = (
            tuple(float(tick) for tick in default_ticks)
            if frequency_axis_options.ticks is None
            else tuple(float(tick) for tick in frequency_axis_options.ticks)
        )
        resolved_labels = (
            tuple(str(label) for label in default_labels)
            if frequency_axis_options.labels is None
            and frequency_axis_options.ticks is None
            else (
                tuple(f"{tick / 1000.0:g}" for tick in resolved_ticks)
                if frequency_axis_options.labels is None
                else tuple(str(label) for label in frequency_axis_options.labels)
            )
        )
        if len(resolved_ticks) != len(resolved_labels):
            raise ValueError("frequency axis ticks and labels must have the same length")
        if x_axis_key == "log" and any(tick <= 0.0 for tick in resolved_ticks):
            raise ValueError("frequency axis ticks must be positive for logarithmic axis")

        visible_pairs = tuple(
            (tick, tick_label)
            for tick, tick_label in zip(resolved_ticks, resolved_labels)
            if resolved_freq_min <= tick <= resolved_freq_max
        )

        resolved_frequency_axis = FrequencyAxisOptions(
            ticks=tuple(tick for tick, _ in visible_pairs),
            labels=tuple(tick_label for _, tick_label in visible_pairs),
            freq_min=resolved_freq_min,
            freq_max=resolved_freq_max,
            margin_ratio=resolved_margin_ratio,
        )
        if ax is None:
            return resolved_frequency_axis

        ticks_khz = [tick / 1000.0 for tick in resolved_frequency_axis.ticks or ()]
        resolved_freq_min_khz = float(resolved_frequency_axis.freq_min) / 1000.0
        resolved_freq_max_khz = float(resolved_frequency_axis.freq_max) / 1000.0
        margin_ratio = float(resolved_frequency_axis.margin_ratio)

        if axis == "x":
            axis_object = ax.xaxis
            set_scale = ax.set_xscale
            set_limits = ax.set_xlim
            set_label = ax.set_xlabel
        elif axis == "y":
            axis_object = ax.yaxis
            set_scale = ax.set_yscale
            set_limits = ax.set_ylim
            set_label = ax.set_ylabel
        else:
            axis_object = getattr(ax, "zaxis", None)
            set_scale = getattr(ax, "set_zscale", None)
            set_limits = getattr(ax, "set_zlim", None)
            set_label = getattr(ax, "set_zlabel", None)
            if axis_object is None or set_scale is None or set_limits is None:
                raise ValueError("z-axis formatting requires a matplotlib 3D axis")
            if set_label is None:
                raise ValueError("z-axis labeling requires a matplotlib 3D axis")
        if label is not None:
            set_label(label)

        if x_axis_key == "log":
            set_scale("log")
            log_min = np.log10(resolved_freq_min_khz)
            log_max = np.log10(resolved_freq_max_khz)
            margin_log = (log_max - log_min) * margin_ratio
            axis_min = 10 ** (log_min - margin_log)
            axis_max = 10 ** (log_max + margin_log)
        else:
            set_scale("linear")
            margin_linear = (resolved_freq_max_khz - resolved_freq_min_khz) * margin_ratio
            axis_min = resolved_freq_min_khz - margin_linear
            axis_max = resolved_freq_max_khz + margin_linear
        set_limits(axis_min, axis_max)
        axis_object.set_major_locator(FixedLocator(ticks_khz))
        axis_object.set_major_formatter(FixedFormatter(resolved_frequency_axis.labels or ()))
        axis_object.set_minor_locator(NullLocator())
        axis_object.set_minor_formatter(NullFormatter())
        if hasattr(axis_object, "offsetText"):
            axis_object.offsetText.set_visible(False)
        return resolved_frequency_axis

    @staticmethod
    def create_magnitude_axis(
        ax: plt.Axes,
        axis: str,
        unit: str,
        selected_positions: np.ndarray,
        position_coordinate_system: str,
        ear: str,
        options: AxisOptions | None = None,
        legend_location: str = "upper left",
    ) -> None:
        labels = Labels()
        default_label = labels.magnitude_db if unit == "db" else labels.magnitude_linear
        Axis.create_panel_axis(
            ax=ax,
            axis=axis,
            default_label=default_label,
            selected_positions=selected_positions,
            position_coordinate_system=position_coordinate_system,
            ear=ear,
            options=options,
            legend_location=legend_location,
        )

    @staticmethod
    def create_amplitude_axis(
        ax: plt.Axes,
        axis: str,
        selected_positions: np.ndarray,
        position_coordinate_system: str,
        ear: str,
        options: AxisOptions | None = None,
    ) -> None:
        labels = Labels()
        Axis.create_panel_axis(
            ax=ax,
            axis=axis,
            default_label=labels.impulse_response,
            selected_positions=selected_positions,
            position_coordinate_system=position_coordinate_system,
            ear=ear,
            options=options,
        )

    @staticmethod
    def create_time_axis(
        ax: plt.Axes,
        axis: str,
        options: AxisOptions | None = None,
    ) -> None:
        labels = Labels()
        Axis.create_label_axis(
            ax=ax,
            axis=axis,
            default_label=labels.time,
            options=options,
        )

    @staticmethod
    def create_samples_axis(
        ax: plt.Axes,
        axis: str,
        options: AxisOptions | None = None,
    ) -> None:
        labels = Labels()
        Axis.create_label_axis(
            ax=ax,
            axis=axis,
            default_label=labels.samples,
            options=options,
        )

    @staticmethod
    def get_azimuth_range_mode(
        options: AxisOptions | None = None,
    ) -> str:
        axis_options = AxisOptions() if options is None else options
        azimuth_axis_options = (
            AzimuthAxisOptions()
            if axis_options.azimuth_axis is None
            else axis_options.azimuth_axis
        )
        resolved_range_mode = (
            Axis.azimuth_range_modes[0]
            if azimuth_axis_options.range_mode is None
            else str(azimuth_axis_options.range_mode).strip()
        )
        if resolved_range_mode not in Axis.azimuth_range_modes:
            raise ValueError(
                "azimuth axis range_mode accepts "
                f"{Axis.azimuth_range_modes[0]} or {Axis.azimuth_range_modes[1]}"
            )
        return resolved_range_mode

    @staticmethod
    def transform_azimuth_values(
        values: np.ndarray,
        options: AxisOptions | None = None,
    ) -> np.ndarray:
        resolved_values = np.asarray(values, dtype=float)
        resolved_range_mode = Axis.get_azimuth_range_mode(options=options)
        if resolved_range_mode == Axis.azimuth_range_modes[0]:
            return np.mod(resolved_values, 360.0)
        transformed_values = np.mod(resolved_values + 180.0, 360.0) - 180.0
        transformed_values[np.isclose(transformed_values, -180.0, atol=1e-8, rtol=0.0)] = 180.0
        return transformed_values

    @staticmethod
    def create_azimuth_axis(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        labels = Labels()
        resolved_range_mode = Axis.get_azimuth_range_mode(options=options)
        transformed_values = (
            None
            if values is None
            else Axis.transform_azimuth_values(values=values, options=options)
        )
        Axis.create_direction_axis(
            ax=ax,
            axis=axis,
            default_label=labels.azimuth,
            values=transformed_values,
            tick_step=Axis.direction_tick_step,
            default_limits=(
                Axis.azimuth_limits_unsigned
                if resolved_range_mode == Axis.azimuth_range_modes[0]
                else Axis.azimuth_limits_signed
            ),
            options=options,
        )

    @staticmethod
    def create_elevation_axis(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        labels = Labels()
        Axis.create_direction_axis(
            ax=ax,
            axis=axis,
            default_label=labels.elevation,
            values=values,
            tick_step=Axis.elevation_tick_step,
            options=options,
        )

    @staticmethod
    def create_lateral_axis(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        labels = Labels()
        Axis.create_direction_axis(
            ax=ax,
            axis=axis,
            default_label=labels.lateral,
            values=values,
            tick_step=Axis.elevation_tick_step,
            default_limits=Axis.lateral_limits,
            options=options,
        )

    @staticmethod
    def create_polar_axis(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        labels = Labels()
        Axis.create_direction_axis(
            ax=ax,
            axis=axis,
            default_label=labels.polar,
            values=values,
            tick_step=Axis.direction_tick_step,
            default_limits=Axis.polar_limits,
            options=options,
        )

    @staticmethod
    def create_plane_angle_axis(
        ax: plt.Axes,
        axis: str,
        plane: str,
        values: np.ndarray | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        plane_key = str(plane).strip().lower()
        if plane_key == "horizontal":
            Axis.create_azimuth_axis(ax=ax, axis=axis, values=values, options=options)
            return
        if plane_key == "median":
            Axis.create_polar_axis(ax=ax, axis=axis, values=values, options=options)
            return
        raise ValueError("plane accepts horizontal or median")

    @staticmethod
    def create_plane_title(
        plane: str,
    ) -> str:
        plane_key = str(plane).strip().lower()
        titles = Titles()
        if plane_key == "horizontal":
            return titles.horizontal_plane
        if plane_key == "median":
            return titles.median_plane
        raise ValueError("plane accepts horizontal or median")

    @staticmethod
    def create_elevation_spectrum_title(
        real_azimuth: float,
    ) -> str:
        titles = Titles()
        return titles.elevation_spectrum.format(angle=float(real_azimuth))


class Plots:
    #  Inheritance. All methods will accept a instance of HRTF , then HRTF will inherit from Plots
    def plot_magnitude(
        self: "HRTF",
        positions: str | list | np.ndarray = "front",
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = 1.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """Plot HRTF magnitude responses for up to four source positions.

        Parameters
        ----------
        positions : str | list | np.ndarray, default="front"
            One position or a collection of positions. Named aliases such as
            ``"front"``, ``"back"``, ``"left"``, and ``"right"`` are accepted.
            Up to four positions can be shown in one figure.
        x_axis : {"linear", "log"}, default="linear"
            Frequency scale used on the x axis.
        unit : {"db", "linear"}, default="db"
            Magnitude representation used on the y axis.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display. When ``"both"`` is selected, left and right
            responses are drawn together in each subplot.
        reference : float | {"max"}, default=1.0
            Reference used when ``unit="db"``. ``"max"`` normalizes the plotted
            magnitude to the maximum selected value.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, legend, frequency-axis, and per-panel overrides.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.

        Returns
        -------
        None

        Use Cases
        ---------
        - Compare magnitude responses across several source positions.
        - Inspect left, right, or binaural magnitude structure at one location.
        - Create figures without showing them immediately.

        Examples
        --------
        Plot the default front position:

        >>> hrtf.plot_magnitude()

        Plot left and right positions with both ears:

        >>> hrtf.plot_magnitude(positions=["left", "right"], ear="both")

        Plot one position in linear magnitude with a logarithmic frequency axis:

        >>> hrtf.plot_magnitude(positions="front", unit="linear", x_axis="log")

        Create the figure without showing it immediately:

        >>> hrtf.plot_magnitude(show=False)
        """
        accepted_parameters = AcceptedParameters()
        if unit not in accepted_parameters.units:
            raise AttributeError(
                f"unit accepts : {accepted_parameters.units[0]} or {accepted_parameters.units[1]}"
            )
        if x_axis not in accepted_parameters.frequency_x_axes:
            raise AttributeError(
                "x_axis accepts "
                f"{accepted_parameters.frequency_x_axes[0]} or "
                f"{accepted_parameters.frequency_x_axes[1]}"
            )
        if ear not in accepted_parameters.ears:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ears[0]}, {accepted_parameters.ears[1]} or {accepted_parameters.ears[2]}"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.TF.values is None or self.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        position_queries = self.Sources.get_position_queries(positions)
        position_count = len(position_queries)
        if position_count == 0:
            raise ValueError("At least one position is required")
        if position_count > 4:
            raise ValueError("plot_magnitude accepts up to 4 positions")

        layout_number = 1 if position_count == 1 else 21 if position_count == 2 else 3
        layout = create_layout(
            layout=layout_number,
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        selected_position_info = [
            self.Sources.get_position_index(
                selected_position_query,
                coordinate_system="spherical",
            )
            for selected_position_query in position_queries
        ]
        tf_magnitude = self.TF.magnitude
        if unit == "db":
            if isinstance(reference, str) and str(reference).strip().lower() == "max":
                selected_indices = [selected_index for selected_index, _ in selected_position_info]
                reference_values = np.asarray(tf_magnitude[selected_indices], dtype=float)
                if ear != "both" and reference_values.ndim >= 3:
                    ear_index = 0 if ear == "left" else 1
                    if reference_values.shape[1] <= ear_index:
                        raise ValueError(f"Requested ear '{ear}' is not available in TF data")
                    reference_values = reference_values[:, ear_index, :]
                plot_reference = float(np.max(reference_values))
                tf_values = magnitude_to_db(tf_magnitude, reference=plot_reference)
            else:
                tf_values = magnitude_to_db(tf_magnitude, reference=reference)
        else:
            tf_values = tf_magnitude
        labels = Labels()
        magnitude_legend_location = "upper right" if x_axis == "linear" else "upper left"

        for index, (_, selected_positions) in enumerate(selected_position_info):
            ax = layout.get_axis(index)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(index))
            resolved_frequency_axis = Axis.create_frequency_axis(
                ax=None,
                axis="x",
                x_axis=x_axis,
                frequency_bins=frequency_bins_hz,
                freq_min=freq_min,
                freq_max=freq_max,
                options=resolved_axis_options.frequency_axis,
            )
            frequency_mask = (
                (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
                & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
            )
            if not np.any(frequency_mask):
                raise ValueError("Selected frequency range produced no TF bins")
            frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0
            frequency_label = (
                labels.frequency
                if resolved_axis_options.xlabel is None
                else resolved_axis_options.xlabel
            )
            idxs = int(selected_position_info[index][0])
            selected_positions = np.asarray(selected_positions, dtype=float)
            y_values = np.asarray(tf_values[idxs][..., frequency_mask], dtype=float)

            if ear == "both":
                if y_values.ndim < 2 or y_values.shape[0] < 2:
                    raise ValueError("Both ears requested but TF data does not contain two ear channels")
                ax.plot(frequency_khz, y_values[0, :], color='blue')
                ax.plot(frequency_khz, y_values[1, :], color='red')
            else:
                if y_values.ndim == 1:
                    selected_y_values = y_values.reshape(-1)
                else:
                    ear_index = 0 if ear == "left" else 1
                    if y_values.shape[0] <= ear_index:
                        raise ValueError(f"Requested ear '{ear}' is not available in TF data")
                    selected_y_values = np.asarray(y_values[ear_index], dtype=float).reshape(-1)
                ax.plot(frequency_khz, selected_y_values, color='blue')

            Axis.create_frequency_axis(
                ax=ax,
                axis="x",
                x_axis=x_axis,
                label=frequency_label,
                options=resolved_frequency_axis,
            )
            Axis.create_magnitude_axis(
                ax=ax,
                axis="y",
                unit=unit,
                selected_positions=selected_positions,
                position_coordinate_system="spherical",
                ear=ear,
                options=resolved_axis_options,
                legend_location=magnitude_legend_location,
            )

        if position_count < layout.axes.size:
            for ax in layout.axes[position_count:]:
                ax.set_visible(False)

        if figure_options.title is not None:
            layout.set_figure_title(figure_options.title)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_amplitude(
        self: "HRTF",
        positions: str | list | np.ndarray = "front",
        ear: str = "both",
        x_axis: str = "time",
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """Plot HRIR amplitude responses for up to four source positions.

        Parameters
        ----------
        positions : str | list | np.ndarray, default="front"
            One position or a collection of positions. Named aliases such as
            ``"front"``, ``"back"``, ``"left"``, and ``"right"`` are accepted.
            Up to four positions can be shown in one figure.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display. When ``"both"`` is selected, left and right
            ear waveforms are drawn together in each subplot.
        x_axis : {"time", "samples"}, default="time"
            Horizontal axis used for the waveform plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, legend, and per-panel overrides.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect HRIR waveform shape for one or several directions.
        - Compare left and right ear impulse responses at the same position.
        - Create waveform figures without showing them immediately.

        Examples
        --------
        Plot the default front position:

        >>> hrtf.plot_amplitude()

        Plot two positions using sample index on the x axis:

        >>> hrtf.plot_amplitude(positions=["front", "left"], x_axis="samples")

        Plot a single ear at a named position:

        >>> hrtf.plot_amplitude(positions="right", ear="left")

        Create the figure without showing it immediately:

        >>> hrtf.plot_amplitude(show=False)
        """
        accepted_parameters = AcceptedParameters()
        if ear not in accepted_parameters.ears:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ears[0]}, {accepted_parameters.ears[1]} or {accepted_parameters.ears[2]}"
            )
        if x_axis not in accepted_parameters.x_axes:
            raise AttributeError(
                f"x_axis accepts : {accepted_parameters.x_axes[0]} or {accepted_parameters.x_axes[1]}"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if x_axis == "time" and self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required when x_axis='time'")

        position_queries = self.Sources.get_position_queries(positions)
        position_count = len(position_queries)
        if position_count == 0:
            raise ValueError("At least one position is required")
        if position_count > 4:
            raise ValueError("plot_amplitude accepts up to 4 positions")

        layout_number = 1 if position_count == 1 else 21 if position_count == 2 else 3
        layout = create_layout(
            layout=layout_number,
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        ir_values = np.asarray(self.IR.values, dtype=float)
        if ir_values.ndim < 2 or ir_values.shape[-1] == 0:
            raise ValueError("IR values must contain at least one sample")
        sample_indexes = np.arange(ir_values.shape[-1], dtype=float)
        if x_axis == "time":
            x_values = sample_indexes / float(self.IR.sample_rate)
        else:
            x_values = sample_indexes

        for index, selected_position_query in enumerate(position_queries):
            ax = layout.get_axis(index)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(index))
            idxs, selected_positions = self.Sources.get_position_index(
                selected_position_query,
                coordinate_system="spherical",
            )
            selected_positions = np.asarray(selected_positions, dtype=float)
            y_values = np.asarray(ir_values[idxs], dtype=float)

            if ear == "both":
                if y_values.ndim < 2 or y_values.shape[0] < 2:
                    raise ValueError("Both ears requested but IR data does not contain two ear channels")
                ax.plot(x_values, y_values[0, :], color="blue")
                ax.plot(x_values, y_values[1, :], color="red")
            else:
                if y_values.ndim == 1:
                    selected_y_values = y_values.reshape(-1)
                else:
                    ear_index = 0 if ear == "left" else 1
                    if y_values.shape[0] <= ear_index:
                        raise ValueError(f"Requested ear '{ear}' is not available in IR data")
                    selected_y_values = np.asarray(y_values[ear_index], dtype=float).reshape(-1)
                ax.plot(x_values, selected_y_values, color="blue")

            if x_axis == "time":
                Axis.create_time_axis(
                    ax=ax,
                    axis="x",
                    options=resolved_axis_options,
                )
            else:
                Axis.create_samples_axis(
                    ax=ax,
                    axis="x",
                    options=resolved_axis_options,
                )
            Axis.create_amplitude_axis(
                ax=ax,
                axis="y",
                selected_positions=selected_positions,
                position_coordinate_system="spherical",
                ear=ear,
                options=resolved_axis_options,
            )

        if position_count < layout.axes.size:
            for ax in layout.axes[position_count:]:
                ax.set_visible(False)

        if figure_options.title is not None:
            layout.set_figure_title(figure_options.title)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_amplitude_and_magnitude(
        self: "HRTF",
        position: str | list | np.ndarray = "front",
        ear: str = "both",
        x_axis: str = "time",
        frequency_x_axis: str = "linear",
        magnitude: str = "db",
        reference: float | str = 1.0,
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """Plot amplitude and magnitude views for a single source position.

        Parameters
        ----------
        position : str | list | np.ndarray, default="front"
            Position query to plot. Exactly one position is accepted. Named
            aliases such as ``"front"``, ``"back"``, ``"left"``, and ``"right"``
            are accepted.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display in both subplots.
        x_axis : {"time", "samples"}, default="time"
            Horizontal axis used for the amplitude subplot.
        frequency_x_axis : {"linear", "log"}, default="linear"
            Frequency-axis scale used on the magnitude subplot.
        magnitude : {"db", "linear"}, default="db"
            Magnitude representation used on the bottom subplot.
        reference : float | {"max"}, default=1.0
            Reference used when ``magnitude="db"`` for the magnitude subplot.
        options : PlotOptions | None, default=None
            Optional figure, axis, legend, frequency-axis, and panel overrides.
            Frequency-range control for the magnitude subplot should be passed
            through ``options.axis.frequency_axis`` or the bottom-panel override.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect time-domain and frequency-domain behavior for the same direction.
        - Compare left and right ear waveform and magnitude structure together.
        - Create a compact two-panel summary for one position.

        Examples
        --------
        Plot the default front position:

        >>> hrtf.plot_amplitude_and_magnitude()

        Plot one position using sample index and linear magnitude:

        >>> hrtf.plot_amplitude_and_magnitude(
        ...     position="left",
        ...     x_axis="samples",
        ...     magnitude="linear",
        ... )

        Plot both ears with a logarithmic frequency axis:

        >>> hrtf.plot_amplitude_and_magnitude(
        ...     position="front",
        ...     ear="both",
        ...     frequency_x_axis="log",
        ... )

        Create the figure without showing it immediately:

        >>> hrtf.plot_amplitude_and_magnitude(show=False)
        """
        accepted_parameters = AcceptedParameters()
        if ear not in accepted_parameters.ears:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ears[0]}, {accepted_parameters.ears[1]} or {accepted_parameters.ears[2]}"
            )
        if x_axis not in accepted_parameters.x_axes:
            raise AttributeError(
                f"x_axis accepts : {accepted_parameters.x_axes[0]} or {accepted_parameters.x_axes[1]}"
            )
        if frequency_x_axis not in accepted_parameters.frequency_x_axes:
            raise AttributeError(
                "frequency_x_axis accepts "
                f"{accepted_parameters.frequency_x_axes[0]} or "
                f"{accepted_parameters.frequency_x_axes[1]}"
            )
        if magnitude not in accepted_parameters.units:
            raise AttributeError(
                "magnitude accepts : "
                f"{accepted_parameters.units[0]} or {accepted_parameters.units[1]}"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.TF.values is None or self.TF.frequency_bins is None:
            raise ValueError("TF data is not available")
        if x_axis == "time" and self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required when x_axis='time'")

        position_queries = self.Sources.get_position_queries(position)
        if len(position_queries) != 1:
            raise ValueError(
                "plot_amplitude_and_magnitude accepts exactly one position"
            )
        selected_position_query = position_queries[0]

        layout = create_layout(
            layout=23,
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        top_axis_options = axis_options.merge(panel_axis_options.get(0))
        bottom_axis_options = axis_options.merge(panel_axis_options.get(1))
        top_axis_panel_options = top_axis_options.merge(AxisOptions(title=""))
        bottom_axis_panel_options = bottom_axis_options.merge(AxisOptions(title=""))

        idxs, selected_positions = self.Sources.get_position_index(
            selected_position_query,
            coordinate_system="spherical",
        )
        selected_positions = np.asarray(selected_positions, dtype=float)

        ir_values = np.asarray(self.IR.values, dtype=float)
        if ir_values.ndim < 2 or ir_values.shape[-1] == 0:
            raise ValueError("IR values must contain at least one sample")
        sample_indexes = np.arange(ir_values.shape[-1], dtype=float)
        x_values = (
            sample_indexes / float(self.IR.sample_rate)
            if x_axis == "time"
            else sample_indexes
        )
        ir_y_values = np.asarray(ir_values[idxs], dtype=float)

        ir_ax = layout.get_axis("top")
        if ear == "both":
            if ir_y_values.ndim < 2 or ir_y_values.shape[0] < 2:
                raise ValueError(
                    "Both ears requested but IR data does not contain two ear channels"
                )
            ir_ax.plot(x_values, ir_y_values[0, :], color="blue")
            ir_ax.plot(x_values, ir_y_values[1, :], color="red")
        else:
            if ir_y_values.ndim == 1:
                selected_ir_y_values = ir_y_values.reshape(-1)
            else:
                ear_index = 0 if ear == "left" else 1
                if ir_y_values.shape[0] <= ear_index:
                    raise ValueError(
                        f"Requested ear '{ear}' is not available in IR data"
                    )
                selected_ir_y_values = np.asarray(
                    ir_y_values[ear_index],
                    dtype=float,
                ).reshape(-1)
            ir_ax.plot(x_values, selected_ir_y_values, color="blue")

        if x_axis == "time":
            Axis.create_time_axis(
                ax=ir_ax,
                axis="x",
                options=top_axis_options,
            )
        else:
            Axis.create_samples_axis(
                ax=ir_ax,
                axis="x",
                options=top_axis_options,
            )
        Axis.create_amplitude_axis(
            ax=ir_ax,
            axis="y",
            selected_positions=selected_positions,
            position_coordinate_system="spherical",
            ear=ear,
            options=top_axis_panel_options,
        )

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        resolved_frequency_axis = Axis.create_frequency_axis(
            ax=None,
            axis="x",
            x_axis=frequency_x_axis,
            frequency_bins=frequency_bins_hz,
            options=bottom_axis_options.frequency_axis,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
            & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0
        tf_magnitude = self.TF.magnitude
        if magnitude == "db":
            if isinstance(reference, str) and str(reference).strip().lower() == "max":
                reference_values = np.asarray(tf_magnitude[idxs], dtype=float)
                if ear != "both" and reference_values.ndim >= 2:
                    ear_index = 0 if ear == "left" else 1
                    if reference_values.shape[0] <= ear_index:
                        raise ValueError(
                            f"Requested ear '{ear}' is not available in TF data"
                        )
                    reference_values = reference_values[ear_index]
                plot_reference = float(np.max(reference_values))
                tf_values = magnitude_to_db(tf_magnitude, reference=plot_reference)
            else:
                tf_values = magnitude_to_db(tf_magnitude, reference=reference)
        else:
            tf_values = tf_magnitude
        magnitude_y_values = np.asarray(tf_values[idxs][..., frequency_mask], dtype=float)

        magnitude_ax = layout.get_axis("bottom")
        if ear == "both":
            if magnitude_y_values.ndim < 2 or magnitude_y_values.shape[0] < 2:
                raise ValueError(
                    "Both ears requested but TF data does not contain two ear channels"
                )
            magnitude_ax.plot(frequency_khz, magnitude_y_values[0, :], color="blue")
            magnitude_ax.plot(frequency_khz, magnitude_y_values[1, :], color="red")
        else:
            if magnitude_y_values.ndim == 1:
                selected_magnitude_y_values = magnitude_y_values.reshape(-1)
            else:
                ear_index = 0 if ear == "left" else 1
                if magnitude_y_values.shape[0] <= ear_index:
                    raise ValueError(
                        f"Requested ear '{ear}' is not available in TF data"
                    )
                selected_magnitude_y_values = np.asarray(
                    magnitude_y_values[ear_index],
                    dtype=float,
                ).reshape(-1)
            magnitude_ax.plot(frequency_khz, selected_magnitude_y_values, color="blue")

        labels = Labels()
        magnitude_legend_location = (
            "upper right" if frequency_x_axis == "linear" else "upper left"
        )
        frequency_label = (
            labels.frequency
            if bottom_axis_options.xlabel is None
            else bottom_axis_options.xlabel
        )
        Axis.create_frequency_axis(
            ax=magnitude_ax,
            axis="x",
            x_axis=frequency_x_axis,
            label=frequency_label,
            options=resolved_frequency_axis,
        )
        Axis.create_magnitude_axis(
            ax=magnitude_ax,
            axis="y",
            unit=magnitude,
            selected_positions=selected_positions,
            position_coordinate_system="spherical",
            ear=ear,
            options=bottom_axis_panel_options,
            legend_location=magnitude_legend_location,
        )

        resolved_figure_title = (
            Axis.create_position_title(
                selected_positions=selected_positions,
                position_coordinate_system="spherical",
            )
            if figure_options.title is None
            else figure_options.title
        )
        layout.set_figure_title(resolved_figure_title)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_plane_spectrum(
        self: "HRTF",
        plane: str = "median",
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = "max",
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """Plot a frequency-angle spectrum heatmap for a canonical HRTF plane.

        Parameters
        ----------
        plane : {"horizontal", "median"}, default="median"
            Canonical plane to visualize. ``"horizontal"`` uses the horizontal
            plane at ``0`` degrees elevation. ``"median"`` uses the canonical
            median plane defined by the front-back sagittal path.
        x_axis : {"linear", "log"}, default="linear"
            Frequency scale used on the x axis.
        unit : {"db", "linear"}, default="db"
            Magnitude representation used for the heatmap values.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display. When ``"both"`` is selected, a separate panel
            is created for each ear.
        reference : float | {"max"}, default="max"
            Reference used when ``unit="db"``. ``"max"`` normalizes the plotted
            plane to its maximum value.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, heatmap, and panel overrides. For the
            horizontal plane, ``options.axis.azimuth_axis`` can be used to choose
            the azimuth plotting convention, for example ``"-180-180"`` or
            ``"0-360"``.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the canonical horizontal-plane spectrum over azimuth.
        - Inspect the canonical median-plane spectrum over polar angle.
        - Compare left and right ear spectral structure in the same plane.
        - Create plane-based HRTF heatmaps without showing them immediately.

        Examples
        --------
        Plot the canonical median plane with default settings:

        >>> hrtf.plot_plane_spectrum()

        Plot the horizontal plane for the left ear only:

        >>> hrtf.plot_plane_spectrum(plane="horizontal", ear="left")

        Plot the horizontal plane using signed azimuth values:

        >>> hrtf.plot_plane_spectrum(
        ...     plane="horizontal",
        ...     options=PlotOptions(
        ...         axis=AxisOptions(
        ...             azimuth_axis=AzimuthAxisOptions(range_mode="-180-180")
        ...         )
        ...     ),
        ... )

        Create the figure without showing it immediately:

        >>> hrtf.plot_plane_spectrum(show=False)
        """
        accepted_parameters = AcceptedParameters()
        if plane not in ("horizontal", "median"):
            raise AttributeError(
                "plot_plane_spectrum plane accepts horizontal or median"
            )
        if unit not in accepted_parameters.units:
            raise AttributeError(
                f"unit accepts : {accepted_parameters.units[0]} or {accepted_parameters.units[1]}"
            )
        if x_axis not in accepted_parameters.frequency_x_axes:
            raise AttributeError(
                "x_axis accepts "
                f"{accepted_parameters.frequency_x_axes[0]} or "
                f"{accepted_parameters.frequency_x_axes[1]}"
            )
        if ear not in accepted_parameters.ears:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ears[0]}, {accepted_parameters.ears[1]} or {accepted_parameters.ears[2]}"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = AxisOptions(
            azimuth_axis=AzimuthAxisOptions(range_mode="-180-180")
        ).merge(plot_options.axis)
        heatmap_options = HeatmapOptions(cmap="magma").merge(plot_options.heatmap)
        heatmap_frequency_axis_options = (
            FrequencyAxisOptions()
            if axis_options.frequency_axis is None
            else axis_options.frequency_axis
        ).merge(FrequencyAxisOptions(margin_ratio=Heatmap.axis_margin_ratio))

        if self.TF.values is None or self.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        plane_key = str(plane).strip().lower()
        layout_number = 22 if ear == "both" else 1
        layout = create_layout(
            layout=layout_number,
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        indices, _ = self.Planes.get_plane_indices(
            plane=plane_key,
            angle=0.0,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError("Selected plane does not contain any source positions")

        spherical_positions = SourcePositionData.create_positions(
            sources=self.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]

        if plane_key == "horizontal":
            plane_axis_values = np.asarray(spherical_positions[:, 0], dtype=float)
        else:
            lateral_polar_positions = self.Sources.spherical_to_lateral_polar(
                spherical_positions,
                angle_unit="degrees",
            )
            plane_axis_values = np.asarray(lateral_polar_positions[:, 1], dtype=float)

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        resolved_frequency_axis = Axis.create_frequency_axis(
            ax=None,
            axis="x",
            x_axis=x_axis,
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            options=heatmap_frequency_axis_options,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
            & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

        tf_magnitude = self.TF.magnitude
        plane_values = np.asarray(tf_magnitude[indices][..., frequency_mask], dtype=float)
        if plane_values.ndim == 2:
            plane_values = plane_values[:, np.newaxis, :]
        if plane_values.ndim != 3:
            raise ValueError("TF values for spectrum must have shape (M, E, F)")
        if unit == "db":
            if isinstance(reference, str) and str(reference).strip().lower() == "max":
                reference_values = plane_values
                if ear != "both":
                    ear_index = 0 if ear == "left" else 1
                    if reference_values.shape[1] <= ear_index:
                        raise ValueError(
                            f"Requested ear '{ear}' is not available in TF data"
                        )
                    reference_values = reference_values[:, ear_index, :]
                plot_reference = float(np.max(reference_values))
                plane_values = magnitude_to_db(plane_values, reference=plot_reference)
            else:
                plane_values = magnitude_to_db(plane_values, reference=reference)
        if ear == "both":
            if plane_values.shape[1] < 2:
                raise ValueError(
                    "Both ears requested but TF data does not contain two ear channels"
                )
            spectrum_matrices = [plane_values[:, 0, :], plane_values[:, 1, :]]
            panel_positions = ["left", "right"]
            default_panel_titles = ["Left Ear", "Right Ear"]
        else:
            if plane_values.shape[1] == 1:
                ear_index = 0
            else:
                ear_index = 0 if ear == "left" else 1
                if plane_values.shape[1] <= ear_index:
                    raise ValueError(
                        f"Requested ear '{ear}' is not available in TF data"
                    )
            spectrum_matrices = [plane_values[:, ear_index, :]]
            panel_positions = ["main"]
            default_panel_titles = [f"{ear.capitalize()} Ear"]

        vmin = min(float(np.min(matrix)) for matrix in spectrum_matrices)
        vmax = max(float(np.max(matrix)) for matrix in spectrum_matrices)
        labels = Labels()
        colorbar_label = labels.magnitude_db if unit == "db" else labels.magnitude_linear
        heatmap_colormap = Heatmap.create_colormap(options=heatmap_options)

        for panel_index, (panel_position, spectrum_matrix, default_panel_title) in enumerate(
            zip(panel_positions, spectrum_matrices, default_panel_titles)
        ):
            ax = layout.get_axis(panel_position)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(panel_index))
            panel_plane_axis_values = (
                Axis.transform_azimuth_values(
                    values=plane_axis_values,
                    options=resolved_axis_options,
                )
                if plane_key == "horizontal"
                else np.asarray(plane_axis_values, dtype=float)
            )
            panel_sort_indices = np.argsort(panel_plane_axis_values)
            sorted_panel_plane_axis_values = panel_plane_axis_values[panel_sort_indices]
            sorted_spectrum_matrix = spectrum_matrix[panel_sort_indices, :]
            mesh = ax.pcolormesh(
                frequency_khz,
                sorted_panel_plane_axis_values,
                sorted_spectrum_matrix,
                shading="auto",
                cmap=heatmap_colormap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.margins(x=0.0, y=0.0)
            frequency_label = (
                labels.frequency
                if resolved_axis_options.xlabel is None
                else resolved_axis_options.xlabel
            )
            Axis.create_frequency_axis(
                ax=ax,
                axis="x",
                x_axis=x_axis,
                label=frequency_label,
                options=resolved_frequency_axis,
            )
            Axis.create_plane_angle_axis(
                ax=ax,
                axis="y",
                plane=plane_key,
                values=sorted_panel_plane_axis_values,
                options=resolved_axis_options,
            )
            resolved_title = (
                default_panel_title
                if resolved_axis_options.title is None
                else resolved_axis_options.title
            )
            ax.set_title(resolved_title)
            grid_enabled = (
                False if resolved_axis_options.grid is None else resolved_axis_options.grid
            )
            if grid_enabled:
                ax.grid(True)
            Heatmap.create_colorbar(
                fig=layout.fig,
                ax=ax,
                mesh=mesh,
                label=colorbar_label,
                options=heatmap_options,
            )
        resolved_figure_title = (
            Axis.create_plane_title(
                plane=plane_key,
            )
            if figure_options.title is None
            else figure_options.title
        )
        layout.set_figure_title(resolved_figure_title)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_elevation_spectrum(
        self: "HRTF",
        azimuth: float | str = 0.0,
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = "max",
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """Plot a fixed-azimuth elevation spectrum heatmap.

        Parameters
        ----------
        azimuth : float | str, default=0.0
            Azimuth used to select the elevation slice. Named aliases such as
            ``"front"``, ``"back"``, ``"left"``, and ``"right"`` are accepted.
            The nearest available azimuth in the source grid is used.
        x_axis : {"linear", "log"}, default="linear"
            Frequency scale used on the x axis.
        unit : {"db", "linear"}, default="db"
            Magnitude representation used for the heatmap values.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display. When ``"both"`` is selected, a separate panel
            is created for each ear.
        reference : float | {"max"}, default="max"
            Reference used when ``unit="db"``. ``"max"`` normalizes the plotted
            slice to its maximum value.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, heatmap, and panel overrides.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect how magnitude changes with elevation at a fixed azimuth.
        - Compare left and right ear spectral structure along one azimuth slice.
        - Create elevation-spectrum heatmaps without showing them immediately.

        Examples
        --------
        Plot the front elevation spectrum with default settings:

        >>> hrtf.plot_elevation_spectrum()

        Plot the left-side elevation spectrum for one ear:

        >>> hrtf.plot_elevation_spectrum(azimuth="left", ear="left")

        Plot a numeric azimuth with logarithmic frequency scaling:

        >>> hrtf.plot_elevation_spectrum(azimuth=30.0, x_axis="log")

        Create the figure without showing it immediately:

        >>> hrtf.plot_elevation_spectrum(show=False)
        """
        accepted_parameters = AcceptedParameters()
        if unit not in accepted_parameters.units:
            raise AttributeError(
                f"unit accepts : {accepted_parameters.units[0]} or {accepted_parameters.units[1]}"
            )
        if x_axis not in accepted_parameters.frequency_x_axes:
            raise AttributeError(
                "x_axis accepts "
                f"{accepted_parameters.frequency_x_axes[0]} or "
                f"{accepted_parameters.frequency_x_axes[1]}"
            )
        if ear not in accepted_parameters.ears:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ears[0]}, {accepted_parameters.ears[1]} or {accepted_parameters.ears[2]}"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )
        heatmap_options = HeatmapOptions(cmap="magma").merge(plot_options.heatmap)
        heatmap_frequency_axis_options = (
            FrequencyAxisOptions()
            if axis_options.frequency_axis is None
            else axis_options.frequency_axis
        ).merge(FrequencyAxisOptions(margin_ratio=Heatmap.axis_margin_ratio))

        if self.TF.values is None or self.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        if isinstance(azimuth, str):
            azimuth_key = str(azimuth).strip().lower()
            named_positions = Sources.get_named_positions(angle_unit="degrees")
            if azimuth_key not in named_positions:
                raise ValueError("azimuth accepts a finite value or: front, back, left, right")
            resolved_azimuth = float(named_positions[azimuth_key][0])
        else:
            if isinstance(azimuth, bool):
                raise ValueError("azimuth must be a finite value")
            resolved_azimuth = float(azimuth)
            if not np.isfinite(resolved_azimuth):
                raise ValueError("azimuth must be a finite value")

        layout_number = 22 if ear == "both" else 1
        layout = create_layout(
            layout=layout_number,
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        spherical_positions = SourcePositionData.create_positions(
            sources=self.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )

        azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
        elevation_values = np.asarray(spherical_positions[:, 1], dtype=float)
        available_azimuths = np.unique(azimuth_values)
        azimuth_deltas = np.mod(available_azimuths - resolved_azimuth + 180.0, 360.0) - 180.0
        real_azimuth = float(available_azimuths[int(np.argmin(np.abs(azimuth_deltas)))])
        selected = np.isclose(
            np.mod(azimuth_values - real_azimuth + 180.0, 360.0) - 180.0,
            0.0,
            atol=1e-8,
            rtol=0.0,
        )
        indices = np.where(selected)[0]
        if indices.size == 0:
            raise ValueError("Selected elevation spectrum does not contain any source positions")

        slice_elevation_values = elevation_values[indices]
        sort_indices = np.argsort(slice_elevation_values)
        sorted_elevation_values = slice_elevation_values[sort_indices]

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        resolved_frequency_axis = Axis.create_frequency_axis(
            ax=None,
            axis="x",
            x_axis=x_axis,
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            options=heatmap_frequency_axis_options,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
            & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

        tf_magnitude = self.TF.magnitude
        slice_values = np.asarray(tf_magnitude[indices][..., frequency_mask], dtype=float)
        if slice_values.ndim == 2:
            slice_values = slice_values[:, np.newaxis, :]
        if slice_values.ndim != 3:
            raise ValueError("TF values for vertical slice spectrum must have shape (M, E, F)")
        if unit == "db":
            if isinstance(reference, str) and str(reference).strip().lower() == "max":
                reference_values = slice_values
                if ear != "both":
                    ear_index = 0 if ear == "left" else 1
                    if reference_values.shape[1] <= ear_index:
                        raise ValueError(
                            f"Requested ear '{ear}' is not available in TF data"
                        )
                    reference_values = reference_values[:, ear_index, :]
                plot_reference = float(np.max(reference_values))
                slice_values = magnitude_to_db(slice_values, reference=plot_reference)
            else:
                slice_values = magnitude_to_db(slice_values, reference=reference)
        slice_values = slice_values[sort_indices]

        if ear == "both":
            if slice_values.shape[1] < 2:
                raise ValueError(
                    "Both ears requested but TF data does not contain two ear channels"
                )
            spectrum_matrices = [slice_values[:, 0, :], slice_values[:, 1, :]]
            panel_positions = ["left", "right"]
            default_panel_titles = ["Left Ear", "Right Ear"]
        else:
            if slice_values.shape[1] == 1:
                ear_index = 0
            else:
                ear_index = 0 if ear == "left" else 1
                if slice_values.shape[1] <= ear_index:
                    raise ValueError(
                        f"Requested ear '{ear}' is not available in TF data"
                    )
            spectrum_matrices = [slice_values[:, ear_index, :]]
            panel_positions = ["main"]
            default_panel_titles = [f"{ear.capitalize()} Ear"]

        vmin = min(float(np.min(matrix)) for matrix in spectrum_matrices)
        vmax = max(float(np.max(matrix)) for matrix in spectrum_matrices)
        labels = Labels()
        colorbar_label = labels.magnitude_db if unit == "db" else labels.magnitude_linear
        heatmap_colormap = Heatmap.create_colormap(options=heatmap_options)

        for panel_index, (panel_position, spectrum_matrix, default_panel_title) in enumerate(
            zip(panel_positions, spectrum_matrices, default_panel_titles)
        ):
            ax = layout.get_axis(panel_position)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(panel_index))
            mesh = ax.pcolormesh(
                frequency_khz,
                sorted_elevation_values,
                spectrum_matrix,
                shading="auto",
                cmap=heatmap_colormap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.margins(x=0.0, y=0.0)
            frequency_label = (
                labels.frequency
                if resolved_axis_options.xlabel is None
                else resolved_axis_options.xlabel
            )
            Axis.create_frequency_axis(
                ax=ax,
                axis="x",
                x_axis=x_axis,
                label=frequency_label,
                options=resolved_frequency_axis,
            )
            Axis.create_elevation_axis(
                ax=ax,
                axis="y",
                values=sorted_elevation_values,
                options=resolved_axis_options,
            )
            resolved_title = (
                default_panel_title
                if resolved_axis_options.title is None
                else resolved_axis_options.title
            )
            ax.set_title(resolved_title)
            grid_enabled = (
                False if resolved_axis_options.grid is None else resolved_axis_options.grid
            )
            if grid_enabled:
                ax.grid(True)
            Heatmap.create_colorbar(
                fig=layout.fig,
                ax=ax,
                mesh=mesh,
                label=colorbar_label,
                options=heatmap_options,
            )
        resolved_figure_title = (
            Axis.create_elevation_spectrum_title(real_azimuth=real_azimuth)
            if figure_options.title is None
            else figure_options.title
        )
        layout.set_figure_title(resolved_figure_title)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_itd_horizontal_plane(
        self: "HRTF",
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot absolute ITD over the canonical horizontal plane in polar coordinates.

        The horizontal plane at ``0`` degrees elevation is selected from the
        current source grid, absolute interaural time differences are computed
        in seconds, and the result is displayed in a polar plot. Azimuth is
        represented on the angular axis and absolute ITD is represented on the
        radial axis.

        Parameters
        ----------
        options : PlotOptions or None, optional
            Plot configuration used to control the figure settings, subplot
            title, margins, and grid behavior. If ``None``, default plotting
            options are used.
        show : bool, optional
            If ``True``, call ``plt.show()`` before finishing the method. If
            ``False``, the figure is created without showing it.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the azimuth-dependent ITD pattern in the horizontal plane.
        - Visualize binaural timing cues using a compact polar representation.
        - Create an ITD figure without showing it immediately.

        Examples
        --------
        >>> hrtf.plot_itd_horizontal_plane()
        >>> hrtf.plot_itd_horizontal_plane(show=False)
        >>> hrtf.plot_itd_horizontal_plane(
        ...     options=PlotOptions(
        ...         figure=FigureOptions(title="Horizontal Plane ITD")
        ...     )
        ... )
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")

        itd_values = np.abs(
            np.asarray(
                calculate_itd(
                    self.IR,
                    output="seconds",
                ),
                dtype=float,
            )
        )
        theta_values, radial_values, sorted_itd_values = Polar.create_horizontal_plane_curve(
            sources=self.Sources,
            planes=self.Planes,
            values=itd_values,
        )

        layout = Projection.create_layout(
            layout=1,
            projection="polar",
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        ax = layout.get_axis("main")

        ax.plot(
            theta_values,
            radial_values,
            color="steelblue",
            linewidth=2.0,
        )
        ax.set_theta_zero_location("N")
        theta_ticks = np.arange(0.0, 360.0, Polar.theta_tick_step, dtype=float)
        ax.set_xticks(np.deg2rad(theta_ticks))
        ax.set_xticklabels([f"{int(tick)}°" for tick in theta_ticks])
        radial_max = float(np.max(sorted_itd_values)) if sorted_itd_values.size > 0 else 0.0
        radial_tick_step = 2e-4
        if np.isclose(radial_max, 0.0):
            resolved_radial_max = radial_tick_step
        else:
            resolved_radial_max = (
                np.ceil((radial_max * 1.1) / radial_tick_step) * radial_tick_step
            )
        radial_ticks = np.arange(
            radial_tick_step,
            resolved_radial_max + (0.5 * radial_tick_step),
            radial_tick_step,
            dtype=float,
        )
        ax.set_ylim(0.0, resolved_radial_max)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels(
            [f"{tick:0.4f}".replace(".", ",") for tick in radial_ticks]
        )
        ax.set_rlabel_position(350.0)
        resolved_title = (
            Labels().itd_seconds if figure_options.title is None else figure_options.title
        )
        if axis_options.title is not None:
            resolved_title = axis_options.title
        ax.set_title(resolved_title)
        grid_enabled = True if axis_options.grid is None else axis_options.grid
        ax.grid(grid_enabled)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_ild_horizontal_plane(
        self: "HRTF",
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot absolute ILD over the canonical horizontal plane in polar coordinates.

        The horizontal plane at ``0`` degrees elevation is selected from the
        current source grid, absolute interaural level differences are computed
        in decibels, and the result is displayed in a polar plot. Azimuth is
        represented on the angular axis and absolute ILD is represented on the
        radial axis.

        Parameters
        ----------
        options : PlotOptions or None, optional
            Plot configuration used to control the figure settings, subplot
            title, margins, and grid behavior. If ``None``, default plotting
            options are used.
        show : bool, optional
            If ``True``, call ``plt.show()`` before finishing the method. If
            ``False``, the figure is created without showing it.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the azimuth-dependent ILD pattern in the horizontal plane.
        - Visualize binaural level cues using a compact polar representation.
        - Create an ILD figure without showing it immediately.

        Examples
        --------
        >>> hrtf.plot_ild_horizontal_plane()
        >>> hrtf.plot_ild_horizontal_plane(show=False)
        >>> hrtf.plot_ild_horizontal_plane(
        ...     options=PlotOptions(
        ...         figure=FigureOptions(title="Horizontal Plane ILD")
        ...     )
        ... )
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")

        ild_values = np.abs(
            np.asarray(
                calculate_ild(
                    self.IR,
                    domain="ir",
                    output="db",
                    mode="broad-band",
                ),
                dtype=float,
            )
        )
        theta_values, radial_values, sorted_ild_values = Polar.create_horizontal_plane_curve(
            sources=self.Sources,
            planes=self.Planes,
            values=ild_values,
        )

        layout = Projection.create_layout(
            layout=1,
            projection="polar",
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        ax = layout.get_axis("main")

        ax.plot(
            theta_values,
            radial_values,
            color="steelblue",
            linewidth=2.0,
        )
        ax.set_theta_zero_location("N")
        theta_ticks = np.arange(0.0, 360.0, Polar.theta_tick_step, dtype=float)
        ax.set_xticks(np.deg2rad(theta_ticks))
        ax.set_xticklabels([f"{int(tick)}°" for tick in theta_ticks])
        radial_max = float(np.max(sorted_ild_values)) if sorted_ild_values.size > 0 else 0.0
        radial_tick_step = 5.0
        if np.isclose(radial_max, 0.0):
            resolved_radial_max = radial_tick_step
        else:
            resolved_radial_max = (
                np.ceil((radial_max * 1.1) / radial_tick_step) * radial_tick_step
            )
        radial_ticks = np.arange(
            radial_tick_step,
            resolved_radial_max + (0.5 * radial_tick_step),
            radial_tick_step,
            dtype=float,
        )
        ax.set_ylim(0.0, resolved_radial_max)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels(
            [f"{int(np.rint(tick))}" for tick in radial_ticks]
        )
        ax.set_rlabel_position(350.0)
        resolved_title = (
            Labels().ild_db if figure_options.title is None else figure_options.title
        )
        if axis_options.title is not None:
            resolved_title = axis_options.title
        ax.set_title(resolved_title)
        grid_enabled = True if axis_options.grid is None else axis_options.grid
        ax.grid(grid_enabled)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_source_grid(
        self: "HRTF",
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot the source grid as an interactive three-dimensional scatter figure.

        The method reads the current source positions from the HRTF instance,
        converts them to Cartesian coordinates when necessary, and renders the
        grid in a 3D Matplotlib axis. Direction arrows for front, right, and up
        are added to make the spatial orientation easier to interpret in the
        default camera view.

        Parameters
        ----------
        options : PlotOptions or None, optional
            Plot configuration used to control the figure settings, axis grid,
            margins, and title behavior. If ``None``, default plotting options
            are used.
        show : bool, optional
            If ``True``, call ``plt.show()`` before finishing the method. If
            ``False``, the figure is created without showing it.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the spatial sampling pattern of a source grid.
        - Check how dense or sparse a dataset is across directions.
        - Visualize the currently selected subset of sources after spatial
          selection or transformation.

        Examples
        --------
        >>> hrtf.plot_source_grid()
        >>> hrtf.plot_source_grid(show=False)
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        cartesian_positions = SourcePositionData.create_positions(
            sources=self.Sources,
            coordinate_system="cartesian",
            angle_unit="degrees",
        )
        layout, ax = ThreeDimensional.create_layout(
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )

        x_values = np.asarray(cartesian_positions[:, 0], dtype=float)
        y_values = np.asarray(cartesian_positions[:, 1], dtype=float)
        z_values = np.asarray(cartesian_positions[:, 2], dtype=float)
        ax.scatter(
            x_values,
            y_values,
            z_values,
            s=28.0,
            color="steelblue",
            edgecolors="black",
            linewidths=0.4,
            depthshade=True,
        )
        axis_half_span = ThreeDimensional.configure_axis(
            ax=ax,
            cartesian_positions=cartesian_positions,
        )
        ThreeDimensional.create_direction_markers(
            ax=ax,
            sources=self.Sources,
            axis_half_span=axis_half_span,
        )

        grid_enabled = True if axis_options.grid is None else axis_options.grid
        ax.grid(grid_enabled)

        resolved_figure_title = (
            "Source Grid" if figure_options.title is None else figure_options.title
        )
        layout.set_figure_title(resolved_figure_title)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_plane_grid(
        self: "HRTF",
        plane: str | list[str] | tuple[str, ...] = "horizontal",
        options: PlotOptions | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot the source grid and highlight canonical spatial planes in 3D.

        The full source grid is displayed as a light background scatter, while
        the selected canonical plane or planes are overlaid with stronger
        colors. The supported planes are the horizontal plane, the median plane,
        and the frontal plane, using the canonical definitions already provided
        by the spatial plane-selection logic in the library.

        Parameters
        ----------
        plane : str or list[str] or tuple[str, ...], optional
            Plane or planes to highlight. Accepted values are ``"horizontal"``,
            ``"median"``, and ``"frontal"``. A single string highlights one
            plane, while a list or tuple highlights multiple planes in the same
            figure.
        options : PlotOptions or None, optional
            Plot configuration used to control the figure settings, axis grid,
            legend behavior, margins, and title behavior. If ``None``, default
            plotting options are used.
        show : bool, optional
            If ``True``, call ``plt.show()`` before finishing the method. If
            ``False``, the figure is created without showing it.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the geometry of the canonical horizontal, median, and frontal
          planes in a dataset.
        - Verify whether a dataset contains the expected plane coverage.
        - Compare several canonical planes in one spatial grid view.

        Examples
        --------
        >>> hrtf.plot_plane_grid()
        >>> hrtf.plot_plane_grid(plane="median")
        >>> hrtf.plot_plane_grid(plane=["horizontal", "median", "frontal"], show=False)
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )
        legend_options = (
            LegendOptions() if axis_options.legend is None else axis_options.legend
        )

        raw_planes = [plane] if isinstance(plane, str) else list(plane)
        if len(raw_planes) == 0:
            raise ValueError("plane must contain at least one value")

        resolved_planes: list[str] = []
        for raw_plane in raw_planes:
            plane_key = str(raw_plane).strip().lower()
            if plane_key not in {"horizontal", "median", "frontal"}:
                raise ValueError("plane accepts: horizontal, median, frontal")
            if plane_key not in resolved_planes:
                resolved_planes.append(plane_key)

        cartesian_positions = SourcePositionData.create_positions(
            sources=self.Sources,
            coordinate_system="cartesian",
            angle_unit="degrees",
        )
        layout, ax = ThreeDimensional.create_layout(
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )

        x_values = np.asarray(cartesian_positions[:, 0], dtype=float)
        y_values = np.asarray(cartesian_positions[:, 1], dtype=float)
        z_values = np.asarray(cartesian_positions[:, 2], dtype=float)

        ax.scatter(
            x_values,
            y_values,
            z_values,
            s=18.0,
            color="#9ecae1",
            edgecolors="none",
            depthshade=True,
            alpha=0.55,
            label="Source Grid",
        )

        plane_colors = {
            "horizontal": "blue",
            "median": "red",
            "frontal": "green",
        }
        plane_titles = {
            "horizontal": "Horizontal Plane Grid",
            "median": "Median Plane Grid",
            "frontal": "Frontal Plane Grid",
        }
        plane_labels = {
            "horizontal": "Horizontal Plane",
            "median": "Median Plane",
            "frontal": "Frontal Plane",
        }

        for plane_key in resolved_planes:
            if plane_key == "horizontal":
                indices, _ = self.Planes.get_horizontal_plane_indices(
                    elevation=0.0,
                    angle_unit="degrees",
                )
            elif plane_key == "median":
                indices, _ = self.Planes.get_median_plane_indices(
                    azimuth=0.0,
                    angle_unit="degrees",
                )
            else:
                indices, _ = self.Planes.get_frontal_plane_indices(
                    azimuth=90.0,
                    angle_unit="degrees",
                )
            plane_positions = np.asarray(cartesian_positions[indices], dtype=float)
            ax.scatter(
                plane_positions[:, 0],
                plane_positions[:, 1],
                plane_positions[:, 2],
                s=34.0,
                color=plane_colors[plane_key],
                edgecolors="black",
                linewidths=0.35,
                depthshade=True,
                label=plane_labels[plane_key],
            )

        axis_half_span = ThreeDimensional.configure_axis(
            ax=ax,
            cartesian_positions=cartesian_positions,
        )
        ThreeDimensional.create_direction_markers(
            ax=ax,
            sources=self.Sources,
            axis_half_span=axis_half_span,
        )

        grid_enabled = True if axis_options.grid is None else axis_options.grid
        ax.grid(grid_enabled)

        legend_enabled = True if legend_options.enabled is None else legend_options.enabled
        if legend_enabled:
            resolved_legend_location = (
                "upper right"
                if legend_options.location is None
                else legend_options.location
            )
            ax.legend(loc=resolved_legend_location)

        if figure_options.title is None:
            if len(resolved_planes) == 1:
                resolved_figure_title = plane_titles[resolved_planes[0]]
            else:
                resolved_figure_title = "Plane Grid"
        else:
            resolved_figure_title = figure_options.title
        layout.set_figure_title(resolved_figure_title)
        if show and plot_options.show:
            plt.show()
        return None
