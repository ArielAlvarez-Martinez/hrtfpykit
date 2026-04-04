from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
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
    fig_title: float = 10


@dataclass(frozen=True)
class Labels:
    frequency: str = "Frequency(kHz)"
    magnitude_db: str = "Magnitude (dB)"
    magnitude_linear: str = "Magnitude"
    time: str = "Time (s)"
    samples: str = "Samples"
    impulse_response: str = "Amplitude"


@dataclass(frozen=True)
class Titles:
    spherical_alias: str = "{name} : [Azimuth= {az}°, Elevation= {el}°]"
    spherical_position: str = "Position : [Azimuth= {az}°, Elevation= {el}°]"
    cartesian_alias: str = "{name} : [x= {x}, y= {y}, z= {z}]"
    cartesian_position: str = "Position : [x= {x}, y= {y}, z= {z}]"
    lateral_polar_alias: str = "{name} : [Lateral= {lateral}°, Polar= {polar}°]"
    lateral_polar_position: str = "Position : [Lateral= {lateral}°, Polar= {polar}°]"


@dataclass(frozen=True)
class AcceptedParameters:
    units: tuple[str, str] = ("db", "linear")
    ears: tuple[str, str, str] = ("left", "right", "both")
    x_axes: tuple[str, str] = ("time", "samples")
    coordinate_systems: tuple[str, str, str] = (
        "spherical",
        "cartesian",
        "lateral-polar",
    )


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
class AxisOptions:
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    shared_x_visible: bool | None = None
    grid: bool | None = None
    legend: LegendOptions | None = None
    frequency_axis: FrequencyAxisOptions | None = None

    def merge(self, options: AxisOptions | None = None) -> AxisOptions:
        if options is None:
            return self
        base_legend = LegendOptions() if self.legend is None else self.legend
        base_frequency_axis = (
            FrequencyAxisOptions()
            if self.frequency_axis is None
            else self.frequency_axis
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
        )


@dataclass(frozen=True)
class PlotOptions:
    figure: FigureOptions | None = None
    axis: AxisOptions | None = None
    panels: dict[int | str, AxisOptions] | None = None
    show: bool = True


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


def configure_rc() -> None:
    rc = Rc()
    plt.rcParams.update(
        {
            "font.size": rc.default,
            "axes.titlesize": rc.fig_title,
            "axes.labelsize": rc.axis_labels,
            "xtick.labelsize": rc.ticks,
            "ytick.labelsize": rc.ticks,
            "legend.fontsize": rc.legend,
            "legend.title_fontsize": rc.legend_title,
            "figure.titlesize": rc.fig_title,
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
        return LayoutFigure(
            fig=fig,
            axes=np.asarray(axes, dtype=object).reshape(-1),
            layout=cls.layout,
            positions=cls.positions,
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


class Layout2Vertical(Layout):
    layout = 2
    rows = 2
    cols = 1
    positions = ("top", "bottom")
    figsize = (
        FigSizeDefault().width,
        FigSizeDefault().height,
    )
    sharex = True


class Layout2VerticalIndependent(Layout):
    layout = 22
    rows = 2
    cols = 1
    positions = ("top", "bottom")
    figsize = (
        FigSizeDefault().width,
        FigSizeDefault().height,
    )


class Layout4(Layout):
    layout = 4
    rows = 2
    cols = 2
    positions = ("top_left", "top_right", "bottom_left", "bottom_right")
    figsize = (
        FigSizeDefault().width + 2,
        FigSizeDefault().height + 1,
    )


class Layout2Horizontal(Layout):
    layout = 12
    rows = 1
    cols = 2
    positions = ("left", "right")
    figsize = (12, 6)


class LayoutFactory:
    registry: dict[int, type[Layout]] = {
        1: Layout1,
        2: Layout2Vertical,
        22: Layout2VerticalIndependent,
        4: Layout4,
        12: Layout2Horizontal,
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

def create_layout(
    layout: int,
    figsize: tuple[float, float] | None = None,
    margins: Margins | None = None,
) -> LayoutFigure:
    return LayoutFactory.create(layout=layout, figsize=figsize, margins=margins)


class Axis:
    shared_x_visible: bool = True
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
        0,
        5000,
        10000,
        15000,
        20000,
    )
    frequency_tick_labels_linear: tuple[str, ...] = (
        "0",
        "5",
        "10",
        "15",
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
        ax.set_title(resolved_title)
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
        unit: str,
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
        requested_freq_min = (
            frequency_axis_options.freq_min if freq_min is None else freq_min
        )
        if requested_freq_min is None:
            if resolved_frequency_bins is None:
                raise ValueError("freq_min is required when frequency_bins are not provided")
            if unit == "db":
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
        if unit == "db" and resolved_freq_min <= 0.0:
            raise ValueError("freq_min must be positive for logarithmic frequency axis")
        if resolved_margin_ratio < 0.0:
            raise ValueError("margin_ratio must be non-negative")

        default_ticks = Axis.frequency_ticks_log if unit == "db" else Axis.frequency_ticks_linear
        default_labels = (
            Axis.frequency_tick_labels_log
            if unit == "db"
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
        if unit == "db" and any(tick <= 0.0 for tick in resolved_ticks):
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

        if unit == "db":
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
            legend_location="upper left",
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


class Plots:
    #  Inheritance. All methods will accept a instance of HRTF , then HRTF will inherit from Plots
    @staticmethod
    def get_panel_axis_options(
        layout: LayoutFigure,
        plot_options: PlotOptions,
    ) -> dict[int, AxisOptions]:
        panel_axis_options: dict[int, AxisOptions] = {}
        if plot_options.panels is None:
            return panel_axis_options
        for panel, panel_options in plot_options.panels.items():
            if isinstance(panel, str):
                if panel not in layout.positions:
                    raise ValueError(
                        f"panel accepts: {', '.join(layout.positions)}"
                    )
                panel_index = layout.positions.index(panel)
            else:
                panel_index = int(panel)
                if panel_index < 0 or panel_index >= layout.axes.size:
                    raise ValueError(
                        f"panel index must be between 0 and {layout.axes.size - 1}"
                    )
            if panel_index in panel_axis_options:
                raise ValueError(
                    f"panel override for subplot {panel_index} is duplicated"
                )
            panel_axis_options[panel_index] = panel_options
        return panel_axis_options

    def plot_magnitude(
        self: "HRTF",
        positions: list | np.ndarray,
        position_coordinate_system: str = "spherical",
        unit: str = "db",
        ear: str = "both",
        reference: float = 1.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
    ) -> LayoutFigure:
        accepted_parameters = AcceptedParameters()
        if unit not in accepted_parameters.units:
            raise AttributeError(
                f"unit accepts : {accepted_parameters.units[0]} or {accepted_parameters.units[1]}"
            )
        if ear not in accepted_parameters.ears:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ears[0]}, {accepted_parameters.ears[1]} or {accepted_parameters.ears[2]}"
            )
        if position_coordinate_system not in accepted_parameters.coordinate_systems:
            raise AttributeError(
                "position_coordinate_system accepts "
                f"{accepted_parameters.coordinate_systems[0]}, "
                f"{accepted_parameters.coordinate_systems[1]} or "
                f"{accepted_parameters.coordinate_systems[2]}"
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

        layout_number = 1 if position_count == 1 else 2 if position_count == 2 else 4
        layout = create_layout(
            layout=layout_number,
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        panel_axis_options = self.get_panel_axis_options(layout, plot_options)

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        tf_values = (
            self.TF.get_magnitude_db(reference=reference)
            if unit == "db"
            else self.TF.magnitude
        )
        labels = Labels()

        for index, selected_position_query in enumerate(position_queries):
            ax = layout.get_axis(index)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(index))
            resolved_frequency_axis = Axis.create_frequency_axis(
                ax=None,
                axis="x",
                unit=unit,
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
            idxs, selected_positions = self.Sources.get_position_index(
                selected_position_query,
                coordinate_system=position_coordinate_system,
            )
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
                unit=unit,
                label=frequency_label,
                options=resolved_frequency_axis,
            )
            Axis.create_magnitude_axis(
                ax=ax,
                axis="y",
                unit=unit,
                selected_positions=selected_positions,
                position_coordinate_system=position_coordinate_system,
                ear=ear,
                options=resolved_axis_options,
            )

        if position_count < layout.axes.size:
            for ax in layout.axes[position_count:]:
                ax.set_visible(False)

        if figure_options.title is not None:
            layout.fig.suptitle(
                figure_options.title,
                y=min(resolved_margins.top + 0.05, 0.98),
            )
        if plot_options.show:
            plt.show()
        return layout

    def plot_amplitude(
        self: "HRTF",
        positions: list | np.ndarray,
        position_coordinate_system: str = "spherical",
        ear: str = "both",
        x_axis: str = "time",
        options: PlotOptions | None = None,
    ) -> LayoutFigure:
        accepted_parameters = AcceptedParameters()
        if ear not in accepted_parameters.ears:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ears[0]}, {accepted_parameters.ears[1]} or {accepted_parameters.ears[2]}"
            )
        if x_axis not in accepted_parameters.x_axes:
            raise AttributeError(
                f"x_axis accepts : {accepted_parameters.x_axes[0]} or {accepted_parameters.x_axes[1]}"
            )
        if position_coordinate_system not in accepted_parameters.coordinate_systems:
            raise AttributeError(
                "position_coordinate_system accepts "
                f"{accepted_parameters.coordinate_systems[0]}, "
                f"{accepted_parameters.coordinate_systems[1]} or "
                f"{accepted_parameters.coordinate_systems[2]}"
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

        layout_number = 1 if position_count == 1 else 2 if position_count == 2 else 4
        layout = create_layout(
            layout=layout_number,
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        panel_axis_options = self.get_panel_axis_options(layout, plot_options)

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
                coordinate_system=position_coordinate_system,
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
                position_coordinate_system=position_coordinate_system,
                ear=ear,
                options=resolved_axis_options,
            )

        if position_count < layout.axes.size:
            for ax in layout.axes[position_count:]:
                ax.set_visible(False)

        if figure_options.title is not None:
            layout.fig.suptitle(
                figure_options.title,
                y=min(resolved_margins.top + 0.05, 0.98),
            )
        if plot_options.show:
            plt.show()
        return layout

    def plot_amplitude_and_magnitude(
        self: "HRTF",
        positions: list | np.ndarray,
        position_coordinate_system: str = "spherical",
        ear: str = "both",
        x_axis: str = "time",
        unit: str = "db",
        reference: float = 1.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
    ) -> LayoutFigure:
        accepted_parameters = AcceptedParameters()
        if ear not in accepted_parameters.ears:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ears[0]}, {accepted_parameters.ears[1]} or {accepted_parameters.ears[2]}"
            )
        if x_axis not in accepted_parameters.x_axes:
            raise AttributeError(
                f"x_axis accepts : {accepted_parameters.x_axes[0]} or {accepted_parameters.x_axes[1]}"
            )
        if unit not in accepted_parameters.units:
            raise AttributeError(
                f"unit accepts : {accepted_parameters.units[0]} or {accepted_parameters.units[1]}"
            )
        if position_coordinate_system not in accepted_parameters.coordinate_systems:
            raise AttributeError(
                "position_coordinate_system accepts "
                f"{accepted_parameters.coordinate_systems[0]}, "
                f"{accepted_parameters.coordinate_systems[1]} or "
                f"{accepted_parameters.coordinate_systems[2]}"
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

        position_queries = self.Sources.get_position_queries(positions)
        if len(position_queries) != 1:
            raise ValueError(
                "plot_amplitude_and_magnitude accepts exactly one position"
            )
        selected_position_query = position_queries[0]

        layout = create_layout(
            layout=12,
            figsize=figure_options.figsize,
            margins=resolved_margins,
        )
        panel_axis_options = self.get_panel_axis_options(layout, plot_options)

        left_axis_options = axis_options.merge(panel_axis_options.get(0))
        right_axis_options = axis_options.merge(panel_axis_options.get(1))
        left_axis_panel_options = left_axis_options.merge(AxisOptions(title=""))
        right_axis_panel_options = right_axis_options.merge(AxisOptions(title=""))

        idxs, selected_positions = self.Sources.get_position_index(
            selected_position_query,
            coordinate_system=position_coordinate_system,
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

        ir_ax = layout.get_axis("left")
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
                options=left_axis_options,
            )
        else:
            Axis.create_samples_axis(
                ax=ir_ax,
                axis="x",
                options=left_axis_options,
            )
        Axis.create_amplitude_axis(
            ax=ir_ax,
            axis="y",
            selected_positions=selected_positions,
            position_coordinate_system=position_coordinate_system,
            ear=ear,
            options=left_axis_panel_options,
        )

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        resolved_frequency_axis = Axis.create_frequency_axis(
            ax=None,
            axis="x",
            unit=unit,
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            options=right_axis_options.frequency_axis,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
            & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0
        tf_values = (
            self.TF.get_magnitude_db(reference=reference)
            if unit == "db"
            else self.TF.magnitude
        )
        magnitude_y_values = np.asarray(tf_values[idxs][..., frequency_mask], dtype=float)

        magnitude_ax = layout.get_axis("right")
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
        frequency_label = (
            labels.frequency
            if right_axis_options.xlabel is None
            else right_axis_options.xlabel
        )
        Axis.create_frequency_axis(
            ax=magnitude_ax,
            axis="x",
            unit=unit,
            label=frequency_label,
            options=resolved_frequency_axis,
        )
        Axis.create_magnitude_axis(
            ax=magnitude_ax,
            axis="y",
            unit=unit,
            selected_positions=selected_positions,
            position_coordinate_system=position_coordinate_system,
            ear=ear,
            options=right_axis_panel_options,
        )

        resolved_figure_title = (
            Axis.create_position_title(
                selected_positions=selected_positions,
                position_coordinate_system=position_coordinate_system,
            )
            if figure_options.title is None
            else figure_options.title
        )
        layout.fig.suptitle(
            resolved_figure_title,
            y=min(resolved_margins.top + 0.05, 0.98),
        )
        if plot_options.show:
            plt.show()
        return layout
