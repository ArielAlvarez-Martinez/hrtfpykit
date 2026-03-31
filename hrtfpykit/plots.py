from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator


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
class AcceptedParameters:
    unit: tuple[str, str] = ("db", "linear")
    ear: tuple[str, str, str] = ("left", "right", "both")


@dataclass(frozen=True)
class FrequencyAxis:
    ticks: tuple[float, ...] = (250, 500, 1000, 2000, 4000, 8000, 16000, 20000)
    labels: tuple[str, ...] = ("0.25", "0.5", "1", "2", "4", "8", "16", "20")
    freq_min: float | None = None
    freq_max: float | None = None
    margin_ratio: float = 0.03

    @staticmethod
    def set_ticks(
        ax: plt.Axes,
        unit: str,
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> None:
        frequency_axis = FrequencyAxis()
        resolved_freq_min = (
            frequency_axis.freq_min if freq_min is None else float(freq_min)
        )
        resolved_freq_max = (
            frequency_axis.freq_max if freq_max is None else float(freq_max)
        )
        margin_ratio = float(frequency_axis.margin_ratio)
        if resolved_freq_min is None or resolved_freq_max is None:
            raise ValueError("freq_min and freq_max are required for frequency axis formatting")
        if not np.isfinite(resolved_freq_min) or not np.isfinite(resolved_freq_max):
            raise ValueError("freq_min and freq_max must be finite values")
        if resolved_freq_min >= resolved_freq_max:
            raise ValueError("freq_min must be smaller than freq_max")
        if unit == "db" and resolved_freq_min <= 0.0:
            raise ValueError("freq_min must be positive for logarithmic frequency axis")

        ticks = []
        labels = []
        for tick, label in zip(frequency_axis.ticks, frequency_axis.labels):
            if resolved_freq_min <= tick <= resolved_freq_max:
                ticks.append(tick)
                labels.append(label)

        ticks_khz = [tick / 1000.0 for tick in ticks]
        resolved_freq_min_khz = resolved_freq_min / 1000.0
        resolved_freq_max_khz = resolved_freq_max / 1000.0
        if unit == "db":
            ax.set_xscale("log")
            log_min = np.log10(resolved_freq_min_khz)
            log_max = np.log10(resolved_freq_max_khz)
            margin_log = (log_max - log_min) * margin_ratio
            x_min = 10 ** (log_min - margin_log)
            x_max = 10 ** (log_max + margin_log)
        else:
            ax.set_xscale("linear")
            margin_linear = (resolved_freq_max_khz - resolved_freq_min_khz) * margin_ratio
            x_min = resolved_freq_min_khz - margin_linear
            x_max = resolved_freq_max_khz + margin_linear
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(FixedLocator(ticks_khz))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.offsetText.set_visible(False)


@dataclass(frozen=True)
class Labels:
    frequency: str = "Frequency(kHz)"
    magnitude_db: str = "Magnitude (dB)"
    magnitude_linear: str = "Magnitude"


@dataclass(frozen=True)
class Titles:
    position: str = "Position : [az.: {az}, el.: {el}]"


@dataclass(frozen=True)
class Legends:
    location: str = "upper left"
    left: str = "Left Ear"
    right: str = "Right Ear"


@dataclass(frozen=True)
class TickLabels:
    shared_x_visible: bool = True


@dataclass(frozen=True)
class AxisStyle:
    xlabel: str
    ylabel: str
    title: str
    use_frequency_axis: bool = False
    freq_min: float | None = None
    freq_max: float | None = None
    shared_x_visible: bool = False
    use_grid: bool = True
    use_ear_legend: bool = True


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


class Layout2(Layout):
    layout = 2
    rows = 2
    cols = 1
    positions = ("top", "bottom")
    figsize = (
        FigSizeDefault().width,
        FigSizeDefault().height + 2,
    )
    sharex = True


class Layout4(Layout):
    layout = 4
    rows = 2
    cols = 2
    positions = ("top_left", "top_right", "bottom_left", "bottom_right")
    figsize = (
        FigSizeDefault().width + 2,
        FigSizeDefault().height + 1,
    )


class LayoutFactory:
    registry: dict[int, type[Layout]] = {
        1: Layout1,
        2: Layout2,
        4: Layout4,
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


def figure(
    figsize: tuple[float, float] = (FigSizeDefault().width, FigSizeDefault().height),
) -> tuple[plt.Figure, plt.Axes]:
    layout = LayoutFactory.create(layout=1, figsize=figsize)
    return layout.fig, layout.get_axis(0)


layouts = tuple(LayoutFactory.registry)


def create_layout(
    layout: int,
    figsize: tuple[float, float] | None = None,
    margins: Margins | None = None,
) -> LayoutFigure:
    return LayoutFactory.create(layout=layout, figsize=figsize, margins=margins)


def grid(ax: plt.Axes | np.ndarray | list[plt.Axes]) -> None:
    axes = np.asarray(ax, dtype=object).reshape(-1)
    for axis in axes:
        axis.grid(True)


class Ear:
    @staticmethod
    def ears_legend(ax: plt.Axes, ear: str) -> None:
        legends = Legends()
        if ear == "both":
            ax.legend(
                labels=[legends.left, legends.right],
                loc=legends.location,
            )
        elif ear == "left":
            ax.legend(labels=[legends.left], loc=legends.location)
        elif ear == "right":
            ax.legend(labels=[legends.right], loc=legends.location)


def apply_axis_style(
    ax: plt.Axes,
    style: AxisStyle,
    unit: str | None = None,
    ear: str | None = None,
) -> None:
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    ax.set_title(style.title)
    if style.use_frequency_axis:
        if unit is None:
            raise ValueError("unit is required when frequency axis formatting is enabled")
        FrequencyAxis.set_ticks(
            ax,
            unit,
            freq_min=style.freq_min,
            freq_max=style.freq_max,
        )
    if style.shared_x_visible:
        ax.tick_params(axis="x", which="both", labelbottom=True)
    if style.use_ear_legend:
        if ear is None:
            raise ValueError("ear is required when legend formatting is enabled")
        Ear.ears_legend(ax, ear)
    if style.use_grid:
        grid(ax)


def create_magnitude_axis_style(
    unit: str,
    selected_positions: np.ndarray,
    freq_min: float | None = None,
    freq_max: float | None = None,
    shared_x_visible: bool | None = None,
) -> AxisStyle:
    labels = Labels()
    titles = Titles()
    tick_labels = TickLabels()
    if shared_x_visible is None:
        shared_x_visible = tick_labels.shared_x_visible
    return AxisStyle(
        xlabel=labels.frequency,
        ylabel=(
            labels.magnitude_db if unit == "db" else labels.magnitude_linear
        ),
        title=titles.position.format(
            az=float(selected_positions[0]),
            el=float(selected_positions[1]),
        ),
        use_frequency_axis=True,
        freq_min=None if freq_min is None else float(freq_min),
        freq_max=None if freq_max is None else float(freq_max),
        shared_x_visible=shared_x_visible,
        use_grid=True,
        use_ear_legend=True,
    )


class Plots:
    # Lets work in Inheritance. All methods will accept a instance of HRTF , then HRTF will inherit from Plots

    def plot_magnitude(
        self: "HRTF",
        position: list | np.ndarray,
        unit: str = "db",
        ear: str = "both",
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> None:
        accepted_parameters = AcceptedParameters()
        if unit not in accepted_parameters.unit:
            raise AttributeError(
                f"unit accepts : {accepted_parameters.unit[0]} or {accepted_parameters.unit[1]}"
            )
        if ear not in accepted_parameters.ear:
            raise AttributeError(
                f"ear accepts {accepted_parameters.ear[0]}, {accepted_parameters.ear[1]} or {accepted_parameters.ear[2]}"
            )
        hrtf = self[ear]

        if hrtf.TF.values is None or hrtf.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        positions = np.asarray(position, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.ndim != 2 or positions.shape[-1] not in {2, 3}:
            raise ValueError("position must have shape (2,), (3,), (K, 2), or (K, 3)")

        position_count = int(positions.shape[0])
        if position_count == 0:
            raise ValueError("At least one position is required")
        if position_count > 4:
            raise ValueError("plot_magnitude accepts up to 4 positions")

        layout_number = 1 if position_count == 1 else 2 if position_count == 2 else 4
        layout = (
            create_layout(layout=1, figsize=(8, 6))
            if layout_number == 1
            else create_layout(layout=layout_number)
        )
        frequency_bins_hz = np.asarray(hrtf.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        if freq_min is None:
            if unit == "db":
                positive_frequency_bins = frequency_bins_hz[frequency_bins_hz > 0.0]
                if positive_frequency_bins.size == 0:
                    raise ValueError(
                        "TF frequency bins must include a positive value for logarithmic frequency axis"
                    )
                resolved_freq_min = float(np.min(positive_frequency_bins))
            else:
                resolved_freq_min = float(np.min(frequency_bins_hz))
        else:
            resolved_freq_min = float(freq_min)
        resolved_freq_max = (
            float(np.max(frequency_bins_hz)) if freq_max is None else float(freq_max)
        )
        frequency_mask = (
            (frequency_bins_hz >= resolved_freq_min)
            & (frequency_bins_hz <= resolved_freq_max)
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        x_khz = frequency_bins_hz[frequency_mask] / 1000.0
        tf_values = hrtf.TF.magnitude_db if unit == "db" else hrtf.TF.magnitude

        for index, selected_position_query in enumerate(positions):
            ax = layout.get_axis(index)
            idxs, selected_positions = hrtf.Sources.get_position_index(
                selected_position_query,
                coordinate_system="spherical",
            )
            y = tf_values[idxs][..., frequency_mask]

            if ear == "both":
                y_left = y[0, :]
                y_right = y[1, :]
                ax.plot(x_khz, y_left)
                ax.plot(x_khz, y_right, linestyle="dashed")
            else:
                ax.plot(x_khz, y)

            axis_style = create_magnitude_axis_style(
                unit=unit,
                selected_positions=selected_positions,
                freq_min=resolved_freq_min,
                freq_max=resolved_freq_max,
            )
            apply_axis_style(
                ax=ax,
                style=axis_style,
                unit=unit,
                ear=ear,
            )

        if position_count < layout.axes.size:
            for ax in layout.axes[position_count:]:
                ax.set_visible(False)

        plt.show()
