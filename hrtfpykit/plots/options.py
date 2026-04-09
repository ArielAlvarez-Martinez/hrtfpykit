from __future__ import annotations

from dataclasses import dataclass

from .default import Margins


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


@dataclass(frozen=True)
class PlotOptions:
    figure: FigureOptions | None = None
    axis: AxisOptions | None = None
    heatmap: HeatmapOptions | None = None
    panels: dict[int | str, AxisOptions] | None = None
    show: bool = True
