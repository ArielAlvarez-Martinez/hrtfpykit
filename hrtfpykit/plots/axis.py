from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

from .axis_helpers import apply_frequency_axis, build_frequency_axis
from .labels import Labels
from .options import AxisOptions, AzimuthAxisOptions, FrequencyAxisOptions


class Axis(ABC):
    @staticmethod
    @abstractmethod
    def apply(*args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def apply_label(
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


class DirectionAxis(Axis, ABC):
    @classmethod
    def get_tick_step(cls) -> float:
        raise NotImplementedError

    @staticmethod
    def apply_direction(
        ax: plt.Axes,
        axis: str,
        default_label: str,
        values: np.ndarray | None = None,
        tick_step: float | None = None,
        default_limits: tuple[float, float] | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        resolved_tick_step = 20.0 if tick_step is None else float(tick_step)
        if resolved_tick_step <= 0.0:
            raise ValueError("tick_step must be positive")
        Axis.apply_label(
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
                raise ValueError(
                    "z-axis directional formatting requires a matplotlib 3D axis"
                )
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

class FrequencyLogAxis(Axis):
    frequency_ticks: tuple[float, ...] = (
        250,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        20000,
    )

    @staticmethod
    def build(
        frequency_bins: np.ndarray | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: FrequencyAxisOptions | None = None,
    ) -> FrequencyAxisOptions:
        return build_frequency_axis(
            scale="log",
            default_ticks=FrequencyLogAxis.frequency_ticks,
            default_labels=Labels.frequency_tick_labels_log,
            frequency_bins=frequency_bins,
            freq_min=freq_min,
            freq_max=freq_max,
            options=options,
        )

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
        options: FrequencyAxisOptions | None = None,
    ) -> None:
        apply_frequency_axis(
            scale="log",
            ax=ax,
            axis=axis,
            label=label,
            options=options,
        )


class FrequencyLinearAxis(Axis):
    frequency_ticks: tuple[float, ...] = (
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

    @staticmethod
    def build(
        frequency_bins: np.ndarray | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: FrequencyAxisOptions | None = None,
    ) -> FrequencyAxisOptions:
        return build_frequency_axis(
            scale="linear",
            default_ticks=FrequencyLinearAxis.frequency_ticks,
            default_labels=Labels.frequency_tick_labels_linear,
            frequency_bins=frequency_bins,
            freq_min=freq_min,
            freq_max=freq_max,
            options=options,
        )

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
        options: FrequencyAxisOptions | None = None,
    ) -> None:
        apply_frequency_axis(
            scale="linear",
            ax=ax,
            axis=axis,
            label=label,
            options=options,
        )


class MagnitudeAxis(Axis):
    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        unit: str,
        options: AxisOptions | None = None,
    ) -> None:
        default_label = (
            Labels.magnitude_db if unit == "db" else Labels.magnitude_linear
        )
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=default_label,
            options=options,
        )


class AmplitudeAxis(Axis):
    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        options: AxisOptions | None = None,
    ) -> None:
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=Labels.impulse_response,
            options=options,
        )


class TimeAxis(Axis):
    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        options: AxisOptions | None = None,
    ) -> None:
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=Labels.time,
            options=options,
        )


class SampleAxis(Axis):
    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        options: AxisOptions | None = None,
    ) -> None:
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=Labels.samples,
            options=options,
        )


class XAxis(Axis):
    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        Axis.apply_label(
            ax=ax,
            axis="x",
            default_label=Labels.three_d_x_label,
            options=options,
        )
        if center is None and half_span is None:
            return
        if center is None or half_span is None:
            raise ValueError("XAxis.apply requires both center and half_span")
        resolved_center = float(center)
        resolved_half_span = float(half_span)
        if not np.isfinite(resolved_center):
            raise ValueError("x-axis center must be finite")
        if not np.isfinite(resolved_half_span) or resolved_half_span <= 0.0:
            raise ValueError("x-axis half_span must be a finite, positive value")
        ax.set_xlim(
            resolved_center - resolved_half_span,
            resolved_center + resolved_half_span,
        )


class YAxis(Axis):
    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        Axis.apply_label(
            ax=ax,
            axis="y",
            default_label=Labels.three_d_y_label,
            options=options,
        )
        if center is None and half_span is None:
            return
        if center is None or half_span is None:
            raise ValueError("YAxis.apply requires both center and half_span")
        resolved_center = float(center)
        resolved_half_span = float(half_span)
        if not np.isfinite(resolved_center):
            raise ValueError("y-axis center must be finite")
        if not np.isfinite(resolved_half_span) or resolved_half_span <= 0.0:
            raise ValueError("y-axis half_span must be a finite, positive value")
        ax.set_ylim(
            resolved_center - resolved_half_span,
            resolved_center + resolved_half_span,
        )


class ZAxis(Axis):
    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        Axis.apply_label(
            ax=ax,
            axis="z",
            default_label=Labels.three_d_z_label,
            options=options,
        )
        if center is None and half_span is None:
            return
        if center is None or half_span is None:
            raise ValueError("ZAxis.apply requires both center and half_span")
        resolved_center = float(center)
        resolved_half_span = float(half_span)
        if not np.isfinite(resolved_center):
            raise ValueError("z-axis center must be finite")
        if not np.isfinite(resolved_half_span) or resolved_half_span <= 0.0:
            raise ValueError("z-axis half_span must be a finite, positive value")
        set_zlim = getattr(ax, "set_zlim", None)
        if set_zlim is None:
            raise ValueError("z-axis limits require a matplotlib 3D axis")
        set_zlim(
            resolved_center - resolved_half_span,
            resolved_center + resolved_half_span,
        )


class AzimuthAnglesAxis(DirectionAxis):
    direction_tick_step: float = 20.0
    azimuth_range_modes: tuple[str, str] = ("0-360", "-180-180")
    azimuth_limits_unsigned: tuple[float, float] = (0.0, 360.0)
    azimuth_limits_signed: tuple[float, float] = (-180.0, 180.0)

    @classmethod
    def get_tick_step(cls) -> float:
        return cls.direction_tick_step

    @staticmethod
    def get_range_mode(
        options: AxisOptions | None = None,
    ) -> str:
        axis_options = AxisOptions() if options is None else options
        azimuth_axis_options = (
            AzimuthAxisOptions()
            if axis_options.azimuth_axis is None
            else axis_options.azimuth_axis
        )
        resolved_range_mode = (
            AzimuthAnglesAxis.azimuth_range_modes[0]
            if azimuth_axis_options.range_mode is None
            else str(azimuth_axis_options.range_mode).strip()
        )
        if resolved_range_mode not in AzimuthAnglesAxis.azimuth_range_modes:
            raise ValueError(
                "azimuth axis range_mode accepts "
                f"{AzimuthAnglesAxis.azimuth_range_modes[0]} or {AzimuthAnglesAxis.azimuth_range_modes[1]}"
            )
        return resolved_range_mode

    @staticmethod
    def transform_values(
        values: np.ndarray,
        options: AxisOptions | None = None,
    ) -> np.ndarray:
        resolved_values = np.asarray(values, dtype=float)
        resolved_range_mode = AzimuthAnglesAxis.get_range_mode(options=options)
        if resolved_range_mode == AzimuthAnglesAxis.azimuth_range_modes[0]:
            return np.mod(resolved_values, 360.0)
        transformed_values = np.mod(resolved_values + 180.0, 360.0) - 180.0
        transformed_values[np.isclose(transformed_values, -180.0, atol=1e-8, rtol=0.0)] = 180.0
        return transformed_values

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        resolved_range_mode = AzimuthAnglesAxis.get_range_mode(options=options)
        transformed_values = (
            None
            if values is None
            else AzimuthAnglesAxis.transform_values(values=values, options=options)
        )
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.azimuth,
            values=transformed_values,
            tick_step=AzimuthAnglesAxis.direction_tick_step,
            default_limits=(
                AzimuthAnglesAxis.azimuth_limits_unsigned
                if resolved_range_mode == AzimuthAnglesAxis.azimuth_range_modes[0]
                else AzimuthAnglesAxis.azimuth_limits_signed
            ),
            options=options,
        )


class ElevationAnglesAxis(DirectionAxis):
    elevation_tick_step: float = 10.0

    @classmethod
    def get_tick_step(cls) -> float:
        return cls.elevation_tick_step

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.elevation,
            values=values,
            tick_step=ElevationAnglesAxis.elevation_tick_step,
            options=options,
        )


class PolarAnglesAxis(DirectionAxis):
    direction_tick_step: float = 20.0
    polar_limits: tuple[float, float] = (-90.0, 270.0)

    @classmethod
    def get_tick_step(cls) -> float:
        return cls.direction_tick_step

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.polar,
            values=values,
            tick_step=PolarAnglesAxis.direction_tick_step,
            default_limits=PolarAnglesAxis.polar_limits,
            options=options,
        )
