from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

from .labels import Labels
from .legends import Ear
from .options import (
    AxisOptions,
    AzimuthAxisOptions,
    FrequencyAxisOptions,
    LegendOptions,
)
from .titles import Titles
from ..hrtf.coordinates import get_position_alias


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
            Ear.apply(
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
        position_alias = get_position_alias(
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
        if x_axis_key not in {"log", "linear"}:
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
        elevation_angle: float = 0.0,
    ) -> str:
        plane_key = str(plane).strip().lower()
        titles = Titles()
        if plane_key == "horizontal":
            if np.isclose(float(elevation_angle), 0.0, atol=1e-8, rtol=0.0):
                return titles.horizontal_plane
            return titles.horizontal_plane_elevation.format(angle=float(elevation_angle))
        if plane_key == "median":
            return titles.median_plane
        raise ValueError("plane accepts horizontal or median")

    @staticmethod
    def create_elevation_spectrum_title(
        real_azimuth: float,
    ) -> str:
        titles = Titles()
        return titles.elevation_spectrum.format(angle=float(real_azimuth))
