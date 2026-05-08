from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

from .axis_helpers import apply_frequency_axis, build_frequency_axis
from .labels import Labels


class Axis(ABC):
    """Base contract for all plotting axis formatters."""

    @staticmethod
    @abstractmethod
    def apply(*args, **kwargs) -> None:
        """Apply axis formatting for the concrete axis formatter."""
        raise NotImplementedError

    @staticmethod
    def apply_label(
        ax: plt.Axes,
        axis: str,
        default_label: str,
        label: str | None = None,
    ) -> None:
        """Apply x, y, or z axis labels with a default fallback.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``"x"``, ``"y"``, or ``"z"``.
        default_label : str
            Default label used when ``label`` is not provided.
        label : str | None, default=None
            Optional explicit label.

        Returns
        -------
        None

        """
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis accepts 'x', 'y', or 'z'")
        resolved_label = default_label if label is None else str(label)
        if axis == "x":
            ax.set_xlabel(resolved_label)
            return
        if axis == "y":
            ax.set_ylabel(resolved_label)
            return
        set_zlabel = getattr(ax, "set_zlabel", None)
        if set_zlabel is None:
            raise ValueError("z-axis labeling requires a matplotlib 3D axis")
        set_zlabel(resolved_label)


class DirectionAxis(Axis, ABC):
    """Base formatter for directional angle axes."""

    @classmethod
    def get_tick_step(cls) -> float:
        """Return the directional tick-step value used by the axis class."""
        raise NotImplementedError

    @staticmethod
    def apply_direction(
        ax: plt.Axes,
        axis: str,
        default_label: str,
        values: np.ndarray | None = None,
        tick_step: float | None = None,
        default_limits: tuple[float, float] | None = None,
        label: str | None = None,
    ) -> None:
        """Apply directional limits and ticks for angle-like axes.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``"x"``, ``"y"``, or ``"z"``.
        default_label : str
            Default label text.
        values : np.ndarray | None, default=None
            Direction values used to resolve limits and interior ticks.
        tick_step : float | None, default=None
            Tick spacing in axis units.
        default_limits : tuple[float, float] | None, default=None
            Fallback limits used when ``values`` is not provided.
        label : str | None, default=None
            Optional explicit label.

        Returns
        -------
        None

        """
        resolved_tick_step = 20.0 if tick_step is None else float(tick_step)
        if resolved_tick_step <= 0.0:
            raise ValueError("tick_step must be positive")
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=default_label,
            label=label,
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
    """Log-frequency axis formatter with standard HRTF-oriented ticks."""

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
        ticks: tuple[float, ...] | list[float] | None = None,
        labels: tuple[str, ...] | list[str] | None = None,
        margin_ratio: float = 0.03,
    ) -> dict[str, float | tuple[float, ...] | tuple[str, ...]]:
        """Build validated configuration for logarithmic frequency axes.

        Parameters
        ----------
        frequency_bins : np.ndarray | None, default=None
            Frequency bins in Hz used to infer limits when not provided.
        freq_min : float | None, default=None
            Minimum frequency in Hz.
        freq_max : float | None, default=None
            Maximum frequency in Hz.
        ticks : tuple[float, ...] | list[float] | None, default=None
            Tick positions in Hz.
        labels : tuple[str, ...] | list[str] | None, default=None
            Tick labels matching ``ticks``.
        margin_ratio : float, default=0.03
            Relative axis margin.

        Returns
        -------
        dict[str, float | tuple[float, ...] | tuple[str, ...]]
            Frequency-axis configuration dictionary.

        """
        return build_frequency_axis(
            scale="log",
            default_ticks=FrequencyLogAxis.frequency_ticks,
            default_labels=Labels.frequency_tick_labels_log,
            frequency_bins=frequency_bins,
            freq_min=freq_min,
            freq_max=freq_max,
            ticks=ticks,
            labels=labels,
            margin_ratio=margin_ratio,
        )

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
        config: dict[str, float | tuple[float, ...] | tuple[str, ...]] | None = None,
    ) -> None:
        """Apply a logarithmic frequency-axis configuration.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``"x"``, ``"y"``, or ``"z"``.
        label : str | None, default=None
            Optional axis label.
        config : dict[str, float | tuple[float, ...] | tuple[str, ...]] | None
            Configuration produced by :meth:`build`.

        Returns
        -------
        None
        """
        if config is None:
            raise ValueError("config is required")
        apply_frequency_axis(
            scale="log",
            ax=ax,
            axis=axis,
            label=label,
            freq_min=float(config["freq_min"]),
            freq_max=float(config["freq_max"]),
            ticks=tuple(config["ticks"]),  # type: ignore[arg-type]
            labels=tuple(config["labels"]),  # type: ignore[arg-type]
            margin_ratio=float(config["margin_ratio"]),
        )


class FrequencyLinearAxis(Axis):
    """Linear-frequency axis formatter with standard HRTF-oriented ticks."""

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
        ticks: tuple[float, ...] | list[float] | None = None,
        labels: tuple[str, ...] | list[str] | None = None,
        margin_ratio: float = 0.03,
    ) -> dict[str, float | tuple[float, ...] | tuple[str, ...]]:
        """Build validated configuration for linear frequency axes.

        Parameters
        ----------
        frequency_bins : np.ndarray | None, default=None
            Frequency bins in Hz used to infer limits when not provided.
        freq_min : float | None, default=None
            Minimum frequency in Hz.
        freq_max : float | None, default=None
            Maximum frequency in Hz.
        ticks : tuple[float, ...] | list[float] | None, default=None
            Tick positions in Hz.
        labels : tuple[str, ...] | list[str] | None, default=None
            Tick labels matching ``ticks``.
        margin_ratio : float, default=0.03
            Relative axis margin.

        Returns
        -------
        dict[str, float | tuple[float, ...] | tuple[str, ...]]
            Frequency-axis configuration dictionary.
        """
        return build_frequency_axis(
            scale="linear",
            default_ticks=FrequencyLinearAxis.frequency_ticks,
            default_labels=Labels.frequency_tick_labels_linear,
            frequency_bins=frequency_bins,
            freq_min=freq_min,
            freq_max=freq_max,
            ticks=ticks,
            labels=labels,
            margin_ratio=margin_ratio,
        )

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
        config: dict[str, float | tuple[float, ...] | tuple[str, ...]] | None = None,
    ) -> None:
        """Apply a linear frequency-axis configuration.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``"x"``, ``"y"``, or ``"z"``.
        label : str | None, default=None
            Optional axis label.
        config : dict[str, float | tuple[float, ...] | tuple[str, ...]] | None
            Configuration produced by :meth:`build`.

        Returns
        -------
        None
        """
        if config is None:
            raise ValueError("config is required")
        apply_frequency_axis(
            scale="linear",
            ax=ax,
            axis=axis,
            label=label,
            freq_min=float(config["freq_min"]),
            freq_max=float(config["freq_max"]),
            ticks=tuple(config["ticks"]),  # type: ignore[arg-type]
            labels=tuple(config["labels"]),  # type: ignore[arg-type]
            margin_ratio=float(config["margin_ratio"]),
        )


class MagnitudeAxis(Axis):
    """Magnitude-axis label formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        unit: str,
        label: str | None = None,
    ) -> None:
        """Apply magnitude axis labeling for linear or decibel units."""
        default_label = Labels.magnitude_db if unit == "db" else Labels.magnitude_linear
        Axis.apply_label(ax=ax, axis=axis, default_label=default_label, label=label)


class AmplitudeAxis(Axis):
    """Amplitude-axis label formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
    ) -> None:
        """Apply impulse-response amplitude label on the selected axis."""
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=Labels.impulse_response,
            label=label,
        )


class TimeAxis(Axis):
    """Time-axis label formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
    ) -> None:
        """Apply time label on the selected axis."""
        Axis.apply_label(ax=ax, axis=axis, default_label=Labels.time, label=label)


class SampleAxis(Axis):
    """Sample-index axis label formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
    ) -> None:
        """Apply sample-index label on the selected axis."""
        Axis.apply_label(ax=ax, axis=axis, default_label=Labels.samples, label=label)


class XAxis(Axis):
    """3D x-axis label and symmetric-limit formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        label: str | None = None,
    ) -> None:
        """Apply x-axis label and optionally symmetric limits around a center.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        center : float | None, default=None
            Center value for x-axis limits.
        half_span : float | None, default=None
            Positive half-span used to create symmetric limits.
        label : str | None, default=None
            Optional explicit label.

        Returns
        -------
        None
        """
        Axis.apply_label(ax=ax, axis="x", default_label=Labels.three_d_x_label, label=label)
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
        ax.set_xlim(resolved_center - resolved_half_span, resolved_center + resolved_half_span)


class YAxis(Axis):
    """3D y-axis label and symmetric-limit formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        label: str | None = None,
    ) -> None:
        """Apply y-axis label and optionally symmetric limits around a center.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        center : float | None, default=None
            Center value for y-axis limits.
        half_span : float | None, default=None
            Positive half-span used to create symmetric limits.
        label : str | None, default=None
            Optional explicit label.

        Returns
        -------
        None
        """
        Axis.apply_label(ax=ax, axis="y", default_label=Labels.three_d_y_label, label=label)
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
        ax.set_ylim(resolved_center - resolved_half_span, resolved_center + resolved_half_span)


class ZAxis(Axis):
    """3D z-axis label and symmetric-limit formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        label: str | None = None,
    ) -> None:
        """Apply z-axis label and optionally symmetric limits around a center.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        center : float | None, default=None
            Center value for z-axis limits.
        half_span : float | None, default=None
            Positive half-span used to create symmetric limits.
        label : str | None, default=None
            Optional explicit label.

        Returns
        -------
        None
        """
        Axis.apply_label(ax=ax, axis="z", default_label=Labels.three_d_z_label, label=label)
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
        set_zlim(resolved_center - resolved_half_span, resolved_center + resolved_half_span)


class AzimuthAnglesAxis(DirectionAxis):
    """Azimuth-axis formatter with configurable signed or unsigned range."""

    direction_tick_step: float = 20.0
    azimuth_range_modes: tuple[str, str] = ("0-360", "-180-180")
    azimuth_limits_unsigned: tuple[float, float] = (0.0, 360.0)
    azimuth_limits_signed: tuple[float, float] = (-180.0, 180.0)

    @classmethod
    def get_tick_step(cls) -> float:
        """Return azimuth tick spacing in degrees."""
        return cls.direction_tick_step

    @staticmethod
    def get_range_mode(range_mode: str | None = None) -> str:
        """Resolve and validate azimuth range mode."""
        resolved_range_mode = (
            AzimuthAnglesAxis.azimuth_range_modes[0]
            if range_mode is None
            else str(range_mode).strip()
        )
        if resolved_range_mode not in AzimuthAnglesAxis.azimuth_range_modes:
            raise ValueError("azimuth range_mode accepts 0-360 or -180-180")
        return resolved_range_mode

    @staticmethod
    def transform_values(
        values: np.ndarray,
        range_mode: str | None = None,
    ) -> np.ndarray:
        """Transform azimuth values to the selected range convention.

        Parameters
        ----------
        values : np.ndarray
            Input azimuth values in degrees.
        range_mode : str | None, default=None
            ``"0-360"`` or ``"-180-180"``.

        Returns
        -------
        np.ndarray
            Transformed azimuth values in degrees.
        """
        resolved_values = np.asarray(values, dtype=float)
        resolved_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=range_mode)
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
        range_mode: str | None = None,
        label: str | None = None,
    ) -> None:
        """Apply azimuth-axis formatting using the selected range mode."""
        resolved_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=range_mode)
        transformed_values = (
            None
            if values is None
            else AzimuthAnglesAxis.transform_values(values=values, range_mode=resolved_range_mode)
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
            label=label,
        )


class AzimuthAnglesAxisPolarProjection(Axis):
    """Azimuth formatter for polar projection axes."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        tick_step: float = 30.0,
    ) -> None:
        """Apply polar-theta ticks with north-up orientation.

        Parameters
        ----------
        ax : plt.Axes
            Target polar axis.
        tick_step : float, default=30.0
            Angular tick spacing in degrees.

        Returns
        -------
        None
        """
        if getattr(ax, "name", "") != "polar":
            raise ValueError("AzimuthAnglesAxisPolarProjection requires a polar axis")
        resolved_tick_step = float(tick_step)
        if not np.isfinite(resolved_tick_step) or resolved_tick_step <= 0.0:
            raise ValueError("tick_step must be a finite, positive value")
        theta_ticks = np.arange(0.0, 360.0, resolved_tick_step, dtype=float)
        ax.set_theta_zero_location("N")
        ax.set_xticks(np.deg2rad(theta_ticks))
        ax.set_xticklabels([f"{int(np.rint(tick))}°" for tick in theta_ticks])


class RadialAxisPolarProjection(Axis):
    """Radial-axis formatter for polar projection axes."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        radial_values: np.ndarray,
        radial_label_default: str,
        tick_step: float = 1.0,
        tick_label_style: str = "integer",
        label_position: float = 350.0,
        label: str | None = None,
    ) -> None:
        """Apply radial limits, ticks, and labels for polar plots.

        Parameters
        ----------
        ax : plt.Axes
            Target polar axis.
        radial_values : np.ndarray
            Radial data values used to resolve axis limits.
        radial_label_default : str
            Default radial-axis label.
        tick_step : float, default=1.0
            Radial tick spacing.
        tick_label_style : str, default="integer"
            Tick-label formatting mode: ``"integer"`` or ``"decimal_comma_4"``.
        label_position : float, default=350.0
            Radial label angular position in degrees.
        label : str | None, default=None
            Optional explicit radial-axis label.

        Returns
        -------
        None

        """
        if getattr(ax, "name", "") != "polar":
            raise ValueError("RadialAxisPolarProjection requires a polar axis")
        resolved_radial_values = np.asarray(radial_values, dtype=float).reshape(-1)
        if resolved_radial_values.size == 0:
            radial_max = 0.0
        else:
            if not np.all(np.isfinite(resolved_radial_values)):
                raise ValueError("radial_values must contain finite values")
            radial_max = float(np.max(resolved_radial_values))

        resolved_tick_step = float(tick_step)
        if not np.isfinite(resolved_tick_step) or resolved_tick_step <= 0.0:
            raise ValueError("radial_tick_step must be a finite, positive value")
        resolved_tick_label_style = str(tick_label_style).strip()
        if resolved_tick_label_style not in {"integer", "decimal_comma_4"}:
            raise ValueError("radial_tick_label_style accepts integer or decimal_comma_4")

        if np.isclose(radial_max, 0.0):
            resolved_radial_max = resolved_tick_step
        else:
            resolved_radial_max = np.ceil((radial_max * 1.1) / resolved_tick_step) * resolved_tick_step
        radial_ticks = np.arange(
            resolved_tick_step,
            resolved_radial_max + (0.5 * resolved_tick_step),
            resolved_tick_step,
            dtype=float,
        )
        ax.set_ylim(0.0, resolved_radial_max)
        ax.set_yticks(radial_ticks)
        if resolved_tick_label_style == "decimal_comma_4":
            radial_tick_labels = [f"{tick:0.4f}".replace(".", ",") for tick in radial_ticks]
        else:
            radial_tick_labels = [f"{int(np.rint(tick))}" for tick in radial_ticks]
        ax.set_yticklabels(radial_tick_labels)

        resolved_label_position = float(label_position)
        if not np.isfinite(resolved_label_position):
            raise ValueError("rlabel_position must be finite")
        ax.set_rlabel_position(resolved_label_position)
        resolved_radial_label = radial_label_default if label is None else str(label)
        ax.set_ylabel(resolved_radial_label, rotation=0)
        ax.yaxis.set_label_coords(0.5, ax.title.get_position()[1], transform=ax.transAxes)
        ax.yaxis.label.set_horizontalalignment("center")
        ax.yaxis.label.set_verticalalignment("bottom")


class ElevationAnglesAxis(DirectionAxis):
    """Elevation-axis formatter."""

    elevation_tick_step: float = 10.0

    @classmethod
    def get_tick_step(cls) -> float:
        """Return elevation tick spacing in degrees."""
        return cls.elevation_tick_step

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        label: str | None = None,
    ) -> None:
        """Apply elevation-axis labeling, limits, and ticks."""
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.elevation,
            values=values,
            tick_step=ElevationAnglesAxis.elevation_tick_step,
            label=label,
        )


class PolarAnglesAxis(DirectionAxis):
    """Lateral-polar angle axis formatter."""

    direction_tick_step: float = 20.0
    polar_limits: tuple[float, float] = (-90.0, 270.0)

    @classmethod
    def get_tick_step(cls) -> float:
        """Return polar-angle tick spacing in degrees."""
        return cls.direction_tick_step

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        label: str | None = None,
    ) -> None:
        """Apply lateral-polar axis labeling, limits, and ticks."""
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.polar,
            values=values,
            tick_step=PolarAnglesAxis.direction_tick_step,
            default_limits=PolarAnglesAxis.polar_limits,
            label=label,
        )
