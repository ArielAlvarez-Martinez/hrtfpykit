from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

from .axis_helpers import apply_frequency_axis, build_frequency_axis
from .labels import Labels
from .options import (
    AxisOptions,
    AzimuthAxisOptions,
    AzimuthPolarAxisOptions,
    FrequencyAxisOptions,
    RadialPolarAxisOptions,
)


class Axis(ABC):
    """Base abstraction for axis-formatting strategies."""

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
        """Apply a resolved label to an axis dimension.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axis that receives the label.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        default_label : str
            Label used when no override is provided in ``options``.
        options : AxisOptions | None, default=None
            Axis options that may provide ``xlabel`` or ``ylabel`` overrides.

        Returns
        -------
        None

        Use Cases
        ---------
        - Apply consistent labels across multiple plotting methods.
        - Respect user overrides while keeping stable defaults.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> Axis.apply_label(ax=ax, axis="x", default_label="Frequency (kHz)")
        """
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
    """Base class for directional axes that use angular ticks."""

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
        """Apply directional limits and ticks to an axis.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axis that receives directional formatting.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        default_label : str
            Label applied when no axis-label override is provided.
        values : np.ndarray | None, default=None
            Direction values used to compute limits and internal ticks.
            When ``None``, ``default_limits`` are used if provided.
        tick_step : float | None, default=None
            Tick spacing in axis units.
        default_limits : tuple[float, float] | None, default=None
            Fallback limits used when ``values`` is not provided.
        options : AxisOptions | None, default=None
            Axis options used for label overrides.

        Returns
        -------
        None

        Use Cases
        ---------
        - Format azimuth, elevation, or polar-angle axes.
        - Keep directional ticks consistent across different plots.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> DirectionAxis.apply_direction(
        ...     ax=ax,
        ...     axis="x",
        ...     default_label="Azimuth",
        ...     values=np.array([-90.0, 0.0, 90.0]),
        ...     tick_step=30.0,
        ... )
        """
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
    """Log-frequency axis formatter."""

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
        """Build resolved options for a logarithmic frequency axis.

        Parameters
        ----------
        frequency_bins : np.ndarray | None, default=None
            Frequency bins in Hz used to resolve default bounds.
        freq_min : float | None, default=None
            Minimum frequency in Hz.
        freq_max : float | None, default=None
            Maximum frequency in Hz.
        options : FrequencyAxisOptions | None, default=None
            Additional frequency-axis options.

        Returns
        -------
        FrequencyAxisOptions
            Resolved logarithmic frequency-axis options.

        Use Cases
        ---------
        - Build consistent log-frequency settings for spectrum plots.

        Examples
        --------
        >>> import numpy as np
        >>> opts = FrequencyLogAxis.build(frequency_bins=np.linspace(20.0, 20000.0, 100))
        >>> float(opts.freq_min) > 0.0
        True
        """
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
        """Apply logarithmic frequency formatting to a Matplotlib axis.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        label : str | None, default=None
            Optional axis label.
        options : FrequencyAxisOptions | None, default=None
            Resolved frequency-axis options.

        Returns
        -------
        None

        Use Cases
        ---------
        - Format spectrum axes in log-frequency scale.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> FrequencyLogAxis.apply(ax=ax, axis="x", label="Frequency (kHz)", options=FrequencyAxisOptions(freq_min=20.0, freq_max=20000.0, ticks=(100.0, 1000.0, 10000.0), labels=("0.1", "1", "10"), margin_ratio=0.0))
        """
        apply_frequency_axis(
            scale="log",
            ax=ax,
            axis=axis,
            label=label,
            options=options,
        )


class FrequencyLinearAxis(Axis):
    """Linear-frequency axis formatter."""

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
        """Build resolved options for a linear frequency axis.

        Parameters
        ----------
        frequency_bins : np.ndarray | None, default=None
            Frequency bins in Hz used to resolve default bounds.
        freq_min : float | None, default=None
            Minimum frequency in Hz.
        freq_max : float | None, default=None
            Maximum frequency in Hz.
        options : FrequencyAxisOptions | None, default=None
            Additional frequency-axis options.

        Returns
        -------
        FrequencyAxisOptions
            Resolved linear frequency-axis options.

        Use Cases
        ---------
        - Build consistent linear-frequency settings for heatmaps.

        Examples
        --------
        >>> import numpy as np
        >>> opts = FrequencyLinearAxis.build(frequency_bins=np.linspace(0.0, 20000.0, 100))
        >>> float(opts.freq_max) >= float(opts.freq_min)
        True
        """
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
        """Apply linear frequency formatting to a Matplotlib axis.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        label : str | None, default=None
            Optional axis label.
        options : FrequencyAxisOptions | None, default=None
            Resolved frequency-axis options.

        Returns
        -------
        None

        Use Cases
        ---------
        - Format frequency axes in linear scale.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> FrequencyLinearAxis.apply(ax=ax, axis="x", label="Frequency (kHz)", options=FrequencyAxisOptions(freq_min=0.0, freq_max=8000.0, ticks=(1000.0, 2000.0), labels=("1", "2"), margin_ratio=0.0))
        """
        apply_frequency_axis(
            scale="linear",
            ax=ax,
            axis=axis,
            label=label,
            options=options,
        )


class MagnitudeAxis(Axis):
    """Magnitude-axis label formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        unit: str,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply magnitude label formatting.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        unit : str
            Magnitude unit. Expected values are ``"db"`` or ``"linear"``.
        options : AxisOptions | None, default=None
            Axis options that may override label text.

        Returns
        -------
        None

        Use Cases
        ---------
        - Label magnitude responses in dB or linear scale.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> MagnitudeAxis.apply(ax=ax, axis="y", unit="db")
        """
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
    """Amplitude-axis label formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply impulse-response amplitude label formatting.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        options : AxisOptions | None, default=None
            Axis options that may override label text.

        Returns
        -------
        None

        Use Cases
        ---------
        - Label HRIR amplitude plots consistently.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> AmplitudeAxis.apply(ax=ax, axis="y")
        """
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=Labels.impulse_response,
            options=options,
        )


class TimeAxis(Axis):
    """Time-axis label formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply time label formatting.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        options : AxisOptions | None, default=None
            Axis options that may override label text.

        Returns
        -------
        None

        Use Cases
        ---------
        - Label waveform plots in seconds.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> TimeAxis.apply(ax=ax, axis="x")
        """
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=Labels.time,
            options=options,
        )


class SampleAxis(Axis):
    """Sample-index axis label formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply sample-index label formatting.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        options : AxisOptions | None, default=None
            Axis options that may override label text.

        Returns
        -------
        None

        Use Cases
        ---------
        - Label waveform plots in samples.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> SampleAxis.apply(ax=ax, axis="x")
        """
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=Labels.samples,
            options=options,
        )


class XAxis(Axis):
    """3D X-axis label and limit formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply X-axis label and optional symmetric limits.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        center : float | None, default=None
            Axis center value.
        half_span : float | None, default=None
            Positive half span around ``center``.
        options : AxisOptions | None, default=None
            Axis options that may override label text.

        Returns
        -------
        None

        Use Cases
        ---------
        - Configure symmetric 3D X limits around a computed center.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> XAxis.apply(ax=ax, center=0.0, half_span=1.0)
        """
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
    """3D Y-axis label and limit formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply Y-axis label and optional symmetric limits.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        center : float | None, default=None
            Axis center value.
        half_span : float | None, default=None
            Positive half span around ``center``.
        options : AxisOptions | None, default=None
            Axis options that may override label text.

        Returns
        -------
        None

        Use Cases
        ---------
        - Configure symmetric 3D Y limits around a computed center.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> YAxis.apply(ax=ax, center=0.0, half_span=1.0)
        """
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
    """3D Z-axis label and limit formatter."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply Z-axis label and optional symmetric limits.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        center : float | None, default=None
            Axis center value.
        half_span : float | None, default=None
            Positive half span around ``center``.
        options : AxisOptions | None, default=None
            Axis options that may override label text.

        Returns
        -------
        None

        Use Cases
        ---------
        - Configure symmetric 3D Z limits around a computed center.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="3d")
        >>> ZAxis.apply(ax=ax, center=0.0, half_span=1.0)
        """
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
    """Azimuth-axis formatter for cartesian and polar-style ranges."""

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
        """Resolve azimuth range mode from axis options.

        Parameters
        ----------
        options : AxisOptions | None, default=None
            Axis options that may include ``azimuth_axis.range_mode``.

        Returns
        -------
        str
            Resolved range mode: ``"0-360"`` or ``"-180-180"``.

        Use Cases
        ---------
        - Keep azimuth transformation and tick logic consistent.

        Examples
        --------
        >>> AzimuthAnglesAxis.get_range_mode()
        '0-360'
        """
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
        """Transform azimuth values into the configured range mode.

        Parameters
        ----------
        values : np.ndarray
            Input azimuth values in degrees.
        options : AxisOptions | None, default=None
            Axis options that control azimuth range mode.

        Returns
        -------
        np.ndarray
            Transformed azimuth values in the selected range convention.

        Use Cases
        ---------
        - Convert source azimuths before sorting or plotting.

        Examples
        --------
        >>> import numpy as np
        >>> AzimuthAnglesAxis.transform_values(np.array([270.0]))
        array([270.])
        """
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
        """Apply azimuth-axis directional formatting.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        values : np.ndarray | None, default=None
            Azimuth values used to resolve limits and ticks.
        options : AxisOptions | None, default=None
            Axis options including azimuth range mode and label overrides.

        Returns
        -------
        None

        Use Cases
        ---------
        - Format horizontal-plane azimuth axes in signed or unsigned mode.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> AzimuthAnglesAxis.apply(ax=ax, axis="x")
        """
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


class AzimuthAnglesAxisPolarProjection(Axis):
    """Azimuth-axis formatter for polar projections."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply azimuth ticks and orientation on a polar axis.

        Parameters
        ----------
        ax : plt.Axes
            Target polar axis.
        options : AxisOptions | None, default=None
            Axis options containing ``azimuth_polar_axis.tick_step``.

        Returns
        -------
        None

        Use Cases
        ---------
        - Configure angular ticks for absolute ITD/ILD polar curves.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="polar")
        >>> AzimuthAnglesAxisPolarProjection.apply(ax=ax)
        """
        if getattr(ax, "name", "") != "polar":
            raise ValueError("AzimuthAnglesAxisPolarProjection requires a polar axis")
        axis_options = AxisOptions() if options is None else options
        azimuth_polar_axis_options = (
            AzimuthPolarAxisOptions()
            if axis_options.azimuth_polar_axis is None
            else axis_options.azimuth_polar_axis
        )
        resolved_tick_step = (
            30.0
            if azimuth_polar_axis_options.tick_step is None
            else float(azimuth_polar_axis_options.tick_step)
        )
        if not np.isfinite(resolved_tick_step) or resolved_tick_step <= 0.0:
            raise ValueError("tick_step must be a finite, positive value")
        theta_ticks = np.arange(0.0, 360.0, resolved_tick_step, dtype=float)
        ax.set_theta_zero_location("N")
        ax.set_xticks(np.deg2rad(theta_ticks))
        ax.set_xticklabels([f"{int(np.rint(tick))}°" for tick in theta_ticks])


class RadialAxisPolarProjection(Axis):
    """Radial-axis formatter for polar projections."""

    @staticmethod
    def apply(
        ax: plt.Axes,
        radial_values: np.ndarray,
        radial_label_default: str,
        options: AxisOptions | None = None,
    ) -> None:
        """Apply radial limits, ticks, and radial label on a polar axis.

        Parameters
        ----------
        ax : plt.Axes
            Target polar axis.
        radial_values : np.ndarray
            Values used to resolve radial maximum and tick spacing.
        radial_label_default : str
            Default radial label.
        options : AxisOptions | None, default=None
            Axis options that may provide radial-axis settings and ylabel override.

        Returns
        -------
        None

        Use Cases
        ---------
        - Format radial scale for absolute ITD and ILD polar plots.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection="polar")
        >>> RadialAxisPolarProjection.apply(ax=ax, radial_values=np.array([0.1, 0.2]), radial_label_default="Value")
        """
        if getattr(ax, "name", "") != "polar":
            raise ValueError("RadialAxisPolarProjection requires a polar axis")
        resolved_axis_options = AxisOptions() if options is None else options
        radial_polar_axis_options = (
            RadialPolarAxisOptions()
            if resolved_axis_options.radial_polar_axis is None
            else resolved_axis_options.radial_polar_axis
        )
        resolved_radial_values = np.asarray(radial_values, dtype=float).reshape(-1)
        if resolved_radial_values.size == 0:
            radial_max = 0.0
        else:
            if not np.all(np.isfinite(resolved_radial_values)):
                raise ValueError("radial_values must contain finite values")
            radial_max = float(np.max(resolved_radial_values))

        resolved_radial_tick_step = (
            1.0
            if radial_polar_axis_options.tick_step is None
            else float(radial_polar_axis_options.tick_step)
        )
        if (
            not np.isfinite(resolved_radial_tick_step)
            or resolved_radial_tick_step <= 0.0
        ):
            raise ValueError("radial_tick_step must be a finite, positive value")
        resolved_tick_label_style = (
            "integer"
            if radial_polar_axis_options.tick_label_style is None
            else str(radial_polar_axis_options.tick_label_style).strip()
        )
        if resolved_tick_label_style not in {"integer", "decimal_comma_4"}:
            raise ValueError(
                "radial_tick_label_style accepts integer or decimal_comma_4"
            )
        if np.isclose(radial_max, 0.0):
            resolved_radial_max = resolved_radial_tick_step
        else:
            resolved_radial_max = (
                np.ceil((radial_max * 1.1) / resolved_radial_tick_step)
                * resolved_radial_tick_step
            )
        radial_ticks = np.arange(
            resolved_radial_tick_step,
            resolved_radial_max + (0.5 * resolved_radial_tick_step),
            resolved_radial_tick_step,
            dtype=float,
        )
        ax.set_ylim(0.0, resolved_radial_max)
        ax.set_yticks(radial_ticks)
        if resolved_tick_label_style == "decimal_comma_4":
            radial_tick_labels = [f"{tick:0.4f}".replace(".", ",") for tick in radial_ticks]
        else:
            radial_tick_labels = [f"{int(np.rint(tick))}" for tick in radial_ticks]
        ax.set_yticklabels(radial_tick_labels)

        resolved_rlabel_position = (
            350.0
            if radial_polar_axis_options.label_position is None
            else float(radial_polar_axis_options.label_position)
        )
        if not np.isfinite(resolved_rlabel_position):
            raise ValueError("rlabel_position must be finite")
        ax.set_rlabel_position(resolved_rlabel_position)
        resolved_radial_label = (
            radial_label_default
            if resolved_axis_options.ylabel is None
            else resolved_axis_options.ylabel
        )
        ax.set_ylabel(resolved_radial_label, rotation=0)
        ax.yaxis.set_label_coords(0.5, ax.title.get_position()[1], transform=ax.transAxes)
        ax.yaxis.label.set_horizontalalignment("center")
        ax.yaxis.label.set_verticalalignment("bottom")


class ElevationAnglesAxis(DirectionAxis):
    """Elevation-axis directional formatter."""

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
        """Apply elevation-axis directional formatting.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        values : np.ndarray | None, default=None
            Elevation values used to resolve limits and ticks.
        options : AxisOptions | None, default=None
            Axis options that may override axis labels.

        Returns
        -------
        None

        Use Cases
        ---------
        - Format elevation axes in elevation-spectrum heatmaps.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ElevationAnglesAxis.apply(ax=ax, axis="y")
        """
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.elevation,
            values=values,
            tick_step=ElevationAnglesAxis.elevation_tick_step,
            options=options,
        )


class PolarAnglesAxis(DirectionAxis):
    """Polar-angle directional formatter."""

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
        """Apply polar-angle directional formatting.

        Parameters
        ----------
        ax : plt.Axes
            Target axis.
        axis : str
            Axis dimension: ``"x"``, ``"y"``, or ``"z"``.
        values : np.ndarray | None, default=None
            Polar-angle values used to resolve limits and ticks.
        options : AxisOptions | None, default=None
            Axis options that may override axis labels.

        Returns
        -------
        None

        Use Cases
        ---------
        - Format median-plane polar-angle axes in heatmaps.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> PolarAnglesAxis.apply(ax=ax, axis="y")
        """
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.polar,
            values=values,
            tick_step=PolarAnglesAxis.direction_tick_step,
            default_limits=PolarAnglesAxis.polar_limits,
            options=options,
        )
