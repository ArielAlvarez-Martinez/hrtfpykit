from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

from .axis_helpers import apply_frequency_axis, build_frequency_axis
from .labels import Labels


class Axis(ABC):
    """Base interface for Matplotlib axis formatters used by hrtfpykit.

    Axis defines the shared formatter contract and provides common label
    application for x, y, and z axes. Concrete subclasses specialize this
    interface for HRTF frequency axes, HRIR time/sample axes, interaural-cue
    axes, polar axes, and 3D source-grid axes.
    """

    @staticmethod
    @abstractmethod
    def apply(*args, **kwargs) -> None:
        """Apply formatting implemented by a concrete axis formatter.

        Concrete subclasses define the accepted parameters because frequency,
        direction, polar, and 3D axes require different context.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Always raised by the abstract base implementation.
        """
        raise NotImplementedError

    @staticmethod
    def apply_label(
        ax: plt.Axes,
        axis: str,
        default_label: str,
        label: str | None = None,
    ) -> None:
        """Apply x, y, or z axis labels with a default fallback.

        The method centralizes label resolution for all axis formatters. If
        label is None, default_label is used; otherwise label is
        converted to str and applied to the selected Matplotlib axis.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        default_label : str
            Default label used when label is not provided.
        label : str | None, default=None
            Optional explicit label.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If axis is not ``x``, ``y``, or ``z``, or if z-axis
            labeling is requested on a non-3D Matplotlib axis.

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
    """Shared formatter for angle-valued direction axes.

    :class:`~hrtfpykit.plots.axis.DirectionAxis` implements the common limit and
    tick logic used by azimuth, elevation, and lateral-polar angle axes.
    Subclasses provide default labels, tick spacing, and optional fixed limits
    while reusing the same Matplotlib locator and formatter behavior.
    """

    @classmethod
    def get_tick_step(cls) -> float:
        """Return the directional tick spacing used by the axis class.

        Returns
        -------
        float
            Tick spacing in degrees.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """
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

        The method applies a label, resolves axis limits from values or
        default_limits, and installs fixed major ticks inside the displayed
        range. Tick labels are integer degree values. Boundary ticks are omitted
        when values define the limits so endpoint labels do not duplicate the
        axis limits.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        default_label : str
            Default label text.
        values : np.ndarray | None, default=None
            Direction values used to resolve limits and interior ticks.
        tick_step : float | None, default=None
            Tick spacing in axis units.
        default_limits : tuple[float, float] | None, default=None
            Fallback limits used when values is not provided.
        label : str | None, default=None
            Optional explicit label.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If axis is invalid, z-axis formatting is requested on a non-3D
            axis, tick_step is not positive, values is empty, or
            values contains non-finite entries.

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
        set_limits: Any
        axis_object: Any
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

        tick_values_array: np.ndarray
        if resolved_values.size == 1:
            axis_limits = (
                float(resolved_values[0]),
                float(resolved_values[0]),
            )
            tick_values_array = np.array([], dtype=float)
        else:
            axis_limits = (
                float(resolved_values[0]),
                float(resolved_values[-1]),
            )
            tick_start = np.floor(axis_limits[0] / resolved_tick_step) + 1.0
            tick_stop = np.ceil(axis_limits[1] / resolved_tick_step) - 1.0
            if tick_start <= tick_stop:
                tick_values_array = np.arange(
                    tick_start * resolved_tick_step,
                    (tick_stop + 1.0) * resolved_tick_step,
                    resolved_tick_step,
                    dtype=float,
                )
            else:
                tick_values_array = np.array([], dtype=float)

        if tick_values_array.size > 0:
            lower_label = int(np.rint(axis_limits[0]))
            upper_label = int(np.rint(axis_limits[1]))
            tick_labels_int = np.rint(tick_values_array).astype(int)
            keep_mask = (tick_labels_int != lower_label) & (tick_labels_int != upper_label)
            tick_values_array = tick_values_array[keep_mask]

        set_limits(*axis_limits)
        tick_positions = tuple(float(value) for value in tick_values_array)
        tick_labels = tuple(f"{int(np.rint(value))}" for value in tick_positions)
        axis_object.set_major_locator(FixedLocator(tick_positions))
        axis_object.set_major_formatter(FixedFormatter(tick_labels))


class FrequencyLogAxis(Axis):
    """Logarithmic frequency-axis formatter for HRTF magnitude plots.

    :class:`~hrtfpykit.plots.axis.FrequencyLogAxis` builds and applies the
    standard logarithmic frequency scale used by HRTF spectrum plots.
    Configuration values are specified in Hz to match the frequency bins stored
    by frequency-domain objects, while
    :func:`~hrtfpykit.plots.axis_helpers.apply_frequency_axis` converts axis
    limits and ticks to kHz for display.

    Attributes
    ----------
    frequency_ticks : tuple[float, ...]
        Default logarithmic tick positions in Hz.
    """

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

        The returned dictionary is intended to be passed to
        :meth:`~hrtfpykit.plots.axis.FrequencyLogAxis.apply`. Frequency bounds
        and ticks are stored in Hz. Tick labels are filtered to the visible range
        and default to the
        hrtfpykit logarithmic frequency labels when custom ticks and labels are
        not provided.

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
            Tick labels matching ticks.
        margin_ratio : float, default=0.03
            Relative axis margin.

        Returns
        -------
        dict[str, float | tuple[float, ...] | tuple[str, ...]]
            Frequency-axis configuration dictionary.

        Raises
        ------
        ValueError
            If frequency bounds cannot be inferred, bounds are non-finite or
            empty, the lower bound is not positive, ticks and labels have
            different lengths, logarithmic ticks are non-positive, or
            margin_ratio is negative.

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

        config must be produced by
        :meth:`~hrtfpykit.plots.axis.FrequencyLogAxis.build` or contain the same
        keys: ``freq_min``, ``freq_max``, ``ticks``, ``labels``, and ``margin_ratio``.
        Values are interpreted in Hz and rendered in kHz on the selected
        Matplotlib axis.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        label : str | None, default=None
            Optional axis label.
        config : dict[str, float | tuple[float, ...] | tuple[str, ...]] | None
            Configuration produced by
            :meth:`~hrtfpykit.plots.axis.FrequencyLogAxis.build`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If config is missing or if delegated frequency-axis application
            receives an invalid axis or incompatible tick labels.

        """
        if config is None:
            raise ValueError("config is required")
        apply_frequency_axis(
            scale="log",
            ax=ax,
            axis=axis,
            label=label,
            freq_min=float(cast(Any, config["freq_min"])),
            freq_max=float(cast(Any, config["freq_max"])),
            ticks=tuple(config["ticks"]),  # type: ignore[arg-type]
            labels=tuple(config["labels"]),  # type: ignore[arg-type]
            margin_ratio=float(cast(Any, config["margin_ratio"])),
        )


class FrequencyLinearAxis(Axis):
    """Linear frequency-axis formatter for HRTF and comparison plots.

    :class:`~hrtfpykit.plots.axis.FrequencyLinearAxis` builds and applies the
    standard linear frequency scale used by HRTF spectrum, heatmap, comparison,
    and spherical-harmonic plots. Configuration values are specified in Hz to
    match stored frequency bins, while the applied Matplotlib axis is displayed
    in kHz.

    Attributes
    ----------
    frequency_ticks : tuple[float, ...]
        Default linear tick positions in Hz.
    """

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

        The returned dictionary is intended to be passed to
        :meth:`~hrtfpykit.plots.axis.FrequencyLinearAxis.apply`. Frequency
        bounds and ticks are stored in Hz. Tick labels are filtered to the
        visible range and default to the hrtfpykit linear frequency labels when
        custom ticks and labels are not provided.

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
            Tick labels matching ticks.
        margin_ratio : float, default=0.03
            Relative axis margin.

        Returns
        -------
        dict[str, float | tuple[float, ...] | tuple[str, ...]]
            Frequency-axis configuration dictionary.

        Raises
        ------
        ValueError
            If frequency bounds cannot be inferred, bounds are non-finite or
            empty, ticks and labels have different lengths, or margin_ratio
            is negative.

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

        config must be produced by
        :meth:`~hrtfpykit.plots.axis.FrequencyLinearAxis.build` or contain the
        same keys: ``freq_min``, ``freq_max``, ``ticks``, ``labels``, and
        ``margin_ratio``. Values are interpreted in Hz and rendered in kHz on the
        selected Matplotlib axis.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        label : str | None, default=None
            Optional axis label.
        config : dict[str, float | tuple[float, ...] | tuple[str, ...]] | None
            Configuration produced by
            :meth:`~hrtfpykit.plots.axis.FrequencyLinearAxis.build`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If config is missing or if delegated frequency-axis application
            receives an invalid axis or incompatible tick labels.

        """
        if config is None:
            raise ValueError("config is required")
        apply_frequency_axis(
            scale="linear",
            ax=ax,
            axis=axis,
            label=label,
            freq_min=float(cast(Any, config["freq_min"])),
            freq_max=float(cast(Any, config["freq_max"])),
            ticks=tuple(config["ticks"]),  # type: ignore[arg-type]
            labels=tuple(config["labels"]),  # type: ignore[arg-type]
            margin_ratio=float(cast(Any, config["margin_ratio"])),
        )


class MagnitudeAxis(Axis):
    """Axis-label formatter for HRTF magnitude values.

    :class:`~hrtfpykit.plots.axis.MagnitudeAxis` chooses the default magnitude
    label used by spectra and heatmaps. When ``unit`` is ``db``, the decibel label
    is selected; all other unit values select the linear magnitude label. Unit
    validation is performed by the calling plot functions.
    """

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        unit: str,
        label: str | None = None,
    ) -> None:
        """Apply a magnitude label to the selected axis.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        unit : str
            Magnitude unit. ``db`` selects Labels.magnitude_db; any other
            value selects Labels.magnitude_linear.
        label : str | None, default=None
            Optional explicit label overriding the unit-derived default.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the selected axis is invalid, or if z-axis labeling is requested
            on a non-3D Matplotlib axis.

        """
        default_label = Labels.magnitude_db if unit == "db" else Labels.magnitude_linear
        Axis.apply_label(ax=ax, axis=axis, default_label=default_label, label=label)


class AmplitudeAxis(Axis):
    """Axis-label formatter for time-domain HRIR amplitude plots.

    :class:`~hrtfpykit.plots.axis.AmplitudeAxis` applies the default ordinate
    label used by impulse-response visualizations. It does not configure limits
    or ticks because amplitude scaling depends on the plotted HRIR samples.
    """

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
    ) -> None:
        """Apply the impulse-response amplitude label to an axis.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        label : str | None, default=None
            Optional explicit label overriding Labels.impulse_response.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the selected axis is invalid, or if z-axis labeling is requested
            on a non-3D Matplotlib axis.

        """
        Axis.apply_label(
            ax=ax,
            axis=axis,
            default_label=Labels.impulse_response,
            label=label,
        )


class TimeAxis(Axis):
    """Axis-label formatter for HRIR time values in milliseconds.

    TimeAxis is used by impulse-response plots when the horizontal
    coordinate has already been converted from sample indices to milliseconds.
    """

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
    ) -> None:
        """Apply the default time label to an axis.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        label : str | None, default=None
            Optional explicit label overriding Labels.time.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the selected axis is invalid, or if z-axis labeling is requested
            on a non-3D Matplotlib axis.

        """
        Axis.apply_label(ax=ax, axis=axis, default_label=Labels.time, label=label)


class SampleAxis(Axis):
    """Axis-label formatter for discrete HRIR sample indices.

    :class:`~hrtfpykit.plots.axis.SampleAxis` is used by impulse-response plots
    that display the raw sample index rather than a physical time vector.
    """

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        label: str | None = None,
    ) -> None:
        """Apply the default sample-index label to an axis.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        label : str | None, default=None
            Optional explicit label overriding Labels.samples.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the selected axis is invalid, or if z-axis labeling is requested
            on a non-3D Matplotlib axis.

        """
        Axis.apply_label(ax=ax, axis=axis, default_label=Labels.samples, label=label)


class XAxis(Axis):
    """3D x-axis label and symmetric-limit formatter.

    :class:`~hrtfpykit.plots.axis.XAxis` is used by source-grid plots after
    spherical SOFA source positions are converted to Cartesian coordinates. When
    ``center`` and ``half_span`` are supplied, the formatter applies limits that
    keep the rendered 3D source cloud visually balanced with the y and z axes.
    """

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        label: str | None = None,
    ) -> None:
        """Apply x-axis label and optionally symmetric limits around a center.

        The label is always applied. Limits are changed only when both
        center and half_span are provided; omitting both leaves the
        current Matplotlib limits unchanged.

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

        Raises
        ------
        ValueError
            If only one of center or half_span is provided, if the
            center is not finite, or if the half-span is not finite and
            positive.

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
    """3D y-axis label and symmetric-limit formatter.

    :class:`~hrtfpykit.plots.axis.YAxis` mirrors
    :class:`~hrtfpykit.plots.axis.XAxis` for the Cartesian y coordinate in 3D
    source-grid views. It is normally used together with
    :class:`~hrtfpykit.plots.axis.XAxis` and
    :class:`~hrtfpykit.plots.axis.ZAxis` so source positions use comparable
    limits on all dimensions.
    """

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        label: str | None = None,
    ) -> None:
        """Apply y-axis label and optionally symmetric limits around a center.

        The label is always applied. Limits are changed only when both
        center and half_span are provided; omitting both leaves the
        current Matplotlib limits unchanged.

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

        Raises
        ------
        ValueError
            If only one of center or half_span is provided, if the
            center is not finite, or if the half-span is not finite and
            positive.

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
    """3D z-axis label and symmetric-limit formatter.

    :class:`~hrtfpykit.plots.axis.ZAxis` completes the Cartesian-axis formatting
    used by 3D source-grid plots. Unlike
    :class:`~hrtfpykit.plots.axis.XAxis` and
    :class:`~hrtfpykit.plots.axis.YAxis`, it requires a Matplotlib 3D axis when
    either labeling or limit application reaches z-axis-specific methods.
    """

    @staticmethod
    def apply(
        ax: plt.Axes,
        center: float | None = None,
        half_span: float | None = None,
        label: str | None = None,
    ) -> None:
        """Apply z-axis label and optionally symmetric limits around a center.

        The label is always applied through the Matplotlib 3D z-axis API.
        Limits are changed only when both center and half_span are
        provided; omitting both leaves the current z limits unchanged.

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

        Raises
        ------
        ValueError
            If ax is not a 3D axis, if only one of center or
            half_span is provided, if the center is not finite, or if the
            half-span is not finite and positive.

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
    """Azimuth-axis formatter with configurable signed or unsigned range.

    :class:`~hrtfpykit.plots.axis.AzimuthAnglesAxis` normalizes azimuth values
    and applies the direction formatter used by horizontal-plane,
    source-position, and cue-difference plots. hrtfpykit follows the SOFA
    spherical convention in degrees: azimuth increases anticlockwise in the
    listener-centered horizontal plane, with front at 0 degrees, listener-left
    at 90 degrees, back at 180 degrees, and listener-right at 270 degrees.

    The default ``0-360`` range shows this convention directly. The signed
    ``-180-180`` range wraps listener-right to -90 degrees while listener-left
    remains +90 degrees. When signed azimuth is applied to the x-axis,
    hrtfpykit reverses the displayed x-axis so listener-left appears on the
    left side of the figure and listener-right appears on the right side.
    Signed azimuth on y-axes keeps the normal Matplotlib numeric orientation.

    Attributes
    ----------
    direction_tick_step : float
        Default azimuth tick spacing in degrees.
    azimuth_range_modes : tuple[str, str]
        Supported range-mode identifiers.
    azimuth_limits_unsigned : tuple[float, float]
        Default axis limits for ``0-360`` mode.
    azimuth_limits_signed : tuple[float, float]
        Default axis limits for ``-180-180`` mode.
    """

    direction_tick_step: float = 40.0
    azimuth_range_modes: tuple[str, str] = ("0-360", "-180-180")
    azimuth_limits_unsigned: tuple[float, float] = (0.0, 360.0)
    azimuth_limits_signed: tuple[float, float] = (-180.0, 180.0)

    @classmethod
    def get_tick_step(cls) -> float:
        """Return the default azimuth tick spacing.

        Returns
        -------
        float
            Tick spacing in degrees used by
            :meth:`~hrtfpykit.plots.axis.AzimuthAnglesAxis.apply`.
        """
        return cls.direction_tick_step

    @staticmethod
    def get_range_mode(range_mode: str | None = None) -> str:
        """Resolve and validate the azimuth range convention.

        Parameters
        ----------
        range_mode : str | None, default=None
            Requested azimuth convention. None resolves to ``-180-180``.

        Returns
        -------
        str
            Validated range mode, either ``0-360`` or ``-180-180``.

        Raises
        ------
        ValueError
            If range_mode is not one of the supported conventions.

        """
        resolved_range_mode = (
            AzimuthAnglesAxis.azimuth_range_modes[1]
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

        Values are interpreted in degrees. In ``0-360`` mode, values wrap
        with modulo 360. In ``-180-180`` mode, values wrap to the signed
        interval and values numerically equal to -180 are represented as
        180 to avoid a duplicate boundary label.

        Parameters
        ----------
        values : np.ndarray
            Input azimuth values in degrees.
        range_mode : str | None, default=None
            ``0-360`` or ``-180-180``.

        Returns
        -------
        np.ndarray
            Transformed azimuth values in degrees.

        Raises
        ------
        ValueError
            If range_mode is not one of the supported conventions.

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
        """Apply azimuth-axis labels, limits, and ticks.

        If values are provided, they are first normalized with
        :meth:`~hrtfpykit.plots.axis.AzimuthAnglesAxis.transform_values`, then
        passed to
        :meth:`~hrtfpykit.plots.axis.DirectionAxis.apply_direction` so limits
        follow the data. If values is None, the selected range mode supplies the
        default azimuth limits. In signed ``-180-180`` mode, x-axis azimuth is
        displayed with +90 degrees on the left side of the figure and
        -90 degrees on the right side. Signed y-axis azimuth keeps the normal
        increasing bottom-to-top numeric orientation.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        values : np.ndarray | None, default=None
            Optional azimuth values in degrees.
        range_mode : str | None, default=None
            Azimuth convention, either ``0-360`` or ``-180-180``.
        label : str | None, default=None
            Optional explicit label overriding Labels.azimuth.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the range mode, axis selector, tick step, or provided direction
            values are invalid.

        """
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
        if axis == "x" and resolved_range_mode == AzimuthAnglesAxis.azimuth_range_modes[1]:
            x_lower, x_upper = ax.get_xlim()
            if x_lower < x_upper:
                ax.set_xlim(x_upper, x_lower)


class AzimuthAnglesAxisPolarProjection(Axis):
    """Azimuth formatter for Matplotlib polar projection axes.

    :class:`~hrtfpykit.plots.axis.AzimuthAnglesAxisPolarProjection` configures
    polar-theta ticks for circular HRTF plots. The zero-angle direction is set
    to north so azimuth plots follow the conventional top-down spatial
    orientation used elsewhere in hrtfpykit.
    """

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

        Raises
        ------
        ValueError
            If ax is not a polar projection, or if tick_step is not a
            finite positive value.

        """
        if getattr(ax, "name", "") != "polar":
            raise ValueError("AzimuthAnglesAxisPolarProjection requires a polar axis")
        resolved_tick_step = float(tick_step)
        if not np.isfinite(resolved_tick_step) or resolved_tick_step <= 0.0:
            raise ValueError("tick_step must be a finite, positive value")
        theta_ticks = np.arange(0.0, 360.0, resolved_tick_step, dtype=float)
        cast(Any, ax).set_theta_zero_location("N")
        ax.set_xticks(np.deg2rad(theta_ticks))
        ax.set_xticklabels([f"{int(np.rint(tick))}°" for tick in theta_ticks])


class RadialAxisPolarProjection(Axis):
    """Radial-axis formatter for Matplotlib polar projection axes.

    :class:`~hrtfpykit.plots.axis.RadialAxisPolarProjection` derives radial
    limits from cue or metric values and formats radial tick labels for polar
    HRTF comparisons. It is used when a circular azimuth view needs a radial
    scale for magnitude, error, directivity, or another scalar plotted against
    direction.
    """

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

        The radial maximum is expanded to the next tick boundary after a small
        margin. Empty radial data produces a one-step radial range so the axis
        remains drawable.

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
        tick_label_style : str, default=``integer``
            Tick-label formatting mode: ``integer`` or ``decimal_comma_4``.
        label_position : float, default=350.0
            Radial label angular position in degrees.
        label : str | None, default=None
            Optional explicit radial-axis label.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ax is not a polar projection, if radial values contain
            non-finite data, if tick_step is not finite and positive, if
            tick_label_style is unsupported, or if label_position is
            not finite.

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
        cast(Any, ax).set_rlabel_position(resolved_label_position)
        resolved_radial_label = radial_label_default if label is None else str(label)
        ax.set_ylabel(resolved_radial_label, rotation=0)
        ax.yaxis.set_label_coords(0.5, ax.title.get_position()[1], transform=ax.transAxes)
        ax.yaxis.label.set_horizontalalignment("center")
        ax.yaxis.label.set_verticalalignment("bottom")


class ElevationAnglesAxis(DirectionAxis):
    """Axis formatter for elevation-angle coordinates in spatial plots.

    :class:`~hrtfpykit.plots.axis.ElevationAnglesAxis` applies the shared
    direction formatting used by source-position and plane-selection plots where
    elevation is measured in degrees above or below the horizontal plane.

    Attributes
    ----------
    elevation_tick_step : float
        Default elevation tick spacing in degrees.
    """

    elevation_tick_step: float = 20.0

    @classmethod
    def get_tick_step(cls) -> float:
        """Return the default elevation tick spacing.

        Returns
        -------
        float
            Tick spacing in degrees used by
            :meth:`~hrtfpykit.plots.axis.ElevationAnglesAxis.apply`.
        """
        return cls.elevation_tick_step

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        label: str | None = None,
    ) -> None:
        """Apply elevation-axis labels, limits, and ticks.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        values : np.ndarray | None, default=None
            Optional elevation values in degrees used to derive axis limits.
        label : str | None, default=None
            Optional explicit label overriding Labels.elevation.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the axis selector is invalid, z-axis formatting is requested on
            a non-3D axis, or provided direction values are empty or non-finite.

        """
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.elevation,
            values=values,
            tick_step=ElevationAnglesAxis.elevation_tick_step,
            label=label,
        )


class PolarAnglesAxis(DirectionAxis):
    """Lateral-polar angle axis formatter.

    :class:`~hrtfpykit.plots.axis.PolarAnglesAxis` formats the lateral-polar
    coordinate used by spherical-harmonic and source-direction visualizations.
    When values are not supplied, it uses the full -90 to 270 degree range
    expected by the local plotting conventions.

    Attributes
    ----------
    direction_tick_step : float
        Default polar-angle tick spacing in degrees.
    polar_limits : tuple[float, float]
        Default lateral-polar axis limits in degrees.
    """

    direction_tick_step: float = 20.0
    polar_limits: tuple[float, float] = (-90.0, 270.0)

    @classmethod
    def get_tick_step(cls) -> float:
        """Return the default lateral-polar tick spacing.

        Returns
        -------
        float
            Tick spacing in degrees used by
            :meth:`~hrtfpykit.plots.axis.PolarAnglesAxis.apply`.
        """
        return cls.direction_tick_step

    @staticmethod
    def apply(
        ax: plt.Axes,
        axis: str,
        values: np.ndarray | None = None,
        label: str | None = None,
    ) -> None:
        """Apply lateral-polar axis labels, limits, and ticks.

        Parameters
        ----------
        ax : plt.Axes
            Target Matplotlib axis.
        axis : str
            Axis selector: ``x``, ``y``, or ``z``.
        values : np.ndarray | None, default=None
            Optional lateral-polar values in degrees used to derive limits.
        label : str | None, default=None
            Optional explicit label overriding Labels.polar.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the axis selector is invalid, z-axis formatting is requested on
            a non-3D axis, or provided direction values are empty or non-finite.

        """
        DirectionAxis.apply_direction(
            ax=ax,
            axis=axis,
            default_label=Labels.polar,
            values=values,
            tick_step=PolarAnglesAxis.direction_tick_step,
            default_limits=PolarAnglesAxis.polar_limits,
            label=label,
        )
