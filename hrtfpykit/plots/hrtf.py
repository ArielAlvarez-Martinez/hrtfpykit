from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from .axis import (
    AmplitudeAxis,
    Axis,
    AzimuthAnglesAxis,
    AzimuthAnglesAxisPolarProjection,
    ElevationAnglesAxis,
    FrequencyLinearAxis,
    FrequencyLogAxis,
    MagnitudeAxis,
    PolarAnglesAxis,
    RadialAxisPolarProjection,
    SampleAxis,
    TimeAxis,
    XAxis,
    YAxis,
    ZAxis,
)
from .axis_helpers import (
    create_sources_grid_direction_markers,
    resolve_three_dimensional_axis_geometry,
)
from .default import Margins
from .figure import Figure
from .labels import Labels
from .layouts import Layout_1, Layout_2Horizontal, Layout_2Vertical, Layout_3
from .legends import Ear
from .polar import create_horizontal_plane_curve
from .titles import Titles
from ..hrtf.coordinates import (
    get_named_positions,
    get_position_queries,
    get_source_positions,
    spherical_to_lateral_polar,
)
from ..hrtf.dsp import magnitude_to_db, tf_from_ir
from ..hrtf.metrics import ild, itd
from ..hrtf.planes import (
    get_frontal_plane,
    get_horizontal_plane,
    get_median_plane,
)


if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


class HRTFPlots:
    """Visualization methods inherited by :class:`~hrtfpykit.hrtf.HRTF`.

    :class:`~hrtfpykit.plots.hrtf.HRTFPlots` groups the plotting operations that operate on an HRTF
    object's current :class:`~hrtfpykit.hrtf.domain.IR`, :class:`~hrtfpykit.hrtf.domain.TF`, and :class:`~hrtfpykit.hrtf.sources.Sources` state. The methods are
    attached to the main :class:`~hrtfpykit.hrtf.HRTF` abstraction so plots
    reflect any previous loading, source selection, domain conversion, ITD
    compensation, windowing, or other transformation applied to the object.

    The plotting surface covers the main inspection workflows used in HRTF and
    HRIR analysis: time-domain impulse responses, frequency-domain magnitude
    responses, frequency-angle spectrum heatmaps, interaural cue curves,
    absolute polar cue summaries, and 3D source-grid geometry. Position queries
    are resolved through the same source manager used by processing methods, so
    named directions, spherical coordinate queries, and plane selections follow
    the library's normal spatial-resolution rules.

    Notes
    -----
    These methods are intended to be called from an :class:`~hrtfpykit.hrtf.HRTF` instance, not from
    a standalone :class:`~hrtfpykit.plots.hrtf.HRTFPlots` object. Every plotting method validates the
    required domain representation before drawing, creates a Matplotlib figure
    through the hrtfpykit plotting wrappers, optionally calls
    matplotlib.pyplot.show(), and returns None.
    """

    def plot_magnitude(
        hrtf: "HRTF",
        positions: str | list | tuple | np.ndarray = ("front", "back", "left", "right"),
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = 1.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot HRTF magnitude responses at selected source positions.

        The method selects one to four source directions from the current HRTF
        source grid, extracts the corresponding transfer-function magnitudes,
        and draws one subplot per direction. Frequency values are displayed in
        kilohertz while freq_min and freq_max are interpreted in hertz,
        matching the frequency bins stored by the frequency-domain object.

        Position queries are resolved in spherical coordinates in degrees.
        Named positions use hrtfpykit's built-in aliases, while numeric queries
        should use [azimuth, elevation]. When unit=``db``, magnitudes are
        converted with magnitude-to-decibel conversion using either the supplied numeric
        reference or the maximum selected value when reference=``max``.

        Parameters
        ----------
        positions : str | list | tuple | np.ndarray, default=(``front``, ``back``, ``left``, ``right``)
            One position or a collection of positions. Named aliases such as
            ``front``, ``back``, ``left``, and ``right`` are accepted.
            Numeric queries must use spherical coordinates in degrees as
            [azimuth, elevation], for example [0.0, 0.0] for the front
            direction. Up to four positions can be shown in one figure.
        x_axis : {``linear``, ``log``}, default=``linear``
            Frequency scale used on the x axis.
        unit : {``db``, ``linear``}, default=``db``
            Magnitude representation used on the y axis.
        ear : {``left``, ``right``, ``both``}, default=``both``
            Ear channel to display. When ``both`` is selected, left and right
            responses are drawn together in each subplot.
        reference : float | {``max``}, default=1.0
            Reference used when unit=``db``. ``max`` normalizes the plotted
            magnitude to the maximum selected value.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress generated default subplot titles.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If unit, x_axis, or ear is not one of the supported
            values.
        ValueError
            If TF values or frequency bins are missing, no positions are
            requested, more than four positions are requested, the selected
            frequency range contains no bins, or the requested ear channel is
            not available.

        Notes
        -----
        One position uses :class:`~hrtfpykit.plots.layouts.Layout_1`, two positions use :class:`~hrtfpykit.plots.layouts.Layout_2Vertical`,
        and three or four positions use :class:`~hrtfpykit.plots.layouts.Layout_3`. With ear=``both``, left
        and right channels are drawn together on each subplot and labelled with
        the shared ear legend.
        """
        if unit not in {"db", "linear"}:
            raise AttributeError(
                "unit accepts : db or linear"
            )
        if x_axis not in {"log", "linear"}:
            raise AttributeError(
                "x_axis accepts log or linear"
            )
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        resolved_margins = Margins()

        if hrtf.TF.values is None or hrtf.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        position_queries = get_position_queries(positions)
        position_count = len(position_queries)
        if position_count == 0:
            raise ValueError("At least one position is required")
        if position_count > 4:
            raise ValueError("plot_magnitude accepts up to 4 positions")

        if position_count == 1:
            resolved_layout = Layout_1(
                figsize=Layout_1().figsize,
                margins=resolved_margins,
            )
        elif position_count == 2:
            resolved_layout = Layout_2Vertical(
                figsize=Layout_2Vertical().figsize,
                margins=resolved_margins,
            )
        else:
            resolved_layout = Layout_3(
                figsize=Layout_3().figsize,
                margins=resolved_margins,
            )
        figure = Figure(resolved_layout)

        frequency_bins_hz = np.asarray(hrtf.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        selected_position_info = [
            hrtf.Sources.get_position_index(
                selected_position_query,
                coordinate_system="spherical",
            )
            for selected_position_query in position_queries
        ]
        tf_magnitude = hrtf.TF.magnitude
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
        magnitude_legend_location = "upper right" if x_axis == "linear" else "upper left"

        for index, (_, selected_positions) in enumerate(selected_position_info):
            ax = figure.get_ax(index)
            frequency_axis = (
                FrequencyLogAxis if x_axis == "log" else FrequencyLinearAxis
            )
            resolved_frequency_axis = frequency_axis.build(
                frequency_bins=frequency_bins_hz,
                freq_min=freq_min,
                freq_max=freq_max,
                )
            frequency_mask = (
                (frequency_bins_hz >= float(resolved_frequency_axis["freq_min"]))
                & (frequency_bins_hz <= float(resolved_frequency_axis["freq_max"]))
            )
            if not np.any(frequency_mask):
                raise ValueError("Selected frequency range produced no TF bins")
            frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0
            frequency_label = Labels.frequency
            idxs = int(selected_position_info[index][0])
            selected_positions = np.asarray(selected_positions, dtype=float)
            y_values = np.asarray(tf_values[idxs][..., frequency_mask], dtype=float)

            if ear == "both":
                if y_values.ndim < 2 or y_values.shape[0] < 2:
                    raise ValueError("Both ears requested but TF data does not contain two ear channels")
                figure.create_two_dimension(
                    ax=ax,
                    x=frequency_khz,
                    y=y_values[0, :],
                    color="blue",
                )
                figure.create_two_dimension(
                    ax=ax,
                    x=frequency_khz,
                    y=y_values[1, :],
                    color="red",
                )
            else:
                if y_values.ndim == 1:
                    selected_y_values = y_values.reshape(-1)
                else:
                    ear_index = 0 if ear == "left" else 1
                    if y_values.shape[0] <= ear_index:
                        raise ValueError(f"Requested ear '{ear}' is not available in TF data")
                    selected_y_values = np.asarray(y_values[ear_index], dtype=float).reshape(-1)
                figure.create_two_dimension(
                    ax=ax,
                    x=frequency_khz,
                    y=selected_y_values,
                    color="blue",
                )

            frequency_axis.apply(
                ax=ax,
                axis="x",
                label=frequency_label,
                config=resolved_frequency_axis,
            )
            MagnitudeAxis.apply(ax=ax, axis="y", unit=unit)
            default_subplot_title = Titles.create_position_title(
                selected_positions=selected_positions,
            )
            resolved_subplot_title = default_subplot_title if titles else ""
            Titles.create_subplots_titles(ax=ax, title=resolved_subplot_title)
            if Figure.shared_x_visible:
                ax.tick_params(axis="x", which="both", labelbottom=True)
            Ear.apply(ax=ax, ear=ear, location=magnitude_legend_location, labels=None)
            ax.grid(True)

        if position_count < figure.axes.size:
            figure.hide_unused_axes(position_count)

        if show:
            plt.show()
        return None

    def plot_amplitude(
        hrtf: "HRTF",
        positions: str | list | tuple | np.ndarray = ("front", "back", "left", "right"),
        ear: str = "both",
        x_axis: str = "time",
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot HRIR amplitude responses for up to four source positions.

        The method selects one to four source directions from the current
        source grid and draws the corresponding time-domain impulse responses.
        Positions are resolved in spherical coordinates in degrees and may be
        provided through named aliases or numeric [azimuth, elevation]
        queries.

        The x-axis can show elapsed time in seconds or raw sample indices. Time
        mode requires :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` because sample positions are converted
        to seconds before plotting.

        Parameters
        ----------
        positions : str | list | tuple | np.ndarray, default=(``front``, ``back``, ``left``, ``right``)
            One position or a collection of positions. Named aliases such as
            ``front``, ``back``, ``left``, and ``right`` are accepted.
            Numeric queries must use spherical coordinates in degrees as
            [azimuth, elevation], for example [0.0, 0.0] for the front
            direction. Up to four positions can be shown in one figure.
        ear : {``left``, ``right``, ``both``}, default=``both``
            Ear channel to display. When ``both`` is selected, left and right
            ear waveforms are drawn together in each subplot.
        x_axis : {``time``, ``samples``}, default=``time``
            Horizontal axis used for the waveform plot.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress generated default subplot titles.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If ear or x_axis is not one of the supported values.
        ValueError
            If IR data is missing, a time axis is requested without a sample
            rate, no positions are requested, more than four positions are
            requested, the IR array has no samples, or the requested ear channel
            is not available.

        Notes
        -----
        One position uses :class:`~hrtfpykit.plots.layouts.Layout_1`, two positions use :class:`~hrtfpykit.plots.layouts.Layout_2Vertical`,
        and three or four positions use :class:`~hrtfpykit.plots.layouts.Layout_3`. With ear=``both``, left
        and right HRIR channels are drawn on the same subplot and labelled with
        the shared ear legend.
        """
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        if x_axis not in {"time", "samples"}:
            raise AttributeError(
                "x_axis accepts : time or samples"
            )
        resolved_margins = Margins()

        if hrtf.IR.values is None:
            raise ValueError("IR data is not available")
        if x_axis == "time" and hrtf.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required when x_axis='time'")

        position_queries = get_position_queries(positions)
        position_count = len(position_queries)
        if position_count == 0:
            raise ValueError("At least one position is required")
        if position_count > 4:
            raise ValueError("plot_amplitude accepts up to 4 positions")

        if position_count == 1:
            resolved_layout = Layout_1(
                figsize=Layout_1().figsize,
                margins=resolved_margins,
            )
        elif position_count == 2:
            resolved_layout = Layout_2Vertical(
                figsize=Layout_2Vertical().figsize,
                margins=resolved_margins,
            )
        else:
            resolved_layout = Layout_3(
                figsize=Layout_3().figsize,
                margins=resolved_margins,
            )
        figure = Figure(resolved_layout)

        ir_values = np.asarray(hrtf.IR.values, dtype=float)
        if ir_values.ndim < 2 or ir_values.shape[-1] == 0:
            raise ValueError("IR values must contain at least one sample")
        sample_indexes = np.arange(ir_values.shape[-1], dtype=float)
        if x_axis == "time":
            x_values = sample_indexes / float(hrtf.IR.sample_rate)
        else:
            x_values = sample_indexes

        for index, selected_position_query in enumerate(position_queries):
            ax = figure.get_ax(index)
            idxs, selected_positions = hrtf.Sources.get_position_index(
                selected_position_query,
                coordinate_system="spherical",
            )
            selected_positions = np.asarray(selected_positions, dtype=float)
            y_values = np.asarray(ir_values[idxs], dtype=float)

            if ear == "both":
                if y_values.ndim < 2 or y_values.shape[0] < 2:
                    raise ValueError("Both ears requested but IR data does not contain two ear channels")
                figure.create_two_dimension(
                    ax=ax,
                    x=x_values,
                    y=y_values[0, :],
                    color="blue",
                )
                figure.create_two_dimension(
                    ax=ax,
                    x=x_values,
                    y=y_values[1, :],
                    color="red",
                )
            else:
                if y_values.ndim == 1:
                    selected_y_values = y_values.reshape(-1)
                else:
                    ear_index = 0 if ear == "left" else 1
                    if y_values.shape[0] <= ear_index:
                        raise ValueError(f"Requested ear '{ear}' is not available in IR data")
                    selected_y_values = np.asarray(y_values[ear_index], dtype=float).reshape(-1)
                figure.create_two_dimension(
                    ax=ax,
                    x=x_values,
                    y=selected_y_values,
                    color="blue",
                )

            if x_axis == "time":
                TimeAxis.apply(ax=ax, axis="x")
            else:
                SampleAxis.apply(ax=ax, axis="x")
            AmplitudeAxis.apply(ax=ax, axis="y")
            default_subplot_title = Titles.create_position_title(
                selected_positions=selected_positions,
            )
            resolved_subplot_title = default_subplot_title if titles else ""
            Titles.create_subplots_titles(ax=ax, title=resolved_subplot_title)
            if Figure.shared_x_visible:
                ax.tick_params(axis="x", which="both", labelbottom=True)
            Ear.apply(ax=ax, ear=ear, location="upper right", labels=None)
            ax.grid(True)

        if position_count < figure.axes.size:
            figure.hide_unused_axes(position_count)

        if show:
            plt.show()
        return None

    def plot_amplitude_and_magnitude(
        hrtf: "HRTF",
        position: str | list | np.ndarray = "front",
        ear: str = "both",
        amplitude_x_axis: str = "time",
        magnitude_x_axis: str = "linear",
        magnitude: str = "db",
        reference: float | str = 1.0,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot amplitude and magnitude views for a single source position.

        This method creates a two-panel summary for one source direction. The
        top subplot shows the time-domain HRIR amplitude response and the
        bottom subplot shows the corresponding frequency-domain HRTF magnitude
        response for the same source. The amplitude subplot can use seconds or
        sample indices, while the magnitude subplot can use a linear or
        logarithmic frequency axis.

        The method requires both the time-domain and frequency-domain
        representations to be available. When magnitude=``db``, the magnitude
        response is converted with magnitude-to-decibel conversion using either a numeric
        reference or the selected direction's maximum magnitude when
        reference=``max``.

        Parameters
        ----------
        position : str | list | np.ndarray, default=``front``
            Position query to plot. Exactly one position is accepted. Named
            aliases such as ``front``, ``back``, ``left``, and ``right``
            are accepted. Numeric queries must use spherical coordinates in
            degrees as [azimuth, elevation], for example [0.0, 0.0].
        ear : {``left``, ``right``, ``both``}, default=``both``
            Ear channel to display in both subplots.
        amplitude_x_axis : {``time``, ``samples``}, default=``time``
            Horizontal axis used for the amplitude subplot.
        magnitude_x_axis : {``linear``, ``log``}, default=``linear``
            Frequency-axis scale used on the magnitude subplot.
        magnitude : {``db``, ``linear``}, default=``db``
            Magnitude representation used on the bottom subplot.
        reference : float | {``max``}, default=1.0
            Reference used when magnitude=``db`` for the magnitude subplot.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress the generated default figure title.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If ear, amplitude_x_axis, magnitude_x_axis, or
            magnitude is not one of the supported values.
        ValueError
            If IR or TF data is missing, a time axis is requested without a
            sample rate, zero or multiple positions are supplied, the selected
            frequency range contains no bins, or the requested ear channel is
            not available in IR or TF data.

        Notes
        -----
        The plot uses a vertical two-panel layout with independent x axes. This
        makes it useful for checking whether a transformation applied to the
        HRIR is reflected in the corresponding magnitude response.
        """
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        if amplitude_x_axis not in {"time", "samples"}:
            raise AttributeError(
                "amplitude_x_axis accepts : time or samples"
            )
        if magnitude_x_axis not in {"log", "linear"}:
            raise AttributeError(
                "magnitude_x_axis accepts log or linear"
            )
        if magnitude not in {"db", "linear"}:
            raise AttributeError(
                "magnitude accepts : db or linear"
            )
        resolved_margins = Margins()

        if hrtf.IR.values is None:
            raise ValueError("IR data is not available")
        if hrtf.TF.values is None or hrtf.TF.frequency_bins is None:
            raise ValueError("TF data is not available")
        if amplitude_x_axis == "time" and hrtf.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required when amplitude_x_axis='time'")

        position_queries = get_position_queries(position)
        if len(position_queries) != 1:
            raise ValueError(
                "plot_amplitude_and_magnitude accepts exactly one position"
            )
        selected_position_query = position_queries[0]

        figure = Figure(
            Layout_2Vertical(
                figsize=(8, 8),
                margins=resolved_margins,
                sharex=False,
            )
        )

        idxs, selected_positions = hrtf.Sources.get_position_index(
            selected_position_query,
            coordinate_system="spherical",
        )
        selected_positions = np.asarray(selected_positions, dtype=float)

        ir_values = np.asarray(hrtf.IR.values, dtype=float)
        if ir_values.ndim < 2 or ir_values.shape[-1] == 0:
            raise ValueError("IR values must contain at least one sample")
        sample_indexes = np.arange(ir_values.shape[-1], dtype=float)
        x_values = (
            sample_indexes / float(hrtf.IR.sample_rate)
            if amplitude_x_axis == "time"
            else sample_indexes
        )
        ir_y_values = np.asarray(ir_values[idxs], dtype=float)

        ir_ax = figure.get_ax("top")
        if ear == "both":
            if ir_y_values.ndim < 2 or ir_y_values.shape[0] < 2:
                raise ValueError(
                    "Both ears requested but IR data does not contain two ear channels"
                )
            figure.create_two_dimension(
                ax=ir_ax,
                x=x_values,
                y=ir_y_values[0, :],
                color="blue",
            )
            figure.create_two_dimension(
                ax=ir_ax,
                x=x_values,
                y=ir_y_values[1, :],
                color="red",
            )
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
            figure.create_two_dimension(
                ax=ir_ax,
                x=x_values,
                y=selected_ir_y_values,
                color="blue",
            )

        if amplitude_x_axis == "time":
            TimeAxis.apply(ax=ir_ax, axis="x")
        else:
            SampleAxis.apply(ax=ir_ax, axis="x")
        AmplitudeAxis.apply(ax=ir_ax, axis="y")
        if Figure.shared_x_visible:
            ir_ax.tick_params(axis="x", which="both", labelbottom=True)
        Ear.apply(ax=ir_ax, ear=ear, location="upper right", labels=None)
        ir_ax.grid(True)

        frequency_bins_hz = np.asarray(hrtf.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        magnitude_frequency_axis = (
            FrequencyLogAxis
            if magnitude_x_axis == "log"
            else FrequencyLinearAxis
        )
        resolved_frequency_axis = magnitude_frequency_axis.build(
            frequency_bins=frequency_bins_hz,
            )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis["freq_min"]))
            & (frequency_bins_hz <= float(resolved_frequency_axis["freq_max"]))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0
        tf_magnitude = hrtf.TF.magnitude
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

        magnitude_ax = figure.get_ax("bottom")
        if ear == "both":
            if magnitude_y_values.ndim < 2 or magnitude_y_values.shape[0] < 2:
                raise ValueError(
                    "Both ears requested but TF data does not contain two ear channels"
                )
            figure.create_two_dimension(
                ax=magnitude_ax,
                x=frequency_khz,
                y=magnitude_y_values[0, :],
                color="blue",
            )
            figure.create_two_dimension(
                ax=magnitude_ax,
                x=frequency_khz,
                y=magnitude_y_values[1, :],
                color="red",
            )
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
            figure.create_two_dimension(
                ax=magnitude_ax,
                x=frequency_khz,
                y=selected_magnitude_y_values,
                color="blue",
            )

        magnitude_legend_location = (
            "upper right" if magnitude_x_axis == "linear" else "upper left"
        )
        frequency_label = Labels.frequency
        magnitude_frequency_axis.apply(
            ax=magnitude_ax,
            axis="x",
            label=frequency_label,
            config=resolved_frequency_axis,
        )
        MagnitudeAxis.apply(ax=magnitude_ax, axis="y", unit=magnitude)
        if Figure.shared_x_visible:
            magnitude_ax.tick_params(axis="x", which="both", labelbottom=True)
        Ear.apply(ax=magnitude_ax, ear=ear, location=magnitude_legend_location, labels=None)
        magnitude_ax.grid(True)

        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                Titles.create_position_title(selected_positions=selected_positions),
            )
        if show:
            plt.show()
        return None

    def plot_spectrum_plane(
        hrtf: "HRTF",
        plane: str = "horizontal",
        elevation_angle: float = 0.0,
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = "max",
        colormap: str = "jet",
        freq_min: float | None = None,
        freq_max: float | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot a frequency-angle spectrum heatmap for an HRTF plane.

        The method selects a spatial plane from the current source grid and
        renders HRTF magnitude as a frequency-by-angle heatmap. Frequency is
        shown on the horizontal axis in kilohertz, while the vertical axis is
        either azimuth for horizontal planes or lateral-polar angle for the
        median plane.

        Horizontal planes are selected by the nearest available elevation to
        elevation_angle. Median-plane plots use the canonical sagittal path
        at azimuth zero. With ear=``both``, the method creates one heatmap for
        the left ear and one heatmap for the right ear using shared color
        limits.

        Parameters
        ----------
        plane : {``horizontal``, ``median``}, default=``horizontal``
            Plane to visualize. ``horizontal`` uses a horizontal plane
            selected by elevation. ``median`` uses the canonical median
            plane defined by the front-back sagittal path.
        elevation_angle : float, default=0.0
            Target elevation used when plane=``horizontal``. The nearest
            available horizontal plane in the grid is selected. This parameter
            is not used for the median plane.
        x_axis : {``linear``, ``log``}, default=``linear``
            Frequency scale used on the x axis.
        unit : {``db``, ``linear``}, default=``db``
            Magnitude representation used for the heatmap values.
        ear : {``left``, ``right``, ``both``}, default=``both``
            Ear channel to display. When ``both`` is selected, a separate subplot
            is created for each ear.
        reference : float | {``max``}, default=``max``
            Reference used when unit=``db``. ``max`` normalizes the plotted
            plane to its maximum value.
        colormap : str, default=``jet``
            Matplotlib colormap name used for the heatmap.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress generated default subplot and figure titles.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If plane, unit, x_axis, or ear is not one of the
            supported values, or if elevation_angle is not finite.
        ValueError
            If TF data is missing, elevation_angle is used with a non-
            horizontal plane, the selected plane has no positions, frequency
            bins are invalid, the selected frequency range contains no bins, or
            the requested ear channel is not available.

        Notes
        -----
        Horizontal-plane azimuths are displayed in the signed -180 .. 180
        convention. When unit=``db`` and reference=``max``, normalization
        is computed over the plotted plane and selected ear channels before
        conversion to decibels.
        """
        if plane not in ("horizontal", "median"):
            raise AttributeError(
                "plot_spectrum_plane plane accepts horizontal or median"
            )
        if isinstance(elevation_angle, bool):
            raise AttributeError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise AttributeError("elevation_angle must be a finite value")
        if unit not in {"db", "linear"}:
            raise AttributeError(
                "unit accepts : db or linear"
            )
        if x_axis not in {"log", "linear"}:
            raise AttributeError(
                "x_axis accepts log or linear"
            )
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        resolved_margins = Margins()
        azimuth_range_mode = "-180-180"
        heatmap_margin_ratio = 0.0

        if hrtf.TF.values is None or hrtf.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        plane_key = str(plane).strip().lower()
        if plane_key != "horizontal" and not np.isclose(
            elevation_angle,
            0.0,
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "elevation_angle only applies when plane='horizontal'"
            )
        if ear == "both":
            resolved_layout = Layout_2Horizontal(
                figsize=Layout_2Horizontal().figsize,
                margins=resolved_margins,
            )
        else:
            resolved_layout = Layout_1(
                figsize=Layout_1().figsize,
                margins=resolved_margins,
            )
        figure = Figure(resolved_layout)

        if plane_key == "horizontal":
            indices, real_plane_elevation = get_horizontal_plane(
                hrtf=hrtf,
                elevation=elevation_angle,
                angle_unit="degrees",
            )
        else:
            indices, _ = get_median_plane(
                hrtf=hrtf,
                azimuth=0.0,
                angle_unit="degrees",
            )
            real_plane_elevation = 0.0
        if indices.size == 0:
            raise ValueError("Selected plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=hrtf.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]

        if plane_key == "horizontal":
            plane_axis_values = np.asarray(spherical_positions[:, 0], dtype=float)
        else:
            lateral_polar_positions = spherical_to_lateral_polar(
                spherical_positions,
                angle_unit="degrees",
            )
            plane_axis_values = np.asarray(lateral_polar_positions[:, 1], dtype=float)

        frequency_bins_hz = np.asarray(hrtf.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        frequency_axis = (
            FrequencyLogAxis if x_axis == "log" else FrequencyLinearAxis
        )
        resolved_frequency_axis = frequency_axis.build(
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            margin_ratio=heatmap_margin_ratio,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis["freq_min"]))
            & (frequency_bins_hz <= float(resolved_frequency_axis["freq_max"]))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

        tf_magnitude = hrtf.TF.magnitude
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
            subplot_positions = ["left", "right"]
            default_subplot_titles = ["Left Ear", "Right Ear"]
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
            subplot_positions = ["main"]
            default_subplot_titles = [f"{ear.capitalize()} Ear"]

        vmin = min(float(np.min(matrix)) for matrix in spectrum_matrices)
        vmax = max(float(np.max(matrix)) for matrix in spectrum_matrices)
        colorbar_label = (
            Labels.magnitude_db if unit == "db" else Labels.magnitude_linear
        )
        for subplot_index, (subplot_position, spectrum_matrix, default_subplot_title) in enumerate(
            zip(subplot_positions, spectrum_matrices, default_subplot_titles)
        ):
            ax = figure.get_ax(subplot_position)
            subplot_plane_axis_values = (
                AzimuthAnglesAxis.transform_values(
                    values=plane_axis_values,
                    range_mode=azimuth_range_mode,
                )
                if plane_key == "horizontal"
                else np.asarray(plane_axis_values, dtype=float)
            )
            subplot_sort_indices = np.argsort(subplot_plane_axis_values)
            sorted_subplot_plane_axis_values = subplot_plane_axis_values[subplot_sort_indices]
            sorted_spectrum_matrix = spectrum_matrix[subplot_sort_indices, :]
            figure.create_heatmap(
                ax=ax,
                x=frequency_khz,
                y=sorted_subplot_plane_axis_values,
                values=sorted_spectrum_matrix,
                label=colorbar_label,
                colormap=colormap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.margins(x=0.0, y=0.0)
            frequency_label = Labels.frequency
            frequency_axis.apply(
                ax=ax,
                axis="x",
                label=frequency_label,
                config=resolved_frequency_axis,
            )
            if plane_key == "horizontal":
                AzimuthAnglesAxis.apply(
                    ax=ax,
                    axis="y",
                    values=sorted_subplot_plane_axis_values,
                    range_mode=azimuth_range_mode,
                )
            else:
                PolarAnglesAxis.apply(
                    ax=ax,
                    axis="y",
                    values=sorted_subplot_plane_axis_values,
                )
            resolved_title = default_subplot_title if titles else ""
            Titles.create_subplots_titles(ax=ax, title=resolved_title)
        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                Titles.create_plane_title(
                    plane=plane_key,
                    elevation_angle=real_plane_elevation,
                ),
            )
        if show:
            plt.show()
        return None

    def plot_elevation_spectrum(
        hrtf: "HRTF",
        azimuth: float | str = 0.0,
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = "max",
        colormap: str = "jet",
        freq_min: float | None = None,
        freq_max: float | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot a fixed-azimuth elevation spectrum heatmap.

        The method selects the nearest azimuth slice in the current source grid
        and renders HRTF magnitude as a frequency-by-elevation heatmap.
        Frequency is shown in kilohertz, elevation is shown in degrees, and
        the selected real azimuth is reported in the generated title when
        titles are enabled.

        Numeric azimuths are interpreted in degrees. Named position aliases use
        their spherical azimuth component, so ``front``, ``back``,
        ``left``, and ``right`` can be used for common vertical slices.
        With ear=``both``, the method creates one heatmap per ear using
        shared color limits.

        Parameters
        ----------
        azimuth : float | str, default=0.0
            Azimuth used to select the elevation slice. Named aliases such as
            ``front``, ``back``, ``left``, and ``right`` are accepted.
            The nearest available azimuth in the source grid is used.
        x_axis : {``linear``, ``log``}, default=``linear``
            Frequency scale used on the x axis.
        unit : {``db``, ``linear``}, default=``db``
            Magnitude representation used for the heatmap values.
        ear : {``left``, ``right``, ``both``}, default=``both``
            Ear channel to display. When ``both`` is selected, a separate subplot
            is created for each ear.
        reference : float | {``max``}, default=``max``
            Reference used when unit=``db``. ``max`` normalizes the plotted
            slice to its maximum value.
        colormap : str, default=``jet``
            Matplotlib colormap name used for the heatmap.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress generated default subplot and figure titles.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If unit, x_axis, or ear is not one of the supported
            values.
        ValueError
            If TF data is missing, azimuth is not finite or is an unknown
            named position, the selected slice has no positions, frequency bins
            are invalid, the selected frequency range contains no bins, or the
            requested ear channel is not available.

        Notes
        -----
        When the requested azimuth is not present exactly in the source grid,
        the nearest available azimuth is selected using circular angular
        distance. When unit=``db`` and reference=``max``, normalization is
        computed over the plotted slice and selected ear channels.
        """
        if unit not in {"db", "linear"}:
            raise AttributeError(
                "unit accepts : db or linear"
            )
        if x_axis not in {"log", "linear"}:
            raise AttributeError(
                "x_axis accepts log or linear"
            )
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        resolved_margins = Margins()
        heatmap_margin_ratio = 0.0

        if hrtf.TF.values is None or hrtf.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        if isinstance(azimuth, str):
            azimuth_key = str(azimuth).strip().lower()
            named_positions = get_named_positions(angle_unit="degrees")
            if azimuth_key not in named_positions:
                raise ValueError("azimuth accepts a finite value or: front, back, left, right")
            resolved_azimuth = float(named_positions[azimuth_key][0])
        else:
            if isinstance(azimuth, bool):
                raise ValueError("azimuth must be a finite value")
            resolved_azimuth = float(azimuth)
            if not np.isfinite(resolved_azimuth):
                raise ValueError("azimuth must be a finite value")

        if ear == "both":
            resolved_layout = Layout_2Horizontal(
                figsize=Layout_2Horizontal().figsize,
                margins=resolved_margins,
            )
        else:
            resolved_layout = Layout_1(
                figsize=Layout_1().figsize,
                margins=resolved_margins,
            )
        figure = Figure(resolved_layout)

        spherical_positions = get_source_positions(
            sources=hrtf.Sources,
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

        frequency_bins_hz = np.asarray(hrtf.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        frequency_axis = (
            FrequencyLogAxis if x_axis == "log" else FrequencyLinearAxis
        )
        resolved_frequency_axis = frequency_axis.build(
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            margin_ratio=heatmap_margin_ratio,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis["freq_min"]))
            & (frequency_bins_hz <= float(resolved_frequency_axis["freq_max"]))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

        tf_magnitude = hrtf.TF.magnitude
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
            subplot_positions = ["left", "right"]
            default_subplot_titles = ["Left Ear", "Right Ear"]
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
            subplot_positions = ["main"]
            default_subplot_titles = [f"{ear.capitalize()} Ear"]

        vmin = min(float(np.min(matrix)) for matrix in spectrum_matrices)
        vmax = max(float(np.max(matrix)) for matrix in spectrum_matrices)
        colorbar_label = (
            Labels.magnitude_db if unit == "db" else Labels.magnitude_linear
        )
        for subplot_index, (subplot_position, spectrum_matrix, default_subplot_title) in enumerate(
            zip(subplot_positions, spectrum_matrices, default_subplot_titles)
        ):
            ax = figure.get_ax(subplot_position)
            figure.create_heatmap(
                ax=ax,
                x=frequency_khz,
                y=sorted_elevation_values,
                values=spectrum_matrix,
                label=colorbar_label,
                colormap=colormap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.margins(x=0.0, y=0.0)
            frequency_label = Labels.frequency
            frequency_axis.apply(
                ax=ax,
                axis="x",
                label=frequency_label,
                config=resolved_frequency_axis,
            )
            ElevationAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_elevation_values,
            )
            resolved_title = default_subplot_title if titles else ""
            Titles.create_subplots_titles(ax=ax, title=resolved_title)
        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                Titles.create_elevation_spectrum_title(real_azimuth=real_azimuth),
            )
        if show:
            plt.show()
        return None

    def plot_itd_curve(
        hrtf: "HRTF",
        elevation_angle: float = 0.0,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot signed ITD over a horizontal plane as azimuth versus time delay.

        The method computes signed interaural time difference from the current
        HRIR data, selects the nearest horizontal plane to elevation_angle,
        and plots ITD in seconds against signed azimuth. The curve is sorted by
        azimuth so the plot follows the horizontal plane continuously.

        Parameters
        ----------
        elevation_angle : float, default=0.0
            Target elevation used to select the horizontal plane. The nearest
            available elevation in the grid is used.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress the generated default figure title.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If IR data or sample rate is missing, elevation_angle is not
            finite, the selected horizontal plane is empty, or the computed ITD
            values do not align with the number of source positions.

        Notes
        -----
        Azimuth is displayed in the signed -180 .. 180 convention, where
        positive azimuth values correspond to the left side and negative values
        correspond to the right side.
        """
        resolved_margins = Margins()
        azimuth_range_mode = "-180-180"

        if hrtf.IR.values is None:
            raise ValueError("IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")
        if isinstance(elevation_angle, bool):
            raise ValueError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise ValueError("elevation_angle must be a finite value")

        itd_values = np.asarray(
            itd(
                hrtf.IR,
                output="seconds",
            ),
            dtype=float,
        )
        if itd_values.ndim != 1:
            itd_values = itd_values.reshape(-1)
        indices, real_elevation = get_horizontal_plane(
            hrtf=hrtf,
            elevation=elevation_angle,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError("Selected horizontal plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=hrtf.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]
        azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
        transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
            values=azimuth_values,
            range_mode=azimuth_range_mode,
        )
        if itd_values.shape[0] != hrtf.Sources.get_positions(angle_unit="degrees").shape[0]:
            raise ValueError("ITD values must match the number of source positions")
        horizontal_itd_values = itd_values[indices]
        sort_indices = np.argsort(transformed_azimuth_values)
        sorted_azimuth_values = transformed_azimuth_values[sort_indices]
        sorted_itd_values = horizontal_itd_values[sort_indices]

        figure = Figure(
            Layout_1(
                figsize=Layout_1().figsize,
                margins=resolved_margins,
            )
        )
        ax = figure.get_ax("main")
        figure.create_two_dimension(
            ax=ax,
            x=sorted_azimuth_values,
            y=sorted_itd_values,
            color="steelblue",
            linewidth=2.0,
        )
        ax.margins(x=0.0)
        AzimuthAnglesAxis.apply(
            ax=ax,
            axis="x",
            values=sorted_azimuth_values,
            range_mode=azimuth_range_mode,
        )
        Axis.apply_label(
            ax=ax,
            axis="y",
            default_label=Labels.itd,
        )
        ax.grid(True)
        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                Titles.create_plane_title(
                    plane="horizontal",
                    elevation_angle=real_elevation,
                ),
            )
        if show:
            plt.show()
        return None

    def plot_absolute_itd(
        hrtf: "HRTF",
        elevation_angle: float = 0.0,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot absolute ITD over a horizontal plane in polar coordinates.

        The method computes absolute interaural time difference from the current
        HRIR data, selects the nearest horizontal plane to elevation_angle,
        and displays the resulting cue magnitude in a polar plot. Azimuth is
        represented on the angular axis and absolute ITD in seconds is
        represented on the radial axis.

        Parameters
        ----------
        elevation_angle : float, default=0.0
            Target elevation used to select the horizontal plane. The nearest
            available elevation in the grid is used.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress the generated default figure title.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If IR data or sample rate is missing, elevation_angle is not
            finite, or the selected horizontal plane cannot be resolved for the
            current source grid.

        Notes
        -----
        The polar azimuth axis uses 30-degree ticks with a north-up orientation.
        The radial label defaults to Labels.itd_seconds and radial ticks use
        the decimal-comma style configured by the polar-axis helper.
        """
        resolved_margins = Margins()
        polar_tick_step = 30.0
        radial_tick_step = 2e-4
        radial_tick_label_style = "decimal_comma_4"
        radial_label_position = 350.0
        polar_curve_color = "steelblue"
        polar_curve_linewidth = 2.0

        if hrtf.IR.values is None:
            raise ValueError("IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")
        if isinstance(elevation_angle, bool):
            raise ValueError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise ValueError("elevation_angle must be a finite value")

        itd_values = np.abs(
            np.asarray(
                itd(
                    hrtf.IR,
                    output="seconds",
                ),
                dtype=float,
            )
        )
        theta_values, radial_values, sorted_itd_values, real_elevation = create_horizontal_plane_curve(
            hrtf=hrtf,
            values=itd_values,
            elevation=elevation_angle,
        )

        figure = Figure(
            Layout_1(
                figsize=(6, 7),
                margins=resolved_margins,
            ),
            projection="polar",
        )
        ax = figure.get_ax("main")

        figure.create_two_dimension(
            ax=ax,
            x=theta_values,
            y=radial_values,
            color=polar_curve_color,
            linewidth=polar_curve_linewidth,
        )
        AzimuthAnglesAxisPolarProjection.apply(
            ax=ax,
            tick_step=polar_tick_step,
        )
        RadialAxisPolarProjection.apply(
            ax=ax,
            radial_values=sorted_itd_values,
            radial_label_default=Labels.itd_seconds,
            tick_step=radial_tick_step,
            tick_label_style=radial_tick_label_style,
            label_position=radial_label_position,
        )
        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                Titles.create_plane_title(
                    plane="horizontal",
                    elevation_angle=real_elevation,
                ),
            )
        ax.grid(True)
        if show:
            plt.show()
        return None

    def plot_ild_plane(
        hrtf: "HRTF",
        plane: str = "horizontal",
        elevation_angle: float = 0.0,
        colormap: str = "jet",
        freq_min: float | None = None,
        freq_max: float | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot a frequency-dependent ILD heatmap for an HRTF plane.

        The method computes frequency-dependent interaural level difference
        from the current HRIR data and renders it as a frequency-by-angle
        heatmap. Frequency is shown in kilohertz. The vertical axis is azimuth
        for horizontal planes and lateral-polar angle for the median plane.

        Horizontal planes are selected by the nearest available elevation to
        elevation_angle. Median-plane plots use the canonical sagittal path
        at azimuth zero. The ILD values are computed in decibels from the
        current impulse responses before plotting.

        Parameters
        ----------
        plane : {``horizontal``, ``median``}, default=``horizontal``
            Plane to visualize. ``horizontal`` uses a horizontal plane
            selected by elevation. ``median`` uses the canonical median
            plane defined by the front-back sagittal path.
        elevation_angle : float, default=0.0
            Target elevation used when plane=``horizontal``. The nearest
            available horizontal plane in the grid is selected. This parameter
            is not used for the median plane.
        colormap : str, default=``jet``
            Matplotlib colormap name used for the heatmap.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress the generated default figure title.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If plane is not ``horizontal`` or ``median``, or if
            elevation_angle is not finite.
        ValueError
            If IR data or sample rate is missing, elevation_angle is used
            with a non-horizontal plane, the selected plane has no positions,
            frequency bins are invalid, the selected frequency range contains
            no bins, or frequency-dependent ILD values do not have the expected
            (positions, frequencies) shape.

        Notes
        -----
        Horizontal-plane azimuths are displayed in the signed -180 .. 180
        convention. The frequency axis is always linear for this plot because
        the method currently builds a :class:`~hrtfpykit.plots.axis.FrequencyLinearAxis` configuration.
        """
        if plane not in ("horizontal", "median"):
            raise AttributeError("plot_ild_plane plane accepts horizontal or median")
        if isinstance(elevation_angle, bool):
            raise AttributeError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise AttributeError("elevation_angle must be a finite value")
        resolved_margins = Margins()
        azimuth_range_mode = "-180-180"
        heatmap_margin_ratio = 0.0

        if hrtf.IR.values is None:
            raise ValueError("IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")

        plane_key = str(plane).strip().lower()
        if plane_key != "horizontal" and not np.isclose(
            elevation_angle,
            0.0,
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError("elevation_angle only applies when plane='horizontal'")

        figure = Figure(
            Layout_1(
                figsize=Layout_1().figsize,
                margins=resolved_margins,
            )
        )

        if plane_key == "horizontal":
            indices, real_plane_elevation = get_horizontal_plane(
                hrtf=hrtf,
                elevation=elevation_angle,
                angle_unit="degrees",
            )
        else:
            indices, _ = get_median_plane(
                hrtf=hrtf,
                azimuth=0.0,
                angle_unit="degrees",
            )
            real_plane_elevation = 0.0
        if indices.size == 0:
            raise ValueError("Selected plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=hrtf.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]
        if plane_key == "horizontal":
            plane_axis_values = np.asarray(spherical_positions[:, 0], dtype=float)
        else:
            lateral_polar_positions = spherical_to_lateral_polar(
                spherical_positions,
                angle_unit="degrees",
            )
            plane_axis_values = np.asarray(lateral_polar_positions[:, 1], dtype=float)

        _, frequency_bins_hz, _ = tf_from_ir(
            np.asarray(hrtf.IR.values, dtype=float),
            sample_rate=hrtf.IR.sample_rate,
            fft_length=hrtf.fft_length,
        )
        frequency_bins_hz = np.asarray(frequency_bins_hz, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        resolved_frequency_axis = FrequencyLinearAxis.build(
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            margin_ratio=heatmap_margin_ratio,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis["freq_min"]))
            & (frequency_bins_hz <= float(resolved_frequency_axis["freq_max"]))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

        ild_values = np.asarray(
            ild(
                hrtf.IR,
                sample_rate=hrtf.IR.sample_rate,
                fft_length=hrtf.fft_length,
                mode="frequency-dependent",
                output="db",
            ),
            dtype=float,
        )
        plane_matrix = np.asarray(ild_values[indices][..., frequency_mask], dtype=float)
        if plane_matrix.ndim != 2:
            raise ValueError("Frequency-dependent ILD values must have shape (M, F)")

        ax = figure.get_ax("main")
        subplot_plane_axis_values = (
            AzimuthAnglesAxis.transform_values(
                values=plane_axis_values,
                range_mode=azimuth_range_mode,
            )
            if plane_key == "horizontal"
            else np.asarray(plane_axis_values, dtype=float)
        )
        subplot_sort_indices = np.argsort(subplot_plane_axis_values)
        sorted_subplot_plane_axis_values = subplot_plane_axis_values[subplot_sort_indices]
        sorted_plane_matrix = plane_matrix[subplot_sort_indices, :]
        figure.create_heatmap(
            ax=ax,
            x=frequency_khz,
            y=sorted_subplot_plane_axis_values,
            values=sorted_plane_matrix,
            label=Labels.ild,
            colormap=colormap,
            shading="auto",
            vmin=float(np.min(sorted_plane_matrix)),
            vmax=float(np.max(sorted_plane_matrix)),
        )
        ax.margins(x=0.0, y=0.0)
        frequency_label = Labels.frequency
        FrequencyLinearAxis.apply(
            ax=ax,
            axis="x",
            label=frequency_label,
            config=resolved_frequency_axis,
        )
        if plane_key == "horizontal":
            AzimuthAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_subplot_plane_axis_values,
                range_mode=azimuth_range_mode,
            )
        else:
            PolarAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_subplot_plane_axis_values,
            )
        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                Titles.create_plane_title(
                    plane=plane_key,
                    elevation_angle=real_plane_elevation,
                ),
            )
        if show:
            plt.show()
        return None

    def plot_ild_curve(
        hrtf: "HRTF",
        elevation_angle: float = 0.0,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot signed ILD over a horizontal plane as azimuth versus level difference.

        The method computes signed broad-band interaural level difference from
        the current HRIR data, selects the nearest horizontal plane to
        elevation_angle, and plots ILD in decibels against signed azimuth.
        The curve is sorted by azimuth so the plot follows the horizontal plane
        continuously.

        Parameters
        ----------
        elevation_angle : float, default=0.0
            Target elevation used to select the horizontal plane. The nearest
            available elevation in the grid is used.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress the generated default figure title.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If IR data or sample rate is missing, elevation_angle is not
            finite, the selected horizontal plane is empty, or the computed ILD
            values do not align with the number of source positions.

        Notes
        -----
        Azimuth is displayed in the signed -180 .. 180 convention, where
        positive azimuth values correspond to the left side and negative values
        correspond to the right side.
        """
        resolved_margins = Margins()
        azimuth_range_mode = "-180-180"

        if hrtf.IR.values is None:
            raise ValueError("IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")
        if isinstance(elevation_angle, bool):
            raise ValueError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise ValueError("elevation_angle must be a finite value")

        ild_values = np.asarray(
            ild(
                hrtf.IR,
                output="db",
                mode="broad-band",
            ),
            dtype=float,
        )
        if ild_values.ndim != 1:
            ild_values = ild_values.reshape(-1)
        indices, real_elevation = get_horizontal_plane(
            hrtf=hrtf,
            elevation=elevation_angle,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError("Selected horizontal plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=hrtf.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]
        azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
        transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
            values=azimuth_values,
            range_mode=azimuth_range_mode,
        )
        if ild_values.shape[0] != hrtf.Sources.get_positions(angle_unit="degrees").shape[0]:
            raise ValueError("ILD values must match the number of source positions")
        horizontal_ild_values = ild_values[indices]
        sort_indices = np.argsort(transformed_azimuth_values)
        sorted_azimuth_values = transformed_azimuth_values[sort_indices]
        sorted_ild_values = horizontal_ild_values[sort_indices]

        figure = Figure(
            Layout_1(
                figsize=Layout_1().figsize,
                margins=resolved_margins,
            )
        )
        ax = figure.get_ax("main")
        figure.create_two_dimension(
            ax=ax,
            x=sorted_azimuth_values,
            y=sorted_ild_values,
            color="steelblue",
            linewidth=2.0,
        )
        ax.margins(x=0.0)
        AzimuthAnglesAxis.apply(
            ax=ax,
            axis="x",
            values=sorted_azimuth_values,
            range_mode=azimuth_range_mode,
        )
        Axis.apply_label(
            ax=ax,
            axis="y",
            default_label=Labels.ild,
        )
        ax.grid(True)
        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                Titles.create_plane_title(
                    plane="horizontal",
                    elevation_angle=real_elevation,
                ),
            )
        if show:
            plt.show()
        return None

    def plot_absolute_ild(
        hrtf: "HRTF",
        elevation_angle: float = 0.0,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot absolute ILD over a horizontal plane in polar coordinates.

        The method computes absolute broad-band interaural level difference from
        the current HRIR data, selects the nearest horizontal plane to
        elevation_angle, and displays the resulting cue magnitude in a polar
        plot. Azimuth is represented on the angular axis and absolute ILD in
        decibels is represented on the radial axis.

        Parameters
        ----------
        elevation_angle : float, default=0.0
            Target elevation used to select the horizontal plane. The nearest
            available elevation in the grid is used.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress the generated default figure title.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If IR data or sample rate is missing, elevation_angle is not
            finite, or the selected horizontal plane cannot be resolved for the
            current source grid.

        Notes
        -----
        The polar azimuth axis uses 30-degree ticks with a north-up orientation.
        The radial label defaults to Labels.ild_db and radial ticks are
        formatted as integer decibel values.
        """
        resolved_margins = Margins()
        polar_tick_step = 30.0
        radial_tick_step = 5.0
        radial_tick_label_style = "integer"
        radial_label_position = 350.0
        polar_curve_color = "steelblue"
        polar_curve_linewidth = 2.0

        if hrtf.IR.values is None:
            raise ValueError("IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")
        if isinstance(elevation_angle, bool):
            raise ValueError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise ValueError("elevation_angle must be a finite value")

        ild_values = np.abs(
            np.asarray(
                ild(
                    hrtf.IR,
                    output="db",
                    mode="broad-band",
                ),
                dtype=float,
            )
        )
        theta_values, radial_values, sorted_ild_values, real_elevation = create_horizontal_plane_curve(
            hrtf=hrtf,
            values=ild_values,
            elevation=elevation_angle,
        )

        figure = Figure(
            Layout_1(
                figsize=(6, 7),
                margins=resolved_margins,
            ),
            projection="polar",
        )
        ax = figure.get_ax("main")

        figure.create_two_dimension(
            ax=ax,
            x=theta_values,
            y=radial_values,
            color=polar_curve_color,
            linewidth=polar_curve_linewidth,
        )
        AzimuthAnglesAxisPolarProjection.apply(
            ax=ax,
            tick_step=polar_tick_step,
        )
        RadialAxisPolarProjection.apply(
            ax=ax,
            radial_values=sorted_ild_values,
            radial_label_default=Labels.ild_db,
            tick_step=radial_tick_step,
            tick_label_style=radial_tick_label_style,
            label_position=radial_label_position,
        )
        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                Titles.create_plane_title(
                    plane="horizontal",
                    elevation_angle=real_elevation,
                ),
            )
        ax.grid(True)
        if show:
            plt.show()
        return None

    def plot_source_grid(
        hrtf: "HRTF",
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot the source grid as an interactive three-dimensional scatter.

        The method reads the current source positions from the HRTF instance,
        resolves them as Cartesian coordinates, and renders the grid in a 3D
        Matplotlib axis. Direction markers for front, right, and up are added
        to make the coordinate orientation clear in the default camera view.

        This plot is useful after loading a SOFA file, selecting a spatial
        subset, or transforming source coordinates because it visualizes the
        exact source grid currently attached to the object.

        Parameters
        ----------
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress the generated default figure title.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If source positions cannot be resolved as a valid Cartesian grid or
            axis geometry cannot be computed from the current positions.

        Notes
        -----
        The plot uses an equal-aspect 3D axis derived from the full Cartesian
        source extent. Axis labels use the library's 3D coordinate labels in
        meters.
        """
        resolved_margins = Margins()
        source_grid_scatter_size = 28.0
        source_grid_scatter_color = "steelblue"
        source_grid_scatter_edgecolors = "black"
        source_grid_scatter_linewidths = 0.4
        source_grid_scatter_depthshade = True

        cartesian_positions = get_source_positions(
            sources=hrtf.Sources,
            coordinate_system="cartesian",
            angle_unit="degrees",
        )
        figure = Figure(
            Layout_1(
                figsize=(6, 7),
                margins=resolved_margins,
            ),
            projection="3d",
        )
        ax = figure.get_ax("main")

        x_values = np.asarray(cartesian_positions[:, 0], dtype=float)
        y_values = np.asarray(cartesian_positions[:, 1], dtype=float)
        z_values = np.asarray(cartesian_positions[:, 2], dtype=float)
        figure.create_three_dimension(
            ax=ax,
            x=x_values,
            y=y_values,
            z=z_values,
            s=source_grid_scatter_size,
            color=source_grid_scatter_color,
            edgecolors=source_grid_scatter_edgecolors,
            linewidths=source_grid_scatter_linewidths,
            depthshade=source_grid_scatter_depthshade,
        )
        x_center, y_center, z_center, axis_half_span = resolve_three_dimensional_axis_geometry(
            cartesian_positions=cartesian_positions
        )
        XAxis.apply(
            ax=ax,
            center=x_center,
            half_span=axis_half_span,
            )
        YAxis.apply(
            ax=ax,
            center=y_center,
            half_span=axis_half_span,
            )
        ZAxis.apply(
            ax=ax,
            center=z_center,
            half_span=axis_half_span,
            )
        ax.set_box_aspect((1.0, 1.0, 1.0))
        create_sources_grid_direction_markers(
            ax=ax,
            sources=hrtf.Sources,
            axis_half_span=axis_half_span,
        )

        ax.grid(True)
        if titles:
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                "Source Grid",
            )
        if show:
            plt.show()
        return None

    def plot_plane_grid(
        hrtf: "HRTF",
        plane: str | list[str] | tuple[str, ...] = "horizontal",
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot the source grid and highlight canonical spatial planes in 3D.

        The full source grid is displayed as a light background scatter, while
        the selected canonical plane or planes are overlaid with stronger
        colors. Plane membership is resolved through the same plane-selection
        helpers used by processing and metric plots, so the highlighted points
        match the library's canonical horizontal, median, and frontal plane
        definitions.

        Parameters
        ----------
        plane : str | list[str] | tuple[str, ...], default=``horizontal``
            Plane or planes to highlight. Accepted values are ``horizontal``,
            ``median``, and ``frontal``. A single string highlights one
            plane, while a list or tuple highlights multiple planes in the same
            figure.
        show : bool, default=True
            If True, call matplotlib.pyplot.show() before returning.
        titles : bool, default=True
            If False, suppress the generated default figure title.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If plane is empty, contains an unsupported plane name, source
            positions cannot be resolved as Cartesian coordinates, or axis
            geometry cannot be computed from the current source grid.

        Notes
        -----
        Duplicate plane names are ignored after the first occurrence. A legend
        is added to distinguish the background source grid from each highlighted
        plane.
        """
        resolved_margins = Margins()
        source_grid_scatter_size = 18.0
        source_grid_scatter_color = "#9ecae1"
        source_grid_scatter_edgecolors = "none"
        source_grid_scatter_depthshade = True
        source_grid_scatter_alpha = 0.55
        source_grid_scatter_label = "Source Grid"
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

        cartesian_positions = get_source_positions(
            sources=hrtf.Sources,
            coordinate_system="cartesian",
            angle_unit="degrees",
        )
        figure = Figure(
            Layout_1(
                figsize=(6, 7),
                margins=resolved_margins,
            ),
            projection="3d",
        )
        ax = figure.get_ax("main")

        x_values = np.asarray(cartesian_positions[:, 0], dtype=float)
        y_values = np.asarray(cartesian_positions[:, 1], dtype=float)
        z_values = np.asarray(cartesian_positions[:, 2], dtype=float)

        figure.create_three_dimension(
            ax=ax,
            x=x_values,
            y=y_values,
            z=z_values,
            s=source_grid_scatter_size,
            color=source_grid_scatter_color,
            edgecolors=source_grid_scatter_edgecolors,
            depthshade=source_grid_scatter_depthshade,
            alpha=source_grid_scatter_alpha,
            label=source_grid_scatter_label,
        )

        plane_colors = {
            "horizontal": "green",
            "median": "red",
            "frontal": "blue",
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
                indices, _ = get_horizontal_plane(
                    hrtf=hrtf,
                    elevation=0.0,
                    angle_unit="degrees",
                )
            elif plane_key == "median":
                indices, _ = get_median_plane(
                    hrtf=hrtf,
                    azimuth=0.0,
                    angle_unit="degrees",
                )
            else:
                indices, _ = get_frontal_plane(
                    hrtf=hrtf,
                    azimuth=90.0,
                    angle_unit="degrees",
                )
            plane_positions = np.asarray(cartesian_positions[indices], dtype=float)
            figure.create_three_dimension(
                ax=ax,
                x=plane_positions[:, 0],
                y=plane_positions[:, 1],
                z=plane_positions[:, 2],
                s=34.0,
                color=plane_colors[plane_key],
                edgecolors="black",
                linewidths=0.35,
                depthshade=True,
                label=plane_labels[plane_key],
            )

        x_center, y_center, z_center, axis_half_span = resolve_three_dimensional_axis_geometry(
            cartesian_positions=cartesian_positions
        )
        XAxis.apply(
            ax=ax,
            center=x_center,
            half_span=axis_half_span,
            )
        YAxis.apply(
            ax=ax,
            center=y_center,
            half_span=axis_half_span,
            )
        ZAxis.apply(
            ax=ax,
            center=z_center,
            half_span=axis_half_span,
            )
        ax.set_box_aspect((1.0, 1.0, 1.0))
        create_sources_grid_direction_markers(
            ax=ax,
            sources=hrtf.Sources,
            axis_half_span=axis_half_span,
        )

        ax.grid(True)
        ax.legend(loc="upper right")
        if titles:
            if len(resolved_planes) == 1:
                resolved_figure_title = plane_titles[resolved_planes[0]]
            else:
                resolved_figure_title = "Plane Grid"
            Titles.create_figure_title(
                figure.fig,
                figure.axes,
                figure.figure_title_y,
                resolved_figure_title,
            )
        if show:
            plt.show()
        return None
