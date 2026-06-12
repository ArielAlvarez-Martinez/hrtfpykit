from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
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
from ..utils.coordinates import (
    get_named_positions,
    get_position_queries,
    get_source_positions,
    spherical_to_lateral_polar,
)
from ..utils.dsp import magnitude_to_db
from ..utils.metrics import ild, itd
from ..utils.planes import (
    get_frontal_plane,
    get_horizontal_plane,
    get_median_plane,
)

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF

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
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot magnitude responses for selected source positions.

    ``plot_magnitude`` reads ``hrtf.TF.magnitude`` and ``hrtf.TF.frequency_bins``, resolves
    each requested source with ``hrtf.Sources.get_position_index(...)``, and draws one
    spectrum panel per source direction. When ``unit="db"``, the plotted values are produced
    to decibels; ``reference="max"`` uses the maximum selected magnitude as the dB
    reference.

    Frequency limits are passed in hertz and displayed in kilohertz. Position queries use
    named directions such as ``"front"`` or spherical ``[azimuth, elevation]`` coordinates
    in degrees.

    Parameters
    ----------
    positions : str | list | tuple | np.ndarray, default=(``front``, ``back``, ``left``, ``right``)
        One position or a collection of positions. Named aliases such as ``front``,
        ``back``, ``left``, and ``right`` are accepted. Numeric queries must use
        spherical coordinates in degrees as [azimuth, elevation], for example [0.0, 0.0]
        for the front direction. Up to four positions can be shown in one figure.
    x_axis : {``linear``, ``log``}, default=``linear``
        Frequency scale used on the x axis.
    unit : {``db``, ``linear``}, default=``db``
        Magnitude representation used on the y axis.
    ear : {``left``, ``right``, ``both``}, default=``both``
        Ear channel to display. When ``both`` is selected, left and right responses are
        drawn together in each subplot.
    reference : float | {``max``}, default=1.0
        Reference used when unit=``db``. ``max`` normalizes the plotted magnitude to the
        maximum selected value.
    freq_min : float | None, default=None
        Minimum frequency in Hz included in the plot.
    freq_max : float | None, default=None
        Maximum frequency in Hz included in the plot.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default subplot titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the selected HRTF magnitude response panels.

    Raises
    ------
    AttributeError
        If unit, x_axis, or ear is not one of the supported values.
    ValueError
        If TF values or frequency bins are missing, no positions are requested, more
        than four positions are requested, the selected frequency range contains no
        bins, or the requested ear channel is not available.

    Notes
    -----
    The figure uses one panel for one requested source, two stacked panels for two requested
    sources, and a compact grid for three or four requested sources. With ``ear="both"``,
    both ear channels are drawn in each panel with the shared ear legend.

    Examples
    --------
    Plot one normalized magnitude response for the front direction on a logarithmic
    frequency axis:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_magnitude
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_magnitude(
    ...     hrtf,
    ...     positions="front",
    ...     x_axis="log",
    ...     ear="both",
    ...     reference="max",
    ...     freq_max=16000.0,
    ... )
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

    resolved_layout: Any
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
            (frequency_bins_hz >= float(cast(Any, resolved_frequency_axis["freq_min"])))
            & (frequency_bins_hz <= float(cast(Any, resolved_frequency_axis["freq_max"])))
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
            label=frequency_label if show_labels else "",
            config=resolved_frequency_axis,
        )
        MagnitudeAxis.apply(ax=ax, axis="y", unit=unit, label=None if show_labels else "")
        default_subplot_title = Titles.create_position_title(
            selected_positions=selected_positions,
        )
        resolved_subplot_title = default_subplot_title if show_titles else ""
        if show_titles:
            Titles.create_subplots_titles(ax=ax, title=resolved_subplot_title)
        if Figure.shared_x_visible:
            ax.tick_params(axis="x", which="both", labelbottom=True)
        if show_legends:
            Ear.apply(ax=ax, ear=ear, location=magnitude_legend_location, labels=None)
        ax.grid(True)

    if position_count < figure.axes.size:
        figure.hide_unused_axes(position_count)

    if show:
        plt.show()
    return figure.fig

def plot_amplitude(
    hrtf: "HRTF",
    positions: str | list | tuple | np.ndarray = ("front", "back", "left", "right"),
    ear: str = "both",
    x_axis: str = "time",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot HRIR amplitude responses for selected source positions.

    ``plot_amplitude`` reads the current time domain data from ``hrtf.IR.values``, resolves
    each requested source with ``hrtf.Sources.get_position_index(...)``, and draws the
    selected impulse responses. With ``ear="both"``, left and right channels are overlaid in
    each source panel.

    The x axis can show sample indices or time in milliseconds. Time mode uses
    ``hrtf.IR.sample_rate`` to convert sample positions to milliseconds.

    Parameters
    ----------
    positions : str | list | tuple | np.ndarray, default=(``front``, ``back``, ``left``, ``right``)
        One position or a collection of positions. Named aliases such as ``front``,
        ``back``, ``left``, and ``right`` are accepted. Numeric queries must use
        spherical coordinates in degrees as [azimuth, elevation], for example [0.0, 0.0]
        for the front direction. Up to four positions can be shown in one figure.
    ear : {``left``, ``right``, ``both``}, default=``both``
        Ear channel to display. When ``both`` is selected, left and right ear waveforms
        are drawn together in each subplot.
    x_axis : {``time``, ``samples``}, default=``time``
        Horizontal axis used for the waveform plot.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default subplot titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the selected HRIR amplitude response panels.

    Raises
    ------
    AttributeError
        If ear or x_axis is not one of the supported values.
    ValueError
        If IR data is missing, time axis plotting lacks a sample rate, no positions are
        requested, more than four positions are requested, the IR array has no samples,
        or the requested ear channel is unavailable.

    Notes
    -----
    The figure uses one panel for one requested source, two stacked panels for two requested
    sources, and a compact grid for three or four requested sources. With ``ear="both"``,
    both ear channels are drawn in each panel with the shared ear legend.

    Examples
    --------
    Plot one front HRIR waveform:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_amplitude
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_amplitude(
    ...     hrtf,
    ...     positions="front",
    ...     ear="both",
    ...     x_axis="samples",
    ... )
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

    resolved_layout: Any
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
        x_values = 1000.0 * sample_indexes / float(cast(Any, hrtf.IR.sample_rate))
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
            TimeAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
        else:
            SampleAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
        AmplitudeAxis.apply(ax=ax, axis="y", label=None if show_labels else "")
        default_subplot_title = Titles.create_position_title(
            selected_positions=selected_positions,
        )
        resolved_subplot_title = default_subplot_title if show_titles else ""
        if show_titles:
            Titles.create_subplots_titles(ax=ax, title=resolved_subplot_title)
        if Figure.shared_x_visible:
            ax.tick_params(axis="x", which="both", labelbottom=True)
        if show_legends:
            Ear.apply(ax=ax, ear=ear, location="upper right", labels=None)
        ax.grid(True)

    if position_count < figure.axes.size:
        figure.hide_unused_axes(position_count)

    if show:
        plt.show()
    return figure.fig

def plot_etc(
    hrtf: "HRTF",
    positions: str | list | tuple | np.ndarray = ("front", "back", "left", "right"),
    ear: str = "both",
    x_axis: str = "time",
    reference: float | str = "max",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot energy time curves for selected source positions.

    ``plot_etc`` reads ``hrtf.IR.values``, resolves each requested source with
    ``hrtf.Sources.get_position_index(...)``, and converts ``abs(IR)`` to decibels. The
    result is an energy time view of the selected HRIR samples.

    The x axis can show sample indices or time in milliseconds. Time mode uses
    ``hrtf.IR.sample_rate`` to convert sample positions to milliseconds.

    Parameters
    ----------
    positions : str | list | tuple | np.ndarray, default=(``front``, ``back``, ``left``, ``right``)
        One position or a collection of positions. Named aliases such as ``front``,
        ``back``, ``left``, and ``right`` are accepted. Numeric queries must use
        spherical coordinates in degrees as [azimuth, elevation], for example [0.0, 0.0]
        for the front direction. Up to four positions can be shown in one figure.
    ear : {``left``, ``right``, ``both``}, default=``both``
        Ear channel to display. When ``both`` is selected, left and right ear ETC traces
        are drawn together in each subplot.
    x_axis : {``time``, ``samples``}, default=``time``
        Horizontal axis used for the ETC plot.
    reference : float | {``max``}, default=``max``
        Reference used for decibel conversion. ``max`` normalizes the selected ETC
        traces to their maximum absolute IR value.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default subplot titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the selected energy time curve panels.

    Raises
    ------
    AttributeError
        If ear or x_axis is not one of the supported values.
    ValueError
        If IR data is missing, time axis plotting lacks a sample rate, no positions are
        requested, more than four positions are requested, the IR array has no samples,
        the reference is invalid, or the requested ear channel is unavailable.

    Notes
    -----
    The figure uses the same source selection and panel arrangement as
    :func:`~hrtfpykit.plots.plot_amplitude`. The y values differ because the selected HRIR
    samples are converted to dB before plotting.

    Examples
    --------
    Plot the front direction ETC for both ears using sample indices:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_etc
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_etc(
    ...     hrtf,
    ...     positions="front",
    ...     ear="both",
    ...     x_axis="samples",
    ...     reference="max",
    ... )
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
        raise ValueError("plot_etc accepts up to 4 positions")

    resolved_layout: Any
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
        x_values = 1000.0 * sample_indexes / float(cast(Any, hrtf.IR.sample_rate))
    else:
        x_values = sample_indexes

    selected_position_info = [
        hrtf.Sources.get_position_index(
            selected_position_query,
            coordinate_system="spherical",
        )
        for selected_position_query in position_queries
    ]
    selected_indices = [selected_index for selected_index, _ in selected_position_info]
    reference_values = np.abs(np.asarray(ir_values[selected_indices], dtype=float))
    if ear != "both" and reference_values.ndim >= 3:
        ear_index = 0 if ear == "left" else 1
        if reference_values.shape[1] <= ear_index:
            raise ValueError(f"Requested ear '{ear}' is not available in IR data")
        reference_values = reference_values[:, ear_index, :]
    plot_reference: float | str
    if isinstance(reference, str) and str(reference).strip().lower() == "max":
        plot_reference = float(np.max(reference_values))
    else:
        plot_reference = reference

    for index, (idxs, selected_positions) in enumerate(selected_position_info):
        ax = figure.get_ax(index)
        selected_positions = np.asarray(selected_positions, dtype=float)
        y_values = np.asarray(ir_values[int(idxs)], dtype=float)
        etc_values = magnitude_to_db(np.abs(y_values), reference=plot_reference)

        if ear == "both":
            if etc_values.ndim < 2 or etc_values.shape[0] < 2:
                raise ValueError("Both ears requested but IR data does not contain two ear channels")
            figure.create_two_dimension(
                ax=ax,
                x=x_values,
                y=etc_values[0, :],
                color="blue",
            )
            figure.create_two_dimension(
                ax=ax,
                x=x_values,
                y=etc_values[1, :],
                color="red",
            )
        else:
            if etc_values.ndim == 1:
                selected_y_values = etc_values.reshape(-1)
            else:
                ear_index = 0 if ear == "left" else 1
                if etc_values.shape[0] <= ear_index:
                    raise ValueError(f"Requested ear '{ear}' is not available in IR data")
                selected_y_values = np.asarray(etc_values[ear_index], dtype=float).reshape(-1)
            figure.create_two_dimension(
                ax=ax,
                x=x_values,
                y=selected_y_values,
                color="blue",
            )

        if x_axis == "time":
            TimeAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
        else:
            SampleAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
        Axis.apply_label(ax=ax, axis="y", default_label=Labels.energy_db if show_labels else "", label=None if show_labels else "")
        default_subplot_title = Titles.create_position_title(
            selected_positions=selected_positions,
        )
        resolved_subplot_title = default_subplot_title if show_titles else ""
        if show_titles:
            Titles.create_subplots_titles(ax=ax, title=resolved_subplot_title)
        if Figure.shared_x_visible:
            ax.tick_params(axis="x", which="both", labelbottom=True)
        if show_legends:
            Ear.apply(ax=ax, ear=ear, location="upper right", labels=None)
        ax.grid(True)

    if position_count < figure.axes.size:
        figure.hide_unused_axes(position_count)

    if show:
        plt.show()
    return figure.fig

def plot_etc_plane(
    hrtf: "HRTF",
    plane: str = "horizontal",
    plane_angle: float = 0.0,
    azimuth_range_mode: str = "0-360",
    ear: str = "both",
    x_axis: str = "time",
    reference: float | str = "max",
    colormap: str = "jet",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot energy time curves across a measured spatial plane.

    ``plot_etc_plane`` selects the nearest measured horizontal or median plane from
    ``hrtf.Sources``. It reads the selected samples from ``hrtf.IR.values``, converts
    ``abs(IR)`` to dB, and draws a direction by time heatmap.

    Horizontal planes use spherical azimuth on the y axis. Median planes use lateral polar
    angle on the y axis, where 0 degrees is front, 90 degrees is up, 180 degrees is back,
    and -90 degrees is down.

    Parameters
    ----------
    plane : {``horizontal``, ``median``}, default=``horizontal``
        Plane to visualize. ``horizontal`` uses a horizontal plane selected by spherical
        elevation. ``median`` uses the nearest measured lateral polar lateral angle.
    plane_angle : float, default=0.0
        Plane coordinate in degrees used to resolve the nearest measured plane. For
        ``plane="horizontal"`` this is spherical elevation. For ``plane="median"`` this
        is lateral polar lateral angle.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention used for horizontal plane azimuth values. ``0-360`` follows
        the SOFA anticlockwise convention: front is 0 degrees, listener left is 90
        degrees, back is 180 degrees, and listener right is 270 degrees. ``-180-180``
        wraps listener right to -90 degrees. Because horizontal plane azimuth is drawn
        on the y axis in this plot, signed azimuth keeps normal Matplotlib numeric
        orientation.
    ear : {``left``, ``right``, ``both``}, default=``both``
        Ear channel to display. When ``both`` is selected, a separate heatmap is created
        for each ear.
    x_axis : {``time``, ``samples``}, default=``time``
        Horizontal axis used for the ETC heatmap.
    reference : float | {``max``}, default=``max``
        Reference used for decibel conversion. ``max`` normalizes the plotted plane to
        its maximum absolute IR value over the selected plane and ear channels.
    colormap : str, default=``jet``
        Colormap name used for the heatmap.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default subplot and figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the selected plane energy time heatmap.

    Raises
    ------
    AttributeError
        If plane, ear, or x_axis is not one of the supported values, or if plane_angle
        is not finite.
    ValueError
        If IR data is missing, time axis plotting lacks a sample rate, the azimuth range
        is invalid, the selected plane has no positions, the IR array has no samples,
        the reference is invalid, or the requested ear channel is unavailable.

    Notes
    -----
    This is the plane view counterpart of :func:`~hrtfpykit.plots.plot_etc`. It keeps the
    same dB reference behavior and shows all measured source positions in the selected plane
    at once.

    Examples
    --------
    Plot a horizontal plane ETC heatmap for both ears using sample indices:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_etc_plane
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_etc_plane(
    ...     hrtf,
    ...     plane="horizontal",
    ...     plane_angle=0.0,
    ...     ear="both",
    ...     x_axis="samples",
    ...     reference="max",
    ... )
    """
    if plane not in ("horizontal", "median"):
        raise AttributeError("plot_etc_plane plane accepts horizontal or median")
    if ear not in {"left", "right", "both"}:
        raise AttributeError("ear accepts left, right or both")
    if x_axis not in {"time", "samples"}:
        raise AttributeError("x_axis accepts : time or samples")
    if isinstance(plane_angle, bool):
        raise AttributeError("plane_angle must be a finite value")
    plane_angle = float(plane_angle)
    if not np.isfinite(plane_angle):
        raise AttributeError("plane_angle must be a finite value")

    if hrtf.IR.values is None:
        raise ValueError("IR data is not available")
    if x_axis == "time" and hrtf.IR.sample_rate is None:
        raise ValueError("IR sample_rate is required when x_axis='time'")

    azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=azimuth_range_mode)
    resolved_margins = Margins()
    plane_key = str(plane).strip().lower()
    resolved_layout: Any
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
        indices, real_plane_angle = get_horizontal_plane(
            hrtf=hrtf,
            plane_angle=plane_angle,
            angle_unit="degrees",
        )
    else:
        indices, real_plane_angle = get_median_plane(
            hrtf=hrtf,
            plane_angle=plane_angle,
            angle_unit="degrees",
        )
    if indices.size == 0:
        raise ValueError("Selected plane does not contain any source positions")

    ir_values = np.asarray(hrtf.IR.values, dtype=float)
    if ir_values.ndim < 2 or ir_values.shape[-1] == 0:
        raise ValueError("IR values must contain at least one sample")
    sample_indexes = np.arange(ir_values.shape[-1], dtype=float)
    if x_axis == "time":
        x_values = 1000.0 * sample_indexes / float(cast(Any, hrtf.IR.sample_rate))
    else:
        x_values = sample_indexes

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

    plane_values = np.abs(np.asarray(ir_values[indices], dtype=float))
    if plane_values.ndim == 2:
        plane_values = plane_values[:, np.newaxis, :]
    if plane_values.ndim != 3:
        raise ValueError("IR values for ETC plane must have shape (M, E, N)")
    reference_values = plane_values
    if ear != "both":
        ear_index = 0 if ear == "left" else 1
        if reference_values.shape[1] <= ear_index:
            raise ValueError(f"Requested ear '{ear}' is not available in IR data")
        reference_values = reference_values[:, ear_index, :]
    plot_reference: float | str
    if isinstance(reference, str) and str(reference).strip().lower() == "max":
        plot_reference = float(np.max(reference_values))
    else:
        plot_reference = reference
    etc_values = magnitude_to_db(plane_values, reference=plot_reference)

    if ear == "both":
        if etc_values.shape[1] < 2:
            raise ValueError("Both ears requested but IR data does not contain two ear channels")
        etc_matrices = [etc_values[:, 0, :], etc_values[:, 1, :]]
        subplot_positions = ["left", "right"]
        default_subplot_titles = ["Left Ear", "Right Ear"]
    else:
        if etc_values.shape[1] == 1:
            ear_index = 0
        else:
            ear_index = 0 if ear == "left" else 1
            if etc_values.shape[1] <= ear_index:
                raise ValueError(f"Requested ear '{ear}' is not available in IR data")
        etc_matrices = [etc_values[:, ear_index, :]]
        subplot_positions = ["main"]
        default_subplot_titles = [f"{ear.capitalize()} Ear"]

    finite_values = np.asarray(etc_values, dtype=float)[np.isfinite(etc_values)]
    if finite_values.size == 0:
        raise ValueError("No finite ETC values available for heatmap rendering")
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))

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

    for subplot_position, etc_matrix, default_subplot_title in zip(
        subplot_positions,
        etc_matrices,
        default_subplot_titles,
    ):
        ax = figure.get_ax(subplot_position)
        sorted_etc_matrix = etc_matrix[subplot_sort_indices, :]
        figure.create_heatmap(
            ax=ax,
            x=x_values,
            y=sorted_subplot_plane_axis_values,
            values=sorted_etc_matrix,
            label=Labels.energy_db if show_labels else "",
            colormap=colormap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.margins(x=0.0, y=0.0)
        if x_axis == "time":
            TimeAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
        else:
            SampleAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
        if plane_key == "horizontal":
            AzimuthAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_subplot_plane_axis_values,
                range_mode=azimuth_range_mode,
                label=None if show_labels else "",
            )
        else:
            PolarAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_subplot_plane_axis_values,
                label=None if show_labels else "",
            )
        resolved_title = default_subplot_title if show_titles else ""
        if show_titles:
            Titles.create_subplots_titles(ax=ax, title=resolved_title)

    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane=plane_key,
                plane_angle=real_plane_angle,
            ),
        )
    if show:
        plt.show()
    return figure.fig

def plot_spectrum_plane(
    hrtf: "HRTF",
    plane: str = "horizontal",
    plane_angle: float = 0.0,
    azimuth_range_mode: str = "0-360",
    x_axis: str = "linear",
    unit: str = "db",
    ear: str = "both",
    reference: float | str = "max",
    colormap: str = "jet",
    freq_min: float | None = None,
    freq_max: float | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot HRTF magnitude across a measured spatial plane.

    ``plot_spectrum_plane`` selects the nearest measured horizontal or median plane from
    ``hrtf.Sources``. It reads ``hrtf.TF.magnitude`` and ``hrtf.TF.frequency_bins``,
    optionally converts magnitude to dB to decibels, and draws a frequency by angle heatmap.

    Horizontal planes use spherical azimuth on the y axis. Median planes use lateral polar
    angle on the y axis, where 0 degrees is front, 90 degrees is up, 180 degrees is back,
    and -90 degrees is down.

    Parameters
    ----------
    plane : {``horizontal``, ``median``}, default=``horizontal``
        Plane to visualize. ``horizontal`` uses a horizontal plane selected by spherical
        elevation. ``median`` uses the nearest measured lateral polar lateral angle.
    plane_angle : float, default=0.0
        Plane coordinate in degrees used to resolve the nearest measured plane. For
        ``plane="horizontal"`` this is spherical elevation. For ``plane="median"`` this
        is lateral polar lateral angle.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention used for horizontal plane azimuth values. ``0-360`` follows
        the SOFA anticlockwise convention: front is 0 degrees, listener left is 90
        degrees, back is 180 degrees, and listener right is 270 degrees. ``-180-180``
        wraps listener right to -90 degrees. Because horizontal plane azimuth is drawn
        on the y axis in this plot, signed azimuth keeps normal Matplotlib numeric
        orientation.
    x_axis : {``linear``, ``log``}, default=``linear``
        Frequency scale used on the x axis.
    unit : {``db``, ``linear``}, default=``db``
        Magnitude representation used for the heatmap values.
    ear : {``left``, ``right``, ``both``}, default=``both``
        Ear channel to display. When ``both`` is selected, a separate subplot is created
        for each ear.
    reference : float | {``max``}, default=``max``
        Reference used when unit=``db``. ``max`` normalizes the plotted plane to its
        maximum value.
    colormap : str, default=``jet``
        Colormap name used for the heatmap.
    freq_min : float | None, default=None
        Minimum frequency in Hz included in the plot.
    freq_max : float | None, default=None
        Maximum frequency in Hz included in the plot.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default subplot and figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the selected plane magnitude heatmap.

    Raises
    ------
    AttributeError
        If plane, unit, x_axis, or ear is not one of the supported values, or if
        plane_angle is not finite.
    ValueError
        If TF data is missing, the azimuth range is invalid, the selected plane has no
        positions, frequency bins are invalid, the selected frequency range contains no
        bins, or the requested ear channel is not available.

    Notes
    -----
    Horizontal plane azimuth follows ``azimuth_range_mode`` on the y axis. In signed mode,
    +90 degrees remains above -90 degrees because hrtfpykit only reverses the visual axis
    when azimuth is plotted on the x axis.

    Examples
    --------
    Plot a horizontal plane spectrum heatmap around ear height:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_spectrum_plane
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_spectrum_plane(
    ...     hrtf,
    ...     plane="horizontal",
    ...     plane_angle=0.0,
    ...     x_axis="linear",
    ...     ear="left",
    ...     freq_max=16000.0,
    ... )
    """
    if plane not in ("horizontal", "median"):
        raise AttributeError(
            "plot_spectrum_plane plane accepts horizontal or median"
        )
    if isinstance(plane_angle, bool):
        raise AttributeError("plane_angle must be a finite value")
    plane_angle = float(plane_angle)
    if not np.isfinite(plane_angle):
        raise AttributeError("plane_angle must be a finite value")
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
    azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=azimuth_range_mode)
    resolved_margins = Margins()
    heatmap_margin_ratio = 0.0

    if hrtf.TF.values is None or hrtf.TF.frequency_bins is None:
        raise ValueError("TF data is not available")

    plane_key = str(plane).strip().lower()
    resolved_layout: Any
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
        indices, real_plane_angle = get_horizontal_plane(
            hrtf=hrtf,
            plane_angle=plane_angle,
            angle_unit="degrees",
        )
    else:
        indices, real_plane_angle = get_median_plane(
            hrtf=hrtf,
            plane_angle=plane_angle,
            angle_unit="degrees",
        )

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
        (frequency_bins_hz >= float(cast(Any, resolved_frequency_axis["freq_min"])))
        & (frequency_bins_hz <= float(cast(Any, resolved_frequency_axis["freq_max"])))
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
            label=colorbar_label if show_labels else "",
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
            label=frequency_label if show_labels else "",
            config=resolved_frequency_axis,
        )
        if plane_key == "horizontal":
            AzimuthAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_subplot_plane_axis_values,
                range_mode=azimuth_range_mode,
                label=None if show_labels else "",
            )
        else:
            PolarAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_subplot_plane_axis_values,
                label=None if show_labels else "",
            )
        resolved_title = default_subplot_title if show_titles else ""
        if show_titles:
            Titles.create_subplots_titles(ax=ax, title=resolved_title)
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane=plane_key,
                plane_angle=real_plane_angle,
            ),
        )
    if show:
        plt.show()
    return figure.fig

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
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot HRTF magnitude across elevation at one azimuth.

    ``plot_elevation_spectrum`` resolves the requested azimuth from a numeric angle or a
    named direction, selects the nearest measured azimuth slice in ``hrtf.Sources``, and
    plots ``hrtf.TF.magnitude`` against frequency and elevation. When ``unit="db"``,
    magnitude values are converted to decibels.

    The selected real azimuth is used in the figure title when titles are enabled. With
    ``ear="both"``, the function creates separate left and right panels with shared color
    limits.

    Parameters
    ----------
    azimuth : float | str, default=0.0
        Azimuth used to select the elevation slice. Named aliases such as ``front``,
        ``back``, ``left``, and ``right`` are accepted. The nearest available azimuth in
        the source grid is used.
    x_axis : {``linear``, ``log``}, default=``linear``
        Frequency scale used on the x axis.
    unit : {``db``, ``linear``}, default=``db``
        Magnitude representation used for the heatmap values.
    ear : {``left``, ``right``, ``both``}, default=``both``
        Ear channel to display. When ``both`` is selected, a separate subplot is created
        for each ear.
    reference : float | {``max``}, default=``max``
        Reference used when unit=``db``. ``max`` normalizes the plotted slice to its
        maximum value.
    colormap : str, default=``jet``
        Colormap name used for the heatmap.
    freq_min : float | None, default=None
        Minimum frequency in Hz included in the plot.
    freq_max : float | None, default=None
        Maximum frequency in Hz included in the plot.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default subplot and figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the selected elevation spectrum heatmap.

    Raises
    ------
    AttributeError
        If unit, x_axis, or ear is not one of the supported values.
    ValueError
        If TF data is missing, azimuth is not finite or is an unknown named position,
        the selected slice has no positions, frequency bins are invalid, the selected
        frequency range contains no bins, or the requested ear channel is not available.

    Notes
    -----
    When the requested azimuth is not present exactly in the source grid, the nearest
    available azimuth is selected by circular angular distance. When ``unit="db"`` and
    ``reference="max"``, normalization is computed from the selected slice and ear channels.

    Examples
    --------
    Plot a front facing elevation spectrum to inspect how magnitude changes from below to
    above the listener:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_elevation_spectrum
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_elevation_spectrum(
    ...     hrtf,
    ...     azimuth="front",
    ...     x_axis="log",
    ...     ear="left",
    ...     freq_max=16000.0,
    ... )
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

    resolved_layout: Any
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
        (frequency_bins_hz >= float(cast(Any, resolved_frequency_axis["freq_min"])))
        & (frequency_bins_hz <= float(cast(Any, resolved_frequency_axis["freq_max"])))
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
            label=colorbar_label if show_labels else "",
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
            label=frequency_label if show_labels else "",
            config=resolved_frequency_axis,
        )
        ElevationAnglesAxis.apply(
            ax=ax,
            axis="y",
            values=sorted_elevation_values,
            label=None if show_labels else "",
        )
        resolved_title = default_subplot_title if show_titles else ""
        if show_titles:
            Titles.create_subplots_titles(ax=ax, title=resolved_title)
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_elevation_spectrum_title(real_azimuth=real_azimuth),
        )
    if show:
        plt.show()
    return figure.fig

def plot_itd(
    hrtf: "HRTF",
    plane_angle: float = 0.0,
    azimuth_range_mode: str = "0-360",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot signed ITD over a horizontal plane.

    ``plot_itd`` calls :func:`~hrtfpykit.hrtf.itd` as ``itd(hrtf, method="threshold",
    output="time", thresh_level=-10.0, upper_cut_freq=3000.0, filter_order=10,
    absolute=False)`` using the current ``hrtf.IR`` data. It then selects the nearest
    measured horizontal plane from ``hrtf.Sources``, sorts the selected sources by azimuth,
    and plots signed ITD in microseconds.

    Positive signed ITD follows the convention of :func:`~hrtfpykit.hrtf.itd`: the left ear
    is delayed relative to the right ear. ``azimuth_range_mode="-180-180"`` reverses the
    displayed x axis so listener left appears on the left side of the figure.

    Parameters
    ----------
    plane_angle : float, default=0.0
        Target horizontal plane elevation used to select the horizontal plane. The
        nearest available elevation in the grid is used.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention used on the x axis. ``0-360`` follows the SOFA anticlockwise
    convention: front is 0 degrees, listener left is 90 degrees, back is 180
        degrees, and listener right is 270 degrees. ``-180-180`` wraps listener right to
        -90 degrees and reverses the displayed x axis so listener left appears visually
        left.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the signed ITD curve.

    Raises
    ------
    ValueError
        If IR data or sample rate is missing, plane_angle is invalid, the azimuth range
        is invalid, the selected horizontal plane is empty, or the ITD value count
        differs from the number of source positions.

    Notes
    -----
    In ``azimuth_range_mode="0-360"``, the x axis follows the SOFA azimuth convention
    directly. In ``azimuth_range_mode="-180-180"``, the x axis is displayed right to left
    numerically so +90 degrees, listener left, appears on the left side of the figure and
    -90 degrees, listener right, appears on the right side.

    Examples
    --------
    Plot signed ITD around the horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_itd
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_itd(hrtf, plane_angle=0.0)
    """
    azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=azimuth_range_mode)
    resolved_margins = Margins()

    if hrtf.IR.values is None:
        raise ValueError("IR data is not available")
    if hrtf.IR.sample_rate is None:
        raise ValueError("IR sample_rate is required")
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    plane_angle = float(plane_angle)
    if not np.isfinite(plane_angle):
        raise ValueError("plane_angle must be a finite value")

    itd_values = np.asarray(
        itd(
            hrtf,
            output="time",
        ),
        dtype=float,
    )
    if itd_values.ndim != 1:
        itd_values = itd_values.reshape(-1)
    indices, real_elevation = get_horizontal_plane(
        hrtf=hrtf,
        plane_angle=plane_angle,
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
        label=None if show_labels else "",
    )
    Axis.apply_label(
        ax=ax,
        axis="y",
        default_label=Labels.itd,
        label=None if show_labels else "",
    )
    ax.grid(True)
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane="horizontal",
                plane_angle=real_elevation,
            ),
        )
    if show:
        plt.show()
    return figure.fig

def plot_absolute_itd(
    hrtf: "HRTF",
    plane_angle: float = 0.0,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot absolute ITD over a horizontal plane in polar form.

    ``plot_absolute_itd`` calls :func:`~hrtfpykit.hrtf.itd` as ``itd(hrtf,
    method="threshold", output="time", thresh_level=-10.0, upper_cut_freq=3000.0,
    filter_order=10, absolute=True)`` using the current ``hrtf.IR`` data. It selects the
    nearest measured horizontal plane from ``hrtf.Sources`` and sorts the polar curve by
    azimuth.

    The angular axis follows the SOFA spherical azimuth convention: front is 0 degrees,
    listener left is 90 degrees, back is 180 degrees, and listener right is 270 degrees. The
    radial axis shows ITD magnitude in microseconds.

    Parameters
    ----------
    plane_angle : float, default=0.0
        Target horizontal plane elevation used to select the horizontal plane. The
        nearest available elevation in the grid is used.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the absolute ITD polar curve.

    Raises
    ------
    ValueError
        If IR data or sample rate is missing, plane_angle is not finite, or the selected
        horizontal plane cannot be resolved for the current source grid.

    Notes
    -----
    The polar angular axis uses spherical azimuth in the SOFA convention. The radial label
    uses ITD time units and radial ticks use integer microsecond labels.

    Examples
    --------
    Plot the absolute ITD cue around the horizontal plane in polar form:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_absolute_itd
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_absolute_itd(hrtf, plane_angle=0.0)
    """
    resolved_margins = Margins()
    polar_tick_step = 30.0
    radial_tick_step = 200.0
    radial_tick_label_style = "integer"
    radial_label_position = 350.0
    polar_curve_color = "steelblue"
    polar_curve_linewidth = 2.0

    if hrtf.IR.values is None:
        raise ValueError("IR data is not available")
    if hrtf.IR.sample_rate is None:
        raise ValueError("IR sample_rate is required")
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    plane_angle = float(plane_angle)
    if not np.isfinite(plane_angle):
        raise ValueError("plane_angle must be a finite value")

    itd_values = np.asarray(
        itd(
            hrtf,
            output="time",
            absolute=True,
        ),
        dtype=float,
    )
    theta_values, radial_values, sorted_itd_values, real_elevation = create_horizontal_plane_curve(
        hrtf=hrtf,
        values=itd_values,
        plane_angle=plane_angle,
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
        radial_label_default=Labels.itd_time if show_labels else "",
        tick_step=radial_tick_step,
        tick_label_style=radial_tick_label_style,
        label_position=radial_label_position,
        label=None if show_labels else "",
    )
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane="horizontal",
                plane_angle=real_elevation,
            ),
        )
    ax.grid(True)
    if show:
        plt.show()
    return figure.fig

def plot_ild_fd(
    hrtf: "HRTF",
    plane: str = "horizontal",
    plane_angle: float = 0.0,
    azimuth_range_mode: str = "0-360",
    colormap: str = "jet",
    freq_min: float | None = None,
    freq_max: float | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot frequency dependent ILD across a measured spatial plane.

    ``plot_ild_fd`` calls :func:`~hrtfpykit.hrtf.ild` as ``ild(hrtf,
    mode="frequency-dependent", epsilon=1e-12, absolute=False)`` using the current
    ``hrtf.TF`` data. It then selects a horizontal plane with
    :func:`~hrtfpykit.utils.planes.get_horizontal_plane` or a median plane with
    :func:`~hrtfpykit.utils.planes.get_median_plane`, restricts the selected frequency bins,
    and draws ILD in dB as a frequency by angle heatmap.

    Horizontal planes use spherical azimuth on the y axis. Median planes use lateral polar
    angle on the y axis, where 0 degrees is front, 90 degrees is up, 180 degrees is back,
    and -90 degrees is down.

    Parameters
    ----------
    plane : {``horizontal``, ``median``}, default=``horizontal``
        Plane to visualize. ``horizontal`` uses a horizontal plane selected by spherical
        elevation. ``median`` uses the nearest measured lateral polar lateral angle.
    plane_angle : float, default=0.0
        Plane coordinate in degrees used to resolve the nearest measured plane. For
        ``plane="horizontal"`` this is spherical elevation. For ``plane="median"`` this
        is lateral polar lateral angle.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention used for horizontal plane azimuth values. ``0-360`` follows
        the SOFA anticlockwise convention: front is 0 degrees, listener left is 90
        degrees, back is 180 degrees, and listener right is 270 degrees. ``-180-180``
        wraps listener right to -90 degrees. Because horizontal plane azimuth is drawn
        on the y axis in this plot, signed azimuth keeps normal Matplotlib numeric
        orientation.
    colormap : str, default=``jet``
        Colormap name used for the heatmap.
    freq_min : float | None, default=None
        Minimum frequency in Hz included in the plot.
    freq_max : float | None, default=None
        Maximum frequency in Hz included in the plot.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the frequency dependent ILD heatmap.

    Raises
    ------
    AttributeError
        If plane is not ``horizontal`` or ``median``, or if plane_angle is not finite.
    ValueError
        If TF data or frequency bins are missing, the azimuth range is invalid, the
        selected plane has no positions, frequency bins are invalid, the selected
        frequency range is empty, or the frequency dependent ILD shape is invalid.

    Notes
    -----
    Horizontal plane azimuth follows ``azimuth_range_mode`` on the y axis. In signed mode,
    +90 degrees remains above -90 degrees because hrtfpykit only reverses the visual axis
    when azimuth is plotted on the x axis.

    Examples
    --------
    Plot frequency dependent ILD over the horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_ild_fd
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_ild_fd(
    ...     hrtf,
    ...     plane="horizontal",
    ...     plane_angle=0.0,
    ...     freq_max=16000.0,
    ... )
    """
    if plane not in ("horizontal", "median"):
        raise AttributeError("plot_ild_fd plane accepts horizontal or median")
    if isinstance(plane_angle, bool):
        raise AttributeError("plane_angle must be a finite value")
    plane_angle = float(plane_angle)
    if not np.isfinite(plane_angle):
        raise AttributeError("plane_angle must be a finite value")
    azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=azimuth_range_mode)
    resolved_margins = Margins()
    heatmap_margin_ratio = 0.0

    if hrtf.TF.values is None:
        raise ValueError("TF data is not available")

    plane_key = str(plane).strip().lower()

    figure = Figure(
        Layout_1(
            figsize=Layout_1().figsize,
            margins=resolved_margins,
        )
    )

    if plane_key == "horizontal":
        indices, real_plane_angle = get_horizontal_plane(
            hrtf=hrtf,
            plane_angle=plane_angle,
            angle_unit="degrees",
        )
    else:
        indices, real_plane_angle = get_median_plane(
            hrtf=hrtf,
            plane_angle=plane_angle,
            angle_unit="degrees",
        )

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

    if hrtf.TF.frequency_bins is None:
        raise ValueError("TF frequency bins are not available")
    frequency_bins_hz = np.asarray(hrtf.TF.frequency_bins, dtype=float)
    if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
        raise ValueError("TF frequency bins must be a non-empty 1D array")
    resolved_frequency_axis = FrequencyLinearAxis.build(
        frequency_bins=frequency_bins_hz,
        freq_min=freq_min,
        freq_max=freq_max,
        margin_ratio=heatmap_margin_ratio,
    )
    frequency_mask = (
        (frequency_bins_hz >= float(cast(Any, resolved_frequency_axis["freq_min"])))
        & (frequency_bins_hz <= float(cast(Any, resolved_frequency_axis["freq_max"])))
    )
    if not np.any(frequency_mask):
        raise ValueError("Selected frequency range produced no TF bins")
    frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

    ild_values = np.asarray(
        ild(
            hrtf,
            mode="frequency-dependent",
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
        label=Labels.ild if show_labels else "",
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
        label=frequency_label if show_labels else "",
        config=resolved_frequency_axis,
    )
    if plane_key == "horizontal":
        AzimuthAnglesAxis.apply(
            ax=ax,
            axis="y",
            values=sorted_subplot_plane_axis_values,
            range_mode=azimuth_range_mode,
            label=None if show_labels else "",
        )
    else:
        PolarAnglesAxis.apply(
            ax=ax,
            axis="y",
            values=sorted_subplot_plane_axis_values,
            label=None if show_labels else "",
        )
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane=plane_key,
                plane_angle=real_plane_angle,
            ),
        )
    if show:
        plt.show()
    return figure.fig

def plot_ild(
    hrtf: "HRTF",
    plane_angle: float = 0.0,
    azimuth_range_mode: str = "0-360",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot signed broad band ILD over a horizontal plane.

    ``plot_ild`` calls :func:`~hrtfpykit.hrtf.ild` as ``ild(hrtf, mode="broad-band",
    epsilon=1e-12, absolute=False)`` using the current ``hrtf.IR`` data. It then selects the
    nearest measured horizontal plane from ``hrtf.Sources``, sorts the selected sources by
    azimuth, and plots signed ILD in dB.

    Positive signed ILD follows the convention of :func:`~hrtfpykit.hrtf.ild`: the left ear
    level is greater than the right ear level. ``azimuth_range_mode="-180-180"`` reverses
    the displayed x axis so listener left appears on the left side of the figure.

    Parameters
    ----------
    plane_angle : float, default=0.0
        Target horizontal plane elevation used to select the horizontal plane. The
        nearest available elevation in the grid is used.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention used on the x axis. ``0-360`` follows the SOFA anticlockwise
    convention: front is 0 degrees, listener left is 90 degrees, back is 180
        degrees, and listener right is 270 degrees. ``-180-180`` wraps listener right to
        -90 degrees and reverses the displayed x axis so listener left appears visually
        left.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the signed broad band ILD curve.

    Raises
    ------
    ValueError
        If IR data is missing, plane_angle is invalid, the azimuth range is invalid, the
        selected horizontal plane is empty, or the ILD value count differs from the
        number of source positions.

    Notes
    -----
    In ``azimuth_range_mode="0-360"``, the x axis follows the SOFA azimuth convention
    directly. In ``azimuth_range_mode="-180-180"``, the x axis is displayed right to left
    numerically so +90 degrees, listener left, appears on the left side of the figure and
    -90 degrees, listener right, appears on the right side.

    Examples
    --------
    Plot signed broad band ILD around the horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_ild
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_ild(hrtf, plane_angle=0.0)
    """
    azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=azimuth_range_mode)
    resolved_margins = Margins()

    if hrtf.IR.values is None:
        raise ValueError("IR data is not available")
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    plane_angle = float(plane_angle)
    if not np.isfinite(plane_angle):
        raise ValueError("plane_angle must be a finite value")

    ild_values = np.asarray(
        ild(
            hrtf,
            mode="broad-band",
        ),
        dtype=float,
    )
    if ild_values.ndim != 1:
        ild_values = ild_values.reshape(-1)
    indices, real_elevation = get_horizontal_plane(
        hrtf=hrtf,
        plane_angle=plane_angle,
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
        label=None if show_labels else "",
    )
    Axis.apply_label(
        ax=ax,
        axis="y",
        default_label=Labels.ild if show_labels else "",
        label=None if show_labels else "",
    )
    ax.grid(True)
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane="horizontal",
                plane_angle=real_elevation,
            ),
        )
    if show:
        plt.show()
    return figure.fig

def plot_absolute_ild(
    hrtf: "HRTF",
    plane_angle: float = 0.0,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot absolute broad band ILD over a horizontal plane in polar form.

    ``plot_absolute_ild`` calls :func:`~hrtfpykit.hrtf.ild` as ``ild(hrtf,
    mode="broad-band", epsilon=1e-12, absolute=True)`` using the current ``hrtf.IR`` data.
    It selects the nearest measured horizontal plane from ``hrtf.Sources`` and sorts the
    polar curve by azimuth.

    The angular axis follows the SOFA spherical azimuth convention: front is 0 degrees,
    listener left is 90 degrees, back is 180 degrees, and listener right is 270 degrees. The
    radial axis shows ILD magnitude in dB.

    Parameters
    ----------
    plane_angle : float, default=0.0
        Target horizontal plane elevation used to select the horizontal plane. The
        nearest available elevation in the grid is used.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the absolute broad band ILD polar curve.

    Raises
    ------
    ValueError
        If IR data is missing, plane_angle is not finite, or the selected horizontal
        plane cannot be resolved for the current source grid.

    Notes
    -----
    The polar angular axis uses spherical azimuth in the SOFA convention. The radial label
    uses ILD dB units and radial ticks are formatted as integer decibel values.

    Examples
    --------
    Plot the unsigned broad band ILD cue around the horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_absolute_ild
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_absolute_ild(hrtf, plane_angle=0.0)
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
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    plane_angle = float(plane_angle)
    if not np.isfinite(plane_angle):
        raise ValueError("plane_angle must be a finite value")

    ild_values = np.asarray(
        ild(
            hrtf,
            mode="broad-band",
            absolute=True,
        ),
        dtype=float,
    )
    theta_values, radial_values, sorted_ild_values, real_elevation = create_horizontal_plane_curve(
        hrtf=hrtf,
        values=ild_values,
        plane_angle=plane_angle,
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
        radial_label_default=Labels.ild_db if show_labels else "",
        tick_step=radial_tick_step,
        tick_label_style=radial_tick_label_style,
        label_position=radial_label_position,
        label=None if show_labels else "",
    )
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane="horizontal",
                plane_angle=real_elevation,
            ),
        )
    ax.grid(True)
    if show:
        plt.show()
    return figure.fig

def plot_source_grid(
    hrtf: "HRTF",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot the current HRTF source grid in 3D.

    ``plot_source_grid`` reads Cartesian source positions from ``hrtf.Sources`` and draws
    the current source grid. It uses equal 3D axis limits and adds direction markers when
    labels are enabled.

    The plot reflects the source grid currently stored in the HRTF object, including any
    source selection already applied to that object.

    Parameters
    ----------
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the 3D source grid.

    Raises
    ------
    ValueError
        If source positions cannot be resolved as a valid Cartesian grid or axis
        geometry cannot be computed from the current positions.

    Notes
    -----
    The 3D axes use the same scale on all dimensions, so the plotted grid keeps the physical
    proportions of the source positions.

    Examples
    --------
    Plot the measurement source grid from a loaded SOFA file:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_source_grid
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_source_grid(hrtf)
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
        label=None if show_labels else "",
    )
    YAxis.apply(
        ax=ax,
        center=y_center,
        half_span=axis_half_span,
        label=None if show_labels else "",
    )
    ZAxis.apply(
        ax=ax,
        center=z_center,
        half_span=axis_half_span,
        label=None if show_labels else "",
    )
    cast(Any, ax).set_box_aspect((1.0, 1.0, 1.0))
    if show_labels:
        create_sources_grid_direction_markers(
            ax=ax,
            sources=hrtf.Sources,
            axis_half_span=axis_half_span,
        )

    ax.grid(True)
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            "Source Grid",
        )
    if show:
        plt.show()
    return figure.fig

def plot_plane_grid(
    hrtf: "HRTF",
    plane: str | list[str] | tuple[str, ...] = "horizontal",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot the source grid and highlight canonical planes.

    ``plot_plane_grid`` reads Cartesian source positions from ``hrtf.Sources`` and overlays
    selected plane memberships. Horizontal, median, and frontal planes are resolved with the
    same plane definitions used by hrtfpykit plot and metric workflows.

    The highlighted points therefore match the same plane definitions used by the metric and
    plane plotting functions.

    Parameters
    ----------
    plane : str | list[str] | tuple[str, ...], default=``horizontal``
        Plane or planes to highlight. Accepted values are ``horizontal``, ``median``,
        and ``frontal``. A single string highlights one plane, while a list or tuple
        highlights multiple planes in the same figure.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated default figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the 3D source grid with highlighted planes.

    Raises
    ------
    ValueError
        If plane is empty, contains an unsupported plane name, source positions cannot
        be resolved as Cartesian coordinates, or axis geometry cannot be computed from
        the current source grid.

    Notes
    -----
    Duplicate plane names are ignored after the first occurrence. A legend identifies the
    full source grid and each highlighted plane.

    Examples
    --------
    Plot the source grid and highlight the canonical horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_plane_grid
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> plot_plane_grid(hrtf, plane="horizontal")
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
                plane_angle=0.0,
                angle_unit="degrees",
            )
        elif plane_key == "median":
            indices, _ = get_median_plane(
                hrtf=hrtf,
                plane_angle=0.0,
                angle_unit="degrees",
            )
        else:
            indices, _ = get_frontal_plane(
                hrtf=hrtf,
                plane_angle=90.0,
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
        label=None if show_labels else "",
    )
    YAxis.apply(
        ax=ax,
        center=y_center,
        half_span=axis_half_span,
        label=None if show_labels else "",
    )
    ZAxis.apply(
        ax=ax,
        center=z_center,
        half_span=axis_half_span,
        label=None if show_labels else "",
    )
    cast(Any, ax).set_box_aspect((1.0, 1.0, 1.0))
    if show_labels:
        create_sources_grid_direction_markers(
            ax=ax,
            sources=hrtf.Sources,
            axis_half_span=axis_half_span,
        )

    ax.grid(True)
    if show_legends:
        ax.legend(loc="upper right")
    if show_titles:
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
    return figure.fig
