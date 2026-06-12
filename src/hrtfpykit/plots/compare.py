from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from .axis import (
    AmplitudeAxis,
    Axis,
    AzimuthAnglesAxis,
    AzimuthAnglesAxisPolarProjection,
    FrequencyLinearAxis,
    FrequencyLogAxis,
    MagnitudeAxis,
    RadialAxisPolarProjection,
    SampleAxis,
    TimeAxis,
    ElevationAnglesAxis,
)
from .default import Margins
from .figure import Figure
from .labels import Labels
from .layouts import Layout_1, Layout_2Horizontal, Layout_2Vertical, Layout_3
from .legends import Subjects
from .titles import Titles
from .types import Heatmap
from ..utils.warnings import HRTFPyKitWarning, warn_user
from ..utils.coordinates import get_position_queries, get_source_positions
from ..utils.dsp import magnitude_to_db
from ..utils.metrics import hrtf_difference, ild, ild_difference, itd, itd_difference
from ..utils.planes import get_horizontal_plane
from .polar import create_horizontal_plane_curve


if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


def compare_magnitude(
    hrtfs: list["HRTF"],
    positions: str | list | tuple | np.ndarray = ("front",),
    ear: str = "left",
    x_axis: str = "linear",
    unit: str = "db",
    reference: float | str = 1.0,
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Compare HRTF magnitude responses from several HRTFs.

    ``compare_magnitude`` reads ``TF.magnitude`` and ``TF.frequency_bins`` from every HRTF
    in ``hrtfs``. For each requested source query, it resolves the nearest measured source
    in every HRTF with ``hrtf.Sources.get_position_index(...)`` and overlays the selected
    magnitude traces. When ``unit="db"``, values are converted to decibels.

    Frequency limits are resolved across all inputs. If the same query resolves to different
    real source coordinates across HRTFs, the function emits an
    :class:`~hrtfpykit.utils.warnings.HRTFPyKitWarning` so the comparison is explicit about
    source mismatch.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least
        2 and at most 5 entries. Every object must contain frequency domain data and
        frequency bins.
    positions : str | list | tuple | np.ndarray, default=(``front``,)
        Position query or collection of position queries. Up to 4 positions are
        accepted. Query resolution uses each HRTF's nearest available source in
        spherical coordinates.
    ear : {``left``, ``right``, ``both``}, default=``left``
        Ear channel selection. ``both`` requires exactly one position and creates
        separate left ear and right ear subplots.
    x_axis : {``linear``, ``log``}, default=``linear``
        Frequency axis scale used for all subplots.
    unit : {``db``, ``linear``}, default=``db``
        Magnitude representation. ``db`` converts magnitudes with decibel conversion;
        ``linear`` plots raw magnitudes.
    reference : float | str, default=1.0
        Reference used when ``unit`` is ``db``. ``max`` normalizes all plotted curves to
        the maximum selected magnitude over the requested positions, frequency range,
        and ear selection.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses the default color cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right`` when ``x_axis`` is ``linear`` and
        ``upper left`` when ``x_axis`` is ``log``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Optional legend anchor tuple (x, y).
    freq_min : float | None, default=None
        Minimum frequency in Hz. If omitted, resolved from all HRTFs.
    freq_max : float | None, default=None
        Maximum frequency in Hz. If omitted, resolved from all HRTFs.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated subplot and figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the overlaid HRTF magnitude comparisons.

    Raises
    ------
    ValueError
        If the HRTF list length, option values, legend/style lengths, requested
        positions, frequency range, TF availability, frequency bins, or ear channels are
        invalid.

    Warns
    -----
    HRTFPyKitWarning
        If the same position query resolves to different real source coordinates in
        different HRTFs.

    Notes
    -----
    With one selected source and ``ear="both"``, the figure places the left and right ear
    comparisons in separate panels. Otherwise, each requested source gets its own panel and
    every HRTF is drawn in that panel.

    Examples
    --------
    Compare left ear magnitude responses from two SOFA files at the front direction, using a
    logarithmic frequency axis and a shared dB reference:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_magnitude
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_magnitude(
    ...     [hrtf_a, hrtf_b],
    ...     positions="front",
    ...     ear="left",
    ...     x_axis="log",
    ...     unit="db",
    ...     reference="max",
    ...     legends=["P0001", "P0002"],
    ...     line_styles=["-", "--"],
    ...     freq_max=16000.0,
    ... )
    """
    if not isinstance(hrtfs, list):
        raise ValueError("hrtfs must be a list[HRTF]")
    hrtf_count = len(hrtfs)
    if hrtf_count < 2:
        raise ValueError("compare_magnitude requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_magnitude accepts up to 5 HRTFs")
    if unit not in {"db", "linear"}:
        raise ValueError("unit accepts db or linear")
    if x_axis not in {"linear", "log"}:
        raise ValueError("x_axis accepts linear or log")
    if ear not in {"left", "right", "both"}:
        raise ValueError("ear accepts left, right, or both")

    position_queries = get_position_queries(positions)
    position_count = len(position_queries)
    if position_count == 0:
        raise ValueError("At least one position is required")
    if position_count > 4:
        raise ValueError("compare_magnitude accepts up to 4 positions")
    if ear == "both" and position_count != 1:
        raise ValueError("ear='both' accepts exactly one position")

    if legends is None:
        resolved_legends = Subjects.create_default_labels(hrtf_count)
    else:
        resolved_legends = [str(value) for value in legends]
        if len(resolved_legends) != hrtf_count:
            raise ValueError("legends length must match len(hrtfs)")

    if line_colors is None:
        default_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_values = (
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
            if default_colors is None
            else default_colors.by_key().get("color", ["tab:blue"])
        )
        if len(color_values) == 0:
            color_values = ["tab:blue"]
        resolved_line_colors = [color_values[index % len(color_values)] for index in range(hrtf_count)]
    else:
        resolved_line_colors = list(line_colors)
        if len(resolved_line_colors) != hrtf_count:
            raise ValueError("line_colors length must match len(hrtfs)")

    if line_styles is None:
        resolved_line_styles = ["-"] * hrtf_count
    else:
        resolved_line_styles = [str(value) for value in line_styles]
        if len(resolved_line_styles) != hrtf_count:
            raise ValueError("line_styles length must match len(hrtfs)")

    resolved_frequency_axis_class = FrequencyLogAxis if x_axis == "log" else FrequencyLinearAxis
    resolved_ear_index = 0 if ear == "left" else 1
    resolved_margins = Margins()

    frequency_bins_by_subject: list[np.ndarray] = []
    frequency_masks_by_subject: list[np.ndarray] = []
    selected_indices_by_subject: list[list[int]] = []
    selected_positions_by_subject: list[list[np.ndarray]] = []
    tf_magnitudes_by_subject: list[np.ndarray] = []

    for subject_index, hrtf in enumerate(hrtfs):
        if hrtf.TF.values is None or hrtf.TF.frequency_bins is None:
            raise ValueError(f"HRTF at index {subject_index} does not contain TF data")
        subject_frequency_bins = np.asarray(hrtf.TF.frequency_bins, dtype=float)
        if subject_frequency_bins.ndim != 1 or subject_frequency_bins.size == 0:
            raise ValueError(f"HRTF at index {subject_index} has invalid TF frequency bins")
        if x_axis == "log":
            subject_positive_bins = subject_frequency_bins[subject_frequency_bins > 0.0]
            if subject_positive_bins.size == 0:
                raise ValueError(f"HRTF at index {subject_index} has no positive TF frequency bins for log axis")

        subject_indices: list[int] = []
        subject_positions: list[np.ndarray] = []
        for query in position_queries:
            idx, selected_position = hrtf.Sources.get_position_index(
                query,
                coordinate_system="spherical",
            )
            subject_indices.append(int(idx))
            subject_positions.append(np.asarray(selected_position, dtype=float).reshape(-1))
        selected_indices_by_subject.append(subject_indices)
        selected_positions_by_subject.append(subject_positions)
        frequency_bins_by_subject.append(subject_frequency_bins)
        tf_magnitudes_by_subject.append(np.asarray(hrtf.TF.magnitude, dtype=float))

    reference_positions = selected_positions_by_subject[0]
    for subject_index in range(1, hrtf_count):
        for position_index, query in enumerate(position_queries):
            reference_position = reference_positions[position_index]
            compared_position = selected_positions_by_subject[subject_index][position_index]
            if not np.allclose(reference_position, compared_position, atol=1e-8, rtol=0.0):
                warn_user(
                    (
                        "compare_magnitude resolved different real positions for "
                        f"query={query!r}: subject_1={reference_position.tolist()} "
                        f"vs subject_{subject_index + 1}={compared_position.tolist()}"
                    ),
                    category=HRTFPyKitWarning,
                )

    if freq_min is None:
        if x_axis == "log":
            resolved_freq_min = max(
                float(np.min(bins[bins > 0.0])) for bins in frequency_bins_by_subject
            )
        else:
            resolved_freq_min = max(float(np.min(bins)) for bins in frequency_bins_by_subject)
    else:
        resolved_freq_min = float(freq_min)
    if freq_max is None:
        resolved_freq_max = min(float(np.max(bins)) for bins in frequency_bins_by_subject)
    else:
        resolved_freq_max = float(freq_max)
    if not np.isfinite(resolved_freq_min) or not np.isfinite(resolved_freq_max):
        raise ValueError("freq_min and freq_max must be finite values")
    if resolved_freq_min >= resolved_freq_max:
        raise ValueError("Resolved frequency range is empty across provided HRTFs")
    if x_axis == "log" and resolved_freq_min <= 0.0:
        raise ValueError("freq_min must be positive for logarithmic frequency axis")

    for subject_bins in frequency_bins_by_subject:
        subject_mask = (
            (subject_bins >= resolved_freq_min)
            & (subject_bins <= resolved_freq_max)
        )
        if not np.any(subject_mask):
            raise ValueError("Selected frequency range produced no TF bins for one HRTF")
        frequency_masks_by_subject.append(subject_mask)

    tf_values_by_subject: list[np.ndarray] = []
    if unit == "db":
        if isinstance(reference, str) and str(reference).strip().lower() == "max":
            reference_candidates: list[float] = []
            for subject_index, tf_magnitude in enumerate(tf_magnitudes_by_subject):
                subject_mask = frequency_masks_by_subject[subject_index]
                for position_index in range(position_count):
                    source_index = selected_indices_by_subject[subject_index][position_index]
                    source_values = np.asarray(tf_magnitude[source_index], dtype=float)
                    if ear == "both":
                        if source_values.ndim < 2 or source_values.shape[0] < 2:
                            raise ValueError(
                                f"Both ears requested but HRTF at index {subject_index} does not contain two ear channels"
                            )
                        reference_candidates.append(float(np.max(source_values[:, subject_mask])))
                    else:
                        if source_values.ndim == 1:
                            reference_candidates.append(float(np.max(source_values[subject_mask])))
                        else:
                            if source_values.shape[0] <= resolved_ear_index:
                                raise ValueError(
                                    f"Requested ear '{ear}' is not available in HRTF at index {subject_index}"
                                )
                            reference_candidates.append(
                                float(np.max(source_values[resolved_ear_index, subject_mask]))
                            )
            if len(reference_candidates) == 0:
                raise ValueError("Unable to resolve reference='max' for comparison")
            resolved_reference = float(np.max(np.asarray(reference_candidates, dtype=float)))
            for tf_magnitude in tf_magnitudes_by_subject:
                tf_values_by_subject.append(
                    np.asarray(magnitude_to_db(tf_magnitude, reference=resolved_reference), dtype=float)
                )
        else:
            for tf_magnitude in tf_magnitudes_by_subject:
                tf_values_by_subject.append(
                    np.asarray(magnitude_to_db(tf_magnitude, reference=reference), dtype=float)
                )
    else:
        tf_values_by_subject = [np.asarray(values, dtype=float) for values in tf_magnitudes_by_subject]

    resolved_layout: Any
    if ear == "both":
        resolved_layout = Layout_2Horizontal(
            figsize=Layout_2Horizontal().figsize,
            margins=resolved_margins,
        )
    elif position_count == 1:
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

    resolved_frequency_axis = resolved_frequency_axis_class.build(
        frequency_bins=None,
        freq_min=resolved_freq_min,
        freq_max=resolved_freq_max,
    )
    magnitude_legend_location = "upper right" if x_axis == "linear" else "upper left"
    resolved_legend_location = (
        magnitude_legend_location if legend_location is None else str(legend_location)
    )
    resolved_legend_bbox_to_anchor = (
        None
        if legend_bbox_to_anchor is None
        else (
            float(legend_bbox_to_anchor[0]),
            float(legend_bbox_to_anchor[1]),
        )
    )

    if ear == "both":
        for axis_name, ear_index, subplot_title in (
            ("left", 0, Titles.left_ear),
            ("right", 1, Titles.right_ear),
        ):
            ax = figure.get_ax(axis_name)
            for subject_index in range(hrtf_count):
                source_index = selected_indices_by_subject[subject_index][0]
                subject_bins = frequency_bins_by_subject[subject_index]
                subject_mask = frequency_masks_by_subject[subject_index]
                subject_values = np.asarray(tf_values_by_subject[subject_index][source_index], dtype=float)
                if subject_values.ndim < 2 or subject_values.shape[0] <= ear_index:
                    raise ValueError(
                        f"Both ears requested but HRTF at index {subject_index} does not contain two ear channels"
                    )
                figure.create_two_dimension(
                    ax=ax,
                    x=subject_bins[subject_mask] / 1000.0,
                    y=np.asarray(subject_values[ear_index, subject_mask], dtype=float).reshape(-1),
                    color=resolved_line_colors[subject_index],
                    linestyle=resolved_line_styles[subject_index],
                    linewidth=2.0,
                )
            resolved_frequency_axis_class.apply(
                ax=ax,
                axis="x",
                label=Labels.frequency if show_labels else "",
                config=resolved_frequency_axis,
            )
            MagnitudeAxis.apply(ax=ax, axis="y", unit=unit, label=None if show_labels else "")
            if show_titles:
                Titles.create_subplots_titles(ax=ax, title=subplot_title)
            if show_legends:
                Subjects.apply(
                    ax=ax,
                    labels=resolved_legends,
                    location=resolved_legend_location,
                    bbox_to_anchor=resolved_legend_bbox_to_anchor,
                )
            ax.grid(True)
    else:
        for position_index in range(position_count):
            ax = figure.get_ax(position_index)
            for subject_index in range(hrtf_count):
                source_index = selected_indices_by_subject[subject_index][position_index]
                subject_bins = frequency_bins_by_subject[subject_index]
                subject_mask = frequency_masks_by_subject[subject_index]
                subject_values = np.asarray(tf_values_by_subject[subject_index][source_index], dtype=float)
                if subject_values.ndim == 1:
                    y_values = np.asarray(subject_values[subject_mask], dtype=float).reshape(-1)
                else:
                    if subject_values.shape[0] <= resolved_ear_index:
                        raise ValueError(
                            f"Requested ear '{ear}' is not available in HRTF at index {subject_index}"
                        )
                    y_values = np.asarray(
                        subject_values[resolved_ear_index, subject_mask],
                        dtype=float,
                    ).reshape(-1)
                figure.create_two_dimension(
                    ax=ax,
                    x=subject_bins[subject_mask] / 1000.0,
                    y=y_values,
                    color=resolved_line_colors[subject_index],
                    linestyle=resolved_line_styles[subject_index],
                    linewidth=2.0,
                )
            resolved_frequency_axis_class.apply(
                ax=ax,
                axis="x",
                label=Labels.frequency if show_labels else "",
                config=resolved_frequency_axis,
            )
            MagnitudeAxis.apply(ax=ax, axis="y", unit=unit, label=None if show_labels else "")
            if show_titles:
                Titles.create_subplots_titles(
                    ax=ax,
                    title=Titles.create_position_title(
                        selected_positions=reference_positions[position_index],
                    ),
                )
            if Figure.shared_x_visible:
                ax.tick_params(axis="x", which="both", labelbottom=True)
            if show_legends:
                Subjects.apply(
                    ax=ax,
                    labels=resolved_legends,
                    location=resolved_legend_location,
                    bbox_to_anchor=resolved_legend_bbox_to_anchor,
                )
            ax.grid(True)

    if ear != "both" and position_count < figure.axes.size:
        figure.hide_unused_axes(position_count)

    if show:
        plt.show()
    return figure.fig


def compare_amplitude(
    hrtfs: list["HRTF"],
    positions: str | list | tuple | np.ndarray = ("front",),
    ear: str = "left",
    x_axis: str = "time",
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Compare HRIR amplitude responses from several HRTFs.

    ``compare_amplitude`` reads ``IR.values`` from every HRTF in ``hrtfs``. For each
    requested source query, it resolves the nearest measured source in every HRTF with
    ``hrtf.Sources.get_position_index(...)`` and overlays the selected impulse responses.

    The x axis can show sample indices or time in milliseconds. Time mode uses each HRTF's
    ``IR.sample_rate``. If the same query resolves to different real source coordinates
    across HRTFs, the function emits an :class:`~hrtfpykit.utils.warnings.HRTFPyKitWarning`.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least
        2 and at most 5 entries. Every object must contain time domain IR data.
    positions : str | list | tuple | np.ndarray, default=(``front``,)
        Position query or collection of position queries. Up to 4 positions are
        accepted. Query resolution uses each HRTF's nearest available source in
        spherical coordinates.
    ear : {``left``, ``right``, ``both``}, default=``left``
        Ear channel selection. ``both`` requires exactly one position and creates
        separate left ear and right ear subplots.
    x_axis : {``time``, ``samples``}, default=``time``
        Horizontal axis mode for waveforms. ``time`` converts samples to milliseconds
        using each HRTF's sample rate.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses the default color cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Optional legend anchor tuple (x, y).
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated subplot and figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the overlaid HRIR amplitude comparisons.

    Raises
    ------
    ValueError
        If the HRTF list length, option values, legend/style lengths, requested
        positions, IR availability, sample rate requirements, IR shape, or ear channels
        are invalid.

    Warns
    -----
    HRTFPyKitWarning
        If the same position query resolves to different real source coordinates in
        different HRTFs.

    Notes
    -----
    With one selected source and ``ear="both"``, the figure places the left and right ear
    comparisons in separate panels. Otherwise, each requested source gets its own panel and
    every HRTF is drawn in that panel.

    Examples
    --------
    Compare left ear front direction impulse responses for two HRTFs, using sample indices
    on the x axis:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_amplitude
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_amplitude(
    ...     [hrtf_a, hrtf_b],
    ...     positions="front",
    ...     ear="left",
    ...     x_axis="samples",
    ...     legends=["P0001", "P0002"],
    ...     line_styles=["-", "--"],
    ... )
    """
    if not isinstance(hrtfs, list):
        raise ValueError("hrtfs must be a list[HRTF]")
    hrtf_count = len(hrtfs)
    if hrtf_count < 2:
        raise ValueError("compare_amplitude requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_amplitude accepts up to 5 HRTFs")
    if ear not in {"left", "right", "both"}:
        raise ValueError("ear accepts left, right, or both")
    if x_axis not in {"time", "samples"}:
        raise ValueError("x_axis accepts time or samples")

    position_queries = get_position_queries(positions)
    position_count = len(position_queries)
    if position_count == 0:
        raise ValueError("At least one position is required")
    if position_count > 4:
        raise ValueError("compare_amplitude accepts up to 4 positions")
    if ear == "both" and position_count != 1:
        raise ValueError("ear='both' accepts exactly one position")

    if legends is None:
        resolved_legends = Subjects.create_default_labels(hrtf_count)
    else:
        resolved_legends = [str(value) for value in legends]
        if len(resolved_legends) != hrtf_count:
            raise ValueError("legends length must match len(hrtfs)")

    if line_colors is None:
        default_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_values = (
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
            if default_colors is None
            else default_colors.by_key().get("color", ["tab:blue"])
        )
        if len(color_values) == 0:
            color_values = ["tab:blue"]
        resolved_line_colors = [color_values[index % len(color_values)] for index in range(hrtf_count)]
    else:
        resolved_line_colors = list(line_colors)
        if len(resolved_line_colors) != hrtf_count:
            raise ValueError("line_colors length must match len(hrtfs)")

    if line_styles is None:
        resolved_line_styles = ["-"] * hrtf_count
    else:
        resolved_line_styles = [str(value) for value in line_styles]
        if len(resolved_line_styles) != hrtf_count:
            raise ValueError("line_styles length must match len(hrtfs)")

    resolved_ear_index = 0 if ear == "left" else 1
    resolved_margins = Margins()

    selected_indices_by_subject: list[list[int]] = []
    selected_positions_by_subject: list[list[np.ndarray]] = []
    ir_values_by_subject: list[np.ndarray] = []
    x_values_by_subject: list[np.ndarray] = []

    for subject_index, hrtf in enumerate(hrtfs):
        if hrtf.IR.values is None:
            raise ValueError(f"HRTF at index {subject_index} does not contain IR data")
        if x_axis == "time" and hrtf.IR.sample_rate is None:
            raise ValueError(
                f"HRTF at index {subject_index} requires sample_rate for x_axis='time'"
            )

        subject_indices: list[int] = []
        subject_positions: list[np.ndarray] = []
        for query in position_queries:
            idx, selected_position = hrtf.Sources.get_position_index(
                query,
                coordinate_system="spherical",
            )
            subject_indices.append(int(idx))
            subject_positions.append(np.asarray(selected_position, dtype=float).reshape(-1))
        selected_indices_by_subject.append(subject_indices)
        selected_positions_by_subject.append(subject_positions)

        subject_ir_values = np.asarray(hrtf.IR.values, dtype=float)
        if subject_ir_values.ndim < 2 or subject_ir_values.shape[-1] == 0:
            raise ValueError(f"HRTF at index {subject_index} has invalid IR values")
        ir_values_by_subject.append(subject_ir_values)

        subject_sample_indexes = np.arange(subject_ir_values.shape[-1], dtype=float)
        if x_axis == "time":
            x_values_by_subject.append(1000.0 * subject_sample_indexes / float(cast(Any, hrtf.IR.sample_rate)))
        else:
            x_values_by_subject.append(subject_sample_indexes)

    reference_positions = selected_positions_by_subject[0]
    for subject_index in range(1, hrtf_count):
        for position_index, query in enumerate(position_queries):
            reference_position = reference_positions[position_index]
            compared_position = selected_positions_by_subject[subject_index][position_index]
            if not np.allclose(reference_position, compared_position, atol=1e-8, rtol=0.0):
                warn_user(
                    (
                        "compare_amplitude resolved different real positions for "
                        f"query={query!r}: subject_1={reference_position.tolist()} "
                        f"vs subject_{subject_index + 1}={compared_position.tolist()}"
                    ),
                    category=HRTFPyKitWarning,
                )

    resolved_layout: Any
    if ear == "both":
        resolved_layout = Layout_2Horizontal(
            figsize=Layout_2Horizontal().figsize,
            margins=resolved_margins,
        )
    elif position_count == 1:
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
    resolved_legend_location = Subjects.location if legend_location is None else str(legend_location)
    resolved_legend_bbox_to_anchor = (
        None
        if legend_bbox_to_anchor is None
        else (
            float(legend_bbox_to_anchor[0]),
            float(legend_bbox_to_anchor[1]),
        )
    )

    if ear == "both":
        for axis_name, ear_index, subplot_title in (
            ("left", 0, Titles.left_ear),
            ("right", 1, Titles.right_ear),
        ):
            ax = figure.get_ax(axis_name)
            for subject_index in range(hrtf_count):
                source_index = selected_indices_by_subject[subject_index][0]
                subject_x_values = x_values_by_subject[subject_index]
                subject_ir_values = np.asarray(ir_values_by_subject[subject_index][source_index], dtype=float)
                if subject_ir_values.ndim < 2 or subject_ir_values.shape[0] <= ear_index:
                    raise ValueError(
                        f"Both ears requested but HRTF at index {subject_index} does not contain two ear channels"
                    )
                figure.create_two_dimension(
                    ax=ax,
                    x=subject_x_values,
                    y=np.asarray(subject_ir_values[ear_index], dtype=float).reshape(-1),
                    color=resolved_line_colors[subject_index],
                    linestyle=resolved_line_styles[subject_index],
                    linewidth=2.0,
                )
            if x_axis == "time":
                TimeAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
            else:
                SampleAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
            AmplitudeAxis.apply(ax=ax, axis="y", label=None if show_labels else "")
            if show_titles:
                Titles.create_subplots_titles(ax=ax, title=subplot_title)
            if show_legends:
                Subjects.apply(
                    ax=ax,
                    labels=resolved_legends,
                    location=resolved_legend_location,
                    bbox_to_anchor=resolved_legend_bbox_to_anchor,
                )
            ax.grid(True)
    else:
        for position_index in range(position_count):
            ax = figure.get_ax(position_index)
            for subject_index in range(hrtf_count):
                source_index = selected_indices_by_subject[subject_index][position_index]
                subject_x_values = x_values_by_subject[subject_index]
                subject_ir_values = np.asarray(ir_values_by_subject[subject_index][source_index], dtype=float)
                if subject_ir_values.ndim == 1:
                    y_values = np.asarray(subject_ir_values, dtype=float).reshape(-1)
                else:
                    if subject_ir_values.shape[0] <= resolved_ear_index:
                        raise ValueError(
                            f"Requested ear '{ear}' is not available in HRTF at index {subject_index}"
                        )
                    y_values = np.asarray(subject_ir_values[resolved_ear_index], dtype=float).reshape(-1)
                figure.create_two_dimension(
                    ax=ax,
                    x=subject_x_values,
                    y=y_values,
                    color=resolved_line_colors[subject_index],
                    linestyle=resolved_line_styles[subject_index],
                    linewidth=2.0,
                )
            if x_axis == "time":
                TimeAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
            else:
                SampleAxis.apply(ax=ax, axis="x", label=None if show_labels else "")
            AmplitudeAxis.apply(ax=ax, axis="y", label=None if show_labels else "")
            if show_titles:
                Titles.create_subplots_titles(
                    ax=ax,
                    title=Titles.create_position_title(
                        selected_positions=reference_positions[position_index],
                    ),
                )
            if Figure.shared_x_visible:
                ax.tick_params(axis="x", which="both", labelbottom=True)
            if show_legends:
                Subjects.apply(
                    ax=ax,
                    labels=resolved_legends,
                    location=resolved_legend_location,
                    bbox_to_anchor=resolved_legend_bbox_to_anchor,
                )
            ax.grid(True)

    if ear != "both" and position_count < figure.axes.size:
        figure.hide_unused_axes(position_count)

    if show:
        plt.show()
    return figure.fig


def compare_absolute_itd(
    hrtfs: list["HRTF"],
    plane_angle: float = 0.0,
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Compare absolute ITD curves from several HRTFs.

    ``compare_absolute_itd`` calls :func:`~hrtfpykit.hrtf.itd` for each HRTF as ``itd(hrtf,
    method="threshold", output="time", thresh_level=-10.0, upper_cut_freq=3000.0,
    filter_order=10, absolute=True)``. It selects the nearest measured horizontal plane for
    each HRTF, sorts each polar curve by azimuth, and overlays the curves in one figure.

    Each HRTF resolves the requested horizontal plane independently. If compared HRTFs
    resolve to different real elevations, the function emits an
    :class:`~hrtfpykit.utils.warnings.HRTFPyKitWarning`.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least
        2 and at most 5 entries. Every object must contain IR data and an IR sample
        rate.
    plane_angle : float, default=0.0
        Requested horizontal plane elevation in degrees. The nearest available elevation
        is selected separately for each HRTF.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses the default color cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Legend anchor tuple (x, y). Defaults to (1.08, 1.08).
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the overlaid absolute ITD polar curves.

    Raises
    ------
    ValueError
        If the HRTF list length, plane_angle, legend/style lengths, IR availability, or
        sample rate availability are invalid.

    Warns
    -----
    HRTFPyKitWarning
        If HRTFs resolve the requested horizontal plane to different real elevations.

    Notes
    -----
    The polar angular axis uses spherical azimuth in the SOFA convention. The radial axis
    uses ITD time units and integer microsecond tick labels.

    Examples
    --------
    Compare absolute ITD on the horizontal plane for two HRTFs:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_absolute_itd
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_absolute_itd(
    ...     [hrtf_a, hrtf_b],
    ...     plane_angle=0.0,
    ...     legends=["P0001", "P0002"],
    ...     line_styles=["-", "--"],
    ... )
    """
    if not isinstance(hrtfs, list):
        raise ValueError("hrtfs must be a list[HRTF]")
    hrtf_count = len(hrtfs)
    if hrtf_count < 2:
        raise ValueError("compare_absolute_itd requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_absolute_itd accepts up to 5 HRTFs")
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    resolved_plane_angle = float(plane_angle)
    if not np.isfinite(resolved_plane_angle):
        raise ValueError("plane_angle must be a finite value")

    if legends is None:
        resolved_legends = Subjects.create_default_labels(hrtf_count)
    else:
        resolved_legends = [str(value) for value in legends]
        if len(resolved_legends) != hrtf_count:
            raise ValueError("legends length must match len(hrtfs)")

    if line_colors is None:
        default_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_values = (
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
            if default_colors is None
            else default_colors.by_key().get("color", ["tab:blue"])
        )
        if len(color_values) == 0:
            color_values = ["tab:blue"]
        resolved_line_colors = [color_values[index % len(color_values)] for index in range(hrtf_count)]
    else:
        resolved_line_colors = list(line_colors)
        if len(resolved_line_colors) != hrtf_count:
            raise ValueError("line_colors length must match len(hrtfs)")

    if line_styles is None:
        resolved_line_styles = ["-"] * hrtf_count
    else:
        resolved_line_styles = [str(value) for value in line_styles]
        if len(resolved_line_styles) != hrtf_count:
            raise ValueError("line_styles length must match len(hrtfs)")

    theta_values_by_subject: list[np.ndarray] = []
    radial_values_by_subject: list[np.ndarray] = []
    sorted_itd_values_by_subject: list[np.ndarray] = []
    real_elevations: list[float] = []

    for subject_index, hrtf in enumerate(hrtfs):
        if hrtf.IR.values is None:
            raise ValueError(f"HRTF at index {subject_index} does not contain IR data")
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"HRTF at index {subject_index} requires IR sample_rate")

        absolute_itd_values = np.asarray(
            itd(
                hrtf,
                output="time",
                absolute=True,
            ),
            dtype=float,
        )
        theta_values, radial_values, sorted_itd_values, real_elevation = create_horizontal_plane_curve(
            hrtf=hrtf,
            values=absolute_itd_values,
            plane_angle=resolved_plane_angle,
        )
        theta_values_by_subject.append(theta_values)
        radial_values_by_subject.append(radial_values)
        sorted_itd_values_by_subject.append(np.asarray(sorted_itd_values, dtype=float).reshape(-1))
        real_elevations.append(float(real_elevation))

    reference_real_elevation = real_elevations[0]
    for subject_index in range(1, hrtf_count):
        compared_real_elevation = real_elevations[subject_index]
        if not np.isclose(
            compared_real_elevation,
            reference_real_elevation,
            atol=1e-8,
            rtol=0.0,
        ):
            warn_user(
                (
                    "compare_absolute_itd resolved different horizontal plane elevations: "
                    f"subject_1={reference_real_elevation:.6f} vs "
                    f"subject_{subject_index + 1}={compared_real_elevation:.6f}"
                ),
                category=HRTFPyKitWarning,
            )

    figure = Figure(
        Layout_1(
            figsize=(6, 7),
            margins=Margins(),
        ),
        projection="polar",
    )
    ax = figure.get_ax("main")

    for subject_index in range(hrtf_count):
        figure.create_two_dimension(
            ax=ax,
            x=theta_values_by_subject[subject_index],
            y=radial_values_by_subject[subject_index],
            color=resolved_line_colors[subject_index],
            linestyle=resolved_line_styles[subject_index],
            linewidth=2.0,
        )

    if len(sorted_itd_values_by_subject) == 0:
        combined_sorted_itd_values = np.array([], dtype=float)
    else:
        combined_sorted_itd_values = np.concatenate(sorted_itd_values_by_subject)

    AzimuthAnglesAxisPolarProjection.apply(
        ax=ax,
        tick_step=30.0,
    )
    RadialAxisPolarProjection.apply(
        ax=ax,
        radial_values=combined_sorted_itd_values,
        radial_label_default=Labels.itd_time if show_labels else "",
        tick_step=200.0,
        tick_label_style="integer",
        label_position=350.0,
        label=None if show_labels else "",
    )
    resolved_legend_location = Subjects.location if legend_location is None else str(legend_location)
    resolved_legend_bbox_to_anchor = (
        (1.08, 1.08)
        if legend_bbox_to_anchor is None
        else (
            float(legend_bbox_to_anchor[0]),
            float(legend_bbox_to_anchor[1]),
        )
    )
    if show_legends:
        Subjects.apply(
            ax=ax,
            labels=resolved_legends,
            location=resolved_legend_location,
            bbox_to_anchor=resolved_legend_bbox_to_anchor,
        )
    ax.grid(True)

    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane="horizontal",
                plane_angle=reference_real_elevation,
            ),
        )
    if show:
        plt.show()
    return figure.fig


def compare_absolute_ild(
    hrtfs: list["HRTF"],
    plane_angle: float = 0.0,
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Compare absolute broad band ILD curves from several HRTFs.

    ``compare_absolute_ild`` calls :func:`~hrtfpykit.hrtf.ild` for each HRTF as ``ild(hrtf,
    mode="broad-band", epsilon=1e-12, absolute=True)``. It selects the nearest measured
    horizontal plane for each HRTF, sorts each polar curve by azimuth, and overlays the
    curves in one figure.

    Each HRTF resolves the requested horizontal plane independently. If compared HRTFs
    resolve to different real elevations, the function emits an
    :class:`~hrtfpykit.utils.warnings.HRTFPyKitWarning`.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least
        2 and at most 5 entries. Every object must contain IR data.
    plane_angle : float, default=0.0
        Requested horizontal plane elevation in degrees. The nearest available elevation
        is selected separately for each HRTF.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses the default color cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Legend anchor tuple (x, y). Defaults to (1.08, 1.08).
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the overlaid absolute broad band ILD polar curves.

    Raises
    ------
    ValueError
        If the HRTF list length, plane_angle, legend/style lengths, or IR availability
        is invalid.

    Warns
    -----
    HRTFPyKitWarning
        If HRTFs resolve the requested horizontal plane to different real elevations.

    Notes
    -----
    The polar angular axis uses spherical azimuth in the SOFA convention. The radial axis
    uses ILD dB units and integer tick labels.

    Examples
    --------
    Compare absolute broad band ILD on the horizontal plane for two HRTFs:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_absolute_ild
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_absolute_ild(
    ...     [hrtf_a, hrtf_b],
    ...     plane_angle=0.0,
    ...     legends=["P0001", "P0002"],
    ...     line_styles=["-", "--"],
    ... )
    """
    if not isinstance(hrtfs, list):
        raise ValueError("hrtfs must be a list[HRTF]")
    hrtf_count = len(hrtfs)
    if hrtf_count < 2:
        raise ValueError("compare_absolute_ild requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_absolute_ild accepts up to 5 HRTFs")
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    resolved_plane_angle = float(plane_angle)
    if not np.isfinite(resolved_plane_angle):
        raise ValueError("plane_angle must be a finite value")

    if legends is None:
        resolved_legends = Subjects.create_default_labels(hrtf_count)
    else:
        resolved_legends = [str(value) for value in legends]
        if len(resolved_legends) != hrtf_count:
            raise ValueError("legends length must match len(hrtfs)")

    if line_colors is None:
        default_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_values = (
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
            if default_colors is None
            else default_colors.by_key().get("color", ["tab:blue"])
        )
        if len(color_values) == 0:
            color_values = ["tab:blue"]
        resolved_line_colors = [color_values[index % len(color_values)] for index in range(hrtf_count)]
    else:
        resolved_line_colors = list(line_colors)
        if len(resolved_line_colors) != hrtf_count:
            raise ValueError("line_colors length must match len(hrtfs)")

    if line_styles is None:
        resolved_line_styles = ["-"] * hrtf_count
    else:
        resolved_line_styles = [str(value) for value in line_styles]
        if len(resolved_line_styles) != hrtf_count:
            raise ValueError("line_styles length must match len(hrtfs)")

    theta_values_by_subject: list[np.ndarray] = []
    radial_values_by_subject: list[np.ndarray] = []
    sorted_ild_values_by_subject: list[np.ndarray] = []
    real_elevations: list[float] = []

    for subject_index, hrtf in enumerate(hrtfs):
        if hrtf.IR.values is None:
            raise ValueError(f"HRTF at index {subject_index} does not contain IR data")
        absolute_ild_values = np.asarray(
            ild(
                hrtf,
                mode="broad-band",
                absolute=True,
            ),
            dtype=float,
        )
        theta_values, radial_values, sorted_ild_values, real_elevation = create_horizontal_plane_curve(
            hrtf=hrtf,
            values=absolute_ild_values,
            plane_angle=resolved_plane_angle,
        )
        theta_values_by_subject.append(theta_values)
        radial_values_by_subject.append(radial_values)
        sorted_ild_values_by_subject.append(np.asarray(sorted_ild_values, dtype=float).reshape(-1))
        real_elevations.append(float(real_elevation))

    reference_real_elevation = real_elevations[0]
    for subject_index in range(1, hrtf_count):
        compared_real_elevation = real_elevations[subject_index]
        if not np.isclose(
            compared_real_elevation,
            reference_real_elevation,
            atol=1e-8,
            rtol=0.0,
        ):
            warn_user(
                (
                    "compare_absolute_ild resolved different horizontal plane elevations: "
                    f"subject_1={reference_real_elevation:.6f} vs "
                    f"subject_{subject_index + 1}={compared_real_elevation:.6f}"
                ),
                category=HRTFPyKitWarning,
            )

    figure = Figure(
        Layout_1(
            figsize=(6, 7),
            margins=Margins(),
        ),
        projection="polar",
    )
    ax = figure.get_ax("main")

    for subject_index in range(hrtf_count):
        figure.create_two_dimension(
            ax=ax,
            x=theta_values_by_subject[subject_index],
            y=radial_values_by_subject[subject_index],
            color=resolved_line_colors[subject_index],
            linestyle=resolved_line_styles[subject_index],
            linewidth=2.0,
        )

    if len(sorted_ild_values_by_subject) == 0:
        combined_sorted_ild_values = np.array([], dtype=float)
    else:
        combined_sorted_ild_values = np.concatenate(sorted_ild_values_by_subject)

    AzimuthAnglesAxisPolarProjection.apply(
        ax=ax,
        tick_step=30.0,
    )
    RadialAxisPolarProjection.apply(
        ax=ax,
        radial_values=combined_sorted_ild_values,
        radial_label_default=Labels.ild_db if show_labels else "",
        tick_step=5.0,
        tick_label_style="integer",
        label_position=350.0,
        label=None if show_labels else "",
    )
    resolved_legend_location = Subjects.location if legend_location is None else str(legend_location)
    resolved_legend_bbox_to_anchor = (
        (1.08, 1.08)
        if legend_bbox_to_anchor is None
        else (
            float(legend_bbox_to_anchor[0]),
            float(legend_bbox_to_anchor[1]),
        )
    )
    if show_legends:
        Subjects.apply(
            ax=ax,
            labels=resolved_legends,
            location=resolved_legend_location,
            bbox_to_anchor=resolved_legend_bbox_to_anchor,
        )
    ax.grid(True)

    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane="horizontal",
                plane_angle=reference_real_elevation,
            ),
        )
    if show:
        plt.show()
    return figure.fig


def compare_itd(
    hrtfs: list["HRTF"],
    plane_angle: float = 0.0,
    azimuth_range_mode: str = "0-360",
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Compare signed ITD curves from several HRTFs.

    ``compare_itd`` calls :func:`~hrtfpykit.hrtf.itd` for each HRTF as ``itd(hrtf,
    method="threshold", output="time", thresh_level=-10.0, upper_cut_freq=3000.0,
    filter_order=10, absolute=False)``. For every HRTF, it selects the nearest measured
    horizontal plane from that HRTF source grid, sorts sources by azimuth, and overlays
    signed ITD in microseconds.

    The plotted values keep the sign convention of :func:`~hrtfpykit.hrtf.itd`.
    ``azimuth_range_mode="-180-180"`` reverses the displayed x axis so listener left appears
    on the left side of the figure.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least
        2 and at most 5 entries. Every object must contain IR data and an IR sample
        rate.
    plane_angle : float, default=0.0
        Requested horizontal plane elevation in degrees. The nearest available elevation
        is selected separately for each HRTF.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention used on the x axis. ``0-360`` follows the SOFA anticlockwise
    convention: front is 0 degrees, listener left is 90 degrees, back is 180
        degrees, and listener right is 270 degrees. ``-180-180`` wraps listener right to
        -90 degrees and reverses the displayed x axis so listener left appears visually
        left.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses the default color cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Optional legend anchor tuple (x, y).
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the overlaid signed ITD curves.

    Raises
    ------
    ValueError
        If the HRTF list length, plane_angle, azimuth range, legend/style lengths, IR
        availability, sample rate availability, selected plane, or computed ITD shape is
        invalid.

    Warns
    -----
    HRTFPyKitWarning
        If HRTFs resolve the requested horizontal plane to different real elevations.

    Notes
    -----
    The plot uses one Cartesian axis and overlays all HRTFs in the same coordinate frame.
    Curves can have different azimuth sample locations when the HRTFs use different source
    grids.

    Examples
    --------
    Compare the signed ITD curve around the horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_itd
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_itd(
    ...     [hrtf_a, hrtf_b],
    ...     plane_angle=0.0,
    ...     legends=["P0001", "P0002"],
    ...     line_styles=["-", "--"],
    ... )
    """
    if not isinstance(hrtfs, list):
        raise ValueError("hrtfs must be a list[HRTF]")
    hrtf_count = len(hrtfs)
    if hrtf_count < 2:
        raise ValueError("compare_itd requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_itd accepts up to 5 HRTFs")
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    resolved_plane_angle = float(plane_angle)
    if not np.isfinite(resolved_plane_angle):
        raise ValueError("plane_angle must be a finite value")
    resolved_azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(
        range_mode=azimuth_range_mode,
    )

    if legends is None:
        resolved_legends = Subjects.create_default_labels(hrtf_count)
    else:
        resolved_legends = [str(value) for value in legends]
        if len(resolved_legends) != hrtf_count:
            raise ValueError("legends length must match len(hrtfs)")

    if line_colors is None:
        default_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_values = (
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
            if default_colors is None
            else default_colors.by_key().get("color", ["tab:blue"])
        )
        if len(color_values) == 0:
            color_values = ["tab:blue"]
        resolved_line_colors = [color_values[index % len(color_values)] for index in range(hrtf_count)]
    else:
        resolved_line_colors = list(line_colors)
        if len(resolved_line_colors) != hrtf_count:
            raise ValueError("line_colors length must match len(hrtfs)")

    if line_styles is None:
        resolved_line_styles = ["-"] * hrtf_count
    else:
        resolved_line_styles = [str(value) for value in line_styles]
        if len(resolved_line_styles) != hrtf_count:
            raise ValueError("line_styles length must match len(hrtfs)")

    sorted_azimuths_by_subject: list[np.ndarray] = []
    sorted_itd_by_subject: list[np.ndarray] = []
    real_elevations: list[float] = []

    for subject_index, hrtf in enumerate(hrtfs):
        if hrtf.IR.values is None:
            raise ValueError(f"HRTF at index {subject_index} does not contain IR data")
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"HRTF at index {subject_index} requires IR sample_rate")

        itd_values = np.asarray(
            itd(
                hrtf,
                output="time",
            ),
            dtype=float,
        ).reshape(-1)
        indices, real_elevation = get_horizontal_plane(
            hrtf=hrtf,
            plane_angle=resolved_plane_angle,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError(f"HRTF at index {subject_index} has no sources in selected horizontal plane")
        source_positions = hrtf.Sources.get_positions(angle_unit="degrees")
        spherical_positions = np.asarray(source_positions[indices], dtype=float)
        azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
        transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
            values=azimuth_values,
            range_mode=resolved_azimuth_range_mode,
        )
        if itd_values.shape[0] != source_positions.shape[0]:
            raise ValueError(f"ITD values must match source count for HRTF at index {subject_index}")
        horizontal_itd_values = itd_values[indices]
        sort_indices = np.argsort(transformed_azimuth_values)
        sorted_azimuths_by_subject.append(np.asarray(transformed_azimuth_values[sort_indices], dtype=float))
        sorted_itd_by_subject.append(np.asarray(horizontal_itd_values[sort_indices], dtype=float))
        real_elevations.append(float(real_elevation))

    reference_real_elevation = real_elevations[0]
    for subject_index in range(1, hrtf_count):
        compared_real_elevation = real_elevations[subject_index]
        if not np.isclose(compared_real_elevation, reference_real_elevation, atol=1e-8, rtol=0.0):
            warn_user(
                (
                    "compare_itd resolved different horizontal plane elevations: "
                    f"subject_1={reference_real_elevation:.6f} vs "
                    f"subject_{subject_index + 1}={compared_real_elevation:.6f}"
                ),
                category=HRTFPyKitWarning,
            )

    figure = Figure(
        Layout_1(
            figsize=Layout_1().figsize,
            margins=Margins(),
        )
    )
    ax = figure.get_ax("main")
    for subject_index in range(hrtf_count):
        figure.create_two_dimension(
            ax=ax,
            x=sorted_azimuths_by_subject[subject_index],
            y=sorted_itd_by_subject[subject_index],
            color=resolved_line_colors[subject_index],
            linestyle=resolved_line_styles[subject_index],
            linewidth=2.0,
        )

    AzimuthAnglesAxis.apply(
        ax=ax,
        axis="x",
        values=np.concatenate(sorted_azimuths_by_subject),
        range_mode=resolved_azimuth_range_mode,
        label=None if show_labels else "",
    )
    Axis.apply_label(
        ax=ax,
        axis="y",
        default_label=Labels.itd,
        label=None if show_labels else "",
    )
    resolved_legend_location = Subjects.location if legend_location is None else str(legend_location)
    resolved_legend_bbox_to_anchor = (
        None
        if legend_bbox_to_anchor is None
        else (
            float(legend_bbox_to_anchor[0]),
            float(legend_bbox_to_anchor[1]),
        )
    )
    if show_legends:
        Subjects.apply(
            ax=ax,
            labels=resolved_legends,
            location=resolved_legend_location,
            bbox_to_anchor=resolved_legend_bbox_to_anchor,
        )
    ax.grid(True)

    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane="horizontal",
                plane_angle=reference_real_elevation,
            ),
        )
    if show:
        plt.show()
    return figure.fig


def compare_ild(
    hrtfs: list["HRTF"],
    plane_angle: float = 0.0,
    azimuth_range_mode: str = "0-360",
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Compare signed broad band ILD curves from several HRTFs.

    ``compare_ild`` calls :func:`~hrtfpykit.hrtf.ild` for each HRTF as ``ild(hrtf,
    mode="broad-band", epsilon=1e-12, absolute=False)``. For every HRTF, it selects the
    nearest measured horizontal plane from that HRTF source grid, sorts sources by azimuth,
    and overlays signed ILD in dB.

    The plotted values keep the sign convention of :func:`~hrtfpykit.hrtf.ild`.
    ``azimuth_range_mode="-180-180"`` reverses the displayed x axis so listener left appears
    on the left side of the figure.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least
        2 and at most 5 entries. Every object must contain IR data.
    plane_angle : float, default=0.0
        Requested horizontal plane elevation in degrees. The nearest available elevation
        is selected separately for each HRTF.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention used on the x axis. ``0-360`` follows the SOFA anticlockwise
    convention: front is 0 degrees, listener left is 90 degrees, back is 180
        degrees, and listener right is 270 degrees. ``-180-180`` wraps listener right to
        -90 degrees and reverses the displayed x axis so listener left appears visually
        left.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses the default color cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Optional legend anchor tuple (x, y).
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If False, suppress generated figure titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the overlaid signed broad band ILD curves.

    Raises
    ------
    ValueError
        If the HRTF list length, plane_angle, azimuth range, legend/style lengths, IR
        availability, selected plane, or computed ILD shape is invalid.

    Warns
    -----
    HRTFPyKitWarning
        If HRTFs resolve the requested horizontal plane to different real elevations.

    Notes
    -----
    The plot uses one Cartesian axis and overlays all HRTFs in the same coordinate frame.
    Curves can have different azimuth sample locations when the HRTFs use different source
    grids.

    Examples
    --------
    Compare the signed ILD curve around the horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_ild
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_ild(
    ...     [hrtf_a, hrtf_b],
    ...     plane_angle=0.0,
    ...     legends=["P0001", "P0002"],
    ...     line_styles=["-", "--"],
    ... )
    """
    if not isinstance(hrtfs, list):
        raise ValueError("hrtfs must be a list[HRTF]")
    hrtf_count = len(hrtfs)
    if hrtf_count < 2:
        raise ValueError("compare_ild requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_ild accepts up to 5 HRTFs")
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    resolved_plane_angle = float(plane_angle)
    if not np.isfinite(resolved_plane_angle):
        raise ValueError("plane_angle must be a finite value")
    resolved_azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(
        range_mode=azimuth_range_mode,
    )

    if legends is None:
        resolved_legends = Subjects.create_default_labels(hrtf_count)
    else:
        resolved_legends = [str(value) for value in legends]
        if len(resolved_legends) != hrtf_count:
            raise ValueError("legends length must match len(hrtfs)")

    if line_colors is None:
        default_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_values = (
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
            if default_colors is None
            else default_colors.by_key().get("color", ["tab:blue"])
        )
        if len(color_values) == 0:
            color_values = ["tab:blue"]
        resolved_line_colors = [color_values[index % len(color_values)] for index in range(hrtf_count)]
    else:
        resolved_line_colors = list(line_colors)
        if len(resolved_line_colors) != hrtf_count:
            raise ValueError("line_colors length must match len(hrtfs)")

    if line_styles is None:
        resolved_line_styles = ["-"] * hrtf_count
    else:
        resolved_line_styles = [str(value) for value in line_styles]
        if len(resolved_line_styles) != hrtf_count:
            raise ValueError("line_styles length must match len(hrtfs)")

    sorted_azimuths_by_subject: list[np.ndarray] = []
    sorted_ild_by_subject: list[np.ndarray] = []
    real_elevations: list[float] = []

    for subject_index, hrtf in enumerate(hrtfs):
        if hrtf.IR.values is None:
            raise ValueError(f"HRTF at index {subject_index} does not contain IR data")
        ild_values = np.asarray(
            ild(
                hrtf,
                mode="broad-band",
            ),
            dtype=float,
        ).reshape(-1)
        indices, real_elevation = get_horizontal_plane(
            hrtf=hrtf,
            plane_angle=resolved_plane_angle,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError(f"HRTF at index {subject_index} has no sources in selected horizontal plane")
        source_positions = hrtf.Sources.get_positions(angle_unit="degrees")
        spherical_positions = np.asarray(source_positions[indices], dtype=float)
        azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
        transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
            values=azimuth_values,
            range_mode=resolved_azimuth_range_mode,
        )
        if ild_values.shape[0] != source_positions.shape[0]:
            raise ValueError(f"ILD values must match source count for HRTF at index {subject_index}")
        horizontal_ild_values = ild_values[indices]
        sort_indices = np.argsort(transformed_azimuth_values)
        sorted_azimuths_by_subject.append(np.asarray(transformed_azimuth_values[sort_indices], dtype=float))
        sorted_ild_by_subject.append(np.asarray(horizontal_ild_values[sort_indices], dtype=float))
        real_elevations.append(float(real_elevation))

    reference_real_elevation = real_elevations[0]
    for subject_index in range(1, hrtf_count):
        compared_real_elevation = real_elevations[subject_index]
        if not np.isclose(compared_real_elevation, reference_real_elevation, atol=1e-8, rtol=0.0):
            warn_user(
                (
                    "compare_ild resolved different horizontal plane elevations: "
                    f"subject_1={reference_real_elevation:.6f} vs "
                    f"subject_{subject_index + 1}={compared_real_elevation:.6f}"
                ),
                category=HRTFPyKitWarning,
            )

    figure = Figure(
        Layout_1(
            figsize=Layout_1().figsize,
            margins=Margins(),
        )
    )
    ax = figure.get_ax("main")
    for subject_index in range(hrtf_count):
        figure.create_two_dimension(
            ax=ax,
            x=sorted_azimuths_by_subject[subject_index],
            y=sorted_ild_by_subject[subject_index],
            color=resolved_line_colors[subject_index],
            linestyle=resolved_line_styles[subject_index],
            linewidth=2.0,
        )

    AzimuthAnglesAxis.apply(
        ax=ax,
        axis="x",
        values=np.concatenate(sorted_azimuths_by_subject),
        range_mode=resolved_azimuth_range_mode,
        label=None if show_labels else "",
    )
    Axis.apply_label(
        ax=ax,
        axis="y",
        default_label=Labels.ild if show_labels else "",
        label=None if show_labels else "",
    )
    resolved_legend_location = Subjects.location if legend_location is None else str(legend_location)
    resolved_legend_bbox_to_anchor = (
        None
        if legend_bbox_to_anchor is None
        else (
            float(legend_bbox_to_anchor[0]),
            float(legend_bbox_to_anchor[1]),
        )
    )
    if show_legends:
        Subjects.apply(
            ax=ax,
            labels=resolved_legends,
            location=resolved_legend_location,
            bbox_to_anchor=resolved_legend_bbox_to_anchor,
        )
    ax.grid(True)

    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.create_plane_title(
                plane="horizontal",
                plane_angle=reference_real_elevation,
            ),
        )
    if show:
        plt.show()
    return figure.fig


def compare_itd_difference(
    hrtf_reference: "HRTF",
    hrtfs: "HRTF | list[HRTF] | tuple[HRTF, ...]",
    method: str = "threshold",
    output: str = "time",
    thresh_level: float = -10.0,
    upper_cut_freq: float = 3000.0,
    filter_order: int = 10,
    absolute: bool = True,
    reduction_method: str = "mean",
    azimuth_range_mode: str = "0-360",
    plot_type: str = "heatmap",
    colormap: str = "jet",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot ITD difference values over the reference source grid.

    ``compare_itd_difference`` calls :func:`~hrtfpykit.hrtf.itd_difference` as
    ``itd_difference(hrtf_reference=hrtf_reference, hrtfs=hrtfs, method=method,
    output=output, thresh_level=thresh_level, upper_cut_freq=upper_cut_freq,
    filter_order=filter_order, absolute=absolute, reduction_axis="itds",
    reduction_method=reduction_method)``. The result is one ITD difference value per source
    position after reducing the compared HRTF axis when several HRTFs are provided.

    Source coordinates are read from ``hrtf_reference.Sources`` in spherical degrees.
    ``plot_type="scatter"`` colors the measured source positions directly, while
    ``plot_type="heatmap"`` interpolates the same source values onto an azimuth and
    elevation image grid.

    Parameters
    ----------
    hrtf_reference : HRTF
        Reference HRTF. It must provide IR data, an IR sample rate, and the source grid
        used for the plot coordinates.
    hrtfs : HRTF or sequence of HRTF
        Compared HRTF object or objects. Every compared HRTF must use the same source
        positions as ``hrtf_reference``. Several compared HRTFs are reduced into one
        source map with ``reduction_method``.
    method : {``threshold``, ``maxiacce``}, default=``threshold``
        ITD estimator passed to :func:`~hrtfpykit.hrtf.itd_difference`.
    output : {``time``, ``samples``}, default=``time``
        Unit used for ITD values and the colorbar label. ``time`` returns microseconds.
        ``samples`` returns sample offsets.
    thresh_level : float, default=-10.0
        Threshold offset passed to the threshold ITD estimator.
    upper_cut_freq : float, default=3000.0
        Low pass cutoff frequency in hertz used by the ITD estimator.
    filter_order : int, default=10
        Filter order used by the ITD estimator.
    absolute : bool, default=True
        Difference sign handling. ``True`` plots absolute ITD differences; ``False``
        plots signed ``compared - reference`` differences.
    reduction_method : {``mean``, ``rms``}, default=``mean``
        Method used to reduce the compared HRTF axis when ``hrtfs`` contains several
        HRTFs.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention applied to the x axis. ``0-360`` follows the SOFA
        anticlockwise convention: front is 0 degrees, listener left is 90 degrees, back
        is 180 degrees, and listener right is 270 degrees. ``-180-180`` wraps listener
        right to -90 degrees and reverses the displayed x axis so listener left appears
        visually left.
    plot_type : {``scatter``, ``heatmap``}, default=``heatmap``
        Source map renderer. ``scatter`` plots measured sources as colored markers.
        ``heatmap`` plots an interpolated azimuth/elevation color image.
    colormap : str, default=``jet``
        Colormap name used for source map coloring.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If True, adds the default figure title.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the ITD difference source map.

    Raises
    ------
    ValueError
        If ITD difference calculation fails, source positions are invalid, or the number
        of ITD difference values differs from the number of source positions.

    Examples
    --------
    Plot an interpolated ITD difference heatmap:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_itd_difference
    >>> hrtf_reference = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_compared = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_itd_difference(
    ...     hrtf_reference,
    ...     hrtf_compared,
    ...     method="threshold",
    ...     output="time",
    ...     absolute=True,
    ...     plot_type="heatmap",
    ...     colormap="viridis",
    ... )
    """
    difference_values = np.asarray(
        itd_difference(
            hrtf_reference=hrtf_reference,
            hrtfs=hrtfs,
            method=method,
            output=output,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
            absolute=absolute,
            reduction_axis="itds",
            reduction_method=reduction_method,
        ),
        dtype=float,
    ).reshape(-1)

    spherical_positions = np.asarray(
        get_source_positions(
            sources=hrtf_reference.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        ),
        dtype=float,
    )
    if spherical_positions.ndim != 2 or spherical_positions.shape[1] < 2:
        raise ValueError("Source positions must have shape (N, 3) in spherical coordinates")
    if spherical_positions.shape[0] != difference_values.shape[0]:
        raise ValueError("ITD difference values must match number of source positions")

    azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=azimuth_range_mode)
    azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
    elevation_values = np.asarray(spherical_positions[:, 1], dtype=float)
    transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
        values=azimuth_values,
        range_mode=azimuth_range_mode,
    )

    figure = Figure(
        Layout_1(
            figsize=(8, 6),
            margins=Margins(),
        )
    )
    ax = figure.get_ax("main")
    output_key = str(output).strip().lower()
    colorbar_label = (
        Labels.compare_itd_difference_time
        if output_key == "time"
        else Labels.compare_itd_difference_samples
    )
    resolved_plot_type = str(plot_type).strip().lower()
    if resolved_plot_type not in {"scatter", "heatmap"}:
        raise ValueError("plot_type accepts scatter or heatmap")
    value_min = float(np.min(difference_values))
    value_max = float(np.max(difference_values))
    source_map: Any
    if resolved_plot_type == "scatter":
        source_map = ax.scatter(
            transformed_azimuth_values,
            elevation_values,
            c=difference_values,
            cmap=colormap,
            s=32.0,
            edgecolors="black",
            linewidths=0.25,
            vmin=value_min,
            vmax=value_max,
        )
    else:
        source_coordinates = np.column_stack((transformed_azimuth_values, elevation_values))
        unique_coordinates, inverse_indices = np.unique(
            source_coordinates,
            axis=0,
            return_inverse=True,
        )
        if unique_coordinates.shape[0] < 3:
            raise ValueError("heatmap plot_type requires at least three source positions")
        heatmap_values = np.zeros(unique_coordinates.shape[0], dtype=float)
        heatmap_counts = np.zeros(unique_coordinates.shape[0], dtype=float)
        np.add.at(heatmap_values, inverse_indices, difference_values)
        np.add.at(heatmap_counts, inverse_indices, 1.0)
        heatmap_values = heatmap_values / heatmap_counts
        azimuth_grid_values = np.linspace(
            float(np.min(unique_coordinates[:, 0])),
            float(np.max(unique_coordinates[:, 0])),
            361,
            dtype=float,
        )
        elevation_grid_values = np.linspace(
            float(np.min(unique_coordinates[:, 1])),
            float(np.max(unique_coordinates[:, 1])),
            181,
            dtype=float,
        )
        pole_mask = np.ones(unique_coordinates.shape[0], dtype=bool)
        pole_coordinate_blocks: list[np.ndarray] = []
        pole_value_blocks: list[np.ndarray] = []
        max_elevation = float(np.max(unique_coordinates[:, 1]))
        max_elevation_indices = np.where(
            np.isclose(unique_coordinates[:, 1], max_elevation)
        )[0]
        if max_elevation_indices.size == 1:
            max_elevation_index = int(max_elevation_indices[0])
            pole_mask[max_elevation_index] = False
            pole_coordinate_blocks.append(
                np.column_stack(
                    (
                        azimuth_grid_values,
                        np.full(azimuth_grid_values.shape, max_elevation, dtype=float),
                    )
                )
            )
            pole_value_blocks.append(
                np.full(
                    azimuth_grid_values.shape,
                    heatmap_values[max_elevation_index],
                    dtype=float,
                )
            )
        min_elevation = float(np.min(unique_coordinates[:, 1]))
        min_elevation_indices = np.where(
            np.isclose(unique_coordinates[:, 1], min_elevation)
        )[0]
        if min_elevation_indices.size == 1 and not np.isclose(min_elevation, max_elevation):
            min_elevation_index = int(min_elevation_indices[0])
            pole_mask[min_elevation_index] = False
            pole_coordinate_blocks.append(
                np.column_stack(
                    (
                        azimuth_grid_values,
                        np.full(azimuth_grid_values.shape, min_elevation, dtype=float),
                    )
                )
            )
            pole_value_blocks.append(
                np.full(
                    azimuth_grid_values.shape,
                    heatmap_values[min_elevation_index],
                    dtype=float,
                )
            )
        interpolation_coordinates = np.vstack(
            [unique_coordinates[pole_mask], *pole_coordinate_blocks]
        )
        interpolation_values = np.concatenate([heatmap_values[pole_mask], *pole_value_blocks])
        azimuth_grid, elevation_grid = np.meshgrid(
            azimuth_grid_values,
            elevation_grid_values,
        )
        try:
            heatmap_grid = griddata(
                interpolation_coordinates,
                interpolation_values,
                (azimuth_grid, elevation_grid),
                method="linear",
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "heatmap plot_type requires at least three non-collinear source positions"
            ) from exc
        heatmap_grid = np.asarray(heatmap_grid, dtype=float)
        nearest_heatmap_grid = np.asarray(
            griddata(
                interpolation_coordinates,
                interpolation_values,
                (azimuth_grid, elevation_grid),
                method="nearest",
            ),
            dtype=float,
        )
        heatmap_grid = np.where(
            np.isnan(heatmap_grid),
            nearest_heatmap_grid,
            heatmap_grid,
        )
        source_map = ax.imshow(
            heatmap_grid,
            origin="lower",
            extent=(
                float(azimuth_grid_values[0]),
                float(azimuth_grid_values[-1]),
                float(elevation_grid_values[0]),
                float(elevation_grid_values[-1]),
            ),
            aspect="auto",
            interpolation="bicubic",
            cmap=colormap,
            vmin=value_min,
            vmax=value_max,
        )
    colorbar_size = f"{Heatmap.colorbar_fraction * 100.0:.1f}%"
    colorbar_axis = make_axes_locatable(ax).append_axes(
        Heatmap.colorbar_location,
        size=colorbar_size,
        pad=Heatmap.colorbar_pad,
    )
    figure.fig.colorbar(source_map, cax=colorbar_axis, label=colorbar_label if show_labels else "")
    AzimuthAnglesAxis.apply(
        ax=ax,
        axis="x",
        values=transformed_azimuth_values,
        range_mode=azimuth_range_mode,
        label=None if show_labels else "",
    )
    x_min = float(np.min(transformed_azimuth_values))
    x_max = float(np.max(transformed_azimuth_values))
    if resolved_plot_type == "scatter":
        x_span = x_max - x_min
        x_padding = 8.0 if np.isclose(x_span, 0.0) else max(8.0, 0.05 * x_span)
        x_lower = x_min - x_padding
        x_upper = x_max + x_padding
    else:
        x_lower = x_min
        x_upper = x_max
    if azimuth_range_mode == AzimuthAnglesAxis.azimuth_range_modes[1]:
        ax.set_xlim(x_upper, x_lower)
    else:
        ax.set_xlim(x_lower, x_upper)
    ElevationAnglesAxis.apply(
        ax=ax,
        axis="y",
        values=elevation_values,
        label=None if show_labels else "",
    )
    y_min = float(np.min(elevation_values))
    y_max = float(np.max(elevation_values))
    if resolved_plot_type == "scatter":
        y_span = y_max - y_min
        y_padding = 2.0 if np.isclose(y_span, 0.0) else max(2.0, 0.04 * y_span)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    else:
        ax.set_ylim(y_min, y_max)
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.compare_itd_difference,
        )
    if show:
        plt.show()
    return figure.fig


def compare_ild_difference(
    hrtf_reference: "HRTF",
    hrtfs: "HRTF | list[HRTF] | tuple[HRTF, ...]",
    epsilon: float = 1e-12,
    absolute: bool = True,
    reduction_method: str = "mean",
    azimuth_range_mode: str = "0-360",
    plot_type: str = "heatmap",
    colormap: str = "jet",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot broad band ILD difference values over the reference source grid.

    ``compare_ild_difference`` calls :func:`~hrtfpykit.hrtf.ild_difference` as
    ``ild_difference(hrtf_reference=hrtf_reference, hrtfs=hrtfs, mode="broad-band",
    epsilon=epsilon, absolute=absolute, reduction_axis="ilds",
    reduction_method=reduction_method)``. The result is one ILD difference value in dB per
    source position after reducing the compared HRTF axis when several HRTFs are provided.

    Source coordinates are read from ``hrtf_reference.Sources`` in spherical degrees.
    ``plot_type="scatter"`` colors the measured source positions directly, while
    ``plot_type="heatmap"`` interpolates the same source values onto an azimuth and
    elevation image grid.

    Parameters
    ----------
    hrtf_reference : HRTF
        Reference HRTF. It must provide IR data and the source grid used for the plot
        coordinates.
    hrtfs : HRTF or sequence of HRTF
        Compared HRTF object or objects. Every compared HRTF must use the same source
        positions as ``hrtf_reference``. Several compared HRTFs are reduced into one
        source map with ``reduction_method``.
    epsilon : float, default=1e-12
        Positive floor passed to :func:`~hrtfpykit.hrtf.ild_difference` before
        level ratio conversion.
    absolute : bool, default=True
        Difference sign handling. ``True`` plots absolute ILD differences; ``False``
        plots signed ``compared - reference`` differences.
    reduction_method : {``mean``, ``rms``}, default=``mean``
        Method used to reduce the compared HRTF axis when ``hrtfs`` contains several
        HRTFs.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention applied to the x axis. ``0-360`` follows the SOFA
        anticlockwise convention: front is 0 degrees, listener left is 90 degrees, back
        is 180 degrees, and listener right is 270 degrees. ``-180-180`` wraps listener
        right to -90 degrees and reverses the displayed x axis so listener left appears
        visually left.
    plot_type : {``scatter``, ``heatmap``}, default=``heatmap``
        Source map renderer. ``scatter`` plots measured sources as colored markers.
        ``heatmap`` plots an interpolated azimuth/elevation color image.
    colormap : str, default=``jet``
        Colormap name used for source map coloring.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If True, adds the default figure title.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the broad band ILD difference source map.

    Raises
    ------
    ValueError
        If ILD difference calculation fails, source positions are invalid, or the number
        of ILD difference values differs from the number of source positions.

    Examples
    --------
    Plot broad band ILD difference magnitudes as measured source markers:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_ild_difference
    >>> hrtf_reference = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_compared = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_ild_difference(
    ...     hrtf_reference,
    ...     hrtf_compared,
    ...     absolute=True,
    ...     plot_type="scatter",
    ...     colormap="plasma",
    ... )
    """
    difference_values = np.asarray(
        ild_difference(
            hrtf_reference=hrtf_reference,
            hrtfs=hrtfs,
            mode="broad-band",
            epsilon=epsilon,
            absolute=absolute,
            reduction_axis="ilds",
            reduction_method=reduction_method,
        ),
        dtype=float,
    ).reshape(-1)

    spherical_positions = np.asarray(
        get_source_positions(
            sources=hrtf_reference.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        ),
        dtype=float,
    )
    if spherical_positions.ndim != 2 or spherical_positions.shape[1] < 2:
        raise ValueError("Source positions must have shape (N, 3) in spherical coordinates")
    if spherical_positions.shape[0] != difference_values.shape[0]:
        raise ValueError("ILD difference values must match number of source positions")

    azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=azimuth_range_mode)
    azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
    elevation_values = np.asarray(spherical_positions[:, 1], dtype=float)
    transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
        values=azimuth_values,
        range_mode=azimuth_range_mode,
    )

    figure = Figure(
        Layout_1(
            figsize=(8, 6),
            margins=Margins(),
        )
    )
    ax = figure.get_ax("main")
    resolved_plot_type = str(plot_type).strip().lower()
    if resolved_plot_type not in {"scatter", "heatmap"}:
        raise ValueError("plot_type accepts scatter or heatmap")
    value_min = float(np.min(difference_values))
    value_max = float(np.max(difference_values))
    colorbar_label = Labels.compare_ild_difference_db
    source_map: Any
    if resolved_plot_type == "scatter":
        source_map = ax.scatter(
            transformed_azimuth_values,
            elevation_values,
            c=difference_values,
            cmap=colormap,
            s=32.0,
            edgecolors="black",
            linewidths=0.25,
            vmin=value_min,
            vmax=value_max,
        )
    else:
        source_coordinates = np.column_stack((transformed_azimuth_values, elevation_values))
        unique_coordinates, inverse_indices = np.unique(
            source_coordinates,
            axis=0,
            return_inverse=True,
        )
        if unique_coordinates.shape[0] < 3:
            raise ValueError("heatmap plot_type requires at least three source positions")
        heatmap_values = np.zeros(unique_coordinates.shape[0], dtype=float)
        heatmap_counts = np.zeros(unique_coordinates.shape[0], dtype=float)
        np.add.at(heatmap_values, inverse_indices, difference_values)
        np.add.at(heatmap_counts, inverse_indices, 1.0)
        heatmap_values = heatmap_values / heatmap_counts
        azimuth_grid_values = np.linspace(
            float(np.min(unique_coordinates[:, 0])),
            float(np.max(unique_coordinates[:, 0])),
            361,
            dtype=float,
        )
        elevation_grid_values = np.linspace(
            float(np.min(unique_coordinates[:, 1])),
            float(np.max(unique_coordinates[:, 1])),
            181,
            dtype=float,
        )
        pole_mask = np.ones(unique_coordinates.shape[0], dtype=bool)
        pole_coordinate_blocks: list[np.ndarray] = []
        pole_value_blocks: list[np.ndarray] = []
        max_elevation = float(np.max(unique_coordinates[:, 1]))
        max_elevation_indices = np.where(
            np.isclose(unique_coordinates[:, 1], max_elevation)
        )[0]
        if max_elevation_indices.size == 1:
            max_elevation_index = int(max_elevation_indices[0])
            pole_mask[max_elevation_index] = False
            pole_coordinate_blocks.append(
                np.column_stack(
                    (
                        azimuth_grid_values,
                        np.full(azimuth_grid_values.shape, max_elevation, dtype=float),
                    )
                )
            )
            pole_value_blocks.append(
                np.full(
                    azimuth_grid_values.shape,
                    heatmap_values[max_elevation_index],
                    dtype=float,
                )
            )
        min_elevation = float(np.min(unique_coordinates[:, 1]))
        min_elevation_indices = np.where(
            np.isclose(unique_coordinates[:, 1], min_elevation)
        )[0]
        if min_elevation_indices.size == 1 and not np.isclose(min_elevation, max_elevation):
            min_elevation_index = int(min_elevation_indices[0])
            pole_mask[min_elevation_index] = False
            pole_coordinate_blocks.append(
                np.column_stack(
                    (
                        azimuth_grid_values,
                        np.full(azimuth_grid_values.shape, min_elevation, dtype=float),
                    )
                )
            )
            pole_value_blocks.append(
                np.full(
                    azimuth_grid_values.shape,
                    heatmap_values[min_elevation_index],
                    dtype=float,
                )
            )
        interpolation_coordinates = np.vstack(
            [unique_coordinates[pole_mask], *pole_coordinate_blocks]
        )
        interpolation_values = np.concatenate([heatmap_values[pole_mask], *pole_value_blocks])
        azimuth_grid, elevation_grid = np.meshgrid(
            azimuth_grid_values,
            elevation_grid_values,
        )
        try:
            heatmap_grid = griddata(
                interpolation_coordinates,
                interpolation_values,
                (azimuth_grid, elevation_grid),
                method="linear",
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "heatmap plot_type requires at least three non-collinear source positions"
            ) from exc
        heatmap_grid = np.asarray(heatmap_grid, dtype=float)
        nearest_heatmap_grid = np.asarray(
            griddata(
                interpolation_coordinates,
                interpolation_values,
                (azimuth_grid, elevation_grid),
                method="nearest",
            ),
            dtype=float,
        )
        heatmap_grid = np.where(
            np.isnan(heatmap_grid),
            nearest_heatmap_grid,
            heatmap_grid,
        )
        source_map = ax.imshow(
            heatmap_grid,
            origin="lower",
            extent=(
                float(azimuth_grid_values[0]),
                float(azimuth_grid_values[-1]),
                float(elevation_grid_values[0]),
                float(elevation_grid_values[-1]),
            ),
            aspect="auto",
            interpolation="bicubic",
            cmap=colormap,
            vmin=value_min,
            vmax=value_max,
        )
    colorbar_size = f"{Heatmap.colorbar_fraction * 100.0:.1f}%"
    colorbar_axis = make_axes_locatable(ax).append_axes(
        Heatmap.colorbar_location,
        size=colorbar_size,
        pad=Heatmap.colorbar_pad,
    )
    figure.fig.colorbar(source_map, cax=colorbar_axis, label=colorbar_label if show_labels else "")
    AzimuthAnglesAxis.apply(
        ax=ax,
        axis="x",
        values=transformed_azimuth_values,
        range_mode=azimuth_range_mode,
        label=None if show_labels else "",
    )
    x_min = float(np.min(transformed_azimuth_values))
    x_max = float(np.max(transformed_azimuth_values))
    if resolved_plot_type == "scatter":
        x_span = x_max - x_min
        x_padding = 8.0 if np.isclose(x_span, 0.0) else max(8.0, 0.05 * x_span)
        x_lower = x_min - x_padding
        x_upper = x_max + x_padding
    else:
        x_lower = x_min
        x_upper = x_max
    if azimuth_range_mode == AzimuthAnglesAxis.azimuth_range_modes[1]:
        ax.set_xlim(x_upper, x_lower)
    else:
        ax.set_xlim(x_lower, x_upper)
    ElevationAnglesAxis.apply(
        ax=ax,
        axis="y",
        values=elevation_values,
        label=None if show_labels else "",
    )
    y_min = float(np.min(elevation_values))
    y_max = float(np.max(elevation_values))
    if resolved_plot_type == "scatter":
        y_span = y_max - y_min
        y_padding = 2.0 if np.isclose(y_span, 0.0) else max(2.0, 0.04 * y_span)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    else:
        ax.set_ylim(y_min, y_max)
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.compare_ild_difference,
        )
    if show:
        plt.show()
    return figure.fig


def compare_hrtf_difference(
    hrtf_reference: "HRTF",
    hrtfs: "HRTF | list[HRTF] | tuple[HRTF, ...]",
    metric: str = "lsd",
    ear: str = "left",
    frequencies: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
    frequency_bands: tuple[float, float] | list[tuple[float, float]] | tuple[tuple[float, float], ...] | np.ndarray | None = None,
    epsilon: float = 1e-12,
    reduction_method: str = "mean",
    azimuth_range_mode: str = "0-360",
    plot_type: str = "heatmap",
    colormap: str = "jet",
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> MplFigure:
    """Plot HRTF difference values over the reference source grid.

    ``compare_hrtf_difference`` calls :func:`~hrtfpykit.hrtf.hrtf_difference` as
    ``hrtf_difference(hrtf_reference=hrtf_reference, hrtfs=hrtfs, metric=metric, ear=ear,
    plane="all", frequencies=frequencies, frequency_bands=frequency_bands,
    reduction_axis=..., reduction_method=reduction_method, epsilon=epsilon)``.
    ``metric="lsd"`` computes log spectral distortion from ``TF`` data, ``metric="rmse"``
    and ``metric="mae"`` compute linear HRIR amplitude error from ``IR`` data, and
    ``metric="nrmse"`` computes reference normalized HRIR error in dB.

    Source coordinates are read from ``hrtf_reference.Sources`` in spherical degrees.
    ``plot_type="scatter"`` colors the measured source positions directly, while
    ``plot_type="heatmap"`` interpolates the same source values onto an azimuth and
    elevation image grid.

    Parameters
    ----------
    hrtf_reference : HRTF
        Reference HRTF. It must provide the source grid used for plot coordinates and
        the domain data required by ``metric``.
    hrtfs : HRTF or sequence of HRTF
        Compared HRTF object or objects. Every compared HRTF must use the same source
        positions and metric data shape as ``hrtf_reference``. Several compared HRTFs
        are reduced into one source map with ``reduction_method``.
    metric : {``lsd``, ``rmse``, ``nrmse``, ``mae``}, default=``lsd``
        HRTF difference metric plotted on the source map.
    ear : {``left``, ``right``, ``both``}, default=``left``
        Ear channel selection passed to :func:`~hrtfpykit.hrtf.hrtf_difference`.
        ``both`` computes both ears and reduces the ear axis before plotting.
    frequencies : float, sequence of float, numpy.ndarray, or None, default=None
        Frequency selector in hertz passed to :func:`~hrtfpykit.hrtf.hrtf_difference`
        for ``metric="lsd"``. Each requested value is mapped to the nearest available TF
        bin.
    frequency_bands : pair, sequence of pairs, numpy.ndarray, or None, default=None
        Inclusive frequency band or bands in hertz passed to
        :func:`~hrtfpykit.hrtf.hrtf_difference` for ``metric="lsd"``.
    epsilon : float, default=1e-12
        Positive floor passed to :func:`~hrtfpykit.hrtf.hrtf_difference`.
    reduction_method : {``mean``, ``rms``}, default=``mean``
        Method used to reduce compared HRTFs and, for ``ear="both"``, ears.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention applied to the x axis. ``0-360`` follows the SOFA
        anticlockwise convention: front is 0 degrees, listener left is 90 degrees, back
        is 180 degrees, and listener right is 270 degrees. ``-180-180`` wraps listener
        right to -90 degrees and reverses the displayed x axis so listener left appears
        visually left.
    plot_type : {``scatter``, ``heatmap``}, default=``heatmap``
        Source map renderer. ``scatter`` plots measured sources as colored markers.
        ``heatmap`` plots an interpolated azimuth/elevation color image.
    colormap : str, default=``jet``
        Colormap name used for source map coloring.
    show : bool, default=True
        If True, display the figure before returning.
    show_titles : bool, default=True
        If True, adds the default figure title.
    show_labels : bool, default=True
        If False, suppress generated axis labels and colorbar labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the selected HRTF difference source map.

    Raises
    ------
    ValueError
        If metric calculation fails, source positions are invalid, or the number of
        metric values differs from the number of source positions.

    Examples
    --------
    Plot a left ear LSD heatmap:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_hrtf_difference
    >>> hrtf_reference = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_compared = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_hrtf_difference(
    ...     hrtf_reference,
    ...     hrtf_compared,
    ...     metric="lsd",
    ...     ear="left",
    ...     plot_type="heatmap",
    ...     colormap="viridis",
    ... )
    """
    metric_key = str(metric).strip().lower()
    if metric_key not in {"lsd", "rmse", "nrmse", "mae"}:
        raise ValueError("metric must be one of: lsd, rmse, nrmse, mae")
    metric_colorbar_labels = {
        "lsd": Labels.compare_hrtf_difference_lsd_db,
        "rmse": Labels.compare_hrtf_difference_rmse,
        "nrmse": Labels.compare_hrtf_difference_nrmse_db,
        "mae": Labels.compare_hrtf_difference_mae,
    }
    metric_title_labels = {
        "lsd": "LSD",
        "rmse": "RMSE",
        "nrmse": "NRMSE",
        "mae": "MAE",
    }
    ear_key = str(ear).strip().lower()
    reduction_axis: str | tuple[str, str]
    if ear_key == "both":
        reduction_axis = ("differences", "ears")
    else:
        reduction_axis = "differences"
    difference_values = np.asarray(
        hrtf_difference(
            hrtf_reference=hrtf_reference,
            hrtfs=hrtfs,
            metric=metric_key,
            ear=ear_key,
            plane="all",
            frequencies=frequencies,
            frequency_bands=frequency_bands,
            reduction_axis=reduction_axis,
            reduction_method=reduction_method,
            epsilon=epsilon,
        ),
        dtype=float,
    ).reshape(-1)

    spherical_positions = np.asarray(
        get_source_positions(
            sources=hrtf_reference.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        ),
        dtype=float,
    )
    if spherical_positions.ndim != 2 or spherical_positions.shape[1] < 2:
        raise ValueError("Source positions must have shape (N, 3) in spherical coordinates")
    if spherical_positions.shape[0] != difference_values.shape[0]:
        raise ValueError("HRTF difference values must match number of source positions")

    azimuth_range_mode = AzimuthAnglesAxis.get_range_mode(range_mode=azimuth_range_mode)
    azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
    elevation_values = np.asarray(spherical_positions[:, 1], dtype=float)
    transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
        values=azimuth_values,
        range_mode=azimuth_range_mode,
    )

    figure = Figure(
        Layout_1(
            figsize=(8, 6),
            margins=Margins(),
        )
    )
    ax = figure.get_ax("main")
    resolved_plot_type = str(plot_type).strip().lower()
    if resolved_plot_type not in {"scatter", "heatmap"}:
        raise ValueError("plot_type accepts scatter or heatmap")
    value_min = float(np.min(difference_values))
    value_max = float(np.max(difference_values))
    colorbar_label = metric_colorbar_labels[metric_key]
    source_map: Any
    if resolved_plot_type == "scatter":
        source_map = ax.scatter(
            transformed_azimuth_values,
            elevation_values,
            c=difference_values,
            cmap=colormap,
            s=32.0,
            edgecolors="black",
            linewidths=0.25,
            vmin=value_min,
            vmax=value_max,
        )
    else:
        source_coordinates = np.column_stack((transformed_azimuth_values, elevation_values))
        unique_coordinates, inverse_indices = np.unique(
            source_coordinates,
            axis=0,
            return_inverse=True,
        )
        if unique_coordinates.shape[0] < 3:
            raise ValueError("heatmap plot_type requires at least three source positions")
        heatmap_values = np.zeros(unique_coordinates.shape[0], dtype=float)
        heatmap_counts = np.zeros(unique_coordinates.shape[0], dtype=float)
        np.add.at(heatmap_values, inverse_indices, difference_values)
        np.add.at(heatmap_counts, inverse_indices, 1.0)
        heatmap_values = heatmap_values / heatmap_counts
        azimuth_grid_values = np.linspace(
            float(np.min(unique_coordinates[:, 0])),
            float(np.max(unique_coordinates[:, 0])),
            361,
            dtype=float,
        )
        elevation_grid_values = np.linspace(
            float(np.min(unique_coordinates[:, 1])),
            float(np.max(unique_coordinates[:, 1])),
            181,
            dtype=float,
        )
        pole_mask = np.ones(unique_coordinates.shape[0], dtype=bool)
        pole_coordinate_blocks: list[np.ndarray] = []
        pole_value_blocks: list[np.ndarray] = []
        max_elevation = float(np.max(unique_coordinates[:, 1]))
        max_elevation_indices = np.where(
            np.isclose(unique_coordinates[:, 1], max_elevation)
        )[0]
        if max_elevation_indices.size == 1:
            max_elevation_index = int(max_elevation_indices[0])
            pole_mask[max_elevation_index] = False
            pole_coordinate_blocks.append(
                np.column_stack(
                    (
                        azimuth_grid_values,
                        np.full(azimuth_grid_values.shape, max_elevation, dtype=float),
                    )
                )
            )
            pole_value_blocks.append(
                np.full(
                    azimuth_grid_values.shape,
                    heatmap_values[max_elevation_index],
                    dtype=float,
                )
            )
        min_elevation = float(np.min(unique_coordinates[:, 1]))
        min_elevation_indices = np.where(
            np.isclose(unique_coordinates[:, 1], min_elevation)
        )[0]
        if min_elevation_indices.size == 1 and not np.isclose(min_elevation, max_elevation):
            min_elevation_index = int(min_elevation_indices[0])
            pole_mask[min_elevation_index] = False
            pole_coordinate_blocks.append(
                np.column_stack(
                    (
                        azimuth_grid_values,
                        np.full(azimuth_grid_values.shape, min_elevation, dtype=float),
                    )
                )
            )
            pole_value_blocks.append(
                np.full(
                    azimuth_grid_values.shape,
                    heatmap_values[min_elevation_index],
                    dtype=float,
                )
            )
        interpolation_coordinates = np.vstack(
            [unique_coordinates[pole_mask], *pole_coordinate_blocks]
        )
        interpolation_values = np.concatenate([heatmap_values[pole_mask], *pole_value_blocks])
        azimuth_grid, elevation_grid = np.meshgrid(
            azimuth_grid_values,
            elevation_grid_values,
        )
        try:
            heatmap_grid = griddata(
                interpolation_coordinates,
                interpolation_values,
                (azimuth_grid, elevation_grid),
                method="linear",
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "heatmap plot_type requires at least three non-collinear source positions"
            ) from exc
        heatmap_grid = np.asarray(heatmap_grid, dtype=float)
        nearest_heatmap_grid = np.asarray(
            griddata(
                interpolation_coordinates,
                interpolation_values,
                (azimuth_grid, elevation_grid),
                method="nearest",
            ),
            dtype=float,
        )
        heatmap_grid = np.where(
            np.isnan(heatmap_grid),
            nearest_heatmap_grid,
            heatmap_grid,
        )
        source_map = ax.imshow(
            heatmap_grid,
            origin="lower",
            extent=(
                float(azimuth_grid_values[0]),
                float(azimuth_grid_values[-1]),
                float(elevation_grid_values[0]),
                float(elevation_grid_values[-1]),
            ),
            aspect="auto",
            interpolation="bicubic",
            cmap=colormap,
            vmin=value_min,
            vmax=value_max,
        )
    colorbar_size = f"{Heatmap.colorbar_fraction * 100.0:.1f}%"
    colorbar_axis = make_axes_locatable(ax).append_axes(
        Heatmap.colorbar_location,
        size=colorbar_size,
        pad=Heatmap.colorbar_pad,
    )
    figure.fig.colorbar(source_map, cax=colorbar_axis, label=colorbar_label if show_labels else "")
    AzimuthAnglesAxis.apply(
        ax=ax,
        axis="x",
        values=transformed_azimuth_values,
        range_mode=azimuth_range_mode,
        label=None if show_labels else "",
    )
    x_min = float(np.min(transformed_azimuth_values))
    x_max = float(np.max(transformed_azimuth_values))
    if resolved_plot_type == "scatter":
        x_span = x_max - x_min
        x_padding = 8.0 if np.isclose(x_span, 0.0) else max(8.0, 0.05 * x_span)
        x_lower = x_min - x_padding
        x_upper = x_max + x_padding
    else:
        x_lower = x_min
        x_upper = x_max
    if azimuth_range_mode == AzimuthAnglesAxis.azimuth_range_modes[1]:
        ax.set_xlim(x_upper, x_lower)
    else:
        ax.set_xlim(x_lower, x_upper)
    ElevationAnglesAxis.apply(
        ax=ax,
        axis="y",
        values=elevation_values,
        label=None if show_labels else "",
    )
    y_min = float(np.min(elevation_values))
    y_max = float(np.max(elevation_values))
    if resolved_plot_type == "scatter":
        y_span = y_max - y_min
        y_padding = 2.0 if np.isclose(y_span, 0.0) else max(2.0, 0.04 * y_span)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    else:
        ax.set_ylim(y_min, y_max)
    if show_titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            f"{Titles.compare_hrtf_difference}: {metric_title_labels[metric_key]}",
        )
    if show:
        plt.show()
    return figure.fig
