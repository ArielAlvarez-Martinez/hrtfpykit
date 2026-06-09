from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np

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
from ..utils.warnings import HRTFPyKitWarning, warn_user
from ..utils.coordinates import get_position_queries, get_source_positions
from ..utils.dsp import magnitude_to_db
from ..utils.metrics import ild, abs_ild_diff, itd, lsd
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
    titles: bool = True,
) -> None:
    """Compare HRTF magnitude responses across multiple :class:`~hrtfpykit.hrtf.HRTF` objects.

    The function overlays one magnitude curve per HRTF for each requested
    source position. It is designed for comparing subjects, measurements,
    model outputs, or processing pipelines that share a broadly comparable
    source grid and frequency range. Each position query is resolved against
    every HRTF independently, then the real resolved positions are checked so
    the caller is warned when a named or numeric query resolves to different
    source coordinates across subjects.

    Frequency limits are resolved across all provided HRTFs. If freq_min or
    freq_max is omitted, the plotted range is restricted to the overlapping
    frequency span available in every input. Frequency values are shown in kHz
    on the x-axis, while freq_min and freq_max are specified in Hz to
    match the stored frequency bins.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least 2 and at most 5
        entries. Every object must contain frequency-domain data and frequency
        bins.
    positions : str | list | tuple | np.ndarray, default=(``front``,)
        Position query or collection of position queries. Up to 4 positions are
        accepted. Query resolution uses each HRTF's nearest available source in
        spherical coordinates.
    ear : {``left``, ``right``, ``both``}, default=``left``
        Ear channel selection. ``both`` requires exactly one position and
        creates separate left-ear and right-ear subplots.
    x_axis : {``linear``, ``log``}, default=``linear``
        Frequency-axis scale used for all subplots.
    unit : {``db``, ``linear``}, default=``db``
        Magnitude representation. ``db`` converts magnitudes with
        magnitude-to-decibel conversion; ``linear`` plots raw
        magnitudes.
    reference : float | str, default=1.0
        Reference used when ``unit`` is ``db``. ``max`` normalizes all plotted
        curves to the maximum selected magnitude over the requested positions,
        frequency range, and ear selection.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses Matplotlib default cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right`` when ``x_axis`` is ``linear``
        and ``upper left`` when ``x_axis`` is ``log``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Optional legend anchor tuple (x, y).
    freq_min : float | None, default=None
        Minimum frequency in Hz. If omitted, resolved from all HRTFs.
    freq_max : float | None, default=None
        Maximum frequency in Hz. If omitted, resolved from all HRTFs.
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        Controls subplot titles in single-ear mode.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the HRTF list length, option values, legend/style lengths, requested
        positions, frequency range, TF availability, frequency bins, or ear
        channels are invalid.

    Warns
    -----
    HRTFPyKitWarning
        If the same position query resolves to different real source
        coordinates in different HRTFs.

    Notes
    -----
    One single-ear position uses :class:`~hrtfpykit.plots.layouts.Layout_1`,
    two positions use :class:`~hrtfpykit.plots.layouts.Layout_2Vertical`, and
    three or four positions use :class:`~hrtfpykit.plots.layouts.Layout_3`.
    When ``ear`` is ``both``, the function uses
    :class:`~hrtfpykit.plots.layouts.Layout_2Horizontal` and one position,
    placing left-ear and right-ear comparisons side by side.

    Examples
    --------
    Compare left-ear magnitude responses from two SOFA files at the front
    direction, using a logarithmic frequency axis and a shared dB reference:

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
        resolved_line_colors = [str(color_values[index % len(color_values)]) for index in range(hrtf_count)]
    else:
        resolved_line_colors = [str(value) for value in line_colors]
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
                label=Labels.frequency,
                config=resolved_frequency_axis,
            )
            MagnitudeAxis.apply(ax=ax, axis="y", unit=unit)
            Titles.create_subplots_titles(ax=ax, title=subplot_title)
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
                label=Labels.frequency,
                config=resolved_frequency_axis,
            )
            MagnitudeAxis.apply(ax=ax, axis="y", unit=unit)
            if titles:
                Titles.create_subplots_titles(
                    ax=ax,
                    title=Titles.create_position_title(
                        selected_positions=reference_positions[position_index],
                    ),
                )
            else:
                Titles.create_subplots_titles(ax=ax, title="")
            if Figure.shared_x_visible:
                ax.tick_params(axis="x", which="both", labelbottom=True)
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
    titles: bool = True,
) -> None:
    """Compare HRIR amplitude curves across multiple HRTF instances.

    The function overlays one time-domain impulse-response waveform per HRTF
    for each requested source position. It is intended for comparing HRIR shape,
    delay, windowing, or preprocessing effects across subjects or pipeline
    outputs. Each position query is resolved independently for every HRTF, and a
    warning is emitted when the real resolved source coordinate differs across
    inputs.

    The horizontal axis can show seconds or sample indices. Time mode requires
    every input HRTF to provide
    :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>`; sample mode
    only requires the impulse-response array.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must
        contain at least 2 and at most 5 entries. Every object must contain
        time-domain IR data.
    positions : str | list | tuple | np.ndarray, default=(``front``,)
        Position query or collection of position queries. Up to 4 positions are
        accepted. Query resolution uses each HRTF's nearest available source in
        spherical coordinates.
    ear : {``left``, ``right``, ``both``}, default=``left``
        Ear channel selection. ``both`` requires exactly one position and
        creates separate left-ear and right-ear subplots.
    x_axis : {``time``, ``samples``}, default=``time``
        Horizontal axis mode for waveforms. ``time`` converts samples to
        seconds using each HRTF's sample rate.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses Matplotlib default cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Optional legend anchor tuple (x, y).
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        Controls subplot titles in single-ear mode.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the HRTF list length, option values, legend/style lengths, requested
        positions, IR availability, sample-rate requirements, IR shape, or ear
        channels are invalid.

    Warns
    -----
    HRTFPyKitWarning
        If the same position query resolves to different real source
        coordinates in different HRTFs.

    Notes
    -----
    The layout selection mirrors
    :func:`~hrtfpykit.plots.compare_magnitude`: one single-ear
    position uses :class:`~hrtfpykit.plots.layouts.Layout_1`, two positions use :class:`~hrtfpykit.plots.layouts.Layout_2Vertical`, three or
    four positions use :class:`~hrtfpykit.plots.layouts.Layout_3`, and ear=``both`` uses
    :class:`~hrtfpykit.plots.layouts.Layout_2Horizontal` for one position.

    Examples
    --------
    Compare left-ear front-direction impulse responses for two HRTFs, using
    sample indices on the x-axis:

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
        resolved_line_colors = [str(color_values[index % len(color_values)]) for index in range(hrtf_count)]
    else:
        resolved_line_colors = [str(value) for value in line_colors]
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
            x_values_by_subject.append(subject_sample_indexes / float(cast(Any, hrtf.IR.sample_rate)))
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
                TimeAxis.apply(ax=ax, axis="x")
            else:
                SampleAxis.apply(ax=ax, axis="x")
            AmplitudeAxis.apply(ax=ax, axis="y")
            Titles.create_subplots_titles(ax=ax, title=subplot_title)
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
                TimeAxis.apply(ax=ax, axis="x")
            else:
                SampleAxis.apply(ax=ax, axis="x")
            AmplitudeAxis.apply(ax=ax, axis="y")
            if titles:
                Titles.create_subplots_titles(
                    ax=ax,
                    title=Titles.create_position_title(
                        selected_positions=reference_positions[position_index],
                    ),
                )
            else:
                Titles.create_subplots_titles(ax=ax, title="")
            if Figure.shared_x_visible:
                ax.tick_params(axis="x", which="both", labelbottom=True)
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


def compare_abs_itd(
    hrtfs: list["HRTF"],
    plane_angle: float = 0.0,
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    titles: bool = True,
) -> None:
    """Compare absolute ITD polar curves across multiple HRTF instances.

    This function computes broad-band interaural time difference for each HRTF,
    converts it to an absolute magnitude in seconds, extracts the nearest
    horizontal-plane curve to ``plane_angle``, and overlays one polar trace
    per HRTF. It is intended for comparing timing-cue magnitude across subjects,
    measured datasets, or processing pipelines without preserving ITD sign.

    Each HRTF resolves the requested horizontal plane independently. The figure
    title uses the real resolved elevation from the first HRTF, and the function
    warns if other inputs resolve to a different elevation.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must
        contain at least 2 and at most 5 entries. Every object must contain IR
        data and an IR sample rate.
    plane_angle : float, default=0.0
        Requested horizontal-plane elevation in degrees. The nearest available
        elevation is selected separately for each HRTF.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses Matplotlib default cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Legend anchor tuple (x, y). Defaults to (1.08, 1.08).
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        If True, renders a figure title using the resolved elevation from
        the first HRTF.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the HRTF list length, plane_angle, legend/style lengths, IR
        availability, or sample-rate availability are invalid.

    Warns
    -----
    HRTFPyKitWarning
        If HRTFs resolve the requested horizontal plane to different real
        elevations.

    Notes
    -----
    The polar theta axis uses a north-up orientation with 30-degree ticks. The
    radial axis uses Labels.itd_seconds and a decimal tick style suitable
    for small ITD values.

    Examples
    --------
    Compare absolute ITD on the horizontal plane for two HRTFs:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_abs_itd
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_abs_itd(
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
        raise ValueError("compare_abs_itd requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_abs_itd accepts up to 5 HRTFs")
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
        resolved_line_colors = [str(color_values[index % len(color_values)]) for index in range(hrtf_count)]
    else:
        resolved_line_colors = [str(value) for value in line_colors]
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

        absolute_itd_values = np.abs(
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
                    "compare_abs_itd resolved different horizontal-plane elevations: "
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
        radial_label_default=Labels.itd_seconds,
        tick_step=2e-4,
        tick_label_style="decimal_comma_4",
        label_position=350.0,
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
    Subjects.apply(
        ax=ax,
        labels=resolved_legends,
        location=resolved_legend_location,
        bbox_to_anchor=resolved_legend_bbox_to_anchor,
    )
    ax.grid(True)

    if titles:
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


def compare_abs_bb_ild(
    hrtfs: list["HRTF"],
    plane_angle: float = 0.0,
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    titles: bool = True,
) -> None:
    """Compare absolute ILD polar curves across multiple HRTF instances.

    This function computes broad-band interaural level difference for each HRTF,
    converts it to an absolute magnitude in decibels, extracts the nearest
    horizontal-plane curve to ``plane_angle``, and overlays one polar trace
    per HRTF. It is intended for comparing level-cue magnitude across subjects,
    measured datasets, or processing pipelines without preserving ILD sign.

    Each HRTF resolves the requested horizontal plane independently. The figure
    title uses the real resolved elevation from the first HRTF, and the function
    warns if other inputs resolve to a different elevation.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must
        contain at least 2 and at most 5 entries. Every object must contain IR
        data and an IR sample rate.
    plane_angle : float, default=0.0
        Requested horizontal-plane elevation in degrees. The nearest available
        elevation is selected separately for each HRTF.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses Matplotlib default cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Legend anchor tuple (x, y). Defaults to (1.08, 1.08).
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        If True, renders a figure title using the resolved elevation from
        the first HRTF.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the HRTF list length, plane_angle, legend/style lengths, IR
        availability, or sample-rate availability are invalid.

    Warns
    -----
    HRTFPyKitWarning
        If HRTFs resolve the requested horizontal plane to different real
        elevations.

    Notes
    -----
    The polar theta axis uses a north-up orientation with 30-degree ticks. The
    radial axis uses Labels.ild_db and integer tick labels because ILD is
    displayed in decibels.

    Examples
    --------
    Compare absolute ILD on the horizontal plane for two HRTFs:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_abs_bb_ild
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_abs_bb_ild(
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
        raise ValueError("compare_abs_bb_ild requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_abs_bb_ild accepts up to 5 HRTFs")
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
        resolved_line_colors = [str(color_values[index % len(color_values)]) for index in range(hrtf_count)]
    else:
        resolved_line_colors = [str(value) for value in line_colors]
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
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"HRTF at index {subject_index} requires IR sample_rate")

        absolute_ild_values = np.abs(
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
                    "compare_abs_bb_ild resolved different horizontal-plane elevations: "
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
        radial_label_default=Labels.ild_db,
        tick_step=5.0,
        tick_label_style="integer",
        label_position=350.0,
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
    Subjects.apply(
        ax=ax,
        labels=resolved_legends,
        location=resolved_legend_location,
        bbox_to_anchor=resolved_legend_bbox_to_anchor,
    )
    ax.grid(True)

    if titles:
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


def compare_signed_itd(
    hrtfs: list["HRTF"],
    plane_angle: float = 0.0,
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    titles: bool = True,
) -> None:
    """Compare signed ITD curves across multiple HRTFs.

    This function computes broad-band signed ITD in seconds for each HRTF,
    extracts the nearest horizontal-plane slice to ``plane_angle``, sorts
    the slice by signed azimuth, and overlays one azimuth-versus-ITD line per
    HRTF. It is intended for inspecting timing-cue directionality across
    subjects, datasets, or processing pipelines.

    Unlike :func:`~hrtfpykit.plots.compare_abs_itd`, this function
    preserves ITD sign. The x-axis uses the signed -180 .. 180 azimuth
    convention.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least 2 and at most 5
        entries. Every object must contain IR data and an IR sample rate.
    plane_angle : float, default=0.0
        Requested horizontal-plane elevation in degrees. The nearest available
        elevation is selected separately for each HRTF.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses Matplotlib default cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Optional legend anchor tuple (x, y).
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        If True, renders a figure title using the resolved elevation from
        the first HRTF.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the HRTF list length, plane_angle, legend/style lengths, IR
        availability, sample-rate availability, selected plane, or computed ITD
        shape is invalid.

    Warns
    -----
    HRTFPyKitWarning
        If HRTFs resolve the requested horizontal plane to different real
        elevations.

    Notes
    -----
    The plot uses one Cartesian axis and overlays all subjects in the same
    coordinate frame. Curves may have different azimuth sample locations if the
    HRTFs use different source grids.

    Examples
    --------
    Compare the signed ITD curve around the horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_signed_itd
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_signed_itd(
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
        raise ValueError("compare_signed_itd requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_signed_itd accepts up to 5 HRTFs")
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
        resolved_line_colors = [str(color_values[index % len(color_values)]) for index in range(hrtf_count)]
    else:
        resolved_line_colors = [str(value) for value in line_colors]
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
                hrtf.IR,
                output="seconds",
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
            range_mode="-180-180",
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
                    "compare_signed_itd resolved different horizontal-plane elevations: "
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
        range_mode="-180-180",
    )
    Axis.apply_label(
        ax=ax,
        axis="y",
        default_label=Labels.itd,
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
    Subjects.apply(
        ax=ax,
        labels=resolved_legends,
        location=resolved_legend_location,
        bbox_to_anchor=resolved_legend_bbox_to_anchor,
    )
    ax.grid(True)

    if titles:
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


def compare_signed_bb_ild(
    hrtfs: list["HRTF"],
    plane_angle: float = 0.0,
    legends: list[str] | tuple[str, ...] | None = None,
    line_colors: list[str] | tuple[str, ...] | None = None,
    line_styles: list[str] | tuple[str, ...] | None = None,
    legend_location: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show: bool = True,
    titles: bool = True,
) -> None:
    """Compare signed ILD curves across multiple HRTFs.

    This function computes broad-band signed ILD in decibels for each HRTF,
    extracts the nearest horizontal-plane slice to ``plane_angle``, sorts
    the slice by signed azimuth, and overlays one azimuth-versus-ILD line per
    HRTF. It is intended for inspecting level-cue directionality across
    subjects, datasets, or processing pipelines.

    Unlike :func:`~hrtfpykit.plots.compare_abs_bb_ild`, this function
    preserves ILD sign. The x-axis uses the signed -180 .. 180 azimuth
    convention.

    Parameters
    ----------
    hrtfs : list[HRTF]
        :class:`~hrtfpykit.hrtf.HRTF` objects to compare. The list must contain at least 2 and at most 5
        entries. Every object must contain IR data and an IR sample rate.
    plane_angle : float, default=0.0
        Requested horizontal-plane elevation in degrees. The nearest available
        elevation is selected separately for each HRTF.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1`` through ``subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses Matplotlib default cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    legend_location : str | None, default=None
        Legend location. Defaults to ``upper right``.
    legend_bbox_to_anchor : tuple[float, float] | None, default=None
        Optional legend anchor tuple (x, y).
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        If True, renders a figure title using the resolved elevation from
        the first HRTF.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the HRTF list length, plane_angle, legend/style lengths, IR
        availability, sample-rate availability, selected plane, or computed ILD
        shape is invalid.

    Warns
    -----
    HRTFPyKitWarning
        If HRTFs resolve the requested horizontal plane to different real
        elevations.

    Notes
    -----
    The plot uses one Cartesian axis and overlays all subjects in the same
    coordinate frame. Curves may have different azimuth sample locations if the
    HRTFs use different source grids.

    Examples
    --------
    Compare the signed ILD curve around the horizontal plane:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import compare_signed_bb_ild
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> compare_signed_bb_ild(
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
        raise ValueError("compare_signed_bb_ild requires at least 2 HRTFs")
    if hrtf_count > 5:
        raise ValueError("compare_signed_bb_ild accepts up to 5 HRTFs")
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
        resolved_line_colors = [str(color_values[index % len(color_values)]) for index in range(hrtf_count)]
    else:
        resolved_line_colors = [str(value) for value in line_colors]
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
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"HRTF at index {subject_index} requires IR sample_rate")

        ild_values = np.asarray(
            ild(
                hrtf.IR,
                output="db",
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
            range_mode="-180-180",
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
                    "compare_signed_bb_ild resolved different horizontal-plane elevations: "
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
        range_mode="-180-180",
    )
    Axis.apply_label(
        ax=ax,
        axis="y",
        default_label=Labels.ild,
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
    Subjects.apply(
        ax=ax,
        labels=resolved_legends,
        location=resolved_legend_location,
        bbox_to_anchor=resolved_legend_bbox_to_anchor,
    )
    ax.grid(True)

    if titles:
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


def plot_abs_itd_diff(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    method: str = "threshold",
    output: str = "seconds",
    thresh_level: float = -10.0,
    upper_cut_freq: float = 3000.0,
    filter_order: int = 10,
    azimuth_range_mode: str = "0-360",
    colormap: str = "jet",
    show: bool = True,
    titles: bool = True,
) -> None:
    """Plot signed ITD differences between two HRTFs across source positions.

    The function computes per-position ITD for both inputs and plots the signed
    difference itd_a - itd_b as a color-coded spatial scatter map. Azimuth
    is shown on the x-axis, elevation is shown on the y-axis, and marker color
    encodes the timing difference in the requested output unit.

    This plot requires both HRTFs to expose the same source grid in the same
    order. It is useful for inspecting where a transformation, model, or
    measurement changes ITD and whether that change is localized to specific
    source directions.

    Parameters
    ----------
    hrtf_a : HRTF
        First HRTF used in the signed subtraction. Must contain IR data, an IR
        sample rate, and a source grid matching hrtf_b.
    hrtf_b : HRTF
        Second HRTF used in the signed subtraction. Must contain IR data, an IR
        sample rate, and a source grid matching hrtf_a.
    method : {``threshold``, ``maxiacce``}, default=``threshold``
        ITD estimator passed to :func:`~hrtfpykit.hrtf.metrics.itd`.
    output : {``seconds``, ``samples``}, default=``seconds``
        Unit of ITD values and colorbar label.
    thresh_level : float, default=-10.0
        Threshold offset in dB when ``method`` is ``threshold``.
    upper_cut_freq : float, default=3000.0
        Low-pass cutoff in Hz used before ITD estimation.
    filter_order : int, default=10
        Butterworth low-pass filter order used in ITD estimation.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``0-360``
        Azimuth convention applied on the x-axis.
    colormap : str, default=``jet``
        Matplotlib colormap name used for marker coloring.
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        If True, applies the figure title.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If either input is not HRTF-like, IR data or sample rates are missing,
        output is unsupported, sample output is requested with unequal
        sample rates, source grids differ, calculated ITD arrays have different
        shapes, source positions are invalid, or the ITD differences cannot be
        aligned with source positions.

    Notes
    -----
    This plotting function computes a signed difference directly. That differs
    from :func:`~hrtfpykit.hrtf.abs_itd_diff`, which returns absolute
    per-position differences. Positive and negative colors therefore retain the
    direction of hrtf_a - hrtf_b.

    Examples
    --------
    Plot the signed ITD difference between two HRTFs that share the same source
    grid:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_abs_itd_diff
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> plot_abs_itd_diff(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     method="threshold",
    ...     output="seconds",
    ...     azimuth_range_mode="-180-180",
    ...     colormap="viridis",
    ... )
    """
    for label, hrtf in (("hrtf_a", hrtf_a), ("hrtf_b", hrtf_b)):
        if not hasattr(hrtf, "IR") or not hasattr(hrtf, "Sources"):
            raise ValueError(f"{label} must be an HRTF instance")
        if hrtf.IR.values is None:
            raise ValueError(f"{label} IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"{label} IR sample_rate is required")

    output_key = str(output).strip().lower()
    if output_key not in {"seconds", "samples"}:
        raise ValueError("output must be one of: seconds, samples")
    if output_key == "samples" and not np.isclose(
        float(cast(Any, hrtf_a.IR.sample_rate)),
        float(cast(Any, hrtf_b.IR.sample_rate)),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("output='samples' requires equal sample_rate in both HRTFs")

    source_positions_a = np.asarray(hrtf_a.Sources.get_positions(angle_unit="degrees"), dtype=float)
    source_positions_b = np.asarray(hrtf_b.Sources.get_positions(angle_unit="degrees"), dtype=float)
    if source_positions_a.shape != source_positions_b.shape:
        raise ValueError("HRTFs must have the same number of source positions")
    if not np.allclose(source_positions_a, source_positions_b, atol=1e-8, rtol=0.0):
        raise ValueError("HRTFs must share the same source positions for ITD difference")

    itd_a = np.asarray(
        itd(
            hrtf_a.IR,
            method=method,
            output=output_key,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        ),
        dtype=float,
    )
    itd_b = np.asarray(
        itd(
            hrtf_b.IR,
            method=method,
            output=output_key,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        ),
        dtype=float,
    )
    if itd_a.shape != itd_b.shape:
        raise ValueError("Calculated ITD arrays must have matching shapes")
    difference_values = np.asarray(itd_a - itd_b, dtype=float).reshape(-1)

    spherical_positions = np.asarray(
        get_source_positions(
            sources=hrtf_a.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        ),
        dtype=float,
    )
    if spherical_positions.ndim != 2 or spherical_positions.shape[1] < 2:
        raise ValueError("Source positions must have shape (N, 3) in spherical coordinates")
    if spherical_positions.shape[0] != difference_values.shape[0]:
        raise ValueError("ITD difference values must match number of source positions")

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
    colorbar_label = (
        Labels.plot_abs_itd_diff_seconds
        if output_key == "seconds"
        else Labels.plot_abs_itd_diff_samples
    )
    scatter = ax.scatter(
        transformed_azimuth_values,
        elevation_values,
        c=difference_values,
        cmap=colormap,
        s=32.0,
        edgecolors="black",
        linewidths=0.25,
        vmin=float(np.min(difference_values)),
        vmax=float(np.max(difference_values)),
    )
    figure.fig.colorbar(scatter, ax=ax, label=colorbar_label)
    AzimuthAnglesAxis.apply(
        ax=ax,
        axis="x",
        values=transformed_azimuth_values,
        range_mode=azimuth_range_mode,
    )
    x_min = float(np.min(transformed_azimuth_values))
    x_max = float(np.max(transformed_azimuth_values))
    x_span = x_max - x_min
    x_padding = 8.0 if np.isclose(x_span, 0.0) else max(8.0, 0.05 * x_span)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ElevationAnglesAxis.apply(
        ax=ax,
        axis="y",
        values=elevation_values,
    )
    y_min = float(np.min(elevation_values))
    y_max = float(np.max(elevation_values))
    y_span = y_max - y_min
    y_padding = 2.0 if np.isclose(y_span, 0.0) else max(2.0, 0.04 * y_span)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    if titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.plot_abs_itd_diff,
        )
    if show:
        plt.show()


def plot_abs_bb_ild_diff(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    output: str = "db",
    epsilon: float = 1e-12,
    azimuth_range_mode: str = "-180-180",
    colormap: str = "jet",
    show: bool = True,
    titles: bool = True,
) -> None:
    """Plot absolute broad-band ILD differences across source positions.

    The function computes one broad-band ILD difference per shared source
    position using :func:`~hrtfpykit.hrtf.abs_ild_diff` with
    ``mode="broad-band"`` and displays those scalar values as a color-coded
    scatter map over azimuth and elevation. Marker color encodes the absolute
    broad-band ILD difference in the requested output representation.

    This plot is intentionally broad-band only. A frequency-dependent ILD
    comparison produces a position-by-frequency matrix, not one scalar per
    source position, and therefore belongs in a frequency-plane visualization
    rather than this source-grid map.

    Parameters
    ----------
    hrtf_a : HRTF
        First HRTF used for ILD comparison. Must contain IR data and a source
        grid compatible with hrtf_b.
    hrtf_b : HRTF
        Second HRTF used for ILD comparison. Must contain IR data and a source
        grid compatible with hrtf_a.
    output : {``db``, ``linear``}, default=``db``
        Broad-band ILD output representation and colorbar label style.
    epsilon : float, default=1e-12
        Positive floor passed to :func:`~hrtfpykit.hrtf.abs_ild_diff`.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``-180-180``
        Azimuth convention applied on the x-axis.
    colormap : str, default=``jet``
        Matplotlib colormap name used for marker coloring.
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        If True, applies the figure title.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If delegated broad-band ILD-difference calculation fails, source
        positions are invalid, or the returned ILD-difference values cannot be
        aligned with the number of source positions.

    Notes
    -----
    This function visualizes absolute broad-band ILD differences returned by
    :func:`~hrtfpykit.hrtf.abs_ild_diff`. Use the lower-level metric directly
    for frequency-dependent ILD difference arrays or signed left/right level
    changes.

    Examples
    --------
    Plot broad-band ILD differences between two HRTFs across the shared source
    grid:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_abs_bb_ild_diff
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> plot_abs_bb_ild_diff(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     output="db",
    ...     colormap="plasma",
    ... )
    """
    difference_values = np.asarray(
        abs_ild_diff(
            hrtf_a=hrtf_a,
            hrtf_b=hrtf_b,
            mode="broad-band",
            output=output,
            epsilon=epsilon,
        ),
        dtype=float,
    ).reshape(-1)

    spherical_positions = np.asarray(
        get_source_positions(
            sources=hrtf_a.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        ),
        dtype=float,
    )
    if spherical_positions.ndim != 2 or spherical_positions.shape[1] < 2:
        raise ValueError("Source positions must have shape (N, 3) in spherical coordinates")
    if spherical_positions.shape[0] != difference_values.shape[0]:
        raise ValueError("ILD difference values must match number of source positions")

    azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
    elevation_values = np.asarray(spherical_positions[:, 1], dtype=float)
    transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
        values=azimuth_values,
        range_mode=azimuth_range_mode,
    )

    figure = Figure(
        Layout_1(
            figsize=(12, 6),
            margins=Margins(),
        )
    )
    ax = figure.get_ax("main")
    output_key = str(output).strip().lower()
    colorbar_label = (
        Labels.plot_abs_bb_ild_diff_db
        if output_key == "db"
        else Labels.plot_abs_bb_ild_diff_linear
    )
    scatter = ax.scatter(
        transformed_azimuth_values,
        elevation_values,
        c=difference_values,
        cmap=colormap,
        s=32.0,
        edgecolors="black",
        linewidths=0.25,
        vmin=float(np.min(difference_values)),
        vmax=float(np.max(difference_values)),
    )
    figure.fig.colorbar(scatter, ax=ax, label=colorbar_label)
    AzimuthAnglesAxis.apply(
        ax=ax,
        axis="x",
        values=transformed_azimuth_values,
        range_mode=azimuth_range_mode,
    )
    x_min = float(np.min(transformed_azimuth_values))
    x_max = float(np.max(transformed_azimuth_values))
    x_span = x_max - x_min
    x_padding = 8.0 if np.isclose(x_span, 0.0) else max(8.0, 0.05 * x_span)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ElevationAnglesAxis.apply(
        ax=ax,
        axis="y",
        values=elevation_values,
    )
    y_min = float(np.min(elevation_values))
    y_max = float(np.max(elevation_values))
    y_span = y_max - y_min
    y_padding = 2.0 if np.isclose(y_span, 0.0) else max(2.0, 0.04 * y_span)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    if titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.plot_abs_bb_ild_diff,
        )
    if show:
        plt.show()


def plot_lsd(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    ear: str = "left",
    epsilon: float = 1e-12,
    azimuth_range_mode: str = "-180-180",
    colormap: str = "jet",
    show: bool = True,
    titles: bool = True,
) -> None:
    """Plot full-grid LSD across source positions as a spatial scatter map.

    The function computes one LSD value per source position with
    :func:`~hrtfpykit.hrtf.lsd`. When ``ear="both"``, the delegated LSD call
    averages the ear domain with ``reduction="ears"`` so
    each source position still maps to one displayed value. The result is shown
    as an azimuth-elevation scatter map with color representing log-spectral
    distance in decibels.

    Frequency selection is delegated to the metric. With frequencies=None,
    the metric uses its default LSD band from 20 Hz to 20 kHz and validates that
    both HRTFs can be compared over the requested source grid and ear channel.

    Parameters
    ----------
    hrtf_a : HRTF
        First HRTF used in the comparison. Must contain TF data and a source
        grid compatible with hrtf_b.
    hrtf_b : HRTF
        Second HRTF used in the comparison. Must contain TF data and a source
        grid compatible with hrtf_a.
    ear : {``left``, ``right``, ``both``}, default=``left``
        Ear channel selection passed to :func:`~hrtfpykit.hrtf.lsd`. ``both``
        computes both ear channels and averages the ear domain inside ``lsd`` so
        the scatter plot receives one value per source position.
    epsilon : float, default=1e-12
        Positive floor passed to :func:`~hrtfpykit.hrtf.lsd` before dB conversion.
    azimuth_range_mode : {``0-360``, ``-180-180``}, default=``-180-180``
        Azimuth convention applied to the x-axis values.
    colormap : str, default=``jet``
        Matplotlib colormap used to encode LSD values.
    show : bool, default=True
        If True, calls matplotlib.pyplot.show().
    titles : bool, default=True
        If True, applies the figure title.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If delegated LSD calculation fails, source positions are invalid, or
        the returned LSD values cannot be aligned with the number of source
        positions.

    Notes
    -----
    This is a spatial summary plot: each source position receives one
    frequency-reduced LSD value. If ``ear="both"``, the displayed value also
    includes the LSD ear-axis reduction. Use
    Use :func:`~hrtfpykit.plots.plot_lsd` when you need a full-grid spatial
    summary of pairwise spectral distance.

    Examples
    --------
    Plot a full-grid log-spectral-distance summary for the right ear:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> from hrtfpykit.plots import plot_lsd
    >>> hrtf_a = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf_b = load_hrtf("P0002_FreeFieldComp_44kHz.sofa")
    >>> plot_lsd(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     ear="right",
    ...     azimuth_range_mode="-180-180",
    ...     colormap="viridis",
    ... )
    """
    ear_key = str(ear).strip().lower()
    reduction: str | None
    if ear_key == "both":
        reduction = "ears"
    else:
        reduction = None
    difference_values = np.asarray(
        lsd(
            hrtf_a=hrtf_a,
            hrtf_b=hrtf_b,
            ear=ear_key,
            plane="all",
            frequencies=None,
            reduction=reduction,
            epsilon=epsilon,
        ),
        dtype=float,
    ).reshape(-1)

    spherical_positions = np.asarray(
        get_source_positions(
            sources=hrtf_a.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        ),
        dtype=float,
    )
    if spherical_positions.ndim != 2 or spherical_positions.shape[1] < 2:
        raise ValueError("Source positions must have shape (N, 3) in spherical coordinates")
    if spherical_positions.shape[0] != difference_values.shape[0]:
        raise ValueError("LSD values must match number of source positions")

    azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
    elevation_values = np.asarray(spherical_positions[:, 1], dtype=float)
    transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
        values=azimuth_values,
        range_mode=azimuth_range_mode,
    )

    figure = Figure(
        Layout_1(
            figsize=(12, 6),
            margins=Margins(),
        )
    )
    ax = figure.get_ax("main")
    scatter = ax.scatter(
        transformed_azimuth_values,
        elevation_values,
        c=difference_values,
        cmap=colormap,
        s=32.0,
        edgecolors="black",
        linewidths=0.25,
        vmin=float(np.min(difference_values)),
        vmax=float(np.max(difference_values)),
    )
    figure.fig.colorbar(scatter, ax=ax, label=Labels.plot_lsd_db)
    AzimuthAnglesAxis.apply(
        ax=ax,
        axis="x",
        values=transformed_azimuth_values,
        range_mode=azimuth_range_mode,
    )
    x_min = float(np.min(transformed_azimuth_values))
    x_max = float(np.max(transformed_azimuth_values))
    x_span = x_max - x_min
    x_padding = 8.0 if np.isclose(x_span, 0.0) else max(8.0, 0.05 * x_span)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ElevationAnglesAxis.apply(
        ax=ax,
        axis="y",
        values=elevation_values,
    )
    y_min = float(np.min(elevation_values))
    y_max = float(np.max(elevation_values))
    y_span = y_max - y_min
    y_padding = 2.0 if np.isclose(y_span, 0.0) else max(2.0, 0.04 * y_span)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    if titles:
        Titles.create_figure_title(
            figure.fig,
            figure.axes,
            figure.figure_title_y,
            Titles.plot_lsd,
        )
    if show:
        plt.show()


