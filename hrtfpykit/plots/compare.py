from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .axis import (
    AmplitudeAxis,
    FrequencyLinearAxis,
    FrequencyLogAxis,
    MagnitudeAxis,
    SampleAxis,
    TimeAxis,
)
from .default import Margins
from .figure import Figure
from .labels import Labels
from .layouts import Layout_1, Layout_2Horizontal, Layout_2Vertical, Layout_3
from .titles import Titles
from .._warnings import HRTFPyKitWarning, warn_user
from ..hrtf.coordinates import get_position_queries
from ..hrtf.dsp import magnitude_to_db


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
    freq_min: float | None = None,
    freq_max: float | None = None,
    show: bool = True,
    titles: bool = True,
) -> None:
    """Compare HRTF magnitude curves across multiple HRTF instances.

    The function overlays one curve per subject for each requested position.
    It supports up to five HRTFs and up to four positions. When ``ear="both"``,
    comparison is restricted to one position and displayed as two horizontal
    subplots: left ear and right ear.

    Parameters
    ----------
    hrtfs : list[HRTF]
        HRTF objects to compare. Requires at least 2 and at most 5 entries.
    positions : str | list | tuple | np.ndarray, default=("front",)
        Position query or collection of position queries. Up to 4 positions are
        accepted. Query resolution uses each HRTF's nearest available source.
    ear : {"left", "right", "both"}, default="left"
        Ear channel selection. ``"both"`` requires exactly one position.
    x_axis : {"linear", "log"}, default="linear"
        Frequency-axis scale.
    unit : {"db", "linear"}, default="db"
        Magnitude unit for plotted values.
    reference : float | str, default=1.0
        Reference used when ``unit="db"``. Supports ``"max"``.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1 ... subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses Matplotlib default cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    freq_min : float | None, default=None
        Minimum frequency in Hz. If omitted, resolved from all HRTFs.
    freq_max : float | None, default=None
        Maximum frequency in Hz. If omitted, resolved from all HRTFs.
    show : bool, default=True
        If ``True``, calls ``matplotlib.pyplot.show()``.
    titles : bool, default=True
        Controls subplot titles in single-ear mode.

    Returns
    -------
    None

    Use Cases
    ---------
    - Compare baseline vs individualized HRTFs at key directions.
    - Compare pipeline variants for one target ear.
    - Inspect both ears at one position with shared subject legends.

    Examples
    --------
    >>> from hrtfpykit.plots.compare import compare_magnitude
    >>> compare_magnitude([h1, h2], positions=["front", "left"], ear="left", show=False)
    >>> compare_magnitude(
    ...     [h1, h2, h3],
    ...     positions="front",
    ...     ear="both",
    ...     legends=["baseline", "pipe_a", "pipe_b"],
    ...     line_styles=["-", "--", ":"],
    ...     show=False,
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
        resolved_legends = [f"subject_{index + 1}" for index in range(hrtf_count)]
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

    if ear == "both":
        for axis_name, ear_index, subplot_title in (
            ("left", 0, "Left Ear"),
            ("right", 1, "Right Ear"),
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
            ax.legend(labels=resolved_legends, loc=magnitude_legend_location)
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
            ax.legend(labels=resolved_legends, loc=magnitude_legend_location)
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
    show: bool = True,
    titles: bool = True,
) -> None:
    """Compare HRIR amplitude curves across multiple HRTF instances.

    The function overlays one waveform per subject for each requested position.
    It supports up to five HRTFs and up to four positions. When ``ear="both"``,
    comparison is restricted to one position and displayed as two horizontal
    subplots: left ear and right ear.

    Parameters
    ----------
    hrtfs : list[HRTF]
        HRTF objects to compare. Requires at least 2 and at most 5 entries.
    positions : str | list | tuple | np.ndarray, default=("front",)
        Position query or collection of position queries. Up to 4 positions are
        accepted. Query resolution uses each HRTF's nearest available source.
    ear : {"left", "right", "both"}, default="left"
        Ear channel selection. ``"both"`` requires exactly one position.
    x_axis : {"time", "samples"}, default="time"
        Horizontal axis mode for waveforms.
    legends : list[str] | tuple[str, ...] | None, default=None
        Subject legend labels. Defaults to ``subject_1 ... subject_n``.
    line_colors : list[str] | tuple[str, ...] | None, default=None
        One line color per subject. Uses Matplotlib default cycle when omitted.
    line_styles : list[str] | tuple[str, ...] | None, default=None
        One line style per subject. Defaults to solid lines.
    show : bool, default=True
        If ``True``, calls ``matplotlib.pyplot.show()``.
    titles : bool, default=True
        Controls subplot titles in single-ear mode.

    Returns
    -------
    None

    Use Cases
    ---------
    - Compare HRIR waveform timing and shape across subjects/pipelines.
    - Compare preprocessing variants in samples or seconds.
    - Inspect both ears at one position with shared subject legends.

    Examples
    --------
    >>> from hrtfpykit.plots.compare import compare_amplitude
    >>> compare_amplitude([h1, h2], positions=["front", "right"], ear="left", show=False)
    >>> compare_amplitude(
    ...     [h1, h2],
    ...     positions="front",
    ...     ear="both",
    ...     x_axis="samples",
    ...     legends=["reference", "candidate"],
    ...     show=False,
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
        resolved_legends = [f"subject_{index + 1}" for index in range(hrtf_count)]
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
            x_values_by_subject.append(subject_sample_indexes / float(hrtf.IR.sample_rate))
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

    if ear == "both":
        for axis_name, ear_index, subplot_title in (
            ("left", 0, "Left Ear"),
            ("right", 1, "Right Ear"),
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
            ax.legend(labels=resolved_legends, loc="upper right")
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
            ax.legend(labels=resolved_legends, loc="upper right")
            ax.grid(True)

    if ear != "both" and position_count < figure.axes.size:
        figure.hide_unused_axes(position_count)

    if show:
        plt.show()
