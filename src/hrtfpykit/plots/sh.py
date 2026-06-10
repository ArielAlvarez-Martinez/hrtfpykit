from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING, Any, cast

from .axis import Axis, FrequencyLinearAxis, FrequencyLogAxis, MagnitudeAxis
from .figure import Figure
from .layouts import Layout_1
from .titles import Titles
from ..utils.coordinates import get_position_queries
from ..utils.dsp import magnitude_to_db
from ..utils.sh import sht_error

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


def sht_reconstruction_comparison(
    hrtf: "HRTF",
    reconstructed_magnitude: np.ndarray,
    position: np.ndarray | list | tuple | str = "front",
    ear: str = "left",
    x_axis: str = "linear",
    unit: str = "db",
    reference: float | str = 1.0,
    freq_min: float | None = None,
    freq_max: float | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> None:
    """Compare original and SH-reconstructed HRTF magnitude spectra.

    This diagnostic plot overlays the magnitude stored in
    :attr:`~hrtfpykit.hrtf.HRTF.TF` with a magnitude matrix reconstructed
    from spherical-harmonic coefficients, typically the output of
    :func:`~hrtfpykit.hrtf.sht_inverse`. It is used to inspect how well a
    selected spherical-harmonic order reproduces the spectral detail of one
    source direction and one ear.

    The function resolves position against the current HRTF source grid through
    the same spherical-position query path used by the main HRTF plotting methods.
    It then selects the corresponding original TF magnitude, extracts the matching
    reconstructed magnitude trace, converts both traces when unit=``db``, and draws
    them on a single frequency-axis plot.

    Notes
    -----
    ``reconstructed_magnitude`` must contain linear reconstructed magnitude
    values, not decibels. Passing ``unit`` as ``db`` affects only the plotted
    values. In dB mode, absolute magnitudes are converted with a small positive
    floor so unconstrained SH reconstructions with negative fitted values can be
    displayed. If ``reference`` is ``max``, both traces are normalized by the
    maximum absolute magnitude across the selected original and reconstructed
    spectra before conversion to decibels.

    The function creates a new :class:`~hrtfpykit.plots.figure.Figure` with a
    single axis and returns None.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.HRTF` object providing the reference
        complex TF data, frequency bins, and source-grid metadata.
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` must have shape
        (positions, ears, frequency_bins) and include at least two ear channels.
    reconstructed_magnitude : np.ndarray
        Reconstructed linear magnitude values. Use shape (N, F) for a single-ear
        spherical-harmonic reconstruction or (N, 2, F) for a two-ear
        reconstruction produced with ear=``both``. The first axis must match the
        HRTF source-position axis and the final axis must match
        :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`.
    position : np.ndarray | list | tuple | str, default=``front``
        Single spatial query resolved on the HRTF source grid. Named positions such
        as ``front``, ``back``, ``left``, and ``right`` are accepted.
        Numeric queries use spherical coordinates in degrees as [azimuth,
        elevation].
    ear : {``left``, ``right``}, default=``left``
        Ear channel used for the original HRTF trace and, when
        reconstructed_magnitude has an ear axis, for the reconstructed trace. For
        a single-ear reconstruction with shape (N, F), choose the ear that was
        used when computing the SH coefficients.
    x_axis : {``linear``, ``log``}, default=``linear``
        Frequency-axis scale used for the plot.
    unit : {``db``, ``linear``}, default=``db``
        Magnitude unit used on the y axis.
    reference : float | str, default=1.0
        Reference used when unit=``db``. Passing ``max`` normalizes both
        traces by the maximum magnitude across the selected original and
        reconstructed spectra.
    freq_min : float | None, default=None
        Lower frequency bound in hertz. When omitted, the minimum available
        frequency bin is used.
    freq_max : float | None, default=None
        Upper frequency bound in hertz. When omitted, the maximum available
        frequency bin is used.
    show : bool, default=True
        If True, call matplotlib.pyplot.show() before returning.
    show_titles : bool, default=True
        If False, suppress generated subplot titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    None
        The function creates and configures a Matplotlib figure as a side effect.

    Raises
    ------
    ValueError
        If TF values or frequency bins are unavailable, if ear, x_axis, or unit
        is unsupported, if position does not resolve to exactly one source
        position, if original or reconstructed arrays have incompatible shapes,
        if the selected position is out of bounds, if the frequency-bin axis
        does not match the selected magnitude trace, if frequency bounds are
        invalid, or if no frequency bins fall inside the selected range.

    Examples
    --------
    Compare a two-ear SH reconstruction against the original left-ear HRTF magnitude
    at the front direction:

    >>> from hrtfpykit.hrtf import load_hrtf, sht, sht_inverse
    >>> from hrtfpykit.plots import sht_reconstruction_comparison
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> sh_representation = sht(hrtf, sh_order=8, ear="both")
    >>> sh_representation.C.shape
    (81, 2, 129)
    >>> reconstructed = sht_inverse(sh_representation)
    >>> reconstructed.shape
    (793, 2, 129)
    >>> sht_reconstruction_comparison(
    ...     hrtf=hrtf,
    ...     reconstructed_magnitude=reconstructed,
    ...     position="front",
    ...     ear="left",
    ...     x_axis="log",
    ...     unit="db",
    ...     reference="max",
    ...     freq_min=200.0,
    ...     freq_max=16000.0,
    ... )
    """
    if hrtf.TF.values is None:
        raise ValueError("TF values are not available")
    if hrtf.TF.frequency_bins is None:
        raise ValueError("TF frequency_bins are not available")

    resolved_ear = str(ear).strip().lower()
    if resolved_ear not in {"left", "right"}:
        raise ValueError("ear accepts left or right")
    resolved_x_axis = str(x_axis).strip().lower()
    if resolved_x_axis not in {"linear", "log"}:
        raise ValueError("x_axis accepts linear or log")
    resolved_unit = str(unit).strip().lower()
    if resolved_unit not in {"db", "linear"}:
        raise ValueError("unit accepts db or linear")

    position_queries = get_position_queries(position)
    if len(position_queries) != 1:
        raise ValueError("position must resolve to exactly one source position")

    selected_index, selected_position = hrtf.Sources.get_position_index(
        position_queries[0],
        coordinate_system="spherical",
    )
    selected_index = int(selected_index)

    tf_values = np.asarray(hrtf.TF.values)
    if tf_values.ndim != 3:
        raise ValueError("TF values must have shape (positions, ears, frequency_bins)")
    if tf_values.shape[1] < 2:
        raise ValueError("TF values must include two ears")

    original_values_all = np.asarray(np.abs(tf_values), dtype=float)
    reconstructed_values_all = np.asarray(reconstructed_magnitude, dtype=float)
    if reconstructed_values_all.ndim not in {2, 3}:
        raise ValueError("reconstructed_magnitude must have shape (N, F) or (N, 2, F)")
    ear_index = 0 if resolved_ear == "left" else 1
    if selected_index < 0 or selected_index >= original_values_all.shape[0]:
        raise ValueError("position index is out of bounds for HRTF TF values")
    if selected_index < 0 or selected_index >= reconstructed_values_all.shape[0]:
        raise ValueError("position index is out of bounds for reconstructed_magnitude")

    if original_values_all.ndim == 3:
        if original_values_all.shape[1] < 2:
            raise ValueError("original_magnitude ear axis must contain two channels")
        original_position_values = np.asarray(
            original_values_all[selected_index, ear_index, :],
            dtype=float,
        )
    else:
        original_position_values = np.asarray(
            original_values_all[selected_index, :],
            dtype=float,
        )

    if reconstructed_values_all.ndim == 3:
        if reconstructed_values_all.shape[1] < 2:
            raise ValueError("reconstructed_magnitude ear axis must contain two channels")
        reconstructed_position_values = np.asarray(
            reconstructed_values_all[selected_index, ear_index, :],
            dtype=float,
        )
    else:
        reconstructed_position_values = np.asarray(
            reconstructed_values_all[selected_index, :],
            dtype=float,
        )

    if reconstructed_position_values.shape != original_position_values.shape:
        raise ValueError("Original and reconstructed magnitudes must have matching shape")

    frequency_bins = np.asarray(hrtf.TF.frequency_bins, dtype=float).reshape(-1)
    if frequency_bins.size != original_position_values.size:
        raise ValueError("TF frequency_bins length must match magnitude frequency axis")
    frequency_axis = FrequencyLogAxis if resolved_x_axis == "log" else FrequencyLinearAxis
    frequency_axis_config = frequency_axis.build(
        frequency_bins=frequency_bins,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    selected_frequency_indices = np.where(
        (frequency_bins >= float(cast(Any, frequency_axis_config["freq_min"])))
        & (frequency_bins <= float(cast(Any, frequency_axis_config["freq_max"])))
    )[0]
    if selected_frequency_indices.size == 0:
        raise ValueError("No frequency bins fall inside the selected frequency range")
    frequency_bins = frequency_bins[selected_frequency_indices]
    original_position_values = original_position_values[selected_frequency_indices]
    reconstructed_position_values = reconstructed_position_values[selected_frequency_indices]
    frequency_khz = frequency_bins / 1000.0

    if resolved_unit == "db":
        resolved_reference: float | str
        if isinstance(reference, str) and str(reference).strip().lower() == "max":
            resolved_reference = float(
                np.max(
                    np.concatenate(
                        [
                            np.abs(original_position_values).reshape(-1),
                            np.abs(reconstructed_position_values).reshape(-1),
                        ]
                    )
                )
            )
        else:
            resolved_reference = reference
        db_floor = 1e-12
        original_plot_values = np.asarray(
            magnitude_to_db(
                np.maximum(np.abs(original_position_values), db_floor),
                reference=resolved_reference,
            ),
            dtype=float,
        )
        reconstructed_plot_values = np.asarray(
            magnitude_to_db(
                np.maximum(np.abs(reconstructed_position_values), db_floor),
                reference=resolved_reference,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(original_plot_values)) or not np.all(np.isfinite(reconstructed_plot_values)):
            raise ValueError("dB magnitude arrays must contain finite values")
    else:
        original_plot_values = original_position_values
        reconstructed_plot_values = reconstructed_position_values

    figure = Figure(Layout_1())
    ax = figure.get_ax("main")
    figure.create_two_dimension(
        ax=ax,
        x=frequency_khz,
        y=original_plot_values,
        color="tab:blue",
    )
    figure.create_two_dimension(
        ax=ax,
        x=frequency_khz,
        y=reconstructed_plot_values,
        color="tab:red",
        linestyle="--",
    )

    frequency_axis.apply(
        ax=ax,
        axis="x",
        config=frequency_axis_config,
    )
    MagnitudeAxis.apply(
        ax=ax,
        axis="y",
        unit=resolved_unit,
        label=None if show_labels else "",
    )

    if show_titles:
        Titles.create_subplots_titles(
            ax=ax,
            title=Titles.create_position_title(np.asarray(selected_position, dtype=float)[:2]),
        )
    if show_legends:
        ax.legend(labels=["Original", "Reconstructed"], loc="upper right")
    if show:
        plt.show()


def sht_reconstruction_error(
    hrtf: "HRTF",
    reconstructed_magnitude: np.ndarray,
    position: np.ndarray | list | tuple | str = "front",
    ear: str = "left",
    x_axis: str = "linear",
    magnitude: str = "linear",
    reference: float | str = 1.0,
    freq_min: float | None = None,
    freq_max: float | None = None,
    show: bool = True,
    show_titles: bool = True,
    show_labels: bool = True,
    show_legends: bool = True,
) -> None:
    """Plot SH reconstruction magnitude error for one direction and ear.

    This diagnostic plot shows the point-wise magnitude error between the
    original HRTF magnitude and a spherical-harmonic reconstruction at one source
    position and ear. The plotted error is original_magnitude -
    reconstructed_magnitude for the selected trace in the requested magnitude
    domain, and the subplot title includes the root-mean-square error across
    frequency.

    Use this function after :func:`~hrtfpykit.hrtf.sht_inverse` to inspect
    where a selected spherical-harmonic order loses spectral detail for a specific
    direction. Unlike
    :func:`~hrtfpykit.plots.sht_reconstruction_comparison`, this function
    plots the error directly instead of overlaying original and reconstructed
    spectra.

    Notes
    -----
    ``reconstructed_magnitude`` must contain linear magnitude values with the same
    source-position and frequency axes as the
    :class:`~hrtfpykit.hrtf.HRTF` object's current TF data. The frequency
    axis is taken from
    :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` and
    displayed in kHz.

    The function creates a new :class:`~hrtfpykit.plots.figure.Figure` with a
    single axis and returns None.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.HRTF` object providing the reference
        complex TF data, frequency bins, and source-grid metadata.
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` must have shape
        (positions, ears, frequency_bins) and include at least two ear channels.
    reconstructed_magnitude : np.ndarray
        Reconstructed linear magnitude values. Use shape (N, F) for a single-ear
        spherical-harmonic reconstruction or (N, 2, F) for a two-ear
        reconstruction produced with ear=``both``. The first axis must match the
        HRTF source-position axis and the final axis must match
        :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`.
    position : np.ndarray | list | tuple | str, default=``front``
        Single spatial query resolved on the HRTF source grid. Named positions such
        as ``front``, ``back``, ``left``, and ``right`` are accepted.
        Numeric queries use spherical coordinates in degrees as [azimuth,
        elevation].
    ear : {``left``, ``right``}, default=``left``
        Ear channel used for the original HRTF trace and, when
        reconstructed_magnitude has an ear axis, for the reconstructed trace. For
        a single-ear reconstruction with shape (N, F), choose the ear that was
        used when computing the SH coefficients.
    x_axis : {``linear``, ``log``}, default=``linear``
        Frequency-axis scale used for the plot.
    magnitude : {``linear``, ``db``}, default=``linear``
        Magnitude domain used for the error trace. ``linear`` plots the raw
        linear-magnitude reconstruction error. ``db`` converts absolute
        magnitudes to decibels before subtracting, so the plotted trace is a
        point-wise LSD-style dB error.
    reference : float | str, default=1.0
        Reference used when magnitude=``db``. Passing ``max`` normalizes both
        original and reconstructed values by the maximum absolute magnitude in
        the selected trace before dB conversion.
    freq_min : float | None, default=None
        Lower frequency bound in hertz. When omitted, the minimum available
        frequency bin is used.
    freq_max : float | None, default=None
        Upper frequency bound in hertz. When omitted, the maximum available
        frequency bin is used.
    show : bool, default=True
        If True, call matplotlib.pyplot.show() before returning.
    show_titles : bool, default=True
        If False, suppress generated subplot titles.
    show_labels : bool, default=True
        If False, suppress generated axis labels.
    show_legends : bool, default=True
        If False, suppress generated legends.

    Returns
    -------
    None
        The function creates and configures a Matplotlib figure as a side effect.

    Raises
    ------
    ValueError
        If TF values or frequency bins are unavailable, if ear, x_axis, or
        magnitude is unsupported, if position does not resolve to exactly one source
        position, if original or reconstructed arrays have incompatible shapes,
        if the selected position is out of bounds, if the frequency-bin axis
        does not match the selected magnitude trace, if frequency bounds are
        invalid, if no frequency bins fall inside the selected range, or if dB
        conversion cannot produce finite values.

    Examples
    --------
    Plot the reconstruction error for the right ear at the left direction:

    >>> from hrtfpykit.hrtf import load_hrtf, sht, sht_inverse
    >>> from hrtfpykit.plots import sht_reconstruction_error
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> sh_representation = sht(hrtf, sh_order=8, ear="both")
    >>> sh_representation.C.shape
    (81, 2, 129)
    >>> reconstructed = sht_inverse(sh_representation)
    >>> reconstructed.shape
    (793, 2, 129)
    >>> sht_reconstruction_error(
    ...     hrtf=hrtf,
    ...     reconstructed_magnitude=reconstructed,
    ...     position="left",
    ...     ear="right",
    ...     x_axis="log",
    ...     magnitude="db",
    ...     reference="max",
    ...     freq_min=200.0,
    ...     freq_max=16000.0,
    ... )
    """
    if hrtf.TF.values is None:
        raise ValueError("TF values are not available")
    if hrtf.TF.frequency_bins is None:
        raise ValueError("TF frequency_bins are not available")

    resolved_ear = str(ear).strip().lower()
    if resolved_ear not in {"left", "right"}:
        raise ValueError("ear accepts left or right")
    resolved_x_axis = str(x_axis).strip().lower()
    if resolved_x_axis not in {"linear", "log"}:
        raise ValueError("x_axis accepts linear or log")
    resolved_magnitude = str(magnitude).strip().lower()
    if resolved_magnitude not in {"linear", "db"}:
        raise ValueError("magnitude accepts linear or db")

    position_queries = get_position_queries(position)
    if len(position_queries) != 1:
        raise ValueError("position must resolve to exactly one source position")

    selected_index, selected_position = hrtf.Sources.get_position_index(
        position_queries[0],
        coordinate_system="spherical",
    )
    selected_index = int(selected_index)

    tf_values = np.asarray(hrtf.TF.values)
    if tf_values.ndim != 3:
        raise ValueError("TF values must have shape (positions, ears, frequency_bins)")
    if tf_values.shape[1] < 2:
        raise ValueError("TF values must include two ears")

    original_values_all = np.asarray(np.abs(tf_values), dtype=float)
    reconstructed_values_all = np.asarray(reconstructed_magnitude, dtype=float)
    if reconstructed_values_all.ndim not in {2, 3}:
        raise ValueError("reconstructed_magnitude must have shape (N, F) or (N, 2, F)")
    ear_index = 0 if resolved_ear == "left" else 1
    if selected_index < 0 or selected_index >= original_values_all.shape[0]:
        raise ValueError("position index is out of bounds for HRTF TF values")
    if selected_index < 0 or selected_index >= reconstructed_values_all.shape[0]:
        raise ValueError("position index is out of bounds for reconstructed_magnitude")

    original_position_values = np.asarray(
        original_values_all[selected_index, ear_index, :],
        dtype=float,
    )

    if reconstructed_values_all.ndim == 3:
        if reconstructed_values_all.shape[1] < 2:
            raise ValueError("reconstructed_magnitude ear axis must contain two channels")
        reconstructed_position_values = np.asarray(
            reconstructed_values_all[selected_index, ear_index, :],
            dtype=float,
        )
    else:
        reconstructed_position_values = np.asarray(
            reconstructed_values_all[selected_index, :],
            dtype=float,
        )

    if reconstructed_position_values.shape != original_position_values.shape:
        raise ValueError("Original and reconstructed magnitudes must have matching shape")

    frequency_bins = np.asarray(hrtf.TF.frequency_bins, dtype=float).reshape(-1)
    if frequency_bins.size != original_position_values.size:
        raise ValueError("TF frequency_bins length must match magnitude frequency axis")
    frequency_axis = FrequencyLogAxis if resolved_x_axis == "log" else FrequencyLinearAxis
    frequency_axis_config = frequency_axis.build(
        frequency_bins=frequency_bins,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    selected_frequency_indices = np.where(
        (frequency_bins >= float(cast(Any, frequency_axis_config["freq_min"])))
        & (frequency_bins <= float(cast(Any, frequency_axis_config["freq_max"])))
    )[0]
    if selected_frequency_indices.size == 0:
        raise ValueError("No frequency bins fall inside the selected frequency range")
    frequency_bins = frequency_bins[selected_frequency_indices]
    original_position_values = original_position_values[selected_frequency_indices]
    reconstructed_position_values = reconstructed_position_values[selected_frequency_indices]
    frequency_khz = frequency_bins / 1000.0

    if resolved_magnitude == "db":
        resolved_reference: float | str
        if isinstance(reference, str) and str(reference).strip().lower() == "max":
            resolved_reference = float(
                np.max(
                    np.concatenate(
                        [
                            np.abs(original_position_values).reshape(-1),
                            np.abs(reconstructed_position_values).reshape(-1),
                        ]
                    )
                )
            )
        else:
            resolved_reference = reference
        db_floor = 1e-12
        original_error_values = np.asarray(
            magnitude_to_db(
                np.maximum(np.abs(original_position_values), db_floor),
                reference=resolved_reference,
            ),
            dtype=float,
        )
        reconstructed_error_values = np.asarray(
            magnitude_to_db(
                np.maximum(np.abs(reconstructed_position_values), db_floor),
                reference=resolved_reference,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(original_error_values)) or not np.all(np.isfinite(reconstructed_error_values)):
            raise ValueError("dB magnitude arrays must contain finite values")
        error_values = original_error_values - reconstructed_error_values
        y_label = "Magnitude Error (dB)"
        rms_label = "LSD RMS"
        rms_unit = " dB"
    else:
        error_values = original_position_values - reconstructed_position_values
        y_label = "Magnitude Error"
        rms_label = "RMS"
        rms_unit = ""
    _, _, rms_error, _ = sht_error(
        original_magnitude=original_position_values,
        reconstructed_magnitude=reconstructed_position_values,
        magnitude=resolved_magnitude,
        reference=reference,
    )

    figure = Figure(Layout_1())
    ax = figure.get_ax("main")
    figure.create_two_dimension(
        ax=ax,
        x=frequency_khz,
        y=error_values,
        color="red",
    )

    frequency_axis.apply(
        ax=ax,
        axis="x",
        config=frequency_axis_config,
    )
    Axis.apply_label(
        ax=ax,
        axis="y",
        default_label=y_label,
        label=None if show_labels else "",
    )
    if show_titles:
        Titles.create_subplots_titles(
            ax=ax,
            title=(
                f"{Titles.create_position_title(np.asarray(selected_position, dtype=float)[:2])}"
                f" | Error | {rms_label}={rms_error:.6f}{rms_unit}"
            ),
        )
    if show:
        plt.show()
