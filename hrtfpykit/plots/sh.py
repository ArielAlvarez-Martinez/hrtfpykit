from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

from .axis import Axis, FrequencyLinearAxis, MagnitudeAxis
from .figure import Figure
from .layouts import Layout_1
from .titles import Titles
from ..hrtf.coordinates import get_position_queries
from ..hrtf.dsp import magnitude_to_db

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


def plot_reconstruction_comparison(
    hrtf: "HRTF",
    reconstructed_magnitude: np.ndarray,
    position: np.ndarray | list | tuple | str = "front",
    ear: str = "left",
    unit: str = "db",
    reference: float | str = 1.0,
    show: bool = True,
) -> None:
    """Plot original vs reconstructed magnitude for one position and ear.

    Parameters
    ----------
    hrtf : HRTF
        HRTF object providing original TF magnitude and frequency bins.
    reconstructed_magnitude : np.ndarray
        Reconstructed magnitude matrix with shape ``(N, F)`` or ``(N, 2, F)``.
    position : np.ndarray | list | tuple | str, default="front"
        Single spatial query resolved on the HRTF source grid.
    ear : str, default="left"
        Ear selection. Accepted values are ``"left"`` and ``"right"``.
    unit : str, default="db"
        Magnitude unit for display. Accepted values are ``"db"`` and ``"linear"``.
    reference : float | str, default=1.0
        Reference used when ``unit="db"``. ``"max"`` uses the maximum value
        across original and reconstructed traces.
    show : bool, default=True
        If ``True``, call ``matplotlib.pyplot.show()``.

    Returns
    -------
    None

    Use Cases
    ---------
    - Visual inspection of SH reconstruction quality at one direction.
    - Compare reconstructed and original magnitude in linear or dB units.

    Examples
    --------
    >>> reconstructed = sht_inverse(sh)
    >>> plot_reconstruction_comparison(
    ...     hrtf=hrtf,
    ...     reconstructed_magnitude=reconstructed,
    ...     position="front",
    ...     ear="left",
    ...     unit="db",
    ...     show=False,
    ... )
    """
    if hrtf.TF.values is None:
        raise ValueError("TF values are not available")
    if hrtf.TF.frequency_bins is None:
        raise ValueError("TF frequency_bins are not available")

    resolved_ear = str(ear).strip().lower()
    if resolved_ear not in {"left", "right"}:
        raise ValueError("ear accepts left or right")
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
    frequency_khz = frequency_bins / 1000.0

    if resolved_unit == "db":
        if isinstance(reference, str) and str(reference).strip().lower() == "max":
            resolved_reference = float(
                np.max(
                    np.concatenate(
                        [
                            original_position_values.reshape(-1),
                            reconstructed_position_values.reshape(-1),
                        ]
                    )
                )
            )
        else:
            resolved_reference = reference
        original_plot_values = np.asarray(
            magnitude_to_db(original_position_values, reference=resolved_reference),
            dtype=float,
        )
        reconstructed_plot_values = np.asarray(
            magnitude_to_db(reconstructed_position_values, reference=resolved_reference),
            dtype=float,
        )
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

    frequency_axis_config = FrequencyLinearAxis.build(frequency_bins=frequency_bins)
    FrequencyLinearAxis.apply(
        ax=ax,
        axis="x",
        config=frequency_axis_config,
    )
    MagnitudeAxis.apply(
        ax=ax,
        axis="y",
        unit=resolved_unit,
    )

    Titles.create_subplots_titles(
        ax=ax,
        title=Titles.create_position_title(np.asarray(selected_position, dtype=float)[:2]),
    )
    ax.legend(labels=["Original", "Reconstructed"], loc="upper right")
    if show:
        plt.show()


def plot_reconstruction_error(
    hrtf: "HRTF",
    reconstructed_magnitude: np.ndarray,
    position: np.ndarray | list | tuple | str = "front",
    ear: str = "left",
    show: bool = True,
) -> None:
    """Plot reconstruction error across frequency for one position and ear.

    Parameters
    ----------
    hrtf : HRTF
        HRTF object providing original TF magnitude and frequency bins.
    reconstructed_magnitude : np.ndarray
        Reconstructed magnitude matrix with shape ``(N, F)`` or ``(N, 2, F)``.
    position : np.ndarray | list | tuple | str, default="front"
        Single spatial query resolved on the HRTF source grid.
    ear : str, default="left"
        Ear selection. Accepted values are ``"left"`` and ``"right"``.
    show : bool, default=True
        If ``True``, call ``matplotlib.pyplot.show()``.

    Returns
    -------
    None

    Use Cases
    ---------
    - Inspect signed reconstruction error versus frequency.
    - Track RMS error shown in the subplot title for one direction.

    Examples
    --------
    >>> reconstructed = sht_inverse(sh)
    >>> plot_reconstruction_error(
    ...     hrtf=hrtf,
    ...     reconstructed_magnitude=reconstructed,
    ...     position="left",
    ...     ear="right",
    ...     show=False,
    ... )
    """
    if hrtf.TF.values is None:
        raise ValueError("TF values are not available")
    if hrtf.TF.frequency_bins is None:
        raise ValueError("TF frequency_bins are not available")

    resolved_ear = str(ear).strip().lower()
    if resolved_ear not in {"left", "right"}:
        raise ValueError("ear accepts left or right")

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
    frequency_khz = frequency_bins / 1000.0

    error_values = original_position_values - reconstructed_position_values
    y_label = "Magnitude Error"
    rms_error = float(np.sqrt(np.mean(np.asarray(error_values, dtype=float) ** 2)))

    figure = Figure(Layout_1())
    ax = figure.get_ax("main")
    figure.create_two_dimension(
        ax=ax,
        x=frequency_khz,
        y=error_values,
        color="red",
    )

    frequency_axis_config = FrequencyLinearAxis.build(frequency_bins=frequency_bins)
    FrequencyLinearAxis.apply(
        ax=ax,
        axis="x",
        config=frequency_axis_config,
    )
    Axis.apply_label(
        ax=ax,
        axis="y",
        default_label=y_label,
    )
    Titles.create_subplots_titles(
        ax=ax,
        title=(
            f"{Titles.create_position_title(np.asarray(selected_position, dtype=float)[:2])}"
            f" | Error | RMS={rms_error:.6f}"
        ),
    )
    if show:
        plt.show()
