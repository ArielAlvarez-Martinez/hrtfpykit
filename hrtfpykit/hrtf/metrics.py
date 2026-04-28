from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

from .dsp import iir_filter, magnitude_to_db, tf_from_ir
from .coordinates import get_position_queries
from .planes import get_horizontal_plane, get_median_plane

if TYPE_CHECKING:
    from .hrtf import HRTF
    from .domain import IR


def itd(
    ir: np.ndarray | "IR",
    method: str = "threshold",
    sample_rate: float | None = None,
    output: str = "samples",
    thresh_level: float = -10.0,
    upper_cut_freq: float = 3000.0,
    filter_order: int = 10,
) -> np.ndarray:
    """Estimate interaural time differences from binaural IR data.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain array or ``IR`` object with layout ``[..., ear, samples]``.
        The ear convention is ``0=left`` and ``1=right``.
    method : {"threshold", "maxiacce"}, default="threshold"
        ITD estimator used to compute the onset difference.
    sample_rate : float | None, default=None
        Sample rate in Hz for NumPy input. When ``ir`` is an ``IR`` object and
        this value is omitted, ``IR.sample_rate`` is used.
    output : {"seconds", "samples"}, default="seconds"
        Output unit of the returned ITD values.
    thresh_level : float, default=-10.0
        Threshold offset in dB used by the ``threshold`` method.
    upper_cut_freq : float, default=3000.0
        Low-pass cutoff in Hz applied before ITD estimation.
    filter_order : int, default=10
        Butterworth low-pass filter order used in the preprocessing stage.

    Returns
    -------
    np.ndarray
        ITD values with shape ``ir.shape[:-2]``. Positive values mean the left
        ear is delayed relative to the right ear.

    Use Cases
    ---------
    - Estimate directional ITD cues directly from HRIR data.
    - Compare onset-delay behavior between two binaural renderings.
    - Build ITD curves over a source grid before spatial plotting.

    Examples
    --------
    Estimate ITD in samples for a short binaural impulse:

    >>> ir = np.array([[[0.0, 0.0, 1.0, 0.0],
    ...                 [0.0, 1.0, 0.0, 0.0]]])
    >>> itd(ir, sample_rate=48000.0, output="samples")
    array([1])

    Convert the same ITD estimate to seconds:

    >>> itd(ir, sample_rate=48000.0, output="seconds")
    array([2.08333333e-05])
    """
    if isinstance(ir, np.ndarray):
        ir_values = ir
        resolved_sample_rate = sample_rate
    else:
        if not hasattr(ir, "values") or not hasattr(ir, "sample_rate"):
            raise ValueError("ir must be a NumPy array or an IR instance")
        ir_values = ir.values
        resolved_sample_rate = sample_rate if sample_rate is not None else ir.sample_rate

    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim < 2:
        raise ValueError("IR data must include at least channel and time axes")

    if resolved_sample_rate is None:
        raise ValueError("sample_rate is required")
    if isinstance(resolved_sample_rate, bool):
        raise ValueError("sample_rate must be a finite, positive value.")
    try:
        resolved_sample_rate = float(resolved_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("sample_rate must be a finite, positive value.") from None
    if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
        raise ValueError("sample_rate must be a finite, positive value.")

    method_key = str(method).strip().lower()
    if method_key not in {"threshold", "maxiacce"}:
        raise ValueError("method must be one of: threshold, maxiacce")
    output_key = str(output).strip().lower()
    if output_key not in {"seconds", "samples"}:
        raise ValueError("output must be one of: seconds, samples")

    ir_channel_last = ir_values
    channel_count = ir_channel_last.shape[-2]
    if channel_count < 2:
        raise ValueError("IR ear axis must contain at least two channels (0=left, 1=right)")

    time_length = int(ir_channel_last.shape[-1])
    if time_length < 2:
        raise ValueError("IR time axis must contain at least two samples")

    flattened = ir_channel_last.reshape(-1, channel_count, time_length)
    itd_values = np.empty(flattened.shape[0], dtype=int)

    if isinstance(filter_order, bool) or not isinstance(filter_order, int):
        raise ValueError("filter_order must be an integer")
    if filter_order <= 0:
        raise ValueError("filter_order must be positive")

    left_signals = flattened[:, 0, :]
    right_signals = flattened[:, 1, :]
    left_processed = iir_filter(
        left_signals,
        filter="lowpass",
        sample_rate=resolved_sample_rate,
        cutoff=upper_cut_freq,
        order=filter_order,
    )
    right_processed = iir_filter(
        right_signals,
        filter="lowpass",
        sample_rate=resolved_sample_rate,
        cutoff=upper_cut_freq,
        order=filter_order,
    )

    if method_key == "threshold":
        if isinstance(thresh_level, bool):
            raise ValueError("thresh_level must be a finite value.")
        try:
            thresh_level = float(thresh_level)
        except (TypeError, ValueError):
            raise ValueError("thresh_level must be a finite value.") from None
        if not np.isfinite(thresh_level):
            raise ValueError("thresh_level must be a finite value.")

        for index in range(flattened.shape[0]):
            left_db = 0.5 * magnitude_to_db(np.square(left_processed[index]))
            right_db = 0.5 * magnitude_to_db(np.square(right_processed[index]))
            left_threshold = float(np.max(left_db)) + thresh_level
            right_threshold = float(np.max(right_db)) + thresh_level
            left_hits = np.where(left_db > left_threshold)[0]
            right_hits = np.where(right_db > right_threshold)[0]
            if left_hits.size == 0 or right_hits.size == 0:
                raise ValueError("threshold mode could not find a valid onset index")
            left_idx = int(left_hits[0])
            right_idx = int(right_hits[0])
            itd_values[index] = int(left_idx - right_idx)
    else:
        lags = signal.correlation_lags(time_length, time_length, mode="full")
        for index in range(flattened.shape[0]):
            left_env = np.abs(signal.hilbert(left_processed[index]))
            right_env = np.abs(signal.hilbert(right_processed[index]))
            cross_corr = signal.correlate(right_env, left_env, mode="full", method="fft")
            peak_lag = lags[int(np.argmax(np.abs(cross_corr)))]
            itd_values[index] = int(-peak_lag)

    if output_key == "seconds":
        itd_values = itd_values.astype(float) / resolved_sample_rate
    output_shape = ir_channel_last.shape[:-2]
    return itd_values.reshape(output_shape)


def ild(
    ir: np.ndarray | "IR",
    sample_rate: float | None = None,
    fft_length: int | None = None,
    mode: str = "broad-band",
    output: str = "db",
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute interaural level differences from binaural IR data.

    Parameters
    ----------
    ir : np.ndarray | IR
        Binaural time-domain signal with layout ``[..., ear, samples]``. The
        ear convention is ``0=left`` and ``1=right``.
    sample_rate : float | None, default=None
        Sample rate in Hz used when ``ir`` is a NumPy array and
        ``mode="frequency-dependent"``.
    fft_length : int | None, default=None
        Optional FFT length used when ``mode="frequency-dependent"`` and the
        IR must be converted internally to TF values.
    mode : {"broad-band", "frequency-dependent"}, default="broad-band"
        ILD mode to compute.
    output : {"db", "linear"}, default="db"
        Output representation of the ILD values.
    epsilon : float, default=1e-12
        Positive floor used to avoid division by zero in level ratios.

    Returns
    -------
    np.ndarray
        ILD array in the requested mode. ``mode="broad-band"`` returns shape
        ``[...]`` and ``mode="frequency-dependent"`` returns
        ``[..., frequency_bins]``. Positive values mean the left-ear level is
        greater than the right-ear level, and negative values mean the
        right-ear level is greater than the left-ear level.

    Use Cases
    ---------
    - Measure broad-band level asymmetry between left and right ears.
    - Compute per-frequency ILD cues for spectral analysis.
    - Build ILD features for comparison and quality metrics.

    Examples
    --------
    Measure the broad-band ILD of a simple binaural impulse:

    >>> ir = np.array([[[1.0, 0.0, 0.0, 0.0],
    ...                 [0.5, 0.0, 0.0, 0.0]]])
    >>> ild(ir, sample_rate=48000.0, mode="broad-band", output="db")
    array([6.02059991])

    Inspect the frequency-dependent ILD shape for the same signal:

    >>> ild(ir, sample_rate=48000.0, mode="frequency-dependent", output="linear").shape
    (1, 3)
    """
    ir_object = None
    if isinstance(ir, np.ndarray):
        ir_values = ir
    elif hasattr(ir, "values"):
        ir_object = ir
        ir_values = ir.values
    else:
        raise ValueError("ir must be a NumPy array or an IR instance")

    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim < 2:
        raise ValueError("IR data must include at least ear and time axes")

    output_key = str(output).strip().lower()
    if output_key not in {"db", "linear"}:
        raise ValueError("output must be one of: db, linear")

    mode_key = str(mode).strip().lower()
    if mode_key not in {"broad-band", "frequency-dependent"}:
        raise ValueError("mode must be one of: broad-band, frequency-dependent")

    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    if ir_object is not None and hasattr(ir_object, "sample_rate"):
        resolved_sample_rate = (
            sample_rate if sample_rate is not None else ir_object.sample_rate
        )
    else:
        resolved_sample_rate = sample_rate

    if ir_values.shape[-2] < 2:
        raise ValueError("Ear axis must contain at least two channels (0=left, 1=right)")

    if mode_key == "frequency-dependent":
        if resolved_sample_rate is None:
            raise ValueError("sample_rate is required for IR NumPy inputs")
        tf_values, _, _ = tf_from_ir(
            ir_values,
            sample_rate=resolved_sample_rate,
            fft_length=fft_length,
        )
    else:
        tf_values = None

    if mode_key == "broad-band":
        left_values = np.asarray(ir_values[..., 0, :], dtype=float)
        right_values = np.asarray(ir_values[..., 1, :], dtype=float)
        left_rms = np.sqrt(np.mean(np.square(left_values), axis=-1))
        right_rms = np.sqrt(np.mean(np.square(right_values), axis=-1))
        ild_linear = (left_rms + epsilon) / (right_rms + epsilon)
        if output_key == "linear":
            return ild_linear
        return magnitude_to_db(ild_linear)

    left_magnitude = np.abs(tf_values[..., 0, :])
    right_magnitude = np.abs(tf_values[..., 1, :])
    ild_linear = (left_magnitude + epsilon) / (right_magnitude + epsilon)

    if output_key == "linear":
        return ild_linear
    return magnitude_to_db(ild_linear)


def itd_difference(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    method: str = "threshold",
    output: str = "seconds",
    thresh_level: float = -10.0,
    upper_cut_freq: float = 3000.0,
    filter_order: int = 10,
) -> np.ndarray:
    """Compute absolute per-position ITD differences between two HRTFs.

    The metric estimates ITD independently for each input HRTF using the same
    estimator configuration and returns the absolute difference between both
    ITD vectors. The output therefore represents the magnitude of ITD change
    per source position.

    Parameters
    ----------
    hrtf_a : HRTF
        Reference HRTF object used for comparison. Must share the same source
        grid (same source positions) as ``hrtf_b``.
    hrtf_b : HRTF
        Compared HRTF object used for comparison. Must share the same source
        grid (same source positions) as ``hrtf_a``.
    method : {"threshold", "maxiacce"}, default="threshold"
        ITD estimator passed to :func:`itd`.
    output : {"seconds", "samples"}, default="seconds"
        Unit of returned absolute ITD differences.
    thresh_level : float, default=-10.0
        Threshold offset in dB used by ``method="threshold"``.
    upper_cut_freq : float, default=3000.0
        Low-pass cutoff in Hz passed to :func:`itd`.
    filter_order : int, default=10
        Butterworth low-pass order passed to :func:`itd`.

    Returns
    -------
    np.ndarray
        Absolute ITD differences per position with shape ``[positions]`` (or
        the corresponding leading source shape for selected subsets).

    Use Cases
    ---------
    - Quantify per-position ITD changes after individualization.
    - Compare ITD impact of two processing pipelines.
    - Build position-wise ITD error curves relative to a reference HRTF.

    Notes
    -----
    Both HRTFs must have the same source grid. If source positions differ,
    this function raises a ``ValueError``.

    Examples
    --------
    >>> from hrtfpykit.hrtf.metrics import itd_difference
    >>> itd_diff = itd_difference(hrtf_a, hrtf_b)
    >>> itd_diff.shape
    (hrtf_a.IR.values.shape[0],)
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
        float(hrtf_a.IR.sample_rate),
        float(hrtf_b.IR.sample_rate),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "output='samples' requires equal sample_rate in both HRTFs"
        )

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
    return np.abs(itd_a - itd_b)


def ild_difference(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    mode: str = "broad-band",
    output: str = "db",
    fft_length: int | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute absolute per-position ILD differences between two HRTFs.

    The metric estimates ILD independently for each input HRTF using the same
    estimator configuration and returns the absolute difference between both
    ILD arrays. The output therefore represents the magnitude of ILD change
    per source position.

    Parameters
    ----------
    hrtf_a : HRTF
        Reference HRTF object used for comparison. Must share the same source
        grid (same source positions) as ``hrtf_b``.
    hrtf_b : HRTF
        Compared HRTF object used for comparison. Must share the same source
        grid (same source positions) as ``hrtf_a``.
    mode : {"broad-band", "frequency-dependent"}, default="broad-band"
        ILD mode passed to :func:`ild`.
    output : {"db", "linear"}, default="db"
        Output representation passed to :func:`ild`.
    fft_length : int | None, default=None
        Optional FFT length used when ``mode="frequency-dependent"``.
    epsilon : float, default=1e-12
        Positive floor passed to :func:`ild`.

    Returns
    -------
    np.ndarray
        Absolute ILD differences per position. For ``mode="broad-band"``, the
        shape is ``[positions]``. For ``mode="frequency-dependent"``, the
        shape is ``[positions, frequency_bins]``.

    Use Cases
    ---------
    - Quantify per-position ILD changes after individualization.
    - Compare ILD impact of two processing pipelines.
    - Build position-wise ILD error curves relative to a reference HRTF.

    Notes
    -----
    Both HRTFs must have the same source grid. If source positions differ,
    this function raises a ``ValueError``.

    Examples
    --------
    >>> from hrtfpykit.hrtf.metrics import ild_difference
    >>> ild_diff = ild_difference(hrtf_a, hrtf_b)
    >>> ild_diff.shape
    (hrtf_a.IR.values.shape[0],)
    """
    for label, hrtf in (("hrtf_a", hrtf_a), ("hrtf_b", hrtf_b)):
        if not hasattr(hrtf, "IR") or not hasattr(hrtf, "Sources"):
            raise ValueError(f"{label} must be an HRTF instance")
        if hrtf.IR.values is None:
            raise ValueError(f"{label} IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"{label} IR sample_rate is required")

    mode_key = str(mode).strip().lower()
    if mode_key not in {"broad-band", "frequency-dependent"}:
        raise ValueError("mode must be one of: broad-band, frequency-dependent")
    output_key = str(output).strip().lower()
    if output_key not in {"db", "linear"}:
        raise ValueError("output must be one of: db, linear")

    source_positions_a = np.asarray(hrtf_a.Sources.get_positions(angle_unit="degrees"), dtype=float)
    source_positions_b = np.asarray(hrtf_b.Sources.get_positions(angle_unit="degrees"), dtype=float)
    if source_positions_a.shape != source_positions_b.shape:
        raise ValueError("HRTFs must have the same number of source positions")
    if not np.allclose(source_positions_a, source_positions_b, atol=1e-8, rtol=0.0):
        raise ValueError("HRTFs must share the same source positions for ILD difference")

    if mode_key == "frequency-dependent" and not np.isclose(
        float(hrtf_a.IR.sample_rate),
        float(hrtf_b.IR.sample_rate),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "mode='frequency-dependent' requires equal sample_rate in both HRTFs"
        )

    ild_a = np.asarray(
        ild(
            hrtf_a.IR,
            sample_rate=float(hrtf_a.IR.sample_rate),
            fft_length=fft_length,
            mode=mode_key,
            output=output_key,
            epsilon=epsilon,
        ),
        dtype=float,
    )
    ild_b = np.asarray(
        ild(
            hrtf_b.IR,
            sample_rate=float(hrtf_b.IR.sample_rate),
            fft_length=fft_length,
            mode=mode_key,
            output=output_key,
            epsilon=epsilon,
        ),
        dtype=float,
    )
    if ild_a.shape != ild_b.shape:
        raise ValueError("Calculated ILD arrays must have matching shapes")
    return np.abs(ild_a - ild_b)


def lsd(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    ear: str = "both",
    plane: str = "all",
    elevation: float = 0.0,
    positions: np.ndarray | list | tuple | str | None = None,
    frequencies: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
    reduction: str = "none",
    epsilon: float = 1e-12,
) -> np.ndarray | float:
    """Compute log-spectral distortion (LSD) between two HRTFs in dB.

    This method compares two HRTFs in the logarithmic (dB) domain using the
    selected ear configuration. It can evaluate the left ear, the right ear,
    or both ears together. When ``ear="both"``, LSD is computed for both ear
    channels and then averaged across the ear axis before any optional output
    reduction. It supports full-grid evaluation, plane-restricted evaluation,
    frequency selection, and reduction modes for per-position, per-frequency,
    or global scalar outputs.

    Design rule
    -----------
    When ``frequencies=None``, LSD is computed only in the 20 Hz to 20 kHz
    band. This excludes DC (0 Hz) by default.

    Parameters
    ----------
    hrtf_a : HRTF
        First HRTF used in the comparison.
    hrtf_b : HRTF
        Second HRTF used in the comparison.
    ear : {"left", "right", "both"}, default="both"
        Ear configuration used during the comparison.
        - ``"left"`` evaluates only the left-ear TF values.
        - ``"right"`` evaluates only the right-ear TF values.
        - ``"both"`` evaluates both ear channels and averages the resulting
          LSD values across ears.
    plane : {"all", "horizontal", "median"}, default="all"
        Spatial subset of source positions:
        - ``"all"`` uses the full source grid.
        - ``"horizontal"`` uses the nearest available horizontal plane at
          ``elevation``.
        - ``"median"`` uses the canonical median plane.
    elevation : float, default=0.0
        Requested elevation in degrees used only when
        ``plane="horizontal"``.
    positions : np.ndarray | list | tuple | str | None, default=None
        Optional position selector. Accepts one position query or a collection
        of queries in the same format used across plot APIs (named positions
        and numeric spherical queries). When provided, the resolved positions
        are intersected with the selected ``plane``.
    frequencies : float | list[float] | tuple[float, ...] | np.ndarray | None, default=None
        Optional frequency selector in Hz. Accepts multiple frequency bins. Each target is mapped 
        to the nearest available TF bin. ``None`` selects bins in the 20 Hz to 20 kHz band. If you
        wanna calculate the lsd for all "avilable" frquency bins pass : 
        frequencies=hrtf.TF.frequency_bins
    reduction : {"none", "locations", "frequencies", "global"}, default="none"
        Aggregation mode applied after dB difference computation:
        - ``"none"``:
          returns absolute dB differences per selected position and frequency.
        - ``"locations"``:
          returns RMS over locations for each selected frequency.
        - ``"frequencies"``:
          returns RMS over frequencies for each selected location.
        - ``"global"``:
          returns one global RMS value across all selected locations and
          frequencies.
    epsilon : float, default=1e-12
        Positive lower bound applied to magnitudes before conversion to dB.
        This avoids invalid values from ``log10(0)``.

    Returns
    -------
    np.ndarray | float
        LSD values in dB. Shape depends on ``reduction`` and on whether a
        single frequency was selected:
        - ``reduction="none"``:
          ``(positions, frequencies)`` or ``(positions,)`` for one frequency.
        - ``reduction="locations"``:
          ``(frequencies,)`` or ``float`` for one frequency.
        - ``reduction="frequencies"``:
          ``(positions,)``.
        - ``reduction="global"``:
          ``float``.

    Use Cases
    ---------
    - Compare two full HRTFs across all positions and all frequencies.
    - Evaluate LSD per frequency while averaging across spatial locations.
    - Evaluate LSD per location while averaging across frequencies.
    - Inspect LSD in a specific horizontal elevation plane.
    - Inspect LSD in the canonical median plane at one target frequency.

    Examples
    --------
    Global LSD scalar for the selected configuration:

    >>> lsd_scalar = lsd(hrtf_a, hrtf_b, reduction="global")

    Full LSD map across all positions and frequencies for the left ear:

    >>> lsd_map = lsd(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     ear="left",
    ...     reduction="none",
    ... )
    >>> lsd_map.ndim
    2

    Per-frequency LSD averaged across locations:

    >>> lsd_per_frequency = lsd(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     ear="left",
    ...     reduction="locations",
    ... )

    Per-location LSD averaged across frequencies in a horizontal plane:

    >>> lsd_per_location = lsd(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     ear="right",
    ...     plane="horizontal",
    ...     elevation=0.0,
    ...     reduction="frequencies",
    ... )

    Global scalar LSD at one selected frequency in the median plane:

    >>> lsd_plane_scalar = lsd(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     ear="left",
    ...     plane="median",
    ...     frequencies=4000.0,
    ...     reduction="global",
    ... )
    """
    for label, hrtf in (("hrtf_a", hrtf_a), ("hrtf_b", hrtf_b)):
        if not hasattr(hrtf, "TF") or not hasattr(hrtf, "Sources"):
            raise ValueError(f"{label} must be an HRTF instance")
        if hrtf.TF.values is None:
            raise ValueError(f"{label} TF data is not available")
        if hrtf.TF.frequency_bins is None:
            raise ValueError(f"{label} TF frequency_bins are required")

    plane_key = str(plane).strip().lower()
    if plane_key not in {"all", "horizontal", "median"}:
        raise ValueError("plane must be one of: all, horizontal, median")

    reduction_key = str(reduction).strip().lower()
    if reduction_key not in {"none", "locations", "frequencies", "global"}:
        raise ValueError("reduction must be one of: none, locations, frequencies, global")

    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    source_positions_a = np.asarray(hrtf_a.Sources.get_positions(angle_unit="degrees"), dtype=float)
    source_positions_b = np.asarray(hrtf_b.Sources.get_positions(angle_unit="degrees"), dtype=float)
    if source_positions_a.shape != source_positions_b.shape:
        raise ValueError("HRTFs must have the same number of source positions")
    if not np.allclose(source_positions_a, source_positions_b, atol=1e-8, rtol=0.0):
        raise ValueError("HRTFs must share the same source positions for LSD")

    tf_a = np.asarray(hrtf_a.TF.values)
    tf_b = np.asarray(hrtf_b.TF.values)
    if tf_a.ndim != 3 or tf_b.ndim != 3:
        raise ValueError("TF values must have shape (positions, ears, frequency_bins)")
    if tf_a.shape != tf_b.shape:
        raise ValueError("HRTFs must have matching TF shapes for LSD")
    if tf_a.shape[0] != source_positions_a.shape[0]:
        raise ValueError("TF positions axis must match source positions count")
    if tf_a.shape[1] < 2:
        raise ValueError("TF ear axis must contain at least two channels (0=left, 1=right)")

    frequency_bins_a = np.asarray(hrtf_a.TF.frequency_bins, dtype=float).reshape(-1)
    frequency_bins_b = np.asarray(hrtf_b.TF.frequency_bins, dtype=float).reshape(-1)
    if frequency_bins_a.shape != frequency_bins_b.shape:
        raise ValueError("HRTFs must have matching TF frequency_bins")
    if not np.allclose(frequency_bins_a, frequency_bins_b, atol=1e-8, rtol=0.0):
        raise ValueError("HRTFs must share the same TF frequency_bins for LSD")
    if frequency_bins_a.size != tf_a.shape[-1]:
        raise ValueError("TF frequency axis length must match frequency_bins length")

    ear_key = str(ear).strip().lower()
    if ear_key not in {"left", "right", "both"}:
        raise ValueError("ear must be one of: left, right, both")
    if ear_key == "left":
        selected_ear_indices = np.array([0], dtype=int)
    elif ear_key == "right":
        selected_ear_indices = np.array([1], dtype=int)
    else:
        selected_ear_indices = np.array([0, 1], dtype=int)

    if plane_key == "all":
        selected_positions = np.arange(source_positions_a.shape[0], dtype=int)
    elif plane_key == "horizontal":
        selected_positions, _ = get_horizontal_plane(hrtf=hrtf_a, elevation=float(elevation))
    else:
        selected_positions, _ = get_median_plane(hrtf=hrtf_a, azimuth=0.0)
    selected_positions = np.asarray(selected_positions, dtype=int).reshape(-1)
    if selected_positions.size == 0:
        raise ValueError("Selected plane has no source positions")

    if positions is not None:
        position_queries = get_position_queries(positions)
        selected_from_queries: list[int] = []
        for query in position_queries:
            position_index, _ = hrtf_a.Sources.get_position_index(
                query,
                coordinate_system="spherical",
            )
            selected_from_queries.append(int(position_index))
        selected_from_queries_array = np.asarray(selected_from_queries, dtype=int)
        selected_positions = np.intersect1d(
            selected_positions,
            selected_from_queries_array,
            assume_unique=False,
        )
        if selected_positions.size == 0:
            raise ValueError("No source positions matched the provided positions and plane filters")

    if frequencies is None:
        selected_frequency_indices = np.where(
            (frequency_bins_a >= 20.0) & (frequency_bins_a <= 20000.0)
        )[0]
        
        if selected_frequency_indices.size == 0:
            raise ValueError(
                "No frequency bins available in the default LSD range [20.0, 20000.0] Hz"
            )
    else:
        if isinstance(frequencies, bool):
            raise ValueError("frequencies must be finite value(s)")
        frequency_values = np.asarray(frequencies, dtype=float).reshape(-1)
        if frequency_values.size == 0:
            raise ValueError("frequencies must contain at least one value when provided")
        if not np.all(np.isfinite(frequency_values)):
            raise ValueError("frequencies must be finite value(s)")
        nearest_frequency_indices = [
            int(np.argmin(np.abs(frequency_bins_a - float(target_frequency))))
            for target_frequency in frequency_values
        ]
        selected_frequency_indices = np.asarray(
            tuple(dict.fromkeys(nearest_frequency_indices)),
            dtype=int,
        )

    tf_values_a = np.asarray(
        tf_a[np.ix_(selected_positions, selected_ear_indices, selected_frequency_indices)],
        dtype=complex,
    )
    tf_values_b = np.asarray(
        tf_b[np.ix_(selected_positions, selected_ear_indices, selected_frequency_indices)],
        dtype=complex,
    )
    if tf_values_a.shape != tf_values_b.shape:
        raise ValueError("Selected TF slices must have matching shapes")

    magnitude_a = np.maximum(np.abs(tf_values_a), epsilon)
    magnitude_b = np.maximum(np.abs(tf_values_b), epsilon)
    db_values_a = magnitude_to_db(magnitude_a)
    db_values_b = magnitude_to_db(magnitude_b)
    difference_db = db_values_a - db_values_b

    if reduction_key == "none":
        absolute_difference_db = np.abs(difference_db)
        absolute_difference_db = np.mean(absolute_difference_db, axis=1)
        if absolute_difference_db.shape[-1] == 1:
            lsd_output: np.ndarray | float = absolute_difference_db[:, 0]
        else:
            lsd_output = absolute_difference_db
    elif reduction_key == "locations":
        rms_over_locations = np.sqrt(np.mean(np.square(difference_db), axis=0))
        rms_over_locations = np.mean(rms_over_locations, axis=0)
        if rms_over_locations.size == 1:
            lsd_output = float(rms_over_locations[0])
        else:
            lsd_output = rms_over_locations
    elif reduction_key == "frequencies":
        if difference_db.shape[-1] == 1:
            raise ValueError("reduction='frequencies' requires multiple selected frequencies")
        rms_over_frequencies = np.sqrt(np.mean(np.square(difference_db), axis=-1))
        rms_over_frequencies = np.mean(rms_over_frequencies, axis=1)
        lsd_output = rms_over_frequencies
    else:
        rms_over_positions_frequencies = np.sqrt(np.mean(np.square(difference_db), axis=(0, 2)))
        lsd_output = float(np.mean(rms_over_positions_frequencies))

    return lsd_output
