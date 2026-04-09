from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

from .dsp import iir_filter, magnitude_to_db, tf_from_ir

if TYPE_CHECKING:
    from .domain import IR


def calculate_itd(
    ir: np.ndarray | "IR",
    method: str = "threshold",
    sample_rate: float | None = None,
    output: str = "seconds",
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

    Examples
    --------
    Estimate ITD in samples for a short binaural impulse:

    >>> ir = np.array([[[0.0, 0.0, 1.0, 0.0],
    ...                 [0.0, 1.0, 0.0, 0.0]]])
    >>> calculate_itd(ir, sample_rate=48000.0, output="samples")
    array([1])

    Convert the same ITD estimate to seconds:

    >>> calculate_itd(ir, sample_rate=48000.0, output="seconds")
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


def calculate_ild(
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

    Examples
    --------
    Measure the broad-band ILD of a simple binaural impulse:

    >>> ir = np.array([[[1.0, 0.0, 0.0, 0.0],
    ...                 [0.5, 0.0, 0.0, 0.0]]])
    >>> calculate_ild(ir, sample_rate=48000.0, mode="broad-band", output="db")
    array([6.02059991])

    Inspect the frequency-dependent ILD shape for the same signal:

    >>> calculate_ild(ir, sample_rate=48000.0, mode="frequency-dependent", output="linear").shape
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
