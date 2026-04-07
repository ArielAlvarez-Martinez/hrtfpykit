from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from .domain import IR, TF


def get_signal_duration(
    signal: np.ndarray | "IR",
    sample_rate: float | None = None,
) -> float:
    """Compute the duration of a time-domain signal.

    Parameters
    ----------
    signal : np.ndarray | IR
        Time-domain array or ``IR`` object with ``.values``.
    sample_rate : float | None, default=None
        Sample rate in Hz. When ``signal`` is an ``IR`` object and this value
        is omitted, ``IR.sample_rate`` is used.

    Returns
    -------
    float
        Duration in seconds.

    Examples
    --------
    >>> get_signal_duration(np.zeros(480), sample_rate=48000.0)
    0.01
    >>> get_signal_duration(np.zeros((2, 960)), sample_rate=48000.0)
    0.02
    """
    if isinstance(signal, np.ndarray):
        signal_values = signal
        resolved_sample_rate = sample_rate
    else:
        if not hasattr(signal, "values") or not hasattr(signal, "sample_rate"):
            raise ValueError("signal must be a NumPy array or an IR instance")
        signal_values = signal.values
        resolved_sample_rate = sample_rate if sample_rate is not None else signal.sample_rate

    if signal_values is None:
        raise ValueError("Signal data is not available")
    if not isinstance(signal_values, np.ndarray):
        raise ValueError("Signal data must be a NumPy array")
    if signal_values.ndim == 0:
        raise ValueError("Signal data must have at least one dimension")

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

    return float(signal_values.shape[-1]) / resolved_sample_rate


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
    >>> ir = np.array([[[0.0, 0.0, 1.0, 0.0],
    ...                 [0.0, 1.0, 0.0, 0.0]]])
    >>> calculate_itd(ir, sample_rate=48000.0, output="samples")
    array([1])
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
    left_processed = apply_iir_filter(
        left_signals,
        filter="lowpass",
        sample_rate=resolved_sample_rate,
        cutoff=upper_cut_freq,
        order=filter_order,
    )
    right_processed = apply_iir_filter(
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
        ``[..., frequency_bins]``.

    Examples
    --------
    >>> ir = np.array([[[1.0, 0.0, 0.0, 0.0],
    ...                 [0.5, 0.0, 0.0, 0.0]]])
    >>> calculate_ild(ir, sample_rate=48000.0, mode="broad-band", output="db")
    array([6.02059991])
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
        tf_values, _, _ = calculate_tf_from_ir(
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


def get_magnitude(tf: np.ndarray | "TF") -> np.ndarray:
    """Return transfer-function magnitudes.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or ``TF`` object with ``.values``.

    Returns
    -------
    np.ndarray
        Magnitude values computed as ``abs(tf)`` with the same shape as the
        input.

    Examples
    --------
    >>> get_magnitude(np.array([1.0 + 1.0j, 0.0 + 2.0j]))
    array([1.41421356, 2.        ])
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    return np.abs(tf_values)


def magnitude_to_db(
    magnitude: np.ndarray,
    reference: float | str = 1.0,
) -> np.ndarray:
    """Convert linear magnitude values to decibels.

    Parameters
    ----------
    magnitude : np.ndarray
        Non-negative magnitude values.
    reference : float | {"max"}, default=1.0
        Positive reference magnitude used in the conversion
        ``20 * log10(magnitude / reference)``. The special value ``"max"``
        uses the maximum magnitude present in the input array.

    Returns
    -------
    np.ndarray
        Magnitude values in dB with the same shape as the input array.

    Examples
    --------
    >>> magnitude_to_db(np.array([1.0, 2.0]))
    array([0.        , 6.02059991])
    >>> magnitude_to_db(np.array([1.0, 2.0]), reference=2.0)
    array([-6.02059991,  0.        ])
    >>> magnitude_to_db(np.array([1.0, 2.0]), reference="max")
    array([-6.02059991,  0.        ])
    """
    magnitude_values = np.asarray(magnitude, dtype=float)
    if np.any(magnitude_values < 0.0):
        raise ValueError("magnitude values must be non-negative")
    if isinstance(reference, str):
        reference_key = str(reference).strip().lower()
        if reference_key != "max":
            raise ValueError("reference must be a finite, positive float or 'max'")
        reference_value = float(np.max(magnitude_values))
        if not np.isfinite(reference_value) or reference_value <= 0.0:
            raise ValueError("reference='max' requires at least one positive magnitude value")
    else:
        reference_value = float(reference)
        if not np.isfinite(reference_value) or reference_value <= 0.0:
            raise ValueError("reference must be a finite, positive float or 'max'")
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(magnitude_values / reference_value)


def db_to_magnitude(
    magnitude_db: np.ndarray,
    reference: float | str = 1.0,
) -> np.ndarray:
    """Convert decibel magnitudes back to linear values.

    Parameters
    ----------
    magnitude_db : np.ndarray
        Magnitude values in decibels.
    reference : float, default=1.0
        Positive reference magnitude used in the inverse conversion.
        ``"max"`` is not supported here.

    Returns
    -------
    np.ndarray
        Linear magnitude values with the same shape as the input array.

    Examples
    --------
    >>> db_to_magnitude(np.array([0.0, 6.02059991]))
    array([1., 2.])
    >>> db_to_magnitude(np.array([-6.02059991, 0.0]), reference=2.0)
    array([1., 2.])
    """
    magnitude_db_values = np.asarray(magnitude_db, dtype=float)
    if isinstance(reference, str):
        reference_key = str(reference).strip().lower()
        if reference_key == "max":
            raise ValueError("db_to_magnitude does not accept reference='max'")
        raise ValueError("reference must be a finite, positive float")
    reference_value = float(reference)
    if not np.isfinite(reference_value) or reference_value <= 0.0:
        raise ValueError("reference must be a finite, positive float")
    return reference_value * (10.0 ** (magnitude_db_values / 20.0))


def get_magnitude_db(
    tf: np.ndarray | "TF",
    reference: float | str = 1.0,
) -> np.ndarray:
    """Return transfer-function magnitudes directly in decibels.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or ``TF`` object with ``.values``.
    reference : float | {"max"}, default=1.0
        Positive reference magnitude used in the dB conversion. The special
        value ``"max"`` uses the maximum magnitude present in the input TF.

    Returns
    -------
    np.ndarray
        Magnitude values in dB with the same shape as the TF input.

    Examples
    --------
    >>> tf = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    >>> get_magnitude_db(tf)
    array([0.        , 6.02059991])
    >>> get_magnitude_db(tf, reference=2.0)
    array([-6.02059991,  0.        ])
    >>> get_magnitude_db(tf, reference="max")
    array([-6.02059991,  0.        ])
    """
    magnitude = get_magnitude(tf)
    return magnitude_to_db(magnitude, reference=reference)


def get_phase(tf: np.ndarray | "TF", unit: str = "degrees") -> np.ndarray:
    """Return transfer-function phase values.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or ``TF`` object with ``.values``.
    unit : str, default="degrees"
        Output unit. Degree and radian aliases are supported.

    Returns
    -------
    np.ndarray
        Phase values in the requested unit with the same shape as the input TF.

    Examples
    --------
    >>> get_phase(np.array([1.0 + 1.0j]), unit="degrees")
    array([45.])
    >>> np.round(get_phase(np.array([1.0 + 1.0j]), unit="radians"), 4)
    array([0.7854])
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    unit_key = str(unit).strip().lower()
    if unit_key in {"degrees", "degree", "deg"}:
        return np.angle(tf_values, deg=True)
    if unit_key in {"radians", "radian", "rad"}:
        return np.angle(tf_values, deg=False)
    raise ValueError("unit must be one of: degrees, radians")


def modify_phase(
    tf: np.ndarray | "TF",
    new_phase: np.ndarray,
    unit: str = "degrees",
) -> np.ndarray:
    """Replace TF phase values while preserving the original magnitude.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or ``TF`` object with ``.values``.
    new_phase : np.ndarray
        Phase array with the same shape as the TF values.
    unit : str, default="degrees"
        Phase unit used by ``new_phase``. Degree and radian aliases are
        supported.

    Returns
    -------
    np.ndarray
        Complex TF values with the original magnitude and the new phase.

    Examples
    --------
    >>> tf = np.array([1.0 + 1.0j])
    >>> np.round(modify_phase(tf, np.array([0.0]), unit="degrees"), 4)
    array([1.4142+0.j])
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")

    new_phase_values = np.asarray(new_phase, dtype=float)
    if new_phase_values.shape != tf_values.shape:
        raise ValueError("new_phase must match TF shape")

    unit_key = str(unit).strip().lower()
    if unit_key in {"degrees", "degree", "deg"}:
        phase_radians = np.deg2rad(new_phase_values)
    elif unit_key in {"radians", "radian", "rad"}:
        phase_radians = new_phase_values
    else:
        raise ValueError("unit must be one of: degrees, radians")

    magnitude_values = np.abs(tf_values)
    return magnitude_values * np.exp(1j * phase_radians)


def modify_magnitude(
    tf: np.ndarray | "TF",
    new_magnitude: np.ndarray,
    scale: str = "linear",
) -> np.ndarray:
    """Replace TF magnitude values while preserving the original phase.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or ``TF`` object with ``.values``.
    new_magnitude : np.ndarray
        Magnitude array with the same shape as the TF values.
    scale : str, default="linear"
        Scale of ``new_magnitude``. Supported values are ``linear``,
        ``lineal``, and ``db``.

    Returns
    -------
    np.ndarray
        Complex TF values with the new magnitude and the original phase.

    Examples
    --------
    >>> tf = np.array([1.0 + 1.0j])
    >>> np.round(modify_magnitude(tf, np.array([2.0])), 4)
    array([1.4142+1.4142j])
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")

    new_magnitude_values = np.asarray(new_magnitude, dtype=float)
    if new_magnitude_values.shape != tf_values.shape:
        raise ValueError("new_magnitude must match TF shape")

    scale_key = str(scale).strip().lower()
    if scale_key in {"linear", "lineal"}:
        magnitude_values = new_magnitude_values
    elif scale_key in {"db", "decibel", "decibels"}:
        magnitude_values = db_to_magnitude(new_magnitude_values)
    else:
        raise ValueError("scale must be one of: linear, lineal, db")

    if np.any(magnitude_values < 0.0):
        raise ValueError("new_magnitude must be non-negative")

    phase_values = np.angle(tf_values)
    return magnitude_values * np.exp(1j * phase_values)


def get_real(tf: np.ndarray | "TF") -> np.ndarray:
    """Return the real part of transfer-function values.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or ``TF`` object with ``.values``.

    Returns
    -------
    np.ndarray
        Real component of the TF values with the same shape as the input.

    Examples
    --------
    >>> get_real(np.array([1.0 + 2.0j, 3.0 - 4.0j]))
    array([1., 3.])
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    return np.real(tf_values)


def get_imag(tf: np.ndarray | "TF") -> np.ndarray:
    """Return the imaginary part of transfer-function values.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or ``TF`` object with ``.values``.

    Returns
    -------
    np.ndarray
        Imaginary component of the TF values with the same shape as the input.

    Examples
    --------
    >>> get_imag(np.array([1.0 + 2.0j, 3.0 - 4.0j]))
    array([ 2., -4.])
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    return np.imag(tf_values)


def upsampling(
    ir: np.ndarray | "IR",
    new_sample_rate: float,
    sample_rate: float | None = None,
) -> tuple[np.ndarray, float]:
    """Upsample an IR signal using polyphase resampling.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain array or ``IR`` object with ``.values``.
    new_sample_rate : float
        Target sample rate in Hz. It must be strictly greater than the current
        sample rate.
    sample_rate : float | None, default=None
        Source sample rate used when ``ir`` is a NumPy array.

    Returns
    -------
    tuple[np.ndarray, float]
        Tuple ``(resampled_ir, resolved_new_sample_rate)``.

    Examples
    --------
    >>> ir = np.array([1.0, 0.0, 0.0, 0.0])
    >>> resampled_ir, sr = upsampling(ir, new_sample_rate=96000.0, sample_rate=48000.0)
    >>> sr
    96000.0
    >>> resampled_ir.shape[-1] > ir.shape[-1]
    True
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
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")

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

    if isinstance(new_sample_rate, bool):
        raise ValueError("new_sample_rate must be a finite, positive value.")
    try:
        new_sample_rate = float(new_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("new_sample_rate must be a finite, positive value.") from None
    if not np.isfinite(new_sample_rate) or new_sample_rate <= 0.0:
        raise ValueError("new_sample_rate must be a finite, positive value.")
    if new_sample_rate <= resolved_sample_rate:
        raise ValueError("new_sample_rate must be greater than current sample_rate for upsampling")

    ratio = Fraction(new_sample_rate / resolved_sample_rate).limit_denominator(10000)
    resampled_ir = signal.resample_poly(
        ir_values,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )
    return resampled_ir, new_sample_rate


def downsampling(
    ir: np.ndarray | "IR",
    new_sample_rate: float,
    sample_rate: float | None = None,
) -> tuple[np.ndarray, float]:
    """Downsample an IR signal using polyphase resampling.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain array or ``IR`` object with ``.values``.
    new_sample_rate : float
        Target sample rate in Hz. It must be strictly lower than the current
        sample rate.
    sample_rate : float | None, default=None
        Source sample rate used when ``ir`` is a NumPy array.

    Returns
    -------
    tuple[np.ndarray, float]
        Tuple ``(resampled_ir, resolved_new_sample_rate)``.

    Examples
    --------
    >>> ir = np.zeros(8, dtype=float)
    >>> resampled_ir, sr = downsampling(ir, new_sample_rate=24000.0, sample_rate=48000.0)
    >>> sr
    24000.0
    >>> resampled_ir.shape[-1] < ir.shape[-1]
    True
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
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")

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

    if isinstance(new_sample_rate, bool):
        raise ValueError("new_sample_rate must be a finite, positive value.")
    try:
        new_sample_rate = float(new_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("new_sample_rate must be a finite, positive value.") from None
    if not np.isfinite(new_sample_rate) or new_sample_rate <= 0.0:
        raise ValueError("new_sample_rate must be a finite, positive value.")
    if new_sample_rate >= resolved_sample_rate:
        raise ValueError("new_sample_rate must be lower than current sample_rate for downsampling")

    ratio = Fraction(new_sample_rate / resolved_sample_rate).limit_denominator(10000)
    resampled_ir = signal.resample_poly(
        ir_values,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )
    return resampled_ir, new_sample_rate


def apply_window(ir: np.ndarray | "IR", window_name: str) -> np.ndarray:
    """Apply a named time-domain window to IR samples.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain array or ``IR`` object with ``.values``.
    window_name : str
        Window identifier. Supported values are ``hann``, ``hamming``,
        ``blackman``, and ``rectangular``.

    Returns
    -------
    np.ndarray
        Windowed IR values.

    Examples
    --------
    >>> np.round(apply_window(np.ones(4), "hann"), 4)
    array([0.  , 0.75, 0.75, 0.  ])
    >>> apply_window(np.ones(4), "rectangular")
    array([1., 1., 1., 1.])

    """
    if isinstance(ir, np.ndarray):
        ir_values = ir
    else:
        if not hasattr(ir, "values"):
            raise ValueError("ir must be a NumPy array or an IR instance")
        ir_values = ir.values
    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")
    length = ir_values.shape[-1]
    if length <= 0:
        raise ValueError("IR data must contain at least one sample")
    key = window_name.strip().lower()
    if key in {"hann", "hanning"}:
        window_values = np.hanning(length)
    elif key in {"rectangular"}:
        window_values = np.ones(length)
    elif key == "hamming":
        window_values = np.hamming(length)
    elif key == "blackman":
        window_values = np.blackman(length)
    else:
        raise ValueError(
            "window_name must be one of: hann, hamming, blackman, rectangular"
        )
    return ir_values * window_values

def apply_padding(
    ir: np.ndarray | "IR",
    padding_length: int,
    location: str = "end",
    value: float | complex = 0,
) -> np.ndarray:
    """Pad IR values along the last axis.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain signal container with ``.values``.
    padding_length : int
        Number of samples added to the IR.
    location : {"start", "end"}, default="end"
        Side where the padding is applied.
    value : float | complex, default=0
        Constant value used in the padded region.

    Returns
    -------
    np.ndarray
        Padded IR array.

    Examples
    --------
    >>> apply_padding(np.array([1.0, 2.0]), padding_length=2, location="end")
    array([1., 2., 0., 0.])
    >>> apply_padding(np.array([1.0, 2.0]), padding_length=2, location="start", value=-1.0)
    array([-1., -1.,  1.,  2.])
    """

    if isinstance(ir, np.ndarray):
        ir_values = ir
    elif hasattr(ir, "values") and hasattr(ir, "sample_rate"):
        ir_values = ir.values
    else:
        ir_values = None
    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if isinstance(padding_length, bool) or not isinstance(padding_length, int):
        raise ValueError("Padding must be an integer")
    if padding_length < 0:
        raise ValueError("Padding must be non-negative")
    if padding_length == 0:
        return ir_values
    location_key = location.strip().lower()
    if location_key == "start":
        before, after = padding_length, 0
    elif location_key == "end":
        before, after = 0, padding_length
    else:
        raise ValueError("Padding location must be 'start' or 'end'")
    pad_width = [(0, 0)] * (ir_values.ndim - 1) + [(before, after)]
    return np.pad(
        ir_values,
        pad_width,
        mode="constant",
        constant_values=value,
    )


def apply_fir_filter(
    ir: np.ndarray | "IR",
    filter: str,
    sample_rate: float | None = None,
    cutoff: float | tuple[float, float] | None = None,
    num_taps: int = 101,
    window: str | None = None,
) -> np.ndarray:
    """Apply an FIR filter to IR data.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain array or ``IR`` object with ``.values``.
    filter : str
        Filter family. Low-pass, high-pass, and band-pass aliases are
        supported.
    sample_rate : float | None, default=None
        Sample rate in Hz.
    cutoff : float | tuple[float, float] | None, default=None
        Cutoff value. Use a scalar for low-pass or high-pass filtering and a
        tuple for band-pass filtering.
    num_taps : int, default=101
        Odd FIR length.
    window : str | None, default=None
        FIR design window. Supported values are ``hann``, ``hamming``,
        ``blackman``, and ``rectangular``.

    Returns
    -------
    np.ndarray
        Filtered IR values with the same shape as the input.

    Examples
    --------
    >>> ir = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> filtered = apply_fir_filter(ir, filter="lowpass", sample_rate=48000.0, cutoff=3000.0, num_taps=5)
    >>> filtered.shape
    (7,)
    """
    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "values"):
            ir = ir.values
        else:
            ir = None
    if ir is None:
        raise ValueError("IR data is not available")

    filter_type = str(filter).strip().lower()
    if sample_rate is None:
        raise ValueError("sample_rate is required for filters")
    if cutoff is None:
        raise ValueError("cutoff is required for filters")
    if isinstance(num_taps, bool) or not isinstance(num_taps, int):
        raise ValueError("num_taps must be an integer")
    if num_taps <= 0:
        raise ValueError("num_taps must be positive")
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd")

    window_value = None
    if window is None:
        window_value = "boxcar"
    else:
        window_type = str(window).strip().lower()
        if window_type in {"hann", "hanning"}:
            window_value = "hann"
        elif window_type in {"rectangular"}:
            window_value = "boxcar"
        elif window_type == "hamming":
            window_value = "hamming"
        elif window_type == "blackman":
            window_value = "blackman"
        else:
            raise ValueError("window must be one of: hann, hamming, blackman, rectangular")
    nyquist = 0.5 * sample_rate
    if filter_type in {"lowpass", "low-pass", "lp"}:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0.0 or cutoff_value >= nyquist:
            raise ValueError("cutoff must be between 0 and Nyquist for lowpass")
        kernel_values = signal.firwin(
            num_taps,
            cutoff_value,
            window=window_value,
            pass_zero=True,
            fs=sample_rate,
        )
    elif filter_type in {"highpass", "high-pass", "hp"}:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0.0 or cutoff_value >= nyquist:
            raise ValueError("cutoff must be between 0 and Nyquist for highpass")
        kernel_values = signal.firwin(
            num_taps,
            cutoff_value,
            window=window_value,
            pass_zero=False,
            fs=sample_rate,
        )
    elif filter_type in {"bandpass", "band-pass", "bp"}:
        if not isinstance(cutoff, tuple) or len(cutoff) != 2:
            raise ValueError("cutoff must be (low, high) for bandpass")
        cutoff_low = float(cutoff[0])
        cutoff_high = float(cutoff[1])
        if cutoff_low <= 0.0 or cutoff_high >= nyquist or cutoff_low >= cutoff_high:
            raise ValueError("cutoff must satisfy 0 < low < high < Nyquist for bandpass")
        kernel_values = signal.firwin(
            num_taps,
            [cutoff_low, cutoff_high],
            window=window_value,
            pass_zero=False,
            fs=sample_rate,
        )
    else:
        raise ValueError("filter must be one of: lowpass, highpass, bandpass")

    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel_values, mode="same"),
        axis=-1,
        arr=ir,
    )


def apply_iir_filter(
    ir: np.ndarray | "IR",
    filter: str,
    sample_rate: float | None = None,
    cutoff: float | tuple[float, float] | None = None,
    order: int = 10,
) -> np.ndarray:
    """Apply an IIR Butterworth filter to IR data.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain array or ``IR`` object with ``.values``.
    filter : str
        Filter family. Low-pass, high-pass, and band-pass aliases are
        supported.
    sample_rate : float | None, default=None
        Sample rate in Hz.
    cutoff : float | tuple[float, float] | None, default=None
        Cutoff value. Use a scalar for low-pass or high-pass filtering and a
        tuple for band-pass filtering.
    order : int, default=10
        Positive Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered IR values with the same shape as the input.

    Examples
    --------
    >>> ir = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> filtered = apply_iir_filter(ir, filter="lowpass", sample_rate=48000.0, cutoff=3000.0, order=4)
    >>> filtered.shape
    (7,)
    """
    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "values"):
            ir = ir.values
        else:
            ir = None
    if ir is None:
        raise ValueError("IR data is not available")

    filter_type = str(filter).strip().lower()
    if sample_rate is None:
        raise ValueError("sample_rate is required for filters")
    if cutoff is None:
        raise ValueError("cutoff is required for filters")
    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError("order must be an integer")
    if order <= 0:
        raise ValueError("order must be positive")

    nyquist = 0.5 * sample_rate
    if filter_type in {"lowpass", "low-pass", "lp"}:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0.0 or cutoff_value >= nyquist:
            raise ValueError("cutoff must be between 0 and Nyquist for lowpass")
        b, a = signal.butter(order, cutoff_value, btype="lowpass", fs=sample_rate)
    elif filter_type in {"highpass", "high-pass", "hp"}:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0.0 or cutoff_value >= nyquist:
            raise ValueError("cutoff must be between 0 and Nyquist for highpass")
        b, a = signal.butter(order, cutoff_value, btype="highpass", fs=sample_rate)
    elif filter_type in {"bandpass", "band-pass", "bp"}:
        if not isinstance(cutoff, tuple) or len(cutoff) != 2:
            raise ValueError("cutoff must be (low, high) for bandpass")
        cutoff_low = float(cutoff[0])
        cutoff_high = float(cutoff[1])
        if cutoff_low <= 0.0 or cutoff_high >= nyquist or cutoff_low >= cutoff_high:
            raise ValueError("cutoff must satisfy 0 < low < high < Nyquist for bandpass")
        b, a = signal.butter(
            order,
            [cutoff_low, cutoff_high],
            btype="bandpass",
            fs=sample_rate,
        )
    else:
        raise ValueError("filter must be one of: lowpass, highpass, bandpass")

    return signal.lfilter(b, a, ir, axis=-1)


def minimum_phase(
    data: np.ndarray | "IR",
    method: str = "homomorphic",
    fft_length: int | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Convert IR data into a minimum-phase IR.

    Parameters
    ----------
    data : np.ndarray | IR
        Real-valued IR samples stored as a NumPy array or ``IR`` object.
    method : {"homomorphic", "cepstrum", "real_cepstrum"}, default="homomorphic"
        Minimum-phase strategy. ``homomorphic`` and ``real_cepstrum`` use a
        log-magnitude real cepstrum, while ``cepstrum`` uses a complex
        cepstrum with unwrapped phase.
    fft_length : int | None, default=None
        Optional FFT length used for cepstral operations.
    epsilon : float, default=1e-12
        Positive floor applied to magnitude values before logarithms.

    Returns
    -------
    np.ndarray
        Minimum-phase IR array with the same trailing length as the resolved
        IR input.

    Examples
    --------
    >>> ir = np.array([1.0, 0.5, 0.25, 0.0])
    >>> minimum_phase(ir).shape
    (4,)
    """
    if isinstance(data, np.ndarray):
        ir_values = data
    else:
        if hasattr(data, "sample_rate"):
            ir_values = data.values
        else:
            raise ValueError("data must be a NumPy array or an IR instance")

    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")

    method_key = str(method).strip().lower()
    if method_key not in {"homomorphic", "cepstrum", "real_cepstrum"}:
        raise ValueError("method must be one of: homomorphic, cepstrum, real_cepstrum")

    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    ir_real = np.real_if_close(ir_values, tol=1000)
    if np.iscomplexobj(ir_real):
        raise ValueError("IR data must be real-valued for minimum-phase conversion")
    ir_array = np.asarray(ir_real, dtype=float)
    ir_length = int(ir_array.shape[-1])

    if fft_length is None:
        fft_length_used = max(2, 2 * max(ir_length - 1, 1))
    else:
        if isinstance(fft_length, bool) or not isinstance(fft_length, int):
            raise ValueError("fft_length must be an integer")
        if fft_length < 2:
            raise ValueError("fft_length must be at least 2")
        fft_length_used = int(fft_length)
    if fft_length_used < ir_length:
        raise ValueError("fft_length must be greater than or equal to IR length")

    minimum_phase_values = np.empty_like(ir_array, dtype=float)
    ir_reshaped = ir_array.reshape(-1, ir_length)
    minimum_phase_reshaped = minimum_phase_values.reshape(-1, ir_length)

    for index in range(ir_reshaped.shape[0]):
        if method_key in {"homomorphic", "real_cepstrum"}:
            spectrum_values = np.fft.rfft(ir_reshaped[index], n=fft_length_used)
            magnitude_values = np.maximum(np.abs(spectrum_values), epsilon)
            log_magnitude = np.log(magnitude_values)
            cepstrum = np.fft.irfft(log_magnitude, n=fft_length_used)

            minimum_cepstrum = np.zeros(fft_length_used, dtype=float)
            minimum_cepstrum[0] = cepstrum[0]
            if fft_length_used % 2 == 0:
                half_index = fft_length_used // 2
                minimum_cepstrum[1:half_index] = 2.0 * cepstrum[1:half_index]
                minimum_cepstrum[half_index] = cepstrum[half_index]
            else:
                half_index = (fft_length_used + 1) // 2
                minimum_cepstrum[1:half_index] = 2.0 * cepstrum[1:half_index]

            minimum_spectrum = np.exp(np.fft.rfft(minimum_cepstrum, n=fft_length_used))
            minimum_ir = np.fft.irfft(minimum_spectrum, n=fft_length_used)
        else:
            spectrum_values = np.fft.fft(ir_reshaped[index], n=fft_length_used)
            magnitude_values = np.maximum(np.abs(spectrum_values), epsilon)
            unwrapped_phase = np.unwrap(np.angle(spectrum_values))
            complex_log_spectrum = np.log(magnitude_values) + 1j * unwrapped_phase
            cepstrum = np.fft.ifft(complex_log_spectrum, n=fft_length_used)

            minimum_cepstrum = np.zeros(fft_length_used, dtype=complex)
            minimum_cepstrum[0] = cepstrum[0]
            if fft_length_used % 2 == 0:
                half_index = fft_length_used // 2
                minimum_cepstrum[1:half_index] = 2.0 * cepstrum[1:half_index]
                minimum_cepstrum[half_index] = cepstrum[half_index]
            else:
                half_index = (fft_length_used + 1) // 2
                minimum_cepstrum[1:half_index] = 2.0 * cepstrum[1:half_index]

            minimum_spectrum = np.exp(np.fft.fft(minimum_cepstrum, n=fft_length_used))
            minimum_ir = np.fft.ifft(minimum_spectrum, n=fft_length_used)
            minimum_ir = np.real_if_close(minimum_ir, tol=1000)
            if np.iscomplexobj(minimum_ir):
                minimum_ir = np.real(minimum_ir)

        minimum_phase_reshaped[index] = np.asarray(minimum_ir[:ir_length], dtype=float)

    return minimum_phase_values


def calculate_tf_from_ir(
    ir: np.ndarray | "IR",
    sample_rate: float | None = None,
    fft_length: int | None = None,
    window_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int] | "TF":
    """Compute TF values from IR values using an FFT.

    Parameters
    ----------
    ir : np.ndarray | IR
        IR array or ``IR`` object.
    sample_rate : float | None, default=None
        Sample rate in Hz for NumPy input. Optional for ``IR`` input when
        ``IR.sample_rate`` is available.
    fft_length : int | None, default=None
        FFT size. When omitted, the IR length is used.
    window_name : str | None, default=None
        Optional time-domain window applied before the FFT.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int] | TF
        For NumPy input, returns ``(tf_values, frequency_bins, fft_length_used)``.
        For ``IR`` input, returns the updated ``TF`` object linked to the same
        ``HRTF`` instance.

    Examples
    --------
    >>> ir = np.array([1.0, 0.0, 0.0, 0.0])
    >>> tf, frequency_bins, fft_length_used = calculate_tf_from_ir(ir, sample_rate=48000.0)
    >>> tf.shape, frequency_bins.shape, fft_length_used
    ((3,), (3,), 4)
    """
    ir_object = None
    if isinstance(ir, np.ndarray):
        ir_values = ir
        if ir_values.size == 0 or np.all(ir_values == 0):
            raise ValueError("NumPy ir array requires non empty values.")
        resolved_sample_rate = sample_rate
    else:
        if not hasattr(ir, "_hrtf") or not hasattr(ir, "values") or not hasattr(ir, "sample_rate"):
            raise ValueError("ir must be a NumPy array or an IR instance")
        ir_object = ir
        ir_values = ir.values
        if ir_values is None:
            raise ValueError("IR data is not available; cannot compute TF.")
        if not isinstance(ir_values, np.ndarray):
            raise ValueError("IR.values must be a NumPy array.")
        if ir_values.size == 0 or np.all(ir_values == 0):
            raise ValueError("IR requires non empty 'values'.")
        resolved_sample_rate = sample_rate if sample_rate is not None else ir.sample_rate

    if resolved_sample_rate is None:
        if ir_object is None:
            raise ValueError("sample_rate is required when ir is a NumPy array")
        raise ValueError("sample_rate is required when IR.sample_rate is unavailable")
    try:
        resolved_sample_rate = float(resolved_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("sample_rate must be a finite, positive value.") from None
    if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
        raise ValueError("sample_rate must be a finite, positive value.")

    if fft_length is None:
        fft_length_used = int(ir_values.shape[-1])
    else:
        if isinstance(fft_length, bool) or not isinstance(fft_length, int):
            raise ValueError("fft_length must be an integer")
        if fft_length <= 0:
            raise ValueError("fft_length must be positive")
        fft_length_used = int(fft_length)

    if fft_length_used < 2:
        raise ValueError("FFT length must contain at least two points.")

    ir_used = ir_values
    if window_name:
        ir_used = apply_window(ir_values, window_name)

    tf_values = np.fft.rfft(ir_used, n=fft_length_used, axis=-1)
    frequency_bins = np.fft.rfftfreq(fft_length_used, d=1.0 / resolved_sample_rate)
    if ir_object is not None:
        tf_object = ir_object._hrtf.TF
        tf_object.values = tf_values
        tf_object.frequency_bins = frequency_bins
        ir_object._hrtf.fft_length = fft_length_used
        return tf_object
    return tf_values, frequency_bins, fft_length_used


def calculate_ir_from_tf(
    tf: np.ndarray | "TF",
    frequency_bins: np.ndarray | None = None,
    sample_rate: float | None = None,
    spectrum_type: str | None = None,
) -> tuple[np.ndarray, float] | "IR":
    """Compute IR values from TF values using inverse FFT routines.

    Parameters
    ----------
    tf : np.ndarray | TF
        TF array or ``TF`` object.
    frequency_bins : np.ndarray | None, default=None
        Optional frequency-bin vector matching the TF length.
    sample_rate : float | None, default=None
        Sample rate used when frequency bins must be inferred for NumPy TF
        input.
    spectrum_type : str | None, default=None
        Required when inferring bins. Supported values are ``"positive"``
        for one-sided spectra and ``"complete"`` for full complex spectra.

    Returns
    -------
    tuple[np.ndarray, float, int] | IR
        For NumPy input, returns ``(ir_values, sample_rate, fft_length_used)``.
        For ``TF`` input, returns the updated ``IR`` object linked to the same
        ``HRTF`` instance.

    Examples
    --------
    >>> tf = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    >>> frequency_bins = np.array([0.0, 12000.0, 24000.0])
    >>> ir, sample_rate, fft_length_used = calculate_ir_from_tf(tf, frequency_bins=frequency_bins)
    >>> ir.shape, sample_rate, fft_length_used
    ((4,), 48000.0, 4)
    """
    tf_object = None
    if isinstance(tf, np.ndarray):
        tf_values = tf
        if tf_values.size == 0 or np.all(tf_values == 0):
            raise ValueError("NumPy tf array requires non empty values.")
    else:
        if not hasattr(tf, "_hrtf") or not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_object = tf
        tf_values = tf.values
        if tf_values is None:
            raise ValueError("TF data is not available; cannot compute IR.")
        if not isinstance(tf_values, np.ndarray):
            raise ValueError("TF.values must be a NumPy array.")
        if tf_values.size == 0 or np.all(tf_values == 0):
            raise ValueError("TF requires non empty 'values'.")

    tf_used = tf_values

    if tf_used.shape[-1] < 2:
        raise ValueError("TF length must contain at least two points.")

    if frequency_bins is None:
        if tf_object is not None:
            raise ValueError(
                "calculate_ir_from_tf requires 'frequency_bins' when tf is a TF instance."
            )
        if sample_rate is None:
            raise ValueError(
                "sample_rate is required when frequency_bins is not provided for NumPy TF."
            )
        try:
            resolved_sample_rate = float(sample_rate)
        except (TypeError, ValueError):
            raise ValueError("sample_rate must be a finite, positive value.") from None
        if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
            raise ValueError("sample_rate must be a finite, positive value.")
        if spectrum_type is None:
            raise ValueError(
                "spectrum_type is required when frequency_bins is not provided for NumPy TF."
            )
        spectrum_key = str(spectrum_type).strip().lower()
        if spectrum_key == "positive":
            inferred_fft_length = 2 * (tf_used.shape[-1] - 1)
            frequency_bins_array = np.fft.rfftfreq(
                inferred_fft_length,
                d=1.0 / resolved_sample_rate,
            )
        elif spectrum_key == "complete":
            inferred_fft_length = tf_used.shape[-1]
            frequency_bins_array = np.fft.fftshift(
                np.fft.fftfreq(
                    inferred_fft_length,
                    d=1.0 / resolved_sample_rate,
                )
            )
        else:
            raise ValueError(
                "spectrum_type must be 'positive' or 'complete' when inferring frequency_bins."
            )
    else:
        frequency_bins_array = np.asarray(frequency_bins, dtype=float)

    if frequency_bins_array.size == 0 or np.all(frequency_bins_array == 0):
        raise ValueError("TF requires non empty 'frequency_bins'.")

    if frequency_bins_array.ndim != 1 or frequency_bins_array.size != tf_used.shape[-1]:
        raise ValueError("frequency_bins must be 1D and match TF length")
    if frequency_bins_array.size < 2:
        raise ValueError("frequency_bins must contain at least two points")

    diffs = np.diff(frequency_bins_array)
    step = float(diffs[0])
    if step <= 0.0 or not np.allclose(diffs, step, rtol=1e-5, atol=1e-8):
        raise ValueError("frequency_bins must be uniformly spaced and increasing")

    if float(np.min(frequency_bins_array)) < 0.0:
        expected_n_fft = frequency_bins_array.size
        fft_length_used = expected_n_fft
        ir_values = np.fft.ifft(tf_used, n=fft_length_used, axis=-1)
        ir_values = np.real_if_close(ir_values, tol=1000)
    else:
        if not np.isclose(frequency_bins_array[0], 0.0):
            raise ValueError("frequency_bins must start at 0 Hz for one-sided spectra")
        expected_n_fft = 2 * (frequency_bins_array.size - 1)
        fft_length_used = expected_n_fft
        ir_values = np.fft.irfft(tf_used, n=fft_length_used, axis=-1)

    sample_rate = step * expected_n_fft
    if tf_object is not None:
        ir_object = tf_object._hrtf.IR
        ir_object.values = ir_values
        ir_object.sample_rate = sample_rate
        tf_object._hrtf.fft_length = fft_length_used
        return ir_object
    return ir_values, sample_rate, fft_length_used
