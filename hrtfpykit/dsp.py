from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import warnings
import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from .domain import IR, TF


def get_signal_duration(
    signal: np.ndarray | "IR",
    sample_rate: float | None = None,
) -> float:
    """General Description:
    Compute the duration of a time-domain signal from its sample count and sample rate.

    Parameters:
    - signal: Time-domain array or `IR` object with `.values`.
    - sample_rate: Optional sample rate in Hz. If omitted for an `IR` object,
      the method uses `IR.sample_rate`.

    Returns:
    - Duration in seconds as a Python `float`.

    Use Cases:
    - Report HRIR length in physical time units.
    - Build time-based crop, window, or plotting settings.

    Examples:
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
    """General Description:
    Estimate interaural time differences from binaural impulse responses.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`. Expected layout is
      `[..., ear, samples]` with ear convention `0=left`, `1=right`.
    - method: ITD estimator. Supported values are `threshold` and `maxiacce`.
    - sample_rate: Optional sample rate in Hz for NumPy input. For `IR` input,
      the method uses `IR.sample_rate` when this argument is omitted.
    - output: Output unit for the ITD values, either `seconds` or `samples`.
    - thresh_level: Threshold offset in dB used by the `threshold` method.
    - upper_cut_freq: Low-pass cutoff in Hz applied before ITD estimation.
    - filter_order: Positive Butterworth filter order used in the preprocessing stage.

    Returns:
    - Array of ITD values with shape `ir.shape[:-2]`. Positive values mean the
      left ear is delayed relative to the right ear.

    Use Cases:
    - Extract binaural timing cues from HRIR datasets.
    - Compare source-dependent ITD behavior across positions.

    Examples:
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
    data: np.ndarray | "IR" | "TF",
    domain: str = "auto",
    sample_rate: float | None = None,
    fft_length: int | None = None,
    mode: str = "broad-band",
    output: str = "db",
    epsilon: float = 1e-12,
) -> np.ndarray:
    """General Description:
    Compute interaural level differences from binaural IR or TF data.

    Parameters:
    - data: Binaural signal container. Accepted inputs are:
      - `np.ndarray` with layout `[..., ear, samples_or_bins]`
      - `IR` object with `.values`
      - `TF` object with `.values`
      Ear convention is `0=left`, `1=right`.
    - domain: Input domain. Supported values are `auto`, `ir`, and `tf`.
      In `auto` mode, `IR` objects map to `ir`, `TF` objects map to `tf`,
      complex NumPy arrays map to `tf`, and real NumPy arrays map to `ir`.
    - sample_rate: Sample rate in Hz used only for NumPy IR inputs.
    - fft_length: Optional FFT length used only when IR data must be converted to TF.
    - mode: ILD mode, either `broad-band` or `frequency-dependent`.
    - output: Output representation, either `db` or `linear`.
    - epsilon: Positive floor used to avoid division by zero in level ratios.

    Returns:
    - ILD array in the requested mode:
      - `mode='broad-band'` returns shape `[...]`
      - `mode='frequency-dependent'` returns shape `[..., frequency_bins]`
      For `output='db'`, the result is `20*log10(left/right)`.

    Use Cases:
    - Extract broadband or spectral binaural level cues from HRIR/HRTF data.
    - Compare left-right level balance across source positions.

    Examples:
    >>> ir = np.array([[[1.0, 0.0, 0.0, 0.0],
    ...                 [0.5, 0.0, 0.0, 0.0]]])
    >>> calculate_ild(ir, domain="ir", sample_rate=48000.0, mode="broad-band", output="db")
    array([6.02059991])
    >>> tf = np.fft.rfft(ir, axis=-1)
    >>> calculate_ild(tf, domain="tf", mode="frequency-dependent", output="linear").shape
    (3,)
    """
    data_object = None
    if isinstance(data, np.ndarray):
        data_values = data
    elif hasattr(data, "values"):
        data_object = data
        data_values = data.values
    else:
        raise ValueError("data must be a NumPy array, IR, or TF instance")

    if data_values is None:
        raise ValueError("Signal data is not available")
    if not isinstance(data_values, np.ndarray):
        raise ValueError("Signal data must be a NumPy array")
    if data_values.size == 0:
        raise ValueError("Signal data must be non-empty")
    if data_values.ndim < 2:
        raise ValueError("Signal data must include at least ear and time/frequency axes")

    domain_key = str(domain).strip().lower()
    if domain_key not in {"auto", "ir", "tf"}:
        raise ValueError("domain must be one of: auto, ir, tf")

    if domain_key == "auto":
        if data_object is not None and hasattr(data_object, "sample_rate"):
            domain_key = "ir"
        elif data_object is not None and hasattr(data_object, "frequency_bins"):
            domain_key = "tf"
        else:
            domain_key = "tf" if np.iscomplexobj(data_values) else "ir"

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

    if domain_key == "ir":
        if data_object is not None and hasattr(data_object, "sample_rate"):
            resolved_sample_rate = sample_rate if sample_rate is not None else data_object.sample_rate
        else:
            resolved_sample_rate = sample_rate
        if resolved_sample_rate is None:
            raise ValueError("sample_rate is required for IR NumPy inputs")
        if mode_key == "frequency-dependent":
            tf_values, _, _ = calculate_tf_from_ir(
                data_values,
                sample_rate=resolved_sample_rate,
                fft_length=fft_length,
            )
        else:
            tf_values = None
    else:
        tf_values = np.asarray(data_values)

    if data_values.shape[-2] < 2:
        raise ValueError("Ear axis must contain at least two channels (0=left, 1=right)")

    if mode_key == "broad-band":
        if domain_key == "ir":
            left_values = np.asarray(data_values[..., 0, :], dtype=float)
            right_values = np.asarray(data_values[..., 1, :], dtype=float)
            left_rms = np.sqrt(np.mean(np.square(left_values), axis=-1))
            right_rms = np.sqrt(np.mean(np.square(right_values), axis=-1))
        else:
            left_values = np.abs(np.asarray(data_values[..., 0, :]))
            right_values = np.abs(np.asarray(data_values[..., 1, :]))
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
    """General Description:
    Return the magnitude of transfer-function values.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.

    Returns:
    - Magnitude array computed as `abs(tf)` with the same shape as the input TF.

    Use Cases:
    - Build magnitude-response curves from complex HRTF values.
    - Prepare TF data for dB conversion or ratio analysis.

    Examples:
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
    """General Description:
    Convert linear magnitude values into decibels using a reference value.

    Parameters:
    - magnitude: Non-negative magnitude values.
    - reference: Positive reference magnitude used in the conversion
      `20 * log10(magnitude / reference)`.

    Returns:
    - Magnitude values in dB with the same shape as the input array.

    Use Cases:
    - Plot frequency responses in logarithmic amplitude.
    - Express magnitude relative to a fixed reference or a dataset-specific peak.

    Examples:
    >>> magnitude_to_db(np.array([1.0, 2.0]))
    array([0.        , 6.02059991])
    >>> magnitude_to_db(np.array([1.0, 2.0]), reference=2.0)
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
    return 20.0 * np.log10(magnitude_values / reference_value)


def db_to_magnitude(
    magnitude_db: np.ndarray,
    reference: float | str = 1.0,
) -> np.ndarray:
    """General Description:
    Convert decibel magnitudes back to linear magnitude values.

    Parameters:
    - magnitude_db: Magnitude values in decibels.
    - reference: Positive reference magnitude used in the inverse conversion.

    Returns:
    - Linear magnitude values with the same shape as the input array.

    Use Cases:
    - Rebuild linear magnitudes after dB-domain processing.
    - Prepare spectra for complex TF reconstruction.

    Examples:
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
    """General Description:
    Return transfer-function magnitudes directly in decibels.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - reference: Positive reference magnitude used in the dB conversion.

    Returns:
    - Magnitude values in dB with the same shape as the TF input.

    Use Cases:
    - Inspect HRTF magnitude responses directly in dB.
    - Build relative-magnitude plots by using a custom reference.

    Examples:
    >>> tf = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    >>> get_magnitude_db(tf)
    array([0.        , 6.02059991])
    >>> get_magnitude_db(tf, reference=2.0)
    array([-6.02059991,  0.        ])
    """
    magnitude = get_magnitude(tf)
    return magnitude_to_db(magnitude, reference=reference)


def get_phase(tf: np.ndarray | "TF", unit: str = "degrees") -> np.ndarray:
    """General Description:
    Return the phase of transfer-function values in degrees or radians.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - unit: Output unit. Supported values are `degrees`, `degree`, `deg`,
      `radians`, `radian`, and `rad`.

    Returns:
    - Phase values in the requested unit with the same shape as the input TF.

    Use Cases:
    - Plot HRTF phase responses.
    - Build phase-aware transforms or diagnostics.

    Examples:
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
    """General Description:
    Replace TF phase values while preserving the original magnitude.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - new_phase: Phase array with the same shape as the TF values.
    - unit: Phase unit used by `new_phase`. Supported values are degree and radian aliases.

    Returns:
    - Complex TF values with the original magnitude and the new phase.

    Use Cases:
    - Apply external phase estimates to measured HRTF magnitudes.
    - Build controlled phase-perturbation experiments.

    Examples:
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
    """General Description:
    Replace TF magnitude values while preserving the original phase.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - new_magnitude: Magnitude array with the same shape as the TF values.
    - scale: Scale of `new_magnitude`. Supported values are `linear`, `lineal`, and `db`.

    Returns:
    - Complex TF values with the new magnitude and the original phase.

    Use Cases:
    - Apply a smoothed target magnitude to a measured phase response.
    - Reconstruct TF values after equalization in the magnitude domain.

    Examples:
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
    """General Description:
    Return the real part of transfer-function values.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.

    Returns:
    - Real component of the TF values with the same shape as the input.

    Use Cases:
    - Serialize TF values into real/imag fields for SOFA-like storage.
    - Inspect numerical behavior of complex spectral transforms.

    Examples:
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
    """General Description:
    Return the imaginary part of transfer-function values.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.

    Returns:
    - Imaginary component of the TF values with the same shape as the input.

    Use Cases:
    - Export complex TF data into split real/imag channels.
    - Inspect numerical artifacts in spectral transforms.

    Examples:
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
    """General Description:
    Upsample an IR signal to a higher sample rate using polyphase resampling.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`.
    - new_sample_rate: Target sample rate in Hz. It must be strictly greater than
      the current sample rate.
    - sample_rate: Optional source sample rate used when `ir` is a NumPy array.

    Returns:
    - Tuple `(resampled_ir, resolved_new_sample_rate)`.

    Use Cases:
    - Increase temporal resolution for later analysis.
    - Match a high-rate rendering or convolution pipeline.

    Examples:
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
    """General Description:
    Downsample an IR signal to a lower sample rate using polyphase resampling.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`.
    - new_sample_rate: Target sample rate in Hz. It must be strictly lower than
      the current sample rate.
    - sample_rate: Optional source sample rate used when `ir` is a NumPy array.

    Returns:
    - Tuple `(resampled_ir, resolved_new_sample_rate)`.

    Use Cases:
    - Reduce storage and compute for large HRIR datasets.
    - Match external systems that require lower sample rates.

    Examples:
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


def apply_window(ir: np.ndarray | "IR", window_name: str) -> np.ndarray | None:
    """General Description:
    Apply a named time-domain window to IR samples.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`.
    - window_name: Window identifier. Supported values are `hann`, `hamming`,
      `blackman`, and `rectangular`.

    Returns:
    - Windowed IR values, or `None` when the input signal is invalid or the
      window name is unsupported.

    Use Cases:
    - Reduce spectral leakage before FFT conversion.
    - Smooth IR edges for controlled truncation experiments.

    Examples:
    >>> np.round(apply_window(np.ones(4), "hann"), 4)
    array([0.  , 0.75, 0.75, 0.  ])
    >>> apply_window(np.ones(4), "rectangular")
    array([1., 1., 1., 1.])
    """

    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "values"):
            ir = ir.values
        else:
            ir = None
    if ir is None:
        return None
    length = ir.shape[-1]
    if length <= 0:
        return None
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
        warnings.warn(
            f"Unsupported window '{window_name}'; proceeding without windowing.",
            UserWarning,
        )
        return None
    return ir * window_values


def apply_ir_crop(
    ir: np.ndarray | "IR",
    start: int | None = None,
    end: int | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    sample_rate: float | None = None,
) -> np.ndarray:
    """General Description:
    Crop IR samples by sample indices or by physical time range.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`.
    - start: Start sample index, inclusive.
    - end: End sample index, exclusive.
    - start_seconds: Start time in seconds.
    - end_seconds: End time in seconds.
    - sample_rate: Optional sample rate used for second-based cropping.

    Returns:
    - Cropped IR values with the same leading dimensions as the input.

    Use Cases:
    - Isolate direct sound or reflection windows.
    - Build fixed-duration HRIR segments for analysis or plotting.

    Examples:
    >>> apply_ir_crop(np.arange(8), start=2, end=5)
    array([2, 3, 4])
    >>> apply_ir_crop(np.arange(8), start_seconds=0.0, end_seconds=0.0000625, sample_rate=48000.0)
    array([0, 1, 2])
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
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")

    using_sample_indices = start is not None or end is not None
    using_seconds = start_seconds is not None or end_seconds is not None
    if using_sample_indices and using_seconds:
        raise ValueError("Use either sample indices (start/end) or seconds (start_seconds/end_seconds)")

    start_index = start
    end_index = end
    if using_seconds:
        if resolved_sample_rate is None:
            raise ValueError("sample_rate is required when using seconds crop")
        if isinstance(resolved_sample_rate, bool):
            raise ValueError("sample_rate must be a finite, positive value.")
        try:
            resolved_sample_rate = float(resolved_sample_rate)
        except (TypeError, ValueError):
            raise ValueError("sample_rate must be a finite, positive value.") from None
        if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
            raise ValueError("sample_rate must be a finite, positive value.")
        if start_seconds is not None:
            if isinstance(start_seconds, bool):
                raise ValueError("start_seconds must be a finite, non-negative value.")
            try:
                start_seconds = float(start_seconds)
            except (TypeError, ValueError):
                raise ValueError("start_seconds must be a finite, non-negative value.") from None
            if not np.isfinite(start_seconds) or start_seconds < 0.0:
                raise ValueError("start_seconds must be a finite, non-negative value.")
            start_index = int(round(start_seconds * resolved_sample_rate))
        else:
            start_index = None
        if end_seconds is not None:
            if isinstance(end_seconds, bool):
                raise ValueError("end_seconds must be a finite, non-negative value.")
            try:
                end_seconds = float(end_seconds)
            except (TypeError, ValueError):
                raise ValueError("end_seconds must be a finite, non-negative value.") from None
            if not np.isfinite(end_seconds) or end_seconds < 0.0:
                raise ValueError("end_seconds must be a finite, non-negative value.")
            end_index = int(round(end_seconds * resolved_sample_rate))
        else:
            end_index = None
    else:
        if start is not None:
            if isinstance(start, bool) or not isinstance(start, int):
                raise ValueError("start must be an integer")
            if start < 0:
                raise ValueError("start must be non-negative")
        if end is not None:
            if isinstance(end, bool) or not isinstance(end, int):
                raise ValueError("end must be an integer")
            if end < 0:
                raise ValueError("end must be non-negative")

    if start_index is not None and end_index is not None and start_index >= end_index:
        raise ValueError("Crop end must be greater than crop start")

    return ir_values[..., slice(start_index, end_index)]


def apply_tf_crop(
    tf: np.ndarray | "TF",
    start: int | None = None,
    end: int | None = None,
    start_frequency: float | None = None,
    end_frequency: float | None = None,
    frequency_bins: np.ndarray | None = None,
) -> np.ndarray:
    """General Description:
    Apply a band-limiting crop to TF values by bin indices or by frequency limits.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - start: Start bin index, inclusive.
    - end: End bin index, exclusive.
    - start_frequency: Lower crop frequency in Hz.
    - end_frequency: Upper crop frequency in Hz.
    - frequency_bins: Optional frequency-bin vector used for NumPy TF inputs.

    Returns:
    - TF array where bins outside the selected region are set to zero.

    Use Cases:
    - Keep only selected spectral regions for analysis.
    - Apply simple band masks in frequency-domain experiments.

    Examples:
    >>> tf = np.array([1+0j, 2+0j, 3+0j, 4+0j])
    >>> apply_tf_crop(tf, start=1, end=3)
    array([0.+0.j, 2.+0.j, 3.+0.j, 0.+0.j])
    >>> bins = np.array([0.0, 1000.0, 2000.0, 3000.0])
    >>> apply_tf_crop(tf, start_frequency=1000.0, end_frequency=2000.0, frequency_bins=bins)
    array([0.+0.j, 2.+0.j, 3.+0.j, 0.+0.j])
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
        frequency_bins_array = frequency_bins
    else:
        if not hasattr(tf, "values") or not hasattr(tf, "frequency_bins"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
        if frequency_bins is not None:
            frequency_bins_array = frequency_bins
        else:
            frequency_bins_array = tf.frequency_bins

    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    if tf_values.ndim == 0:
        raise ValueError("TF data must have at least one dimension")


    using_indices = start is not None or end is not None
    using_frequencies = start_frequency is not None or end_frequency is not None
    if using_indices and using_frequencies:
        raise ValueError(
            "Use either index crop (start/end) or frequency crop (start_frequency/end_frequency)"
        )

    tf_cropped = np.array(tf_values, copy=True)

    if using_indices:
        if start is not None:
            if isinstance(start, bool) or not isinstance(start, int):
                raise ValueError("start must be an integer")
            if start < 0:
                raise ValueError("start must be non-negative")
        if end is not None:
            if isinstance(end, bool) or not isinstance(end, int):
                raise ValueError("end must be an integer")
            if end < 0:
                raise ValueError("end must be non-negative")
        if start is not None and end is not None and start >= end:
            raise ValueError("Crop end must be greater than crop start")

        mask = np.zeros(tf_values.shape[-1], dtype=bool)
        mask[slice(start, end)] = True
        tf_cropped[..., ~mask] = 0
        return tf_cropped

    if using_frequencies:
        if frequency_bins_array is None:
            raise ValueError("frequency_bins is required for frequency-domain crop")
        frequency_bins_array = np.asarray(frequency_bins_array, dtype=float)
        if frequency_bins_array.ndim != 1:
            raise ValueError("frequency_bins must be a 1D array")
        if frequency_bins_array.size != tf_values.shape[-1]:
            raise ValueError("frequency_bins must match TF length")

        if start_frequency is not None:
            if isinstance(start_frequency, bool):
                raise ValueError("start_frequency must be a finite, non-negative value.")
            try:
                start_frequency = float(start_frequency)
            except (TypeError, ValueError):
                raise ValueError("start_frequency must be a finite, non-negative value.") from None
            if not np.isfinite(start_frequency) or start_frequency < 0.0:
                raise ValueError("start_frequency must be a finite, non-negative value.")
        else:
            start_frequency = 0.0

        if end_frequency is not None:
            if isinstance(end_frequency, bool):
                raise ValueError("end_frequency must be a finite, non-negative value.")
            try:
                end_frequency = float(end_frequency)
            except (TypeError, ValueError):
                raise ValueError("end_frequency must be a finite, non-negative value.") from None
            if not np.isfinite(end_frequency) or end_frequency < 0.0:
                raise ValueError("end_frequency must be a finite, non-negative value.")
        else:
            end_frequency = float(np.max(np.abs(frequency_bins_array)))

        if start_frequency >= end_frequency:
            raise ValueError("Crop end frequency must be greater than crop start frequency")

        frequency_magnitude = np.abs(frequency_bins_array)
        mask = (frequency_magnitude >= start_frequency) & (frequency_magnitude <= end_frequency)
        tf_cropped[..., ~mask] = 0
        return tf_cropped

    return tf_cropped


def apply_padding(
    data: np.ndarray | "IR" | "TF",
    padding_length: int,
    location: str = "end",
    value: float | complex = 0,
) -> np.ndarray:
    """General Description:
    Pad signal values at the start or end along the last axis.

    Parameters:
    - data: Signal container (`np.ndarray`, `IR`, or `TF`) with `.values`.
    - padding_length: Number of samples or bins to add.
    - location: Padding side, either `start` or `end`.
    - value: Constant value used in the padded region.

    Returns:
    - Padded signal array.

    Use Cases:
    - Extend IR length before FFT analysis.
    - Extend TF vectors for controlled frequency-domain experiments.

    Examples:
    >>> apply_padding(np.array([1.0, 2.0]), padding_length=2, location="end")
    array([1., 2., 0., 0.])
    >>> apply_padding(np.array([1.0, 2.0]), padding_length=2, location="start", value=-1.0)
    array([-1., -1.,  1.,  2.])
    """

    if isinstance(data, np.ndarray):
        signal_values = data
    elif hasattr(data, "values"):
        signal_values = data.values
    else:
        signal_values = None
    if signal_values is None:
        raise ValueError("Signal data is not available")
    if not isinstance(signal_values, np.ndarray):
        raise ValueError("Signal data must be a NumPy array")
    if signal_values.size == 0:
        raise ValueError("Signal data must be non-empty")
    if isinstance(padding_length, bool) or not isinstance(padding_length, int):
        raise ValueError("Padding must be an integer")
    if padding_length < 0:
        raise ValueError("Padding must be non-negative")
    if padding_length == 0:
        return signal_values
    location_key = location.strip().lower()
    if location_key == "start":
        before, after = padding_length, 0
    elif location_key == "end":
        before, after = 0, padding_length
    else:
        raise ValueError("Padding location must be 'start' or 'end'")
    pad_width = [(0, 0)] * (signal_values.ndim - 1) + [(before, after)]
    return np.pad(
        signal_values,
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
    """General Description:
    Apply an FIR filter to IR data.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`.
    - filter: Filter family. Supported values are low-pass, high-pass, and band-pass aliases.
    - sample_rate: Sample rate in Hz.
    - cutoff: Cutoff value. Use a scalar for low/high-pass and a tuple for band-pass.
    - num_taps: Odd FIR length.
    - window: FIR design window, one of `hann`, `hamming`, `blackman`, or `rectangular`.

    Returns:
    - Filtered IR values with the same shape as the input.

    Use Cases:
    - Remove undesired frequency regions from HRIRs.
    - Precondition responses before feature extraction.

    Examples:
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
    """General Description:
    Apply an IIR Butterworth filter to IR data.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`.
    - filter: Filter family. Supported values are low-pass, high-pass, and band-pass aliases.
    - sample_rate: Sample rate in Hz.
    - cutoff: Cutoff value. Use a scalar for low/high-pass and a tuple for band-pass.
    - order: Positive Butterworth filter order.

    Returns:
    - Filtered IR values with the same shape as the input.

    Use Cases:
    - Reproduce IIR preprocessing chains for ITD estimation.
    - Apply lightweight recursive filtering before feature extraction.

    Examples:
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
    """General Description:
    Convert IR data into a minimum-phase IR.

    Parameters:
    - data: `np.ndarray` or `IR` object containing real-valued IR samples.
    - method: Minimum-phase strategy. `homomorphic` and `real_cepstrum` use a
      log-magnitude real cepstrum, while `cepstrum` uses a complex cepstrum with
      unwrapped phase.
    - fft_length: Optional FFT length used for cepstral operations.
    - epsilon: Positive floor applied to magnitude values before logarithms.

    Returns:
    - Minimum-phase IR array with the same trailing length as the resolved IR input.

    Use Cases:
    - Create minimum-phase HRIR approximations for low-latency processing.
    - Standardize phase behavior before comparisons or model fitting.

    Examples:
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
    """General Description:
    Compute transfer-function values from IR values using an FFT.

    Parameters:
    - ir: IR array or `IR` object.
    - sample_rate: Sample rate in Hz for NumPy input. Optional for `IR` input.
    - fft_length: Optional FFT size. If omitted, the IR length is used.
    - window_name: Optional time-domain window applied before the FFT.

    Returns:
    - For NumPy input: `(tf_values, frequency_bins, fft_length_used)`.
    - For `IR` input: the updated `TF` object linked to the same `HRTF`.

    Use Cases:
    - Build TF representations after IR editing.
    - Control frequency resolution via explicit FFT lengths.

    Examples:
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
        windowed = apply_window(ir_values, window_name)
        if windowed is not None:
            ir_used = windowed

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
    """General Description:
    Compute IR values from TF values using inverse FFT routines.

    Parameters:
    - tf: TF array or `TF` object.
    - frequency_bins: Optional frequency-bin vector matching the TF length.
    - sample_rate: Optional sample rate used when inferring bins for NumPy TF input.
    - spectrum_type: Required when inferring bins. Supported values are `positive`
      for one-sided spectra and `complete` for full complex spectra.

    Returns:
    - For NumPy input: `(ir_values, sample_rate, fft_length_used)`.
    - For `TF` input: the updated `IR` object linked to the same `HRTF`.

    Use Cases:
    - Reconstruct HRIRs after TF-domain edits.
    - Convert loaded HRTF datasets back into time-domain form.

    Examples:
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
