from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import warnings
import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from .domain import IR, TF


def apply_normalization(
    data: np.ndarray | "IR" | "TF",
    value: float,
) -> np.ndarray | None:
    """General Description:
    Apply amplitude normalization to signal values.

    Parameters:
    - data: Signal container (`np.ndarray`, `IR`, or `TF`) holding `.values`.
    - value: Non-zero normalization factor.

    Returns:
    - Normalized signal values, or `None` when input/normalization is invalid.

    Use Cases:
    - Standardize levels before comparisons or transforms.
    - Prepare signals for consistent plotting.

    Best Practices:
    - Use finite non-zero normalization factors.
    - Handle `None` outputs when input validation fails.
    """

    if isinstance(data, np.ndarray):
        signal = data
    elif hasattr(data, "values"):
        signal = data.values
    else:
        signal = None
    if signal is None:
        warnings.warn("Signal data is not available; cannot apply normalization.", UserWarning)
        return None
    try:
        norm_value = float(value)
    except (TypeError, ValueError):
        warnings.warn("Normalization value is invalid; cannot apply normalization.", UserWarning)
        return None
    if np.isclose(norm_value, 0.0):
        warnings.warn("Normalization value is zero; cannot apply normalization.", UserWarning)
        return None
    return signal / norm_value


def undo_normalization(
    data: np.ndarray | "IR" | "TF",
    value: float,
) -> np.ndarray | None:
    """General Description:
    Undo a previously applied amplitude normalization.

    Parameters:
    - data: Signal container (`np.ndarray`, `IR`, or `TF`) holding `.values`.
    - value: Non-zero normalization factor used previously.

    Returns:
    - Denormalized signal values, or `None` when input/normalization is invalid.

    Use Cases:
    - Restore physical amplitude after normalized-domain processing.
    - Reconstruct original scale for export.

    Best Practices:
    - Keep normalization factors tracked in metadata.
    - Use finite non-zero factors to avoid invalid scaling.
    """

    if isinstance(data, np.ndarray):
        signal = data
    elif hasattr(data, "values"):
        signal = data.values
    else:
        signal = None
    if signal is None:
        warnings.warn("Signal data is not available; cannot undo normalization.", UserWarning)
        return None
    try:
        norm_value = float(value)
    except (TypeError, ValueError):
        warnings.warn("Normalization value is invalid; cannot undo normalization.", UserWarning)
        return None
    if np.isclose(norm_value, 0.0):
        warnings.warn("Normalization value is zero; cannot undo normalization.", UserWarning)
        return None
    return signal * norm_value


def get_signal_duration(
    signal: np.ndarray | "IR",
    sample_rate: float | None = None,
) -> float:
    """General Description:
    Compute signal duration in seconds from sample count and sample rate.

    Parameters:
    - signal: Time-domain array or `IR` object with `.values`.
    - sample_rate: Optional sample rate in Hz. If omitted for `IR`, uses `IR.sample_rate`.

    Returns:
    - Duration in seconds.

    Use Cases:
    - Report IR window length in absolute time.
    - Build time-based crop or analysis settings.

    Best Practices:
    - Provide finite positive sample rates.
    - Ensure signal values are initialized before querying duration.
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
    Estimate interaural time differences (ITD) from binaural IR signals.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`.
      Expected layout is `[..., ear, samples]` with ear convention `0=left`, `1=right`.
    - method: ITD estimator (`threshold` or `maxiacce`).
    - sample_rate: Optional sample rate in Hz for NumPy input. For `IR`, defaults to `IR.sample_rate`.
    - output: ITD unit (`seconds` or `samples`).
    - thresh_level: Threshold offset in dB for `threshold` mode.
    - upper_cut_freq: Low-pass cutoff in Hz applied before ITD estimation.
    - filter_order: Positive IIR Butterworth order for low-pass preprocessing.

    Returns:
    - Array of ITD values in selected `output` units. Positive means left-ear delay relative to right-ear.

    Use Cases:
    - Extract ITD features from HRIR datasets.
    - Compare binaural timing cues across positions.

    Best Practices:
    - Keep input channels in HRTF convention (`0=left`, `1=right`).
    - Use `threshold` for onset-based ITD and `maxiacce` for envelope-correlation ITD.
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
    Compute interaural level differences (ILD) from binaural IR or TF data.

    Parameters:
    - data: Binaural signal container. Accepts:
      - `np.ndarray` with layout `[..., ear, samples_or_bins]`
      - `IR` object with `.values`
      - `TF` object with `.values`
      Ear convention is `0=left`, `1=right`.
    - domain: Input domain (`auto`, `ir`, `tf`). In `auto` mode:
      - `IR` objects are treated as `ir`
      - `TF` objects are treated as `tf`
      - NumPy arrays are treated as `tf` when complex, otherwise `ir`
    - sample_rate: Sample rate in Hz used only when `domain='ir'` and `data` is a NumPy array.
    - fft_length: Optional FFT length used only for IR-domain inputs.
    - mode: ILD computation mode (`broad-band` or `frequency-dependent`).
    - output: ILD output representation (`db` or `linear`).
    - epsilon: Positive floor used to avoid division by zero in magnitude ratios.

    Returns:
    - ILD array in selected mode:
      - `mode='broad-band'`: shape `[...]` (one ILD value per position).
      - `mode='frequency-dependent'`: shape `[..., frequency_bins]`.
      For `output='db'`: `20*log10(ratio)`. For `output='linear'`: `ratio`.

    Use Cases:
    - Extract broadband or frequency-dependent binaural level cues from HRIR/HRTF data.
    - Compare left-right spectral imbalance across source positions.

    Best Practices:
    - Keep ear ordering in HRTF convention (`0=left`, `1=right`).
    - Prefer `output='db'` for analysis and reporting.
    - Use IR-domain inputs when you need ILD derived from current time-domain edits.
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
    Return the magnitude spectrum of transfer-function values.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.

    Returns:
    - Magnitude values (`abs(tf)`).

    Use Cases:
    - Build spectral envelopes.
    - Prepare data for dB conversion.

    Best Practices:
    - Ensure TF values are initialized and numeric.
    - Use this helper instead of repeated `np.abs` calls across codepaths.
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


def magnitude_to_db(magnitude: np.ndarray, reference: float = 1.0) -> np.ndarray:
    """General Description:
    Convert linear magnitude values to decibels.

    Parameters:
    - magnitude: Non-negative magnitude values.

    Returns:
    - Magnitude values in dB (`20*log10(magnitude)`).

    Use Cases:
    - Plot frequency responses in logarithmic amplitude.
    - Compare attenuation/boost using dB scales.

    Best Practices:
    - Clamp or validate data upstream when zeros are expected.
    - Keep units explicit when mixing linear and dB values.
    """
    magnitude_values = np.asarray(magnitude, dtype=float)
    if np.any(magnitude_values < 0.0):
        raise ValueError("magnitude values must be non-negative")
    reference_value = float(reference)
    if not np.isfinite(reference_value) or reference_value <= 0.0:
        raise ValueError("reference must be a finite, positive float")
    return 20.0 * np.log10(magnitude_values / reference_value)


def db_to_magnitude(magnitude_db: np.ndarray, reference: float = 1.0) -> np.ndarray:
    """General Description:
    Convert decibel magnitudes to linear scale.

    Parameters:
    - magnitude_db: Magnitude values in decibels.

    Returns:
    - Linear magnitude values.

    Use Cases:
    - Rebuild linear magnitudes after dB-domain processing.
    - Prepare spectra for complex reconstruction.

    Best Practices:
    - Keep dB conventions consistent (`20*log10` for magnitude).
    - Prefer float arrays for numerical stability.
    """
    magnitude_db_values = np.asarray(magnitude_db, dtype=float)
    reference_value = float(reference)
    if not np.isfinite(reference_value) or reference_value <= 0.0:
        raise ValueError("reference must be a finite, positive float")
    return reference_value * (10.0 ** (magnitude_db_values / 20.0))


def get_magnitude_db(tf: np.ndarray | "TF", reference: float = 1.0) -> np.ndarray:
    """General Description:
    Return TF magnitudes directly in decibels.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.

    Returns:
    - Magnitude values in dB.

    Use Cases:
    - Fast response-curve inspection.
    - Thresholding/analysis in perceptual amplitude scale.

    Best Practices:
    - Validate TF initialization before calling.
    - Use together with consistent dB reference assumptions.
    """
    magnitude = get_magnitude(tf)
    return magnitude_to_db(magnitude, reference=reference)


def get_phase(tf: np.ndarray | "TF", unit: str = "degrees") -> np.ndarray:
    """General Description:
    Return the phase of TF values in degrees or radians.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - unit: Output unit (`degrees` or `radians`, aliases supported).

    Returns:
    - Phase values in requested unit.

    Use Cases:
    - Phase response visualization.
    - Building phase-aware transforms.

    Best Practices:
    - Keep angle units explicit in downstream code.
    - Use radians for numerical transforms and degrees for reporting.
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
    Replace the phase of TF values while preserving magnitude.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - new_phase: Phase array with same shape as TF values.
    - unit: Phase unit for `new_phase` (`degrees` or `radians`, aliases supported).

    Returns:
    - Complex TF values with original magnitude and replaced phase.

    Use Cases:
    - Apply externally estimated phase while keeping measured magnitude.
    - Build controlled phase perturbation experiments.

    Best Practices:
    - Keep `new_phase` shape identical to TF values for deterministic behavior.
    - Use radians in pipelines that already operate in angular frequency math.
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
    Replace the magnitude of TF values while preserving phase.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - new_magnitude: Magnitude array with same shape as TF values.
    - scale: Magnitude scale (`linear`, `lineal`, or `db`).

    Returns:
    - Complex TF values with replaced magnitude and original phase.

    Use Cases:
    - Apply smoothed target magnitudes to measured phase.
    - Reconstruct TFs after magnitude-domain equalization.

    Best Practices:
    - Use non-negative linear magnitudes.
    - Use explicit `db` scale when values are in decibels.
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
    Return the real part of TF values.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.

    Returns:
    - Real component of TF values.

    Use Cases:
    - Serialize TF into real/imag convention fields.
    - Debug spectral symmetry issues.

    Best Practices:
    - Ensure TF arrays are initialized before access.
    - Pair with `get_imag` when reconstructing complex spectra.
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
    Return the imaginary part of TF values.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.

    Returns:
    - Imaginary component of TF values.

    Use Cases:
    - Export complex TF data to split real/imag channels.
    - Debug numerical artifacts in spectral transforms.

    Best Practices:
    - Keep complex dtype through spectral pipelines.
    - Pair with `get_real` for consistent roundtrips.
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
    - new_sample_rate: Target sample rate in Hz, strictly greater than current rate.
    - sample_rate: Optional source sample rate when `ir` is a NumPy array.

    Returns:
    - Tuple `(resampled_ir, resolved_new_sample_rate)`.

    Use Cases:
    - Increase temporal resolution for later analysis.
    - Match a high-rate processing/rendering pipeline.

    Best Practices:
    - Ensure source and target sample rates are finite and positive.
    - Recompute TF after resampling to keep domains synchronized.
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
    - new_sample_rate: Target sample rate in Hz, strictly lower than current rate.
    - sample_rate: Optional source sample rate when `ir` is a NumPy array.

    Returns:
    - Tuple `(resampled_ir, resolved_new_sample_rate)`.

    Use Cases:
    - Reduce storage and compute for large datasets.
    - Match external systems requiring lower sample rates.

    Best Practices:
    - Ensure target rate preserves required bandwidth.
    - Recompute TF after resampling to maintain consistency.
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
    - window_name: Window identifier (`hann`, `hamming`, `blackman`, `rectangular`).

    Returns:
    - Windowed IR values, or `None` when input/window is invalid.

    Use Cases:
    - Reduce leakage before FFT conversion.
    - Smooth IR edges for controlled truncation.

    Best Practices:
    - Use supported window names and handle `None` outputs.
    - Keep window selection explicit for reproducibility.
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
    Crop IR samples by index range or by time range.

    Parameters:
    - ir: Time-domain array or `IR` object with `.values`.
    - start: Start sample index (inclusive).
    - end: End sample index (exclusive).
    - start_seconds: Start time in seconds.
    - end_seconds: End time in seconds.
    - sample_rate: Optional sample rate used for second-based crop.

    Returns:
    - Cropped IR values.

    Use Cases:
    - Isolate direct sound or specific reflection windows.
    - Build fixed-duration IR segments.

    Best Practices:
    - Use either index-based or second-based bounds in one call.
    - Provide finite positive sample rates for time-based crop.
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
    Apply a band-limiting crop to TF values by indices or frequencies.

    Parameters:
    - tf: Frequency-domain array or `TF` object with `.values`.
    - start: Start bin index (inclusive).
    - end: End bin index (exclusive).
    - start_frequency: Lower crop frequency in Hz.
    - end_frequency: Upper crop frequency in Hz.
    - frequency_bins: Optional frequency-bin vector for NumPy TF inputs.

    Returns:
    - TF values with bins outside selected region set to zero.

    Use Cases:
    - Keep only selected spectral regions.
    - Apply brickwall-style masks in controlled experiments.

    Best Practices:
    - Use either index or frequency mode, not both.
    - Pass consistent frequency bins matching TF length.
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
    - padding_length: Number of samples/bins to add.
    - location: Padding side (`start` or `end`).
    - value: Constant padding value.

    Returns:
    - Padded signal array.

    Use Cases:
    - Extend IR length before FFT analysis.
    - Extend TF vectors for spectral-domain experimentation.

    Best Practices:
    - Keep padding lengths explicit and non-negative.
    - Maintain matching metadata when padding TF bins externally.
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
    - filter: Filter family (`lowpass`, `highpass`, `bandpass`, aliases supported).
    - sample_rate: Sample rate in Hz.
    - cutoff: Cutoff value (scalar for low/high, tuple for bandpass).
    - num_taps: Odd FIR length.
    - window: FIR design window (`hann`, `hamming`, `blackman`, `rectangular`).

    Returns:
    - Filtered IR values.

    Use Cases:
    - Remove undesired frequency regions from HRIRs.
    - Precondition responses before feature extraction.

    Best Practices:
    - Keep `num_taps` odd for linear-phase FIR behavior.
    - Ensure cutoffs remain strictly inside valid Nyquist bounds.
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
    - filter: Filter family (`lowpass`, `highpass`, `bandpass`, aliases supported).
    - sample_rate: Sample rate in Hz.
    - cutoff: Cutoff value (scalar for low/high, tuple for bandpass).
    - order: Positive Butterworth filter order.

    Returns:
    - Filtered IR values.

    Use Cases:
    - Reproduce legacy IIR-based preprocessing chains for ITD estimation.
    - Apply lightweight recursive filtering with low computational cost.

    Best Practices:
    - Keep `order` moderate to avoid numerical instability on extreme settings.
    - Ensure cutoffs remain strictly inside valid Nyquist bounds.
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
    Convert IR data to a minimum-phase IR.

    Parameters:
    - data: `np.ndarray` or `IR`.
    - method: Minimum-phase strategy.
      `homomorphic`/`real_cepstrum` use log-magnitude real cepstrum,
      `cepstrum` uses complex cepstrum with unwrapped phase.
    - fft_length: Optional FFT length for cepstral operations.
    - epsilon: Positive floor for magnitude values before logarithms.

    Returns:
    - Minimum-phase IR array with the same trailing length as the resolved IR input.

    Use Cases:
    - Create minimum-phase HRIR approximations for low-latency processing.
    - Standardize phase behavior before comparisons or model fitting.

    Best Practices:
    - Provide explicit `fft_length` for reproducible cepstral behavior.
    - Keep `epsilon` small but finite to avoid invalid log magnitudes.
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
    ir_normalization: float | None = None,
    normalization_action: str = "apply",
) -> tuple[np.ndarray, np.ndarray, int] | "TF":
    """General Description:
    Compute transfer-function values from IR values using FFT.

    Parameters:
    - ir: IR array or `IR` object.
    - sample_rate: Sample rate in Hz for NumPy input, optional for `IR` object.
    - fft_length: Optional FFT size.
    - window_name: Optional time-domain window applied before FFT.
    - ir_normalization: Optional normalization factor to apply/undo before FFT.
    - normalization_action: Normalization mode (`apply` or `undo`).

    Returns:
    - For NumPy input: `(tf_values, frequency_bins, fft_length_used)`.
    - For `IR` input: updated `TF` object linked to the same `HRTF`.

    Use Cases:
    - Build TF representation after IR editing.
    - Control frequency resolution via explicit FFT lengths.

    Best Practices:
    - Keep sample rates finite and positive.
    - Use explicit `fft_length` when reproducibility across runs is required.
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

    action = normalization_action.strip().lower()
    if action not in {"apply", "undo"}:
        raise ValueError("normalization_action must be 'apply' or 'undo'")

    ir_used = ir_values
    if window_name:
        windowed = apply_window(ir_values, window_name)
        if windowed is not None:
            ir_used = windowed
    if ir_normalization is not None:
        try:
            norm_value = float(ir_normalization)
        except (TypeError, ValueError):
            norm_value = None
        if norm_value is not None and not np.isclose(norm_value, 0.0):
            if action == "apply":
                ir_used = ir_used / norm_value
            else:
                ir_used = ir_used * norm_value

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
    tf_normalization: float | None = None,
    normalization_action: str = "undo",
    sample_rate: float | None = None,
    spectrum_type: str | None = None,
) -> tuple[np.ndarray, float] | "IR":
    """General Description:
    Compute IR values from TF values using inverse FFT routines.

    Parameters:
    - tf: TF array or `TF` object.
    - frequency_bins: Optional frequency-bin vector matching TF length.
    - tf_normalization: Optional normalization factor to apply/undo before inverse FFT.
    - normalization_action: Normalization mode (`apply` or `undo`).
    - sample_rate: Optional sample rate used when inferring bins for NumPy TF.
    - spectrum_type: Required when inferring bins (`positive` or `complete`).

    Returns:
    - For NumPy input: `(ir_values, sample_rate, fft_length_used)`.
    - For `TF` input: updated `IR` object linked to the same `HRTF`.

    Use Cases:
    - Reconstruct HRIRs after TF-domain edits.
    - Convert loaded HRTF datasets to time-domain analysis form.

    Best Practices:
    - Keep `frequency_bins` uniformly spaced and increasing.
    - Provide explicit `spectrum_type` when bins are not directly available.
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

    action = normalization_action.strip().lower()
    if action not in {"apply", "undo"}:
        raise ValueError("normalization_action must be 'apply' or 'undo'")

    tf_used = tf_values
    if tf_normalization is not None:
        try:
            norm_value = float(tf_normalization)
        except (TypeError, ValueError):
            norm_value = None
        if norm_value is not None and not np.isclose(norm_value, 0.0):
            if action == "apply":
                tf_used = tf_values / norm_value
            else:
                tf_used = tf_values * norm_value

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
