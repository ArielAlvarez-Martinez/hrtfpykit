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


def signal_duration(
    signal: np.ndarray | "IR",
    sample_rate: float | None = None,
) -> float:
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


def get_magnitude(tf: np.ndarray | "TF") -> np.ndarray:
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


def magnitude_to_db(magnitude: np.ndarray) -> np.ndarray:
    magnitude_values = np.asarray(magnitude, dtype=float)
    if np.any(magnitude_values < 0.0):
        raise ValueError("magnitude values must be non-negative")
    return 20.0 * np.log10(magnitude_values)


def db_to_magnitude(magnitude_db: np.ndarray) -> np.ndarray:
    magnitude_db_values = np.asarray(magnitude_db, dtype=float)
    return 10.0 ** (magnitude_db_values / 20.0)


def get_magnitude_db(tf: np.ndarray | "TF") -> np.ndarray:
    magnitude = get_magnitude(tf)
    return magnitude_to_db(magnitude)


def get_phase(tf: np.ndarray | "TF", unit: str = "degrees") -> np.ndarray:
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


def get_real(tf: np.ndarray | "TF") -> np.ndarray:
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


def apply_crop(
    ir: np.ndarray | "IR",
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:
    return apply_ir_crop(ir, start=start, end=end)


def apply_padding(
    data: np.ndarray | "IR" | "TF",
    padding_length: int,
    location: str = "end",
    value: float | complex = 0,
) -> np.ndarray:

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


def apply_filter(
    ir: np.ndarray | "IR",
    filter: str,
    sample_rate: float | None = None,
    cutoff: float | tuple[float, float] | None = None,
    num_taps: int = 101,
    window: str | None = None,
) -> np.ndarray:
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

def calculate_tf_from_ir(
    ir: np.ndarray | "IR",
    sample_rate: float | None = None,
    fft_length: int | None = None,
    window_name: str | None = None,
    ir_normalization: float | None = None,
    normalization_action: str = "apply",
) -> tuple[np.ndarray, np.ndarray, int] | "TF":
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
