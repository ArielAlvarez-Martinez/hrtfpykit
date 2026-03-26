from __future__ import annotations

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


def apply_crop(
    ir: np.ndarray | "IR",
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:

    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "values"):
            ir = ir.values
        else:
            ir = None
    if ir is None:
        raise ValueError("IR data is not available")
    return ir[..., slice(start, end)]


def apply_padding(
    ir: np.ndarray | "IR",
    padding_length: int,
    location: str = "end",
    value: int = 0,
) -> np.ndarray:

    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "values"):
            ir = ir.values
        else:
            ir = None
    if ir is None:
        raise ValueError("IR data is not available")
    if isinstance(padding_length, bool) or not isinstance(padding_length, int):
        raise ValueError("Padding must be an integer")
    if padding_length < 0:
        raise ValueError("Padding must be non-negative")
    if padding_length == 0:
        return ir
    location_key = location.strip().lower()
    if location_key == "start":
        before, after = padding_length, 0
    elif location_key == "end":
        before, after = 0, padding_length
    else:
        raise ValueError("Padding location must be 'start' or 'end'")
    pad_width = [(0, 0)] * (ir.ndim - 1) + [(before, after)]
    return np.pad(
        ir,
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
    ir: np.ndarray,
    sample_rate: float,
    fft_length: int | None = None,
    window_name: str | None = None,
    ir_normalization: float | None = None,
    normalization_action: str = "apply",
) -> tuple[np.ndarray, np.ndarray, int]:
   
    if ir is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if sample_rate is None:
        raise ValueError("sample_rate is required when ir is a NumPy array")
    n_fft = fft_length if fft_length is not None else ir.shape[-1]

    signal = ir
    if window_name:
        windowed = apply_window(ir, window_name)
        if windowed is not None:
            signal = windowed
    if ir_normalization is not None:
        try:
            norm_value = float(ir_normalization)
        except (TypeError, ValueError):
            norm_value = None
        if norm_value is not None and not np.isclose(norm_value, 0.0):
            action = normalization_action.strip().lower()
            if action not in {"apply", "undo"}:
                raise ValueError("normalization_action must be 'apply' or 'undo'")
            if action == "apply":
                signal = signal / norm_value
            else:
                signal = signal * norm_value

    tf = np.fft.rfft(signal, n=n_fft, axis=-1)
    frequency_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    fft_length = int(n_fft)
    return tf, frequency_bins, fft_length


def calculate_ir_from_tf(
    tf: np.ndarray,
    frequency_bins: np.ndarray,
    fft_length: int | None = None,
    tf_normalization: float | None = None,
    normalization_action: str = "undo",
) -> tuple[np.ndarray | None, float | None]:
    
    if tf is None:
        warnings.warn("TF data is not available; cannot compute IR.", UserWarning)
        return None, None
    if not isinstance(tf, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    if frequency_bins is None:
        raise ValueError("frequency_bins is required to compute IR")
    action = normalization_action.strip().lower()
    if action not in {"apply", "undo"}:
        raise ValueError("normalization_action must be 'apply' or 'undo'")

    tf_used = tf
    if tf_normalization is not None:
        try:
            norm_value = float(tf_normalization)
        except (TypeError, ValueError):
            norm_value = None
        if norm_value is not None and not np.isclose(norm_value, 0.0):
            if action == "apply":
                tf_used = tf / norm_value
            else:
                tf_used = tf * norm_value

    if tf_used.shape[-1] < 2:
        warnings.warn("TF length is too short to compute IR.", UserWarning)
        return None, None

    if frequency_bins.ndim != 1 or frequency_bins.size != tf_used.shape[-1]:
        raise ValueError("frequency_bins must be 1D and match TF length")
    if frequency_bins.size < 2:
        raise ValueError("frequency_bins must contain at least two points")
    diffs = np.diff(frequency_bins)
    step = float(diffs[0])
    if step <= 0.0 or not np.allclose(diffs, step, rtol=1e-5, atol=1e-8):
        raise ValueError("frequency_bins must be uniformly spaced and increasing")
    if float(np.min(frequency_bins)) < 0.0:
        expected_n_fft = frequency_bins.size
        if fft_length is not None and fft_length != expected_n_fft:
            warnings.warn(
                "FFT length does not match the provided frequency bins; using inferred length.",
                UserWarning,
            )
        fft_length_used = expected_n_fft
        ir = np.fft.ifft(tf_used, n=fft_length_used, axis=-1)
        ir = np.real_if_close(ir, tol=1000)
        sample_rate = step * expected_n_fft
        return ir, sample_rate
    if not np.isclose(frequency_bins[0], 0.0):
        raise ValueError("frequency_bins must start at 0 Hz for one-sided spectra")
    expected_n_fft = 2 * (frequency_bins.size - 1)
    if fft_length is not None and fft_length != expected_n_fft:
        warnings.warn(
            "FFT length does not match the provided frequency bins; using inferred length.",
            UserWarning,
        )
    fft_length_used = expected_n_fft
    ir = np.fft.irfft(tf_used, n=fft_length_used, axis=-1)
    sample_rate = step * expected_n_fft
    return ir, sample_rate
