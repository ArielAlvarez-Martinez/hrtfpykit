from __future__ import annotations

from typing import TYPE_CHECKING

import warnings
import numpy as np

if TYPE_CHECKING:
    from .domain import FrequencyDomain, TimeDomain


def apply_normalization(
    data: np.ndarray | "TimeDomain" | "FrequencyDomain",
    value: float,
) -> np.ndarray | None:
    if isinstance(data, np.ndarray):
        signal = data
    elif hasattr(data, "ir"):
        signal = data.ir
    elif hasattr(data, "tf"):
        signal = data.tf
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
    data: np.ndarray | "TimeDomain" | "FrequencyDomain",
    value: float,
) -> np.ndarray | None:
    if isinstance(data, np.ndarray):
        signal = data
    elif hasattr(data, "ir"):
        signal = data.ir
    elif hasattr(data, "tf"):
        signal = data.tf
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


def window(ir: np.ndarray | "TimeDomain", window_name: str) -> np.ndarray | None:
    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "ir"):
            ir = ir.ir
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


def compute_tf_from_ir(
    ir: np.ndarray | "TimeDomain",
    sample_rate: float,
    fft_length: int | None = None,
    window_name: str | None = None,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "ir"):
            ir = ir.ir
    if ir is None:
        raise ValueError("IR data is not available")
    n_fft = fft_length if fft_length is not None else ir.shape[-1]

    signal = ir
    if window_name:
        windowed = window(ir, window_name)
        if windowed is not None:
            signal = windowed

    tf = np.fft.rfft(signal, n=n_fft, axis=-1)
    if normalize and n_fft:
        tf = tf / float(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return tf, freqs, int(n_fft)


def compute_ir_from_tf(
    tf: np.ndarray | "FrequencyDomain",
    frequency_bins: np.ndarray | None = None,
    fft_length: int | None = None,
    tf_normalization: float | None = None,
    normalization_action: str = "undo",
) -> np.ndarray | None:
    """Convert a transfer function to an impulse response.

    Parameters
    ----------
    tf : numpy.ndarray or FrequencyDomain
        Complex transfer function values. If a FrequencyDomain is provided,
        its ``tf`` (and optionally ``frequency_bins``) are used.
    frequency_bins : numpy.ndarray, optional
        Frequency axis in Hz. If omitted, a one-sided positive spectrum is assumed.
    fft_length : int, optional
        FFT length for IFFT/IRFFT and for inferring frequency bins.
    tf_normalization : float, optional
        Normalization factor to apply or undo on the transfer function.
    normalization_action : {"apply", "undo"}, optional
        Controls how the transfer function is scaled before IFFT:
        "apply" divides the TF by ``tf_normalization``,
        "undo" multiplies the TF by ``tf_normalization``.

    Returns
    -------
    numpy.ndarray or None
        Time-domain impulse response. Returns None when conversion fails.

    Examples
    --------
    ir = compute_ir_from_tf(tf, frequency_bins=freqs, fft_length=512)
    ir = compute_ir_from_tf(freq_domain, tf_normalization=2.0, normalization_action="undo")

    Best Practices
    --------------
    - Provide ``frequency_bins`` when available to avoid ambiguity.
    - Keep ``fft_length`` consistent with the TF length.

    Warnings / Errors
    -----------------
    - Raises ValueError if ``normalization_action`` is invalid.
    - Returns None when TF length is too short or FFT length is invalid.
    """
    tf_array = tf
    if not isinstance(tf, np.ndarray):
        if hasattr(tf, "tf"):
            tf_array = tf.tf
            if frequency_bins is None and hasattr(tf, "frequency_bins"):
                frequency_bins = tf.frequency_bins
    if tf_array is None:
        warnings.warn("TF data is not available; cannot compute IR.", UserWarning)
        return None
    n_fft = fft_length

    action = normalization_action.strip().lower()
    if action not in {"apply", "undo"}:
        raise ValueError("normalization_action must be 'apply' or 'undo'")

    tf_used = tf_array
    if tf_normalization is not None:
        try:
            norm_value = float(tf_normalization)
        except (TypeError, ValueError):
            norm_value = None
        if norm_value is not None and not np.isclose(norm_value, 0.0):
            if action == "apply":
                tf_used = tf_array / norm_value
            else:
                tf_used = tf_array * norm_value

    if tf_used.shape[-1] < 2:
        warnings.warn("TF length is too short to compute IR.", UserWarning)
        return None

    freqs = frequency_bins

    if freqs is not None and freqs.ndim == 1 and freqs.size == tf_used.shape[-1]:
        step = None
        if freqs.size >= 2:
            diffs = np.diff(freqs)
            first = float(diffs[0])
            if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
                step = first
        if step is not None:
            if float(np.min(freqs)) < 0.0:
                n_fft_used = n_fft or freqs.size
                ir = np.fft.ifft(tf_used, n=n_fft_used, axis=-1)
                ir = np.real_if_close(ir, tol=1000)
                return ir
            n_fft_used = n_fft or (2 * (freqs.size - 1))
            ir = np.fft.irfft(tf_used, n=n_fft_used, axis=-1)
            return ir

    n_fft_used = n_fft or (2 * (tf_used.shape[-1] - 1))
    if n_fft_used <= 0:
        warnings.warn("FFT length is invalid; cannot compute IR.", UserWarning)
        return None
    ir = np.fft.irfft(tf_used, n=n_fft_used, axis=-1)
    return ir
