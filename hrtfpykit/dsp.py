from __future__ import annotations

import warnings
import numpy as np


def window(ir: np.ndarray, name: str) -> np.ndarray | None:
    length = ir.shape[-1]
    if length <= 0:
        return None
    key = name.strip().lower()
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
            f"Unsupported window '{name}'; proceeding without windowing.",
            UserWarning,
        )
        return None
    return ir * window_values


def compute_tf_from_ir(
    ir: np.ndarray,
    sample_rate: float,
    fft_length: int | None = None,
    window_name: str | None = None,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, int | None]:
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
    tf: np.ndarray,
    freqs: np.ndarray | None,
    fft_length: int | None = None,
    normalization: float | None = None,
) -> tuple[np.ndarray | None, float | None, int | None]:
    n_fft = fft_length

    scale = None
    if normalization is not None:
        try:
            norm_value = float(normalization)
        except (TypeError, ValueError):
            norm_value = None
        if norm_value is not None and not np.isclose(norm_value, 0.0):
            scale = 1.0 / norm_value
    if scale is None:
        scale = 1.0

    if tf.shape[-1] < 2:
        return None, None, None

    if freqs is not None and freqs.ndim == 1 and freqs.size == tf.shape[-1]:
        step = None
        if freqs.size >= 2:
            diffs = np.diff(freqs)
            first = float(diffs[0])
            if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
                step = first
            if step is not None:
                if float(np.min(freqs)) < 0.0:
                    n_fft_used = n_fft or freqs.size
                    samplerate = step * n_fft_used
                    tf_used = tf * scale
                    ir = np.fft.ifft(tf_used, n=n_fft_used, axis=-1)
                    ir = np.real_if_close(ir, tol=1000)
                    return ir, float(samplerate), int(n_fft_used)
                n_fft_used = n_fft or (2 * (freqs.size - 1))
                samplerate = step * n_fft_used
                tf_used = tf * scale
                ir = np.fft.irfft(tf_used, n=n_fft_used, axis=-1)
                return ir, float(samplerate), int(n_fft_used)

    n_fft_used = n_fft or (2 * (tf.shape[-1] - 1))
    if n_fft_used <= 0:
        return None, None, None
    tf_used = tf * scale
    ir = np.fft.irfft(tf_used, n=n_fft_used, axis=-1)
    return ir, None, int(n_fft_used)
