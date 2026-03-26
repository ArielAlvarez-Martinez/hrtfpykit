from __future__ import annotations

from typing import TYPE_CHECKING

import warnings
import numpy as np

from .dsp import apply_crop, apply_filter, apply_padding, apply_window, calculate_tf_from_ir

if TYPE_CHECKING:
    from .hrtf import HRTF


class IR:
    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.sample_rate: float | None = None

    @property
    def ir_length(self) -> int | None:
        values = self.values
        if values is None:
            return None
        return int(values.shape[-1])

    def apply_crop(self, start: int | None = None, end: int | None = None) -> None:
        values = self.values
        if values is None:
            raise ValueError("IR data is not available")
        self.values = apply_crop(values, start=start, end=end)
        self._recompute_tf()

    def apply_window(self, window_name: str) -> None:
        values = self.values
        if values is None:
            raise ValueError("IR data is not available")
        windowed = apply_window(values, window_name)
        if windowed is None:
            raise ValueError(f"Unsupported window '{window_name}'")
        self.values = windowed
        self._recompute_tf()
    
    def apply_padding(
        self,
        padding_length: int,
        location: str = "end",
        value: int = 0,
    ) -> None:
        values = self.values
        if values is None:
            raise ValueError("IR data is not available")
        if isinstance(padding_length, bool) or not isinstance(padding_length, int):
            raise ValueError("Padding must be an integer")
        if padding_length < 0:
            raise ValueError("Padding must be non-negative")
        if padding_length == 0:
            return
        self.values = apply_padding(
            values,
            padding_length=padding_length,
            location=location,
            value=value,
        )
        self._recompute_tf()

    def apply_filter(
        self,
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        num_taps: int = 101,
        window: str | None = None,
    ) -> None:
        values = self.values
        if values is None:
            raise ValueError("IR data is not available")
        sample_rate = self.sample_rate
        if sample_rate is None:
            raise ValueError("sample_rate is required for filters")
        self.values = apply_filter(
            values,
            filter=filter,
            sample_rate=sample_rate,
            cutoff=cutoff,
            num_taps=num_taps,
            window=window,
        )
        self._recompute_tf()

    def modify_fft_length(self, new_fft_length: int) -> None:
        if self.values is None:
            raise ValueError("IR data is not available")
        self._hrtf.fft_length = int(new_fft_length)
        self._recompute_tf(fft_length=int(new_fft_length))

    def _recompute_tf(
        self,
        fft_length: int | None = None,
        window: str | None = None,
    ) -> None:
        values = self.values
        if values is None:
            raise ValueError("IR data is not available")
        sample_rate = self.sample_rate
        if sample_rate is None:
            warnings.warn("Missing samplerate; cannot compute TF from IR.", UserWarning)
            return
        fft_length_value = fft_length if fft_length is not None else self._hrtf.fft_length
        window_value = window if window is not None else None
        tf, frequency_bins, fft_length = calculate_tf_from_ir(
            values,
            sample_rate,
            fft_length=fft_length_value,
            window_name=window_value,
        )
        self._hrtf.TF.values = tf
        self._hrtf.TF.frequency_bins = frequency_bins
        self._hrtf.fft_length = fft_length


class TF:
    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.frequency_bins: np.ndarray | None = None

    @property
    def tf_length(self) -> int | None:
        values = self.values
        if values is None:
            return None
        return int(values.shape[-1])

    @property
    def frequency_bins_step(self) -> float | None:
        frequency_bins = self.frequency_bins
        if frequency_bins is None:
            return None
        if frequency_bins.size < 2:
            return None
        diffs = np.diff(frequency_bins)
        first = float(diffs[0])
        if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
            return first
        return None

    @property
    def min_frequency_bin(self) -> float | None:
        frequency_bins = self.frequency_bins
        if frequency_bins is None:
            return None
        if frequency_bins.size == 0:
            return None
        return float(np.min(frequency_bins))

    @property
    def max_frequency_bin(self) -> float | None:
        frequency_bins = self.frequency_bins
        if frequency_bins is None:
            return None
        if frequency_bins.size == 0:
            return None
        return float(np.max(frequency_bins))

    @property
    def magnitude(self) -> np.ndarray | None:
        values = self.values
        if values is None:
            return None
        return np.abs(values)

    @property
    def magnitude_db(self) -> np.ndarray | None:
        magnitude = self.magnitude
        if magnitude is None:
            return None
        return 20.0 * np.log10(magnitude + 1e-12)

    @property
    def phase(self) -> np.ndarray | None:
        values = self.values
        if values is None:
            return None
        return np.angle(values)

    @property
    def real(self) -> np.ndarray | None:
        values = self.values
        if values is None:
            return None
        return np.real(values)

    @property
    def imaginary(self) -> np.ndarray | None:
        values = self.values
        if values is None:
            return None
        return np.imag(values)

    @property
    def fft_length(self) -> int | None:
        if self._hrtf.fft_length is None:
            return None
        return int(self._hrtf.fft_length)
