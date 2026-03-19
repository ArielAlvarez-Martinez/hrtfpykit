from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hrtf import HRTF


class FrequencyDomainWrapper:
    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf

    @property
    def tf(self) -> np.ndarray | None:
        return self._hrtf.TF

    @property
    def frequency_bins(self) -> np.ndarray | None:
        return self._hrtf.FrequencyBins

    @property
    def magnitude(self) -> np.ndarray | None:
        if self._hrtf.TF is None:
            return None
        return np.abs(self._hrtf.TF)

    @property
    def magnitude_db(self) -> np.ndarray | None:
        magnitude = self.magnitude
        if magnitude is None:
            return None
        return 20.0 * np.log10(magnitude + 1e-12)

    @property
    def phase(self) -> np.ndarray | None:
        if self._hrtf.TF is None:
            return None
        return np.angle(self._hrtf.TF)

    @property
    def real(self) -> np.ndarray | None:
        if self._hrtf.TF is None:
            return None
        return np.real(self._hrtf.TF)

    @property
    def imaginary(self) -> np.ndarray | None:
        if self._hrtf.TF is None:
            return None
        return np.imag(self._hrtf.TF)

    @property
    def fft_length(self) -> int | None:
        if self._hrtf.FFT_length is None:
            return None
        return int(self._hrtf.FFT_length)
