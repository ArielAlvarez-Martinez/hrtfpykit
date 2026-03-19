from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hrtf import HRTF


class TimeDomainWrapper:
    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf

    @property
    def ir(self) -> np.ndarray | None:
        return self._hrtf.IR

    def apply_crop(self, start: int | None = None, end: int | None = None) -> None:
        if self._hrtf.IR is None:
            raise ValueError("IR data is not available")
        self._hrtf.IR = self._hrtf.IR[..., slice(start, end)]
        self._hrtf._recompute_tf_from_ir()

    def apply_window(self, window: str) -> None:
        if self._hrtf.IR is None:
            raise ValueError("IR data is not available")
        window_values = self._hrtf._window(window, self._hrtf.IR.shape[-1])
        if window_values is None:
            raise ValueError(f"Unsupported window '{window}'")
        self._hrtf.IR = self._hrtf.IR * window_values
        self._hrtf._recompute_tf_from_ir()

    def apply_itd_shift(self, samples: int) -> None:
        if self._hrtf.IR is None:
            raise ValueError("IR data is not available")
        self._hrtf.IR = self._hrtf._shift_ir(self._hrtf.IR, samples)
        self._hrtf._recompute_tf_from_ir()

    def apply_filter(self, kernel: np.ndarray) -> None:
        if self._hrtf.IR is None:
            raise ValueError("IR data is not available")
        kernel_arr = np.asarray(kernel)
        if kernel_arr.ndim != 1:
            raise ValueError("Filter kernel must be 1D")
        self._hrtf.IR = np.apply_along_axis(
            lambda x: np.convolve(x, kernel_arr, mode="same"),
            axis=-1,
            arr=self._hrtf.IR,
        )
        self._hrtf._recompute_tf_from_ir()

    def apply_fft_length(self, fft_length: int) -> None:
        if self._hrtf.IR is None:
            raise ValueError("IR data is not available")
        self._hrtf.FFT_length = int(fft_length)
        self._hrtf._recompute_tf_from_ir(FFT_length=int(fft_length))
