from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .dsp import (
    imag,
    magnitude,
    magnitude_db,
    phase,
    real,
    signal_duration,
)
from .metrics import calculate_itd

if TYPE_CHECKING:
    from .hrtf import HRTF


class IR:
    """Time-domain view of an HRTF dataset."""

    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.sample_rate: float | None = None

    @property
    def ir_length(self) -> int:
        """Return the number of IR samples along the last axis."""
        return int(self.values.shape[-1])

    @property
    def ir_duration(self) -> float:
        """Return the IR duration in seconds."""
        return signal_duration(self)

    def get_itd(
        self,
        method: str = "threshold",
        output: str = "seconds",
        thresh_level: float = -10.0,
        upper_cut_freq: float = 3000.0,
        filter_order: int = 10,
    ) -> np.ndarray:
        """General Description:
        Compute interaural time difference (ITD) from current IR data.

        Parameters:
        - method: ITD estimator (`threshold` or `maxiacce`).
        - output: Output unit (`seconds` or `samples`).
        - thresh_level: Threshold offset in dB for `threshold` method.
        - upper_cut_freq: Low-pass cutoff in Hz applied before ITD estimation.
        - filter_order: Positive IIR Butterworth order for preprocessing.

        Returns:
        - Array of ITD values in selected `output` units. Positive means left-ear delay relative to right-ear.

        """
        return calculate_itd(
            self,
            method=method,
            output=output,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        )


class TF:
    """Frequency-domain view of an HRTF dataset."""

    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.frequency_bins: np.ndarray | None = None

    @property
    def tf_length(self) -> int:
        """Return the number of TF bins along the last axis."""
        return int(self.values.shape[-1])

    @property
    def frequency_bins_step(self) -> float | None:
        """Return bin spacing when frequency bins are uniformly spaced."""
        frequency_bins = self.frequency_bins
        diffs = np.diff(frequency_bins)
        first = float(diffs[0])
        if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
            return first
        return None

    @property
    def min_frequency_bin(self) -> float:
        """Return the minimum frequency bin value."""
        return float(np.min(self.frequency_bins))

    @property
    def max_frequency_bin(self) -> float:
        """Return the maximum frequency bin value."""
        return float(np.max(self.frequency_bins))

    @property
    def magnitude(self) -> np.ndarray:
        """Return TF magnitude."""
        return magnitude(self)

    def get_magnitude_db(self, reference: float | str = 1.0) -> np.ndarray:
        return magnitude_db(self, reference=reference)

    @property
    def phase(self) -> np.ndarray:
        """Return TF phase in degrees."""
        return phase(self)

    @property
    def real(self) -> np.ndarray:
        """Return the real part of TF values."""
        return real(self)

    @property
    def imag(self) -> np.ndarray:
        """Return the imaginary part of TF values."""
        return imag(self)
