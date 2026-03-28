from __future__ import annotations

"""Domain abstractions for IR and TF data.

Returns:
    None.
"""

from typing import TYPE_CHECKING

import warnings
import numpy as np

from .dsp import (
    apply_ir_crop,
    apply_filter,
    apply_padding,
    apply_window,
    calculate_tf_from_ir,
    get_imag,
    get_magnitude,
    get_magnitude_db,
    get_phase,
    get_real,
    signal_duration,
)

if TYPE_CHECKING:
    from .hrtf import HRTF


class IR:
    """Time-domain view of an HRTF dataset.

    Returns:
        IR instance.
    """

    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.sample_rate: float | None = None

    @property
    def ir_length(self) -> int | None:
        """Return the number of IR samples along the last axis.

        Returns:
            int | None.
        """
        
        return int(self.values.shape[-1])

    @property
    def ir_duration(self) -> float:
        """Return the IR duration in seconds.

        Returns:
            float.
        """

        return signal_duration(self)

    def apply_crop(
        self,
        start: int | None = None,
        end: int | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> None:
        """General Description:
        Crop the IR using sample indices or seconds and refresh TF.

        Parameters:
        - start: Start sample index (inclusive).
        - end: End sample index (exclusive).
        - start_seconds: Start time in seconds.
        - end_seconds: End time in seconds.

        Returns:
        None.

        Use Cases:
        - Trim early reflections or isolate a time segment.
        - Work in seconds when sample-rate-independent editing is needed.

        Best Practices:
        - Use either sample indices or seconds mode, not both at once.
        - Recompute linked TF via this method instead of manual array slicing.
        """
    
        self.values = apply_ir_crop(
            self,
            start=start,
            end=end,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        self._recompute_tf()

    def apply_window(self, window_name: str) -> None:
        """General Description:
        Apply a named window to IR values and refresh TF.

        Parameters:
        - window_name: Window identifier (for example hann, hamming, blackman).

        Returns:
        None.

        Use Cases:
        - Reduce spectral leakage before frequency analysis or conversion.

        Best Practices:
        - Use supported window names only.
        - Apply windowing intentionally because it modifies amplitude/energy distribution.
        """

        windowed = apply_window(self.values, window_name)
        self.values = windowed
        self._recompute_tf()
    
    def apply_padding(
        self,
        padding_length: int,
        location: str = "end",
        value: int = 0,
    ) -> None:
        """General Description:
        Add constant-value padding to IR values and refresh TF.

        Parameters:
        - padding_length: Number of samples to add.
        - location: Padding side, start or end.
        - value: Constant value used for padded samples.

        Returns:
        None.

        Use Cases:
        - Increase IR length prior to FFT operations.
        - Align signals for subsequent processing.

        Best Practices:
        - Prefer end padding for most HRIR workflows.
        - Keep padding length explicit and documented in experiments.
        """
       
        if padding_length == 0:
            return
        self.values = apply_padding(
            self.values,
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
        """General Description:
        Apply FIR filtering in time domain and refresh TF.

        Parameters:
        - filter: Filter type (lowpass, highpass, bandpass aliases supported).
        - cutoff: Cutoff frequency or tuple of frequencies for bandpass.
        - num_taps: FIR kernel length.
        - window: Window type used for FIR design.

        Returns:
        None.

        Use Cases:
        - Remove frequency regions or isolate frequency bands in HRIR data.

        Best Practices:
        - Use odd num_taps for linear-phase FIR behavior.
        - Keep cutoff values inside valid Nyquist limits.
        """
        
        self.values = apply_filter(
            self.values,
            filter=filter,
            sample_rate=self.sample_rate,
            cutoff=cutoff,
            num_taps=num_taps,
            window=window,
        )
        self._recompute_tf()

    def _recompute_tf(
        self,
        fft_length: int | None = None,
        window: str | None = None,
    ) -> None:
        """General Description:
        Recompute TF from current IR values and sample rate.

        Parameters:
        - fft_length: Optional FFT length override.
        - window: Optional window name applied before FFT.

        Returns:
        None.

        Use Cases:
        - Internal synchronization after IR-domain edits.

        Best Practices:
        - Keep this method as the single IR->TF synchronization point.
        - Avoid calling it with missing sample_rate.
        """
    
        fft_length_value = fft_length if fft_length is not None else self._hrtf.fft_length
        window_value = window if window is not None else None
        tf, frequency_bins, fft_length = calculate_tf_from_ir(
            self.values,
            self.sample_rate,
            fft_length=fft_length_value,
            window_name=window_value,
        )
        self._hrtf.TF.values = tf
        self._hrtf.TF.frequency_bins = frequency_bins
        self._hrtf.fft_length = fft_length


class TF:
    """Frequency-domain view of an HRTF dataset.

    Returns:
        TF instance.
    """

    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.frequency_bins: np.ndarray | None = None

    @property
    def tf_length(self) -> int:
        """Return the number of TF bins along the last axis.

        Returns:
            int.
        """
        return int(self.values.shape[-1])

    @property
    def frequency_bins_step(self) -> float | None:
        """Return bin spacing when frequency bins are uniformly spaced.

        Returns:
            float | None.
        """
        frequency_bins = self.frequency_bins
        diffs = np.diff(frequency_bins)
        first = float(diffs[0])
        if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
            return first
        return None

    @property
    def min_frequency_bin(self) -> float | None:
        """Return the minimum frequency bin value.

        Returns:
            float | None.
        """
        return float(np.min(self.frequency_bins))

    @property
    def max_frequency_bin(self) -> float | None:
        """Return the maximum frequency bin value.

        Returns:
            float | None.
        """
        return float(np.max(self.frequency_bins))

    @property
    def magnitude(self) -> np.ndarray:
        """Return TF magnitude.

        Returns:
            np.ndarray.
        """
        return get_magnitude(self)

    @property
    def magnitude_db(self) -> np.ndarray:
        """Return TF magnitude in decibels.

        Returns:
            np.ndarray.
        """
        return get_magnitude_db(self)

    @property
    def phase(self) -> np.ndarray:
        """Return TF phase in degrees.

        Returns:
            np.ndarray.
        """
        return get_phase(self)

    @property
    def real(self) -> np.ndarray:
        """Return the real part of TF values.

        Returns:
            np.ndarray.
        """
        return get_real(self)

    @property
    def imag(self) -> np.ndarray:
        """Return the imaginary part of TF values.

        Returns:
            np.ndarray.
        """
        return get_imag(self)
