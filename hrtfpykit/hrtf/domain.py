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
from .metrics import itd

if TYPE_CHECKING:
    from .hrtf import HRTF


class IR:
    """Time-domain representation attached to an :class:`HRTF` object.

    ``IR`` stores the HRIR sample array and sample-rate metadata used by the
    parent HRTF abstraction. It provides convenience properties and DSP-backed
    calculations that operate on the time-domain representation without
    requiring callers to access the parent object internals.

    Attributes
    ----------
    values : numpy.ndarray or None
        Time-domain impulse-response values, usually arranged with source
        position and ear axes before the final sample axis.
    sample_rate : float or None
        Sampling rate in hertz for the impulse-response data.
    """

    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.sample_rate: float | None = None

    @property
    def ir_length(self) -> int:
        """Return the number of HRIR samples along the final axis.

        The value is derived from ``IR.values.shape[-1]`` and therefore
        reflects the current in-memory time-domain representation, including
        any padding, resampling, or replacement performed through
        ``HRTF.transform``.
        """
        return int(self.values.shape[-1])

    @property
    def ir_duration(self) -> float:
        """Return the current HRIR duration in seconds.

        Duration is computed from the sample count and ``IR.sample_rate`` using
        the shared DSP duration helper. It is meaningful only when both
        time-domain values and a valid sample rate are available.
        """
        return signal_duration(self)

    def get_itd(
        self,
        method: str = "threshold",
        output: str = "samples",
        thresh_level: float = -10.0,
        upper_cut_freq: float = 3000.0,
        filter_order: int = 10,
    ) -> np.ndarray:
        """Compute interaural time difference from the current IR values.

        Parameters
        ----------
        method : {'threshold', 'maxiacce'}, default='threshold'
            ITD estimation method.
        output : {'seconds', 'samples'}, default='samples'
            Unit used for the returned ITD values.
        thresh_level : float, default=-10.0
            Threshold offset in decibels used by the threshold estimator.
        upper_cut_freq : float, default=3000.0
            Low-pass cutoff frequency in hertz applied before estimation.
        filter_order : int, default=10
            Positive IIR Butterworth filter order used during preprocessing.

        Returns
        -------
        numpy.ndarray
            ITD values in the selected unit. Positive values indicate a
            left-ear delay relative to the right ear.

        """
        return itd(
            self,
            method=method,
            output=output,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        )


class TF:
    """Frequency-domain representation attached to an :class:`HRTF` object.

    ``TF`` stores the complex HRTF frequency-response array and its frequency
    bins for the parent HRTF abstraction. It exposes derived magnitude, phase,
    real, and imaginary views through the shared DSP utilities.

    Attributes
    ----------
    values : numpy.ndarray or None
        Frequency-domain transfer-function values, usually arranged with source
        position and ear axes before the final frequency-bin axis.
    frequency_bins : numpy.ndarray or None
        Frequency-bin values in hertz.
    """

    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.frequency_bins: np.ndarray | None = None

    @property
    def tf_length(self) -> int:
        """Return the number of HRTF frequency bins along the final axis.

        The value is derived from ``TF.values.shape[-1]`` and corresponds to
        the current one-sided frequency-domain representation used by the
        HRTF object.
        """
        return int(self.values.shape[-1])

    @property
    def frequency_bins_step(self) -> float | None:
        """Return the frequency-bin spacing in hertz when it is uniform.

        The method compares consecutive entries in ``TF.frequency_bins``. It
        returns the common spacing for uniformly sampled spectra and ``None``
        when the bins are not uniformly spaced.
        """
        frequency_bins = self.frequency_bins
        diffs = np.diff(frequency_bins)
        first = float(diffs[0])
        if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
            return first
        return None

    @property
    def min_frequency_bin(self) -> float:
        """Return the minimum available frequency bin in hertz.

        This is normally 0 Hz for one-sided spectra loaded from HRIR data, but
        it reflects the current ``TF.frequency_bins`` array exactly.
        """
        return float(np.min(self.frequency_bins))

    @property
    def max_frequency_bin(self) -> float:
        """Return the maximum available frequency bin in hertz.

        For uniformly sampled one-sided spectra, this usually corresponds to
        the Nyquist frequency implied by the IR sample rate and FFT length.
        """
        return float(np.max(self.frequency_bins))

    @property
    def magnitude(self) -> np.ndarray:
        """Return the linear magnitude of the complex HRTF values.

        The result has the same source, ear, and frequency-bin layout as
        ``TF.values`` and is computed through the shared DSP magnitude helper.
        """
        return magnitude(self)

    def get_magnitude_db(self, reference: float | str = 1.0) -> np.ndarray:
        """Return TF magnitude in decibels.

        Parameters
        ----------
        reference : float or str, default=1.0
            Reference magnitude passed to the shared decibel conversion
            routine.

        Returns
        -------
        numpy.ndarray
            Magnitude values in decibels.
        """
        return magnitude_db(self, reference=reference)

    @property
    def phase(self) -> np.ndarray:
        """Return the phase of the complex HRTF values in degrees.

        The result has the same shape as ``TF.values`` and uses the library
        phase helper so phase handling is consistent with transform methods.
        """
        return phase(self)

    @property
    def real(self) -> np.ndarray:
        """Return the real component of the complex HRTF values.

        This property exposes ``Data.Real``-style values for the current
        frequency-domain representation without modifying the parent HRTF.
        """
        return real(self)

    @property
    def imag(self) -> np.ndarray:
        """Return the imaginary component of the complex HRTF values.

        This property exposes ``Data.Imag``-style values for the current
        frequency-domain representation without modifying the parent HRTF.
        """
        return imag(self)
