from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..utils.dsp import (
    imag,
    magnitude,
    magnitude_db,
    phase,
    real,
    signal_duration,
)
from ..utils.metrics import itd

if TYPE_CHECKING:
    from .hrtf import HRTF


class IR:
    def __init__(self, hrtf: HRTF) -> None:
        """Represent the time-domain HRIR view owned by an :class:`~hrtfpykit.hrtf.HRTF` object.

        :class:`~hrtfpykit.hrtf.domain.IR` stores the HRIR sample array and
        sample-rate metadata used by the parent
        :class:`~hrtfpykit.hrtf.HRTF` abstraction. It is created lazily by
        :attr:`~hrtfpykit.hrtf.HRTF.IR` and acts as the in-memory
        time-domain view of SOFA ``Data.IR`` data or data reconstructed from
        SimpleFreeFieldHRTF frequency-domain files.

        The object does not own independent source metadata. Its leading axes
        are expected to stay aligned with
        :attr:`~hrtfpykit.hrtf.HRTF.Sources` and with the sibling
        :attr:`~hrtfpykit.hrtf.HRTF.TF` representation. Time-domain
        transforms update :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>`
        and :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>`
        first, then synchronize
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and
        :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`
        through FFT helpers.

        Parameters
        ----------
        hrtf : :class:`~hrtfpykit.hrtf.HRTF`
            Parent :class:`~hrtfpykit.hrtf.HRTF` object that owns this
            time-domain representation.

        Attributes
        ----------
        values : numpy.ndarray or None
            Time-domain impulse-response values. Standard HRTF data use layout
            (positions, ears, samples), but methods generally treat any leading
            axes before the final sample axis as preserved metadata axes.
        sample_rate : float or None
            Sampling rate in hertz for the impulse-response data.
        """
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.sample_rate: float | None = None

    @property
    def ir_length(self) -> int:
        """Return the number of HRIR samples along the final axis.

        The value is derived from the final axis of
        :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` and therefore
        reflects the current in-memory time-domain representation, including
        any padding, resampling, or replacement performed through
        :attr:`~hrtfpykit.hrtf.HRTF.transform`.

        Returns
        -------
        int
            Number of samples on the final axis of :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>`.

        Raises
        ------
        AttributeError
            If :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` is None.
        """
        if self.values is None:
            raise ValueError("IR values are not available")
        return int(self.values.shape[-1])

    @property
    def ir_duration(self) -> float:
        """Return the current HRIR duration in seconds.

        Duration is computed from the sample count and
        :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` using the
        shared DSP duration helper. Leading source and ear axes do not affect
        the result; only the final sample axis is used.

        Returns
        -------
        float
            Duration in seconds, computed from
            :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` and
            :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>`.

        Raises
        ------
        ValueError
            If :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` is missing,
            not a NumPy array, has no sample axis, or
            :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is
            missing or not finite and positive.
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

        This is the domain-object convenience wrapper around
        :func:`~hrtfpykit.hrtf.metrics.itd`. It estimates ITD from
        :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` using
        :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` and
        preserves any leading source axes before the final (ear, samples)
        layout. The ear convention is 0=left and 1=right.

        Parameters
        ----------
        method : {``threshold``, ``maxiacce``}, default=``threshold``
            ITD estimator. ``threshold`` detects per-ear onset samples after
            low-pass filtering, while ``maxiacce`` uses envelope
            cross-correlation.
        output : {``seconds``, ``samples``}, default=``samples``
            Unit used for the returned ITD values.
        thresh_level : float, default=-10.0
            Threshold offset in decibels used when ``method`` is ``threshold``.
        upper_cut_freq : float, default=3000.0
            Low-pass cutoff frequency in hertz applied before estimation.
        filter_order : int, default=10
            Positive IIR Butterworth filter order used during preprocessing.

        Returns
        -------
        numpy.ndarray
            ITD values in the selected unit. The output shape is
            IR.values.shape[:-2]. Positive values indicate a left-ear delay
            relative to the right ear.

        Raises
        ------
        ValueError
            If :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` or :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is unavailable, the IR array
            does not contain at least two ears and two samples, the estimator
            configuration is invalid, or the threshold estimator cannot find a
            valid onset.
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
    def __init__(self, hrtf: HRTF) -> None:
        """Represent the frequency-domain HRTF view owned by an :class:`~hrtfpykit.hrtf.HRTF` object.

        :class:`~hrtfpykit.hrtf.domain.TF` stores the complex HRTF
        frequency-response array and its frequency bins for the parent
        :class:`~hrtfpykit.hrtf.HRTF` abstraction. It is created lazily by
        :attr:`~hrtfpykit.hrtf.HRTF.TF` and acts as the in-memory
        frequency-domain view of SOFA ``Data.Real`` and ``Data.Imag`` data or
        data computed from HRIR files.

        The object is expected to stay aligned with the sibling
        :attr:`~hrtfpykit.hrtf.HRTF.IR` and
        :attr:`~hrtfpykit.hrtf.HRTF.Sources` representations.
        Frequency-domain transforms update
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and
        :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`
        first, then synchronize
        :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` and
        :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` through
        inverse FFT helpers.

        Parameters
        ----------
        hrtf : :class:`~hrtfpykit.hrtf.HRTF`
            Parent :class:`~hrtfpykit.hrtf.HRTF` object that owns this frequency-domain representation.

        Attributes
        ----------
        values : numpy.ndarray or None
            Complex frequency-domain transfer-function values. Standard HRTF data
            use layout (positions, ears, frequency_bins), but derived properties
            preserve any leading axes before the final frequency axis.
        frequency_bins : numpy.ndarray or None
            One-dimensional frequency-bin values in hertz corresponding to the final
            axis of :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.
        """
        self._hrtf = hrtf
        self.values: np.ndarray | None = None
        self.frequency_bins: np.ndarray | None = None

    @property
    def tf_length(self) -> int:
        """Return the number of HRTF frequency bins along the final axis.

        The value is derived from the final axis of
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and corresponds to
        the current one-sided frequency-domain representation used by the
        :class:`~hrtfpykit.hrtf.HRTF` object.

        Returns
        -------
        int
            Number of frequency bins on the final axis of :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.

        Raises
        ------
        AttributeError
            If :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` is None.
        """
        if self.values is None:
            raise ValueError("TF values are not available")
        return int(self.values.shape[-1])

    @property
    def frequency_bins_step(self) -> float | None:
        """Return the frequency-bin spacing in hertz when it is uniform.

        The method compares consecutive entries in :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`. It
        returns the common spacing for uniformly sampled spectra and None
        when the bins are not uniformly spaced.

        Returns
        -------
        float | None
            Uniform spacing in hertz, or None when consecutive frequency
            differences are not equal within the local tolerance.

        Raises
        ------
        ValueError
            If :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` is None.
        IndexError
            If fewer than two frequency bins are available.
        """
        frequency_bins = self.frequency_bins
        if frequency_bins is None:
            raise ValueError("TF frequency_bins are not available")
        diffs = np.diff(frequency_bins)
        first = float(diffs[0])
        if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
            return first
        return None

    @property
    def min_frequency_bin(self) -> float:
        """Return the minimum available frequency bin in hertz.

        This is normally 0 Hz for one-sided spectra loaded from HRIR data, but
        it reflects the current :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` array exactly.

        Returns
        -------
        float
            Minimum value in :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`.

        Raises
        ------
        TypeError
            If :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` is None.
        ValueError
            If :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` is empty.
        """
        if self.frequency_bins is None:
            raise ValueError("TF frequency_bins are not available")
        return float(np.min(self.frequency_bins))

    @property
    def max_frequency_bin(self) -> float:
        """Return the maximum available frequency bin in hertz.

        For uniformly sampled one-sided spectra, this usually corresponds to
        the Nyquist frequency implied by the IR sample rate and FFT length.

        Returns
        -------
        float
            Maximum value in :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`.

        Raises
        ------
        TypeError
            If :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` is None.
        ValueError
            If :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` is empty.
        """
        if self.frequency_bins is None:
            raise ValueError("TF frequency_bins are not available")
        return float(np.max(self.frequency_bins))

    @property
    def magnitude(self) -> np.ndarray:
        """Return the linear magnitude of the complex HRTF values.

        The result has the same source, ear, and frequency-bin layout as
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and is computed through the shared DSP magnitude helper.

        Returns
        -------
        numpy.ndarray
            Linear magnitude values with the same shape as :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.

        Raises
        ------
        ValueError
            If :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` is missing or is not a NumPy array.
        """
        return magnitude(self)

    def get_magnitude_db(self, reference: float | str = 1.0) -> np.ndarray:
        """Return TF magnitude in decibels.

        This method computes the linear magnitude of :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and converts
        it with the shared dB conversion helper. It is the domain-object
        convenience API used by plotting and spectral inspection workflows.

        Parameters
        ----------
        reference : float | {``max``}, default=1.0
            Positive reference magnitude used for 20 * log10(magnitude /
            reference). The special value ``max`` normalizes to the
            maximum magnitude present in :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.

        Returns
        -------
        numpy.ndarray
            Magnitude values in decibels with the same shape as :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.

        Raises
        ------
        ValueError
            If :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` is missing or invalid, if a magnitude is invalid,
            or if reference is not accepted by the dB conversion helper.
        """
        return magnitude_db(self, reference=reference)

    @property
    def phase(self) -> np.ndarray:
        """Return the phase of the complex HRTF values in degrees.

        The result has the same shape as :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and uses the library
        phase helper so phase handling is consistent with transform methods.

        Returns
        -------
        numpy.ndarray
            Phase values in degrees with the same shape as :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.

        Raises
        ------
        ValueError
            If :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` is missing or is not a NumPy array.
        """
        return phase(self)

    @property
    def real(self) -> np.ndarray:
        """Return the real component of the complex HRTF values.

        This property exposes ``Data.Real``-style values for the current
        frequency-domain representation without modifying the parent HRTF.

        Returns
        -------
        numpy.ndarray
            Real component of :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` with the same shape.

        Raises
        ------
        ValueError
            If :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` is missing or is not a NumPy array.
        """
        return real(self)

    @property
    def imag(self) -> np.ndarray:
        """Return the imaginary component of the complex HRTF values.

        This property exposes ``Data.Imag``-style values for the current
        frequency-domain representation without modifying the parent HRTF.

        Returns
        -------
        numpy.ndarray
            Imaginary component of :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` with the same shape.

        Raises
        ------
        ValueError
            If :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` is missing or is not a NumPy array.
        """
        return imag(self)
