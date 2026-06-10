from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ..utils.dsp import (
    downsampling,
    fir_filter,
    iir_filter,
    ir_from_tf,
    minimum_phase,
    modify_magnitude,
    modify_phase,
    padding,
    tf_gain,
    tf_from_ir,
    upsampling,
    window,
)
from ..utils.metrics import itd
from .domain import IR, TF
from ..utils.directivity import ctf_from_hrtf, dtf_from_hrtf

if TYPE_CHECKING:
    from .hrtf import HRTF


class Transform:
    def __init__(self, hrtf: "HRTF") -> None:
        """Provide immutable HRTF-processing operations for one parent object.

        :class:`~hrtfpykit.hrtf.transforms.Transform` is accessed through
        :attr:`~hrtfpykit.hrtf.HRTF.transform` and provides non-mutating
        HRTF-processing operations. Each method clones the parent
        :class:`~hrtfpykit.hrtf.HRTF` object, applies one operation to the
        clone, resynchronizes the affected domain representation, marks the
        returned object as transformed, and leaves the original HRTF unchanged.

        Time-domain operations modify the impulse-response array stored in
        :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>`, then refresh the
        frequency-response array stored in
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`. Frequency-domain
        operations do the reverse: they modify the transfer-function array and
        refresh the impulse-response array. This keeps both acoustic representations
        available for later plotting, metric calculation, SOFA synchronization,
        and export.

        Parameters
        ----------
        hrtf : :class:`~hrtfpykit.hrtf.HRTF`
            Parent :class:`~hrtfpykit.hrtf.HRTF` object used as the
            source for cloned transform results.

        Examples
        --------
        Access the transform namespace from a loaded HRTF and apply a
        preprocessing step without changing the original object:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> windowed = hrtf.transform.apply_window("hann")
        >>> hrtf.is_transformed()
        False
        >>> windowed.is_transformed()
        True
        """
        self._hrtf = hrtf

    def apply_window(
        self,
        window_name: str,
        start_sample: int | None = None,
        end_sample: int | None = None,
    ) -> "HRTF":
        """Apply a time-domain window to IR values and rebuild TF.

        The window is applied along the final IR sample axis. By default, it
        spans the complete HRIR. ``start_sample`` and ``end_sample`` restrict
        the window to one interval while samples outside that interval remain
        unchanged. The returned HRTF keeps the original source and ear layout,
        stores the windowed IR, and recomputes the frequency-domain
        representation with the current fft_length.

        Parameters
        ----------
        window_name : str
            Window identifier passed to the DSP layer, for example ``hann``,
            ``hamming``, ``blackman``, or ``rectangular``.
        start_sample : int or None, default=None
            First sample included in the windowed interval. None starts at
            sample 0.
        end_sample : int or None, default=None
            First sample after the windowed interval. None uses the full IR
            length.

        Returns
        -------
        HRTF
            A new HRTF instance with windowed IR values and refreshed TF data.

        Raises
        ------
        ValueError
            If IR data are unavailable, the requested window is unsupported, or
            the requested sample interval is invalid.

        Examples
        --------
        Apply a Hann window to the HRIR samples and keep the source and ear
        layout unchanged:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> windowed = hrtf.transform.apply_window(
        ...     "hann",
        ...     start_sample=0,
        ...     end_sample=128,
        ... )
        >>> windowed.IR.values.shape
        (793, 2, 256)
        >>> windowed.TF.values.shape
        (793, 2, 129)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = window(
            ir,
            window_name,
            start_sample=start_sample,
            end_sample=end_sample,
        )
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def apply_padding(
        self,
        padding_length: int,
        location: str = "end",
        value: float = 0,
    ) -> "HRTF":
        """Pad IR values along the sample axis and rebuild TF.

        Padding is applied to the final IR axis while preserving source and ear
        layout. The frequency-domain representation is recomputed from the
        padded IR in the returned HRTF.

        Parameters
        ----------
        padding_length : int
            Number of samples added to the IR.
        location : {``start``, ``end``}, default=``end``
            Side where the padding is applied.
        value : float, default=0
            Constant value used in the padded region.

        Returns
        -------
        HRTF
            A new HRTF instance with padded IR values and refreshed TF data.

        Raises
        ------
        ValueError
            If IR data are unavailable, padding length is invalid, or
            location is not supported.

        Examples
        --------
        Append silent samples to every HRIR and rebuild the frequency-domain
        representation from the padded signals:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.IR.values.shape
        (793, 2, 256)
        >>> padded = hrtf.transform.apply_padding(32, location="end")
        >>> padded.IR.values.shape
        (793, 2, 288)
        >>> padded.TF.values.shape
        (793, 2, 129)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = padding(
            ir,
            padding_length=padding_length,
            location=location,
            value=value,
        )
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def upsampling(
        self,
        new_sample_rate: float,
    ) -> "HRTF":
        """Upsample impulse-response values and resynchronize TF data.

        The transform changes :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>`
        and :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` in
        the returned object, then recomputes
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and frequency bins
        from the resampled IR.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in Hz. It must be strictly greater than the
            current IR sample rate.

        Returns
        -------
        HRTF
            A new HRTF instance with upsampled IR values, updated IR sample
            rate, and refreshed TF data.

        Raises
        ------
        ValueError
            If IR data or sample-rate metadata are unavailable, or if the
            target sample rate is not finite and greater than the current rate.

        Examples
        --------
        Resample a SOFA-loaded HRTF to a higher sampling rate before later
        time-domain processing:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.IR.sample_rate
        44100.0
        >>> hrtf.IR.values.shape
        (793, 2, 256)
        >>> upsampled = hrtf.transform.upsampling(88200.0)
        >>> upsampled.IR.sample_rate
        88200.0
        >>> upsampled.IR.values.shape
        (793, 2, 512)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values, ir.sample_rate = upsampling(
            ir,
            new_sample_rate=new_sample_rate,
        )
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def downsampling(
        self,
        new_sample_rate: float,
    ) -> "HRTF":
        """Downsample impulse-response values and resynchronize TF data.

        The transform changes :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>`
        and :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` in
        the returned object, then recomputes
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and frequency bins
        from the resampled IR.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in Hz. It must be strictly lower than the
            current IR sample rate.

        Returns
        -------
        HRTF
            A new HRTF instance with downsampled IR values, updated IR sample
            rate, and refreshed TF data.

        Raises
        ------
        ValueError
            If IR data or sample-rate metadata are unavailable, or if the
            target sample rate is not finite and lower than the current rate.

        Examples
        --------
        Resample a SOFA-loaded HRTF to a lower sampling rate and keep TF data
        synchronized with the new HRIR samples:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.IR.sample_rate
        44100.0
        >>> hrtf.IR.values.shape
        (793, 2, 256)
        >>> downsampled = hrtf.transform.downsampling(22050.0)
        >>> downsampled.IR.sample_rate
        22050.0
        >>> downsampled.IR.values.shape
        (793, 2, 128)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values, ir.sample_rate = downsampling(
            ir,
            new_sample_rate=new_sample_rate,
        )
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def apply_fir_filter(
        self,
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        num_taps: int = 101,
        window: str | None = None,
    ) -> "HRTF":
        """Apply FIR filtering to IR values and rebuild TF.

        Filtering is performed in the time domain along the final IR sample
        axis. The returned object stores the filtered IR and recomputes the TF
        representation from it.

        Parameters
        ----------
        filter : str
            Filter type. Low-pass, high-pass, and band-pass aliases are
            accepted by the DSP layer.
        cutoff : float | tuple[float, float] | None, default=None
            Cutoff frequency or frequency pair in Hz.
        num_taps : int, default=101
            FIR filter length.
        window : str | None, default=None
            Optional FIR design window.

        Returns
        -------
        HRTF
            A new HRTF instance with filtered IR values and refreshed TF data.

        Raises
        ------
        ValueError
            If IR data or sample-rate metadata are unavailable, filter
            arguments are invalid, or cutoff values are incompatible with the
            sample rate.

        Examples
        --------
        Low-pass filter the HRIRs with an FIR design and use the returned HRTF
        for subsequent metric or plotting workflows:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> filtered = hrtf.transform.apply_fir_filter(
        ...     filter="lowpass",
        ...     cutoff=3000.0,
        ...     num_taps=31,
        ... )
        >>> filtered.IR.values.shape
        (793, 2, 256)
        >>> filtered.is_transformed()
        True
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = fir_filter(
            ir,
            filter=filter,
            sample_rate=ir.sample_rate,
            cutoff=cutoff,
            num_taps=num_taps,
            window=window,
        )
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def apply_iir_filter(
        self,
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        order: int = 10,
    ) -> "HRTF":
        """Apply IIR filtering to IR values and rebuild TF.

        Filtering is performed in the time domain along the final IR sample
        axis. The returned object stores the filtered IR and recomputes the TF
        representation from it.

        Parameters
        ----------
        filter : str
            Filter type. Low-pass, high-pass, and band-pass aliases are
            accepted by the DSP layer.
        cutoff : float | tuple[float, float] | None, default=None
            Cutoff frequency or frequency pair in Hz.
        order : int, default=10
            Butterworth filter order.

        Returns
        -------
        HRTF
            A new HRTF instance with filtered IR values and refreshed TF data.

        Raises
        ------
        ValueError
            If IR data or sample-rate metadata are unavailable, filter
            arguments are invalid, or cutoff values are incompatible with the
            sample rate.

        Examples
        --------
        Apply a Butterworth low-pass filter to the HRIRs and keep the derived
        transfer functions available on the returned object:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> filtered = hrtf.transform.apply_iir_filter(
        ...     filter="lowpass",
        ...     cutoff=3000.0,
        ...     order=4,
        ... )
        >>> filtered.IR.values.shape
        (793, 2, 256)
        >>> filtered.TF.values.shape
        (793, 2, 129)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = iir_filter(
            ir,
            filter=filter,
            sample_rate=ir.sample_rate,
            cutoff=cutoff,
            order=order,
        )
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def minimum_phase(
        self,
        method: str = "homomorphic",
        fft_length: int | None = None,
        epsilon: float = 1e-12,
    ) -> "HRTF":
        """Convert IR values to minimum phase and rebuild TF.

        The transform replaces each current HRIR with a minimum-phase version
        derived from its magnitude response. The returned object then refreshes
        the TF representation from the minimum-phase IR.

        Parameters
        ----------
        method : str, default=``homomorphic``
            Minimum-phase method key passed to the DSP layer.
        fft_length : int | None, default=None
            Optional FFT length used during cepstral reconstruction.
        epsilon : float, default=1e-12
            Small positive floor used for numerical stability.

        Returns
        -------
        HRTF
            A new HRTF instance with minimum-phase IR values and refreshed TF data.

        Raises
        ------
        ValueError
            If IR data are unavailable or minimum-phase parameters are invalid.

        Examples
        --------
        Convert the HRIRs to a minimum-phase representation while preserving
        the current HRTF layout:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> minimum = hrtf.transform.minimum_phase()
        >>> minimum.IR.values.shape
        (793, 2, 256)
        >>> minimum.is_transformed()
        True
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = minimum_phase(
            ir,
            method=method,
            fft_length=fft_length,
            epsilon=epsilon,
        )
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def to_ctf(
        self,
        weights: bool = False,
        magnitude_average: str = "log",
        attenuation: float | None = None,
    ) -> "HRTF":
        """Convert the current HRTF into its common transfer function (CTF).

        The CTF collapses the source axis into a single common response per ear.
        The returned HRTF keeps a singleton source axis for compatibility with
        source-based plotting and SOFA update workflows.

        Parameters
        ----------
        weights : bool, optional
            If False, all source positions contribute equally. If True,
            diffuse-field weights are derived internally from the HRTF source
            positions using spherical Voronoi areas.
        magnitude_average : {``log``, ``linear``}, optional
            Rule used to average source magnitudes before the minimum-phase
            CTF reconstruction. ``log`` computes a log-magnitude average
            (geometric mean in linear magnitude). ``linear`` computes a
            direct linear-magnitude average (arithmetic mean).
        attenuation : float | None, optional
            Optional attenuation in dB applied to the CTF magnitude before the
            minimum-phase reconstruction. If None, no attenuation is
            applied.

        Returns
        -------
        HRTF
            A new HRTF instance containing the CTF. The output keeps a
            singleton compatibility source axis.

        Raises
        ------
        ValueError
            If TF data, IR reference length, source geometry, weighting
            inputs, or averaging parameters are invalid.

        Examples
        --------
        Estimate the common transfer function of a SOFA-loaded HRTF and inspect
        the singleton source axis kept in the result:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.TF.values.shape
        (793, 2, 129)
        >>> ctf = hrtf.transform.to_ctf(weights=False)
        >>> ctf.TF.values.shape
        (1, 2, 129)
        """
        transformed_hrtf = ctf_from_hrtf(
            hrtf=self._hrtf,
            weights=weights,
            magnitude_average=magnitude_average,
            attenuation=attenuation,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def to_dtf(
        self,
        weights: bool = False,
        magnitude_average: str = "log",
        attenuation: float | None = None,
    ) -> "HRTF":
        """Convert the current HRTF into its directional transfer function (DTF).

        The DTF removes a common transfer component estimated from the current
        HRTF. The returned HRTF preserves the source layout of the input object
        and rebuilds IR data from the DTF-domain TF values.

        Parameters
        ----------
        weights : bool, optional
            If False, all source positions contribute equally to the
            internal CTF estimate. If True, diffuse-field weights are
            derived internally from the HRTF source positions using spherical
            Voronoi areas.
        magnitude_average : {``log``, ``linear``}, optional
            Rule used to estimate the internal CTF magnitude before the DTF
            division. ``log`` computes a log-magnitude average
            (geometric mean in linear magnitude). ``linear`` computes a
            direct linear-magnitude average (arithmetic mean).
        attenuation : float | None, optional
            Optional attenuation in dB applied to the DTF after the CTF
            division. If None, no attenuation is applied.

        Returns
        -------
        HRTF
            A new HRTF instance containing the DTF while preserving the
            source layout of the current HRTF.

        Raises
        ------
        ValueError
            If TF data, IR reference length, source geometry, weighting
            inputs, or averaging parameters are invalid.

        Examples
        --------
        Remove the common transfer component while keeping the original source
        grid layout:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.TF.values.shape
        (793, 2, 129)
        >>> dtf = hrtf.transform.to_dtf(weights=False)
        >>> dtf.TF.values.shape
        (793, 2, 129)
        >>> dtf.IR.values.shape
        (793, 2, 256)
        """
        transformed_hrtf = dtf_from_hrtf(
            hrtf=self._hrtf,
            weights=weights,
            magnitude_average=magnitude_average,
            attenuation=attenuation,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def modify_ir(
        self,
        new_ir: np.ndarray | IR | "HRTF",
    ) -> "HRTF":
        """Replace time-domain IR values and rebuild TF data.

        new_ir replaces the full current IR array. The leading dimensions
        before the final sample axis must match the current spatial and ear
        layout. When ``new_ir`` is an :class:`~hrtfpykit.hrtf.domain.IR` or
        :class:`~hrtfpykit.hrtf.HRTF` object and provides a sample rate,
        that sample rate is copied into the returned HRTF before TF
        recomputation.

        Parameters
        ----------
        new_ir : np.ndarray | IR | HRTF
            Time-domain data used to replace the current IR values. NumPy
            arrays must keep the same spatial and ear layout as the current
            HRTF. :class:`~hrtfpykit.hrtf.domain.IR` and :class:`~hrtfpykit.hrtf.HRTF` inputs contribute their IR values, and
            when available their sample rate.

        Returns
        -------
        HRTF
            A new HRTF instance with modified IR values and rebuilt TF data.

        Raises
        ------
        ValueError
            If replacement data are missing, empty, not array-like in the
            expected way, or do not match the current leading IR/TF layout.

        Examples
        --------
        Replace HRIR values with an edited copy and let the transform rebuild
        the transfer functions:

        >>> import numpy as np
        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> ir = np.array(hrtf.IR.values, copy=True)
        >>> ir[..., :8] = 0.0
        >>> modified = hrtf.transform.modify_ir(ir)
        >>> modified.IR.values[0, 0, :8]
        array([0., 0., 0., 0., 0., 0., 0., 0.])
        >>> modified.IR.values.shape
        (793, 2, 256)
        >>> modified.TF.values.shape
        (793, 2, 129)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR

        resolved_sample_rate = ir.sample_rate
        new_ir_values: np.ndarray | None
        if isinstance(new_ir, np.ndarray):
            new_ir_values = new_ir
        elif isinstance(new_ir, self._hrtf.__class__):
            new_ir_values = new_ir.IR.values
            if new_ir.IR.sample_rate is not None:
                resolved_sample_rate = new_ir.IR.sample_rate
        elif isinstance(new_ir, IR):
            new_ir_values = new_ir.values
            if new_ir.sample_rate is not None:
                resolved_sample_rate = new_ir.sample_rate
        else:
            raise ValueError("new_ir must be a NumPy array, IR, or HRTF instance")

        if new_ir_values is None:
            raise ValueError("IR data is not available")
        if not isinstance(new_ir_values, np.ndarray):
            raise ValueError("new_ir values must be a NumPy array")
        if new_ir_values.size == 0:
            raise ValueError("new_ir must be non-empty")
        if new_ir_values.ndim == 0:
            raise ValueError("new_ir must have at least one dimension")

        if ir.values is not None and new_ir_values.shape[:-1] != ir.values.shape[:-1]:
            raise ValueError("new_ir leading shape must match the current IR layout")
        if ir.values is None and transformed_hrtf.TF.values is not None:
            if new_ir_values.shape[:-1] != transformed_hrtf.TF.values.shape[:-1]:
                raise ValueError("new_ir leading shape must match the current TF layout")

        ir.values = np.array(new_ir_values, copy=True)
        ir.sample_rate = resolved_sample_rate

        if transformed_hrtf.fft_length is not None:
            transformed_hrtf.fft_length = max(
                int(transformed_hrtf.fft_length),
                int(ir.values.shape[-1]),
            )

        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def modify_phase(
        self,
        new_phase: np.ndarray,
        unit: str = "degrees",
    ) -> "HRTF":
        """Replace TF phase values and rebuild IR.

        The transform preserves the current TF magnitude and replaces only the
        phase component. The replacement phase must be compatible with the
        current TF layout and is interpreted according to unit.

        Parameters
        ----------
        new_phase : np.ndarray
            Phase array with the same TF layout as the current HRTF.
        unit : {``degrees``, ``radians``}, default=``degrees``
            Unit used by new_phase.

        Returns
        -------
        HRTF
            A new HRTF instance with modified TF phase and rebuilt IR data.

        Raises
        ------
        ValueError
            If TF data or frequency bins are unavailable, unit is invalid,
            or the replacement phase cannot be broadcast to the TF layout.

        Examples
        --------
        Replace phase values with an edited phase array while preserving the
        current TF magnitude:

        >>> import numpy as np
        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> phase = np.array(hrtf.TF.phase, copy=True)
        >>> phase[..., 1:] *= 0.95
        >>> modified = hrtf.transform.modify_phase(phase, unit="radians")
        >>> modified.TF.values.shape
        (793, 2, 129)
        >>> modified.IR.values.shape
        (793, 2, 256)
        """
        transformed_hrtf = self._hrtf.clone()
        tf = transformed_hrtf.TF
        tf.values = modify_phase(
            tf,
            new_phase=new_phase,
            unit=unit,
        )
        ir_from_tf(
            tf,
            frequency_bins=tf.frequency_bins,
            mesh2hrtf_compatible=transformed_hrtf.mesh2hrtf_compatible,
            n_shift=transformed_hrtf.mesh2hrtf_n_shift,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def modify_tf(
        self,
        new_tf: np.ndarray | TF | "HRTF",
    ) -> "HRTF":
        """Replace frequency-domain TF values and rebuild IR data.

        new_tf replaces the full complex TF array. The leading dimensions
        before the frequency axis must match the current TF layout, or the
        current IR leading layout when no TF data are present. Frequency bins
        are copied from :class:`~hrtfpykit.hrtf.domain.TF` or
        :class:`~hrtfpykit.hrtf.HRTF` inputs when available; otherwise
        they are reused or inferred from the current sample rate when the TF
        length changes.

        Parameters
        ----------
        new_tf : np.ndarray | TF | HRTF
            Frequency-domain data used to replace the current TF values. NumPy
            arrays must keep the same spatial and ear layout as the current
            HRTF. :class:`~hrtfpykit.hrtf.domain.TF` and
            :class:`~hrtfpykit.hrtf.HRTF` inputs contribute their TF
            values and, when available, their frequency bins.

        Returns
        -------
        HRTF
            A new HRTF instance with modified TF values and rebuilt IR data.

        Raises
        ------
        ValueError
            If replacement data are missing, empty, have incompatible leading
            shape, contain too few frequency bins, or require frequency-bin
            inference without valid sample-rate metadata.

        Examples
        --------
        Replace TF values with a scaled complex copy and rebuild the
        time-domain representation:

        >>> import numpy as np
        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> round(float(hrtf.TF.magnitude[0, 0, 1]), 6)
        0.209696
        >>> tf = np.array(hrtf.TF.values, copy=True) * 0.98
        >>> modified = hrtf.transform.modify_tf(tf)
        >>> round(float(modified.TF.magnitude[0, 0, 1]), 6)
        0.205502
        >>> modified.TF.values.shape
        (793, 2, 129)
        >>> modified.IR.values.shape
        (793, 2, 256)
        """
        transformed_hrtf = self._hrtf.clone()
        tf = transformed_hrtf.TF

        new_tf_values: np.ndarray | None
        new_frequency_bins: np.ndarray | None
        if isinstance(new_tf, np.ndarray):
            new_tf_values = new_tf
            new_frequency_bins = None
        elif isinstance(new_tf, self._hrtf.__class__):
            new_tf_values = new_tf.TF.values
            new_frequency_bins = new_tf.TF.frequency_bins
        elif isinstance(new_tf, TF):
            new_tf_values = new_tf.values
            new_frequency_bins = new_tf.frequency_bins
        else:
            raise ValueError("new_tf must be a NumPy array, TF, or HRTF instance")

        if new_tf_values is None:
            raise ValueError("TF data is not available")
        if not isinstance(new_tf_values, np.ndarray):
            raise ValueError("new_tf values must be a NumPy array")
        if new_tf_values.size == 0:
            raise ValueError("new_tf must be non-empty")
        if new_tf_values.ndim == 0:
            raise ValueError("new_tf must have at least one dimension")
        if new_tf_values.shape[-1] < 2:
            raise ValueError("new_tf length must contain at least two points")

        if tf.values is not None and new_tf_values.shape[:-1] != tf.values.shape[:-1]:
            raise ValueError("new_tf leading shape must match the current TF layout")
        if tf.values is None and transformed_hrtf.IR.values is not None:
            if new_tf_values.shape[:-1] != transformed_hrtf.IR.values.shape[:-1]:
                raise ValueError("new_tf leading shape must match the current IR layout")

        tf.values = np.array(new_tf_values, copy=True)
        if new_frequency_bins is not None:
            if not isinstance(new_frequency_bins, np.ndarray):
                raise ValueError("new_tf frequency_bins must be a NumPy array")
            if new_frequency_bins.ndim != 1:
                raise ValueError("new_tf frequency_bins must be 1D")
            if new_frequency_bins.size != tf.values.shape[-1]:
                raise ValueError("new_tf frequency_bins must match TF length")
            tf.frequency_bins = np.array(new_frequency_bins, copy=True)

        if tf.frequency_bins is None or tf.frequency_bins.shape[-1] != tf.values.shape[-1]:
            if tf.frequency_bins is not None:
                frequency_bins = np.asarray(tf.frequency_bins, dtype=float)
                if frequency_bins.ndim != 1 or frequency_bins.size < 2:
                    raise ValueError("frequency_bins must be 1D and contain at least two points")
                diffs = np.diff(frequency_bins)
                step = float(diffs[0])
                if step <= 0.0 or not np.allclose(diffs, step, rtol=1e-5, atol=1e-8):
                    raise ValueError("frequency_bins must be uniformly spaced and increasing")
                if float(np.min(frequency_bins)) < 0.0:
                    raise ValueError("Only one-sided non-negative frequency_bins are supported")
                sample_rate = step * (2 * (frequency_bins.size - 1))
                tf.frequency_bins = np.fft.rfftfreq(
                    2 * (tf.values.shape[-1] - 1),
                    d=1.0 / sample_rate,
                )
            else:
                resolved_tf_sample_rate = transformed_hrtf.IR.sample_rate
                if resolved_tf_sample_rate is None:
                    raise ValueError(
                        "sample_rate is required to infer frequency_bins when TF length changes"
                    )
                if isinstance(resolved_tf_sample_rate, bool):
                    raise ValueError("sample_rate must be a finite, positive value.")
                try:
                    resolved_tf_sample_rate = float(resolved_tf_sample_rate)
                except (TypeError, ValueError):
                    raise ValueError("sample_rate must be a finite, positive value.") from None
                if not np.isfinite(resolved_tf_sample_rate) or resolved_tf_sample_rate <= 0.0:
                    raise ValueError("sample_rate must be a finite, positive value.")
                tf.frequency_bins = np.fft.rfftfreq(
                    2 * (tf.values.shape[-1] - 1),
                    d=1.0 / resolved_tf_sample_rate,
                )

        ir_from_tf(
            tf,
            frequency_bins=tf.frequency_bins,
            mesh2hrtf_compatible=transformed_hrtf.mesh2hrtf_compatible,
            n_shift=transformed_hrtf.mesh2hrtf_n_shift,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def modify_magnitude(
        self,
        new_magnitude: np.ndarray,
        scale: str = "linear",
    ) -> "HRTF":
        """Replace TF magnitude values and rebuild IR.

        The transform preserves the current TF phase and replaces only the
        magnitude component. new_magnitude must be compatible with the
        current TF layout and is interpreted according to scale.

        Parameters
        ----------
        new_magnitude : np.ndarray
            Magnitude array with the same TF layout as the current HRTF.
        scale : {``linear``, ``db``}, default=``linear``
            Magnitude scale used by new_magnitude.

        Returns
        -------
        HRTF
            A new HRTF instance with modified TF magnitude and rebuilt IR data.

        Raises
        ------
        ValueError
            If TF data or frequency bins are unavailable, scale is invalid,
            or the replacement magnitude cannot be broadcast to the TF layout.

        Examples
        --------
        Replace the TF magnitude with a slightly attenuated copy and keep the
        original phase:

        >>> import numpy as np
        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> round(float(hrtf.TF.magnitude[0, 0, 1]), 6)
        0.209696
        >>> magnitude = np.array(hrtf.TF.magnitude, copy=True) * 0.95
        >>> modified = hrtf.transform.modify_magnitude(magnitude, scale="linear")
        >>> round(float(modified.TF.magnitude[0, 0, 1]), 6)
        0.199212
        >>> modified.TF.values.shape
        (793, 2, 129)
        >>> modified.IR.values.shape
        (793, 2, 256)
        """
        transformed_hrtf = self._hrtf.clone()
        tf = transformed_hrtf.TF
        tf.values = modify_magnitude(
            tf,
            new_magnitude=new_magnitude,
            scale=scale,
        )
        ir_from_tf(
            tf,
            frequency_bins=tf.frequency_bins,
            mesh2hrtf_compatible=transformed_hrtf.mesh2hrtf_compatible,
            n_shift=transformed_hrtf.mesh2hrtf_n_shift,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def apply_gain(
        self,
        gain: float | np.ndarray,
        scale: str = "db",
    ) -> "HRTF":
        """Apply a TF-domain gain and rebuild IR.

        Gain modifies TF magnitude while preserving phase. Scalar gains apply
        globally; array gains can target sources, ears, or frequency bins when
        they are broadcast-compatible with :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.

        Parameters
        ----------
        gain : float | np.ndarray
            Gain applied to the current TF magnitude while preserving phase.
            Scalar gains affect every source, ear, and bin equally. Array
            gains must be broadcast-compatible with the current TF shape. In
            scale=``db``, negative values attenuate and positive values
            amplify.
        scale : {``linear``, ``db``}, default=``db``
            Scale used by gain.

        Returns
        -------
        HRTF
            A new HRTF instance with gain-adjusted TF values and rebuilt IR
            data.

        Raises
        ------
        ValueError
            If TF data or frequency bins are unavailable, scale is invalid,
            or gain cannot be broadcast to the current TF layout.

        Examples
        --------
        Apply a broadband attenuation in dB to all source positions and ears:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> round(float(hrtf.TF.magnitude[0, 0, 1]), 6)
        0.209696
        >>> quieter = hrtf.transform.apply_gain(-3.0, scale="db")
        >>> round(float(quieter.TF.magnitude[0, 0, 1]), 6)
        0.148454
        >>> quieter.TF.values.shape
        (793, 2, 129)
        >>> quieter.is_transformed()
        True
        """
        transformed_hrtf = self._hrtf.clone()
        tf = transformed_hrtf.TF
        tf.values = tf_gain(
            tf,
            gain=gain,
            scale=scale,
        )
        ir_from_tf(
            tf,
            frequency_bins=tf.frequency_bins,
            mesh2hrtf_compatible=transformed_hrtf.mesh2hrtf_compatible,
            n_shift=transformed_hrtf.mesh2hrtf_n_shift,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def modify_fft_length(self, new_fft_length: int) -> "HRTF":
        """Set the HRTF FFT length and recompute TF from the current IR.

        The returned object keeps the current IR unchanged and rebuilds TF
        using new_fft_length. This changes frequency-bin spacing and TF
        length but does not add time-domain information to the HRIR.

        Parameters
        ----------
        new_fft_length : int
            FFT size used for IR-to-TF conversion.

        Returns
        -------
        HRTF
            A new HRTF instance with updated FFT length and recomputed TF data.

        Raises
        ------
        ValueError
            If IR data are unavailable or the FFT length is invalid for
            real-FFT conversion.

        Examples
        --------
        Increase the FFT length used for IR-to-TF conversion and inspect the
        resulting frequency-bin count:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.fft_length
        256
        >>> hrtf.TF.values.shape
        (793, 2, 129)
        >>> modified = hrtf.transform.modify_fft_length(512)
        >>> modified.fft_length
        512
        >>> modified.TF.values.shape
        (793, 2, 257)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        if ir.values is None:
            raise ValueError("IR data is not available")
        transformed_hrtf.fft_length = int(new_fft_length)
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def add_itd(
        self,
        itd: float,
        unit: str = "samples",
    ) -> "HRTF":
        """Add an interaural time delay to IR values and rebuild TF.

        Positive ITD values delay the left ear; negative values delay the right
        ear. A scalar delay is applied to every source position. Array delays
        must match the leading IR shape before the ear and sample axes, which
        allows position-dependent ITD perturbations.

        Parameters
        ----------
        itd : float
            ITD value to apply. Positive values delay the left ear and
            negative values delay the right ear.
        unit : {``time``, ``samples``}, default=``samples``
            Unit used by itd. ``time`` is interpreted in microseconds.

        Returns
        -------
        HRTF
            A new HRTF instance with ITD-modified IR values and refreshed TF data.

        Raises
        ------
        ValueError
            If IR data do not contain two ear channels, delay values are
            non-finite, delay-array shape is incompatible with the IR layout,
            time values are requested without sample-rate metadata, or the absolute
            delay is not smaller than the IR length.

        Examples
        --------
        Add a two-sample delay to the left ear for every source position:

        >>> from hrtfpykit.hrtf import load_hrtf, itd
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> itd(hrtf, output="samples")[:5]
        array([-3, -4, -3, -4, -4])
        >>> delayed = hrtf.transform.add_itd(2, unit="samples")
        >>> itd(delayed, output="samples")[:5]
        array([-1, -2, -1, -2, -2])
        >>> delayed.IR.values.shape
        (793, 2, 256)
        >>> delayed.is_transformed()
        True
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        if ir.values is None:
            raise ValueError("IR data is not available")
        if ir.values.ndim < 2:
            raise ValueError("IR data must include ear and time axes")
        if ir.values.shape[-2] < 2:
            raise ValueError("IR ear axis must contain at least two channels (0=left, 1=right)")

        unit_key = str(unit).strip().lower()
        if unit_key not in {"time", "samples"}:
            raise ValueError("unit must be one of: time, samples")

        if isinstance(itd, bool):
            raise ValueError("itd must be finite value(s)")
        itd_values = np.asarray(itd, dtype=float)
        if itd_values.size == 0:
            raise ValueError("itd must contain at least one value")
        if not np.all(np.isfinite(itd_values)):
            raise ValueError("itd must be finite value(s)")

        if unit_key == "time":
            if ir.sample_rate is None:
                raise ValueError("IR sample_rate is required when unit='time'")
            itd_samples = np.rint(itd_values * float(ir.sample_rate) / 1_000_000.0).astype(int)
        else:
            itd_samples = np.rint(itd_values).astype(int)

        ir_values = np.asarray(ir.values, dtype=float)
        leading_shape = ir_values.shape[:-2]
        channel_count = ir_values.shape[-2]
        sample_count = ir_values.shape[-1]
        if itd_samples.ndim == 0:
            itd_samples = np.full(leading_shape, int(itd_samples), dtype=int)
        else:
            itd_samples = np.asarray(itd_samples, dtype=int)
            if itd_samples.shape != leading_shape:
                raise ValueError(
                    "itd array shape must match IR leading shape "
                    f"{leading_shape}, got {itd_samples.shape}"
                )

        if np.all(itd_samples == 0):
            tf_from_ir(
                ir,
                fft_length=transformed_hrtf.fft_length,
            )
            transformed_hrtf._transformed = True
            return transformed_hrtf

        if np.max(np.abs(itd_samples)) >= sample_count:
            raise ValueError("Absolute ITD in samples must be smaller than IR length")

        flattened = ir_values.reshape(-1, channel_count, sample_count)
        flattened_itd = itd_samples.reshape(-1)
        for index in range(flattened.shape[0]):
            delay = int(abs(flattened_itd[index]))
            if delay == 0:
                continue
            if flattened_itd[index] > 0:
                left = np.array(flattened[index, 0, :], copy=True)
                delayed_left = np.zeros_like(left)
                delayed_left[delay:] = left[:-delay]
                flattened[index, 0, :] = delayed_left
            else:
                right = np.array(flattened[index, 1, :], copy=True)
                delayed_right = np.zeros_like(right)
                delayed_right[delay:] = right[:-delay]
                flattened[index, 1, :] = delayed_right

        ir.values = flattened.reshape(*leading_shape, channel_count, sample_count)
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def delete_itd(
        self,
        method: str = "threshold",
        thresh_level: float = -10.0,
        upper_cut_freq: float = 3000.0,
        filter_order: int = 10,
    ) -> "HRTF":
        """Estimate and remove ITD from the current IR values, then resync TF.

        The ITD sign convention follows itd: positive ITD means
        left-ear delay relative to right-ear and negative ITD means right-ear
        delay relative to left-ear. Compensation is applied per source by
        advancing the delayed channel and zero-filling the tail introduced by
        the shift.

        Parameters
        ----------
        method : {``threshold``, ``maxiacce``}, default=``threshold``
            ITD estimator used to compute the delay per position.
        thresh_level : float, default=-10.0
            Threshold offset in dB used when method=``threshold``.
        upper_cut_freq : float, default=3000.0
            Low-pass cutoff in Hz applied before ITD estimation.
        filter_order : int, default=10
            Butterworth low-pass filter order used before ITD estimation.

        Returns
        -------
        HRTF
            A new HRTF instance with ITD-compensated IR values and refreshed
            TF data. Compensation is applied per source position.

        Raises
        ------
        ValueError
            If IR data do not contain two ear channels, ITD estimation
            parameters are invalid, or an estimated delay is not smaller than
            the IR length.

        Examples
        --------
        Estimate and remove ITD from each source position before comparing
        magnitude-focused features:

        >>> from hrtfpykit.hrtf import load_hrtf, itd
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> itd(hrtf, output="samples")[:5]
        array([-3, -4, -3, -4, -4])
        >>> no_itd = hrtf.transform.delete_itd()
        >>> itd(no_itd, output="samples")[:5]
        array([0, 0, 0, 0, 0])
        >>> no_itd.IR.values.shape
        (793, 2, 256)
        >>> no_itd.TF.values.shape
        (793, 2, 129)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        if ir.values is None:
            raise ValueError("IR data is not available")
        if ir.values.ndim < 2:
            raise ValueError("IR data must include ear and time axes")
        if ir.values.shape[-2] < 2:
            raise ValueError("IR ear axis must contain at least two channels (0=left, 1=right)")

        itd_samples = itd(
            transformed_hrtf,
            method=method,
            output="samples",
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        )
        itd_samples = np.asarray(itd_samples, dtype=int)

        ir_values = np.asarray(ir.values, dtype=float)
        leading_shape = ir_values.shape[:-2]
        channel_count = ir_values.shape[-2]
        sample_count = ir_values.shape[-1]
        flattened = ir_values.reshape(-1, channel_count, sample_count)
        flattened_itd = itd_samples.reshape(-1)

        for index in range(flattened.shape[0]):
            delay = int(abs(flattened_itd[index]))
            if delay == 0:
                continue
            if delay >= sample_count:
                raise ValueError("Estimated ITD in samples must be smaller than IR length")
            if flattened_itd[index] > 0:
                left = np.array(flattened[index, 0, :], copy=True)
                advanced_left = np.zeros_like(left)
                advanced_left[:-delay] = left[delay:]
                flattened[index, 0, :] = advanced_left
            else:
                right = np.array(flattened[index, 1, :], copy=True)
                advanced_right = np.zeros_like(right)
                advanced_right[:-delay] = right[delay:]
                flattened[index, 1, :] = advanced_right

        ir.values = flattened.reshape(*leading_shape, channel_count, sample_count)
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf
