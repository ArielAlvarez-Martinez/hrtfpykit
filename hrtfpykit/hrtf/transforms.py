from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from .dsp import (
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
from .metrics import itd
from .domain import IR, TF
from .directivity import ctf_from_hrtf, dtf_from_hrtf

if TYPE_CHECKING:
    from .hrtf import HRTF


class Transform:
    """HRTF transform operations."""

    def __init__(self, hrtf: "HRTF") -> None:
        self._hrtf = hrtf

    def apply_window(self, window_name: str) -> "HRTF":
        """Apply a time-domain window to IR values and resync TF.

        Parameters
        ----------
        window_name : str
            Window identifier passed to the DSP layer, for example ``"hann"``,
            ``"hamming"``, ``"blackman"``, or ``"rectangular"``.

        Returns
        -------
        HRTF
            A new HRTF instance with windowed IR values and refreshed TF data.

        Use Cases
        ---------
        - Reduce spectral leakage before FFT conversion.
        - Prepare HRIR data for spectral analysis.

        Examples
        --------
        Apply a Hann window before inspecting the front direction:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> windowed = hrtf.transform.apply_window("hann")
        >>> windowed.plot_amplitude(positions="front", show=False)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = window(ir, window_name)
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
        """Pad IR values in time and resync TF.

        Parameters
        ----------
        padding_length : int
            Number of samples added to the IR.
        location : {"start", "end"}, default="end"
            Side where the padding is applied.
        value : float, default=0
            Constant value used in the padded region.

        Returns
        -------
        HRTF
            A new HRTF instance with padded IR values and refreshed TF data.

        Use Cases
        ---------
        - Increase IR length before FFT-based workflows.
        - Add leading or trailing samples for later processing.

        Examples
        --------
        Pad one front-facing HRIR before plotting it:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> padded = hrtf.transform.apply_padding(32, location="end")
        >>> padded.plot_amplitude(positions="front", x_axis="samples", show=False)
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
        """Upsample IR values and resync TF.

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

        Use Cases
        ---------
        - Match the sample rate required by another dataset or model.
        - Increase temporal resolution before later time-domain processing.

        Examples
        --------
        Upsample one front-facing HRIR set to 96 kHz:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> upsampled = hrtf.transform.upsampling(96000.0)
        >>> upsampled.IR.sample_rate
        96000.0
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
        """Downsample IR values and resync TF.

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

        Use Cases
        ---------
        - Match the sample rate of a lower-rate dataset or playback pipeline.

        Examples
        --------
        Downsample one front-facing HRIR set to 24 kHz:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> downsampled = hrtf.transform.downsampling(24000.0)
        >>> downsampled.IR.sample_rate
        24000.0
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
        """Apply FIR filtering to IR values and resync TF.

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

        Use Cases
        ---------
        - Remove undesired frequency content from HRIR data.
        - Isolate a band before feature extraction.

        Examples
        --------
        Low-pass one front direction with an FIR design:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> lowpassed = hrtf.transform.apply_fir_filter(
        ...     "lowpass",
        ...     cutoff=3000.0,
        ...     num_taps=31,
        ... )
        >>> lowpassed.plot_magnitude(positions="front", show=False)
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
        """Apply IIR filtering to IR values and resync TF.

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

        Use Cases
        ---------
        - Reproduce legacy IIR-based preprocessing chains.
        - Apply low-latency recursive filtering before analysis.

        Examples
        --------
        Smooth one front direction with an IIR low-pass filter:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> smoothed = hrtf.transform.apply_iir_filter(
        ...     "lowpass",
        ...     cutoff=3000.0,
        ...     order=4,
        ... )
        >>> smoothed.plot_magnitude(positions="front", show=False)
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
        """Convert IR values to minimum phase and resync TF.

        Parameters
        ----------
        method : str, default="homomorphic"
            Minimum-phase method key passed to the DSP layer.
        fft_length : int | None, default=None
            Optional FFT length used during cepstral reconstruction.
        epsilon : float, default=1e-12
            Small positive floor used for numerical stability.

        Returns
        -------
        HRTF
            A new HRTF instance with minimum-phase IR values and refreshed TF data.

        Use Cases
        ---------
        - Build minimum-phase HRIR approximations for reduced-latency processing.
        - Standardize phase behavior across datasets before analysis.

        Examples
        --------
        Convert one direction into a minimum-phase version:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> minimum_phase_hrtf = hrtf.transform.minimum_phase()
        >>> minimum_phase_hrtf.plot_amplitude(positions="front", show=False)
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

        Parameters
        ----------
        weights : bool, optional
            If ``False``, all source positions contribute equally. If ``True``,
            diffuse-field weights are derived internally from the HRTF source
            positions using spherical Voronoi areas.
        magnitude_average : {"log", "linear"}, optional
            Rule used to average source magnitudes before the minimum-phase
            CTF reconstruction. ``"log"`` computes a log-magnitude average
            (geometric mean in linear magnitude). ``"linear"`` computes a
            direct linear-magnitude average (arithmetic mean).
        attenuation : float | None, optional
            Optional attenuation in dB applied to the CTF magnitude before the
            minimum-phase reconstruction. If ``None``, no attenuation is
            applied.

        Returns
        -------
        HRTF
            A new HRTF instance containing the CTF. The output keeps a
            singleton compatibility source axis.

        Use Cases
        ---------
        - Derive the common spectral component of the current HRTF.
        - Prepare a CTF object for later DTF decomposition or reconstruction.

        Examples
        --------
        Collapse a loaded HRTF into a single common transfer function:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> ctf = hrtf.transform.to_ctf(weights=True, magnitude_average="linear")
        >>> ctf.plot_magnitude(
        ...     positions=ctf.Sources.get_positions(angle_unit="degrees")[0, :2],
        ...     ear="both",
        ...     show=False,
        ... )
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

        Parameters
        ----------
        weights : bool, optional
            If ``False``, all source positions contribute equally to the
            internal CTF estimate. If ``True``, diffuse-field weights are
            derived internally from the HRTF source positions using spherical
            Voronoi areas.
        magnitude_average : {"log", "linear"}, optional
            Rule used to estimate the internal CTF magnitude before the DTF
            division. ``"log"`` computes a log-magnitude average
            (geometric mean in linear magnitude). ``"linear"`` computes a
            direct linear-magnitude average (arithmetic mean).
        attenuation : float | None, optional
            Optional attenuation in dB applied to the DTF after the CTF
            division. If ``None``, no attenuation is applied.

        Returns
        -------
        HRTF
            A new HRTF instance containing the DTF while preserving the
            source layout of the current HRTF.

        Use Cases
        ---------
        - Remove the common spectral component of the current HRTF while
          preserving its directional structure.
        - Prepare DTF data for directivity analysis or later recombination
          with a CTF.

        Examples
        --------
        Remove the common transfer function and inspect the directional component:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> dtf = hrtf.transform.to_dtf(weights=True, attenuation=20.0)
        >>> dtf.plot_magnitude(
        ...     positions=["front", "left"],
        ...     ear="both",
        ...     show=False,
        ... )
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
        """Replace IR values and rebuild TF.

        Parameters
        ----------
        new_ir : np.ndarray | IR | HRTF
            Time-domain data used to replace the current IR values. NumPy
            arrays must keep the same spatial and ear layout as the current
            HRTF. ``IR`` and ``HRTF`` inputs contribute their IR values, and
            when available their sample rate.

        Returns
        -------
        HRTF
            A new HRTF instance with modified IR values and rebuilt TF data.

        Use Cases
        ---------
        - Replace the current HRIR values with edited or externally generated IR data.
        - Update the IR length while preserving the current source and ear layout.

        Examples
        --------
        Replace one direction with a gated HRIR:

        >>> import numpy as np
        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> edited_ir = np.array(hrtf.IR.values, copy=True)
        >>> edited_ir[..., -32:] = 0.0
        >>> gated = hrtf.transform.modify_ir(edited_ir)
        >>> gated.plot_amplitude(positions="front", show=False)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR

        resolved_sample_rate = ir.sample_rate
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

        Parameters
        ----------
        new_phase : np.ndarray
            Phase array with the same TF layout as the current HRTF.
        unit : {"degrees", "radians"}, default="degrees"
            Unit used by ``new_phase``.

        Returns
        -------
        HRTF
            A new HRTF instance with modified TF phase and rebuilt IR data.

        Use Cases
        ---------
        - Apply target phase profiles while preserving measured magnitudes.
        - Build controlled phase perturbation experiments.

        Examples
        --------
        Replace one transfer function with a zero-phase version:

        >>> import numpy as np
        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> zero_phase = np.zeros_like(hrtf.TF.phase)
        >>> phase_aligned = hrtf.transform.modify_phase(zero_phase, unit="degrees")
        >>> phase_aligned.plot_amplitude(positions="front", show=False)
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
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def modify_tf(
        self,
        new_tf: np.ndarray | TF | "HRTF",
    ) -> "HRTF":
        """Replace TF values and rebuild IR.

        Parameters
        ----------
        new_tf : np.ndarray | TF | HRTF
            Frequency-domain data used to replace the current TF values. NumPy
            arrays must keep the same spatial and ear layout as the current
            HRTF. ``TF`` and ``HRTF`` inputs contribute their TF values, and
            when available their frequency bins.

        Returns
        -------
        HRTF
            A new HRTF instance with modified TF values and rebuilt IR data.

        Use Cases
        ---------
        - Replace the current HRTF values with edited or externally generated TF data.
        - Update the TF length while preserving the current source and ear layout.
        - Reuse TF data and frequency bins from another ``TF`` object or ``HRTF`` instance.

        Examples
        --------
        Soften the highest bins before replacing one transfer function:

        >>> import numpy as np
        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> edited_tf = np.array(hrtf.TF.values, copy=True)
        >>> edited_tf[..., -24:] *= 0.5
        >>> softened = hrtf.transform.modify_tf(edited_tf)
        >>> softened.plot_magnitude(positions="front", show=False)
        """
        transformed_hrtf = self._hrtf.clone()
        tf = transformed_hrtf.TF

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
                sample_rate = transformed_hrtf.IR.sample_rate
                if sample_rate is None:
                    raise ValueError(
                        "sample_rate is required to infer frequency_bins when TF length changes"
                    )
                if isinstance(sample_rate, bool):
                    raise ValueError("sample_rate must be a finite, positive value.")
                try:
                    sample_rate = float(sample_rate)
                except (TypeError, ValueError):
                    raise ValueError("sample_rate must be a finite, positive value.") from None
                if not np.isfinite(sample_rate) or sample_rate <= 0.0:
                    raise ValueError("sample_rate must be a finite, positive value.")
                tf.frequency_bins = np.fft.rfftfreq(
                    2 * (tf.values.shape[-1] - 1),
                    d=1.0 / sample_rate,
                )

        ir_from_tf(
            tf,
            frequency_bins=tf.frequency_bins,
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def modify_magnitude(
        self,
        new_magnitude: np.ndarray,
        scale: str = "linear",
    ) -> "HRTF":
        """Replace TF magnitude values and rebuild IR.

        Parameters
        ----------
        new_magnitude : np.ndarray
            Magnitude array with the same TF layout as the current HRTF.
        scale : {"linear", "db"}, default="linear"
            Magnitude scale used by ``new_magnitude``.

        Returns
        -------
        HRTF
            A new HRTF instance with modified TF magnitude and rebuilt IR data.

        Use Cases
        ---------
        - Apply target magnitude responses while preserving phase.
        - Evaluate perceptual effects of magnitude-only modifications.

        Examples
        --------
        Tilt the magnitude while preserving the original phase:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> tilted_magnitude = hrtf.TF.magnitude * 0.9
        >>> softened = hrtf.transform.modify_magnitude(tilted_magnitude, scale="linear")
        >>> softened.plot_magnitude(positions="front", show=False)
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
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def apply_gain(
        self,
        gain: float | np.ndarray,
        scale: str = "db",
    ) -> "HRTF":
        """Apply a TF-domain gain and rebuild IR.

        Parameters
        ----------
        gain : float | np.ndarray
            Gain applied to the current TF magnitude while preserving phase.
            Scalar gains affect every source, ear, and bin equally. Array
            gains must be broadcast-compatible with the current TF shape. In
            ``scale="db"``, negative values attenuate and positive values
            amplify.
        scale : {"linear", "db"}, default="db"
            Scale used by ``gain``.

        Returns
        -------
        HRTF
            A new HRTF instance with gain-adjusted TF values and rebuilt IR
            data.

        Use Cases
        ---------
        - Apply a global attenuation or amplification in the frequency domain.
        - Create controlled level offsets while preserving the original phase.
        - Apply broadcastable per-ear or per-bin gains when TF shapes match.

        Examples
        --------
        Attenuate one selected direction by 6 dB:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> quieter = hrtf.transform.apply_gain(-6.0, scale="db")
        >>> quieter.plot_magnitude(positions="front", show=False)

        Apply a gentle linear gain boost to the whole TF:

        >>> louder = hrtf.transform.apply_gain(1.1, scale="linear")
        >>> louder.plot_magnitude(positions="front", show=False)
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
        )
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def modify_fft_length(self, new_fft_length: int) -> "HRTF":
        """Set the HRTF FFT length and recompute TF from the current IR.

        Parameters
        ----------
        new_fft_length : int
            FFT size used for IR-to-TF conversion.

        Returns
        -------
        HRTF
            A new HRTF instance with updated FFT length and recomputed TF data.

        Use Cases
        ---------
        - Adjust spectral resolution for analysis or interpolation pipelines.

        Examples
        --------
        Increase FFT resolution before inspecting the magnitude response:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> dense_tf = hrtf.transform.modify_fft_length(1024)
        >>> dense_tf.TF.values.shape[-1]
        513
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

    def modify_source_coordinate_system(
        self,
        coordinate_system: str,
    ) -> "HRTF":
        """Update the target source coordinate system.

        Parameters
        ----------
        coordinate_system : {"spherical", "cartesian", "lateral-polar"}
            Target coordinate system used by ``Sources`` when positions are read.

        Returns
        -------
        HRTF
            A new HRTF instance with updated ``Sources.source_coordinate_system``.

        Use Cases
        ---------
        - Switch source representation without modifying SOFA-stored coordinates.
        - Prepare source grids for coordinate-specific analysis workflows.

        Examples
        --------
        Switch the source grid to cartesian coordinates before plotting it:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> cartesian = hrtf.transform.modify_source_coordinate_system("cartesian")
        >>> cartesian.plot_source_grid(show=False)
        """
        coordinate_system = str(coordinate_system).strip().lower()
        allowed_coordinate_systems = {"spherical", "cartesian", "lateral-polar"}
        if coordinate_system not in allowed_coordinate_systems:
            raise ValueError(
                "coordinate_system must be one of: spherical, cartesian, lateral-polar"
            )

        transformed_hrtf = self._hrtf.clone()
        transformed_hrtf.Sources.source_coordinate_system = coordinate_system
        transformed_hrtf._transformed = True
        return transformed_hrtf

    def add_itd(
        self,
        itd: float,
        unit: str = "samples",
    ) -> "HRTF":
        """Add a fixed ITD to the current IR values and resync TF.

        Parameters
        ----------
        itd : float
            ITD value to apply. Positive values delay the left ear and
            negative values delay the right ear.
        unit : {"seconds", "samples"}, default="seconds"
            Unit used by ``itd``.

        Returns
        -------
        HRTF
            A new HRTF instance with ITD-modified IR values and refreshed TF data.

        Use Cases
        ---------
        - Introduce controlled binaural delay for experiments.
        - Simulate additional interaural timing offset.

        Examples
        --------
        Add a fixed ITD to one front-facing direction:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> delayed = hrtf.transform.add_itd(4, unit="samples")
        >>> delayed.plot_amplitude(positions="front", show=False)
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
        if unit_key not in {"seconds", "samples"}:
            raise ValueError("unit must be one of: seconds, samples")

        if isinstance(itd, bool):
            raise ValueError("itd must be a finite value")
        itd_value = float(itd)
        if not np.isfinite(itd_value):
            raise ValueError("itd must be a finite value")

        if unit_key == "seconds":
            if ir.sample_rate is None:
                raise ValueError("IR sample_rate is required when unit='seconds'")
            itd_samples = int(np.round(itd_value * float(ir.sample_rate)))
        else:
            itd_samples = int(np.round(itd_value))

        if itd_samples == 0:
            tf_from_ir(
                ir,
                fft_length=transformed_hrtf.fft_length,
            )
            transformed_hrtf._transformed = True
            return transformed_hrtf

        ir_values = np.asarray(ir.values, dtype=float)
        sample_count = ir_values.shape[-1]
        delay = abs(itd_samples)
        if delay >= sample_count:
            raise ValueError("Absolute ITD in samples must be smaller than IR length")

        if itd_samples > 0:
            left = np.array(ir_values[..., 0, :], copy=True)
            delayed_left = np.zeros_like(left)
            delayed_left[..., delay:] = left[..., :-delay]
            ir_values[..., 0, :] = delayed_left
        else:
            right = np.array(ir_values[..., 1, :], copy=True)
            delayed_right = np.zeros_like(right)
            delayed_right[..., delay:] = right[..., :-delay]
            ir_values[..., 1, :] = delayed_right

        ir.values = ir_values
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

        The ITD sign convention follows ``itd``: positive ITD means
        left-ear delay relative to right-ear and negative ITD means right-ear
        delay relative to left-ear.

        Parameters
        ----------
        method : {"threshold", "maxiacce"}, default="threshold"
            ITD estimator used to compute the delay per position.
        thresh_level : float, default=-10.0
            Threshold offset in dB used when ``method="threshold"``.
        upper_cut_freq : float, default=3000.0
            Low-pass cutoff in Hz applied before ITD estimation.
        filter_order : int, default=10
            Butterworth low-pass filter order used before ITD estimation.

        Returns
        -------
        HRTF
            A new HRTF instance with ITD-compensated IR values and refreshed
            TF data. Compensation is applied per source position.

        Use Cases
        ---------
        - Align binaural arrival times while avoiding additional latency.
        - Remove measured interaural delay before comparative analysis.
        - Standardize onset alignment before feature extraction or metric computation.

        Examples
        --------
        Remove measured ITD before plotting the horizontal trend:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> aligned = hrtf.transform.delete_itd(method="threshold")
        >>> aligned.plot_itd_curve(show=False)
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
            ir,
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
