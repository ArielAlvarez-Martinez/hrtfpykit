from __future__ import annotations

import numpy as np
from scipy.special import sph_harm
import math
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from .dsp import (
    convolve,
    deconvolve,
    downsampling,
    fir_filter,
    iir_filter,
    ir_from_tf,
    minimum_phase,
    modify_magnitude,
    modify_phase,
    padding,
    tf_from_ir,
    upsampling,
    window,
)
from .metrics import calculate_itd
from .domain import IR, TF

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
        >>> transformed = hrtf.transform.apply_window("hann")
        >>> transformed = hrtf.transform.apply_window("blackman")
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = window(ir, window_name)
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
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
        >>> transformed = hrtf.transform.apply_padding(32)
        >>> transformed = hrtf.transform.apply_padding(16, location="start", value=0.0)
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
        >>> transformed = hrtf.transform.apply_fir_filter("lowpass", cutoff=3000.0)
        >>> transformed = hrtf.transform.apply_fir_filter("bandpass", cutoff=(1000.0, 8000.0))
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
        >>> transformed = hrtf.transform.apply_iir_filter("lowpass", cutoff=3000.0)
        >>> transformed = hrtf.transform.apply_iir_filter("bandpass", cutoff=(1000.0, 8000.0), order=4)
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
        return transformed_hrtf

    def convolve(
        self,
        ir_2: np.ndarray | "IR" | "HRTF",
        mode: str = "same",
        method: str = "auto",
    ) -> "HRTF":
        """Convolve the current IR with another IR and resync TF.

        Parameters
        ----------
        ir_2 : np.ndarray | IR | HRTF
            Second impulse response used in the convolution. When an ``HRTF``
            instance is provided, its ``IR`` domain is used.
        mode : {"full", "same", "valid"}, default="same"
            Convolution output mode passed to the DSP layer. The default keeps
            the current IR length.
        method : {"auto", "direct", "fft"}, default="auto"
            Convolution method passed to the DSP layer.

        Returns
        -------
        HRTF
            A new HRTF instance with convolved IR values and refreshed TF data.

        Use Cases
        ---------
        - Cascade the current HRIR with another measured or designed IR.
        - Apply one IR to every source and ear through broadcasting.
        - Use another HRTF object's IR data as the convolution target.

        Examples
        --------
        >>> transformed = hrtf.transform.convolve(np.array([1.0, 0.5, 0.0]))
        >>> transformed = hrtf.transform.convolve(other_hrtf)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        if ir.values is None:
            raise ValueError("IR data is not available")

        if isinstance(ir_2, self._hrtf.__class__):
            ir_2_values = ir_2.IR
        else:
            ir_2_values = ir_2

        ir.values = convolve(
            ir_1=ir,
            ir_2=ir_2_values,
            mode=mode,
            method=method,
        )

        if transformed_hrtf.fft_length is not None:
            transformed_hrtf.fft_length = max(
                int(transformed_hrtf.fft_length),
                int(ir.values.shape[-1]),
            )

        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        return transformed_hrtf

    def compensate(
        self,
        ir_2: np.ndarray | "IR" | "HRTF",
        fft_length: int | None = None,
        output_length: int | None = None,
        regularization: float = 1e-8,
    ) -> "HRTF":
        """Remove an IR from the current IR through regularized deconvolution.

        Parameters
        ----------
        ir_2 : np.ndarray | IR | HRTF
            IR to remove from the current IR. When an ``HRTF`` instance is
            provided, its ``IR`` domain is used.
        fft_length : int | None, default=None
            FFT length used by the DSP deconvolution step.
        output_length : int | None, default=None
            Number of samples returned in the compensated IR. When omitted, the
            current IR length is used.
        regularization : float, default=1e-8
            Positive stabilization value passed to the DSP deconvolution step.

        Returns
        -------
        HRTF
            A new HRTF instance with compensated IR values and refreshed TF
            data.

        Use Cases
        ---------
        - Remove a known room or device IR from the current HRIR.
        - Use another HRTF object's IR data as the compensation reference.

        Examples
        --------
        >>> transformed = hrtf.transform.compensate(np.array([1.0, 0.5, 0.0]))
        >>> transformed = hrtf.transform.compensate(other_hrtf, output_length=256)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        if ir.values is None:
            raise ValueError("IR data is not available")

        if isinstance(ir_2, self._hrtf.__class__):
            ir_2_values = ir_2.IR
        else:
            ir_2_values = ir_2

        if output_length is None:
            output_length = int(ir.values.shape[-1])

        ir.values = deconvolve(
            ir_1=ir,
            ir_2=ir_2_values,
            fft_length=fft_length,
            output_length=output_length,
            regularization=regularization,
        )

        if transformed_hrtf.fft_length is not None:
            transformed_hrtf.fft_length = max(
                int(transformed_hrtf.fft_length),
                int(ir.values.shape[-1]),
            )

        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
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
        >>> transformed = hrtf.transform.minimum_phase()
        >>> transformed = hrtf.transform.minimum_phase(method="hilbert", fft_length=1024)
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
        >>> new_ir = np.zeros_like(hrtf.IR.values)
        >>> transformed = hrtf.transform.modify_ir(new_ir)
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
        >>> phase = np.zeros_like(hrtf.TF.phase)
        >>> transformed = hrtf.transform.modify_phase(phase, unit="degrees")
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
        >>> new_tf = np.ones_like(hrtf.TF.values, dtype=complex)
        >>> transformed = hrtf.transform.modify_tf(new_tf)
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
                    sample_rate = step * frequency_bins.size
                    tf.frequency_bins = np.fft.fftshift(
                        np.fft.fftfreq(
                            tf.values.shape[-1],
                            d=1.0 / sample_rate,
                        )
                    )
                else:
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
        scale : {"linear", "lineal", "db"}, default="linear"
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
        >>> magnitude = np.ones_like(hrtf.TF.magnitude)
        >>> transformed = hrtf.transform.modify_magnitude(magnitude, scale="linear")
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
        return transformed_hrtf

    def upsampling(self, new_sample_rate: float) -> "HRTF":
        """Upsample IR values to a higher sample rate and resync TF.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in Hz. It must be higher than the current IR
            sample rate.

        Returns
        -------
        HRTF
            A new HRTF instance with resampled IR values and refreshed TF data.

        Use Cases
        ---------
        - Increase temporal resolution for analysis or rendering.

        Examples
        --------
        >>> transformed = hrtf.transform.upsampling(96000.0)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        resampled_ir, resampled_sample_rate = upsampling(
            ir,
            new_sample_rate=new_sample_rate,
        )
        ir.values = resampled_ir
        ir.sample_rate = resampled_sample_rate
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        return transformed_hrtf

    def downsampling(self, new_sample_rate: float) -> "HRTF":
        """Downsample IR values to a lower sample rate and resync TF.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in Hz. It must be lower than the current IR
            sample rate.

        Returns
        -------
        HRTF
            A new HRTF instance with resampled IR values and refreshed TF data.

        Use Cases
        ---------
        - Reduce processing and storage footprint.
        - Match external systems that require lower sample rates.

        Examples
        --------
        >>> transformed = hrtf.transform.downsampling(24000.0)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        resampled_ir, resampled_sample_rate = downsampling(
            ir,
            new_sample_rate=new_sample_rate,
        )
        ir.values = resampled_ir
        ir.sample_rate = resampled_sample_rate
        tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
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
        >>> transformed = hrtf.transform.modify_fft_length(1024)
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
        >>> transformed = hrtf.transform.modify_source_coordinate_system("cartesian")
        >>> transformed = hrtf.transform.modify_source_coordinate_system("lateral-polar")
        """
        coordinate_system = str(coordinate_system).strip().lower()
        allowed_coordinate_systems = {"spherical", "cartesian", "lateral-polar"}
        if coordinate_system not in allowed_coordinate_systems:
            raise ValueError(
                "coordinate_system must be one of: spherical, cartesian, lateral-polar"
            )

        transformed_hrtf = self._hrtf.clone()
        transformed_hrtf.Sources.source_coordinate_system = coordinate_system
        return transformed_hrtf

    def add_itd(
        self,
        itd: float,
        unit: str = "seconds",
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
        >>> transformed = hrtf.transform.add_itd(0.0002, unit="seconds")
        >>> transformed = hrtf.transform.add_itd(4, unit="samples")
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
        return transformed_hrtf

    def delete_itd(
        self,
        method: str = "threshold",
        thresh_level: float = -10.0,
        upper_cut_freq: float = 3000.0,
        filter_order: int = 10,
    ) -> "HRTF":
        """Estimate and remove ITD from the current IR values, then resync TF.

        The ITD sign convention follows ``calculate_itd``: positive ITD means
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
        >>> transformed = hrtf.transform.delete_itd()
        >>> transformed = hrtf.transform.delete_itd(method="maxiacce", upper_cut_freq=1500.0)
        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        if ir.values is None:
            raise ValueError("IR data is not available")
        if ir.values.ndim < 2:
            raise ValueError("IR data must include ear and time axes")
        if ir.values.shape[-2] < 2:
            raise ValueError("IR ear axis must contain at least two channels (0=left, 1=right)")

        itd_samples = calculate_itd(
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
        return transformed_hrtf

    
'''

Spherical Harmonic Transformation

'''
def _factorial(n):
    if n < 0:
        raise ValueError("Factorial of negative number")
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result

def _double_factorial(n):
    if n <= 0:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result

def _P_mm(m, x):
    return ((-1) ** m) * _double_factorial(2 * m - 1) * (1 - x**2) ** (m / 2)

def _P_m1_m(m, x):
    return x * (2 * m + 1) * _P_mm(m, x)

def _assoc_legendre(n, m, x):
    if m < 0 or m > n:
        raise ValueError("Require 0 <= m <= n")

    if n == m:
        return _P_mm(m, x)

    if n == m + 1:
        return _P_m1_m(m, x)

    P_nm2 = _P_mm(m, x)
    P_nm1 = _P_m1_m(m, x)

    for ell in range(m + 2, n + 1):
        P_n = (
            ((2 * ell - 1) * x * P_nm1 - (ell + m - 1) * P_nm2)
            / (ell - m)
        )
        P_nm2, P_nm1 = P_nm1, P_n

    return P_nm1

def _sh_normalization(n, m):
    m = abs(m)
    return math.sqrt(
        (2 * n + 1) / (4 * math.pi)
        * _factorial(n - m) / _factorial(n + m)
    )

def _sph_harm_manual(m, n, phi, theta):
    x = np.cos(theta)
    m_abs = abs(m)

    P = _assoc_legendre(n, m_abs, x)
    N = _sh_normalization(n, m_abs)

    Y = N * P * np.exp(1j * m * phi)

    # Handle negative m (Condon–Shortley phase)
    if m < 0:
        Y = ((-1) ** m_abs) * np.conj(Y)

    return Y

def _sh_vector_from_scratch(sh_order, az, el):
    theta = np.pi / 2.0 - el   # elevation -> colatitude
    phi = az

    y = []

    for n in range(sh_order + 1):
        for m in range(-n, n + 1):
            Y = _sph_harm_manual(m, n, phi, theta)

            if m < 0:
                y.append(np.sqrt(2) * (-1)**m * Y.imag)
            elif m == 0:
                y.append(Y.real)
            else:
                y.append(np.sqrt(2) * (-1)**m * Y.real)

    return np.asarray(y, dtype=float)

def sht_core_from_scratch(f, dirs, sh_order, epsilon=1e-6):
    """
    Compute the spherical harmonics (SH) decomposition of a spatially sampled
    function using a fully manual SH basis.

    This function projects a discrete spatial function f(azimuth, elevation)
    onto a real spherical harmonics basis up to a given order using a
    regularized least-squares formulation.

    Parameters
    ----------
    f : array_like, shape (N,)
        Sampled values of the spatial function to be decomposed.
        Each element corresponds to one spatial direction defined in `dirs`.
        In the context of HRTFs, this typically represents the HRTF magnitude
        at a fixed frequency across N spatial directions.

    dirs : array_like, shape (N, 2)
        Spatial directions associated with `f`.
        Each row must be of the form:
            [azimuth, elevation]
        with angles expressed in radians.
        The azimuth is assumed to lie in [0, 2π) and elevation in [-π/2, π/2].

    sh_order : int
        Maximum spherical harmonics order used for the decomposition.
        The total number of SH coefficients is:
            (sh_order + 1)^2

    epsilon : float, optional
        Tikhonov regularization parameter used to stabilize the least-squares
        inversion. This term improves numerical robustness in the presence of
        noisy measurements or non-uniform spatial sampling.
        Default is 1e-6.

    Returns
    -------
    C : ndarray, shape ((sh_order + 1)^2,)
        Vector of real spherical harmonics coefficients corresponding to the
        SH decomposition of `f`. The coefficients are ordered according to the
        standard (n, m) indexing, with increasing order n and degree m.

    f_recons : ndarray, shape (N,)
        Reconstructed approximation of the input function obtained by
        evaluating the SH expansion at the original spatial directions.
        This can be used to assess reconstruction accuracy.

    Y : ndarray, shape (N, (sh_order + 1)^2)
        Spherical harmonics basis matrix evaluated at the input directions.
        Each row corresponds to one spatial direction, and each column
        corresponds to one SH basis function.

    Notes
    -----
    The SH coefficients are obtained by solving the regularized normal equations:

        f ≈ Y C
        C = (Yᵀ Y + ε I)⁻¹ Yᵀ f

    where Y is the SH basis matrix and ε is the regularization parameter.

    This implementation uses a fully manual computation of the spherical
    harmonics basis, without relying on external special-function libraries.
    """

    f = np.asarray(f).reshape(-1)
    dirs = np.asarray(dirs)

    N = f.size
    n_coeffs = (sh_order + 1) ** 2

    Y = np.zeros((N, n_coeffs))

    for i in range(N):
        az, el = dirs[i]
        Y[i, :] = _sh_vector_from_scratch(sh_order, az, el)

    A = Y.T @ Y + epsilon * np.eye(n_coeffs)
    b = Y.T @ f

    C = np.linalg.solve(A, b)
    f_recons = Y @ C

    return C, f_recons, Y

def _sh_vector(sh_order, az, el):
    

    theta = np.pi / 2.0 - el   # elevation -> colatitude
    phi = az

    y = []
    for n in range(sh_order + 1):
        for m in range(-n, n + 1):
            Y = sph_harm(m, n, phi, theta)

            if m < 0:
                y.append(np.sqrt(2) * (-1)**m * Y.imag)
            elif m == 0:
                y.append(Y.real)
            else:
                y.append(np.sqrt(2) * (-1)**m * Y.real)

    return np.asarray(y, dtype=float)

def sht_core(f, dirs, sh_order, epsilon=1e-6):
    """
    Compute the spherical harmonics (SH) decomposition of a spatially sampled
    function using a predefined spherical harmonics basis.

    This function projects a discrete spatial function f(azimuth, elevation)
    onto a real spherical harmonics basis up to a given order using a
    regularized least-squares formulation. The SH basis functions are generated
    by the helper function `_sh_vector`.

    Parameters
    ----------
    f : array_like, shape (N,)
        Sampled values of the spatial function to be decomposed.
        Each element corresponds to one spatial direction defined in `dirs`.
        In the context of HRTFs, this typically represents the HRTF magnitude
        at a fixed frequency across N spatial directions.

    dirs : array_like, shape (N, 2)
        Spatial directions associated with `f`.
        Each row must be of the form:
            [azimuth, elevation]
        with angles expressed in radians.
        The azimuth is assumed to lie in [0, 2π) and elevation in [-π/2, π/2].

    sh_order : int
        Maximum spherical harmonics order used for the decomposition.
        The total number of SH coefficients is:
            (sh_order + 1)^2

    epsilon : float, optional
        Tikhonov regularization parameter used to stabilize the least-squares
        inversion. This term improves numerical robustness in the presence of
        noisy measurements or non-uniform spatial sampling.
        Default is 1e-6.

    Returns
    -------
    C : ndarray, shape ((sh_order + 1)^2,)
        Vector of real spherical harmonics coefficients corresponding to the
        SH decomposition of `f`. The coefficients are ordered according to the
        standard (n, m) indexing, with increasing order n and degree m.

    f_recons : ndarray, shape (N,)
        Reconstructed approximation of the input function obtained by
        evaluating the SH expansion at the original spatial directions.
        This can be used to assess reconstruction accuracy.

    Y : ndarray, shape (N, (sh_order + 1)^2)
        Spherical harmonics basis matrix evaluated at the input directions.
        Each row corresponds to one spatial direction, and each column
        corresponds to one SH basis function.
    
    """

    f = np.asarray(f).reshape(-1)
    dirs = np.asarray(dirs)

    N = f.size
    n_coeffs = (sh_order + 1) ** 2

    Y = np.zeros((N, n_coeffs))

    for i in range(N):
        az, el = dirs[i]
        Y[i, :] = _sh_vector(sh_order, az, el)

    A = Y.T @ Y + epsilon * np.eye(n_coeffs)
    b = Y.T @ f

    C = np.linalg.solve(A, b)
    f_recons = Y @ C

    return C, f_recons, Y

'''

HRTF magnitude reconstruction for one frequency bin

'''

def sht_reconstruction(N, dirs, sh_order, C_matrix):
    """
    Reconstruct a spatial function (e.g. HRTF magnitude) from spherical
    harmonics coefficients.

    This function evaluates the spherical harmonics expansion at a given
    set of spatial directions using precomputed SH coefficients.

    Parameters
    ----------
    N : int
        Number of spatial positions (directions) at which the function
        will be reconstructed.

    dirs : array_like, shape (N, 2)
        Spatial directions at which the function is reconstructed.
        Each row must be:
            [azimuth, elevation]
        with angles expressed in radians.

    sh_order : int
        Maximum spherical harmonics order used for the reconstruction.
        The number of coefficients must be:
            (sh_order + 1)^2

    C_matrix : array_like, shape ((sh_order + 1)^2,)
        Vector of spherical harmonics coefficients.

    Returns
    -------
    f_reconstructed : ndarray, shape (N,)
        Reconstructed spatial function evaluated at the input directions.
        In the HRTF context, this corresponds to the reconstructed HRTF
        magnitude at the given spatial positions.

    Notes
    -----
    The reconstruction is performed as:

        f̂ = Y C

    where Y is the spherical harmonics basis matrix evaluated at `dirs`
    and C contains the SH coefficients.
    """
    dirs = np.asarray(dirs)
    C_matrix = np.asarray(C_matrix).reshape(-1)

    n_coeffs = (sh_order + 1) ** 2

    # To be sure that the HRTFs are compatible aka "they were measured using the same spatial setup/grid"
    if C_matrix.size != n_coeffs:
        raise ValueError(
            "C_matrix has incompatible size. "
            f"Expected {n_coeffs}, got {C_matrix.size}."
        )

    Y = np.zeros((N, n_coeffs))

    for i in range(N):
        az, el = dirs[i]
        Y[i, :] = _sh_vector(sh_order, az, el)

    f_reconstructed = Y @ C_matrix

    return f_reconstructed

'''

Spherical Harmonic Transformation : visualization, comparison and validation

'''

def reconstruction_error(f, f_recons):

    f = np.asarray(f).reshape(-1)
    f_recons = np.asarray(f_recons).reshape(-1)

    diff = f - f_recons

    abs_err = np.linalg.norm(diff)
    rel_err = abs_err / np.linalg.norm(f)
    rms_err = np.sqrt(np.mean(diff**2))

    return abs_err, rel_err, rms_err


def print_validation_report(f, f_recons, label=""):
    abs_err, rel_err, rms_err = reconstruction_error(f, f_recons)

    print("---- Validation Report", label, "----")
    print(f"Absolute error  : {abs_err:.8f}")
    print(f"Relative error  : {rel_err:.8f}")
    print(f"RMS error       : {rms_err:.8f}")
    print(f"Max |diff|      : {np.max(np.abs(f - f_recons)):.8f}")
    print("-----------------------------------")

def plot_sht_reconstruction_comparison(f, f_recons, label=""):
    """
    Plot original signal and reconstructed signal, and display reconstruction errors.

    Parameters
    ----------
    f : array_like, shape (N,)
        Original signal.
    f_recons : array_like, shape (N,)
        Reconstructed signal.
    label : str, optional
        Label to identify the comparison (e.g. frequency, SH order).
    """

    f = np.asarray(f).reshape(-1)
    f_recons = np.asarray(f_recons).reshape(-1)

    if f.shape != f_recons.shape:
        raise ValueError("f and f_recons must have the same shape.")

    # --- error metrics ---
    abs_err, rel_err, rms_err = reconstruction_error(f, f_recons)
    diff = f - f_recons

    # --- plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Signal comparison
    axes[0].plot(f, label="Original", linewidth=2)
    axes[0].plot(f_recons, '--', label="Reconstructed", linewidth=2)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"SHT Reconstruction Comparison {label}")
    axes[0].grid(linestyle='--')
    axes[0].legend()

    # Error plot
    axes[1].plot(diff, color='red', label="Error (f - f̂)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Error")
    axes[1].grid(linestyle='--')

    # Error text box
    textstr = (
        f"Absolute L2 error : {abs_err:.6f}\n"
        f"Relative L2 error : {rel_err:.6f}\n"
        f"RMS error         : {rms_err:.6f}"
    )

    axes[1].text(
        0.02, 0.95, textstr,
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    plt.tight_layout()
    plt.show()
