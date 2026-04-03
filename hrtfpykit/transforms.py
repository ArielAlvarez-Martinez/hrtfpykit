import numpy as np
from scipy.special import sph_harm
import math
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from .dsp import (
    apply_fir_filter,
    apply_iir_filter,
    apply_padding,
    apply_window,
    calculate_itd,
    calculate_ir_from_tf,
    calculate_tf_from_ir,
    downsampling,
    minimum_phase,
    modify_magnitude,
    modify_phase,
    upsampling,
)

if TYPE_CHECKING:
    from .hrtf import HRTF


class Transform:
    """HRTF transform operations."""

    def __init__(self, hrtf: "HRTF") -> None:
        self._hrtf = hrtf

    def apply_window(self, window_name: str) -> "HRTF":
        """General Description:
        Apply a time-domain window to IR values and resync TF.

        Parameters:
        - window_name: Window identifier (for example hann, hamming, blackman).

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Reduce spectral leakage before FFT conversion.

        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        windowed = apply_window(ir, window_name)
        if windowed is None:
            raise ValueError(f"Unsupported window '{window_name}'")
        ir.values = windowed
        calculate_tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        return transformed_hrtf

    def apply_padding(
        self,
        padding_length: int,
        location: str = "end",
        value: float | complex = 0,
        domain: str = "time",
    ) -> "HRTF":
        """General Description:
        Pad IR or TF values and resync the paired domain.

        Parameters:
        - padding_length: Number of samples or bins to add.
        - location: Padding side, start or end.
        - value: Constant pad value.
        - domain: Domain to apply padding, time or frequency.

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Increase IR length before FFT-based workflows.
        - Extend TF length for FFT-size exploration.

        """
        transformed_hrtf = self._hrtf.clone()
        domain_key = str(domain).strip().lower()
        if domain_key == "time":
            ir = transformed_hrtf.IR
            ir.values = apply_padding(
                ir,
                padding_length=padding_length,
                location=location,
                value=value,
            )
            calculate_tf_from_ir(
                ir,
                fft_length=transformed_hrtf.fft_length,
            )
            return transformed_hrtf

        if domain_key == "frequency":
            tf = transformed_hrtf.TF
            tf_values = apply_padding(
                tf,
                padding_length=padding_length,
                location=location,
                value=value,
            )
            frequency_bins = tf.frequency_bins
            if frequency_bins is None:
                raise ValueError("TF frequency bins are required for TF padding")
            frequency_bins = np.asarray(frequency_bins, dtype=float)
            if frequency_bins.ndim != 1 or frequency_bins.size < 2:
                raise ValueError("TF frequency bins must be 1D with at least two values")
            diffs = np.diff(frequency_bins)
            step = float(diffs[0])
            if not np.allclose(diffs, step, rtol=1e-5, atol=1e-8):
                raise ValueError("TF frequency bins must be uniformly spaced for TF padding")
            location_key = str(location).strip().lower()
            if location_key == "start":
                new_bins = frequency_bins[0] - step * np.arange(padding_length, 0, -1)
                tf.frequency_bins = np.concatenate((new_bins, frequency_bins))
            elif location_key == "end":
                new_bins = frequency_bins[-1] + step * np.arange(1, padding_length + 1)
                tf.frequency_bins = np.concatenate((frequency_bins, new_bins))
            else:
                raise ValueError("Padding location must be 'start' or 'end'")
            tf.values = tf_values
            calculate_ir_from_tf(
                tf,
                frequency_bins=tf.frequency_bins,
            )
            return transformed_hrtf

        raise ValueError("domain must be 'time' or 'frequency'")

    def apply_fir_filter(
        self,
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        num_taps: int = 101,
        window: str | None = None,
    ) -> "HRTF":
        """General Description:
        Apply FIR filtering on IR values and resync TF.

        Parameters:
        - filter: Filter type (lowpass, highpass, bandpass aliases supported).
        - cutoff: Cutoff frequency or cutoff pair for bandpass.
        - num_taps: FIR filter length.
        - window: Optional FIR design window.

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Remove undesired frequency content from HRIR data.
        - Isolate a band before feature extraction.

        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = apply_fir_filter(
            ir,
            filter=filter,
            sample_rate=ir.sample_rate,
            cutoff=cutoff,
            num_taps=num_taps,
            window=window,
        )
        calculate_tf_from_ir(
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
        """General Description:
        Apply IIR filtering on IR values and resync TF.

        Parameters:
        - filter: Filter type (lowpass, highpass, bandpass aliases supported).
        - cutoff: Cutoff frequency or cutoff pair for bandpass.
        - order: IIR Butterworth filter order.

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Reproduce legacy IIR-based preprocessing chains.
        - Apply low-latency recursive filtering before analysis.

        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = apply_iir_filter(
            ir,
            filter=filter,
            sample_rate=ir.sample_rate,
            cutoff=cutoff,
            order=order,
        )
        calculate_tf_from_ir(
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
        """General Description:
        Convert IR values to minimum-phase and resync TF.

        Parameters:
        - method: Minimum-phase method key.
        - fft_length: Optional FFT length used in cepstral reconstruction.
        - epsilon: Small positive floor for log-magnitude stability.

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Build minimum-phase HRIR approximations for reduced-latency processing.
        - Standardize phase behavior across datasets before analysis.

        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        ir.values = minimum_phase(
            ir,
            method=method,
            fft_length=fft_length,
            epsilon=epsilon,
        )
        calculate_tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        return transformed_hrtf

    def modify_phase(
        self,
        new_phase: np.ndarray,
        unit: str = "degrees",
    ) -> "HRTF":
        """General Description:
        Replace TF phase values and rebuild IR.

        Parameters:
        - new_phase: Phase array matching TF shape.
        - unit: Phase unit (`degrees` or `radians`).

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Apply target phase profiles while preserving measured magnitudes.
        - Build controlled phase perturbation experiments.

        """
        transformed_hrtf = self._hrtf.clone()
        tf = transformed_hrtf.TF
        tf.values = modify_phase(
            tf,
            new_phase=new_phase,
            unit=unit,
        )
        calculate_ir_from_tf(
            tf,
            frequency_bins=tf.frequency_bins,
        )
        return transformed_hrtf

    def modify_magnitude(
        self,
        new_magnitude: np.ndarray,
        scale: str = "linear",
    ) -> "HRTF":
        """General Description:
        Replace TF magnitude values and rebuild IR.

        Parameters:
        - new_magnitude: Magnitude array matching TF shape.
        - scale: Magnitude scale (`linear`, `lineal`, or `db`).

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Apply target magnitude responses while preserving phase.
        - Evaluate perceptual effects of magnitude-only modifications.

        """
        transformed_hrtf = self._hrtf.clone()
        tf = transformed_hrtf.TF
        tf.values = modify_magnitude(
            tf,
            new_magnitude=new_magnitude,
            scale=scale,
        )
        calculate_ir_from_tf(
            tf,
            frequency_bins=tf.frequency_bins,
        )
        return transformed_hrtf

    def upsampling(self, new_sample_rate: float) -> "HRTF":
        """General Description:
        Upsample IR values to a higher sample rate and resync TF.

        Parameters:
        - new_sample_rate: Target sample rate in Hz, higher than current IR sample rate.

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Increase temporal resolution for analysis or rendering.

        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        resampled_ir, resampled_sample_rate = upsampling(
            ir,
            new_sample_rate=new_sample_rate,
        )
        ir.values = resampled_ir
        ir.sample_rate = resampled_sample_rate
        calculate_tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        return transformed_hrtf

    def downsampling(self, new_sample_rate: float) -> "HRTF":
        """General Description:
        Downsample IR values to a lower sample rate and resync TF.

        Parameters:
        - new_sample_rate: Target sample rate in Hz, lower than current IR sample rate.

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Reduce processing and storage footprint.
        - Match external systems that require lower sample rates.

        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        resampled_ir, resampled_sample_rate = downsampling(
            ir,
            new_sample_rate=new_sample_rate,
        )
        ir.values = resampled_ir
        ir.sample_rate = resampled_sample_rate
        calculate_tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        return transformed_hrtf

    def modify_fft_length(self, new_fft_length: int) -> "HRTF":
        """General Description:
        Set HRTF FFT length and recompute TF from current IR.

        Parameters:
        - new_fft_length: FFT size used for IR-to-TF conversion.

        Returns:
        - A new HRTF instance with transformed IR/TF values.

        Use Cases:
        - Adjust spectral resolution for analysis or interpolation pipelines.

        """
        transformed_hrtf = self._hrtf.clone()
        ir = transformed_hrtf.IR
        if ir.values is None:
            raise ValueError("IR data is not available")
        transformed_hrtf.fft_length = int(new_fft_length)
        calculate_tf_from_ir(
            ir,
            fft_length=transformed_hrtf.fft_length,
        )
        return transformed_hrtf

    def modify_source_coordinate_system(
        self,
        coordinate_system: str,
    ) -> "HRTF":
        """General Description:
        Update Sources target coordinate system and refresh source positions.

        Parameters:
        - coordinate_system: Target source coordinate system (`spherical`, `cartesian`, or `lateral-polar`).

        Returns:
        - A new HRTF instance with updated `Sources.source_coordinate_system`.

        Use Cases:
        - Switch source representation without modifying underlying SOFA stored coordinates.
        - Prepare source grids for coordinate-specific analysis workflows.

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
        """General Description:
        Add a fixed ITD to current IR values and resync TF.

        Parameters:
        - itd: ITD value to apply. Positive delays left ear; negative delays right ear.
        - unit: ITD unit (`seconds` or `samples`).

        Returns:
        - A new HRTF instance with ITD-modified IR/TF values.
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
            calculate_tf_from_ir(
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
        calculate_tf_from_ir(
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
        """General Description:
        Estimate and remove ITD from current IR values by deducting delay on the delayed ear, then resync TF.
        The ITD sign convention follows `calculate_itd`: positive ITD means left-ear delay relative to right-ear
        (`ear axis: 0=left, 1=right`), negative ITD means right-ear delay relative to left-ear.

        Parameters:
        - method: ITD estimator (`threshold` or `maxiacce`) used to compute per-position ITD.
        - thresh_level: Threshold offset in dB for `threshold` mode.
        - upper_cut_freq: Low-pass cutoff in Hz applied before ITD estimation.
        - filter_order: Positive IIR Butterworth order for low-pass preprocessing.

        Returns:
        - A new HRTF instance with ITD-compensated IR/TF values.
        - Compensation is performed per IR position.

        Use Cases:
        - Align binaural arrival times while avoiding additional latency.
        - Remove measured interaural delay before comparative analysis.
        - Standardize onset alignment before ML feature extraction or metric computation.

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
        calculate_tf_from_ir(
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
