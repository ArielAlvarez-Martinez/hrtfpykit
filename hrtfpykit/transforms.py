import numpy as np
from scipy.special import sph_harm
import math
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from .dsp import (
    apply_filter as dsp_apply_filter,
    apply_ir_crop as dsp_apply_ir_crop,
    apply_padding as dsp_apply_padding,
    apply_tf_crop as dsp_apply_tf_crop,
    apply_window as dsp_apply_window,
    calculate_ir_from_tf,
    calculate_tf_from_ir,
    downsampling as dsp_downsampling,
    upsampling as dsp_upsampling,
)

if TYPE_CHECKING:
    from .domain import IR, TF
    from .spatial import Sources


class _Transform:
    def __init__(self, domain: "IR | TF | Sources") -> None:
        self._domain = domain
        self._hrtf = domain._hrtf


class TransformIR(_Transform):
    """IR-domain transform operations."""

    def __init__(self, ir: "IR") -> None:
        super().__init__(ir)

    def apply_ir_crop(
        self,
        start: int | None = None,
        end: int | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> None:
        """General Description:
        Crop IR values in time by sample indices or seconds and resync TF.

        Parameters:
        - start: Start sample index (inclusive).
        - end: End sample index (exclusive).
        - start_seconds: Start time in seconds.
        - end_seconds: End time in seconds.

        Returns:
        None.

        Use Cases:
        - Trim early reflections or isolate a time segment.
        - Apply sample-rate-aware cropping using seconds.

        Best Practices:
        - Use either sample indices or seconds in a single call.
        - Let this method handle IR/TF synchronization.
        """
        self._domain.values = dsp_apply_ir_crop(
            self._domain,
            start=start,
            end=end,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        calculate_tf_from_ir(
            self._domain,
            fft_length=self._hrtf.fft_length,
        )

    def apply_window(self, window_name: str) -> None:
        """General Description:
        Apply a time-domain window to IR values and resync TF.

        Parameters:
        - window_name: Window identifier (for example hann, hamming, blackman).

        Returns:
        None.

        Use Cases:
        - Reduce spectral leakage before FFT conversion.

        Best Practices:
        - Use supported window names only.
        - Apply windowing intentionally because it changes signal energy distribution.
        """
        windowed = dsp_apply_window(self._domain, window_name)
        if windowed is None:
            raise ValueError(f"Unsupported window '{window_name}'")
        self._domain.values = windowed
        calculate_tf_from_ir(
            self._domain,
            fft_length=self._hrtf.fft_length,
        )

    def apply_padding(
        self,
        padding_length: int,
        location: str = "end",
        value: float | complex = 0,
    ) -> None:
        """General Description:
        Pad IR values in time domain and resync TF.

        Parameters:
        - padding_length: Number of samples to add.
        - location: Padding side, start or end.
        - value: Constant pad value.

        Returns:
        None.

        Use Cases:
        - Increase IR length before FFT-based workflows.
        - Align impulse responses for downstream comparisons.

        Best Practices:
        - Prefer end padding for most HRIR workflows.
        - Keep padding length explicit for reproducibility.
        """
        self._domain.values = dsp_apply_padding(
            self._domain,
            padding_length=padding_length,
            location=location,
            value=value,
        )
        calculate_tf_from_ir(
            self._domain,
            fft_length=self._hrtf.fft_length,
        )

    def apply_filter(
        self,
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        num_taps: int = 101,
        window: str | None = None,
    ) -> None:
        """General Description:
        Apply FIR filtering on IR values and resync TF.

        Parameters:
        - filter: Filter type (lowpass, highpass, bandpass aliases supported).
        - cutoff: Cutoff frequency or cutoff pair for bandpass.
        - num_taps: FIR filter length.
        - window: Optional FIR design window.

        Returns:
        None.

        Use Cases:
        - Remove undesired frequency content from HRIR data.
        - Isolate a band before feature extraction.

        Best Practices:
        - Use odd `num_taps` for linear-phase behavior.
        - Keep cutoff values inside valid Nyquist limits.
        """
        self._domain.values = dsp_apply_filter(
            self._domain,
            filter=filter,
            sample_rate=self._domain.sample_rate,
            cutoff=cutoff,
            num_taps=num_taps,
            window=window,
        )
        calculate_tf_from_ir(
            self._domain,
            fft_length=self._hrtf.fft_length,
        )

    def upsampling(self, new_sample_rate: float) -> None:
        """General Description:
        Upsample IR values to a higher sample rate and resync TF.

        Parameters:
        - new_sample_rate: Target sample rate in Hz, higher than current IR sample rate.

        Returns:
        None.

        Use Cases:
        - Increase temporal resolution for analysis or rendering.

        Best Practices:
        - Use only when IR values and sample rate are initialized.
        - Centralize resampling here to keep IR and TF synchronized.
        """
        resampled_ir, resampled_sample_rate = dsp_upsampling(
            self._domain,
            new_sample_rate=new_sample_rate,
        )
        self._domain.values = resampled_ir
        self._domain.sample_rate = resampled_sample_rate
        calculate_tf_from_ir(
            self._domain,
            fft_length=self._hrtf.fft_length,
        )

    def downsampling(self, new_sample_rate: float) -> None:
        """General Description:
        Downsample IR values to a lower sample rate and resync TF.

        Parameters:
        - new_sample_rate: Target sample rate in Hz, lower than current IR sample rate.

        Returns:
        None.

        Use Cases:
        - Reduce processing and storage footprint.
        - Match external systems that require lower sample rates.

        Best Practices:
        - Ensure target rate preserves the required frequency bandwidth.
        - Use this method instead of manual resampling to preserve IR/TF consistency.
        """
        resampled_ir, resampled_sample_rate = dsp_downsampling(
            self._domain,
            new_sample_rate=new_sample_rate,
        )
        self._domain.values = resampled_ir
        self._domain.sample_rate = resampled_sample_rate
        calculate_tf_from_ir(
            self._domain,
            fft_length=self._hrtf.fft_length,
        )

    def modify_fft_length(self, new_fft_length: int) -> None:
        """General Description:
        Set HRTF FFT length and recompute TF from current IR.

        Parameters:
        - new_fft_length: FFT size used for IR-to-TF conversion.

        Returns:
        None.

        Use Cases:
        - Adjust spectral resolution for analysis or interpolation pipelines.

        Best Practices:
        - Ensure IR values exist before changing FFT length.
        - Keep FFT-length changes centralized to maintain consistent metadata.
        """
        if self._domain.values is None:
            raise ValueError("IR data is not available")
        self._hrtf.fft_length = int(new_fft_length)
        calculate_tf_from_ir(
            self._domain,
            fft_length=self._hrtf.fft_length,
        )


class TransformTF(_Transform):
    def __init__(self, tf: "TF") -> None:
        super().__init__(tf)

    def apply_padding(
        self,
        padding_length: int,
        location: str = "end",
        value: float | complex = 0,
    ) -> None:
        """General Description:
        Pad TF values in frequency domain and rebuild IR.

        Parameters:
        - padding_length: Number of bins to add.
        - location: Padding side, start or end.
        - value: Constant pad value.

        Returns:
        None.

        Use Cases:
        - Extend TF length for FFT-size exploration.
        - Create controlled spectral-domain zero regions at boundaries.

        Best Practices:
        - Ensure frequency bins are available and uniformly spaced.
        - Use with awareness that IR will be recomputed after TF changes.
        """
        self._domain.values = dsp_apply_padding(
            self._domain,
            padding_length=padding_length,
            location=location,
            value=value,
        )
        frequency_bins = self._domain.frequency_bins
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
            self._domain.frequency_bins = np.concatenate((new_bins, frequency_bins))
        elif location_key == "end":
            new_bins = frequency_bins[-1] + step * np.arange(1, padding_length + 1)
            self._domain.frequency_bins = np.concatenate((frequency_bins, new_bins))
        else:
            raise ValueError("Padding location must be 'start' or 'end'")
        calculate_ir_from_tf(
            self._domain,
            frequency_bins=self._domain.frequency_bins,
        )

    def apply_tf_crop(
        self,
        start: int | None = None,
        end: int | None = None,
        start_frequency: float | None = None,
        end_frequency: float | None = None,
    ) -> None:
        """General Description:
        Crop TF bins by indices or frequency range and rebuild IR.

        Parameters:
        - start: Start TF bin index (inclusive) for index crop.
        - end: End TF bin index (exclusive) for index crop.
        - start_frequency: Lower frequency bound in Hz for frequency crop.
        - end_frequency: Upper frequency bound in Hz for frequency crop.

        Returns:
        None.

        Use Cases:
        - Keep only a selected spectral band.
        - Build brickwall-like masks for controlled experiments.

        Best Practices:
        - Use either index crop or frequency crop in one call.
        - Verify resulting IR behavior because hard spectral boundaries can introduce ringing.
        """
        self._domain.values = dsp_apply_tf_crop(
            self._domain,
            start=start,
            end=end,
            start_frequency=start_frequency,
            end_frequency=end_frequency,
        )
        calculate_ir_from_tf(
            self._domain,
            frequency_bins=self._domain.frequency_bins,
        )


class TransformSources(_Transform):

    def __init__(self, sources: "Sources") -> None:
        super().__init__(sources)

    


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
