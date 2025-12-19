import numpy as np
from scipy.special import sph_harm
import math
import matplotlib.pyplot as plt

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
