from __future__ import annotations


import numpy as np
from scipy.special import sph_harm
from typing import TYPE_CHECKING
from dataclasses import dataclass

from .coordinates import get_source_positions

if TYPE_CHECKING:
    from .domain import TF
    from .hrtf import HRTF

@dataclass
class SH:
    """Spherical-harmonic-domain representation of HRTF magnitudes.

    Stores the SH coefficient matrix and the SH basis matrix used during
    decomposition, together with metadata required for inverse reconstruction.

    Attributes
    ----------
    C : np.ndarray
        SH coefficient matrix.
    Y : np.ndarray
        SH basis matrix evaluated at the source directions.
    sh_order : int
        SH order used to compute `C`.
    N : int
        Number of source positions used during decomposition.
    """

    C: np.ndarray
    Y: np.ndarray
    sh_order: int
    N: int

    def get_coefficients(self) -> np.ndarray:
        """Return the SH coefficient matrix.

        Returns
        -------
        np.ndarray
            SH coefficient matrix `C`.
        """
        return self.C


def sht(
    tf: "TF | HRTF",
    sh_order: int,
    ear: str = "left",
    epsilon: float = 1e-6,
) -> SH:
    """Compute spherical harmonic decomposition from HRTF magnitudes.

    This method projects magnitude responses over the spatial source grid into
    spherical harmonic coefficients. It supports decomposition for one ear
    (`left` or `right`) or both ears (`both`) across all available frequency bins.

    Parameters
    ----------
    tf : TF | HRTF
        Input frequency-domain data or an HRTF object containing TF data.
    sh_order : int
        Non-negative SH order.
    ear : str, default="left"
        Ear selection. Accepted values: `left`, `right`, `both`.
    epsilon : float, default=1e-6
        Tikhonov regularization factor added to the normal matrix.

    Returns
    -------
    SH
        SH container with:
        - `C`: coefficient matrix of shape `(n_coeffs, n_freqs)` or
          `(n_coeffs, 2, n_freqs)` for `ear="both"`.
        - `Y`: basis matrix of shape `(N, n_coeffs)`.
        - `sh_order` and `N` metadata.

    Use Cases
    ---------
    - Compress HRTF magnitude data into SH coefficients.
    - Prepare SH-domain data for interpolation or reconstruction workflows.

    Examples
    --------
    >>> sh = sht(hrtf, sh_order=10, ear="left")
    >>> C = sh.get_coefficients()
    >>> C.shape
    (121, n_freqs)
    """
    if hasattr(tf, "TF") and hasattr(tf, "Sources"):
        hrtf = tf
        tf_domain = hrtf.TF
    elif hasattr(tf, "values") and hasattr(tf, "frequency_bins") and hasattr(tf, "_hrtf"):
        tf_domain = tf
        hrtf = tf._hrtf
    else:
        raise ValueError("tf must be a TF or HRTF instance")

    if tf_domain.values is None:
        raise ValueError("TF values are not available")
    if tf_domain.frequency_bins is None:
        raise ValueError("TF frequency_bins are not available")

    ear_key = str(ear).strip().lower()
    if ear_key not in {"left", "right", "both"}:
        raise ValueError("ear must be one of: left, right, both")
    if isinstance(sh_order, bool) or not isinstance(sh_order, int) or sh_order < 0:
        raise ValueError("sh_order must be a non-negative integer")
    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    tf_values = np.asarray(tf_domain.values)
    if tf_values.ndim != 3:
        raise ValueError("TF values must have shape (positions, ears, frequency_bins)")
    if tf_values.shape[1] < 2:
        raise ValueError("TF values must include two ears")
    frequency_bins = np.asarray(tf_domain.frequency_bins, dtype=float).reshape(-1)
    if frequency_bins.size != tf_values.shape[-1]:
        raise ValueError("TF frequency_bins length must match TF frequency axis")
    if ear_key == "left":
        magnitude_values = np.asarray(np.abs(tf_values[:, 0, :]), dtype=float)
    elif ear_key == "right":
        magnitude_values = np.asarray(np.abs(tf_values[:, 1, :]), dtype=float)
    else:
        magnitude_values = np.asarray(np.abs(tf_values[:, 0:2, :]), dtype=float)
    dirs = np.asarray(
        get_source_positions(
            sources=hrtf.Sources,
            coordinate_system="spherical",
            angle_unit="radians",
        )[:, :2],
        dtype=float,
    )
    if dirs.ndim != 2 or dirs.shape[1] != 2:
        raise ValueError("Resolved directions must have shape (N, 2)")
    if dirs.shape[0] != magnitude_values.shape[0]:
        raise ValueError("TF positions and source directions count must match")

    N = magnitude_values.shape[0]
    n_coeffs = (sh_order + 1) ** 2

    azimuth = dirs[:, 0]
    elevation = dirs[:, 1]
    theta = (np.pi / 2.0) - elevation
    phi = azimuth

    Y = np.zeros((N, n_coeffs), dtype=float)
    coeff_index = 0
    for n in range(sh_order + 1):
        for m in range(-n, n + 1):
            complex_basis = sph_harm(m, n, phi, theta)
            if m < 0:
                Y[:, coeff_index] = np.sqrt(2.0) * ((-1) ** m) * complex_basis.imag
            elif m == 0:
                Y[:, coeff_index] = complex_basis.real
            else:
                Y[:, coeff_index] = np.sqrt(2.0) * ((-1) ** m) * complex_basis.real
            coeff_index += 1

    A = Y.T @ Y + epsilon * np.eye(n_coeffs)
    if magnitude_values.ndim == 2:
        b = Y.T @ magnitude_values
        C = np.linalg.solve(A, b)
    elif magnitude_values.ndim == 3:
        n_ears = magnitude_values.shape[1]
        n_freqs = magnitude_values.shape[2]
        C = np.zeros((n_coeffs, n_ears, n_freqs), dtype=float)
        for ear_index in range(n_ears):
            b = Y.T @ magnitude_values[:, ear_index, :]
            C[:, ear_index, :] = np.linalg.solve(A, b)
    else:
        raise ValueError("Resolved magnitudes must have shape (N, F) or (N, E, F)")

    return SH(C=C, Y=Y, sh_order=sh_order, N=N)

def sht_inverse(sh: SH):
    """Reconstruct magnitude matrix from SH coefficients.

    Parameters
    ----------
    sh : SH
        SH-domain object returned by :func:`sht`.

    Returns
    -------
    np.ndarray
        Reconstructed magnitude matrix:
        - `(N, n_freqs)` when `sh.C` is 2D.
        - `(N, 2, n_freqs)` when `sh.C` is 3D (both ears).

    Use Cases
    ---------
    - Recover spatial magnitude values from SH coefficients.
    - Evaluate SH approximation quality against original magnitudes.

    Examples
    --------
    >>> sh = sht(hrtf, sh_order=8, ear="both")
    >>> magnitude_reconstructed = sht_inverse(sh)
    >>> magnitude_reconstructed.shape[0] == sh.N
    True
    """
    C = np.asarray(sh.C)
    Y = np.asarray(sh.Y)

    n_coeffs = (int(sh.sh_order) + 1) ** 2
    if Y.ndim != 2:
        raise ValueError("SH.Y must have shape (N, n_coefficients)")
    if Y.shape[0] != int(sh.N):
        raise ValueError("SH.N must match SH.Y first dimension")
    if Y.shape[1] != n_coeffs:
        raise ValueError(
            "SH.Y has incompatible coefficient dimension for SH.sh_order"
        )
    if C.shape[0] != n_coeffs:
        raise ValueError(
            "SH.C has incompatible coefficient dimension for SH.sh_order"
        )

    if C.ndim == 2:
        return Y @ C
    if C.ndim == 3:
        return np.einsum("nc,cef->nef", Y, C)
    raise ValueError("SH.C must have shape (n_coefficients, n_freqs) or (n_coefficients, 2, n_freqs)")


def sht_error(
    original_magnitude: np.ndarray,
    reconstructed_magnitude: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute reconstruction error metrics between two magnitude tensors.

    Parameters
    ----------
    original_magnitude : np.ndarray
        Reference magnitude values. Must have the same shape as
        ``reconstructed_magnitude``.
    reconstructed_magnitude : np.ndarray
        Reconstructed magnitude values to evaluate against the reference.

    Returns
    -------
    tuple[float, float, float, float]
        Error metrics returned as:
        - ``absolute_error``: L2 norm of the difference.
        - ``relative_error``: absolute error divided by reference L2 norm.
        - ``rms_error``: root-mean-square error.
        - ``max_absolute_error``: maximum absolute point-wise error.

    Use Cases
    ---------
    - Quantify global SH reconstruction quality after ``sht_inverse``.
    - Compare reconstruction quality across SH orders.
    - Report compact error metrics for selected ears/positions/frequencies.

    Examples
    --------
    >>> reconstructed = sht_inverse(sh)
    >>> abs_err, rel_err, rms_err, max_err = sht_error(
    ...     original_magnitude=np.abs(hrtf.TF.values[:, 0, :]),
    ...     reconstructed_magnitude=reconstructed,
    ... )
    >>> rms_err >= 0.0
    True
    """
    original_values = np.asarray(original_magnitude, dtype=float)
    reconstructed_values = np.asarray(reconstructed_magnitude, dtype=float)
    if original_values.shape != reconstructed_values.shape:
        raise ValueError("original_magnitude and reconstructed_magnitude must have the same shape")
    if original_values.size == 0:
        raise ValueError("magnitude arrays must be non-empty")
    if not np.all(np.isfinite(original_values)) or not np.all(np.isfinite(reconstructed_values)):
        raise ValueError("magnitude arrays must contain finite values")

    difference = original_values - reconstructed_values
    absolute_error = float(np.linalg.norm(difference))
    original_norm = float(np.linalg.norm(original_values))
    if np.isclose(original_norm, 0.0, atol=1e-15, rtol=0.0):
        relative_error = float("inf")
    else:
        relative_error = float(absolute_error / original_norm)
    rms_error = float(np.sqrt(np.mean(difference**2)))
    max_absolute_error = float(np.max(np.abs(difference)))
    return absolute_error, relative_error, rms_error, max_absolute_error
