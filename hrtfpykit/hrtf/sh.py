from __future__ import annotations


from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import sph_harm_y

from .coordinates import get_source_positions

if TYPE_CHECKING:
    from .domain import TF
    from .hrtf import HRTF


@dataclass
class SH:
    """Spherical-harmonic representation of HRTF magnitude data.

    :class:`~hrtfpykit.hrtf.SH` is the container returned by
    :func:`~hrtfpykit.hrtf.sht`. It stores the
    regularized spherical-harmonic coefficient matrix together with the basis
    matrix evaluated at the HRTF source directions. The representation is built
    from linear HRTF magnitudes, not complex transfer functions, so phase is
    not encoded in the coefficients.

    The coefficient axis follows the implementation order used by
    :func:`~hrtfpykit.hrtf.sht`: for each degree n from 0 through
    ``sh_order``, it stores orders m from -n through n. The total coefficient
    count is (sh_order + 1) ** 2.

    Attributes
    ----------
    C : np.ndarray
        Spherical-harmonic coefficient matrix. For one selected ear the shape
        is (n_coefficients, n_frequencies). For ear=``both`` the shape is
        (n_coefficients, 2, n_frequencies).
    Y : np.ndarray
        Real-valued spherical-harmonic basis matrix with shape
        (N, n_coefficients), evaluated at the source directions used during
        decomposition.
    sh_order : int
        Non-negative spherical-harmonic order used to create C and Y.
    N : int
        Number of source positions used during decomposition.
    """

    C: np.ndarray
    Y: np.ndarray
    sh_order: int
    N: int

    def get_coefficients(self) -> np.ndarray:
        """Return the spherical-harmonic coefficient matrix.

        This method returns :attr:`~hrtfpykit.hrtf.SH.C` unchanged. The
        first axis indexes spherical-harmonic coefficients; remaining axes
        follow the selected ear layout and frequency-bin axis produced by
        :func:`~hrtfpykit.hrtf.sht`.

        Returns
        -------
        np.ndarray
            The coefficient matrix stored in C.
        """
        return self.C


def sht(
    tf: "TF | HRTF",
    sh_order: int,
    ear: str = "left",
    epsilon: float = 1e-6,
) -> SH:
    """Compute a spherical-harmonic decomposition of HRTF magnitudes.

    The function projects linear HRTF magnitudes from the source grid into a
    real-valued spherical-harmonic basis. It accepts either an
    :class:`~hrtfpykit.hrtf.hrtf.HRTF` object or its linked
    :class:`~hrtfpykit.hrtf.domain.TF` domain object. In both cases, the linked
    HRTF source grid is used to evaluate the basis. The complex transfer functions
    stored in :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` are decomposed
    by magnitude.

    The source coordinates are read as spherical positions in radians. The
    implementation maps SOFA-style azimuth/elevation to spherical-harmonic
    angles where phi is the azimuth and theta is pi / 2 minus elevation before
    evaluating scipy.special.sph_harm_y. Coefficients are solved with Tikhonov
    regularization.

    Parameters
    ----------
    tf : TF | HRTF
        Input frequency-domain domain object or
        :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.
        The values stored in
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` must have shape
        (positions, ears, frequency_bins) and contain at least two ear channels.
        The bins stored in
        :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` must
        match the final TF axis.
    sh_order : int
        Non-negative spherical-harmonic order. The coefficient count is
        (sh_order + 1) ** 2.
    ear : {``left``, ``right``, ``both``}, default=``left``
        Ear channel used for the decomposition. ``left`` and ``right``
        produce a two-dimensional coefficient matrix. ``both`` preserves a
        two-ear axis in the coefficient matrix.
    epsilon : float, default=1e-6
        Positive Tikhonov regularization factor added to the normal matrix.
        Larger values stabilize ill-conditioned source grids more strongly but
        increase regularization bias.

    Returns
    -------
    SH
        SH-domain container with C shaped
        (n_coefficients, n_frequencies) for one ear or
        (n_coefficients, 2, n_frequencies) for both ears, and Y shaped
        (N, n_coefficients).

    Raises
    ------
    ValueError
        If tf is not an HRTF or linked TF object, TF values or frequency
        bins are missing, TF shape is incompatible, the source grid does not
        match the TF position axis, ear is invalid, sh_order is not a
        non-negative integer, or epsilon is not finite and positive.

    Examples
    --------
    >>> sh = sht(hrtf, sh_order=10, ear="left")
    >>> C = sh.get_coefficients()
    >>> C.shape[0]
    121
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
            complex_basis = sph_harm_y(n, m, theta, phi)
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
    """Reconstruct HRTF magnitudes on the original source grid.

    The reconstruction multiplies the basis matrix stored in
    :attr:`SH.Y <hrtfpykit.hrtf.SH.Y>` by the coefficient matrix stored in
    :attr:`SH.C <hrtfpykit.hrtf.SH.C>`. It reconstructs magnitudes only and
    uses the same source directions that were used to build the
    :class:`~hrtfpykit.hrtf.SH` object; it does not evaluate the
    spherical-harmonic model on new directions.

    Parameters
    ----------
    sh : SH
        SH-domain object returned by :func:`~hrtfpykit.hrtf.sht`.
        :attr:`~hrtfpykit.hrtf.SH.Y` must have shape (N, n_coefficients)
        and :attr:`~hrtfpykit.hrtf.SH.C` must start with the same
        coefficient count implied by :attr:`~hrtfpykit.hrtf.SH.sh_order`.

    Returns
    -------
    np.ndarray
        Reconstructed linear magnitude matrix. The shape is
        (N, n_frequencies) when sh.C is two-dimensional, or
        (N, 2, n_frequencies) when sh.C is three-dimensional.

    Raises
    ------
    ValueError
        If :attr:`~hrtfpykit.hrtf.SH.Y`,
        :attr:`~hrtfpykit.hrtf.SH.C`, :attr:`~hrtfpykit.hrtf.SH.N`, or
        :attr:`~hrtfpykit.hrtf.SH.sh_order` are mutually inconsistent, or
        if :attr:`~hrtfpykit.hrtf.SH.C` is not two- or three-dimensional.

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

    The function compares two linear-magnitude arrays element by element. It
    is intended for evaluating the output of
    :func:`~hrtfpykit.hrtf.sht_inverse` against the
    original magnitudes used for the decomposition, but it accepts any two
    finite arrays with matching shape.

    Parameters
    ----------
    original_magnitude : np.ndarray
        Reference linear-magnitude values. Must have the same shape as
        reconstructed_magnitude.
    reconstructed_magnitude : np.ndarray
        Reconstructed linear-magnitude values to evaluate against the reference.

    Returns
    -------
    tuple[float, float, float, float]
        Error metrics returned as:

        - absolute_error: L2 norm of original - reconstructed.
        - relative_error: absolute error divided by the reference L2 norm.
          If the reference norm is zero, this value is inf.
        - rms_error: root-mean-square point-wise error.
        - max_absolute_error: maximum absolute point-wise error.

    Raises
    ------
    ValueError
        If the input arrays have different shapes, are empty, or contain
        non-finite values.

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
