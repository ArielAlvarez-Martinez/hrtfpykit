from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import SphericalVoronoi

from .coordinates import get_spherical_positions, spherical_to_cartesian
from .dsp import ir_from_tf, magnitude, minimum_phase, tf_from_ir

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


def ctf_from_hrtf(
    hrtf: "HRTF",
    weights: bool = False,
    magnitude_average: str = "log",
    attenuation: float | None = None,
) -> "HRTF":
    """Estimate a common transfer function from an :class:`~hrtfpykit.hrtf.HRTF` object.

    A common transfer function (CTF) describes the source-independent spectral
    component of an HRTF. This function estimates that component by averaging
    the magnitude response over the source axis, reconstructing a
    minimum-phase response from the averaged magnitude, and returning the
    result as a new :class:`~hrtfpykit.hrtf.HRTF` instance. The output keeps a singleton source axis
    so it can still be used with the same plotting, transformation, and SOFA
    synchronization workflows as ordinary :class:`~hrtfpykit.hrtf.HRTF` objects.

    The computation is performed on :attr:`~hrtfpykit.hrtf.HRTF.TF`. If
    :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` is available, its
    final-axis length is treated as the reference HRIR support:
    the reconstructed CTF impulse response is cropped or padded with zeros to that
    length and the TF is rebuilt on the original FFT grid. If no IR values are
    available, the inverse transform length implied by the TF grid is kept.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.HRTF`
        Input :class:`~hrtfpykit.hrtf.HRTF` object.
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` supplies the
        complex transfer functions,
        :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`
        supplies the active frequency grid, and
        :attr:`~hrtfpykit.hrtf.HRTF.Sources` supplies source geometry when
        diffuse-field weighting is requested.
    weights : bool, optional
        If False, all source positions contribute equally. If True,
        source weights are derived from the spherical Voronoi area associated
        with each source direction. Weighted estimation is useful for
        irregular measurement grids where equal source weights would
        over-represent densely sampled regions.
    magnitude_average : {``log``, ``linear``}, optional
        Averaging rule applied to the source magnitudes. ``log`` averages
        log magnitudes, which is equivalent to a geometric mean in linear
        magnitude. ``linear`` averages linear magnitudes directly, which is
        equivalent to an arithmetic mean.
    attenuation : float | None, optional
        Optional attenuation in dB applied to the CTF magnitude before the
        minimum-phase reconstruction. If None, the averaged magnitude is
        used without additional attenuation.

    Returns
    -------
    HRTF
        New :class:`~hrtfpykit.hrtf.HRTF` object containing the CTF. For a typical binaural input with
        TF.values.shape == (M, 2, F), the output uses
        TF.values.shape == (1, 2, F) and a matching singleton-source IR
        representation.

    Raises
    ------
    ValueError
        If hrtf does not expose the expected HRTF interface, TF data are
        missing, empty, not NumPy, or have fewer than two frequency bins, TF
        frequency bins are missing, weights is not boolean,
        magnitude_average is not ``log`` or ``linear``,
        attenuation is not finite and non-negative, the TF grid contains
        negative frequency bins, or diffuse-field weights cannot be derived
        from the source grid.

    Notes
    -----
    weights=True requires a spherical source grid with one position per TF
    source, at least four unique directions, and strictly positive source
    radii. The source radii are normalized to one before computing Voronoi
    areas because the weighting represents directional coverage on the unit
    sphere, not physical source distance.

    The minimum-phase reconstruction currently assumes one-sided,
    non-negative TF frequency bins. A larger FFT length increases spectral
    sampling density but does not add physical HRIR support; when an input IR
    is present, its length remains the time-domain reference for the returned
    object.
    """
    try:
        tf = hrtf.TF
        sources = hrtf.Sources
    except AttributeError:
        raise ValueError("hrtf must be an HRTF instance")

    tf_values = tf.values
    frequency_bins = tf.frequency_bins

    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    if tf_values.size == 0:
        raise ValueError("TF data must be non-empty")
    if tf_values.ndim < 2:
        raise ValueError("TF data must have at least source and frequency dimensions")
    if tf_values.shape[-1] < 2:
        raise ValueError("TF data must contain at least two frequency bins")
    if frequency_bins is None:
        raise ValueError("TF frequency_bins are required")
    target_ir_length = None
    if hrtf.IR.values is not None:
        target_ir_length = int(np.asarray(hrtf.IR.values).shape[-1])

    source_count = int(tf_values.shape[0])
    tiny = np.finfo(float).tiny
    selected_indices = sources._selected_indices

    if not isinstance(weights, bool):
        raise ValueError("weights must be a boolean")
    magnitude_average_key = str(magnitude_average).strip().lower()
    if magnitude_average_key not in {"log", "linear"}:
        raise ValueError("magnitude_average must be one of: log, linear")

    if attenuation is not None:
        if isinstance(attenuation, bool):
            raise ValueError("attenuation must be a finite, non-negative value.")
        try:
            attenuation = float(attenuation)
        except (TypeError, ValueError):
            raise ValueError("attenuation must be a finite, non-negative value.") from None
        if not np.isfinite(attenuation) or attenuation < 0.0:
            raise ValueError("attenuation must be a finite, non-negative value.")

    if weights:
        spherical_positions = get_spherical_positions(
            sources,
            angle_unit="radians",
        )
        if spherical_positions.shape[0] != source_count:
            raise ValueError("Source positions must match the TF source dimension")
        if source_count < 4:
            raise ValueError("Diffuse-field weights require at least four source positions")

        unit_spherical_positions = np.array(spherical_positions, copy=True)
        radii = unit_spherical_positions[..., 2]
        if np.any(radii <= 0.0):
            raise ValueError("Diffuse-field weights require strictly positive source radii")
        unit_spherical_positions[..., 2] = 1.0
        unit_cartesian_positions = spherical_to_cartesian(
            unit_spherical_positions,
            angle_unit="radians",
        )

        rounded_positions = np.round(unit_cartesian_positions, decimals=12)
        if np.unique(rounded_positions, axis=0).shape[0] != unit_cartesian_positions.shape[0]:
            raise ValueError("Diffuse-field weights require unique source directions")

        try:
            voronoi = SphericalVoronoi(unit_cartesian_positions)
            source_weights = voronoi.calculate_areas()
        except ValueError as exc:
            raise ValueError(
                "Diffuse-field weights could not be derived from the source positions"
            ) from exc

        source_weights = np.asarray(source_weights, dtype=float)
        if source_weights.ndim != 1 or source_weights.shape[0] != source_count:
            raise ValueError("Derived diffuse-field weights must match the TF source dimension")
        if np.any(source_weights <= 0.0):
            raise ValueError("Derived diffuse-field weights must be positive")
        source_weights = source_weights / np.sum(source_weights)
    else:
        source_weights = np.full(
            source_count,
            1.0 / source_count,
            dtype=float,
        )

    magnitude_values = np.maximum(magnitude(tf), tiny)

    if magnitude_average_key == "log":
        ctf_magnitude = np.exp(
            np.tensordot(source_weights, np.log(magnitude_values), axes=(0, 0))
        )
    else:
        ctf_magnitude = np.tensordot(source_weights, magnitude_values, axes=(0, 0))

    if attenuation is not None:
        ctf_magnitude = ctf_magnitude / np.power(10.0, attenuation / 20.0)

    ctf_hrtf = hrtf.clone()
    ctf_hrtf.TF.values = np.asarray(ctf_magnitude, dtype=np.complex128)[np.newaxis, ...]
    ctf_hrtf.TF.frequency_bins = np.array(frequency_bins, copy=True)
    ir_from_tf(
        ctf_hrtf.TF,
        frequency_bins=ctf_hrtf.TF.frequency_bins,
        mesh2hrtf_compatible=ctf_hrtf.mesh2hrtf_compatible,
        n_shift=ctf_hrtf.mesh2hrtf_n_shift,
    )

    if np.min(np.asarray(ctf_hrtf.TF.frequency_bins, dtype=float)) < 0.0:
        raise ValueError("minimum-phase CTF currently requires one-sided TF data")
    ctf_hrtf.IR.values = minimum_phase(ctf_hrtf.IR)
    if target_ir_length is not None:
        current_ir_length = int(ctf_hrtf.IR.values.shape[-1])
        if current_ir_length > target_ir_length:
            ctf_hrtf.IR.values = np.asarray(
                ctf_hrtf.IR.values[..., :target_ir_length],
                dtype=float,
            )
        elif current_ir_length < target_ir_length:
            pad_width = [(0, 0)] * (ctf_hrtf.IR.values.ndim - 1) + [
                (0, target_ir_length - current_ir_length)
            ]
            ctf_hrtf.IR.values = np.pad(
                np.asarray(ctf_hrtf.IR.values, dtype=float),
                pad_width,
                mode="constant",
                constant_values=0.0,
            )
    tf_from_ir(
        ctf_hrtf.IR,
        fft_length=ctf_hrtf.fft_length,
    )

    if ctf_hrtf.Sofa is not None:
        if selected_indices is None:
            ctf_hrtf.Sources._selected_indices = np.array([0], dtype=int)
        else:
            ctf_hrtf.Sources._selected_indices = np.array(
                [int(np.asarray(selected_indices, dtype=int)[0])],
                dtype=int,
            )

    return ctf_hrtf


def dtf_from_hrtf(
    hrtf: "HRTF",
    weights: bool = False,
    magnitude_average: str = "log",
    attenuation: float | None = None,
) -> "HRTF":
    """Estimate a directional transfer function from an :class:`~hrtfpykit.hrtf.HRTF` object.

    A directional transfer function (DTF) isolates the source-dependent part
    of an HRTF by removing an internally estimated common transfer function
    (CTF). This function computes the CTF with
    :func:`~hrtfpykit.utils.directivity.ctf_from_hrtf`, divides
    the original complex TF values by that CTF on the active frequency grid,
    and returns the directional result as a new :class:`~hrtfpykit.hrtf.HRTF` instance.

    The output preserves the source layout of the input object. If
    :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` is available, its
    final-axis length is used as the reference HRIR support for the reconstructed DTF impulse responses. If no
    IR values are available, the inverse transform length implied by the TF
    grid is kept.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.HRTF`
        Input :class:`~hrtfpykit.hrtf.HRTF` object.
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` supplies the
        complex transfer functions,
        :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`
        supplies the active frequency grid, and
        :attr:`~hrtfpykit.hrtf.HRTF.Sources` supplies source geometry when
        diffuse-field weighting is requested for the internal CTF.
    weights : bool, optional
        If False, all source positions contribute equally to the internal
        CTF estimate. If True, source weights are derived from spherical
        Voronoi areas. Weighted estimation is recommended for irregular source
        grids when the DTF should represent a diffuse-field normalization
        rather than the sampling density of the measurement set.
    magnitude_average : {``log``, ``linear``}, optional
        Averaging rule used to estimate the internal CTF magnitude before the
        DTF division. ``log`` averages log magnitudes, which is equivalent
        to a geometric mean in linear magnitude. ``linear`` averages linear
        magnitudes directly, which is equivalent to an arithmetic mean.
    attenuation : float | None, optional
        Optional attenuation in dB applied after the HRTF is divided by the
        internal CTF. If None, the DTF magnitude is not attenuated.

    Returns
    -------
    HRTF
        New :class:`~hrtfpykit.hrtf.HRTF` object containing the DTF. For a typical binaural input with
        TF.values.shape == (M, 2, F), the output keeps the same source and
        ear layout and returns DTF values on the same frequency grid.

    Raises
    ------
    ValueError
        If hrtf does not expose the expected HRTF interface, TF data are
        missing, empty, not NumPy, or have fewer than two frequency bins, TF
        frequency bins are missing, attenuation is not finite and
        non-negative, or the internal CTF computation fails because weighting,
        averaging, source geometry, or frequency-bin requirements are not met.

    Notes
    -----
    The internal CTF is computed without attenuation and the optional
    attenuation value is applied only to the final DTF. During the TF
    division, CTF bins with vanishing magnitude are replaced by the smallest
    positive floating-point magnitude with the same phase, preventing numerical
    division by zero while keeping phase continuity.
    """
    try:
        tf = hrtf.TF
    except AttributeError:
        raise ValueError("hrtf must be an HRTF instance")

    tf_values = tf.values
    frequency_bins = tf.frequency_bins

    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    if tf_values.size == 0:
        raise ValueError("TF data must be non-empty")
    if tf_values.ndim < 2:
        raise ValueError("TF data must have at least source and frequency dimensions")
    if tf_values.shape[-1] < 2:
        raise ValueError("TF data must contain at least two frequency bins")
    if frequency_bins is None:
        raise ValueError("TF frequency_bins are required")
    target_ir_length = None
    if hrtf.IR.values is not None:
        target_ir_length = int(np.asarray(hrtf.IR.values).shape[-1])

    if attenuation is not None:
        if isinstance(attenuation, bool):
            raise ValueError("attenuation must be a finite, non-negative value.")
        try:
            attenuation = float(attenuation)
        except (TypeError, ValueError):
            raise ValueError("attenuation must be a finite, non-negative value.") from None
        if not np.isfinite(attenuation) or attenuation < 0.0:
            raise ValueError("attenuation must be a finite, non-negative value.")

    ctf_hrtf = ctf_from_hrtf(
        hrtf=hrtf,
        weights=weights,
        magnitude_average=magnitude_average,
        attenuation=None,
    )

    tiny = np.finfo(float).tiny
    ctf_tf_values = np.asarray(ctf_hrtf.TF.values, dtype=np.complex128)
    ctf_magnitude_values = np.abs(ctf_tf_values)
    safe_ctf_tf_values = np.where(
        ctf_magnitude_values > tiny,
        ctf_tf_values,
        tiny * np.exp(1j * np.angle(ctf_tf_values)),
    )

    dtf_hrtf = hrtf.clone()
    dtf_hrtf.TF.values = np.asarray(tf_values, dtype=np.complex128) / safe_ctf_tf_values
    dtf_hrtf.TF.frequency_bins = np.array(frequency_bins, copy=True)

    if attenuation is not None:
        dtf_hrtf.TF.values = dtf_hrtf.TF.values / np.power(10.0, attenuation / 20.0)

    ir_from_tf(
        dtf_hrtf.TF,
        frequency_bins=dtf_hrtf.TF.frequency_bins,
        mesh2hrtf_compatible=dtf_hrtf.mesh2hrtf_compatible,
        n_shift=dtf_hrtf.mesh2hrtf_n_shift,
    )
    if target_ir_length is not None:
        if dtf_hrtf.IR.values is None:
            raise ValueError("DTF IR values are not available")
        dtf_ir_values = dtf_hrtf.IR.values
        current_ir_length = int(dtf_ir_values.shape[-1])
        if current_ir_length > target_ir_length:
            dtf_hrtf.IR.values = np.asarray(
                dtf_ir_values[..., :target_ir_length],
                dtype=float,
            )
        elif current_ir_length < target_ir_length:
            pad_width = [(0, 0)] * (dtf_ir_values.ndim - 1) + [
                (0, target_ir_length - current_ir_length)
            ]
            dtf_hrtf.IR.values = np.pad(
                np.asarray(dtf_ir_values, dtype=float),
                pad_width,
                mode="constant",
                constant_values=0.0,
            )
        tf_from_ir(
            dtf_hrtf.IR,
            fft_length=dtf_hrtf.fft_length,
        )

    return dtf_hrtf


def hrtf_from_dtf_and_ctf(
    dtf: "HRTF",
    ctf: "HRTF",
) -> "HRTF":
    """Reconstruct an HRTF from DTF and CTF components.

    ``hrtf_from_dtf_and_ctf`` combines a directional transfer function
    (DTF) with a common transfer function (CTF) to recover a full HRTF
    object. The reconstruction is performed in the complex frequency domain:
    the DTF values are multiplied by the CTF values on the same frequency
    grid, then the time domain impulse responses are rebuilt from the
    reconstructed transfer functions.

    This function is intended for workflows where the directional and common
    parts of an HRTF are handled separately. A common example is a deep
    learning experiment that predicts DTF values and then needs to combine
    them with a measured, estimated, or predicted CTF before saving, plotting,
    or evaluating the reconstructed HRTF.

    The returned object is cloned from ``dtf``. The DTF source layout,
    metadata, SOFA handle, and HRTF interface therefore remain the reference,
    while the TF and IR values are replaced by the reconstructed HRTF values.
    If ``dtf.IR.values`` is available, its final-axis length is used as the
    HRIR support for the reconstructed impulse responses. If no DTF IR values
    are available, the inverse transform length implied by the TF grid is
    kept.

    Parameters
    ----------
    dtf : HRTF
        :class:`~hrtfpykit.hrtf.HRTF` object containing the directional
        transfer function. ``dtf.TF.values`` define the directional spectral
        component, ``dtf.TF.frequency_bins`` define the output frequency grid,
        and the DTF source layout defines the source layout of the
        reconstructed HRTF. In normal hrtfpykit workflows, ``dtf`` is produced
        with :meth:`~hrtfpykit.hrtf.transforms.Transform.to_dtf`.
    ctf : HRTF
        :class:`~hrtfpykit.hrtf.HRTF` object containing the common transfer
        function. In normal hrtfpykit workflows, ``ctf`` is produced with
        :meth:`~hrtfpykit.hrtf.transforms.Transform.to_ctf`. Its frequency
        bins must match the DTF frequency bins, and its leading TF dimensions
        must broadcast to the DTF layout without expanding that layout.

    Returns
    -------
    HRTF
        New :class:`~hrtfpykit.hrtf.HRTF` object containing the reconstructed
        transfer functions and impulse responses. The source layout follows
        ``dtf``. The frequency grid follows ``dtf.TF.frequency_bins`` and must
        match ``ctf.TF.frequency_bins``.

    Raises
    ------
    ValueError
        If either input does not expose the expected HRTF interface, TF data
        are missing, empty, not NumPy, or have fewer than two frequency bins,
        frequency bins are missing or do not match, TF lengths differ, or the
        CTF leading dimensions cannot broadcast to the DTF leading dimensions
        without expanding the DTF layout.

    Notes
    -----
    DTF alone is not enough to recover the original HRTF. The CTF carries the
    common spectral component that was removed during DTF construction, so a
    reconstruction workflow must provide both components.

    If a reference DTF IR length is available, reconstructed HRIRs are cropped
    or padded with zeros to that support and the TF is rebuilt with
    :attr:`~hrtfpykit.hrtf.HRTF.fft_length`.

    Examples
    --------
    Reconstruct an HRTF after separating it into DTF and CTF components:

    >>> from hrtfpykit.hrtf import hrtf_from_dtf_and_ctf, load_hrtf
    >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
    >>> dtf = hrtf.transform.to_dtf()
    >>> ctf = hrtf.transform.to_ctf()
    >>> reconstructed = hrtf_from_dtf_and_ctf(dtf, ctf)
    >>> reconstructed.TF.values.shape == hrtf.TF.values.shape
    True

    """
    try:
        dtf_tf = dtf.TF
        ctf_tf = ctf.TF
    except AttributeError:
        raise ValueError("dtf and ctf must be HRTF instances")

    dtf_tf_values = dtf_tf.values
    ctf_tf_values = ctf_tf.values
    dtf_frequency_bins = dtf_tf.frequency_bins
    ctf_frequency_bins = ctf_tf.frequency_bins

    if dtf_tf_values is None:
        raise ValueError("DTF TF data is not available")
    if ctf_tf_values is None:
        raise ValueError("CTF TF data is not available")
    if not isinstance(dtf_tf_values, np.ndarray):
        raise ValueError("DTF TF data must be a NumPy array")
    if not isinstance(ctf_tf_values, np.ndarray):
        raise ValueError("CTF TF data must be a NumPy array")
    if dtf_tf_values.size == 0:
        raise ValueError("DTF TF data must be non-empty")
    if ctf_tf_values.size == 0:
        raise ValueError("CTF TF data must be non-empty")
    if dtf_tf_values.ndim < 2:
        raise ValueError("DTF TF data must have at least source and frequency dimensions")
    if ctf_tf_values.ndim < 2:
        raise ValueError("CTF TF data must have at least source and frequency dimensions")
    if dtf_tf_values.shape[-1] < 2:
        raise ValueError("DTF TF data must contain at least two frequency bins")
    if ctf_tf_values.shape[-1] < 2:
        raise ValueError("CTF TF data must contain at least two frequency bins")
    if dtf_frequency_bins is None:
        raise ValueError("DTF TF frequency_bins are required")
    if ctf_frequency_bins is None:
        raise ValueError("CTF TF frequency_bins are required")
    target_ir_length = None
    if dtf.IR.values is not None:
        target_ir_length = int(np.asarray(dtf.IR.values).shape[-1])

    dtf_frequency_bins = np.asarray(dtf_frequency_bins, dtype=float)
    ctf_frequency_bins = np.asarray(ctf_frequency_bins, dtype=float)
    if dtf_frequency_bins.shape != ctf_frequency_bins.shape:
        raise ValueError("DTF and CTF frequency_bins must have the same shape")
    if not np.allclose(dtf_frequency_bins, ctf_frequency_bins, rtol=1e-8, atol=1e-12):
        raise ValueError("DTF and CTF frequency_bins must match")

    if ctf_tf_values.shape[-1] != dtf_tf_values.shape[-1]:
        raise ValueError("DTF and CTF TF lengths must match")

    try:
        broadcast_shape = np.broadcast_shapes(
            dtf_tf_values.shape[:-1],
            ctf_tf_values.shape[:-1],
        )
    except ValueError:
        raise ValueError(
            "CTF TF leading shape must be broadcastable to the DTF TF leading shape"
        ) from None

    if broadcast_shape != dtf_tf_values.shape[:-1]:
        raise ValueError(
            "CTF TF leading shape must not expand the DTF TF leading shape"
        )

    hrtf = dtf.clone()
    hrtf.TF.values = (
        np.asarray(dtf_tf_values, dtype=np.complex128)
        * np.asarray(ctf_tf_values, dtype=np.complex128)
    )
    hrtf.TF.frequency_bins = np.array(dtf_frequency_bins, copy=True)
    ir_from_tf(
        hrtf.TF,
        frequency_bins=hrtf.TF.frequency_bins,
        mesh2hrtf_compatible=hrtf.mesh2hrtf_compatible,
        n_shift=hrtf.mesh2hrtf_n_shift,
    )
    if target_ir_length is not None:
        if hrtf.IR.values is None:
            raise ValueError("HRTF IR values are not available")
        hrtf_ir_values = hrtf.IR.values
        current_ir_length = int(hrtf_ir_values.shape[-1])
        if current_ir_length > target_ir_length:
            hrtf.IR.values = np.asarray(
                hrtf_ir_values[..., :target_ir_length],
                dtype=float,
            )
        elif current_ir_length < target_ir_length:
            pad_width = [(0, 0)] * (hrtf_ir_values.ndim - 1) + [
                (0, target_ir_length - current_ir_length)
            ]
            hrtf.IR.values = np.pad(
                np.asarray(hrtf_ir_values, dtype=float),
                pad_width,
                mode="constant",
                constant_values=0.0,
            )
        tf_from_ir(
            hrtf.IR,
            fft_length=hrtf.fft_length,
        )

    return hrtf
