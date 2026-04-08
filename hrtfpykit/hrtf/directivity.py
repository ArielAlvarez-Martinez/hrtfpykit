from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import SphericalVoronoi

from .coordinates import get_spherical_positions, spherical_to_cartesian
from .dsp import ir_from_tf, magnitude as tf_magnitude, minimum_phase, tf_from_ir

if TYPE_CHECKING:
    from .hrtf import HRTF


def ctf_from_hrtf(
    hrtf: "HRTF",
    weights: bool = False,
    magnitude_average: str = "log",
    attenuation: float | None = None,
) -> "HRTF":
    """Compute a common transfer function (CTF) from an HRTF object.

    The CTF is derived by collapsing the source axis of the input HRTF into a
    single common spectral response for each ear. The resulting magnitude is
    then converted to a minimum-phase transfer function and returned as a new
    HRTF object for compatibility with the rest of the HRTF API.

    Parameters
    ----------
    hrtf : HRTF
        Input HRTF object. Its TF data and source geometry are used to compute
        the CTF.
    weights : bool, optional
        If ``False``, all source positions contribute equally. If ``True``,
        diffuse-field weights are derived internally from the HRTF source
        positions using spherical Voronoi areas.
    magnitude_average : {"log", "linear"}, optional
        Rule used to average source magnitudes before the minimum-phase
        reconstruction. ``"log"`` computes a log-magnitude average
        (geometric mean in linear magnitude). ``"linear"`` computes a direct
        linear-magnitude average (arithmetic mean).
    attenuation : float | None, optional
        Optional attenuation in dB applied to the CTF magnitude before the
        minimum-phase reconstruction. If ``None``, no attenuation is applied.

    Returns
    -------
    HRTF
        New HRTF object containing the CTF. The output keeps a singleton
        source axis for compatibility, so a typical binaural output has
        ``TF.values.shape == (1, 2, F)`` and ``IR.values.shape == (1, 2, N)``.

    Use Cases
    ---------
    - Derive a common spectral reference from a measured HRTF.
    - Prepare a CTF object for later DTF-style directivity decomposition.
    - Export a common binaural response while preserving compatibility with the
      HRTF API.

    Best Practices
    --------------
    - Use ``weights=True`` when the source grid represents a full sphere and
      you want a diffuse-field-style CTF.
    - Use ``magnitude_average="log"`` as the default choice for directivity
      analysis because it is less dominated by large spectral peaks.
    - Treat the returned singleton source axis as a compatibility axis, not as
      a real directional measurement.

    Examples
    --------
    >>> from hrtfpykit.hrtf.directivity import ctf_from_hrtf
    >>> ctf = ctf_from_hrtf(hrtf)
    >>> ctf.TF.values.shape[0]
    1

    >>> ctf = ctf_from_hrtf(
    ...     hrtf,
    ...     weights=True,
    ...     magnitude_average="linear",
    ...     attenuation=20.0,
    ... )
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

    magnitude_values = np.maximum(tf_magnitude(tf), tiny)

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
    )

    if np.min(np.asarray(ctf_hrtf.TF.frequency_bins, dtype=float)) < 0.0:
        raise ValueError("minimum-phase CTF currently requires one-sided TF data")
    ctf_hrtf.IR.values = minimum_phase(ctf_hrtf.IR)
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
    """Compute a directional transfer function (DTF) from an HRTF object.

    The DTF is obtained by dividing the input HRTF by a common transfer
    function (CTF) derived from the same HRTF. The CTF is internally computed
    as a minimum-phase response with :func:`ctf_from_hrtf`, while the returned
    DTF preserves the original source axis of the input HRTF.

    Parameters
    ----------
    hrtf : HRTF
        Input HRTF object. Its TF data are divided by the internally derived
        CTF.
    weights : bool, optional
        If ``False``, all source positions contribute equally to the internal
        CTF estimate. If ``True``, diffuse-field weights are derived internally
        from the HRTF source positions using spherical Voronoi areas.
    magnitude_average : {"log", "linear"}, optional
        Rule used to estimate the internal CTF magnitude before the DTF
        division. ``"log"`` computes a log-magnitude average
        (geometric mean in linear magnitude). ``"linear"`` computes a direct
        linear-magnitude average (arithmetic mean).
    attenuation : float | None, optional
        Optional attenuation in dB applied to the DTF after the CTF division.
        If ``None``, no attenuation is applied.

    Returns
    -------
    HRTF
        New HRTF object containing the DTF. The output preserves the source
        layout of the input HRTF, so a typical binaural output keeps
        ``TF.values.shape == (M, 2, F)`` and ``IR.values.shape == (M, 2, N)``.

    Use Cases
    ---------
    - Remove the common spectral component of an HRTF while preserving the
      directional structure.
    - Prepare DTF data for directivity analysis or later recombination with a
      CTF.
    - Export an HRTF-shaped directional response that remains compatible with
      the rest of the HRTF API.

    Best Practices
    --------------
    - Use ``weights=True`` when the source grid represents a full sphere and
      you want a diffuse-field-style DTF decomposition.
    - Use ``magnitude_average="log"`` as the default choice for directivity
      analysis because it is less dominated by large spectral peaks.
    - Interpret ``attenuation`` as a playback or export headroom control for
      the DTF itself, not as part of the CTF estimation.

    Examples
    --------
    >>> from hrtfpykit.hrtf.directivity import dtf_from_hrtf
    >>> dtf = dtf_from_hrtf(hrtf)
    >>> dtf.TF.values.shape[0] == hrtf.TF.values.shape[0]
    True

    >>> dtf = dtf_from_hrtf(
    ...     hrtf,
    ...     weights=True,
    ...     magnitude_average="linear",
    ...     attenuation=20.0,
    ... )
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
    )

    return dtf_hrtf
