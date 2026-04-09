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
    HRTF object for compatibility with the rest of the HRTF API. The returned
    TF keeps the FFT grid of the input HRTF, but the returned IR is cropped or
    zero-padded so that its length matches ``hrtf.IR.ir_length``.

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
        ``TF.values.shape == (1, 2, F)`` and
        ``IR.values.shape == (1, 2, hrtf.IR.ir_length)``.

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

    Design Notes
    ------------
    - The CTF is estimated in the frequency domain, so its inverse FFT length
      is naturally tied to ``TF.frequency_bins`` and therefore to the active
      FFT grid.
    - A larger FFT length increases spectral sampling density, but it does not
      create additional physical HRIR information. It only changes how finely
      the same spectrum is sampled.
    - For that reason, this function does not expose the raw inverse-FFT
      length as the final ``IR.ir_length``. Instead, it treats
      ``hrtf.IR.ir_length`` as the time-domain reference support.
    - In this API that reference IR is not optional. Loaded HRTF objects are
      expected to provide a valid ``IR`` representation, and that IR length is
      used as the design target for the returned CTF.
    - If the inverse FFT produces a longer IR, the extra tail is cropped. If
      it produces a shorter IR, zeros are appended at the end. The TF is then
      resynchronized with the same ``fft_length``.
    - This behavior is deliberate: ``TF.tf_length`` expresses FFT resolution,
      while ``IR.ir_length`` expresses the intended HRIR support.

    Examples
    --------
    Derive a common transfer function and inspect its binaural magnitude:

    >>> from hrtfpykit.hrtf.directivity import ctf_from_hrtf
    >>> from hrtfpykit import HRTF
    >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
    >>> ctf = ctf_from_hrtf(
    ...     hrtf,
    ...     weights=True,
    ...     magnitude_average="linear",
    ...     attenuation=20.0,
    ... )
    >>> ctf.plot_magnitude(
    ...     positions=ctf.Sources.get_positions(angle_unit="degrees")[0, :2],
    ...     ear="both",
    ...     show=False,
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
    """Compute a directional transfer function (DTF) from an HRTF object.

    The DTF is obtained by dividing the input HRTF by a common transfer
    function (CTF) derived from the same HRTF. The CTF is internally computed
    as a minimum-phase response with :func:`ctf_from_hrtf`, while the returned
    DTF preserves the original source axis of the input HRTF. The returned TF
    keeps the FFT grid of the input HRTF, but the returned IR is cropped or
    zero-padded so that its length matches ``hrtf.IR.ir_length``.

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
        ``TF.values.shape == (M, 2, F)`` and
        ``IR.values.shape == (M, 2, hrtf.IR.ir_length)``.

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

    Design Notes
    ------------
    - The DTF division is carried out on the active TF grid, so the raw
      inverse FFT would otherwise return an IR length implied only by
      ``TF.frequency_bins``.
    - That raw inverse-FFT length is not treated as the final DTF support,
      because changing FFT length changes spectral resolution, not the amount
      of meaningful time-domain HRTF information.
    - By design, ``hrtf.IR.ir_length`` is the reference time-domain support
      for the returned DTF.
    - In this API that reference IR is not optional. Loaded HRTF objects are
      expected to provide a valid ``IR`` representation, and that IR length is
      used as the design target for the returned DTF.
    - If the inverse FFT produces a longer IR, the extra tail is cropped. If
      it produces a shorter IR, zeros are appended at the end. The TF is then
      recomputed with the same ``fft_length`` so the spectral grid stays
      unchanged.

    Examples
    --------
    Remove the common transfer component and compare two canonical directions:

    >>> from hrtfpykit.hrtf.directivity import dtf_from_hrtf
    >>> from hrtfpykit import HRTF
    >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
    >>> dtf = dtf_from_hrtf(
    ...     hrtf,
    ...     weights=True,
    ...     magnitude_average="linear",
    ...     attenuation=20.0,
    ... )
    >>> dtf.plot_magnitude(
    ...     positions=["front", "left"],
    ...     ear="both",
    ...     show=False,
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
    )
    if target_ir_length is not None:
        current_ir_length = int(dtf_hrtf.IR.values.shape[-1])
        if current_ir_length > target_ir_length:
            dtf_hrtf.IR.values = np.asarray(
                dtf_hrtf.IR.values[..., :target_ir_length],
                dtype=float,
            )
        elif current_ir_length < target_ir_length:
            pad_width = [(0, 0)] * (dtf_hrtf.IR.values.ndim - 1) + [
                (0, target_ir_length - current_ir_length)
            ]
            dtf_hrtf.IR.values = np.pad(
                np.asarray(dtf_hrtf.IR.values, dtype=float),
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
    """Reconstruct an HRTF from a DTF and a CTF.

    The reconstruction is performed in the frequency domain by multiplying the
    directional transfer function (DTF) by the common transfer function (CTF).
    The returned object keeps the source layout of the DTF input and rebuilds
    its IR representation from the reconstructed TF. The reconstructed TF keeps
    the active FFT grid, but the reconstructed IR is cropped or zero-padded so
    that its length matches ``dtf.IR.ir_length``.

    Parameters
    ----------
    dtf : HRTF
        HRTF object containing the directional transfer function. Its source
        layout defines the source layout of the reconstructed HRTF.
    ctf : HRTF
        HRTF object containing the common transfer function. It is typically a
        singleton-source compatibility object produced by
        :func:`ctf_from_hrtf`.

    Returns
    -------
    HRTF
        New HRTF object containing the reconstructed transfer function and
        impulse response. The reconstructed source layout follows the DTF, and
        the reconstructed IR length follows ``dtf.IR.ir_length``.

    Use Cases
    ---------
    - Reconstruct an HRTF after separate CTF and DTF analysis.
    - Verify that a DTF/CTF decomposition is internally consistent.
    - Recombine a directional response with a common reference response while
      staying inside the HRTF API.

    Best Practices
    --------------
    - Use a DTF and CTF derived from the same original HRTF and the same FFT
      grid.
    - Treat the DTF as the source-layout reference and the CTF as a
      broadcast-compatible common response.
    - Treat the DTF as the time-domain reference as well: the reconstruction
      keeps the DTF source layout and the DTF IR support.

    Design Notes
    ------------
    - The multiplication ``DTF * CTF`` is performed on the TF grid, so the raw
      inverse FFT is controlled by ``TF.frequency_bins`` and ``fft_length``.
    - As in ``ctf_from_hrtf`` and ``dtf_from_hrtf``, that raw inverse-FFT
      length is not considered authoritative for the final HRIR support.
    - By design, the DTF is the reference object for reconstruction: it
      defines the directional layout and also the intended ``IR.ir_length``.
    - In this API that DTF IR is not optional. It is the explicit time-domain
      reference used to size the reconstructed HRTF.
    - If the inverse FFT produces a longer IR, the extra tail is cropped. If
      it produces a shorter IR, zeros are appended at the end. The TF is then
      recomputed with the same ``fft_length`` so the reconstruction keeps the
      chosen spectral resolution without silently changing HRIR support.

    Examples
    --------
    Reconstruct an HRTF after separate CTF and DTF analysis:

    >>> from hrtfpykit.hrtf.directivity import (
    ...     ctf_from_hrtf,
    ...     dtf_from_hrtf,
    ...     hrtf_from_dtf_and_ctf,
    ... )
    >>> from hrtfpykit import HRTF
    >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
    >>> dtf = dtf_from_hrtf(hrtf)
    >>> ctf = ctf_from_hrtf(hrtf)
    >>> hrtf_reconstructed = hrtf_from_dtf_and_ctf(dtf, ctf)
    >>> hrtf_reconstructed.plot_magnitude(
    ...     positions=["front", "left"],
    ...     ear="both",
    ...     show=False,
    ... )
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
    )
    if target_ir_length is not None:
        current_ir_length = int(hrtf.IR.values.shape[-1])
        if current_ir_length > target_ir_length:
            hrtf.IR.values = np.asarray(
                hrtf.IR.values[..., :target_ir_length],
                dtype=float,
            )
        elif current_ir_length < target_ir_length:
            pad_width = [(0, 0)] * (hrtf.IR.values.ndim - 1) + [
                (0, target_ir_length - current_ir_length)
            ]
            hrtf.IR.values = np.pad(
                np.asarray(hrtf.IR.values, dtype=float),
                pad_width,
                mode="constant",
                constant_values=0.0,
            )
        tf_from_ir(
            hrtf.IR,
            fft_length=hrtf.fft_length,
        )

    return hrtf
