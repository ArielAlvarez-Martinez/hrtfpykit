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
    magnitude: str = "db",
    attenuation: float | None = None,
) -> "HRTF":
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
    magnitude_key = str(magnitude).strip().lower()
    if magnitude_key not in {"db", "linear"}:
        raise ValueError("magnitude must be one of: db, linear")

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

    if magnitude_key == "db":
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
