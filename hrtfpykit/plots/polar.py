from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..hrtf.coordinates import get_source_positions
from ..hrtf.planes import get_horizontal_plane


if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


def create_horizontal_plane_curve(
    hrtf: "HRTF",
    values: np.ndarray,
    elevation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    indices, real_elevation = get_horizontal_plane(
        hrtf=hrtf,
        elevation=elevation,
        angle_unit="degrees",
    )
    if indices.size == 0:
        raise ValueError("Horizontal plane does not contain any source positions")

    spherical_positions = get_source_positions(
        sources=hrtf.Sources,
        coordinate_system="spherical",
        angle_unit="degrees",
    )[indices]
    azimuth_values = np.mod(np.asarray(spherical_positions[:, 0], dtype=float), 360.0)
    plane_values = np.asarray(values, dtype=float)[indices]
    if plane_values.ndim != 1:
        plane_values = np.asarray(plane_values, dtype=float).reshape(-1)

    sort_indices = np.argsort(azimuth_values)
    sorted_azimuth_values = azimuth_values[sort_indices]
    sorted_plane_values = plane_values[sort_indices]
    if sorted_azimuth_values.size > 1:
        theta_values = np.deg2rad(
            np.concatenate(
                (
                    sorted_azimuth_values,
                    np.array([sorted_azimuth_values[0] + 360.0], dtype=float),
                )
            )
        )
        radial_values = np.concatenate(
            (
                sorted_plane_values,
                np.array([sorted_plane_values[0]], dtype=float),
            )
        )
    else:
        theta_values = np.deg2rad(sorted_azimuth_values)
        radial_values = sorted_plane_values
    return theta_values, radial_values, sorted_plane_values, float(real_elevation)
