from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .hrtf import HRTF


def _get_plane_indices(
    hrtf: "HRTF",
    plane: str,
    angle: float = 0.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, np.ndarray]:
    """Return source indices for a requested plane and selected plane angles."""
    plane_key = str(plane).strip().lower()
    if plane_key not in {"horizontal", "median", "frontal"}:
        raise ValueError("plane must be one of: horizontal, median, frontal")
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")
    if isinstance(angle, bool):
        raise ValueError("angle must be a finite value")
    angle = float(angle)
    if not np.isfinite(angle):
        raise ValueError("angle must be a finite value")

    grid_system = str(hrtf.Sources.get_source_coordinate_system()).strip().lower()
    grid_positions = hrtf.Sources.get_positions(angle_unit=unit)
    if grid_positions.ndim != 2 or grid_positions.shape[-1] != 3:
        raise ValueError("Source positions grid must have shape (N, 3)")

    if grid_system == "spherical":
        spherical_positions = grid_positions
    elif grid_system == "cartesian":
        spherical_positions = hrtf.Sources.cartesian_to_spherical(
            grid_positions,
            angle_unit=unit,
        )
    elif grid_system == "lateral-polar":
        spherical_positions = hrtf.Sources.lateral_polar_to_spherical(
            grid_positions,
            angle_unit=unit,
        )
    else:
        raise ValueError(f"Unsupported source coordinate system: {grid_system!r}")

    azimuth = np.asarray(spherical_positions[..., 0], dtype=float)
    elevation = np.asarray(spherical_positions[..., 1], dtype=float)
    full = 360.0 if unit == "degrees" else 2.0 * np.pi
    half = full / 2.0

    if plane_key == "horizontal":
        available_elevations = np.unique(elevation)
        elevation_deltas = np.abs(available_elevations - angle)
        real_elevation = float(available_elevations[int(np.argmin(elevation_deltas))])
        indices = np.where(np.isclose(elevation, real_elevation, atol=1e-8, rtol=0.0))[0]
        real_plane_angles = np.round(np.array([real_elevation], dtype=float), 2)
        return indices.astype(int), real_plane_angles

    available_azimuths = np.unique(azimuth)
    azimuth_deltas = np.mod(available_azimuths - angle + half, full) - half
    real_primary = float(available_azimuths[int(np.argmin(np.abs(azimuth_deltas)))])
    opposite_target = np.mod(real_primary + half, full)
    opposite_deltas = np.mod(available_azimuths - opposite_target + half, full) - half
    real_opposite = float(available_azimuths[int(np.argmin(np.abs(opposite_deltas)))])

    delta_primary = np.mod(azimuth - real_primary + half, full) - half
    delta_opposite = np.mod(azimuth - real_opposite + half, full) - half
    primary_indices = np.where(np.isclose(delta_primary, 0.0, atol=1e-8, rtol=0.0))[0]
    opposite_indices = np.where(np.isclose(delta_opposite, 0.0, atol=1e-8, rtol=0.0))[0]
    indices = np.unique(np.concatenate((primary_indices, opposite_indices))).astype(int)
    real_plane_angles = np.round(np.array([real_primary, real_opposite], dtype=float), 2)
    return indices, real_plane_angles


def get_horizontal_plane(
    hrtf: "HRTF",
    elevation: float = 0.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, float]:
    """Return indices of the horizontal plane nearest to requested elevation."""
    indices, real_plane_angles = _get_plane_indices(
        hrtf=hrtf,
        plane="horizontal",
        angle=elevation,
        angle_unit=angle_unit,
    )
    return indices, float(real_plane_angles[0])


def get_median_plane(
    hrtf: "HRTF",
    azimuth: float = 0.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of the median plane nearest to requested azimuth."""
    return _get_plane_indices(
        hrtf=hrtf,
        plane="median",
        angle=azimuth,
        angle_unit=angle_unit,
    )


def get_frontal_plane(
    hrtf: "HRTF",
    azimuth: float = 90.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of the frontal plane nearest to requested azimuth."""
    return _get_plane_indices(
        hrtf=hrtf,
        plane="frontal",
        angle=azimuth,
        angle_unit=angle_unit,
    )
