from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from .coordinates import cartesian_to_spherical, lateral_polar_to_spherical


if TYPE_CHECKING:
    from .hrtf import HRTF


def _get_plane_indices(
    hrtf: "HRTF",
    plane: str,
    angle: float = 0.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, np.ndarray]:
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

    grid_system = str(hrtf.Sources.source_coordinate_system).strip().lower()
    grid_positions = hrtf.Sources.get_positions(angle_unit=unit)
    if grid_positions.ndim != 2 or grid_positions.shape[-1] != 3:
        raise ValueError("Source positions grid must have shape (N, 3)")

    if grid_system == "spherical":
        spherical_positions = grid_positions
    elif grid_system == "cartesian":
        spherical_positions = cartesian_to_spherical(
            grid_positions,
            angle_unit=unit,
        )
    elif grid_system == "lateral-polar":
        spherical_positions = lateral_polar_to_spherical(
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
    """Return the horizontal plane nearest to a requested elevation.

    Parameters
    ----------
    hrtf : HRTF
        HRTF instance that provides the source grid to inspect.
    elevation : float, default=0.0
        Requested horizontal-plane elevation.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used by ``elevation`` and by the returned real elevation.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(indices, real_elevation)`` where ``indices`` contains the source-grid
        indices in the selected horizontal plane and ``real_elevation`` is the
        actual elevation present in the grid.

    Examples
    --------
    Resolve the nearest horizontal plane and inspect its source grid:

    >>> from hrtfpykit import HRTF
    >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
    >>> indices, real_elevation = get_horizontal_plane(hrtf, elevation=0.0)
    >>> horizontal = hrtf.select(plane="horizontal", plane_angle=real_elevation)
    >>> horizontal.plot_source_grid(show=False)
    """
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
    """Return the median plane nearest to a requested azimuth.

    Parameters
    ----------
    hrtf : HRTF
        HRTF instance that provides the source grid to inspect.
    azimuth : float, default=0.0
        Requested azimuth used to resolve the nearest median plane.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used by ``azimuth`` and by the returned real azimuths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(indices, real_azimuths)`` where ``indices`` contains the source-grid
        indices in the selected median plane and ``real_azimuths`` contains the
        two opposite azimuths that define that plane in the grid.

    Examples
    --------
    Resolve the nearest median plane and inspect that sagittal slice:

    >>> from hrtfpykit import HRTF
    >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
    >>> indices, real_azimuths = get_median_plane(hrtf, azimuth=0.0)
    >>> median = hrtf.select(plane="median", plane_angle=real_azimuths[0])
    >>> median.plot_source_grid(show=False)
    """
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
    """Return the frontal plane nearest to a requested azimuth.

    Parameters
    ----------
    hrtf : HRTF
        HRTF instance that provides the source grid to inspect.
    azimuth : float, default=90.0
        Requested azimuth used to resolve the nearest frontal plane.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used by ``azimuth`` and by the returned real azimuths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(indices, real_azimuths)`` where ``indices`` contains the source-grid
        indices in the selected frontal plane and ``real_azimuths`` contains the
        two opposite azimuths that define that plane in the grid.

    Examples
    --------
    Resolve the nearest frontal plane and inspect that coronal slice:

    >>> from hrtfpykit import HRTF
    >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
    >>> indices, real_azimuths = get_frontal_plane(hrtf, azimuth=90.0)
    >>> frontal = hrtf.select(plane="frontal", plane_angle=real_azimuths[0])
    >>> frontal.plot_source_grid(show=False)
    """
    return _get_plane_indices(
        hrtf=hrtf,
        plane="frontal",
        angle=azimuth,
        angle_unit=angle_unit,
    )
