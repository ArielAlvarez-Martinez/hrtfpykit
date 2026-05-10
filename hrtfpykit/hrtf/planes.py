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
    """Resolve source indices for the nearest measured spatial plane.

    The helper normalizes the current :attr:`~hrtfpykit.hrtf.hrtf.HRTF.Sources` grid to spherical
    coordinates before resolving a plane. It respects any active source subset
    already stored on the :class:`~hrtfpykit.hrtf.hrtf.HRTF` object, so returned indices are relative to the
    current source view.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.hrtf.HRTF` object whose source grid is inspected.
    plane : {"horizontal", "median", "frontal"}
        Plane family to resolve. "horizontal" selects a constant elevation.
        "median" and "frontal" select the nearest requested azimuth
        together with the nearest opposite azimuth.
    angle : float, default=0.0
        Requested elevation for "horizontal" or requested azimuth for
        "median" and "frontal".
    angle_unit : {"degrees", "radians"}, default="degrees"
        Unit used by angle and by returned plane angles.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (indices, real_plane_angles). indices contains integer source
        indices in the current source view. real_plane_angles contains one
        rounded elevation for a horizontal plane or two rounded azimuths for
        median and frontal planes.

    Raises
    ------
    ValueError
        If plane or angle_unit is unsupported, angle is boolean or
        non-finite, source positions are not an (N, 3) grid, or the active
        source coordinate system cannot be converted to spherical coordinates.
    """
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
    """Return source indices for the horizontal plane nearest to an elevation.

    A horizontal plane is represented as all source positions whose spherical
    elevation equals the nearest available elevation in the current source
    grid. The source grid may be stored as spherical, cartesian, or
    lateral-polar coordinates; it is converted to spherical coordinates before
    matching.

    This function is used by HRTF selection, horizontal-plane spectrum plots,
    interaural-cue plots, and comparison metrics. If hrtf already
    represents a selected spatial subset, the returned indices address that
    selected source view.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.hrtf.HRTF` object whose :class:`~hrtfpykit.hrtf.sources.Sources` grid is inspected.
    elevation : float, default=0.0
        Requested horizontal-plane elevation. The nearest measured elevation in
        the grid is used when an exact match is unavailable.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used by elevation and by the returned
        real_elevation.

    Returns
    -------
    tuple[np.ndarray, float]
        (indices, real_elevation) where indices contains integer
        source-grid indices in the current source view and real_elevation
        is the actual grid elevation rounded to two decimals.

    Raises
    ------
    ValueError
        If elevation is boolean or non-finite, angle_unit is
        unsupported, the source grid has an invalid shape, or source positions
        cannot be converted to spherical coordinates.
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
    """Return source indices for the median plane nearest to an azimuth.

    A median-plane query selects the nearest measured azimuth to azimuth and
    the nearest measured azimuth opposite to it. With the default angle, this
    resolves the sagittal 0/180 degree plane when those azimuths are
    present in the grid. The source grid may be stored as spherical,
    cartesian, or lateral-polar coordinates; it is converted to spherical
    coordinates before matching.

    This function is used by HRTF selection, plane visualizations, and
    comparison metrics that need a sagittal source slice. If hrtf already
    represents a selected spatial subset, returned indices address that
    selected source view.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.hrtf.HRTF` object whose :class:`~hrtfpykit.hrtf.sources.Sources` grid is inspected.
    azimuth : float, default=0.0
        Requested azimuth used to resolve the primary side of the median plane.
        The opposite side is resolved at azimuth + 180 degrees or
        azimuth + pi radians.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used by azimuth and by returned real azimuths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (indices, real_azimuths) where indices contains integer
        source-grid indices in the current source view and real_azimuths
        contains the two actual grid azimuths, rounded to two decimals, that
        define the resolved plane.

    Raises
    ------
    ValueError
        If azimuth is boolean or non-finite, angle_unit is unsupported,
        the source grid has an invalid shape, or source positions cannot be
        converted to spherical coordinates.
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
    """Return source indices for the frontal plane nearest to an azimuth.

    A frontal-plane query selects the nearest measured azimuth to azimuth
    and the nearest measured azimuth opposite to it. With the default angle,
    this resolves the coronal 90/270 degree plane when those azimuths
    are present in the grid. The source grid may be stored as spherical,
    cartesian, or lateral-polar coordinates; it is converted to spherical
    coordinates before matching.

    This function is used by HRTF selection and source-grid plane plots. If
    hrtf already represents a selected spatial subset, returned indices
    address that selected source view.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.hrtf.HRTF` object whose :class:`~hrtfpykit.hrtf.sources.Sources` grid is inspected.
    azimuth : float, default=90.0
        Requested azimuth used to resolve the primary side of the frontal
        plane. The opposite side is resolved at azimuth + 180 degrees or
        azimuth + pi radians.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used by azimuth and by returned real azimuths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (indices, real_azimuths) where indices contains integer
        source-grid indices in the current source view and real_azimuths
        contains the two actual grid azimuths, rounded to two decimals, that
        define the resolved plane.

    Raises
    ------
    ValueError
        If azimuth is boolean or non-finite, angle_unit is unsupported,
        the source grid has an invalid shape, or source positions cannot be
        converted to spherical coordinates.
    """
    return _get_plane_indices(
        hrtf=hrtf,
        plane="frontal",
        angle=azimuth,
        angle_unit=angle_unit,
    )
