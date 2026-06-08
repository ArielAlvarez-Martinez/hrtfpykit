from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .coordinates import (
    cartesian_to_lateral_polar,
    cartesian_to_spherical,
    lateral_polar_to_spherical,
    spherical_to_lateral_polar,
)


if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


def _get_plane_indices(
    hrtf: "HRTF",
    plane: str,
    plane_angle: float = 0.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve source indices for the nearest measured spatial plane.

    The helper normalizes the current :attr:`~hrtfpykit.hrtf.HRTF.Sources` grid
    before resolving a plane. Horizontal and frontal selection use spherical
    coordinates. Median selection uses lateral-polar coordinates so
    ``plane_angle`` follows the natural lateral-angle coordinate of the median
    plane family. Returned indices are always relative to the current HRTF
    source view and therefore respect any active spatial subset.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.HRTF` object whose source grid is inspected.
    plane : {``horizontal``, ``median``, ``frontal``}
        Plane family to resolve. ``horizontal`` selects a constant spherical
        elevation. ``median`` selects a constant lateral-polar lateral angle.
        ``frontal`` selects the nearest requested spherical azimuth together
        with the nearest opposite azimuth.
    plane_angle : float, default=0.0
        Plane coordinate used to resolve the nearest measured plane. For
        ``horizontal`` this is spherical elevation. For ``median`` this is
        lateral-polar lateral angle. For ``frontal`` this is spherical azimuth.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Unit used by ``plane_angle`` and by returned real plane angles.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (indices, real_plane_angles). For ``horizontal`` the real plane angle is
        the matched elevation. For ``median`` it is the matched lateral angle.
        For ``frontal`` it contains the two matched azimuths that define the
        resolved coronal plane.

    Raises
    ------
    ValueError
        If plane or angle_unit is unsupported, plane_angle is boolean or
        non-finite, source positions are not an (N, 3) grid, or the active
        source coordinate system cannot be converted to the coordinate system
        needed by the selected plane family.
    """
    plane_key = str(plane).strip().lower()
    if plane_key not in {"horizontal", "median", "frontal"}:
        raise ValueError("plane must be one of: horizontal, median, frontal")
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")
    if isinstance(plane_angle, bool):
        raise ValueError("plane_angle must be a finite value")
    plane_angle = float(plane_angle)
    if not np.isfinite(plane_angle):
        raise ValueError("plane_angle must be a finite value")

    grid_system = str(hrtf.Sources.source_coordinate_system).strip().lower()
    grid_positions = hrtf.Sources.get_positions(
        angle_unit=unit,
        coordinate_system=grid_system,
    )
    if grid_positions.ndim != 2 or grid_positions.shape[-1] != 3:
        raise ValueError("Source positions grid must have shape (N, 3)")

    if grid_system == "spherical":
        spherical_positions = grid_positions
        lateral_polar_positions = spherical_to_lateral_polar(
            grid_positions,
            angle_unit=unit,
        )
    elif grid_system == "cartesian":
        spherical_positions = cartesian_to_spherical(
            grid_positions,
            angle_unit=unit,
        )
        lateral_polar_positions = cartesian_to_lateral_polar(
            grid_positions,
            angle_unit=unit,
        )
    elif grid_system == "lateral-polar":
        lateral_polar_positions = grid_positions
        spherical_positions = lateral_polar_to_spherical(
            grid_positions,
            angle_unit=unit,
        )
    else:
        raise ValueError(f"Unsupported source coordinate system: {grid_system!r}")

    azimuth = np.asarray(spherical_positions[..., 0], dtype=float)
    elevation = np.asarray(spherical_positions[..., 1], dtype=float)
    lateral = np.asarray(lateral_polar_positions[..., 0], dtype=float)
    full = 360.0 if unit == "degrees" else 2.0 * np.pi
    half = full / 2.0

    if plane_key == "horizontal":
        available_elevations = np.unique(elevation)
        elevation_deltas = np.abs(available_elevations - plane_angle)
        real_elevation = float(available_elevations[int(np.argmin(elevation_deltas))])
        indices = np.where(np.isclose(elevation, real_elevation, atol=1e-8, rtol=0.0))[0]
        real_plane_angles = np.round(np.array([real_elevation], dtype=float), 2)
        return indices.astype(int), real_plane_angles

    if plane_key == "median":
        available_lateral_angles = np.unique(lateral)
        lateral_deltas = np.abs(available_lateral_angles - plane_angle)
        real_lateral = float(available_lateral_angles[int(np.argmin(lateral_deltas))])
        indices = np.where(np.isclose(lateral, real_lateral, atol=1e-8, rtol=0.0))[0]
        real_plane_angles = np.round(np.array([real_lateral], dtype=float), 2)
        return indices.astype(int), real_plane_angles

    available_azimuths = np.unique(azimuth)
    azimuth_deltas = np.mod(available_azimuths - plane_angle + half, full) - half
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
    plane_angle: float = 0.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, float]:
    """Return source indices for the horizontal plane nearest to an elevation.

    A horizontal plane is represented as all source positions whose spherical
    elevation equals the nearest available elevation in the current source
    grid. ``plane_angle`` is interpreted as spherical elevation. The source
    grid may be stored as spherical, cartesian, or lateral-polar coordinates;
    it is converted as needed before matching.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.HRTF` object whose :class:`~hrtfpykit.hrtf.sources.Sources` grid is inspected.
    plane_angle : float, default=0.0
        Requested horizontal-plane elevation. The nearest measured elevation in
        the grid is used when an exact match is unavailable.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used by plane_angle and by the returned real_elevation.

    Returns
    -------
    tuple[np.ndarray, float]
        (indices, real_elevation) where indices contains integer source-grid
        indices in the current source view and real_elevation is the actual
        grid elevation rounded to two decimals.

    Raises
    ------
    ValueError
        If plane_angle is boolean or non-finite, angle_unit is unsupported, the
        source grid has an invalid shape, or source positions cannot be
        converted to spherical coordinates.
    """
    indices, real_plane_angles = _get_plane_indices(
        hrtf=hrtf,
        plane="horizontal",
        plane_angle=plane_angle,
        angle_unit=angle_unit,
    )
    return indices, float(real_plane_angles[0])


def get_median_plane(
    hrtf: "HRTF",
    plane_angle: float = 0.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, float]:
    """Return source indices for the median plane nearest to a lateral angle.

    A median-plane query selects all source positions whose lateral-polar
    lateral angle equals the nearest available lateral angle in the current
    source grid. With the default angle, this resolves the sagittal plane at
    lateral angle 0 when that lateral angle is present in the grid.

    This function is used by HRTF selection, plane visualizations, and
    comparison metrics that need a median source slice. If hrtf already
    represents a selected spatial subset, returned indices address that
    selected source view.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.HRTF` object whose :class:`~hrtfpykit.hrtf.sources.Sources` grid is inspected.
    plane_angle : float, default=0.0
        Requested lateral-polar lateral angle. The nearest measured lateral
        angle in the grid is used when an exact match is unavailable.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used by plane_angle and by the returned real_lateral.

    Returns
    -------
    tuple[np.ndarray, float]
        (indices, real_lateral) where indices contains integer source-grid
        indices in the current source view and real_lateral is the actual
        lateral angle rounded to two decimals.

    Raises
    ------
    ValueError
        If plane_angle is boolean or non-finite, angle_unit is unsupported, the
        source grid has an invalid shape, or source positions cannot be
        converted to lateral-polar coordinates.
    """
    indices, real_plane_angles = _get_plane_indices(
        hrtf=hrtf,
        plane="median",
        plane_angle=plane_angle,
        angle_unit=angle_unit,
    )
    return indices, float(real_plane_angles[0])


def get_frontal_plane(
    hrtf: "HRTF",
    plane_angle: float = 90.0,
    angle_unit: str = "degrees",
) -> tuple[np.ndarray, np.ndarray]:
    """Return source indices for the frontal plane nearest to an azimuth.

    A frontal-plane query selects the nearest measured spherical azimuth to
    plane_angle and the nearest measured azimuth opposite to it. With the
    default angle, this resolves the coronal 90/270 degree plane when those
    azimuths are present in the grid.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.HRTF` object whose :class:`~hrtfpykit.hrtf.sources.Sources` grid is inspected.
    plane_angle : float, default=90.0
        Requested spherical azimuth used to resolve the primary side of the
        frontal plane. The opposite side is resolved at plane_angle + 180
        degrees or plane_angle + pi radians.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used by plane_angle and by returned real azimuths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (indices, real_azimuths) where indices contains integer source-grid
        indices in the current source view and real_azimuths contains the two
        actual grid azimuths, rounded to two decimals, that define the resolved
        plane.

    Raises
    ------
    ValueError
        If plane_angle is boolean or non-finite, angle_unit is unsupported, the
        source grid has an invalid shape, or source positions cannot be
        converted to spherical coordinates.
    """
    return _get_plane_indices(
        hrtf=hrtf,
        plane="frontal",
        plane_angle=plane_angle,
        angle_unit=angle_unit,
    )
