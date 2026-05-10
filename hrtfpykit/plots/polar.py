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
    """Create polar-plot curve arrays for a horizontal HRTF plane.

    :func:`~hrtfpykit.plots.polar.create_horizontal_plane_curve` prepares the
    data used by ITD, ILD, and comparison plots that summarize a source-grid
    metric over a horizontal plane. It resolves the requested elevation through
    :func:`~hrtfpykit.hrtf.planes.get_horizontal_plane`, reads the corresponding
    spherical source positions, sorts the selected values by azimuth, converts azimuth
    degrees to Matplotlib polar radians, and closes the returned radial curve when
    more than one point is available.

    The input values should contain one scalar metric per source position in the
    current HRTF view. Typical callers pass per-source ITD or ILD values already
    reduced over ear and frequency axes. The function does not compute acoustic
    metrics; it only aligns and orders caller-provided values with the selected source
    positions.

    Parameters
    ----------
    hrtf : :class:`~hrtfpykit.hrtf.hrtf.HRTF`
        :class:`~hrtfpykit.hrtf.hrtf.HRTF` object whose
        :class:`~hrtfpykit.hrtf.sources.Sources` manager provides the current
        source grid. Spatial selections already applied to the HRTF are
        reflected by the source positions used here.
    values : np.ndarray
        Per-source scalar values aligned with the HRTF source-position axis. The first
        axis must correspond to source positions because the horizontal-plane indices
        are applied directly to this array.
    elevation : float, default=0.0
        Requested horizontal-plane elevation in degrees. The returned
        real_elevation may differ when the source grid does not contain the exact
        requested elevation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        (theta_values, radial_values, sorted_plane_values, real_elevation),
        where theta_values are azimuth angles in radians for Matplotlib polar
        axes, radial_values are the sorted plane values with the first value
        repeated at the end when the curve can be closed, sorted_plane_values are
        the sorted per-azimuth values without the closing sample, and
        real_elevation is the horizontal-plane elevation actually selected from
        the HRTF grid.

    Raises
    ------
    ValueError
        If the selected horizontal plane has no source positions. Errors from source
        coordinate conversion or horizontal-plane resolution are propagated from the
        HRTF source and plane utilities.
    IndexError
        If values does not contain entries for the selected source indices.

    Notes
    -----
    theta_values and radial_values are the arrays intended for plotting with
    a polar Matplotlib axis. sorted_plane_values is returned separately so callers
    can derive radial limits or tick labels from the unclosed metric values. When the
    selected plane contains a single point, the curve is not closed because repeating
    the point would not add an angular segment.

    """
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
