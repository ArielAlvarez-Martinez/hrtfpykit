from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .sources import Sources


def get_position_queries(
    positions: np.ndarray | list | tuple | str,
) -> list[np.ndarray | str]:
    """Normalize one or more position queries.

    Parameters
    ----------
    positions : np.ndarray | list | tuple | str
        Position query or collection of queries. Each numeric query must have
        shape ``(2,)`` or ``(3,)``. String queries are normalized to lowercase.

    Returns
    -------
    list[np.ndarray | str]
        Normalized list of queries, where numeric entries are returned as
        ``np.ndarray`` and string entries are returned as lowercase strings.

    Use Cases
    ---------
    - Normalize user input before resolving source-grid positions.
    - Accept either one query or many queries through the same API.
    - Support mixed named and numeric position inputs.

    Examples
    --------
    >>> get_position_queries("front")
    ['front']
    >>> get_position_queries([0.0, 0.0])
    [array([0., 0.])]
    >>> get_position_queries([[0.0, 0.0], [90.0, 0.0]])
    [array([0., 0.]), array([90.,  0.])]
    """
    error_message = "positions must have shape (2,), (3,), (K, 2), or (K, 3)"
    accepted_lengths = {2, 3}

    if isinstance(positions, str):
        return [str(positions).strip().lower()]

    if isinstance(positions, np.ndarray) and positions.dtype != object:
        numeric_positions = np.asarray(positions, dtype=float)
        if numeric_positions.ndim == 1:
            if numeric_positions.shape[0] not in accepted_lengths:
                raise ValueError(error_message)
            return [numeric_positions]
        if numeric_positions.ndim == 2 and numeric_positions.shape[-1] in accepted_lengths:
            return [
                np.asarray(position, dtype=float)
                for position in numeric_positions
            ]

    raw_positions = np.asarray(positions, dtype=object)
    if raw_positions.ndim == 0:
        value = raw_positions.item()
        if isinstance(value, str):
            return [str(value).strip().lower()]
        position = np.asarray(value, dtype=float)
        if position.ndim != 1 or position.shape[0] not in accepted_lengths:
            raise ValueError(error_message)
        return [position]

    if raw_positions.ndim == 1:
        values = raw_positions.tolist()
        if len(values) == 0:
            raise ValueError("At least one position is required")
        if all(isinstance(value, str) for value in values):
            return [str(value).strip().lower() for value in values]
        if all(np.isscalar(value) and not isinstance(value, str) for value in values):
            if raw_positions.shape[0] not in accepted_lengths:
                raise ValueError(error_message)
            return [np.asarray(raw_positions, dtype=float)]
        position_queries: list[np.ndarray | str] = []
        for value in values:
            if isinstance(value, str):
                position_queries.append(str(value).strip().lower())
                continue
            position = np.asarray(value, dtype=float)
            if position.ndim != 1 or position.shape[0] not in accepted_lengths:
                raise ValueError(error_message)
            position_queries.append(position)
        return position_queries

    if raw_positions.ndim == 2 and raw_positions.shape[-1] in accepted_lengths:
        numeric_positions = np.asarray(raw_positions, dtype=float)
        return [
            np.asarray(position, dtype=float)
            for position in numeric_positions
        ]

    raise ValueError(error_message)


def get_named_positions(
    angle_unit: str = "degrees",
) -> dict[str, np.ndarray]:
    """Return canonical named positions in spherical coordinates.

    Parameters
    ----------
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used for the returned azimuth and elevation values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from ``{"front", "left", "back", "right"}`` to spherical
        ``(azimuth, elevation)`` position arrays.

    Use Cases
    ---------
    - Resolve named source queries into numeric coordinates.
    - Generate canonical front, left, back, and right references.
    - Build plot labels or directional selections.

    Examples
    --------
    >>> get_named_positions()["front"]
    array([0., 0.])
    >>> get_named_positions(angle_unit="radians")["left"]
    array([1.57079633, 0.        ])
    """
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")
    named_positions = {
        "front": np.array([0.0, 0.0], dtype=float),
        "left": np.array([90.0, 0.0], dtype=float),
        "back": np.array([180.0, 0.0], dtype=float),
        "right": np.array([270.0, 0.0], dtype=float),
    }
    if unit == "radians":
        return {
            name: np.deg2rad(position)
            for name, position in named_positions.items()
        }
    return named_positions


def get_position_alias(
    position: np.ndarray | list[float] | tuple[float, ...],
    coordinate_system: str = "spherical",
    angle_unit: str = "degrees",
) -> str | None:
    """Return the canonical alias of a horizontal cardinal position.

    Parameters
    ----------
    position : np.ndarray | list[float] | tuple[float, ...]
        Position to evaluate.
    coordinate_system : {"spherical", "cartesian", "lateral-polar"}, default="spherical"
        Coordinate system used by ``position``.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit for spherical and lateral-polar inputs.

    Returns
    -------
    str | None
        One of ``{"front", "left", "back", "right"}`` when the position
        matches a canonical horizontal cardinal direction, otherwise ``None``.

    Use Cases
    ---------
    - Replace explicit coordinates with directional aliases in plot titles.
    - Recognize canonical positions after coordinate conversion.
    - Build cleaner user-facing labels for horizontal positions.

    Examples
    --------
    >>> get_position_alias([0.0, 0.0])
    'front'
    >>> get_position_alias([90.0, 0.0])
    'left'
    >>> get_position_alias([0.0, 30.0])
    """
    system = str(coordinate_system).strip().lower()
    if system not in {"spherical", "cartesian", "lateral-polar"}:
        raise ValueError(
            "coordinate_system must be one of: spherical, cartesian, lateral-polar"
        )
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")

    resolved_position = np.asarray(position, dtype=float)
    if system == "cartesian":
        if resolved_position.shape != (3,):
            raise ValueError("For cartesian, position must have shape (3,)")
        spherical_position = cartesian_to_spherical(
            resolved_position,
            angle_unit=unit,
        )
    elif system == "lateral-polar":
        if resolved_position.shape not in {(2,), (3,)}:
            raise ValueError(
                "For lateral-polar, position must have shape (2,) or (3,)"
            )
        spherical_position = lateral_polar_to_spherical(
            resolved_position,
            angle_unit=unit,
        )
    else:
        if resolved_position.shape not in {(2,), (3,)}:
            raise ValueError(
                "For spherical, position must have shape (2,) or (3,)"
            )
        spherical_position = resolved_position

    azimuth = float(np.asarray(spherical_position, dtype=float)[0])
    elevation = float(np.asarray(spherical_position, dtype=float)[1])
    if not np.isfinite(azimuth) or not np.isfinite(elevation):
        raise ValueError("position must contain finite values")

    full = 360.0 if unit == "degrees" else 2.0 * np.pi
    half = full / 2.0
    if not np.isclose(elevation, 0.0, atol=1e-6, rtol=0.0):
        return None
    for name, named_position in get_named_positions(angle_unit=unit).items():
        azimuth_delta = np.mod(azimuth - float(named_position[0]) + half, full) - half
        elevation_delta = elevation - float(named_position[1])
        if np.isclose(azimuth_delta, 0.0, atol=1e-6, rtol=0.0) and np.isclose(
            elevation_delta,
            0.0,
            atol=1e-6,
            rtol=0.0,
        ):
            return name
    return None


def get_spherical_positions(
    sources: "Sources",
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Return a source grid expressed in spherical coordinates.

    Parameters
    ----------
    sources : Sources
        Source-grid view used to read and convert positions.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used for the returned azimuth and elevation values.

    Returns
    -------
    np.ndarray
        Source positions with shape ``(N, 3)`` in spherical
        ``(azimuth, elevation, radius)`` coordinates.

    Use Cases
    ---------
    - Normalize any source-grid coordinate system to spherical.
    - Reuse one spherical conversion path across multiple source queries.
    - Prepare angle-based analyses from cartesian or lateral-polar grids.

    Examples
    --------
    >>> spherical = get_spherical_positions(hrtf.Sources)
    >>> spherical.shape[-1]
    3
    """
    positions = sources.get_positions(angle_unit=angle_unit)
    target_system = str(sources.source_coordinate_system).strip().lower()
    if target_system == "spherical":
        return np.asarray(positions, dtype=float)
    if target_system == "cartesian":
        return np.asarray(
            cartesian_to_spherical(
                positions,
                angle_unit=angle_unit,
            ),
            dtype=float,
        )
    if target_system == "lateral-polar":
        cartesian = lateral_polar_to_cartesian(
            positions,
            angle_unit=angle_unit,
        )
        return np.asarray(
            cartesian_to_spherical(
                cartesian,
                angle_unit=angle_unit,
            ),
            dtype=float,
        )
    raise ValueError(f"Unsupported target coordinate system: {target_system!r}")


def spherical_to_cartesian(
    coordinates: np.ndarray,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Convert spherical coordinates into cartesian coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape ``(..., 3)`` containing spherical
        ``(azimuth, elevation, radius)`` values.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit of the azimuth and elevation values.

    Returns
    -------
    np.ndarray
        Array with shape ``(..., 3)`` containing cartesian ``(x, y, z)``
        coordinates.

    Use Cases
    ---------
    - Convert spherical source grids for 3D plotting.
    - Prepare source positions for cartesian geometric processing.

    Examples
    --------
    >>> spherical_to_cartesian(np.array([[0.0, 0.0, 1.0]]))
    array([[1., 0., 0.]])
    >>> spherical_to_cartesian(np.array([[90.0, 0.0, 1.0]]))
    array([[0., 1., 0.]])
    >>> spherical_to_cartesian(np.array([[0.0, 90.0, 1.0]]))
    array([[0., 0., 1.]])
    """
    spherical = np.asarray(coordinates, dtype=float)
    if spherical.shape[-1] != 3:
        raise ValueError("Spherical coordinates must have shape (..., 3)")
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")

    azimuth = spherical[..., 0]
    elevation = spherical[..., 1]
    radius = spherical[..., 2]
    if np.any(radius < 0.0):
        raise ValueError("Spherical radius must be non-negative")

    if unit == "degrees":
        if np.any((elevation < -90.0) | (elevation > 90.0)):
            raise ValueError("Spherical elevation must be in [-90, 90] degrees")
        azimuth = np.mod(azimuth, 360.0)
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    else:
        half_pi = np.pi / 2.0
        if np.any((elevation < -half_pi) | (elevation > half_pi)):
            raise ValueError("Spherical elevation must be in [-pi/2, pi/2] radians")
        azimuth = np.mod(azimuth, 2.0 * np.pi)

    cos_elevation = np.cos(elevation)
    x = radius * cos_elevation * np.cos(azimuth)
    y = radius * cos_elevation * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return np.stack((x, y, z), axis=-1)


def cartesian_to_spherical(
    coordinates: np.ndarray,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Convert cartesian coordinates into spherical coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape ``(..., 3)`` containing cartesian ``(x, y, z)`` values.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used for the returned azimuth and elevation values.

    Returns
    -------
    np.ndarray
        Array with shape ``(..., 3)`` containing spherical
        ``(azimuth, elevation, radius)`` coordinates.

    Use Cases
    ---------
    - Inspect cartesian grids in SOFA-style azimuth and elevation.
    - Convert cartesian source data before plane or direction queries.

    Examples
    --------
    >>> cartesian_to_spherical(np.array([[1.0, 0.0, 0.0]]))
    array([[0., 0., 1.]])
    >>> cartesian_to_spherical(np.array([[0.0, 1.0, 0.0]]))
    array([[90., 0., 1.]])
    >>> cartesian_to_spherical(np.array([[0.0, 0.0, 1.0]]))
    array([[0., 90., 1.]])
    """
    cartesian = np.asarray(coordinates, dtype=float)
    if cartesian.shape[-1] != 3:
        raise ValueError("Cartesian coordinates must have shape (..., 3)")
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")

    x = cartesian[..., 0]
    y = cartesian[..., 1]
    z = cartesian[..., 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    azimuth = np.mod(azimuth, 2.0 * np.pi)

    if unit == "degrees":
        azimuth = np.rad2deg(azimuth)
        elevation = np.rad2deg(elevation)

    return np.stack((azimuth, elevation, radius), axis=-1)


def cartesian_to_lateral_polar(
    coordinates: np.ndarray,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Convert cartesian coordinates into lateral-polar coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape ``(..., 3)`` containing cartesian ``(x, y, z)`` values.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used for the returned lateral and polar values.

    Returns
    -------
    np.ndarray
        Array with shape ``(..., 3)`` containing lateral-polar
        ``(lateral, polar, radius)`` coordinates.

    Use Cases
    ---------
    - Convert cartesian source grids into interaural-style coordinates.
    - Prepare source data for median-plane analyses.

    Examples
    --------
    >>> cartesian_to_lateral_polar(np.array([[1.0, 0.0, 0.0]]))
    array([[0., 0., 1.]])
    >>> cartesian_to_lateral_polar(np.array([[0.0, 0.0, 1.0]]))
    array([[0., 90., 1.]])
    >>> cartesian_to_lateral_polar(np.array([[0.0, 1.0, 0.0]]))
    array([[90., 0., 1.]])
    """
    cartesian = np.asarray(coordinates, dtype=float)
    if cartesian.shape[-1] != 3:
        raise ValueError("Cartesian coordinates must have shape (..., 3)")
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")

    x = cartesian[..., 0]
    y = cartesian[..., 1]
    z = cartesian[..., 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    y_over_r = np.divide(y, radius, out=np.zeros_like(y), where=radius != 0.0)
    y_over_r = np.clip(y_over_r, -1.0, 1.0)
    lateral = np.arcsin(y_over_r)
    polar = np.arctan2(z, x)
    pole_mask = (radius == 0.0) | np.isclose(np.abs(lateral), np.pi / 2.0, atol=1e-12)
    polar = np.where(pole_mask, 0.0, polar)
    polar = np.where(polar < -np.pi / 2.0, polar + 2.0 * np.pi, polar)

    if unit == "degrees":
        lateral = np.rad2deg(lateral)
        polar = np.rad2deg(polar)

    return np.stack((lateral, polar, radius), axis=-1)


def lateral_polar_to_cartesian(
    coordinates: np.ndarray,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Convert lateral-polar coordinates into cartesian coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape ``(..., 3)`` containing
        ``(lateral, polar, radius)`` values.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit of the lateral and polar values.

    Returns
    -------
    np.ndarray
        Array with shape ``(..., 3)`` containing cartesian ``(x, y, z)``
        coordinates.

    Use Cases
    ---------
    - Convert interaural-style source definitions into cartesian form.
    - Feed lateral-polar datasets into 3D plotting or geometric processing.

    Examples
    --------
    >>> lateral_polar_to_cartesian(np.array([[0.0, 0.0, 1.0]]))
    array([[1., 0., 0.]])
    >>> lateral_polar_to_cartesian(np.array([[0.0, 90.0, 1.0]]))
    array([[0., 0., 1.]])
    >>> lateral_polar_to_cartesian(np.array([[90.0, 0.0, 1.0]]))
    array([[0., 1., 0.]])
    """
    lateral_polar = np.asarray(coordinates, dtype=float)
    if lateral_polar.shape[-1] != 3:
        raise ValueError("Lateral-polar coordinates must have shape (..., 3)")
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")

    lateral = lateral_polar[..., 0]
    polar = lateral_polar[..., 1]
    radius = lateral_polar[..., 2]
    if np.any(radius < 0.0):
        raise ValueError("Lateral-polar radius must be non-negative")

    if unit == "degrees":
        if np.any((lateral < -90.0) | (lateral > 90.0)):
            raise ValueError("Lateral angle must be in [-90, 90] degrees")
        polar = np.mod(polar + 90.0, 360.0) - 90.0
        lateral = np.deg2rad(lateral)
        polar = np.deg2rad(polar)
    else:
        half_pi = np.pi / 2.0
        if np.any((lateral < -half_pi) | (lateral > half_pi)):
            raise ValueError("Lateral angle must be in [-pi/2, pi/2] radians")
        polar = np.mod(polar + half_pi, 2.0 * np.pi) - half_pi

    cos_lateral = np.cos(lateral)
    x = radius * cos_lateral * np.cos(polar)
    y = radius * np.sin(lateral)
    z = radius * cos_lateral * np.sin(polar)
    return np.stack((x, y, z), axis=-1)


def spherical_to_lateral_polar(
    coordinates: np.ndarray,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Convert spherical coordinates into lateral-polar coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape ``(..., 3)`` containing spherical
        ``(azimuth, elevation, radius)`` values.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used for both the input and output angles.

    Returns
    -------
    np.ndarray
        Array with shape ``(..., 3)`` containing lateral-polar
        ``(lateral, polar, radius)`` coordinates.

    Use Cases
    ---------
    - Convert SOFA spherical data into interaural-style coordinates.
    - Prepare spherical grids for median-plane and lateral-angle workflows.

    Examples
    --------
    >>> spherical_to_lateral_polar(np.array([[0.0, 0.0, 1.0]]))
    array([[0., 0., 1.]])
    >>> spherical_to_lateral_polar(np.array([[90.0, 0.0, 1.0]]))
    array([[90., 0., 1.]])
    """
    cartesian = spherical_to_cartesian(coordinates, angle_unit=angle_unit)
    return cartesian_to_lateral_polar(cartesian, angle_unit=angle_unit)


def lateral_polar_to_spherical(
    coordinates: np.ndarray,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Convert lateral-polar coordinates into spherical coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape ``(..., 3)`` containing lateral-polar
        ``(lateral, polar, radius)`` values.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit used for both the input and output angles.

    Returns
    -------
    np.ndarray
        Array with shape ``(..., 3)`` containing spherical
        ``(azimuth, elevation, radius)`` coordinates.

    Use Cases
    ---------
    - Convert interaural-style coordinates back into SOFA-style angles.
    - Compare lateral-polar datasets with spherical source grids.

    Examples
    --------
    >>> lateral_polar_to_spherical(np.array([[0.0, 0.0, 1.0]]))
    array([[0., 0., 1.]])
    >>> lateral_polar_to_spherical(np.array([[0.0, 90.0, 1.0]]))
    array([[0., 90., 1.]])
    """
    cartesian = lateral_polar_to_cartesian(coordinates, angle_unit=angle_unit)
    return cartesian_to_spherical(cartesian, angle_unit=angle_unit)


def get_closest_position_index(
    query_position: np.ndarray | list[float] | tuple[float, ...],
    grid_positions: np.ndarray,
    coordinate_system: str = "cartesian",
    angle_unit: str = "degrees",
) -> int:
    """Return the index of the exact or nearest query match in a coordinate grid.

    Parameters
    ----------
    query_position : np.ndarray | list[float] | tuple[float, ...]
        Query coordinates. For spherical/lateral-polar, accepts ``(2,)`` angle-only
        or ``(3,)`` full coordinates. For cartesian, requires ``(3,)``.
    grid_positions : np.ndarray
        Candidate grid in ``coordinate_system`` with shape ``(N, 3)``.
    coordinate_system : {"spherical", "cartesian", "lateral-polar"}, default="cartesian"
        Coordinate system used by both query and grid.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Angular unit for spherical/lateral-polar cases.

    Returns
    -------
    int
        Exact-match index when available; otherwise nearest-match index.

    Use Cases
    ---------
    - Resolve numeric queries against a discrete source grid.
    - Match angle-only spherical or lateral-polar queries with wrap-aware distance.
    - Select the nearest available source position when an exact match is absent.

    Examples
    --------
    >>> get_closest_position_index(
    ...     query_position=[0.0, 0.0],
    ...     grid_positions=np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0]]),
    ...     coordinate_system="spherical",
    ... )
    0
    """
    system = str(coordinate_system).strip().lower()
    if system not in {"spherical", "cartesian", "lateral-polar"}:
        raise ValueError(
            "coordinate_system must be one of: spherical, cartesian, lateral-polar"
        )
    unit = str(angle_unit).strip().lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")

    grid = np.asarray(grid_positions, dtype=float)
    if grid.ndim != 2 or grid.shape[-1] != 3:
        raise ValueError("grid_positions must have shape (N, 3)")

    query = np.asarray(query_position, dtype=float)
    if system == "cartesian" and query.shape != (3,):
        raise ValueError("For cartesian, query_position must have shape (3,)")
    if system in {"spherical", "lateral-polar"} and query.shape not in {(2,), (3,)}:
        raise ValueError(
            "For spherical/lateral-polar, query_position must have shape (2,) or (3,)"
        )

    if system in {"spherical", "lateral-polar"} and query.shape == (2,):
        angle_deltas = grid[..., :2] - query
        first_angle_delta = angle_deltas[..., 0]
        second_angle_delta = angle_deltas[..., 1]
        full = 360.0 if unit == "degrees" else 2.0 * np.pi
        if system == "spherical":
            first_angle_delta = np.mod(first_angle_delta + full / 2.0, full) - full / 2.0
        else:
            second_angle_delta = np.mod(second_angle_delta + full / 2.0, full) - full / 2.0
        distances = np.sqrt(first_angle_delta**2 + second_angle_delta**2)
        exact_matches = np.where(np.isclose(distances, 0.0, atol=1e-8, rtol=0.0))[0]
        if exact_matches.size > 0:
            return int(exact_matches[0])
        return int(np.argmin(distances))

    if system == "cartesian":
        query_cartesian = query
        grid_cartesian = grid
    elif system == "spherical":
        query_cartesian = spherical_to_cartesian(query, angle_unit=unit)
        grid_cartesian = spherical_to_cartesian(grid, angle_unit=unit)
    else:
        query_cartesian = lateral_polar_to_cartesian(query, angle_unit=unit)
        grid_cartesian = lateral_polar_to_cartesian(grid, angle_unit=unit)

    deltas = grid_cartesian - query_cartesian
    distances = np.linalg.norm(deltas, axis=-1)
    exact_matches = np.where(np.isclose(distances, 0.0, atol=1e-8, rtol=0.0))[0]
    if exact_matches.size > 0:
        return int(exact_matches[0])
    return int(np.argmin(distances))
