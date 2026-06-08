from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ..hrtf.sources import Sources


def get_position_queries(
    positions: np.ndarray | list | tuple | str,
) -> list[np.ndarray | str]:
    """Normalize user-facing position queries into a predictable list.

    Position queries are accepted throughout hrtfpykit by source selection,
    plotting, metrics, and spherical-harmonic helpers. This function converts
    those flexible inputs into a uniform list while preserving the order in
    which the caller provided them. Named positions are returned as stripped,
    lowercase strings. Numeric positions are converted to floating-point
    arrays and may contain either angular coordinates only, (a, b), or a
    full coordinate triplet, (a, b, r).

    Parameters
    ----------
    positions : np.ndarray | list | tuple | str
        Single query or collection of queries. Accepted forms include a named
        query such as ``front``, a single numeric position with shape
        (2,) or (3,), a numeric array with shape (K, 2) or
        (K, 3), a sequence of named queries, or a mixed sequence containing
        named and numeric queries.

    Returns
    -------
    list[np.ndarray | str]
        Normalized position queries. String entries are stripped and
        lowercased. Numeric entries are returned as float arrays with
        shape (2,) or (3,).

    Raises
    ------
    ValueError
        If an empty one-dimensional collection is provided, or if a numeric
        query does not have shape (2,), (3,), (K, 2), or
        (K, 3).

    Notes
    -----
    This function only normalizes query structure. It does not check whether a
    named query exists in a source grid, whether a numeric position is finite,
    or whether the position belongs to a specific coordinate system.
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
    """Return canonical horizontal named positions.

    The returned positions are the built-in aliases used by hrtfpykit when a
    caller requests directions such as ``front`` or ``left``. They are
    expressed as spherical angle pairs (azimuth, elevation) without an
    explicit radius. The convention matches the rest of the HRTF source-grid
    code: front is azimuth 0, left is 90, back is 180, right is
    270, and all four aliases lie on the horizontal plane with elevation
    0.

    Parameters
    ----------
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used for the returned azimuth and elevation values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from {``front``, ``left``, ``back``, ``right``} to spherical
        (azimuth, elevation) position arrays.

    Raises
    ------
    ValueError
        If angle_unit is not ``degrees`` or ``radians``.

    Notes
    -----
    Radius is intentionally omitted because these aliases represent directions
    rather than physical measurement distances. Callers that need full
    spherical coordinates can append the radius appropriate for their source
    grid before matching.
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
    """Resolve a position to a built-in horizontal alias when possible.

    This helper maps positions onto the canonical names returned by
    :func:`~hrtfpykit.hrtf.coordinates.get_named_positions`. It is used by plot titles and other
    user-facing labels to display stable names for exact cardinal directions.
    Spherical inputs may be angle-only (2,) positions or full (3,)
    positions; the radius component is ignored for alias matching. Cartesian
    inputs must be full (x, y, z) triplets. Lateral-polar inputs are
    converted through the lateral-polar conversion path, so use full
    (lateral, polar, radius) triplets for that coordinate system.

    Parameters
    ----------
    position : np.ndarray | list[float] | tuple[float, ...]
        Position to evaluate. Shape must be (3,) for cartesian and
        lateral-polar inputs. Spherical inputs may use shape (2,) or
        (3,).
    coordinate_system : {``spherical``, ``cartesian``, ``lateral-polar``}, default=``spherical``
        Coordinate system used by position.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit for spherical and lateral-polar inputs and for the
        returned internal comparison grid.

    Returns
    -------
    str | None
        One of {``front``, ``left``, ``back``, ``right``} when the position
        matches a canonical horizontal cardinal direction. Returns None for
        valid positions that are not on those four horizontal directions.

    Raises
    ------
    ValueError
        If coordinate_system or angle_unit is unsupported, if the input
        shape is invalid for the coordinate system, if the coordinate
        conversion fails, or if azimuth/elevation values are not finite.

    Notes
    -----
    Matching uses a small absolute tolerance and wraps azimuth before
    comparison, so equivalent angles such as 0 and 360 degrees resolve
    to the same alias. Elevation must be approximately zero; elevated or
    depressed directions are valid positions but do not receive a cardinal
    alias.
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
    """Return an HRTF source grid expressed in spherical coordinates.

    :class:`~hrtfpykit.hrtf.sources.Sources` objects may expose their grid in
    spherical, cartesian, or lateral-polar coordinates depending on
    :attr:`~hrtfpykit.hrtf.sources.Sources.source_coordinate_system`. This helper normalizes that view to spherical
    (azimuth, elevation, radius) coordinates so algorithms such as
    metrics, spherical harmonics, plane extraction, and diffuse-field
    weighting can work from a consistent representation.

    Parameters
    ----------
    sources : Sources
        Source-grid manager attached to an :class:`~hrtfpykit.hrtf.HRTF`
        object. The object must provide
        :meth:`~hrtfpykit.hrtf.sources.Sources.get_positions` and
        :attr:`~hrtfpykit.hrtf.sources.Sources.source_coordinate_system`.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used for returned azimuth and elevation values and for
        angular conversions performed while normalizing the grid.

    Returns
    -------
    np.ndarray
        Source positions with shape (N, 3) in spherical
        (azimuth, elevation, radius) coordinates.

    Raises
    ------
    ValueError
        If the source coordinate system is not ``spherical``,
        ``cartesian``, or ``lateral-polar``, or if the needed coordinate
        conversion rejects the source positions or angle unit.

    Notes
    -----
    The returned grid reflects the current
    :class:`~hrtfpykit.hrtf.sources.Sources` view. If the owning HRTF has been
    spatially selected,
    :meth:`~hrtfpykit.hrtf.sources.Sources.get_positions` is responsible for
    applying that selection before this function converts the coordinates.
    """
    target_system = str(sources.source_coordinate_system).strip().lower()
    positions = sources.get_positions(
        angle_unit=angle_unit,
        coordinate_system=target_system,
    )
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


def get_source_positions(
    sources: "Sources",
    coordinate_system: str,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Return an HRTF source grid in a requested coordinate system.

    This is the general source-grid conversion helper used by plotting,
    comparison, and spherical-harmonic code. It reads the current positions
    from a :class:`~hrtfpykit.hrtf.sources.Sources` manager, validates that
    they form an (M, 3) grid, and converts them from
    :attr:`~hrtfpykit.hrtf.sources.Sources.source_coordinate_system` into the
    requested target coordinate system when necessary.

    Parameters
    ----------
    sources : Sources
        Source-grid manager attached to an :class:`~hrtfpykit.hrtf.HRTF`
        object. The object must provide
        :meth:`~hrtfpykit.hrtf.sources.Sources.get_positions` and
        :attr:`~hrtfpykit.hrtf.sources.Sources.source_coordinate_system`.
    coordinate_system : {``spherical``, ``cartesian``, ``lateral-polar``}
        Coordinate system requested for the returned grid.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used when reading and returning spherical or
        lateral-polar coordinates. Cartesian coordinates are unaffected by
        angular units.

    Returns
    -------
    np.ndarray
        Source positions with shape (M, 3) in coordinate_system.
        Columns are (azimuth, elevation, radius) for spherical,
        (x, y, z) for cartesian, and (lateral, polar, radius) for
        lateral-polar coordinates.

    Raises
    ------
    ValueError
        If source positions do not have shape (M, 3), if the current or
        target coordinate system is unsupported, if angle_unit is
        unsupported by a conversion, or if a coordinate array violates the
        validation rules of the conversion being applied.

    Notes
    -----
    The function delegates the initial read to
    :meth:`~hrtfpykit.hrtf.sources.Sources.get_positions`. That
    means selected HRTF views and SOFA unit conversion are handled by
    :class:`~hrtfpykit.hrtf.sources.Sources` before this function performs the
    final coordinate-system normalization.
    """
    source_system = str(sources.source_coordinate_system).strip().lower()
    source_positions = np.asarray(
        sources.get_positions(
            angle_unit=angle_unit,
            coordinate_system=source_system,
        ),
        dtype=float,
    )
    if (
        source_positions.ndim != 2
        or source_positions.shape[0] == 0
        or source_positions.shape[1] != 3
    ):
        raise ValueError("Source positions must have shape (M, 3)")

    target_system = str(coordinate_system).strip().lower()
    if target_system == source_system:
        return source_positions
    if target_system == "spherical":
        return get_spherical_positions(
            sources=sources,
            angle_unit=angle_unit,
        )
    if target_system == "cartesian":
        if source_system == "spherical":
            return spherical_to_cartesian(
                source_positions,
                angle_unit=angle_unit,
            )
        if source_system == "lateral-polar":
            return lateral_polar_to_cartesian(
                source_positions,
                angle_unit=angle_unit,
            )
    if target_system == "lateral-polar":
        if source_system == "cartesian":
            return cartesian_to_lateral_polar(
                source_positions,
                angle_unit=angle_unit,
            )
        if source_system == "spherical":
            return spherical_to_lateral_polar(
                source_positions,
                angle_unit=angle_unit,
            )
    raise ValueError(
        f"Unsupported source coordinate system conversion: {source_system!r} -> {target_system!r}"
    )


def spherical_to_cartesian(
    coordinates: np.ndarray,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Convert spherical HRTF coordinates into cartesian coordinates.

    Spherical coordinates use the SOFA-style convention implemented across
    hrtfpykit: azimuth rotates in the horizontal plane, elevation is
    positive upward from the horizontal plane, and radius is the
    non-negative distance from the listener. Cartesian output uses +x for
    front, +y for left, and +z for up.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape (..., 3) containing spherical
        (azimuth, elevation, radius) values. Any leading dimensions are
        preserved.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit of the input azimuth and elevation values.

    Returns
    -------
    np.ndarray
        Array with shape (..., 3) containing cartesian (x, y, z)
        coordinates.

    Raises
    ------
    ValueError
        If coordinates does not end with length 3, if angle_unit is
        unsupported, if any radius is negative, or if elevation is outside
        [-90, 90] degrees or [-pi/2, pi/2] radians.

    Notes
    -----
    Azimuth is normalized modulo a full turn before conversion. This makes
    equivalent directions such as -90 and 270 degrees produce the same
    cartesian result.
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
    """Convert cartesian HRTF coordinates into spherical coordinates.

    Cartesian coordinates are interpreted with the hrtfpykit listener-centered
    axes: +x points front, +y points left, and +z points up. The
    returned spherical coordinates follow the SOFA-style
    (azimuth, elevation, radius) convention used by :class:`~hrtfpykit.hrtf.sources.Sources` and the
    HRTF processing helpers.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape (..., 3) containing cartesian (x, y, z)
        values. Any leading dimensions are preserved.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used for the returned azimuth and elevation values.

    Returns
    -------
    np.ndarray
        Array with shape (..., 3) containing spherical
        (azimuth, elevation, radius) coordinates.

    Raises
    ------
    ValueError
        If coordinates does not end with length 3 or if
        angle_unit is not ``degrees`` or ``radians``.

    Notes
    -----
    Azimuth is normalized to [0, 360) degrees or [0, 2*pi) radians.
    Elevation is measured from the horizontal plane. The zero vector is
    converted deterministically to (0, 0, 0) because both arctan2 calls
    receive zero-valued inputs.
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
    """Convert cartesian HRTF coordinates into lateral-polar coordinates.

    Lateral-polar coordinates represent each source as
    (lateral, polar, radius). lateral is positive toward the left ear,
    polar rotates in the median plane from the front direction, and
    radius is the non-negative distance from the listener. Cartesian input
    uses the same listener-centered axes as the rest of hrtfpykit:
    +x front, +y left, and +z up.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape (..., 3) containing cartesian (x, y, z)
        values. Any leading dimensions are preserved.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used for the returned lateral and polar values.

    Returns
    -------
    np.ndarray
        Array with shape (..., 3) containing lateral-polar
        (lateral, polar, radius) coordinates.

    Raises
    ------
    ValueError
        If coordinates does not end with length 3 or if
        angle_unit is not ``degrees`` or ``radians``.

    Notes
    -----
    The returned lateral angle lies in [-90, 90] degrees or
    [-pi/2, pi/2] radians. Polar is normalized to [-90, 270) degrees or
    [-pi/2, 3*pi/2) radians. At zero radius and at the lateral poles,
    polar is singular; this implementation returns polar = 0 for those
    positions so downstream code receives deterministic coordinates.
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
    """Convert lateral-polar HRTF coordinates into cartesian coordinates.

    Lateral-polar input is interpreted as (lateral, polar, radius) where
    lateral is positive left, polar rotates in the median plane from
    the front direction, and radius is the distance from the listener. The
    cartesian output uses +x for front, +y for left, and +z for up.

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape (..., 3) containing
        (lateral, polar, radius) values. Any leading dimensions are
        preserved.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit of the lateral and polar values.

    Returns
    -------
    np.ndarray
        Array with shape (..., 3) containing cartesian (x, y, z)
        coordinates.

    Raises
    ------
    ValueError
        If coordinates does not end with length 3, if angle_unit is
        unsupported, if any radius is negative, or if lateral angle is outside
        [-90, 90] degrees or [-pi/2, pi/2] radians.

    Notes
    -----
    Polar is normalized to [-90, 270) degrees or [-pi/2, 3*pi/2)
    radians before conversion. This preserves equivalent polar directions
    while keeping the lateral-polar convention consistent with
    :func:`~hrtfpykit.hrtf.coordinates.cartesian_to_lateral_polar`.
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
    """Convert spherical HRTF coordinates into lateral-polar coordinates.

    This convenience conversion uses cartesian coordinates as the intermediate
    representation. The input follows the SOFA-style spherical convention
    (azimuth, elevation, radius) and the output follows the hrtfpykit
    lateral-polar convention (lateral, polar, radius).

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape (..., 3) containing spherical
        (azimuth, elevation, radius) values. Any leading dimensions are
        preserved.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used for both the input and output angles.

    Returns
    -------
    np.ndarray
        Array with shape (..., 3) containing lateral-polar
        (lateral, polar, radius) coordinates.

    Raises
    ------
    ValueError
        If the spherical input fails validation in
        :func:`~hrtfpykit.hrtf.coordinates.spherical_to_cartesian`, or if the
        intermediate cartesian data cannot be converted by
        :func:`~hrtfpykit.hrtf.coordinates.cartesian_to_lateral_polar`.

    Notes
    -----
    The same angle unit is used for input and output. Azimuth and polar are
    normalized by the underlying conversion functions.
    """
    cartesian = spherical_to_cartesian(coordinates, angle_unit=angle_unit)
    return cartesian_to_lateral_polar(cartesian, angle_unit=angle_unit)


def lateral_polar_to_spherical(
    coordinates: np.ndarray,
    angle_unit: str = "degrees",
) -> np.ndarray:
    """Convert lateral-polar HRTF coordinates into spherical coordinates.

    This convenience conversion uses cartesian coordinates as the intermediate
    representation. The input follows the hrtfpykit lateral-polar convention
    (lateral, polar, radius) and the output follows the SOFA-style
    spherical convention (azimuth, elevation, radius).

    Parameters
    ----------
    coordinates : np.ndarray
        Array with shape (..., 3) containing lateral-polar
        (lateral, polar, radius) values. Any leading dimensions are
        preserved.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit used for both the input and output angles.

    Returns
    -------
    np.ndarray
        Array with shape (..., 3) containing spherical
        (azimuth, elevation, radius) coordinates.

    Raises
    ------
    ValueError
        If the lateral-polar input fails validation in
        :func:`~hrtfpykit.hrtf.coordinates.lateral_polar_to_cartesian`, or if the
        intermediate cartesian data cannot be converted by
        :func:`~hrtfpykit.hrtf.coordinates.cartesian_to_spherical`.

    Notes
    -----
    The same angle unit is used for input and output. Polar and azimuth are
    normalized by the underlying conversion functions.
    """
    cartesian = lateral_polar_to_cartesian(coordinates, angle_unit=angle_unit)
    return cartesian_to_spherical(cartesian, angle_unit=angle_unit)


def get_closest_position_index(
    query_position: np.ndarray | list[float] | tuple[float, ...],
    grid_positions: np.ndarray,
    coordinate_system: str = "cartesian",
    angle_unit: str = "degrees",
) -> int:
    """Return the index of the exact or nearest position in a source grid.

    This helper resolves numeric position queries against an HRTF source grid.
    It first returns the first exact match when one is available within a small
    tolerance. Otherwise it returns the nearest candidate. Full 3-D queries are
    compared in cartesian space so radius contributes to the distance. Angle
    only spherical or lateral-polar queries are compared in angular space and
    ignore radius.

    Parameters
    ----------
    query_position : np.ndarray | list[float] | tuple[float, ...]
        Query coordinates. For spherical and lateral-polar grids, accepts
        (2,) angle-only queries or full (3,) coordinates. For
        cartesian grids, requires a full (3,) coordinate.
    grid_positions : np.ndarray
        Candidate source grid in coordinate_system with shape (N, 3).
    coordinate_system : {``spherical``, ``cartesian``, ``lateral-polar``}, default=``cartesian``
        Coordinate system used by both query and grid.
    angle_unit : {``degrees``, ``radians``}, default=``degrees``
        Angular unit for spherical and lateral-polar queries and grids.

    Returns
    -------
    int
        Index of the first exact match when available; otherwise the index of
        the nearest candidate in grid_positions.

    Raises
    ------
    ValueError
        If coordinate_system or angle_unit is unsupported, if
        grid_positions does not have shape (N, 3), if
        query_position has an invalid shape for the coordinate system, if a
        coordinate conversion rejects the query or grid values, or if no grid
        candidate can be selected.

    Notes
    -----
    For spherical angle-only queries, azimuth wraps around a full turn before
    angular distance is measured. For lateral-polar angle-only queries, polar
    wraps around a full turn. Full spherical and lateral-polar queries are
    converted to cartesian coordinates before distance calculation.
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
