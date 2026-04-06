from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hrtf import HRTF


class Sources:
    """Spatial-source view and coordinate-conversion utilities.

    Notes
    -----
    Conventions implemented by this class:

    - Spherical (SOFA-style): ``(azimuth, elevation, radius)`` with azimuth in
      ``[0, 360)`` degrees (anticlockwise in horizontal plane), elevation in
      ``[-90, 90]`` degrees (positive up), and non-negative radius.
    - Lateral-polar: ``(lateral, polar, radius)``
      with lateral in ``[-90, 90]`` degrees (positive left), polar normalized to
      ``[-90, 270)`` degrees, and non-negative radius.
    - Cartesian: ``(x, y, z)`` with ``+y`` as left and ``+z`` as up.

    At lateral poles (``|lateral| = 90``) and at zero radius, polar is singular.
    This implementation uses a deterministic placeholder ``polar = 0``.
    """

    def __init__(
        self,
        hrtf: "HRTF | None" = None,
    ) -> None:
        self._hrtf = hrtf
        self.source_coordinate_system = self._hrtf.Sofa.VariableAttributes.get("SourcePosition:Type").value
        self._selected_indices: np.ndarray | None = None

    def get_source_coordinate_system(self) -> str:
        """Return the currently selected source coordinate system.

        Returns
        -------
        str
            Coordinate-system name used as target by :meth:`get_positions`.
        """
        return str(self.source_coordinate_system)

    @staticmethod
    def get_position_queries(
        positions: np.ndarray | list | tuple | str,
    ) -> list[np.ndarray | str]:
        if isinstance(positions, str):
            return [str(positions).strip().lower()]

        if isinstance(positions, np.ndarray) and positions.dtype != object:
            numeric_positions = np.asarray(positions, dtype=float)
            if numeric_positions.ndim == 1:
                if numeric_positions.shape[0] not in {2, 3}:
                    raise ValueError(
                        "positions must have shape (2,), (3,), (K, 2), or (K, 3)"
                    )
                return [numeric_positions]
            if numeric_positions.ndim == 2 and numeric_positions.shape[-1] in {2, 3}:
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
            if position.ndim != 1 or position.shape[0] not in {2, 3}:
                raise ValueError(
                    "positions must have shape (2,), (3,), (K, 2), or (K, 3)"
                )
            return [position]

        if raw_positions.ndim == 1:
            values = raw_positions.tolist()
            if len(values) == 0:
                raise ValueError("At least one position is required")
            if all(isinstance(value, str) for value in values):
                return [str(value).strip().lower() for value in values]
            if all(np.isscalar(value) and not isinstance(value, str) for value in values):
                if raw_positions.shape[0] not in {2, 3}:
                    raise ValueError(
                        "positions must have shape (2,), (3,), (K, 2), or (K, 3)"
                    )
                return [np.asarray(raw_positions, dtype=float)]
            position_queries: list[np.ndarray | str] = []
            for value in values:
                if isinstance(value, str):
                    position_queries.append(str(value).strip().lower())
                    continue
                position = np.asarray(value, dtype=float)
                if position.ndim != 1 or position.shape[0] not in {2, 3}:
                    raise ValueError(
                        "positions must have shape (2,), (3,), (K, 2), or (K, 3)"
                    )
                position_queries.append(position)
            return position_queries

        if raw_positions.ndim == 2 and raw_positions.shape[-1] in {2, 3}:
            numeric_positions = np.asarray(raw_positions, dtype=float)
            return [
                np.asarray(position, dtype=float)
                for position in numeric_positions
            ]

        raise ValueError(
            "positions must have shape (2,), (3,), (K, 2), or (K, 3)"
        )

    @staticmethod
    def get_named_positions(
        angle_unit: str = "degrees",
    ) -> dict[str, np.ndarray]:
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

    @staticmethod
    def get_position_alias(
        position: np.ndarray | list[float] | tuple[float, ...],
        coordinate_system: str = "spherical",
        angle_unit: str = "degrees",
    ) -> str | None:
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
            spherical_position = Sources.cartesian_to_spherical(
                resolved_position,
                angle_unit=unit,
            )
        elif system == "lateral-polar":
            if resolved_position.shape not in {(2,), (3,)}:
                raise ValueError(
                    "For lateral-polar, position must have shape (2,) or (3,)"
                )
            spherical_position = Sources.lateral_polar_to_spherical(
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
        for name, named_position in Sources.get_named_positions(angle_unit=unit).items():
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
   
    @staticmethod
    def spherical_to_cartesian(
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """General Description:
        Convert spherical source coordinates into cartesian coordinates.

        The spherical convention used by this library is
        ``(azimuth, elevation, radius)``:

        - ``azimuth`` rotates in the horizontal plane from ``+x`` toward ``+y``
        - ``elevation`` measures vertical angle from the horizontal plane toward ``+z``
        - ``radius`` is the source distance from the origin

        The returned cartesian convention is ``(x, y, z)``, where ``+y`` points
        to the listener's left and ``+z`` points upward.

        Parameters:
        - coordinates: Array with shape ``(..., 3)`` containing
          ``(azimuth, elevation, radius)`` values.
        - angle_unit: Angular unit of ``azimuth`` and ``elevation``.

        Returns:
        - Array with shape ``(..., 3)`` containing cartesian
          ``(x, y, z)`` coordinates.

        Use Cases:
        - Convert SOFA spherical source grids into cartesian coordinates for 3D plotting.
        - Prepare spherical source positions for algorithms that operate in ``(x, y, z)``.

        Examples:
        >>> Sources.spherical_to_cartesian(np.array([[0.0, 0.0, 1.0]]))
        array([[1., 0., 0.]])
        >>> Sources.spherical_to_cartesian(np.array([[90.0, 0.0, 1.0]]))
        array([[0., 1., 0.]])
        >>> Sources.spherical_to_cartesian(np.array([[0.0, 90.0, 1.0]]))
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

    @staticmethod
    def cartesian_to_spherical(
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """General Description:
        Convert cartesian source coordinates into spherical coordinates.

        The input cartesian convention is ``(x, y, z)``. The returned spherical
        convention is ``(azimuth, elevation, radius)``, where azimuth is
        normalized to the canonical positive range and elevation stays in the
        vertical interval supported by the SOFA-style spherical definition.

        Parameters:
        - coordinates: Array with shape ``(..., 3)`` containing cartesian
          ``(x, y, z)`` values.
        - angle_unit: Angular unit used for the returned azimuth and elevation.

        Returns:
        - Array with shape ``(..., 3)`` containing spherical
          ``(azimuth, elevation, radius)`` coordinates.

        Use Cases:
        - Inspect a cartesian source grid in SOFA-style azimuth and elevation.
        - Convert cartesian data before selecting horizontal, median, or frontal planes.

        Examples:
        >>> Sources.cartesian_to_spherical(np.array([[1.0, 0.0, 0.0]]))
        array([[0., 0., 1.]])
        >>> Sources.cartesian_to_spherical(np.array([[0.0, 1.0, 0.0]]))
        array([[90., 0., 1.]])
        >>> Sources.cartesian_to_spherical(np.array([[0.0, 0.0, 1.0]]))
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

    @staticmethod
    def cartesian_to_lateral_polar(
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """General Description:
        Convert cartesian source coordinates into lateral-polar coordinates.

        The lateral-polar convention used here is
        ``(lateral, polar, radius)``:

        - ``lateral`` is positive to the left and negative to the right
        - ``polar`` describes the position around the interaural axis
        - ``radius`` is the distance from the origin

        This is useful when working with median-plane and interaural-style
        spatial representations instead of SOFA spherical coordinates.

        Parameters:
        - coordinates: Array with shape ``(..., 3)`` containing cartesian
          ``(x, y, z)`` values.
        - angle_unit: Angular unit used for ``lateral`` and ``polar``.

        Returns:
        - Array with shape ``(..., 3)`` containing lateral-polar
          ``(lateral, polar, radius)`` coordinates.

        Use Cases:
        - Convert cartesian source grids into interaural-style coordinates.
        - Prepare data for median-plane analyses where lateral angle is the primary descriptor.

        Examples:
        >>> Sources.cartesian_to_lateral_polar(np.array([[1.0, 0.0, 0.0]]))
        array([[0., 0., 1.]])
        >>> Sources.cartesian_to_lateral_polar(np.array([[0.0, 0.0, 1.0]]))
        array([[0., 90., 1.]])
        >>> Sources.cartesian_to_lateral_polar(np.array([[0.0, 1.0, 0.0]]))
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

    @staticmethod
    def lateral_polar_to_cartesian(
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """General Description:
        Convert lateral-polar coordinates into cartesian coordinates.

        The input convention is ``(lateral, polar, radius)`` and the output is
        cartesian ``(x, y, z)``. Lateral is limited to the interaural interval
        and polar is normalized internally to the range used by this library.

        Parameters:
        - coordinates: Array with shape ``(..., 3)`` containing
          ``(lateral, polar, radius)`` values.
        - angle_unit: Angular unit of ``lateral`` and ``polar``.

        Returns:
        - Array with shape ``(..., 3)`` containing cartesian
          ``(x, y, z)`` coordinates.

        Use Cases:
        - Convert interaural-style source definitions into a cartesian grid.
        - Feed lateral-polar datasets into 3D plotting or geometric processing code.

        Examples:
        >>> Sources.lateral_polar_to_cartesian(np.array([[0.0, 0.0, 1.0]]))
        array([[1., 0., 0.]])
        >>> Sources.lateral_polar_to_cartesian(np.array([[0.0, 90.0, 1.0]]))
        array([[0., 0., 1.]])
        >>> Sources.lateral_polar_to_cartesian(np.array([[90.0, 0.0, 1.0]]))
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

    @staticmethod
    def spherical_to_lateral_polar(
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """General Description:
        Convert spherical coordinates into lateral-polar coordinates.

        This conversion follows the library's full coordinate chain:
        spherical ``(azimuth, elevation, radius)``
        -> cartesian ``(x, y, z)``
        -> lateral-polar ``(lateral, polar, radius)``.

        Parameters:
        - coordinates: Array with shape ``(..., 3)`` containing
          spherical ``(azimuth, elevation, radius)`` values.
        - angle_unit: Angular unit used for both the input and output angles.

        Returns:
        - Array with shape ``(..., 3)`` containing lateral-polar
          ``(lateral, polar, radius)`` coordinates.

        Use Cases:
        - Compare SOFA spherical datasets against interaural-style spatial analyses.
        - Prepare spherical grids for median-plane or lateral-angle workflows.

        Examples:
        >>> Sources.spherical_to_lateral_polar(np.array([[0.0, 0.0, 1.0]]))
        array([[0., 0., 1.]])
        >>> Sources.spherical_to_lateral_polar(np.array([[90.0, 0.0, 1.0]]))
        array([[90., 0., 1.]])
        """
        cartesian = Sources.spherical_to_cartesian(coordinates, angle_unit=angle_unit)
        return Sources.cartesian_to_lateral_polar(cartesian, angle_unit=angle_unit)

    @staticmethod
    def lateral_polar_to_spherical(
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """General Description:
        Convert lateral-polar coordinates into spherical coordinates.

        This conversion follows the inverse chain:
        lateral-polar ``(lateral, polar, radius)``
        -> cartesian ``(x, y, z)``
        -> spherical ``(azimuth, elevation, radius)``.

        Parameters:
        - coordinates: Array with shape ``(..., 3)`` containing
          lateral-polar ``(lateral, polar, radius)`` values.
        - angle_unit: Angular unit used for both the input and output angles.

        Returns:
        - Array with shape ``(..., 3)`` containing spherical
          ``(azimuth, elevation, radius)`` coordinates.

        Use Cases:
        - Convert interaural-style coordinates back into SOFA-style azimuth and elevation.
        - Compare median-plane datasets with spherical source grids.

        Examples:
        >>> Sources.lateral_polar_to_spherical(np.array([[0.0, 0.0, 1.0]]))
        array([[0., 0., 1.]])
        >>> Sources.lateral_polar_to_spherical(np.array([[0.0, 90.0, 1.0]]))
        array([[0., 90., 1.]])
        """
        cartesian = Sources.lateral_polar_to_cartesian(coordinates, angle_unit=angle_unit)
        return Sources.cartesian_to_spherical(cartesian, angle_unit=angle_unit)

    def get_positions(
        self,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """Return source positions in the currently selected coordinate system.

        Parameters
        ----------
        angle_unit : {"degrees", "radians"}, default="degrees"
            Angular unit for angular coordinate systems.

        Returns
        -------
        np.ndarray
            Source grid with shape ``(N, 3)`` as float values.

        Notes
        -----
        Source data are read from SOFA ``SourcePosition`` and converted to
        ``self.source_coordinate_system``.
        """
        source_positions = self._hrtf.Sofa.Variables.get("SourcePosition").value
        source_system = self._hrtf.Sofa.VariableAttributes.get("SourcePosition:Type").value
        source_units = self._hrtf.Sofa.VariableAttributes.get("SourcePosition:Units").value
        target_system = self.source_coordinate_system

        requested_angle_unit = str(angle_unit).strip().lower()
        if requested_angle_unit not in {"degrees", "radians"}:
            raise ValueError("angle_unit must be 'degrees' or 'radians'")

        source_system = str(source_system).strip().lower()
        target_system = str(target_system).strip().lower()
        allowed_coordinate_systems = {"spherical", "cartesian", "lateral-polar"}
        if source_system not in allowed_coordinate_systems:
            raise ValueError(f"Unsupported source coordinate system: {source_system!r}")
        if target_system not in allowed_coordinate_systems:
            raise ValueError(f"Unsupported target coordinate system: {target_system!r}")

        source_units = str(source_units).strip().lower()
        source_angle_unit = "degrees"
        if "radian" in source_units:
            source_angle_unit = "radians"
        elif "degree" not in source_units:
            if source_system != "cartesian":
                raise ValueError(
                    "SourcePosition:Units must include degree or radian for angular coordinate systems"
                )

        source_positions = np.asarray(source_positions, dtype=float)
        if self._selected_indices is not None:
            source_positions = np.take(
                source_positions,
                np.asarray(self._selected_indices, dtype=int),
                axis=0,
            )

        if source_system == target_system:
            if target_system == "cartesian" or source_angle_unit == requested_angle_unit:
                return np.asarray(source_positions, dtype=float)
            if target_system == "spherical":
                cartesian = self.spherical_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                )
                return np.asarray(
                    self.cartesian_to_spherical(
                        cartesian,
                        angle_unit=requested_angle_unit,
                    ),
                    dtype=float,
                )
            cartesian = self.lateral_polar_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                self.cartesian_to_lateral_polar(
                    cartesian,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )

        if source_system == "spherical" and target_system == "cartesian":
            return np.asarray(
                self.spherical_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "cartesian" and target_system == "spherical":
            return np.asarray(
                self.cartesian_to_spherical(
                    source_positions,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "cartesian" and target_system == "lateral-polar":
            return np.asarray(
                self.cartesian_to_lateral_polar(
                    source_positions,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "lateral-polar" and target_system == "cartesian":
            return np.asarray(
                self.lateral_polar_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "spherical" and target_system == "lateral-polar":
            cartesian = self.spherical_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                self.cartesian_to_lateral_polar(
                    cartesian,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "lateral-polar" and target_system == "spherical":
            cartesian = self.lateral_polar_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                self.cartesian_to_spherical(
                    cartesian,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )

        raise ValueError(
            f"Unsupported conversion from {source_system!r} to {target_system!r}"
        )

    def get_azimuth_angles(
        self,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """Return unique source-grid azimuth angles.

        Parameters
        ----------
        angle_unit : {"degrees", "radians"}, default="degrees"
            Angular unit used when reading positions and returning azimuth values.

        Returns
        -------
        np.ndarray
            One-dimensional array of unique azimuth angles rounded to two decimals.

        Use Cases
        ---------
        - Inspect the available azimuth sampling of a source grid.
        - Build plane or panel selections from the source layout.
        - Validate angular coverage before plotting or interpolation.

        """
        target_system = str(self.source_coordinate_system).strip().lower()
        positions = self.get_positions(angle_unit=angle_unit)
        if target_system == "spherical":
            azimuth = positions[..., 0]
        elif target_system == "cartesian":
            spherical = self.cartesian_to_spherical(positions, angle_unit=angle_unit)
            azimuth = spherical[..., 0]
        elif target_system == "lateral-polar":
            cartesian = self.lateral_polar_to_cartesian(positions, angle_unit=angle_unit)
            spherical = self.cartesian_to_spherical(cartesian, angle_unit=angle_unit)
            azimuth = spherical[..., 0]
        else:
            raise ValueError(f"Unsupported target coordinate system: {target_system!r}")
        return np.unique(np.round(np.asarray(azimuth, dtype=float), 2))

    def get_elevation_angles(
        self,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """Return unique source-grid elevation angles.

        Parameters
        ----------
        angle_unit : {"degrees", "radians"}, default="degrees"
            Angular unit used when reading positions and returning elevation values.

        Returns
        -------
        np.ndarray
            One-dimensional array of unique elevation angles rounded to two decimals.

        Use Cases
        ---------
        - Inspect the available elevation sampling of a source grid.
        - Select horizontal slices from the source layout.
        - Validate vertical coverage before plotting or spatial selection.

        """
        target_system = str(self.source_coordinate_system).strip().lower()
        positions = self.get_positions(angle_unit=angle_unit)
        if target_system == "spherical":
            elevation = positions[..., 1]
        elif target_system == "cartesian":
            spherical = self.cartesian_to_spherical(positions, angle_unit=angle_unit)
            elevation = spherical[..., 1]
        elif target_system == "lateral-polar":
            cartesian = self.lateral_polar_to_cartesian(positions, angle_unit=angle_unit)
            spherical = self.cartesian_to_spherical(cartesian, angle_unit=angle_unit)
            elevation = spherical[..., 1]
        else:
            raise ValueError(f"Unsupported target coordinate system: {target_system!r}")
        return np.unique(np.round(np.asarray(elevation, dtype=float), 2))

    def get_elevation_angles_for_azimuth(
        self,
        azimuth: float,
        angle_unit: str = "degrees",
    ) -> tuple[np.ndarray, float]:
        """Return available elevation angles for the nearest azimuth in the grid.

        Parameters
        ----------
        azimuth : float
            Requested azimuth angle used to query the source grid.
        angle_unit : {"degrees", "radians"}, default="degrees"
            Angular unit for ``azimuth`` and returned elevation values.

        Returns
        -------
        tuple[np.ndarray, float]
            ``(elevation_angles, real_azimuth)`` where ``elevation_angles`` is a
            one-dimensional array of unique elevations available at the matched
            azimuth, and ``real_azimuth`` is the actual azimuth selected from the grid.

        Use Cases
        ---------
        - Inspect vertical sampling for a requested azimuth slice.
        - Build elevation selectors for plotting or interactive tools.
        - Query the real grid coverage before extracting directional data.

        """
        if isinstance(azimuth, bool):
            raise ValueError("azimuth must be a finite value")
        azimuth = float(azimuth)
        if not np.isfinite(azimuth):
            raise ValueError("azimuth must be a finite value")

        target_system = str(self.source_coordinate_system).strip().lower()
        positions = self.get_positions(angle_unit=angle_unit)
        if target_system == "spherical":
            spherical = positions
        elif target_system == "cartesian":
            spherical = self.cartesian_to_spherical(positions, angle_unit=angle_unit)
        elif target_system == "lateral-polar":
            cartesian = self.lateral_polar_to_cartesian(positions, angle_unit=angle_unit)
            spherical = self.cartesian_to_spherical(cartesian, angle_unit=angle_unit)
        else:
            raise ValueError(f"Unsupported target coordinate system: {target_system!r}")

        unit = str(angle_unit).strip().lower()
        if unit not in {"degrees", "radians"}:
            raise ValueError("angle_unit must be 'degrees' or 'radians'")

        azimuth_angles = np.asarray(spherical[..., 0], dtype=float)
        elevation_angles = np.asarray(spherical[..., 1], dtype=float)
        available_azimuths = np.unique(azimuth_angles)
        full = 360.0 if unit == "degrees" else 2.0 * np.pi
        half = full / 2.0
        azimuth_deltas = np.mod(available_azimuths - azimuth + half, full) - half
        real_azimuth = float(available_azimuths[int(np.argmin(np.abs(azimuth_deltas)))])
        selected = np.isclose(
            np.mod(azimuth_angles - real_azimuth + half, full) - half,
            0.0,
            atol=1e-8,
            rtol=0.0,
        )
        return np.unique(np.round(elevation_angles[selected], 2)), round(real_azimuth, 2)

    def get_azimuth_angles_for_elevation(
        self,
        elevation: float,
        angle_unit: str = "degrees",
    ) -> tuple[np.ndarray, float]:
        """Return available azimuth angles for the nearest elevation in the grid.

        Parameters
        ----------
        elevation : float
            Requested elevation angle used to query the source grid.
        angle_unit : {"degrees", "radians"}, default="degrees"
            Angular unit for ``elevation`` and returned azimuth values.

        Returns
        -------
        tuple[np.ndarray, float]
            ``(azimuth_angles, real_elevation)`` where ``azimuth_angles`` is a
            one-dimensional array of unique azimuths available at the matched
            elevation, and ``real_elevation`` is the actual elevation selected
            from the grid.

        Use Cases
        ---------
        - Inspect horizontal sampling for a requested elevation slice.
        - Build azimuth selectors for plotting or interactive tools.
        - Query the real grid coverage before extracting directional data.

        """
        if isinstance(elevation, bool):
            raise ValueError("elevation must be a finite value")
        elevation = float(elevation)
        if not np.isfinite(elevation):
            raise ValueError("elevation must be a finite value")

        target_system = str(self.source_coordinate_system).strip().lower()
        positions = self.get_positions(angle_unit=angle_unit)
        if target_system == "spherical":
            spherical = positions
        elif target_system == "cartesian":
            spherical = self.cartesian_to_spherical(positions, angle_unit=angle_unit)
        elif target_system == "lateral-polar":
            cartesian = self.lateral_polar_to_cartesian(positions, angle_unit=angle_unit)
            spherical = self.cartesian_to_spherical(cartesian, angle_unit=angle_unit)
        else:
            raise ValueError(f"Unsupported target coordinate system: {target_system!r}")

        elevation_angles = np.asarray(spherical[..., 1], dtype=float)
        azimuth_angles = np.asarray(spherical[..., 0], dtype=float)
        available_elevations = np.unique(elevation_angles)
        real_elevation = float(
            available_elevations[int(np.argmin(np.abs(available_elevations - elevation)))]
        )
        selected = np.isclose(
            elevation_angles,
            real_elevation,
            atol=1e-8,
            rtol=0.0,
        )
        return np.unique(np.round(azimuth_angles[selected], 2)), round(real_elevation, 2)

    @staticmethod
    def get_closest_position_index(
        query_position: np.ndarray | list[float] | tuple[float, ...],
        grid_positions: np.ndarray,
        coordinate_system: str = "cartesian",
        angle_unit: str = "degrees",
    ) -> int:
        """Return index of exact-or-nearest query match in a coordinate grid.

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

        Notes
        -----
        Wrap-aware angle distance is applied in angle-only mode for spherical azimuth
        and lateral-polar polar angle.
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
            query_cartesian = Sources.spherical_to_cartesian(query, angle_unit=unit)
            grid_cartesian = Sources.spherical_to_cartesian(grid, angle_unit=unit)
        else:
            query_cartesian = Sources.lateral_polar_to_cartesian(query, angle_unit=unit)
            grid_cartesian = Sources.lateral_polar_to_cartesian(grid, angle_unit=unit)

        deltas = grid_cartesian - query_cartesian
        distances = np.linalg.norm(deltas, axis=-1)
        exact_matches = np.where(np.isclose(distances, 0.0, atol=1e-8, rtol=0.0))[0]
        if exact_matches.size > 0:
            return int(exact_matches[0])
        return int(np.argmin(distances))

    def get_position_index(
        self,
        position: np.ndarray | list[float] | tuple[float, float, float] | str,
        coordinate_system: str = "spherical",
        angle_unit: str = "degrees",
    ) -> tuple[int, np.ndarray]:
        """Return matched source index and matched real position.

        Parameters
        ----------
        position : np.ndarray | list[float] | tuple[float, float, float]
            Query position in ``coordinate_system``.
        coordinate_system : {"spherical", "cartesian", "lateral-polar"}, default="spherical"
            Coordinate system of ``position`` and returned ``real_position``.
        angle_unit : {"degrees", "radians"}, default="degrees"
            Angular unit for spherical/lateral-polar inputs and outputs.

        Returns
        -------
        tuple[int, np.ndarray]
            ``(idx, real_position)`` where ``idx`` is the selected grid index and
            ``real_position`` is the selected grid coordinate rounded to two decimals.
        """
        system = str(coordinate_system).strip().lower()
        if system not in {"spherical", "cartesian", "lateral-polar"}:
            raise ValueError(
                "coordinate_system must be one of: spherical, cartesian, lateral-polar"
            )
        unit = str(angle_unit).strip().lower()
        if unit not in {"degrees", "radians"}:
            raise ValueError("angle_unit must be 'degrees' or 'radians'")

        grid_system = str(self.source_coordinate_system).strip().lower()
        grid_positions = self.get_positions(angle_unit=unit)
        if grid_positions.ndim != 2 or grid_positions.shape[-1] != 3:
            raise ValueError("Source positions grid must have shape (N, 3)")

        if grid_system == system:
            grid_in_query_system = grid_positions
        elif grid_system == "cartesian" and system == "spherical":
            grid_in_query_system = self.cartesian_to_spherical(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "cartesian" and system == "lateral-polar":
            grid_in_query_system = self.cartesian_to_lateral_polar(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "spherical" and system == "cartesian":
            grid_in_query_system = self.spherical_to_cartesian(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "spherical" and system == "lateral-polar":
            grid_in_query_system = self.spherical_to_lateral_polar(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "lateral-polar" and system == "cartesian":
            grid_in_query_system = self.lateral_polar_to_cartesian(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "lateral-polar" and system == "spherical":
            grid_in_query_system = self.lateral_polar_to_spherical(
                grid_positions,
                angle_unit=unit,
            )
        else:
            raise ValueError(
                f"Unsupported conversion from {grid_system!r} to {system!r}"
            )

        if isinstance(position, str):
            named_position = str(position).strip().lower()
            named_positions = Sources.get_named_positions(angle_unit=unit)
            if named_position not in named_positions:
                raise ValueError(
                    "named position accepts: front, back, left, or right"
                )
            if grid_system == "spherical":
                grid_in_spherical = grid_positions
            elif grid_system == "cartesian":
                grid_in_spherical = self.cartesian_to_spherical(
                    grid_positions,
                    angle_unit=unit,
                )
            elif grid_system == "lateral-polar":
                grid_in_spherical = self.lateral_polar_to_spherical(
                    grid_positions,
                    angle_unit=unit,
                )
            else:
                raise ValueError(
                    f"Unsupported conversion from {grid_system!r} to 'spherical'"
                )
            query_position = np.asarray(named_positions[named_position], dtype=float)
            idx = Sources.get_closest_position_index(
                query_position=query_position,
                grid_positions=grid_in_spherical,
                coordinate_system="spherical",
                angle_unit=unit,
            )
            return idx, np.round(np.asarray(grid_in_query_system[idx], dtype=float), 2)

        query_position = np.asarray(position, dtype=float)
        idx = Sources.get_closest_position_index(
            query_position=query_position,
            grid_positions=grid_in_query_system,
            coordinate_system=system,
            angle_unit=unit,
        )
        return idx, np.round(np.asarray(grid_in_query_system[idx], dtype=float), 2)




class Planes:
    """Plane-selection API for source grids."""

    def __init__(
        self,
        hrtf: "HRTF | None" = None,
    ) -> None:
        self._hrtf = hrtf

    def get_plane_indices(
        self,
        plane: str,
        angle: float = 0.0,
        angle_unit: str = "degrees",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return source indices for a requested plane and selected plane angles.

        Parameters
        ----------
        plane : {"horizontal", "median", "frontal"}
            Plane type to select.
        angle : float, default=0.0
            Target plane angle. For horizontal this is elevation. For
            median/frontal this is azimuth reference.
        angle_unit : {"degrees", "radians"}, default="degrees"
            Angular unit for ``angle`` and returned plane angles.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(indices, real_plane_angles)`` where:
            - ``indices`` are source-grid indices in the selected plane.
            - ``real_plane_angles`` are the actual angle(s) present in the grid
              used for selection.

        Notes
        -----
        If exact plane angles are not present in the grid, nearest available
        angle(s) are selected.
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

        grid_system = str(self._hrtf.Sources.get_source_coordinate_system()).strip().lower()
        grid_positions = self._hrtf.Sources.get_positions(angle_unit=unit)
        if grid_positions.ndim != 2 or grid_positions.shape[-1] != 3:
            raise ValueError("Source positions grid must have shape (N, 3)")

        if grid_system == "spherical":
            spherical_positions = grid_positions
        elif grid_system == "cartesian":
            spherical_positions = self._hrtf.Sources.cartesian_to_spherical(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "lateral-polar":
            spherical_positions = self._hrtf.Sources.lateral_polar_to_spherical(
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

    def get_horizontal_plane_indices(
        self,
        elevation: float = 0.0,
        angle_unit: str = "degrees",
    ) -> tuple[np.ndarray, float]:
        """Return indices of the horizontal plane nearest to requested elevation."""
        indices, real_plane_angles = self.get_plane_indices(
            plane="horizontal",
            angle=elevation,
            angle_unit=angle_unit,
        )
        return indices, float(real_plane_angles[0])

    def get_median_plane_indices(
        self,
        azimuth: float = 0.0,
        angle_unit: str = "degrees",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return indices of the median (sagittal) plane nearest to requested azimuth."""
        return self.get_plane_indices(
            plane="median",
            angle=azimuth,
            angle_unit=angle_unit,
        )

    def get_frontal_plane_indices(
        self,
        azimuth: float = 90.0,
        angle_unit: str = "degrees",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return indices of the frontal plane nearest to requested azimuth."""
        return self.get_plane_indices(
            plane="frontal",
            angle=azimuth,
            angle_unit=angle_unit,
        )

    
