from typing import TYPE_CHECKING

import numpy as np
from .coordinates import (
    cartesian_to_lateral_polar,
    cartesian_to_spherical,
    get_closest_position_index,
    get_named_positions,
    get_spherical_positions,
    lateral_polar_to_cartesian,
    lateral_polar_to_spherical,
    spherical_to_cartesian,
    spherical_to_lateral_polar,
)

if TYPE_CHECKING:
    from .hrtf import HRTF


class Sources:
    """Spatial-source view and source-grid utilities.

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
                cartesian = spherical_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                )
                return np.asarray(
                    cartesian_to_spherical(
                        cartesian,
                        angle_unit=requested_angle_unit,
                    ),
                    dtype=float,
                )
            cartesian = lateral_polar_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                cartesian_to_lateral_polar(
                    cartesian,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )

        if source_system == "spherical" and target_system == "cartesian":
            return np.asarray(
                spherical_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "cartesian" and target_system == "spherical":
            return np.asarray(
                cartesian_to_spherical(
                    source_positions,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "cartesian" and target_system == "lateral-polar":
            return np.asarray(
                cartesian_to_lateral_polar(
                    source_positions,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "lateral-polar" and target_system == "cartesian":
            return np.asarray(
                lateral_polar_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "spherical" and target_system == "lateral-polar":
            cartesian = spherical_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                cartesian_to_lateral_polar(
                    cartesian,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "lateral-polar" and target_system == "spherical":
            cartesian = lateral_polar_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                cartesian_to_spherical(
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

        """
        spherical = get_spherical_positions(self, angle_unit=angle_unit)
        azimuth = spherical[..., 0]
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

        """
        spherical = get_spherical_positions(self, angle_unit=angle_unit)
        elevation = spherical[..., 1]
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

        """
        if isinstance(azimuth, bool):
            raise ValueError("azimuth must be a finite value")
        azimuth = float(azimuth)
        if not np.isfinite(azimuth):
            raise ValueError("azimuth must be a finite value")

        spherical = get_spherical_positions(self, angle_unit=angle_unit)

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

        """
        if isinstance(elevation, bool):
            raise ValueError("elevation must be a finite value")
        elevation = float(elevation)
        if not np.isfinite(elevation):
            raise ValueError("elevation must be a finite value")

        spherical = get_spherical_positions(self, angle_unit=angle_unit)

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
            grid_in_query_system = cartesian_to_spherical(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "cartesian" and system == "lateral-polar":
            grid_in_query_system = cartesian_to_lateral_polar(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "spherical" and system == "cartesian":
            grid_in_query_system = spherical_to_cartesian(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "spherical" and system == "lateral-polar":
            grid_in_query_system = spherical_to_lateral_polar(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "lateral-polar" and system == "cartesian":
            grid_in_query_system = lateral_polar_to_cartesian(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "lateral-polar" and system == "spherical":
            grid_in_query_system = lateral_polar_to_spherical(
                grid_positions,
                angle_unit=unit,
            )
        else:
            raise ValueError(
                f"Unsupported conversion from {grid_system!r} to {system!r}"
            )

        if isinstance(position, str):
            named_position = str(position).strip().lower()
            named_positions = get_named_positions(angle_unit=unit)
            if named_position not in named_positions:
                raise ValueError(
                    "named position accepts: front, back, left, or right"
                )
            if grid_system == "spherical":
                grid_in_spherical = grid_positions
            elif grid_system == "cartesian":
                grid_in_spherical = cartesian_to_spherical(
                    grid_positions,
                    angle_unit=unit,
                )
            elif grid_system == "lateral-polar":
                grid_in_spherical = lateral_polar_to_spherical(
                    grid_positions,
                    angle_unit=unit,
                )
            else:
                raise ValueError(
                    f"Unsupported conversion from {grid_system!r} to 'spherical'"
                )
            query_position = np.asarray(named_positions[named_position], dtype=float)
            idx = get_closest_position_index(
                query_position=query_position,
                grid_positions=grid_in_spherical,
                coordinate_system="spherical",
                angle_unit=unit,
            )
            return idx, np.round(np.asarray(grid_in_query_system[idx], dtype=float), 2)

        query_position = np.asarray(position, dtype=float)
        idx = get_closest_position_index(
            query_position=query_position,
            grid_positions=grid_in_query_system,
            coordinate_system=system,
            angle_unit=unit,
        )
        return idx, np.round(np.asarray(grid_in_query_system[idx], dtype=float), 2)
