from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hrtf import HRTF


class Sources:
    def __init__(
        self,
        hrtf: "HRTF | None" = None,
    ) -> None:
        self._hrtf = hrtf
        self.source_coordinate_system = self._hrtf.Sofa.VariableAttributes.get("SourcePosition:Type").value
        self._positions = self.get_positions()

    def get_source_coordinate_system(self) -> str:
        return str(self.source_coordinate_system)
   
    def get_azimuth_angles(
        self,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
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

    @staticmethod
    def spherical_to_cartesian(
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
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
        cartesian = Sources.spherical_to_cartesian(coordinates, angle_unit=angle_unit)
        return Sources.cartesian_to_lateral_polar(cartesian, angle_unit=angle_unit)

    @staticmethod
    def lateral_polar_to_spherical(
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        cartesian = Sources.lateral_polar_to_cartesian(coordinates, angle_unit=angle_unit)
        return Sources.cartesian_to_spherical(cartesian, angle_unit=angle_unit)

    def get_positions(
        self,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        def _round(values: np.ndarray) -> np.ndarray:
            return np.round(np.asarray(values, dtype=float), 2)

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

        if source_system == target_system:
            if target_system == "cartesian" or source_angle_unit == requested_angle_unit:
                return _round(source_positions)
            if target_system == "spherical":
                cartesian = self.spherical_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                )
                return _round(
                    self.cartesian_to_spherical(
                        cartesian,
                        angle_unit=requested_angle_unit,
                    )
                )
            cartesian = self.lateral_polar_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return _round(
                self.cartesian_to_lateral_polar(
                    cartesian,
                    angle_unit=requested_angle_unit,
                )
            )

        if source_system == "spherical" and target_system == "cartesian":
            return _round(
                self.spherical_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                )
            )
        if source_system == "cartesian" and target_system == "spherical":
            return _round(
                self.cartesian_to_spherical(
                    source_positions,
                    angle_unit=requested_angle_unit,
                )
            )
        if source_system == "cartesian" and target_system == "lateral-polar":
            return _round(
                self.cartesian_to_lateral_polar(
                    source_positions,
                    angle_unit=requested_angle_unit,
                )
            )
        if source_system == "lateral-polar" and target_system == "cartesian":
            return _round(
                self.lateral_polar_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                )
            )
        if source_system == "spherical" and target_system == "lateral-polar":
            cartesian = self.spherical_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return _round(
                self.cartesian_to_lateral_polar(
                    cartesian,
                    angle_unit=requested_angle_unit,
                )
            )
        if source_system == "lateral-polar" and target_system == "spherical":
            cartesian = self.lateral_polar_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return _round(
                self.cartesian_to_spherical(
                    cartesian,
                    angle_unit=requested_angle_unit,
                )
            )

        raise ValueError(
            f"Unsupported conversion from {source_system!r} to {target_system!r}"
        )

    @staticmethod
    def get_closest_position_index(
        query_position: np.ndarray | list[float] | tuple[float, ...],
        grid_positions: np.ndarray,
        coordinate_system: str = "cartesian",
        angle_unit: str = "degrees",
    ) -> int:
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
        position: np.ndarray | list[float] | tuple[float, float, float],
        coordinate_system: str = "spherical",
        angle_unit: str = "degrees",
    ) -> tuple[int, np.ndarray]:
        system = str(coordinate_system).strip().lower()
        if system not in {"spherical", "cartesian", "lateral-polar"}:
            raise ValueError(
                "coordinate_system must be one of: spherical, cartesian, lateral-polar"
            )
        unit = str(angle_unit).strip().lower()
        if unit not in {"degrees", "radians"}:
            raise ValueError("angle_unit must be 'degrees' or 'radians'")

        query_position = np.asarray(position, dtype=float)
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

        idx = Sources.get_closest_position_index(
            query_position=query_position,
            grid_positions=grid_in_query_system,
            coordinate_system=system,
            angle_unit=unit,
        )
        return idx, np.round(np.asarray(grid_in_query_system[idx], dtype=float), 2)




class Planes:
    """Plane-selection helper """

    def __init__(
        self,
        hrtf: "HRTF | None" = None,
    ) -> None:
        self._hrtf = hrtf

    
