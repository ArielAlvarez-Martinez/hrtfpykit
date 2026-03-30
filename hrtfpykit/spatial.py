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

    def spherical_to_cartesian(
        self,
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

        if unit == "degrees":
            azimuth = np.deg2rad(azimuth)
            elevation = np.deg2rad(elevation)

        cos_elevation = np.cos(elevation)
        x = radius * cos_elevation * np.cos(azimuth)
        y = radius * cos_elevation * np.sin(azimuth)
        z = radius * np.sin(elevation)
        return np.stack((x, y, z), axis=-1)

    def cartesian_to_spherical(
        self,
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

        if unit == "degrees":
            azimuth = np.rad2deg(azimuth)
            elevation = np.rad2deg(elevation)

        return np.stack((azimuth, elevation, radius), axis=-1)

    def cartesian_to_lateral_polar(
        self,
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
        lateral = np.arcsin(np.divide(y, radius, out=np.zeros_like(y), where=radius != 0.0))
        polar = np.arctan2(z, x)

        if unit == "degrees":
            lateral = np.rad2deg(lateral)
            polar = np.rad2deg(polar)

        return np.stack((lateral, polar, radius), axis=-1)

    def lateral_polar_to_cartesian(
        self,
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

        if unit == "degrees":
            lateral = np.deg2rad(lateral)
            polar = np.deg2rad(polar)

        cos_lateral = np.cos(lateral)
        x = radius * cos_lateral * np.cos(polar)
        y = radius * np.sin(lateral)
        z = radius * cos_lateral * np.sin(polar)
        return np.stack((x, y, z), axis=-1)

    def spherical_to_lateral_polar(
        self,
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        cartesian = self.spherical_to_cartesian(coordinates, angle_unit=angle_unit)
        return self.cartesian_to_lateral_polar(cartesian, angle_unit=angle_unit)

    def lateral_polar_to_spherical(
        self,
        coordinates: np.ndarray,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        cartesian = self.lateral_polar_to_cartesian(coordinates, angle_unit=angle_unit)
        return self.cartesian_to_spherical(cartesian, angle_unit=angle_unit)

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
    
    @property
    def azimuth_angles(self) -> np.ndarray | None:
        pass

    @property
    def elevation_angles(self) -> np.ndarray | None:
        pass
    




class Planes:
    """Plane-selection helper """

    def __init__(
        self,
        hrtf: "HRTF | None" = None,
    ) -> None:
        self._hrtf = hrtf

    
