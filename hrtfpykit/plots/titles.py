from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Titles:
    spherical_alias: str = "{name} : [Azimuth= {az}°, Elevation= {el}°]"
    spherical_position: str = "Position : [Azimuth= {az}°, Elevation= {el}°]"
    cartesian_alias: str = "{name} : [x= {x}, y= {y}, z= {z}]"
    cartesian_position: str = "Position : [x= {x}, y= {y}, z= {z}]"
    lateral_polar_alias: str = "{name} : [Lateral= {lateral}°, Polar= {polar}°]"
    lateral_polar_position: str = "Position : [Lateral= {lateral}°, Polar= {polar}°]"
    horizontal_plane: str = "Horizontal Plane"
    horizontal_plane_elevation: str = "Horizontal Plane : [Elevation= {angle}°]"
    median_plane: str = "Median Plane"
    elevation_spectrum: str = "Elevation Spectrum : [Azimuth= {angle}°]"
