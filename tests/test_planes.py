import numpy as np

from hrtfpykit.coordinates import spherical_to_cartesian
from hrtfpykit.planes import (
    get_frontal_plane,
    get_horizontal_plane,
    get_median_plane,
)


class DummySources:
    def __init__(
        self,
        positions: np.ndarray,
        source_coordinate_system: str = "spherical",
    ) -> None:
        self._positions = np.asarray(positions, dtype=float)
        self.source_coordinate_system = source_coordinate_system

    def get_positions(
        self,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        if angle_unit != "degrees":
            raise ValueError("DummySources only supports degrees in tests")
        return np.asarray(self._positions, dtype=float)


class DummyHRTF:
    def __init__(
        self,
        positions: np.ndarray,
        source_coordinate_system: str = "spherical",
    ) -> None:
        self.Sources = DummySources(
            positions=positions,
            source_coordinate_system=source_coordinate_system,
        )


def test_get_horizontal_plane_returns_nearest_available_elevation() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [90.0, 0.0, 1.0],
            [180.0, 0.0, 1.0],
            [270.0, 0.0, 1.0],
            [0.0, 20.0, 1.0],
            [180.0, 20.0, 1.0],
        ],
        dtype=float,
    )
    hrtf = DummyHRTF(positions)

    indices, real_elevation = get_horizontal_plane(hrtf=hrtf, elevation=10.0)

    assert np.array_equal(indices, np.array([0, 1, 2, 3], dtype=int))
    assert real_elevation == 0.0


def test_get_median_plane_returns_canonical_front_back_pair() -> None:
    positions = np.array(
        [
            [0.0, -30.0, 1.0],
            [0.0, 30.0, 1.0],
            [180.0, -30.0, 1.0],
            [180.0, 30.0, 1.0],
            [90.0, 0.0, 1.0],
            [270.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    hrtf = DummyHRTF(positions)

    indices, real_azimuths = get_median_plane(hrtf=hrtf, azimuth=5.0)

    assert np.array_equal(indices, np.array([0, 1, 2, 3], dtype=int))
    assert np.array_equal(real_azimuths, np.array([0.0, 180.0], dtype=float))


def test_get_frontal_plane_supports_cartesian_source_grids() -> None:
    spherical_positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [90.0, -20.0, 1.0],
            [90.0, 20.0, 1.0],
            [270.0, -20.0, 1.0],
            [270.0, 20.0, 1.0],
            [180.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cartesian_positions = spherical_to_cartesian(spherical_positions)
    hrtf = DummyHRTF(cartesian_positions, source_coordinate_system="cartesian")

    indices, real_azimuths = get_frontal_plane(hrtf=hrtf, azimuth=100.0)

    assert np.array_equal(indices, np.array([1, 2, 3, 4], dtype=int))
    assert np.array_equal(real_azimuths, np.array([90.0, 270.0], dtype=float))
