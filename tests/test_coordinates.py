import numpy as np

from hrtfpykit.hrtf.coordinates import (
    cartesian_to_spherical,
    get_closest_position_index,
    get_named_positions,
    get_position_alias,
    get_position_queries,
    get_spherical_positions,
    lateral_polar_to_spherical,
    spherical_to_cartesian,
    spherical_to_lateral_polar,
)


class DummySources:
    def __init__(
        self,
        positions: np.ndarray,
        source_coordinate_system: str,
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


def test_get_position_queries_normalizes_strings_and_numeric_inputs() -> None:
    assert get_position_queries("Front") == ["front"]

    single_numeric = get_position_queries([0.0, 0.0])
    assert len(single_numeric) == 1
    assert np.array_equal(single_numeric[0], np.array([0.0, 0.0], dtype=float))

    many_numeric = get_position_queries([[0.0, 0.0], [90.0, 0.0]])
    assert len(many_numeric) == 2
    assert np.array_equal(many_numeric[0], np.array([0.0, 0.0], dtype=float))
    assert np.array_equal(many_numeric[1], np.array([90.0, 0.0], dtype=float))


def test_get_named_positions_and_alias_resolve_cardinal_directions() -> None:
    named_positions = get_named_positions()

    assert np.array_equal(named_positions["front"], np.array([0.0, 0.0], dtype=float))
    assert np.array_equal(named_positions["left"], np.array([90.0, 0.0], dtype=float))
    assert get_position_alias([0.0, 0.0]) == "front"
    assert get_position_alias([90.0, 0.0]) == "left"
    assert get_position_alias([270.0, 0.0]) == "right"
    assert get_position_alias([0.0, 30.0]) is None


def test_coordinate_conversions_round_trip_between_supported_systems() -> None:
    spherical = np.array(
        [
            [0.0, 0.0, 1.0],
            [90.0, 0.0, 1.0],
            [180.0, 30.0, 1.5],
            [270.0, -20.0, 2.0],
        ],
        dtype=float,
    )

    cartesian = spherical_to_cartesian(spherical)
    spherical_round_trip = cartesian_to_spherical(cartesian)
    assert np.allclose(spherical_round_trip, spherical, atol=1e-10)

    lateral_polar = spherical_to_lateral_polar(spherical)
    spherical_from_lateral_polar = lateral_polar_to_spherical(lateral_polar)
    assert np.allclose(spherical_from_lateral_polar, spherical, atol=1e-10)


def test_get_closest_position_index_handles_spherical_wrap_around() -> None:
    grid_positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [90.0, 0.0, 1.0],
            [180.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    idx = get_closest_position_index(
        query_position=[359.0, 0.0],
        grid_positions=grid_positions,
        coordinate_system="spherical",
    )

    assert idx == 0


def test_get_spherical_positions_converts_cartesian_source_grid() -> None:
    sources = DummySources(
        positions=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        source_coordinate_system="cartesian",
    )

    spherical_positions = get_spherical_positions(sources)

    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [90.0, 0.0, 1.0],
            [0.0, 90.0, 1.0],
        ],
        dtype=float,
    )
    assert np.allclose(spherical_positions, expected, atol=1e-10)
