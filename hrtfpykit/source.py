import numpy as np


class SourceWrapper:
    def __init__(
        self,
        positions: np.ndarray | None = None,
        position_type: str | None = None,
        position_units: str | None = None,
    ) -> None:
        self._positions = positions
        self._position_type = position_type
        self._position_units = position_units

    @property
    def positions(self) -> np.ndarray | None:
        return self._positions

    @property
    def position_type(self) -> str | None:
        return self._position_type

    @property
    def position_units(self) -> str | None:
        return self._position_units

    @property
    def azimuth(self) -> np.ndarray | None:
        if self.position_type != "spherical":
            return None
        if self.positions is None or self.positions.shape[-1] < 1:
            return None
        return self.positions[..., 0]

    @property
    def elevation(self) -> np.ndarray | None:
        if self.position_type != "spherical":
            return None
        if self.positions is None or self.positions.shape[-1] < 2:
            return None
        return self.positions[..., 1]

    @property
    def distance(self) -> np.ndarray | None:
        if self.position_type != "spherical":
            return None
        if self.positions is None or self.positions.shape[-1] < 3:
            return None
        return self.positions[..., 2]
