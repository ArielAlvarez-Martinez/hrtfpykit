from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hrtf import HRTF


class SourceWrapper:
    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf

    @property
    def positions(self) -> np.ndarray | None:
        return self._hrtf.source_positions

    @property
    def position_type(self) -> str | None:
        return self._hrtf.source_position_type

    @property
    def position_units(self) -> str | None:
        return self._hrtf.source_position_units

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
