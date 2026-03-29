from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from .transforms import TransformSources

if TYPE_CHECKING:
    from .hrtf import HRTF


class Sources:
    def __init__(
        self,
        hrtf: "HRTF | None" = None,
    ) -> None:
        self._hrtf = hrtf

    @cached_property
    def transform(self) -> "TransformSources":
        return TransformSources(self)

    #TODO :Convert properties to methods .

    @property
    def positions(self) -> np.ndarray | None:
        if self._hrtf is None:
            return None
        if self._hrtf.Sofa is None:
            return None
        variables = self._hrtf.Sofa.Variables
        if variables is None:
            return None
        if "SourcePosition" not in set(variables.get_names()):
            return None
        return np.asarray(variables.get("SourcePosition").value, dtype=float)

    @property
    def position_type(self) -> str | None:
        if self._hrtf is None:
            return None
        if self._hrtf.Sofa is None:
            return None
        var_attrs = self._hrtf.Sofa.VariableAttributes
        if var_attrs is None:
            return None
        try:
            return var_attrs.get("SourcePosition:Type").value
        except ValueError:
            return None

    @property
    def position_units(self) -> str | None:
        if self._hrtf is None:
            return None
        if self._hrtf.Sofa is None:
            return None
        var_attrs = self._hrtf.Sofa.VariableAttributes
        if var_attrs is None:
            return None
        try:
            return var_attrs.get("SourcePosition:Units").value
        except ValueError:
            return None

    @property
    def azimuth_angles(self) -> np.ndarray | None:
        if self.position_type != "spherical":
            return None
        if self.positions is None or self.positions.shape[-1] < 1:
            return None
        return self.positions[..., 0]

    @property
    def elevation_angles(self) -> np.ndarray | None:
        if self.position_type != "spherical":
            return None
        if self.positions is None or self.positions.shape[-1] < 2:
            return None
        return self.positions[..., 1]

    @property
    def radius(self) -> np.ndarray | None:
        if self.position_type != "spherical":
            return None
        if self.positions is None or self.positions.shape[-1] < 3:
            return None
        return self.positions[..., 2]


class Planes:
    #TODO : class Planes will allow select specific planes from a HRTF
    pass 
