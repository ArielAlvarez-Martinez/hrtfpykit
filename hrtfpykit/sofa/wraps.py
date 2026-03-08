from typing import Union

import numpy as np


class DimensionsWrap:
    def __init__(self, name: str, _netCDF4_dataset_dimensions):
        self.name = name
        self.value = _netCDF4_dataset_dimensions[name].size
        self.is_unlimited = bool(getattr(self.value, "isunlimited", lambda: False)())
        self._netCDF4_dataset_dimensions = _netCDF4_dataset_dimensions
    
    def __repr__(self) -> str:
        return f"DimensionsWrap(name = {self.name!r}, value = {self.value}, unlimited = {self.is_unlimited})"


class AttributesWrap:
    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"AttributesWrap(name={self.name!r}, value={self.value!r})"


class VariablesWrap:
    def __init__(self, name: str, var):
        self.name = name
        self._var = var
        
    @property
    def value(self) -> Union[int, np.ndarray]:
        data = np.array(self._var[:])
        if data.size == 1 and np.issubdtype(data.dtype, np.number):
            return int(data.reshape(-1)[0])
        return data

    def __repr__(self) -> str:
        value = self.value
        return f"VariablesWrap(name={self.name!r}, dimension= {self._var.shape}, dtype= {type(value)})"
