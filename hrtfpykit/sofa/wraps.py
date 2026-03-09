from typing import Any, Dict, Union
import numpy as np


class DimensionsWrap:
    def __init__(self, name: str, _netCDF4_dataset_dimensions):
        self.name = name
        self.value = _netCDF4_dataset_dimensions[name].size
        self.is_unlimited = bool(_netCDF4_dataset_dimensions[name].isunlimited())
        self._netCDF4_dataset_dimensions = _netCDF4_dataset_dimensions
    
    def __repr__(self) -> str:
        return f"DimensionsWrap(name = {self.name!r}, value = {self.value}, unlimited = {self.is_unlimited})"


class AttributesWrap:
    def __init__(self, name: str, value: Any, type: str):
        self.name = name
        self.value = value
        self.type = type

    def __repr__(self) -> str:
        return f"AttributesWrap(name={self.name!r}, value={self.value!r}, type={self.type!r})"


class VariablesWrap:
    def __init__(self, name: str, var):
        self.name = name
        self._var = var
        
    @property
    def value(self) -> Union[float, np.ndarray]:
        data = np.array(self._var[:])
        return data

    @property
    def attributes(self) -> Dict[str, AttributesWrap]:
        attributes: Dict[str, AttributesWrap] = {}
        for attr_name in self._var.ncattrs():
            full_name = f"{self.name}:{attr_name}"
            attributes[full_name] = AttributesWrap(
                full_name,
                getattr(self._var, attr_name),
                "VariableAttribute",
            )
        return attributes

    def __repr__(self) -> str:
        value = self.value
        return (
            f"VariablesWrap(name={self.name!r}, dimension={self._var.shape}, "
            f"dtype={type(value)}, value={value!r}, attributes={self.attributes!r}"
        )
