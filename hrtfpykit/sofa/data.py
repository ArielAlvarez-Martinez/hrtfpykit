from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Iterator
import netCDF4
import numpy as np
from .wraps import DimensionsWrap, VariablesWrap, AttributesWrap


class _Data(ABC):

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("Dataset is required")
        self._netCDF4_dataset = dataset

    @abstractmethod
    def get_names(self):
        pass

    @abstractmethod
    def get_values(self):
        pass

    @abstractmethod
    def get(self, name: str): 
        pass
    
    @abstractmethod
    def get_all(self):
        pass
    
    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def __getitem__(self, name):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod    
    def __len__(self):
        pass


class _Dimensions(_Data):
    
    def __init__(self, dataset : netCDF4.Dataset = None):
        super().__init__(dataset)

    def get_names(self) -> list[str]:
        return list(self._netCDF4_dataset.dimensions.keys())
    
    def get_values(self) -> list[int]:
        return [dim.size for dim in self._netCDF4_dataset.dimensions.values()]

    def get(self, name: str) -> Optional[DimensionsWrap]:
        if name not in self._netCDF4_dataset.dimensions:
            print("Please insert a valid dimension name")
            return None
        return DimensionsWrap(name, self._netCDF4_dataset.dimensions)

    def get_all(self) -> Dict[str, DimensionsWrap]:
        return {
            k: DimensionsWrap(k, self._netCDF4_dataset.dimensions)
            for k in self._netCDF4_dataset.dimensions.keys()
            }
   
    def summary(self) -> str:
        lines = []
        for name, dim in self._netCDF4_dataset.dimensions.items():
            lines.append(f"{name} = {dim.size}")
        return "\n".join(lines)

    def __getitem__(self, name: str) -> Optional[DimensionsWrap]:
        return self.get(name)

    def __iter__(self) -> Iterator[DimensionsWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self._netCDF4_dataset.dimensions)
    

class _AttributesBase(_Data):

    def __init__(self, dataset : netCDF4.Dataset = None):
        super().__init__(dataset)

    @abstractmethod
    def _iter_items(self) -> Iterator[tuple[str, Any]]:
        pass

    @abstractmethod
    def _get_value(self, name: str) -> Optional[Any]:
        pass

    def _invalid_name_message(self) -> str:
        return "Please insert a valid attribute name"

    def get_names(self) -> list[str]:
        return [name for name, _ in self._iter_items()]

    def get_values(self) -> list[Any]:
        return [value for _, value in self._iter_items()]

    def get(self, name: str) -> Optional[AttributesWrap]:
        value = self._get_value(name)
        if value is None:
            print(self._invalid_name_message())
            return None
        return AttributesWrap(name, value)

    def get_all(self) -> Dict[str, AttributesWrap]:
        return {name: AttributesWrap(name, value) for name, value in self._iter_items()}

    def summary(self) -> str:
        lines = [f"{name} = {value}" for name, value in self._iter_items()]
        return "\n".join(lines)

    def __getitem__(self, key: str) -> Optional[AttributesWrap]:
        return self.get(key)

    def __iter__(self) -> Iterator[AttributesWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self.get_names())


class _GlobalAttributes(_AttributesBase):

    def _iter_items(self) -> Iterator[tuple[str, Any]]:
        for name in self._netCDF4_dataset.ncattrs():
            yield name, getattr(self._netCDF4_dataset, name)

    def _get_value(self, name: str) -> Optional[Any]:
        if name not in self._netCDF4_dataset.ncattrs():
            return None
        return getattr(self._netCDF4_dataset, name)

    def _invalid_name_message(self) -> str:
        return "Please insert a valid global attribute name"


class _VariableAttributes(_AttributesBase):

    def _iter_items(self) -> Iterator[tuple[str, Any]]:
        for var_name, var in self._netCDF4_dataset.variables.items():
            for attr_name in var.ncattrs():
                yield f"{var_name}:{attr_name}", getattr(var, attr_name)

    def _get_value(self, name: str) -> Optional[Any]:
        if ":" not in name:
            return None
        var_name, attr_name = name.split(":", 1)
        if var_name not in self._netCDF4_dataset.variables:
            return None
        var = self._netCDF4_dataset.variables[var_name]
        if attr_name not in var.ncattrs():
            return None
        return getattr(var, attr_name)

    def _invalid_name_message(self) -> str:
        return "Please insert a valid variable attribute name"


class _Attributes(_Data):

    def __init__(self, dataset : netCDF4.Dataset = None):
        super().__init__(dataset)
        self._global = _GlobalAttributes(dataset)
        self._variable = _VariableAttributes(dataset)

    @property
    def GlobalAttributes(self) -> _GlobalAttributes:
        return self._global

    @property
    def VariableAttributes(self) -> _VariableAttributes:
        return self._variable

    def get_names(self) -> list[str]:
        return self._global.get_names() + self._variable.get_names()

    def get_values(self) -> list[Any]:
        return self._global.get_values() + self._variable.get_values()

    def get(self, name: str) -> Optional[AttributesWrap]:
        if ":" in name:
            return self._variable.get(name)
        return self._global.get(name)

    def get_all(self) -> Dict[str, AttributesWrap]:
        return {**self._global.get_all(), **self._variable.get_all()}

    def summary(self) -> str:
        lines = []
        global_summary = self._global.summary()
        variable_summary = self._variable.summary()
        if global_summary:
            lines.append(global_summary)
        if variable_summary:
            lines.append(variable_summary)
        return "\n".join(lines)

    def __getitem__(self, key: str) -> Optional[AttributesWrap]:
        return self.get(key)

    def __iter__(self) -> Iterator[AttributesWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self.get_names())


class _Variables(_Data):

    def __init__(self, dataset : netCDF4.Dataset = None):
        super().__init__(dataset)

    def get_names(self) -> list[str]:
        return list(self._netCDF4_dataset.variables.keys())

    def get_values(self) -> list[np.ndarray]:
        return [np.array(v[:]) for v in self._netCDF4_dataset.variables.values()]

    def get(self, name: str) -> Optional[VariablesWrap]:
        if name not in self._netCDF4_dataset.variables:
            print("Please insert a valid variable name")
            return None
        return VariablesWrap(name, self._netCDF4_dataset.variables[name])

    def get_all(self) -> Dict[str, VariablesWrap]:
        return {
            k: VariablesWrap(k, v) for k, v in self._netCDF4_dataset.variables.items()
        }

    def summary(self) -> str:
        lines = []
        for name, var in self._netCDF4_dataset.variables.items():
            dims = []
            for dim_name in var.dimensions:
                if dim_name in self._netCDF4_dataset.dimensions:
                    dim_size = self._netCDF4_dataset.dimensions[dim_name].size
                else:
                    dim_size = "?"
                dims.append(f"{dim_name}={dim_size}")
            dims_str = ", ".join(dims)
            lines.append(
                f"{name} : dimensions = ({dims_str}) "
            )
        return "\n".join(lines)

    def __getitem__(self, key: str) -> Optional[VariablesWrap]:
        return self.get(key)

    def __iter__(self) -> Iterator[VariablesWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self._netCDF4_dataset.variables)

