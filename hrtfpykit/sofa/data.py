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
            raise ValueError(f"Dimension not found: {name}")
        return DimensionsWrap(name, self._netCDF4_dataset.dimensions)

    def get_all(self) -> Dict[str, DimensionsWrap]:
        return {
            k: DimensionsWrap(k, self._netCDF4_dataset.dimensions)
            for k in self._netCDF4_dataset.dimensions.keys()
            }
   
    def summary(self) -> str:
        lines = []
        for name in sorted(self._netCDF4_dataset.dimensions.keys()):
            dim = self._netCDF4_dataset.dimensions[name]
            lines.append(f"{name} = {dim.size}")
        return "\n".join(lines)

    def __getitem__(self, name: str) -> Optional[DimensionsWrap]:
        return self.get(name)

    def __iter__(self) -> Iterator[DimensionsWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self._netCDF4_dataset.dimensions)
    

class _AttributesBase(_Data):

    def __init__(self, dataset: netCDF4.Dataset = None, attribute_type: str = "Attribute") -> None:
        super().__init__(dataset)
        self._attribute_type = attribute_type

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
            label = self._attribute_type
            if label.endswith("Attribute"):
                label = f"{label[:-9]} attribute"
            else:
                label = f"{label} attribute"
            raise ValueError(f"{label} not found: {name}")
        return AttributesWrap(name, value, self._attribute_type)

    def get_all(self) -> Dict[str, AttributesWrap]:
        return {
            name: AttributesWrap(name, value, self._attribute_type)
            for name, value in self._iter_items()
        }

    def summary(self) -> str:
        pass

    def __getitem__(self, key: str) -> Optional[AttributesWrap]:
        return self.get(key)

    def __iter__(self) -> Iterator[AttributesWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self.get_names())


class _GlobalAttributes(_AttributesBase):

    def __init__(self, dataset: netCDF4.Dataset = None) -> None:
        super().__init__(dataset, attribute_type="GlobalAttribute")

    def _iter_items(self) -> Iterator[tuple[str, Any]]:
        for name in self._netCDF4_dataset.ncattrs():
            yield name, getattr(self._netCDF4_dataset, name)

    def _get_value(self, name: str) -> Optional[Any]:
        if name not in self._netCDF4_dataset.ncattrs():
            return None
        return getattr(self._netCDF4_dataset, name)

    def _invalid_name_message(self) -> str:
        return "Please insert a valid global attribute name"

    def summary(self) -> str:
        items = list(self._iter_items())
        if not items:
            return ""
        lines = [
            "****************************",
            "   GLOBAL ATTRIBUTES",
            "****************************",
        ]
        lines.extend(f"GLOBAL:{name} = {value}" for name, value in items)
        return "\n".join(lines)


class _VariableAttributes(_AttributesBase):

    def __init__(self, dataset: netCDF4.Dataset = None) -> None:
        super().__init__(dataset, attribute_type="VariableAttribute")

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

    def summary(self) -> str:
        items = list(self._iter_items())
        if not items:
            return ""
        lines = [
            "****************************",
            "   VARIABLE ATTRIBUTES",
            "****************************",
        ]
        lines.extend(f"{name} = {value}" for name, value in items)
        return "\n".join(lines)

class _Variables(_Data):

    def __init__(self, dataset : netCDF4.Dataset = None):
        super().__init__(dataset)

    def get_names(self) -> list[str]:
        return list(self._netCDF4_dataset.variables.keys())

    def get_values(self) -> list[np.ndarray]:
        return [np.array(v[:]) for v in self._netCDF4_dataset.variables.values()]

    def get(self, name: str) -> Optional[VariablesWrap]:
        if name not in self._netCDF4_dataset.variables:
            raise ValueError(f"Variable not found: {name}")
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
            lines.append(f"{name} : dimensions= ({dims_str})")
            attrs = list(var.ncattrs())
            if attrs:
                lines.append("      attributes:")
                for attr_name in attrs:
                    value = getattr(var, attr_name)
                    lines.append(f"      {name}:{attr_name}= {value}")
        return "\n".join(lines)

    def __getitem__(self, key: str) -> Optional[VariablesWrap]:
        return self.get(key)

    def __iter__(self) -> Iterator[VariablesWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self._netCDF4_dataset.variables)
