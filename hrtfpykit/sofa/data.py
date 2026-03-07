from abc import ABC, abstractmethod
from typing import Optional, Dict, Iterator
import netCDF4
import numpy as np
from .wraps import DimensionsWrap, VariablesWrap, AttributesWrap


class _Data(ABC):

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("Dataset is required")
        self.netCDF4_dataset = dataset

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
        return list(self.netCDF4_dataset.dimensions.keys())
    
    def get_values(self) -> list[int]:
        return [dim.size for dim in self.netCDF4_dataset.dimensions.values()]

    def get(self, name: str) -> Optional[DimensionsWrap]:
        if name not in self.netCDF4_dataset.dimensions:
            print("Please insert a valid dimension name")
            return None
        return DimensionsWrap(name, self.netCDF4_dataset.dimensions)

    def get_all(self) -> Dict[str, DimensionsWrap]:
        return {
            k: DimensionsWrap(k, self.netCDF4_dataset.dimensions)
            for k in self.netCDF4_dataset.dimensions.keys()
            }
   
    def summary(self) -> None:
        lines = []
        for name, dim in self.netCDF4_dataset.dimensions.items():
            lines.append(f"{name} = {dim.size}")
        print("\n".join(lines))

    def __getitem__(self, name: str) -> Optional[DimensionsWrap]:
        return self.get(name)

    def __iter__(self) -> Iterator[DimensionsWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self.netCDF4_dataset.dimensions)
    

class _Attributes(_Data):

    def get_all(self):
        pass
    def info(self):
        pass
    def get_names(self):
        pass

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("Dataset is required")
        self._netCDF4_dataset = dataset

    def get(self, name: Optional[str] = None):
        if name is None:
            return {k: AttributesWrap(k, getattr(self._netCDF4_dataset, k)) for k in self._netCDF4_dataset.ncattrs()}
        if name not in self._netCDF4_dataset.ncattrs():
            return None
        return AttributesWrap(name, getattr(self._netCDF4_dataset, name))

    def summary(self) -> None:
        lines = []
        for name in self._netCDF4_dataset.ncattrs():
            lines.append(f"{name} = {getattr(self._netCDF4_dataset, name)}")
        print("\n".join(lines))

    def __getitem__(self, key):
        return self.get(key)

    def __iter__(self):
        return iter(self._netCDF4_dataset.ncattrs())

    def __len__(self):
        return len(self._netCDF4_dataset.ncattrs())


class _Variables(_Data):

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("Dataset is required")
        self.netCDF4_dataset = dataset


    def get(self, name: Optional[str] = None):
        if name is None:
            return {k: VariablesWrap(k, v) for k, v in self.netCDF4_dataset.variables.items()}
        if name not in self.netCDF4_dataset.variables:
            return None
        return VariablesWrap(name, self.netCDF4_dataset.variables[name])

    def get_values(self, name: Optional[str] = None):
        if name is None:
            return {k: np.array(v[:]) for k, v in self.netCDF4_dataset.variables.items()}
        if name not in self.netCDF4_dataset.variables:
            return None
        return np.array(self.netCDF4_dataset.variables[name][:])

    def summary(self) -> None:
        lines = []
        for name, var in self.netCDF4_dataset.variables.items():
            dims = ",".join(var.dimensions)
            data = np.array(var[:])
            lines.append(
                f"{name} :  shape ({dims})  , current_shape = {data.shape} , data_type = {data.dtype}"
            )
        print("\n".join(lines))

    def __getitem__(self, key: str) -> Optional[VariablesWrap]:
        return self.get(key)

    def __iter__(self):
        return iter(self.netCDF4_dataset.variables)

    def __len__(self):
        return len(self.netCDF4_dataset.variables)
