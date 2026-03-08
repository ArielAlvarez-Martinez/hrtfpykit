from abc import ABC, abstractmethod
from typing import Optional, Dict, Iterator, Union
import netCDF4
import numpy as np
import pathlib
from .wraps import DimensionsWrap, VariablesWrap, AttributesWrap
from .check import check_hrtf, check_path


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
    

class _Attributes(_Data):

    def __init__(self, dataset : netCDF4.Dataset = None):
        super().__init__(dataset)

    def get_names(self) -> list[str]:
        return list(self._netCDF4_dataset.ncattrs())

    def get_values(self) -> list[str]:
        return [getattr(self._netCDF4_dataset, name) for name in self._netCDF4_dataset.ncattrs()]

    def get(self, name: str) -> Optional[AttributesWrap]:
        if name not in self._netCDF4_dataset.ncattrs():
            print("Please insert a valid attribute name")
            return None
        return AttributesWrap(name, getattr(self._netCDF4_dataset, name))

    def get_all(self) -> Dict[str, AttributesWrap]:
        return {
            k: AttributesWrap(k, getattr(self._netCDF4_dataset, k))
            for k in self._netCDF4_dataset.ncattrs()
        }

    def summary(self) -> str:
        lines = []
        for name in self._netCDF4_dataset.ncattrs():
            lines.append(f"{name} = {getattr(self._netCDF4_dataset, name)}")
        return "\n".join(lines)

    def __getitem__(self, key: str) -> Optional[AttributesWrap]:
        return self.get(key)

    def __iter__(self) -> Iterator[AttributesWrap]:
        return iter(self.get_all().values())

    def __len__(self) -> int:
        return len(self._netCDF4_dataset.ncattrs())


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


class SOFA:

    def __init__(self):
        self.netCDF4_dataset: Optional[netCDF4.Dataset] = None
        self.path = None

    def _open(self,path : Union[str, pathlib.Path], mode : str = "r", parallel : bool = False, check_sofa : bool = True):
        check_path(path)
        if check_sofa is True:
            check_hrtf(path)    
        self.netCDF4_dataset = netCDF4.Dataset(path,mode=mode, parallel=parallel)
        self.path = path
        return self

    
    @classmethod
    def load(cls,path : Union[str, pathlib.Path], mode : str = "r", parallel : bool = False, check_sofa : bool = True):
        Sofa_object = cls()
        Sofa_object._open(path, mode, parallel, check_sofa)
        return Sofa_object

    @property
    def Dimensions(self) -> _Dimensions:
        if self.netCDF4_dataset is None:
            return None
        return _Dimensions(self.netCDF4_dataset)
    
    @property 
    def Attributes(self) -> _Attributes :
        if self.netCDF4_dataset is None:
            return None
        return _Attributes(self.netCDF4_dataset)

    @property
    def Variables(self) -> _Variables:
        if self.netCDF4_dataset is None:
            return None
        return _Variables(self.netCDF4_dataset)

