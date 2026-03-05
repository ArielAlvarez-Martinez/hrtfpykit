from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Dict, overload
import netCDF4
import numpy as np
from .wraps import DimensionsWrap, VariablesWrap, AttributesWrap

class _Data(ABC):

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("Dataset is required")
        self.netCDF4_dataset = dataset
        
    @abstractmethod
    def get(self, name: str): 
        pass

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def names(self):
        pass
    
    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod    
    def __len__(self):
        pass

class Dimensions:

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("dataset is required")
        self.netCDF4_dataset = dataset

    @property
    def description(self) -> None:
        description = [
            ("I", "Singleton axis for global listener/receiver metadata (one set for the whole HRTF set)"),
            ("C", "3D coordinate triplet used for HRTF geometry (e.g., source/ear positions: x,y,z or az,el,r)"),
            ("R", "Receivers = ears (left/right HRTF)"),
            ("E", "Emitter(s) (typically one loudspeaker/source definition)"),
            ("N", "Number of samples (time-domain HRIR) or frequency bins (frequency-domain HRTF)"),
            ("M", "Measurements = directions/positions around the head (one HRTF per direction per ear)"),
            ("S", "String-length dimension (if 0 no string-array fields stored)"),
        ]
        lines = [f"{dim}: {desc}" for dim, desc in description]
        header = "DIMENSIONS"
        width = max(len(header), max(len(line) for line in lines))
        rule = "-" * (width + 4)
        out = [rule, f"| {header.center(width)} |", rule]
        for line in lines:
            out.append(f"| {line.ljust(width)} |")
            out.append(rule)
        print("\n".join(out))

    @overload
    def get(self, name: Optional[str] = None) -> Dict[str, DimensionsWrap] | DimensionsWrap | None:
        ...

    def get(self, name: Optional[str] = None) -> Union[None, DimensionsWrap, Dict[str, DimensionsWrap]]:
        if name is None:
            return {k: DimensionsWrap(k, v) for k, v in self.netCDF4_dataset.dimensions.items()}
        if name not in self.netCDF4_dataset.dimensions:
            return None
        return DimensionsWrap(name, self.netCDF4_dataset.dimensions[name])
    
    def summary(self) -> None:
        lines = []
        for name, dim in self.netCDF4_dataset.dimensions.items():
            lines.append(f"{name} = {dim.size}")
        print("\n".join(lines))

    def __getitem__(self, key: str) -> Optional[DimensionsWrap]:
        return self.get(key)

    def __iter__(self):
        return iter(self.netCDF4_dataset.dimensions)

    def __len__(self):
        return len(self.netCDF4_dataset.dimensions)

class Attributes:

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("Dataset is required")
        self.netCDF4_dataset = dataset

    def get(self, name: Optional[str] = None):
        if name is None:
            return {k: AttributesWrap(k, getattr(self.netCDF4_dataset, k)) for k in self.netCDF4_dataset.ncattrs()}
        if name not in self.netCDF4_dataset.ncattrs():
            return None
        return AttributesWrap(name, getattr(self.netCDF4_dataset, name))

    def summary(self) -> None:
        lines = []
        for name in self.netCDF4_dataset.ncattrs():
            lines.append(f"{name} = {getattr(self.netCDF4_dataset, name)}")
        print("\n".join(lines))

    def __getitem__(self, key):
        return self.get(key)

    def __iter__(self):
        return iter(self.netCDF4_dataset.ncattrs())

    def __len__(self):
        return len(self.netCDF4_dataset.ncattrs())

class Variables:

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("Dataset is required")
        self.netCDF4_dataset = dataset


    @overload
    def get(self, name: None = None) -> Dict[str, VariablesWrap]:
        ...

    @overload
    def get(self, name: str) -> Optional[VariablesWrap]:
        ...

    def get(self, name: Optional[str] = None):
        if name is None:
            return {k: VariablesWrap(v) for k, v in self.netCDF4_dataset.variables.items()}
        if name not in self.netCDF4_dataset.variables:
            return None
        return VariablesWrap(self.netCDF4_dataset.variables[name])

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

    def dimensions(self) -> None:
        lines = []
        for name, var in self.netCDF4_dataset.variables.items():
            dims = ",".join(var.dimensions)
            dtype = getattr(var, "dtype", None)
            lines.append(f"{name} = {dims} ({dtype})")
        print("\n".join(lines))

    def __getitem__(self, key: str) -> Optional[VariablesWrap]:
        return self.get(key)

    def __iter__(self):
        return iter(self.netCDF4_dataset.variables)

    def __len__(self):
        return len(self.netCDF4_dataset.variables)
