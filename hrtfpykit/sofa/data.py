from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Iterator
import netCDF4
import numpy as np
from .wraps import DimensionsWrap, VariablesWrap, AttributesWrap


class _Data(ABC):
    """Base wrapper for SOFA netCDF4 collections.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        Open SOFA dataset. Must not be None.
    """

    def __init__(self, dataset : netCDF4.Dataset = None):
        if dataset is None:
            raise ValueError("Dataset is required")
        self._netCDF4_dataset = dataset

    @abstractmethod
    def get_names(self):
        """Return the list of available item names."""
        pass

    @abstractmethod
    def get_values(self):
        """Return the list of raw item values."""
        pass

    @abstractmethod
    def get(self, name: str): 
        """Return a single item by name."""
        pass
    
    @abstractmethod
    def get_all(self):
        """Return all items as a mapping of name -> wrap."""
        pass
    
    @abstractmethod
    def summary(self):
        """Return a formatted text summary for this collection."""
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
    """Access wrapper for SOFA dimension metadata."""
    
    def __init__(self, dataset : netCDF4.Dataset = None):
        super().__init__(dataset)

    def get_names(self) -> list[str]:
        """Return all dimension names in the dataset."""
        return list(self._netCDF4_dataset.dimensions.keys())
    
    def get_values(self) -> list[int]:
        """Return all dimension sizes in the dataset."""
        return [dim.size for dim in self._netCDF4_dataset.dimensions.values()]

    def get(self, name: str) -> Optional[DimensionsWrap]:
        """Return a wrapped dimension by name.

        Parameters
        ----------
        name : str
            Dimension name.

        Returns
        -------
        DimensionsWrap
            Wrapped dimension metadata.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> sofa.Dimensions.get("M").value
        """
        if name not in self._netCDF4_dataset.dimensions:
            raise ValueError(f"Dimension not found: {name}")
        return DimensionsWrap(name, self._netCDF4_dataset.dimensions)

    def get_all(self) -> Dict[str, DimensionsWrap]:
        """Return all dimensions as wrapped objects.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> list(sofa.Dimensions.get_all().keys())
        """
        return {
            k: DimensionsWrap(k, self._netCDF4_dataset.dimensions)
            for k in self._netCDF4_dataset.dimensions.keys()
            }
   
    def summary(self) -> str:
        """Return a formatted summary of dimensions and sizes.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> print(sofa.Dimensions.summary())
        """
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
    """Base wrapper for SOFA attributes collections."""

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
        """Return all attribute names."""
        return [name for name, _ in self._iter_items()]

    def get_values(self) -> list[Any]:
        """Return all attribute values."""
        return [value for _, value in self._iter_items()]

    def get(self, name: str) -> Optional[AttributesWrap]:
        """Return a wrapped attribute by name.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        AttributesWrap
            Wrapped attribute metadata.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> sofa.GlobalAttributes.get("Title").value
        """
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
        """Return all attributes as wrapped objects.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> list(sofa.GlobalAttributes.get_all().keys())
        """
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
    """Access wrapper for global SOFA attributes."""

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
        """Return a formatted summary of global attributes.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> print(sofa.GlobalAttributes.summary())
        """
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
    """Access wrapper for variable SOFA attributes."""

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
        """Return a formatted summary of variable attributes.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> print(sofa.VariableAttributes.summary())
        """
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
    """Access wrapper for SOFA variables."""

    def __init__(self, dataset : netCDF4.Dataset = None):
        super().__init__(dataset)

    def get_names(self) -> list[str]:
        """Return all variable names in the dataset."""
        return list(self._netCDF4_dataset.variables.keys())

    def get_values(self) -> list[np.ndarray]:
        """Return all variable values as NumPy arrays."""
        return [np.array(v[:]) for v in self._netCDF4_dataset.variables.values()]

    def get(self, name: str) -> Optional[VariablesWrap]:
        """Return a wrapped variable by name.

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        VariablesWrap
            Wrapped variable data and metadata.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> sofa.Variables.get("Data.IR").value.shape
        """
        if name not in self._netCDF4_dataset.variables:
            raise ValueError(f"Variable not found: {name}")
        return VariablesWrap(name, self._netCDF4_dataset.variables[name])

    def get_all(self) -> Dict[str, VariablesWrap]:
        """Return all variables as wrapped objects.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> list(sofa.Variables.get_all().keys())
        """
        return {
            k: VariablesWrap(k, v) for k, v in self._netCDF4_dataset.variables.items()
        }

    def summary(self) -> str:
        """Return a formatted summary of variables and their attributes.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> print(sofa.Variables.summary())
        """
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
