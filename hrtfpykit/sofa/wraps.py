from typing import Any, Dict
import numpy as np


class DimensionsWrap:
    """Wrap a netCDF4 Dimension for SOFA inspection.

    Parameters
    ----------
    name : str
        Dimension name.
    _netCDF4_dataset_dimensions : Any
        Mapping-like container of netCDF4 dimensions (for example,
        ``Dataset.dimensions``).

    Attributes
    ----------
    name : str
        Dimension name.
    value : int
        Dimension size.
    is_unlimited : bool
        Whether the dimension is unlimited.
    """

    def __init__(self, name: str, _netCDF4_dataset_dimensions: Any) -> None:
        self.name = name
        self.value = _netCDF4_dataset_dimensions[name].size
        self.is_unlimited = bool(_netCDF4_dataset_dimensions[name].isunlimited())
        self._netCDF4_dataset_dimensions = _netCDF4_dataset_dimensions

    def __repr__(self) -> str:
        return f"DimensionsWrap(name = {self.name!r}, value = {self.value}, unlimited = {self.is_unlimited})"



class AttributesWrap:
    """Wrap a netCDF4 attribute for SOFA-friendly access.

    Parameters
    ----------
    name : str
        Attribute name (for variables, use ``Variable:Attribute``).
    value : Any
        Attribute value.
    type : str
        Attribute type label used by the SOFA API.
    """

    def __init__(self, name: str, value: Any, type: str) -> None:
        self.name = name
        self.value = value
        self.type = type

    def __repr__(self) -> str:
        return f"AttributesWrap(name={self.name!r}, value={self.value!r}, type={self.type!r})"



class VariablesWrap:
    """Wrap a netCDF4 Variable with convenience accessors.

    Parameters
    ----------
    name : str
        Variable name.
    var : Any
        netCDF4 Variable instance.

    """

    def __init__(self, name: str, var: Any) -> None:
        self.name = name
        self._var = var

    @property
    def value(self) -> np.ndarray:
        """Return the variable data as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Copy of the variable data.
        """
        data = np.array(self._var[:])
        return data

    @property
    def attributes(self) -> Dict[str, AttributesWrap]:
        """Return variable attributes as a mapping of wrappers.

        Returns
        -------
        dict
            of str to AttributesWrap
            Attribute wrappers keyed by ``Variable:Attribute``.
        """
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
