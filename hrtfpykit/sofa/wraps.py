import numpy as np


class DimensionsWrap:
    def __init__(self, name: str, value, is_unlimited: bool = False):
        self.name = name
        self.value = value
        self.is_unlimited = bool(is_unlimited)

    def __repr__(self) -> str:
        return f"DimensionsWrap(name={self.name!r}, value={self.value}, unlimited={self.is_unlimited})"


class VariablesWrap:
    def __init__(self, name: str, var):
        self._name = name
        self._var = var

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self):
        return self._var.dtype

    @property
    def shape(self):
        return self._var.shape

    def value(self):
        return np.array(self._var[:])

    def __repr__(self) -> str:
        return f"VariablesWrap(name={self._name!r}, shape={self.shape}, dtype={self.dtype})"


class AttributesWrap:
    def __init__(self, name: str, value: str):
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"AttributesWrap(name={self._name!r}, value={self._value!r})"
