from typing import Optional, Union
import pathlib
import netCDF4
from .check import check_hrtf, check_path
from .data import _Attributes, _Dimensions, _GlobalAttributes, _VariableAttributes, _Variables


class SOFA:

    def __init__(self):
        self.netCDF4_dataset: Optional[netCDF4.Dataset] = None
        self.path = None

    def _open(self, path: Union[str, pathlib.Path], mode: str = "r", parallel: bool = False, check_sofa: bool = True):
        check_path(path)
        if check_sofa is True:
            check_hrtf(path)
        self.netCDF4_dataset = netCDF4.Dataset(path, mode=mode, parallel=parallel)
        self.path = path
        return self

    @classmethod
    def load(cls, path: Union[str, pathlib.Path], mode: str = "r", parallel: bool = False, check_sofa: bool = True):
        sofa_object = cls()
        sofa_object._open(path, mode, parallel, check_sofa)
        return sofa_object

    @property
    def Dimensions(self) -> Optional[_Dimensions]:
        if self.netCDF4_dataset is None:
            return None
        return _Dimensions(self.netCDF4_dataset)

    @property
    def Attributes(self) -> Optional[_Attributes]:
        if self.netCDF4_dataset is None:
            return None
        return _Attributes(self.netCDF4_dataset)

    @property
    def GlobalAttributes(self) -> Optional[_GlobalAttributes]:
        if self.netCDF4_dataset is None:
            return None
        return _GlobalAttributes(self.netCDF4_dataset)

    @property
    def VariableAttributes(self) -> Optional[_VariableAttributes]:
        if self.netCDF4_dataset is None:
            return None
        return _VariableAttributes(self.netCDF4_dataset)

    @property
    def Variables(self) -> Optional[_Variables]:
        if self.netCDF4_dataset is None:
            return None
        return _Variables(self.netCDF4_dataset)