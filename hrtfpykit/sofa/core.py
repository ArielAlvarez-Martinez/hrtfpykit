print("core.py")
from typing import Union, Optional
import pathlib
import netCDF4 as ncdf
from .data import _Dimensions, _Attributes, _Variables
from .check import check_hrtf, check_path


class SOFA:

    def __init__(self):
        self.netCDF4_dataset: ncdf.Dataset | None = None
        self.path = None

    def _open(self,path : Union[str, pathlib.Path], mode : str = "r", parallel : bool = False, check_sofa : bool = True):
        check_path(path)
        if check_sofa is True:
            check_hrtf(path)    
        self.netCDF4_dataset = ncdf.Dataset(path,mode=mode, parallel=parallel)
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
    

so = SOFA.load("<local-projects>/hrtfpykit/hrtfs/hrtf_24.sofa")

for i in so.Dimensions:
    print(i.value)