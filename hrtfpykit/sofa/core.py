print("core.py")
from typing import Union
import pathlib
import netCDF4 as ncdf
from .data import Dimensions, Attributes, Variables
from .check import check_hrtf, check_path


class SOFA:

    def __init__(self):
        self.netCDF4_dataset : ncdf.Dataset= None
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
    def Dimensions(self):
        if self.netCDF4_dataset is None:
            return None
        return Dimensions(self.netCDF4_dataset)
    
    @property 
    def Attributes(self):
        if self.netCDF4_dataset is None:
            return None
        return Attributes(self.netCDF4_dataset)

    @property
    def Variables(self):
        if self.netCDF4_dataset is None:
            return None
        return Variables(self.netCDF4_dataset)
     
path = ""

sofa_object = SOFA.load(path)

i = sofa_object.Dimensions.get("I")

