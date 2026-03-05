print("core.py")
from typing import Optional, Union
import pathlib
import netCDF4 as ncdf
import time
from functools import wraps
from .data import Dimensions, Attributes, Variables


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

def print_return(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result!r}")
        return result
    return wrapper


class SOFA:

    def __init__(self):
        self.netCDF4_dataset : ncdf.Dataset= None
        self.path = None

    @staticmethod
    def _check_path(path: Optional[Union[str, pathlib.Path]]):
        if not isinstance(path, pathlib.Path):
           path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SOFA file not found: {path}")
        if path.suffix.lower() != ".sofa":
            raise ValueError(f"SOFA file must end with .sofa: {path}")

    def _open(self,path: Optional[Union[str, pathlib.Path]], mode : str = "r", parallel : bool = False):
        self._check_path(path)
        self.netCDF4_dataset = ncdf.Dataset(path,mode=mode, parallel=parallel)
        return self

    @classmethod
    def load(cls,path: Optional[Union[str, pathlib.Path]], mode : str = "r", parallel : bool = False):
        Sofa_object = cls()
        Sofa_object._open(path, mode, parallel)
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
 
