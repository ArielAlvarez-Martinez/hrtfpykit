from typing import Dict, Optional, Union
import datetime
import importlib.metadata
import pathlib
import platform
import sys
import netCDF4
import numpy as np
from .check import check_hrtf, check_path
from .conventions import CONVENTIONS
from .data import  _Dimensions, _GlobalAttributes, _VariableAttributes, _Variables


class SOFA:

    def __init__(self):
        self.netCDF4_dataset: Optional[netCDF4.Dataset] = None
        self.path = None

    def _require_dataset(self) -> netCDF4.Dataset:
        if self.netCDF4_dataset is None:
            raise ValueError("Dataset is not loaded")
        return self.netCDF4_dataset

    @property
    def Dimensions(self) -> Optional[_Dimensions]:
        if self.netCDF4_dataset is None:
            return None
        return _Dimensions(self.netCDF4_dataset)

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
    
    def create_dimension(self, name: str, value: int):
        dataset = self._require_dataset()
        if name in dataset.dimensions:
            raise ValueError(f"Dimension attribute already exists: {name}")
        spec = CONVENTIONS[dataset.SOFAConventions][dataset.SOFAConventionsVersion]
        

        dataset.createDimension(name, value)
        
    def rename_dimension(self, old_name: str, new_name: str):
        dataset = self._require_dataset()
        dataset.renameDimension(old_name , new_name)

    def create_global_attribute(self, name: str, value: Optional[str] = None) -> None:
        dataset = self._require_dataset()
        if name in dataset.ncattrs():
            raise ValueError(f"Global attribute already exists: {name}")
        setattr(dataset, name, "" if value is None else value)

    def modify_global_attribute(self, name: str, value: str) -> None:
        dataset = self._require_dataset()
        if name not in dataset.ncattrs():
            raise ValueError(f"Global attribute not found: {name}")
        setattr(dataset, name, value)

    def delete_global_attribute(self, name: str) -> None:
        dataset = self._require_dataset()
        if name not in dataset.ncattrs():
            raise ValueError(f"Global attribute not found: {name}")
        delattr(dataset, name)

    def create_variable_attribute(self, name: str, value: Optional[str] = None) -> None:
        dataset = self._require_dataset()
        if ":" not in name:
            raise ValueError("Variable attribute name must be in format 'Variable:Attribute'")
        var_name, attr_name = name.split(":", 1)
        if var_name not in dataset.variables:
            raise ValueError(f"Variable not found: {var_name}")
        var = dataset.variables[var_name]
        if attr_name in var.ncattrs():
            raise ValueError(f"Variable attribute already exists: {name}")
        setattr(var, attr_name, "" if value is None else value)

    def modify_variable_attribute(self, name: str, value: str) -> None:
        dataset = self._require_dataset()
        if ":" not in name:
            raise ValueError("Variable attribute name must be in format 'Variable:Attribute'")
        var_name, attr_name = name.split(":", 1)
        if var_name not in dataset.variables:
            raise ValueError(f"Variable not found: {var_name}")
        var = dataset.variables[var_name]
        if attr_name not in var.ncattrs():
            raise ValueError(f"Variable attribute not found: {name}")
        setattr(var, attr_name, value)

    def delete_variable_attribute(self, name: str) -> None:
        dataset = self._require_dataset()
        if ":" not in name:
            raise ValueError("Variable attribute name must be in format 'Variable:Attribute'")
        var_name, attr_name = name.split(":", 1)
        if var_name not in dataset.variables:
            raise ValueError(f"Variable not found: {var_name}")
        var = dataset.variables[var_name]
        if attr_name not in var.ncattrs():
            raise ValueError(f"Variable attribute not found: {name}")
        delattr(var, attr_name)

    def _open(self, path: Union[str, pathlib.Path], mode: str = "r", parallel: bool = False, check_sofa: bool = True):
        check_path(path)
        if check_sofa is True:
            check_hrtf(path)
        self.netCDF4_dataset = netCDF4.Dataset(path, mode=mode, parallel=parallel)
        self.path = path
        return self

    @classmethod
    def load(cls, path: Union[str, pathlib.Path], mode: str = "r", parallel: bool = False, check_sofa: bool = True) -> "SOFA": 
        print(f"Loading SOFA file from: {path}")
        sofa_object = cls()
        sofa_object._open(path, mode, parallel, check_sofa)
        print("SOFA load complete")
        return sofa_object

    @classmethod
    def create_dummy(
        cls,
        sofa_conventions: str,
        version: Optional[str] = None,
        dim_sizes: Optional[Dict[str, int]] = None,
        custom_global_attributes: Optional[Dict[str, str]] = None,
        override_default_global_attributes: bool = False,
    ) -> "SOFA":
        print("Creating in-memory dummy SOFA dataset")
        print(f"SOFA conventions: {sofa_conventions}")
        if sofa_conventions not in CONVENTIONS:
            raise ValueError(
                f"Unsupported SOFAConventions '{sofa_conventions}'. "
                f"Supported: {', '.join(sorted(CONVENTIONS.keys()))}"
            )
        available_versions = CONVENTIONS[sofa_conventions]
        if version is None:
            def _version_key(value: str) -> tuple:
                parts = value.split(".")
                if all(part.isdigit() for part in parts):
                    return tuple(int(part) for part in parts)
                return (value,)
            version = max(available_versions.keys(), key=_version_key)
            print(f"No version provided, using latest available: {version}")
        else:
            print(f"Requested conventions version: {version}")

        if version not in available_versions:
            raise ValueError(
                f"Unsupported SOFAConventionsVersion '{version}' for {sofa_conventions}. "
                f"Supported: {', '.join(sorted(available_versions.keys()))}"
            )

        spec = available_versions[version]

        def _first_dim_option(dim_spec: Optional[str]) -> list[str]:
            if not dim_spec:
                return []
            option = dim_spec.split(",", 1)[0].strip()
            option = option.replace(" ", "")
            return [letter.upper() for letter in option]

        def _dtype_for(var_type: Optional[str]) -> object:
            if var_type is None:
                return "f8"
            key = var_type.lower()
            if key in ("double", "float64"):
                return "f8"
            if key in ("float", "float32"):
                return "f4"
            if key in ("int", "int32"):
                return "i4"
            if key in ("short", "int16"):
                return "i2"
            if key in ("char", "string"):
                return str
            return "f8"

        def _reshape_for_broadcast(data: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
            if data.shape == target_shape:
                return data
            if data.shape == ():
                return data
            non_singleton = tuple(dim for dim in target_shape if dim != 1)
            if data.shape != non_singleton:
                return data
            reshaped: list[int] = []
            data_index = 0
            for dim in target_shape:
                if dim == 1:
                    reshaped.append(1)
                else:
                    reshaped.append(data.shape[data_index])
                    data_index += 1
            return data.reshape(tuple(reshaped))

        default_dim_sizes: Dict[str, int] = {"R": 2,"E": 1,"M": 1,"N": 1,"C": 3,"I": 1,"S": 0}
        user_dim_sizes: Dict[str, int] = {}
        
        if dim_sizes is not None:
            user_dim_sizes = {str(k).upper(): int(v) for k, v in dim_sizes.items()}
        if user_dim_sizes:
            ordered = ", ".join(f"{k}={v}" for k, v in sorted(user_dim_sizes.items()))
            print(f"User dimension overrides: {ordered}")

        dim_sizes: Dict[str, int] = {}
        for name, entry in spec.items():
            if name.startswith("GLOBAL:") or ":" in name:
                continue
            dim_names = _first_dim_option(entry.get("dimensions"))
            if not dim_names:
                continue
            default = entry.get("default")
            shape = None
            if isinstance(default, (list, tuple, np.ndarray)):
                try:
                    shape = np.array(default).shape
                except Exception:
                    shape = None
            if shape is None or len(shape) != len(dim_names):
                shape = tuple(user_dim_sizes.get(dim_name, default_dim_sizes.get(dim_name, 1)) for dim_name in dim_names)
            for dim_name, size in zip(dim_names, shape):
                base_size = user_dim_sizes.get(dim_name, default_dim_sizes.get(dim_name, 1))
                dim_sizes[dim_name] = max(dim_sizes.get(dim_name, base_size), base_size, int(size))

        for dim_name, size in default_dim_sizes.items():
            if dim_name not in dim_sizes:
                dim_sizes[dim_name] = user_dim_sizes.get(dim_name, size)
        for dim_name, size in user_dim_sizes.items():
            if dim_name not in dim_sizes:
                dim_sizes[dim_name] = size

        if "S" not in dim_sizes:
            dim_sizes["S"] = 0
        ordered = ", ".join(f"{k}={v}" for k, v in sorted(dim_sizes.items()))
        print(f"Final dimension sizes: {ordered}")
        dataset = netCDF4.Dataset(
            f"inmemory_{sofa_conventions}_{version}",
            mode="w",
            diskless=True,
            persist=False,
        )
        try:
            for dim_name in sorted(dim_sizes.keys()):
                size = dim_sizes[dim_name]
                if dim_name == "S" and size == 0:
                    dataset.createDimension(dim_name, None)
                else:
                    dataset.createDimension(dim_name, size)

            for name, entry in spec.items():
                if not name.startswith("GLOBAL:"):
                    continue
                attr_name = name.split("GLOBAL:", 1)[1]
                default = entry.get("default")
                if default is None:
                    continue
                setattr(dataset, attr_name, default)

            for name, entry in spec.items():
                if name.startswith("GLOBAL:") or ":" in name:
                    continue
                dim_names = _first_dim_option(entry.get("dimensions"))
                dtype = _dtype_for(entry.get("type"))
                var = dataset.createVariable(name, dtype, tuple(dim_names))
                default = entry.get("default")
                if default is None:
                    continue
                if len(dim_names) == 0:
                    var[...] = default
                    continue
                shape = tuple(dim_sizes.get(dim_name, default_dim_sizes.get(dim_name, 1)) for dim_name in dim_names)
                data = np.array(default)
                if data.shape == shape:
                    var[:] = data
                elif data.shape == ():
                    var[:] = np.full(shape, data)
                else:
                    try:
                        var[:] = np.broadcast_to(data, shape)
                    except Exception:
                        try:
                            reshaped = _reshape_for_broadcast(data, shape)
                            var[:] = np.broadcast_to(reshaped, shape)
                        except Exception:
                            var[:] = np.zeros(shape)

            for name, entry in spec.items():
                if name.startswith("GLOBAL:") or ":" not in name:
                    continue
                var_name, attr_name = name.split(":", 1)
                if var_name not in dataset.variables:
                    continue
                default = entry.get("default")
                if default is None:
                    continue
                setattr(dataset.variables[var_name], attr_name, default)
        except Exception:
            dataset.close()
            raise

        dataset.SOFAConventions = sofa_conventions
        dataset.SOFAConventionsVersion = version
        cls._complete_global_attributes(
            dataset,
            custom_global_attributes,
            override_default_global_attributes,
        )

        sofa_object = cls()
        sofa_object.netCDF4_dataset = dataset
        sofa_object.path = None
        print("Dummy SOFA dataset ready")
        return sofa_object

    @staticmethod
    def _complete_global_attributes(
        dataset: netCDF4.Dataset,
        custom_global_attributes: Optional[Dict[str, str]] = None,
        override_default_global_attributes: bool = False,
    ) -> None:
        def _is_missing(value: object) -> bool:
            if value is None:
                return True
            if isinstance(value, str) and value.strip() == "":
                return True
            return False

        try:
            api_version = importlib.metadata.version("hrtfpykit")
        except importlib.metadata.PackageNotFoundError:
            api_version = "unknown"

        compiler = platform.python_compiler()
        implementation = sys.implementation.name
        if implementation:
            implementation = implementation.capitalize()
        else:
            implementation = "Python"
        python_version = f"{platform.python_version()} [{implementation} - {compiler}]"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        default_custom_attributes: Dict[str, str] = {
            "APIName": "hrtfpykit",
            "APIVersion": api_version,
            "ApplicationName": "Python",
            "ApplicationVersion": python_version,
            "DateCreated": now,
            "DateModified": now,
        }

        resolved = default_custom_attributes
        if custom_global_attributes:
            resolved = {**default_custom_attributes, **custom_global_attributes}

        for attr_name, value in resolved.items():
            if override_default_global_attributes or _is_missing(getattr(dataset, attr_name, None)):
                setattr(dataset, attr_name, value)

    def save(self, path: Optional[Union[str, pathlib.Path]] = None, overwrite: bool = False) -> pathlib.Path:
        if self.netCDF4_dataset is None:
            raise ValueError("Dataset is not loaded")

        if path is None:
            print("Saving SOFA file to original path")
            self.netCDF4_dataset.sync()
            if self.path is None:
                raise ValueError("No path available to save the dataset")
            print("SOFA save complete")
            return pathlib.Path(self.path)

        target_path = pathlib.Path(path)
        print(f"Saving SOFA file to: {target_path}")
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"SOFA file already exists: {target_path}")

        src = self.netCDF4_dataset
        file_format = getattr(src, "file_format", "NETCDF4")
        dst = netCDF4.Dataset(str(target_path), mode="w", format=file_format)
        try:
            for name, dim in src.dimensions.items():
                size = None if dim.isunlimited() else dim.size
                dst.createDimension(name, size)

            dst.setncatts({name: getattr(src, name) for name in src.ncattrs()})

            for name, var in src.variables.items():
                dst_var = dst.createVariable(name, var.datatype, var.dimensions)
                dst_var.setncatts({attr: getattr(var, attr) for attr in var.ncattrs()})
                dst_var[:] = var[:]
        finally:
            dst.close()

        print("SOFA save complete")
        return target_path

    def copy(self) -> "SOFA":
        if self.netCDF4_dataset is None:
            raise ValueError("Dataset is not loaded")

        src = self.netCDF4_dataset
        file_format = getattr(src, "file_format", "NETCDF4")
        dst = netCDF4.Dataset(
            f"inmemory_{id(self)}",
            mode="w",
            diskless=True,
            persist=False,
            format=file_format,
        )
        try:
            for name, dim in src.dimensions.items():
                size = None if dim.isunlimited() else dim.size
                dst.createDimension(name, size)

            dst.setncatts({name: getattr(src, name) for name in src.ncattrs()})

            for name, var in src.variables.items():
                dst_var = dst.createVariable(name, var.datatype, var.dimensions)
                dst_var.setncatts({attr: getattr(var, attr) for attr in var.ncattrs()})
                dst_var[:] = var[:]
        except Exception:
            dst.close()
            raise

        sofa_object = SOFA()
        sofa_object.netCDF4_dataset = dst
        sofa_object.path = None
        return sofa_object

    def summary(self) -> str:
        if self.netCDF4_dataset is None:
            raise ValueError("Dataset is not loaded")

        dataset = self.netCDF4_dataset
        lines: list[str] = [
            "****************************",
            "     GLOBAL ATTRIBUTES",
            "****************************",
        ]
        for name in dataset.ncattrs():
            lines.append(f"GLOBAL:{name} : {getattr(dataset, name)}")

        lines.extend(
            [
                "*******************************************",
                "    VARIABLES AND VARIABLES ATTRIBUTES",
                "*******************************************",
            ]
        )
        for name, var in dataset.variables.items():
            dims = []
            for dim_name in var.dimensions:
                if dim_name in dataset.dimensions:
                    dim_size = dataset.dimensions[dim_name].size
                else:
                    dim_size = "?"
                dims.append(f"{dim_name}={dim_size}")
            dims_str = ", ".join(dims)
            lines.append(f"{name} : dimensions= ({dims_str})")
            attrs = list(var.ncattrs())
            if attrs:
                lines.append("    attributes:")
                for attr_name in attrs:
                    value = getattr(var, attr_name)
                    lines.append(f"        {name}:{attr_name}= {value}")
        return "\n".join(lines)

