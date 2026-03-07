from typing import Any, Optional, Union
import warnings
import netCDF4
import numpy as np
import pathlib
from .conventions import CONVENTIONS


def _formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _formatwarning


def check_path(path : Union[str, pathlib.Path]):
        if not isinstance(path, pathlib.Path):
           path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SOFA file not found: {path}")
        if path.suffix.lower() != ".sofa":
            raise ValueError(f"SOFA file must end with .sofa: {path}")


def check_hrtf(target: Union[str,netCDF4.Dataset], convention_name: Optional[str] = None, version: Optional[str] = None):
    """Check a HRTF SOFA object against SOFA conventions.

    Raises ValueError for unsupported conventions or DataType.
    Emits warnings for missing mandatory fields or read-only mismatches.
    """
    dataset, _closer = _resolve_dataset(target)
    try:
        if convention_name is None:
            convention_name = getattr(dataset, "SOFAConventions", None)

        if not convention_name:
            raise ValueError("Missing SOFAConventions on dataset")
        if convention_name not in CONVENTIONS:
            raise ValueError(
                f"Unsupported SOFAConventions '{convention_name}'. "
                f"Supported: {', '.join(sorted(CONVENTIONS.keys()))}"
            )

        expected_data_type = None
        if convention_name == "SimpleFreeFieldHRIR":
            expected_data_type = "FIR"
        elif convention_name == "SimpleFreeFieldHRTF":
            expected_data_type = "TF"

        data_type = getattr(dataset, "DataType", None)
        if expected_data_type and data_type != expected_data_type:
            raise ValueError(
                f"Unsupported DataType '{data_type}', expected '{expected_data_type}'"
            )

        if version is None:
            version = getattr(dataset, "SOFAConventionsVersion", None)

        if not version or version not in CONVENTIONS[convention_name]:
            warnings.warn(
                f"Unsupported or missing SOFAConventionsVersion '{version}' for {convention_name}. "
                f"Supported: {', '.join(sorted(CONVENTIONS[convention_name].keys()))}"
            )
            return {"convention": {"name": convention_name, "version": version}}

        spec = CONVENTIONS[convention_name][version]
        value_check_vars = []
        if convention_name == "SimpleFreeFieldHRIR":
            value_check_vars = ["Data.IR", "Data.SamplingRate"]
        elif convention_name == "SimpleFreeFieldHRTF":
            value_check_vars = ["Data.Real", "Data.Imag", "N"]

        for name, entry in spec.items():
            flags = set(entry.get("flags") or "")
            default = entry.get("default")

            kind = _parse_name(name)

            if kind[0] == "global_attr":
                attr_name = kind[1]
                exists = attr_name in dataset.ncattrs()
                if "m" in flags and not exists:
                    warnings.warn(f"Missing global attribute: {attr_name}")
                    continue
                if exists and "r" in flags and default not in ("", None):
                    value = getattr(dataset, attr_name)
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Global attribute {attr_name} should be '{default}', got '{value}'"
                        )

            elif kind[0] == "var_attr":
                var_name, attr_name = kind[1], kind[2]
                if var_name not in dataset.variables:
                    if "m" in flags:
                        warnings.warn(f"Missing variable for attribute: {var_name}")
                    continue
                var = dataset.variables[var_name]
                exists = attr_name in var.ncattrs()
                if "m" in flags and not exists:
                    warnings.warn(f"Missing attribute {attr_name} on variable {var_name}")
                    continue
                if exists and "r" in flags and default not in ("", None):
                    value = getattr(var, attr_name)
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Attribute {var_name}:{attr_name} should be '{default}', got '{value}'"
                        )

            else:
                var_name = kind[1]
                if var_name not in dataset.variables:
                    if "m" in flags:
                        warnings.warn(f"Missing variable: {var_name}")
                    continue
                var = dataset.variables[var_name]
                dim_spec = entry.get("dimensions")
                _warn_dim_mismatch(dataset, var_name, var, dim_spec)

                if "r" in flags and default not in ("", None):
                    value = np.array(var[:])
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Variable {var_name} does not match default value"
                        )

                if var_name in value_check_vars:
                    values = np.array(var[:])
                    if _is_invalid_values(values):
                        raise ValueError(f"{var_name} has invalid values (zero/None/missing)")

        return {"convention": {"name": convention_name, "version": version}}
    finally:
        if _closer is not None:
            _closer.close()

def _resolve_dataset(target: Union[str, netCDF4.Dataset]):
    if hasattr(target, "netCDF4_dataset"):
        return target.netCDF4_dataset, None
    if hasattr(target, "variables") and hasattr(target, "ncattrs"):
        return target, None
    ds = netCDF4.Dataset(str(target), "r")
    return ds, ds


def _parse_name(name: str):
    if name.startswith("GLOBAL:"):
        return ("global_attr", name.split("GLOBAL:", 1)[1])
    if ":" in name:
        var, attr = name.split(":", 1)
        return ("var_attr", var, attr)
    return ("variable", name)


def _compare_default(default: Any, value: Any) -> bool:
    if default in ("", None):
        return True
    try:
        if isinstance(value, np.ndarray) or isinstance(default, (list, tuple)):
            return np.array_equal(np.array(value), np.array(default))
        return value == default
    except Exception:
        return False


def _split_dim_options(dimensions: Optional[str]) -> list[str]:
    if not dimensions:
        return []
    return [opt.strip() for opt in dimensions.split(",") if opt.strip()]


def _matches_dim_option(var_dims: tuple[str, ...], option: str) -> bool:
    letters = [c for c in option if c.strip()]
    if len(var_dims) != len(letters):
        return False
    for dim_name, letter in zip(var_dims, letters):
        if dim_name.upper() != letter.upper():
            return False
    return True


def _warn_dim_mismatch(dataset: netCDF4.Dataset, var_name: str, var, dim_spec: Optional[str]) -> None:
    options = _split_dim_options(dim_spec)
    if options and not any(_matches_dim_option(var.dimensions, opt) for opt in options):
        warnings.warn(
            f"Variable {var_name} has dims {var.dimensions}, expected one of {options}"
        )
    for dim_name, size in zip(var.dimensions, var.shape):
        if dim_name in dataset.dimensions:
            if dataset.dimensions[dim_name].size != size:
                warnings.warn(
                    f"Variable {var_name} dimension {dim_name} size {size} "
                    f"does not match Dimensions {dataset.dimensions[dim_name].size}"
                )


def _is_invalid_values(values: np.ndarray) -> bool:
    if values is None:
        return True
    try:
        arr = np.array(values)
    except Exception:
        return True
    if arr.size == 0:
        return True
    if arr.dtype == object:
        for item in arr.ravel().tolist():
            if item is None or item == "":
                return True
    if np.issubdtype(arr.dtype, np.number):
        if np.all(arr == 0):
            return True
    return False
