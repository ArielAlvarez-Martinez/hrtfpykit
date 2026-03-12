from typing import Any, Optional, Union
import warnings
import netCDF4
import numpy as np
from .conventions import CONVENTIONS


def _formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _formatwarning


def check_sofa_against_conventions(
    target: Union[str, netCDF4.Dataset],
    convention_name: Optional[str] = None,
    version: Optional[str] = None,
):
    """Check a SOFA file against SOFA conventions.
    Emits warnings for missing mandatory fields or read-only mismatches.
    """
    dataset, _closer = _resolve_dataset(target)
    try:
        if convention_name is None:
            convention_name = getattr(dataset, "SOFAConventions", None)

        if not convention_name:
            warnings.warn("Missing SOFAConventions on dataset", UserWarning)
            return {"convention": {"name": convention_name, "version": version}}
        if convention_name not in CONVENTIONS:
            warnings.warn(
                (
                    f"Unsupported SOFAConventions '{convention_name}'. "
                    "API may not behave as expected. "
                    f"Supported: {', '.join(sorted(CONVENTIONS.keys()))}"
                ),
                UserWarning,
            )
            return {"convention": {"name": convention_name, "version": version}}

        if version is None:
            version = getattr(dataset, "SOFAConventionsVersion", None)

        if not version or version not in CONVENTIONS[convention_name]:
            warnings.warn(
                f"Unsupported or missing SOFAConventionsVersion '{version}' for {convention_name}. "
                f"Supported: {', '.join(sorted(CONVENTIONS[convention_name].keys()))}",
                UserWarning,
            )
            return {"convention": {"name": convention_name, "version": version}}

        spec = CONVENTIONS[convention_name][version]
        spec_global_attrs = {
            name.split("GLOBAL:", 1)[1] for name in spec.keys() if name.startswith("GLOBAL:")
        }
        spec_var_attrs = {
            name for name in spec.keys() if ":" in name and not name.startswith("GLOBAL:")
        }
        spec_vars = {
            name for name in spec.keys() if not name.startswith("GLOBAL:") and ":" not in name
        }
        spec_dim_letters = set()
        for entry in spec.values():
            dim_spec = entry.get("dimensions")
            if not dim_spec:
                continue
            for option in _split_dim_options(dim_spec):
                for letter in option:
                    if letter.strip():
                        spec_dim_letters.add(letter.upper())
        default_dim_letters = {"R", "E", "M", "N", "C", "I", "S"}
        expected_dim_letters = spec_dim_letters.union(default_dim_letters)

        for name, entry in spec.items():
            flags = set(entry.get("flags") or "")
            default = entry.get("default")

            kind = _parse_name(name)

            if kind[0] == "global_attr":
                attr_name = kind[1]
                exists = attr_name in dataset.ncattrs()
                if "m" in flags and not exists:
                    warnings.warn(f"Missing global attribute: {attr_name}", UserWarning)
                    continue
                if exists and "r" in flags and default not in ("", None):
                    value = getattr(dataset, attr_name)
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Global attribute {attr_name} should be '{default}', got '{value}'",
                            UserWarning,
                        )

            elif kind[0] == "var_attr":
                var_name, attr_name = kind[1], kind[2]
                if var_name not in dataset.variables:
                    if "m" in flags:
                        warnings.warn(
                            f"Missing variable for attribute: {var_name}", UserWarning
                        )
                    continue
                var = dataset.variables[var_name]
                exists = attr_name in var.ncattrs()
                if "m" in flags and not exists:
                    warnings.warn(
                        f"Missing attribute {attr_name} on variable {var_name}",
                        UserWarning,
                    )
                    continue
                if exists and "r" in flags and default not in ("", None):
                    value = getattr(var, attr_name)
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Attribute {var_name}:{attr_name} should be '{default}', got '{value}'",
                            UserWarning,
                        )

            else:
                var_name = kind[1]
                if var_name not in dataset.variables:
                    if "m" in flags:
                        warnings.warn(f"Missing variable: {var_name}", UserWarning)
                    continue
                var = dataset.variables[var_name]
                dim_spec = entry.get("dimensions")
                _warn_dim_mismatch(dataset, var_name, var, dim_spec)

                if "r" in flags and default not in ("", None):
                    value = np.array(var[:])
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Variable {var_name} does not match default value",
                            UserWarning,
                        )

        extra_global_attrs = sorted(
            attr for attr in dataset.ncattrs() if attr not in spec_global_attrs
        )
        if extra_global_attrs:
            warnings.warn(
                f"Custom global attributes found: {extra_global_attrs}",
                UserWarning,
            )

        extra_vars = sorted(
            var_name for var_name in dataset.variables.keys() if var_name not in spec_vars
        )
        if extra_vars:
            warnings.warn(
                f"Custom variables found: {extra_vars}",
                UserWarning,
            )

        extra_var_attrs: list[str] = []
        for var_name, var in dataset.variables.items():
            for attr_name in var.ncattrs():
                full_name = f"{var_name}:{attr_name}"
                if full_name not in spec_var_attrs:
                    extra_var_attrs.append(full_name)
        if extra_var_attrs:
            warnings.warn(
                f"Custom variable attributes found: {sorted(extra_var_attrs)}",
                UserWarning,
            )

        extra_dims = sorted(
            dim_name
            for dim_name in dataset.dimensions.keys()
            if dim_name.upper() not in expected_dim_letters and dim_name.upper() != "S"
        )
        if extra_dims:
            warnings.warn(
                f"Custom dimensions found: {extra_dims}",
                UserWarning,
            )

        missing_dims = sorted(
            dim for dim in expected_dim_letters if dim not in {d.upper() for d in dataset.dimensions.keys()}
        )
        if missing_dims:
            warnings.warn(
                f"Missing dimensions found: {missing_dims}",
                UserWarning,
            )

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
            f"Variable {var_name} has dims {var.dimensions}, expected one of {options}",
            UserWarning,
        )
    for dim_name, size in zip(var.dimensions, var.shape):
        if dim_name in dataset.dimensions:
            if dataset.dimensions[dim_name].size != size:
                warnings.warn(
                    f"Variable {var_name} dimension {dim_name} size {size} "
                    f"does not match Dimensions {dataset.dimensions[dim_name].size}",
                    UserWarning,
                )
