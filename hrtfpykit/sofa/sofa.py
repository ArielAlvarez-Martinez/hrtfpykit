from typing import Any, Dict, Optional, Union
import pathlib
from uuid import uuid4
import netCDF4
import numpy as np
from .conventions import CONVENTIONS
from .data import  _Dimensions, _GlobalAttributes, _VariableAttributes, _Variables
from .sofa_helpers import (
    complete_global_attributes,
    dtype_for,
    ensure_broadcastable,
    first_dim_option,
    open_sofa,
    require_dataset,
    reshape_for_broadcast,
    version_key,
    warn_dimension_shape_mismatch,
)


class SOFA:
    """High-level SOFA file handler backed by netCDF4.

    This class wraps a netCDF4 SOFA dataset and provides CRUD helpers for
    dimensions, attributes, and variables, along with utilities for loading,
    copying, saving, and summarizing SOFA files.

    Notes
    -----
    - No hidden I/O is performed. Files are only read/written when you call
      ``load()``, ``save()``, or other explicit methods.
    - The underlying dataset is a netCDF4 Dataset, so standard netCDF4
      rules and constraints apply.

    Examples
    --------
    Load, edit, and save in place:

    >>> sofa = SOFA.load("my.sofa")
    >>> sofa_clone = sofa.clone()
    >>> sofa_clone.create_global_attribute("Title", "My HRTF")
    >>> sofa_clone.save("my_copy.sofa")

    Create a dummy in-memory writable dataset:

    >>> sofa = SOFA.create_dummy("SimpleFreeFieldHRIR", version="1.2")
    >>> print(sofa.summary())
    """

    def __init__(self):
        self.netCDF4_dataset: Optional[netCDF4.Dataset] = None
        self.path = None

    @property
    def Dimensions(self) -> Optional[_Dimensions]:
        """Return the dimension access wrapper.

        Returns
        -------
        Optional[_Dimensions]
            Wrapper for dimension access, or None if no dataset is loaded.
        """
        if self.netCDF4_dataset is None:
            return None
        return _Dimensions(self.netCDF4_dataset)

    @property
    def GlobalAttributes(self) -> Optional[_GlobalAttributes]:
        """Return the global attribute access wrapper.

        Returns
        -------
        Optional[_GlobalAttributes]
            Wrapper for global attributes, or None if no dataset is loaded.
        """
        if self.netCDF4_dataset is None:
            return None
        return _GlobalAttributes(self.netCDF4_dataset)

    @property
    def Variables(self) -> Optional[_Variables]:
        """Return the variable access wrapper.

        Returns
        -------
        Optional[_Variables]
            Wrapper for variables, or None if no dataset is loaded.
        """
        if self.netCDF4_dataset is None:
            return None
        return _Variables(self.netCDF4_dataset)

    @property
    def VariableAttributes(self) -> Optional[_VariableAttributes]:
        """Return the variable attribute access wrapper.

        Returns
        -------
        Optional[_VariableAttributes]
            Wrapper for variable attributes, or None if no dataset is loaded.
        """
        if self.netCDF4_dataset is None:
            return None
        return _VariableAttributes(self.netCDF4_dataset)

    def create_dimension(self, name: str, value: int) -> None:
        """Create a new SOFA dimension.

        Parameters
        ----------
        name : str
            Dimension name.
        value : int
            Dimension size.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.create_dimension("X", 3)
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.create_dimension("X", 3)
        """
        dataset = require_dataset(self)
        if name in dataset.dimensions:
            raise ValueError(f"Dimension attribute already exists: {name}")
        dataset.createDimension(name, value)
        print(f"Dimension: '{name}' created succesfully")
        
    def rename_dimension(self, old_name: str, new_name: str) -> None:
        """Rename an existing SOFA dimension.

        Parameters
        ----------
        old_name : str
            Existing dimension name.
        new_name : str
            New dimension name.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.rename_dimension("M", "Measurements")
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.rename_dimension("M", "Measurements")
        """
        dataset = require_dataset(self)
        if old_name not in dataset.dimensions:
            print(f"Dimension: '{old_name}' not found")
            raise ValueError(f"Dimension not found: {old_name}")
        dataset.renameDimension(old_name , new_name)
        print(f"Dimension: '{old_name}' renamed succesfully")

    def create_global_attribute(self, name: str, value: Optional[str] = None) -> None:
        """Create a global SOFA attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Optional[str], optional
            Attribute value. Empty string is used when None.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.create_global_attribute("Title", "My HRTF")
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.create_global_attribute("Title", "My HRTF")
        """
        dataset = require_dataset(self)
        if name in dataset.ncattrs():
            raise ValueError(f"Global attribute already exists: {name}")
        stored_value = "" if value is None else value
        setattr(dataset, name, stored_value)
        print(f"Global attribute: '{name}' created succesfully")

    def modify_global_attribute(self, name: str, value: str) -> None:
        """Modify an existing global SOFA attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        value : str
            New attribute value.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.modify_global_attribute("Title", "Updated title")
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.modify_global_attribute("Title", "Updated title")
        """
        dataset = require_dataset(self)
        if name not in dataset.ncattrs():
            raise ValueError(f"Global attribute not found: {name}")
        setattr(dataset, name, value)
        print(f"Global attribute: '{name}' modified succesfully")

    def delete_global_attribute(self, name: str) -> None:
        """Delete a global SOFA attribute.

        Parameters
        ----------
        name : str
            Attribute name to remove.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.delete_global_attribute("Comment")
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.delete_global_attribute("Comment")
        """
        dataset = require_dataset(self)
        if name not in dataset.ncattrs():
            raise ValueError(f"Global attribute not found: {name}")
        delattr(dataset, name)
        print(f"Global attribute: '{name}' deleted succesfully")

    def create_variable_attribute(self, name: str, value: Optional[str] = None) -> None:
        """Create a variable attribute.

        Parameters
        ----------
        name : str
            Attribute name in the form ``"Variable:Attribute"``.
        value : Optional[str], optional
            Attribute value. Empty string is used when None.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.create_variable_attribute("Data.SamplingRate:Units", "hertz")
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.create_variable_attribute("Data.SamplingRate:Units", "hertz")
        """
        dataset = require_dataset(self)
        if ":" not in name:
            raise ValueError("Variable attribute name must be in format 'Variable:Attribute'")
        var_name, attr_name = name.split(":", 1)
        if var_name not in dataset.variables:
            raise ValueError(f"Variable not found: {var_name}")
        var = dataset.variables[var_name]
        if attr_name in var.ncattrs():
            raise ValueError(f"Variable attribute already exists: {name}")
        stored_value = "" if value is None else value
        setattr(var, attr_name, stored_value)
        print(f"Variable attribute: '{name}' created succesfully")

    def modify_variable_attribute(self, name: str, value: str) -> None:
        """Modify an existing variable attribute.

        Parameters
        ----------
        name : str
            Attribute name in the form ``"Variable:Attribute"``.
        value : str
            New attribute value.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.modify_variable_attribute("Data.IR:Units", "Pa")
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.modify_variable_attribute("Data.IR:Units", "Pa")
        """
        dataset = require_dataset(self)
        if ":" not in name:
            raise ValueError("Variable attribute name must be in format 'Variable:Attribute'")
        var_name, attr_name = name.split(":", 1)
        if var_name not in dataset.variables:
            raise ValueError(f"Variable not found: {var_name}")
        var = dataset.variables[var_name]
        if attr_name not in var.ncattrs():
            raise ValueError(f"Variable attribute not found: {name}")
        setattr(var, attr_name, value)
        print(f"Variable attribute: '{name}' modified succesfully")

    def delete_variable_attribute(self, name: str) -> None:
        """Delete a variable attribute.

        Parameters
        ----------
        name : str
            Attribute name in the form ``"Variable:Attribute"``.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.delete_variable_attribute("Data.IR:Units")
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.delete_variable_attribute("Data.IR:Units")
        """
        dataset = require_dataset(self)
        if ":" not in name:
            raise ValueError("Variable attribute name must be in format 'Variable:Attribute'")
        var_name, attr_name = name.split(":", 1)
        if var_name not in dataset.variables:
            raise ValueError(f"Variable not found: {var_name}")
        var = dataset.variables[var_name]
        if attr_name not in var.ncattrs():
            raise ValueError(f"Variable attribute not found: {name}")
        delattr(var, attr_name)
        print(f"Variable attribute: '{name}' deleted succesfully")

    def create_variable(
        self,
        name: str,
        data: Union[np.ndarray, list],
        dimensions: Union[tuple[str, ...], list[str]],
        dtype: Optional[Union[str, np.dtype]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a SOFA variable and optionally its attributes.

        Parameters
        ----------
        name : str
            Variable name.
        data : Union[np.ndarray, list]
            Variable data.
        dimensions : Union[tuple[str, ...], list[str]]
            Dimension names in order, matching the data shape.
        dtype : Optional[Union[str, np.dtype]], optional
            Data type for the variable. Defaults to the array dtype.
        attributes : Optional[Dict[str, Any]], optional
            Optional attributes to set on the variable.

        Notes
        -----
        The function warns when dimension sizes do not coincide with the
        dataset dimensions and raises an error if the data cannot be
        broadcast to the target shape.
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> data = np.zeros((sofa_clone.netCDF4_dataset.dimensions["M"].size,))
        >>> sofa_clone.create_variable("Custom", data, ("M",), attributes={"Units": "unitless"})
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> data = np.zeros((sofa.netCDF4_dataset.dimensions["M"].size,))
        >>> sofa.create_variable("Custom", data, ("M",), attributes={"Units": "unitless"})
        """
        dataset = require_dataset(self)
        if name in dataset.variables:
            raise ValueError(f"Variable already exists: {name}")

        dims = tuple(dimensions)
        missing_dims = [dim_name for dim_name in dims if dim_name not in dataset.dimensions]
        if missing_dims:
            missing_str = ", ".join(missing_dims)
            raise ValueError(f"Dimensions not found: {missing_str}")

        array = np.array(data)
        if dtype is None:
            dtype = array.dtype

        warn_dimension_shape_mismatch(name, dims, array.shape, dataset)
        target_shape: list[int] = []
        for idx, dim_name in enumerate(dims):
            dim = dataset.dimensions[dim_name]
            if dim.isunlimited():
                if idx < array.ndim:
                    target_shape.append(array.shape[idx])
                else:
                    target_shape.append(1)
            else:
                target_shape.append(dim.size)
        ensure_broadcastable(name, array, tuple(target_shape))

        var = dataset.createVariable(name, dtype, dims)
        var[...] = array

        if attributes:
            for attr_name, attr_value in attributes.items():
                setattr(var, attr_name, attr_value)

        print(f"Variable: '{name}' created succesfully")

    def modify_variable(self, name: str, data: Union[np.ndarray, list]) -> None:
        """Overwrite data for an existing SOFA variable.

        Parameters
        ----------
        name : str
            Variable name.
        data : Union[np.ndarray, list]
            New variable data.

        Notes
        -----
        The function warns when dimension sizes do not coincide with the
        dataset dimensions and raises an error if the data cannot be
        broadcast to the target shape.
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> new_data = np.zeros((sofa_clone.netCDF4_dataset.dimensions["M"].size, 2, 256))
        >>> sofa_clone.modify_variable("Data.IR", new_data)
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> new_data = np.zeros((sofa.netCDF4_dataset.dimensions["M"].size, 2, 256))
        >>> sofa.modify_variable("Data.IR", new_data)
        """
        dataset = require_dataset(self)
        if name not in dataset.variables:
            raise ValueError(f"Variable not found: {name}")
        var = dataset.variables[name]
        array = np.array(data)
        warn_dimension_shape_mismatch(name, var.dimensions, array.shape, dataset)
        target_shape: list[int] = []
        for idx, dim_name in enumerate(var.dimensions):
            dim = dataset.dimensions[dim_name]
            if dim.isunlimited():
                if idx < array.ndim:
                    target_shape.append(array.shape[idx])
                else:
                    target_shape.append(var.shape[idx])
            else:
                target_shape.append(var.shape[idx])
        ensure_broadcastable(name, array, tuple(target_shape))
        var[...] = array
        print(f"Variable: '{name}' modified succesfully")

    def delete_variable(self, name: str) -> None:
        """Delete a SOFA variable.

        Parameters
        ----------
        name : str
            Variable name to remove.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original dataset, create an in-memory
        clone with ``clone()``, apply edits to the clone, and save when
        you are ready. Direct in-place editing with ``mode="r+"`` is
        still available for expert users.

        Examples
        --------
        Recommended (safe) workflow:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.delete_variable("Custom")
        >>> sofa_clone.save("my_copy.sofa")

        Direct edit (expert users):

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.delete_variable("Custom")
        """
        dataset = require_dataset(self)
        if name not in dataset.variables:
            raise ValueError(f"Variable not found: {name}")
        del dataset.variables[name]
        print(f"Variable: '{name}' deleted succesfully")

    @classmethod
    def load(cls, path: Union[str, pathlib.Path], mode: str = "r", parallel: bool = False, check_sofa_against_conventions: bool = True) -> "SOFA": 
        """Load a SOFA file and return a SOFA class instance.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            Path to the SOFA file.
        mode : str, optional
            netCDF4 open mode (e.g., "r", "r+").
        parallel : bool, optional
            Whether to open in parallel mode.
        check_sofa_against_conventions : bool, optional
            If True, validates against SOFA conventions on open.

        Returns
        -------
        SOFA
            Loaded SOFA instance.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        """
        print(f"Loading SOFA file from: {path}")
        sofa_object = cls()
        open_sofa(sofa_object, path, mode, parallel, check_sofa_against_conventions)
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
        """Create an in-memory dummy SOFA dataset following a convention.

        Parameters
        ----------
        sofa_conventions : str
            SOFA convention name.
        version : Optional[str], optional
            Convention version. If None, the latest available is used.
        dim_sizes : Optional[Dict[str, int]], optional
            Dimension size overrides.
        custom_global_attributes : Optional[Dict[str, str]], optional
            Additional global attributes to set.
        override_default_global_attributes : bool, optional
            If True, custom attributes override defaults even if set.

        Returns
        -------
        SOFA
            In-memory SOFA instance.

        Notes
        -----
        The dummy dataset always includes an unlimited ``S`` dimension with
        initial size 0. Passing ``S`` in ``dim_sizes`` raises an error because
        ``S`` is reserved by the SOFA conventions. To create other unlimited
        dimensions, set their size to 0 in ``dim_sizes``.

        Examples
        --------
        Basic dummy:

        >>> sofa = SOFA.create_dummy("SimpleFreeFieldHRIR", version="1.2")
        >>> print(sofa.Dimensions.summary())

        Override fixed dimensions:

        >>> sofa = SOFA.create_dummy(
        ...     "SimpleFreeFieldHRIR",
        ...     dim_sizes={"R": 2, "C": 3, "M": 100, "N": 256},
        ... )

        Create an additional unlimited dimension:

        >>> sofa = SOFA.create_dummy(
        ...     "SimpleFreeFieldHRIR",
        ...     dim_sizes={"R": 2, "C": 3, "K": 0},
        ... )

        Passing ``S`` raises an error (``S`` is always unlimited and starts at 0):

        >>> SOFA.create_dummy("SimpleFreeFieldHRIR", dim_sizes={"S": 10})
        Traceback (most recent call last):
        ...
        ValueError: dim_sizes must not include 'S' (reserved unlimited dimension).
        """
        print("Creating in-memory dummy SOFA dataset")
        print(f"SOFA conventions: {sofa_conventions}")
        if sofa_conventions not in CONVENTIONS:
            raise ValueError(
                f"Unsupported SOFAConventions '{sofa_conventions}'. "
                f"Supported: {', '.join(sorted(CONVENTIONS.keys()))}"
            )
        available_versions = CONVENTIONS[sofa_conventions]
        if version is None:
            version = max(available_versions.keys(), key=version_key)
            print(f"No version provided, using latest available: {version}")
        else:
            print(f"Requested conventions version: {version}")

        if version not in available_versions:
            raise ValueError(
                f"Unsupported SOFAConventionsVersion '{version}' for {sofa_conventions}. "
                f"Supported: {', '.join(sorted(available_versions.keys()))}"
            )

        spec = available_versions[version]

        user_dim_sizes: Dict[str, int] = {}

        if dim_sizes is not None:
            if any(str(k).upper() == "S" for k in dim_sizes.keys()):
                raise ValueError(
                    "dim_sizes must not include 'S' (reserved unlimited dimension). "
                    "S is always created with size 0 and unlimited. "
                    "To create other unlimited dimensions, pass size 0 (e.g., {'K': 0})."
                )
            user_dim_sizes = {str(k).upper(): int(v) for k, v in dim_sizes.items()}
        if user_dim_sizes:
            ordered = ", ".join(f"{k}={v}" for k, v in sorted(user_dim_sizes.items()))
            print(f"User dimension overrides: {ordered}")

        effective_dim_sizes = dict(user_dim_sizes)
        effective_dim_sizes["S"] = 0

        unlimited_dims = {name for name, size in effective_dim_sizes.items() if size == 0}
        unlimited_dims.add("S")

        dim_sizes: Dict[str, int] = {}
        for name, entry in spec.items():
            if name.startswith("GLOBAL:") or ":" in name:
                continue
            dim_names = first_dim_option(entry.get("dimensions"))
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
                shape = tuple(effective_dim_sizes.get(dim_name, 1) for dim_name in dim_names)
            for dim_name, size in zip(dim_names, shape):
                base_size = effective_dim_sizes.get(dim_name, 1)
                dim_sizes[dim_name] = max(dim_sizes.get(dim_name, base_size), base_size, int(size))
        for dim_name, size in user_dim_sizes.items():
            if dim_name not in dim_sizes:
                dim_sizes[dim_name] = size

        if "S" not in dim_sizes:
            dim_sizes["S"] = effective_dim_sizes.get("S", 0)

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
                if dim_name in unlimited_dims:
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
                dim_names = first_dim_option(entry.get("dimensions"))
                dtype = dtype_for(entry.get("type"))
                var = dataset.createVariable(name, dtype, tuple(dim_names))
                default = entry.get("default")
                if default is None:
                    continue
                if len(dim_names) == 0:
                    var[...] = default
                    continue
                shape = tuple(dim_sizes.get(dim_name, 1) for dim_name in dim_names)
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
                            reshaped = reshape_for_broadcast(data, shape)
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
        complete_global_attributes(
            dataset,
            custom_global_attributes,
            override_default_global_attributes,
        )

        sofa_object = cls()
        sofa_object.netCDF4_dataset = dataset
        sofa_object.path = None
        print("Dummy SOFA dataset ready")
        return sofa_object

    def save(self, path: Optional[Union[str, pathlib.Path]] = None, overwrite: bool = False) -> pathlib.Path:
        """Save the SOFA dataset to disk.

        Parameters
        ----------
        path : Optional[Union[str, pathlib.Path]], optional
            Target path. If None, saves to the original path.
        overwrite : bool, optional
            If True, allows overwriting an existing file.

        Returns
        -------
        pathlib.Path
            Path to the saved SOFA file.

        Notes
        -----
        Cloned SOFA objects are independent in-memory datasets, so creating
        multiple clones from the same source object is supported. Saving
        still follows normal filesystem rules: if you try to overwrite the
        same on-disk path while another dataset keeps that file open, close
        the original dataset first or save to a different path.

        Examples
        --------
        Save a clone to a new file path:

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()
        >>> sofa_clone.save("my_copy.sofa")

        Save the currently opened writable dataset back to its original path:

        >>> sofa = SOFA.load("my.sofa", mode="r+")
        >>> sofa.save()

        Overwrite the original path from a separate clone:

        >>> sofa_ro = SOFA.load("my.sofa")
        >>> sofa_clone = sofa_ro.clone()
        >>> sofa_ro.netCDF4_dataset.close()
        >>> sofa_clone.save("my.sofa", overwrite=True)
        """
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

    def clone(self) -> "SOFA":
        """Create an in-memory writable clone of the current SOFA object.

        Returns
        -------
        SOFA
            A new SOFA instance backed by a diskless dataset.

        Notes
        -----
        The clone is an in-memory writable copy of the current dataset.
        Each call creates a new independent diskless NetCDF dataset, so
        cloning the same SOFA object multiple times is supported.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone = sofa.clone()

        >>> sofa = SOFA.load("my.sofa")
        >>> sofa_clone_1 = sofa.clone()
        >>> sofa_clone_2 = sofa.clone()
        """
        if self.netCDF4_dataset is None:
            raise ValueError("Dataset is not loaded")

        src = self.netCDF4_dataset
        file_format = getattr(src, "file_format", "NETCDF4")
        dst = netCDF4.Dataset(
            f"inmemory_{uuid4().hex}",
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

    def copy_with(
        self,
        dim_sizes: dict[str, int] | None = None,
        global_attributes: dict[str, Any] | None = None,
        variable_attributes: dict[str, dict[str, Any]] | None = None,
        variables: dict[str, np.ndarray] | None = None,
    ) -> "SOFA":
        """Create a modified in-memory copy of the current SOFA object.

        Parameters
        ----------
        dim_sizes : dict[str, int] | None
            Dimension size overrides. Only fixed dimensions may be overridden.
        global_attributes : dict[str, Any] | None
            Global attributes to add or replace.
        variable_attributes : dict[str, dict[str, Any]] | None
            Per-variable attributes to add or replace.
        variables : dict[str, numpy.ndarray] | None
            Variable data overrides. Only existing variable names are supported.

        Returns
        -------
        SOFA
            A new SOFA instance backed by a diskless dataset.

        Raises
        ------
        ValueError
            If the dataset is not loaded, an override refers to a missing
            dimension or variable, or a provided array cannot be broadcast
            to the target variable shape.

        Warnings
        --------
        - If you override a dimension size, you must also provide replacement
          arrays for variables that depend on that dimension (e.g., ``Data.IR``),
          otherwise shape checks will fail.
        - Only existing variables can be overridden; use a separate creation
          workflow if you need to add brand-new variables.

        Examples
        --------
        Resize the ``N`` dimension and replace ``Data.IR`` accordingly:

        >>> sofa = SOFA.load("my_sofa.sofa")
        >>> new_ir = np.zeros((1550, 2, 200))
        >>> sofa_mod = sofa.copy_with(
        ...     dim_sizes={"N": 200},
        ...     variables={"Data.IR": new_ir},
        ... )

        Override global and variable attributes:

        >>> sofa_mod = sofa.copy_with(
        ...     global_attributes={"Title": "Modified HRTF"},
        ...     variable_attributes={"Data.IR": {"Units": "Pa"}},
        ... )

        """
        if self.netCDF4_dataset is None:
            raise ValueError("Dataset is not loaded")

        src = self.netCDF4_dataset
        file_format = getattr(src, "file_format", "NETCDF4")
        dst = netCDF4.Dataset(
            f"inmemory_{uuid4().hex}",
            mode="w",
            diskless=True,
            persist=False,
            format=file_format,
        )
        try:
            override_dims = dim_sizes or {}
            missing_dims = set(override_dims) - set(src.dimensions)
            if missing_dims:
                missing_str = ", ".join(sorted(missing_dims))
                raise ValueError(f"Dimensions not found: {missing_str}")

            for name, dim in src.dimensions.items():
                if dim.isunlimited():
                    if name in override_dims:
                        raise ValueError(f"Cannot override unlimited dimension: {name}")
                    size = None
                else:
                    size = override_dims.get(name, dim.size)
                dst.createDimension(name, size)

            dst.setncatts({name: getattr(src, name) for name in src.ncattrs()})
            if global_attributes:
                for attr_name, value in global_attributes.items():
                    setattr(dst, attr_name, value)

            override_vars = variables or {}
            missing_vars = set(override_vars) - set(src.variables)
            if missing_vars:
                missing_str = ", ".join(sorted(missing_vars))
                raise ValueError(f"Variables not found: {missing_str}")

            var_attr_overrides = variable_attributes or {}
            missing_attr_vars = set(var_attr_overrides) - set(src.variables)
            if missing_attr_vars:
                missing_str = ", ".join(sorted(missing_attr_vars))
                raise ValueError(f"Variables not found: {missing_str}")

            for name, var in src.variables.items():
                dst_var = dst.createVariable(name, var.datatype, var.dimensions)
                dst_var.setncatts({attr: getattr(var, attr) for attr in var.ncattrs()})
                if name in var_attr_overrides:
                    for attr_name, value in var_attr_overrides[name].items():
                        setattr(dst_var, attr_name, value)

                data = override_vars[name] if name in override_vars else var[:]
                array = np.array(data)
                target_shape: list[int] = []
                for idx, dim_name in enumerate(var.dimensions):
                    dim = dst.dimensions[dim_name]
                    if dim.isunlimited():
                        if idx < array.ndim:
                            target_shape.append(array.shape[idx])
                        else:
                            target_shape.append(dst_var.shape[idx])
                    else:
                        target_shape.append(dim.size)
                ensure_broadcastable(name, array, tuple(target_shape))
                dst_var[...] = array
        except Exception:
            dst.close()
            raise

        sofa_object = SOFA()
        sofa_object.netCDF4_dataset = dst
        sofa_object.path = None
        return sofa_object

    def summary(self) -> str:
        """Return a formatted summary of GLOBALS Attributes, Variables and Variables Attributes.

        Returns
        -------
        str
            Multi-line summary string.

        Examples
        --------
        >>> sofa = SOFA.load("my.sofa")
        >>> print(sofa.summary())
        """
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
