from typing import Any, Dict, Optional, Union
import pathlib
from uuid import uuid4
import os
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


def load_sofa(
    path: Union[str, pathlib.Path],
    mode: str = "r",
    parallel: bool = False,
    check_sofa_against_conventions: bool = True,
) -> "SOFA":
    """Load a SOFA file into a :class:`~hrtfpykit.sofa.SOFA` object.

    This is the public entry point for inspecting or editing SOFA files with
    hrtfpykit. The file is opened with netCDF4, wrapped in a
    :class:`~hrtfpykit.sofa.SOFA` instance, and optionally checked
    against the SOFA convention declared by its global attributes.
    :class:`~hrtfpykit.hrtf.HRTF` objects use this function internally
    when loading SimpleFreeFieldHRIR and SimpleFreeFieldHRTF files.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        Path to an existing ``.sofa`` file.
    mode : str, default=``r``
        netCDF4 open mode used to open the file, such as ``r`` for
        read-only access or ``r+`` for direct in-place edits.
    parallel : bool, default=False
        Whether to request netCDF4 parallel I/O support when opening the
        netCDF4 object. This is forwarded directly to netCDF4.Dataset.
    check_sofa_against_conventions : bool, default=True
        Whether to validate the file against its declared
        ``SOFAConventions`` metadata before opening it as a
        :class:`~hrtfpykit.sofa.SOFA` object.
        Convention mismatches are reported by the validation utility, usually
        as SOFA convention warnings.

    Returns
    -------
    :class:`~hrtfpykit.sofa.SOFA`
        :class:`~hrtfpykit.sofa.SOFA` object whose
        :attr:`~hrtfpykit.sofa.SOFA.netCDF4_dataset` attribute is an open
        netCDF4 storage handle and whose
        :attr:`~hrtfpykit.sofa.SOFA.path` attribute points to ``path``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` does not end in ``.sofa``.
    OSError
        If netCDF4 cannot open the file with the requested mode or parallel
        setting.

    Notes
    -----
    The returned object owns an open netCDF4 handle. Close
    :attr:`~hrtfpykit.sofa.SOFA.netCDF4_dataset` when the loaded file is no longer needed, or save
    to a separate path with :meth:`~hrtfpykit.sofa.SOFA.save` when working
    from a clone.

    Examples
    --------
    Open a SimpleFreeFieldHRIR convention SOFA file, inspect its convention
    metadata and HRIR array shape, then close the underlying netCDF4 handle
    when inspection is finished:

    >>> from hrtfpykit.sofa import load_sofa
    >>> sofa = load_sofa("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
    >>> try:
    ...     convention = sofa.GlobalAttributes.get("SOFAConventions").value
    ...     ir_shape = sofa.Variables.get("Data.IR").value.shape
    ... finally:
    ...     sofa.netCDF4_dataset.close()
    >>> convention
    'SimpleFreeFieldHRIR'
    >>> ir_shape
    (793, 2, 256)
    """
    print(f"Loading SOFA file from: {path}")
    sofa_object = SOFA()
    open_sofa(
        sofa_object,
        path,
        mode,
        parallel,
        check_sofa_against_conventions,
    )
    print("SOFA load complete")
    return sofa_object


class SOFA:
    def __init__(self):
        """Represent a SOFA file and its netCDF4 storage handle.

        :class:`~hrtfpykit.sofa.SOFA` is the library abstraction around an
        open netCDF4 object that follows a SOFA convention. It provides
        controlled access to dimensions, global attributes, variables, and
        variable attributes through hrtfpykit collection wrappers, plus
        explicit methods for creating, modifying, cloning, saving, and
        summarizing SOFA files.

        The class is used directly for SOFA inspection workflows and indirectly by
        :class:`~hrtfpykit.hrtf.HRTF`, where it acts as the persistence layer for
        HRIR/HRTF data, source positions, sampling metadata, and convention metadata.

        Notes
        -----
        No hidden I/O is performed. Files are only read or written when you call
        :func:`~hrtfpykit.sofa.load_sofa`,
        :meth:`~hrtfpykit.sofa.SOFA.save`, or other explicit editing
        methods. The underlying storage object is a netCDF4 Dataset, so
        standard netCDF4 rules and constraints apply. Methods that mutate SOFA
        content require a writable netCDF4 handle; the safest workflow is to
        call :meth:`~hrtfpykit.sofa.SOFA.clone` or
        :meth:`~hrtfpykit.sofa.SOFA.copy_with`, modify the in-memory copy,
        then save the result explicitly.

        Attributes
        ----------
        netCDF4_dataset : netCDF4.Dataset | None
            Open netCDF4 storage handle backing this
            :class:`~hrtfpykit.sofa.SOFA` object. None means no SOFA file
            has been loaded or created.
        path : pathlib.Path | None
            Original or most recent disk path associated with the SOFA object. In-memory
            clones and dummy SOFA objects start with None.

        """
        self.netCDF4_dataset: Optional[netCDF4.Dataset] = None
        self.path = None

    @property
    def Dimensions(self) -> Optional[_Dimensions]:
        """Return the SOFA dimension collection wrapper.

        The wrapper exposes dimension names, sizes, and per-dimension
        :class:`~hrtfpykit.sofa.wraps.DimensionsWrap` objects from the current netCDF4 storage handle.

        Returns
        -------
        Optional[_Dimensions]
            Dimension access wrapper, or None when no SOFA file is loaded.
        """
        if self.netCDF4_dataset is None:
            return None
        return _Dimensions(self.netCDF4_dataset)

    @property
    def GlobalAttributes(self) -> Optional[_GlobalAttributes]:
        """Return the SOFA global-attribute collection wrapper.

        Global attributes include convention metadata such as
        ``SOFAConventions``, ``SOFAConventionsVersion``, ``DataType``,
        application metadata, and file lifecycle timestamps.

        Returns
        -------
        Optional[_GlobalAttributes]
            Global-attribute access wrapper, or None when no SOFA file is
            loaded.
        """
        if self.netCDF4_dataset is None:
            return None
        return _GlobalAttributes(self.netCDF4_dataset)

    @property
    def Variables(self) -> Optional[_Variables]:
        """Return the SOFA variable collection wrapper.

        Variables contain the numeric and string arrays stored in the SOFA
        file, including HRTF/HRIR data arrays, source positions, sampling
        rates, and frequency vectors.

        Returns
        -------
        Optional[_Variables]
            Variable access wrapper, or None when no SOFA file is loaded.
        """
        if self.netCDF4_dataset is None:
            return None
        return _Variables(self.netCDF4_dataset)

    @property
    def VariableAttributes(self) -> Optional[_VariableAttributes]:
        """Return the SOFA variable-attribute collection wrapper.

        Variable attributes are exposed with ``Variable:Attribute`` keys,
        such as ``SourcePosition:Type`` or ``Data.SamplingRate:Units``.
        These attributes describe coordinate systems, units, and semantic
        labels required by SOFA-based HRTF workflows.

        Returns
        -------
        Optional[_VariableAttributes]
            Variable-attribute access wrapper, or None when no SOFA file is
            loaded.
        """
        if self.netCDF4_dataset is None:
            return None
        return _VariableAttributes(self.netCDF4_dataset)

    def create_dimension(self, name: str, value: int) -> None:
        """Create a fixed-size dimension in the loaded SOFA object.

        Dimensions define the axes used by SOFA variables. For example,
        ``M`` commonly indexes measurements or source positions, ``R`` indexes
        receivers, ``N`` indexes time samples, and ``E`` indexes string
        lengths or emitters depending on the convention.

        Parameters
        ----------
        name : str
            Dimension name to create.
        value : int
            Dimension size passed to netCDF4.Dataset.createDimension.

        Raises
        ------
        ValueError
            If no SOFA file is loaded or if the dimension already exists.
        Exception
            Propagates errors raised by netCDF4 when the netCDF4 storage handle is not
            writable or the dimension cannot be created.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

        """
        dataset = require_dataset(self)
        if name in dataset.dimensions:
            raise ValueError(f"Dimension attribute already exists: {name}")
        dataset.createDimension(name, value)
        print(f"Dimension: '{name}' created succesfully")

    def rename_dimension(self, old_name: str, new_name: str) -> None:
        """Rename an existing dimension in the loaded SOFA object.

        The operation delegates to netCDF4 renameDimension and therefore
        updates the dimension name at the storage layer. Use this only when you
        also understand how existing SOFA variables depend on the dimension.

        Parameters
        ----------
        old_name : str
            Existing dimension name.
        new_name : str
            New dimension name.

        Raises
        ------
        ValueError
            If no SOFA file is loaded or if old_name does not exist.
        Exception
            Propagates errors raised by netCDF4, including attempts to rename
            dimensions on read-only netCDF4 storage handles or to invalid names.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

        """
        dataset = require_dataset(self)
        if old_name not in dataset.dimensions:
            print(f"Dimension: '{old_name}' not found")
            raise ValueError(f"Dimension not found: {old_name}")
        dataset.renameDimension(old_name , new_name)
        print(f"Dimension: '{old_name}' renamed succesfully")

    def create_global_attribute(self, name: str, value: Optional[str] = None) -> None:
        """Create a global attribute on the loaded SOFA object.

        Global attributes describe file-level SOFA metadata such as
        ``SOFAConventions``, ``SOFAConventionsVersion``, ``DataType``,
        application metadata, and date fields. This method adds a new global
        attribute and refuses to overwrite an existing one.

        Parameters
        ----------
        name : str
            Global attribute name to create.
        value : Optional[str], optional
            Attribute value. An empty string is stored when None is
            supplied.

        Raises
        ------
        ValueError
            If no SOFA file is loaded or if the global attribute already
            exists.
        Exception
            Propagates errors raised by netCDF4 when the attribute cannot be
            written.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

        """
        dataset = require_dataset(self)
        if name in dataset.ncattrs():
            raise ValueError(f"Global attribute already exists: {name}")
        stored_value = "" if value is None else value
        setattr(dataset, name, stored_value)
        print(f"Global attribute: '{name}' created succesfully")

    def modify_global_attribute(self, name: str, value: str) -> None:
        """Modify an existing global attribute on the loaded SOFA object.

        This method updates file-level metadata that already exists on the
        netCDF4 storage handle. It is commonly used by HRTF save workflows to keep
        ``SOFAConventions``, ``DataType``, ``DateModified``, and hrtfpykit API
        metadata synchronized after IR/TF changes.

        Parameters
        ----------
        name : str
            Existing global attribute name.
        value : str
            New attribute value.

        Raises
        ------
        ValueError
            If no SOFA file is loaded or if the global attribute does not
            exist.
        Exception
            Propagates errors raised by netCDF4 when the attribute cannot be
            written.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

        """
        dataset = require_dataset(self)
        if name not in dataset.ncattrs():
            raise ValueError(f"Global attribute not found: {name}")
        setattr(dataset, name, value)
        print(f"Global attribute: '{name}' modified succesfully")

    def delete_global_attribute(self, name: str) -> None:
        """Delete a global attribute from the loaded SOFA object.

        Removing global attributes can make a SOFA file invalid if required
        convention metadata is deleted. Use this method for controlled editing
        workflows where the resulting file will be validated or completed
        before saving.

        Parameters
        ----------
        name : str
            Attribute name to remove.

        Raises
        ------
        ValueError
            If no SOFA file is loaded or if the global attribute does not
            exist.
        Exception
            Propagates errors raised by netCDF4 when the attribute cannot be
            removed.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

        """
        dataset = require_dataset(self)
        if name not in dataset.ncattrs():
            raise ValueError(f"Global attribute not found: {name}")
        delattr(dataset, name)
        print(f"Global attribute: '{name}' deleted succesfully")

    def create_variable_attribute(self, name: str, value: Optional[str] = None) -> None:
        """Create an attribute on an existing SOFA variable.

        Variable attributes describe variable-specific metadata such as units,
        coordinate system type, and long names. The public key format is
        ``Variable:Attribute`` so the same style can be used by
        :attr:`~hrtfpykit.sofa.SOFA.VariableAttributes` and
        :attr:`~hrtfpykit.sofa.wraps.VariablesWrap.attributes`.

        Parameters
        ----------
        name : str
            Attribute key in the form ``Variable:Attribute``.
        value : Optional[str], optional
            Attribute value. An empty string is stored when None is
            supplied.

        Raises
        ------
        ValueError
            If no SOFA file is loaded, ``name`` is malformed, the target
            variable does not exist, or the variable attribute already
            exists.
        Exception
            Propagates errors raised by netCDF4 when the attribute cannot be
            written.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

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
        """Modify an existing attribute on a SOFA variable.

        Use this method to update variable metadata without changing the
        variable data array. For HRTF files this is typically used for units
        and coordinate-system metadata, for example ``SourcePosition:Type`` or
        ``Data.SamplingRate:Units``.

        Parameters
        ----------
        name : str
            Attribute key in the form ``Variable:Attribute``.
        value : str
            New attribute value.

        Raises
        ------
        ValueError
            If no SOFA file is loaded, ``name`` is malformed, the target
            variable does not exist, or the variable attribute does not
            exist.
        Exception
            Propagates errors raised by netCDF4 when the attribute cannot be
            written.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

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
        """Delete an attribute from a SOFA variable.

        Removing variable attributes can make a SOFA file ambiguous or invalid
        when the attribute is required by the declared convention. For example,
        source-position units and coordinate-system type are required by
        hrtfpykit source-grid logic.

        Parameters
        ----------
        name : str
            Attribute key in the form ``Variable:Attribute``.

        Raises
        ------
        ValueError
            If no SOFA file is loaded, ``name`` is malformed, the target
            variable does not exist, or the variable attribute does not
            exist.
        Exception
            Propagates errors raised by netCDF4 when the attribute cannot be
            removed.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

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
        """Create a SOFA variable and optionally attach metadata.

        The new variable is created with the provided dimension names and data.
        Data are converted to a NumPy array, checked against the target
        dimension shape, and assigned through netCDF4 broadcasting. This is
        used by HRTF update workflows when a converted representation requires
        variables that were not present in the original SOFA file, such as
        creating frequency-domain variables when saving as
        SimpleFreeFieldHRTF.

        Parameters
        ----------
        name : str
            Variable name to create, for example ``Data.IR``,
            ``Data.Real``, ``Data.Imag``, or ``N``.
        data : Union[np.ndarray, list]
            Data assigned to the new variable. The value is converted with
            ``np.array`` before shape checks and assignment.
        dimensions : Union[tuple[str, ...], list[str]]
            Existing netCDF4 dimension names in storage order.
        dtype : Optional[Union[str, np.dtype]], optional
            Data type passed to netCDF4.Dataset.createVariable. If ``dtype`` is
            None, the converted data array dtype is used.
        attributes : Optional[Dict[str, Any]], optional
            Optional variable attributes to set after variable creation.

        Raises
        ------
        ValueError
            If no SOFA file is loaded, the variable already exists, one or more
            dimensions are missing, or data cannot be broadcast to the
            requested shape.
        Exception
            Propagates errors raised by netCDF4 during variable creation,
            attribute assignment, or data assignment.

        Notes
        -----
        The function warns when dimension sizes do not coincide with the
        netCDF4 dimensions and raises an error if the data cannot be broadcast
        to the target shape. Unlimited dimensions use the supplied data size on
        that axis when available.

        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

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

        The replacement data are converted to a NumPy array, validated against
        the variable's dimension shape, and assigned through netCDF4. The
        variable definition, dimensions, dtype, and attributes are preserved;
        only its stored values are replaced.

        Parameters
        ----------
        name : str
            Existing variable name.
        data : Union[np.ndarray, list]
            New variable data. The value is converted with ``np.array`` before
            shape checks and assignment.

        Raises
        ------
        ValueError
            If no SOFA file is loaded, the variable does not exist, or data
            cannot be broadcast to the variable shape.
        Exception
            Propagates errors raised by netCDF4 during data assignment.

        Notes
        -----
        The function warns when dimension sizes do not coincide with the
        netCDF4 dimensions and raises an error if the data cannot be broadcast
        to the target shape. For unlimited dimensions, the replacement array's
        axis length is accepted when provided.

        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

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
        """Delete a variable from the loaded SOFA object.

        This method removes a variable by name from the underlying netCDF4
        storage handle. Deleting convention-required variables can make a SOFA file
        invalid; validate or rebuild required content before saving the
        result.

        Parameters
        ----------
        name : str
            Variable name to remove.

        Raises
        ------
        ValueError
            If no SOFA file is loaded or if the variable does not exist.
        Exception
            Propagates errors raised by netCDF4 when the variable cannot be
            removed.

        Notes
        -----
        Editing a SOFA file requires writable access. The recommended
        workflow is to load the original SOFA file, create an in-memory
        clone with :meth:`~hrtfpykit.sofa.SOFA.clone`, apply edits to the clone, and save when
        you are ready. Direct in-place editing with mode ``r+`` is
        still available for expert users.

        """
        dataset = require_dataset(self)
        if name not in dataset.variables:
            raise ValueError(f"Variable not found: {name}")
        del dataset.variables[name]
        print(f"Variable: '{name}' deleted succesfully")

    @classmethod
    def create_dummy(
        cls,
        sofa_conventions: str,
        version: Optional[str] = None,
        dim_sizes: Optional[Dict[str, int]] = None,
        custom_global_attributes: Optional[Dict[str, str]] = None,
        override_default_global_attributes: bool = False,
    ) -> "SOFA":
        """Create an in-memory SOFA object from a supported convention.

        The returned object is backed by a diskless netCDF4 Dataset and is
        useful for synthetic examples, tests, and workflows that construct
        SOFA content before saving it to disk. Dimensions, variables, global
        attributes, and variable attributes are derived from hrtfpykit's local
        SOFA convention specifications.

        Parameters
        ----------
        sofa_conventions : str
            Supported SOFA convention name, such as
            ``SimpleFreeFieldHRIR`` or ``SimpleFreeFieldHRTF``.
        version : Optional[str], optional
            Convention version. If ``version`` is None, the latest version available in
            the local convention table is used.
        dim_sizes : Optional[Dict[str, int]], optional
            Dimension size overrides. Keys are normalized to uppercase.
            Passing size 0 for a non-reserved dimension creates that
            dimension as unlimited. The reserved ``S`` dimension must not be
            supplied and is always created as unlimited.
        custom_global_attributes : Optional[Dict[str, str]], optional
            Additional global attributes used to complete or override the
            default hrtfpykit metadata fields.
        override_default_global_attributes : bool, optional
            If ``True``, ``custom_global_attributes`` override existing default
            values. If ``False``, custom values fill only missing or empty
            metadata fields.

        Returns
        -------
        SOFA
            In-memory :class:`~hrtfpykit.sofa.SOFA` object backed by a writable diskless netCDF4
            netCDF4 storage handle. The returned object's :attr:`~hrtfpykit.sofa.SOFA.path` is None until it is
            saved.

        Raises
        ------
        ValueError
            If the convention name or version is unsupported, or if
            ``dim_sizes`` tries to override the reserved unlimited ``S``
            dimension.
        Exception
            Propagates errors raised by netCDF4 while creating dimensions,
            variables, attributes, or assigning default data.

        Notes
        -----
        The dummy :class:`~hrtfpykit.sofa.SOFA` object always includes an unlimited ``S`` dimension with
        initial size 0. Passing ``S`` in ``dim_sizes`` raises an error because
        ``S`` is reserved by the SOFA conventions. To create other unlimited
        dimensions, set their size to 0 in ``dim_sizes``.

        Defaults from the convention table are assigned when available. If a
        non-scalar default cannot be broadcast to the resolved variable shape,
        the variable is initialized with zeros so that the SOFA object remains
        structurally usable and can be filled by later calls to
        :meth:`~hrtfpykit.sofa.SOFA.modify_variable`.

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
        """Save the current SOFA object to disk.

        The method writes the currently loaded SOFA object content to disk.
        When ``path`` is omitted, the object is synchronized back to its
        original file path recorded in
        :attr:`~hrtfpykit.sofa.SOFA.path`. When ``path`` is provided, a
        complete netCDF4 copy is written to a temporary file and then moved
        into place.

        Parameters
        ----------
        path : Optional[Union[str, pathlib.Path]], optional
            Target path. If ``path`` is None, the loaded SOFA object is synchronized to the
            original path recorded in :attr:`~hrtfpykit.sofa.SOFA.path`.
        overwrite : bool, optional
            If ``True``, allows replacing an existing destination file when
            ``path`` is provided.

        Returns
        -------
        pathlib.Path
            Path to the saved SOFA file.

        Raises
        ------
        ValueError
            If no SOFA file is loaded or if ``path`` is omitted and the SOFA
            object has no original path.
        FileExistsError
            If the destination already exists and ``overwrite`` is ``False``.
        Exception
            Propagates errors raised by netCDF4 or the filesystem while
            copying dimensions, attributes, variables, or replacing the target
            file.

        Notes
        -----
        Calling :meth:`~hrtfpykit.sofa.SOFA.save` without a path synchronizes the loaded SOFA content
        back to the original file recorded in :attr:`~hrtfpykit.sofa.SOFA.path`.

        Cloned :class:`~hrtfpykit.sofa.SOFA` objects and objects returned by :meth:`~hrtfpykit.sofa.SOFA.copy_with` are
        independent in-memory SOFA objects. If one of those objects should replace
        an existing SOFA file on disk, save it to that filename with
        ``overwrite=True``. In that case the method writes a temporary copy
        first and then replaces the destination file.

        This method copies dimensions, global attributes, variables, variable
        attributes, and variable values. It does not run SOFA convention
        validation before writing; call the validation utilities explicitly
        when validation is part of the workflow.

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
        temp_path = target_path.with_name(f".{target_path.name}.{uuid4().hex}.tmp")
        dst = netCDF4.Dataset(str(temp_path), mode="w", format=file_format)
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
            if temp_path.exists():
                temp_path.unlink()
            raise
        else:
            dst.close()
            os.replace(temp_path, target_path)

        self.path = target_path
        print("SOFA save complete")
        return target_path

    def clone(self) -> "SOFA":
        """Create an in-memory writable clone of the current :class:`~hrtfpykit.sofa.SOFA` object.

        The clone contains a full copy of dimensions, global attributes,
        variables, variable attributes, and variable values from the current
        SOFA object. It is backed by a diskless netCDF4 Dataset and has no file
        path until saved.

        Returns
        -------
        SOFA
            New SOFA instance backed by an independent diskless netCDF4 storage handle.

        Raises
        ------
        ValueError
            If no SOFA file is loaded.
        Exception
            Propagates errors raised by netCDF4 while copying SOFA content.

        Notes
        -----
        The clone is an in-memory writable copy of the current SOFA object.
        Each call creates a new independent diskless netCDF4 storage handle, so
        cloning the same :class:`~hrtfpykit.sofa.SOFA` object multiple times is supported. Because the
        clone is independent from the original NetCDF handle, you can later
        save it to a new filename or replace an existing file with
        :meth:`~hrtfpykit.sofa.SOFA.save` with overwrite enabled.

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
        """Create a modified in-memory copy of the current :class:`~hrtfpykit.sofa.SOFA` object.

        :meth:`~hrtfpykit.sofa.SOFA.copy_with` is a structured cloning helper for workflows that need
        to resize fixed dimensions or replace existing arrays while preserving
        the rest of the SOFA file. It is used by HRTF save/update logic when
        transformed IR/TF data no longer match the original SOFA dimensions.

        Parameters
        ----------
        dim_sizes : dict[str, int] | None
            Size overrides for existing fixed dimensions. Unlimited dimensions
            cannot be overridden.
        global_attributes : dict[str, Any] | None
            Global attributes to add or replace in the copied SOFA object.
        variable_attributes : dict[str, dict[str, Any]] | None
            Per-variable attributes to add or replace. The outer keys are
            existing variable names and inner keys are attribute names without
            the ``Variable:`` prefix.
        variables : dict[str, numpy.ndarray] | None
            Replacement data for existing variables. New variables cannot be
            created with this method.

        Returns
        -------
        SOFA
            New SOFA instance backed by an independent diskless netCDF4 storage handle with
            the requested overrides applied.

        Raises
        ------
        ValueError
            If the SOFA file is not loaded, an override refers to a missing
            dimension or variable, or a provided array cannot be broadcast
            to the target variable shape.
        Exception
            Propagates errors raised by netCDF4 while copying or assigning
            SOFA content.

        Notes
        -----
        If you override a dimension size, provide replacement arrays for all
        variables that depend on that dimension unless their existing values
        can still broadcast to the new target shape. Use
        :meth:`~hrtfpykit.sofa.SOFA.create_variable`
        on a writable copy when you need to add brand-new variables.

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
        """Return a text summary of the loaded SOFA object.

        The summary is intended for quick inspection in notebooks, terminals,
        and debugging logs. It lists global attributes first, then each
        variable with its dimension names, dimension sizes, and variable
        attributes.

        Returns
        -------
        str
            Multi-line summary of global attributes, variables, dimensions,
            and variable attributes.

        Raises
        ------
        ValueError
            If no SOFA file is loaded.

        Notes
        -----
        This method does not validate the file and does not print by itself.
        It only builds and returns the summary string.

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
