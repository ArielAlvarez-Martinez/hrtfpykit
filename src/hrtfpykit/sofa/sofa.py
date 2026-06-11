from typing import Any, Dict, Optional, Union, cast
import datetime
import importlib.metadata
import pathlib
from uuid import uuid4
import os
import netCDF4
import numpy as np
from .data import  _Dimensions, _GlobalAttributes, _VariableAttributes, _Variables
from .sofa_helpers import (
    ensure_broadcastable,
    get_variable_creation_options,
    open_sofa,
    require_dataset,
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
    The returned object owns an open netCDF4 handle. Call
    :meth:`~hrtfpykit.sofa.SOFA.close` when SOFA-backed metadata and variable
    access are no longer needed. A file-backed SOFA object can later reopen
    the same path with :meth:`~hrtfpykit.sofa.SOFA.open`.

    Examples
    --------
    Open a SimpleFreeFieldHRIR convention SOFA file and inspect its convention
    metadata, source grid, and HRIR array through the SOFA wrappers:

    >>> from hrtfpykit.sofa import load_sofa
    >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
    >>> sofa.GlobalAttributes.get("SOFAConventions").value
    'SimpleFreeFieldHRIR'
    >>> sofa.Variables.get_names()
    ['ListenerPosition', 'ReceiverPosition', 'SourcePosition', 'EmitterPosition', 'ListenerUp', 'ListenerView', 'Data.IR', 'Data.SamplingRate']
    >>> sofa.Variables.get("SourcePosition").value.shape
    (793, 3)
    >>> sofa.Variables.get("Data.IR").value.shape
    (793, 2, 256)
    """
    sofa_object = SOFA()
    open_sofa(
        sofa_object,
        path,
        mode,
        parallel,
        check_sofa_against_conventions,
    )
    return sofa_object



class SOFA:
    def __init__(self) -> None:
        """Represent a SOFA file and its netCDF4 storage handle.

        :class:`~hrtfpykit.sofa.SOFA` is the library abstraction around an
        open netCDF4 object that follows a SOFA convention. It provides
        controlled access to dimensions, global attributes, variables, and
        variable attributes through hrtfpykit collection wrappers, plus
        explicit methods for opening, closing, creating, modifying, cloning,
        saving, and summarizing SOFA files.

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
            clones start with None.

        Examples
        --------
        Load a SOFA file and access its main metadata and array collections
        through a :class:`~hrtfpykit.sofa.SOFA` object:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.path.name
        'P0001_FreeFieldComp_44kHz.sofa'
        >>> sofa.Dimensions.get("M").value
        793
        >>> sofa.Dimensions.get("R").value
        2
        >>> sofa.Dimensions.get("N").value
        256
        >>> sofa.GlobalAttributes.get("SOFAConventions").value
        'SimpleFreeFieldHRIR'
        >>> sofa.VariableAttributes.get("SourcePosition:Type").value
        'spherical'
        >>> sofa.Variables.get("Data.IR").value.shape
        (793, 2, 256)
        """
        self.netCDF4_dataset: Optional[netCDF4.Dataset] = None
        self.path: pathlib.Path | None = None
        self._modified: bool = False
        self._change_messages: list[str] = []

    @property
    def Dimensions(self) -> _Dimensions:
        """Return the SOFA dimension collection wrapper.

        The wrapper exposes dimension names, sizes, and per-dimension
        :class:`~hrtfpykit.sofa.wraps.DimensionsWrap` objects from the current
        open netCDF4 storage handle.

        Returns
        -------
        _Dimensions
            Dimension access wrapper for the open SOFA dataset.

        Raises
        ------
        ValueError
            If no dataset is attached or if the SOFA dataset is closed.

        Examples
        --------
        Inspect the measurement, receiver, and sample dimensions of a loaded
        SimpleFreeFieldHRIR SOFA file:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.Dimensions.get("M").value
        793
        >>> sofa.Dimensions.get("R").value
        2
        >>> sofa.Dimensions.get("N").value
        256
        """
        dataset = require_dataset(self)
        return _Dimensions(dataset)

    @property
    def GlobalAttributes(self) -> _GlobalAttributes:
        """Return the SOFA global-attribute collection wrapper.

        Global attributes include convention metadata such as
        ``SOFAConventions``, ``SOFAConventionsVersion``, ``DataType``,
        application metadata, and file lifecycle timestamps.

        Returns
        -------
        _GlobalAttributes
            Global-attribute access wrapper for the open SOFA dataset.

        Raises
        ------
        ValueError
            If no dataset is attached or if the SOFA dataset is closed.

        Examples
        --------
        Read convention-level metadata from a loaded SOFA file:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.GlobalAttributes.get("SOFAConventions").value
        'SimpleFreeFieldHRIR'
        >>> sofa.GlobalAttributes.get("DataType").value
        'FIR'
        """
        dataset = require_dataset(self)
        return _GlobalAttributes(dataset)

    @property
    def Variables(self) -> _Variables:
        """Return the SOFA variable collection wrapper.

        Variables contain the numeric and string arrays stored in the SOFA
        file, including HRTF/HRIR data arrays, source positions, sampling
        rates, and frequency vectors.

        Returns
        -------
        _Variables
            Variable access wrapper for the open SOFA dataset.

        Raises
        ------
        ValueError
            If no dataset is attached or if the SOFA dataset is closed.

        Examples
        --------
        Read acoustic data, source positions, and sample-rate values from a
        loaded SOFA file:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.Variables.get("Data.IR").value.shape
        (793, 2, 256)
        >>> sofa.Variables.get("SourcePosition").value.shape
        (793, 3)
        >>> sofa.Variables.get("Data.SamplingRate").value
        array([44100.])
        """
        dataset = require_dataset(self)
        return _Variables(dataset)

    @property
    def VariableAttributes(self) -> _VariableAttributes:
        """Return the SOFA variable-attribute collection wrapper.

        Variable attributes are exposed with ``Variable:Attribute`` keys,
        such as ``SourcePosition:Type`` or ``Data.SamplingRate:Units``.
        These attributes describe coordinate systems, units, and semantic
        labels required by SOFA-based HRTF workflows.

        Returns
        -------
        _VariableAttributes
            Variable-attribute access wrapper for the open SOFA dataset.

        Raises
        ------
        ValueError
            If no dataset is attached or if the SOFA dataset is closed.

        Examples
        --------
        Read coordinate-system and units metadata for source positions:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.VariableAttributes.get("SourcePosition:Type").value
        'spherical'
        >>> sofa.VariableAttributes.get("SourcePosition:Units").value
        'degree, degree, metre'
        """
        dataset = require_dataset(self)
        return _VariableAttributes(dataset)

    def is_open(self) -> bool:
        """Return whether the SOFA netCDF4 dataset is currently open.

        The method reports the state of the backing netCDF4 handle owned by
        this :class:`~hrtfpykit.sofa.SOFA` object. It returns ``False`` when no
        dataset has been attached or when the attached dataset has been closed.

        Returns
        -------
        bool
            ``True`` when the backing dataset is open, otherwise ``False``.

        Examples
        --------
        Check the storage state before and after closing a loaded SOFA file:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.is_open()
        True
        >>> sofa.close()
        >>> sofa.is_open()
        False
        """
        if self.netCDF4_dataset is None:
            return False
        is_open = getattr(self.netCDF4_dataset, "isopen", None)
        if callable(is_open):
            return bool(is_open())
        return True

    def close(self) -> None:
        """Close the backing SOFA netCDF4 dataset.

        Closing releases the netCDF4 file handle owned by this
        :class:`~hrtfpykit.sofa.SOFA` object. The object keeps its ``path`` and
        ``netCDF4_dataset`` reference, but SOFA-backed accessors such as
        :attr:`Dimensions`, :attr:`GlobalAttributes`, :attr:`Variables`,
        :attr:`VariableAttributes`, and :meth:`summary` require an open dataset
        and raise ``ValueError`` after close. File-backed SOFA objects can be
        reopened with :meth:`open`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no dataset is attached or if the dataset is already closed.

        Examples
        --------
        Close a loaded SOFA file after reading the metadata needed by the
        application:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.close()
        >>> sofa.is_open()
        False
        """
        dataset = require_dataset(self)
        dataset.close()

    def open(
        self,
        mode: str = "r",
        parallel: bool = False,
        check_sofa_against_conventions: bool = True,
    ) -> None:
        """Open or reopen the SOFA netCDF4 dataset from ``path``.

        This method opens the file stored in :attr:`path` and attaches the new
        netCDF4 handle to the current :class:`~hrtfpykit.sofa.SOFA` object. If
        the dataset is already open, the method returns without replacing the
        active handle. Objects without a path, such as unsaved in-memory SOFA
        clones, cannot be reopened from disk.

        Parameters
        ----------
        mode : str, default=``r``
            netCDF4 file mode used to open the file.
        parallel : bool, default=False
            Whether to request netCDF4 parallel I/O support.
        check_sofa_against_conventions : bool, default=True
            Whether to validate the file against its declared SOFA convention
            before opening it.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no path is available for this SOFA object.
        FileNotFoundError
            If the stored path does not exist.
        OSError
            If netCDF4 cannot open the file with the requested options.

        Examples
        --------
        Reopen a file-backed SOFA object after closing its dataset handle:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.close()
        >>> sofa.open()
        >>> sofa.is_open()
        True
        """
        if self.is_open():
            return
        if self.path is None:
            raise ValueError("No path available to open the dataset")
        open_sofa(
            self,
            self.path,
            mode,
            parallel,
            check_sofa_against_conventions,
        )

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
        attributes, variable values, and variable storage metadata such as
        compression filters, chunking, endian settings, checksums, quantization
        options, and fill values. It does not run SOFA convention validation
        before writing; call the validation utilities explicitly when validation
        is part of the workflow.

        Examples
        --------
        Save an edited clone to a relative output path:

        >>> from pathlib import Path
        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_global_attribute("ExampleNote", "saved copy")
        >>> output_dir = Path("processed")
        >>> output_dir.mkdir(exist_ok=True)
        >>> saved_path = editable.save(
        ...     output_dir / "P0001_sofa_copy.sofa",
        ...     overwrite=True,
        ... )
        >>> saved_path.name
        'P0001_sofa_copy.sofa'
        """
        dataset = require_dataset(self)

        target_path: pathlib.Path | None = None
        original_path: pathlib.Path | None = None
        if path is None:
            if self.path is None:
                raise ValueError("No path available to save the dataset")
            original_path = pathlib.Path(self.path)
        else:
            target_path = pathlib.Path(path)
            if target_path.exists() and not overwrite:
                raise FileExistsError(f"SOFA file already exists: {target_path}")

        self._change_messages = []

        if self._modified:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                hrtfpykit_version = importlib.metadata.version("hrtfpykit")
            except importlib.metadata.PackageNotFoundError:
                hrtfpykit_version = "unknown"
            hrtfpykit_stamp = (
                f"Modified using hrtfpykit v{hrtfpykit_version} on {now}"
            )
            if "DateModified" in dataset.ncattrs():
                self.modify_global_attribute("DateModified", now)
            else:
                self.create_global_attribute("DateModified", now)
            if "hrtfpykit" in dataset.ncattrs():
                self.modify_global_attribute("hrtfpykit", hrtfpykit_stamp)
            else:
                self.create_global_attribute("hrtfpykit", hrtfpykit_stamp)

        if path is None:
            dataset.sync()
            self._modified = False
            if original_path is None:
                raise ValueError("No path available to save the dataset")
            return original_path

        src = dataset
        if target_path is None:
            raise ValueError("No path available to save the dataset")
        file_format = getattr(src, "file_format", "NETCDF4")
        temp_path = target_path.with_name(f".{target_path.name}.{uuid4().hex}.tmp")
        dst = netCDF4.Dataset(str(temp_path), mode="w", format=cast(Any, file_format))
        try:
            for name, dim in src.dimensions.items():
                size = None if dim.isunlimited() else dim.size
                dst.createDimension(name, size)

            dst.setncatts({name: getattr(src, name) for name in src.ncattrs()})

            for name, var in src.variables.items():
                creation_options = get_variable_creation_options(var)
                if "chunksizes" in creation_options:
                    target_chunksizes: list[int] = []
                    for dim_name, chunk_size in zip(var.dimensions, creation_options["chunksizes"]):
                        dim = dst.dimensions[dim_name]
                        if dim.isunlimited():
                            target_chunksizes.append(int(chunk_size))
                        else:
                            target_chunksizes.append(min(int(chunk_size), int(dim.size)))
                    creation_options["chunksizes"] = tuple(target_chunksizes)
                dst_var = dst.createVariable(
                    name,
                    var.datatype,
                    var.dimensions,
                    **creation_options,
                )
                dst_var.setncatts(
                    {
                        attr: getattr(var, attr)
                        for attr in var.ncattrs()
                        if attr != "_FillValue"
                    }
                )
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
        self._modified = False
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
            If no dataset is attached or if the SOFA dataset is closed.
        Exception
            Propagates errors raised by netCDF4 while copying SOFA content.

        Notes
        -----
        The clone is an in-memory writable copy of the current SOFA object.
        Each call creates a new independent diskless netCDF4 storage handle, so
        cloning the same :class:`~hrtfpykit.sofa.SOFA` object multiple times is supported. Because the
        clone is independent from the original NetCDF handle, you can later
        save it to a new filename or replace an existing file with
        :meth:`~hrtfpykit.sofa.SOFA.save` with overwrite enabled. Variable
        storage metadata such as compression filters, chunking, endian settings,
        checksums, quantization options, and fill values are preserved.

        Examples
        --------
        Clone a loaded SOFA object before editing metadata, leaving the source
        SOFA object unchanged:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_global_attribute("ExampleNote", "clone only")
        >>> editable.path is None
        True
        >>> editable.GlobalAttributes.get("ExampleNote").value
        'clone only'
        >>> "ExampleNote" in sofa.GlobalAttributes.get_names()
        False
        """
        src = require_dataset(self)
        file_format = getattr(src, "file_format", "NETCDF4")
        dst = netCDF4.Dataset(
            f"inmemory_{uuid4().hex}",
            mode="w",
            diskless=True,
            persist=False,
            format=cast(Any, file_format),
        )
        try:
            for name, dim in src.dimensions.items():
                size = None if dim.isunlimited() else dim.size
                dst.createDimension(name, size)

            dst.setncatts({name: getattr(src, name) for name in src.ncattrs()})

            for name, var in src.variables.items():
                creation_options = get_variable_creation_options(var)
                if "chunksizes" in creation_options:
                    target_chunksizes: list[int] = []
                    for dim_name, chunk_size in zip(var.dimensions, creation_options["chunksizes"]):
                        dim = dst.dimensions[dim_name]
                        if dim.isunlimited():
                            target_chunksizes.append(int(chunk_size))
                        else:
                            target_chunksizes.append(min(int(chunk_size), int(dim.size)))
                    creation_options["chunksizes"] = tuple(target_chunksizes)
                dst_var = dst.createVariable(
                    name,
                    var.datatype,
                    var.dimensions,
                    **creation_options,
                )
                dst_var.setncatts(
                    {
                        attr: getattr(var, attr)
                        for attr in var.ncattrs()
                        if attr != "_FillValue"
                    }
                )
                dst_var[:] = var[:]
        except Exception:
            dst.close()
            raise

        sofa_object = SOFA()
        sofa_object.netCDF4_dataset = dst
        sofa_object.path = None
        sofa_object._modified = self._modified
        sofa_object._change_messages = list(self._change_messages)
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
        can still broadcast to the new target shape. Variable storage metadata
        such as compression filters, chunking, endian settings, checksums,
        quantization options, and fill values are preserved. Chunk sizes are
        clipped to resized fixed dimensions when needed. Use
        :meth:`~hrtfpykit.sofa.SOFA.create_variable`
        on a writable copy when you need to add brand-new variables.

        Examples
        --------
        Create a resized copy with shorter HRIR data while keeping the original
        SOFA object unchanged:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> ir = sofa.Variables.get("Data.IR").value
        >>> cropped = sofa.copy_with(
        ...     dim_sizes={"N": 128},
        ...     variables={"Data.IR": ir[..., :128]},
        ... )
        >>> sofa.Variables.get("Data.IR").value.shape
        (793, 2, 256)
        >>> cropped.Variables.get("Data.IR").value.shape
        (793, 2, 128)
        """
        src = require_dataset(self)
        file_format = getattr(src, "file_format", "NETCDF4")
        change_messages: list[str] = []
        dst = netCDF4.Dataset(
            f"inmemory_{uuid4().hex}",
            mode="w",
            diskless=True,
            persist=False,
            format=cast(Any, file_format),
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
                    if name in override_dims and size != dim.size:
                        change_messages.append(
                            f"Dimension: '{name}' modified succesfully"
                        )
                dst.createDimension(name, size)

            dst.setncatts({name: getattr(src, name) for name in src.ncattrs()})
            if global_attributes:
                for attr_name, value in global_attributes.items():
                    if attr_name in src.ncattrs():
                        if getattr(src, attr_name) != value:
                            change_messages.append(
                                f"Global attribute: '{attr_name}' modified succesfully"
                            )
                    else:
                        change_messages.append(
                            f"Global attribute: '{attr_name}' created succesfully"
                        )
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
                creation_options = get_variable_creation_options(var)
                if "chunksizes" in creation_options:
                    target_chunksizes: list[int] = []
                    for dim_name, chunk_size in zip(var.dimensions, creation_options["chunksizes"]):
                        dim = dst.dimensions[dim_name]
                        if dim.isunlimited():
                            target_chunksizes.append(int(chunk_size))
                        else:
                            target_chunksizes.append(min(int(chunk_size), int(dim.size)))
                    creation_options["chunksizes"] = tuple(target_chunksizes)
                dst_var = dst.createVariable(
                    name,
                    var.datatype,
                    var.dimensions,
                    **creation_options,
                )
                dst_var.setncatts(
                    {
                        attr: getattr(var, attr)
                        for attr in var.ncattrs()
                        if attr != "_FillValue"
                    }
                )
                if name in var_attr_overrides:
                    for attr_name, value in var_attr_overrides[name].items():
                        attribute_name = f"{name}:{attr_name}"
                        if attr_name in var.ncattrs():
                            if getattr(var, attr_name) != value:
                                change_messages.append(
                                    f"Variable attribute: '{attribute_name}' modified succesfully"
                                )
                        else:
                            change_messages.append(
                                f"Variable attribute: '{attribute_name}' created succesfully"
                            )
                        setattr(dst_var, attr_name, value)

                data = override_vars[name] if name in override_vars else var[:]
                array = np.array(data)
                if name in override_vars:
                    source_array = np.array(var[:])
                    variable_modified = True
                    try:
                        comparable_array = np.broadcast_to(array, source_array.shape)
                    except ValueError:
                        variable_modified = True
                    else:
                        if (
                            np.issubdtype(source_array.dtype, np.number)
                            and np.issubdtype(comparable_array.dtype, np.number)
                        ):
                            variable_modified = not np.allclose(
                                source_array,
                                comparable_array,
                            )
                        else:
                            variable_modified = not np.array_equal(
                                source_array,
                                comparable_array,
                            )
                    if variable_modified:
                        change_messages.append(
                            f"Variable: '{name}' modified succesfully"
                        )
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
        sofa_object._modified = self._modified or bool(
            dim_sizes or global_attributes or variable_attributes or variables
        )
        sofa_object._change_messages = list(self._change_messages) + change_messages
        return sofa_object

    def summary(self) -> str:
        """Return a text summary of the loaded SOFA object.

        The summary is intended for quick inspection in notebooks, terminals,
        and debugging logs. It lists global attributes first, then each
        variable with its dimension names, dimension sizes, and variable
        attributes. The backing SOFA dataset must be open.

        Returns
        -------
        str
            Multi-line summary of global attributes, variables, dimensions,
            and variable attributes.

        Raises
        ------
        ValueError
            If no dataset is attached or if the SOFA dataset is closed.

        Notes
        -----
        This method does not validate the file and does not print by itself.
        It only builds and returns the summary string.

        Examples
        --------
        Build a text summary and check that it includes the main SOFA sections
        and HRIR variable dimensions:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> summary = sofa.summary()
        >>> summary.splitlines()[:3]
        ['****************************', '     GLOBAL ATTRIBUTES', '****************************']
        >>> "Data.IR : dimensions= (M=793, R=2, N=256)" in summary
        True
        """
        dataset = require_dataset(self)
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
                dim_size: int | str
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

        Examples
        --------
        Add a custom dimension to an in-memory clone and inspect the new
        dimension metadata:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_dimension("Q", 3)
        >>> editable.Dimensions.get("Q").value
        3
        >>> editable.Dimensions.get("Q").is_unlimited
        False
        """
        dataset = require_dataset(self)
        if name in dataset.dimensions:
            raise ValueError(f"Dimension attribute already exists: {name}")
        dataset.createDimension(name, value)
        self._modified = True
        self._change_messages.append(f"Dimension: '{name}' created succesfully")

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

        Examples
        --------
        Rename a custom dimension on an in-memory clone before adding variables
        that depend on it:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_dimension("Q", 2)
        >>> editable.rename_dimension("Q", "Q2")
        >>> editable.Dimensions.get("Q2").value
        2
        """
        dataset = require_dataset(self)
        if old_name not in dataset.dimensions:
            self._change_messages.append(f"Dimension: '{old_name}' not found")
            raise ValueError(f"Dimension not found: {old_name}")
        dataset.renameDimension(old_name , new_name)
        self._modified = True
        self._change_messages.append(f"Dimension: '{old_name}' renamed succesfully")

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

        Examples
        --------
        Add file-level metadata to a clone without changing the source SOFA
        file:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_global_attribute(
        ...     "ExampleNote",
        ...     "created from a clone",
        ... )
        >>> editable.GlobalAttributes.get("ExampleNote").value
        'created from a clone'
        """
        dataset = require_dataset(self)
        if name in dataset.ncattrs():
            raise ValueError(f"Global attribute already exists: {name}")
        stored_value = "" if value is None else value
        setattr(dataset, name, stored_value)
        self._modified = True
        self._change_messages.append(f"Global attribute: '{name}' created succesfully")

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

        Examples
        --------
        Update a global attribute on a cloned SOFA object and read the edited
        value back through the global-attribute wrapper:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_global_attribute("ExampleNote", "initial note")
        >>> editable.modify_global_attribute("ExampleNote", "updated note")
        >>> editable.GlobalAttributes.get("ExampleNote").value
        'updated note'
        """
        dataset = require_dataset(self)
        if name not in dataset.ncattrs():
            raise ValueError(f"Global attribute not found: {name}")
        setattr(dataset, name, value)
        self._modified = True
        self._change_messages.append(f"Global attribute: '{name}' modified succesfully")

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

        Examples
        --------
        Remove a custom global attribute from an in-memory clone:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_global_attribute("ExampleNote", "temporary")
        >>> editable.delete_global_attribute("ExampleNote")
        >>> "ExampleNote" in editable.GlobalAttributes.get_names()
        False
        """
        dataset = require_dataset(self)
        if name not in dataset.ncattrs():
            raise ValueError(f"Global attribute not found: {name}")
        delattr(dataset, name)
        self._modified = True
        self._change_messages.append(f"Global attribute: '{name}' deleted succesfully")

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

        Examples
        --------
        Add metadata to the HRIR variable on a cloned SOFA object:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_variable_attribute(
        ...     "Data.IR:ExampleNote",
        ...     "time-domain data",
        ... )
        >>> editable.VariableAttributes.get("Data.IR:ExampleNote").value
        'time-domain data'
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
        self._modified = True
        self._change_messages.append(f"Variable attribute: '{name}' created succesfully")

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

        Examples
        --------
        Update metadata on the HRIR variable of a cloned SOFA object:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_variable_attribute("Data.IR:ExampleNote", "initial")
        >>> editable.modify_variable_attribute("Data.IR:ExampleNote", "edited copy")
        >>> editable.VariableAttributes.get("Data.IR:ExampleNote").value
        'edited copy'
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
        self._modified = True
        self._change_messages.append(f"Variable attribute: '{name}' modified succesfully")

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

        Examples
        --------
        Delete a custom variable attribute after using it during an editing
        workflow:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_variable_attribute("Data.IR:ExampleNote", "temporary")
        >>> editable.delete_variable_attribute("Data.IR:ExampleNote")
        >>> "Data.IR:ExampleNote" in editable.VariableAttributes.get_names()
        False
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
        self._modified = True
        self._change_messages.append(f"Variable attribute: '{name}' deleted succesfully")

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

        Examples
        --------
        Create a small derived variable on a cloned SOFA object and attach
        units metadata to it:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_dimension("Q", 3)
        >>> editable.create_variable(
        ...     "ExampleVector",
        ...     [1.0, 2.0, 3.0],
        ...     ("Q",),
        ...     attributes={"Units": "1"},
        ... )
        >>> editable.Variables.get("ExampleVector").value
        array([1., 2., 3.])
        >>> editable.VariableAttributes.get("ExampleVector:Units").value
        '1'
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

        self._modified = True
        self._change_messages.append(f"Variable: '{name}' created succesfully")

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

        Examples
        --------
        Replace HRIR samples on a cloned SOFA object while preserving the
        ``Data.IR`` variable definition and metadata:

        >>> import numpy as np
        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> ir = editable.Variables.get("Data.IR").value
        >>> edited_ir = np.array(ir, copy=True)
        >>> edited_ir[..., :8] = 0.0
        >>> editable.modify_variable("Data.IR", edited_ir)
        >>> editable.Variables.get("Data.IR").value[0, 0, :8]
        array([0., 0., 0., 0., 0., 0., 0., 0.])
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
        self._modified = True
        self._change_messages.append(f"Variable: '{name}' modified succesfully")

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

        Examples
        --------
        Remove a custom variable from a cloned SOFA object after using it as
        temporary metadata:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> editable = sofa.clone()
        >>> editable.create_dimension("Q", 3)
        >>> editable.create_variable("ExampleVector", [1.0, 2.0, 3.0], ("Q",))
        >>> editable.delete_variable("ExampleVector")
        >>> "ExampleVector" in editable.Variables.get_names()
        False
        """
        dataset = require_dataset(self)
        if name not in dataset.variables:
            raise ValueError(f"Variable not found: {name}")
        del dataset.variables[name]
        self._modified = True
        self._change_messages.append(f"Variable: '{name}' deleted succesfully")
