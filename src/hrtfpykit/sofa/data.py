from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Iterator
import netCDF4
import numpy as np
from .wraps import DimensionsWrap, VariablesWrap, AttributesWrap


class _Data(ABC):
    def __init__(self, dataset: netCDF4.Dataset | None = None):
        """Define shared behavior for SOFA collection wrappers.

        ``_Data`` stores the open netCDF4 storage handle used by a
        :class:`~hrtfpykit.sofa.SOFA` instance and defines the collection
        interface shared by dimensions, global attributes, variable
        attributes, and variables. Concrete subclasses expose SOFA storage
        objects through hrtfpykit wrapper classes so callers can inspect SOFA
        content without working directly with raw netCDF4 mappings.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            Open netCDF4 storage handle containing SOFA dimensions, variables, and
            attributes. Must not be None.

        Raises
        ------
        ValueError
            If ``dataset`` is None.

        Attributes
        ----------
        _netCDF4_dataset : netCDF4.Dataset
            netCDF4 storage handle backing the collection wrapper.
        """
        if dataset is None:
            raise ValueError("Dataset is required")
        self._netCDF4_dataset = dataset

    @abstractmethod
    def get(self, name: str):
        """Return a wrapped item by name.

        Parameters
        ----------
        name : str
            Item name or key.

        Returns
        -------
        object
            Collection-specific wrapper object.

        Raises
        ------
        ValueError
            If ``name`` cannot be resolved by the concrete collection.
        """
        pass

    @abstractmethod
    def get_names(self):
        """Return available item names for the wrapped collection.

        Returns
        -------
        list[str]
            Names or keys that can be passed to
            :meth:`~hrtfpykit.sofa.data._Data.get`.
        """
        pass

    @abstractmethod
    def get_values(self):
        """Return raw values for the wrapped collection.

        Returns
        -------
        list
            Values associated with the collection items.
        """
        pass

    @abstractmethod
    def get_all(self):
        """Return all wrapped items keyed by name.

        Returns
        -------
        dict
            Mapping from collection-specific names to wrapper objects.
        """
        pass

    @abstractmethod
    def summary(self):
        """Return a formatted text summary for this collection.

        Returns
        -------
        str
            Human-readable summary of the collection content.
        """
        pass


class _Dimensions(_Data):
    def __init__(self, dataset: netCDF4.Dataset | None = None):
        """Expose SOFA dimensions from a netCDF4-backed SOFA storage handle.

        ``_Dimensions`` backs :attr:`~hrtfpykit.sofa.SOFA.Dimensions`. It exposes
        dimension names, sizes, wrapped dimension objects, and a compact text summary.
        Dimension wrappers are used to inspect SOFA axes such as ``M`` for
        measurements, ``R`` for receivers, and ``N`` for samples or frequency bins,
        depending on the active convention.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            Open netCDF4 storage handle for a SOFA file.

        Raises
        ------
        ValueError
            If ``dataset`` is None.
        """
        super().__init__(dataset)

    def get(self, name: str) -> Optional[DimensionsWrap]:
        """Return one wrapped SOFA dimension by name.

        Parameters
        ----------
        name : str
            Dimension name to resolve, such as ``M``, ``R``, ``N``, or ``E``.

        Returns
        -------
        DimensionsWrap
            Wrapped dimension metadata containing the name, size, and
            unlimited flag.

        Raises
        ------
        ValueError
            If ``name`` is not present in the netCDF4 dimensions.

        Examples
        --------
        Read the measurement dimension size from a SOFA file:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.Dimensions.get("M").value
        793
        """
        if name not in self._netCDF4_dataset.dimensions:
            raise ValueError(f"Dimension not found: {name}")
        return DimensionsWrap(name, self._netCDF4_dataset.dimensions)

    def get_names(self) -> list[str]:
        """Return all dimension names in storage order.

        Returns
        -------
        list[str]
            Names from ``dataset.dimensions`` on the netCDF4 storage handle.

        Examples
        --------
        List the dimension keys exposed by the SOFA file:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.Dimensions.get_names()
        ['M', 'R', 'E', 'N', 'C', 'I']
        """
        return list(self._netCDF4_dataset.dimensions.keys())

    def get_values(self) -> list[int]:
        """Return all dimension sizes in storage order.

        Returns
        -------
        list[int]
            Dimension sizes matching the order returned by
            :meth:`~hrtfpykit.sofa.data._Dimensions.get_names`.
            Unlimited dimensions report their current netCDF4 size.

        Examples
        --------
        Read dimension sizes in the same order as
        :meth:`get_names`:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.Dimensions.get_values()
        [793, 2, 3, 256, 3, 1]
        """
        return [dim.size for dim in self._netCDF4_dataset.dimensions.values()]

    def get_all(self) -> Dict[str, DimensionsWrap]:
        """Return all dimensions as wrapped objects.

        Returns
        -------
        dict[str, DimensionsWrap]
            Mapping from dimension names to wrapped dimension metadata.

        Examples
        --------
        Collect dimension wrappers and inspect one dimension:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> dimensions = sofa.Dimensions.get_all()
        >>> dimensions["M"].value
        793
        """
        return {
            k: DimensionsWrap(k, self._netCDF4_dataset.dimensions)
            for k in self._netCDF4_dataset.dimensions.keys()
        }

    def summary(self) -> str:
        """Return a formatted summary of SOFA dimensions.

        Returns
        -------
        str
            Multi-line text where each line has name = size. Dimensions
            are sorted alphabetically to keep the summary stable.

        Examples
        --------
        Print a stable text summary of the SOFA dimensions:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> print(sofa.Dimensions.summary())
        C = 3
        E = 3
        I = 1
        M = 793
        N = 256
        R = 2
        """
        lines = []
        for name in sorted(self._netCDF4_dataset.dimensions.keys()):
            dim = self._netCDF4_dataset.dimensions[name]
            lines.append(f"{name} = {dim.size}")
        return "\n".join(lines)


class _AttributesBase(_Data):
    def __init__(self, dataset: netCDF4.Dataset | None = None, attribute_type: str = "Attribute") -> None:
        """Provide lookup, iteration, and wrapping for SOFA attributes.

        ``_AttributesBase`` implements the shared lookup, iteration, and wrapping
        behavior for global attributes and variable attributes. Concrete subclasses
        provide the collection-specific iteration and lookup logic because global
        attributes live on the SOFA file while variable attributes live on
        each netCDF4 variable.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            Open netCDF4 storage handle for a SOFA file.
        attribute_type : str, default=``Attribute``
            Type label attached to returned attribute wrappers.

        Raises
        ------
        ValueError
            If ``dataset`` is None.
        """
        super().__init__(dataset)
        self._attribute_type = attribute_type

    @abstractmethod
    def _iter_items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over attribute names and raw values.

        Returns
        -------
        Iterator[tuple[str, Any]]
            Iterator yielding (name, value) pairs in the concrete
            collection's key format.
        """
        pass

    @abstractmethod
    def _get_value(self, name: str) -> Optional[Any]:
        """Return a raw attribute value by collection-specific name.

        Parameters
        ----------
        name : str
            Attribute name or key.

        Returns
        -------
        Any | None
            Raw attribute value when present, otherwise None.
        """
        pass

    def _invalid_name_message(self) -> str:
        """Return the generic invalid-name message for this collection.

        Returns
        -------
        str
            Human-readable message used by subclasses or diagnostics.
        """
        return "Please insert a valid attribute name"

    def get(self, name: str) -> Optional[AttributesWrap]:
        """Return one wrapped attribute by name.

        Parameters
        ----------
        name : str
            Attribute key in the format expected by the concrete collection.
            Global attributes use the plain attribute name. Variable attributes
            use ``Variable:Attribute``.

        Returns
        -------
        AttributesWrap
            Wrapped attribute metadata with the concrete collection's
            ``attribute_type`` label.

        Raises
        ------
        ValueError
            If ``name`` does not resolve to an available attribute in the
            wrapped collection.

        Examples
        --------
        Read one global attribute and one variable attribute by key:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.GlobalAttributes.get("SOFAConventions").value
        'SimpleFreeFieldHRIR'
        >>> sofa.VariableAttributes.get("SourcePosition:Type").value
        'spherical'
        """
        value = self._get_value(name)
        if value is None:
            label = self._attribute_type
            if label.endswith("Attribute"):
                label = f"{label[:-9]} attribute"
            else:
                label = f"{label} attribute"
            raise ValueError(f"{label} not found: {name}")
        return AttributesWrap(name, value, self._attribute_type)

    def get_names(self) -> list[str]:
        """Return all attribute names in collection order.

        Returns
        -------
        list[str]
            Attribute keys produced by
            :meth:`~hrtfpykit.sofa.data._AttributesBase._iter_items`.

        Examples
        --------
        Check whether expected global and variable attribute keys are present:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> "SOFAConventions" in sofa.GlobalAttributes.get_names()
        True
        >>> "SourcePosition:Type" in sofa.VariableAttributes.get_names()
        True
        """
        return [name for name, _ in self._iter_items()]

    def get_values(self) -> list[Any]:
        """Return all raw attribute values in collection order.

        Returns
        -------
        list[Any]
            Attribute values produced by
            :meth:`~hrtfpykit.sofa.data._AttributesBase._iter_items`.

        Examples
        --------
        Inspect raw attribute values without wrapping them first:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> "SimpleFreeFieldHRIR" in sofa.GlobalAttributes.get_values()
        True
        """
        return [value for _, value in self._iter_items()]

    def get_all(self) -> Dict[str, AttributesWrap]:
        """Return all attributes as wrapped objects.

        Returns
        -------
        dict[str, AttributesWrap]
            Mapping from attribute keys to wrapped attribute metadata.

        Examples
        --------
        Collect wrapped attributes and inspect metadata from each collection:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> global_attributes = sofa.GlobalAttributes.get_all()
        >>> global_attributes["SOFAConventions"].value
        'SimpleFreeFieldHRIR'
        >>> variable_attributes = sofa.VariableAttributes.get_all()
        >>> variable_attributes["SourcePosition:Type"].value
        'spherical'
        """
        return {
            name: AttributesWrap(name, value, self._attribute_type)
            for name, value in self._iter_items()
        }

    def summary(self) -> str:
        """Return the base attribute summary placeholder.

        Concrete subclasses override this method with formatted summaries for
        their specific attribute collection. The base implementation is kept to
        satisfy the shared ``_Data`` protocol.

        Returns
        -------
        str
            Empty string.

        Examples
        --------
        Users normally call ``summary()`` on ``sofa.GlobalAttributes`` or
        ``sofa.VariableAttributes``, which use the concrete summary
        implementations.
        """
        return ""


class _GlobalAttributes(_AttributesBase):
    def __init__(self, dataset: netCDF4.Dataset | None = None) -> None:
        """Expose file-level SOFA metadata from a netCDF4 storage handle.

        ``_GlobalAttributes`` backs
        :attr:`~hrtfpykit.sofa.SOFA.GlobalAttributes`. It exposes file-level
        metadata such as ``SOFAConventions``, ``SOFAConventionsVersion``,
        ``DataType``, application metadata, and date fields through the shared
        attribute-wrapper API.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            Open netCDF4 storage handle for a SOFA file.

        Raises
        ------
        ValueError
            If ``dataset`` is None.
        """
        super().__init__(dataset, attribute_type="GlobalAttribute")

    def _iter_items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over global attribute names and values.

        Returns
        -------
        Iterator[tuple[str, Any]]
            Iterator yielding (attribute_name, value) pairs from
            ``dataset.ncattrs()``.
        """
        for name in self._netCDF4_dataset.ncattrs():
            yield name, getattr(self._netCDF4_dataset, name)

    def _get_value(self, name: str) -> Optional[Any]:
        """Return a global attribute value by name.

        Parameters
        ----------
        name : str
            Global attribute name.

        Returns
        -------
        Any | None
            Attribute value when present, otherwise None.
        """
        if name not in self._netCDF4_dataset.ncattrs():
            return None
        return getattr(self._netCDF4_dataset, name)

    def _invalid_name_message(self) -> str:
        """Return the invalid-name message for global attributes.

        Returns
        -------
        str
            Message describing the expected global-attribute key.
        """
        return "Please insert a valid global attribute name"

    def summary(self) -> str:
        """Return a formatted summary of global SOFA attributes.

        Returns
        -------
        str
            Multi-line summary with a header followed by ``GLOBAL:name`` =
            value lines. Returns an empty string when the SOFA file has no
            global attributes.

        Examples
        --------
        Check the formatted global-attribute summary text:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> "SOFAConventions" in sofa.GlobalAttributes.summary()
        True
        """
        items = list(self._iter_items())
        if not items:
            return ""
        lines = [
            "****************************",
            "   GLOBAL ATTRIBUTES",
            "****************************",
        ]
        lines.extend(f"GLOBAL:{name} = {value}" for name, value in items)
        return "\n".join(lines)


class _VariableAttributes(_AttributesBase):
    def __init__(self, dataset: netCDF4.Dataset | None = None) -> None:
        """Expose attributes attached to individual SOFA variables.

        ``_VariableAttributes`` backs
        :attr:`~hrtfpykit.sofa.SOFA.VariableAttributes`. It exposes attributes
        attached to individual variables using the hrtfpykit key format
        ``Variable:Attribute``. These attributes describe units, coordinate
        systems, and semantic labels required by SOFA-based HRTF workflows.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            Open netCDF4 storage handle for a SOFA file.

        Raises
        ------
        ValueError
            If ``dataset`` is None.
        """
        super().__init__(dataset, attribute_type="VariableAttribute")

    def _iter_items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over variable attribute keys and values.

        Returns
        -------
        Iterator[tuple[str, Any]]
            Iterator yielding (``Variable:Attribute``, value) pairs for
            every attribute on every variable.
        """
        for var_name, var in self._netCDF4_dataset.variables.items():
            for attr_name in var.ncattrs():
                yield f"{var_name}:{attr_name}", getattr(var, attr_name)

    def _get_value(self, name: str) -> Optional[Any]:
        """Return a variable attribute value by fully qualified key.

        Parameters
        ----------
        name : str
            Attribute key in the form ``Variable:Attribute``.

        Returns
        -------
        Any | None
            Attribute value when the variable and attribute exist, otherwise
            None.
        """
        if ":" not in name:
            return None
        var_name, attr_name = name.split(":", 1)
        if var_name not in self._netCDF4_dataset.variables:
            return None
        var = self._netCDF4_dataset.variables[var_name]
        if attr_name not in var.ncattrs():
            return None
        return getattr(var, attr_name)

    def _invalid_name_message(self) -> str:
        """Return the invalid-name message for variable attributes.

        Returns
        -------
        str
            Message describing the expected variable-attribute key.
        """
        return "Please insert a valid variable attribute name"

    def summary(self) -> str:
        """Return a formatted summary of SOFA variable attributes.

        Returns
        -------
        str
            Multi-line summary with a header followed by
            ``Variable:Attribute`` = value lines. Returns an empty string when
            no variable attributes are present.

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

        Check the formatted variable-attribute summary text:

        >>> "SourcePosition:Type" in sofa.VariableAttributes.summary()
        True
        """
        items = list(self._iter_items())
        if not items:
            return ""
        lines = [
            "****************************",
            "   VARIABLE ATTRIBUTES",
            "****************************",
        ]
        lines.extend(f"{name} = {value}" for name, value in items)
        return "\n".join(lines)


class _Variables(_Data):
    def __init__(self, dataset: netCDF4.Dataset | None = None):
        """Expose SOFA variable data and metadata from a netCDF4 storage handle.

        ``_Variables`` backs :attr:`~hrtfpykit.sofa.SOFA.Variables`. It exposes
        variable names, NumPy values, wrapped variable objects, and a summary of
        variable dimensions and attributes. This collection is the primary read path
        for SOFA arrays such as ``Data.IR``, ``Data.Real``, ``Data.Imag``,
        ``SourcePosition``, ``Data.SamplingRate``, and ``N``.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            Open netCDF4 storage handle for a SOFA file.

        Raises
        ------
        ValueError
            If ``dataset`` is None.
        """
        super().__init__(dataset)

    def get(self, name: str) -> Optional[VariablesWrap]:
        """Return one wrapped SOFA variable by name.

        Parameters
        ----------
        name : str
            Variable name to resolve, such as ``Data.IR``, ``SourcePosition``,
            or ``Data.SamplingRate``.

        Returns
        -------
        VariablesWrap
            Wrapped variable exposing NumPy data through
            :attr:`~hrtfpykit.sofa.wraps.VariablesWrap.value` and variable
            attributes through
            :attr:`~hrtfpykit.sofa.wraps.VariablesWrap.attributes`.

        Raises
        ------
        ValueError
            If ``name`` is not present in the netCDF4 variables.

        Examples
        --------
        Read the HRIR data variable and inspect its array shape:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> sofa.Variables.get("Data.IR").value.shape
        (793, 2, 256)
        """
        if name not in self._netCDF4_dataset.variables:
            raise ValueError(f"Variable not found: {name}")
        return VariablesWrap(name, self._netCDF4_dataset.variables[name])

    def get_names(self) -> list[str]:
        """Return all variable names in storage order.

        Returns
        -------
        list[str]
            Names from ``dataset.variables`` on the netCDF4 storage handle.

        Examples
        --------
        Check whether a standard SOFA data variable is present:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> "Data.IR" in sofa.Variables.get_names()
        True
        """
        return list(self._netCDF4_dataset.variables.keys())

    def get_values(self) -> list[np.ndarray]:
        """Return all variable values as NumPy arrays.

        Returns
        -------
        list[np.ndarray]
            Complete variable data arrays in storage order. Each variable is
            read with full slicing and converted with :func:`numpy.array`.

        Raises
        ------
        Exception
            Propagates errors raised by netCDF4 when a variable cannot be
            read.

        Examples
        --------
        Load every SOFA variable value and compare the count with the names:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> values = sofa.Variables.get_values()
        >>> len(values) == len(sofa.Variables.get_names())
        True
        """
        return [np.array(v[:]) for v in self._netCDF4_dataset.variables.values()]

    def get_all(self) -> Dict[str, VariablesWrap]:
        """Return all variables as wrapped objects.

        Returns
        -------
        dict[str, VariablesWrap]
            Mapping from variable names to wrapped variable objects.

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
        >>> editable.Variables.get_all()["ExampleVector"].value
        array([1., 2., 3.])
        >>> editable.VariableAttributes.get("ExampleVector:Units").value
        '1'
        """
        return {
            k: VariablesWrap(k, v) for k, v in self._netCDF4_dataset.variables.items()
        }

    def summary(self) -> str:
        """Return a formatted summary of SOFA variables.

        Returns
        -------
        str
            Multi-line summary listing each variable, its dimension names and
            current dimension sizes, followed by any variable attributes.
            Missing dimension references are displayed with ?.

        Examples
        --------
        Check the formatted variable summary text:

        >>> from hrtfpykit.sofa import load_sofa
        >>> sofa = load_sofa("P0001_FreeFieldComp_44kHz.sofa")
        >>> "Data.IR" in sofa.Variables.summary()
        True
        """
        lines = []
        for name, var in self._netCDF4_dataset.variables.items():
            dims = []
            for dim_name in var.dimensions:
                dim_size: int | str
                if dim_name in self._netCDF4_dataset.dimensions:
                    dim_size = self._netCDF4_dataset.dimensions[dim_name].size
                else:
                    dim_size = "?"
                dims.append(f"{dim_name}={dim_size}")
            dims_str = ", ".join(dims)
            lines.append(f"{name} : dimensions= ({dims_str})")
            attrs = list(var.ncattrs())
            if attrs:
                lines.append("      attributes:")
                for attr_name in attrs:
                    value = getattr(var, attr_name)
                    lines.append(f"      {name}:{attr_name}= {value}")
        return "\n".join(lines)

