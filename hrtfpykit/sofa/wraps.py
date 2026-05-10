from typing import Any, Dict
import numpy as np


class DimensionsWrap:
    def __init__(self, name: str, _netCDF4_dataset_dimensions: Any) -> None:
        """Represent one SOFA dimension for inspection workflows.

        :class:`~hrtfpykit.sofa.wraps.DimensionsWrap` is the lightweight
        object returned by :meth:`~hrtfpykit.sofa.data._Dimensions.get` and
        related dimension collection methods. It exposes the dimension name,
        size, and unlimited flag in the same wrapper-based style used for SOFA
        variables and attributes. This keeps dimension inspection independent
        from direct netCDF4 access while still preserving the original
        dimension metadata needed by HRTF and SOFA validation workflows.

        Notes
        -----
        This wrapper is intended for inspection. It does not mutate the underlying
        SOFA file; dimension edits are handled by methods on
        :class:`~hrtfpykit.sofa.sofa.SOFA`.

        Parameters
        ----------
        name : str
            Name of the dimension to wrap, such as "M", "R", "N", or
            "E" in common SOFA files.
        _netCDF4_dataset_dimensions : Any
            Mapping-like container of netCDF4 dimensions, typically
            netCDF4.Dataset.dimensions. The mapping must contain "name" and
            each entry must provide a size and an "isunlimited()" method.

        Raises
        ------
        KeyError
            If "name" is not present in "_netCDF4_dataset_dimensions".

        Attributes
        ----------
        value : int
            Dimension size captured when the wrapper is created.
        is_unlimited : bool
            Whether the wrapped netCDF4 dimension is unlimited.
        """
        self.name = name
        self.value = _netCDF4_dataset_dimensions[name].size
        self.is_unlimited = bool(_netCDF4_dataset_dimensions[name].isunlimited())
        self._netCDF4_dataset_dimensions = _netCDF4_dataset_dimensions

    def __repr__(self) -> str:
        """Return a developer-facing representation of the wrapped dimension.

        Returns
        -------
        str
            Representation containing the dimension name, captured size, and
            unlimited flag.
        """
        return f"DimensionsWrap(name = {self.name!r}, value = {self.value}, unlimited = {self.is_unlimited})"



class AttributesWrap:
    def __init__(self, name: str, value: Any, type: str) -> None:
        """Represent one global or variable SOFA attribute.

        :class:`~hrtfpykit.sofa.wraps.AttributesWrap` is returned by
        global-attribute and variable-attribute accessors. It stores the
        attribute name, the raw attribute value read from netCDF4, and a small
        type label that tells callers whether the attribute came from the SOFA
        file itself or from a specific variable. Variable attributes use the
        canonical hrtfpykit key form "Variable:Attribute".

        Notes
        -----
        The wrapper is a small read object. To create, modify, or delete SOFA
        attributes, use the corresponding methods on
        :class:`~hrtfpykit.sofa.sofa.SOFA`.

        Parameters
        ----------
        name : str
            Attribute name. Global attributes use their netCDF4 attribute name
            directly. Variable attributes use "Variable:Attribute".
        value : Any
            Attribute value as returned by netCDF4.
        type : str
            Attribute type label used by the SOFA API, typically
            "GlobalAttribute" or "VariableAttribute".

        """
        self.name = name
        self.value = value
        self.type = type

    def __repr__(self) -> str:
        """Return a developer-facing representation of the wrapped attribute.

        Returns
        -------
        str
            Representation containing the attribute name, value, and type
            label.
        """
        return f"AttributesWrap(name={self.name!r}, value={self.value!r}, type={self.type!r})"



class VariablesWrap:
    def __init__(self, name: str, var: Any) -> None:
        """Represent one SOFA variable and expose array/metadata views.

        :class:`~hrtfpykit.sofa.wraps.VariablesWrap` is returned by
        :meth:`~hrtfpykit.sofa.data._Variables.get` and related variable
        collection methods. It keeps a reference to the underlying netCDF4
        variable and exposes two convenience views used throughout hrtfpykit:
        :attr:`~hrtfpykit.sofa.wraps.VariablesWrap.value` for NumPy data
        access and :attr:`~hrtfpykit.sofa.wraps.VariablesWrap.attributes` for
        the variable's SOFA metadata. This is the wrapper used for core
        HRTF/HRIR arrays such as
        "Data.IR", "Data.Real", "Data.Imag", "SourcePosition", and
        "Data.SamplingRate".

        Notes
        -----
        :class:`~hrtfpykit.sofa.wraps.VariablesWrap` does not cache variable
        data. Each access to :attr:`~hrtfpykit.sofa.wraps.VariablesWrap.value`
        reads from the current netCDF4 variable and returns a NumPy array copy.
        This keeps inspection accurate after in-memory SOFA edits while
        avoiding direct mutation of the underlying netCDF4 variable through
        the wrapper.

        Parameters
        ----------
        name : str
            SOFA variable name.
        var : Any
            netCDF4 variable instance. The object must support full slicing and
            must provide "shape" and "ncattrs()".

        """
        self.name = name
        self._var = var

    @property
    def value(self) -> np.ndarray:
        """Return the variable data as a NumPy array copy.

        The complete netCDF4 variable is read with full slicing and converted
        with :func:`numpy.array`. This gives callers a normal NumPy value for
        HRTF, HRIR, position, and metadata processing without exposing the
        mutable netCDF4 variable object directly.

        Returns
        -------
        numpy.ndarray
            Variable data copied from the underlying netCDF4 variable. The
            returned shape follows the SOFA variable shape, for example
            ("M", "R", "N") for "Data.IR" in a SimpleFreeFieldHRIR file.

        Raises
        ------
        Exception
            Propagates errors raised by the underlying netCDF4 variable when
            it cannot be read.
        """
        data = np.array(self._var[:])
        return data

    @property
    def attributes(self) -> Dict[str, AttributesWrap]:
        """Return variable attributes as wrapped SOFA metadata.

        Variable attributes are exposed with fully qualified keys of the form
        "Variable:Attribute". For example, the "Type" attribute on
        "SourcePosition" is returned under "SourcePosition:Type". This
        matches the lookup convention used by
        :attr:`~hrtfpykit.sofa.sofa.SOFA.VariableAttributes` and by HRTF
        source-coordinate code.

        Returns
        -------
        dict[str, AttributesWrap]
            Attribute wrappers keyed by fully qualified
            "Variable:Attribute" names. Each wrapper uses the
            "VariableAttribute" type label.

        Raises
        ------
        Exception
            Propagates errors raised by the underlying netCDF4 variable when
            its attributes cannot be inspected.
        """
        attributes: Dict[str, AttributesWrap] = {}
        for attr_name in self._var.ncattrs():
            full_name = f"{self.name}:{attr_name}"
            attributes[full_name] = AttributesWrap(
                full_name,
                getattr(self._var, attr_name),
                "VariableAttribute",
            )
        return attributes

    def __repr__(self) -> str:
        """Return a developer-facing representation of the wrapped variable.

        Returns
        -------
        str
            Representation containing the variable name, netCDF4 shape, NumPy
            value type, copied value, and wrapped variable attributes.

        Notes
        -----
        Building the representation reads
        :attr:`~hrtfpykit.sofa.wraps.VariablesWrap.value` and
        :attr:`~hrtfpykit.sofa.wraps.VariablesWrap.attributes`.
        For large SOFA variables this can be expensive, so prefer direct
        inspection of the raw variable name, shape, or attributes when a concise
        diagnostic is enough.
        """
        value = self.value
        return (
            f"VariablesWrap(name={self.name!r}, dimension={self._var.shape}, "
            f"dtype={type(value)}, value={value!r}, attributes={self.attributes!r}"
        )
