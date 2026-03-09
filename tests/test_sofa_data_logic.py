from pathlib import Path

import netCDF4 as ncdf
import numpy as np

from hrtfpykit.sofa.core import SOFA


def _create_sofa_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.sofa"
    ds = ncdf.Dataset(str(path), mode="w")
    try:
        ds.SOFAConventions = "SimpleFreeFieldHRIR"
        ds.SOFAConventionsVersion = "1.0"
        ds.DataType = "FIR"
        ds.DatabaseName = "TestDB"

        ds.createDimension("m", 2)
        ds.createDimension("R", 2)
        ds.createDimension("n", 4)
        ds.createDimension("I", 1)

        ir = ds.createVariable("Data.IR", "f8", ("m", "R", "n"))
        ir[:] = np.ones((2, 2, 4))
        ir.Units = "pascal"

        sr = ds.createVariable("Data.SamplingRate", "f8", ("I",))
        sr[:] = np.array([44100.0])
        sr.Units = "hertz"
    finally:
        ds.close()
    return path


def test_sofa_properties(tmp_path: Path) -> None:
    path = _create_sofa_file(tmp_path)
    sofa = SOFA.load(path, check_sofa=True)
    try:
        assert sofa.Dimensions is not None
        assert sofa.Attributes is not None
        assert sofa.GlobalAttributes is not None
        assert sofa.VariableAttributes is not None
        assert sofa.Variables is not None
    finally:
        sofa.netCDF4_dataset.close()


def test_dimensions_logic(tmp_path: Path) -> None:
    path = _create_sofa_file(tmp_path)
    sofa = SOFA.load(path, check_sofa=True)
    try:
        dims = sofa.Dimensions
        assert dims is not None
        assert dims.get("m") is not None
        assert dims.get("m").value == 2
        assert "m" in dims.get_names()
        assert len(dims) == 4
        assert "m = 2" in dims.summary()
    finally:
        sofa.netCDF4_dataset.close()


def test_global_and_variable_attributes(tmp_path: Path) -> None:
    path = _create_sofa_file(tmp_path)
    sofa = SOFA.load(path, check_sofa=True)
    try:
        global_attrs = sofa.GlobalAttributes
        variable_attrs = sofa.VariableAttributes
        assert global_attrs is not None
        assert variable_attrs is not None

        global_wrap = global_attrs.get("DatabaseName")
        assert global_wrap is not None
        assert global_wrap.value == "TestDB"
        assert "DatabaseName" in global_attrs.get_names()

        var_wrap = variable_attrs.get("Data.IR:Units")
        assert var_wrap is not None
        assert var_wrap.value == "pascal"
        assert "Data.IR:Units" in variable_attrs.get_names()
    finally:
        sofa.netCDF4_dataset.close()


def test_attributes_facade(tmp_path: Path) -> None:
    path = _create_sofa_file(tmp_path)
    sofa = SOFA.load(path, check_sofa=True)
    try:
        attrs = sofa.Attributes
        assert attrs is not None

        global_wrap = attrs.get("DatabaseName")
        assert global_wrap is not None
        assert global_wrap.value == "TestDB"

        var_wrap = attrs.get("Data.IR:Units")
        assert var_wrap is not None
        assert var_wrap.value == "pascal"

        names = attrs.get_names()
        assert "DatabaseName" in names
        assert "Data.IR:Units" in names
    finally:
        sofa.netCDF4_dataset.close()


def test_variables_logic(tmp_path: Path) -> None:
    path = _create_sofa_file(tmp_path)
    sofa = SOFA.load(path, check_sofa=True)
    try:
        variables = sofa.Variables
        assert variables is not None

        ir = variables.get("Data.IR")
        assert ir is not None
        assert isinstance(ir.value, np.ndarray)

        sr = variables.get("Data.SamplingRate")
        assert sr is not None
        assert sr.value == 44100
        assert isinstance(sr.value, int)

        summary = variables.summary()
        assert "Data.IR : dimensions = (m=2, R=2, n=4)" in summary
    finally:
        sofa.netCDF4_dataset.close()
