import numpy as np
import pytest

from hrtfpykit.sofa.core import SOFA


def _make_dummy_sofa() -> SOFA:
    pytest.importorskip("netCDF4")
    return SOFA.create_dummy("SimpleFreeFieldHRIR", version="1.2")


def test_dimension_crud() -> None:
    sofa = _make_dummy_sofa()
    try:
        dataset = sofa.netCDF4_dataset
        assert dataset is not None

        sofa.create_dimension("X", 3)
        assert "X" in dataset.dimensions
        assert dataset.dimensions["X"].size == 3

        sofa.rename_dimension("X", "Y")
        assert "Y" in dataset.dimensions
        assert "X" not in dataset.dimensions
    finally:
        sofa.netCDF4_dataset.close()


def test_global_attribute_crud() -> None:
    sofa = _make_dummy_sofa()
    try:
        dataset = sofa.netCDF4_dataset
        assert dataset is not None

        sofa.create_global_attribute("TestGlobal", "alpha")
        assert "TestGlobal" in dataset.ncattrs()
        assert getattr(dataset, "TestGlobal") == "alpha"

        sofa.modify_global_attribute("TestGlobal", "beta")
        assert getattr(dataset, "TestGlobal") == "beta"

        sofa.delete_global_attribute("TestGlobal")
        assert "TestGlobal" not in dataset.ncattrs()
    finally:
        sofa.netCDF4_dataset.close()


def test_variable_attribute_crud() -> None:
    sofa = _make_dummy_sofa()
    try:
        dataset = sofa.netCDF4_dataset
        assert dataset is not None

        var = dataset.variables["Data.IR"]
        sofa.create_variable_attribute("Data.IR:Units", "pascal")
        assert "Units" in var.ncattrs()
        assert getattr(var, "Units") == "pascal"

        sofa.modify_variable_attribute("Data.IR:Units", "Pa")
        assert getattr(var, "Units") == "Pa"

        sofa.delete_variable_attribute("Data.IR:Units")
        assert "Units" not in var.ncattrs()
    finally:
        sofa.netCDF4_dataset.close()


def test_variable_crud() -> None:
    sofa = _make_dummy_sofa()
    try:
        dataset = sofa.netCDF4_dataset
        assert dataset is not None

        m_size = dataset.dimensions["M"].size
        data = np.ones((m_size,), dtype=np.float64)

        sofa.create_variable(
            name="TestVar",
            data=data,
            dimensions=("M",),
            attributes={"Units": "unitless", "Type": ""},
        )
        assert "TestVar" in dataset.variables
        assert dataset.variables["TestVar"][:].shape == (m_size,)
        assert dataset.variables["TestVar"].Units == "unitless"

        new_data = np.full((m_size,), 2.0)
        sofa.modify_variable("TestVar", new_data)
        assert np.allclose(dataset.variables["TestVar"][:], new_data)

        sofa.delete_variable("TestVar")
        assert "TestVar" not in dataset.variables
    finally:
        sofa.netCDF4_dataset.close()
