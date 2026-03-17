from pathlib import Path

import pytest

from hrtfpykit.sofa.conventions_manager import ConventionsManager


def test_list_conventions_specifications_returns_mapping() -> None:
    listing = ConventionsManager.list_conventions_specifications()
    assert isinstance(listing, dict)
    assert "SimpleFreeFieldHRIR" in listing


def test_inspect_and_add_delete_convention_specification() -> None:
    name = "TempConvention"
    version = "0.1"
    spec = {
        "GLOBAL:Conventions": {
            "default": "SOFA",
            "flags": "rm",
            "dimensions": None,
            "type": "attribute",
            "comment": "",
        }
    }

    ConventionsManager.add_convention_specification(name, version, spec, overwrite=True)
    fetched = ConventionsManager.inspect_sofa_specification(name, version)
    assert fetched["GLOBAL:Conventions"]["default"] == "SOFA"

    ConventionsManager.delete_convention_specification_version(name, version)
    with pytest.raises(KeyError):
        ConventionsManager.inspect_sofa_specification(name, version)


def test_export_convention_specification_json(tmp_path: Path) -> None:
    name = "SimpleFreeFieldHRIR"
    version = "1.2"
    out_path = tmp_path / "spec.json"
    ConventionsManager.export_convention_specification_json(name, version, out_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "\"convention\"" in content
