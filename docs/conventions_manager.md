# Conventions Manager

The conventions manager provides a small registry API for adding, removing,
inspecting, and serializing SOFA convention specifications without editing
`conventions.py` directly.

## Usage

```python
from hrtfpykit.sofa.conventions_manager import ConventionsManager

manager = ConventionsManager()
spec = manager.inspect_sofa_specification("SimpleFreeFieldHRIR", "1.2")

manager.export_convention_specification_json("SimpleFreeFieldHRIR", "1.2", "spec.json")
manager.add_convention_specification_from_json("spec.json")
```

## JSON format

Single convention:

```json
{
  "convention": "SimpleFreeFieldHRIR",
  "version": "1.2",
  "spec": {
    "GLOBAL:Conventions": {
      "default": "SOFA",
      "flags": "rm",
      "dimensions": null,
      "type": "attribute",
      "comment": ""
    }
  }
}
```

Registry payload:

```json
{
  "registry": {
    "SimpleFreeFieldHRIR": {
      "1.2": { "...": "..." }
    }
  }
}
```
