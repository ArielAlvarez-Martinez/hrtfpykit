# SOFA API

This document describes the stable SOFA API in `hrtfpykit.sofa`. It focuses on
loading, inspecting, validating, and editing SOFA files with explicit I/O and
reproducible behavior.

---

## Contents

- [Overview](#overview)
- [Design principles](#design-principles)
- [Quick start](#quick-start)
- [Recommended edit workflow](#recommended-edit-workflow)
- [Public surface](#public-surface)
- [Wrappers](#wrappers)
- [SOFA Main Methods](#sofa-main-methods)
- [SOFA CRUD Methods](#sofa-crud-methods)
- [Validation and security](#validation-and-security)
- [ConventionsManager](#conventionsmanager)
- [SOFA conventions registry](#sofa-conventions-registry-conventions)
- [Common pitfalls](#common-pitfalls)
- [Use cases](#use-cases)

---

## Overview

The SOFA API provides high-level access to SOFA.
It wraps common operations for dimensions, variables, and attributes, while
preserving the underlying SOFA conventions and constraints.

You can:
- Load, handle and save SOFA files.
- Inspect or validate metadata against the conventions registry.
- Create in-memory SOFA objects for tests.
- Safely modify files via an explicit copy → save workflow.

---

## Design principles

- No hidden I/O: you must call `load()` or `save()` explicitly.
- Deterministic behavior: no implicit shuffling or randomization.
- Explicit validation: use `check_sofa_against_conventions` and
  `check_sofa_security` when needed.
- Safe editing: prefer `copy()` then `save()` to a new path.
- NetCDF4-backed: the underlying SOFA object is a `netCDF4.Dataset` and can be
  accessed via `sofa.netCDF4_dataset` for advanced use.

---

## Quick start

```python
from hrtfpykit.sofa import SOFA, check_sofa_against_conventions

# Load
sofa = SOFA.load("my.sofa")

# Inspect
print(sofa.summary())

# Access variables
ir = sofa.Variables["Data.IR"].value
sr = sofa.Variables["Data.SamplingRate"].value

# Validate
check_sofa_against_conventions(sofa)
```

---

## Recommended edit workflow

```python
# Read-only open
sofa = SOFA.load("my.sofa")

# Make changes on a copy
sofa_copy = sofa.copy()
sofa_copy.modify_global_attribute("Title", "Updated title")

# Save to a new path
sofa_copy.save("my_updated.sofa")

# Close the original object if no longer needed
sofa.netCDF4_dataset.close()
```

---

## Public surface

### Classes
- `SOFA`
- `ConventionsManager`

### Functions
- `check_sofa_against_conventions`
- `check_sofa_security`

---

## Wrappers

The SOFA object exposes wrappers for dimensions, attributes, and variables.
If a SOFA object is not loaded, these properties return `None`.

### `SOFA.Dimensions`

**Purpose**
Access dimensions via a wrapper interface.

**Returns**
- `_Dimensions` wrapper

**Example**
```python
dims = sofa.Dimensions
print(dims.get_names())
```

**Wrap fields**
- `name`
- `value` (dimension size)
- `is_unlimited`

---

### `SOFA.GlobalAttributes`

**Purpose**
Access global attributes via a wrapper interface.

**Returns**
- `_GlobalAttributes` wrapper

**Example**
```python
ga = sofa.GlobalAttributes
print(ga.get("Title").value)
```

**Wrap fields**
- `name`
- `value`
- `type` (always `"GlobalAttribute"` here)

---

### `SOFA.VariableAttributes`

**Purpose**
Access variable attributes via a wrapper interface.

**Returns**
- `_VariableAttributes` wrapper

**Example**
```python
va = sofa.VariableAttributes
print(va.get("Data.IR:Units").value)
```

**Wrap fields**
- `name` (format `Variable:Attribute`)
- `value`
- `type` (always `"VariableAttribute"` here)

---

### `SOFA.Variables`

**Purpose**
Access variables via a wrapper interface.

**Returns**
- `_Variables` wrapper

**Example**
```python
vars_wrap = sofa.Variables
print(vars_wrap.get("Data.IR").value.shape)
```

**Wrap fields**
- `VariablesWrap.value` (numpy array copy)
- `VariablesWrap.attributes` (mapping of variable attributes)

---

### Wrapper methods (shared)

All wrappers implement the same access pattern. The methods below apply to
`Dimensions`, `GlobalAttributes`, `VariableAttributes`, and `Variables`.

#### `get(name)`

**Purpose**
Return a wrap object by name.

**Parameters**
- `name : str`
  Item name.

**Returns**
- Wrap object (`DimensionsWrap`, `AttributesWrap`, or `VariablesWrap`).

**Raises**
- `ValueError` if the item is not found.

**Example**
```python
ir = sofa.Variables.get("Data.IR")
```

---

#### `get_all()`

**Purpose**
Return all items as a name → wrap mapping.

**Returns**
- `dict[str, Wrap]`

**Example**
```python
vars_map = sofa.Variables.get_all()
```

---

#### `get_names()`

**Purpose**
Return a list of item names.

**Returns**
- `list[str]`

**Example**
```python
names = sofa.Variables.get_names()
```

---

#### `get_values()`

**Purpose**
Return raw values for all items.

**Returns**
- `list[Any]` (dimension sizes, attribute values, or numpy arrays)

**Warnings**
- For `Variables`, this returns copies and can be memory heavy.

**Example**
```python
values = sofa.Variables.get_values()
```

---

#### `summary()`

**Purpose**
Return a formatted summary string.

**Returns**
- `str`

**Example**
```python
print(sofa.Variables.summary())
```

---

#### `__getitem__(name)`

**Purpose**
Alias for `get(name)`.

**Parameters**
- `name : str`

**Returns**
- Wrap object

**Raises**
- `ValueError` if the item is not found.

**Example**
```python
ir = sofa.Variables["Data.IR"]
```

---

#### `__iter__()`

**Purpose**
Iterate over wrap objects.

**Returns**
- Iterator of wrap objects

**Example**
```python
for dim in sofa.Dimensions:
    print(dim.name, dim.value)
```

---

#### `__len__()`

**Purpose**
Return the number of items.

**Returns**
- `int`

**Example**
```python
print(len(sofa.Variables))
```

---

## SOFA Main Methods

### `SOFA.load(path, mode="r", parallel=False, check_sofa_against_conventions=True)`

**Purpose**
Load a SOFA file and return a `SOFA` instance.

**Parameters**
- `path : str | pathlib.Path`
  SOFA file path.
- `mode : str`
  netCDF4 open mode (`"r"`, `"r+"`).
- `parallel : bool`
  Open in parallel mode if supported by the netCDF build.
- `check_sofa_against_conventions : bool`
  If True, validates the file at load.

**Returns**
- `SOFA`

**Raises**
- `FileNotFoundError` if the file does not exist.
- `ValueError` if the file extension is not `.sofa`.

**Example**
```python
sofa = SOFA.load("example.sofa")
```

**Notes**
- Call `sofa.netCDF4_dataset.close()` when you are done with the SOFA object.

---

### `SOFA.create_dummy(sofa_conventions, version=None, dim_sizes=None, custom_global_attributes=None, override_default_global_attributes=False)`

**Purpose**
Create an in-memory SOFA object that follows a convention spec.

**Parameters**
- `sofa_conventions : str`
  Convention name (e.g., `"SimpleFreeFieldHRIR"`).
- `version : str | None`
  Convention version. If `None`, the latest version is used.
- `dim_sizes : dict[str, int] | None`
  Optional dimension overrides.
- `custom_global_attributes : dict[str, str] | None`
  Custom global attributes to set.
- `override_default_global_attributes : bool`
  If True, custom attributes override defaults.

**Returns**
- `SOFA` (in-memory object)

**Raises**
- `ValueError` if the convention or version is unsupported.
- `ValueError` if `dim_sizes` includes `"S"` (reserved unlimited dim).

**Example**
```python
sofa = SOFA.create_dummy("SimpleFreeFieldHRIR", version="1.2")
```

---

### `SOFA.save(path=None, overwrite=False)`

**Purpose**
Save the SOFA file to disk.

**Parameters**
- `path : str | pathlib.Path | None`
  Target path. If `None`, saves to the original path.
- `overwrite : bool`
  If True, allows overwriting an existing file.

**Returns**
- `pathlib.Path`

**Raises**
- `ValueError` if no SOFA object is loaded or no path is available.
- `FileExistsError` if the target path exists and overwrite is False.

**Example**
```python
sofa_copy = sofa.copy()
sofa_copy.save("new.sofa")
```

**Notes**
- If you overwrite the same file, ensure the original SOFA object is closed first.

---

### `SOFA.copy()`

**Purpose**
Create an in-memory, writable copy of the SOFA object.

**Returns**
- `SOFA`

**Raises**
- `ValueError` if no SOFA object is loaded.

**Example**
```python
sofa_copy = sofa.copy()
```

---

### `SOFA.summary()`

**Purpose**
Return a formatted summary of global attributes, variables, and variable attributes.

**Returns**
- `str`

**Raises**
- `ValueError` if no SOFA object is loaded.

**Example**
```python
print(sofa.summary())
```

---

## SOFA CRUD Methods

### Dimensions

#### `SOFA.create_dimension(name, value)`

**Purpose**
Create a new dimension.

**Parameters**
- `name : str`
  Dimension name.
- `value : int`
  Dimension size.

**Returns**
- `None`

**Raises**
- `ValueError` if the dimension already exists.

**Example**
```python
sofa_copy.create_dimension("X", 3)
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

#### `SOFA.rename_dimension(old_name, new_name)`

**Purpose**
Rename an existing dimension.

**Parameters**
- `old_name : str`
  Existing dimension name.
- `new_name : str`
  New dimension name.

**Returns**
- `None`

**Raises**
- `ValueError` if the old dimension does not exist.

**Example**
```python
sofa_copy.rename_dimension("M", "Measurements")
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

### Global attributes

#### `SOFA.create_global_attribute(name, value=None)`

**Purpose**
Create a global attribute.

**Parameters**
- `name : str`
  Attribute name.
- `value : str | None`
  Attribute value. Uses empty string when `None`.

**Returns**
- `None`

**Raises**
- `ValueError` if the attribute already exists.

**Example**
```python
sofa_copy.create_global_attribute("Title", "My HRTF")
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

#### `SOFA.modify_global_attribute(name, value)`

**Purpose**
Modify a global attribute.

**Parameters**
- `name : str`
  Attribute name.
- `value : str`
  New attribute value.

**Returns**
- `None`

**Raises**
- `ValueError` if the attribute does not exist.

**Example**
```python
sofa_copy.modify_global_attribute("Title", "Updated")
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

#### `SOFA.delete_global_attribute(name)`

**Purpose**
Delete a global attribute.

**Parameters**
- `name : str`
  Attribute name.

**Returns**
- `None`

**Raises**
- `ValueError` if the attribute does not exist.

**Example**
```python
sofa_copy.delete_global_attribute("Comment")
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

### Variable attributes

#### `SOFA.create_variable_attribute(name, value=None)`

**Purpose**
Create a variable attribute.

**Parameters**
- `name : str`
  Attribute name in the form `"Variable:Attribute"`.
- `value : str | None`
  Attribute value. Uses empty string when `None`.

**Returns**
- `None`

**Raises**
- `ValueError` if the variable does not exist or the attribute already exists.
- `ValueError` if `name` is not in `"Variable:Attribute"` format.

**Example**
```python
sofa_copy.create_variable_attribute("Data.IR:Units", "pascal")
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

#### `SOFA.modify_variable_attribute(name, value)`

**Purpose**
Modify a variable attribute.

**Parameters**
- `name : str`
  Attribute name in the form `"Variable:Attribute"`.
- `value : str`
  New attribute value.

**Returns**
- `None`

**Raises**
- `ValueError` if the variable or attribute does not exist.
- `ValueError` if `name` is not in `"Variable:Attribute"` format.

**Example**
```python
sofa_copy.modify_variable_attribute("Data.IR:Units", "Pa")
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

#### `SOFA.delete_variable_attribute(name)`

**Purpose**
Delete a variable attribute.

**Parameters**
- `name : str`
  Attribute name in the form `"Variable:Attribute"`.

**Returns**
- `None`

**Raises**
- `ValueError` if the variable or attribute does not exist.
- `ValueError` if `name` is not in `"Variable:Attribute"` format.

**Example**
```python
sofa_copy.delete_variable_attribute("Data.IR:Units")
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

### Variables

#### `SOFA.create_variable(name, data, dimensions, dtype=None, attributes=None)`

**Purpose**
Create a variable and optionally set its attributes.

**Parameters**
- `name : str`
  Variable name.
- `data : numpy.ndarray | list`
  Variable data.
- `dimensions : tuple[str, ...] | list[str]`
  Dimension names in order.
- `dtype : str | numpy.dtype | None`
  Data type for the variable. Defaults to the array dtype.
- `attributes : dict[str, Any] | None`
  Optional attributes to set on the variable.

**Returns**
- `None`

**Raises**
- `ValueError` if the variable already exists.
- `ValueError` if dimensions are missing.
- `ValueError` if data cannot be broadcast to the target shape.

**Warnings**
- Warns if data shape does not match declared dimensions.

**Example**
```python
data = np.zeros((sofa_copy.netCDF4_dataset.dimensions["M"].size,))
sofa_copy.create_variable("Custom", data, ("M",), attributes={"Units": "unitless"})
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

#### `SOFA.modify_variable(name, data)`

**Purpose**
Overwrite data for an existing variable.

**Parameters**
- `name : str`
  Variable name.
- `data : numpy.ndarray | list`
  New variable data.

**Returns**
- `None`

**Raises**
- `ValueError` if the variable does not exist.
- `ValueError` if data cannot be broadcast to the target shape.

**Warnings**
- Warns if data shape does not match declared dimensions.

**Example**
```python
new_data = np.zeros((100, 2, 256))
sofa_copy.modify_variable("Data.IR", new_data)
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

#### `SOFA.delete_variable(name)`

**Purpose**
Delete a variable.

**Parameters**
- `name : str`
  Variable name.

**Returns**
- `None`

**Raises**
- `ValueError` if the variable does not exist.

**Example**
```python
sofa_copy.delete_variable("Custom")
```

**Notes**
- Requires a writable SOFA object.
- Prints a status message on success.

---

## Validation and security

### `check_sofa_against_conventions(target, convention_name=None, version=None)`

**Purpose**
Validate a SOFA file or SOFA object against the conventions registry.

**Parameters**
- `target : str | pathlib.Path | netCDF4.Dataset | SOFA`
  File path, netCDF4 dataset, or SOFA object.
- `convention_name : str | None`
  Convention name to validate against. Defaults to the file metadata.
- `version : str | None`
  Convention version. Defaults to the file metadata.

**Returns**
- `dict`
  Summary dict: `{ "convention": {"name": ..., "version": ...} }`.

**Warnings**
- Emits warnings for missing mandatory metadata, shape mismatches, and
  custom (non-standard) attributes.

**Example**
```python
report = check_sofa_against_conventions("my.sofa")
```

**Notes**
- This function reports most issues as warnings, not exceptions.

---

### `check_sofa_security(target=None, hdf5_version=None, min_safe_hdf5="1.14.4", print_report=True, paranoid_mode=False)`

**Purpose**
Run security checks for SOFA/HDF5 handling.

**Parameters**
- `target : str | pathlib.Path | netCDF4.Dataset | SOFA | None`
  SOFA file path or object. Required for paranoid mode.
- `hdf5_version : str | None`
  HDF5 version to validate. If `None`, attempts auto-detection.
- `min_safe_hdf5 : str`
  Minimum acceptable HDF5 version.
- `print_report : bool`
  Print a formatted report if True.
- `paranoid_mode : bool`
  If True, scans raw file bytes only (no parsing).

**Returns**
- `dict`
  Report dictionary with `passed`, `hdf5_version`, and `checks`.

**Raises**
- `ValueError` in paranoid mode on failure or when no file path is provided.

**Example**
```python
report = check_sofa_security("my.sofa", print_report=False)
```

**Notes**
- Standard mode returns a report even if checks fail.
- Paranoid mode raises `ValueError` on failure and never parses the SOFA file.

---

## ConventionsManager

`ConventionsManager` provides a registry interface for adding, deleting,
and serializing convention specs without editing `conventions.py`. Registry
changes live in memory; export to JSON if you need persistence.

### `ConventionsManager.available_conventions_specifications()`

**Purpose**
Print a table of available conventions and versions.

**Returns**
- `None`

**Raises**
- `ValueError` if no conventions are registered.

**Example**
```python
ConventionsManager.available_conventions_specifications()
```

---

### `ConventionsManager.list_conventions_specifications()`

**Purpose**
Return a mapping of convention name → list of versions.

**Returns**
- `dict[str, list[str]]`

**Example**
```python
listing = ConventionsManager.list_conventions_specifications()
```

---

### `ConventionsManager.inspect_sofa_specification(name, version)`

**Purpose**
Return the spec dict for a convention version.

**Parameters**
- `name : str`
  Convention name.
- `version : str`
  Convention version.

**Returns**
- `dict`
  Spec dictionary.

**Raises**
- `KeyError` if the convention or version is not found.

**Example**
```python
spec = ConventionsManager.inspect_sofa_specification("SimpleFreeFieldHRIR", "1.2")
```

---

### `ConventionsManager.add_convention_specification(name, version, spec, overwrite=False)`

**Purpose**
Add or update a convention spec.

**Parameters**
- `name : str`
  Convention name.
- `version : str`
  Convention version.
- `spec : Mapping[str, Mapping[str, Any]]`
  Convention specification.
- `overwrite : bool`
  Allow overwriting an existing version.

**Returns**
- `None`

**Raises**
- `ValueError` if the version exists and `overwrite=False`.
- `ValueError` if the spec is missing required fields
  (`default`, `flags`, `dimensions`, `type`, `comment`).

**Example**
```python
ConventionsManager.add_convention_specification(
    "TempConvention",
    "0.1",
    {"GLOBAL:Conventions": {"default": "SOFA", "flags": "rm", "dimensions": None, "type": "attribute", "comment": ""}},
    overwrite=True,
)
```

---

### `ConventionsManager.delete_convention_specification_version(name, version)`

**Purpose**
Delete a specific convention version.

**Parameters**
- `name : str`
- `version : str`

**Returns**
- `None`

**Raises**
- `KeyError` if the convention or version is not found.

**Example**
```python
ConventionsManager.delete_convention_specification_version("TempConvention", "0.1")
```

---

### `ConventionsManager.delete_convention_specification(name)`

**Purpose**
Delete all versions for a convention.

**Parameters**
- `name : str`

**Returns**
- `None`

**Raises**
- `KeyError` if the convention is not found.

**Example**
```python
ConventionsManager.delete_convention_specification("TempConvention")
```

---

### `ConventionsManager.export_convention_specification_json(name, version, path)`

**Purpose**
Export a convention spec to JSON.

**Parameters**
- `name : str`
- `version : str`
- `path : str | pathlib.Path`

**Returns**
- `None`

**Raises**
- `KeyError` if the convention or version is not found.

**Example**
```python
ConventionsManager.export_convention_specification_json(
    "SimpleFreeFieldHRIR",
    "1.2",
    "spec.json",
)
```

---

### `ConventionsManager.add_convention_specification_from_json(path, overwrite=False)`

**Purpose**
Import a convention spec or registry from JSON.

**Parameters**
- `path : str | pathlib.Path`
- `overwrite : bool`
  Allow overwriting existing versions.

**Returns**
- `None`

**Raises**
- `FileNotFoundError` if the JSON file does not exist.
- `ValueError` if the JSON payload is invalid.

**Example**
```python
ConventionsManager.add_convention_specification_from_json("spec.json")
```

---

## SOFA conventions registry (`CONVENTIONS`)

The SOFA conventions are stored in a Python dictionary in
`hrtfpykit.sofa.conventions` named `CONVENTIONS`. Each entry maps:

- convention name → version → spec dictionary

Each spec entry describes a **global attribute**, **variable**, or
**variable attribute** using a common schema:

- `default`: default value required by the convention
- `flags`: requirement flags (see below)
- `dimensions`: expected dimension pattern
- `type`: storage type (`double`, `float`, `int`, `string`, etc.)
- `comment`: human-readable description

### Flags

Common flags used in the spec:

- `m` (mandatory): the item must exist in the file
- `r` (read-only): must match the default value

### Dimension strings

Dimension fields are declared in compact form (e.g., `MRN` or `IC, MC`).
Uppercase letters indicate required dimension types; lowercase letters
allow the variable size to determine the dimension length.

### Defaults

Defaults represent the expected baseline values for required fields. When
`r` is present, the value **must** match exactly. For `m` without `r`, the
field must be present but can differ.

This project ships an up-to-date local registry of available SOFA conventions.
It is still useful to consult the official SOFA convention documentation for
additional background and context.

---

## Common pitfalls

- **Missing write permissions**: use `SOFA.load(..., mode="r+")` only when
  you need in-place edits.
- **Overwriting a file while open**: close the original SOFA object or save to a new path.
- **Broadcasting errors**: check that your variable data can broadcast to
  the declared dimensions.
- **Missing convention metadata**: validation will warn if required fields
  are missing or inconsistent.
- **Forgetting to close SOFA objects**: call `sofa.netCDF4_dataset.close()` when done.

---

## Use cases

- Validate SOFA files.
- Inspect SOFA object metadata without using netCDF4 directly.
- Create minimal in-memory SOFA objects for tests.
- Export or import convention specs for reproducibility.
