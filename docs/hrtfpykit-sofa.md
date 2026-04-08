# SOFA API

This guide describes the public SOFA API in `hrtfpykit.sofa`.
It covers the main objects, the normal edit workflows, and the methods used to
inspect, validate, copy, and save SOFA files and SOFA objects.

---

## Contents

- [Overview](#overview)
- [Main Objects](#main-objects)
- [Recommended Workflows](#recommended-workflows)
- [Public Surface](#public-surface)
- [Wrapper Collections](#wrapper-collections)
- [Core SOFA Methods](#core-sofa-methods)
- [CRUD Methods](#crud-methods)
- [Dimensions](#dimensions)
- [Global Attributes](#global-attributes)
- [Variable Attributes](#variable-attributes)
- [Variables](#variables)
- [Validation and Security](#validation-and-security)
- [ConventionsManager](#conventionsmanager)
- [The `CONVENTIONS` Registry](#the-conventions-registry)
- [Common Pitfalls](#common-pitfalls)
- [Typical Use Cases](#typical-use-cases)

---

## Overview

The SOFA layer is the library entry point for low-level SOFA file handling.
It wraps a `netCDF4.Dataset` and exposes a small high-level API for:

- loading existing `.sofa` files
- creating in-memory SOFA objects
- reading dimensions, variables, and attributes through wrapper objects
- editing SOFA objects through explicit copy or writable-open workflows
- validating files against the local conventions registry

The API is explicit by design:

- `load()` opens a SOFA file
- `clone()` and `copy_with()` create new in-memory SOFA objects
- `save()` writes a SOFA object to disk

There is no hidden file I/O.

---

## Main Objects

### `SOFA`

`SOFA` is the main high-level object.
It owns a `netCDF4.Dataset` and exposes convenience methods for SOFA-specific
CRUD operations and validation workflows.

Use `SOFA` when you want to:

- inspect a SOFA file without working directly with raw `netCDF4`
- edit metadata or variables with shape checks and SOFA-oriented naming
- prepare an in-memory modified copy before saving to disk

The underlying `netCDF4.Dataset` is still available as `sofa.netCDF4_dataset` for
advanced use.

### Wrapper Collections

A loaded `SOFA` object exposes four wrapper collections:

- `sofa.Dimensions`
- `sofa.GlobalAttributes`
- `sofa.VariableAttributes`
- `sofa.Variables`

These wrappers provide a consistent access pattern:

- `get(name)`
- `get_all()`
- `get_names()`
- `get_values()`
- `summary()`
- `__getitem__(name)`
- iteration
- `len(...)`

If no SOFA file is loaded, these properties return `None`.

### Wrap Objects

The wrapper collections return small inspection objects:

- `DimensionsWrap`
  - `name`
  - `value`
  - `is_unlimited`
- `AttributesWrap`
  - `name`
  - `value`
  - `type`
- `VariablesWrap`
  - `name`
  - `value`
  - `attributes`

`VariablesWrap.value` returns a NumPy copy of the variable data.

---

## Recommended Workflows

### Inspect a File

Use this when you want to read metadata, inspect variables, or validate a file.

```python
from hrtfpykit.sofa import SOFA, check_sofa_against_conventions

sofa = SOFA.load("my.sofa")

print(sofa.summary())
print(sofa.Dimensions["M"].value)
print(sofa.Variables["Data.IR"].value.shape)

check_sofa_against_conventions(sofa)
```

### Edit Safely in Memory

This is the default editing workflow.
Read the original file, clone or copy it in memory, make changes there, and
save the result to a new file path.

```python
sofa = SOFA.load("my.sofa")

sofa_clone = sofa.clone()
sofa_clone.modify_global_attribute("Title", "Updated title")
sofa_clone.save("my_updated.sofa")
```

`clone()` and `copy_with()` create independent in-memory SOFA objects.
They can be called repeatedly on the same source object.

### Edit in Place

Use this only when you intentionally want to modify the original file directly.

```python
sofa = SOFA.load("my.sofa", mode="r+")
sofa.modify_global_attribute("Title", "Updated title")
sofa.save()
```

### Build a Modified Copy in One Step

Use `copy_with()` when you already know the overrides you want to apply.

```python
sofa_mod = sofa.copy_with(
    dim_sizes={"N": 512},
    variables={"Data.IR": new_ir},
    global_attributes={"Title": "Modified HRTF"},
)
```

---

## Public Surface

### Classes

- `SOFA`
- `ConventionsManager`

### Functions

- `check_sofa_against_conventions`
- `check_sofa_security`

### Core SOFA Methods

- `SOFA.load`
- `SOFA.create_dummy`
- `SOFA.clone`
- `SOFA.copy_with`
- `SOFA.save`
- `SOFA.summary`

### SOFA CRUD Methods

- `SOFA.create_dimension`
- `SOFA.rename_dimension`
- `SOFA.create_global_attribute`
- `SOFA.modify_global_attribute`
- `SOFA.delete_global_attribute`
- `SOFA.create_variable_attribute`
- `SOFA.modify_variable_attribute`
- `SOFA.delete_variable_attribute`
- `SOFA.create_variable`
- `SOFA.modify_variable`
- `SOFA.delete_variable`

---

## Wrapper Collections

### `SOFA.Dimensions`

Access SOFA dimensions through `DimensionsWrap` objects.

```python
dims = sofa.Dimensions
print(dims.get_names())
print(dims["M"].value)
print(dims["N"].is_unlimited)
```

### `SOFA.GlobalAttributes`

Access global SOFA attributes through `AttributesWrap` objects.

```python
ga = sofa.GlobalAttributes
print(ga.get("Title").value)
```

### `SOFA.VariableAttributes`

Access variable attributes using `Variable:Attribute` names.

```python
va = sofa.VariableAttributes
print(va.get("Data.IR:Units").value)
```

### `SOFA.Variables`

Access SOFA variables through `VariablesWrap` objects.

```python
vars_wrap = sofa.Variables
ir = vars_wrap["Data.IR"].value
ir_attrs = vars_wrap["Data.IR"].attributes
```

### Shared Wrapper Methods

All four wrapper collections implement the same access model:

#### `get(name)`

Return one wrapped item by name.
Raises `ValueError` if the item does not exist.

#### `get_all()`

Return a `dict[name, wrap]` for the whole collection.

#### `get_names()`

Return the list of available names.

#### `get_values()`

Return the raw values of the collection.
For `Variables`, this can be memory heavy because it materializes NumPy copies.

#### `summary()`

Return a formatted text summary of that collection.

#### `__getitem__(name)`, iteration, and `len(...)`

The wrappers support dictionary-like access and iteration:

```python
ir = sofa.Variables["Data.IR"]

for dim in sofa.Dimensions:
    print(dim.name, dim.value)

print(len(sofa.GlobalAttributes))
```

---

## Core SOFA Methods

### `SOFA.load(path, mode="r", parallel=False, check_sofa_against_conventions=True)`

Load a SOFA file and return a `SOFA` object.

Parameters:
- `path`: path to the `.sofa` file
- `mode`: netCDF open mode such as `"r"` or `"r+"`
- `parallel`: open in parallel mode if supported by the local netCDF build
- `check_sofa_against_conventions`: run convention validation on load

Returns:
- `SOFA`

Use `mode="r"` for normal inspection and clone-based editing.
Use `mode="r+"` only when you want in-place changes.

### `SOFA.create_dummy(sofa_conventions, version=None, dim_sizes=None, custom_global_attributes=None, override_default_global_attributes=False)`

Create an in-memory SOFA object from a registered convention specification.

This is useful for:

- prototyping
- examples
- generating minimal SOFA objects

Important behavior:

- the reserved dimension `S` is always unlimited and created internally
- passing `S` in `dim_sizes` raises `ValueError`
- any other dimension with size `0` is treated as unlimited

### `SOFA.clone()`

Create an in-memory writable copy of the current SOFA object.

Use `clone()` when you want to:

- duplicate the current SOFA object as-is
- make manual edits through the CRUD methods
- save to a new output path without touching the original file

Each call creates a new independent in-memory SOFA object.

### `SOFA.copy_with(dim_sizes=None, global_attributes=None, variable_attributes=None, variables=None)`

Create an in-memory copy of the current SOFA object with explicit overrides.

Use `copy_with()` when you already know the modifications you want to apply.
It is especially useful for:

- resizing fixed dimensions and replacing dependent arrays at the same time
- updating metadata while keeping the rest of the dataset unchanged
- preparing a derived SOFA object in one expression

Important behavior:

- only fixed dimensions can be overridden
- overriding a dimension may require replacing variables that depend on it
- only existing variables can be overridden in this method
- repeated `copy_with()` calls on the same source object are supported

### `SOFA.save(path=None, overwrite=False)`

Write the current SOFA object to disk.

Behavior:

- `path=None` saves back to the original loaded path
- `path="new.sofa"` writes a new file
- `overwrite=False` prevents replacing an existing file

Use `save()` after:

- editing a SOFA file opened with `mode="r+"`
- creating a clone and preparing an output file
- creating a modified in-memory SOFA object with `copy_with()`

When writing from a separate SOFA object to the exact same on-disk path, normal
filesystem locking rules still apply. In that case, save to a new file path or
ensure the original writable object is no longer holding that path open.

### `SOFA.summary()`

Return a formatted summary of:

- global attributes
- variables
- variable attributes

Use it for fast inspection of a SOFA object without manually traversing the
underlying `netCDF4.Dataset`.

---

## CRUD Methods

These methods operate on a writable `SOFA` object, typically one produced by
`clone()`, `copy_with()`, `create_dummy()`, or `load(..., mode="r+")`.

### Dimensions

#### `SOFA.create_dimension(name, value)`

Create a new dimension in the SOFA object.

Raises `ValueError` if the dimension already exists.

#### `SOFA.rename_dimension(old_name, new_name)`

Rename an existing dimension.

Raises `ValueError` if the original dimension does not exist.

### Global Attributes

#### `SOFA.create_global_attribute(name, value=None)`

Create a new global attribute.

If `value` is `None`, an empty string is stored.

#### `SOFA.modify_global_attribute(name, value)`

Modify an existing global attribute.

Raises `ValueError` if the attribute does not exist.

#### `SOFA.delete_global_attribute(name)`

Delete a global attribute.

Raises `ValueError` if the attribute does not exist.

### Variable Attributes

#### `SOFA.create_variable_attribute(name, value=None)`

Create a variable attribute using the `Variable:Attribute` naming pattern.

Example:

```python
sofa_clone.create_variable_attribute("Data.IR:Units", "pascal")
```

#### `SOFA.modify_variable_attribute(name, value)`

Modify an existing variable attribute.

#### `SOFA.delete_variable_attribute(name)`

Delete a variable attribute.

All variable-attribute methods:

- require the `Variable:Attribute` naming pattern
- raise `ValueError` for invalid names or missing targets

### Variables

#### `SOFA.create_variable(name, data, dimensions, dtype=None, attributes=None)`

Create a new variable and optionally assign variable attributes.

Behavior:

- dimensions must already exist
- data must be broadcastable to the target variable shape
- the method warns when provided shapes do not match dimension sizes cleanly

Example:

```python
data = np.zeros((sofa_clone.Dimensions["M"].value,))
sofa_clone.create_variable(
    "Custom",
    data,
    ("M",),
    attributes={"Units": "unitless"},
)
```

#### `SOFA.modify_variable(name, data)`

Replace the contents of an existing variable.

The new data must be broadcastable to the stored variable shape.

#### `SOFA.delete_variable(name)`

Delete an existing variable.

All variable-editing methods raise `ValueError` when the target variable does
not exist or when the provided data cannot fit the declared dimensions.

---

## Validation and Security

### `check_sofa_against_conventions(target, convention_name=None, version=None)`

Validate a SOFA file or SOFA object against the local `CONVENTIONS` registry.

Accepted targets:

- file path
- `netCDF4.Dataset`
- `SOFA`

The report is warning-oriented.
Most problems are reported as warnings rather than exceptions, which makes the
function useful for inspection and diagnostics.

Use it to catch issues such as:

- missing required metadata
- shape mismatches
- non-standard custom attributes
- convention/version mismatches

### `check_sofa_security(target=None, hdf5_version=None, min_safe_hdf5="1.14.4", print_report=True, paranoid_mode=False)`

Run security-oriented checks related to SOFA and HDF5 handling.

Use standard mode when you want a report.
Use `paranoid_mode=True` when you want a stricter byte-level path that raises
on failure and avoids normal SOFA parsing.

---

## `ConventionsManager`

`ConventionsManager` manages the in-memory SOFA conventions registry.
Use it when you want to inspect, extend, import, export, or remove convention
specifications without editing `conventions.py` directly.

Registry changes are in-memory changes unless you export them.

### Available Methods

#### `ConventionsManager.available_conventions_specifications()`

Print the registered convention names and their available versions.

#### `ConventionsManager.inspect_sofa_specification(name, version)`

Return the specification dictionary for one convention version.

#### `ConventionsManager.add_convention_specification(name, version, spec, overwrite=False)`

Register a new convention specification or replace an existing one.

The specification must include the required fields used by the registry:

- `default`
- `flags`
- `dimensions`
- `type`
- `comment`

#### `ConventionsManager.delete_convention_specification_version(name, version)`

Delete one version of a convention.

#### `ConventionsManager.delete_convention_specification(name)`

Delete a convention and all of its versions.

#### `ConventionsManager.export_convention_specification_json(name, version, path)`

Export one convention version to JSON.

#### `ConventionsManager.add_convention_specification_from_json(path, overwrite=False)`

Load convention data from JSON.

Supported payload styles:

- a single convention payload
- a full registry payload

---

## The `CONVENTIONS` Registry

The SOFA conventions registry lives in `hrtfpykit.sofa.conventions` as the
`CONVENTIONS` dictionary.

The hierarchy is:

- convention name
- version
- specification dictionary

Each specification entry describes one global attribute, variable, or variable
attribute using fields such as:

- `default`
- `flags`
- `dimensions`
- `type`
- `comment`

Common flags:

- `m`: mandatory
- `r`: read-only, must match the registered default

The registry is the local reference used by validation and dummy creation.

---

## Common Pitfalls

- Opening a file in `mode="r+"` when you only need inspection. Prefer
  `load(..., mode="r")` plus `clone()` or `copy_with()`.
- Resizing a fixed dimension without also replacing the variables that depend
  on that dimension.
- Using `copy_with()` to add brand-new variables. `copy_with()` only overrides
  existing variables; use `create_variable()` for new ones.
- Passing variable attributes without the `Variable:Attribute` naming pattern.
- Expecting convention validation to raise on every problem. Most validation
  issues are emitted as warnings.
- Overwriting the exact same output path from a separate dataset handle instead
  of saving to a new file path or using the original writable object.

---

## Typical Use Cases

- inspect a SOFA dataset without writing raw `netCDF4` code
- validate a SOFA file against the local conventions registry
- prepare an edited in-memory copy and save it as a new file
- create dummy SOFA datasets for examples or tooling
- modify convention specifications and export them to JSON
