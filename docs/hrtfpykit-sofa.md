# SOFA API

The SOFA API provides the low-level interface for working with
SOFA files in `hrtfpykit`.

SOFA files are standardized netCDF/HDF5 files used to store spatial acoustic
data such as HRTFs, HRIRs, source positions, listener metadata, measurement
conditions, and convention-specific metadata. The SOFA API gives users a
Pythonic way to load, inspect, validate, edit, create, and save these files
without having to interact directly with the raw `netCDF4.Dataset` interface.

This API is designed for users who need direct control over the structure of a 
SOFA file. It exposes the main SOFA components — dimensions, global attributes, 
variable attributes, and variables — through dedicated wrapper collections and 
explicit editing methods. In addition, the API validates the file structure against 
the official SOFA conventions, raising warnings or errors when inconsistencies,
missing required elements, or incompatible definitions are detected. This makes 
it possible to work close to the SOFA data model while still relying on a safer,
more structured, and convention-aware object-oriented interface.

With the SOFA API, users can:

- open existing `.sofa` files as high-level `SOFA` objects
- inspect dimensions, variables, global attributes, and variable attributes
- access SOFA data arrays as NumPy-compatible values
- clone SOFA files into independent in-memory objects
- create modified copies with explicit metadata and data overrides
- edit dimensions, attributes, and variables through CRUD-style methods
- create convention-backed dummy SOFA objects for testing or prototyping
- validate files against the local SOFA conventions registry
- run security-oriented checks related to HDF5 handling and suspicious content
- import, export, inspect, add, or remove SOFA convention specifications

The SOFA API is the right layer to use when working directly with the structure of
a SOFA file. Within hrtfpykit, it serves as the foundational abstraction on which 
the higher level acoustic HRTF interface is built, and it also provides the file 
backbone for the dataset API designed to support HRTF datasets and deep learning workflows. For acoustic analysis, transformation, and HRTF processing, users may 
prefer the HRTF interface. For dataset construction, model training pipelines, and machine learning experiments, the dataset API provides a dedicated higher level abstraction. For inspection of SOFA files, metadata editing, convention validation, custom SOFA generation, or registry management, hrtfpykit.sofa is the appropriate 
entry point.

The canonical loader is:

```python
from hrtfpykit import load_sofa

sofa = load_sofa("example.sofa")
```

A loaded `SOFA` object exposes structured access to the file content through:

```python
sofa.Dimensions
sofa.GlobalAttributes
sofa.VariableAttributes
sofa.Variables
```

The underlying raw dataset remains available for expert workflows as:

```python
sofa.netCDF4_dataset
```

Official SOFA conventions reference:

- https://www.sofaconventions.org/mediawiki/index.php/SOFA_conventions


---

## Contents

- [Overview](#overview)
- [Import Patterns](#import-patterns)
- [Main Objects](#main-objects)
- [Recommended Workflows](#recommended-workflows)
- [Top-Level Public Surface](#top-level-public-surface)
- [The `load_sofa()` Entry Point](#the-load_sofa-entry-point)
- [The `SOFA` Object](#the-sofa-object)
- [Wrapper Collections](#wrapper-collections)
- [CRUD Methods](#crud-methods)
- [Validation and Security](#validation-and-security)
- [Conventions Management](#conventions-management)
- [The `CONVENTIONS` Registry](#the-conventions-registry)
- [Common Pitfalls](#common-pitfalls)
- [Typical End-to-End Examples](#typical-end-to-end-examples)

---

## Overview

The SOFA layer is the low-level file and metadata layer of `hrtfpykit`.
It wraps `netCDF4.Dataset` objects and provides a higher-level API for:

- loading existing `.sofa` files
- reading dimensions, variables, and attributes through wrapper collections
- editing SOFA objects through explicit in-memory copies or writable-open workflows
- creating new convention-backed dummy SOFA objects
- validating files against the local conventions registry
- running security checks related to HDF5 versioning and suspicious content

The SOFA layer is intentionally explicit:

- `load_sofa()` opens a file and returns a `SOFA` object
- `clone()` creates a new in-memory copy
- `copy_with()` creates a modified in-memory copy with explicit overrides
- `save()` writes the current SOFA object to disk

There is no hidden file I/O.
There is no implicit mutation of the original file unless you intentionally
open it in writable mode and save back to the same path.

---

## Import Patterns

The current canonical import patterns are:

### 1. Main loader entry point

```python
from hrtfpykit import load_sofa
```

This is the normal way to open a SOFA file.

### 2. SOFA package utilities

```python
from hrtfpykit.sofa import (
    load_sofa,
    check_sofa_against_conventions,
    check_sofa_security,
    ConventionsManager,
)
```

This is the normal import style when you want the SOFA package-level tools.

### 3. Direct access to the `SOFA` class

```python
from hrtfpykit.sofa.sofa import SOFA
```

This is the current direct import path for the `SOFA` class itself.

Practical rule:

- use `load_sofa()` for file loading
- use `SOFA` for object construction helpers such as `create_dummy(...)`
- use `ConventionsManager` and the check functions for registry and validation workflows

---

## Main Objects

### `SOFA`

`SOFA` is the main high-level SOFA object.
It wraps a `netCDF4.Dataset` and exposes:

- wrapper collections for dimensions, attributes, and variables
- CRUD-style methods for editing metadata and variables
- helper methods for cloning, copying, saving, and summarizing

Use `SOFA` when you want to:

- inspect SOFA content without working directly with raw `netCDF4`
- edit metadata or variable arrays with SOFA-aware checks
- create modified in-memory copies before saving
- build convention-backed dummy SOFA objects

The underlying raw dataset is still available as:

- `sofa.netCDF4_dataset`

for expert workflows.

### Wrapper Collections

A loaded `SOFA` object exposes four wrapper collections:

- `sofa.Dimensions`
- `sofa.GlobalAttributes`
- `sofa.VariableAttributes`
- `sofa.Variables`

These wrappers provide a consistent access model:

- `get(name)`
- `get_all()`
- `get_names()`
- `get_values()`
- `summary()`
- `__getitem__(name)`
- iteration
- `len(...)`

If no dataset is loaded, these properties return `None`.

### Wrap Objects

The wrappers return small inspection objects.

Main wrap types:

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

Important behavior:

- `VariablesWrap.value` returns a NumPy copy of the data
- wrapper objects are for inspection and access, not direct in-place mutation

### `ConventionsManager`

`ConventionsManager` is the conventions-registry management API.

Use it when you want to:

- inspect registered convention specifications
- add a new convention specification
- delete one version or a whole convention
- export one convention spec to JSON
- import one convention spec or a whole registry payload from JSON

---

## Recommended Workflows

### Inspect a file

Use this when you want to read metadata, inspect variables, or validate a file.

```python
from hrtfpykit import load_sofa
from hrtfpykit.sofa import check_sofa_against_conventions

sofa = load_sofa("my.sofa")

print(sofa.summary())
print(sofa.Dimensions["M"].value)
print(sofa.Variables["Data.IR"].value.shape)

check_sofa_against_conventions(sofa)
```

### Edit safely in memory

This is the default editing workflow.
Read the original file, clone it in memory, make changes there, and save the
result to a new path.

```python
from hrtfpykit import load_sofa

sofa = load_sofa("my.sofa")

sofa_clone = sofa.clone()
sofa_clone.modify_global_attribute("Title", "Updated title")
sofa_clone.save("my_updated.sofa", overwrite=True)
```

### Edit in place

Use this only when you intentionally want to modify the original file directly.

```python
from hrtfpykit import load_sofa

sofa = load_sofa("my.sofa", mode="r+")
sofa.modify_global_attribute("Title", "Updated title")
sofa.save()
```

### Build a modified copy in one step

Use `copy_with()` when you already know the exact overrides you want to apply.

```python
sofa_mod = sofa.copy_with(
    dim_sizes={"N": 512},
    variables={"Data.IR": new_ir},
    global_attributes={"Title": "Modified HRTF"},
)
```

### Create a dummy SOFA object

Use this when you want a convention-backed in-memory object for examples,
prototyping, or data generation.

```python
from hrtfpykit.sofa.sofa import SOFA

sofa = SOFA.create_dummy("SimpleFreeFieldHRIR", version="1.2")
print(sofa.summary())
```

---

## Top-Level Public Surface

### Top-level exports from `hrtfpykit.sofa`

- `load_sofa`
- `check_sofa_against_conventions`
- `check_sofa_security`
- `ConventionsManager`

### Core object

- `SOFA`
  - currently imported from `hrtfpykit.sofa.sofa`

### Main `SOFA` methods

- `create_dummy(...)`
- `clone()`
- `copy_with(...)`
- `save(...)`
- `summary()`

### CRUD methods

- `create_dimension(...)`
- `rename_dimension(...)`
- `create_global_attribute(...)`
- `modify_global_attribute(...)`
- `delete_global_attribute(...)`
- `create_variable_attribute(...)`
- `modify_variable_attribute(...)`
- `delete_variable_attribute(...)`
- `create_variable(...)`
- `modify_variable(...)`
- `delete_variable(...)`

---

## The `load_sofa()` Entry Point

### Purpose

`load_sofa()` is the canonical loader for SOFA files.

It opens the file, validates its path and extension, optionally checks it
against the declared convention, and returns a `SOFA` object backed by an
open `netCDF4.Dataset`.

### Signature

```python
load_sofa(
    path,
    mode="r",
    parallel=False,
    check_sofa_against_conventions=True,
)
```

### Parameters

#### `path`

Path to the `.sofa` file.

#### `mode`

netCDF open mode.

Typical values:

- `"r"`
  - read-only inspection and clone-based workflows
- `"r+"`
  - expert in-place editing workflows

#### `parallel`

Whether the dataset should be opened in parallel mode.
This is an expert option and depends on the local netCDF/HDF5 stack.

#### `check_sofa_against_conventions`

Whether the loader should run the conventions check when opening the file.

Recommended default:

- `True`

### Returns

- `SOFA`

### Examples

Load a file for standard inspection:

```python
from hrtfpykit import load_sofa

sofa = load_sofa("my.sofa")
```

Open a file for in-place editing:

```python
sofa = load_sofa("my.sofa", mode="r+")
```

Disable convention checks temporarily:

```python
sofa = load_sofa("my.sofa", check_sofa_against_conventions=False)
```

---

## The `SOFA` Object

### Role

`SOFA` is the main editable object of the SOFA layer.

It holds:

- the backed `netCDF4.Dataset`
- the original `path` when loaded from disk
- wrapper-based views over dimensions, global attributes, variable attributes, and variables

### Properties

- `Dimensions`
- `GlobalAttributes`
- `Variables`
- `VariableAttributes`

### Main object lifecycle methods

- `create_dummy(...)`
- `clone()`
- `copy_with(...)`
- `save(...)`
- `summary()`

### `create_dummy(...)`

Create an in-memory SOFA object from a registered convention specification.

Main parameters:

- `sofa_conventions`
- `version`
- `dim_sizes`
- `custom_global_attributes`
- `override_default_global_attributes`

Important behaviour:

- reserved dimension `S` is always created internally as unlimited
- passing `S` in `dim_sizes` raises `ValueError`
- any other dimension with size `0` is treated as unlimited

Example:

```python
from hrtfpykit.sofa.sofa import SOFA

sofa = SOFA.create_dummy(
    "SimpleFreeFieldHRIR",
    version="1.2",
    dim_sizes={"M": 10, "N": 256},
)
```

### `clone()`

Create a new independent in-memory SOFA copy.

Use it when:

- you want the safest edit workflow
- you want multiple editing branches from the same source file

Example:

```python
sofa_clone = sofa.clone()
```

### `copy_with(...)`

Create a modified in-memory copy using explicit overrides.

Main parameters:

- `dim_sizes`
- `global_attributes`
- `variable_attributes`
- `variables`

Use it when:

- you want a one-expression modified copy
- you are resizing dimensions and replacing dependent arrays together
- you want a derived SOFA object without a sequence of manual CRUD calls

Example:

```python
sofa_mod = sofa.copy_with(
    dim_sizes={"N": 512},
    variables={"Data.IR": new_ir},
)
```

### `save(path=None, overwrite=False)`

Write the current SOFA object to disk.

Parameters:

- `path`
  - `None` means save back to the original loaded path
- `overwrite`
  - whether an existing target path may be replaced

Important behaviour:

- save without `path` requires that the object has an original path
- save to an existing file with `overwrite=False` raises `FileExistsError`
- the method writes a temporary copy first and then replaces the target

Example:

```python
sofa.save("copy.sofa", overwrite=True)
```

### `summary()`

Return a formatted summary of the whole SOFA object.

Use it when:

- you want a quick inspection of dimensions, variables, and attributes
- you want a debugging overview before editing

Example:

```python
print(sofa.summary())
```

---

## Wrapper Collections

The wrapper collections are the read interface of the `SOFA` object.

They are designed to give one consistent access style across dimensions,
attributes, and variables.

### Shared wrapper methods

All four wrappers support:

- `get(name)`
- `get_all()`
- `get_names()`
- `get_values()`
- `summary()`
- `__getitem__(name)`
- iteration
- `len(...)`

### `SOFA.Dimensions`

Use this collection for dimension metadata.

Example:

```python
dims = sofa.Dimensions

print(dims.get_names())
print(dims["M"].value)
print(dims["N"].is_unlimited)
```

### `SOFA.GlobalAttributes`

Use this collection for global SOFA attributes.

Example:

```python
ga = sofa.GlobalAttributes
print(ga.get("Title").value)
```

### `SOFA.VariableAttributes`

Use this collection for variable attributes addressed as `Variable:Attribute`.

Example:

```python
va = sofa.VariableAttributes
print(va.get("Data.IR:Units").value)
```

### `SOFA.Variables`

Use this collection for SOFA variables.

Example:

```python
variables = sofa.Variables
ir = variables["Data.IR"].value
ir_attrs = variables["Data.IR"].attributes
```

### Wrapper object semantics

#### Dimensions

Wrapped dimensions expose:

- `name`
- `value`
- `is_unlimited`

#### Attributes

Wrapped attributes expose:

- `name`
- `value`
- `type`

#### Variables

Wrapped variables expose:

- `name`
- `value`
- `attributes`

Important detail:

- `VariablesWrap.value` returns a NumPy copy
- reading many large variables through `get_values()` can be memory-heavy

---

## CRUD Methods

The CRUD methods are the explicit editing interface of the `SOFA` object.

These methods are grouped by what they edit:

- dimensions
- global attributes
- variable attributes
- variables

### Dimensions

#### `create_dimension(name, value)`

Create a new dimension.

Use it when:

- you are adding a custom dimension to a writable or cloned SOFA object

Example:

```python
sofa_clone.create_dimension("X", 3)
```

#### `rename_dimension(old_name, new_name)`

Rename an existing dimension.

Example:

```python
sofa_clone.rename_dimension("M", "Measurements")
```

### Global attributes

#### `create_global_attribute(name, value=None)`

Create a new global attribute.

Example:

```python
sofa_clone.create_global_attribute("DemoNote", "synthetic example")
```

#### `modify_global_attribute(name, value)`

Modify an existing global attribute.

Example:

```python
sofa_clone.modify_global_attribute("Title", "Updated title")
```

#### `delete_global_attribute(name)`

Delete a global attribute.

Example:

```python
sofa_clone.delete_global_attribute("DemoNote")
```

### Variable attributes

Variable attribute names use the form:

- `"Variable:Attribute"`

#### `create_variable_attribute(name, value=None)`

Example:

```python
sofa_clone.create_variable_attribute("Data.IR:DemoTag", "raw")
```

#### `modify_variable_attribute(name, value)`

Example:

```python
sofa_clone.modify_variable_attribute("Data.IR:DemoTag", "windowed")
```

#### `delete_variable_attribute(name)`

Example:

```python
sofa_clone.delete_variable_attribute("Data.IR:DemoTag")
```

### Variables

#### `create_variable(name, data, dimensions, dtype=None, attributes=None)`

Create a new variable and optionally attach attributes.

Important parameters:

- `name`
- `data`
- `dimensions`
- `dtype`
- `attributes`

Important behaviour:

- dimensions must exist first
- shapes are validated for broadcast compatibility
- dimension mismatch warnings can be emitted by the helper layer

Example:

```python
import numpy as np

data = np.zeros((sofa_clone.netCDF4_dataset.dimensions["M"].size,))
sofa_clone.create_variable(
    "Custom",
    data,
    ("M",),
    attributes={"Units": "unitless"},
)
```

#### `modify_variable(name, data)`

Overwrite one existing variable with new data.

Example:

```python
sofa_clone.modify_variable("Data.IR", new_ir)
```

#### `delete_variable(name)`

Delete one variable.

Example:

```python
sofa_clone.delete_variable("Custom")
```

---

## Validation and Security

The SOFA layer includes two distinct validation tools:

- convention validation
- security checks

### `check_sofa_against_conventions(...)`

Validate a SOFA file or dataset against its convention specification.

Accepted target types:

- file path
- open `netCDF4.Dataset`
- loaded `SOFA` object

Main parameters:

- `target`
- `convention_name`
- `version`

This check emits warnings for things such as:

- missing mandatory attributes or variables
- default mismatches on read-only fields
- dimension mismatches
- custom attributes, variables, or dimensions not present in the spec

Example:

```python
from hrtfpykit.sofa import check_sofa_against_conventions

report = check_sofa_against_conventions("my.sofa")
print(report)
```

### `check_sofa_security(...)`

Run security-oriented checks for SOFA and HDF5 handling.

Main parameters:

- `target`
- `hdf5_version`
- `min_safe_hdf5`
- `print_report`
- `paranoid_mode`

This check covers:

- HDF5 runtime version against a minimum safe baseline
- suspicious URLs in attributes or raw file content
- suspicious file extensions in attributes or raw file content

Modes:

- standard mode
  - parses dataset attributes through `netCDF4`
- paranoid mode
  - scans raw file bytes only and never parses the dataset

Example:

```python
from hrtfpykit.sofa import check_sofa_security

report = check_sofa_security("my.sofa", print_report=False)
print(report["passed"])
```

Paranoid mode:

```python
report = check_sofa_security(
    "my.sofa",
    print_report=False,
    paranoid_mode=True,
)
```

---

## Conventions Management

`ConventionsManager` is the explicit API for working with the local SOFA
conventions registry.

### Main methods

- `available_conventions_specifications()`
- `inspect_sofa_specification(name, version)`
- `add_convention_specification(name, version, spec, overwrite=False)`
- `delete_convention_specification_version(name, version)`
- `delete_convention_specification(name)`
- `export_convention_specification_json(name, version, path)`
- `add_convention_specification_from_json(path, overwrite=False)`

### `available_conventions_specifications()`

Print the registered conventions and versions.

Example:

```python
from hrtfpykit.sofa import ConventionsManager

ConventionsManager.available_conventions_specifications()
```

### `inspect_sofa_specification(name, version)`

Return one convention specification dictionary.

Example:

```python
spec = ConventionsManager.inspect_sofa_specification(
    "SimpleFreeFieldHRIR",
    "1.2",
)
```

### `add_convention_specification(...)`

Register one new convention specification.

Important parameters:

- `name`
- `version`
- `spec`
- `overwrite`

### `delete_convention_specification_version(...)`

Delete one specific version of a convention.

### `delete_convention_specification(...)`

Delete one whole convention and all its versions.

### `export_convention_specification_json(...)`

Export one convention specification to JSON.

### `add_convention_specification_from_json(...)`

Import a convention specification or a registry payload from JSON.

Supported JSON shapes:

- single convention payload
- whole registry payload

---

## The `CONVENTIONS` Registry

The local registry lives in:

- `hrtfpykit.sofa.conventions.CONVENTIONS`

It is the specification source used by:

- `SOFA.create_dummy(...)`
- `check_sofa_against_conventions(...)`
- `ConventionsManager`

Conceptually, `CONVENTIONS` defines:

- available conventions
- supported versions
- expected globals
- expected variables
- expected variable attributes
- expected dimensions
- default values and flags

Practical rule:

- treat the registry as the low-level source of truth
- use `ConventionsManager` for normal registry workflows instead of mutating it blindly

---

## Common Pitfalls

### Using stale loader examples

The current canonical loader is:

- `load_sofa(...)`

not:

- `SOFA.load(...)`

If you are updating older examples or notebooks, this is the first change to make.

### Confusing inspection with mutation

Wrapper objects are inspection helpers.
They do not replace the CRUD methods.

Use:

- wrapper collections to read
- CRUD methods to edit

### Editing the original file unintentionally

If you open with:

- `mode="r+"`

and then save without a new path, you are working on the original file.

Default safe workflow:

- open read-only
- clone
- edit the clone
- save to a new file

### Forgetting shape compatibility when modifying variables

When replacing variable data:

- dimensions must exist
- shapes must be broadcast-compatible
- resizing dimensions may require coordinated updates of dependent variables

### Using `get_values()` on large variable collections carelessly

For `Variables`, `get_values()` materializes NumPy copies and can be memory-heavy.

Use it carefully for large datasets.

### Treating security checks as convention validation

These are different tools:

- `check_sofa_against_conventions(...)`
  - checks semantic/spec compliance
- `check_sofa_security(...)`
  - checks HDF5/version/content risk signals

Use the right one for the right problem.

---

## Typical End-to-End Examples

### 1. Inspect one SOFA file

```python
from hrtfpykit import load_sofa

sofa = load_sofa("my.sofa")

print(sofa.summary())
print(sofa.Dimensions.get_names())
print(sofa.GlobalAttributes.get_names())
print(sofa.Variables["Data.IR"].value.shape)
```

### 2. Create a safe edited copy

```python
from hrtfpykit import load_sofa

sofa = load_sofa("my.sofa")
sofa_clone = sofa.clone()
sofa_clone.modify_global_attribute("Title", "Updated title")
sofa_clone.save("updated.sofa", overwrite=True)
```

### 3. Build a modified copy with explicit overrides

```python
sofa_mod = sofa.copy_with(
    global_attributes={"Title": "Modified HRTF"},
    variables={"Data.IR": new_ir},
)
sofa_mod.save("modified.sofa", overwrite=True)
```

### 4. Create a dummy convention-backed object

```python
from hrtfpykit.sofa.sofa import SOFA

sofa = SOFA.create_dummy("SimpleFreeFieldHRIR", version="1.2")
print(sofa.summary())
```

### 5. Validate one file against its convention

```python
from hrtfpykit.sofa import check_sofa_against_conventions

report = check_sofa_against_conventions("my.sofa")
print(report)
```

### 6. Run a security check

```python
from hrtfpykit.sofa import check_sofa_security

report = check_sofa_security("my.sofa", print_report=False)
print(report["passed"])
```

### 7. Export and import a convention specification

```python
from hrtfpykit.sofa import ConventionsManager

ConventionsManager.export_convention_specification_json(
    "SimpleFreeFieldHRIR",
    "1.2",
    "simplefreefieldhrir_1_2.json",
)

ConventionsManager.add_convention_specification_from_json(
    "simplefreefieldhrir_1_2.json",
    overwrite=True,
)
```
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
