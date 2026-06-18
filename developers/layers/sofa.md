# `hrtfpykit.sofa` layer

The SOFA layer owns file-level SOFA access, validation, editing, cloning, and saving. It is the lowest public layer in `hrtfpykit`.

## Public entry points

From `src/hrtfpykit/sofa/__init__.py`:

- `hrtfpykit.sofa.sofa.SOFA`
- `hrtfpykit.sofa.sofa.load_sofa()`
- `hrtfpykit.sofa.check.check_sofa_against_conventions()`
- `hrtfpykit.sofa.check.check_sofa_security()`

## Main objects

### `hrtfpykit.sofa.sofa.SOFA`

`hrtfpykit.sofa.sofa.SOFA` is defined in `src/hrtfpykit/sofa/sofa.py`.

It composes, rather than inherits from, `netCDF4.Dataset`:

```python
self.netCDF4_dataset: Optional[netCDF4.Dataset] = None
self.path: pathlib.Path | None = None
self._modified: bool = False
self._change_messages: list[str] = []
```

The core design is a controlled wrapper around netCDF4 storage. The wrapper lets the HRTF layer use SOFA files without directly depending on raw netCDF4 calls everywhere.

### Accessor composition

`hrtfpykit.sofa.sofa.SOFA` exposes storage collections as properties:

- `hrtfpykit.sofa.sofa.SOFA.Dimensions` returns `hrtfpykit.sofa.data._Dimensions`
- `hrtfpykit.sofa.sofa.SOFA.GlobalAttributes` returns `hrtfpykit.sofa.data._GlobalAttributes`
- `hrtfpykit.sofa.sofa.SOFA.Variables` returns `hrtfpykit.sofa.data._Variables`
- `hrtfpykit.sofa.sofa.SOFA.VariableAttributes` returns `hrtfpykit.sofa.data._VariableAttributes`

These classes are in `src/hrtfpykit/sofa/data.py`. They wrap raw netCDF4 dimensions, variables, and attributes into `hrtfpykit.sofa.wraps.DimensionsWrap`, `hrtfpykit.sofa.wraps.VariablesWrap`, and `hrtfpykit.sofa.wraps.AttributesWrap` from `src/hrtfpykit/sofa/wraps.py`.

The inheritance inside `data.py` is:

```text
_Data(ABC)
├── _Dimensions
├── _AttributesBase
│   ├── _GlobalAttributes
│   └── _VariableAttributes
└── _Variables
```

This inheritance is internal. The public API should remain the `hrtfpykit.sofa.sofa.SOFA` properties.

## Load workflow

```text
hrtfpykit.sofa.sofa.load_sofa(path)
    ↓
hrtfpykit.sofa.sofa.SOFA()
    ↓
hrtfpykit.sofa.sofa_helpers.open_sofa(
    sofa_object,
    path,
    mode,
    parallel,
    check_sofa_against_conventions,
)
    ↓
hrtfpykit.sofa.sofa.SOFA with open netCDF4_dataset
```

`hrtfpykit.sofa.sofa.load_sofa()` does not parse acoustic HRTF semantics. It opens the file and optionally validates convention metadata. Acoustic interpretation belongs to `hrtfpykit.hrtf.hrtf.load_hrtf()` in the HRTF layer.

## Validation workflow

`hrtfpykit.sofa.check.check_sofa_against_conventions()` lives in `src/hrtfpykit/sofa/check.py` and uses `hrtfpykit.sofa.conventions.CONVENTIONS` from `src/hrtfpykit/sofa/conventions.py`.

It checks:

- declared `SOFAConventions`;
- declared `SOFAConventionsVersion`;
- mandatory global attributes;
- mandatory variables;
- mandatory variable attributes;
- default/read-only values;
- dimension compatibility;
- custom global attributes, variables, variable attributes, and dimensions.

The function emits `hrtfpykit.utils.warnings.SOFAConventionWarning` through `hrtfpykit.utils.warnings.warn_user()` rather than rewriting the file.

`hrtfpykit.sofa.check.check_sofa_security()` is also in `check.py`. It performs defensive inspection of SOFA paths and attribute strings for suspicious links, file extensions, and HDF5 version concerns.

## Handle lifecycle

`hrtfpykit.sofa.sofa.SOFA.is_open()` checks whether the underlying `netCDF4.Dataset` is open.

`hrtfpykit.sofa.sofa.SOFA.close()` closes the current dataset handle.

`hrtfpykit.sofa.sofa.SOFA.open()` reopens from `hrtfpykit.sofa.sofa.SOFA.path`, using `hrtfpykit.sofa.sofa_helpers.open_sofa()` again. Objects with `path is None`, such as diskless clones, cannot be reopened from disk.

This lifecycle matters because `hrtfpykit.hrtf.hrtf.load_hrtf(..., sofa_open=False)` can close the SOFA handle after the HRTF arrays and source positions are loaded into memory.

## Mutation and saving

Mutation methods include:

- dimensions: `hrtfpykit.sofa.sofa.SOFA.create_dimension()`, `hrtfpykit.sofa.sofa.SOFA.rename_dimension()`
- global attributes: `hrtfpykit.sofa.sofa.SOFA.create_global_attribute()`, `hrtfpykit.sofa.sofa.SOFA.modify_global_attribute()`, `hrtfpykit.sofa.sofa.SOFA.delete_global_attribute()`
- variable attributes: `hrtfpykit.sofa.sofa.SOFA.create_variable_attribute()`, `hrtfpykit.sofa.sofa.SOFA.modify_variable_attribute()`, `hrtfpykit.sofa.sofa.SOFA.delete_variable_attribute()`
- variables: `hrtfpykit.sofa.sofa.SOFA.create_variable()`, `hrtfpykit.sofa.sofa.SOFA.modify_variable()`, `hrtfpykit.sofa.sofa.SOFA.delete_variable()`

These methods update the netCDF4 dataset and mark `_modified` so `hrtfpykit.sofa.sofa.SOFA.save()` can stamp metadata such as `DateModified` and `hrtfpykit`.

`hrtfpykit.sofa.sofa.SOFA.save()` either syncs the original dataset or writes a full temporary copy and atomically replaces the destination. It preserves variable storage metadata through `hrtfpykit.sofa.sofa_helpers.get_variable_creation_options()`.

## Cloning and structured copies

`hrtfpykit.sofa.sofa.SOFA.clone()` creates an independent diskless netCDF4 copy. It copies dimensions, global attributes, variables, variable attributes, values, and storage options.

`hrtfpykit.sofa.sofa.SOFA.copy_with()` creates a modified diskless copy with selected overrides:

- `dim_sizes`
- `global_attributes`
- `variable_attributes`
- `variables`

`hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` uses `hrtfpykit.sofa.sofa.SOFA.copy_with()` when transformed HRTF data need to resize SOFA dimensions or replace arrays safely.

## Logic example: clone, edit, and save SOFA storage

This example shows the intended file-level workflow. The SOFA layer owns the
netCDF4 handle and storage mutation; it does not interpret the acoustic meaning
of `Data.IR` beyond variable names, dimensions, and values.

```python
from hrtfpykit.sofa import load_sofa

sofa = load_sofa("subject.sofa", mode="r")
editable = sofa.clone()
editable.modify_variable("Data.IR", edited_ir)
editable.save("subject_edited.sofa", overwrite=True)
```

Calling flow:

```text
hrtfpykit.sofa.sofa.load_sofa("subject.sofa", mode="r")
    -> hrtfpykit.sofa.sofa.SOFA()
    -> hrtfpykit.sofa.sofa_helpers.open_sofa(
           sofa_object, path, mode, parallel, check_sofa_against_conventions
       )
       -> netCDF4.Dataset(...)
    -> return hrtfpykit.sofa.sofa.SOFA

hrtfpykit.sofa.sofa.SOFA.clone(sofa)
    -> hrtfpykit.sofa.sofa_helpers.require_dataset(self)
    -> netCDF4.Dataset(..., diskless=True, persist=False)
    -> hrtfpykit.sofa.sofa_helpers.get_variable_creation_options(var)
       for each copied variable
    -> return cloned hrtfpykit.sofa.sofa.SOFA

hrtfpykit.sofa.sofa.SOFA.modify_variable(editable, "Data.IR", edited_ir)
    -> hrtfpykit.sofa.sofa_helpers.require_dataset(self)
    -> numpy.array(data)
    -> hrtfpykit.sofa.sofa_helpers.warn_dimension_shape_mismatch(
           name, var.dimensions, array.shape, dataset
       )
    -> hrtfpykit.sofa.sofa_helpers.ensure_broadcastable(
           name, array, target_shape
       )
    -> netCDF4.Variable.__setitem__(dataset.variables["Data.IR"], ..., array)

hrtfpykit.sofa.sofa.SOFA.save(editable, "subject_edited.sofa", overwrite=True)
    -> hrtfpykit.sofa.sofa_helpers.require_dataset(self)
    -> hrtfpykit.sofa.sofa.SOFA.create_global_attribute(...)
       or hrtfpykit.sofa.sofa.SOFA.modify_global_attribute(...)
    -> netCDF4.Dataset(temp_path, mode="w", format=...)
    -> netCDF4.Dataset.createDimension(...) for each source dimension
    -> netCDF4.Dataset.setncatts(...) for global attributes
    -> netCDF4.Dataset.createVariable(
           ..., **hrtfpykit.sofa.sofa_helpers.get_variable_creation_options(var)
       )
    -> netCDF4.Variable.setncatts(...)
    -> netCDF4.Variable.__setitem__(dst_var, slice(None), var[:])
    -> os.replace(temp_path, target_path)
    -> return pathlib.Path
```

What each step does:

1. `hrtfpykit.sofa.sofa.load_sofa()` creates a `hrtfpykit.sofa.sofa.SOFA` object and delegates the netCDF4 open logic to
   `hrtfpykit.sofa.sofa_helpers.open_sofa()` from `src/hrtfpykit/sofa/sofa_helpers.py`.
2. `hrtfpykit.sofa.sofa.SOFA.clone()` reads the current `netCDF4.Dataset`, creates a diskless
   writable copy, and copies dimensions, global attributes, variables, variable
   attributes, values, and variable creation options.
3. `hrtfpykit.sofa.sofa.SOFA.modify_variable()` calls `hrtfpykit.sofa.sofa_helpers.require_dataset()`, converts the replacement
   values with `numpy.array`, checks that the data can be broadcast to the existing
   variable shape, writes the values into the netCDF4 variable, and marks the
   object as modified.
4. `hrtfpykit.sofa.sofa.SOFA.save()` writes the current storage representation to the requested
   destination. When saving to a new path, it creates a temporary netCDF4 file,
   copies the full storage content, preserves variable storage options through
   `hrtfpykit.sofa.sofa_helpers.get_variable_creation_options()`, and then replaces the destination path.

When dimensions or several variables must change together, prefer
`hrtfpykit.sofa.sofa.SOFA.copy_with(dim_sizes=..., variables=...)` over a sequence of ad hoc
mutations. That is the structured path used by `hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` when the
active HRTF arrays no longer match the original SOFA dimensions.

## Invariants

- SOFA file access should go through `hrtfpykit.sofa.sofa.SOFA`, `hrtfpykit.sofa.sofa.load_sofa()`, and helpers in `hrtfpykit.sofa`.
- `hrtfpykit.sofa.sofa.SOFA` should remain file/storage oriented; it should not own acoustic transform logic.
- HRTF semantics such as `Data.IR` vs `Data.Real`/`Data.Imag` interpretation belong in `hrtfpykit.hrtf.hrtf.load_hrtf()` and `hrtfpykit.hrtf.hrtf.HRTF.update_sofa()`, not generic SOFA accessors.
- New SOFA editing behavior should preserve storage metadata when copying variables.
