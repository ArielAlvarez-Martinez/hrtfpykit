# HRTF API

The HRTF API is the acoustic working layer of `hrtfpykit`. It is the right layer to use when working with HRTF or HRIR data as acoustic objects rather than as raw SOFA file components. The API is built around the `HRTF` object, which loads SOFA based data and keeps the time domain impulse responses and frequency domain transfer functions available through a single interface.

Within `hrtfpykit`, the HRTF API sits above the SOFA API. The SOFA layer provides the structural representation of the file, while the HRTF layer provides the acoustic abstraction used for analysis, transformation, comparison, spherical harmonic processing, and export. This abstraction also provides the acoustic foundation used by the dataset API for HRTF datasets and deep learning workflows.

The HRTF API is designed for workflows where users need to inspect source positions, select spatial or spectral regions, move between impulse responses and transfer functions, apply acoustic transformations, compute binaural and spectral metrics, derive spherical harmonic representations, and write processed results back to SOFA when required. For acoustic analysis and processing, `hrtfpykit.hrtf` is the appropriate entry point. Each `HRTF` object retains its backed `SOFA` object through `hrtf.Sofa`, allowing acoustic workflows to remain connected to the original SOFA structure for inspection, metadata access, convention checks, editing, and export.

With the HRTF API, users can:

- open existing `.sofa` files as high level `HRTF` objects
- load both `SimpleFreeFieldHRIR` and `SimpleFreeFieldHRTF` data
- access time domain impulse responses through the `IR` view
- access frequency domain transfer functions through the `TF` view
- inspect and query the source grid through the `Sources` interface
- select spatial, temporal, spectral, or ear specific subsets
- move between impulse responses and transfer functions without manual bookkeeping
- create independent acoustic copies for parallel processing workflows
- apply non destructive acoustic transformations that return new `HRTF` objects
- modify impulse responses, transfer functions, magnitude, phase, gain, or FFT length
- compute binaural and spectral metrics such as ITD, ILD, and LSD
- derive spherical harmonic representations of HRTF magnitude data
- synchronize processed acoustic data back into the backed SOFA object
- save processed HRTF or HRIR content as a new `.sofa` file

The canonical loader is:

```
from hrtfpykit import load_hrtf

hrtf = load_hrtf("example.sofa")
```

A loaded `HRTF` object exposes structured access to the acoustic content through:

```
hrtf.IR
hrtf.TF
hrtf.Sources
hrtf.transform
```

The backed SOFA representation remains available through:

```
hrtf.Sofa
```

The SOFA convention associated with the loaded object can be inspected as:

```
hrtf.SOFAConventions
```

---

## Contents

- [Overview](#overview)
- [Import Patterns](#import-patterns)
- [Main Objects](#main-objects)
- [Recommended Workflows](#recommended-workflows)
- [Top-Level Public Surface](#top-level-public-surface)
- [The `load_hrtf()` Entry Point](#the-load_hrtf-entry-point)
- [The `HRTF` Object](#the-hrtf-object)
- [Domain Views: `IR` and `TF`](#domain-views-ir-and-tf)
- [Source Grid API: `Sources`](#source-grid-api-sources)
- [Transform API](#transform-api)
- [DSP API](#dsp-api)
- [Metrics API](#metrics-api)
- [Spherical Harmonics API](#spherical-harmonics-api)
- [Common Pitfalls](#common-pitfalls)
- [Typical End-to-End Examples](#typical-end-to-end-examples)

---

## Overview

The HRTF layer is the acoustic core of `hrtfpykit`.
It is built around one main abstraction:

- `HRTF`

This object is designed to load both:

- `SimpleFreeFieldHRIR`
- `SimpleFreeFieldHRTF`

and keep both acoustic representations available after loading:

- `IR`
  - time-domain impulse responses
- `TF`
  - frequency-domain transfer functions

This is the key design rule of the module:

- one object
- two synchronized acoustic views

The HRTF layer is intentionally explicit:

- `load_hrtf()` loads one SOFA file into an `HRTF` object
- `select()` creates an in-memory selected copy
- `transform` creates derived HRTF objects
- `update_sofa()` writes current in-memory acoustic data back into the backed SOFA object
- `save()` persists the result to disk

There is no hidden acoustic synchronization.
If you transform an HRTF in memory and want those changes reflected in SOFA
variables before export, you call `update_sofa()` explicitly.

---

## Import Patterns

There are three main import levels you will typically use.

### 1. Main user entry points

Use this when you want the normal high-level workflow.

```python
from hrtfpykit import load_hrtf
```

### 2. HRTF package-level acoustic functions

Use this when you want top-level HRTF metrics or SH functions.

```python
from hrtfpykit.hrtf import itd, ild, lsd, sht
```

### 3. Submodule-level APIs

Use this when you want one specific subsystem directly.

```python
from hrtfpykit.hrtf.dsp import tf_from_ir, magnitude_db
from hrtfpykit.hrtf.metrics import itd_difference
from hrtfpykit.hrtf.sh import SH, sht_inverse
```

Practical rule:

- use `load_hrtf()` and the `HRTF` object for normal workflows
- use submodules directly only when you need lower-level control

---

## Main Objects

### `HRTF`

`HRTF` is the main acoustic object.
It owns the backed `SOFA` object and coordinates the rest of the HRTF API.

It exposes:

- `IR`
- `TF`
- `Sources`
- `transform`

Use `HRTF` when you want to:

- load one SOFA file and work with it as one acoustic object
- move between time and frequency domains without manual bookkeeping
- select a subset of positions or acoustic regions
- derive modified HRTFs while preserving the original object
- export the current state back into SOFA

### `IR`

`IR` is the time-domain view of the HRTF.

It stores:

- `values`
- `sample_rate`

It also exposes time-domain convenience properties and operations such as:

- `ir_length`
- `ir_duration`
- `get_itd(...)`

### `TF`

`TF` is the frequency-domain view of the HRTF.

It stores:

- `values`
- `frequency_bins`

It also exposes convenience properties and methods such as:

- `tf_length`
- `frequency_bins_step`
- `min_frequency_bin`
- `max_frequency_bin`
- `magnitude`
- `get_magnitude_db(...)`
- `phase`
- `real`
- `imag`

### `Sources`

`Sources` is the spatial source-grid view of the HRTF.

It resolves and converts the SOFA `SourcePosition` data so you can work with
the source grid through one interface.

Use it when you want to:

- read positions in spherical, cartesian, or lateral-polar coordinates
- inspect azimuth and elevation coverage
- map user queries to the nearest real source-grid position
- support selection and plotting workflows tied to source positions

### `Transform`

`Transform` is the non-destructive acoustic transformation API.

Every transform returns a new `HRTF` object.
The current object is not overwritten.

Use it when you want to:

- window or pad HRIRs
- resample the IR
- apply FIR or IIR filtering
- modify magnitude or phase
- change FFT length
- add or remove ITD
- derive DTF or CTF forms

---

## Recommended Workflows

### Load and inspect

Use this when you first want to understand what one dataset contains.

```python
from hrtfpykit import load_hrtf

hrtf = load_hrtf("my_hrtf.sofa")

print(hrtf.SOFAConventions)
print(hrtf.IR.values.shape)
print(hrtf.IR.sample_rate)
print(hrtf.TF.values.shape)
print(hrtf.TF.frequency_bins.shape)
print(hrtf.Sources.get_positions().shape)
```

### Select before analysing

This is the normal workflow when you only need a spatial or acoustic subset.

```python
front = hrtf.select(positions=["front"])
horizontal = hrtf.select(plane="horizontal")
early_ir = hrtf.select(start_seconds=0.0, end_seconds=0.01)
```

### Transform safely in memory

This is the default workflow for acoustic edits.

```python
windowed = hrtf.transform.apply_window("hann")
minimum_phase = hrtf.transform.minimum_phase()
resampled = hrtf.transform.downsampling(24000.0)
```

### Export an edited copy

Use this when the in-memory result should become a SOFA file.

```python
processed = hrtf.transform.apply_padding(32, location="end")
processed.update_sofa(change_sofa_dimensions=True)
processed.save("processed.sofa", overwrite=True)
```

---

## Top-Level Public Surface

This section describes the package-level public surface currently exposed by
`hrtfpykit.hrtf`.

### Top-level exports

- `load_hrtf`
- `itd`
- `ild`
- `itd_difference`
- `ild_difference`
- `lsd`
- `sht`
- `sht_inverse`
- `sht_error`

These names are meant for the most common workflows.

### Important submodules

The broader HRTF layer also includes these module-level APIs:

- `hrtfpykit.hrtf.dsp`
- `hrtfpykit.hrtf.metrics`
- `hrtfpykit.hrtf.sh`
- `hrtfpykit.hrtf.transforms`
- `hrtfpykit.hrtf.sources`
- `hrtfpykit.hrtf.domain`

These are part of the practical public documentation surface even when not
every symbol is re-exported from `hrtfpykit.hrtf.__init__`.

---

## The `load_hrtf()` Entry Point

### Purpose

`load_hrtf()` is the normal way to construct an `HRTF` object from a SOFA file.

It supports both:

- `SimpleFreeFieldHRIR`
- `SimpleFreeFieldHRTF`

and ensures the returned object has:

- `IR.values`
- `IR.sample_rate`
- `TF.values`
- `TF.frequency_bins`
- `SOFAConventions`
- `fft_length`

### Signature

```python
load_hrtf(
    path,
    mode="r",
    parallel=False,
    check_sofa_against_conventions=True,
    fft_length=None,
    mesh2hrtf_compatible=False,
    mesh2hrtf_n_shift=30,
)
```

### Parameters

#### `path`

Path to the SOFA file.

#### `mode`

SOFA/netCDF opening mode.
Use `"r"` for standard read-only workflows.
Use `"r+"` only when you intentionally need writable access to the backed SOFA dataset.

#### `parallel`

Whether the SOFA API should open the file in parallel mode.
This is an expert option.

#### `check_sofa_against_conventions`

Whether the loader should validate the file against the declared SOFA convention during loading.

Recommended default:

- `True`

#### `fft_length`

Optional FFT length used when deriving TF from HRIR content.

Use it when:

- you need a specific transform size
- you want reproducible frequency resolution across objects

#### `mesh2hrtf_compatible`

When loading `SimpleFreeFieldHRTF`, this activates Mesh2HRTF-compatible TF-to-IR reconstruction rules.

#### `mesh2hrtf_n_shift`

Optional circular shift used during Mesh2HRTF-compatible reconstruction.

### Examples

Load a standard HRIR-based SOFA file:

```python
from hrtfpykit import load_hrtf

hrtf = load_hrtf("my_hrir.sofa")
```

Load an HRTF-based SOFA file and preserve its TF bins:

```python
hrtf = load_hrtf("my_hrtf_tf.sofa")
```

Load with an explicit FFT length:

```python
hrtf = load_hrtf("my_hrir.sofa", fft_length=1024)
```

Load a Mesh2HRTF export:

```python
hrtf = load_hrtf(
    "mesh2hrtf_export.sofa",
    mesh2hrtf_compatible=True,
    mesh2hrtf_n_shift=30,
)
```

---

## The `HRTF` Object

### Role

`HRTF` is the owner object for:

- acoustic data
- source-grid state
- transformation state
- export synchronization state

It is the object you keep through the whole pipeline.

### Main attributes and views

- `Sofa`
- `SOFAConventions`
- `fft_length`
- `IR`
- `TF`
- `Sources`
- `transform`

### Main methods

- `clone()`
- `reset()`
- `is_transformed()`
- `update_sofa(...)`
- `save(...)`
- `select(...)`

### `clone()`

Create a deep in-memory copy of the current HRTF object.

Use it when:

- you want independent processing branches
- you want to compare multiple transform pipelines from the same starting point

Example:

```python
branch = hrtf.clone()
branch = branch.transform.apply_gain(-3.0, scale="db")
```

### `reset()`

Restore the in-memory object from the currently backed SOFA data.

Use it when:

- you want to discard current in-memory transforms
- you want to go back to the last backed SOFA state without reloading the file manually

Example:

```python
hrtf = hrtf.transform.apply_window("hann")
hrtf.reset()
```

### `is_transformed()`

Return whether the current in-memory state differs from the backed SOFA content.

Use it when:

- you want to know if `update_sofa()` is necessary
- you want to guard save/export logic

Example:

```python
if hrtf.is_transformed():
    hrtf.update_sofa(change_sofa_dimensions=True)
```

### `update_sofa(change_sofa_dimensions=False, sofa_convention="same")`

Synchronize current in-memory acoustic data into the backed SOFA object.

Important parameters:

- `change_sofa_dimensions`
  - allow resizing of SOFA dimensions when transformed data shape changed
- `sofa_convention`
  - `"same"`
  - `"SimpleFreeFieldHRIR"`
  - `"SimpleFreeFieldHRTF"`

Use it when:

- you want to export transformed IR/TF values into SOFA variables
- you want to convert the backed acoustic representation before saving

Example:

```python
processed.update_sofa(
    change_sofa_dimensions=True,
    sofa_convention="SimpleFreeFieldHRTF",
)
```

### `save(...)`

Write the current object to disk.

Typical parameters:

- `path`
- `overwrite`
- `change_sofa_dimensions`
- `sofa_convention`

Typical use:

```python
processed.save("processed.sofa", overwrite=True)
```

### `select(...)`

Return a selected copy of the HRTF.

This method is central because it lets you create subsets without mutating the
original object.

It supports combinations of:

- `positions`
- `position_coordinate_system`
- `plane`
- `plane_angle`
- `angle_unit`
- `ear`
- `start`
- `end`
- `start_seconds`
- `end_seconds`

Use it when:

- you want one named position such as `"front"`
- you want a whole plane
- you want a time-domain interval
- you want one ear or both ears

Examples:

Select one named position:

```python
front = hrtf.select(positions=["front"])
```

Select a horizontal plane:

```python
horizontal = hrtf.select(plane="horizontal")
```

Select one time interval:

```python
early_ir = hrtf.select(
    start_seconds=0.0,
    end_seconds=0.01,
)
```

---

## Domain Views: `IR` and `TF`

### `IR`

`IR` is the time-domain container attached to the HRTF object.

#### Main data

- `IR.values`
- `IR.sample_rate`

#### Convenience properties

- `IR.ir_length`
  - number of samples
- `IR.ir_duration`
  - duration in seconds

#### Main method

- `IR.get_itd(...)`

#### `get_itd(...)`

This is a convenience wrapper around the metric-level `itd(...)` function.

Main parameters:

- `method`
  - `"threshold"` or `"maxiacce"`
- `output`
  - `"seconds"` or `"samples"`
- `thresh_level`
- `upper_cut_freq`
- `filter_order`

Example:

```python
itd_values = hrtf.IR.get_itd(
    method="threshold",
    output="samples",
)
```

### `TF`

`TF` is the frequency-domain container attached to the HRTF object.

#### Main data

- `TF.values`
- `TF.frequency_bins`

#### Convenience properties

- `TF.tf_length`
- `TF.frequency_bins_step`
- `TF.min_frequency_bin`
- `TF.max_frequency_bin`
- `TF.magnitude`
- `TF.phase`
- `TF.real`
- `TF.imag`

#### Main method

- `TF.get_magnitude_db(reference=...)`

Example:

```python
magnitude_db = hrtf.TF.get_magnitude_db(reference="max")
```

Practical rule:

- use `IR` for time-domain operations
- use `TF` for spectral operations

---

## Source Grid API: `Sources`

`Sources` is the spatial API attached to the HRTF object.

It lets you read and query the source grid in multiple coordinate systems
without rewriting the underlying SOFA data.

### Coordinate systems

Supported coordinate systems:

- `spherical`
- `cartesian`
- `lateral-polar`

### Main methods

- `get_positions(...)`
- `get_azimuth_angles(...)`
- `get_elevation_angles(...)`
- `get_elevation_angles_for_azimuth(...)`
- `get_azimuth_angles_for_elevation(...)`
- `get_position_index(...)`

### `get_positions(angle_unit="degrees")`

Return the source grid in the current target coordinate system.

Use it when:

- you want the whole source grid
- you want positions after a coordinate-system change

Example:

```python
positions = hrtf.Sources.get_positions(angle_unit="degrees")
```

### `get_azimuth_angles(...)`

Return the unique azimuth angles in the current grid.

Example:

```python
azimuths = hrtf.Sources.get_azimuth_angles()
```

### `get_elevation_angles(...)`

Return the unique elevation angles in the current grid.

Example:

```python
elevations = hrtf.Sources.get_elevation_angles()
```

### `get_elevation_angles_for_azimuth(azimuth, angle_unit="degrees")`

Return:

- the available elevation values for the nearest real azimuth
- the matched real azimuth

Use it when:

- you want to inspect vertical coverage at one azimuth slice

Example:

```python
elevations, real_azimuth = hrtf.Sources.get_elevation_angles_for_azimuth(30.0)
```

### `get_azimuth_angles_for_elevation(elevation, angle_unit="degrees")`

Return:

- the available azimuth values for the nearest real elevation
- the matched real elevation

Example:

```python
azimuths, real_elevation = hrtf.Sources.get_azimuth_angles_for_elevation(0.0)
```

### `get_position_index(position, coordinate_system="spherical", angle_unit="degrees")`

Return:

- the matched source-grid index
- the matched real position

This method accepts:

- numeric coordinate queries
- named positions such as `"front"`, `"back"`, `"left"`, `"right"`

Example:

```python
idx, real_position = hrtf.Sources.get_position_index(
    "front",
    coordinate_system="spherical",
)
```

---

## Transform API

The transform API is exposed as:

```python
hrtf.transform
```

Each transform returns a new `HRTF` object.

### Transform philosophy

The transform layer is for object-level workflows.
If you want low-level signal utilities on arrays, use the DSP layer directly.

### Time-domain transforms

#### `apply_window(window_name)`

Apply a time-domain window to IR values and rebuild TF.

Parameter:

- `window_name`
  - example values: `"hann"`, `"hamming"`, `"blackman"`

Example:

```python
windowed = hrtf.transform.apply_window("hann")
```

#### `apply_padding(padding_length, location="end", value=0)`

Pad the IR in time and rebuild TF.

Important parameters:

- `padding_length`
- `location`
  - `"start"` or `"end"`
- `value`

Example:

```python
padded = hrtf.transform.apply_padding(64, location="end")
```

#### `upsampling(new_sample_rate)`

Upsample IR values to a higher sample rate.

Example:

```python
upsampled = hrtf.transform.upsampling(96000.0)
```

#### `downsampling(new_sample_rate)`

Downsample IR values to a lower sample rate.

Example:

```python
downsampled = hrtf.transform.downsampling(24000.0)
```

#### `apply_fir_filter(filter, cutoff=None, num_taps=101, window=None)`

Apply an FIR filter in the time domain.

Important parameters:

- `filter`
  - low-pass, high-pass, or band-pass aliases supported by the DSP layer
- `cutoff`
- `num_taps`
- `window`

Example:

```python
filtered = hrtf.transform.apply_fir_filter(
    "lowpass",
    cutoff=3000.0,
    num_taps=31,
)
```

#### `apply_iir_filter(filter, cutoff=None, order=10)`

Apply an IIR filter in the time domain.

Important parameters:

- `filter`
- `cutoff`
- `order`

Example:

```python
filtered = hrtf.transform.apply_iir_filter(
    "lowpass",
    cutoff=3000.0,
    order=4,
)
```

#### `minimum_phase(method="homomorphic", fft_length=None, epsilon=1e-12)`

Convert IR values to minimum phase and rebuild TF.

Use it when:

- you want minimum-phase approximations
- you want to standardize phase behaviour before analysis

Example:

```python
minimum_phase_hrtf = hrtf.transform.minimum_phase()
```

### HRTF-domain transforms

#### `to_ctf(weights=False, magnitude_average="log", attenuation=None)`

Convert the current HRTF into its common transfer function.

Important parameters:

- `weights`
- `magnitude_average`
  - `"log"` or `"linear"`
- `attenuation`

Example:

```python
ctf = hrtf.transform.to_ctf(weights=True, magnitude_average="linear")
```

#### `to_dtf(weights=False, magnitude_average="log", attenuation=None)`

Convert the current HRTF into its directional transfer function.

Example:

```python
dtf = hrtf.transform.to_dtf(weights=True, attenuation=20.0)
```

### Replace / edit acoustic content

#### `modify_ir(new_ir)`

Replace IR values and rebuild TF.

Accepted input types:

- `np.ndarray`
- `IR`
- `HRTF`

Example:

```python
edited_ir = hrtf.IR.values.copy()
edited_ir[..., -16:] = 0.0
gated = hrtf.transform.modify_ir(edited_ir)
```

#### `modify_tf(new_tf)`

Replace TF values and rebuild IR.

Accepted input types:

- `np.ndarray`
- `TF`
- `HRTF`

Example:

```python
edited_tf = hrtf.TF.values.copy()
edited_tf[..., -8:] *= 0.5
softened = hrtf.transform.modify_tf(edited_tf)
```

#### `modify_phase(new_phase, unit="degrees")`

Replace TF phase values and rebuild IR.

Example:

```python
zero_phase = np.zeros_like(hrtf.TF.phase)
phase_aligned = hrtf.transform.modify_phase(zero_phase)
```

#### `modify_magnitude(new_magnitude, scale="linear")`

Replace TF magnitude and rebuild IR.

Important parameters:

- `new_magnitude`
- `scale`
  - `"linear"` or `"db"`

Example:

```python
softened = hrtf.transform.modify_magnitude(
    hrtf.TF.magnitude * 0.9,
    scale="linear",
)
```

#### `apply_gain(gain, scale="db")`

Apply a gain to TF magnitude while preserving phase.

Example:

```python
quieter = hrtf.transform.apply_gain(-6.0, scale="db")
```

#### `modify_fft_length(new_fft_length)`

Change FFT length and recompute TF from the current IR.

Example:

```python
dense_tf = hrtf.transform.modify_fft_length(1024)
```

#### `modify_source_coordinate_system(coordinate_system)`

Change the target coordinate system used by `Sources`.
This does not rewrite the SOFA source coordinates. It changes how they are resolved.

Example:

```python
cartesian = hrtf.transform.modify_source_coordinate_system("cartesian")
```

#### `add_itd(itd, unit="samples")`

Add a fixed binaural delay.
Positive values delay the left ear.
Negative values delay the right ear.

Example:

```python
delayed = hrtf.transform.add_itd(4, unit="samples")
```

#### `delete_itd(method="threshold", thresh_level=-10.0, upper_cut_freq=3000.0, filter_order=10)`

Estimate existing ITD and remove it.

Use it when:

- you want to centre binaural timing
- you want an ITD-neutralized version before other processing

---

## DSP API

The DSP layer provides lower-level signal operations.

These functions are designed to accept:

- raw `np.ndarray`
- the corresponding domain objects (`IR` or `TF`) where appropriate

Use this layer when:

- you want array-level processing
- you do not need to create a derived `HRTF` object
- you want to combine library DSP functions in your own pipeline

### Core inspection functions

- `signal_duration(signal, sample_rate=None)`
- `magnitude(tf)`
- `magnitude_to_db(magnitude, reference=1.0)`
- `db_to_magnitude(magnitude_db, reference=1.0)`
- `magnitude_db(tf, reference=1.0)`
- `phase(tf, unit="degrees")`
- `real(tf)`
- `imag(tf)`

Example:

```python
from hrtfpykit.hrtf.dsp import magnitude_db

magnitude = magnitude_db(hrtf.TF, reference="max")
```

### Acoustic editing functions

- `modify_phase(...)`
- `modify_magnitude(...)`
- `tf_gain(...)`
- `window(...)`
- `padding(...)`
- `upsampling(...)`
- `downsampling(...)`
- `fir_filter(...)`
- `iir_filter(...)`
- `convolve(...)`
- `deconvolve(...)`
- `minimum_phase(...)`

### Domain conversion functions

- `tf_from_ir(...)`
- `ir_from_tf(...)`

These are the core conversion rules:

- HRIR -> HRTF uses FFT
- HRTF -> HRIR uses inverse FFT

Example:

```python
from hrtfpykit.hrtf.dsp import tf_from_ir

tf, frequency_bins, fft_length = tf_from_ir(hrtf.IR)
```

---

## Metrics API

The metrics layer provides binaural cue and spectral-difference functions.

### `itd(...)`

Estimate interaural time difference from binaural IR data.

Important parameters:

- `method`
  - `"threshold"` or `"maxiacce"`
- `sample_rate`
- `output`
  - `"seconds"` or `"samples"`
- `thresh_level`
- `upper_cut_freq`
- `filter_order`

Example:

```python
from hrtfpykit.hrtf import itd

itd_values = itd(hrtf.IR, output="samples")
```

### `ild(...)`

Estimate interaural level difference from binaural IR data.

Important parameters:

- `sample_rate`
- `fft_length`
- `mode`
  - `"broad-band"` or `"frequency-dependent"`
- `output`
  - `"db"` or `"linear"`
- `epsilon`

Example:

```python
from hrtfpykit.hrtf import ild

ild_values = ild(hrtf.IR, mode="broad-band", output="db")
```

### `itd_difference(...)`

Compute absolute per-position ITD difference between two HRTFs.

Use it when:

- you compare processing pipelines
- you measure individualization error

Example:

```python
from hrtfpykit.hrtf import itd_difference

itd_diff = itd_difference(hrtf_a, hrtf_b, output="seconds")
```

### `ild_difference(...)`

Compute absolute per-position ILD difference between two HRTFs.

Example:

```python
from hrtfpykit.hrtf import ild_difference

ild_diff = ild_difference(hrtf_a, hrtf_b, mode="broad-band")
```

### `lsd(...)`

Compute log-spectral distortion between two HRTFs in dB.

Important parameters:

- `mean_lsd`
- `ear`
  - `"left"`, `"right"`, `"both"`
- `plane`
  - `"all"`, `"horizontal"`, `"median"`
- `elevation`
- `positions`
- `frequencies`
- `reduction`
  - `"none"`, `"locations"`, `"frequencies"`, `"global"`
- `epsilon`

Examples:

Full map:

```python
from hrtfpykit.hrtf import lsd

lsd_map = lsd(
    hrtf_a,
    hrtf_b,
    ear="left",
    reduction="none",
)
```

Global scalar:

```python
lsd_scalar = lsd(
    hrtf_a,
    hrtf_b,
    ear="both",
    reduction="global",
)
```

---

## Spherical Harmonics API

The SH layer provides spherical-harmonic decomposition and reconstruction of
HRTF magnitude data.

### Main objects and functions

- `SH`
- `sht(...)`
- `sht_inverse(...)`
- `sht_error(...)`

### `SH`

`SH` is the coefficient container returned by `sht(...)`.

It stores:

- `C`
  - coefficient matrix
- `Y`
  - basis matrix
- `sh_order`
- `N`

It also exposes:

- `get_coefficients()`

### `sht(tf, sh_order, ear="left", epsilon=1e-6)`

Project HRTF magnitude data into spherical-harmonic coefficients.

Important parameters:

- `tf`
  - `TF` or `HRTF`
- `sh_order`
- `ear`
  - `"left"`, `"right"`, `"both"`
- `epsilon`

Example:

```python
from hrtfpykit.hrtf import sht

sh = sht(hrtf, sh_order=8, ear="both")
```

### `sht_inverse(sh)`

Reconstruct magnitude data from SH coefficients.

Example:

```python
from hrtfpykit.hrtf import sht_inverse

reconstructed = sht_inverse(sh)
```

### `sht_error(sh, magnitude_target)`

Evaluate SH reconstruction error against a target magnitude field.

Use SH workflows when you want to:

- compress magnitude data
- reconstruct directional fields
- evaluate approximation quality

---

## Common Pitfalls

### Confusing object-level transforms with DSP functions

Use:

- `hrtf.transform.*`
  - when you want a new HRTF object

Use:

- `hrtfpykit.hrtf.dsp.*`
  - when you want lower-level array/domain operations

These are related, but not the same workflow.

### Forgetting that `select()` returns a new object

`select()` does not mutate the original HRTF.

Wrong mental model:

- “select changes the current object”

Correct mental model:

- “select creates a selected copy”

### Forgetting `update_sofa()` before export workflows

Transforms change the in-memory object first.
If you need the backed SOFA variables updated before save, call:

```python
hrtf.update_sofa(change_sofa_dimensions=True)
```

especially when:

- IR length changed
- TF length changed
- source subset changed
- output convention changed

### Mixing raw arrays and domain objects incorrectly

Many DSP and metric functions accept both arrays and domain objects, but only
when the domain object contains the correct data.

Examples:

- `itd(...)` expects time-domain data
- `magnitude_db(...)` expects frequency-domain data
- `tf_from_ir(...)` expects IR data and a valid sample rate

### Assuming arbitrary TF bins are always valid

For `SimpleFreeFieldHRTF`, the TF loader expects one-sided positive frequency
bins that are compatible with IR reconstruction.
Malformed or incompatible layouts should fail fast.

---

## Typical End-to-End Examples

### 1. Inspect one HRTF file

```python
from hrtfpykit import load_hrtf

hrtf = load_hrtf("my_hrtf.sofa")

print(hrtf.SOFAConventions)
print(hrtf.IR.values.shape)
print(hrtf.IR.sample_rate)
print(hrtf.TF.values.shape)
print(hrtf.Sources.get_positions().shape)
```

### 2. Analyse one spatial subset

```python
horizontal = hrtf.select(plane="horizontal")
itd_curve = horizontal.IR.get_itd(output="samples")
```

### 3. Create a processed export

```python
processed = hrtf.transform.apply_window("hann")
processed = processed.transform.apply_iir_filter(
    "lowpass",
    cutoff=3000.0,
    order=4,
)
processed.update_sofa(change_sofa_dimensions=True)
processed.save("processed.sofa", overwrite=True)
```

### 4. Compare two HRTFs

```python
from hrtfpykit.hrtf import lsd, itd_difference, ild_difference

lsd_value = lsd(hrtf_a, hrtf_b, ear="both", reduction="global")
itd_diff = itd_difference(hrtf_a, hrtf_b)
ild_diff = ild_difference(hrtf_a, hrtf_b)
```

### 5. Build SH coefficients

```python
from hrtfpykit.hrtf import sht, sht_inverse

sh = sht(hrtf, sh_order=10, ear="left")
coefficients = sh.get_coefficients()
reconstructed = sht_inverse(sh)
```

### 6. Use low-level DSP directly

```python
from hrtfpykit.hrtf.dsp import magnitude_db, tf_from_ir

magnitude = magnitude_db(hrtf.TF, reference="max")
tf, frequency_bins, fft_length = tf_from_ir(hrtf.IR)
```
