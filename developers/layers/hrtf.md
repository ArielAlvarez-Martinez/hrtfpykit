# `hrtfpykit.hrtf` layer

The HRTF layer owns the central acoustic abstraction. It converts SOFA file content into synchronized time-domain, frequency-domain, and source-grid views.

## Public entry points

From `src/hrtfpykit/hrtf/__init__.py`:

- `hrtfpykit.hrtf.hrtf.HRTF`, `hrtfpykit.hrtf.hrtf.load_hrtf()`
- `hrtfpykit.utils.metrics.itd()`, `hrtfpykit.utils.metrics.ild()`, `hrtfpykit.utils.metrics.rms()`, `hrtfpykit.utils.metrics.itd_difference()`, `hrtfpykit.utils.metrics.ild_difference()`, `hrtfpykit.utils.metrics.hrtf_difference()`
- `hrtfpykit.utils.sh.SH`, `hrtfpykit.utils.sh.sht()`, `hrtfpykit.utils.sh.sht_inverse()`, `hrtfpykit.utils.sh.sht_error()`
- `hrtfpykit.utils.directivity.hrtf_from_dtf_and_ctf()`

## Main composition

`hrtfpykit.hrtf.hrtf.HRTF` is defined in `src/hrtfpykit/hrtf/hrtf.py`.

```text
hrtfpykit.hrtf.hrtf.HRTF
├── Sofa: hrtfpykit.sofa.sofa.SOFA | None
├── SOFAConventions: str | None
├── fft_length: int | None
├── mesh2hrtf_compatible: bool
├── mesh2hrtf_n_shift: int | None
├── IR: hrtfpykit.hrtf.domain.IR
├── TF: hrtfpykit.hrtf.domain.TF
├── Sources: hrtfpykit.hrtf.sources.Sources
└── transform: hrtfpykit.hrtf.transforms.Transform
```

`hrtfpykit.hrtf.hrtf.HRTF` composes these objects. It does not inherit from them.

The interface properties `hrtfpykit.hrtf.domain.IR`, `hrtfpykit.hrtf.domain.TF`, `hrtfpykit.hrtf.sources.Sources`, and `transform` are `cached_property` objects. They are created lazily and cached on the HRTF instance.

## Domain objects

`hrtfpykit.hrtf.domain.IR` and `hrtfpykit.hrtf.domain.TF` are defined in `src/hrtfpykit/hrtf/domain.py`.

`hrtfpykit.hrtf.domain.IR` owns:

- `hrtfpykit.hrtf.domain.IR.values`
- `hrtfpykit.hrtf.domain.IR.sample_rate`
- `hrtfpykit.hrtf.domain.IR.ir_length`
- `hrtfpykit.hrtf.domain.IR.ir_duration`

`hrtfpykit.hrtf.domain.TF` owns:

- `hrtfpykit.hrtf.domain.TF.values`
- `hrtfpykit.hrtf.domain.TF.frequency_bins`
- `hrtfpykit.hrtf.domain.TF.tf_length`
- `hrtfpykit.hrtf.domain.TF.frequency_bins_step`
- `hrtfpykit.hrtf.domain.TF.min_frequency_bin`
- `hrtfpykit.hrtf.domain.TF.max_frequency_bin`
- derived views: `hrtfpykit.hrtf.domain.TF.magnitude`, `hrtfpykit.hrtf.domain.TF.get_magnitude_db()`, `hrtfpykit.hrtf.domain.TF.phase`, `hrtfpykit.hrtf.domain.TF.real`, `hrtfpykit.hrtf.domain.TF.imag`

Standard shapes are:

```text
hrtfpykit.hrtf.domain.IR.values -> (positions, ears, samples)
hrtfpykit.hrtf.domain.TF.values -> (positions, ears, frequency_bins)
```

The final axis is the signal axis. Leading axes are source and ear metadata axes.

## Source-grid object

`hrtfpykit.hrtf.sources.Sources` is defined in `src/hrtfpykit/hrtf/sources.py`.

It reads SOFA `SourcePosition`, `SourcePosition:Type`, and `SourcePosition:Units` into memory. It exposes:

- `hrtfpykit.hrtf.sources.Sources.get_positions()`
- `hrtfpykit.hrtf.sources.Sources.get_azimuth_angles()`
- `hrtfpykit.hrtf.sources.Sources.get_elevation_angles()`
- `hrtfpykit.hrtf.sources.Sources.get_elevation_angles_for_azimuth()`
- `hrtfpykit.hrtf.sources.Sources.get_azimuth_angles_for_elevation()`
- `hrtfpykit.hrtf.sources.Sources.get_position_index()`

`hrtfpykit.hrtf.sources.Sources.get_positions()` supports coordinate systems:

- `spherical`
- `cartesian`
- `lateral-polar`

It also supports plane filters:

- `horizontal`
- `median`
- `frontal`

Plane selection delegates to utilities such as `hrtfpykit.utils.planes.get_horizontal_plane()`, `hrtfpykit.utils.planes.get_median_plane()`, and `hrtfpykit.utils.planes.get_frontal_plane()`.

## Load workflow

`hrtfpykit.hrtf.hrtf.load_hrtf()` supports `SimpleFreeFieldHRIR` and `SimpleFreeFieldHRTF`.

For `SimpleFreeFieldHRIR`:

- reads `Data.IR`;
- reads `Data.SamplingRate`;
- calls `hrtfpykit.utils.dsp.tf_from_ir()`;
- populates `hrtfpykit.hrtf.domain.IR.values`, `hrtfpykit.hrtf.domain.IR.sample_rate`, `hrtfpykit.hrtf.domain.TF.values`, and `hrtfpykit.hrtf.domain.TF.frequency_bins`.

For `SimpleFreeFieldHRTF`:

- reads `Data.Real`, `Data.Imag`, and `N`;
- combines real/imaginary values into complex `hrtfpykit.hrtf.domain.TF.values`;
- calls `hrtfpykit.utils.dsp.prepend_missing_dc()` when needed;
- calls `hrtfpykit.utils.dsp.ir_from_tf()`;
- populates `hrtfpykit.hrtf.domain.TF.values`, `hrtfpykit.hrtf.domain.TF.frequency_bins`, `hrtfpykit.hrtf.domain.IR.values`, and `hrtfpykit.hrtf.domain.IR.sample_rate`.

`hrtfpykit.hrtf.hrtf.load_hrtf(..., sofa_open=False)` closes the backing `hrtfpykit.sofa.sofa.SOFA` handle after loading arrays and source positions into memory.

## Transform interface

`hrtfpykit.hrtf.transforms.Transform` is defined in `src/hrtfpykit/hrtf/transforms.py` and is exposed as `hrtfpykit.hrtf.hrtf.HRTF.transform`.

Transform methods include:

- time-domain operations: `hrtfpykit.hrtf.transforms.Transform.apply_window()`, `hrtfpykit.hrtf.transforms.Transform.apply_crop()`, `hrtfpykit.hrtf.transforms.Transform.apply_padding()`, `hrtfpykit.hrtf.transforms.Transform.upsampling()`, `hrtfpykit.hrtf.transforms.Transform.downsampling()`, `hrtfpykit.hrtf.transforms.Transform.apply_fir_filter()`, `hrtfpykit.hrtf.transforms.Transform.apply_iir_filter()`, `hrtfpykit.hrtf.transforms.Transform.minimum_phase()`
- representation transforms: `hrtfpykit.hrtf.transforms.Transform.to_ctf()`, `hrtfpykit.hrtf.transforms.Transform.to_dtf()`
- direct modification: `hrtfpykit.hrtf.transforms.Transform.modify_ir()`, `hrtfpykit.hrtf.transforms.Transform.modify_phase()`, `hrtfpykit.hrtf.transforms.Transform.modify_tf()`, `hrtfpykit.hrtf.transforms.Transform.modify_magnitude()`, `hrtfpykit.hrtf.transforms.Transform.apply_gain()`, `hrtfpykit.hrtf.transforms.Transform.modify_fft_length()`
- binaural cue edits: `hrtfpykit.hrtf.transforms.Transform.add_itd()`, `hrtfpykit.hrtf.transforms.Transform.delete_itd()`, `hrtfpykit.hrtf.transforms.Transform.add_ild()`, `hrtfpykit.hrtf.transforms.Transform.delete_ild()`

The transform interface is designed around cloning. Transform methods operate on a copy from `hrtfpykit.hrtf.hrtf.HRTF.clone()` and return a derived `hrtfpykit.hrtf.hrtf.HRTF`, leaving the original object unchanged from the user perspective.

## Metrics and SH utilities

Metrics live in `src/hrtfpykit/utils/metrics.py` but are re-exported from `hrtfpykit.hrtf`:

- `hrtfpykit.utils.metrics.itd()`
- `hrtfpykit.utils.metrics.ild()`
- `hrtfpykit.utils.metrics.rms()`
- `hrtfpykit.utils.metrics.itd_difference()`
- `hrtfpykit.utils.metrics.ild_difference()`
- `hrtfpykit.utils.metrics.hrtf_difference()`

Spherical harmonic utilities live in `src/hrtfpykit/utils/sh.py` and are also re-exported:

- `hrtfpykit.utils.sh.SH`
- `hrtfpykit.utils.sh.sht()`
- `hrtfpykit.utils.sh.sht_inverse()`
- `hrtfpykit.utils.sh.sht_error()`

These are considered part of the HRTF layer because they operate on HRTF acoustic state and source grids.

## SOFA synchronization

`hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` is the synchronization boundary. It writes current in-memory acoustic state back into the backing `hrtfpykit.sofa.sofa.SOFA` object.

`hrtfpykit.hrtf.hrtf.HRTF.save()` calls `hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` and then `hrtfpykit.sofa.sofa.SOFA.save()`.

Do not make transform methods silently save files. The intended flow is:

```text
hrtfpykit.hrtf.hrtf.load_hrtf()
    ↓
hrtfpykit.hrtf.hrtf.HRTF.transform.* or hrtfpykit.hrtf.hrtf.HRTF.select()
    ↓
hrtfpykit.hrtf.hrtf.HRTF.update_sofa()
    ↓
hrtfpykit.hrtf.hrtf.HRTF.save()
```

## Logic example: load, transform, select, and save an HRTF

This example shows the normal acoustic workflow. The HRTF layer owns the
in-memory `hrtfpykit.hrtf.domain.IR`, `hrtfpykit.hrtf.domain.TF`, and `hrtfpykit.hrtf.sources.Sources` state. File persistence still goes through
the SOFA layer only when `hrtfpykit.hrtf.hrtf.HRTF.save()` is called.

```python
from hrtfpykit.hrtf import load_hrtf

hrtf = load_hrtf("subject.sofa")
delayed = hrtf.transform.apply_padding(32, location="start", ear="left")
selected = delayed.select(positions=["front", "left", "right"])
selected.save(
    "subject_left_delay_subset.sofa",
    overwrite=True,
    change_sofa_dimensions=True,
)
```

Calling flow:

```text
hrtfpykit.hrtf.hrtf.load_hrtf("subject.sofa")
    -> hrtfpykit.sofa.sofa.load_sofa(...)
    -> hrtfpykit.sofa.data._GlobalAttributes.get("SOFAConventions")

    if SOFAConventions == "SimpleFreeFieldHRIR":
        -> hrtfpykit.sofa.data._Variables.get("Data.IR")
        -> hrtfpykit.sofa.data._Variables.get("Data.SamplingRate")
        -> hrtfpykit.utils.dsp.tf_from_ir(...)
        -> hrtfpykit.hrtf.hrtf.HRTF(Sofa)
        -> hrtfpykit.hrtf.domain.IR.values
        -> hrtfpykit.hrtf.domain.IR.sample_rate
        -> hrtfpykit.hrtf.domain.TF.values
        -> hrtfpykit.hrtf.domain.TF.frequency_bins
        -> hrtfpykit.hrtf.sources.Sources
        -> return hrtfpykit.hrtf.hrtf.HRTF

    if SOFAConventions == "SimpleFreeFieldHRTF":
        -> hrtfpykit.sofa.data._Variables.get("Data.Real")
        -> hrtfpykit.sofa.data._Variables.get("Data.Imag")
        -> hrtfpykit.sofa.data._Variables.get("N")
        -> hrtfpykit.utils.dsp.prepend_missing_dc(...)
        -> hrtfpykit.utils.dsp.ir_from_tf(...)
        -> hrtfpykit.hrtf.hrtf.HRTF(Sofa)
        -> hrtfpykit.hrtf.domain.TF.values
        -> hrtfpykit.hrtf.domain.TF.frequency_bins
        -> hrtfpykit.hrtf.domain.IR.values
        -> hrtfpykit.hrtf.domain.IR.sample_rate
        -> hrtfpykit.hrtf.sources.Sources
        -> return hrtfpykit.hrtf.hrtf.HRTF

hrtf.transform.apply_padding(32, location="start", ear="left")
    -> hrtfpykit.hrtf.hrtf.HRTF.transform
    -> hrtfpykit.hrtf.transforms.Transform.apply_padding(...)
    -> hrtfpykit.hrtf.hrtf.HRTF.clone()
    -> hrtfpykit.hrtf.transforms.normalize_ear(ear)
    -> hrtfpykit.utils.dsp.padding(...) for both ears, or selected-ear padding branch
    -> hrtfpykit.utils.dsp.tf_from_ir(
           ir, fft_length=transformed_hrtf.fft_length
       )
    -> hrtfpykit.hrtf.hrtf.HRTF._transformed = True
    -> return hrtfpykit.hrtf.hrtf.HRTF

delayed.select(positions=["front", "left", "right"])
    -> hrtfpykit.hrtf.hrtf.HRTF.select(...)
    -> hrtfpykit.hrtf.hrtf.HRTF.clone()
    -> hrtfpykit.utils.coordinates.get_position_queries(positions)
    -> hrtfpykit.hrtf.sources.Sources.get_position_index(...)
       for each named position
    -> numpy.take(hrtfpykit.hrtf.domain.IR.values, selected_indices, axis=0)
    -> numpy.take(hrtfpykit.hrtf.domain.TF.values, selected_indices, axis=0)
    -> hrtfpykit.hrtf.sources.Sources._selected_indices = source_selected_indices
    -> return hrtfpykit.hrtf.hrtf.HRTF

selected.save("subject_left_delay_subset.sofa", overwrite=True, change_sofa_dimensions=True)
    -> hrtfpykit.hrtf.hrtf.HRTF.save(...)
    -> hrtfpykit.hrtf.hrtf.HRTF.update_sofa(...)
    -> hrtfpykit.sofa.sofa.SOFA.copy_with(...)
       or hrtfpykit.sofa.sofa.SOFA.modify_variable(...) depending on shape changes
    -> hrtfpykit.sofa.sofa.SOFA.save(path, overwrite=overwrite)
    -> return pathlib.Path
```

What each step does:

1. `hrtfpykit.hrtf.hrtf.load_hrtf()` calls `hrtfpykit.sofa.sofa.load_sofa()` and verifies that the loaded
   file declares an HRTF convention supported by the HRTF loader.
2. For `SimpleFreeFieldHRIR`, `hrtfpykit.hrtf.hrtf.load_hrtf()` reads `Data.IR` and
   `Data.SamplingRate`, then derives `hrtfpykit.hrtf.domain.TF.values` and `hrtfpykit.hrtf.domain.TF.frequency_bins` with
   `hrtfpykit.utils.dsp.tf_from_ir()`.
3. For `SimpleFreeFieldHRTF`, `hrtfpykit.hrtf.hrtf.load_hrtf()` reads `Data.Real`, `Data.Imag`, and
   `N`, builds complex `hrtfpykit.hrtf.domain.TF.values`, normalizes a missing DC bin with
   `hrtfpykit.utils.dsp.prepend_missing_dc()` when required, and derives `hrtfpykit.hrtf.domain.IR.values` with
   `hrtfpykit.utils.dsp.ir_from_tf()`.
4. `hrtfpykit.hrtf.hrtf.HRTF.transform` returns the lazily created `hrtfpykit.hrtf.transforms.Transform` object associated
   with the current `hrtfpykit.hrtf.hrtf.HRTF`.
5. `hrtfpykit.hrtf.transforms.Transform.apply_padding()` creates a derived `hrtfpykit.hrtf.hrtf.HRTF` with `hrtfpykit.hrtf.hrtf.HRTF.clone()`,
   applies DSP padding to the selected ear data using `padding_length`,
   `location`, and `ear`, and keeps `hrtfpykit.hrtf.domain.IR` and `hrtfpykit.hrtf.domain.TF` synchronized for the
   returned object.
6. `hrtfpykit.hrtf.hrtf.HRTF.select()` resolves named positions through `hrtfpykit.hrtf.sources.Sources.get_position_index()`
   and source-grid helpers, then slices `hrtfpykit.hrtf.domain.IR.values`, `hrtfpykit.hrtf.domain.TF.values`, and active
   `hrtfpykit.hrtf.sources.Sources` state so the first acoustic axis remains aligned.
7. `hrtfpykit.hrtf.hrtf.HRTF.save()` calls `hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` first. `hrtfpykit.hrtf.hrtf.HRTF.update_sofa()` converts
   the current in-memory acoustic state back into SOFA variables and uses
   `hrtfpykit.sofa.sofa.SOFA.copy_with()` when `change_sofa_dimensions=True` allows coordinated dimension and array replacement.
8. `hrtfpykit.sofa.sofa.SOFA.save()` performs the final file write.

The important invariant is that transforms and selections produce HRTF objects
in memory. They do not persist intermediate states. Persistence is explicit at
the `hrtfpykit.hrtf.hrtf.HRTF.save()` boundary.

## Invariants

- `hrtfpykit.hrtf.domain.IR.values.shape[0]` and `hrtfpykit.hrtf.domain.TF.values.shape[0]` should match the active source count.
- `hrtfpykit.hrtf.domain.IR.values.shape[1]` and `hrtfpykit.hrtf.domain.TF.values.shape[1]` should represent ears.
- `hrtfpykit.hrtf.domain.IR.values.shape[-1]` is samples.
- `hrtfpykit.hrtf.domain.TF.values.shape[-1]` is frequency bins.
- `hrtfpykit.hrtf.sources.Sources.get_positions()` must return rows aligned with the first acoustic axis.
- If TF changes, IR must be regenerated through `hrtfpykit.utils.dsp.ir_from_tf()` when synchronization is required.
- If IR changes, TF must be regenerated through `hrtfpykit.utils.dsp.tf_from_ir()` when synchronization is required.

## Do not do this

- Do not import `hrtfpykit.plots` into `hrtfpykit.hrtf`.
- Do not put dataset-specific filename or resource logic inside `hrtfpykit.hrtf.hrtf.HRTF` or `hrtfpykit.hrtf.transforms.Transform`.
- Do not bypass `hrtfpykit.sofa` for SOFA communication inside HRTF workflows.
- Do not add hidden persistence to transforms; saving must remain explicit through `hrtfpykit.hrtf.hrtf.HRTF.save()`.
