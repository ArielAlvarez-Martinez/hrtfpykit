from functools import cached_property
from pathlib import Path
import datetime
import importlib.metadata

import hrtfpykit.sofa
import numpy as np
from .coordinates import get_position_queries
from .dsp import (
    ir_from_tf,
    tf_from_ir,
)
from .planes import (
    get_frontal_plane,
    get_horizontal_plane,
    get_median_plane,
)
from ..plots.hrtf import HRTFPlots
from ..sofa.sofa import SOFA
from .sources import Sources
from .domain import IR, TF
from .transforms import Transform


def load_hrtf(
    path: str | Path,
    mode: str = "r",
    parallel: bool = False,
    check_sofa_against_conventions: bool = True,
    fft_length: int | None = None,
    mesh2hrtf_compatible: bool = False,
    mesh2hrtf_n_shift: int | None = 30,
) -> "HRTF":
    """Load a SOFA file as an :class:`HRTF` object.

    It loads SOFA content intothe central ``HRTF`` abstraction and guarantees 
    that both domains are available after loading:

    - ``IR`` (time domain)
    - ``TF`` (frequency domain)

    Supported conventions:

    - ``SimpleFreeFieldHRIR``: loaded from ``Data.IR`` and converted to TF.
    - ``SimpleFreeFieldHRTF``: loaded from ``Data.Real``/``Data.Imag``/``N``
      and converted to IR.

    For ``SimpleFreeFieldHRTF``, reconstruction uses positive frequency bins,
    because HRTF impulse responses are real-valued and the negative frequency
    bins are redundant. The expected format is uniformly spaced,
    non-negative, increasing bins. DC (0 Hz) should be present. If DC is
    missing and bins start at one-bin step (``Δf``), ``hrtfpykit`` prepends
    DC with value ``1+0j`` (0 dB attenuation at DC) to keep reconstruction
    consistent.

    Parameters
    ----------
    path : str | Path
        Path to the SOFA file.
    mode : str, default='r'
        File mode used by the SOFA API.
    parallel : bool, default=False
        Whether to enable parallel loading in the SOFA API.
    check_sofa_against_conventions : bool, default=True
        Whether to run convention checks when reading the SOFA file.
    fft_length : int | None, default=None
        Optional FFT length used when deriving TF from HRIR content.
    mesh2hrtf_compatible : bool, default=False
        If ``True``, use Mesh2HRTF-style TF-to-IR reconstruction when loading
        ``SimpleFreeFieldHRTF`` files.
    mesh2hrtf_n_shift : int | None, default=30
        Optional circular shift in samples applied after TF-to-IR
        reconstruction when ``mesh2hrtf_compatible=True``.

    Returns
    -------
    HRTF
        Loaded HRTF object with ``IR``, ``TF``, ``SOFAConventions``, and
        ``fft_length`` populated.

    Use Cases
    ---------
    - Load HRIR-based SOFA files and work in both domains.
    - Load HRTF-based SOFA files while preserving original frequency bins.
    - Enable Mesh2HRTF-compatible reconstruction when required by the source
      convention pipeline.

    Best Practices
    --------------
    - Keep ``check_sofa_against_conventions=True`` in production pipelines.
    - Use ``fft_length`` only when a fixed transform size is explicitly needed.
    - Fail fast on malformed SOFA variables instead of bypassing validation.
    - Keep one-sided ``N`` vectors in ``SimpleFreeFieldHRTF`` files.
    - Include DC explicitly in exported ``SimpleFreeFieldHRTF`` data whenever possible.

    Examples
    --------
    >>> from hrtfpykit import load_hrtf
    >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf.SOFAConventions
    'SimpleFreeFieldHRIR'

    >>> hrtf_tf = load_hrtf("hrtfs/HRTF_TF.sofa")
    >>> hrtf_tf.SOFAConventions
    'SimpleFreeFieldHRTF'

    >>> hrtf_m2h = load_hrtf(
    ...     "hrtfs/HRTF_ARI_44100.sofa",
    ...     mesh2hrtf_compatible=True,
    ... )
    """
    Sofa = hrtfpykit.sofa.load_sofa(
        path,
        mode=mode,
        parallel=parallel,
        check_sofa_against_conventions=check_sofa_against_conventions,
    )
    allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
    global_attrs = Sofa.GlobalAttributes
    variables = Sofa.Variables
    if global_attrs is None or variables is None:
        raise ValueError("SOFA dataset is not loaded")

    try:
        convention = global_attrs.get("SOFAConventions").value
    except ValueError:
        convention = None
    if convention not in allowed:
        raise ValueError(
            "SOFAConventions is not an HRTF convention. "
            f"Expected one of {sorted(allowed)}, got {convention!r} "
            f"for {path!s}."
        )
    variable_names = set(variables.get_names())

    if convention == "SimpleFreeFieldHRIR":
        if "Data.IR" not in variable_names:
            raise ValueError(
                "SimpleFreeFieldHRIR requires variable 'Data.IR', but it is missing."
            )
        ir = np.asarray(variables.get("Data.IR").value)
        if ir.size == 0 or np.all(ir == 0):
            raise ValueError("SimpleFreeFieldHRIR requires non empty 'Data.IR'.")
        if "Data.SamplingRate" not in variable_names:
            raise ValueError(
                "SimpleFreeFieldHRIR requires variable 'Data.SamplingRate', but it is missing."
            )
        sample_rate_data = np.asarray(
            variables.get("Data.SamplingRate").value,
            dtype=float,
        )
        if sample_rate_data.size == 0 or np.all(sample_rate_data == 0):
            raise ValueError(
                "SimpleFreeFieldHRIR requires non empty 'Data.SamplingRate'."
            )
        resolved_sample_rate = float(sample_rate_data.flat[0])
        if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
            raise ValueError(
                "SimpleFreeFieldHRIR requires a finite, positive 'Data.SamplingRate' value."
            )

        tf, frequency_bins, fft_length_used = tf_from_ir(
            ir,
            resolved_sample_rate,
            fft_length=fft_length,
        )
        hrtf = HRTF(Sofa)
        hrtf.IR.values = ir
        hrtf.IR.sample_rate = resolved_sample_rate
        hrtf.TF.values = tf
        hrtf.TF.frequency_bins = frequency_bins
        hrtf.fft_length = fft_length
        if fft_length_used is not None:
            hrtf.fft_length = fft_length_used
        hrtf.SOFAConventions = convention
        return hrtf

    required_variables = ("Data.Real", "Data.Imag", "N")
    missing_variables = [
        name for name in required_variables if name not in variable_names
    ]
    if missing_variables:
        raise ValueError(
            "SimpleFreeFieldHRTF requires variables "
            f"{required_variables}, but missing: {missing_variables}."
        )

    real = np.asarray(variables.get("Data.Real").value, dtype=float)
    if real.size == 0 or np.all(real == 0):
        raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Real'.")

    imag = np.asarray(variables.get("Data.Imag").value, dtype=float)
    if imag.size == 0 or np.all(imag == 0):
        raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Imag'.")

    frequency_bins = np.asarray(variables.get("N").value, dtype=float)
    if frequency_bins.size == 0 or np.all(frequency_bins == 0):
        raise ValueError("SimpleFreeFieldHRTF requires non empty 'N'.")

    tf = real + 1j * imag
    ir, sample_rate, fft_length_used = ir_from_tf(
        tf,
        frequency_bins=frequency_bins,
        mesh2hrtf_compatible=mesh2hrtf_compatible,
        n_shift=mesh2hrtf_n_shift,
    )
    if fft_length is not None and fft_length != fft_length_used:
        raise ValueError("FFT length does not match the provided frequency bins.")
    hrtf = HRTF(Sofa)
    hrtf.IR.values = ir
    hrtf.IR.sample_rate = sample_rate
    hrtf.TF.values = tf
    hrtf.TF.frequency_bins = frequency_bins
    hrtf.fft_length = fft_length_used
    hrtf.SOFAConventions = convention
    return hrtf


class HRTF(HRTFPlots):
    """SOFA-backed HRTF container with synchronized time and frequency domains.

    This class is the main entry point for loading, inspecting, selecting,
    transforming, plotting, and exporting HRTF/HRIR data stored in SOFA files.
    It supports ``SimpleFreeFieldHRIR`` and ``SimpleFreeFieldHRTF``
    conventions, and keeps both domain representations available and aligned:

    - ``IR``: time-domain impulse responses
    - ``TF``: frequency-domain transfer functions

    The class also exposes a spatial interface through ``Sources``. That
    interface resolves source-grid positions, coordinate-system metadata,
    geometric queries (for example named positions), and plane-based selection
    workflows used by both processing and visualization methods.

    The object exposes plotting workflows for HRTF inspection and analysis.
    This includes source-grid visualizations, spectral views, plane projections,
    and metric-oriented plots that operate directly on the current in-memory
    state (including any selection or transformation).

    Typical lifecycle
    -----------------
    1. Load an object with ``hrtfpykit.load_hrtf``.
    2. Inspect or subset data with ``Sources`` and :meth:`select`.
    3. Apply transforms through ``transform``.
    4. Visualize using plotting methods exposed by the object.
    5. Synchronize and export with :meth:`update_sofa` and :meth:`save`.

    Notes
    -----
    The instance keeps a reference to the backed SOFA object (`Sofa`) and tracks
    whether in-memory data differs from that object through ``_transformed``.
    Use :meth:`update_sofa` to synchronize in-memory data back to SOFA variables
    before persistence. This separation keeps transformation workflows explicit:
    in-memory edits can be explored, plotted, and validated first, and committed
    to SOFA only when requested.
    """

    def __init__(
        self,
        Sofa: SOFA | None = None,
    ) -> None:
        """Initialize an HRTF object.

        Parameters
        ----------
        Sofa : SOFA | None, default=None
            Backed SOFA object. When ``None``, the object is created empty and
            should be populated later.
        """
        self.Sofa: SOFA | None = Sofa
        self.SOFAConventions: str | None = None
        self.fft_length: int | None = None
        self._transformed: bool = False

    @cached_property
    def IR(self) -> "IR":
        """Time-domain representation manager."""
        return IR(self)

    @cached_property
    def TF(self) -> "TF":
        """Frequency-domain representation manager."""
        return TF(self)

    @cached_property
    def Sources(self) -> "Sources":
        """Source-grid access and selection manager."""
        return Sources(self)

    @cached_property
    def transform(self) -> "Transform":
        """Transformation API for producing derived HRTFs."""
        return Transform(self)

    def clone(self) -> "HRTF":
        """Create a deep clone of the current HRTF object.

        Returns
        -------
        HRTF
            New object with copied IR, TF, source-selection state, and metadata.

        Use Cases
        ---------
        - Branch a processing pipeline without mutating the original object.
        - Preserve current selection while testing alternative transforms.

        Examples
        --------
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf_branch = hrtf.clone()
        >>> _ = hrtf_branch.transform.apply_gain(-3.0, scale="db")

        Best Practices
        --------------
        - Clone before destructive experimentation when reproducibility matters.
        - Treat cloned and original objects as independent processing branches.
        """
        sofa_clone = self.Sofa
        if self.Sofa is not None:
            try:
                sofa_clone = self.Sofa.clone()
            except ValueError:
                sofa_clone = self.Sofa
        hrtf = HRTF(Sofa=sofa_clone)
        hrtf.SOFAConventions = self.SOFAConventions
        hrtf.fft_length = self.fft_length
        hrtf._transformed = self._transformed
        if self.IR.values is not None:
            hrtf.IR.values = np.array(self.IR.values, copy=True)
        if self.IR.sample_rate is not None:
            hrtf.IR.sample_rate = float(self.IR.sample_rate)
        if self.TF.values is not None:
            hrtf.TF.values = np.array(self.TF.values, copy=True)
        if self.TF.frequency_bins is not None:
            hrtf.TF.frequency_bins = np.array(self.TF.frequency_bins, copy=True)
        if "Sources" in self.__dict__:
            hrtf.Sources.source_coordinate_system = self.Sources.source_coordinate_system
            if self.Sources._selected_indices is not None:
                hrtf.Sources._selected_indices = np.array(
                    self.Sources._selected_indices,
                    dtype=int,
                    copy=True,
                )
        return hrtf

    def reset(self) -> "HRTF":
        """Reset in-memory HRTF data to the backed SOFA content.

        Returns
        -------
        HRTF
            Current instance after restoring IR/TF, source-state, and metadata
            from the backed SOFA object.

        Use Cases
        ---------
        - Discard all in-memory transforms and return to original data.
        - Recover a clean baseline before running a new processing pipeline.

        Examples
        --------
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf = hrtf.transform.apply_window("hann")
        >>> hrtf.is_transformed()
        True
        >>> hrtf.reset()
        HRTF(...)
        >>> hrtf.is_transformed()
        False

        Best Practices
        --------------
        - Use ``reset`` instead of reloading from disk when the same backed SOFA
          object should be preserved.
        - Run ``save`` before reset if transformed data must be kept.
        """
        if self.Sofa is None:
            raise ValueError("Cannot reset an HRTF without a loaded SOFA dataset")
        if self.Sofa.GlobalAttributes is None or self.Sofa.Variables is None:
            raise ValueError("SOFA dataset is not loaded")

        global_attrs = self.Sofa.GlobalAttributes
        variables = self.Sofa.Variables
        allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
        try:
            convention = global_attrs.get("SOFAConventions").value
        except ValueError:
            convention = None
        if convention not in allowed:
            raise ValueError(
                "SOFAConventions is not an HRTF convention. "
                f"Expected one of {sorted(allowed)}, got {convention!r}."
            )

        variable_names = set(variables.get_names())
        if convention == "SimpleFreeFieldHRIR":
            if "Data.IR" not in variable_names:
                raise ValueError(
                    "SimpleFreeFieldHRIR requires variable 'Data.IR', but it is missing."
                )
            if "Data.SamplingRate" not in variable_names:
                raise ValueError(
                    "SimpleFreeFieldHRIR requires variable 'Data.SamplingRate', but it is missing."
                )

            ir = np.asarray(variables.get("Data.IR").value)
            if ir.size == 0 or np.all(ir == 0):
                raise ValueError("SimpleFreeFieldHRIR requires non empty 'Data.IR'.")
            sample_rate_data = np.asarray(
                variables.get("Data.SamplingRate").value,
                dtype=float,
            )
            if sample_rate_data.size == 0 or np.all(sample_rate_data == 0):
                raise ValueError(
                    "SimpleFreeFieldHRIR requires non empty 'Data.SamplingRate'."
                )
            resolved_sample_rate = float(sample_rate_data.flat[0])
            if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
                raise ValueError(
                    "SimpleFreeFieldHRIR requires a finite, positive 'Data.SamplingRate' value."
                )

            tf, frequency_bins, fft_length_used = tf_from_ir(
                ir,
                resolved_sample_rate,
                fft_length=None,
            )
            self.IR.values = np.array(ir, copy=True)
            self.IR.sample_rate = resolved_sample_rate
            self.TF.values = np.array(tf, copy=True)
            self.TF.frequency_bins = np.array(frequency_bins, copy=True)
            self.fft_length = fft_length_used
        else:
            required_variables = ("Data.Real", "Data.Imag", "N")
            missing_variables = [name for name in required_variables if name not in variable_names]
            if missing_variables:
                raise ValueError(
                    "SimpleFreeFieldHRTF requires variables "
                    f"{required_variables}, but missing: {missing_variables}."
                )
            real = np.asarray(variables.get("Data.Real").value, dtype=float)
            if real.size == 0 or np.all(real == 0):
                raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Real'.")
            imag = np.asarray(variables.get("Data.Imag").value, dtype=float)
            if imag.size == 0 or np.all(imag == 0):
                raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Imag'.")
            frequency_bins = np.asarray(variables.get("N").value, dtype=float)
            if frequency_bins.size == 0 or np.all(frequency_bins == 0):
                raise ValueError("SimpleFreeFieldHRTF requires non empty 'N'.")

            tf = real + 1j * imag
            ir, sample_rate, fft_length_used = ir_from_tf(
                tf,
                frequency_bins=frequency_bins,
            )
            self.IR.values = np.array(ir, copy=True)
            self.IR.sample_rate = float(sample_rate)
            self.TF.values = np.array(tf, copy=True)
            self.TF.frequency_bins = np.array(frequency_bins, copy=True)
            self.fft_length = fft_length_used

        if "Sources" in self.__dict__:
            self.Sources.source_coordinate_system = (
                self.Sofa.VariableAttributes.get("SourcePosition:Type").value
            )
            self.Sources._selected_indices = None
        self.SOFAConventions = convention
        self._transformed = False
        return self

    def is_transformed(self) -> bool:
        """Return whether the in-memory object differs from backed SOFA data.

        Returns
        -------
        bool
            ``True`` if at least one transformation or selection modified current
            in-memory data; ``False`` otherwise.

        Use Cases
        ---------
        - Check if :meth:`update_sofa` is needed before saving.
        - Build guard logic in processing pipelines.

        Examples
        --------
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.is_transformed()
        False
        >>> transformed = hrtf.transform.apply_window("hann")
        >>> transformed.is_transformed()
        True

        Best Practices
        --------------
        - Use this check before expensive SOFA synchronization workflows.
        """
        return self._transformed

    def update_sofa(
        self,
        change_sofa_dimensions: bool = False,
        sofa_convention: str = "same",
    ) -> None:
        """Synchronize in-memory IR/TF data into the backed SOFA object.

        Parameters
        ----------
        change_sofa_dimensions : bool, default=False
            If ``True``, allows resizing fixed SOFA dimensions when transformed
            data shape differs from backed variables.
        sofa_convention : {'same', 'SimpleFreeFieldHRIR', 'SimpleFreeFieldHRTF'}, default='same'
            Output SOFA convention to enforce during synchronization. ``'same'``
            keeps the original backed SOFA convention.

        Returns
        -------
        None
            This method updates ``self.Sofa`` in-place and does not return data.

        Use Cases
        ---------
        - Persist transformed HRTF/HRIR values into SOFA variables before save.
        - Export the current object in HRIR or HRTF SOFA convention.
        - Commit selected-position subsets into a resized SOFA structure.

        Examples
        --------
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> transformed = hrtf.transform.apply_padding(padding_length=16, location="end")
        >>> transformed.update_sofa(change_sofa_dimensions=True)
        >>> transformed.save("hrtfs/padded.sofa", overwrite=True)

        >>> selected = hrtf.select(positions=["front", "left", "right"])
        >>> selected.update_sofa(
        ...     change_sofa_dimensions=True,
        ...     sofa_convention="SimpleFreeFieldHRTF",
        ... )

        Best Practices
        --------------
        - Keep ``change_sofa_dimensions=False`` unless a shape change is expected.
        - Use ``sofa_convention='same'`` for metadata-preserving updates.
        - Use explicit convention switching only for deliberate export workflows.
        """
        if self.Sofa is None or self.Sofa.netCDF4_dataset is None:
            raise ValueError("SOFA dataset is not loaded")

        if self.Sofa.GlobalAttributes is None or self.Sofa.Variables is None:
            raise ValueError("SOFA dataset is not loaded")
        try:
            backed_convention = self.Sofa.GlobalAttributes.get("SOFAConventions").value
        except ValueError:
            backed_convention = self.SOFAConventions
        if backed_convention not in {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}:
            raise ValueError(
                "SOFAConventions is not an HRTF convention. "
                f"Expected one of ['SimpleFreeFieldHRIR', 'SimpleFreeFieldHRTF'], got {backed_convention!r}."
            )
        resolved_sofa_convention = str(sofa_convention).strip()
        if resolved_sofa_convention == "same":
            resolved_sofa_convention = backed_convention
        if resolved_sofa_convention not in {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}:
            raise ValueError(
                "sofa_convention must be one of: same, SimpleFreeFieldHRIR, SimpleFreeFieldHRTF"
            )
        selected_indices: np.ndarray | None = None
        if "Sources" in self.__dict__:
            selected_indices = self.Sources._selected_indices
        has_selected_subset = selected_indices is not None
        if (
            not self._transformed
            and resolved_sofa_convention == backed_convention
            and not has_selected_subset
        ):
            print("HRTF is not transformed. SOFA-backed object is already up to date.")
            return

        if resolved_sofa_convention == "SimpleFreeFieldHRIR":
            if self.IR.values is not None and self.IR.sample_rate is not None:
                resolved_ir_values = np.asarray(self.IR.values)
                resolved_sample_rate = float(self.IR.sample_rate)
            else:
                if self.TF.values is None:
                    raise ValueError("IR values are not available and TF values are not available")
                if self.TF.frequency_bins is None:
                    raise ValueError("TF frequency_bins are not available")
                resolved_ir_values, resolved_sample_rate, _ = ir_from_tf(
                    np.asarray(self.TF.values),
                    frequency_bins=np.asarray(self.TF.frequency_bins, dtype=float),
                )
            target_variables: dict[str, np.ndarray] = {
                "Data.IR": np.asarray(resolved_ir_values),
                "Data.SamplingRate": np.asarray(resolved_sample_rate, dtype=float),
            }
        else:
            if self.TF.values is not None and self.TF.frequency_bins is not None:
                resolved_tf_values = np.asarray(self.TF.values)
                resolved_frequency_bins = np.asarray(self.TF.frequency_bins, dtype=float)
            else:
                if self.IR.values is None:
                    raise ValueError("TF values are not available and IR values are not available")
                if self.IR.sample_rate is None:
                    raise ValueError("IR sample_rate is required to derive TF values")
                resolved_tf_values, resolved_frequency_bins, _ = tf_from_ir(
                    np.asarray(self.IR.values),
                    float(self.IR.sample_rate),
                    fft_length=self.fft_length,
                )
            target_variables = {
                "Data.Real": np.asarray(np.real(resolved_tf_values), dtype=float),
                "Data.Imag": np.asarray(np.imag(resolved_tf_values), dtype=float),
                "N": np.asarray(resolved_frequency_bins, dtype=float),
            }

        if resolved_sofa_convention == "SimpleFreeFieldHRIR":
            obsolete_variables = ("Data.Real", "Data.Imag", "N")
        else:
            obsolete_variables = ("Data.IR", "Data.SamplingRate")
        if resolved_sofa_convention != backed_convention:
            working_sofa = self.Sofa.clone()
            working_dataset = working_sofa.netCDF4_dataset
            if working_dataset is None:
                raise ValueError("SOFA dataset is not loaded")
            for obsolete_variable in obsolete_variables:
                if obsolete_variable in working_dataset.variables:
                    working_sofa.delete_variable(obsolete_variable)
        else:
            working_sofa = self.Sofa

        dataset = working_sofa.netCDF4_dataset
        if dataset is None:
            raise ValueError("SOFA dataset is not loaded")
        existing_target_variables = [
            variable_name
            for variable_name in target_variables
            if variable_name in dataset.variables
        ]
        missing_target_variables = [
            variable_name
            for variable_name in target_variables
            if variable_name not in dataset.variables
        ]

        missing_variable_dimensions: dict[str, tuple[str, ...]] = {}
        dimension_overrides: dict[str, int] = {}
        for variable_name in missing_target_variables:
            target_values = np.asarray(target_variables[variable_name])
            if variable_name == "N":
                variable_dimensions = ("N",)
            elif variable_name == "Data.SamplingRate":
                variable_dimensions = ("I",)
            else:
                if "Data.IR" in dataset.variables:
                    variable_dimensions = tuple(dataset.variables["Data.IR"].dimensions)
                elif "Data.Real" in dataset.variables:
                    variable_dimensions = tuple(dataset.variables["Data.Real"].dimensions)
                elif "Data.Imag" in dataset.variables:
                    variable_dimensions = tuple(dataset.variables["Data.Imag"].dimensions)
                else:
                    variable_dimensions = ("M", "R", "N")
            missing_variable_dimensions[variable_name] = variable_dimensions
            if target_values.ndim > len(variable_dimensions):
                raise ValueError(
                    f"Variable '{variable_name}' has incompatible rank for SOFA dimensions"
                )
            aligned_target_shape = (
                (1,) * (len(variable_dimensions) - target_values.ndim)
                + tuple(target_values.shape)
            )
            for axis_index, dimension_name in enumerate(variable_dimensions):
                if dimension_name not in dataset.dimensions:
                    continue
                dimension = dataset.dimensions[dimension_name]
                if dimension.isunlimited():
                    continue
                target_size = int(aligned_target_shape[axis_index])
                if target_size == 1:
                    continue
                current_size = int(dimension.size)
                if current_size == target_size:
                    continue
                if dimension_name in dimension_overrides:
                    if dimension_overrides[dimension_name] != target_size:
                        raise ValueError(
                            f"Conflicting target size for dimension '{dimension_name}'"
                        )
                else:
                    dimension_overrides[dimension_name] = target_size

        mismatched_variables: list[str] = []
        for variable_name in existing_target_variables:
            target_values = target_variables[variable_name]
            variable_shape = tuple(dataset.variables[variable_name].shape)
            target_array = np.asarray(target_values)
            if target_array.ndim > len(variable_shape):
                mismatched_variables.append(variable_name)
                continue
            try:
                np.broadcast_to(target_array, variable_shape)
            except ValueError:
                mismatched_variables.append(variable_name)
        if (mismatched_variables or dimension_overrides) and not change_sofa_dimensions:
            mismatch_text = ", ".join(mismatched_variables)
            raise ValueError(
                "Transformed HRTF dimensions differ from SOFA-backed variables "
                f"({mismatch_text}). Set change_sofa_dimensions=True to allow resizing."
            )

        updated_sofa: SOFA
        if not mismatched_variables and not dimension_overrides:
            updated_sofa = working_sofa.clone()
            for variable_name in existing_target_variables:
                target_values = target_variables[variable_name]
                updated_sofa.modify_variable(variable_name, target_values)
        else:
            for variable_name in mismatched_variables:
                variable = dataset.variables[variable_name]
                target_array = np.asarray(target_variables[variable_name])
                if target_array.ndim > len(variable.dimensions):
                    raise ValueError(
                        f"Variable '{variable_name}' has incompatible rank for SOFA dimensions"
                    )
                target_shape = (
                    (1,) * (len(variable.dimensions) - target_array.ndim)
                    + tuple(target_array.shape)
                )
                for axis_index, dimension_name in enumerate(variable.dimensions):
                    dimension = dataset.dimensions[dimension_name]
                    if dimension.isunlimited():
                        continue
                    target_size = int(target_shape[axis_index])
                    if target_size == 1:
                        continue
                    if dimension_name in dimension_overrides:
                        if dimension_overrides[dimension_name] != target_size:
                            raise ValueError(
                                f"Conflicting target size for dimension '{dimension_name}'"
                            )
                    elif int(dimension.size) != target_size:
                        dimension_overrides[dimension_name] = target_size

            if dimension_overrides:
                overridden_dimension_names = set(dimension_overrides)
                dependent_variable_overrides: dict[str, np.ndarray] = {}
                for variable_name, variable in dataset.variables.items():
                    if variable_name in target_variables:
                        continue
                    affected_dimensions = tuple(
                        dimension_name
                        for dimension_name in variable.dimensions
                        if dimension_name in overridden_dimension_names
                    )
                    if not affected_dimensions:
                        continue
                    if (
                        len(affected_dimensions) == 1
                        and affected_dimensions[0] == "M"
                        and selected_indices is not None
                    ):
                        variable_values = np.asarray(variable[:])
                        axis_index = variable.dimensions.index("M")
                        variable_values = np.take(
                            variable_values,
                            np.asarray(selected_indices, dtype=int),
                            axis=axis_index,
                        )
                        dependent_variable_overrides[variable_name] = variable_values
                        continue
                    if any(
                        dimension_name in overridden_dimension_names
                        for dimension_name in variable.dimensions
                    ):
                        raise ValueError(
                            "Cannot resize SOFA dimensions because dependent variable "
                            f"'{variable_name}' is not handled by update_sofa"
                        )
                updated_sofa = working_sofa.copy_with(
                    dim_sizes=dimension_overrides,
                    variables={
                        variable_name: target_variables[variable_name]
                        for variable_name in existing_target_variables
                    }
                    | dependent_variable_overrides,
                )
            else:
                updated_sofa = working_sofa.clone()
                for variable_name in existing_target_variables:
                    target_values = target_variables[variable_name]
                    updated_sofa.modify_variable(variable_name, target_values)

        updated_dataset = updated_sofa.netCDF4_dataset
        if updated_dataset is None:
            raise ValueError("SOFA dataset is not loaded")
        for variable_name in missing_target_variables:
            target_values = np.asarray(target_variables[variable_name])
            variable_dimensions = missing_variable_dimensions[variable_name]

            if target_values.ndim > len(variable_dimensions):
                raise ValueError(
                    f"Variable '{variable_name}' has incompatible rank for SOFA dimensions"
                )
            aligned_target_shape = (
                (1,) * (len(variable_dimensions) - target_values.ndim)
                + tuple(target_values.shape)
            )
            for axis_index, dimension_name in enumerate(variable_dimensions):
                if dimension_name not in updated_dataset.dimensions:
                    if not change_sofa_dimensions:
                        raise ValueError(
                            f"Dimension '{dimension_name}' is missing in SOFA dataset. "
                            "Set change_sofa_dimensions=True to allow creating missing dimensions."
                        )
                    updated_sofa.create_dimension(
                        dimension_name,
                        int(aligned_target_shape[axis_index]),
                    )
                    continue
                dimension = updated_dataset.dimensions[dimension_name]
                if dimension.isunlimited():
                    continue
                current_size = int(dimension.size)
                target_size = int(aligned_target_shape[axis_index])
                if target_size in {1, current_size}:
                    continue
                raise ValueError(
                    f"Cannot create variable '{variable_name}': dimension '{dimension_name}' "
                    f"size mismatch ({current_size} != {target_size})"
                )

            variable_attributes: dict[str, str] | None = None
            if variable_name == "Data.SamplingRate":
                variable_attributes = {"Units": "hertz"}
            elif variable_name == "N":
                variable_attributes = {
                    "Units": "hertz",
                    "LongName": "frequency",
                }
            updated_sofa.create_variable(
                name=variable_name,
                data=target_values,
                dimensions=variable_dimensions,
                attributes=variable_attributes,
            )

        for obsolete_variable in obsolete_variables:
            if obsolete_variable in updated_dataset.variables:
                updated_sofa.delete_variable(obsolete_variable)

        updated_sofa.path = self.Sofa.path
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            hrtfpykit_version = importlib.metadata.version("hrtfpykit")
        except importlib.metadata.PackageNotFoundError:
            hrtfpykit_version = "unknown"
        existing_date_created = None
        try:
            existing_date_created = updated_sofa.GlobalAttributes.get("DateCreated").value
        except ValueError:
            existing_date_created = None
        if isinstance(existing_date_created, str) and existing_date_created.strip() == "":
            existing_date_created = None
        resolved_global_attributes = {
            "SOFAConventions": resolved_sofa_convention,
            "DataType": (
                "FIR"
                if resolved_sofa_convention == "SimpleFreeFieldHRIR"
                else "TF"
            ),
            "APIName": "hrtfpykit-sofa",
            "APIVersion": hrtfpykit.sofa.__version__,
            "ApplicationName": "hrtfpykit",
            "ApplicationVersion": hrtfpykit_version,
            "DateModified": now,
        }
        if existing_date_created is None:
            resolved_global_attributes["DateCreated"] = now
        for attribute_name, attribute_value in resolved_global_attributes.items():
            try:
                updated_sofa.modify_global_attribute(attribute_name, attribute_value)
            except ValueError:
                updated_sofa.create_global_attribute(attribute_name, attribute_value)
        self.Sofa = updated_sofa
        self.SOFAConventions = resolved_sofa_convention
        return

    def save(
        self,
        path: str | Path | None = None,
        overwrite: bool = False,
        change_sofa_dimensions: bool = False,
        sofa_convention: str = "same",
    ) -> Path:
        """Update SOFA variables and save to disk.

        Parameters
        ----------
        path : str | Path | None, default=None
            Output file path. If ``None``, the backed SOFA path is used.
        overwrite : bool, default=False
            If ``True``, allows overwriting an existing file.
        change_sofa_dimensions : bool, default=False
            Forwarded to :meth:`update_sofa` to control SOFA dimension resizing.
        sofa_convention : {'same', 'SimpleFreeFieldHRIR', 'SimpleFreeFieldHRTF'}, default='same'
            Forwarded to :meth:`update_sofa` to select output convention.

        Returns
        -------
        Path
            Path to the saved SOFA file.

        Use Cases
        ---------
        - Export transformed HRTF files.
        - Save selected subsets to standalone SOFA files.

        Examples
        --------
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> selected = hrtf.select(positions=["front", "left", "right"])
        >>> selected.save(
        ...     path="hrtfs/selected_front_left_right.sofa",
        ...     overwrite=True,
        ...     change_sofa_dimensions=True,
        ... )

        Best Practices
        --------------
        - For position subsets or FFT-length changes, enable
          ``change_sofa_dimensions=True``.
        - Keep convention explicit when creating deliverables for external tools.
        """
        if self.Sofa is None:
            raise ValueError("SOFA dataset is not loaded")
        self.update_sofa(
            change_sofa_dimensions=change_sofa_dimensions,
            sofa_convention=sofa_convention,
        )
        return self.Sofa.save(path=path, overwrite=overwrite)

    def select(
        self,
        positions: np.ndarray | list[list[float]] | list[float] | None = None,
        position_coordinate_system: str = "spherical",
        plane: str | None = None,
        plane_angle: float = 0.0,
        ear: str = "both",
        angle_unit: str = "degrees",
        start: int | None = None,
        end: int | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> "HRTF":
        """Select a spatial subset, ear subset, and/or IR crop from the HRTF.

        Parameters
        ----------
        positions : np.ndarray | list[list[float]] | list[float] | None, default=None
            Explicit positions or named aliases to select.
        position_coordinate_system : str, default='spherical'
            Coordinate system used by ``positions``.
        plane : str | None, default=None
            Plane name to filter positions. Supported values are
            ``horizontal``, ``median``, and ``frontal``.
        plane_angle : float, default=0.0
            Plane angle for the selected ``plane``.
        ear : {'both', 'left', 'right'}, default='both'
            Ear selection.
        angle_unit : str, default='degrees'
            Angle unit used for spatial queries.
        start : int | None, default=None
            IR crop start index (samples).
        end : int | None, default=None
            IR crop end index (samples).
        start_seconds : float | None, default=None
            IR crop start time (seconds). Mutually exclusive with ``start``.
        end_seconds : float | None, default=None
            IR crop end time (seconds). Mutually exclusive with ``end``.

        Returns
        -------
        HRTF
            New HRTF object containing the selected subset.

        Use Cases
        ---------
        - Isolate a spatial plane for analysis and plotting.
        - Build single-ear datasets.
        - Crop impulse responses for latency-window studies.

        Examples
        --------
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> horizontal = hrtf.select(plane="horizontal", plane_angle=0.0, ear="left")
        >>> front_slice = hrtf.select(positions=["front"], start_seconds=0.0, end_seconds=0.01)

        Best Practices
        --------------
        - Use either index-based crop (``start/end``) or time-based crop
          (``start_seconds/end_seconds``), never both.
        - Prefer ``plane`` selection for reproducible geometric subsets.
        - Keep ``ear='both'`` unless unilateral analysis is required.
        """
        transformed_hrtf = self.clone()
        selected_indices: np.ndarray | None = None
        ear_key = str(ear).strip().lower()
        if ear_key not in {"both", "left", "right"}:
            raise ValueError("ear must be one of: both, left, right")

        selecting_spatial = positions is not None or plane is not None
        if selecting_spatial:
            if transformed_hrtf.Sofa is None:
                raise ValueError("Spatial selection requires a loaded SOFA dataset")
            current_source_indices = transformed_hrtf.Sources._selected_indices
            source_positions = transformed_hrtf.Sources.get_positions(angle_unit=angle_unit)
            if source_positions.ndim != 2 or source_positions.shape[-1] != 3:
                raise ValueError("Source positions grid must have shape (N, 3)")
            source_count = int(source_positions.shape[0])

            if positions is not None:
                position_indices: list[int] = []
                for position in get_position_queries(positions):
                    idx, _ = transformed_hrtf.Sources.get_position_index(
                        position=position,
                        coordinate_system=position_coordinate_system,
                        angle_unit=angle_unit,
                    )
                    if idx not in position_indices:
                        position_indices.append(int(idx))
                selected_indices = np.asarray(position_indices, dtype=int)

            if plane is not None:
                plane_key = str(plane).strip().lower()
                if plane_key == "horizontal":
                    plane_indices, _ = get_horizontal_plane(
                        hrtf=transformed_hrtf,
                        elevation=plane_angle,
                        angle_unit=angle_unit,
                    )
                elif plane_key == "median":
                    plane_indices, _ = get_median_plane(
                        hrtf=transformed_hrtf,
                        azimuth=plane_angle,
                        angle_unit=angle_unit,
                    )
                elif plane_key == "frontal":
                    plane_indices, _ = get_frontal_plane(
                        hrtf=transformed_hrtf,
                        azimuth=plane_angle,
                        angle_unit=angle_unit,
                    )
                else:
                    raise ValueError("plane must be one of: horizontal, median, frontal")
                plane_indices = np.asarray(plane_indices, dtype=int)
                if selected_indices is None:
                    selected_indices = plane_indices
                else:
                    selected_indices = np.intersect1d(selected_indices, plane_indices)

            if selected_indices is None:
                selected_indices = np.arange(source_count, dtype=int)
            if selected_indices.size == 0:
                raise ValueError("Selection produced no source positions")

            if current_source_indices is None:
                source_selected_indices = np.asarray(selected_indices, dtype=int)
            else:
                source_selected_indices = np.take(
                    np.asarray(current_source_indices, dtype=int),
                    np.asarray(selected_indices, dtype=int),
                    axis=0,
                )
            transformed_hrtf.Sources._selected_indices = source_selected_indices

            if transformed_hrtf.IR.values is not None:
                transformed_hrtf.IR.values = np.take(
                    transformed_hrtf.IR.values,
                    selected_indices,
                    axis=0,
                )
            if transformed_hrtf.TF.values is not None:
                transformed_hrtf.TF.values = np.take(
                    transformed_hrtf.TF.values,
                    selected_indices,
                    axis=0,
                )

        cropping_ir = (
            start is not None
            or end is not None
            or start_seconds is not None
            or end_seconds is not None
        )
        if cropping_ir:
            if transformed_hrtf.IR.values is None:
                raise ValueError("IR data is not available")
            ir_values = transformed_hrtf.IR.values
            if not isinstance(ir_values, np.ndarray):
                raise ValueError("IR data must be a NumPy array")
            if ir_values.ndim == 0:
                raise ValueError("IR data must have at least one dimension")

            using_sample_indices = start is not None or end is not None
            using_seconds = start_seconds is not None or end_seconds is not None
            if using_sample_indices and using_seconds:
                raise ValueError(
                    "Use either sample indices (start/end) or seconds (start_seconds/end_seconds)"
                )

            start_index = start
            end_index = end
            if using_seconds:
                if transformed_hrtf.IR.sample_rate is None:
                    raise ValueError("sample_rate is required when using seconds crop")
                resolved_sample_rate = transformed_hrtf.IR.sample_rate
                if isinstance(resolved_sample_rate, bool):
                    raise ValueError("sample_rate must be a finite, positive value.")
                try:
                    resolved_sample_rate = float(resolved_sample_rate)
                except (TypeError, ValueError):
                    raise ValueError("sample_rate must be a finite, positive value.") from None
                if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
                    raise ValueError("sample_rate must be a finite, positive value.")
                if start_seconds is not None:
                    if isinstance(start_seconds, bool):
                        raise ValueError("start_seconds must be a finite, non-negative value.")
                    try:
                        start_seconds = float(start_seconds)
                    except (TypeError, ValueError):
                        raise ValueError("start_seconds must be a finite, non-negative value.") from None
                    if not np.isfinite(start_seconds) or start_seconds < 0.0:
                        raise ValueError("start_seconds must be a finite, non-negative value.")
                    start_index = int(round(start_seconds * resolved_sample_rate))
                else:
                    start_index = None
                if end_seconds is not None:
                    if isinstance(end_seconds, bool):
                        raise ValueError("end_seconds must be a finite, non-negative value.")
                    try:
                        end_seconds = float(end_seconds)
                    except (TypeError, ValueError):
                        raise ValueError("end_seconds must be a finite, non-negative value.") from None
                    if not np.isfinite(end_seconds) or end_seconds < 0.0:
                        raise ValueError("end_seconds must be a finite, non-negative value.")
                    end_index = int(round(end_seconds * resolved_sample_rate))
                else:
                    end_index = None
            else:
                if start is not None:
                    if isinstance(start, bool) or not isinstance(start, int):
                        raise ValueError("start must be an integer")
                    if start < 0:
                        raise ValueError("start must be non-negative")
                if end is not None:
                    if isinstance(end, bool) or not isinstance(end, int):
                        raise ValueError("end must be an integer")
                    if end < 0:
                        raise ValueError("end must be non-negative")

            if start_index is not None and end_index is not None and start_index >= end_index:
                raise ValueError("Crop end must be greater than crop start")

            transformed_hrtf.IR.values = ir_values[..., slice(start_index, end_index)]
            tf_from_ir(
                transformed_hrtf.IR,
                fft_length=transformed_hrtf.fft_length,
            )

        if ear_key != "both":
            ear_index = 0 if ear_key == "left" else 1
            if transformed_hrtf.IR.values is not None:
                if transformed_hrtf.IR.values.shape[-2] <= ear_index:
                    raise ValueError(f"Requested ear '{ear_key}' is not available in IR data")
                transformed_hrtf.IR.values = np.take(
                    transformed_hrtf.IR.values,
                    indices=ear_index,
                    axis=-2,
                )
            if transformed_hrtf.TF.values is not None:
                if transformed_hrtf.TF.values.shape[-2] <= ear_index:
                    raise ValueError(f"Requested ear '{ear_key}' is not available in TF data")
                transformed_hrtf.TF.values = np.take(
                    transformed_hrtf.TF.values,
                    indices=ear_index,
                    axis=-2,
                )

        return transformed_hrtf
