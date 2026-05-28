from functools import cached_property
from pathlib import Path
from typing import Any, cast

import hrtfpykit.sofa
import numpy as np
from ..utils.coordinates import get_position_queries, get_spherical_positions
from ..utils.dsp import (
    ir_from_tf,
    prepend_missing_dc,
    tf_from_ir,
)
from ..utils.planes import (
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
    """Load a SOFA file as an :class:`~hrtfpykit.hrtf.HRTF` object.

    This function is the public loader for SOFA-based HRTF workflows in
    hrtfpykit. It reads the file through the package SOFA API, verifies
    that the declared SOFA convention is an HRTF convention, and populates the
    central :class:`~hrtfpykit.hrtf.HRTF` abstraction with synchronized
    time- and frequency-domain data. Loaded objects keep the original SOFA
    handle in :attr:`~hrtfpykit.hrtf.HRTF.Sofa` while exposing NumPy
    arrays through :attr:`~hrtfpykit.hrtf.HRTF.IR` and
    :attr:`~hrtfpykit.hrtf.HRTF.TF` for processing, plotting, selection,
    and export.

    - :class:`~hrtfpykit.hrtf.domain.IR` (time domain)
    - :class:`~hrtfpykit.hrtf.domain.TF` (frequency domain)

    Supported conventions:

    - SimpleFreeFieldHRIR: loaded from ``Data.IR`` and converted to TF.
    - SimpleFreeFieldHRTF: loaded from ``Data.Real``, ``Data.Imag``, and ``N``
      and converted to IR.

    HRIR files derive :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` and
    :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` with a
    real FFT. HRTF files rebuild
    :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` with inverse real FFT
    from the one-sided positive-frequency representation. For
    SimpleFreeFieldHRTF input,
    frequency bins must be non-negative, increasing, and uniformly spaced. DC
    (0 Hz) should be present. If DC is missing and bins start at one-bin step
    (Delta f), hrtfpykit prepends DC with value 1+0j to keep
    reconstruction consistent. The normalized TF is stored on
    :attr:`~hrtfpykit.hrtf.HRTF.TF`, so
    :meth:`~hrtfpykit.hrtf.HRTF.update_sofa` and
    :meth:`~hrtfpykit.hrtf.HRTF.save` write the DC bin back when the object is
    synchronized.

    Mesh2HRTF compatible reconstruction options are stored on the returned
    object. They are reused by :meth:`~hrtfpykit.hrtf.HRTF.reset`, preserved by
    :meth:`~hrtfpykit.hrtf.HRTF.clone`, and forwarded by TF-domain workflows
    that rebuild HRIR data from the current HRTF values.

    Parameters
    ----------
    path : str | Path
        Path to the SOFA file.
    mode : str, default=``r``
        File mode used by the SOFA API.
    parallel : bool, default=False
        Whether to enable parallel loading in the SOFA API.
    check_sofa_against_conventions : bool, default=True
        Whether to run convention checks when reading the SOFA file.
    fft_length : int | None, default=None
        Optional FFT length used when deriving TF from HRIR content. For HRTF
        files, a provided value must match the FFT length implied by
        N/frequency bins.
    mesh2hrtf_compatible : bool, default=False
        If True, use Mesh2HRTF-style TF-to-IR reconstruction when loading
        SimpleFreeFieldHRTF files. The selected value is stored on the returned
        object and reused by reset and transform workflows.
    mesh2hrtf_n_shift : int | None, default=30
        Optional circular shift in samples applied after TF-to-IR
        reconstruction when mesh2hrtf_compatible=True. The selected value is
        stored on the returned object.

    Returns
    -------
    HRTF
        Loaded :class:`~hrtfpykit.hrtf.HRTF` object with
        :class:`~hrtfpykit.hrtf.domain.IR`,
        :class:`~hrtfpykit.hrtf.domain.TF`,
        :attr:`~hrtfpykit.hrtf.HRTF.SOFAConventions`, and
        :attr:`~hrtfpykit.hrtf.HRTF.fft_length` populated. For
        SimpleFreeFieldHRTF input, Mesh2HRTF compatible load options are also
        stored on the object.

    Raises
    ------
    ValueError
        If the SOFA file is unavailable, declares an unsupported convention,
        omits required SOFA variables, contains empty acoustic data, has an
        invalid sample rate, or provides frequency bins incompatible with the
        requested FFT length.

    Examples
    --------
    Load a SimpleFreeFieldHRIR convention SOFA file and inspect the synchronized
    time- and frequency-domain views:

    >>> from hrtfpykit.hrtf import load_hrtf
    >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf.SOFAConventions
    'SimpleFreeFieldHRIR'
    >>> hrtf.IR.values.shape
    (793, 2, 256)
    >>> hrtf.TF.values.shape
    (793, 2, 129)
    >>> hrtf.IR.sample_rate
    44100.0

    Load a SimpleFreeFieldHRTF file with Mesh2HRTF compatible reconstruction
    and keep those reconstruction settings available for reset and later TF
    workflows:

    >>> hrtf = load_hrtf(
    ...     "hrtfs/HRTF_ARI_44100.sofa",
    ...     mesh2hrtf_compatible=True,
    ...     mesh2hrtf_n_shift=30,
    ... )
    >>> hrtf.mesh2hrtf_compatible
    True
    >>> hrtf.mesh2hrtf_n_shift
    30
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
        convention = cast(Any, global_attrs.get("SOFAConventions")).value
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
        ir = np.asarray(cast(Any, variables.get("Data.IR")).value)
        if ir.size == 0 or np.all(ir == 0):
            raise ValueError("SimpleFreeFieldHRIR requires non empty 'Data.IR'.")
        if "Data.SamplingRate" not in variable_names:
            raise ValueError(
                "SimpleFreeFieldHRIR requires variable 'Data.SamplingRate', but it is missing."
            )
        sample_rate_data = np.asarray(
            cast(Any, variables.get("Data.SamplingRate")).value,
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

    real = np.asarray(cast(Any, variables.get("Data.Real")).value, dtype=float)
    if real.size == 0 or np.all(real == 0):
        raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Real'.")

    imag = np.asarray(cast(Any, variables.get("Data.Imag")).value, dtype=float)
    if imag.size == 0 or np.all(imag == 0):
        raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Imag'.")

    frequency_bins = np.asarray(cast(Any, variables.get("N")).value, dtype=float)
    if frequency_bins.size == 0 or np.all(frequency_bins == 0):
        raise ValueError("SimpleFreeFieldHRTF requires non empty 'N'.")

    tf = real + 1j * imag
    tf, frequency_bins = prepend_missing_dc(tf, frequency_bins)
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
    hrtf.mesh2hrtf_compatible = bool(mesh2hrtf_compatible)
    hrtf.mesh2hrtf_n_shift = mesh2hrtf_n_shift
    return hrtf


class HRTF(HRTFPlots):
    def __init__(
        self,
        Sofa: SOFA | None = None,
    ) -> None:
        """Represent an HRTF or HRIR object loaded from SOFA data.

        :class:`~hrtfpykit.hrtf.HRTF` is the main in-memory object used
        to inspect, subset, transform, plot, synchronize, and save HRTF/HRIR
        data loaded from SOFA files. It supports the SimpleFreeFieldHRIR and
        SimpleFreeFieldHRTF conventions and keeps both acoustic representations
        available:

        - :attr:`~hrtfpykit.hrtf.HRTF.IR`: time-domain impulse responses
        - :attr:`~hrtfpykit.hrtf.HRTF.TF`: frequency-domain transfer functions

        The object stores acoustic arrays separately from the backing
        :class:`~hrtfpykit.sofa.SOFA` object. Transformations and
        selections operate on the in-memory arrays first; SOFA variables are
        updated only when :meth:`~hrtfpykit.hrtf.HRTF.update_sofa` or
        :meth:`~hrtfpykit.hrtf.HRTF.save` is called. This makes processing
        workflows explicit and prevents intermediate edits from being written
        to the file representation automatically.

        Spatial metadata and source-grid operations are exposed through
        :class:`~hrtfpykit.hrtf.sources.Sources`. That object resolves SOFA
        ``SourcePosition`` data, named directions, coordinate-system conversion,
        and plane-based queries used by both processing and visualization methods.

        The object exposes plotting workflows for HRTF inspection and analysis. This
        includes source-grid visualizations, spectral views, plane projections, and
        metric-oriented plots that operate directly on the current in-memory state,
        including any selection or transformation.

        The object stores the optional SOFA backing object and starts with
        empty metadata fields. Domain interface objects such as
        :class:`~hrtfpykit.hrtf.domain.IR`,
        :class:`~hrtfpykit.hrtf.domain.TF`,
        :class:`~hrtfpykit.hrtf.sources.Sources`, and
        :class:`~hrtfpykit.hrtf.transforms.Transform` are created lazily when
        their properties are first accessed.
        Raw SOFA variables are not parsed and missing domains are not derived here;
        :func:`~hrtfpykit.hrtf.load_hrtf` performs that loading and synchronization
        work.

        Notes
        -----
        A typical workflow is to load an object with
        :func:`~hrtfpykit.hrtf.load_hrtf`, inspect or subset data with
        :class:`~hrtfpykit.hrtf.sources.Sources` and
        :meth:`~hrtfpykit.hrtf.HRTF.select`, apply transforms through
        :attr:`~hrtfpykit.hrtf.HRTF.transform`, visualize using plotting
        methods, then synchronize and export with
        :meth:`~hrtfpykit.hrtf.HRTF.update_sofa` and
        :meth:`~hrtfpykit.hrtf.HRTF.save`.

        The instance keeps a reference to the backing
        :class:`~hrtfpykit.sofa.SOFA` object in
        :attr:`~hrtfpykit.hrtf.HRTF.Sofa`. Transformation state is tracked
        internally; selected source subsets are tracked separately by
        :class:`~hrtfpykit.hrtf.sources.Sources`. Mesh2HRTF compatible loading
        state is also stored so reset and TF-domain workflows can reconstruct
        HRIR data with the same convention used at load time.

        Parameters
        ----------
        Sofa : :class:`~hrtfpykit.sofa.SOFA` | None, default=None
            :class:`~hrtfpykit.sofa.SOFA` object that backs the HRTF
            instance. When None, the object is created empty and should be
            populated later.

        Attributes
        ----------
        Sofa : :class:`~hrtfpykit.sofa.SOFA` or None
            Backing :class:`~hrtfpykit.sofa.SOFA` object used for source
            metadata, persistence, and SOFA synchronization.
        SOFAConventions : str or None
            Active SOFA convention associated with the loaded or constructed HRTF
            object.
        fft_length : int or None
            FFT length used when synchronizing between IR and TF representations.
        mesh2hrtf_compatible : bool
            Whether SimpleFreeFieldHRTF data should use Mesh2HRTF compatible
            TF-to-IR reconstruction.
        mesh2hrtf_n_shift : int or None
            Circular shift in samples used when
            :attr:`mesh2hrtf_compatible` is True.
        _transformed : bool
            Internal flag indicating whether the in-memory acoustic data were produced
            by a transform workflow.

        """
        self.Sofa: SOFA | None = Sofa
        self.SOFAConventions: str | None = None
        self.fft_length: int | None = None
        self.mesh2hrtf_compatible: bool = False
        self.mesh2hrtf_n_shift: int | None = None
        self._transformed: bool = False

    @cached_property
    def IR(self) -> "IR":
        """Access the time-domain HRIR representation object.

        This object stores :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>`
        and :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` for
        the parent :class:`~hrtfpykit.hrtf.HRTF` object and exposes
        time-domain inspection helpers such as sample length, duration, and
        ITD calculation.

        Examples
        --------
        Load a SOFA file and access the HRIR samples, sample-rate metadata,
        signal length, duration, and ITD values through ``hrtf.IR``:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> ir_values = hrtf.IR.values
        >>> sample_rate = hrtf.IR.sample_rate
        >>> first_position_left_ir = hrtf.IR.values[0, 0, :]
        >>> itd_samples = hrtf.IR.get_itd(output="samples")
        >>> ir_values.shape
        (793, 2, 256)
        >>> sample_rate
        44100.0
        >>> first_position_left_ir.shape
        (256,)
        >>> hrtf.IR.ir_length
        256
        >>> hrtf.IR.ir_duration
        0.005804988662131519
        >>> itd_samples.shape
        (793,)
        """
        return IR(self)

    @cached_property
    def TF(self) -> "TF":
        """Access the frequency-domain HRTF representation object.

        This object stores complex :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`
        and :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`
        for the parent object and exposes derived magnitude, phase, real, and
        imaginary views used by transforms, metrics, and plots.

        Examples
        --------
        Load a SOFA file and access the frequency-domain HRTF values, frequency
        axis, bin metadata, and derived spectral arrays through ``hrtf.TF``:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.TF.values.shape
        (793, 2, 129)
        >>> hrtf.TF.values.dtype
        dtype('complex128')
        >>> hrtf.TF.frequency_bins[:5]
        array([  0.      , 172.265625, 344.53125 , 516.796875, 689.0625  ])
        >>> hrtf.TF.tf_length
        129
        >>> hrtf.TF.frequency_bins_step
        172.265625
        >>> hrtf.TF.min_frequency_bin
        0.0
        >>> hrtf.TF.max_frequency_bin
        22050.0
        >>> hrtf.TF.magnitude.shape
        (793, 2, 129)
        >>> hrtf.TF.get_magnitude_db().shape
        (793, 2, 129)
        >>> hrtf.TF.phase.shape
        (793, 2, 129)
        >>> hrtf.TF.real.shape
        (793, 2, 129)
        >>> hrtf.TF.imag.shape
        (793, 2, 129)
        """
        return TF(self)

    @cached_property
    def Sources(self) -> "Sources":
        """Access the spatial source-grid object.

        :class:`~hrtfpykit.hrtf.sources.Sources` reads SOFA ``SourcePosition``
        data, converts between the supported coordinate systems, resolves named
        positions, and tracks selected source indices after spatial subsetting.

        Examples
        --------
        Load a SOFA file and access source-grid positions, coordinate-system
        metadata, available angles, nearest-position matches, and selected
        source views through ``hrtf.Sources``:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.Sources.source_coordinate_system
        'spherical'
        >>> hrtf.Sources.get_positions().shape
        (793, 3)
        >>> hrtf.Sources.get_positions()[:3]
        array([[  0. , -45. ,   1.5],
               [  0. , -30. ,   1.5],
               [  0. , -20. ,   1.5]])
        >>> hrtf.Sources.get_positions(angle_unit="radians")[0]
        array([ 0.        , -0.78539816,  1.5       ])
        >>> hrtf.Sources.get_azimuth_angles()[:5]
        array([ 0.,  5., 10., 15., 20.])
        >>> hrtf.Sources.get_elevation_angles()
        array([-45., -30., -20., -10.,   0.,  10.,  20.,  30.,  45.,  60.,  75.,
                90.])
        >>> elevations_at_front, real_azimuth = (
        ...     hrtf.Sources.get_elevation_angles_for_azimuth(0.0)
        ... )
        >>> elevations_at_front[:5]
        array([-45., -30., -20., -10.,   0.])
        >>> real_azimuth
        0.0
        >>> azimuths_on_horizontal, real_elevation = (
        ...     hrtf.Sources.get_azimuth_angles_for_elevation(0.0)
        ... )
        >>> azimuths_on_horizontal[:5]
        array([ 0.,  5., 10., 15., 20.])
        >>> real_elevation
        0.0
        >>> hrtf.Sources.get_position_index("front")
        (4, array([0. , 0. , 1.5]))
        >>> selected = hrtf.select(positions=["front", "left", "right"])
        >>> selected.Sources.get_positions().shape
        (3, 3)
        """
        return Sources(self)

    @cached_property
    def transform(self) -> "Transform":
        """Access the immutable transformation interface for this HRTF.

        Methods on this object clone the current HRTF, apply one processing
        operation, synchronize the affected IR or TF representation, and return
        the derived HRTF without mutating the original instance.
        """
        return Transform(self)

    def clone(self) -> "HRTF":
        """Create a deep clone of the current :class:`~hrtfpykit.hrtf.HRTF` object.

        The clone receives copied IR and TF arrays, sample-rate and
        frequency-bin metadata, FFT length, Mesh2HRTF compatible reconstruction
        settings, transformation state, and source selection state. When the
        backing :class:`~hrtfpykit.sofa.SOFA` object can be cloned, the clone
        receives an independent SOFA handle; otherwise the original handle is
        retained.

        Returns
        -------
        HRTF
            New object with copied acoustic arrays, domain metadata, source
            selection state, SOFA convention metadata, Mesh2HRTF compatible
            loading state, and transformation flag.

        Examples
        --------
        Clone a loaded HRTF before changing array values so the original object
        remains unchanged:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf_copy = hrtf.clone()
        >>> hrtf_copy.IR.values[0, 0, 0] += 1.0
        >>> hrtf_copy.IR.values[0, 0, 0] == hrtf.IR.values[0, 0, 0]
        False
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
        hrtf.mesh2hrtf_compatible = self.mesh2hrtf_compatible
        hrtf.mesh2hrtf_n_shift = self.mesh2hrtf_n_shift
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

        This method discards current in-memory acoustic edits and reloads the
        active domain data from :attr:`~hrtfpykit.hrtf.HRTF.Sofa`. HRIR
        files are restored from ``Data.IR`` and ``Data.SamplingRate`` and then
        converted to TF. HRTF files are restored from ``Data.Real``,
        ``Data.Imag``, and ``N``
        and then converted to IR. If the object was loaded with
        Mesh2HRTF compatible reconstruction, reset uses the same compatibility
        flag and sample shift. Source selections are cleared when the
        :class:`~hrtfpykit.hrtf.sources.Sources` object has already been
        initialized.

        Returns
        -------
        HRTF
            Current instance after restoring IR/TF, source-state, and metadata
            from the backed :class:`~hrtfpykit.sofa.SOFA` object while keeping
            the stored Mesh2HRTF compatible reconstruction settings.

        Raises
        ------
        ValueError
            If no SOFA file is attached, the SOFA file is not loaded, the
            convention is unsupported, or required acoustic variables are
            missing or empty.

        Examples
        --------
        Restore a selected HRTF object from its backed SOFA content:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> selected = hrtf.select(positions=["front", "left", "right"])
        >>> selected.IR.values.shape
        (3, 2, 256)
        >>> restored = selected.reset()
        >>> restored.IR.values.shape
        (793, 2, 256)
        >>> restored.is_transformed()
        False
        """
        if self.Sofa is None:
            raise ValueError("Cannot reset an HRTF without a loaded SOFA dataset")
        if self.Sofa.GlobalAttributes is None or self.Sofa.Variables is None:
            raise ValueError("SOFA dataset is not loaded")

        global_attrs = self.Sofa.GlobalAttributes
        variables = self.Sofa.Variables
        allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
        try:
            convention = cast(Any, global_attrs.get("SOFAConventions")).value
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

            ir = np.asarray(cast(Any, variables.get("Data.IR")).value)
            if ir.size == 0 or np.all(ir == 0):
                raise ValueError("SimpleFreeFieldHRIR requires non empty 'Data.IR'.")
            sample_rate_data = np.asarray(
                cast(Any, variables.get("Data.SamplingRate")).value,
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
            real = np.asarray(cast(Any, variables.get("Data.Real")).value, dtype=float)
            if real.size == 0 or np.all(real == 0):
                raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Real'.")
            imag = np.asarray(cast(Any, variables.get("Data.Imag")).value, dtype=float)
            if imag.size == 0 or np.all(imag == 0):
                raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Imag'.")
            frequency_bins = np.asarray(cast(Any, variables.get("N")).value, dtype=float)
            if frequency_bins.size == 0 or np.all(frequency_bins == 0):
                raise ValueError("SimpleFreeFieldHRTF requires non empty 'N'.")

            tf = real + 1j * imag
            tf, frequency_bins = prepend_missing_dc(tf, frequency_bins)
            ir, sample_rate, fft_length_used = ir_from_tf(
                tf,
                frequency_bins=frequency_bins,
                mesh2hrtf_compatible=self.mesh2hrtf_compatible,
                n_shift=self.mesh2hrtf_n_shift,
            )
            self.IR.values = np.array(ir, copy=True)
            self.IR.sample_rate = float(sample_rate)
            self.TF.values = np.array(tf, copy=True)
            self.TF.frequency_bins = np.array(frequency_bins, copy=True)
            self.fft_length = fft_length_used

        if "Sources" in self.__dict__:
            self.Sources.source_coordinate_system = (
                cast(Any, cast(Any, self.Sofa.VariableAttributes).get("SourcePosition:Type")).value
            )
            self.Sources._selected_indices = None
        self.SOFAConventions = convention
        self._transformed = False
        return self

    def is_transformed(self) -> bool:
        """Return the current HRTF transformation flag.

        The flag is set by transform workflows that modify acoustic data. It is
        a workflow state indicator, not a byte-by-byte comparison between
        in-memory arrays and the backing SOFA object. Source selection is
        tracked separately on :class:`~hrtfpykit.hrtf.sources.Sources` and is handled independently by
        :meth:`~hrtfpykit.hrtf.HRTF.update_sofa`.

        Returns
        -------
        bool
            True if a transform workflow has marked the object as
            transformed; False otherwise.

        Examples
        --------
        Check whether a transform returned a derived HRTF while the original
        object stayed unchanged:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> windowed = hrtf.transform.apply_window("hann")
        >>> hrtf.is_transformed()
        False
        >>> windowed.is_transformed()
        True
        """
        return self._transformed

    def update_sofa(
        self,
        change_sofa_dimensions: bool = False,
        sofa_convention: str = "same",
    ) -> None:
        """Synchronize in-memory IR/TF data into the backed :class:`~hrtfpykit.sofa.SOFA` object.

        The method converts the current acoustic representation into the
        requested SOFA convention and writes the corresponding SOFA variables
        on a cloned or updated :class:`~hrtfpykit.sofa.SOFA` object. It
        updates HRIR output through ``Data.IR`` and ``Data.SamplingRate``; HRTF
        output through ``Data.Real``, ``Data.Imag``, and ``N``. Obsolete variables from the
        opposite convention are removed when the output convention changes.
        If a SimpleFreeFieldHRTF file omitted the DC bin at load time,
        synchronization writes the normalized TF with the inserted DC bin.

        Dimension handling is conservative. If transformed data no longer fit
        existing SOFA dimensions, synchronization raises unless
        ``change_sofa_dimensions=True``. When resizing is allowed, supported
        dependent variables on the measurement axis are subset with the current
        source selection; unsupported dependent variables still raise explicit
        errors to avoid silently corrupting SOFA structure.

        The method updates the in-memory
        :attr:`~hrtfpykit.hrtf.HRTF.Sofa` object only. Use
        :meth:`~hrtfpykit.hrtf.HRTF.save` to persist the synchronized
        SOFA object to disk. When TF values must be converted back to IR during
        synchronization, the stored Mesh2HRTF compatible reconstruction
        settings are reused.

        Parameters
        ----------
        change_sofa_dimensions : bool, default=False
            If True, allows resizing fixed SOFA dimensions when transformed
            data shape differs from backed variables.
        sofa_convention : {``same``, ``SimpleFreeFieldHRIR``, ``SimpleFreeFieldHRTF``}, default=``same``
            Output SOFA convention to enforce during synchronization. ``same``
            keeps the original backed SOFA convention.

        Returns
        -------
        None
            This method updates :attr:`~hrtfpykit.hrtf.HRTF.Sofa`
            in-place and does not return data.

        Raises
        ------
        ValueError
            If no SOFA file is loaded, the convention is unsupported,
            required domain values are missing, transformed shapes cannot be
            represented by the SOFA dimensions, or requested dimension changes
            would affect variables that the method cannot update safely.

        Examples
        --------
        Synchronize a selected source subset into the backed SOFA object before
        saving or inspecting SOFA variables:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> selected = hrtf.select(positions=["front", "left", "right"])
        >>> selected.update_sofa(change_sofa_dimensions=True)
        >>> source_positions = selected.Sofa.Variables.get("SourcePosition").value
        >>> source_positions.shape
        (3, 3)
        """
        if self.Sofa is None or self.Sofa.netCDF4_dataset is None:
            raise ValueError("SOFA dataset is not loaded")

        if self.Sofa.GlobalAttributes is None or self.Sofa.Variables is None:
            raise ValueError("SOFA dataset is not loaded")
        try:
            backed_convention = cast(Any, self.Sofa.GlobalAttributes.get("SOFAConventions")).value
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
        backed_state_matches = True
        if (
            not self._transformed
            and resolved_sofa_convention == backed_convention
            and not has_selected_subset
        ):
            dataset = self.Sofa.netCDF4_dataset
            if dataset is None:
                raise ValueError("SOFA dataset is not loaded")
            if resolved_sofa_convention == "SimpleFreeFieldHRIR":
                if (
                    self.IR.values is not None
                    and self.IR.sample_rate is not None
                    and "Data.IR" in dataset.variables
                    and "Data.SamplingRate" in dataset.variables
                ):
                    backed_ir = np.asarray(dataset.variables["Data.IR"][:])
                    backed_sample_rate = np.asarray(
                        dataset.variables["Data.SamplingRate"][:],
                        dtype=float,
                    )
                    backed_state_matches = (
                        backed_ir.shape == np.asarray(self.IR.values).shape
                        and np.allclose(backed_ir, np.asarray(self.IR.values))
                        and np.allclose(
                            backed_sample_rate,
                            np.asarray(float(self.IR.sample_rate)),
                        )
                    )
                else:
                    backed_state_matches = False
            else:
                if (
                    self.TF.values is not None
                    and self.TF.frequency_bins is not None
                    and "Data.Real" in dataset.variables
                    and "Data.Imag" in dataset.variables
                    and "N" in dataset.variables
                ):
                    backed_real = np.asarray(
                        dataset.variables["Data.Real"][:],
                        dtype=float,
                    )
                    backed_imag = np.asarray(
                        dataset.variables["Data.Imag"][:],
                        dtype=float,
                    )
                    backed_frequency_bins = np.asarray(
                        dataset.variables["N"][:],
                        dtype=float,
                    )
                    backed_state_matches = (
                        backed_real.shape == np.asarray(self.TF.values).shape
                        and backed_imag.shape == np.asarray(self.TF.values).shape
                        and backed_frequency_bins.shape
                        == np.asarray(self.TF.frequency_bins).shape
                        and np.allclose(backed_real, np.real(np.asarray(self.TF.values)))
                        and np.allclose(backed_imag, np.imag(np.asarray(self.TF.values)))
                        and np.allclose(
                            backed_frequency_bins,
                            np.asarray(self.TF.frequency_bins, dtype=float),
                        )
                    )
                else:
                    backed_state_matches = False
            if backed_state_matches:
                print(
                    "HRTF is not transformed. SOFA-backed object is already up to date."
                )
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
                    mesh2hrtf_compatible=self.mesh2hrtf_compatible,
                    n_shift=self.mesh2hrtf_n_shift,
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
            obsolete_variables: tuple[str, ...] = ("Data.Real", "Data.Imag", "N")
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

        dataset = cast(Any, working_sofa.netCDF4_dataset)
        if dataset is None:
            raise ValueError("SOFA dataset is not loaded")
        missing_dc_normalization = False
        if (
            resolved_sofa_convention == "SimpleFreeFieldHRTF"
            and backed_convention == "SimpleFreeFieldHRTF"
            and not self._transformed
            and "Data.Real" in dataset.variables
            and "Data.Imag" in dataset.variables
            and "N" in dataset.variables
        ):
            backed_tf = (
                np.asarray(dataset.variables["Data.Real"][:], dtype=float)
                + 1j * np.asarray(dataset.variables["Data.Imag"][:], dtype=float)
            )
            backed_frequency_bins = np.asarray(dataset.variables["N"][:], dtype=float)
            normalized_tf, normalized_frequency_bins = prepend_missing_dc(
                backed_tf,
                backed_frequency_bins,
            )
            missing_dc_normalization = (
                normalized_tf.shape == np.asarray(target_variables["Data.Real"]).shape
                and normalized_frequency_bins.shape == np.asarray(target_variables["N"]).shape
                and normalized_tf.shape[-1] != backed_tf.shape[-1]
                and np.allclose(np.real(normalized_tf), target_variables["Data.Real"])
                and np.allclose(np.imag(normalized_tf), target_variables["Data.Imag"])
                and np.allclose(normalized_frequency_bins, target_variables["N"])
            )
        allow_dimension_changes = change_sofa_dimensions or missing_dc_normalization
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
                    variable_dimensions: tuple[str, ...] = ("N",)
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
        if (mismatched_variables or dimension_overrides) and not allow_dimension_changes:
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
                current_values = np.asarray(dataset.variables[variable_name][:])
                target_array = np.asarray(target_values)
                try:
                    comparable_target = np.broadcast_to(target_array, current_values.shape)
                except ValueError:
                    updated_sofa.modify_variable(variable_name, target_values)
                else:
                    if not np.allclose(current_values, comparable_target):
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
                    current_values = np.asarray(dataset.variables[variable_name][:])
                    target_array = np.asarray(target_values)
                    try:
                        comparable_target = np.broadcast_to(target_array, current_values.shape)
                    except ValueError:
                        updated_sofa.modify_variable(variable_name, target_values)
                    else:
                        if not np.allclose(current_values, comparable_target):
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
                    if not allow_dimension_changes:
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
        resolved_global_attributes = {
            "SOFAConventions": resolved_sofa_convention,
            "DataType": (
                "FIR"
                if resolved_sofa_convention == "SimpleFreeFieldHRIR"
                else "TF"
            ),
        }
        for attribute_name, attribute_value in resolved_global_attributes.items():
            if attribute_name in updated_dataset.ncattrs():
                current_value = getattr(updated_dataset, attribute_name)
                if current_value != attribute_value:
                    updated_sofa.modify_global_attribute(attribute_name, attribute_value)
            else:
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

        This is the persistence endpoint for the HRTF workflow. It first calls
        :meth:`~hrtfpykit.hrtf.HRTF.update_sofa` so the backed
        :class:`~hrtfpykit.sofa.SOFA` object reflects the current in-memory
        IR/TF state and requested convention, then delegates the disk write to
        :meth:`~hrtfpykit.sofa.SOFA.save`.

        Parameters
        ----------
        path : str | Path | None, default=None
            Output file path. If None, the current backed SOFA path is
            used by the underlying :class:`~hrtfpykit.sofa.SOFA` object.
        overwrite : bool, default=False
            If True, allows overwriting an existing file.
        change_sofa_dimensions : bool, default=False
            Forwarded to :meth:`~hrtfpykit.hrtf.HRTF.update_sofa` to control SOFA dimension resizing.
        sofa_convention : {``same``, ``SimpleFreeFieldHRIR``, ``SimpleFreeFieldHRTF``}, default=``same``
            Forwarded to :meth:`~hrtfpykit.hrtf.HRTF.update_sofa` to select output convention.

        Returns
        -------
        Path
            Path to the saved SOFA file.

        Raises
        ------
        ValueError
            If no SOFA file is attached or synchronization fails.
        FileExistsError
            If the target path already exists and overwrite=False.

        Examples
        --------
        Save a processed HRTF to a new SOFA file using a relative output path:

        >>> from pathlib import Path
        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> windowed = hrtf.transform.apply_window("hann")
        >>> output_dir = Path("processed")
        >>> output_dir.mkdir(exist_ok=True)
        >>> saved_path = windowed.save(
        ...     output_dir / "P0001_windowed.sofa",
        ...     overwrite=True,
        ... )
        >>> saved_path.name
        'P0001_windowed.sofa'
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
        azimuth_angles: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
        elevation_angles: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
        ear: str = "both",
        angle_unit: str = "degrees",
        start: int | None = None,
        end: int | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> "HRTF":
        """Select a spatial subset, ear subset, and/or IR crop from the HRTF.

        Selection returns a cloned :class:`~hrtfpykit.hrtf.HRTF` object
        and leaves the original object unchanged. Spatial selection can be
        expressed with explicit positions, named positions, a geometric plane,
        or azimuth/elevation angle filters. These source-selection modes are
        mutually exclusive: ``positions`` cannot be combined with ``plane`` or
        angle filters, and ``plane`` cannot be combined with angle filters.
        ``azimuth_angles`` and ``elevation_angles`` may be used together; when
        both are provided, selected sources must satisfy both filters. Source
        selections are applied along the leading source axis of both
        :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>` and
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` when those domains are
        available.

        Ear selection keeps both ears by default. Selecting ``left`` or
        ``right`` removes the ear axis entry from available IR and TF arrays.
        IR cropping is applied along the final sample axis and automatically
        recomputes the TF representation from the cropped IR using the cropped
        IR length as the FFT length. Crop boundaries can be provided either as
        sample indices or as seconds, but not both in the same call.

        Parameters
        ----------
        positions : np.ndarray | list[list[float]] | list[float] | None, default=None
            Explicit positions or named aliases to select. Named positions use
            the source-grid aliases such as ``front``, ``back``,
            ``left``, and ``right``. Numeric positions are interpreted in
            position_coordinate_system.
        position_coordinate_system : {``spherical``, ``cartesian``, ``lateral-polar``}, default=``spherical``
            Coordinate system used by numeric positions queries.
        plane : str | None, default=None
            Plane name to filter positions. Supported values are
            horizontal, median, and frontal.
        plane_angle : float, default=0.0
            Plane angle used to resolve the nearest available plane. For the
            horizontal plane this is elevation; for median and frontal planes
            this is azimuth.
        azimuth_angles : float, sequence of float, numpy.ndarray, or None, default=None
            Azimuth angle or angles used to keep matching source positions.
            Requested values resolve to the nearest available source-grid
            azimuth in spherical coordinates and may be combined only with
            elevation_angles.
        elevation_angles : float, sequence of float, numpy.ndarray, or None, default=None
            Elevation angle or angles used to keep matching source positions.
            Requested values resolve to the nearest available source-grid
            elevation in spherical coordinates and may be combined only with
            azimuth_angles.
        ear : {``both``, ``left``, ``right``}, default=``both``
            Ear selection.
        angle_unit : {``degrees``, ``radians``}, default=``degrees``
            Angle unit used for spatial queries and plane angles.
        start : int | None, default=None
            IR crop start index (samples).
        end : int | None, default=None
            IR crop end index (samples).
        start_seconds : float | None, default=None
            IR crop start time (seconds). Mutually exclusive with start.
        end_seconds : float | None, default=None
            IR crop end time (seconds). Mutually exclusive with end.

        Returns
        -------
        HRTF
            New :class:`~hrtfpykit.hrtf.HRTF` object containing the selected subset.

        Raises
        ------
        ValueError
            If the requested selection is invalid, no positions remain after
            filtering, more than one source-selection mode is requested, the
            requested ear is unavailable, crop boundaries are invalid,
            seconds-based cropping is requested without a sample rate, or IR
            data are unavailable for cropping.

        Examples
        --------
        Select three named directions, keep only the left ear, and crop the
        HRIR samples used in the returned HRTF:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.IR.values.shape
        (793, 2, 256)
        >>> selected = hrtf.select(
        ...     positions=["front", "left", "right"],
        ...     ear="left",
        ...     start=0,
        ...     end=128,
        ... )
        >>> selected.IR.values.shape
        (3, 128)
        >>> selected.TF.values.shape
        (3, 65)
        >>> selected.fft_length
        128
        """
        transformed_hrtf = self.clone()
        selected_indices: np.ndarray | None = None
        ear_key = str(ear).strip().lower()
        if ear_key not in {"both", "left", "right"}:
            raise ValueError("ear must be one of: both, left, right")

        selecting_positions = positions is not None
        selecting_plane = plane is not None
        selecting_angles = azimuth_angles is not None or elevation_angles is not None
        if sum((selecting_positions, selecting_plane, selecting_angles)) > 1:
            raise ValueError(
                "Use only one source selection mode: positions, plane, or azimuth_angles/elevation_angles"
            )

        selecting_spatial = selecting_positions or selecting_plane or selecting_angles
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

            if selecting_angles:
                spherical_positions = get_spherical_positions(
                    transformed_hrtf.Sources,
                    angle_unit=angle_unit,
                )
                if spherical_positions.ndim != 2 or spherical_positions.shape[-1] != 3:
                    raise ValueError("Source positions grid must have shape (N, 3)")
                selected_mask = np.ones(source_count, dtype=bool)
                angle_unit_key = str(angle_unit).strip().lower()
                if angle_unit_key == "degrees":
                    full_azimuth = 360.0
                elif angle_unit_key == "radians":
                    full_azimuth = float(2.0 * np.pi)
                else:
                    raise ValueError("angle_unit must be 'degrees' or 'radians'")
                angle_filters = (
                    ("azimuth_angles", "azimuth", azimuth_angles, 0, True),
                    ("elevation_angles", "elevation", elevation_angles, 1, False),
                )
                for angle_name, angle_label, angle_values, column_index, wrap_angle in angle_filters:
                    if angle_values is None:
                        continue
                    raw_angles = np.asarray(angle_values, dtype=object)
                    if raw_angles.ndim == 0:
                        raw_values = np.asarray([raw_angles.item()], dtype=object)
                    elif raw_angles.ndim == 1:
                        raw_values = raw_angles
                    else:
                        raise ValueError(
                            f"{angle_name} must be a finite scalar or one-dimensional sequence"
                        )
                    if raw_values.size == 0:
                        raise ValueError(f"At least one {angle_label} angle is required")
                    if any(isinstance(value, bool | np.bool_) for value in raw_values.tolist()):
                        raise ValueError(f"{angle_name} must contain finite numeric values")
                    try:
                        requested_angles = np.asarray(raw_values, dtype=float)
                    except (TypeError, ValueError):
                        raise ValueError(f"{angle_name} must contain finite numeric values") from None
                    if not np.all(np.isfinite(requested_angles)):
                        raise ValueError(f"{angle_name} must contain finite numeric values")
                    resolved_angles: list[float] = []
                    for requested_angle in requested_angles:
                        if wrap_angle:
                            _, resolved_angle = (
                                transformed_hrtf.Sources.get_elevation_angles_for_azimuth(
                                    float(requested_angle),
                                    angle_unit=angle_unit,
                                )
                            )
                        else:
                            _, resolved_angle = (
                                transformed_hrtf.Sources.get_azimuth_angles_for_elevation(
                                    float(requested_angle),
                                    angle_unit=angle_unit,
                                )
                            )
                        if resolved_angle not in resolved_angles:
                            resolved_angles.append(float(resolved_angle))
                    source_angles = np.asarray(
                        spherical_positions[:, int(column_index)],
                        dtype=float,
                    )
                    requested_angles = np.asarray(resolved_angles, dtype=float)
                    if wrap_angle:
                        requested_angles = np.mod(requested_angles, full_azimuth)
                        source_angles = np.mod(source_angles, full_azimuth)
                    selected_mask &= np.isin(
                        np.round(source_angles, 2),
                        np.round(requested_angles, 2),
                    )
                selected_indices = np.flatnonzero(selected_mask).astype(int)
                if selected_indices.size == 0:
                    raise ValueError(
                        "Selection produced no source positions for the requested azimuth/elevation angles"
                    )

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
                fft_length=None,
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
