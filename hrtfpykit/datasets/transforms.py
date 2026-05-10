from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..hrtf.domain import IR, TF
    from ..hrtf.hrtf import HRTF


class HRTFTransform:
    """Factory namespace for reusable HRTF preprocessing callables.

    :class:`~hrtfpykit.datasets.HRTFTransform` creates small callables that
    receive a subject :class:`~hrtfpykit.hrtf.hrtf.HRTF`, run one HRTF
    operation, and return the transformed object. These callables can be used
    as dataset-level transforms through ``dataset_hrtf_transform`` or as
    HRTF-level spec transforms, for example through
    :class:`~hrtfpykit.datasets.HRTFSpec`.

    The factory methods are thin adapters over
    :attr:`~hrtfpykit.hrtf.hrtf.HRTF.transform` and
    :meth:`~hrtfpykit.hrtf.hrtf.HRTF.select`. They do not load files, mutate
    dataset state, or change the original SOFA resources. They only package
    transform arguments so the same preprocessing can be applied consistently to
    every loaded subject.

    Notes
    -----
    When used as ``dataset_hrtf_transform``, the returned callables are executed
    after a subject HRTF is read and before it is cached. When used on an
    HRTF-level spec, they are executed before that spec extracts its value. If
    the callable returns an object that does not behave like an HRTF, dataset
    loading or value extraction raises an error before sample extraction
    continues.

    """

    @staticmethod
    def build(
        method_name: str,
        *args: object,
        **kwargs: object,
    ) -> Callable[[object], object]:
        """Create an HRTF-level callable from a transform method name.

        This is the generic factory used by the named convenience methods below. It
        stores the requested transform method name and arguments, then delays method
        lookup until a real :class:`~hrtfpykit.hrtf.hrtf.HRTF` object is loaded.
        When executed, the returned callable looks up method_name on
        :attr:`~hrtfpykit.hrtf.hrtf.HRTF.transform`, calls it with the stored
        arguments, and returns the transformed HRTF.

        Parameters
        ----------
        method_name : str
            Name of the method available through
            :attr:`~hrtfpykit.hrtf.hrtf.HRTF.transform`.
        *args : object
            Positional arguments forwarded to the transform method.
        **kwargs : object
            Keyword arguments forwarded to the transform method.

        Returns
        -------
        callable
            Callable that accepts an :class:`~hrtfpykit.hrtf.hrtf.HRTF` object and
            returns the transformed HRTF object.

        Raises
        ------
        TypeError
            Raised by the returned callable if the supplied object does not expose a
            transform attribute.
        AttributeError
            Raised by the returned callable if method_name is not available or not
            callable on the object's transform namespace.

        Notes
        -----
        The factory marks the returned callable with ``__hrtf_transform__`` so callers
        can distinguish HRTF transform wrappers from arbitrary user callables when
        introspection is needed.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.build("apply_padding", 16, location="end")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """

        def transform(hrtf: object) -> object:
            if not hasattr(hrtf, "transform"):
                raise TypeError(
                    "HRTFTransform expects an HRTF-like object with a transform attribute"
                )
            method = getattr(hrtf.transform, method_name, None)
            if method is None or not callable(method):
                raise AttributeError(
                    f"HRTF transform {method_name!r} is not available on {type(hrtf)!r}"
                )
            return method(*args, **kwargs)

        transform.__hrtf_transform__ = True
        return transform

    @staticmethod
    def select(
        *args: object,
        **kwargs: object,
    ) -> Callable[[object], object]:
        """Create an HRTF-level source selection callable.

        Selection is special because it calls
        :meth:`~hrtfpykit.hrtf.hrtf.HRTF.select` directly rather than a method under
        :attr:`~hrtfpykit.hrtf.hrtf.HRTF.transform`. The returned callable lets a
        dataset reduce or reorder source positions before specs extract acoustic
        values. The returned callable can be used through
        ``dataset_hrtf_transform`` or through HRTF-level spec transform hooks.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to :meth:`~hrtfpykit.hrtf.hrtf.HRTF.select`.
        **kwargs : object
            Keyword arguments forwarded to :meth:`~hrtfpykit.hrtf.hrtf.HRTF.select`.

        Returns
        -------
        callable
            Callable that accepts an :class:`~hrtfpykit.hrtf.hrtf.HRTF` object and
            returns the selected HRTF.

        Raises
        ------
        AttributeError
            Raised by the returned callable if the supplied object does not expose a
            callable :meth:`~hrtfpykit.hrtf.hrtf.HRTF.select`-compatible method.

        Notes
        -----
        Use this factory when the training or evaluation dataset should expose a
        source subset, for example a fixed plane or a small set of named positions,
        without rewriting the source SOFA files.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     dataset_hrtf_transform=HRTFTransform.select(positions=[0, 1]),
        ... )
        """

        def transform(hrtf: object) -> object:
            method = getattr(hrtf, "select", None)
            if method is None or not callable(method):
                raise AttributeError(
                    f"HRTF select is not available on {type(hrtf)!r}"
                )
            return method(*args, **kwargs)

        transform.__hrtf_transform__ = True
        return transform

    @staticmethod
    def apply_window(window_name: str) -> Callable[[object], object]:
        """Create a callable that applies a named window to each subject HRIR.

        The returned callable forwards window_name to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.apply_window`. The transform
        operates on time-domain IR values, applies the window along the final sample
        axis, and refreshes the TF representation before dataset specs read values.

        Parameters
        ----------
        window_name : str
            Window identifier forwarded to the HRTF transform layer, for example
            ``hann``, ``hamming``, ``blackman``, or ``rectangular``.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Windowing is useful when dataset samples should suppress late HRIR samples
        or enforce a consistent time-domain taper before frequency-domain features
        are computed.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.apply_window("hann")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build("apply_window", window_name)

    @staticmethod
    def apply_padding(
        padding_length: int,
        location: str = "end",
        value: float = 0,
    ) -> Callable[[object], object]:
        """Create a callable that pads each subject HRIR.

        The returned callable forwards padding_length, location, and value to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.apply_padding`. The transform
        pads the final IR sample axis and rebuilds TF values with the active FFT
        length before sample extraction.

        Parameters
        ----------
        padding_length : int
            Number of samples added to each impulse response.
        location : {``start``, ``end``}, default=``end``
            Side of the IR sample axis where padding is applied.
        value : float, default=0
            Constant value used in the padded region.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Padding changes the time-domain sample count available to sample-indexed
        specs and can also change derived frequency-domain values after TF
        resynchronization.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.apply_padding(16, location="end", value=0.0)
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "apply_padding",
            padding_length,
            location=location,
            value=value,
        )

    @staticmethod
    def upsampling(new_sample_rate: float) -> Callable[[object], object]:
        """Create a callable that upsamples each subject HRIR.

        The returned callable forwards new_sample_rate to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.upsampling`. The transform
        resamples IR values to the target rate, updates IR sample-rate metadata, and
        recomputes TF values and frequency bins from the resampled IR.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in hertz. It must be greater than the current subject
            IR sample rate.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Dataset-level upsampling keeps all loaded subjects in the same transformed
        sampling context before HRTF, ITD, ILD, SH, or sample-indexed specs resolve
        their values.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.upsampling(96000.0)
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build("upsampling", new_sample_rate)

    @staticmethod
    def downsampling(new_sample_rate: float) -> Callable[[object], object]:
        """Create a callable that downsamples each subject HRIR.

        The returned callable forwards new_sample_rate to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.downsampling`. The transform
        resamples IR values to the target rate, updates IR sample-rate metadata, and
        recomputes TF values and frequency bins from the resampled IR.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in hertz. It must be lower than the current subject
            IR sample rate.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Downsampling is intended for workflows that need a lower acoustic sampling
        rate than the original SOFA files while keeping the dataset interface
        unchanged.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.downsampling(44100.0)
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build("downsampling", new_sample_rate)

    @staticmethod
    def apply_fir_filter(
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        num_taps: int = 101,
        window: str | None = None,
    ) -> Callable[[object], object]:
        """Create a callable that applies FIR filtering to each subject HRIR.

        The returned callable forwards filter, cutoff, num_taps, and window to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.apply_fir_filter`. Filtering is
        performed in the time domain, then TF values are rebuilt before specs read
        the loaded subject.

        Parameters
        ----------
        filter : str
            FIR filter type accepted by the HRTF transform layer, such as
            ``lowpass``, ``highpass``, or ``bandpass``.
        cutoff : float | tuple[float, float] | None, default=None
            Cutoff frequency or frequency pair in hertz.
        num_taps : int, default=101
            FIR tap count used to design the filter.
        window : str | None, default=None
            Optional FIR design window.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        FIR filtering is appropriate when a reproducible linear-phase preprocessing
        step should be applied uniformly across all subjects selected by a dataset.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.apply_fir_filter(
        ...     "lowpass",
        ...     cutoff=8000.0,
        ...     num_taps=101,
        ... )
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "apply_fir_filter",
            filter,
            cutoff=cutoff,
            num_taps=num_taps,
            window=window,
        )

    @staticmethod
    def apply_iir_filter(
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        order: int = 10,
    ) -> Callable[[object], object]:
        """Create a callable that applies IIR filtering to each subject HRIR.

        The returned callable forwards filter, cutoff, and order to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.apply_iir_filter`. Filtering is
        performed in the time domain, then TF values are rebuilt before specs read
        the loaded subject.

        Parameters
        ----------
        filter : str
            IIR filter type accepted by the HRTF transform layer, such as
            ``lowpass``, ``highpass``, or ``bandpass``.
        cutoff : float | tuple[float, float] | None, default=None
            Cutoff frequency or frequency pair in hertz.
        order : int, default=10
            Butterworth filter order.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        IIR filtering is useful when compact Butterworth-style preprocessing is more
        important than linear-phase behavior.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.apply_iir_filter(
        ...     "highpass",
        ...     cutoff=200.0,
        ...     order=4,
        ... )
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "apply_iir_filter",
            filter,
            cutoff=cutoff,
            order=order,
        )

    @staticmethod
    def minimum_phase(
        method: str = "homomorphic",
        fft_length: int | None = None,
        epsilon: float = 1e-12,
    ) -> Callable[[object], object]:
        """Create a callable that converts each subject HRIR to minimum phase.

        The returned callable forwards method, fft_length, and epsilon to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.minimum_phase`. The transform
        replaces each HRIR with a minimum-phase version derived from its magnitude
        response, then rebuilds TF values for the loaded subject.

        Parameters
        ----------
        method : str, default=``homomorphic``
            Minimum-phase reconstruction method passed to the HRTF transform layer.
        fft_length : int | None, default=None
            Optional FFT length used during minimum-phase reconstruction.
        epsilon : float, default=1e-12
            Positive numerical floor used by the reconstruction routine.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Minimum-phase conversion is a dataset-side preprocessing step only. It lets
        a dataset expose minimum-phase acoustics without modifying the original SOFA
        files.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.minimum_phase(method="homomorphic")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "minimum_phase",
            method=method,
            fft_length=fft_length,
            epsilon=epsilon,
        )

    @staticmethod
    def to_ctf(
        weights: bool = False,
        magnitude_average: str = "log",
        attenuation: float | None = None,
    ) -> Callable[[object], object]:
        """Create a callable that converts each subject HRTF to CTF form.

        The returned callable forwards weights, magnitude_average, and attenuation
        to :meth:`~hrtfpykit.hrtf.transforms.Transform.to_ctf`. The transform
        collapses the source axis into a singleton common transfer function per ear
        before dataset specs consume the subject HRTF.

        Parameters
        ----------
        weights : bool, default=False
            Whether source-position weights are used when estimating the common
            transfer function.
        magnitude_average : {``log``, ``linear``}, default=``log``
            Magnitude averaging rule used by the CTF transform.
        attenuation : float | None, default=None
            Optional attenuation in decibels applied by the CTF transform.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        The CTF output has a singleton source axis. Dataset specs that index by
        position should be configured with that changed source layout in mind.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.to_ctf(weights=False, magnitude_average="log")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "to_ctf",
            weights=weights,
            magnitude_average=magnitude_average,
            attenuation=attenuation,
        )

    @staticmethod
    def to_dtf(
        weights: bool = False,
        magnitude_average: str = "log",
        attenuation: float | None = None,
    ) -> Callable[[object], object]:
        """Create a callable that converts each subject HRTF to DTF form.

        The returned callable forwards weights, magnitude_average, and attenuation
        to :meth:`~hrtfpykit.hrtf.transforms.Transform.to_dtf`. The transform
        estimates and removes a common transfer component while preserving the
        subject source layout.

        Parameters
        ----------
        weights : bool, default=False
            Whether source-position weights are used when estimating the common
            transfer function removed from the HRTF.
        magnitude_average : {``log``, ``linear``}, default=``log``
            Magnitude averaging rule used during DTF calculation.
        attenuation : float | None, default=None
            Optional attenuation in decibels applied by the DTF transform.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        DTF conversion is useful when learning or analysis should focus on
        direction-dependent spectral structure rather than the common ear response.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.to_dtf(weights=False, magnitude_average="log")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "to_dtf",
            weights=weights,
            magnitude_average=magnitude_average,
            attenuation=attenuation,
        )

    @staticmethod
    def modify_ir(new_ir: np.ndarray | IR | HRTF) -> Callable[[object], object]:
        """Create a callable that replaces each subject IR data array.

        The returned callable forwards new_ir to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.modify_ir`. Replacement data
        become the loaded subject's time-domain values, and TF values are rebuilt
        before dataset specs extract outputs.

        Parameters
        ----------
        new_ir : np.ndarray | IR | HRTF
            Replacement time-domain data forwarded to
            :meth:`~hrtfpykit.hrtf.transforms.Transform.modify_ir`.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        The replacement leading shape must be compatible with each loaded subject's
        source and ear layout. Use this for controlled experiments where dataset
        metadata and indexing come from real resources but acoustic arrays are
        supplied externally.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.modify_ir(np.zeros((440, 2, 256)))
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build("modify_ir", new_ir)

    @staticmethod
    def modify_phase(
        new_phase: np.ndarray,
        unit: str = "degrees",
    ) -> Callable[[object], object]:
        """Create a callable that replaces each subject TF phase.

        The returned callable forwards new_phase and unit to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.modify_phase`. The transform
        preserves current TF magnitude, replaces phase values, and rebuilds IR
        values before sample extraction.

        Parameters
        ----------
        new_phase : np.ndarray
            Replacement phase array with the same TF layout expected by the
            HRTF transform layer.
        unit : {``degrees``, ``radians``}, default=``degrees``
            Angular unit used by new_phase.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        The replacement phase must be compatible with the TF layout of every loaded
        subject that will receive the callable.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.modify_phase(
        ...     np.zeros((440, 2, 129)),
        ...     unit="radians",
        ... )
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "modify_phase",
            new_phase,
            unit=unit,
        )

    @staticmethod
    def modify_tf(new_tf: np.ndarray | TF | HRTF) -> Callable[[object], object]:
        """Create a callable that replaces each subject TF data array.

        The returned callable forwards new_tf to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.modify_tf`. Replacement data
        become the loaded subject's complex frequency-domain values, and IR values
        are rebuilt before dataset specs extract outputs.

        Parameters
        ----------
        new_tf : np.ndarray | TF | HRTF
            Replacement complex frequency-domain data forwarded to the HRTF
            transform layer.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        The replacement leading shape must be compatible with each loaded subject's
        source and ear layout. Frequency-bin metadata are copied from TF-like inputs
        when available.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.modify_tf(
        ...     np.zeros((440, 2, 129), dtype=complex),
        ... )
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build("modify_tf", new_tf)

    @staticmethod
    def modify_magnitude(
        new_magnitude: np.ndarray,
        scale: str = "linear",
    ) -> Callable[[object], object]:
        """Create a callable that replaces each subject TF magnitude.

        The returned callable forwards new_magnitude and scale to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.modify_magnitude`. The transform
        preserves current TF phase, replaces magnitude values, and rebuilds IR values
        before dataset specs consume the loaded subject.

        Parameters
        ----------
        new_magnitude : np.ndarray
            Replacement magnitude array with the same TF layout expected by
            the HRTF transform layer.
        scale : {``linear``, ``db``}, default=``linear``
            Magnitude scale used by new_magnitude.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        The replacement magnitude must be broadcast-compatible with the TF layout of
        every loaded subject that will receive the callable.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.modify_magnitude(
        ...     np.ones((440, 2, 129)),
        ...     scale="linear",
        ... )
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "modify_magnitude",
            new_magnitude,
            scale=scale,
        )

    @staticmethod
    def apply_gain(
        gain: float | np.ndarray,
        scale: str = "db",
    ) -> Callable[[object], object]:
        """Create a callable that applies TF-domain gain to each subject.

        The returned callable forwards gain and scale to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.apply_gain`. Gain modifies TF
        magnitude while preserving phase, then IR values are rebuilt from the
        adjusted TF.

        Parameters
        ----------
        gain : float | np.ndarray
            Scalar or broadcast-compatible gain applied to each loaded HRTF.
        scale : {``db``, ``linear``}, default=``db``
            Scale used to interpret gain.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Scalar gains apply to every source, ear, and frequency bin. Array gains must
        be broadcast-compatible with each loaded subject's TF layout.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.apply_gain(3.0, scale="db")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "apply_gain",
            gain,
            scale=scale,
        )

    @staticmethod
    def modify_fft_length(new_fft_length: int) -> Callable[[object], object]:
        """Create a callable that changes each subject FFT length.

        The returned callable forwards new_fft_length to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.modify_fft_length`. The
        transform keeps current IR values unchanged and recomputes TF values with
        the requested FFT size.

        Parameters
        ----------
        new_fft_length : int
            FFT length used when recomputing the frequency-domain
            representation.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Changing FFT length affects the frequency-bin count and spacing exposed to
        frequency-domain specs, but it does not add new time-domain information to
        the HRIR.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.modify_fft_length(512)
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build("modify_fft_length", new_fft_length)

    @staticmethod
    def modify_source_coordinate_system(coordinate_system: str) -> Callable[[object], object]:
        """Create a callable that changes source coordinate-system metadata.

        The returned callable forwards coordinate_system to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.modify_source_coordinate_system`.
        The transform changes how the subject source manager reports positions,
        while leaving the source layout and acoustic arrays intact.

        Parameters
        ----------
        coordinate_system : {``spherical``, ``cartesian``, ``lateral-polar``}
            Source coordinate system requested for each loaded HRTF.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Use this factory when downstream specs or custom code should read source
        positions in a consistent coordinate system without editing SOFA
        SourcePosition values.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.modify_source_coordinate_system("spherical")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "modify_source_coordinate_system",
            coordinate_system,
        )

    @staticmethod
    def add_itd(
        itd: float,
        unit: str = "samples",
    ) -> Callable[[object], object]:
        """Create a callable that adds ITD to each subject HRIR.

        The returned callable forwards itd and unit to
        :meth:`~hrtfpykit.hrtf.transforms.Transform.add_itd`. Positive ITD values
        delay the left ear, negative values delay the right ear, and TF values are
        rebuilt after the time-domain shift.

        Parameters
        ----------
        itd : float
            Interaural time difference added to each loaded HRTF. Positive values
            delay the left ear and negative values delay the right ear.
        unit : {``samples``, ``seconds``}, default=``samples``
            Unit used by itd.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        This factory is useful for controlled binaural timing perturbations in
        dataset workflows. The underlying transform requires two ear channels and a
        delay smaller than the IR length.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.add_itd(4.0, unit="samples")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "add_itd",
            itd,
            unit=unit,
        )

    @staticmethod
    def delete_itd(
        method: str = "threshold",
        thresh_level: float = -10.0,
        upper_cut_freq: float = 3000.0,
        filter_order: int = 10,
    ) -> Callable[[object], object]:
        """Create a callable that estimates and removes ITD from each subject.

        The returned callable forwards method, thresh_level, upper_cut_freq, and
        filter_order to :meth:`~hrtfpykit.hrtf.transforms.Transform.delete_itd`.
        The transform estimates per-source interaural delay, advances the delayed
        channel, zero-fills the shifted tail, and rebuilds TF values.

        Parameters
        ----------
        method : {``threshold``, ``maxiacce``}, default=``threshold``
            ITD estimator used before delay compensation.
        thresh_level : float, default=-10.0
            Threshold offset in decibels used by the threshold estimator.
        upper_cut_freq : float, default=3000.0
            Low-pass cutoff in hertz used before ITD estimation.
        filter_order : int, default=10
            Butterworth low-pass filter order used before ITD estimation.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an
            :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.

        Notes
        -----
        Use this factory when a dataset should expose ITD-normalized acoustics while
        preserving the rest of the HRTF workflow. The underlying transform requires
        valid two-ear IR data and sample-rate metadata for the estimator path.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets import HRTFSpec
        >>> from hrtfpykit.datasets import HRTFTransform
        >>> transform = HRTFTransform.delete_itd(method="threshold")
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(),
        ...     dataset_hrtf_transform=transform,
        ... )
        """
        return HRTFTransform.build(
            "delete_itd",
            method=method,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        )
