from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..hrtf.domain import IR, TF
    from ..hrtf.hrtf import HRTF


class HRTFTransform:
    """Factory for dataset-level HRTF transform callables.

    ``HRTFTransform`` creates small callables that can be passed as
    ``dataset_hrtf_transform`` to dataset constructors. The callable receives a
    loaded HRTF object, calls the corresponding HRTF transform method, and returns
    the transformed HRTF before spec values are extracted.

    Examples
    --------
    >>> from hrtfpykit.datasets import HUTUBS
    >>> from hrtfpykit.datasets.specs import HRTFSpec
    >>> from hrtfpykit.datasets.transforms import HRTFTransform
    >>> dataset = HUTUBS(
    ...     root="datasets/hutubs",
    ...     inputs=HRTFSpec(),
    ...     dataset_hrtf_transform=HRTFTransform.apply_padding(16),
    ... )
    """

    @staticmethod
    def build(
        method_name: str,
        *args: object,
        **kwargs: object,
    ) -> Callable[[object], object]:
        """Create a dataset-level transform from an HRTF transform method name.

        This is the generic factory used by the named convenience methods below. It
        delays method lookup until a real HRTF object is loaded, then forwards the
        stored arguments to ``hrtf.transform.<method_name>`` so the same transform can
        be reused across subjects.

        Parameters
        ----------
        method_name : str
            Name of the method available under ``hrtf.transform``.
        *args : object
            Positional arguments forwarded to the transform method.
        **kwargs : object
            Keyword arguments forwarded to the transform method.

        Returns
        -------
        callable
            Callable that accepts an HRTF object and returns the transformed HRTF
        object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a dataset-level source selection transform.

        Selection is special because it calls ``hrtf.select`` directly rather than a
        method under ``hrtf.transform``. The returned callable lets datasets reduce
        source positions before specs extract values, while preserving the same
        transform interface as other factories.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``hrtf.select``.
        **kwargs : object
            Keyword arguments forwarded to ``hrtf.select``.

        Returns
        -------
        callable
            Callable that accepts an HRTF object and returns the selected HRTF.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that applies a named window to HRTF IR data.

        The factory stores the window name and returns a callable suitable for
        ``dataset_hrtf_transform``. Each loaded subject HRTF receives the same window
        operation before spec values are selected.

        Parameters
        ----------
        window_name : str
            Window identifier forwarded to
            :meth:`hrtfpykit.hrtf.transforms.Transform.apply_window`.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that pads HRTF IR data.

        The factory stores padding length, location, and fill value, then forwards
        them to the HRTF transform API for every loaded subject. It applies IR length
        adjustment before extraction.

        Parameters
        ----------
        padding_length : int
            Number of samples added to each impulse response.
        location : {"start", "end"}, default="end"
            Side of the IR sample axis where padding is applied.
        value : float, default=0
            Constant value used in the padded region.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that upsamples HRTF data.

        The returned callable forwards the requested sample rate to the HRTF transform
        layer. Using it at dataset construction keeps all subject HRTFs in the same
        transformed sampling context.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in hertz. It is forwarded to the HRTF
            upsampling transform for every loaded subject.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that downsamples HRTF data.

        The returned callable forwards the requested sample rate to the HRTF transform
        layer. This supports model inputs that use a lower acoustic sample rate than the
        source dataset files.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in hertz. It is forwarded to the HRTF
            downsampling transform for every loaded subject.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that applies an FIR filter.

        The factory captures filter type, cutoff, tap count, and optional window, then
        applies that FIR filter to every loaded subject HRTF. It keeps filtering
        configuration close to dataset construction.

        Parameters
        ----------
        filter : str
            FIR filter type accepted by the HRTF transform layer.
        cutoff : float | tuple[float, float] | None, default=None
            Cutoff frequency or frequency pair in hertz.
        num_taps : int, default=101
            FIR tap count used to design the filter.
        window : str | None, default=None
            Optional FIR design window.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that applies an IIR filter.

        The factory captures filter type, cutoff, and order, then applies that IIR
        filter to every loaded subject HRTF. It applies dataset-level preprocessing.

        Parameters
        ----------
        filter : str
            IIR filter type accepted by the HRTF transform layer.
        cutoff : float | tuple[float, float] | None, default=None
            Cutoff frequency or frequency pair in hertz.
        order : int, default=10
            Butterworth filter order.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that converts HRTF data to minimum phase.

        The factory stores minimum-phase method parameters and applies the conversion
        to every loaded subject HRTF. It lets datasets expose transformed acoustics
        without modifying original SOFA files.

        Parameters
        ----------
        method : str, default="homomorphic"
            Minimum-phase reconstruction method passed to the HRTF transform.
        fft_length : int | None, default=None
            Optional FFT length used during minimum-phase reconstruction.
        epsilon : float, default=1e-12
            Positive numerical floor used by the reconstruction routine.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that converts HRTF data to common transfer function form.

        The factory stores CTF averaging options and applies them across subjects before
        specs consume common-transfer-function-normalized HRTFs.

        Parameters
        ----------
        weights : bool, default=False
            Whether source-position weights are used when estimating the common
            transfer function.
        magnitude_average : {"log", "linear"}, default="log"
            Magnitude averaging rule used by the CTF transform.
        attenuation : float | None, default=None
            Optional attenuation in decibels applied by the CTF transform.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that converts HRTF data to directional transfer function
        form.

        The factory stores DTF averaging options and applies them across subjects before
        specs consume directional-transfer-function-normalized HRTFs.

        Parameters
        ----------
        weights : bool, default=False
            Whether source-position weights are used when estimating the common
            transfer function removed from the HRTF.
        magnitude_average : {"log", "linear"}, default="log"
            Magnitude averaging rule used during DTF calculation.
        attenuation : float | None, default=None
            Optional attenuation in decibels applied by the DTF transform.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that replaces IR data.

        The factory stores replacement IR-like data and forwards it to the HRTF
        transform layer. It supports controlled experiments where dataset
        resources provide metadata/context but acoustic arrays are replaced.

        Parameters
        ----------
        new_ir : np.ndarray | IR | HRTF
            Replacement time-domain data forwarded to
            :meth:`hrtfpykit.hrtf.transforms.Transform.modify_ir`.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that replaces phase data.

        The factory stores replacement phase data and unit metadata, then forwards
        both to the HRTF transform layer. It keeps phase modification reusable across
        every loaded subject.

        Parameters
        ----------
        new_phase : np.ndarray
            Replacement phase array with the same TF layout expected by the
            HRTF transform layer.
        unit : {"degrees", "radians"}, default="degrees"
            Angular unit used by ``new_phase``.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that replaces TF data.

        The factory stores replacement transfer-function data and forwards it to the
        HRTF transform layer. It supports experiments that operate directly on
        frequency-domain representations.

        Parameters
        ----------
        new_tf : np.ndarray | TF | HRTF
            Replacement complex frequency-domain data forwarded to the HRTF
            transform layer.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that replaces magnitude data.

        The factory stores replacement magnitude data and scale information, then
        forwards both to the HRTF transform layer. It allows magnitude-domain
        preprocessing to be expressed as a dataset transform.

        Parameters
        ----------
        new_magnitude : np.ndarray
            Replacement magnitude array with the same TF layout expected by
            the HRTF transform layer.
        scale : {"linear", "db"}, default="linear"
            Magnitude scale used by ``new_magnitude``.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that applies gain to HRTF data.

        The factory stores gain and scale arguments and applies them uniformly to
        every loaded subject. It normalizes or perturbs amplitudes
        during dataset construction.

        Parameters
        ----------
        gain : float | np.ndarray
            Scalar or broadcast-compatible gain applied to each loaded HRTF.
        scale : {"db", "linear"}, default="db"
            Scale used to interpret ``gain``.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that changes FFT length.

        The factory stores a target FFT length and forwards it to the HRTF transform
        layer. Use it when frequency-domain specs need a shared bin count
        across loaded resources.

        Parameters
        ----------
        new_fft_length : int
            FFT length used when recomputing the frequency-domain
            representation.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that changes source coordinate system metadata.

        The factory stores the target coordinate system and applies the conversion or
        metadata update through the HRTF transform API. It keeps coordinate handling
        explicit at dataset construction time.

        Parameters
        ----------
        coordinate_system : {"spherical", "cartesian", "lateral-polar"}
            Source coordinate system requested for each loaded HRTF.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that adds interaural time difference.

        The factory stores ITD amount and unit, then applies the shift to every loaded
        subject HRTF. It supports controlled binaural timing perturbations in dataset
        workflows.

        Parameters
        ----------
        itd : float
            Interaural time difference added to each loaded HRTF.
        unit : {"samples", "seconds"}, default="samples"
            Unit used by ``itd``.

        Returns
        -------
        callable
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
        """Create a transform that removes interaural time difference.

        The factory stores ITD estimation parameters and removes timing differences
        through the HRTF transform API. Use it for datasets that should expose
        ITD-normalized acoustics.

        Parameters
        ----------
        method : {"threshold", "maxiacce"}, default="threshold"
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
            Dataset-level transform callable accepting and returning an HRTF object.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> from hrtfpykit.datasets.transforms import HRTFTransform
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
