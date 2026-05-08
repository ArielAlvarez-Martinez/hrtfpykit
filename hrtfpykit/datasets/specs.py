from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


class HRTFSpec:

    def __init__(
        self,
        domain: str = "time",
        signal: str = "ir",
        positions: str | tuple[int, ...] | list[int] | np.ndarray = "all",
        plane: str | tuple[object, ...] | dict[str, object] | None = None,
        ears: str | tuple[str, ...] = "both",
        index_by: str | tuple[str, ...] = ("subject",),
        position_one_hot: bool = False,
        position_index: bool = False,
        ear_one_hot: bool = False,
        ear_index: bool = False,
        frequency_one_hot: bool = False,
        frequency_index: bool = False,
        sample_one_hot: bool = False,
        sample_index: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose HRTF or HRIR values.

        ``HRTFSpec`` is the main acoustic dataset specification. It tells a dataset
        which HRTF representation to read, which source positions or plane to keep,
        which ears to use, which axes create dataset rows, and which optional
        encodings should be added to ``sample['inputs']``. The spec does not load
        files by itself; it is consumed by dataset classes such as ``HUTUBS`` and
        ``SONICOM`` during construction and by ``__getitem__`` when samples are
        extracted.

        Parameters
        ----------
        domain : {'time', 'frequency'}, default='time'
            Acoustic domain to return. ``'time'`` returns HRIR-style sample data;
            ``'frequency'`` returns HRTF-style frequency data.
        signal : str, default='ir'
            Signal component to extract from the loaded HRTF object.
        positions : {'all'} or sequence of int, default='all'
            Source-position indices to include.
        plane : str, tuple, dict, or None, default=None
            Optional horizontal, median, or frontal plane selector.
        ears : {'both', 'left', 'right'} or sequence of str, default='both'
            Ear axis selection when the spec is indexed by ear.
        index_by : str or tuple of str, default=('subject',)
            Dataset row axes for this spec. Supported axes depend on ``domain``.
        position_one_hot, position_index, ear_one_hot, ear_index : bool, default=False
            Whether row context encodings are exposed in the sample inputs.
        frequency_one_hot, frequency_index, sample_one_hot, sample_index : bool, default=False
            Whether frequency/sample context encodings are exposed in the sample
            inputs.
        transform : callable or None, default=None
            Optional value transform applied after dataset-level HRTF loading.
        name : str or None, default=None
            Optional public key used in ``sample['inputs']`` or ``sample['target']``.

        Returns
        -------
        HRTFSpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec(index_by=("subject", "position")))
        >>> sample = dataset[0]
        """
        self.domain = domain
        self.signal = signal
        self.positions = positions
        self.plane = plane
        self.ears = ears
        self.index_by = index_by
        self.position_one_hot = position_one_hot
        self.position_index = position_index
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.frequency_one_hot = frequency_one_hot
        self.frequency_index = frequency_index
        self.sample_one_hot = sample_one_hot
        self.sample_index = sample_index
        self.transform = transform
        self.name = name

class ITDSpec:

    def __init__(
        self,
        positions: str | tuple[int, ...] | list[int] | np.ndarray = "all",
        plane: str | tuple[object, ...] | dict[str, object] | None = None,
        index_by: str | tuple[str, ...] = ("subject",),
        position_one_hot: bool = False,
        position_index: bool = False,
        method: str = "threshold",
        output: str = "samples",
        thresh_level: float = -10.0,
        upper_cut_freq: float = 3000.0,
        filter_order: int = 10,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose interaural time difference values.

        ``ITDSpec`` derives ITD values from the HRTF files selected by the dataset. It
        derives binaural timing features from dataset HRTFs
        instead of full HRIR/HRTF arrays. The spec controls the position subset, the
        optional plane selection, the row indexing axes, and the ITD calculation
        parameters. It is evaluated through the dataset sample pipeline, not as a
        standalone object.

        Parameters
        ----------
        positions : {'all'} or sequence of int, default='all'
            Source-position indices used before ITD calculation.
        plane : str, tuple, dict, or None, default=None
            Optional plane selector used instead of explicit position indices.
        index_by : str or tuple of str, default=('subject',)
            Dataset row axes. ITD supports subject-only and position-indexed rows.
        position_one_hot, position_index : bool, default=False
            Whether position context encodings are exposed in sample inputs.
        method : str, default='threshold'
            ITD estimation method forwarded to the DSP metric.
        output : str, default='samples'
            Output unit or representation requested from the ITD metric.
        thresh_level : float, default=-10.0
            Threshold level used by threshold-based ITD methods.
        upper_cut_freq : float, default=3000.0
            Upper cutoff frequency used by filtered ITD methods.
        filter_order : int, default=10
            Filter order used by filtered ITD methods.
        transform : callable or None, default=None
            Optional transform applied to the calculated ITD value.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        ITDSpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import ITDSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=ITDSpec(index_by=("subject", "position")))
        >>> itd_value = dataset[0]["inputs"]["itd"]
        """
        self.positions = positions
        self.plane = plane
        self.index_by = index_by
        self.position_one_hot = position_one_hot
        self.position_index = position_index
        self.method = method
        self.output = output
        self.thresh_level = thresh_level
        self.upper_cut_freq = upper_cut_freq
        self.filter_order = filter_order
        self.transform = transform
        self.name = name

class ILDSpec:

    def __init__(
        self,
        positions: str | tuple[int, ...] | list[int] | np.ndarray = "all",
        plane: str | tuple[object, ...] | dict[str, object] | None = None,
        index_by: str | tuple[str, ...] = ("subject",),
        position_one_hot: bool = False,
        position_index: bool = False,
        frequency_one_hot: bool = False,
        frequency_index: bool = False,
        mode: str = "broad-band",
        output: str = "db",
        fft_length: int | None = None,
        epsilon: float = 1e-12,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose interaural level difference values.

        ``ILDSpec`` derives ILD values from the HRTF files selected by the dataset. It
        derives binaural level cues instead of full
        acoustic arrays. The spec controls the position subset, broad-band or
        frequency-dependent output, row indexing, and numerical parameters used by the
        ILD calculation. It becomes active only when passed into a dataset ``inputs``
        or ``target`` argument.

        Parameters
        ----------
        positions : {'all'} or sequence of int, default='all'
            Source-position indices used before ILD calculation.
        plane : str, tuple, dict, or None, default=None
            Optional plane selector used instead of explicit position indices.
        index_by : str or tuple of str, default=('subject',)
            Dataset row axes. Frequency indexing requires ``mode='frequency-
            dependent'``.
        position_one_hot, position_index : bool, default=False
            Whether position context encodings are exposed in sample inputs.
        frequency_one_hot, frequency_index : bool, default=False
            Whether frequency context encodings are exposed in sample inputs.
        mode : str, default='broad-band'
            ILD mode forwarded to the DSP metric.
        output : str, default='db'
            Output scale or representation requested from the ILD metric.
        fft_length : int or None, default=None
            FFT length used for frequency-dependent ILD calculation.
        epsilon : float, default=1e-12
            Numerical floor used by level-ratio calculations.
        transform : callable or None, default=None
            Optional transform applied to the calculated ILD value.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        ILDSpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import ILDSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=ILDSpec(mode="frequency-dependent", index_by=("subject", "frequency")),
        ... )
        >>> ild_value = dataset[0]["inputs"]["ild"]
        """
        self.positions = positions
        self.plane = plane
        self.index_by = index_by
        self.position_one_hot = position_one_hot
        self.position_index = position_index
        self.frequency_one_hot = frequency_one_hot
        self.frequency_index = frequency_index
        self.mode = mode
        self.output = output
        self.fft_length = fft_length
        self.epsilon = epsilon
        self.transform = transform
        self.name = name

class SHSpec:

    def __init__(
        self,
        sh_order: int,
        ears: str | tuple[str, ...] = "both",
        index_by: str | tuple[str, ...] = ("subject",),
        ear_one_hot: bool = False,
        ear_index: bool = False,
        frequency_one_hot: bool = False,
        frequency_index: bool = False,
        epsilon: float = 1e-6,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose spherical-harmonic HRTF features.

        ``SHSpec`` converts the dataset-selected HRTF data into spherical-harmonic
        coefficients. The spec lets a model or processing pipeline work in
        a spherical-harmonic basis instead of raw position-indexed HRTFs. The spec
        controls SH order, ear selection, frequency indexing, and optional row
        encodings. It is evaluated by the dataset when samples are requested.

        Parameters
        ----------
        sh_order : int
            Spherical harmonic order used for the decomposition.
        ears : {'both', 'left', 'right'} or sequence of str, default='both'
            Ear selection when the dataset is indexed by ear.
        index_by : str or tuple of str, default=('subject',)
            Dataset row axes. SH specs support ear and frequency indexing.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        frequency_one_hot, frequency_index : bool, default=False
            Whether frequency context encodings are exposed in sample inputs.
        epsilon : float, default=1e-6
            Numerical regularization used by the SH transform.
        transform : callable or None, default=None
            Optional transform applied to the SH value.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        SHSpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import SHSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=SHSpec(sh_order=4, index_by=("subject", "ear", "frequency")),
        ... )
        >>> sh_value = dataset[0]["inputs"]["sh"]
        """
        self.sh_order = sh_order
        self.ears = ears
        self.index_by = index_by
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.frequency_one_hot = frequency_one_hot
        self.frequency_index = frequency_index
        self.epsilon = epsilon
        self.transform = transform
        self.name = name

class MeshSpec:

    def __init__(
        self,
        path: str | Path | None = None,
        extensions: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose subject mesh resources.

        ``MeshSpec`` asks a dataset to include the mesh resource associated with each
        selected subject. The dataset resolves configured mesh variants, optional
        local override paths, extensions, excluded subjects, and split selection
        before sample extraction. The returned value is normally a mesh path unless a
        transform is provided.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Optional root or file path overriding the dataset mesh location.
        extensions : tuple of str or None, default=None
            Optional mesh extensions to search.
        transform : callable or None, default=None
            Optional transform applied to the selected mesh path.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        MeshSpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import SONICOM
        >>> from hrtfpykit.datasets.specs import MeshSpec
        >>> dataset = SONICOM(root="datasets/sonicom", inputs=MeshSpec())
        >>> mesh_path = dataset[0]["inputs"]["mesh"]
        """
        self.path = path
        self.extensions = extensions
        self.transform = transform
        self.name = name

class AnthropometrySpec:

    def __init__(
        self,
        path: str | Path | None = None,
        extensions: tuple[str, ...] | None = None,
        exclude_row: int | Sequence[int] | None = None,
        exclude_column: int | Sequence[int] | None = None,
        accessed_by: str = "row",
        grouped_by: str | tuple[str, ...] = ("subject",),
        subject_id: bool = True,
        ear: str | None = None,
        ear_one_hot: bool = False,
        ear_index: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose anthropometric table values.

        ``AnthropometrySpec`` asks a dataset to load physical measurement tables such
        as head, pinna, or ear measurements and align those rows or columns to the
        dataset subject identifiers. It is separate from ``MetadataSpec`` because
        anthropometry describes physical measurements, while metadata describes
        general annotations. The spec controls table access direction, subject-id
        handling, ear grouping, row/column exclusion, and optional value transforms.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Optional table path overriding the dataset configuration.
        extensions : tuple of str or None, default=None
            Optional table extensions to allow.
        exclude_row, exclude_column : int, sequence of int, or None, default=None
            Row or column indices to remove while loading the table.
        accessed_by : {'row', 'column'}, default='row'
            Whether subjects are represented by rows or columns.
        grouped_by : {'subject'} or ('subject', 'ear'), default=('subject',)
            Dataset grouping used to select anthropometry values.
        subject_id : bool, default=True
            Whether the table includes a leading subject identifier row or column.
        ear : {'both', 'left', 'right'} or None, default=None
            Optional ear selection for ear-grouped anthropometry.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        transform : callable or None, default=None
            Optional transform applied to the selected table value.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        AnthropometrySpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import AnthropometrySpec, HRTFSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=[HRTFSpec(), AnthropometrySpec(grouped_by=("subject", "ear"))],
        ... )
        >>> anthropometry = dataset[0]["inputs"]["anthropometry"]
        """
        self.exclude_row = exclude_row
        self.extensions = extensions
        self.exclude_column = exclude_column
        self.grouped_by = grouped_by
        self.subject_id = subject_id
        self.ear = ear
        self.accessed_by = accessed_by
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.path = path
        self.transform = transform
        self.name = name

class MetadataSpec:

    def __init__(
        self,
        path: str | Path | None = None,
        extensions: tuple[str, ...] | None = None,
        exclude_row: int | Sequence[int] | None = None,
        exclude_column: int | Sequence[int] | None = None,
        accessed_by: str = "row",
        grouped_by: str | tuple[str, ...] = ("subject",),
        subject_id: bool = True,
        ear: str | None = None,
        ear_one_hot: bool = False,
        ear_index: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose general metadata table values.

        ``MetadataSpec`` asks a dataset to load subject or sample annotations and
        align them to the dataset subject identifiers. It shares table-style behavior
        with ``AnthropometrySpec`` but keeps a separate resource identity, state
        field, split intersection, and sample key. This separation allows a dataset to
        expose both physical measurements and general metadata at the same time
        without path or value collisions.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Optional table path overriding the dataset configuration.
        extensions : tuple of str or None, default=None
            Optional table extensions to allow.
        exclude_row, exclude_column : int, sequence of int, or None, default=None
            Row or column indices to remove while loading the table.
        accessed_by : {'row', 'column'}, default='row'
            Whether subjects are represented by rows or columns.
        grouped_by : {'subject'} or ('subject', 'ear'), default=('subject',)
            Dataset grouping used to select metadata values.
        subject_id : bool, default=True
            Whether the table includes a leading subject identifier row or column.
        ear : {'both', 'left', 'right'} or None, default=None
            Optional ear selection for ear-grouped metadata.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        transform : callable or None, default=None
            Optional transform applied to the selected table value.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        MetadataSpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import SONICOM
        >>> from hrtfpykit.datasets.specs import HRTFSpec, MetadataSpec
        >>> dataset = SONICOM(root="datasets/sonicom", inputs=[HRTFSpec(), MetadataSpec()])
        >>> metadata = dataset[0]["inputs"]["metadata"]
        """
        self.exclude_row = exclude_row
        self.extensions = extensions
        self.exclude_column = exclude_column
        self.grouped_by = grouped_by
        self.subject_id = subject_id
        self.ear = ear
        self.accessed_by = accessed_by
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.path = path
        self.transform = transform
        self.name = name

class ImageSpec:

    def __init__(
        self,
        path: str | Path | None = None,
        grouped_by: str | tuple[str, ...] = ("subject",),
        extensions: tuple[str, ...] | None = None,
        ear_one_hot: bool = False,
        ear_index: bool = False,
        concatenate: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose subject image resources.

        ``ImageSpec`` asks a dataset to index image files by subject, or by subject
        and ear when images are stored in left/right groups. The dataset scans the
        image root, intersects available subjects with the other requested resources,
        and returns the selected image paths or transformed image values in each
        sample.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Optional root path overriding the dataset image location.
        grouped_by : {'subject'} or ('subject', 'ear'), default=('subject',)
            Whether images are grouped only by subject or by subject and ear.
        extensions : tuple of str or None, default=None
            Optional image extensions to search.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        concatenate : bool, default=False
            Whether loaded image values should be concatenated by the value selector.
        transform : callable or None, default=None
            Optional transform applied to image values.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        ImageSpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec, ImageSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=[
        ...         HRTFSpec(index_by=("subject", "ear")),
        ...         ImageSpec(path="ear_images", grouped_by=("subject", "ear")),
        ...     ],
        ... )
        >>> image_paths = dataset[0]["inputs"]["image"]
        """
        self.path = path
        self.grouped_by = grouped_by
        self.extensions = extensions
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.concatenate = concatenate
        self.transform = transform
        self.name = name

class VideoSpec:

    def __init__(
        self,
        path: str | Path | None = None,
        grouped_by: str | tuple[str, ...] = ("subject",),
        extensions: tuple[str, ...] | None = None,
        ear_one_hot: bool = False,
        ear_index: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """Define how a dataset should expose subject video resources.

        ``VideoSpec`` asks a dataset to index video files by subject, or by subject
        and ear when videos are stored in left/right groups. The dataset scans the
        video root, intersects available subjects with the other requested resources,
        and returns the selected video paths or transformed video values in each
        sample.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Optional root path overriding the dataset video location.
        grouped_by : {'subject'} or ('subject', 'ear'), default=('subject',)
            Whether videos are grouped only by subject or by subject and ear.
        extensions : tuple of str or None, default=None
            Optional video extensions to search.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        transform : callable or None, default=None
            Optional transform applied to video values.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        VideoSpec Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec, VideoSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=[
        ...         HRTFSpec(index_by=("subject", "ear")),
        ...         VideoSpec(path="ear_videos", grouped_by=("subject", "ear")),
        ...     ],
        ... )
        >>> video_paths = dataset[0]["inputs"]["video"]
        """
        self.path = path
        self.grouped_by = grouped_by
        self.extensions = extensions
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.transform = transform
        self.name = name
