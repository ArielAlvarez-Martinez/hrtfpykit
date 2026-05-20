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
        """Define one HRTF or HRIR array returned by a dataset sample.

        :class:`~hrtfpykit.datasets.HRTFSpec` is the main acoustic value spec for
        dataset classes such as :class:`~hrtfpykit.datasets.HUTUBS` and
        :class:`~hrtfpykit.datasets.SONICOM`. It tells a dataset which HRTF
        representation should be extracted, which source positions and ears should
        be kept, which axes should create dataset rows, and which row encodings
        should be added to sample inputs. The spec is a configuration object; it
        does not load files by itself.

        During dataset construction, the spec makes the HRTF resource family
        required and contributes to the row layout. During indexing, the dataset
        loads the subject :class:`~hrtfpykit.hrtf.HRTF` object, applies the
        optional dataset level HRTF transform, applies this spec transform when
        provided, and extracts the requested IR or TF value.

        The ``transform`` callable is used when this spec should read a modified
        HRTF version before extracting values. It receives the loaded
        :class:`~hrtfpykit.hrtf.HRTF` object and must return the HRTF object that
        should be used by this spec. It does not receive the final NumPy array.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"hrtf"``. The returned value is a
        :class:`numpy.ndarray`.

        The output axes follow the selected HRTF state. Axes named in
        ``index_by`` are selected from the current dataset row and removed from the
        returned value. Axes not named in ``index_by`` remain in the output. The
        natural acoustic order is source positions, ears, and then samples for
        ``domain="time"`` or frequency bins for ``domain="frequency"``. Selecting
        one ear with ``ears="left"`` or ``ears="right"`` squeezes the ear axis
        unless ``ear`` is part of ``index_by``.

        Parameters
        ----------
        domain : {``time``, ``frequency``}, default=``time``
            Acoustic domain to return. ``time`` returns HRIR sample data;
            ``frequency`` returns HRTF frequency data.
        signal : str, default=``ir``
            Signal component to extract from the loaded :class:`~hrtfpykit.hrtf.HRTF` object.
        positions : {``all``} or sequence of int, default=``all``
            Source position indices to include.
        plane : str, tuple, dict, or None, default=None
            Optional horizontal, median, or frontal plane selector.
        ears : {``both``, ``left``, ``right``} or sequence of str, default=``both``
            Ear axis selection when the spec is indexed by ear.
        index_by : str or tuple of str, default=(``subject``,)
            Dataset row axes for this spec. Supported axes depend on domain.
        position_one_hot, position_index, ear_one_hot, ear_index : bool, default=False
            Whether row context encodings are exposed in the sample inputs.
        frequency_one_hot, frequency_index, sample_one_hot, sample_index : bool, default=False
            Whether frequency/sample context encodings are exposed in the sample
            inputs.
        transform : callable or None, default=None
            Optional HRTF transform applied after the dataset level
            ``dataset_hrtf_transform`` and before IR or TF values are extracted.
        name : str or None, default=None
            Optional public key used in sample inputs or sample targets.

        Returns
        -------
        HRTFSpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HRTFSpec, HUTUBS
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(
        ...         domain="frequency",
        ...         signal="tf_magnitude_db",
        ...         ears="left",
        ...         index_by=("subject",),
        ...         name="hrtf",
        ...     ),
        ... )
        >>> sample = dataset[0]
        >>> hrtf = sample["inputs"]["hrtf"]
        >>> print(type(hrtf).__name__)
        ndarray
        >>> print(hrtf.shape)
        (440, 129)
        >>> print(np.round(hrtf[:2, :3], 2))  # doctest: +ELLIPSIS
        [[...]]
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
        """Define interaural time difference values returned by a sample.

        :class:`~hrtfpykit.datasets.ITDSpec` derives ITD values from the HRTF
        resource selected by the dataset. It does not store ITDs itself. During
        indexing, the dataset loads the subject :class:`~hrtfpykit.hrtf.HRTF`,
        applies the optional dataset level HRTF transform, applies this spec
        transform when provided, computes ITD from the resulting HRIR data, and
        returns the selected value.

        The ``transform`` callable is used when ITD should be calculated from a
        modified HRTF version. It receives the loaded
        :class:`~hrtfpykit.hrtf.HRTF` object before ITD calculation and must
        return the HRTF object that should be used for the metric. It does not
        receive the calculated ITD array.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"itd"``. The returned value is a :class:`numpy.ndarray`.

        With ``index_by=("subject",)``, the output is a vector with one ITD value
        per selected source position. With ``index_by=("subject", "position")``,
        each dataset row selects one source position and the output is a 0D array.
        ``position_index`` and ``position_one_hot`` add the
        same row context to sample inputs when requested.

        Parameters
        ----------
        positions : {``all``} or sequence of int, default=``all``
            Source position indices used before ITD calculation.
        plane : str, tuple, dict, or None, default=None
            Optional plane selector used instead of explicit position indices.
        index_by : str or tuple of str, default=(``subject``,)
            Dataset row axes. ITD supports subject only and position indexed rows.
        position_one_hot, position_index : bool, default=False
            Whether position context encodings are exposed in sample inputs.
        method : str, default=``threshold``
            ITD estimation method forwarded to the DSP metric.
        output : str, default=``samples``
            Output unit or representation requested from the ITD metric.
        thresh_level : float, default=-10.0
            Threshold level used by threshold based ITD methods.
        upper_cut_freq : float, default=3000.0
            Upper cutoff frequency used by filtered ITD methods.
        filter_order : int, default=10
            Filter order used by filtered ITD methods.
        transform : callable or None, default=None
            Optional HRTF transform applied before ITD calculation. This transform
            receives the loaded HRTF object, not the calculated ITD value.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        ITDSpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS, ITDSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=ITDSpec(
        ...         index_by=("subject", "position"),
        ...         position_index=True,
        ...         output="samples",
        ...         name="itd",
        ...     ),
        ... )
        >>> sample = dataset[0]
        >>> itd_value = sample["inputs"]["itd"]
        >>> print(type(itd_value).__name__)
        ndarray
        >>> print(itd_value.shape)
        ()
        >>> print(sample["inputs"]["position_index"])
        0
        >>> np.asarray(np.round(itd_value, 3))  # doctest: +ELLIPSIS
        array(...)
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
        """Define interaural level difference values returned by a sample.

        :class:`~hrtfpykit.datasets.ILDSpec` derives ILD values from the HRTF
        resource selected by the dataset. It does not store ILDs itself. During
        indexing, the dataset loads the subject :class:`~hrtfpykit.hrtf.HRTF`,
        applies the optional dataset level HRTF transform, applies this spec
        transform when provided, computes ILD from the resulting HRIR data, and
        returns the selected value.

        The ``transform`` callable is used when ILD should be calculated from a
        modified HRTF version. It receives the loaded
        :class:`~hrtfpykit.hrtf.HRTF` object before ILD calculation and must
        return the HRTF object that should be used for the metric. It does not
        receive the calculated ILD array.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"ild"``. The returned value is a :class:`numpy.ndarray`.

        In ``mode="broad-band"``, subject only rows return one value per selected
        source position, while position indexed rows return one 0D array. In
        ``mode="frequency-dependent"``, a frequency axis
        is kept unless ``frequency`` is included in ``index_by``.

        Parameters
        ----------
        positions : {``all``} or sequence of int, default=``all``
            Source position indices used before ILD calculation.
        plane : str, tuple, dict, or None, default=None
            Optional plane selector used instead of explicit position indices.
        index_by : str or tuple of str, default=(``subject``,)
            Dataset row axes. Frequency indexing requires ``mode`` set to
            ``frequency-dependent``.
        position_one_hot, position_index : bool, default=False
            Whether position context encodings are exposed in sample inputs.
        frequency_one_hot, frequency_index : bool, default=False
            Whether frequency context encodings are exposed in sample inputs.
        mode : str, default=``broad-band``
            ILD mode forwarded to the DSP metric.
        output : str, default=``db``
            Output scale or representation requested from the ILD metric.
        fft_length : int or None, default=None
            FFT length used for frequency dependent ILD calculation.
        epsilon : float, default=1e-12
            Numerical floor used by level ratio calculations.
        transform : callable or None, default=None
            Optional HRTF transform applied before ILD calculation. This transform
            receives the loaded HRTF object, not the calculated ILD value.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        ILDSpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS, ILDSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=ILDSpec(
        ...         mode="broad-band",
        ...         index_by=("subject", "position"),
        ...         position_index=True,
        ...         name="ild",
        ...     ),
        ... )
        >>> sample = dataset[0]
        >>> ild_value = sample["inputs"]["ild"]
        >>> print(type(ild_value).__name__)
        ndarray
        >>> print(ild_value.shape)
        ()
        >>> print(sample["inputs"]["position_index"])
        0
        >>> np.asarray(np.round(ild_value, 3))  # doctest: +ELLIPSIS
        array(...)
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
        """Define spherical harmonic HRTF coefficients returned by a sample.

        :class:`~hrtfpykit.datasets.SHSpec` derives spherical harmonic
        coefficients from the HRTF resource selected by the dataset. During
        indexing, the dataset loads the subject :class:`~hrtfpykit.hrtf.HRTF`,
        applies the optional dataset level HRTF transform, applies this spec
        transform when provided, and runs the spherical harmonic transform on the
        resulting HRTF state.

        The ``transform`` callable is used when SH coefficients should be
        calculated from a modified HRTF version. It receives the loaded
        :class:`~hrtfpykit.hrtf.HRTF` object before the spherical harmonic
        transform and must return the HRTF object that should be used. It does not
        receive the calculated coefficient array.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"sh"``. The returned value is a :class:`numpy.ndarray`.

        The first output axis is always the coefficient axis and has
        ``(sh_order + 1) ** 2`` values. With ``ears="both"``, the output keeps an
        ear axis between coefficients and frequency bins. With ``ears="left"`` or
        ``ears="right"``, the ear axis is squeezed unless ``ear`` is included in
        ``index_by``. Including ``frequency`` in ``index_by`` selects one frequency
        bin per row.

        Parameters
        ----------
        sh_order : int
            Spherical harmonic order used for the decomposition.
        ears : {``both``, ``left``, ``right``} or sequence of str, default=``both``
            Ear selection when the dataset is indexed by ear.
        index_by : str or tuple of str, default=(``subject``,)
            Dataset row axes. SH specs support ear and frequency indexing.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        frequency_one_hot, frequency_index : bool, default=False
            Whether frequency context encodings are exposed in sample inputs.
        epsilon : float, default=1e-6
            Numerical regularization used by the SH transform.
        transform : callable or None, default=None
            Optional HRTF transform applied before spherical harmonic
            decomposition. This transform receives the loaded HRTF object, not the
            calculated coefficient array.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        SHSpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.datasets import HUTUBS, SHSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=SHSpec(
        ...         sh_order=9,
        ...         ears="left",
        ...         index_by=("subject",),
        ...         name="sh",
        ...     ),
        ... )
        >>> sh_value = dataset[0]["inputs"]["sh"]
        >>> print(type(sh_value).__name__)
        ndarray
        >>> print(sh_value.shape)
        (100, 129)
        >>> print(np.round(sh_value[:2, :3], 3))  # doctest: +ELLIPSIS
        [[...]]
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
        """Define subject mesh paths returned by a dataset sample.

        :class:`~hrtfpykit.datasets.MeshSpec` asks a dataset to include the mesh
        resource associated with each selected subject. The dataset can use mesh
        resources declared by a public dataset, such as SONICOM meshes, or a
        custom mesh root passed through ``path``. Custom mesh roots still use the
        mesh filename pattern declared by the dataset configuration, so subject
        IDs and subject numbers remain aligned with the HRTF resources. The
        dataset resolves the configured mesh variant, optional path override,
        allowed extensions, excluded subjects, and split selection before sample
        extraction.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"mesh"``. By default, the returned value is a ``str``
        path. The mesh file is not parsed by hrtfpykit unless ``transform`` is
        provided, in which case the returned type is whatever the transform returns.

        This path first behavior is intentional. hrtfpykit organizes the mesh
        resources and keeps the subject alignment, while the user decides how the
        mesh should be opened, parsed, preprocessed, or converted for a particular
        experiment. The ``transform`` callable is the hook for that custom mesh
        pipeline.

        Mesh specs affect subject availability. When a mesh spec is requested,
        subjects without a matching mesh file are removed before rows are built.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Optional root path overriding the dataset mesh location. Relative
            paths are resolved from the dataset root.
        extensions : tuple of str or None, default=None
            Optional mesh extensions to search.
        transform : callable or None, default=None
            Optional transform applied to the selected mesh path string. Use it to
            define how the mesh resource should be opened, preprocessed, or
            converted before it is returned in the sample.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        MeshSpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> from pathlib import Path
        >>> from hrtfpykit.datasets import HUTUBS, MeshSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=MeshSpec(name="mesh"))
        >>> mesh_path = dataset[0]["inputs"]["mesh"]
        >>> print(type(mesh_path).__name__)
        str
        >>> print(Path(mesh_path).suffix)
        .ply
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
        """Define subject anthropometry values returned by a sample.

        :class:`~hrtfpykit.datasets.AnthropometrySpec` asks a dataset to load
        physical measurement data such as head, pinna, or ear measurements and
        align those values to the selected subject IDs. The table can be the
        anthropometry resource declared by the dataset, or a custom table passed
        through ``path``. This makes the spec useful both for official HUTUBS or
        SONICOM measurements and for experiment-specific measurements extracted
        from another source. The spec controls table access direction,
        subject-id handling, ear grouping, row or column exclusion, and optional
        value transforms.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"anthropometry"``.

        The default output type depends on the loaded anthropometry layout. For
        HUTUBS, the default anthropometry resource returns a ``dict`` mapping
        measurement names to values. If the source is a matrix style file, the
        output can be a :class:`numpy.ndarray`. If ``transform`` is provided, the
        returned type is whatever the transform returns. HUTUBS also filters
        left and right prefixed fields when the spec or row carries an ear context.

        The ``transform`` callable receives the selected anthropometry value after
        subject and optional ear selection. Use it when the selected value should
        be reshaped, filtered, normalized, or converted into the structure expected
        by the custom pipeline.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Optional table path overriding the dataset configuration. Relative
            paths are resolved from the dataset root.
        extensions : tuple of str or None, default=None
            Optional table extensions to allow.
        exclude_row, exclude_column : int, sequence of int, or None, default=None
            Row or column indices to remove while loading the table.
        accessed_by : {``row``, ``column``}, default=``row``
            Whether subjects are represented by rows or columns.
        grouped_by : {``subject``} or (``subject``, ``ear``), default=(``subject``,)
            Dataset grouping used to select anthropometry values.
        subject_id : bool, default=True
            Whether the table includes a leading subject identifier row or column.
        ear : {``both``, ``left``, ``right``} or None, default=None
            Optional ear selection for ear-grouped anthropometry.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        transform : callable or None, default=None
            Optional transform applied to the selected table value after subject
            and optional ear selection.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        AnthropometrySpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import AnthropometrySpec, HUTUBS
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=AnthropometrySpec(name="anthropometry"),
        ... )
        >>> anthropometry = dataset[0]["inputs"]["anthropometry"]
        >>> print(type(anthropometry).__name__)
        dict
        >>> print(list(anthropometry)[:3])  # doctest: +ELLIPSIS
        [...]
        >>> print({key: anthropometry[key] for key in list(anthropometry)[:2]})  # doctest: +ELLIPSIS
        {...}
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
        """Define subject metadata values returned by a sample.

        :class:`~hrtfpykit.datasets.MetadataSpec` asks a dataset to load general
        subject annotations and align them to the selected subject IDs. Metadata
        can come from a public dataset configuration, such as SONICOM metadata, or
        from a custom table passed through ``path``. Metadata is kept separate
        from :class:`~hrtfpykit.datasets.AnthropometrySpec` so a dataset can
        expose physical measurements and general annotations under different
        resource families and sample keys.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"metadata"``.

        The default output type depends on the loaded metadata layout. A row based
        CSV usually returns a ``dict`` mapping metadata field names to values. A
        matrix style file can return a :class:`numpy.ndarray` or scalar value. If
        ``transform`` is provided, the returned type is whatever the transform
        returns. HUTUBS does not declare an official metadata resource in its
        hrtfpykit config; use a dataset that declares metadata, such as SONICOM, or
        provide a custom metadata path.

        The ``transform`` callable receives the selected metadata value after
        subject and optional ear selection. Use it when the selected value should
        be reshaped, filtered, normalized, or converted into the structure expected
        by the custom pipeline.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Optional table path overriding the dataset configuration. Relative
            paths are resolved from the dataset root.
        extensions : tuple of str or None, default=None
            Optional table extensions to allow.
        exclude_row, exclude_column : int, sequence of int, or None, default=None
            Row or column indices to remove while loading the table.
        accessed_by : {``row``, ``column``}, default=``row``
            Whether subjects are represented by rows or columns.
        grouped_by : {``subject``} or (``subject``, ``ear``), default=(``subject``,)
            Dataset grouping used to select metadata values.
        subject_id : bool, default=True
            Whether the table includes a leading subject identifier row or column.
        ear : {``both``, ``left``, ``right``} or None, default=None
            Optional ear selection for ear-grouped metadata.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        transform : callable or None, default=None
            Optional transform applied to the selected metadata value after subject
            and optional ear selection.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        MetadataSpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> from hrtfpykit.datasets import MetadataSpec, SONICOM
        >>> dataset = SONICOM(
        ...     root="datasets/sonicom",
        ...     inputs=MetadataSpec(name="metadata"),
        ... )
        >>> metadata = dataset[0]["inputs"]["metadata"]
        >>> print(type(metadata).__name__)
        dict
        >>> print(list(metadata)[:3])  # doctest: +ELLIPSIS
        [...]
        >>> print({key: metadata[key] for key in list(metadata)[:2]})  # doctest: +ELLIPSIS
        {...}
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
        """Define subject image paths or transformed image values.

        :class:`~hrtfpykit.datasets.ImageSpec` asks a dataset to index image files
        by subject, or by subject and ear when images are stored in left and right
        groups. The image files can be custom resources prepared for an experiment,
        for example ear images rendered from 3D meshes and used as inputs for HRTF
        individualization. The dataset scans the image root, intersects available
        subjects with the other requested resources, and returns the selected
        image value in each sample. HUTUBS declares image scanning support, but
        hrtfpykit does not ship official HUTUBS image files; ``path`` should point
        to the local image folder you want to align with the dataset subjects.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"image"``.

        By default, the returned value is a ``str`` path when one file matches the
        subject key, or a ``list[str]`` when multiple files match. The image is not
        opened by hrtfpykit unless ``transform`` is provided. A transform receives
        each image path and returns the image representation chosen by the user.
        When ``concatenate=True``, a transform is required and the transformed
        values are concatenated with :func:`numpy.concatenate` along axis 0.

        This path first behavior is intentional. hrtfpykit organizes image files
        by subject or by subject and ear, while the user decides how images should
        be opened, resized, normalized, augmented, or converted for a particular
        experiment. The ``transform`` callable is the hook for that custom image
        pipeline.

        Image roots must contain one folder per dataset subject. Subject folders
        can be named with the canonical dataset subject ID, ``subjectN``, or
        ``subject_N``. When ``grouped_by=("subject", "ear")``, each subject
        folder must contain ear folders such as ``left`` and ``right``. Files are
        discovered recursively inside the matched subject or ear folder.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Root path containing custom image resources. Relative paths are
            resolved from the dataset root.
        grouped_by : {``subject``} or (``subject``, ``ear``), default=(``subject``,)
            Whether images are grouped only by subject or by subject and ear.
        extensions : tuple of str or None, default=None
            Optional image extensions to search.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        concatenate : bool, default=False
            Whether transformed image values should be concatenated by the value selector.
        transform : callable or None, default=None
            Optional transform applied to each image path string. Use it to define
            how each image should be opened, preprocessed, augmented, or converted
            before it is returned in the sample.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        ImageSpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> from pathlib import Path
        >>> from hrtfpykit.datasets import HUTUBS, ImageSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=ImageSpec(
        ...         path="ear_images",
        ...         grouped_by="subject",
        ...         name="image",
        ...     ),
        ... )
        >>> image_value = dataset[0]["inputs"]["image"]
        >>> first_image_path = image_value[0] if isinstance(image_value, list) else image_value
        >>> print(type(first_image_path).__name__)
        str
        >>> print(Path(first_image_path).suffix)
        .png
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
        """Define subject video paths or transformed video values.

        :class:`~hrtfpykit.datasets.VideoSpec` asks a dataset to index video files
        by subject, or by subject and ear when videos are stored in left and right
        groups. The video files can be custom resources prepared for an experiment,
        for example turntable renders, measurement recordings, or other visual
        captures aligned with the HRTF subject IDs. The dataset scans the video
        root, intersects available subjects with the other requested resources,
        and returns the selected video value in each sample. HUTUBS declares video
        scanning support, but hrtfpykit does not ship official HUTUBS video files;
        ``path`` should point to the local video folder you want to align with the
        dataset subjects.

        If the spec is passed to ``inputs``, its value appears under
        ``dataset[0]["inputs"][name]``. If it is passed to ``target``, its value
        appears under ``dataset[0]["target"][name]``. When ``name`` is None, the
        default key is ``"video"``.

        By default, the returned value is a ``str`` path when one file matches the
        subject key, or a ``list[str]`` when multiple files match. The video is not
        decoded by hrtfpykit unless ``transform`` is provided. A transform receives
        each video path and returns the video representation chosen by the user.

        This path first behavior is intentional. hrtfpykit organizes video files
        by subject or by subject and ear, while the user decides how videos should
        be opened, sampled, preprocessed, augmented, or converted for a particular
        experiment. The ``transform`` callable is the hook for that custom video
        pipeline.

        Video roots must contain one folder per dataset subject. Subject folders
        can be named with the canonical dataset subject ID, ``subjectN``, or
        ``subject_N``. When ``grouped_by=("subject", "ear")``, each subject
        folder must contain ear folders such as ``left`` and ``right``. Files are
        discovered recursively inside the matched subject or ear folder.

        Parameters
        ----------
        path : str, Path, or None, default=None
            Root path containing custom video resources. Relative paths are
            resolved from the dataset root.
        grouped_by : {``subject``} or (``subject``, ``ear``), default=(``subject``,)
            Whether videos are grouped only by subject or by subject and ear.
        extensions : tuple of str or None, default=None
            Optional video extensions to search.
        ear_one_hot, ear_index : bool, default=False
            Whether ear context encodings are exposed in sample inputs.
        transform : callable or None, default=None
            Optional transform applied to each video path string. Use it to define
            how each video should be opened, sampled, preprocessed, or converted
            before it is returned in the sample.
        name : str or None, default=None
            Optional public key used in sample dictionaries.

        Returns
        -------
        VideoSpec
            Specification object consumed by dataset construction.

        Examples
        --------
        >>> from pathlib import Path
        >>> from hrtfpykit.datasets import HUTUBS, VideoSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=VideoSpec(
        ...         path="ear_videos",
        ...         grouped_by="subject",
        ...         name="video",
        ...     ),
        ... )
        >>> video_value = dataset[0]["inputs"]["video"]
        >>> first_video_path = video_value[0] if isinstance(video_value, list) else video_value
        >>> print(type(first_video_path).__name__)
        str
        >>> print(Path(first_video_path).suffix)
        .mp4
        """
        self.path = path
        self.grouped_by = grouped_by
        self.extensions = extensions
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.transform = transform
        self.name = name
