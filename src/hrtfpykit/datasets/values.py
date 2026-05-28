import numpy as np
from typing import TYPE_CHECKING, Any, cast

from .sanitize import sanitize_ears, sanitize_grouped_by, sanitize_index_by
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MetadataSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
)
from .specs_registry import get_spec_descriptor
from ..utils.dsp import imag, magnitude, magnitude_db, phase, real
from ..utils.metrics import ild, itd
from ..utils.sh import sht
from .split import DatasetSplitPlanner

if TYPE_CHECKING:
    from .base import BaseDataset


class DatasetSampleValueSelector:
    """Resolve concrete sample values for dataset specs during indexing.

    :class:`~hrtfpykit.datasets.values.DatasetSampleValueSelector` is the value
    extraction layer used by
    :meth:`~hrtfpykit.datasets.base.BaseDataset.__getitem__`. For each row and
    input or target spec, it reads the constructed
    :class:`~hrtfpykit.datasets.state.DatasetState`, resolves the requested
    resource or derived acoustic representation, applies row-level indexing, and
    returns the value placed under the spec name in the sample inputs or sample
    target dictionary.

    The selector is intentionally stateless. Caching, resource paths, selected axes,
    loaded table rows, and acoustic context are all stored on the dataset state.
    Dataset subclasses may provide methods with the same value-method names as the
    registry descriptors to customize one resource family while leaving the generic
    fallback implementations available.

    Notes
    -----
    The class is a namespace of static methods because value resolution is driven by
    a dataset instance, a spec instance, and one row dictionary. It does not own
    resources and does not mutate the row table.

    """

    @staticmethod
    def get_sample_value(
        dataset: "BaseDataset",
        spec: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        """Dispatch one spec to its registered value selector.

        This method is the central runtime dispatcher used by
        :meth:`~hrtfpykit.datasets.base.BaseDataset.__getitem__`. It asks the spec
        registry which value method belongs to the spec, prefers a dataset subclass
        override when one exists, and otherwise calls the generic method on
        :class:`~hrtfpykit.datasets.values.DatasetSampleValueSelector`.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns state, resources, and optional subclass
            selector overrides.
        spec : dataset spec
            Spec object to resolve. The spec type must be registered in
            the dataset specs registry.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context containing selected subject, position, ear, frequency, and
            sample indices.

        Returns
        -------
        object
            Concrete sample value for the given spec and row.

        Raises
        ------
        ValueError
            If the spec type is not registered.
        AttributeError
            If the registry points to a selector method that is not available on the
            dataset override or generic selector.

        Notes
        -----
        Subclass overrides receive ``spec``, ``subject_id``, and ``row`` after Python
        binds the dataset instance as ``self``. Generic static methods receive the
        dataset explicitly.

        """

        descriptor = get_spec_descriptor(spec)
        value_method = getattr(dataset, descriptor.value_method_name, None)
        if value_method is not None:
            return value_method(spec, subject_id, row)
        return getattr(DatasetSampleValueSelector, descriptor.value_method_name)(dataset, spec, subject_id, row)

    @staticmethod
    def get_mesh_spec_value(
        dataset: "BaseDataset",
        spec: MeshSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        """Resolve a :class:`~hrtfpykit.datasets.MeshSpec` value for one dataset row.

        Mesh resources are subject-level. The selector reads the validated mesh path
        for subject_id from the dataset state, converts it to a string, and then
        applies the optional spec transform. It does not load mesh geometry directly;
        loading or parsing is left to user transforms or downstream code.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns mesh resource paths.
        spec : MeshSpec
            Mesh spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context. Mesh values are subject-level and do not inspect row axes.

        Returns
        -------
        object
            Mesh path string or transformed mesh value.

        Raises
        ------
        KeyError
            If subject_id does not have a selected mesh path in dataset state.

        Notes
        -----
        The optional transform receives the path string, not a loaded mesh object.

        """

        value: object = str(dataset._state.mesh_paths[subject_id])
        if spec.transform is not None:
            value = spec.transform(value)
        return value

    @staticmethod
    def get_image_spec_value(
        dataset: "BaseDataset",
        spec: ImageSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        """Resolve an :class:`~hrtfpykit.datasets.ImageSpec` value for one dataset row.

        The selector builds a media key from subject_id and, when the spec is
        grouped by ear, the row ear value. It reads every indexed image for that key,
        applies the optional spec transform to each file value, and returns either a
        single value, a list of values, or one concatenated array depending on the
        number of matched files and spec.concatenate.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns image resource indexes.
        spec : ImageSpec
            Image spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context used to select an ear group when the spec is ear-grouped.

        Returns
        -------
        object
            Image path, list of image paths, transformed image values, or a
            concatenated array when spec.concatenate is true.

        Raises
        ------
        KeyError
            If the subject and optional ear media key is not present in the image
            index.
        ValueError
            Raised by NumPy if spec.concatenate is true and transformed values cannot
            be concatenated along axis zero.

        Notes
        -----
        Ear selection is ignored unless the spec grouped_by value includes ``ear``.
        Position-indexed image keys are not selected here because the current image
        index uses subject and optional ear context.

        """

        grouped_by = sanitize_grouped_by(spec.grouped_by)
        ear = (
            None
            if row["ear"] is None
            else str(row["ear"])
            if grouped_by is not None and "ear" in grouped_by
            else None
        )
        media_key = (subject_id, None, ear)
        values: list[object] = []
        state = dataset._state
        for value in state.image_index[media_key]:
            if spec.transform is not None:
                value = spec.transform(value)
            values.append(value)
        if spec.concatenate:
            if spec.transform is None:
                raise ValueError(
                    "ImageSpec(concatenate=True) cannot concatenate raw image paths. "
                    "Pass transform=... to load each image into an array before "
                    "concatenation."
                )
            arrays = [np.asarray(value) for value in values]
            if any(array.ndim == 0 for array in arrays):
                raise ValueError(
                    "ImageSpec(concatenate=True) requires transform to return "
                    "array-like image values with at least one dimension."
                )
            return np.concatenate(arrays, axis=0)
        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def get_video_spec_value(
        dataset: "BaseDataset",
        spec: VideoSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        """Resolve a :class:`~hrtfpykit.datasets.VideoSpec` value for one dataset row.

        The selector mirrors image resolution for video resources. It builds a media
        key from subject_id and, when the spec is grouped by ear, the row ear value.
        It then reads every indexed video for that key, applies the optional spec
        transform to each file value, and returns either one value or a list.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns video resource indexes.
        spec : VideoSpec
            Video spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context used to select an ear group when the spec is ear-grouped.

        Returns
        -------
        object
            Video path, list of video paths, or transformed video values.

        Raises
        ------
        KeyError
            If the subject and optional ear media key is not present in the video
            index.

        Notes
        -----
        Unlike image specs, video specs do not currently concatenate multiple
        transformed values in this selector.

        """

        grouped_by = sanitize_grouped_by(spec.grouped_by)
        ear = (
            None
            if row["ear"] is None
            else str(row["ear"])
            if grouped_by is not None and "ear" in grouped_by
            else None
        )
        media_key = (subject_id, None, ear)
        values: list[object] = []
        state = dataset._state
        for value in state.video_index[media_key]:
            if spec.transform is not None:
                value = spec.transform(value)
            values.append(value)
        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def get_hrtf_spec_value(
        dataset: "BaseDataset",
        spec: HRTFSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        """Resolve an :class:`~hrtfpykit.datasets.HRTFSpec` value for one dataset row.

        This selector loads the subject HRTF through the dataset cache, optionally
        applies the spec-level HRTF transform, chooses the requested domain and
        signal representation, and slices position, ear, frequency, or sample axes
        according to the spec indexing mode and row context. It is the main acoustic
        array extraction path for raw HRIR, HRTF, magnitude, phase, real, and
        imaginary values.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns HRTF paths, cache, and acoustic context.
        spec : HRTFSpec
            HRTF spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context containing selected position, ear, frequency, or sample
            indices.

        Returns
        -------
        np.ndarray
            HRTF or HRIR value selected from the loaded subject HRTF. Output shape
            depends on ``spec.index_by``, ``spec.ears``, ``spec.domain``,
            ``spec.signal``, and any selected positions stored in dataset state.

        Raises
        ------
        KeyError
            If subject_id cannot be loaded through the dataset HRTF path index.
        ValueError
            If a transformed HRTF changes the position axis in a way that prevents
            applying the original spec position selection, or if the row ear is not
            allowed by spec.ears.
        IndexError
            If row position, ear, frequency, or sample indices are outside the
            selected array shape.

        Notes
        -----
        Spec-level transforms are cached per subject and transform identity. Dataset-
        level transforms are applied earlier by the HRTF loading path. When domain is
        ``time``, this selector reads IR values and uses the ``samples`` row axis. For
        frequency-domain values, it reads TF values and can expose complex values,
        real part, imaginary part, magnitude, decibel magnitude, or phase.

        """

        spec_index_by = sanitize_index_by(spec.index_by)
        spec_ears = sanitize_ears(spec.ears)
        domain = str(spec.domain).strip().lower()
        signal = str(spec.signal).strip().lower()
        state = dataset._state
        hrtf = dataset.get_subject_hrtf(subject_id)
        transformed_hrtf = None
        if spec.transform is not None:
            transform_cache_key = ("hrtf_transform", subject_id, id(spec.transform))
            transformed_hrtf = state.cache.get(transform_cache_key)
            if transformed_hrtf is None:
                transformed_hrtf = spec.transform(hrtf)
                state.cache[transform_cache_key] = transformed_hrtf

        selected_hrtf = cast(Any, hrtf if transformed_hrtf is None else transformed_hrtf)
        if domain == "time":
            values = np.asarray(selected_hrtf.IR.values, dtype=float)
            sample_axis_name = "samples"
        else:
            tf_values = np.asarray(selected_hrtf.TF.values)
            sample_axis_name = "frequency"
            if signal == "tf_complex":
                values = tf_values
            elif signal == "tf_real":
                values = real(tf_values)
            elif signal == "tf_imag":
                values = imag(tf_values)
            elif signal == "tf_magnitude":
                values = magnitude(tf_values)
            elif signal == "tf_magnitude_db":
                values = magnitude_db(tf_values)
            elif signal == "tf_phase":
                values = phase(tf_values)
            else:
                values = tf_values

        axis_names = ["position", "ear", sample_axis_name]
        selected_position_indices = state.spec_position_indices.get(id(spec), state.selected_position_indices)
        if "position" not in spec_index_by:
            position_axis = axis_names.index("position")
            if (
                selected_position_indices is not None
                and len(selected_position_indices) != values.shape[position_axis]
            ):
                if transformed_hrtf is None:
                    values = np.take(values, selected_position_indices, axis=position_axis)
                elif state.positions is not None and values.shape[position_axis] == state.positions.shape[0]:
                    values = np.take(values, selected_position_indices, axis=position_axis)
                else:
                    raise ValueError(
                        "HRTFSpec positions cannot be applied after transform because "
                        f"the transformed HRTF position axis has {values.shape[position_axis]} values, "
                        f"but the original dataset has {None if state.positions is None else state.positions.shape[0]} positions "
                        f"and the spec selected {len(selected_position_indices)} positions"
                    )
        else:
            position_axis = axis_names.index("position")
            position_index = int(cast(Any, row["position_index"]))
            values = np.take(values, [position_index], axis=position_axis)
            values = np.squeeze(values, axis=position_axis)
            axis_names.pop(position_axis)

        if "ear" not in spec_index_by:
            if len(spec_ears) != 2:
                ear_axis = axis_names.index("ear")
                ear_indices = [int(ear_index) for _, ear_index in spec_ears]
                values = np.take(values, ear_indices, axis=ear_axis)
                if len(ear_indices) == 1:
                    values = np.squeeze(values, axis=ear_axis)
                    axis_names.pop(ear_axis)
        else:
            ear_axis = axis_names.index("ear")
            row_ear = str(row["ear"])
            allowed_ears = {ear_name: ear_index for ear_name, ear_index in spec_ears}
            if row_ear not in allowed_ears:
                raise ValueError(
                    f"HRTFSpec with ears={spec.ears!r} cannot provide row ear {row_ear!r}"
                )
            ear_index = int(allowed_ears[row_ear])
            values = np.take(values, [ear_index], axis=ear_axis)
            values = np.squeeze(values, axis=ear_axis)
            axis_names.pop(ear_axis)

        if "frequency" in spec_index_by:
            frequency_axis = axis_names.index("frequency")
            frequency_index = int(cast(Any, row["frequency_index"]))
            values = np.take(values, [frequency_index], axis=frequency_axis)
            values = np.squeeze(values, axis=frequency_axis)
        if "samples" in spec_index_by:
            sample_axis = axis_names.index("samples")
            sample_index = int(cast(Any, row["sample_index"]))
            values = np.take(values, [sample_index], axis=sample_axis)
            values = np.squeeze(values, axis=sample_axis)
        return np.asarray(values)

    @staticmethod
    def get_itd_spec_value(
        dataset: "BaseDataset",
        spec: ITDSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        """Resolve an :class:`~hrtfpykit.datasets.ITDSpec` value for one dataset row.

        The selector loads the subject HRTF, applies the optional spec transform
        to that HRTF, computes interaural time difference once per subject and
        spec, stores the metric result in the dataset cache, and applies
        selected-position filtering or row-level position indexing. It turns the
        selected HRTF version into an ITD feature.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns HRTF paths, cache, and acoustic context.
        spec : ITDSpec
            ITD spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context used for position-indexed ITD values.

        Returns
        -------
        np.ndarray
            ITD value selected from the calculated subject ITD array. The result may
            be a scalar-like array for position-indexed rows or a vector when
            position is not part of ``spec.index_by``.

        Raises
        ------
        KeyError
            If subject_id cannot be loaded through the dataset HRTF path index.
        ValueError
            If ITD calculation fails because the underlying HRIR data or estimator
            parameters are invalid.
        IndexError
            If row position_index is outside the calculated ITD array.

        Notes
        -----
        Cache keys include subject_id and the spec object identity, so different ITD
        specs with different estimator parameters do not share metric arrays.

        """

        metric_cache_key = ("itd", subject_id, id(spec))
        state = dataset._state
        value = state.cache.get(metric_cache_key)
        if value is None:
            hrtf = dataset.get_subject_hrtf(subject_id)
            transformed_hrtf = None
            if spec.transform is not None:
                transform_cache_key = ("hrtf_transform", subject_id, id(spec.transform))
                transformed_hrtf = state.cache.get(transform_cache_key)
                if transformed_hrtf is None:
                    transformed_hrtf = spec.transform(hrtf)
                    state.cache[transform_cache_key] = transformed_hrtf
            selected_hrtf = cast(Any, hrtf if transformed_hrtf is None else transformed_hrtf)
            value = np.asarray(
                itd(
                    selected_hrtf.IR,
                    method=spec.method,
                    output=spec.output,
                    thresh_level=spec.thresh_level,
                    upper_cut_freq=spec.upper_cut_freq,
                    filter_order=spec.filter_order,
                )
            )
            state.cache[metric_cache_key] = value
        value = cast(np.ndarray, value)
        spec_index_by = sanitize_index_by(spec.index_by)
        selected_position_indices = state.spec_position_indices.get(id(spec), state.selected_position_indices)
        if "position" not in spec_index_by:
            if len(selected_position_indices) != value.shape[0]:
                value = np.take(value, selected_position_indices, axis=0)
        else:
            value = np.asarray(value[int(cast(Any, row["position_index"]))])
        return value

    @staticmethod
    def get_ild_spec_value(
        dataset: "BaseDataset",
        spec: ILDSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        """Resolve an :class:`~hrtfpykit.datasets.ILDSpec` value for one dataset row.

        The selector loads the subject HRTF, applies the optional spec transform
        to that HRTF, computes interaural level difference once per subject and
        spec, stores the metric result in the dataset cache, applies
        selected-position filtering, and applies row-level position or frequency
        indexing when requested. It supports both broadband and frequency-dependent
        ILD workflows.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns HRTF paths, cache, and acoustic context.
        spec : ILDSpec
            ILD spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context used for position or frequency indexed ILD values.

        Returns
        -------
        np.ndarray
            ILD value selected from the calculated subject ILD array. The shape
            depends on whether position or frequency appears in ``spec.index_by``.

        Raises
        ------
        KeyError
            If subject_id cannot be loaded through the dataset HRTF path index.
        ValueError
            If ILD calculation fails because the underlying HRIR data, sample rate,
            FFT length, mode, or output parameters are invalid.
        IndexError
            If row position_index or frequency_index is outside the calculated ILD
            array.

        Notes
        -----
        Cache keys include subject_id and the spec object identity. Position
        filtering is applied only when the metric output has a leading axis matching
        the dataset source-position count.

        """

        metric_cache_key = ("ild", subject_id, id(spec))
        state = dataset._state
        value = state.cache.get(metric_cache_key)
        if value is None:
            hrtf = dataset.get_subject_hrtf(subject_id)
            transformed_hrtf = None
            if spec.transform is not None:
                transform_cache_key = ("hrtf_transform", subject_id, id(spec.transform))
                transformed_hrtf = state.cache.get(transform_cache_key)
                if transformed_hrtf is None:
                    transformed_hrtf = spec.transform(hrtf)
                    state.cache[transform_cache_key] = transformed_hrtf
            selected_hrtf = cast(Any, hrtf if transformed_hrtf is None else transformed_hrtf)
            value = np.asarray(
                ild(
                    selected_hrtf.IR,
                    sample_rate=state.sample_rate,
                    fft_length=spec.fft_length,
                    mode=spec.mode,
                    output=spec.output,
                    epsilon=spec.epsilon,
                )
            )
            state.cache[metric_cache_key] = value
        value = cast(np.ndarray, value)
        spec_index_by = sanitize_index_by(spec.index_by)
        selected_position_indices = state.spec_position_indices.get(id(spec), state.selected_position_indices)
        if "position" not in spec_index_by:
            if state.positions is not None and value.shape[0] == state.positions.shape[0]:
                if len(selected_position_indices) != value.shape[0]:
                    value = np.take(value, selected_position_indices, axis=0)
        else:
            if state.positions is not None and value.shape[0] == state.positions.shape[0]:
                value = np.asarray(value[int(cast(Any, row["position_index"]))])
        if "frequency" in spec_index_by:
            value = np.asarray(value[..., int(cast(Any, row["frequency_index"]))])
        return value

    @staticmethod
    def get_sh_spec_value(
        dataset: "BaseDataset",
        spec: SHSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        """Resolve an :class:`~hrtfpykit.datasets.SHSpec` value for one dataset row.

        The selector loads the subject HRTF, applies the optional spec transform
        to that HRTF, computes spherical-harmonic coefficients once per subject
        and spec, stores the coefficient array in the dataset cache, applies ear
        selection or row-level ear indexing, and applies frequency indexing when
        requested. It gives datasets an SH-domain representation of the selected
        HRTF version.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns HRTF paths, cache, and acoustic context.
        spec : SHSpec
            Spherical-harmonic spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context used for ear or frequency indexed SH values.

        Returns
        -------
        np.ndarray
            Spherical-harmonic coefficient value selected for the current row. The
            coefficient axis is always retained; ear and frequency axes depend on
            ``spec.index_by`` and ``spec.ears``.

        Raises
        ------
        KeyError
            If subject_id cannot be loaded through the dataset HRTF path index.
        ValueError
            If SH decomposition fails, or if row ear is not allowed by spec.ears.
        IndexError
            If row ear or frequency indices are outside the coefficient array.

        Notes
        -----
        The SH decomposition receives ``both`` when both ears are requested; otherwise
        it receives the selected ear name. Cached values are keyed by subject_id and
        spec object identity.

        """

        sh_cache_key = ("sh", subject_id, id(spec))
        state = dataset._state
        value = state.cache.get(sh_cache_key)
        if value is None:
            hrtf = dataset.get_subject_hrtf(subject_id)
            transformed_hrtf = None
            if spec.transform is not None:
                transform_cache_key = ("hrtf_transform", subject_id, id(spec.transform))
                transformed_hrtf = state.cache.get(transform_cache_key)
                if transformed_hrtf is None:
                    transformed_hrtf = spec.transform(hrtf)
                    state.cache[transform_cache_key] = transformed_hrtf
            selected_hrtf = cast(Any, hrtf if transformed_hrtf is None else transformed_hrtf)
            spec_ears = sanitize_ears(spec.ears)
            sh_ear = "both" if len(spec_ears) == 2 else spec_ears[0][0]
            value = np.asarray(
                sht(
                    selected_hrtf,
                    sh_order=spec.sh_order,
                    ear=sh_ear,
                    epsilon=spec.epsilon,
                ).C
            )
            state.cache[sh_cache_key] = value
        value = cast(np.ndarray, value)
        spec_index_by = sanitize_index_by(spec.index_by)
        spec_ears = sanitize_ears(spec.ears)
        axis_names = ["coefficient", "frequency"]
        if value.ndim == 3:
            axis_names = ["coefficient", "ear", "frequency"]
        if "ear" not in spec_index_by:
            if "ear" in axis_names and len(spec_ears) != 2:
                ear_axis = axis_names.index("ear")
                ear_indices = [int(ear_index) for _, ear_index in spec_ears]
                value = np.take(value, ear_indices, axis=ear_axis)
                if len(ear_indices) == 1:
                    value = np.squeeze(value, axis=ear_axis)
                    axis_names.pop(ear_axis)
        else:
            row_ear = str(row["ear"])
            allowed_ears = {ear_name: ear_index for ear_name, ear_index in spec_ears}
            if row_ear not in allowed_ears:
                raise ValueError(
                    f"SHSpec with ears={spec.ears!r} cannot provide row ear {row_ear!r}"
                )
            if "ear" in axis_names:
                ear_axis = axis_names.index("ear")
                value = np.take(value, [int(allowed_ears[row_ear])], axis=ear_axis)
                value = np.squeeze(value, axis=ear_axis)
                axis_names.pop(ear_axis)
        if "frequency" in spec_index_by:
            frequency_axis = axis_names.index("frequency")
            value = np.take(value, [int(cast(Any, row["frequency_index"]))], axis=frequency_axis)
            value = np.squeeze(value, axis=frequency_axis)
        return value

    @staticmethod
    def get_anthropometry_spec_value(
        dataset: "BaseDataset",
        spec: AnthropometrySpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        """Resolve an :class:`~hrtfpykit.datasets.AnthropometrySpec` value for one dataset row.

        The selector maps the row subject to the normalized anthropometry table
        values stored on the dataset state, applies the dataset specific anthropometry
        selector when one is installed, and finally applies the optional spec
        transform. CSV and MAT parsing, row or column layout, subject matching,
        exclusions, and complete-row filtering are handled by table loading before
        sample selection reaches this method.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns loaded anthropometry table values.
        spec : AnthropometrySpec
            Anthropometry spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context used for ear-grouped anthropometry selection.

        Returns
        -------
        object
            Selected anthropometry table value or transformed value.

        Raises
        ------
        KeyError
            If subject_id cannot be resolved to loaded anthropometry data.

        """

        state = dataset._state
        if state.config is None:
            raise ValueError("Dataset config is not initialized")
        rows = state.anthropometry_rows
        mapped_subject_id = DatasetSplitPlanner.map_subject_id(
            subject_id,
            tuple(state.config.subject_ids),
        )
        if mapped_subject_id not in rows:
            raise KeyError(f"Anthropometry subject {subject_id!r} was not found")
        raw_value: object = dict(cast(dict[str, object], rows[mapped_subject_id]))

        selector = state.anthropometry_value_selector
        if selector is not None and callable(selector):
            raw_value = selector(
                spec=spec,
                row=row,
                value=raw_value,
            )
        if spec.transform is not None:
            raw_value = spec.transform(raw_value)
        return raw_value

    @staticmethod
    def get_metadata_spec_value(
        dataset: "BaseDataset",
        spec: MetadataSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        """Resolve a :class:`~hrtfpykit.datasets.MetadataSpec` value for one dataset row.

        The selector maps the row subject to the normalized metadata table values
        stored on the dataset state, applies the dataset specific metadata selector when
        one is installed, and finally applies the optional spec transform. It
        mirrors anthropometry extraction while keeping general annotations separate
        from physical measurements. CSV and MAT parsing, row or column layout,
        subject matching, exclusions, and complete-row filtering are handled by
        table loading before sample selection reaches this method.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset instance that owns loaded metadata table values.
        spec : MetadataSpec
            Metadata spec to resolve.
        subject_id : str
            Canonical subject ID for the current row.
        row : dict
            Row context used for grouped metadata selection.

        Returns
        -------
        object
            Selected metadata table value or transformed value.

        Raises
        ------
        KeyError
            If subject_id cannot be resolved to loaded metadata data.

        """

        state = dataset._state
        if state.config is None:
            raise ValueError("Dataset config is not initialized")
        rows = state.metadata_rows
        mapped_subject_id = DatasetSplitPlanner.map_subject_id(
            subject_id,
            tuple(state.config.subject_ids),
        )
        if mapped_subject_id not in rows:
            raise KeyError(f"Metadata subject {subject_id!r} was not found")
        raw_value: object = dict(cast(dict[str, object], rows[mapped_subject_id]))

        selector = state.metadata_value_selector
        if selector is not None and callable(selector):
            raw_value = selector(
                spec=spec,
                row=row,
                value=raw_value,
            )
        if spec.transform is not None:
            raw_value = spec.transform(raw_value)
        return raw_value
