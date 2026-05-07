import numpy as np
from typing import TYPE_CHECKING

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
from ..hrtf.dsp import imag, magnitude, magnitude_db, phase, real
from ..hrtf.metrics import ild, itd
from ..hrtf.sh import sht
from .split import DatasetSplitPlanner

if TYPE_CHECKING:
    from .base import BaseDataset


class DatasetSampleValueSelector:
    """Resolve concrete sample values for dataset specs.

    ``DatasetSampleValueSelector`` is the value extraction layer used by
    ``BaseDataset.__getitem__``. It receives the dataset state, the current spec,
    the current subject ID, and the current row context, then returns the value
    stored under that spec name in ``sample['inputs']`` or ``sample['target']``.
    Dataset subclasses can override selector methods with the same names to
    customize specific resource behavior.

    Use Cases
    ---------
    - Extract acoustic, mesh, table, image, and video values from rows.
    - Apply spec-level transforms after row selection.
    - Provide extension points for dataset-specific value behavior.
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
        ``BaseDataset.__getitem__``. It asks the registry which selector belongs to
        the spec, prefers a dataset subclass override when present, and falls back to
        the generic implementation on ``DatasetSampleValueSelector``.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns state, resources, and optional subclass
            selector overrides.
        spec : dataset spec
            Spec object to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context containing selected position, ear, frequency, and sample
            indices.

        Returns
        -------
        object Concrete sample value for the given spec and row.

        Use Cases
        ---------
        - Route specs through the centralized registry.
        - Keep ``BaseDataset.__getitem__`` independent of concrete spec types.
        - Allow dataset subclasses to override resource-specific selection.
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
        """Resolve a ``MeshSpec`` value for one dataset row.

        Mesh resources are subject-level, so the selector maps the current subject to
        its validated mesh path and applies an optional spec transform. It does not
        load geometry directly, leaving that choice to user transforms or downstream
        code.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns mesh resource paths.
        spec : MeshSpec
            Mesh spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context. Mesh values are subject-level and do not use row axes.

        Returns
        -------
        object Mesh path string or transformed mesh value.

        Use Cases
        ---------
        - Pair subject meshes with HRTF samples.
        - Return local mesh paths for downstream geometry loading.
        - Apply a mesh transform before returning the value.
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
        """Resolve an ``ImageSpec`` value for one dataset row.

        The selector uses row ear context when images are grouped by subject-ear,
        collects all indexed files for that media key, applies optional transforms per
        file, and returns a single value, list, or concatenated array depending on the
        spec configuration.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns image resource indexes.
        spec : ImageSpec
            Image spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context used to select an ear group when the spec is ear-grouped.

        Returns
        -------
        object Image path, list of image paths, transformed image values, or a
        concatenated array when ``spec.concatenate=True``.

        Use Cases
        ---------
        - Pair images with acoustic samples.
        - Select left/right media for ear-indexed datasets.
        - Apply image loading or preprocessing transforms.
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
            arrays = [np.asarray(value) for value in values]
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
        """Resolve a ``VideoSpec`` value for one dataset row.

        The selector mirrors image behavior for video resources: it uses subject and
        optional ear context to find indexed files, applies optional transforms, and
        returns a single value or list depending on how many videos match the row.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns video resource indexes.
        spec : VideoSpec
            Video spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context used to select an ear group when the spec is ear-grouped.

        Returns
        -------
        object Video path, list of video paths, or transformed video values.

        Use Cases
        ---------
        - Pair videos with acoustic samples.
        - Select left/right media for ear-indexed datasets.
        - Apply video loading or preprocessing transforms.
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
        """Resolve an ``HRTFSpec`` value for one dataset row.

        This selector loads the subject HRTF through the dataset cache, applies
        optional spec-level HRTF transforms, selects domain and signal representation,
        then slices position, ear, frequency, or sample axes according to the row
        context. It is the main acoustic array extraction path.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns HRTF paths, cache, and acoustic context.
        spec : HRTFSpec
            HRTF spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context containing selected position, ear, frequency, or sample
            indices.

        Returns
        -------
        numpy.ndarray HRTF or HRIR value selected from the loaded subject HRTF.

        Use Cases
        ---------
        - Extract full or indexed HRIR/HRTF arrays.
        - Apply spec-level transforms after HRTF loading.
        - Return signal variants such as magnitude, phase, real, or imaginary TF.
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

        selected_hrtf = hrtf if transformed_hrtf is None else transformed_hrtf
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
            position_index = int(row["position_index"])
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
            frequency_index = int(row["frequency_index"])
            values = np.take(values, [frequency_index], axis=frequency_axis)
            values = np.squeeze(values, axis=frequency_axis)
        if "samples" in spec_index_by:
            sample_axis = axis_names.index("samples")
            sample_index = int(row["sample_index"])
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
        """Resolve an ``ITDSpec`` value for one dataset row.

        The selector computes ITD once per subject/spec pair, caches the metric
        result, applies position selection or row indexing, and finally applies an
        optional spec transform. It turns full HRTF resources into ITD features.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns HRTF paths, cache, and acoustic context.
        spec : ITDSpec
            ITD spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context used for position-indexed ITD values.

        Returns
        -------
        numpy.ndarray ITD value selected from the calculated subject ITD array.

        Use Cases
        ---------
        - Derive ITD features from subject HRTFs.
        - Cache repeated metric calculations per subject and spec.
        - Return full-position or single-position ITD values.
        """

        metric_cache_key = ("itd", subject_id, id(spec))
        state = dataset._state
        value = state.cache.get(metric_cache_key)
        if value is None:
            hrtf = dataset.get_subject_hrtf(subject_id)
            value = np.asarray(
                itd(
                    hrtf.IR,
                    method=spec.method,
                    output=spec.output,
                    thresh_level=spec.thresh_level,
                    upper_cut_freq=spec.upper_cut_freq,
                    filter_order=spec.filter_order,
                )
            )
            state.cache[metric_cache_key] = value
        spec_index_by = sanitize_index_by(spec.index_by)
        selected_position_indices = state.spec_position_indices.get(id(spec), state.selected_position_indices)
        if "position" not in spec_index_by:
            if len(selected_position_indices) != value.shape[0]:
                value = np.take(value, selected_position_indices, axis=0)
        else:
            value = np.asarray(value[int(row["position_index"])])
        if spec.transform is not None:
            value = spec.transform(value)
        return value

    @staticmethod
    def get_ild_spec_value(
        dataset: "BaseDataset",
        spec: ILDSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        """Resolve an ``ILDSpec`` value for one dataset row.

        The selector computes ILD once per subject/spec pair, caches the metric
        result, applies position and frequency row selection when requested, and then
        applies an optional spec transform. It supports both broad-band and frequency-
        dependent ILD workflows.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns HRTF paths, cache, and acoustic context.
        spec : ILDSpec
            ILD spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context used for position or frequency indexed ILD values.

        Returns
        -------
        numpy.ndarray ILD value selected from the calculated subject ILD array.

        Use Cases
        ---------
        - Derive broad-band or frequency-dependent ILD features.
        - Cache repeated metric calculations per subject and spec.
        - Return full-position, single-position, or single-frequency ILD values.
        """

        metric_cache_key = ("ild", subject_id, id(spec))
        state = dataset._state
        value = state.cache.get(metric_cache_key)
        if value is None:
            hrtf = dataset.get_subject_hrtf(subject_id)
            value = np.asarray(
                ild(
                    hrtf.IR,
                    sample_rate=state.sample_rate,
                    fft_length=spec.fft_length,
                    mode=spec.mode,
                    output=spec.output,
                    epsilon=spec.epsilon,
                )
            )
            state.cache[metric_cache_key] = value
        spec_index_by = sanitize_index_by(spec.index_by)
        selected_position_indices = state.spec_position_indices.get(id(spec), state.selected_position_indices)
        if "position" not in spec_index_by:
            if state.positions is not None and value.shape[0] == state.positions.shape[0]:
                if len(selected_position_indices) != value.shape[0]:
                    value = np.take(value, selected_position_indices, axis=0)
        else:
            if state.positions is not None and value.shape[0] == state.positions.shape[0]:
                value = np.asarray(value[int(row["position_index"])])
        if "frequency" in spec_index_by:
            value = np.asarray(value[..., int(row["frequency_index"])])
        if spec.transform is not None:
            value = spec.transform(value)
        return value

    @staticmethod
    def get_sh_spec_value(
        dataset: "BaseDataset",
        spec: SHSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        """Resolve an ``SHSpec`` value for one dataset row.

        The selector computes spherical-harmonic coefficients once per subject/spec
        pair, caches the result, and slices ear or frequency axes according to row
        context. It gives datasets an SH-domain representation of HRTF data
        without changing source resources.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns HRTF paths, cache, and acoustic context.
        spec : SHSpec
            Spherical-harmonic spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context used for ear or frequency indexed SH values.

        Returns
        -------
        numpy.ndarray Spherical-harmonic coefficient value selected for the current
        row.

        Use Cases
        ---------
        - Convert subject HRTFs into SH-domain features.
        - Cache SH decomposition per subject and spec.
        - Return full or indexed ear/frequency coefficient values.
        """

        sh_cache_key = ("sh", subject_id, id(spec))
        state = dataset._state
        value = state.cache.get(sh_cache_key)
        if value is None:
            hrtf = dataset.get_subject_hrtf(subject_id)
            spec_ears = sanitize_ears(spec.ears)
            sh_ear = "both" if len(spec_ears) == 2 else spec_ears[0][0]
            value = np.asarray(
                sht(
                    hrtf,
                    sh_order=spec.sh_order,
                    ear=sh_ear,
                    epsilon=spec.epsilon,
                ).C
            )
            state.cache[sh_cache_key] = value
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
            value = np.take(value, [int(row["frequency_index"])], axis=frequency_axis)
            value = np.squeeze(value, axis=frequency_axis)
        if spec.transform is not None:
            value = spec.transform(value)
        return value

    @staticmethod
    def get_anthropometry_spec_value(
        dataset: "BaseDataset",
        spec: AnthropometrySpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        """Resolve an ``AnthropometrySpec`` value for one dataset row.

        The selector maps the row subject to loaded anthropometry table data, supports
        row-oriented, column-oriented, and MAT matrix access, applies dataset-specific
        anthropometry selectors, and finally applies an optional spec transform. It is
        shared table extraction with anthropometry-specific hooks.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns loaded anthropometry rows.
        spec : AnthropometrySpec
            Anthropometry spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context used for ear-grouped anthropometry selection.

        Returns
        -------
        object Anthropometry row, column, matrix slice, or transformed value.

        Use Cases
        ---------
        - Attach physical measurements to HRTF samples.
        - Select ear-specific anthropometry fields.
        - Apply dataset-specific anthropometry value selectors.
        """

        state = dataset._state
        rows = state.anthropometry_rows
        mapped_subject_id = DatasetSplitPlanner.map_subject_id(
            subject_id,
            tuple(state.config.subject_ids),
        )
        try:
            subject_position = list(state.selected_subjects).index(mapped_subject_id)
        except ValueError as exc:
            raise KeyError(f"Anthropometry subject {subject_id!r} was not found") from exc

        if not isinstance(rows, dict) or not all(
            isinstance(value, dict) for value in rows.values()
        ):
            matrix_values: object = rows
            if isinstance(rows, dict):
                matrix_candidates = {
                    key: value
                    for key, value in rows.items()
                    if not str(key).startswith("__")
                }
                if len(matrix_candidates) != 1:
                    raise ValueError(
                        "MAT anthropometry access requires exactly one data matrix variable"
                    )
                matrix_values = next(iter(matrix_candidates.values()))
            matrix = np.asarray(matrix_values)
            if matrix.ndim < 2:
                raise ValueError(
                    "Anthropometry matrix access requires a two-dimensional value"
                )
            if spec.accessed_by == "row":
                if subject_position < 0 or subject_position >= matrix.shape[0]:
                    raise IndexError(
                        f"Anthropometry row index {subject_position} is out of range for "
                        f"{matrix.shape[0]} rows"
                    )
                raw_value = matrix[subject_position]
            else:
                if subject_position < 0 or subject_position >= matrix.shape[1]:
                    raise IndexError(
                        f"Anthropometry column index {subject_position} is out of range for "
                        f"{matrix.shape[1]} columns"
                    )
                raw_value = matrix[:, subject_position]
        else:
            if mapped_subject_id not in rows:
                if spec.accessed_by == "column":
                    column_values: dict[str, float | str | None] = {}
                    for row_key, row_values in rows.items():
                        if not isinstance(row_values, dict):
                            continue
                        if mapped_subject_id in row_values:
                            column_values[row_key] = row_values[mapped_subject_id]
                    if len(column_values) == 0:
                        raise KeyError(
                            f"Anthropometry subject {subject_id!r} was not found"
                        )
                    raw_value = column_values
                else:
                    raise KeyError(
                        f"Anthropometry subject {subject_id!r} was not found"
                    )
            else:
                row_values = dict(rows[mapped_subject_id])
                if spec.accessed_by == "row":
                    raw_value = row_values
                else:
                    subject_position = list(state.selected_subjects).index(mapped_subject_id)
                    column_keys = tuple(row_values)
                    if subject_position < 0 or subject_position >= len(column_keys):
                        raise IndexError(
                            f"Anthropometry column index {subject_position} is out of range for "
                            f"{len(column_keys)} columns"
                        )
                    column_key = column_keys[subject_position]
                    raw_value = {
                        column_subject_id: row_values_by_subject[column_key]
                        for column_subject_id, row_values_by_subject in rows.items()
                    }

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
        """Resolve a ``MetadataSpec`` value for one dataset row.

        The selector maps the row subject to loaded metadata table data, supports row-
        oriented, column-oriented, and MAT matrix access, applies dataset-specific
        metadata selectors, and finally applies an optional spec transform. It mirrors
        anthropometry extraction while keeping metadata resource identity separate.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance that owns loaded metadata rows.
        spec : MetadataSpec
            Metadata spec to resolve.
        subject_id : str
            Subject ID for the current row.
        row : dict
            Row context used for grouped metadata selection.

        Returns
        -------
        object Metadata row, column, matrix slice, or transformed value.

        Use Cases
        ---------
        - Attach subject annotations to HRTF samples.
        - Keep metadata values separate from anthropometry values.
        - Apply metadata transforms during sample extraction.
        """

        state = dataset._state
        rows = state.metadata_rows
        mapped_subject_id = DatasetSplitPlanner.map_subject_id(
            subject_id,
            tuple(state.config.subject_ids),
        )
        try:
            subject_position = list(state.selected_subjects).index(mapped_subject_id)
        except ValueError as exc:
            raise KeyError(f"Metadata subject {subject_id!r} was not found") from exc

        if not isinstance(rows, dict) or not all(
            isinstance(value, dict) for value in rows.values()
        ):
            matrix_values: object = rows
            if isinstance(rows, dict):
                matrix_candidates = {
                    key: value
                    for key, value in rows.items()
                    if not str(key).startswith("__")
                }
                if len(matrix_candidates) != 1:
                    raise ValueError(
                        "MAT metadata access requires exactly one data matrix variable"
                    )
                matrix_values = next(iter(matrix_candidates.values()))
            matrix = np.asarray(matrix_values)
            if matrix.ndim < 2:
                raise ValueError(
                    "Metadata matrix access requires a two-dimensional value"
                )
            if spec.accessed_by == "row":
                if subject_position < 0 or subject_position >= matrix.shape[0]:
                    raise IndexError(
                        f"Metadata row index {subject_position} is out of range for "
                        f"{matrix.shape[0]} rows"
                    )
                raw_value = matrix[subject_position]
            else:
                if subject_position < 0 or subject_position >= matrix.shape[1]:
                    raise IndexError(
                        f"Metadata column index {subject_position} is out of range for "
                        f"{matrix.shape[1]} columns"
                    )
                raw_value = matrix[:, subject_position]
        else:
            if mapped_subject_id not in rows:
                if spec.accessed_by == "column":
                    column_values: dict[str, float | str | None] = {}
                    for row_key, row_values in rows.items():
                        if not isinstance(row_values, dict):
                            continue
                        if mapped_subject_id in row_values:
                            column_values[row_key] = row_values[mapped_subject_id]
                    if len(column_values) == 0:
                        raise KeyError(
                            f"Metadata subject {subject_id!r} was not found"
                        )
                    raw_value = column_values
                else:
                    raise KeyError(
                        f"Metadata subject {subject_id!r} was not found"
                    )
            else:
                row_values = dict(rows[mapped_subject_id])
                if spec.accessed_by == "row":
                    raw_value = row_values
                else:
                    subject_position = list(state.selected_subjects).index(mapped_subject_id)
                    column_keys = tuple(row_values)
                    if subject_position < 0 or subject_position >= len(column_keys):
                        raise IndexError(
                            f"Metadata column index {subject_position} is out of range for "
                            f"{len(column_keys)} columns"
                        )
                    column_key = column_keys[subject_position]
                    raw_value = {
                        column_subject_id: row_values_by_subject[column_key]
                        for column_subject_id, row_values_by_subject in rows.items()
                    }

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
