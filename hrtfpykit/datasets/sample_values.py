from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .normalization import normalize_ears, normalize_grouped_by, normalize_index_by
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
)
from ..hrtf.dsp import imag, magnitude, magnitude_db, phase, real
from ..hrtf.metrics import ild, itd
from ..hrtf.sh import sht

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


class DatasetSampleValueResolver:
    @staticmethod
    def apply_path_transform(
        paths: list[str],
        transform: Callable | None,
        concatenate: bool = False,
    ) -> object:
        values: list[object] = []
        for path in paths:
            value: object = path
            if transform is not None:
                value = transform(value)
            values.append(value)
        if concatenate:
            arrays = [np.asarray(value) for value in values]
            return np.concatenate(arrays, axis=0)
        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def get_media_key(
        subject_id: str,
        row: dict[str, str | int | None],
        grouped_by: tuple[str, ...] | None,
    ) -> tuple[str, int | None, str | None]:
        ear = (
            None
            if row["ear"] is None
            else str(row["ear"])
            if grouped_by is not None and "ear" in grouped_by
            else None
        )
        return subject_id, None, ear

    @staticmethod
    def select_image_value(
        index: dict[tuple[str, int | None, str | None], list[str]],
        subject_id: str,
        row: dict[str, str | int | None],
        grouped_by: tuple[str, ...] | None,
        transform: Callable | None,
        concatenate: bool = False,
    ) -> object:
        image_key = DatasetSampleValueResolver.get_media_key(
            subject_id,
            row,
            grouped_by,
        )
        return DatasetSampleValueResolver.apply_path_transform(
            index[image_key],
            transform,
            concatenate=concatenate,
        )

    @staticmethod
    def select_video_value(
        index: dict[tuple[str, int | None, str | None], list[str]],
        subject_id: str,
        row: dict[str, str | int | None],
        grouped_by: tuple[str, ...] | None,
        transform: Callable | None,
    ) -> object:
        video_key = DatasetSampleValueResolver.get_media_key(
            subject_id,
            row,
            grouped_by,
        )
        return DatasetSampleValueResolver.apply_path_transform(
            index[video_key],
            transform,
        )

    @staticmethod
    def select_hrtf_value(
        hrtf: "HRTF",
        row: dict[str, str | int | None],
        selected_position_indices: list[int] | None,
        spec: HRTFSpec,
    ) -> np.ndarray:
        spec_index_by = normalize_index_by(spec.index_by)
        spec_ears = normalize_ears(spec.ears)
        domain = str(spec.domain).strip().lower()
        signal = str(spec.signal).strip().lower()
        if domain == "time":
            values = np.asarray(hrtf.IR.values, dtype=float)
            sample_axis_name = "samples"
        else:
            tf_values = np.asarray(hrtf.TF.values)
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
        if "position" not in spec_index_by:
            position_axis = axis_names.index("position")
            if (
                selected_position_indices is not None
                and len(selected_position_indices) != values.shape[position_axis]
            ):
                values = np.take(values, selected_position_indices, axis=position_axis)
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
            axis_names.pop(frequency_axis)
        if "samples" in spec_index_by:
            sample_axis = axis_names.index("samples")
            sample_index = int(row["sample_index"])
            values = np.take(values, [sample_index], axis=sample_axis)
            values = np.squeeze(values, axis=sample_axis)
        return np.asarray(values)

    @staticmethod
    def select_sh_value(
        values: np.ndarray,
        row: dict[str, str | int | None],
        spec: SHSpec,
    ) -> np.ndarray:
        spec_index_by = normalize_index_by(spec.index_by)
        spec_ears = normalize_ears(spec.ears)
        axis_names = ["coefficient", "frequency"]
        if values.ndim == 3:
            axis_names = ["coefficient", "ear", "frequency"]
        if "ear" not in spec_index_by:
            if "ear" in axis_names and len(spec_ears) != 2:
                ear_axis = axis_names.index("ear")
                ear_indices = [int(ear_index) for _, ear_index in spec_ears]
                values = np.take(values, ear_indices, axis=ear_axis)
                if len(ear_indices) == 1:
                    values = np.squeeze(values, axis=ear_axis)
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
                values = np.take(values, [int(allowed_ears[row_ear])], axis=ear_axis)
                values = np.squeeze(values, axis=ear_axis)
                axis_names.pop(ear_axis)
        if "frequency" in spec_index_by:
            frequency_axis = axis_names.index("frequency")
            values = np.take(values, [int(row["frequency_index"])], axis=frequency_axis)
            values = np.squeeze(values, axis=frequency_axis)
        return np.asarray(values)

    def get_sample_value(
        self,
        spec: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        if isinstance(spec, HRTFSpec):
            return self.get_hrtf_spec_value(spec, subject_id, row)
        if isinstance(spec, ITDSpec):
            return self.get_itd_spec_value(spec, subject_id, row)
        if isinstance(spec, ILDSpec):
            return self.get_ild_spec_value(spec, subject_id, row)
        if isinstance(spec, SHSpec):
            return self.get_sh_spec_value(spec, subject_id, row)
        if isinstance(spec, MeshSpec):
            value: object = str(self._mesh_paths[subject_id])
            if spec.transform is not None:
                value = spec.transform(value)
            return value
        if isinstance(spec, AnthropometrySpec):
            return self.get_anthropometry_value(spec, subject_id, row)
        if isinstance(spec, ImageSpec):
            return self.select_image_value(
                self._image_index,
                subject_id,
                row,
                normalize_grouped_by(spec.grouped_by),
                spec.transform,
                concatenate=spec.concatenate,
            )
        if isinstance(spec, VideoSpec):
            return self.select_video_value(
                self._video_index,
                subject_id,
                row,
                normalize_grouped_by(spec.grouped_by),
                spec.transform,
            )
        raise TypeError(f"Unsupported dataset spec: {type(spec)!r}")

    def get_hrtf_spec_value(
        self,
        spec: HRTFSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        hrtf = self.get_subject_hrtf(subject_id)
        transformed_hrtf = None
        if spec.transform is not None:
            transform_cache_key = ("hrtf_transform", subject_id, id(spec.transform))
            transformed_hrtf = self._cache.get(transform_cache_key)
            if transformed_hrtf is None:
                transformed_hrtf = spec.transform(hrtf)
                self._cache[transform_cache_key] = transformed_hrtf
        value = self.select_hrtf_value(
            hrtf=hrtf if transformed_hrtf is None else transformed_hrtf,
            row=row,
            selected_position_indices=(
                None
                if transformed_hrtf is not None
                else self._spec_position_indices.get(id(spec), self._selected_position_indices)
            ),
            spec=spec,
        )
        return value

    def get_itd_spec_value(
        self,
        spec: ITDSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        metric_cache_key = ("itd", subject_id, id(spec))
        value = self._cache.get(metric_cache_key)
        if value is None:
            hrtf = self.get_subject_hrtf(subject_id)
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
            self._cache[metric_cache_key] = value
        spec_index_by = normalize_index_by(spec.index_by)
        selected_position_indices = self._spec_position_indices.get(id(spec), self._selected_position_indices)
        if "position" not in spec_index_by:
            if len(selected_position_indices) != value.shape[0]:
                value = np.take(value, selected_position_indices, axis=0)
        else:
            value = np.asarray(value[int(row["position_index"])])
        if spec.transform is not None:
            value = spec.transform(value)
        return value

    def get_ild_spec_value(
        self,
        spec: ILDSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        metric_cache_key = ("ild", subject_id, id(spec))
        value = self._cache.get(metric_cache_key)
        if value is None:
            hrtf = self.get_subject_hrtf(subject_id)
            value = np.asarray(
                ild(
                    hrtf.IR,
                    sample_rate=self._dataset_sample_rate,
                    fft_length=spec.fft_length,
                    mode=spec.mode,
                    output=spec.output,
                    epsilon=spec.epsilon,
                )
            )
            self._cache[metric_cache_key] = value
        spec_index_by = normalize_index_by(spec.index_by)
        selected_position_indices = self._spec_position_indices.get(id(spec), self._selected_position_indices)
        if "position" not in spec_index_by:
            if value.shape[0] == self._dataset_source_positions.shape[0]:
                if len(selected_position_indices) != value.shape[0]:
                    value = np.take(value, selected_position_indices, axis=0)
        else:
            if value.shape[0] == self._dataset_source_positions.shape[0]:
                value = np.asarray(value[int(row["position_index"])])
        if "frequency" in spec_index_by:
            value = np.asarray(value[..., int(row["frequency_index"])])
        if spec.transform is not None:
            value = spec.transform(value)
        return value

    def get_sh_spec_value(
        self,
        spec: SHSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        sh_cache_key = ("sh", subject_id, id(spec))
        value = self._cache.get(sh_cache_key)
        if value is None:
            hrtf = self.get_subject_hrtf(subject_id)
            spec_ears = normalize_ears(spec.ears)
            sh_ear = "both" if len(spec_ears) == 2 else spec_ears[0][0]
            value = np.asarray(
                sht(
                    hrtf,
                    sh_order=spec.sh_order,
                    ear=sh_ear,
                    epsilon=spec.epsilon,
                ).C
            )
            self._cache[sh_cache_key] = value
        value = self.select_sh_value(
            values=value,
            row=row,
            spec=spec,
        )
        if spec.transform is not None:
            value = spec.transform(value)
        return value

    def get_anthropometry_value(
        self,
        spec: AnthropometrySpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        raw_value = self._resolve_anthropometry_value_default(
            spec=spec,
            subject_id=subject_id,
            row=row,
        )
        resolver = getattr(self, "_anthropometry_value_resolver", None)
        if resolver is not None and callable(resolver):
            raw_value = resolver(
                spec=spec,
                subject_id=subject_id,
                row=row,
                value=raw_value,
            )
        if spec.transform is not None:
            raw_value = spec.transform(raw_value)
        return raw_value

    def _resolve_anthropometry_value_default(
        self,
        spec: AnthropometrySpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        rows = self._anthropometry_rows
        if spec.accessed_by not in {"row", "column"}:
            raise ValueError("AnthropometrySpec accessed_by must be 'row' or 'column'")
        resolved_subject_id = type(self)._resolve_dataset_subject_id(
            subject_id,
            tuple(self._config.subject_ids),
        )
        try:
            subject_position = list(self._subject_ids).index(resolved_subject_id)
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
                value = matrix[subject_position]
            else:
                if subject_position < 0 or subject_position >= matrix.shape[1]:
                    raise IndexError(
                        f"Anthropometry column index {subject_position} is out of range for "
                        f"{matrix.shape[1]} columns"
                    )
                value = matrix[:, subject_position]
        else:
            if resolved_subject_id not in rows:
                raise KeyError(
                    f"Anthropometry subject {subject_id!r} was not found"
                )
            row_values = dict(rows[resolved_subject_id])
            if spec.accessed_by == "row":
                value = row_values
            else:
                subject_ids = tuple(rows)
                if resolved_subject_id not in subject_ids:
                    raise KeyError(
                        f"Anthropometry subject {subject_id!r} was not found"
                    )
                subject_position = list(subject_ids).index(resolved_subject_id)
                column_keys = tuple(row_values)
                if subject_position < 0 or subject_position >= len(column_keys):
                    raise IndexError(
                        f"Anthropometry column index {subject_position} is out of range for "
                        f"{len(column_keys)} columns"
                    )
                column_key = column_keys[subject_position]
                value = {
                    column_subject_id: row_values_by_subject[column_key]
                    for column_subject_id, row_values_by_subject in rows.items()
                }
        return value
