from collections.abc import Callable

import numpy as np

from .index import normalize_index_by
from .resolver import DatasetResourceResolver
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


def apply_media_transform(
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
        if len(values) == 0:
            raise ValueError("Cannot concatenate an empty image sequence")
        arrays = [np.asarray(value) for value in values]
        reference_shape = arrays[0].shape
        if len(reference_shape) != 3:
            raise ValueError(
                "ImageSpec(concatenate=True) requires each transformed image to have shape (C, H, W)"
            )
        for index, array in enumerate(arrays[1:], start=1):
            if array.shape != reference_shape:
                raise ValueError(
                    "ImageSpec(concatenate=True) requires all transformed images to share the same shape, "
                    f"but image 0 has shape {reference_shape} and image {index} has shape {array.shape}"
                )
        return np.concatenate(arrays, axis=0)
    if len(values) == 1:
        return values[0]
    return values


def select_anthropometry_value(
    values: dict[str, float | str | None],
    select: str | tuple[str, ...] | list[str] | None,
    ear: str,
    dataset_name: str,
) -> dict[str, float | str | None]:
    selected = DatasetResourceResolver.normalize_anthropometry_select(select)
    normalized_ear = DatasetResourceResolver.normalize_anthropometry_ear(ear)
    unsupported_ear_message = (
        f"{dataset_name} does not define dataset-specific anthropometry ear handling. "
        f"Requested ear={normalized_ear!r}. "
        "Generic anthropometry access only supports ear='both' and exact raw column names."
    )
    if selected == "complete":
        if normalized_ear != "both":
            raise ValueError(unsupported_ear_message)
        return dict(values)
    missing = [name for name in selected if name not in values]
    if len(missing) > 0:
        raise ValueError(
            f"{dataset_name} could not resolve anthropometry select values {missing} "
            "as exact raw column names. "
            "This dataset does not define dataset-specific anthropometry selection aliases."
        )
    if normalized_ear != "both":
        raise ValueError(unsupported_ear_message)
    return {name: values[name] for name in selected}


class DatasetValueResolver:
    @staticmethod
    def select_media_value(
        index: dict[tuple[str, int | None, str | None], list[str]],
        subject_id: str,
        row: dict[str, str | int | None],
        align_by: tuple[str, ...] | None,
        transform: Callable | None,
        concatenate: bool = False,
        resource_name: str = "media",
    ) -> object:
        if align_by is None:
            raise ValueError(f"{resource_name} align_by is not configured")
        media_key = (
            subject_id,
            None,
            None if row["ear"] is None else str(row["ear"]) if "ear" in align_by else None,
        )
        if media_key not in index:
            raise ValueError(f"No {resource_name} found for sample {media_key}")
        return apply_media_transform(
            index[media_key],
            transform,
            concatenate=concatenate,
        )

    @staticmethod
    def select_hrtf_value(
        hrtf,
        row: dict[str, str | int | None],
        selected_position_indices: list[int],
        selected_ears: list[tuple[str, int]],
        spec: HRTFSpec,
    ) -> np.ndarray:
        spec_index_by = normalize_index_by(spec.index_by)
        domain = str(spec.domain).strip().lower()
        signal = str(spec.signal).strip().lower()
        if domain == "time":
            if signal != "ir":
                raise ValueError("HRTFSpec with domain='time' requires signal='ir'")
            values = np.asarray(hrtf.IR.values, dtype=float)
            sample_axis_name = "samples"
        elif domain == "frequency":
            if signal == "ir":
                raise ValueError("HRTFSpec with domain='frequency' cannot use signal='ir'")
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
                raise ValueError(f"Unsupported signal {signal!r}")
        else:
            raise ValueError(f"Unsupported domain {domain!r}")

        axis_names = ["position", "ear", sample_axis_name]
        if "position" not in spec_index_by:
            position_axis = axis_names.index("position")
            if len(selected_position_indices) != values.shape[position_axis]:
                values = np.take(values, selected_position_indices, axis=position_axis)
        else:
            if row["position_index"] is None:
                raise ValueError(
                    f"HRTFSpec(index_by={spec.index_by!r}) requires position-resolved rows"
                )
            position_axis = axis_names.index("position")
            values = np.take(values, [int(row["position_index"])], axis=position_axis)
            values = np.squeeze(values, axis=position_axis)
            axis_names.pop(position_axis)
        if "ear" not in spec_index_by:
            if len(selected_ears) == 1:
                ear_axis = axis_names.index("ear")
                values = np.take(values, [int(selected_ears[0][1])], axis=ear_axis)
                values = np.squeeze(values, axis=ear_axis)
                axis_names.pop(ear_axis)
        else:
            if row["ear_index"] is None:
                raise ValueError(
                    f"HRTFSpec(index_by={spec.index_by!r}) requires ear-resolved rows"
                )
            ear_axis = axis_names.index("ear")
            values = np.take(values, [int(row["ear_index"])], axis=ear_axis)
            values = np.squeeze(values, axis=ear_axis)
            axis_names.pop(ear_axis)
        if "frequency" in spec_index_by:
            if domain != "frequency":
                raise ValueError(
                    f"HRTFSpec(index_by={spec.index_by!r}) with frequency indexing requires domain='frequency'"
                )
            if row["frequency_index"] is None:
                raise ValueError(
                    f"HRTFSpec(index_by={spec.index_by!r}) requires frequency-resolved rows"
                )
            frequency_axis = axis_names.index("frequency")
            values = np.take(values, [int(row["frequency_index"])], axis=frequency_axis)
            values = np.squeeze(values, axis=frequency_axis)
            axis_names.pop(frequency_axis)
        if "samples" in spec_index_by:
            if domain != "time" or signal != "ir":
                raise ValueError(
                    f"HRTFSpec(index_by={spec.index_by!r}) with sample indexing requires domain='time' and signal='ir'"
                )
            if row["sample_index"] is None:
                raise ValueError(
                    f"HRTFSpec(index_by={spec.index_by!r}) requires sample-resolved rows"
                )
            sample_axis = axis_names.index("samples")
            values = np.take(values, [int(row["sample_index"])], axis=sample_axis)
            values = np.squeeze(values, axis=sample_axis)
        return np.asarray(values)

    @staticmethod
    def select_sh_value(
        values: np.ndarray,
        row: dict[str, str | int | None],
        selected_ears: list[tuple[str, int]],
        spec: SHSpec,
    ) -> np.ndarray:
        spec_index_by = normalize_index_by(spec.index_by)
        axis_names = ["coefficient", "frequency"]
        if values.ndim == 3:
            axis_names = ["coefficient", "ear", "frequency"]
        if "ear" not in spec_index_by:
            if "ear" in axis_names and len(selected_ears) == 1:
                ear_axis = axis_names.index("ear")
                values = np.take(values, [int(selected_ears[0][1])], axis=ear_axis)
                values = np.squeeze(values, axis=ear_axis)
                axis_names.pop(ear_axis)
        else:
            if row["ear_index"] is None:
                raise ValueError(
                    f"SHSpec(index_by={spec.index_by!r}) requires ear-resolved rows"
                )
            if "ear" not in axis_names:
                raise ValueError(
                    f"SHSpec(index_by={spec.index_by!r}) including 'ear' requires ears='both'"
                )
            ear_axis = axis_names.index("ear")
            values = np.take(values, [int(row["ear_index"])], axis=ear_axis)
            values = np.squeeze(values, axis=ear_axis)
            axis_names.pop(ear_axis)
        if "frequency" in spec_index_by:
            if row["frequency_index"] is None:
                raise ValueError(
                    f"SHSpec(index_by={spec.index_by!r}) requires frequency-resolved rows"
                )
            frequency_axis = axis_names.index("frequency")
            values = np.take(values, [int(row["frequency_index"])], axis=frequency_axis)
            values = np.squeeze(values, axis=frequency_axis)
        return np.asarray(values)

    def get_spec_value(
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
            return self.get_anthropometry_value(spec, subject_id)
        if isinstance(spec, ImageSpec):
            return self.select_media_value(
                self._image_index,
                subject_id,
                row,
                self._image_align_by,
                spec.transform,
                concatenate=spec.concatenate,
                resource_name="image",
            )
        if isinstance(spec, VideoSpec):
            return self.select_media_value(
                self._video_index,
                subject_id,
                row,
                self._video_align_by,
                spec.transform,
                resource_name="video",
            )
        raise TypeError(f"Unsupported dataset spec: {type(spec)!r}")

    def get_hrtf_spec_value(
        self,
        spec: HRTFSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        hrtf = self.get_subject_hrtf(subject_id)
        use_hrtf_transform = (
            spec.transform is not None and self.is_explicit_hrtf_transform(spec.transform)
        )
        if (
            spec.transform is not None
            and not use_hrtf_transform
            and self.is_raw_hrtf_transform_method(spec.transform)
        ):
            raise ValueError(
                "Raw Transform methods are not supported in HRTFSpec.transform. "
                "Use hrtfpykit.datasets.HRTFTransform instead."
            )
        transformed_hrtf = None
        if use_hrtf_transform:
            transform_cache_key = (subject_id, id(spec.transform))
            transformed_hrtf = self._transformed_hrtf_cache.get(transform_cache_key)
            if transformed_hrtf is None:
                transformed_hrtf = spec.transform(hrtf)
                if not self.is_hrtf_object(transformed_hrtf):
                    raise ValueError(
                        "HRTFTransform callables used in HRTFSpec.transform must return an HRTF object"
                    )
                if self._cache_hrtf:
                    self._transformed_hrtf_cache[transform_cache_key] = transformed_hrtf
        value = self.select_hrtf_value(
            hrtf=hrtf if transformed_hrtf is None else transformed_hrtf,
            row=row,
            selected_position_indices=self._selected_position_indices,
            selected_ears=self._selected_ears,
            spec=spec,
        )
        if spec.transform is not None and not use_hrtf_transform:
            value = spec.transform(value)
        return value

    def get_itd_spec_value(
        self,
        spec: ITDSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> np.ndarray:
        metric_cache_key = (subject_id, id(spec))
        value = self._metric_cache.get(metric_cache_key)
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
            self._metric_cache[metric_cache_key] = value
        spec_index_by = normalize_index_by(spec.index_by)
        if "position" not in spec_index_by:
            if len(self._selected_position_indices) != value.shape[0]:
                value = np.take(value, self._selected_position_indices, axis=0)
        else:
            if row["position_index"] is None:
                raise ValueError(
                    f"ITDSpec(index_by={spec.index_by!r}) requires position-resolved rows"
                )
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
        metric_cache_key = (subject_id, id(spec))
        value = self._metric_cache.get(metric_cache_key)
        if value is None:
            hrtf = self.get_subject_hrtf(subject_id)
            value = np.asarray(
                ild(
                    hrtf.IR,
                    sample_rate=self.sample_rate,
                    fft_length=spec.fft_length,
                    mode=spec.mode,
                    output=spec.output,
                    epsilon=spec.epsilon,
                )
            )
            self._metric_cache[metric_cache_key] = value
        spec_index_by = normalize_index_by(spec.index_by)
        if "position" not in spec_index_by:
            if value.shape[0] == self.available_positions.shape[0]:
                if len(self._selected_position_indices) != value.shape[0]:
                    value = np.take(value, self._selected_position_indices, axis=0)
        else:
            if row["position_index"] is None:
                raise ValueError(
                    f"ILDSpec(index_by={spec.index_by!r}) requires position-resolved rows"
                )
            if value.shape[0] == self.available_positions.shape[0]:
                value = np.asarray(value[int(row["position_index"])])
        if "frequency" in spec_index_by:
            if row["frequency_index"] is None:
                raise ValueError(
                    f"ILDSpec(index_by={spec.index_by!r}) requires frequency-resolved rows"
                )
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
        sh_cache_key = (subject_id, id(spec))
        value = self._sh_cache.get(sh_cache_key)
        if value is None:
            hrtf = self.get_subject_hrtf(subject_id)
            value = np.asarray(
                sht(
                    hrtf,
                    sh_order=spec.sh_order,
                    ear=spec.ears,
                    epsilon=spec.epsilon,
                ).C
            )
            self._sh_cache[sh_cache_key] = value
        value = self.select_sh_value(
            values=value,
            row=row,
            selected_ears=self._selected_ears,
            spec=spec,
        )
        if spec.transform is not None:
            value = spec.transform(value)
        return value

    def get_anthropometry_value(
        self,
        spec: AnthropometrySpec,
        subject_id: str,
    ) -> dict[str, float | str | None]:
        value = select_anthropometry_value(
            values=self._anthropometry_rows[subject_id],
            select=spec.select,
            ear=spec.ear,
            dataset_name=self.name,
        )
        if spec.transform is not None:
            value = spec.transform(value)
        return value
