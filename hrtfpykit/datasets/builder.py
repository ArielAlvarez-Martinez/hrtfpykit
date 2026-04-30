from collections.abc import Sequence
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from .config import DatasetConfig
from .resources import resolve_dataset_resources
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
    get_spec_name,
    normalize_specs,
)
from ..hrtf.coordinates import get_spherical_positions
from ..hrtf.planes import get_frontal_plane, get_horizontal_plane, get_median_plane
from .summary import DatasetSummary
from .normalization import (
    normalize_grouped_by,
    normalize_index_by,
    normalize_ears,
    normalize_positions,
)

if TYPE_CHECKING:
    from .base import BaseDataset
    from ..hrtf.hrtf import HRTF


@dataclass(frozen=True)
class DatasetBuildPlan:
    input_specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
    target_specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
    specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
    input_names: tuple[str, ...]
    target_names: tuple[str, ...]
    index_by: tuple[str, ...]
    selected_ears: tuple[tuple[str, int], ...]
    position_one_hot: bool
    position_index: bool
    frequency_one_hot: bool
    frequency_index: bool
    sample_one_hot: bool
    sample_index: bool
    ear_one_hot: bool
    ear_index: bool


class DatasetSubjectResolver:
    @staticmethod
    def normalize_subject_id(value: str) -> str:
        return str(value).strip().lower()

    @classmethod
    def resolve_subject_id(
        cls,
        value: str | int,
        subject_ids: tuple[str, ...],
    ) -> str:
        if len(subject_ids) == 0:
            raise ValueError("subject_ids must not be empty")
        if isinstance(value, int):
            index = int(value)
            if index < 1 or index > len(subject_ids):
                raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
            return subject_ids[index - 1]
        text = str(value).strip()
        if text == "":
            raise ValueError("Subject reference must not be empty")
        subject_map = {subject_id.lower(): subject_id for subject_id in subject_ids}
        if text.lower() in subject_map:
            return subject_map[text.lower()]
        subject_index_match = re.fullmatch(r"subject[_-]?(\d+)", text.strip().lower())
        if subject_index_match is not None:
            index = int(subject_index_match.group(1))
            if index < 1 or index > len(subject_ids):
                raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
            return subject_ids[index - 1]
        if text.isdigit():
            index = int(text)
            if index < 1 or index > len(subject_ids):
                raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
            return subject_ids[index - 1]
        normalized = cls.normalize_subject_id(text)
        if normalized.lower() in subject_map:
            return subject_map[normalized.lower()]
        raise ValueError(f"Unknown subject reference {value!r}")

    @classmethod
    def resolve_subject_ids(
        cls,
        values: str | int | tuple[str | int, ...] | list[str | int] | None,
        subject_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if values is None:
            return tuple()
        if isinstance(values, (str, int)):
            return (cls.resolve_subject_id(values, subject_ids),)
        return tuple(
            dict.fromkeys(
                cls.resolve_subject_id(value, subject_ids)
                for value in values
            )
        )

    @staticmethod
    def sort_subject_ids(subject_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        def subject_sort_key(value: str) -> tuple[int, str]:
            match = re.search(r"(\d+)$", str(value))
            if match is None:
                return (0, str(value).lower())
            return (int(match.group(1)), str(value).lower())

        return sorted(subject_ids, key=subject_sort_key)


class DatasetSpecPlanner:
    def __init__(self, config: type[DatasetConfig] | DatasetConfig) -> None:
        self.config = config

    def build(
        self,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None,
    ) -> DatasetBuildPlan:
        input_specs = normalize_specs(inputs)
        target_specs = normalize_specs(target)
        specs = input_specs + target_specs
        if len(specs) == 0:
            raise ValueError("Dataset requires at least one dataset spec in inputs or target")

        for spec_type, resource_name in (
            (AnthropometrySpec, "anthropometry"),
            (ImageSpec, "image"),
            (VideoSpec, "video"),
        ):
            explicit_paths = tuple(
                str(spec.path)
                for spec in specs
                if isinstance(spec, spec_type) and getattr(spec, "path", None) is not None
            )
            if len(set(explicit_paths)) > 1:
                raise ValueError(
                    f"{resource_name} specs must not define different paths in the same dataset"
                )

        dataset_index_by = None
        dataset_index_by_spec: str | None = None
        for spec in self.get_indexed_specs(specs):
            spec_index_by = normalize_index_by(spec.index_by)
            spec_name = get_spec_name(spec)
            spec_axes = set(spec_index_by[1:])
            if isinstance(spec, HRTFSpec):
                domain = str(spec.domain).strip().lower()
                supported_axes = {"position", "ear", "samples"} if domain == "time" else {"position", "ear", "frequency"}
                if domain == "time":
                    supported_index_by = (
                        "('subject',), ('subject', 'position'), ('subject', 'ear'), "
                        "('subject', 'samples'), ('subject', 'position', 'ear'), "
                        "('subject', 'position', 'samples'), ('subject', 'ear', 'samples'), "
                        "('subject', 'position', 'ear', 'samples')"
                    )
                else:
                    supported_index_by = (
                        "('subject',), ('subject', 'position'), ('subject', 'ear'), "
                        "('subject', 'frequency'), ('subject', 'position', 'ear'), "
                        "('subject', 'position', 'frequency'), ('subject', 'ear', 'frequency'), "
                        "('subject', 'position', 'ear', 'frequency')"
                    )
            elif isinstance(spec, ITDSpec):
                supported_axes = {"position"}
                supported_index_by = "('subject',), ('subject', 'position')"
            elif isinstance(spec, ILDSpec):
                supported_axes = {"position"}
                if str(spec.mode).strip().lower() == "frequency-dependent":
                    supported_axes.add("frequency")
                    supported_index_by = (
                        "('subject',), ('subject', 'position'), ('subject', 'frequency'), "
                        "('subject', 'position', 'frequency')"
                    )
                else:
                    supported_index_by = "('subject',), ('subject', 'position')"
            elif isinstance(spec, SHSpec):
                supported_axes = {"ear", "frequency"}
                supported_index_by = (
                    "('subject',), ('subject', 'ear'), ('subject', 'frequency'), "
                    "('subject', 'ear', 'frequency')"
                )
            else:
                supported_axes = set()
                supported_index_by = "()"

            unsupported_axes = sorted(spec_axes - supported_axes)
            if len(unsupported_axes) > 0:
                compatibility_hint = ""
                if isinstance(spec, HRTFSpec):
                    domain = str(spec.domain).strip().lower()
                    if "frequency" in unsupported_axes and domain == "time":
                        compatibility_hint = (
                            " In HRTFSpec, the 'frequency' axis is available only when domain='frequency'."
                        )
                    elif "samples" in unsupported_axes and domain == "frequency":
                        compatibility_hint = (
                            " In HRTFSpec, the 'samples' axis is available only when domain='time'."
                        )
                elif isinstance(spec, ILDSpec):
                    if "frequency" in unsupported_axes and str(spec.mode).strip().lower() != "frequency-dependent":
                        compatibility_hint = " In ILDSpec, enable frequency indexing by setting mode='frequency-dependent'."

                raise ValueError(
                    f"{type(spec).__name__} index_by={spec_index_by!r} uses unsupported axes: "
                    + ", ".join(unsupported_axes)
                    + ". "
                    f"Supported index_by combinations for {type(spec).__name__}: {supported_index_by}."
                    + compatibility_hint
                )

            for flag_name, axis_name in (
                ("position_one_hot", "position"),
                ("position_index", "position"),
                ("ear_one_hot", "ear"),
                ("ear_index", "ear"),
                ("frequency_one_hot", "frequency"),
                ("frequency_index", "frequency"),
                ("sample_one_hot", "samples"),
                ("sample_index", "samples"),
            ):
                if bool(getattr(spec, flag_name, False)) and axis_name not in spec_index_by:
                    compatibility_hint = ""
                    if isinstance(spec, HRTFSpec):
                        domain = str(spec.domain).strip().lower()
                        if axis_name == "frequency" and domain == "time":
                            compatibility_hint = (
                                " In HRTFSpec, set domain='frequency' to use frequency-indexed specs."
                            )
                        elif axis_name == "samples" and domain == "frequency":
                            compatibility_hint = (
                                " In HRTFSpec, set domain='time' to use sample-indexed specs."
                            )
                    elif isinstance(spec, ILDSpec):
                        if axis_name == "frequency":
                            compatibility_hint = (
                                " In ILDSpec, set mode='frequency-dependent' to use frequency indexing."
                            )

                    raise ValueError(
                        f"{type(spec).__name__}.{flag_name} requires index_by to include {axis_name!r}. "
                        f"Supported index_by combinations for {type(spec).__name__}: {supported_index_by}."
                        + compatibility_hint
                    )
            if dataset_index_by is None:
                dataset_index_by = spec_index_by
                dataset_index_by_spec = spec_name
            elif spec_index_by != dataset_index_by:
                raise ValueError(
                    "All indexed specs in a dataset must use the same index_by. "
                    f"{spec_name!r} uses {spec_index_by!r}, but {dataset_index_by_spec!r} uses {dataset_index_by!r}. "
                    "Pick one index_by for the full dataset."
                )

        for spec in self.filter_specs((ImageSpec, VideoSpec, AnthropometrySpec), specs):
            grouped_by = normalize_grouped_by(spec.grouped_by)
            if isinstance(spec, (ImageSpec, VideoSpec)):
                config = self.config.image if isinstance(spec, ImageSpec) else self.config.video
                if config is not None and grouped_by not in tuple(config.supported_grouped_by):
                    raise ValueError(
                        f"{type(spec).__name__} grouped_by={grouped_by!r} is not supported by {self.config.name}"
                    )

        input_names = tuple(get_spec_name(spec) for spec in input_specs)
        target_names = tuple(get_spec_name(spec) for spec in target_specs)

        indexed_specs = self.get_indexed_specs(specs)
        index_by = ("subject",) if len(indexed_specs) == 0 else normalize_index_by(indexed_specs[0].index_by)
        media_specs = self.filter_specs((ImageSpec, VideoSpec, AnthropometrySpec), specs)
        if index_by == ("subject",) and any(
            "ear" in normalize_grouped_by(spec.grouped_by) for spec in media_specs
        ):
            index_by = ("subject", "ear")
        if "ear" not in index_by:
            for spec in media_specs:
                grouped_by = normalize_grouped_by(spec.grouped_by)
                if "ear" in grouped_by:
                    raise ValueError(
                        f"{type(spec).__name__} grouped_by={grouped_by!r} requires an ear-indexed dataset row"
                    )
                if bool(spec.ear_one_hot) or bool(spec.ear_index):
                    raise ValueError(
                        f"{type(spec).__name__} ear encodings require an ear-indexed dataset row"
                    )

        selected_ears: tuple[tuple[str, int], ...] = tuple()
        if "ear" in index_by:
            ear_specs = self.filter_specs((HRTFSpec, SHSpec, AnthropometrySpec), specs)
            for spec in indexed_specs:
                if "ear" not in normalize_index_by(spec.index_by):
                    continue
                if isinstance(spec, (HRTFSpec, SHSpec)):
                    spec_ears = tuple(normalize_ears(spec.ears))
                else:
                    continue
                if len(selected_ears) == 0:
                    selected_ears = spec_ears
                elif spec_ears != selected_ears:
                    raise ValueError(
                        "All ear-indexed specs must use the same ear axis. "
                        f"Expected {selected_ears!r}, got {spec_ears!r} for {type(spec).__name__}"
                    )
            if len(selected_ears) == 0:
                for spec in ear_specs:
                    if not isinstance(spec, AnthropometrySpec):
                        continue
                    if "ear" not in normalize_grouped_by(spec.grouped_by):
                        continue
                    spec_ears = normalize_ears(spec.ear if spec.ear is not None else "both")
                    if len(selected_ears) == 0:
                        selected_ears = tuple(spec_ears)
                    elif tuple(spec_ears) != selected_ears:
                        raise ValueError(
                            "All ear-indexed specs must use the same ear axis. "
                            f"Expected {selected_ears!r}, got {tuple(spec_ears)!r} for {type(spec).__name__}"
                        )
            selected_ears = tuple(normalize_ears("both")) if len(selected_ears) == 0 else selected_ears

        return DatasetBuildPlan(
            input_specs=input_specs,
            target_specs=target_specs,
            specs=specs,
            input_names=input_names,
            target_names=target_names,
            index_by=index_by,
            selected_ears=selected_ears,
            position_one_hot=any(
                bool(spec.position_one_hot)
                for spec in self.filter_specs((HRTFSpec, ITDSpec, ILDSpec), specs)
            ),
            position_index=any(
                bool(spec.position_index)
                for spec in self.filter_specs((HRTFSpec, ITDSpec, ILDSpec), specs)
            ),
            frequency_one_hot=any(
                bool(spec.frequency_one_hot)
                for spec in self.filter_specs((HRTFSpec, ILDSpec, SHSpec), specs)
            ),
            frequency_index=any(
                bool(spec.frequency_index)
                for spec in self.filter_specs((HRTFSpec, ILDSpec, SHSpec), specs)
            ),
            sample_one_hot=any(bool(spec.sample_one_hot) for spec in self.filter_specs((HRTFSpec,), specs)),
            sample_index=any(bool(spec.sample_index) for spec in self.filter_specs((HRTFSpec,), specs)),
            ear_one_hot=any(
                bool(spec.ear_one_hot)
                for spec in self.filter_specs((HRTFSpec, SHSpec, ImageSpec, VideoSpec, AnthropometrySpec), specs)
            ),
            ear_index=any(
                bool(spec.ear_index)
                for spec in self.filter_specs((HRTFSpec, SHSpec, ImageSpec, VideoSpec, AnthropometrySpec), specs)
            ),
        )

    @staticmethod
    def get_indexed_specs(
        specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...],
    ) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...]:
        return cast(
            tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...],
            tuple(spec for spec in specs if isinstance(spec, (HRTFSpec, ITDSpec, ILDSpec, SHSpec))),
        )

    @staticmethod
    def filter_specs(
        spec_types: type[object] | tuple[type[object], ...],
        specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...],
    ) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]:
        return tuple(spec for spec in specs if isinstance(spec, spec_types))


@dataclass(frozen=True)
class DatasetSubjectSplitPlan:
    available_subject_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    split: str
    split_ratio: tuple[float, float, float]
    split_seed: int


class DatasetSubjectSelectionPlanner:
    @staticmethod
    def split_subject_ids(
        subject_ids: Sequence[str],
        split: str,
        split_ratio: tuple[float, float, float],
        split_seed: int,
    ) -> list[str]:
        split_key = str(split).strip().lower()
        if split_key == "all":
            return list(subject_ids)
        if split_key not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: all, train, validation, test")
        if len(split_ratio) != 3:
            raise ValueError("split_ratio must contain three values")
        train_ratio, validation_ratio, test_ratio = split_ratio
        total = float(train_ratio + validation_ratio + test_ratio)
        if not np.isclose(total, 1.0):
            raise ValueError("split_ratio values must sum to 1.0")
        rng = np.random.default_rng(split_seed)
        shuffled = list(subject_ids)
        if len(shuffled) > 1:
            shuffled = [shuffled[index] for index in rng.permutation(len(shuffled))]
        raw_counts = np.asarray(
            [
                len(shuffled) * float(train_ratio),
                len(shuffled) * float(validation_ratio),
                len(shuffled) * float(test_ratio),
            ],
            dtype=float,
        )
        counts = np.floor(raw_counts).astype(int)
        remainder = int(len(shuffled) - int(counts.sum()))
        if remainder > 0:
            for index in np.argsort(-(raw_counts - counts))[:remainder]:
                counts[int(index)] += 1
        train_end = int(counts[0])
        validation_end = int(counts[0] + counts[1])
        if split_key == "train":
            return shuffled[:train_end]
        if split_key == "validation":
            return shuffled[train_end:validation_end]
        return shuffled[validation_end:]

    def build(
        self,
        dataset: "BaseDataset",
        split: str,
        split_ratio: tuple[float, float, float],
        split_seed: int,
    ) -> DatasetSubjectSplitPlan:
        required_subject_sets: list[set[str]] = []
        if len(dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))) > 0:
            required_subject_sets.append(set(dataset._hrtf_paths))
        if len(dataset._get_specs(MeshSpec)) > 0:
            required_subject_sets.append(set(dataset._mesh_paths))
        if len(dataset._get_specs(AnthropometrySpec)) > 0:
            required_subject_sets.append(set(dataset._anthropometry_rows))
        if len(dataset._get_specs(ImageSpec)) > 0:
            required_subject_sets.append({key[0] for key in dataset._image_index})
        if len(dataset._get_specs(VideoSpec)) > 0:
            required_subject_sets.append({key[0] for key in dataset._video_index})

        if len(required_subject_sets) == 0:
            subject_ids = type(dataset)._sort_subject_ids(list(dataset._included_subject_ids))
        else:
            subject_ids = type(dataset)._sort_subject_ids(set.intersection(*required_subject_sets))
        if len(subject_ids) == 0 and len(required_subject_sets) > 0:
            available_counts = []
            if len(dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))) > 0:
                available_counts.append(f"hrtf={len(dataset._hrtf_paths)}")
            if len(dataset._get_specs(MeshSpec)) > 0:
                available_counts.append(f"mesh={len(dataset._mesh_paths)}")
            if len(dataset._get_specs(AnthropometrySpec)) > 0:
                available_counts.append(f"anthropometry={len(dataset._anthropometry_rows)}")
            if len(dataset._get_specs(ImageSpec)) > 0:
                available_counts.append(f"image={len({key[0] for key in dataset._image_index})}")
            if len(dataset._get_specs(VideoSpec)) > 0:
                available_counts.append(f"video={len({key[0] for key in dataset._video_index})}")
            raise ValueError(
                "No subjects match the selected dataset configuration. "
                f"Selected specs: {', '.join(sorted(set(dataset._input_names + dataset._target_names)))}. "
                f"Available subject counts by spec: {', '.join(available_counts)}. "
                f"Root: {dataset._root}\n"
                f"{DatasetSummary.format_resource_summary(dataset._resource_summary)}"
            )

        selected_subject_ids = self.split_subject_ids(subject_ids, split, split_ratio, split_seed)
        if len(selected_subject_ids) == 0:
            raise ValueError(f"Split {split!r} produced an empty dataset")

        return DatasetSubjectSplitPlan(
            available_subject_ids=tuple(subject_ids),
            subject_ids=tuple(selected_subject_ids),
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )


@dataclass(frozen=True)
class DatasetAcousticContext:
    dataset_sample_rate: float | None
    dataset_source_positions: np.ndarray | None
    available_azimuth_angles: np.ndarray | None
    available_elevation_angles: np.ndarray | None
    azimuth_angles: np.ndarray | None
    elevation_angles: np.ndarray | None
    frequency_bins: np.ndarray | None
    sample_indices: np.ndarray | None
    selected_position_indices: tuple[int, ...]
    selected_frequency_indices: tuple[int, ...]
    selected_sample_indices: tuple[int, ...]
    spec_position_indices: tuple[tuple[int, tuple[int, ...]], ...]


class DatasetAcousticContextBuilder:
    @staticmethod
    def resolve_positions_selection(
        positions: str | tuple[int, ...] | list[int] | np.ndarray,
        plane: str | tuple[object, ...] | dict[str, object] | None,
        hrtf: "HRTF",
    ) -> list[int]:
        position_count = int(hrtf.Sources.get_positions().shape[0])
        if plane is None:
            return normalize_positions(positions, position_count)
        if not isinstance(positions, str) or str(positions).strip().lower() != "all":
            raise ValueError("plane selection cannot be combined with custom positions")
        if isinstance(plane, str):
            plane_key = str(plane).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = default_angle
            angle_unit = "degrees"
        elif isinstance(plane, tuple):
            if len(plane) not in {2, 3} or not isinstance(plane[0], str):
                raise ValueError(
                    "Plane selection must be ('horizontal'|'median'|'frontal', angle[, angle_unit])"
                )
            plane_key = str(plane[0]).strip().lower()
            angle = plane[1]
            angle_unit = "degrees" if len(plane) == 2 else str(plane[2]).strip().lower()
        else:
            plane_key = str(plane.get("plane")).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = plane.get("angle", plane.get("plane_angle", default_angle))
            angle_unit = str(plane.get("angle_unit", "degrees")).strip().lower()
        if plane_key not in {"horizontal", "median", "frontal"}:
            raise ValueError("plane must be horizontal, median, or frontal")
        if plane_key == "horizontal":
            indices, _ = get_horizontal_plane(
                hrtf=hrtf,
                elevation=float(angle),
                angle_unit=angle_unit,
            )
        elif plane_key == "median":
            indices, _ = get_median_plane(
                hrtf=hrtf,
                azimuth=float(angle),
                angle_unit=angle_unit,
            )
        else:
            indices, _ = get_frontal_plane(
                hrtf=hrtf,
                azimuth=float(angle),
                angle_unit=angle_unit,
            )
        return [int(index) for index in np.asarray(indices, dtype=int).reshape(-1)]

    def build(self, dataset: "BaseDataset") -> DatasetAcousticContext:
        acoustic_specs = dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))
        if len(acoustic_specs) == 0:
            return DatasetAcousticContext(
                dataset_sample_rate=None,
                dataset_source_positions=None,
                available_azimuth_angles=None,
                available_elevation_angles=None,
                azimuth_angles=None,
                elevation_angles=None,
                frequency_bins=None,
                sample_indices=None,
                selected_position_indices=(),
                selected_frequency_indices=(),
                selected_sample_indices=(),
                spec_position_indices=(),
            )

        sample_subject_id = dataset._subject_ids[0]
        sample_hrtf = dataset.get_subject_hrtf(sample_subject_id)
        dataset_sample_rate = (
            None if sample_hrtf.IR.sample_rate is None else float(sample_hrtf.IR.sample_rate)
        )
        dataset_source_positions = np.asarray(
            sample_hrtf.Sources.get_positions(angle_unit="degrees"),
            dtype=float,
        )
        frequency_bins = (
            None if sample_hrtf.TF.frequency_bins is None else np.asarray(sample_hrtf.TF.frequency_bins, dtype=float)
        )
        selected_frequency_indices = () if frequency_bins is None else tuple(range(int(frequency_bins.shape[0])))
        sample_indices = np.arange(sample_hrtf.IR.values.shape[-1], dtype=int)
        selected_sample_indices = tuple(range(int(sample_indices.shape[0])))

        position_axis: tuple[int, ...] | None = None
        position_axis_spec: str | None = None
        frequency_count: int | None = None
        frequency_count_spec: str | None = None
        sample_count: int | None = None
        sample_count_spec: str | None = None
        spec_position_indices: list[tuple[int, tuple[int, ...]]] = []

        for spec in DatasetSpecPlanner.filter_specs((HRTFSpec, ITDSpec, ILDSpec), dataset._specs):
            indices = DatasetAcousticContextBuilder.resolve_positions_selection(
                spec.positions,
                spec.plane,
                sample_hrtf,
            )
            spec_position_indices.append((id(spec), tuple(indices)))
            if "position" not in normalize_index_by(spec.index_by):
                continue
            axis = tuple(indices)
            if position_axis is None:
                position_axis = axis
                position_axis_spec = get_spec_name(spec)
            elif axis != position_axis:
                current_spec_name = get_spec_name(spec)
                raise ValueError(
                    "All position-indexed specs in a dataset must use the same selected positions. "
                    f"{current_spec_name!r} selects {len(axis)} positions, but {position_axis_spec!r} selects {len(position_axis)}. "
                    "Pick one position selection for the full dataset."
                )

        for spec in DatasetSpecPlanner.get_indexed_specs(dataset._specs):
            spec_name = get_spec_name(spec)
            spec_index_by = normalize_index_by(spec.index_by)
            if "frequency" in spec_index_by:
                if isinstance(spec, (HRTFSpec, SHSpec)):
                    if sample_hrtf.TF.frequency_bins is None:
                        raise ValueError("Frequency-indexed specs require available HRTF frequency bins")
                    current_frequency_count = int(np.asarray(sample_hrtf.TF.frequency_bins).reshape(-1).shape[0])
                elif isinstance(spec, ILDSpec):
                    fft_length = (
                        int(spec.fft_length)
                        if spec.fft_length is not None
                        else int(sample_hrtf.IR.values.shape[-1])
                    )
                    current_frequency_count = int(fft_length // 2 + 1)
                else:
                    continue
                if frequency_count is None:
                    frequency_count = current_frequency_count
                    frequency_count_spec = spec_name
                elif current_frequency_count != frequency_count:
                    raise ValueError(
                        "All frequency-indexed specs must use the same number of frequency bins. "
                        f"{spec_name!r} selects {current_frequency_count} bins, but "
                        f"{frequency_count_spec!r} selects {frequency_count}. "
                        "Pick one frequency selection for the full dataset."
                    )
            if "samples" in spec_index_by:
                current_sample_count = int(sample_hrtf.IR.values.shape[-1])
                if sample_count is None:
                    sample_count = current_sample_count
                    sample_count_spec = spec_name
                elif current_sample_count != sample_count:
                    raise ValueError(
                        "All sample-indexed specs must use the same number of samples. "
                        f"{spec_name!r} selects {current_sample_count} samples, "
                        f"but {sample_count_spec!r} selects {sample_count}. "
                        "Pick one sample selection for the full dataset."
                    )

        selected_position_indices = () if position_axis is None else position_axis
        if frequency_count is not None:
            selected_frequency_indices = tuple(range(int(frequency_count)))
        if sample_count is not None:
            selected_sample_indices = tuple(range(int(sample_count)))

        spherical_positions = np.asarray(
            get_spherical_positions(sample_hrtf.Sources, angle_unit="degrees"),
            dtype=float,
        )
        available_azimuth_angles = np.unique(np.round(spherical_positions[:, 0], 2))
        available_elevation_angles = np.unique(np.round(spherical_positions[:, 1], 2))
        if len(selected_position_indices) > 0:
            selected_spherical_positions = np.asarray(
                spherical_positions[list(selected_position_indices)],
                dtype=float,
            )
            azimuth_angles = np.unique(np.round(selected_spherical_positions[:, 0], 2))
            elevation_angles = np.unique(np.round(selected_spherical_positions[:, 1], 2))
        else:
            azimuth_angles = None
            elevation_angles = None

        return DatasetAcousticContext(
            dataset_sample_rate=dataset_sample_rate,
            dataset_source_positions=dataset_source_positions,
            available_azimuth_angles=available_azimuth_angles,
            available_elevation_angles=available_elevation_angles,
            azimuth_angles=azimuth_angles,
            elevation_angles=elevation_angles,
            frequency_bins=frequency_bins,
            sample_indices=sample_indices,
            selected_position_indices=selected_position_indices,
            selected_frequency_indices=selected_frequency_indices,
            selected_sample_indices=selected_sample_indices,
            spec_position_indices=tuple(spec_position_indices),
        )


class DatasetRowsBuilder:
    @staticmethod
    def build(
        subject_ids: tuple[str, ...],
        index_by: tuple[str, ...],
        selected_position_indices: tuple[int, ...],
        selected_ears: tuple[tuple[str, int], ...] | list[tuple[str, int]],
        selected_frequency_indices: tuple[int, ...],
        selected_sample_indices: tuple[int, ...],
    ) -> list[dict[str, str | int | None]]:
        rows: list[dict[str, str | int | None]] = []
        include_position = "position" in index_by
        include_ear = "ear" in index_by
        include_frequency = "frequency" in index_by
        include_samples = "samples" in index_by
        for subject_id in subject_ids:
            position_values = (
                [(None, None)]
                if not include_position
                else [
                    (int(position_index), int(selected_position_index))
                    for selected_position_index, position_index in enumerate(selected_position_indices)
                ]
            )
            ear_values = (
                [(None, None, None)]
                if not include_ear
                else [
                    (ear_name, int(ear_index), int(selected_ear_index))
                    for selected_ear_index, (ear_name, ear_index) in enumerate(selected_ears)
                ]
            )
            frequency_values = (
                [(None, None)]
                if not include_frequency
                else [
                    (int(frequency_index), int(selected_frequency_index))
                    for selected_frequency_index, frequency_index in enumerate(selected_frequency_indices)
                ]
            )
            sample_values = (
                [(None, None)]
                if not include_samples
                else [
                    (int(sample_index), int(selected_sample_index))
                    for selected_sample_index, sample_index in enumerate(selected_sample_indices)
                ]
            )
            for position_index, selected_position_index in position_values:
                for ear_name, ear_index, selected_ear_index in ear_values:
                    for frequency_index, selected_frequency_index in frequency_values:
                        for sample_index, selected_sample_index in sample_values:
                            rows.append(
                                {
                                    "subject_id": subject_id,
                                    "position_index": position_index,
                                    "selected_position_index": selected_position_index,
                                    "ear": ear_name,
                                    "ear_index": ear_index,
                                    "selected_ear_index": selected_ear_index,
                                    "frequency_index": frequency_index,
                                    "selected_frequency_index": selected_frequency_index,
                                    "sample_index": sample_index,
                                    "selected_sample_index": selected_sample_index,
                                }
                            )
        return rows
