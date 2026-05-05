from dataclasses import dataclass
from collections.abc import Sequence
import re

import numpy as np
from typing import TYPE_CHECKING

from .specs_registry import has_specs
from .sanitize import sanitize_subject_id

if TYPE_CHECKING:
    from .base import BaseDataset


@dataclass(frozen=True)
class DatasetSubjectSplitPlan:
    available_subjects: tuple[str, ...]
    selected_subjects: tuple[str, ...]
    split: str
    split_ratio: tuple[float, float, float]
    split_seed: int


class DatasetSubjectSplitPlanner:
    @classmethod
    def map_subject_id(
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
        normalized = sanitize_subject_id(text)
        if normalized.lower() in subject_map:
            return subject_map[normalized.lower()]
        raise ValueError(f"Unknown subject reference {value!r}")

    @classmethod
    def map_subject_ids(
        cls,
        values: str | int | tuple[str | int, ...] | list[str | int] | None,
        subject_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if values is None:
            return tuple()
        if isinstance(values, (str, int)):
            return (cls.map_subject_id(values, subject_ids),)
        return tuple(dict.fromkeys(cls.map_subject_id(value, subject_ids) for value in values))

    @staticmethod
    def sort_subject_ids(subject_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        def subject_sort_key(value: str) -> tuple[int, str]:
            match = re.search(r"(\d+)$", str(value))
            if match is None:
                return (0, str(value).lower())
            return (int(match.group(1)), str(value).lower())

        return sorted(subject_ids, key=subject_sort_key)

    @staticmethod
    def build_subject_number_map(subject_ids: tuple[str, ...]) -> dict[str, int]:
        return {
            str(subject_id): int(match.group(1))
            if (match is not None and match.group(1) != "")
            else index
            for index, (match, subject_id) in enumerate(
                (
                    (re.search(r"(\d+)$", str(subject_id)), str(subject_id))
                    for subject_id in subject_ids
                ),
                start=1,
            )
        }

    @staticmethod
    def prepare_subject_scope(
        dataset: "BaseDataset",
    ) -> tuple[tuple[str, ...], dict[str, int]]:
        state = dataset._state
        config = state.config
        if config is None:
            raise ValueError("Dataset config is not initialized")
        excluded_subjects = set(state.excluded_subjects)
        sorted_subjects = tuple(DatasetSubjectSplitPlanner.sort_subject_ids(config.subject_ids))
        available_subjects = tuple(
            subject_id
            for subject_id in sorted_subjects
            if subject_id not in excluded_subjects
        )
        subject_numbers = DatasetSubjectSplitPlanner.build_subject_number_map(
            sorted_subjects
        )
        return available_subjects, subject_numbers

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
                len(shuffled) * validation_ratio,
                len(shuffled) * test_ratio,
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

    @staticmethod
    def build(
        dataset: "BaseDataset",
        split: str,
        split_ratio: tuple[float, float, float],
        split_seed: int,
    ) -> DatasetSubjectSplitPlan:
        state = dataset._state
        config = state.config
        if config is None:
            raise ValueError("Dataset config is not initialized")
        resource_subjects, _ = DatasetSubjectSplitPlanner.prepare_subject_scope(dataset)
        has_acoustic_specs = has_specs(state.specs, resource_name="hrtf")
        has_mesh_specs = has_specs(state.specs, resource_name="mesh")
        has_anthro_specs = has_specs(state.specs, resource_name="anthropometry")
        has_metadata_specs = has_specs(state.specs, resource_name="metadata")
        has_image_specs = has_specs(state.specs, resource_name="image")
        has_video_specs = has_specs(state.specs, resource_name="video")
        required_subject_sets: list[set[str]] = []
        if has_acoustic_specs:
            required_subject_sets.append(set(state.hrtf_paths))
        if has_mesh_specs:
            required_subject_sets.append(set(state.mesh_paths))
        if has_anthro_specs:
            required_subject_sets.append(set(state.anthropometry_rows))
        if has_metadata_specs:
            required_subject_sets.append(set(state.metadata_rows))
        if has_image_specs:
            required_subject_sets.append({key[0] for key in state.image_index})
        if has_video_specs:
            required_subject_sets.append({key[0] for key in state.video_index})

        if len(required_subject_sets) == 0:
            subject_ids = DatasetSubjectSplitPlanner.sort_subject_ids(
                list(resource_subjects)
            )
        else:
            subject_ids = DatasetSubjectSplitPlanner.sort_subject_ids(
                set.intersection(*required_subject_sets)
            )
        if len(subject_ids) == 0 and len(required_subject_sets) > 0:
            available_counts = []
            if has_acoustic_specs:
                available_counts.append(f"hrtf={len(state.hrtf_paths)}")
            if has_mesh_specs:
                available_counts.append(f"mesh={len(state.mesh_paths)}")
            if has_anthro_specs:
                available_counts.append(f"anthropometry={len(state.anthropometry_rows)}")
            if has_metadata_specs:
                available_counts.append(f"metadata={len(state.metadata_rows)}")
            if has_image_specs:
                available_counts.append(f"image={len({key[0] for key in state.image_index})}")
            if has_video_specs:
                available_counts.append(f"video={len({key[0] for key in state.video_index})}")
            if len(state.resource_summary) == 0:
                resource_lines = ["Resource summary: none"]
            else:
                resource_lines = ["Resource summary:"]
                for resource_name, summary in state.resource_summary.items():
                    parts = [str(resource_name)]
                    for key in (
                        "pattern",
                        "path",
                        "hrtf_variant",
                        "extensions",
                        "checked",
                        "found",
                        "valid",
                        "invalid",
                        "missing",
                        "subjects",
                        "rows",
                    ):
                        if key in summary:
                            parts.append(f"{key}={summary[key]!r}")
                    if "missing_subject_ids" in summary:
                        missing_subject_ids = tuple(summary["missing_subject_ids"])
                        if len(missing_subject_ids) > 0:
                            preview = ", ".join(
                                str(value) for value in missing_subject_ids[:5]
                            )
                            if len(missing_subject_ids) > 5:
                                preview = f"{preview}, ..."
                            parts.append(f"missing_subject_ids={preview}")
                    if "invalid_subject_ids" in summary:
                        invalid_subject_ids = tuple(summary["invalid_subject_ids"])
                        if len(invalid_subject_ids) > 0:
                            preview = ", ".join(
                                str(value) for value in invalid_subject_ids[:5]
                            )
                            if len(invalid_subject_ids) > 5:
                                preview = f"{preview}, ..."
                            parts.append(f"invalid_subject_ids={preview}")
                    resource_lines.append("  " + parts[0] + ": " + ", ".join(parts[1:]))
            resource_summary_text = "\n".join(resource_lines)
            raise ValueError(
                "No subjects match the selected dataset configuration. "
                f"Selected specs: {', '.join(sorted(set(state.input_names + state.target_names)))}. "
                f"Available subject counts by spec: {', '.join(available_counts)}. "
                f"Root: {state.root}\n"
                f"{resource_summary_text}"
            )

        selected_subject_ids = DatasetSubjectSplitPlanner.split_subject_ids(
            subject_ids,
            split,
            split_ratio,
            split_seed,
        )
        if len(selected_subject_ids) == 0:
            raise ValueError(f"Split {split!r} produced an empty dataset")

        return DatasetSubjectSplitPlan(
            available_subjects=tuple(subject_ids),
            selected_subjects=tuple(selected_subject_ids),
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
