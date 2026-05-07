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
class DatasetSplitPlan:
    """Store available and selected subjects after split planning.

    Parameters
    ----------
    available_subjects : tuple of str
        Subjects available after resource intersection and exclusions.
    selected_subjects : tuple of str
        Subjects selected for the requested split.
    split : str
        Requested split name.
    split_ratio : tuple of float
        Train, validation, and test split ratios.
    split_seed : int
        Seed used for deterministic split assignment.

    Returns
    -------
    DatasetSplitPlan Immutable split plan consumed by ``DatasetBuilder``.

    Use Cases
    ---------
    - Keep resource availability separate from train/validation/test selection.
    - Build dataset rows from selected split subjects.
    - Report available and selected subject counts in summaries.
    """

    available_subjects: tuple[str, ...]
    selected_subjects: tuple[str, ...]
    split: str
    split_ratio: tuple[float, float, float]
    split_seed: int


class DatasetSplitPlanner:
    """Plan dataset subject mapping, sorting, exclusion, and split selection.

    This utility normalizes subject references, intersects resource availability,
    and produces deterministic train, validation, and test subject selections.

    Use Cases
    ---------
    - Normalize subject references such as ``1``, ``'pp1'``, or ``'subject_1'``.
    - Produce deterministic train/validation/test subject splits.
    - Intersect selected specs with available resource subject sets.
    """

    @classmethod
    def map_subject_id(
        cls,
        value: str | int,
        subject_ids: tuple[str, ...],
    ) -> str:
        """Map one user subject reference to a canonical subject ID.

        The mapper accepts exact IDs, integer positions, numeric strings, and
        ``subject1``/``subject_1`` style references. It always returns the canonical
        ID from the dataset config so downstream state never mixes user aliases with
        official subject IDs.

        Parameters
        ----------
        value : str or int
            User subject reference.
        subject_ids : tuple of str
            Canonical subject IDs accepted by the dataset.

        Returns
        -------
        str Canonical subject ID.

        Use Cases
        ---------
        - Accept integer subject references.
        - Accept ``subject1`` or ``subject_1`` style references.
        - Normalize subject references in loading and exclusion paths.
        """

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
        """Map one or more subject references to canonical IDs.

        This helper applies ``map_subject_id`` to optional scalar or sequence inputs,
        removes duplicates while preserving order, and returns a tuple. It is used for
        config exclusions, user exclusions, and download filtering.

        Parameters
        ----------
        values : str, int, sequence, or None
            User subject references.
        subject_ids : tuple of str
            Canonical subject IDs.

        Returns
        -------
        tuple of str Unique canonical subject IDs preserving input order.

        Use Cases
        ---------
        - Normalize config and user exclusions.
        - Accept mixed integer and string subject references.
        """
        if values is None:
            return tuple()
        if isinstance(values, (str, int)):
            return (cls.map_subject_id(values, subject_ids),)
        return tuple(dict.fromkeys(cls.map_subject_id(value, subject_ids) for value in values))

    @staticmethod
    def sort_subject_ids(subject_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        """Sort subject IDs by trailing numeric value.

        Dataset subject IDs often contain prefixes such as ``pp`` or ``P`` followed by
        numbers. This helper keeps natural numeric ordering stable, so ``pp2`` appears
        before ``pp10`` in splits, summaries, and path numbering.

        Parameters
        ----------
        subject_ids : set, list, or tuple
            Subject IDs to sort.

        Returns
        -------
        list of str Naturally sorted subject IDs.

        Use Cases
        ---------
        - Keep pp1 before pp10.
        - Build deterministic subject scopes.
        """
        def subject_sort_key(value: str) -> tuple[int, str]:
            match = re.search(r"(\d+)$", str(value))
            if match is None:
                return (0, str(value).lower())
            return (int(match.group(1)), str(value).lower())

        return sorted(subject_ids, key=subject_sort_key)

    @staticmethod
    def build_subject_number_map(subject_ids: tuple[str, ...]) -> dict[str, int]:
        """Build numeric subject identifiers from canonical IDs.

        Some resource path patterns require a subject number in addition to the
        canonical ID. This helper extracts trailing digits when available and falls
        back to one-based ordering for non-numeric IDs.

        Parameters
        ----------
        subject_ids : tuple of str
            Canonical subject IDs.

        Returns
        -------
        dict Mapping from subject ID to numeric identifier.

        Use Cases
        ---------
        - Format resource path patterns requiring subject numbers.
        """
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
        """Prepare the non-excluded subject scope for a dataset.

        This method combines dataset config subjects with normalized exclusions and
        builds the subject-number map used by path formatters. It provides the pre-
        resource-intersection subject universe for later split planning.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset with initialized config and exclusions.

        Returns
        -------
        tuple Available subject IDs and subject number map.

        Use Cases
        ---------
        - Initialize split planning from config subjects.
        - Apply exclusions before resource intersection.
        """
        state = dataset._state
        config = state.config
        if config is None:
            raise ValueError("Dataset config is not initialized")
        excluded_subjects = set(state.excluded_subjects)
        sorted_subjects = tuple(DatasetSplitPlanner.sort_subject_ids(config.subject_ids))
        available_subjects = tuple(
            subject_id
            for subject_id in sorted_subjects
            if subject_id not in excluded_subjects
        )
        subject_numbers = DatasetSplitPlanner.build_subject_number_map(
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
        """Split subject IDs into all, train, validation, or test subsets.

        The splitter shuffles subjects deterministically with the provided seed,
        converts ratios into integer counts, assigns remainders deterministically, and
        returns only the requested subset. It is deliberately independent of resource
        scanning so it can be tested in isolation.

        Parameters
        ----------
        subject_ids : sequence of str
            Ordered subject IDs available for splitting.
        split : {'all', 'train', 'validation', 'test'}
            Split subset to return.
        split_ratio : tuple of float
            Train, validation, and test ratios. Values must sum to one.
        split_seed : int
            Random seed used to shuffle subjects deterministically.

        Returns
        -------
        list of str Subject IDs selected for the requested split.

        Use Cases
        ---------
        - Build reproducible train/validation/test datasets.
        - Test split behavior without constructing a full dataset.
        - Reuse split logic across dataset integrations.
        """

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
    ) -> DatasetSplitPlan:
        """Build a split plan from dataset resources and selected specs.

        This method intersects available subjects across every resource family
        required by the current specs, reports detailed availability context when the
        intersection is empty, and then applies the requested train/validation/test
        split. It is the boundary between resource availability and row-generation
        subject selection.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset with resource and spec state initialized.
        split : str
            Requested split name.
        split_ratio : tuple of float
            Train, validation, and test ratios.
        split_seed : int
            Deterministic split seed.

        Returns
        -------
        DatasetSplitPlan Available and selected subjects for the dataset.

        Use Cases
        ---------
        - Intersect resource availability across selected specs.
        - Build subject splits after resource scanning.
        """
        state = dataset._state
        config = state.config
        if config is None:
            raise ValueError("Dataset config is not initialized")
        resource_subjects, _ = DatasetSplitPlanner.prepare_subject_scope(dataset)
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
            subject_ids = DatasetSplitPlanner.sort_subject_ids(
                list(resource_subjects)
            )
        else:
            subject_ids = DatasetSplitPlanner.sort_subject_ids(
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
                        "mesh_variant",
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

        selected_subject_ids = DatasetSplitPlanner.split_subject_ids(
            subject_ids,
            split,
            split_ratio,
            split_seed,
        )
        if len(selected_subject_ids) == 0:
            raise ValueError(f"Split {split!r} produced an empty dataset")

        return DatasetSplitPlan(
            available_subjects=tuple(subject_ids),
            selected_subjects=tuple(selected_subject_ids),
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
