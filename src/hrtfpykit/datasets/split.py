from dataclasses import dataclass
from collections.abc import Sequence
import re

import numpy as np
from typing import TYPE_CHECKING, Any, cast

from .specs_registry import has_specs
from .sanitize import sanitize_subject_id

if TYPE_CHECKING:
    from .base import BaseDataset


@dataclass(frozen=True)
class DatasetSplitPlan:
    """Immutable result produced by dataset subject split planning.

    A split plan records the canonical subject IDs that remain after configured
    exclusions and resource intersection, then stores the subset selected for the
    requested split. :class:`~hrtfpykit.datasets.build.DatasetBuilder` copies the
    plan into dataset state after local resources have been scanned and before
    sample rows are generated.

    Notes
    -----
    The object contains subject IDs only. It does not store resource paths, loaded
    table rows, acoustic axes, or sample rows. Those values remain in the dataset
    state built around the plan.

    Attributes
    ----------
    available_subjects : tuple of str
        Canonical subject IDs that satisfy the active exclusions and required
        resource availability.
    selected_subjects : tuple of str
        Canonical subject IDs selected for the requested split.
    split : str
        Requested split name, usually ``all``, ``train``, ``validation``, or ``test``.
    split_ratio : tuple of float
        Train, validation, and test ratios used when selecting a split other than
        ``all``.
    split_seed : int
        Seed used for deterministic split assignment.

    """

    available_subjects: tuple[str, ...]
    selected_subjects: tuple[str, ...]
    split: str
    split_ratio: tuple[float, float, float]
    split_seed: int


class DatasetSplitPlanner:
    """Coordinate subject normalization and deterministic split selection.

    The planner is a stateless collection of helpers used by dataset construction,
    resource discovery, download planning, and table value resolution. It maps user
    subject references to canonical dataset IDs, preserves natural subject ordering,
    derives numeric subject identifiers for path templates, and builds the final
    :class:`~hrtfpykit.datasets.split.DatasetSplitPlan` consumed by
    :class:`~hrtfpykit.datasets.build.DatasetBuilder`.

    Notes
    -----
    All methods are class or static methods because splitting is derived from
    dataset configuration and scanned resource state. The planner does not cache
    paths or mutate dataset rows; callers are responsible for storing the returned
    values in their own state.

    """

    @classmethod
    def map_subject_id(
        cls,
        value: str | int,
        subject_ids: tuple[str, ...],
    ) -> str:
        """Map one user subject reference to a canonical subject ID.

        The mapper accepts case-insensitive exact IDs, one-based integer
        positions, numeric strings, ``subject1``, ``subject_1``, ``subject-1``, and
        strings that match after
        :func:`~hrtfpykit.datasets.sanitize.sanitize_subject_id` normalization.
        It always returns the canonical ID from the dataset configuration, so
        downstream state does not mix aliases with official subject IDs.

        Parameters
        ----------
        value : str or int
            User subject reference to resolve. Integer and numeric string values
            are interpreted as one-based subject positions.
        subject_ids : tuple of str
            Canonical subject IDs accepted by the dataset, in dataset order.

        Returns
        -------
        str
            Canonical subject ID.

        Raises
        ------
        ValueError
            If no subject IDs are available, the reference is empty, a positional
            reference is outside the valid one-based range, or the reference cannot
            be resolved to a canonical ID.

        Notes
        -----
        Positional references are one-based because they are intended for user-facing
        dataset configuration and download filters, not zero-based Python indexing.

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

        This method applies
        :meth:`~hrtfpykit.datasets.split.DatasetSplitPlanner.map_subject_id` to an
        optional scalar or sequence input, removes duplicate canonical IDs while
        preserving first occurrence order, and returns a tuple. It is used for
        configured exclusions, user exclusions, and download subject filters.

        Parameters
        ----------
        values : str, int, sequence, or None
            User subject reference, sequence of references, or None. None means no
            requested subjects.
        subject_ids : tuple of str
            Canonical subject IDs accepted by the dataset, in dataset order.

        Returns
        -------
        tuple of str
            Unique canonical subject IDs preserving input order.

        Raises
        ------
        ValueError
            If any supplied reference cannot be resolved by
            :meth:`~hrtfpykit.datasets.split.DatasetSplitPlanner.map_subject_id`.

        """
        if values is None:
            return tuple()
        if isinstance(values, (str, int)):
            return (cls.map_subject_id(values, subject_ids),)
        return tuple(dict.fromkeys(cls.map_subject_id(value, subject_ids) for value in values))

    @staticmethod
    def sort_subject_ids(subject_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        """Sort subject IDs by trailing numeric value.

        Dataset subject IDs often combine a dataset-specific prefix with a numeric
        suffix. This method sorts by that trailing integer when it is present, so
        ``pp2`` appears before ``pp10`` in splits, resource summaries, and subject
        number maps. IDs without trailing digits are ordered before numeric IDs and
        sorted case-insensitively by their full text.

        Parameters
        ----------
        subject_ids : set, list, or tuple
            Subject IDs to sort.

        Returns
        -------
        list of str
            Naturally sorted subject IDs.

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

        Some dataset resource patterns need both the canonical subject ID and a
        numeric subject token. This method extracts trailing digits when a subject ID
        has them, and falls back to the one-based position in the supplied sequence
        when the ID has no numeric suffix.

        Parameters
        ----------
        subject_ids : tuple of str
            Canonical subject IDs in the order that should define fallback numbers.

        Returns
        -------
        dict of str to int
            Mapping from subject ID to numeric identifier.

        Notes
        -----
        The returned map is used by resource path formatting and download planning.
        It intentionally keeps canonical IDs as keys so callers can resolve paths
        without re-normalizing subject labels.

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

        This method combines the configured subject IDs or user-requested subject
        scope with normalized exclusions and builds the subject-number map used by
        path formatters. The returned subject tuple is the pre-resource-intersection
        universe used by later split planning, while the number map is based on the
        full sorted configuration so path numbering remains stable after subject
        selection and exclusions.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset with initialized configuration and exclusion state.

        Returns
        -------
        available_subjects : tuple of str
            Canonical subject IDs that remain after exclusions, sorted in natural
            subject order.
        subject_numbers : dict of str to int
            Numeric identifier for each configured subject ID.

        Raises
        ------
        ValueError
            If dataset configuration has not been initialized.

        """
        state = dataset._state
        config = state.config
        if config is None:
            raise ValueError("Dataset config is not initialized")
        excluded_subjects = set(state.excluded_subjects)
        sorted_subjects = tuple(DatasetSplitPlanner.sort_subject_ids(config.subject_ids))
        scoped_subjects = (
            sorted_subjects
            if state.requested_subjects is None
            else tuple(DatasetSplitPlanner.sort_subject_ids(state.requested_subjects))
        )
        available_subjects = tuple(
            subject_id
            for subject_id in scoped_subjects
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

        For ``all``, this method returns the input subject IDs as a list without
        shuffling or validating ratios. For ``train``, ``validation``, and ``test``, it
        shuffles the subject IDs with a deterministic NumPy generator, converts the
        three ratios into integer counts, distributes any remainder to the largest
        fractional counts, and returns only the requested subset.

        Parameters
        ----------
        subject_ids : sequence of str
            Ordered canonical subject IDs available for splitting.
        split : {``all``, ``train``, ``validation``, ``test``}
            Split subset to return.
        split_ratio : tuple of float
            Train, validation, and test ratios. For split values other than ``all``,
            the tuple must contain three values and their sum must be close to one.
        split_seed : int
            Random seed used to shuffle subjects deterministically.

        Returns
        -------
        list of str
            Subject IDs selected for the requested split.

        Raises
        ------
        ValueError
            If split is not one of the supported values, split_ratio does not contain
            three values, or the ratio values do not sum to one for a split other
            than ``all``.

        Notes
        -----
        This method is deliberately independent of resource scanning. Callers pass
        the already available subject IDs, which makes the split operation reusable
        for dataset construction and simple unit checks.

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

        This method inspects the dataset specs to determine which resource families
        are required, intersects the subjects available across those resources, and
        applies the requested split to the resulting subject set. It is the boundary
        between resource availability and row-generation subject selection in
        :class:`~hrtfpykit.datasets.build.DatasetBuilder`.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset with configuration, specs, resource paths, loaded table rows, and
            media indexes already initialized.
        split : str
            Requested split name, usually ``all``, ``train``, ``validation``, or ``test``.
        split_ratio : tuple of float
            Train, validation, and test ratios passed to
            :meth:`~hrtfpykit.datasets.split.DatasetSplitPlanner.split_subject_ids`.
        split_seed : int
            Deterministic split seed.

        Returns
        -------
        :class:`~hrtfpykit.datasets.split.DatasetSplitPlan`
            Available and selected subjects for the dataset.

        Raises
        ------
        ValueError
            If dataset configuration is not initialized, required resource families
            have no common subjects, split validation fails, or the requested split
            produces an empty selected-subject set.

        Notes
        -----
        A resource family participates in the intersection only when the current
        input or target specs require that family. If no active spec requires a
        scanned resource, the split is applied to the configured subject scope after
        exclusions.

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
                    summary = cast(dict[str, Any], summary)
                    if "subjects_checked" in summary:
                        resource_lines.append(f"  {resource_name}:")
                        subject_parts = [f"checked={summary['subjects_checked']!r}"]
                        if "subjects_available" in summary:
                            subject_parts.append(f"available={summary['subjects_available']!r}")
                        if "subjects_missing" in summary:
                            subject_parts.append(f"missing={summary['subjects_missing']!r}")
                        resource_lines.append(f"    subjects: {', '.join(subject_parts)}")
                        if "files" in summary:
                            resource_lines.append(f"    files: {summary['files']!r}")
                    else:
                        parts = [str(resource_name)]
                        for key in (
                            "checked",
                            "found",
                            "missing",
                        ):
                            if key in summary:
                                parts.append(f"{key}={summary[key]!r}")
                        for key in (
                            "pattern",
                            "path",
                            "hrtf_variant",
                            "mesh_variant",
                            "extensions",
                            "valid",
                            "invalid",
                            "subjects",
                            "rows",
                        ):
                            if key in summary:
                                parts.append(f"{key}={summary[key]!r}")
                        resource_lines.append("  " + parts[0] + ": " + ", ".join(parts[1:]))
                    if "missing_subject_ids" in summary:
                        missing_subject_ids = tuple(summary["missing_subject_ids"])
                        if len(missing_subject_ids) > 0:
                            preview = ", ".join(
                                str(value) for value in missing_subject_ids[:5]
                            )
                            if len(missing_subject_ids) > 5:
                                preview = f"{preview}, ..."
                            resource_lines.append(f"    missing_subject_ids: {preview}")
                    if "invalid_subject_ids" in summary:
                        invalid_subject_ids = tuple(summary["invalid_subject_ids"])
                        if len(invalid_subject_ids) > 0:
                            preview = ", ".join(
                                str(value) for value in invalid_subject_ids[:5]
                            )
                            if len(invalid_subject_ids) > 5:
                                preview = f"{preview}, ..."
                            resource_lines.append(f"    invalid_subject_ids: {preview}")
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
