from collections.abc import Sequence

import numpy as np


def normalize_index_by(index_by: str | Sequence[str]) -> tuple[str, ...]:
    allowed_axes = {"position", "ear", "frequency", "samples"}
    if isinstance(index_by, str):
        value = str(index_by).strip().lower()
        if value == "subject":
            normalized = ("subject",)
        elif value.startswith("subject-"):
            normalized = tuple(part for part in value.split("-") if part != "")
        else:
            normalized = (value,)
    else:
        normalized = tuple(str(value).strip().lower() for value in index_by)
    values = normalized
    if len(values) == 0:
        raise ValueError("index_by must not be empty")
    if values[0] != "subject":
        raise ValueError("index_by must start with 'subject'")
    if len(set(values)) != len(values):
        raise ValueError("index_by must not contain duplicate axes")
    invalid_axes = [value for value in values[1:] if value not in allowed_axes]
    if invalid_axes:
        raise ValueError(
            "index_by axes after 'subject' must be chosen from: "
            "'position', 'ear', 'frequency', 'samples'"
        )
    if "frequency" in values and "samples" in values:
        raise ValueError("index_by cannot include both 'frequency' and 'samples'")
    return values


def normalize_ears(ears: str | Sequence[str]) -> list[tuple[str, int]]:
    if isinstance(ears, str):
        value = str(ears).strip().lower()
        if value == "both":
            return [("left", 0), ("right", 1)]
        if value == "left":
            return [("left", 0)]
        if value == "right":
            return [("right", 1)]
        raise ValueError("ears must be 'both', 'left', 'right', or a sequence")
    values = [str(value).strip().lower() for value in ears]
    if len(values) == 0:
        raise ValueError("ears must not be empty")
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    mapping = {"left": 0, "right": 1}
    for value in values:
        if value not in mapping:
            raise ValueError("ears sequence only supports 'left' and 'right'")
        if value in seen:
            raise ValueError("ears must not contain duplicates")
        seen.add(value)
        result.append((value, mapping[value]))
    return result


def normalize_positions(
    positions: str | Sequence[int] | np.ndarray,
    position_count: int,
) -> list[int]:
    if isinstance(positions, str):
        value = str(positions).strip().lower()
        if value != "all":
            raise ValueError("positions must be 'all' or a sequence of position indices")
        return list(range(position_count))
    values = np.asarray(positions, dtype=int).reshape(-1)
    if values.size == 0:
        raise ValueError("positions must not be empty")
    result = [int(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError("positions must not contain duplicates")
    for value in result:
        if value < 0 or value >= position_count:
            raise ValueError(
                f"Position index {value} is out of range for {position_count} positions"
            )
    return result


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


def build_rows(
    subject_ids: Sequence[str],
    index_by: tuple[str, ...],
    position_indices: Sequence[int],
    ears: Sequence[tuple[str, int]],
    frequency_indices: Sequence[int],
    sample_indices: Sequence[int],
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
            else [(int(position_index), int(selected_position_index)) for selected_position_index, position_index in enumerate(position_indices)]
        )
        ear_values = (
            [(None, None, None)]
            if not include_ear
            else [
                (ear_name, int(ear_index), int(selected_ear_index))
                for selected_ear_index, (ear_name, ear_index) in enumerate(ears)
            ]
        )
        frequency_values = (
            [(None, None)]
            if not include_frequency
            else [(int(frequency_index), int(selected_frequency_index)) for selected_frequency_index, frequency_index in enumerate(frequency_indices)]
        )
        sample_values = (
            [(None, None)]
            if not include_samples
            else [(int(sample_index), int(selected_sample_index)) for selected_sample_index, sample_index in enumerate(sample_indices)]
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
