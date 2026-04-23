from collections.abc import Sequence

import numpy as np


def normalize_index_by(index_by: str | Sequence[str]) -> tuple[str, ...]:
    allowed = {
        ("subject",),
        ("subject", "position"),
        ("subject", "ear"),
        ("subject", "position", "ear"),
    }
    if isinstance(index_by, str):
        value = str(index_by).strip().lower()
        if value in {"subject", "subject-position", "subject-ear", "subject-position-ear"}:
            normalized = tuple(value.split("-"))
        else:
            normalized = (value,)
        if normalized not in allowed:
            raise ValueError(
                "index_by must be one of: "
                "'subject', 'subject-position', 'subject-ear', 'subject-position-ear', "
                "('subject',), ('subject', 'position'), ('subject', 'ear'), "
                "('subject', 'position', 'ear')"
            )
        return normalized
    values = tuple(str(value).strip().lower() for value in index_by)
    if values not in allowed:
        raise ValueError(
            "index_by must be one of: "
            "('subject',), ('subject', 'position'), ('subject', 'ear'), "
            "('subject', 'position', 'ear')"
        )
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
) -> list[dict[str, str | int | None]]:
    rows: list[dict[str, str | int | None]] = []
    include_position = "position" in index_by
    include_ear = "ear" in index_by
    for subject_id in subject_ids:
        if include_position and include_ear:
            for selected_position_index, position_index in enumerate(position_indices):
                for selected_ear_index, (ear_name, ear_index) in enumerate(ears):
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "position_index": int(position_index),
                            "selected_position_index": int(selected_position_index),
                            "ear": ear_name,
                            "ear_index": int(ear_index),
                            "selected_ear_index": int(selected_ear_index),
                        }
                    )
            continue
        if include_position:
            for selected_position_index, position_index in enumerate(position_indices):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "position_index": int(position_index),
                        "selected_position_index": int(selected_position_index),
                        "ear": None,
                        "ear_index": None,
                        "selected_ear_index": None,
                    }
                )
            continue
        if include_ear:
            for selected_ear_index, (ear_name, ear_index) in enumerate(ears):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "position_index": None,
                        "selected_position_index": None,
                        "ear": ear_name,
                        "ear_index": int(ear_index),
                        "selected_ear_index": int(selected_ear_index),
                    }
                )
            continue
        rows.append(
            {
                "subject_id": subject_id,
                "position_index": None,
                "selected_position_index": None,
                "ear": None,
                "ear_index": None,
                "selected_ear_index": None,
            }
        )
    return rows
