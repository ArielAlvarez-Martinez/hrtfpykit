from collections.abc import Callable
from pathlib import Path
import csv

from .specs import normalize_anthropometry_ear, normalize_anthropometry_select


def convert_table_value(value: str) -> float | str | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def load_anthropometry_rows(
    path: Path,
    subject_column_candidates: tuple[str, ...],
    subject_ids: tuple[str, ...],
    resolve_subject_id: Callable[[str, tuple[str, ...]], str],
) -> dict[str, dict[str, float | str | None]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or len(reader.fieldnames) == 0:
            raise ValueError(f"Anthropometry file {path} does not contain headers")
        fieldnames = {fieldname.lower(): fieldname for fieldname in reader.fieldnames}
        subject_column = reader.fieldnames[0]
        for candidate in subject_column_candidates:
            if candidate.lower() in fieldnames:
                subject_column = fieldnames[candidate.lower()]
                break
        rows: dict[str, dict[str, float | str | None]] = {}
        for row in reader:
            raw_subject_id = row.get(subject_column)
            if raw_subject_id is None or str(raw_subject_id).strip() == "":
                continue
            try:
                subject_id = resolve_subject_id(raw_subject_id, subject_ids)
            except ValueError:
                continue
            converted: dict[str, float | str | None] = {}
            for key, value in row.items():
                if key is None:
                    continue
                if key == subject_column:
                    continue
                converted[key] = convert_table_value("" if value is None else value)
            rows[subject_id] = converted
    return rows


def select_anthropometry_value(
    values: dict[str, float | str | None],
    select: str | tuple[str, ...] | list[str] | None,
    ear: str,
    dataset_name: str,
) -> dict[str, float | str | None]:
    selected = normalize_anthropometry_select(select)
    normalized_ear = normalize_anthropometry_ear(ear)

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
