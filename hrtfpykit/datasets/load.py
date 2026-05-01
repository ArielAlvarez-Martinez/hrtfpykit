from pathlib import Path
from collections.abc import Sequence
import csv
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import loadmat

from .. import hrtf
from .split import DatasetSubjectSelectionPlanner

if TYPE_CHECKING:
    from .base import BaseDataset
    from ..hrtf.hrtf import HRTF


def load_hrtf(
    dataset: "BaseDataset",
    subject_id: str | int,
    subject_ids: tuple[str, ...] | None = None,
) -> "HRTF":
    if subject_ids is None:
        subject_ids = tuple(dataset._subject_ids)
    if dataset._config is None:
        raise ValueError("Dataset config is not initialized")
    mapped_subject_id = DatasetSubjectSelectionPlanner.map_subject_id(
        subject_id,
        subject_ids,
    )
    if mapped_subject_id not in dataset._hrtf_paths:
        raise KeyError(
            f"Subject {subject_id!r} mapped to {mapped_subject_id!r} but does not have an available HRTF file"
        )
    path = dataset._hrtf_paths[mapped_subject_id]
    if not path.exists():
        warnings.warn(
            f"{dataset._name}: subject {mapped_subject_id} HRTF path is missing: {path}",
            stacklevel=2,
        )
        raise FileNotFoundError(
            f"HRTF path is missing for subject {mapped_subject_id}: {path}"
        )
    cache_key = ("hrtf", mapped_subject_id)
    loaded_hrtf = dataset._cache.get(cache_key)
    if loaded_hrtf is None:
        try:
            loaded_hrtf = hrtf.load_hrtf(path)
            if dataset._dataset_hrtf_transform is not None:
                loaded_hrtf = dataset._dataset_hrtf_transform(loaded_hrtf)
                if not (
                    hasattr(loaded_hrtf, "IR")
                    and hasattr(loaded_hrtf, "TF")
                    and hasattr(loaded_hrtf, "Sources")
                    and hasattr(loaded_hrtf, "transform")
                ):
                    raise ValueError("dataset_hrtf_transform must return an HRTF object")
        except Exception as exc:
            warnings.warn(
                f"{dataset._name}: subject {mapped_subject_id} HRTF file could not be loaded: {path} ({exc})",
                stacklevel=2,
            )
            raise
        dataset._cache[cache_key] = loaded_hrtf
    return loaded_hrtf


def load_anthropometry(
    dataset: "BaseDataset",
    path: str | Path,
    extension: str | None = None,
    exclude_row: int | Sequence[int] | None = None,
    exclude_column: int | Sequence[int] | None = None,
    accessed_by: str = "row",
) -> dict[str, dict[str, float | str | None]] | dict[str, object]:
    if dataset._config is None:
        raise ValueError("Dataset config is not initialized")
    if str(accessed_by).strip().lower() not in {"row", "column"}:
        raise ValueError("accessed_by must be 'row' or 'column'")
    dataset_subject_ids = tuple(dataset._config.subject_ids)
    mapped_path = Path(path).expanduser()
    mapped_extension = (
        mapped_path.suffix.lower()
        if extension is None
        else str(extension).strip().lower()
    )
    if not mapped_extension.startswith("."):
        mapped_extension = f".{mapped_extension}"
    if mapped_extension == ".csv":
        if dataset._config.anthropometry is not None:
            subject_column_candidates = tuple(dataset._config.anthropometry.subject_column_candidates)
        else:
            subject_column_candidates = (
                "subject_id",
                "subject",
                "id",
                "participant",
                "pp",
            )
        with mapped_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or len(reader.fieldnames) == 0:
                raise ValueError(f"Anthropometry file {mapped_path} does not contain headers")
            fieldnames = {fieldname.lower(): fieldname for fieldname in reader.fieldnames}
            subject_column = reader.fieldnames[0]
            for candidate in subject_column_candidates:
                if candidate.lower() in fieldnames:
                    subject_column = fieldnames[candidate.lower()]
                    break
            parsed_rows: list[tuple[str, dict[str, float | str | None]]] = []
            for row in reader:
                raw_subject_id = row.get(subject_column)
                if raw_subject_id is None or str(raw_subject_id).strip() == "":
                    continue
                try:
                    subject_id = DatasetSubjectSelectionPlanner.map_subject_id(
                        raw_subject_id,
                        dataset_subject_ids,
                    )
                except ValueError:
                    continue
                converted: dict[str, float | str | None] = {}
                for key, value in row.items():
                    if key is None or key == subject_column:
                        continue
                    text = str("" if value is None else value).strip()
                    if text == "":
                        converted[key] = None
                        continue
                    try:
                        converted[key] = float(text)
                    except ValueError:
                        converted[key] = text
                parsed_rows.append((subject_id, converted))
            raw_rows = {subject_id: values for subject_id, values in parsed_rows}
            row_keys = tuple(raw_rows)
            row_indices: list[int] = []
            if exclude_row is not None:
                if isinstance(exclude_row, int):
                    row_indices = [int(exclude_row)]
                else:
                    row_indices = [int(value) for value in exclude_row]
                for index in row_indices:
                    if index < 0 or index >= len(row_keys):
                        raise ValueError(
                            f"Anthropometry row index {index} is out of range for {len(row_keys)} rows"
                        )
            selected_row_keys = tuple(
                key
                for index, key in enumerate(row_keys)
                if index not in set(row_indices)
            )
            column_indices: list[int] = []
            if exclude_column is not None:
                if isinstance(exclude_column, int):
                    column_indices = [int(exclude_column)]
                else:
                    column_indices = [int(value) for value in exclude_column]
            rows: dict[str, dict[str, float | str | None]] = {}
            for row_key in selected_row_keys:
                row_values = raw_rows[row_key]
                column_keys = tuple(row_values)
                if len(set(column_indices)) != len(column_indices):
                    column_indices = list(dict.fromkeys(column_indices))
                for index in column_indices:
                    if index < 0 or index >= len(column_keys):
                        raise ValueError(
                            f"Anthropometry column index {index} is out of range for {len(column_keys)} columns"
                        )
                selected_columns = tuple(
                    key
                    for index, key in enumerate(column_keys)
                    if index not in set(column_indices)
                )
                selected_rows: dict[str, float | str | None] = {}
                for key in selected_columns:
                    selected_rows[key] = row_values[key]
                rows[row_key] = selected_rows
        return rows
    if mapped_extension == ".mat":
        data = loadmat(mapped_path)
        if not isinstance(exclude_row, type(None)):
            row_indices = (
                (int(exclude_row),)
                if isinstance(exclude_row, int)
                else tuple(int(value) for value in exclude_row)
            )
            data = {
                key: (
                    np.delete(value, row_indices, axis=0)
                    if isinstance(value, np.ndarray) and value.ndim >= 1
                    else value
                )
                for key, value in data.items()
            }
        if not isinstance(exclude_column, type(None)):
            column_indices = (
                (int(exclude_column),)
                if isinstance(exclude_column, int)
                else tuple(int(value) for value in exclude_column)
            )
            data = {
                key: (
                    np.delete(value, column_indices, axis=1)
                    if isinstance(value, np.ndarray) and value.ndim >= 2
                    else value
                )
                for key, value in data.items()
            }
        return data
    raise ValueError(
        "Anthropometry extension must be one of: .csv, .mat"
    )
