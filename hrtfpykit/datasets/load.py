from pathlib import Path
from collections.abc import Sequence
import csv
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import loadmat

from .. import hrtf
from .split import DatasetSubjectSplitPlanner

if TYPE_CHECKING:
    from .base import BaseDataset
    from ..hrtf.hrtf import HRTF


def load_hrtf(
    dataset: "BaseDataset",
    subject_id: str | int,
    subject_ids: tuple[str, ...] | None = None,
    hrtf_paths: dict[str, Path] | None = None,
    cache: dict[object, object] | None = None,
) -> "HRTF":
    state = dataset._state
    if subject_ids is None:
        subject_ids = tuple(state.available_subjects)
    if state.config is None:
        raise ValueError("Dataset config is not initialized")
    mapped_subject_id = DatasetSubjectSplitPlanner.map_subject_id(
        subject_id,
        subject_ids,
    )
    selected_hrtf_paths = state.hrtf_paths if hrtf_paths is None else hrtf_paths
    selected_cache = state.cache if cache is None else cache
    if mapped_subject_id not in selected_hrtf_paths:
        raise KeyError(
            f"Subject {subject_id!r} mapped to {mapped_subject_id!r} but does not have an available HRTF file"
        )
    path = selected_hrtf_paths[mapped_subject_id]
    if not path.exists():
        warnings.warn(
            f"{state.name}: subject {mapped_subject_id} HRTF path is missing: {path}",
            stacklevel=2,
        )
        raise FileNotFoundError(
            f"HRTF path is missing for subject {mapped_subject_id}: {path}"
        )
    cache_key = ("hrtf", mapped_subject_id)
    loaded_hrtf = selected_cache.get(cache_key)
    if loaded_hrtf is None:
        try:
            loaded_hrtf = hrtf.load_hrtf(path)
            if state.dataset_hrtf_transform is not None:
                loaded_hrtf = state.dataset_hrtf_transform(loaded_hrtf)
                if not (
                    hasattr(loaded_hrtf, "IR")
                    and hasattr(loaded_hrtf, "TF")
                    and hasattr(loaded_hrtf, "Sources")
                    and hasattr(loaded_hrtf, "transform")
                ):
                    raise ValueError("dataset_hrtf_transform must return an HRTF object")
        except Exception as exc:
            warnings.warn(
                f"{state.name}: subject {mapped_subject_id} HRTF file could not be loaded: {path} ({exc})",
                stacklevel=2,
            )
            raise
        selected_cache[cache_key] = loaded_hrtf
    return loaded_hrtf


def load_anthropometry(
    dataset: "BaseDataset",
    path: str | Path,
    extension: str | None = None,
    exclude_row: int | Sequence[int] | None = None,
    exclude_column: int | Sequence[int] | None = None,
    accessed_by: str = "row",
) -> dict[str, dict[str, float | str | None]] | dict[str, object]:
    state = dataset._state
    if state.config is None:
        raise ValueError("Dataset config is not initialized")
    accessed_by = str(accessed_by).strip().lower()
    if accessed_by not in {"row", "column"}:
        raise ValueError("accessed_by must be 'row' or 'column'")
    dataset_subject_ids = tuple(state.config.subject_ids)
    mapped_path = Path(path).expanduser()
    mapped_extension = (
        mapped_path.suffix.lower()
        if extension is None
        else str(extension).strip().lower()
    )
    if not mapped_extension.startswith("."):
        mapped_extension = f".{mapped_extension}"
    if mapped_extension == ".csv":
        with mapped_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or len(reader.fieldnames) == 0:
                raise ValueError(f"Anthropometry file {mapped_path} does not contain headers")
            fieldnames = tuple(reader.fieldnames)
            if accessed_by == "row":
                subject_column = fieldnames[0]
                raw_rows: list[tuple[str, dict[str, float | str | None]]] = []
                for row in reader:
                    raw_subject_id = row.get(subject_column)
                    if raw_subject_id is None or str(raw_subject_id).strip() == "":
                        continue
                    try:
                        subject_id = DatasetSubjectSplitPlanner.map_subject_id(
                            raw_subject_id,
                            dataset_subject_ids,
                        )
                    except ValueError:
                        continue
                    converted: dict[str, float | str | None] = {}
                    for key in fieldnames:
                        if key is None or key == subject_column:
                            continue
                        value = row.get(key)
                        text = str("" if value is None else value).strip()
                        if text == "":
                            converted[str(key)] = None
                            continue
                        try:
                            converted[str(key)] = float(text)
                        except ValueError:
                            converted[str(key)] = text
                    raw_rows.append((subject_id, converted))

                row_keys = tuple(subject_id for subject_id, _ in raw_rows)
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
                selected_row_keys = set(row_indices)
                rows: dict[str, dict[str, float | str | None]] = {}
                for index, subject_id in enumerate(row_keys):
                    if index in selected_row_keys:
                        continue
                    row_values = dict(raw_rows[index][1])
                    column_keys = tuple(row_values)
                    column_indices: list[int] = []
                    if exclude_column is not None:
                        if isinstance(exclude_column, int):
                            column_indices = [int(exclude_column)]
                        else:
                            column_indices = [int(value) for value in exclude_column]
                        for column_index in column_indices:
                            if column_index < 0 or column_index >= len(column_keys):
                                raise ValueError(
                                    f"Anthropometry column index {column_index} is out of range for {len(column_keys)} columns"
                                )
                    selected_column_keys = {
                        key
                        for index, key in enumerate(column_keys)
                        if index not in set(column_indices)
                    }
                    rows[subject_id] = {
                        key: row_values[key]
                        for key in row_values
                        if key in selected_column_keys
                    }
                return rows

            subject_columns = tuple(fieldnames[1:]) if len(fieldnames) > 1 else tuple()
            if len(subject_columns) == 0:
                raise ValueError(f"Anthropometry file {mapped_path} has no columns for subject IDs")
            recognized_subject_columns: list[tuple[str, str]] = []
            for subject_column in subject_columns:
                try:
                    mapped_subject_id = DatasetSubjectSplitPlanner.map_subject_id(
                        subject_column,
                        dataset_subject_ids,
                    )
                    recognized_subject_columns.append((subject_column, mapped_subject_id))
                except ValueError:
                    continue
            if len(recognized_subject_columns) == 0:
                raise ValueError(f"Anthropometry file {mapped_path} has no recognized subject columns")

            rows: dict[str, dict[str, float | str | None]] = {}
            raw_rows: list[tuple[str, dict[str, float | str | None]]] = []
            for row in reader:
                row_key = row.get(fieldnames[0], "")
                row_label = str("" if row_key is None else row_key).strip()
                if row_label == "":
                    continue
                row_values: dict[str, float | str | None] = {}
                for source_column, mapped_subject_id in recognized_subject_columns:
                    value = row.get(source_column)
                    text = str("" if value is None else value).strip()
                    if text == "":
                        row_values[mapped_subject_id] = None
                    else:
                        try:
                            row_values[mapped_subject_id] = float(text)
                        except ValueError:
                            row_values[mapped_subject_id] = text
                raw_rows.append((row_label, row_values))

            row_indices: list[int] = []
            if exclude_row is not None:
                if isinstance(exclude_row, int):
                    row_indices = [int(exclude_row)]
                else:
                    row_indices = [int(value) for value in exclude_row]
                for index in row_indices:
                    if index < 0 or index >= len(raw_rows):
                        raise ValueError(
                            f"Anthropometry row index {index} is out of range for {len(raw_rows)} rows"
                        )
            column_indices: list[int] = []
            if exclude_column is not None:
                if isinstance(exclude_column, int):
                    column_indices = [int(exclude_column)]
                else:
                    column_indices = [int(value) for value in exclude_column]
                for index in column_indices:
                    if index < 0 or index >= len(recognized_subject_columns):
                        raise ValueError(
                            f"Anthropometry column index {index} is out of range for {len(recognized_subject_columns)} columns"
                        )
            selected_row_indices = set(row_indices)
            removed_subject_positions = set(column_indices)
            kept_subject_columns = [
                (source_column, mapped_subject_id)
                for position, (source_column, mapped_subject_id) in enumerate(recognized_subject_columns)
                if position not in removed_subject_positions
            ]
            kept_subject_ids = tuple(subject_id for _, subject_id in kept_subject_columns)
            for index, (row_label, row_values) in enumerate(raw_rows):
                if index in selected_row_indices:
                    continue
                filtered_row: dict[str, float | str | None] = {}
                for mapped_subject_id in tuple(row_values):
                    if mapped_subject_id in kept_subject_ids:
                        filtered_row[mapped_subject_id] = row_values[mapped_subject_id]
                rows[row_label] = filtered_row
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
