from pathlib import Path
from collections.abc import Sequence
import csv
import io
import warnings
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import loadmat

from .. import hrtf
from .split import DatasetSplitPlanner

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
    """Load one subject HRTF through an initialized dataset context.

    This function is the low-level implementation behind
    :meth:`~hrtfpykit.datasets.base.BaseDataset.get_subject_hrtf` and acoustic
    value extraction in
    :meth:`~hrtfpykit.datasets.base.BaseDataset.__getitem__`. It maps a user
    subject reference through
    :class:`~hrtfpykit.datasets.split.DatasetSplitPlanner`, resolves the selected
    HRTF path, reuses the dataset cache, loads the SOFA-based object through
    :func:`~hrtfpykit.hrtf.load_hrtf`, applies the dataset-level HRTF transform,
    and wraps loader failures with dataset, subject, and path context.

    The optional subject_ids, hrtf_paths, and cache arguments are used
    by the resource validator while dataset construction is still in progress.
    Normal user-facing calls should go through
    :meth:`~hrtfpykit.datasets.base.BaseDataset.get_subject_hrtf`, which forwards
    the active dataset state automatically.

    Parameters
    ----------
    dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
        Dataset instance whose internal
        :class:`~hrtfpykit.datasets.state.DatasetState` contains configuration,
        resource paths, transform settings, verbosity, and cache objects.
    subject_id : str or int
        Subject reference to map and load. Strings can be canonical dataset
        subject IDs or supported subject aliases; integers are one-based subject
        positions in subject_ids.
    subject_ids : tuple of str or None, default=None
        Optional subject scope used for subject mapping. None uses the
        dataset state's available subjects.
    hrtf_paths : dict[str, Path] or None, default=None
        Optional subject-to-path map. None uses the HRTF paths stored in the
        dataset state.
    cache : dict or None, default=None
        Optional cache dictionary. None uses the dataset state cache. Loaded
        objects are stored under a subject-level HRTF cache key.

    Returns
    -------
    HRTF
        Loaded :class:`~hrtfpykit.hrtf.hrtf.HRTF` object after applying any
        dataset-level HRTF transform.

    Raises
    ------
    ValueError
        If the dataset configuration is not initialized, subject mapping fails,
        the HRTF file cannot be loaded, or the dataset-level transform does not
        return an object compatible with :class:`~hrtfpykit.hrtf.hrtf.HRTF`.
    KeyError
        If the mapped subject does not have an available HRTF path.
    FileNotFoundError
        If the mapped subject has a path entry but the file is missing.

    Notes
    -----
    When dataset._state.verbose is false, stdout emitted by
    :func:`~hrtfpykit.hrtf.load_hrtf` is captured so dataset indexing stays quiet.
    Missing files emit a warning before :class:`FileNotFoundError` is raised.

    """

    state = dataset._state
    if subject_ids is None:
        subject_ids = tuple(state.available_subjects)
    if state.config is None:
        raise ValueError("Dataset config is not initialized")
    mapped_subject_id = DatasetSplitPlanner.map_subject_id(
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
            if state.verbose:
                loaded_hrtf = hrtf.load_hrtf(path)
            else:
                with redirect_stdout(io.StringIO()):
                    loaded_hrtf = hrtf.load_hrtf(path)
            if state.dataset_hrtf_transform is not None:
                loaded_hrtf = state.dataset_hrtf_transform(loaded_hrtf)
                if not (
                    hasattr(loaded_hrtf, "IR")
                    and hasattr(loaded_hrtf, "TF")
                    and hasattr(loaded_hrtf, "Sources")
                    and hasattr(loaded_hrtf, "transform")
                ):
                    raise ValueError("dataset_hrtf_transform must return an :class:`~hrtfpykit.hrtf.hrtf.HRTF` object")
        except Exception as exc:
            raise ValueError(
                f"{state.name}: subject {mapped_subject_id} HRTF file could not be loaded: {path} ({exc})"
            ) from exc
        selected_cache[cache_key] = loaded_hrtf
    return loaded_hrtf


def load_table(
    dataset: "BaseDataset",
    path: str | Path,
    extension: str | None = None,
    exclude_row: int | Sequence[int] | None = None,
    exclude_column: int | Sequence[int] | None = None,
    accessed_by: str = "row",
    subject_id: bool = True,
    resource_name: str = "Table",
) -> dict[str, dict[str, float | str | None]] | dict[str, object]:
    """Load a CSV or MAT table for dataset metadata-style resources.

    This function implements the shared table behavior used by metadata and
    anthropometry resources. It maps row- or column-oriented tables onto canonical
    dataset subject IDs, applies row and column exclusions, converts simple CSV
    cell values to floats when possible, preserves non-numeric text, stores empty
    CSV cells as None, and keeps MAT variables available for matrix-style
    access.

    CSV files are interpreted in one of two orientations. With
    accessed_by="row", each data row is assigned to the corresponding subject
    in :attr:`~hrtfpykit.datasets.config.DatasetConfig.subject_ids`, and the
    returned mapping is keyed by subject ID. With accessed_by="column", each
    subject column is assigned by position to the dataset subject order, and the
    returned mapping is keyed by row label. MAT files are returned as the variable
    dictionary produced by SciPy after optional row and column deletion.

    Parameters
    ----------
    dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
        Dataset instance whose configuration provides canonical subject IDs.
    path : str or Path
        CSV or MAT table path to load. User home markers are expanded before
        reading.
    extension : str or None, default=None
        Explicit extension override. None uses the suffix from path.
        Values may be passed with or without the leading dot.
    exclude_row, exclude_column : int, sequence of int, or None, default=None
        Zero-based row or column positions removed from the loaded table. For
        CSV data, indices are validated against the parsed orientation. For MAT
        data, indices are forwarded to NumPy deletion on compatible arrays.
    accessed_by : {'row', 'column'}, default='row'
        Whether CSV subjects are represented by table rows or table columns. This
        argument does not change MAT loading.
    subject_id : bool, default=True
        Whether a CSV table includes a leading identifier column in row-oriented
        mode, or a leading label column in column-oriented mode. When false, all
        CSV columns are treated as data or subject columns.
    resource_name : str, default='Table'
        Resource label used in validation errors.

    Returns
    -------
    dict
        For row-oriented CSV files, a mapping from subject ID to field values. For
        column-oriented CSV files, a mapping from row label to per-subject values.
        For MAT files, a variable dictionary containing the loaded arrays and
        metadata entries returned by SciPy.

    Raises
    ------
    ValueError
        If the dataset configuration is not initialized, accessed_by is not
        "row" or "column", a CSV file lacks headers, a requested exclusion
        index is out of range for CSV data, or the extension is unsupported.
    OSError
        If the file cannot be opened or read.

    Notes
    -----
    CSV subject identifiers are aligned by table order, not by matching header or
    row text against subject IDs. Dataset configs therefore need to declare the
    same subject order used by the table source.

    """

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
                raise ValueError(f"{resource_name} file {mapped_path} does not contain headers")
            fieldnames = tuple(reader.fieldnames)
            if accessed_by == "row":
                raw_rows: list[tuple[str, dict[str, float | str | None]]] = []
                data_fieldnames = fieldnames[1:] if subject_id else fieldnames
                for row_index, row in enumerate(reader):
                    if row_index >= len(dataset_subject_ids):
                        break
                    mapped_subject_id = dataset_subject_ids[row_index]
                    converted: dict[str, float | str | None] = {}
                    for key in data_fieldnames:
                        if key is None:
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
                    raw_rows.append((mapped_subject_id, converted))

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
                                f"{resource_name} row index {index} is out of range for {len(row_keys)} rows"
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
                                    f"{resource_name} column index {column_index} is out of range for {len(column_keys)} columns"
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

            subject_columns = tuple(fieldnames[1:]) if subject_id else fieldnames
            if len(subject_columns) == 0:
                raise ValueError(f"{resource_name} file {mapped_path} has no columns for subject IDs")
            recognized_subject_columns: list[tuple[str, str]] = [
                (subject_column, dataset_subject_ids[index])
                for index, subject_column in enumerate(subject_columns[:len(dataset_subject_ids)])
            ]
            if len(recognized_subject_columns) == 0:
                raise ValueError(f"{resource_name} file {mapped_path} has no recognized subject columns")

            rows: dict[str, dict[str, float | str | None]] = {}
            raw_rows: list[tuple[str, dict[str, float | str | None]]] = []
            for row in reader:
                if subject_id:
                    row_key = row.get(fieldnames[0], "")
                    row_label = str("" if row_key is None else row_key).strip()
                    if row_label == "":
                        continue
                else:
                    row_label = str(len(raw_rows))
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
                            f"{resource_name} row index {index} is out of range for {len(raw_rows)} rows"
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
                            f"{resource_name} column index {index} is out of range for {len(recognized_subject_columns)} columns"
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
        f"{resource_name} extension must be one of: .csv, .mat"
    )
