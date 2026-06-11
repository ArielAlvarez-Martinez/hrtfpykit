from pathlib import Path
from collections.abc import Sequence
import csv
import io
import warnings
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.io import loadmat

from .. import hrtf
from .split import DatasetSplitPlanner

if TYPE_CHECKING:
    from .base import BaseDataset
    from ..hrtf.hrtf import HRTF


def load_dataset_hrtf(
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
        resource paths, transform settings, SOFA loading options, verbosity,
        and cache objects.
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
        Loaded :class:`~hrtfpykit.hrtf.HRTF` object after applying any
        dataset-level HRTF transform.

    Raises
    ------
    ValueError
        If the dataset configuration is not initialized, subject mapping fails,
        the HRTF file cannot be loaded, or the dataset-level transform does not
        return an object compatible with :class:`~hrtfpykit.hrtf.HRTF`.
    KeyError
        If the mapped subject does not have an available HRTF path.
    FileNotFoundError
        If the mapped subject has a path entry but the file is missing.

    Notes
    -----
    Dataset state controls whether SOFA convention checks run and whether
    loaded HRTFs keep their backing SOFA netCDF datasets open. When
    dataset._state.verbose is false, stdout emitted by
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
            f"Subject {subject_id!r} mapped to {mapped_subject_id!r} but does "
            "not have an available HRTF file"
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
                loaded_hrtf = hrtf.load_hrtf(
                    path,
                    check_sofa_against_conventions=state.check_sofa_against_conventions,
                    sofa_open=state.sofa_open,
                )
            else:
                with redirect_stdout(io.StringIO()):
                    loaded_hrtf = hrtf.load_hrtf(
                        path,
                        check_sofa_against_conventions=state.check_sofa_against_conventions,
                        sofa_open=state.sofa_open,
                    )
            if state.dataset_hrtf_transform is not None:
                loaded_hrtf = state.dataset_hrtf_transform(loaded_hrtf)
                if not (
                    hasattr(loaded_hrtf, "IR")
                    and hasattr(loaded_hrtf, "TF")
                    and hasattr(loaded_hrtf, "Sources")
                    and hasattr(loaded_hrtf, "transform")
                ):
                    raise ValueError(
                        "dataset_hrtf_transform must return an "
                        ":class:`~hrtfpykit.hrtf.HRTF` object"
                    )
        except Exception as exc:
            raise ValueError(
                f"{state.name}: subject {mapped_subject_id} HRTF file could not "
                f"be loaded: {path} ({exc})"
            ) from exc
        selected_cache[cache_key] = loaded_hrtf
    return cast("HRTF", loaded_hrtf)


def load_table(
    dataset: "BaseDataset",
    path: str | Path,
    extension: str | None = None,
    exclude_row: int | Sequence[int] | None = None,
    exclude_column: int | Sequence[int] | None = None,
    accessed_by: str = "row",
    subject_id: bool = True,
    resource_name: str = "Table",
) -> dict[str, dict[str, object]]:
    """Load a CSV or MAT table for dataset metadata-style resources.

    This function implements the shared table behavior used by metadata and
    anthropometry resources. CSV and MAT files are normalized to the same internal
    structure: a dictionary keyed by canonical dataset subject ID, where each
    value is a dictionary of table fields for that subject.

    With accessed_by=``row``, subjects are expected along table rows. With
    accessed_by=``column``, subjects are expected along table columns. When
    subject_id is true, subject references are read from the table and mapped to
    canonical dataset subject IDs. Those references can be canonical IDs or
    supported aliases such as ``subject1`` or ``subject_1``. When subject_id is
    false, table subjects are assigned by order: the first row or column maps to
    the first configured dataset subject, the second maps to the second, and so on.
    A shorter table is valid and represents a leading ordered subset.

    Subjects with any missing, empty, NaN, or infinite field value after row and
    column exclusions are omitted from the returned mapping. This keeps requested
    table specs from producing samples with partial metadata or anthropometry.

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
        Zero based row or column positions removed from the loaded table. When
        subjects are stored in rows, rows are subjects and columns are fields.
        When subjects are stored in columns, columns are subjects and rows are
        fields.
    accessed_by : {``row``, ``column``}, default=``row``
        Whether subjects are represented by table rows or table columns.
    subject_id : bool, default=True
        Whether the table provides explicit subject references. For CSV files
        with subjects in rows, this means the first column contains subject IDs.
        For CSV files with subjects in columns, this means subject IDs are read
        from column headers after the first label column. For MAT files, this
        means a variable named ``subject``, ``subjects``, ``subject_id``,
        ``subject_ids``, ``id``, or ``ids`` provides subject references.
    resource_name : str, default=``Table``
        Resource label used in validation errors.

    Returns
    -------
    dict
        Mapping from canonical subject ID to complete table field values.

    Raises
    ------
    ValueError
        If the dataset configuration is not initialized, accessed_by is not
        ``row`` or ``column``, a CSV file lacks headers, explicit subject IDs are
        missing or cannot be mapped, a requested exclusion index is out of range,
        MAT variables cannot be interpreted as one table, or the extension is
        unsupported.
    OSError
        If the file cannot be opened or read.

    Notes
    -----
    When subject_id is false, subject identity is positional. The table can be
    shorter than the dataset subject list, but it cannot skip subjects in the
    middle because there is no ID telling the loader which subject was skipped.

    """

    state = dataset._state
    if state.config is None:
        raise ValueError("Dataset config is not initialized")
    accessed_by = str(accessed_by).strip().lower()
    if accessed_by not in {"row", "column"}:
        raise ValueError("accessed_by must be 'row' or 'column'")
    dataset_subject_ids = tuple(state.config.subject_ids)
    subject_id_examples = ", ".join(str(value) for value in dataset_subject_ids[:5])
    if len(dataset_subject_ids) > 5:
        subject_id_examples = f"{subject_id_examples}, ..."
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
            source_fieldnames = tuple(reader.fieldnames)
            fieldnames = tuple(
                str(fieldname).strip() if str(fieldname).strip() != "" else f"field_{index + 1}"
                for index, fieldname in enumerate(source_fieldnames)
            )
            table_rows = list(reader)
            if accessed_by == "row":
                excluded_rows: set[int] = set()
                if exclude_row is not None:
                    excluded_rows = (
                        {int(exclude_row)}
                        if isinstance(exclude_row, int)
                        else {int(value) for value in exclude_row}
                    )
                    for index in excluded_rows:
                        if index < 0 or index >= len(table_rows):
                            raise ValueError(
                                f"{resource_name} row index {index} is out of range "
                                f"for {len(table_rows)} rows"
                            )
                data_source_fieldnames = source_fieldnames[1:] if subject_id else source_fieldnames
                data_fieldnames = fieldnames[1:] if subject_id else fieldnames
                excluded_columns: set[int] = set()
                if exclude_column is not None:
                    excluded_columns = (
                        {int(exclude_column)}
                        if isinstance(exclude_column, int)
                        else {int(value) for value in exclude_column}
                    )
                    for index in excluded_columns:
                        if index < 0 or index >= len(data_fieldnames):
                            raise ValueError(
                                f"{resource_name} column index {index} is out of range "
                                f"for {len(data_fieldnames)} columns"
                            )
                rows: dict[str, dict[str, object]] = {}
                for row_index, row in enumerate(table_rows):
                    if row_index in excluded_rows:
                        continue
                    if subject_id:
                        subject_value = str(
                            ""
                            if row.get(source_fieldnames[0]) is None
                            else row.get(source_fieldnames[0])
                        ).strip()
                        if subject_value == "":
                            raise ValueError(f"{resource_name} subject ID must not be empty")
                        try:
                            mapped_subject_id = DatasetSplitPlanner.map_subject_id(
                                subject_value,
                                dataset_subject_ids,
                            )
                        except ValueError as exc:
                            raise ValueError(
                                f"{resource_name} subject ID {subject_value!r} does not "
                                f"match {state.name} subject IDs "
                                f"(examples: {subject_id_examples}). "
                                f"Use a {resource_name.lower()} file with {state.name} "
                                "subject IDs, or rename the IDs."
                            ) from exc
                    else:
                        if row_index >= len(dataset_subject_ids):
                            raise ValueError(
                                f"{resource_name} has {len(table_rows)} data rows, but "
                                f"{state.name} has {len(dataset_subject_ids)} subjects. "
                                "With subject_id=False, rows are mapped by dataset "
                                "order; remove extra rows or use subject_id=True with "
                                "matching IDs."
                            )
                        mapped_subject_id = dataset_subject_ids[row_index]
                    row_values: dict[str, object] = {}
                    row_is_complete = True
                    for column_index, source_key in enumerate(data_source_fieldnames):
                        if column_index in excluded_columns:
                            continue
                        value = row.get(source_key)
                        text = str("" if value is None else value).strip()
                        if text == "":
                            row_is_complete = False
                            row_values[data_fieldnames[column_index]] = None
                            continue
                        try:
                            converted_value: object = float(text)
                        except ValueError:
                            converted_value = text
                        if isinstance(converted_value, float) and not np.isfinite(converted_value):
                            row_is_complete = False
                        row_values[data_fieldnames[column_index]] = converted_value
                    if len(row_values) == 0:
                        row_is_complete = False
                    if row_is_complete:
                        if mapped_subject_id in rows:
                            raise ValueError(
                                f"{resource_name} contains duplicate values for subject "
                                f"{mapped_subject_id!r}"
                            )
                        rows[mapped_subject_id] = row_values
                return rows

            subject_source_columns = (
                tuple(source_fieldnames[1:]) if subject_id else source_fieldnames
            )
            subject_columns = tuple(fieldnames[1:]) if subject_id else fieldnames
            if len(subject_source_columns) == 0:
                raise ValueError(
                    f"{resource_name} file {mapped_path} has no columns for subject IDs"
                )
            excluded_subject_columns: set[int] = set()
            if exclude_column is not None:
                excluded_subject_columns = (
                    {int(exclude_column)}
                    if isinstance(exclude_column, int)
                    else {int(value) for value in exclude_column}
                )
                for index in excluded_subject_columns:
                    if index < 0 or index >= len(subject_source_columns):
                        raise ValueError(
                            f"{resource_name} column index {index} is out of range "
                            f"for {len(subject_source_columns)} columns"
                        )
            recognized_subject_columns: list[tuple[str, str]] = []
            for column_index, source_column in enumerate(subject_source_columns):
                if column_index in excluded_subject_columns:
                    continue
                if subject_id:
                    subject_value = str(subject_columns[column_index]).strip()
                    if subject_value == "":
                        raise ValueError(f"{resource_name} subject ID must not be empty")
                    try:
                        mapped_subject_id = DatasetSplitPlanner.map_subject_id(
                            subject_value,
                            dataset_subject_ids,
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"{resource_name} subject ID {subject_value!r} does not "
                            f"match {state.name} subject IDs "
                            f"(examples: {subject_id_examples}). "
                            f"Use a {resource_name.lower()} file with {state.name} "
                            "subject IDs, or rename the IDs."
                        ) from exc
                else:
                    if column_index >= len(dataset_subject_ids):
                        raise ValueError(
                            f"{resource_name} has {len(subject_source_columns)} subject "
                            f"columns but {state.name} has {len(dataset_subject_ids)} "
                            "subjects. With subject_id=False, columns are mapped by "
                            "dataset order; remove extra columns or use subject_id=True "
                            "with matching IDs."
                        )
                    mapped_subject_id = dataset_subject_ids[column_index]
                recognized_subject_columns.append((source_column, mapped_subject_id))
            if len(recognized_subject_columns) == 0:
                raise ValueError(
                    f"{resource_name} file {mapped_path} has no recognized "
                    "subject columns"
                )

            excluded_field_rows: set[int] = set()
            if exclude_row is not None:
                excluded_field_rows = (
                    {int(exclude_row)}
                    if isinstance(exclude_row, int)
                    else {int(value) for value in exclude_row}
                )
                for index in excluded_field_rows:
                    if index < 0 or index >= len(table_rows):
                        raise ValueError(
                            f"{resource_name} row index {index} is out of range "
                            f"for {len(table_rows)} rows"
                        )
            rows = {}
            for _, mapped_subject_id in recognized_subject_columns:
                if mapped_subject_id in rows:
                    raise ValueError(
                        f"{resource_name} contains duplicate values for subject "
                        f"{mapped_subject_id!r}"
                    )
                rows[mapped_subject_id] = {}
            used_field_names: set[str] = set()
            for row_index, row in enumerate(table_rows):
                if row_index in excluded_field_rows:
                    continue
                if subject_id:
                    field_name = str(
                        ""
                        if row.get(source_fieldnames[0]) is None
                        else row.get(source_fieldnames[0])
                    ).strip()
                    if field_name == "":
                        continue
                else:
                    field_name = f"field_{row_index + 1}"
                if field_name in used_field_names:
                    raise ValueError(f"{resource_name} contains duplicate field {field_name!r}")
                used_field_names.add(field_name)
                for source_column, mapped_subject_id in recognized_subject_columns:
                    value = row.get(source_column)
                    text = str("" if value is None else value).strip()
                    if text == "":
                        rows[mapped_subject_id][field_name] = None
                        continue
                    try:
                        converted_value = float(text)
                    except ValueError:
                        converted_value = text
                    rows[mapped_subject_id][field_name] = converted_value
            complete_rows: dict[str, dict[str, object]] = {}
            for mapped_subject_id, row_values in rows.items():
                row_is_complete = len(row_values) > 0
                for value in row_values.values():
                    if value is None:
                        row_is_complete = False
                    elif isinstance(value, float) and not np.isfinite(value):
                        row_is_complete = False
                if row_is_complete:
                    complete_rows[mapped_subject_id] = row_values
            return complete_rows
    if mapped_extension == ".mat":
        data = {
            key: value
            for key, value in loadmat(mapped_path).items()
            if not str(key).startswith("__")
        }
        subject_variable_name = None
        if subject_id:
            for key in data:
                if str(key).strip().lower() in {
                    "subject",
                    "subjects",
                    "subject_id",
                    "subject_ids",
                    "id",
                    "ids",
                }:
                    subject_variable_name = key
                    break
            if subject_variable_name is None:
                raise ValueError(
                    f"{resource_name} MAT files with subject_id=True require a "
                    "subject ID variable"
                )
        variables = {
            key: value
            for key, value in data.items()
            if key != subject_variable_name
        }
        if len(variables) == 0:
            raise ValueError(
                f"{resource_name} MAT file {mapped_path} does not contain usable "
                "variables"
            )
        subject_axis = 0 if accessed_by == "row" else 1
        subject_axis_sizes = []
        for key, value in variables.items():
            array = np.asarray(value)
            if array.ndim <= subject_axis:
                raise ValueError(
                    f"{resource_name} MAT variable {key!r} does not have a "
                    f"subject {accessed_by}"
                )
            subject_axis_sizes.append(int(array.shape[subject_axis]))
        if len(set(subject_axis_sizes)) != 1:
            raise ValueError(
                f"{resource_name} MAT variables do not share the same subject "
                "axis size"
            )
        subject_axis_size = subject_axis_sizes[0]
        if subject_id:
            subject_values = tuple(np.asarray(data[subject_variable_name]).ravel())
            if len(subject_values) != subject_axis_size:
                raise ValueError(
                    f"{resource_name} MAT subject ID variable has "
                    f"{len(subject_values)} values, "
                    f"but the data contains {subject_axis_size} subjects"
                )
        elif subject_axis_size > len(dataset_subject_ids):
            raise ValueError(
                f"{resource_name} MAT file has {subject_axis_size} subject entries, "
                f"but {state.name} has {len(dataset_subject_ids)} subjects. With "
                "subject_id=False, entries are mapped by dataset order; remove extra "
                "entries or use subject_id=True with matching IDs."
            )

        excluded_subjects: set[int] = set()
        subject_exclusion_source = exclude_row if accessed_by == "row" else exclude_column
        if subject_exclusion_source is not None:
            excluded_subjects = (
                {int(subject_exclusion_source)}
                if isinstance(subject_exclusion_source, int)
                else {int(value) for value in subject_exclusion_source}
            )
            for index in excluded_subjects:
                if index < 0 or index >= subject_axis_size:
                    axis_name = "row" if accessed_by == "row" else "column"
                    raise ValueError(
                        f"{resource_name} {axis_name} index {index} is out of "
                        f"range for {subject_axis_size} {axis_name}s"
                    )

        rows = {}
        for subject_position in range(subject_axis_size):
            if subject_position in excluded_subjects:
                continue
            if subject_id:
                mat_subject_value: object = np.asarray(data[subject_variable_name]).ravel()[
                    subject_position
                ]
                if isinstance(mat_subject_value, np.generic):
                    mat_subject_value = mat_subject_value.item()
                if isinstance(mat_subject_value, np.ndarray):
                    mat_subject_value = np.squeeze(mat_subject_value)
                    if mat_subject_value.shape == ():
                        mat_subject_value = mat_subject_value.item()
                    else:
                        mat_subject_value = "".join(
                            str(item) for item in mat_subject_value.ravel()
                        ).strip()
                if isinstance(mat_subject_value, bytes):
                    mat_subject_value = mat_subject_value.decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                if isinstance(mat_subject_value, float) and mat_subject_value.is_integer():
                    mat_subject_value = int(mat_subject_value)
                try:
                    mapped_subject_id = DatasetSplitPlanner.map_subject_id(
                        cast(str | int, mat_subject_value),
                        dataset_subject_ids,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{resource_name} subject ID {mat_subject_value!r} does not "
                        f"match {state.name} subject IDs "
                        f"(examples: {subject_id_examples}). "
                        f"Use a {resource_name.lower()} file with {state.name} "
                        "subject IDs, or rename the IDs."
                    ) from exc
            else:
                mapped_subject_id = dataset_subject_ids[subject_position]
            if mapped_subject_id in rows:
                raise ValueError(
                    f"{resource_name} contains duplicate values for subject "
                    f"{mapped_subject_id!r}"
                )
            row_values = dict[str, object]()
            for key, value in variables.items():
                selected_value = np.take(
                    np.asarray(value),
                    subject_position,
                    axis=subject_axis,
                )
                selected_value = np.squeeze(selected_value)
                if isinstance(selected_value, np.ndarray) and selected_value.shape == ():
                    selected_value = selected_value.item()
                if isinstance(selected_value, np.generic):
                    selected_value = selected_value.item()
                if isinstance(selected_value, bytes):
                    selected_value = selected_value.decode("utf-8", errors="replace").strip()
                if isinstance(selected_value, str):
                    row_values[str(key)] = selected_value.strip()
                elif isinstance(selected_value, np.ndarray):
                    if selected_value.dtype.kind in {"U", "S"}:
                        row_values[str(key)] = "".join(
                            str(item) for item in selected_value.ravel()
                        ).strip()
                    elif selected_value.dtype == object and selected_value.size == 1:
                        row_values[str(key)] = selected_value.item()
                    elif selected_value.size == 1:
                        row_values[str(key)] = selected_value.reshape(()).item()
                    else:
                        for value_index, item in enumerate(selected_value.ravel(), start=1):
                            row_values[f"{key}_{value_index}"] = (
                                item.item() if isinstance(item, np.generic) else item
                            )
                else:
                    row_values[str(key)] = selected_value
            rows[mapped_subject_id] = row_values

        field_exclusion_source = exclude_column if accessed_by == "row" else exclude_row
        if len(rows) > 0 and field_exclusion_source is not None:
            field_names = tuple(next(iter(rows.values())).keys())
            excluded_fields = (
                {int(field_exclusion_source)}
                if isinstance(field_exclusion_source, int)
                else {int(value) for value in field_exclusion_source}
            )
            for index in excluded_fields:
                if index < 0 or index >= len(field_names):
                    axis_name = "column" if accessed_by == "row" else "row"
                    raise ValueError(
                        f"{resource_name} {axis_name} index {index} is out of "
                        f"range for {len(field_names)} {axis_name}s"
                    )
            kept_fields = {
                field_name
                for field_index, field_name in enumerate(field_names)
                if field_index not in excluded_fields
            }
            rows = {
                mapped_subject_id: {
                    field_name: field_value
                    for field_name, field_value in row_values.items()
                    if field_name in kept_fields
                }
                for mapped_subject_id, row_values in rows.items()
            }

        complete_rows = {}
        for mapped_subject_id, row_values in rows.items():
            row_is_complete = len(row_values) > 0
            for value in row_values.values():
                if value is None:
                    row_is_complete = False
                elif isinstance(value, str) and value.strip() == "":
                    row_is_complete = False
                elif isinstance(value, float | complex) and not np.isfinite(value):
                    row_is_complete = False
                elif isinstance(value, np.ndarray):
                    if value.size == 0:
                        row_is_complete = False
                    elif value.dtype.kind in {"f", "c"} and np.any(~np.isfinite(value)):
                        row_is_complete = False
                    elif value.dtype.kind in {"U", "S"} and any(
                        str(item).strip() == "" for item in value.ravel()
                    ):
                        row_is_complete = False
                    elif value.dtype == object and any(item is None for item in value.ravel()):
                        row_is_complete = False
            if row_is_complete:
                complete_rows[mapped_subject_id] = row_values
        return complete_rows
    raise ValueError(
        f"{resource_name} extension must be one of: .csv, .mat"
    )
