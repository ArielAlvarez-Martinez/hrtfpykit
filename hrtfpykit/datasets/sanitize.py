from collections.abc import Sequence
from copy import copy

from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ITDSpec,
    ILDSpec,
    SHSpec,
    MeshSpec,
    MetadataSpec,
    VideoSpec,
    ImageSpec,
)
from .specs_registry import get_spec_name
import numpy as np


def sanitize_subject_id(value: str) -> str:
    """Normalize a subject identifier for loose user matching.

    Subject references can arrive with whitespace or different casing, while
    dataset configs keep canonical IDs. This helper provides the normalized
    form used by :class:`~hrtfpykit.datasets.split.DatasetSplitPlanner` when it
    compares user aliases with configured subject identifiers. The function does
    not validate that the subject exists and does not replace the canonical value
    returned to callers.

    Parameters
    ----------
    value : str
        Subject identifier or subject-like value to normalize.

    Returns
    -------
    str
        Lowercase stripped subject identifier.

    """
    return str(value).strip().lower()


def sanitize_index_by(index_by: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize and validate dataset row-axis declarations.

    Indexed specs must agree on a subject-first row structure, and only supported
    axes can appear after subject. This helper converts string shorthand into
    tuples and rejects duplicate, incompatible, or ambiguous axes early in spec
    planning. It is used by
    :class:`~hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow`,
    :class:`~hrtfpykit.datasets.acoustic_context.DatasetAcousticContext`, and
    sample value selectors so construction and indexing interpret row axes
    consistently.

    A string equal to "subject" becomes a single-axis tuple. A string such as
    "subject-position" is split on hyphens. Sequence inputs are stripped and
    lowercased element by element.

    Parameters
    ----------
    index_by : str or sequence of str
        Requested dataset row axes. The first axis must be "subject". Optional
        following axes are "position", "ear", "frequency", or "samples".

    Returns
    -------
    tuple of str
        Normalized row axes.

    Raises
    ------
    ValueError
        If no axes are provided, if the first axis is not "subject", if axes are
        duplicated, if an unsupported axis is present, or if both "frequency" and
        "samples" are requested in the same row definition.

    """
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


def sanitize_grouped_by(grouped_by: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize and validate grouped resource axes.

    Table and media resources can be grouped by subject or subject-ear. This
    helper keeps that contract explicit so scanners and value selectors do not
    need to interpret arbitrary grouping combinations. It accepts the same
    hyphenated shorthand as
    :func:`~hrtfpykit.datasets.sanitize.sanitize_index_by`, but only the two
    grouping layouts supported by the dataset pipeline are valid.

    Parameters
    ----------
    grouped_by : str or sequence of str
        Requested grouping axes. Supported normalized values are ("subject",) and
        ("subject", "ear").

    Returns
    -------
    tuple of str
        Normalized grouping axes.

    Raises
    ------
    ValueError
        If the requested grouping is not subject-only or subject-ear.

    """
    if isinstance(grouped_by, str):
        value = str(grouped_by).strip().lower()
        if value == "subject":
            normalized = ("subject",)
        elif value.startswith("subject-"):
            normalized = tuple(part for part in value.split("-") if part != "")
        else:
            normalized = (value,)
    else:
        normalized = tuple(str(value).strip().lower() for value in grouped_by)
    if normalized not in {("subject",), ("subject", "ear")}:
        raise ValueError("grouped_by must be ('subject',) or ('subject', 'ear')")
    return normalized


def sanitize_ear(ear: str | None) -> str | None:
    """Normalize a table ear selector.

    Anthropometry and metadata specs can request no ear, both ears, or one side
    when grouped by ear. This helper centralizes accepted values so workflow
    validation and value selection use the same vocabulary. Empty strings are
    treated as no fixed ear selection.

    Parameters
    ----------
    ear : str or None
        Ear selector from a table-style spec.

    Returns
    -------
    str or None
        Normalized ear selector: None, "both", "left", or "right".

    Raises
    ------
    ValueError
        If the selector is not None, not empty, and not one of "both", "left",
        or "right".

    """
    if ear is None or str(ear).strip() == "":
        return None
    ear_value = str(ear).strip().lower()
    if ear_value == "both":
        return "both"
    if ear_value in {"left", "right"}:
        return ear_value
    raise ValueError("AnthropometrySpec ear must be None, 'both', 'left', or 'right'")


def sanitize_accessed_by(accessed_by: str) -> str:
    """Normalize table access direction.

    Table specs support row-oriented or column-oriented subject layouts. This
    helper validates the setting before table loading so row/column indexing
    errors remain tied to spec planning rather than surfacing later from
    :func:`~hrtfpykit.datasets.load.load_table`.

    Parameters
    ----------
    accessed_by : str
        Table access direction requested by metadata or anthropometry specs.

    Returns
    -------
    str
        Normalized access direction, either "row" or "column".

    Raises
    ------
    ValueError
        If the access direction is not "row" or "column".

    """
    accessed_by_value = str(accessed_by).strip().lower()
    if accessed_by_value not in {"row", "column"}:
        raise ValueError("AnthropometrySpec accessed_by must be 'row' or 'column'")
    return accessed_by_value


def sanitize_ears(ears: str | Sequence[str]) -> list[tuple[str, int]]:
    """Normalize HRTF ear selection into labels and source indices.

    HRTF-like resources use left/right source-ear indices, while user specs use
    readable names such as "both" or "left". This helper returns both
    representations and rejects duplicate or unsupported ear requests. The
    returned integer indices match the binaural ear axis used by loaded HRTF and
    HRIR arrays: left is index 0 and right is index 1.

    Parameters
    ----------
    ears : str or sequence of str
        Ear selection from HRTF-style specs. Strings can be "both", "left", or
        "right". Sequence inputs can contain "left" and "right".

    Returns
    -------
    list of tuple
        Ear labels paired with source ear indices.

    Raises
    ------
    ValueError
        If the selector is unsupported, if a sequence is empty, if a sequence
        contains duplicate ears, or if a sequence contains any value other than
        "left" or "right".

    """
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


def sanitize_positions(
    positions: str | Sequence[int] | np.ndarray,
    position_count: int,
) -> list[int]:
    """Normalize source position selection against a known position count.

    Position selectors can request all positions or explicit integer indices. This
    helper validates emptiness, duplicates, and bounds so acoustic context
    building receives a validated index list. It does not resolve named planes or
    geometric subsets; that is handled by
    :class:`~hrtfpykit.datasets.acoustic_context.DatasetAcousticContext` before
    this helper is called.

    Parameters
    ----------
    positions : str, sequence, or numpy.ndarray
        Requested position selection. The string "all" selects every available
        source position. Non-string inputs are converted to a one-dimensional
        integer array.
    position_count : int
        Number of positions available.

    Returns
    -------
    list of int
        Validated position indices.

    Raises
    ------
    ValueError
        If a string other than "all" is provided, if the explicit index list is
        empty, if duplicate indices are present, or if any index is outside the
        available position range.

    """
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


def sanitize_extensions(
    resource_name: str,
    extensions: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Normalize resource file extension filters.

    Specs and configs may provide extensions with or without a leading dot. This
    helper validates entries, rejects path-like values, lowercases them, and
    removes duplicates before resource scanning. A None value means no explicit
    extension filter was provided; callers can then fall back to dataset
    configuration defaults.

    Parameters
    ----------
    resource_name : str
        Resource label used in validation errors.
    extensions : tuple, list, or None
        Extension values to normalize. Entries can include or omit the leading
        dot.

    Returns
    -------
    tuple of str
        Unique lowercase extensions beginning with a dot.

    Raises
    ------
    ValueError
        If any extension entry is empty, is only ".", or contains path separators.

    """
    if extensions is None:
        return tuple()
    normalized: list[str] = []
    for extension in tuple(extensions):
        extension_text = str(extension).strip()
        if extension_text == "":
            raise ValueError(
                f"{resource_name} extension entries must be non-empty strings"
            )
        normalized_extension = (
            extension_text
            if extension_text.startswith(".")
            else f".{extension_text}"
        )
        if normalized_extension == ".":
            raise ValueError(
                f"{resource_name} extension '.' is invalid; provide a file extension like 'png' or '.png'"
            )
        if "/" in normalized_extension or "\\" in normalized_extension:
            raise ValueError(
                f"{resource_name} extension {extension_text!r} is invalid; do not include path separators"
            )
        normalized.append(normalized_extension.lower())
    return tuple(dict.fromkeys(normalized))


def sanitize_specs(
    specs: HRTFSpec
    | ITDSpec
    | ILDSpec
    | SHSpec
    | MeshSpec
    | AnthropometrySpec
    | MetadataSpec
    | ImageSpec
    | VideoSpec
    | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec]
    | None,
) -> tuple[
    HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec,
    ...,
]:
    """Normalize input or target specs into copied spec tuples.

    Dataset construction should never mutate caller-owned spec objects. This
    helper accepts None, a single spec, or a sequence, copies each spec, and
    rejects duplicate public names before workflow validation continues. It is the
    first normalization step used by
    :class:`~hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow`, before axis,
    grouping, path, and resource compatibility checks run.

    Parameters
    ----------
    specs : spec, sequence of specs, or None
        User-provided dataset specs for an input or target collection.

    Returns
    -------
    tuple of specs
        Copied and name-validated specs.

    Raises
    ------
    TypeError
        If a string is passed instead of a dataset spec object.
    ValueError
        If two specs resolve to the same public sample name.

    """
    if specs is None:
        return tuple()
    if isinstance(specs, str):
        raise TypeError("inputs and target must use dataset spec objects, not strings")
    if isinstance(
        specs,
        (
            HRTFSpec,
            ITDSpec,
            ILDSpec,
            SHSpec,
            MeshSpec,
            AnthropometrySpec,
            MetadataSpec,
            ImageSpec,
            VideoSpec,
        ),
    ):
        values = (specs,)
    else:
        values = tuple(specs)
    names: set[str] = set()
    normalized: list[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec
    ] = []
    for spec in values:
        name = get_spec_name(spec)
        if name in names:
            raise ValueError(f"Duplicate dataset spec name {name!r} is not allowed")
        names.add(name)
        normalized.append(copy(spec))
    return tuple(normalized)
