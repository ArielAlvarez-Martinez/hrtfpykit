from collections.abc import Sequence

from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ITDSpec,
    ILDSpec,
    SHSpec,
    MeshSpec,
    VideoSpec,
    ImageSpec,
)
import numpy as np


def sanitize_index_by(index_by: str | Sequence[str]) -> tuple[str, ...]:
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


def sanitize_ears(ears: str | Sequence[str]) -> list[tuple[str, int]]:
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
    | ImageSpec
    | VideoSpec
    | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec]
    | None,
) -> tuple[
    HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
    ...,
]:
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
            ImageSpec,
            VideoSpec,
        ),
    ):
        values = (specs,)
    else:
        values = tuple(specs)
    names: set[str] = set()
    normalized: list[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec
    ] = []
    from .specs_workflow import DatasetSpecWorkflow

    for spec in values:
        name = DatasetSpecWorkflow.get_spec_name(spec)
        if name in names:
            raise ValueError(f"Duplicate dataset spec name {name!r} is not allowed")
        names.add(name)
        normalized.append(spec)
    return tuple(normalized)
