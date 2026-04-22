from collections.abc import Callable
from pathlib import Path
import re

try:
    from PIL import Image
except ImportError:
    Image = None


def resolve_image_subject_id_from_path(path: Path, subject_ids: tuple[str, ...]) -> str | None:
    parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    for subject_id in sorted(subject_ids, key=len, reverse=True):
        subject_key = subject_id.lower()
        if subject_key in parts:
            return subject_id
        if re.search(rf"(?<![a-z0-9]){re.escape(subject_key)}(?![a-z0-9])", stem):
            return subject_id
    return None


def resolve_image_position_from_path(path: Path) -> int | None:
    text = " ".join(part.lower() for part in path.parts)
    match = re.search(r"(?:pos|position)[-_]?(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def resolve_image_ear_from_path(path: Path) -> str | None:
    text = " ".join(part.lower() for part in path.parts)
    if re.search(r"(?<![a-z0-9])left(?![a-z0-9])", text):
        return "left"
    if re.search(r"(?<![a-z0-9])right(?![a-z0-9])", text):
        return "right"
    return None


def build_image_key(
    subject_id: str,
    align_by: tuple[str, ...],
    position_index: int | None,
    ear: str | None,
) -> tuple[str, int | None, str | None]:
    return (
        subject_id,
        position_index if "position" in align_by else None,
        ear if "ear" in align_by else None,
    )


def scan_image_paths(
    path: Path,
    subject_ids: tuple[str, ...],
    extensions: tuple[str, ...],
    align_by: tuple[str, ...],
) -> dict[tuple[str, int | None, str | None], list[str]]:
    index: dict[tuple[str, int | None, str | None], list[str]] = {}
    if not path.exists():
        raise ValueError(f"Image path does not exist: {path}")
    normalized_extensions = {extension.lower() for extension in extensions}
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in normalized_extensions:
            continue
        subject_id = resolve_image_subject_id_from_path(file, subject_ids)
        if subject_id is None:
            continue
        position_index = None
        ear = None
        if "position" in align_by:
            position_index = resolve_image_position_from_path(file)
            if position_index is None:
                continue
        if "ear" in align_by:
            ear = resolve_image_ear_from_path(file)
            if ear is None:
                continue
        key = build_image_key(subject_id, align_by, position_index, ear)
        index.setdefault(key, []).append(str(file))
    return index


def apply_image_transform(
    paths: list[str],
    transform: Callable | None,
    image_size: int | tuple[int, int] | None = None,
) -> object:
    normalized_image_size = None
    if image_size is not None and Image is None:
        raise ImportError("Pillow is required to use ImageSpec(image_size=...)")
    if image_size is not None:
        if isinstance(image_size, int):
            if image_size <= 0:
                raise ValueError("image_size must be a positive integer or a tuple of two positive integers")
            normalized_image_size = (int(image_size), int(image_size))
        else:
            if len(image_size) != 2:
                raise ValueError("image_size tuple must contain exactly two integers")
            width = int(image_size[0])
            height = int(image_size[1])
            if width <= 0 or height <= 0:
                raise ValueError("image_size values must be positive integers")
            normalized_image_size = (width, height)
        resampling = (
            Image.Resampling.BILINEAR
            if hasattr(Image, "Resampling")
            else Image.BILINEAR
        )
    values: list[object] = []
    for path in paths:
        value: object = path
        if normalized_image_size is not None:
            with Image.open(path) as image:
                image.load()
                value = image.resize(normalized_image_size, resample=resampling)
        if transform is not None:
            value = transform(value)
        values.append(value)
    if len(values) == 1:
        return values[0]
    return values
