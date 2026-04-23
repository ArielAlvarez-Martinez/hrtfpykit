from collections.abc import Callable
from pathlib import Path
import re


def resolve_video_subject_folder(
    root: Path,
    subject_id: str,
    subject_number: int,
) -> Path | None:
    candidate_names = (
        str(subject_id).strip().lower(),
        f"subject{subject_number}",
    )
    matches = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.strip().lower() in candidate_names
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Video path {root} contains multiple folders for subject {subject_id!r}: "
            + ", ".join(str(path.name) for path in matches)
        )
    if len(matches) == 0:
        return None
    return matches[0]


def resolve_video_position_from_path(path: Path) -> int | None:
    text = " ".join(part.lower() for part in path.parts)
    match = re.search(r"(?:pos|position)[-_]?(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def resolve_video_ear_from_path(path: Path) -> str | None:
    text = " ".join(part.lower() for part in path.parts)
    if re.search(r"(?<![a-z0-9])left(?![a-z0-9])", text):
        return "left"
    if re.search(r"(?<![a-z0-9])right(?![a-z0-9])", text):
        return "right"
    return None


def build_video_key(
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


def collect_video_files(
    path: Path,
    extensions: tuple[str, ...],
) -> list[str]:
    normalized_extensions = {extension.lower() for extension in extensions}

    def sort_key(file: Path) -> tuple[int, str, int | float, str]:
        stem = file.stem.strip().lower()
        match = re.fullmatch(r"([a-z_ -]*?)(\d+)", stem)
        if match is None:
            return (1, stem, float("inf"), file.name.lower())
        prefix = match.group(1).strip()
        return (0, prefix, int(match.group(2)), file.name.lower())

    return sorted(
        (
            str(file)
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in normalized_extensions
        ),
        key=lambda file: sort_key(Path(file)),
    )


def scan_video_paths(
    path: Path,
    subject_ids: tuple[str, ...],
    subject_numbers: dict[str, int],
    extensions: tuple[str, ...],
    align_by: tuple[str, ...],
) -> tuple[
    dict[tuple[str, int | None, str | None], list[str]],
    dict[str, int],
    tuple[str, ...],
]:
    index: dict[tuple[str, int | None, str | None], list[str]] = {}
    if not path.exists():
        raise ValueError(f"Video path does not exist: {path}")
    subject_video_counts: dict[str, int] = {}
    missing_subject_ids: list[str] = []
    for subject_id in subject_ids:
        subject_folder = resolve_video_subject_folder(
            path,
            subject_id,
            int(subject_numbers[subject_id]),
        )
        if subject_folder is None:
            missing_subject_ids.append(subject_id)
            continue
        subject_count = 0
        if "ear" in align_by:
            for ear in ("left", "right"):
                ear_folder = subject_folder / ear
                if not ear_folder.is_dir():
                    raise ValueError(
                        f"Video path {path} is incompatible with align_by={align_by}: "
                        f"subject {subject_id!r} is missing the {ear!r} folder"
                    )
                files = collect_video_files(ear_folder, extensions)
                if len(files) == 0:
                    raise ValueError(
                        f"Video path {path} is incompatible with align_by={align_by}: "
                        f"subject {subject_id!r} has no videos in {ear_folder}"
                    )
                subject_count += len(files)
                if "position" in align_by:
                    for file in files:
                        position_index = resolve_video_position_from_path(Path(file))
                        if position_index is None:
                            raise ValueError(
                                f"Video path {path} is incompatible with align_by={align_by}: "
                                f"subject {subject_id!r} has a video without a position token: {file}"
                            )
                        key = build_video_key(subject_id, align_by, position_index, ear)
                        index.setdefault(key, []).append(file)
                else:
                    key = build_video_key(subject_id, align_by, None, ear)
                    index[key] = files
        else:
            files = collect_video_files(subject_folder, extensions)
            if len(files) == 0:
                raise ValueError(
                    f"Video path {path} is incompatible with align_by={align_by}: "
                    f"subject {subject_id!r} has no videos in {subject_folder}"
                )
            subject_count = len(files)
            if "position" in align_by:
                for file in files:
                    position_index = resolve_video_position_from_path(Path(file))
                    if position_index is None:
                        raise ValueError(
                            f"Video path {path} is incompatible with align_by={align_by}: "
                            f"subject {subject_id!r} has a video without a position token: {file}"
                        )
                    key = build_video_key(subject_id, align_by, position_index, None)
                    index.setdefault(key, []).append(file)
            else:
                key = build_video_key(subject_id, align_by, None, None)
                index[key] = files
        subject_video_counts[subject_id] = subject_count
    return index, subject_video_counts, tuple(missing_subject_ids)


def apply_video_transform(
    paths: list[str],
    transform: Callable | None,
) -> object:
    values: list[object] = []
    for path in paths:
        value: object = path
        if transform is not None:
            value = transform(value)
        values.append(value)
    if len(values) == 1:
        return values[0]
    return values
