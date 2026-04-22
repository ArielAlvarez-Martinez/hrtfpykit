from collections.abc import Callable
from pathlib import Path
import re


def discover_mesh_paths(
    root: Path,
    filename_pattern: str,
    extensions: tuple[str, ...],
    subject_ids: tuple[str, ...],
    resolve_subject_id: Callable[[str, tuple[str, ...]], str],
) -> dict[str, Path]:
    pattern = re.compile(filename_pattern, flags=re.IGNORECASE)
    allowed_extensions = {extension.lower() for extension in extensions}
    paths: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_extensions:
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        try:
            subject_id = resolve_subject_id(match.group("subject_id"), subject_ids)
        except ValueError:
            continue
        paths[subject_id] = path
    return paths
