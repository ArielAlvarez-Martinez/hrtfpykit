from collections.abc import Callable
from pathlib import Path
import csv
import re
from typing import cast
import warnings

import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

from ..hrtf.planes import get_frontal_plane, get_horizontal_plane, get_median_plane
from ..hrtf.dsp import imag, magnitude, magnitude_db, phase, real
from ..hrtf.coordinates import get_spherical_positions
from ..main import load_hrtf
from .download import BaseDownload
from .index import (
    build_rows,
    normalize_ears,
    normalize_index_by,
    normalize_positions,
    split_subject_ids,
)
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    MeshSpec,
    VideoSpec,
    get_spec_name,
    normalize_columns,
    normalize_specs,
)


def normalize_subject_id(value: str) -> str:
    subject_id = str(value).strip().lower()
    if subject_id.startswith("pp"):
        suffix = subject_id[2:]
        if suffix.isdigit():
            return f"pp{int(suffix)}"
        return subject_id
    if subject_id.isdigit():
        return f"pp{int(subject_id)}"
    return subject_id


def resolve_dataset_subject_id(
    value: str | int,
    subject_ids: tuple[str, ...],
) -> str:
    if len(subject_ids) == 0:
        raise ValueError("subject_ids must not be empty")
    if isinstance(value, int):
        index = int(value)
        if index < 1 or index > len(subject_ids):
            raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
        return subject_ids[index - 1]
    text = str(value).strip()
    if text == "":
        raise ValueError("Subject reference must not be empty")
    subject_map = {subject_id.lower(): subject_id for subject_id in subject_ids}
    if text.lower() in subject_map:
        return subject_map[text.lower()]
    if text.isdigit():
        index = int(text)
        if index < 1 or index > len(subject_ids):
            raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
        return subject_ids[index - 1]
    normalized = normalize_subject_id(text)
    if normalized.lower() in subject_map:
        return subject_map[normalized.lower()]
    raise ValueError(f"Unknown subject reference {value!r}")


def sort_subject_ids(subject_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    def subject_sort_key(value: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", str(value))
        if match is None:
            return (0, str(value).lower())
        return (int(match.group(1)), str(value).lower())

    return sorted(subject_ids, key=subject_sort_key)


def build_value_signature(value: object) -> object:
    if isinstance(value, Path):
        return ("path", str(value.expanduser()))
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return ("array", tuple(array.shape), build_value_signature(array.tolist()))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), build_value_signature(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(build_value_signature(item) for item in value)
    return value


def discover_hrtf_paths(
    root: Path,
    spec: HRTFSpec,
    variant: str,
    subject_ids: tuple[str, ...],
) -> dict[str, Path]:
    if spec.filename_pattern is None:
        raise ValueError("Dataset hrtf spec is missing filename_pattern")
    pattern = re.compile(spec.filename_pattern, flags=re.IGNORECASE)
    variant_key = str(variant).strip().lower()
    paths: dict[str, Path] = {}
    for path in root.rglob("*.sofa"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        if str(match.group("variant")).strip().lower() != variant_key:
            continue
        try:
            subject_id = resolve_dataset_subject_id(match.group("subject_id"), subject_ids)
        except ValueError:
            continue
        paths[subject_id] = path
    return paths


def discover_mesh_paths(
    root: Path,
    spec: MeshSpec,
    subject_ids: tuple[str, ...],
) -> dict[str, Path]:
    if spec.filename_pattern is None:
        raise ValueError("Dataset mesh spec is missing filename_pattern")
    pattern = re.compile(spec.filename_pattern, flags=re.IGNORECASE)
    allowed_extensions = {extension.lower() for extension in spec.extensions}
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
            subject_id = resolve_dataset_subject_id(match.group("subject_id"), subject_ids)
        except ValueError:
            continue
        paths[subject_id] = path
    return paths


def find_anthropometry_path(root: Path, spec: AnthropometrySpec) -> Path | None:
    if spec.filename is None:
        raise ValueError("Dataset anthropometry spec is missing filename")
    for path in root.rglob(spec.filename):
        if path.is_file():
            return path
    return None


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
    spec: AnthropometrySpec,
    subject_ids: tuple[str, ...],
) -> dict[str, dict[str, float | str | None]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or len(reader.fieldnames) == 0:
            raise ValueError(f"Anthropometry file {path} does not contain headers")
        fieldnames = {fieldname.lower(): fieldname for fieldname in reader.fieldnames}
        subject_column = reader.fieldnames[0]
        for candidate in spec.subject_column_candidates:
            if candidate.lower() in fieldnames:
                subject_column = fieldnames[candidate.lower()]
                break
        rows: dict[str, dict[str, float | str | None]] = {}
        for row in reader:
            raw_subject_id = row.get(subject_column)
            if raw_subject_id is None or str(raw_subject_id).strip() == "":
                continue
            try:
                subject_id = resolve_dataset_subject_id(raw_subject_id, subject_ids)
            except ValueError:
                continue
            converted: dict[str, float | str | None] = {}
            for key, value in row.items():
                if key is None:
                    continue
                converted[key] = convert_table_value("" if value is None else value)
            rows[subject_id] = converted
    return rows


def resolve_subject_id_from_path(path: Path, subject_ids: tuple[str, ...]) -> str | None:
    parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    for subject_id in sorted(subject_ids, key=len, reverse=True):
        subject_key = subject_id.lower()
        if subject_key in parts:
            return subject_id
        if re.search(rf"(?<![a-z0-9]){re.escape(subject_key)}(?![a-z0-9])", stem):
            return subject_id
    return None


def resolve_position_from_path(path: Path) -> int | None:
    text = " ".join(part.lower() for part in path.parts)
    match = re.search(r"(?:pos|position)[-_]?(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def resolve_ear_from_path(path: Path) -> str | None:
    text = " ".join(part.lower() for part in path.parts)
    if re.search(r"(?<![a-z0-9])left(?![a-z0-9])", text):
        return "left"
    if re.search(r"(?<![a-z0-9])right(?![a-z0-9])", text):
        return "right"
    return None


def build_media_key(
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


def scan_media_paths(
    path: Path,
    subject_ids: tuple[str, ...],
    extensions: tuple[str, ...],
    align_by: tuple[str, ...],
) -> dict[tuple[str, int | None, str | None], list[str]]:
    index: dict[tuple[str, int | None, str | None], list[str]] = {}
    if not path.exists():
        raise ValueError(f"Media path does not exist: {path}")
    normalized_extensions = {extension.lower() for extension in extensions}
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in normalized_extensions:
            continue
        subject_id = resolve_subject_id_from_path(file, subject_ids)
        if subject_id is None:
            continue
        position_index = None
        ear = None
        if "position" in align_by:
            position_index = resolve_position_from_path(file)
            if position_index is None:
                continue
        if "ear" in align_by:
            ear = resolve_ear_from_path(file)
            if ear is None:
                continue
        key = build_media_key(subject_id, align_by, position_index, ear)
        index.setdefault(key, []).append(str(file))
    return index


def apply_media_transform(
    paths: list[str],
    transform: Callable | None,
    image_size: int | tuple[int, int] | None = None,
) -> object:
    normalized_image_size = None
    if image_size is not None:
        if Image is None:
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


def resolve_positions_selection(
    positions: str | tuple[object, ...] | list[int] | np.ndarray | dict[str, object],
    hrtf,
) -> list[int]:
    if isinstance(positions, dict):
        plane = positions.get("plane")
        if plane is None:
            raise ValueError("positions dict must include 'plane'")
        plane_key = str(plane).strip().lower()
        default_angle = 90.0 if plane_key == "frontal" else 0.0
        angle = positions.get("angle", positions.get("plane_angle", default_angle))
        angle_unit = str(positions.get("angle_unit", "degrees")).strip().lower()
    elif isinstance(positions, str):
        position_key = str(positions).strip().lower()
        if position_key == "all":
            return normalize_positions("all", int(hrtf.Sources.get_positions().shape[0]))
        if position_key in {"horizontal", "median", "frontal"}:
            plane_key = position_key
            angle = 90.0 if plane_key == "frontal" else 0.0
            angle_unit = "degrees"
        else:
            raise ValueError(
                "positions must be 'all', a sequence of indices, or a plane selection"
            )
    elif isinstance(positions, tuple) and len(positions) in {2, 3} and isinstance(positions[0], str):
        plane_key = str(positions[0]).strip().lower()
        if plane_key not in {"horizontal", "median", "frontal"}:
            raise ValueError("Plane selection must be horizontal, median, or frontal")
        angle = positions[1]
        angle_unit = "degrees" if len(positions) == 2 else str(positions[2]).strip().lower()
    else:
        return normalize_positions(
            positions,
            int(hrtf.Sources.get_positions().shape[0]),
        )

    if plane_key == "horizontal":
        indices, _ = get_horizontal_plane(
            hrtf=hrtf,
            elevation=float(angle),
            angle_unit=angle_unit,
        )
    elif plane_key == "median":
        indices, _ = get_median_plane(
            hrtf=hrtf,
            azimuth=float(angle),
            angle_unit=angle_unit,
        )
    else:
        indices, _ = get_frontal_plane(
            hrtf=hrtf,
            azimuth=float(angle),
            angle_unit=angle_unit,
        )
    return [int(index) for index in np.asarray(indices, dtype=int).reshape(-1)]


def select_hrtf_signal(
    hrtf,
    row: dict[str, str | int | None],
    selected_position_indices: list[int],
    selected_ears: list[tuple[str, int]],
    domain: str,
    signal: str,
) -> np.ndarray:
    if domain == "ir":
        values = np.asarray(hrtf.IR.values, dtype=float)
    else:
        tf_values = np.asarray(hrtf.TF.values)
        if signal in {"tf_complex", "complex"}:
            values = tf_values
        elif signal == "magnitude":
            values = magnitude(tf_values)
        elif signal == "magnitude_db":
            values = magnitude_db(tf_values)
        elif signal == "phase":
            values = phase(tf_values)
        elif signal == "real":
            values = real(tf_values)
        elif signal == "imag":
            values = imag(tf_values)
        else:
            raise ValueError(f"Unsupported signal {signal!r}")

    if row["position_index"] is None:
        if len(selected_position_indices) != values.shape[0]:
            values = np.take(values, selected_position_indices, axis=0)
    else:
        values = np.asarray(values[int(row["position_index"])])

    if row["ear_index"] is None:
        if len(selected_ears) == 1:
            values = np.asarray(values[int(selected_ears[0][1])])
    else:
        values = np.asarray(values[int(row["ear_index"])])

    return np.asarray(values)


def select_anthropometry_columns(
    values: dict[str, float | str | None],
    columns: tuple[str, ...] | None,
) -> dict[str, float | str | None]:
    if columns is None:
        return dict(values)
    missing = [column for column in columns if column not in values]
    if missing:
        raise ValueError(f"Anthropometry columns are missing: {missing}")
    return {column: values[column] for column in columns}


class BaseDataset:
    dataset_name: str = ""
    dataset_subject_ids: tuple[str, ...] = tuple()
    dataset_base_url: str | None = None
    dataset_download_resources: tuple[str, ...] = tuple()
    dataset_download_class = None
    dataset_hrtf_spec: HRTFSpec | None = None
    dataset_mesh_spec: MeshSpec | None = None
    dataset_anthropometry_spec: AnthropometrySpec | None = None
    dataset_image_spec: ImageSpec | None = None
    dataset_video_spec: VideoSpec | None = None

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_version: str = "all",
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...] = HRTFSpec(),
        target: HRTFSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
        | None = None,
        index_by: str | tuple[str, ...] = ("subject",),
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
    ) -> None:
        self.dataset_name = str(type(self).dataset_name)
        self.dataset_subject_ids = tuple(type(self).dataset_subject_ids)
        self.dataset_base_url = type(self).dataset_base_url
        self.dataset_download_resources = tuple(type(self).dataset_download_resources)
        self.dataset_download_class = type(self).dataset_download_class
        self.dataset_hrtf_spec = type(self).dataset_hrtf_spec
        self.dataset_mesh_spec = type(self).dataset_mesh_spec
        self.dataset_anthropometry_spec = type(self).dataset_anthropometry_spec
        self.dataset_image_spec = type(self).dataset_image_spec
        self.dataset_video_spec = type(self).dataset_video_spec
        self.root = Path(root)
        if exclude_subject_ids is None:
            self.exclude_subject_ids = tuple()
        elif isinstance(exclude_subject_ids, (str, int)):
            self.exclude_subject_ids = (
                resolve_dataset_subject_id(exclude_subject_ids, self.dataset_subject_ids),
            )
        else:
            self.exclude_subject_ids = tuple(
                dict.fromkeys(
                    resolve_dataset_subject_id(subject_id, self.dataset_subject_ids)
                    for subject_id in exclude_subject_ids
                )
            )
        self.download_manager = None
        if self.dataset_download_class is not None:
            self.download_manager = cast(
                BaseDownload,
                self.dataset_download_class(
                    root=self.root,
                    excluded_subject_ids=self.exclude_subject_ids,
                    hrtf_spec=self.dataset_hrtf_spec,
                    mesh_spec=self.dataset_mesh_spec,
                    anthropometry_spec=self.dataset_anthropometry_spec,
                ),
            )
            self.root = self.download_manager.root
        self.input_specs = normalize_specs(inputs)
        self.target_specs = normalize_specs(target)
        self.input_names = tuple(get_spec_name(spec) for spec in self.input_specs)
        self.target_names = tuple(get_spec_name(spec) for spec in self.target_specs)
        self.index_by = normalize_index_by(index_by)
        all_specs = self.input_specs + self.target_specs
        self.hrtf_specs = tuple(
            spec for spec in all_specs if isinstance(spec, HRTFSpec)
        )
        self.mesh_specs = tuple(
            spec for spec in all_specs if isinstance(spec, MeshSpec)
        )
        self.anthropometry_specs = tuple(
            spec for spec in all_specs if isinstance(spec, AnthropometrySpec)
        )
        self.image_specs = tuple(
            spec for spec in all_specs if isinstance(spec, ImageSpec)
        )
        self.video_specs = tuple(
            spec for spec in all_specs if isinstance(spec, VideoSpec)
        )
        input_hrtf_specs = tuple(
            spec for spec in self.input_specs if isinstance(spec, HRTFSpec)
        )
        input_mesh_specs = tuple(
            spec for spec in self.input_specs if isinstance(spec, MeshSpec)
        )
        input_anthropometry_specs = tuple(
            spec for spec in self.input_specs if isinstance(spec, AnthropometrySpec)
        )
        input_image_specs = tuple(
            spec for spec in self.input_specs if isinstance(spec, ImageSpec)
        )
        input_video_specs = tuple(
            spec for spec in self.input_specs if isinstance(spec, VideoSpec)
        )

        if len(self.hrtf_specs) > 1:
            resolved_variants = set()
            for spec in self.hrtf_specs:
                resolved_variant = (
                    None
                    if self.dataset_hrtf_spec is None or self.dataset_hrtf_spec.default_variant is None
                    else str(self.dataset_hrtf_spec.default_variant).strip().lower()
                )
                if spec.variant is not None:
                    resolved_variant = str(spec.variant).strip().lower()
                resolved_variants.add(resolved_variant)
            if len(resolved_variants) > 1:
                raise ValueError(
                    "All HRTFSpec objects must resolve to the same variant when hrtf is used in both inputs and target"
                )
            if len({build_value_signature(spec.positions) for spec in self.hrtf_specs}) > 1:
                raise ValueError(
                    "All HRTFSpec objects must use the same positions when hrtf is used in both inputs and target"
                )
            if len({build_value_signature(spec.ears) for spec in self.hrtf_specs}) > 1:
                raise ValueError(
                    "All HRTFSpec objects must use the same ears when hrtf is used in both inputs and target"
                )

        if len(self.image_specs) > 1:
            image_path_signatures = {
                build_value_signature(Path(spec.path))
                for spec in self.image_specs
                if spec.path is not None
            }
            if len(image_path_signatures) > 1:
                raise ValueError(
                    "All ImageSpec objects must use the same path when image is used in both inputs and target"
                )
            if len({normalize_index_by(spec.align_by) for spec in self.image_specs}) > 1:
                raise ValueError(
                    "All ImageSpec objects must use the same align_by when image is used in both inputs and target"
                )

        if len(self.video_specs) > 1:
            video_path_signatures = {
                build_value_signature(Path(spec.path))
                for spec in self.video_specs
                if spec.path is not None
            }
            if len(video_path_signatures) > 1:
                raise ValueError(
                    "All VideoSpec objects must use the same path when video is used in both inputs and target"
                )
            if len({normalize_index_by(spec.align_by) for spec in self.video_specs}) > 1:
                raise ValueError(
                    "All VideoSpec objects must use the same align_by when video is used in both inputs and target"
                )

        self.hrtf_spec = (
            input_hrtf_specs[0]
            if len(input_hrtf_specs) > 0
            else (None if len(self.hrtf_specs) == 0 else self.hrtf_specs[0])
        )
        self.mesh_spec = (
            input_mesh_specs[0]
            if len(input_mesh_specs) > 0
            else (None if len(self.mesh_specs) == 0 else self.mesh_specs[0])
        )
        self.anthropometry_spec = (
            input_anthropometry_specs[0]
            if len(input_anthropometry_specs) > 0
            else (
                None
                if len(self.anthropometry_specs) == 0
                else self.anthropometry_specs[0]
            )
        )
        self.image_spec = next(
            (
                spec
                for spec in input_image_specs
                if spec.path is not None
            ),
            None,
        )
        if self.image_spec is None:
            self.image_spec = next(
                (
                    spec
                    for spec in self.image_specs
                    if spec.path is not None
                ),
                None,
            )
        if self.image_spec is None and len(input_image_specs) > 0:
            self.image_spec = input_image_specs[0]
        if self.image_spec is None and len(self.image_specs) > 0:
            self.image_spec = self.image_specs[0]
        self.video_spec = next(
            (
                spec
                for spec in input_video_specs
                if spec.path is not None
            ),
            None,
        )
        if self.video_spec is None:
            self.video_spec = next(
                (
                    spec
                    for spec in self.video_specs
                    if spec.path is not None
                ),
                None,
            )
        if self.video_spec is None and len(input_video_specs) > 0:
            self.video_spec = input_video_specs[0]
        if self.video_spec is None and len(self.video_specs) > 0:
            self.video_spec = self.video_specs[0]
        self.spec_map: dict[str, HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] = {}
        if self.hrtf_spec is not None:
            self.spec_map["hrtf"] = self.hrtf_spec
        if self.mesh_spec is not None:
            self.spec_map["mesh"] = self.mesh_spec
        if self.anthropometry_spec is not None:
            self.spec_map["anthropometry"] = self.anthropometry_spec
        if self.image_spec is not None:
            self.spec_map["image"] = self.image_spec
        if self.video_spec is not None:
            self.spec_map["video"] = self.video_spec
        self.cache_hrtf = True if len(self.hrtf_specs) == 0 else any(bool(spec.cache) for spec in self.hrtf_specs)
        self.hrtf_cache: dict[str, object] = {}
        self.image_path = (
            None
            if self.image_spec is None or self.image_spec.path is None
            else Path(self.image_spec.path)
        )
        self.video_path = (
            None
            if self.video_spec is None or self.video_spec.path is None
            else Path(self.video_spec.path)
        )
        self.image_align_by = (
            None if self.image_spec is None else normalize_index_by(self.image_spec.align_by)
        )
        self.video_align_by = (
            None if self.video_spec is None else normalize_index_by(self.video_spec.align_by)
        )
        self.available_positions: np.ndarray | None = None
        self.selected_positions: np.ndarray | None = None
        self.available_azimuth_angles: np.ndarray | None = None
        self.available_elevation_angles: np.ndarray | None = None
        self.azimuth_angles: np.ndarray | None = None
        self.elevation_angles: np.ndarray | None = None
        self.frequency_bins: np.ndarray | None = None
        self.selected_position_indices: list[int] = []
        self.selected_ears = [] if self.hrtf_spec is None else normalize_ears(self.hrtf_spec.ears)
        self.position_encoding = (
            "none" if self.hrtf_spec is None else str(self.hrtf_spec.position_encoding).strip().lower()
        )
        self.ear_encoding = (
            "none" if self.hrtf_spec is None else str(self.hrtf_spec.ear_encoding).strip().lower()
        )
        self.hrtf_variant = None
        if len(self.hrtf_specs) > 0:
            if self.dataset_hrtf_spec is not None:
                self.hrtf_variant = self.dataset_hrtf_spec.default_variant
                if self.hrtf_variant is not None:
                    self.hrtf_variant = str(self.hrtf_variant).strip().lower()
            for spec in self.hrtf_specs:
                if spec.variant is not None:
                    self.hrtf_variant = str(spec.variant).strip().lower()
                    break

        if len(self.input_specs) == 0 and len(self.target_specs) == 0:
            raise ValueError("inputs and target cannot both be empty")
        if self.position_encoding not in {"none", "one-hot"}:
            raise ValueError("position_encoding must be 'none' or 'one-hot'")
        if self.ear_encoding not in {"none", "one-hot"}:
            raise ValueError("ear_encoding must be 'none' or 'one-hot'")

        if self.position_encoding != "none" and "position" not in self.index_by:
            raise ValueError("position_encoding requires index_by to include 'position'")
        if self.ear_encoding != "none" and "ear" not in self.index_by:
            raise ValueError("ear_encoding requires index_by to include 'ear'")

        if self.image_spec is not None:
            if self.dataset_image_spec is None:
                raise ValueError(f"{self.dataset_name} does not define an image pipeline")
            if self.image_path is None:
                raise ValueError("ImageSpec.path is required when image is selected")
            if self.dataset_image_spec.supported_align_by is None:
                raise ValueError(f"{self.dataset_name} image spec is missing supported_align_by")
            if self.image_align_by not in self.dataset_image_spec.supported_align_by:
                raise ValueError(
                    f"image align_by must be one of {self.dataset_image_spec.supported_align_by}"
                )
            if "position" in self.image_align_by and "position" not in self.index_by:
                raise ValueError("image align_by including 'position' requires index_by to include 'position'")
            if "ear" in self.image_align_by and "ear" not in self.index_by:
                raise ValueError("image align_by including 'ear' requires index_by to include 'ear'")
        if self.video_spec is not None:
            if self.dataset_video_spec is None:
                raise ValueError(f"{self.dataset_name} does not define a video pipeline")
            if self.video_path is None:
                raise ValueError("VideoSpec.path is required when video is selected")
            if self.dataset_video_spec.supported_align_by is None:
                raise ValueError(f"{self.dataset_name} video spec is missing supported_align_by")
            if self.video_align_by not in self.dataset_video_spec.supported_align_by:
                raise ValueError(
                    f"video align_by must be one of {self.dataset_video_spec.supported_align_by}"
                )
            if "position" in self.video_align_by and "position" not in self.index_by:
                raise ValueError("video align_by including 'position' requires index_by to include 'position'")
            if "ear" in self.video_align_by and "ear" not in self.index_by:
                raise ValueError("video align_by including 'ear' requires index_by to include 'ear'")

        if ("position" in self.index_by or "ear" in self.index_by) and self.hrtf_spec is None:
            raise ValueError(
                "index_by including 'position' or 'ear' currently requires hrtf in inputs or target"
            )

        if self.hrtf_spec is not None:
            if self.dataset_hrtf_spec is None:
                raise ValueError(f"{self.dataset_name} does not provide hrtf data")
            if self.hrtf_variant is None:
                raise ValueError("hrtf_variant could not be resolved")
            if self.dataset_hrtf_spec.variants is None:
                raise ValueError(f"{self.dataset_name} hrtf spec is missing variants")
            if self.hrtf_variant not in self.dataset_hrtf_spec.variants:
                raise ValueError(
                    f"Unsupported hrtf_variant {self.hrtf_variant!r}. "
                    f"Expected one of {self.dataset_hrtf_spec.variants}"
                )
            for spec in self.hrtf_specs:
                domain = str(spec.domain).strip().lower()
                signal = str(spec.signal).strip().lower()
                if domain not in self.dataset_hrtf_spec.supported_domains:
                    raise ValueError(
                        f"Unsupported domain {domain!r}. Expected one of {self.dataset_hrtf_spec.supported_domains}"
                    )
                if signal not in self.dataset_hrtf_spec.supported_signals:
                    raise ValueError(
                        f"Unsupported signal {signal!r}. Expected one of {self.dataset_hrtf_spec.supported_signals}"
                    )
                if domain == "ir" and signal != "ir":
                    raise ValueError("signal must be 'ir' when domain='ir'")
                if domain == "tf" and signal == "ir":
                    raise ValueError("signal 'ir' requires domain='ir'")

        if download:
            if self.download_manager is None:
                raise ValueError(f"{self.dataset_name} does not define a download manager")
            self.download_manager.download(
                download_resources=download_resources,
                download_hrtf_version=download_hrtf_version,
            )

        if self.download_manager is not None:
            self.hrtf_paths = (
                {}
                if self.dataset_hrtf_spec is None
                else self.download_manager.get_hrtf_paths(
                    "measured" if self.hrtf_variant is None else self.hrtf_variant
                )
            )
            self.mesh_paths = (
                {}
                if self.dataset_mesh_spec is None
                else self.download_manager.get_mesh_paths()
            )
            anthropometry_path = (
                None
                if self.dataset_anthropometry_spec is None
                else self.download_manager.get_anthropometry_path()
            )
        else:
            self.hrtf_paths = (
                {}
                if self.dataset_hrtf_spec is None
                else discover_hrtf_paths(
                    self.root,
                    self.dataset_hrtf_spec,
                    "measured" if self.hrtf_variant is None else self.hrtf_variant,
                    self.dataset_subject_ids,
                )
            )
            self.mesh_paths = (
                {}
                if self.dataset_mesh_spec is None
                else discover_mesh_paths(
                    self.root,
                    self.dataset_mesh_spec,
                    self.dataset_subject_ids,
                )
            )
            anthropometry_path = (
                None
                if self.dataset_anthropometry_spec is None
                else find_anthropometry_path(
                    self.root,
                    self.dataset_anthropometry_spec,
                )
            )
        self.anthropometry_rows: dict[str, dict[str, float | str | None]] = {}
        if self.dataset_anthropometry_spec is not None and anthropometry_path is not None:
            self.anthropometry_rows = load_anthropometry_rows(
                anthropometry_path,
                self.dataset_anthropometry_spec,
                self.dataset_subject_ids,
            )
            if len(self.anthropometry_rows) > 0:
                first_row = next(iter(self.anthropometry_rows.values()))
                for spec in self.anthropometry_specs:
                    columns = normalize_columns(spec.columns)
                    if columns is not None:
                        select_anthropometry_columns(first_row, columns)

        excluded_subject_ids = set(self.exclude_subject_ids)
        if self.hrtf_spec is not None:
            missing_hrtf_subject_ids = [
                subject_id
                for subject_id in self.dataset_subject_ids
                if subject_id not in excluded_subject_ids
                if subject_id not in self.hrtf_paths
            ]
            if len(missing_hrtf_subject_ids) > 0:
                preview = ", ".join(missing_hrtf_subject_ids[:5])
                suffix = "" if len(missing_hrtf_subject_ids) <= 5 else ", ..."
                warnings.warn(
                    f"{self.dataset_name}: {len(missing_hrtf_subject_ids)} subjects do not have a matching HRTF file under "
                    f"{self.root} and will be excluded ({preview}{suffix})",
                    stacklevel=2,
                )
            validated_hrtf_paths: dict[str, Path] = {}
            for subject_id, path in self.hrtf_paths.items():
                if not path.exists():
                    warnings.warn(
                        f"{self.dataset_name}: subject {subject_id} HRTF path is missing and will be excluded: {path}",
                        stacklevel=2,
                    )
                    continue
                try:
                    hrtf = load_hrtf(path)
                except Exception as exc:
                    warnings.warn(
                        f"{self.dataset_name}: subject {subject_id} HRTF file could not be loaded and will be excluded: "
                        f"{path} ({exc})",
                        stacklevel=2,
                    )
                    continue
                validated_hrtf_paths[subject_id] = path
                if self.cache_hrtf:
                    self.hrtf_cache[subject_id] = hrtf
            self.hrtf_paths = validated_hrtf_paths

        self.image_index: dict[tuple[str, int | None, str | None], list[str]] = {}
        if self.image_path is not None and self.dataset_image_spec is not None and self.image_align_by is not None:
            self.image_index = scan_media_paths(
                self.image_path,
                self.dataset_subject_ids,
                self.dataset_image_spec.extensions,
                self.image_align_by,
            )
        self.video_index: dict[tuple[str, int | None, str | None], list[str]] = {}
        if self.video_path is not None and self.dataset_video_spec is not None and self.video_align_by is not None:
            self.video_index = scan_media_paths(
                self.video_path,
                self.dataset_subject_ids,
                self.dataset_video_spec.extensions,
                self.video_align_by,
            )

        required_subject_sets: list[set[str]] = []
        if self.hrtf_spec is not None:
            required_subject_sets.append(set(self.hrtf_paths))
        if self.mesh_spec is not None:
            if self.dataset_mesh_spec is None:
                raise ValueError(f"{self.dataset_name} does not provide mesh data")
            required_subject_sets.append(set(self.mesh_paths))
        if self.anthropometry_spec is not None:
            if self.dataset_anthropometry_spec is None:
                raise ValueError(f"{self.dataset_name} does not provide anthropometry")
            required_subject_sets.append(set(self.anthropometry_rows))
        if self.image_spec is not None:
            required_subject_sets.append({key[0] for key in self.image_index})
        if self.video_spec is not None:
            required_subject_sets.append({key[0] for key in self.video_index})
        if len(required_subject_sets) == 0:
            raise ValueError("Could not resolve subjects for the selected specs")

        subject_ids = sort_subject_ids(set.intersection(*required_subject_sets))
        if len(subject_ids) == 0:
            available_counts = []
            if self.hrtf_spec is not None:
                available_counts.append(f"hrtf={len(self.hrtf_paths)}")
            if self.mesh_spec is not None:
                available_counts.append(f"mesh={len(self.mesh_paths)}")
            if self.anthropometry_spec is not None:
                available_counts.append(f"anthropometry={len(self.anthropometry_rows)}")
            if self.image_spec is not None:
                available_counts.append(f"image={len({key[0] for key in self.image_index})}")
            if self.video_spec is not None:
                available_counts.append(f"video={len({key[0] for key in self.video_index})}")
            selected_specs = ", ".join(sorted(set(self.input_names + self.target_names)))
            counts_text = ", ".join(available_counts)
            raise ValueError(
                "No subjects match the selected dataset configuration. "
                f"Selected specs: {selected_specs}. "
                f"Available subject counts by spec: {counts_text}"
            )
        if len(self.exclude_subject_ids) > 0:
            subject_ids = [
                subject_id for subject_id in subject_ids if subject_id not in excluded_subject_ids
            ]
            if len(subject_ids) == 0:
                raise ValueError("No subjects remain after applying exclude_subject_ids")
        subject_ids = split_subject_ids(subject_ids, split, split_ratio, split_seed)
        if len(subject_ids) == 0:
            raise ValueError(f"Split {split!r} produced an empty dataset")
        self.subject_ids = tuple(subject_ids)

        if self.hrtf_spec is not None:
            reference_subject_id = self.subject_ids[0]
            reference_hrtf = load_hrtf(self.hrtf_paths[reference_subject_id])
            if self.cache_hrtf:
                self.hrtf_cache[reference_subject_id] = reference_hrtf
            self.available_positions = np.asarray(
                reference_hrtf.Sources.get_positions(angle_unit="degrees"),
                dtype=float,
            )
            if reference_hrtf.TF.frequency_bins is not None:
                self.frequency_bins = np.asarray(reference_hrtf.TF.frequency_bins, dtype=float)
            self.selected_position_indices = resolve_positions_selection(
                self.hrtf_spec.positions,
                reference_hrtf,
            )
            self.selected_positions = np.asarray(
                self.available_positions[self.selected_position_indices],
                dtype=float,
            )
            spherical_positions = np.asarray(
                get_spherical_positions(reference_hrtf.Sources, angle_unit="degrees"),
                dtype=float,
            )
            self.available_azimuth_angles = np.unique(
                np.round(spherical_positions[:, 0], 2)
            )
            self.available_elevation_angles = np.unique(
                np.round(spherical_positions[:, 1], 2)
            )
            selected_spherical_positions = np.asarray(
                spherical_positions[self.selected_position_indices],
                dtype=float,
            )
            self.azimuth_angles = np.unique(
                np.round(selected_spherical_positions[:, 0], 2)
            )
            self.elevation_angles = np.unique(
                np.round(selected_spherical_positions[:, 1], 2)
            )
        else:
            self.selected_position_indices = []
            self.selected_positions = None
            self.available_azimuth_angles = None
            self.available_elevation_angles = None
            self.azimuth_angles = None
            self.elevation_angles = None

        self.rows = build_rows(
            subject_ids=self.subject_ids,
            index_by=self.index_by,
            position_indices=self.selected_position_indices,
            ears=self.selected_ears,
        )

    def get_subject_hrtf(self, subject_id: str | int):
        resolved_subject_id = resolve_dataset_subject_id(subject_id, self.subject_ids)
        if resolved_subject_id not in self.hrtf_paths:
            raise KeyError(
                f"Subject {subject_id!r} resolved to {resolved_subject_id!r} but does not have an available HRTF file"
            )
        path = self.hrtf_paths[resolved_subject_id]
        if not path.exists():
            warnings.warn(
                f"{self.dataset_name}: subject {resolved_subject_id} HRTF path is missing: {path}",
                stacklevel=2,
            )
            raise FileNotFoundError(
                f"HRTF path is missing for subject {resolved_subject_id}: {path}"
            )
        hrtf = self.hrtf_cache.get(resolved_subject_id)
        if hrtf is None:
            try:
                hrtf = load_hrtf(path)
            except Exception as exc:
                warnings.warn(
                    f"{self.dataset_name}: subject {resolved_subject_id} HRTF file could not be loaded: {path} ({exc})",
                    stacklevel=2,
                )
                raise
            if self.cache_hrtf:
                self.hrtf_cache[resolved_subject_id] = hrtf
        return hrtf

    def get_spec_value(
        self,
        spec: HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        if isinstance(spec, HRTFSpec):
            hrtf = self.get_subject_hrtf(subject_id)
            value = select_hrtf_signal(
                hrtf=hrtf,
                row=row,
                selected_position_indices=self.selected_position_indices,
                selected_ears=self.selected_ears,
                domain=str(spec.domain).strip().lower(),
                signal=str(spec.signal).strip().lower(),
            )
            if spec.transform is not None:
                value = spec.transform(value)
            return value
        if isinstance(spec, MeshSpec):
            value: object = str(self.mesh_paths[subject_id])
            if spec.transform is not None:
                value = spec.transform(value)
            return value
        if isinstance(spec, AnthropometrySpec):
            value: object = select_anthropometry_columns(
                self.anthropometry_rows[subject_id],
                normalize_columns(spec.columns),
            )
            if spec.transform is not None:
                value = spec.transform(value)
            return value
        if isinstance(spec, ImageSpec):
            if self.image_align_by is None:
                raise ValueError("image_align_by is not configured")
            image_key = build_media_key(
                subject_id,
                self.image_align_by,
                None if row["position_index"] is None else int(row["position_index"]),
                None if row["ear"] is None else str(row["ear"]),
            )
            if image_key not in self.image_index:
                raise ValueError(f"No image found for sample {image_key}")
            return apply_media_transform(
                self.image_index[image_key],
                spec.transform,
                image_size=spec.image_size,
            )
        if isinstance(spec, VideoSpec):
            if self.video_align_by is None:
                raise ValueError("video_align_by is not configured")
            video_key = build_media_key(
                subject_id,
                self.video_align_by,
                None if row["position_index"] is None else int(row["position_index"]),
                None if row["ear"] is None else str(row["ear"]),
            )
            if video_key not in self.video_index:
                raise ValueError(f"No video found for sample {video_key}")
            return apply_media_transform(
                self.video_index[video_key],
                spec.transform,
            )
        raise TypeError(f"Unsupported dataset spec: {type(spec)!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | str) -> dict[str, object]:
        if isinstance(index, str):
            subject_id = resolve_dataset_subject_id(index, self.subject_ids)
            matches = [
                row_index
                for row_index, row in enumerate(self.rows)
                if row["subject_id"] == subject_id
            ]
            if len(matches) == 0:
                raise KeyError(f"Subject {index!r} is not available in this dataset instance")
            if len(matches) > 1:
                raise KeyError(
                    f"Subject {index!r} maps to multiple samples under index_by={self.index_by}"
                )
            row = self.rows[matches[0]]
        else:
            row = self.rows[int(index)]

        subject_id = str(row["subject_id"])
        inputs: dict[str, object] = {}
        for spec in self.input_specs:
            inputs[get_spec_name(spec)] = self.get_spec_value(
                spec,
                subject_id,
                row,
            )

        if self.position_encoding == "one-hot" and row["position_offset"] is not None:
            position_encoding = np.zeros(len(self.selected_position_indices), dtype=float)
            position_encoding[int(row["position_offset"])] = 1.0
            inputs["position"] = position_encoding
        if self.ear_encoding == "one-hot" and row["ear_offset"] is not None:
            ear_encoding = np.zeros(len(self.selected_ears), dtype=float)
            ear_encoding[int(row["ear_offset"])] = 1.0
            inputs["ear"] = ear_encoding

        sample: dict[str, object] = {
            "inputs": inputs,
            "subject_id": subject_id,
        }
        if row["position_index"] is not None and self.available_positions is not None:
            sample["position_index"] = int(row["position_index"])
            sample["position"] = np.asarray(
                self.available_positions[int(row["position_index"])],
                dtype=float,
            )
        if row["ear"] is not None:
            sample["ear"] = str(row["ear"])

        if len(self.target_names) > 0:
            target_values: dict[str, object] = {}
            for spec in self.target_specs:
                target_values[get_spec_name(spec)] = self.get_spec_value(
                    spec,
                    subject_id,
                    row,
                )
            if len(target_values) == 1:
                sample["target"] = next(iter(target_values.values()))
            else:
                sample["target"] = target_values

        return sample
