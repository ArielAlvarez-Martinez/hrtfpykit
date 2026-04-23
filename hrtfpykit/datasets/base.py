from pathlib import Path
import re
from typing import TypeVar
import warnings

import numpy as np

from .anthropometry import (
    load_anthropometry_rows,
    select_anthropometry_value,
)
from .config import DatasetConfig
from ..hrtf.planes import get_frontal_plane, get_horizontal_plane, get_median_plane
from ..hrtf.dsp import imag, magnitude, magnitude_db, phase, real
from ..hrtf.coordinates import get_spherical_positions
from ..main import load_hrtf
from .image import apply_image_transform, build_image_key, scan_image_paths
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
    normalize_specs,
)
from .video import apply_video_transform, build_video_key, scan_video_paths


SUPPORTED_HRTF_DOMAINS = (
    "time",
    "frequency",
)


SUPPORTED_HRTF_SIGNALS = (
    "ir",
    "tf_complex",
    "tf_real",
    "tf_imag",
    "tf_magnitude",
    "tf_magnitude_db",
    "tf_phase",
)


SpecType = TypeVar("SpecType")


class BaseDataset:
    config: DatasetConfig | None = None

    @staticmethod
    def normalize_subject_id(value: str) -> str:
        return str(value).strip().lower()

    @classmethod
    def resolve_dataset_subject_id(
        cls,
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
        subject_index_match = re.fullmatch(r"subject[_-]?(\d+)", text.strip().lower())
        if subject_index_match is not None:
            index = int(subject_index_match.group(1))
            if index < 1 or index > len(subject_ids):
                raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
            return subject_ids[index - 1]
        if text.isdigit():
            index = int(text)
            if index < 1 or index > len(subject_ids):
                raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
            return subject_ids[index - 1]
        normalized = cls.normalize_subject_id(text)
        if normalized.lower() in subject_map:
            return subject_map[normalized.lower()]
        raise ValueError(f"Unknown subject reference {value!r}")

    @staticmethod
    def sort_subject_ids(subject_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        def subject_sort_key(value: str) -> tuple[int, str]:
            match = re.search(r"(\d+)$", str(value))
            if match is None:
                return (0, str(value).lower())
            return (int(match.group(1)), str(value).lower())

        return sorted(subject_ids, key=subject_sort_key)

    @staticmethod
    def preview_values(values: tuple[str, ...] | list[str], limit: int = 5) -> str:
        if len(values) == 0:
            return "none"
        preview = ", ".join(str(value) for value in values[:limit])
        if len(values) > limit:
            preview = f"{preview}, ..."
        return preview

    @classmethod
    def format_resource_summary(cls, resource_summary: dict[str, dict[str, object]]) -> str:
        if len(resource_summary) == 0:
            return "Resource summary: none"
        lines = ["Resource summary:"]
        for resource_name, summary in resource_summary.items():
            parts = [str(resource_name)]
            for key in (
                "pattern",
                "path",
                "variant",
                "extensions",
                "checked",
                "found",
                "valid",
                "invalid",
                "missing",
                "subjects",
                "rows",
            ):
                if key in summary:
                    parts.append(f"{key}={summary[key]!r}")
            if "missing_subject_ids" in summary:
                missing_subject_ids = tuple(summary["missing_subject_ids"])
                if len(missing_subject_ids) > 0:
                    parts.append(
                        f"missing_subject_ids={cls.preview_values(missing_subject_ids)}"
                    )
            if "invalid_subject_ids" in summary:
                invalid_subject_ids = tuple(summary["invalid_subject_ids"])
                if len(invalid_subject_ids) > 0:
                    parts.append(
                        f"invalid_subject_ids={cls.preview_values(invalid_subject_ids)}"
                    )
            lines.append("  " + parts[0] + ": " + ", ".join(parts[1:]))
        return "\n".join(lines)

    @classmethod
    def build_value_signature(cls, value: object) -> object:
        if isinstance(value, Path):
            return ("path", str(value.expanduser()))
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            return ("array", tuple(array.shape), cls.build_value_signature(array.tolist()))
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key).strip().lower(), cls.build_value_signature(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls.build_value_signature(item) for item in value)
        return value

    @staticmethod
    def select_primary_spec(
        input_specs: tuple[SpecType, ...],
        all_specs: tuple[SpecType, ...],
        prefer_path: bool = False,
    ) -> SpecType | None:
        if prefer_path:
            preferred = next(
                (spec for spec in input_specs if getattr(spec, "path", None) is not None),
                None,
            )
            if preferred is not None:
                return preferred
            preferred = next(
                (spec for spec in all_specs if getattr(spec, "path", None) is not None),
                None,
            )
            if preferred is not None:
                return preferred
        if len(input_specs) > 0:
            return input_specs[0]
        if len(all_specs) > 0:
            return all_specs[0]
        return None

    @staticmethod
    def resolve_optional_path(
        path: str | Path | None,
        root: Path,
    ) -> Path | None:
        if path is None:
            return None
        resolved_path = Path(path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = root / resolved_path
        return resolved_path

    @staticmethod
    def validate_aligned_asset_spec(
        dataset_name: str,
        asset_name: str,
        spec,
        supported_align_by: tuple[tuple[str, ...], ...] | None,
        asset_path: Path | None,
        asset_align_by: tuple[str, ...] | None,
        index_by: tuple[str, ...],
    ) -> None:
        if spec is None:
            return
        if supported_align_by is None:
            raise ValueError(f"{dataset_name} does not define a {asset_name} pipeline")
        if asset_path is None:
            raise ValueError(f"{type(spec).__name__}.path is required when {asset_name} is selected")
        if asset_align_by not in supported_align_by:
            raise ValueError(f"{asset_name} align_by must be one of {supported_align_by}")
        if asset_align_by is not None and "position" in asset_align_by and "position" not in index_by:
            raise ValueError(
                f"{asset_name} align_by including 'position' requires index_by to include 'position'"
            )
        if asset_align_by is not None and "ear" in asset_align_by and "ear" not in index_by:
            raise ValueError(
                f"{asset_name} align_by including 'ear' requires index_by to include 'ear'"
            )

    @staticmethod
    def resolve_positions_selection(
        positions: str | tuple[int, ...] | list[int] | np.ndarray,
        plane: str | tuple[object, ...] | dict[str, object] | None,
        hrtf,
    ) -> list[int]:
        position_count = int(hrtf.Sources.get_positions().shape[0])
        if plane is None:
            return normalize_positions(
                positions,
                position_count,
            )
        if not isinstance(positions, str) or str(positions).strip().lower() != "all":
            raise ValueError("plane selection cannot be combined with custom positions")

        if isinstance(plane, str):
            plane_key = str(plane).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = default_angle
            angle_unit = "degrees"
        elif isinstance(plane, tuple):
            if len(plane) not in {2, 3} or not isinstance(plane[0], str):
                raise ValueError(
                    "Plane selection must be ('horizontal'|'median'|'frontal', angle[, angle_unit])"
                )
            plane_key = str(plane[0]).strip().lower()
            angle = plane[1]
            angle_unit = "degrees" if len(plane) == 2 else str(plane[2]).strip().lower()
        else:
            plane_key = str(plane.get("plane")).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = plane.get("angle", plane.get("plane_angle", default_angle))
            angle_unit = str(plane.get("angle_unit", "degrees")).strip().lower()

        if plane_key not in {"horizontal", "median", "frontal"}:
            raise ValueError("plane must be horizontal, median, or frontal")

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

    @staticmethod
    def select_hrtf_value(
        hrtf,
        row: dict[str, str | int | None],
        selected_position_indices: list[int],
        selected_ears: list[tuple[str, int]],
        spec: HRTFSpec,
    ) -> np.ndarray:
        spec_index_by = normalize_index_by(spec.index_by)
        domain = str(spec.domain).strip().lower()
        signal = str(spec.signal).strip().lower()
        if domain == "time":
            if signal != "ir":
                raise ValueError("HRTFSpec with domain='time' requires signal='ir'")
            values = np.asarray(hrtf.IR.values, dtype=float)
        elif domain == "frequency":
            if signal == "ir":
                raise ValueError("HRTFSpec with domain='frequency' cannot use signal='ir'")
            tf_values = np.asarray(hrtf.TF.values)
            if signal == "tf_complex":
                values = tf_values
            elif signal == "tf_real":
                values = real(tf_values)
            elif signal == "tf_imag":
                values = imag(tf_values)
            elif signal == "tf_magnitude":
                values = magnitude(tf_values)
            elif signal == "tf_magnitude_db":
                values = magnitude_db(tf_values)
            elif signal == "tf_phase":
                values = phase(tf_values)
            else:
                raise ValueError(f"Unsupported signal {signal!r}")
        else:
            raise ValueError(f"Unsupported domain {domain!r}")

        if "position" not in spec_index_by:
            if len(selected_position_indices) != values.shape[0]:
                values = np.take(values, selected_position_indices, axis=0)
        else:
            if row["position_index"] is None:
                raise ValueError(
                    f"HRTFSpec(index_by={spec.index_by!r}) requires position-resolved rows"
                )
            values = np.asarray(values[int(row["position_index"])])

        if "ear" not in spec_index_by:
            if len(selected_ears) == 1:
                values = np.asarray(values[int(selected_ears[0][1])])
        else:
            if row["ear_index"] is None:
                raise ValueError(
                    f"HRTFSpec(index_by={spec.index_by!r}) requires ear-resolved rows"
                )
            values = np.asarray(values[int(row["ear_index"])])

        return np.asarray(values)

    @staticmethod
    def is_hrtf_object(value: object) -> bool:
        return (
            hasattr(value, "IR")
            and hasattr(value, "TF")
            and hasattr(value, "Sources")
            and hasattr(value, "transform")
        )

    @staticmethod
    def is_explicit_hrtf_transform(transform) -> bool:
        return bool(getattr(transform, "__hrtf_transform__", False))

    @staticmethod
    def is_raw_hrtf_transform_method(transform) -> bool:
        transform_module = str(getattr(transform, "__module__", ""))
        transform_qualname = str(getattr(transform, "__qualname__", ""))
        return transform_module.endswith(".transforms") and transform_qualname.startswith("Transform.")

    @classmethod
    def apply_hrtf_spec_transform(
        cls,
        hrtf,
        transform,
    ):
        transformed_hrtf = transform(hrtf)
        if not cls.is_hrtf_object(transformed_hrtf):
            raise ValueError(
                "HRTFTransform callables used in HRTFSpec.transform must return an HRTF object"
            )
        return transformed_hrtf

    def __init__(
        self,
        root: str | Path,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
        | None = None,
        target: HRTFSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
        | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
    ) -> None:
        if type(self).config is None:
            raise ValueError(f"{type(self).__name__} must define a dataset config")
        self.config = type(self).config
        self.name = str(self.config.name)
        self.root = Path(root).expanduser()
        if exclude_subject_ids is None:
            self.exclude_subject_ids = tuple()
        elif isinstance(exclude_subject_ids, (str, int)):
            self.exclude_subject_ids = (
                self.resolve_dataset_subject_id(exclude_subject_ids, tuple(self.config.subject_ids)),
            )
        else:
            self.exclude_subject_ids = tuple(
                dict.fromkeys(
                    self.resolve_dataset_subject_id(subject_id, tuple(self.config.subject_ids))
                    for subject_id in exclude_subject_ids
                )
            )

        self._input_specs = normalize_specs(inputs)
        self._target_specs = normalize_specs(target)
        input_names = tuple(get_spec_name(spec) for spec in self._input_specs)
        target_names = tuple(get_spec_name(spec) for spec in self._target_specs)
        all_specs = self._input_specs + self._target_specs
        self.resource_summary: dict[str, dict[str, object]] = {}
        hrtf_specs = tuple(spec for spec in all_specs if isinstance(spec, HRTFSpec))
        mesh_specs = tuple(spec for spec in all_specs if isinstance(spec, MeshSpec))
        anthropometry_specs = tuple(
            spec for spec in all_specs if isinstance(spec, AnthropometrySpec)
        )
        image_specs = tuple(spec for spec in all_specs if isinstance(spec, ImageSpec))
        video_specs = tuple(spec for spec in all_specs if isinstance(spec, VideoSpec))

        input_hrtf_specs = tuple(
            spec for spec in self._input_specs if isinstance(spec, HRTFSpec)
        )
        input_mesh_specs = tuple(
            spec for spec in self._input_specs if isinstance(spec, MeshSpec)
        )
        input_anthropometry_specs = tuple(
            spec for spec in self._input_specs if isinstance(spec, AnthropometrySpec)
        )
        input_image_specs = tuple(
            spec for spec in self._input_specs if isinstance(spec, ImageSpec)
        )
        input_video_specs = tuple(
            spec for spec in self._input_specs if isinstance(spec, VideoSpec)
        )

        if len(hrtf_specs) > 1:
            if len({self.build_value_signature(spec.positions) for spec in hrtf_specs}) > 1:
                raise ValueError(
                    "All HRTFSpec objects must use the same positions when hrtf is used in both inputs and target"
                )
            if len({self.build_value_signature(spec.plane) for spec in hrtf_specs}) > 1:
                raise ValueError(
                    "All HRTFSpec objects must use the same plane when hrtf is used in both inputs and target"
                )
            if len({self.build_value_signature(spec.ears) for spec in hrtf_specs}) > 1:
                raise ValueError(
                    "All HRTFSpec objects must use the same ears when hrtf is used in both inputs and target"
                )

        if len(image_specs) > 1:
            image_path_signatures = {
                self.build_value_signature(Path(spec.path))
                for spec in image_specs
                if spec.path is not None
            }
            if len(image_path_signatures) > 1:
                raise ValueError(
                    "All ImageSpec objects must use the same path when image is used in both inputs and target"
                )
            if len({normalize_index_by(spec.align_by) for spec in image_specs}) > 1:
                raise ValueError(
                    "All ImageSpec objects must use the same align_by when image is used in both inputs and target"
                )

        if len(video_specs) > 1:
            video_path_signatures = {
                self.build_value_signature(Path(spec.path))
                for spec in video_specs
                if spec.path is not None
            }
            if len(video_path_signatures) > 1:
                raise ValueError(
                    "All VideoSpec objects must use the same path when video is used in both inputs and target"
                )
            if len({normalize_index_by(spec.align_by) for spec in video_specs}) > 1:
                raise ValueError(
                    "All VideoSpec objects must use the same align_by when video is used in both inputs and target"
                )

        if len(anthropometry_specs) > 1:
            anthropometry_path_signatures = set()
            for spec in anthropometry_specs:
                if spec.path is None:
                    continue
                path = Path(spec.path).expanduser()
                if not path.is_absolute():
                    path = self.root / path
                anthropometry_path_signatures.add(self.build_value_signature(path))
            if len(anthropometry_path_signatures) > 1:
                raise ValueError(
                    "All AnthropometrySpec objects must use the same path when anthropometry is used in both inputs and target"
                )

        primary_hrtf_spec = self.select_primary_spec(input_hrtf_specs, hrtf_specs)
        primary_mesh_spec = self.select_primary_spec(input_mesh_specs, mesh_specs)
        primary_anthropometry_spec = self.select_primary_spec(
            input_anthropometry_specs,
            anthropometry_specs,
            prefer_path=True,
        )
        primary_image_spec = self.select_primary_spec(
            input_image_specs,
            image_specs,
            prefer_path=True,
        )
        primary_video_spec = self.select_primary_spec(
            input_video_specs,
            video_specs,
            prefer_path=True,
        )

        self.index_by = ("subject",)
        if primary_hrtf_spec is not None:
            include_position = any(
                "position" in normalize_index_by(spec.index_by)
                for spec in hrtf_specs
            )
            include_ear = any(
                "ear" in normalize_index_by(spec.index_by)
                for spec in hrtf_specs
            )
            index_by_values = ["subject"]
            if include_position:
                index_by_values.append("position")
            if include_ear:
                index_by_values.append("ear")
            self.index_by = tuple(index_by_values)

        self._cache_hrtf = True if len(hrtf_specs) == 0 else any(bool(spec.cache) for spec in hrtf_specs)
        self._hrtf_cache: dict[str, object] = {}
        self._transformed_hrtf_cache: dict[tuple[str, int], object] = {}
        self.sample_rate: float | None = None
        self._image_path = self.resolve_optional_path(
            None if primary_image_spec is None else primary_image_spec.path,
            self.root,
        )
        self._video_path = self.resolve_optional_path(
            None if primary_video_spec is None else primary_video_spec.path,
            self.root,
        )
        anthropometry_path = self.resolve_optional_path(
            None if primary_anthropometry_spec is None else primary_anthropometry_spec.path,
            self.root,
        )
        self._image_align_by = (
            None if primary_image_spec is None else normalize_index_by(primary_image_spec.align_by)
        )
        self._video_align_by = (
            None if primary_video_spec is None else normalize_index_by(primary_video_spec.align_by)
        )
        self._selected_ears = (
            [] if primary_hrtf_spec is None else normalize_ears(primary_hrtf_spec.ears)
        )
        self._position_encoding = (
            "none"
            if primary_hrtf_spec is None
            else str(primary_hrtf_spec.position_encoding).strip().lower()
        )
        self._ear_encoding = (
            "none"
            if primary_hrtf_spec is None
            else str(primary_hrtf_spec.ear_encoding).strip().lower()
        )
        self._selected_position_indices: list[int] = []
        preset_variant = getattr(self, "variant", None)
        self.variant = None
        if len(hrtf_specs) > 0:
            if self.config.hrtf is not None:
                self.variant = str(self.config.hrtf.default_variant).strip().lower()
            if preset_variant is not None:
                self.variant = preset_variant

        if self._position_encoding not in {"none", "one-hot"}:
            raise ValueError("position_encoding must be 'none' or 'one-hot'")
        if self._ear_encoding not in {"none", "one-hot"}:
            raise ValueError("ear_encoding must be 'none' or 'one-hot'")
        if self._position_encoding == "one-hot" and "position" in input_names:
            raise ValueError(
                "Input spec name 'position' conflicts with position_encoding='one-hot'"
            )
        if self._ear_encoding == "one-hot" and "ear" in input_names:
            raise ValueError(
                "Input spec name 'ear' conflicts with ear_encoding='one-hot'"
            )
        if self._position_encoding != "none" and "position" not in self.index_by:
            raise ValueError("position_encoding requires index_by to include 'position'")
        if self._ear_encoding != "none" and "ear" not in self.index_by:
            raise ValueError("ear_encoding requires index_by to include 'ear'")

        image_supported_align_by = (
            None if self.config.image is None else tuple(self.config.image.supported_align_by)
        )
        video_supported_align_by = (
            None if self.config.video is None else tuple(self.config.video.supported_align_by)
        )

        self.validate_aligned_asset_spec(
            dataset_name=self.name,
            asset_name="image",
            spec=primary_image_spec,
            supported_align_by=image_supported_align_by,
            asset_path=self._image_path,
            asset_align_by=self._image_align_by,
            index_by=self.index_by,
        )
        self.validate_aligned_asset_spec(
            dataset_name=self.name,
            asset_name="video",
            spec=primary_video_spec,
            supported_align_by=video_supported_align_by,
            asset_path=self._video_path,
            asset_align_by=self._video_align_by,
            index_by=self.index_by,
        )

        if ("position" in self.index_by or "ear" in self.index_by) and primary_hrtf_spec is None:
            raise ValueError(
                "HRTFSpec.index_by including 'position' or 'ear' currently requires hrtf in inputs or target"
            )

        if primary_hrtf_spec is not None:
            if self.config.hrtf is None:
                raise ValueError(f"{self.name} does not provide hrtf data")
            if self.variant is None:
                raise ValueError("variant could not be resolved")
            if self.variant not in self.config.hrtf.variants:
                raise ValueError(
                    f"Unsupported variant {self.variant!r}. "
                    f"Expected one of {self.config.hrtf.variants}"
                )
            for spec in hrtf_specs:
                domain = str(spec.domain).strip().lower()
                signal = str(spec.signal).strip().lower()
                if domain not in SUPPORTED_HRTF_DOMAINS:
                    raise ValueError(
                        f"Unsupported domain {spec.domain!r}. Expected one of {SUPPORTED_HRTF_DOMAINS}"
                    )
                if signal not in SUPPORTED_HRTF_SIGNALS:
                    raise ValueError(
                        f"Unsupported signal {spec.signal!r}. Expected one of {SUPPORTED_HRTF_SIGNALS}"
                    )
                if domain == "time" and signal != "ir":
                    raise ValueError("HRTFSpec with domain='time' requires signal='ir'")
                if domain == "frequency" and signal == "ir":
                    raise ValueError("HRTFSpec with domain='frequency' cannot use signal='ir'")
                if isinstance(spec.positions, str):
                    if str(spec.positions).strip().lower() != "all":
                        raise ValueError(
                            "HRTFSpec.positions must be 'all' or a sequence of position indices"
                        )
                normalize_index_by(spec.index_by)
                if spec.plane is not None:
                    if not isinstance(spec.positions, str) or str(spec.positions).strip().lower() != "all":
                        raise ValueError(
                            "HRTFSpec.plane cannot be combined with custom positions"
                        )
                    if isinstance(spec.plane, str):
                        plane_key = str(spec.plane).strip().lower()
                        if plane_key not in {"horizontal", "median", "frontal"}:
                            raise ValueError(
                                "HRTFSpec.plane must be horizontal, median, frontal, "
                                "a tuple-based plane selection, or a dict with a 'plane' key"
                            )
                    elif isinstance(spec.plane, tuple):
                        if len(spec.plane) not in {2, 3} or not isinstance(spec.plane[0], str):
                            raise ValueError(
                                "Tuple plane selection must be ('horizontal'|'median'|'frontal', angle[, angle_unit])"
                            )
                        plane_key = str(spec.plane[0]).strip().lower()
                        if plane_key not in {"horizontal", "median", "frontal"}:
                            raise ValueError("Tuple plane selection must use horizontal, median, or frontal")
                    elif isinstance(spec.plane, dict):
                        plane_name = spec.plane.get("plane")
                        if plane_name is None:
                            raise ValueError("Dict plane selection must include a 'plane' key")
                        plane_key = str(plane_name).strip().lower()
                        if plane_key not in {"horizontal", "median", "frontal"}:
                            raise ValueError("Dict plane selection must use horizontal, median, or frontal")
                    else:
                        raise ValueError(
                            "HRTFSpec.plane must be None, a string, a tuple, or a dict"
                        )

        if primary_mesh_spec is not None and self.config.mesh is None:
            raise ValueError(f"{self.name} does not provide mesh data")
        if (
            primary_anthropometry_spec is not None
            and anthropometry_path is None
            and self.config.anthropometry is None
        ):
            raise ValueError(f"{self.name} does not provide anthropometry")
        if anthropometry_path is not None:
            if not anthropometry_path.exists():
                raise ValueError(f"AnthropometrySpec.path does not exist: {anthropometry_path}")
            if not anthropometry_path.is_file():
                raise ValueError(f"AnthropometrySpec.path is not a file: {anthropometry_path}")

        excluded_subject_ids = set(self.exclude_subject_ids)
        included_subject_ids = tuple(
            subject_id
            for subject_id in self.config.subject_ids
            if subject_id not in excluded_subject_ids
        )
        subject_numbers = {
            subject_id: index
            for index, subject_id in enumerate(tuple(self.config.subject_ids), start=1)
        }
        self._hrtf_paths: dict[str, Path] = {}
        if self.config.hrtf is not None and primary_hrtf_spec is not None:
            hrtf_subject_ids = (
                tuple(self.config.subject_ids)
                if self.config.hrtf.subject_ids is None
                else tuple(self.config.hrtf.subject_ids)
            )
            checked_hrtf_subject_ids = tuple(
                subject_id for subject_id in hrtf_subject_ids if subject_id not in excluded_subject_ids
            )
            for subject_id in checked_hrtf_subject_ids:
                relative_path = self.config.hrtf.path_pattern.format(
                    subject_id=subject_id,
                    variant=self.variant,
                )
                candidate = (self.root / relative_path).expanduser()
                if candidate.is_file():
                    self._hrtf_paths[subject_id] = candidate
            missing_hrtf_subject_ids = tuple(
                subject_id
                for subject_id in checked_hrtf_subject_ids
                if subject_id not in self._hrtf_paths
            )
            self.resource_summary["hrtf"] = {
                "pattern": self.config.hrtf.path_pattern,
                "variant": self.variant,
                "checked": len(checked_hrtf_subject_ids),
                "found": len(
                    [
                        subject_id
                        for subject_id in checked_hrtf_subject_ids
                        if subject_id in self._hrtf_paths
                    ]
                ),
                "missing": len(missing_hrtf_subject_ids),
                "missing_subject_ids": missing_hrtf_subject_ids,
            }

        self._mesh_paths: dict[str, Path] = {}
        if self.config.mesh is not None and primary_mesh_spec is not None:
            mesh_subject_ids = (
                tuple(self.config.subject_ids)
                if self.config.mesh.subject_ids is None
                else tuple(self.config.mesh.subject_ids)
            )
            checked_mesh_subject_ids = tuple(
                subject_id for subject_id in mesh_subject_ids if subject_id not in excluded_subject_ids
            )
            for subject_id in checked_mesh_subject_ids:
                for extension in self.config.mesh.extensions:
                    relative_path = self.config.mesh.path_pattern.format(
                        subject_id=subject_id,
                        extension=extension,
                    )
                    candidate = (self.root / relative_path).expanduser()
                    if candidate.is_file():
                        self._mesh_paths[subject_id] = candidate
                        break
            missing_mesh_subject_ids = tuple(
                subject_id
                for subject_id in checked_mesh_subject_ids
                if subject_id not in self._mesh_paths
            )
            self.resource_summary["mesh"] = {
                "pattern": self.config.mesh.path_pattern,
                "extensions": tuple(self.config.mesh.extensions),
                "checked": len(checked_mesh_subject_ids),
                "found": len(
                    [
                        subject_id
                        for subject_id in checked_mesh_subject_ids
                        if subject_id in self._mesh_paths
                    ]
                ),
                "missing": len(missing_mesh_subject_ids),
                "missing_subject_ids": missing_mesh_subject_ids,
            }

        if anthropometry_path is None and self.config.anthropometry is not None:
            candidate = (self.root / self.config.anthropometry.path).expanduser()
            if candidate.is_file():
                anthropometry_path = candidate
        if primary_anthropometry_spec is not None:
            self.resource_summary["anthropometry"] = {
                "path": None if anthropometry_path is None else str(anthropometry_path),
                "found": anthropometry_path is not None and anthropometry_path.is_file(),
            }

        self._anthropometry_rows: dict[str, dict[str, float | str | None]] = {}
        if primary_anthropometry_spec is not None and anthropometry_path is not None:
            subject_column_candidates = (
                (
                    "subject_id",
                    "subject",
                    "id",
                    "participant",
                    "pp",
                )
                if self.config.anthropometry is None
                else tuple(self.config.anthropometry.subject_column_candidates)
            )
            self._anthropometry_rows = load_anthropometry_rows(
                anthropometry_path,
                subject_column_candidates,
                tuple(self.config.subject_ids),
                self.resolve_dataset_subject_id,
            )
            if len(self._anthropometry_rows) > 0:
                first_subject_id = next(iter(self._anthropometry_rows))
                for spec in anthropometry_specs:
                    self.get_anthropometry_value(spec, first_subject_id)
            self.resource_summary["anthropometry"]["subjects"] = len(self._anthropometry_rows)
            self.resource_summary["anthropometry"]["rows"] = len(self._anthropometry_rows)

        if primary_hrtf_spec is not None:
            missing_hrtf_subject_ids = list(
                self.resource_summary.get("hrtf", {}).get("missing_subject_ids", tuple())
            )
            if len(missing_hrtf_subject_ids) > 0:
                preview = ", ".join(missing_hrtf_subject_ids[:5])
                suffix = "" if len(missing_hrtf_subject_ids) <= 5 else ", ..."
                warnings.warn(
                    f"{self.name}: {len(missing_hrtf_subject_ids)} subjects do not have a matching HRTF file under "
                    f"{self.root} and will be excluded ({preview}{suffix})",
                    stacklevel=2,
                )
            validated_hrtf_paths: dict[str, Path] = {}
            resolved_sample_rate: float | None = None
            for subject_id, path in self._hrtf_paths.items():
                if not path.exists():
                    warnings.warn(
                        f"{self.name}: subject {subject_id} HRTF path is missing and will be excluded: {path}",
                        stacklevel=2,
                    )
                    continue
                try:
                    hrtf = load_hrtf(path)
                except Exception as exc:
                    warnings.warn(
                        f"{self.name}: subject {subject_id} HRTF file could not be loaded and will be excluded: "
                        f"{path} ({exc})",
                        stacklevel=2,
                    )
                    continue
                current_sample_rate = (
                    None if hrtf.IR.sample_rate is None else float(hrtf.IR.sample_rate)
                )
                if resolved_sample_rate is None:
                    resolved_sample_rate = current_sample_rate
                elif current_sample_rate != resolved_sample_rate:
                    raise ValueError(
                        f"{self.name} requires a consistent sample_rate across loaded HRTFs, "
                        f"but subject {subject_id!r} has sample_rate={current_sample_rate} "
                        f"and previous subjects use sample_rate={resolved_sample_rate}"
                    )
                validated_hrtf_paths[subject_id] = path
                if self._cache_hrtf:
                    self._hrtf_cache[subject_id] = hrtf
            invalid_hrtf_subject_ids = tuple(
                subject_id
                for subject_id in self._hrtf_paths
                if subject_id not in validated_hrtf_paths
            )
            self._hrtf_paths = validated_hrtf_paths
            self.resource_summary["hrtf"]["valid"] = len(self._hrtf_paths)
            self.resource_summary["hrtf"]["invalid"] = len(invalid_hrtf_subject_ids)
            self.resource_summary["hrtf"]["invalid_subject_ids"] = invalid_hrtf_subject_ids
            self.sample_rate = resolved_sample_rate
        if primary_mesh_spec is not None:
            missing_mesh_subject_ids = tuple(
                self.resource_summary.get("mesh", {}).get("missing_subject_ids", tuple())
            )
            if len(missing_mesh_subject_ids) > 0:
                preview = self.preview_values(missing_mesh_subject_ids)
                warnings.warn(
                    f"{self.name}: {len(missing_mesh_subject_ids)} subjects do not have a matching mesh file under "
                    f"{self.root} and will be excluded when mesh is required ({preview})",
                    stacklevel=2,
                )

        self._image_index: dict[tuple[str, int | None, str | None], list[str]] = {}
        self._image_counts: dict[str, int] = {}
        if self._image_path is not None and image_supported_align_by is not None and self._image_align_by is not None:
            self._image_index, self._image_counts, missing_image_subject_ids = scan_image_paths(
                self._image_path,
                included_subject_ids,
                subject_numbers,
                tuple(self.config.image.extensions),
                self._image_align_by,
            )
        else:
            missing_image_subject_ids = tuple()
        if primary_image_spec is not None:
            self.resource_summary["image"] = {
                "path": None if self._image_path is None else str(self._image_path),
                "found": len({key[0] for key in self._image_index}),
                "subjects": len({key[0] for key in self._image_index}),
                "missing": len(missing_image_subject_ids),
                "missing_subject_ids": tuple(missing_image_subject_ids),
            }
            if len(missing_image_subject_ids) > 0:
                raise ValueError(
                    f"{self.name} image path is incompatible with the selected dataset subjects. "
                    f"Missing subject folders under {self._image_path}: "
                    f"{self.preview_values(tuple(missing_image_subject_ids))}"
                )
            distinct_image_counts = set(self._image_counts.values())
            if len(distinct_image_counts) > 1:
                warnings.warn(
                    f"{self.name}: subjects do not all have the same number of images under {self._image_path} "
                    f"({', '.join(f'{subject_id}={count}' for subject_id, count in sorted(self._image_counts.items())[:5])}"
                    f"{'' if len(self._image_counts) <= 5 else ', ...'})",
                    stacklevel=2,
                )
        self._video_index: dict[tuple[str, int | None, str | None], list[str]] = {}
        if self._video_path is not None and video_supported_align_by is not None and self._video_align_by is not None:
            self._video_index = scan_video_paths(
                self._video_path,
                tuple(self.config.subject_ids),
                tuple(self.config.video.extensions),
                self._video_align_by,
            )
        if primary_video_spec is not None:
            self.resource_summary["video"] = {
                "path": None if self._video_path is None else str(self._video_path),
                "found": len({key[0] for key in self._video_index}),
                "subjects": len({key[0] for key in self._video_index}),
            }

        required_subject_sets: list[set[str]] = []
        if primary_hrtf_spec is not None:
            required_subject_sets.append(set(self._hrtf_paths))
        if primary_mesh_spec is not None:
            required_subject_sets.append(set(self._mesh_paths))
        if primary_anthropometry_spec is not None:
            required_subject_sets.append(set(self._anthropometry_rows))
        if primary_image_spec is not None:
            required_subject_sets.append({key[0] for key in self._image_index})
        if primary_video_spec is not None:
            required_subject_sets.append({key[0] for key in self._video_index})
        if len(required_subject_sets) == 0:
            subject_ids = self.sort_subject_ids(
                [
                    subject_id
                    for subject_id in included_subject_ids
                ]
            )
        else:
            subject_ids = self.sort_subject_ids(set.intersection(*required_subject_sets))
        if len(subject_ids) == 0 and len(required_subject_sets) > 0:
            available_counts = []
            if len(hrtf_specs) > 0:
                available_counts.append(f"hrtf={len(self._hrtf_paths)}")
            if primary_mesh_spec is not None:
                available_counts.append(f"mesh={len(self._mesh_paths)}")
            if primary_anthropometry_spec is not None:
                available_counts.append(f"anthropometry={len(self._anthropometry_rows)}")
            if primary_image_spec is not None:
                available_counts.append(f"image={len({key[0] for key in self._image_index})}")
            if primary_video_spec is not None:
                available_counts.append(f"video={len({key[0] for key in self._video_index})}")
            selected_specs = ", ".join(sorted(set(input_names + target_names)))
            counts_text = ", ".join(available_counts)
            raise ValueError(
                "No subjects match the selected dataset configuration. "
                f"Selected specs: {selected_specs}. "
                f"Available subject counts by spec: {counts_text}. "
                f"Root: {self.root}\n"
                f"{self.format_resource_summary(self.resource_summary)}"
            )
        if len(self.exclude_subject_ids) > 0 and len(required_subject_sets) > 0:
            subject_ids = [
                subject_id for subject_id in subject_ids if subject_id not in excluded_subject_ids
            ]
            if len(subject_ids) == 0:
                raise ValueError("No subjects remain after applying exclude_subject_ids")
        self.available_subject_ids = tuple(subject_ids)
        split_subjects = split_subject_ids(subject_ids, split, split_ratio, split_seed)
        if len(split_subjects) == 0:
            raise ValueError(f"Split {split!r} produced an empty dataset")
        self.subject_ids = tuple(split_subjects)

        self.available_positions: np.ndarray | None = None
        self.selected_positions: np.ndarray | None = None
        self.available_azimuth_angles: np.ndarray | None = None
        self.available_elevation_angles: np.ndarray | None = None
        self.azimuth_angles: np.ndarray | None = None
        self.elevation_angles: np.ndarray | None = None
        self.frequency_bins: np.ndarray | None = None

        if primary_hrtf_spec is not None:
            reference_subject_id = self.subject_ids[0]
            reference_hrtf = load_hrtf(self._hrtf_paths[reference_subject_id])
            if self._cache_hrtf:
                self._hrtf_cache[reference_subject_id] = reference_hrtf
            self.available_positions = np.asarray(
                reference_hrtf.Sources.get_positions(angle_unit="degrees"),
                dtype=float,
            )
            if reference_hrtf.TF.frequency_bins is not None:
                self.frequency_bins = np.asarray(reference_hrtf.TF.frequency_bins, dtype=float)
            self._selected_position_indices = self.resolve_positions_selection(
                primary_hrtf_spec.positions,
                primary_hrtf_spec.plane,
                reference_hrtf,
            )
            self.selected_positions = np.asarray(
                self.available_positions[self._selected_position_indices],
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
                spherical_positions[self._selected_position_indices],
                dtype=float,
            )
            self.azimuth_angles = np.unique(
                np.round(selected_spherical_positions[:, 0], 2)
            )
            self.elevation_angles = np.unique(
                np.round(selected_spherical_positions[:, 1], 2)
            )

        self._rows = build_rows(
            subject_ids=self.subject_ids,
            index_by=self.index_by,
            position_indices=self._selected_position_indices,
            ears=self._selected_ears,
        )

    def get_hrtf(self, subject_id: str | int):
        return self.get_subject_hrtf(subject_id)

    def get_subject_hrtf(self, subject_id: str | int):
        resolved_subject_id = self.resolve_dataset_subject_id(subject_id, self.subject_ids)
        if resolved_subject_id not in self._hrtf_paths:
            raise KeyError(
                f"Subject {subject_id!r} resolved to {resolved_subject_id!r} but does not have an available HRTF file"
            )
        path = self._hrtf_paths[resolved_subject_id]
        if not path.exists():
            warnings.warn(
                f"{self.name}: subject {resolved_subject_id} HRTF path is missing: {path}",
                stacklevel=2,
            )
            raise FileNotFoundError(
                f"HRTF path is missing for subject {resolved_subject_id}: {path}"
            )
        hrtf = self._hrtf_cache.get(resolved_subject_id)
        if hrtf is None:
            try:
                hrtf = load_hrtf(path)
            except Exception as exc:
                warnings.warn(
                    f"{self.name}: subject {resolved_subject_id} HRTF file could not be loaded: {path} ({exc})",
                    stacklevel=2,
                )
                raise
            if self._cache_hrtf:
                self._hrtf_cache[resolved_subject_id] = hrtf
        return hrtf

    def get_anthropometry_value(
        self,
        spec: AnthropometrySpec,
        subject_id: str,
    ) -> dict[str, float | str | None]:
        return select_anthropometry_value(
            values=self._anthropometry_rows[subject_id],
            select=spec.select,
            ear=spec.ear,
            dataset_name=self.name,
        )

    def get_spec_value(
        self,
        spec: HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
        subject_id: str,
        row: dict[str, str | int | None],
    ) -> object:
        if isinstance(spec, HRTFSpec):
            hrtf = self.get_subject_hrtf(subject_id)
            use_hrtf_transform = (
                spec.transform is not None and self.is_explicit_hrtf_transform(spec.transform)
            )
            if (
                spec.transform is not None
                and not use_hrtf_transform
                and self.is_raw_hrtf_transform_method(spec.transform)
            ):
                raise ValueError(
                    "Raw Transform methods are not supported in HRTFSpec.transform. "
                    "Use hrtfpykit.datasets.HRTFTransform instead."
                )
            transformed_hrtf = None
            if use_hrtf_transform:
                transform_cache_key = (subject_id, id(spec.transform))
                transformed_hrtf = self._transformed_hrtf_cache.get(transform_cache_key)
                if transformed_hrtf is None:
                    transformed_hrtf = self.apply_hrtf_spec_transform(hrtf, spec.transform)
                    if self._cache_hrtf:
                        self._transformed_hrtf_cache[transform_cache_key] = transformed_hrtf
            value = self.select_hrtf_value(
                hrtf=hrtf if transformed_hrtf is None else transformed_hrtf,
                row=row,
                selected_position_indices=self._selected_position_indices,
                selected_ears=self._selected_ears,
                spec=spec,
            )
            if spec.transform is not None and not use_hrtf_transform:
                value = spec.transform(value)
            return value
        if isinstance(spec, MeshSpec):
            value: object = str(self._mesh_paths[subject_id])
            if spec.transform is not None:
                value = spec.transform(value)
            return value
        if isinstance(spec, AnthropometrySpec):
            value: object = self.get_anthropometry_value(spec, subject_id)
            if spec.transform is not None:
                value = spec.transform(value)
            return value
        if isinstance(spec, ImageSpec):
            if self._image_align_by is None:
                raise ValueError("image align_by is not configured")
            image_key = build_image_key(
                subject_id,
                self._image_align_by,
                None if row["position_index"] is None else int(row["position_index"]),
                None if row["ear"] is None else str(row["ear"]),
            )
            if image_key not in self._image_index:
                raise ValueError(f"No image found for sample {image_key}")
            return apply_image_transform(
                self._image_index[image_key],
                spec.transform,
            )
        if isinstance(spec, VideoSpec):
            if self._video_align_by is None:
                raise ValueError("video align_by is not configured")
            video_key = build_video_key(
                subject_id,
                self._video_align_by,
                None if row["position_index"] is None else int(row["position_index"]),
                None if row["ear"] is None else str(row["ear"]),
            )
            if video_key not in self._video_index:
                raise ValueError(f"No video found for sample {video_key}")
            return apply_video_transform(
                self._video_index[video_key],
                spec.transform,
            )
        raise TypeError(f"Unsupported dataset spec: {type(spec)!r}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        if not isinstance(index, int):
            raise TypeError("Dataset indexing only supports integer indices")
        row = self._rows[int(index)]

        subject_id = str(row["subject_id"])
        inputs: dict[str, object] | None = None
        if len(self._input_specs) > 0:
            inputs = {}
            for spec in self._input_specs:
                inputs[get_spec_name(spec)] = self.get_spec_value(
                    spec,
                    subject_id,
                    row,
                )

        if inputs is not None and self._position_encoding == "one-hot" and row["selected_position_index"] is not None:
            position_encoding = np.zeros(len(self._selected_position_indices), dtype=float)
            position_encoding[int(row["selected_position_index"])] = 1.0
            inputs["position"] = position_encoding
        if inputs is not None and self._ear_encoding == "one-hot" and row["selected_ear_index"] is not None:
            ear_encoding = np.zeros(len(self._selected_ears), dtype=float)
            ear_encoding[int(row["selected_ear_index"])] = 1.0
            inputs["ear"] = ear_encoding

        sample: dict[str, object] = {
            "inputs": inputs,
            "target": None,
        }
        if len(self._target_specs) > 0:
            target_values: dict[str, object] = {}
            for spec in self._target_specs:
                target_values[get_spec_name(spec)] = self.get_spec_value(
                    spec,
                    subject_id,
                    row,
                )
            sample["target"] = target_values

        return sample
