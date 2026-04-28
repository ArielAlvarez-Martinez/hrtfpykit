from pathlib import Path

import numpy as np

from .index import normalize_ears, normalize_index_by
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
    get_spec_name,
    normalize_specs,
)


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


SUPPORTED_ILD_MODES = (
    "broad-band",
    "frequency-dependent",
)


SUPPORTED_ILD_OUTPUTS = (
    "db",
    "linear",
)


SUPPORTED_ITD_OUTPUTS = (
    "seconds",
    "samples",
)

# TODO : move specs methods to specs 

class DatasetSpecPlanner:
    @staticmethod
    def validate_plane_selection(
        plane,
        positions,
        spec_name: str,
    ) -> None:
        if plane is None:
            return
        if not isinstance(positions, str) or str(positions).strip().lower() != "all":
            raise ValueError(f"{spec_name}.plane cannot be combined with custom positions")
        if isinstance(plane, str):
            plane_key = str(plane).strip().lower()
            if plane_key not in {"horizontal", "median", "frontal"}:
                raise ValueError(
                    f"{spec_name}.plane must be horizontal, median, frontal, "
                    "a tuple-based plane selection, or a dict with a 'plane' key"
                )
            return
        if isinstance(plane, tuple):
            if len(plane) not in {2, 3} or not isinstance(plane[0], str):
                raise ValueError(
                    "Tuple plane selection must be ('horizontal'|'median'|'frontal', angle[, angle_unit])"
                )
            plane_key = str(plane[0]).strip().lower()
            if plane_key not in {"horizontal", "median", "frontal"}:
                raise ValueError("Tuple plane selection must use horizontal, median, or frontal")
            return
        if isinstance(plane, dict):
            plane_name = plane.get("plane")
            if plane_name is None:
                raise ValueError("Dict plane selection must include a 'plane' key")
            plane_key = str(plane_name).strip().lower()
            if plane_key not in {"horizontal", "median", "frontal"}:
                raise ValueError("Dict plane selection must use horizontal, median, or frontal")
            return
        raise ValueError(
            f"{spec_name}.plane must be None, a string, a tuple, or a dict"
        )

    # this go to values 
    @classmethod
    def normalize_value_signature(cls, value: object) -> object:
        if isinstance(value, Path):
            return ("path", str(value.expanduser()))
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            return ("array", tuple(array.shape), cls.normalize_value_signature(array.tolist()))
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key).strip().lower(), cls.normalize_value_signature(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls.normalize_value_signature(item) for item in value)
        return value

    @staticmethod
    def choose_primary_spec(
        input_specs: tuple[object, ...],
        all_specs: tuple[object, ...],
        prefer_path: bool = False,
    ) -> object | None:
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

    def plan_dataset_specs(self, inputs, target) -> None:
        self._input_specs = normalize_specs(inputs)
        self._target_specs = normalize_specs(target)
        self.input_names = tuple(get_spec_name(spec) for spec in self._input_specs)
        self.target_names = tuple(get_spec_name(spec) for spec in self._target_specs)
        self.classify_dataset_specs()
        self.validate_shared_dataset_specs()
        self.choose_primary_specs()
        self.configure_dataset_indexing()
        self.configure_dataset_encodings()
        self.validate_dataset_transform()
        self.validate_dataset_requirements()
        self.validate_hrtf_specs()
        self.validate_itd_specs()
        self.validate_ild_specs()
        self.validate_sh_specs()

    def classify_dataset_specs(self) -> None:
        all_specs = self._input_specs + self._target_specs
        self.hrtf_specs = tuple(spec for spec in all_specs if isinstance(spec, HRTFSpec))
        self.itd_specs = tuple(spec for spec in all_specs if isinstance(spec, ITDSpec))
        self.ild_specs = tuple(spec for spec in all_specs if isinstance(spec, ILDSpec))
        self.sh_specs = tuple(spec for spec in all_specs if isinstance(spec, SHSpec))
        self.mesh_specs = tuple(spec for spec in all_specs if isinstance(spec, MeshSpec))
        self.anthropometry_specs = tuple(spec for spec in all_specs if isinstance(spec, AnthropometrySpec))
        self.image_specs = tuple(spec for spec in all_specs if isinstance(spec, ImageSpec))
        self.video_specs = tuple(spec for spec in all_specs if isinstance(spec, VideoSpec))

        self.input_hrtf_specs = tuple(spec for spec in self._input_specs if isinstance(spec, HRTFSpec))
        self.input_itd_specs = tuple(spec for spec in self._input_specs if isinstance(spec, ITDSpec))
        self.input_ild_specs = tuple(spec for spec in self._input_specs if isinstance(spec, ILDSpec))
        self.input_sh_specs = tuple(spec for spec in self._input_specs if isinstance(spec, SHSpec))
        self.input_mesh_specs = tuple(spec for spec in self._input_specs if isinstance(spec, MeshSpec))
        self.input_anthropometry_specs = tuple(
            spec for spec in self._input_specs if isinstance(spec, AnthropometrySpec)
        )
        self.input_image_specs = tuple(spec for spec in self._input_specs if isinstance(spec, ImageSpec))
        self.input_video_specs = tuple(spec for spec in self._input_specs if isinstance(spec, VideoSpec))

        self.spatial_specs = self.hrtf_specs + self.itd_specs + self.ild_specs
        self.input_spatial_specs = self.input_hrtf_specs + self.input_itd_specs + self.input_ild_specs
        self.hrtf_backed_specs = self.hrtf_specs + self.itd_specs + self.ild_specs + self.sh_specs
        self.input_hrtf_backed_specs = (
            self.input_hrtf_specs + self.input_itd_specs + self.input_ild_specs + self.input_sh_specs
        )
        self.ear_axis_specs = self.hrtf_specs + self.sh_specs
        self.input_ear_axis_specs = self.input_hrtf_specs + self.input_sh_specs
        self.ear_aligned_media_specs = tuple(
            spec
            for spec in self.image_specs + self.video_specs
            if "ear" in normalize_index_by(spec.align_by)
        )
        self.input_ear_aligned_media_specs = tuple(
            spec
            for spec in self.input_image_specs + self.input_video_specs
            if "ear" in normalize_index_by(spec.align_by)
        )
        self.ear_conditioned_specs = self.ear_axis_specs + self.ear_aligned_media_specs
        self.input_ear_conditioned_specs = self.input_ear_axis_specs + self.input_ear_aligned_media_specs

    def validate_shared_spec_attribute(
        self,
        specs: tuple[object, ...],
        attribute: str,
        message: str,
    ) -> None:
        if len(specs) <= 1:
            return
        signatures = {
            self.normalize_value_signature(getattr(spec, attribute))
            for spec in specs
        }
        if len(signatures) > 1:
            raise ValueError(message)

    def validate_anthropometry_specs(self) -> None:
        if len(self.anthropometry_specs) <= 1:
            return
        anthropometry_path_signatures = set()
        for spec in self.anthropometry_specs:
            if spec.path is None:
                continue
            path = Path(spec.path).expanduser()
            if not path.is_absolute():
                path = self.root / path
            anthropometry_path_signatures.add(self.normalize_value_signature(path))
        if len(anthropometry_path_signatures) > 1:
            raise ValueError(
                "All AnthropometrySpec objects must use the same path when anthropometry is used in both inputs and target"
            )

    def validate_shared_dataset_specs(self) -> None:
        self.validate_shared_spec_attribute(
            self.spatial_specs,
            "positions",
            "All acoustic specs must use the same positions when hrtf, itd, or ild are used together",
        )
        self.validate_shared_spec_attribute(
            self.spatial_specs,
            "plane",
            "All acoustic specs must use the same plane when hrtf, itd, or ild are used together",
        )
        self.validate_shared_spec_attribute(
            self.ear_axis_specs,
            "ears",
            "All HRTFSpec and SHSpec objects must use the same ears when they are used together",
        )
        self.validate_media_specs(self.image_specs, "ImageSpec", "image")
        self.validate_media_specs(self.video_specs, "VideoSpec", "video")
        self.validate_anthropometry_specs()

    def choose_primary_specs(self) -> None:
        self.primary_hrtf_spec = self.choose_primary_spec(self.input_hrtf_specs, self.hrtf_specs)
        self.primary_sh_spec = self.choose_primary_spec(self.input_sh_specs, self.sh_specs)
        self.primary_mesh_spec = self.choose_primary_spec(self.input_mesh_specs, self.mesh_specs)
        self.primary_anthropometry_spec = self.choose_primary_spec(
            self.input_anthropometry_specs,
            self.anthropometry_specs,
            prefer_path=True,
        )
        self.primary_image_spec = self.choose_primary_spec(
            self.input_image_specs,
            self.image_specs,
            prefer_path=True,
        )
        self.primary_video_spec = self.choose_primary_spec(
            self.input_video_specs,
            self.video_specs,
            prefer_path=True,
        )
        self.primary_spatial_spec = self.choose_primary_spec(self.input_spatial_specs, self.spatial_specs)
        self.primary_hrtf_backed_spec = self.choose_primary_spec(
            self.input_hrtf_backed_specs,
            self.hrtf_backed_specs,
        )

    # TODO : move the method to index 
    def configure_dataset_indexing(self) -> None:
        self.index_by = ("subject",)
        self._selected_ears = []
        if self.primary_hrtf_backed_spec is None:
            return
        include_position = any("position" in normalize_index_by(spec.index_by) for spec in self.spatial_specs)
        include_ear = any("ear" in normalize_index_by(spec.index_by) for spec in self.ear_axis_specs) or len(self.ear_aligned_media_specs) > 0
        include_frequency = any(
            "frequency" in normalize_index_by(spec.index_by)
            for spec in self.hrtf_specs + self.ild_specs + self.sh_specs
        )
        include_samples = any("samples" in normalize_index_by(spec.index_by) for spec in self.hrtf_specs)
        index_by_values = ["subject"]
        if include_position:
            index_by_values.append("position")
        if include_ear:
            index_by_values.append("ear")
        if include_frequency:
            index_by_values.append("frequency")
        if include_samples:
            index_by_values.append("samples")
        self.index_by = tuple(index_by_values)
        if len(self.ear_conditioned_specs) == 0:
            self._selected_ears = []
        elif self.primary_hrtf_spec is not None:
            self._selected_ears = normalize_ears(self.primary_hrtf_spec.ears)
        elif self.primary_sh_spec is not None:
            self._selected_ears = normalize_ears(self.primary_sh_spec.ears)
        else:
            self._selected_ears = normalize_ears("both")

    def configure_dataset_encodings(self) -> None:
        input_position_one_hot = tuple(
            bool(spec.position_one_hot)
            for spec in self.input_spatial_specs
            if hasattr(spec, "position_one_hot")
        )
        input_position_index = tuple(
            bool(spec.position_index)
            for spec in self.input_spatial_specs
            if hasattr(spec, "position_index")
        )
        input_frequency_one_hot = tuple(
            bool(spec.frequency_one_hot)
            for spec in self.input_hrtf_specs + self.input_ild_specs + self.input_sh_specs
            if hasattr(spec, "frequency_one_hot")
        )
        input_frequency_index = tuple(
            bool(spec.frequency_index)
            for spec in self.input_hrtf_specs + self.input_ild_specs + self.input_sh_specs
            if hasattr(spec, "frequency_index")
        )
        input_sample_one_hot = tuple(
            bool(spec.sample_one_hot)
            for spec in self.input_hrtf_specs
            if hasattr(spec, "sample_one_hot")
        )
        input_sample_index = tuple(
            bool(spec.sample_index)
            for spec in self.input_hrtf_specs
            if hasattr(spec, "sample_index")
        )
        self._position_one_hot = any(input_position_one_hot)
        self._position_index = any(input_position_index)
        self._frequency_one_hot = any(input_frequency_one_hot)
        self._frequency_index = any(input_frequency_index)
        self._sample_one_hot = any(input_sample_one_hot)
        self._sample_index = any(input_sample_index)
        if len(self.input_ear_conditioned_specs) == 0:
            self._ear_one_hot = False
            self._ear_index = False
        else:
            ear_one_hot = tuple(
                bool(spec.ear_one_hot)
                for spec in self.input_ear_conditioned_specs
            )
            ear_indices = tuple(
                bool(spec.ear_index)
                for spec in self.input_ear_conditioned_specs
            )
            self._ear_one_hot = any(ear_one_hot)
            self._ear_index = any(ear_indices)

        conflicting_input_names = (
            ("position_one_hot", self._position_one_hot),
            ("position_index", self._position_index),
            ("frequency_one_hot", self._frequency_one_hot),
            ("frequency_index", self._frequency_index),
            ("sample_one_hot", self._sample_one_hot),
            ("sample_index", self._sample_index),
            ("ear_one_hot", self._ear_one_hot),
            ("ear_index", self._ear_index),
        )
        for name, enabled in conflicting_input_names:
            if enabled and name in self.input_names:
                raise ValueError(f"Input spec name {name!r} conflicts with categorical dataset output {name!r}")
        if (self._position_one_hot or self._position_index) and "position" not in self.index_by:
            raise ValueError("position_one_hot and position_index require index_by to include 'position'")
        if (self._frequency_one_hot or self._frequency_index) and "frequency" not in self.index_by:
            raise ValueError("frequency_one_hot and frequency_index require index_by to include 'frequency'")
        if (self._sample_one_hot or self._sample_index) and "samples" not in self.index_by:
            raise ValueError("sample_one_hot and sample_index require index_by to include 'samples'")
        if (self._ear_one_hot or self._ear_index) and "ear" not in self.index_by:
            raise ValueError("ear_one_hot and ear_index require index_by to include 'ear'")

    def validate_dataset_transform(self) -> None:
        if (
            self.dataset_hrtf_transform is not None
            and not self.is_explicit_hrtf_transform(self.dataset_hrtf_transform)
        ):
            raise ValueError(
                "dataset_hrtf_transform must be created with hrtfpykit.datasets.HRTFTransform"
            )

    def validate_dataset_requirements(self) -> None:
        if any(axis in self.index_by for axis in ("position", "ear", "frequency", "samples")) and self.primary_hrtf_backed_spec is None:
            raise ValueError(
                "Acoustic index axes 'position', 'ear', 'frequency', and 'samples' require "
                "hrtf, itd, ild, or sh in inputs or target"
            )
        if self.primary_hrtf_backed_spec is None:
            return
        if self.config.hrtf is None:
            raise ValueError(f"{self.name} does not provide hrtf data")
        if self.variant is None:
            raise ValueError("variant could not be resolved")
        if self.variant not in self.config.hrtf.variants:
            raise ValueError(
                f"Unsupported variant {self.variant!r}. "
                f"Expected one of {self.config.hrtf.variants}"
            )

    @staticmethod
    def validate_positions_selection(positions, spec_name: str) -> None:
        if isinstance(positions, str) and str(positions).strip().lower() != "all":
            raise ValueError(
                f"{spec_name}.positions must be 'all' or a sequence of position indices"
            )

    def validate_hrtf_specs(self) -> None:
        for spec in self.hrtf_specs:
            domain = str(spec.domain).strip().lower()
            signal = str(spec.signal).strip().lower()
            if not isinstance(spec.position_one_hot, bool):
                raise ValueError("HRTFSpec.position_one_hot must be a boolean")
            if not isinstance(spec.position_index, bool):
                raise ValueError("HRTFSpec.position_index must be a boolean")
            if not isinstance(spec.ear_one_hot, bool):
                raise ValueError("HRTFSpec.ear_one_hot must be a boolean")
            if not isinstance(spec.ear_index, bool):
                raise ValueError("HRTFSpec.ear_index must be a boolean")
            if not isinstance(spec.frequency_one_hot, bool):
                raise ValueError("HRTFSpec.frequency_one_hot must be a boolean")
            if not isinstance(spec.frequency_index, bool):
                raise ValueError("HRTFSpec.frequency_index must be a boolean")
            if not isinstance(spec.sample_one_hot, bool):
                raise ValueError("HRTFSpec.sample_one_hot must be a boolean")
            if not isinstance(spec.sample_index, bool):
                raise ValueError("HRTFSpec.sample_index must be a boolean")
            if domain not in SUPPORTED_HRTF_DOMAINS:
                raise ValueError(
                    f"Unsupported domain {spec.domain!r}. Expected one of {SUPPORTED_HRTF_DOMAINS}"
                )
            if signal not in SUPPORTED_HRTF_SIGNALS:
                raise ValueError(
                    f"Unsupported signal {spec.signal!r}. Expected one of {SUPPORTED_HRTF_SIGNALS}"
                )
            if spec.transform is not None and not self.is_explicit_hrtf_transform(spec.transform):
                raise ValueError(
                    "HRTFSpec.transform must be created with hrtfpykit.datasets.HRTFTransform"
                )
            if domain == "time" and signal != "ir":
                raise ValueError("HRTFSpec with domain='time' requires signal='ir'")
            if domain == "frequency" and signal == "ir":
                raise ValueError("HRTFSpec with domain='frequency' cannot use signal='ir'")
            spec_index_by = normalize_index_by(spec.index_by)
            self.validate_positions_selection(spec.positions, "HRTFSpec")
            if "frequency" in spec_index_by and domain != "frequency":
                raise ValueError("HRTFSpec.index_by including 'frequency' requires domain='frequency'")
            if "samples" in spec_index_by and (domain != "time" or signal != "ir"):
                raise ValueError(
                    "HRTFSpec.index_by including 'samples' requires domain='time' and signal='ir'"
                )
            self.validate_plane_selection(spec.plane, spec.positions, "HRTFSpec")

    def validate_itd_specs(self) -> None:
        for spec in self.itd_specs:
            if not isinstance(spec.position_one_hot, bool):
                raise ValueError("ITDSpec.position_one_hot must be a boolean")
            if not isinstance(spec.position_index, bool):
                raise ValueError("ITDSpec.position_index must be a boolean")
            output = str(spec.output).strip().lower()
            if output not in SUPPORTED_ITD_OUTPUTS:
                raise ValueError(
                    f"Unsupported ITD output {spec.output!r}. Expected one of {SUPPORTED_ITD_OUTPUTS}"
                )
            self.validate_positions_selection(spec.positions, "ITDSpec")
            spec_index_by = normalize_index_by(spec.index_by)
            if "ear" in spec_index_by or "frequency" in spec_index_by or "samples" in spec_index_by:
                raise ValueError("ITDSpec.index_by only supports 'subject' and optional 'position'")
            self.validate_plane_selection(spec.plane, spec.positions, "ITDSpec")

    def validate_ild_specs(self) -> None:
        for spec in self.ild_specs:
            if not isinstance(spec.position_one_hot, bool):
                raise ValueError("ILDSpec.position_one_hot must be a boolean")
            if not isinstance(spec.position_index, bool):
                raise ValueError("ILDSpec.position_index must be a boolean")
            if not isinstance(spec.frequency_one_hot, bool):
                raise ValueError("ILDSpec.frequency_one_hot must be a boolean")
            if not isinstance(spec.frequency_index, bool):
                raise ValueError("ILDSpec.frequency_index must be a boolean")
            mode = str(spec.mode).strip().lower()
            output = str(spec.output).strip().lower()
            if mode not in SUPPORTED_ILD_MODES:
                raise ValueError(
                    f"Unsupported ILD mode {spec.mode!r}. Expected one of {SUPPORTED_ILD_MODES}"
                )
            if output not in SUPPORTED_ILD_OUTPUTS:
                raise ValueError(
                    f"Unsupported ILD output {spec.output!r}. Expected one of {SUPPORTED_ILD_OUTPUTS}"
                )
            self.validate_positions_selection(spec.positions, "ILDSpec")
            spec_index_by = normalize_index_by(spec.index_by)
            if "ear" in spec_index_by:
                raise ValueError("ILDSpec.index_by does not support 'ear'")
            if "samples" in spec_index_by:
                raise ValueError("ILDSpec.index_by does not support 'samples'")
            if "frequency" in spec_index_by and mode != "frequency-dependent":
                raise ValueError(
                    "ILDSpec.index_by including 'frequency' requires mode='frequency-dependent'"
                )
            self.validate_plane_selection(spec.plane, spec.positions, "ILDSpec")

    def validate_sh_specs(self) -> None:
        for spec in self.sh_specs:
            if not isinstance(spec.ear_one_hot, bool):
                raise ValueError("SHSpec.ear_one_hot must be a boolean")
            if not isinstance(spec.ear_index, bool):
                raise ValueError("SHSpec.ear_index must be a boolean")
            if not isinstance(spec.frequency_one_hot, bool):
                raise ValueError("SHSpec.frequency_one_hot must be a boolean")
            if not isinstance(spec.frequency_index, bool):
                raise ValueError("SHSpec.frequency_index must be a boolean")
            if isinstance(spec.sh_order, bool) or not isinstance(spec.sh_order, int) or spec.sh_order < 0:
                raise ValueError("SHSpec.sh_order must be a non-negative integer")
            if isinstance(spec.epsilon, bool):
                raise ValueError("SHSpec.epsilon must be a finite, positive value")
            try:
                epsilon = float(spec.epsilon)
            except (TypeError, ValueError):
                raise ValueError("SHSpec.epsilon must be a finite, positive value") from None
            if not np.isfinite(epsilon) or epsilon <= 0.0:
                raise ValueError("SHSpec.epsilon must be a finite, positive value")
            spec_index_by = normalize_index_by(spec.index_by)
            if "position" in spec_index_by:
                raise ValueError("SHSpec.index_by does not support 'position'")
            if "samples" in spec_index_by:
                raise ValueError("SHSpec.index_by does not support 'samples'")
            if "ear" in spec_index_by and len(normalize_ears(spec.ears)) != 2:
                raise ValueError("SHSpec.index_by including 'ear' requires ears='both'")

    def validate_media_specs(
        self,
        specs: tuple[ImageSpec | VideoSpec, ...],
        spec_name: str,
        resource_name: str,
    ) -> None:
        if len(specs) > 1:
            path_signatures = {
                self.normalize_value_signature(Path(spec.path))
                for spec in specs
                if spec.path is not None
            }
            if len(path_signatures) > 1:
                raise ValueError(
                    f"All {spec_name} objects must use the same path when {resource_name} is used in both inputs and target"
                )
            align_by_values = {normalize_index_by(spec.align_by) for spec in specs}
            if len(align_by_values) > 1:
                raise ValueError(
                    f"All {spec_name} objects must use the same align_by when {resource_name} is used in both inputs and target"
                )
        for spec in specs:
            if not isinstance(spec.ear_one_hot, bool):
                raise ValueError(f"{spec_name}.ear_one_hot must be a boolean")
            if not isinstance(spec.ear_index, bool):
                raise ValueError(f"{spec_name}.ear_index must be a boolean")
            align_by = normalize_index_by(spec.align_by)
            if (spec.ear_one_hot or spec.ear_index) and "ear" not in align_by:
                raise ValueError(f"{spec_name}.ear_one_hot and {spec_name}.ear_index require align_by to include 'ear'")
