from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
)
from .sanitize import (
    sanitize_accessed_by,
    sanitize_grouped_by,
    sanitize_index_by,
    sanitize_ear,
    sanitize_ears,
    sanitize_specs,
)

from .config import DatasetConfig


SUPPORTED_MEDIA_GROUPED_BY = (("subject",), ("subject", "ear"))


@dataclass(frozen=True)
class DatasetSpecPlan:
    input_specs: tuple[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
        ...,
    ]
    target_specs: tuple[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
        ...,
    ]
    specs: tuple[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
        ...,
    ]
    input_names: tuple[str, ...]
    target_names: tuple[str, ...]
    index_by: tuple[str, ...]
    selected_ears: tuple[tuple[str, int], ...]
    position_one_hot: bool
    position_index: bool
    frequency_one_hot: bool
    frequency_index: bool
    sample_one_hot: bool
    sample_index: bool
    ear_one_hot: bool
    ear_index: bool

class DatasetSpecWorkflow:
    @classmethod
    def build(
        cls,
        config: type[DatasetConfig] | DatasetConfig,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None,
    ) -> DatasetSpecPlan:
        input_specs = sanitize_specs(inputs)
        target_specs = sanitize_specs(target)
        specs = input_specs + target_specs
        if len(specs) == 0:
            raise ValueError("Dataset requires at least one dataset spec in inputs or target")

        for spec_type, resource_name in (
            (AnthropometrySpec, "anthropometry"),
            (ImageSpec, "image"),
            (VideoSpec, "video"),
            (MeshSpec, "mesh"),
        ):
            explicit_paths = tuple(
                str(spec.path)
                for spec in specs
                if isinstance(spec, spec_type) and getattr(spec, "path", None) is not None
            )
            if len(set(explicit_paths)) > 1:
                raise ValueError(
                    f"{resource_name} specs must not define different paths in the same dataset"
                )

        dataset_index_by = None
        dataset_index_by_spec: str | None = None
        for spec in cls.get_indexed_specs(specs):
            spec_index_by = sanitize_index_by(spec.index_by)
            spec_name = cls.get_spec_name(spec)
            spec_axes = set(spec_index_by[1:])
            if isinstance(spec, HRTFSpec):
                domain = str(spec.domain).strip().lower()
                supported_axes = {"position", "ear", "samples"} if domain == "time" else {"position", "ear", "frequency"}
                supported_index_by = (
                    "('subject',), ('subject', 'position'), ('subject', 'ear'), "
                    "('subject', 'samples'), ('subject', 'position', 'ear'), "
                    "('subject', 'position', 'samples'), ('subject', 'ear', 'samples'), "
                    "('subject', 'position', 'ear', 'samples')"
                ) if domain == "time" else (
                    "('subject',), ('subject', 'position'), ('subject', 'ear'), "
                    "('subject', 'frequency'), ('subject', 'position', 'ear'), "
                    "('subject', 'position', 'frequency'), ('subject', 'ear', 'frequency'), "
                    "('subject', 'position', 'ear', 'frequency')"
                )
            elif isinstance(spec, ITDSpec):
                supported_axes = {"position"}
                supported_index_by = "('subject',), ('subject', 'position')"
            elif isinstance(spec, ILDSpec):
                supported_axes = {"position"}
                if str(spec.mode).strip().lower() == "frequency-dependent":
                    supported_axes.add("frequency")
                    supported_index_by = (
                        "('subject',), ('subject', 'position'), ('subject', 'frequency'), "
                        "('subject', 'position', 'frequency')"
                    )
                else:
                    supported_index_by = "('subject',), ('subject', 'position')"
            elif isinstance(spec, SHSpec):
                supported_axes = {"ear", "frequency"}
                supported_index_by = (
                    "('subject',), ('subject', 'ear'), ('subject', 'frequency'), "
                    "('subject', 'ear', 'frequency')"
                )
            else:
                supported_axes = set()
                supported_index_by = "()"

            unsupported_axes = sorted(spec_axes - supported_axes)
            if len(unsupported_axes) > 0:
                compatibility_hint = ""
                if isinstance(spec, HRTFSpec):
                    domain = str(spec.domain).strip().lower()
                    if "frequency" in unsupported_axes and domain == "time":
                        compatibility_hint = (
                            " In HRTFSpec, the 'frequency' axis is available only when domain='frequency'."
                        )
                    elif "samples" in unsupported_axes and domain == "frequency":
                        compatibility_hint = (
                            " In HRTFSpec, the 'samples' axis is available only when domain='time'."
                        )
                elif isinstance(spec, ILDSpec):
                    if "frequency" in unsupported_axes and str(spec.mode).strip().lower() != "frequency-dependent":
                        compatibility_hint = " In ILDSpec, enable frequency indexing by setting mode='frequency-dependent'."

                raise ValueError(
                    f"{type(spec).__name__} index_by={spec_index_by!r} uses unsupported axes: "
                    + ", ".join(unsupported_axes)
                    + ". "
                    f"Supported index_by combinations for {type(spec).__name__}: {supported_index_by}."
                    + compatibility_hint
                )

            for flag_name, axis_name in (
                ("position_one_hot", "position"),
                ("position_index", "position"),
                ("ear_one_hot", "ear"),
                ("ear_index", "ear"),
                ("frequency_one_hot", "frequency"),
                ("frequency_index", "frequency"),
                ("sample_one_hot", "samples"),
                ("sample_index", "samples"),
            ):
                if bool(getattr(spec, flag_name, False)) and axis_name not in spec_index_by:
                    compatibility_hint = ""
                    if isinstance(spec, HRTFSpec):
                        domain = str(spec.domain).strip().lower()
                        if axis_name == "frequency" and domain == "time":
                            compatibility_hint = (
                                " In HRTFSpec, set domain='frequency' to use frequency-indexed specs."
                            )
                        elif axis_name == "samples" and domain == "frequency":
                            compatibility_hint = (
                                " In HRTFSpec, set domain='time' to use sample-indexed specs."
                            )
                    elif isinstance(spec, ILDSpec):
                        if axis_name == "frequency":
                            compatibility_hint = (
                                " In ILDSpec, set mode='frequency-dependent' to use frequency indexing."
                            )

                    raise ValueError(
                        f"{type(spec).__name__}.{flag_name} requires index_by to include {axis_name!r}. "
                        f"Supported index_by combinations for {type(spec).__name__}: {supported_index_by}."
                        + compatibility_hint
                    )
            if dataset_index_by is None:
                dataset_index_by = spec_index_by
                dataset_index_by_spec = spec_name
            elif spec_index_by != dataset_index_by:
                raise ValueError(
                    "All indexed specs in a dataset must use the same index_by. "
                    f"{spec_name!r} uses {spec_index_by!r}, but {dataset_index_by_spec!r} uses {dataset_index_by!r}. "
                    "Pick one index_by for the full dataset."
                )

        for spec in cls.filter_specs((ImageSpec, VideoSpec, AnthropometrySpec), specs):
            grouped_by = sanitize_grouped_by(spec.grouped_by)
            if isinstance(spec, (ImageSpec, VideoSpec)):
                if grouped_by not in SUPPORTED_MEDIA_GROUPED_BY:
                    raise ValueError(
                        f"{type(spec).__name__} grouped_by={grouped_by!r} is not supported. "
                        f"Supported values: {SUPPORTED_MEDIA_GROUPED_BY}"
                    )
            spec.grouped_by = grouped_by
            if isinstance(spec, AnthropometrySpec):
                spec.accessed_by = sanitize_accessed_by(spec.accessed_by)
                spec.ear = sanitize_ear(spec.ear)
            if bool(spec.ear_one_hot) or bool(spec.ear_index):
                if "ear" not in grouped_by:
                    raise ValueError(
                        f"{type(spec).__name__} ear encodings require grouped_by to include 'ear' "
                        f"(got grouped_by={grouped_by!r})."
                    )

        input_names = tuple(cls.get_spec_name(spec) for spec in input_specs)
        target_names = tuple(cls.get_spec_name(spec) for spec in target_specs)

        indexed_specs = cls.get_indexed_specs(specs)
        index_by = ("subject",) if len(indexed_specs) == 0 else sanitize_index_by(indexed_specs[0].index_by)
        media_specs = cls.filter_specs((ImageSpec, VideoSpec, AnthropometrySpec), specs)
        if index_by == ("subject",) and any(
            "ear" in sanitize_grouped_by(spec.grouped_by) for spec in media_specs
        ):
            index_by = ("subject", "ear")
        if "ear" not in index_by:
            for spec in media_specs:
                grouped_by = sanitize_grouped_by(spec.grouped_by)
                if "ear" in grouped_by:
                    raise ValueError(
                        f"{type(spec).__name__} grouped_by={grouped_by!r} requires an ear-indexed dataset row"
                    )
                if bool(spec.ear_one_hot) or bool(spec.ear_index):
                    raise ValueError(
                        f"{type(spec).__name__} ear encodings require an ear-indexed dataset row"
                    )

        selected_ears: tuple[tuple[str, int], ...] = tuple()
        if "ear" in index_by:
            ear_specs = cls.filter_specs((HRTFSpec, SHSpec, AnthropometrySpec), specs)
            for spec in indexed_specs:
                if "ear" not in sanitize_index_by(spec.index_by):
                    continue
                if isinstance(spec, (HRTFSpec, SHSpec)):
                    spec_ears = tuple(sanitize_ears(spec.ears))
                else:
                    continue
                if len(selected_ears) == 0:
                    selected_ears = spec_ears
                elif spec_ears != selected_ears:
                    raise ValueError(
                        "All ear-indexed specs must use the same ear axis. "
                        f"Expected {selected_ears!r}, got {spec_ears!r} for {type(spec).__name__}"
                    )
            if len(selected_ears) == 0:
                for spec in ear_specs:
                    if not isinstance(spec, AnthropometrySpec):
                        continue
                    if "ear" not in sanitize_grouped_by(spec.grouped_by):
                        continue
                    spec_ears = sanitize_ears(spec.ear if spec.ear is not None else "both")
                    if len(selected_ears) == 0:
                        selected_ears = tuple(spec_ears)
                    elif tuple(spec_ears) != selected_ears:
                        raise ValueError(
                            "All ear-indexed specs must use the same ear axis. "
                            f"Expected {selected_ears!r}, got {tuple(spec_ears)!r} for {type(spec).__name__}"
                        )
            selected_ears = tuple(sanitize_ears("both")) if len(selected_ears) == 0 else selected_ears

        return DatasetSpecPlan(
            input_specs=input_specs,
            target_specs=target_specs,
            specs=specs,
            input_names=input_names,
            target_names=target_names,
            index_by=index_by,
            selected_ears=selected_ears,
            position_one_hot=any(
                bool(spec.position_one_hot)
                for spec in cls.filter_specs((HRTFSpec, ITDSpec, ILDSpec), specs)
            ),
            position_index=any(
                bool(spec.position_index)
                for spec in cls.filter_specs((HRTFSpec, ITDSpec, ILDSpec), specs)
            ),
            frequency_one_hot=any(
                bool(spec.frequency_one_hot)
                for spec in cls.filter_specs((HRTFSpec, ILDSpec, SHSpec), specs)
            ),
            frequency_index=any(
                bool(spec.frequency_index)
                for spec in cls.filter_specs((HRTFSpec, ILDSpec, SHSpec), specs)
            ),
            sample_one_hot=any(bool(spec.sample_one_hot) for spec in cls.filter_specs((HRTFSpec,), specs)),
            sample_index=any(bool(spec.sample_index) for spec in cls.filter_specs((HRTFSpec,), specs)),
            ear_one_hot=any(
                bool(spec.ear_one_hot)
                for spec in cls.filter_specs((HRTFSpec, SHSpec, ImageSpec, VideoSpec, AnthropometrySpec), specs)
            ),
            ear_index=any(
                bool(spec.ear_index)
                for spec in cls.filter_specs((HRTFSpec, SHSpec, ImageSpec, VideoSpec, AnthropometrySpec), specs)
            ),
        )


    @staticmethod
    def get_indexed_specs(
        specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...],
    ) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...]:
        return cast(
            tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...],
            tuple(
                spec for spec in specs if isinstance(spec, (HRTFSpec, ITDSpec, ILDSpec, SHSpec))
            ),
        )

    @staticmethod
    def filter_specs(
        spec_types: type[object] | tuple[type[object], ...],
        specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...],
    ) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]:
        return tuple(spec for spec in specs if isinstance(spec, spec_types))

    @staticmethod
    def get_spec_name(
        spec: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
    ) -> str:
        explicit_name = getattr(spec, "name", None)
        if explicit_name is not None:
            name = str(explicit_name).strip()
            if name == "":
                raise ValueError("Dataset spec name must not be empty")
            return name
        if isinstance(spec, HRTFSpec):
            return "hrtf"
        if isinstance(spec, ITDSpec):
            return "itd"
        if isinstance(spec, ILDSpec):
            return "ild"
        if isinstance(spec, SHSpec):
            return "sh"
        if isinstance(spec, MeshSpec):
            return "mesh"
        if isinstance(spec, AnthropometrySpec):
            return "anthropometry"
        if isinstance(spec, ImageSpec):
            return "image"
        if isinstance(spec, VideoSpec):
            return "video"
        raise TypeError(f"Unsupported dataset spec: {type(spec)!r}")
