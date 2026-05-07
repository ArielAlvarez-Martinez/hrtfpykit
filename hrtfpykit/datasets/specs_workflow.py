from collections.abc import Sequence
from dataclasses import dataclass

from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MetadataSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
)
from .specs_registry import (
    SPEC_DESCRIPTORS,
    get_axis_compatibility_hint,
    get_flag_compatibility_hint,
    get_spec_descriptor,
    get_spec_name,
    get_specs,
    get_supported_index,
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
    """Store normalized specs and shared dataset indexing decisions.

    Parameters
    ----------
    input_specs : tuple of specs
        Specs exposed under ``sample['inputs']`` after sanitization.
    target_specs : tuple of specs
        Specs exposed under ``sample['target']`` after sanitization.
    specs : tuple of specs
        Combined input and target specs.
    input_names, target_names : tuple of str
        Public sample keys for input and target values.
    index_by : tuple of str
        Shared dataset row axes selected from the indexed specs.
    selected_ears : tuple of tuple
        Ear labels and source ear indices used by ear-indexed rows.
    position_one_hot, position_index, frequency_one_hot, frequency_index : bool
        Flags indicating which row context encodings should be added.
    sample_one_hot, sample_index, ear_one_hot, ear_index : bool
        Flags indicating which sample and ear context encodings should be added.

    Returns
    -------
    DatasetSpecPlan Immutable plan consumed by ``DatasetBuilder``.

    Use Cases
    ---------
    - Make spec normalization output explicit.
    - Keep one row-indexing contract across all indexed specs.
    - Tell sample extraction which context encodings to include.
    """

    input_specs: tuple[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec,
        ...,
    ]
    target_specs: tuple[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec,
        ...,
    ]
    specs: tuple[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec,
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
    """Normalize and validate dataset specs before resource scanning.

    ``DatasetSpecWorkflow`` turns user-provided input and target specs into one
    coherent dataset plan. It validates duplicate names, path consistency,
    supported index axes, shared ``index_by`` behavior, grouped media behavior,
    ear selection, and requested row-context encodings.

    This utility returns the normalized spec plan consumed by ``DatasetBuilder``.

    Use Cases
    ---------
    - Validate specs before resource scanning.
    - Enforce one shared dataset row-axis contract.
    - Normalize copied specs without mutating caller-owned spec objects.
    """

    @classmethod
    def build(
        cls,
        config: type[DatasetConfig] | DatasetConfig,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None,
    ) -> DatasetSpecPlan:
        """Build a normalized spec plan from input and target specs.

        This method is the spec validation pipeline for dataset construction. It
        copies user specs, validates names and path consistency, enforces one shared
        indexed row structure, checks spec-specific axis compatibility, normalizes
        grouped resources, and decides which row context encodings must be produced.

        Parameters
        ----------
        config : DatasetConfig or type[DatasetConfig]
            Dataset configuration used for resource availability rules.
        inputs : spec, sequence of specs, or None
            Specs exposed under ``sample['inputs']``.
        target : spec, sequence of specs, or None
            Specs exposed under ``sample['target']``.

        Returns
        -------
        DatasetSpecPlan Normalized spec plan used by dataset construction.

        Use Cases
        ---------
        - Validate all specs before scanning resources.
        - Compute the shared dataset ``index_by`` axes.
        - Determine whether row context encodings are needed.
        """

        input_specs = sanitize_specs(inputs)
        target_specs = sanitize_specs(target)
        specs = input_specs + target_specs
        indexed_specs = get_specs(specs, indexed=True)
        grouped_specs = get_specs(specs, grouped=True)
        position_selectable_specs = get_specs(specs, position_selectable=True)
        ear_selectable_specs = get_specs(specs, ear_selectable=True)

        for descriptor in SPEC_DESCRIPTORS:
            if not descriptor.path_based:
                continue
            explicit_paths = tuple(
                str(spec.path)
                for spec in specs
                if isinstance(spec, descriptor.spec_type) and getattr(spec, "path", None) is not None
            )
            if len(set(explicit_paths)) > 1:
                raise ValueError(
                    f"{descriptor.resource_name} specs must not define different paths in the same dataset"
                )

        dataset_index_by = None
        dataset_index_by_spec: str | None = None
        for spec in indexed_specs:
            spec_index_by = sanitize_index_by(spec.index_by)
            spec_name = cls.get_spec_name(spec)
            spec_axes = set(spec_index_by[1:])
            supported_axes, supported_index_by = get_supported_index(spec)

            unsupported_axes = sorted(spec_axes - supported_axes)
            if len(unsupported_axes) > 0:
                compatibility_hint = "".join(
                    get_axis_compatibility_hint(spec, axis_name)
                    for axis_name in unsupported_axes
                )

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
                    compatibility_hint = get_flag_compatibility_hint(spec, axis_name)

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

        for spec in grouped_specs:
            descriptor = get_spec_descriptor(spec)
            grouped_by = sanitize_grouped_by(spec.grouped_by)
            if descriptor.media:
                if grouped_by not in SUPPORTED_MEDIA_GROUPED_BY:
                    raise ValueError(
                        f"{type(spec).__name__} grouped_by={grouped_by!r} is not supported. "
                        f"Supported values: {SUPPORTED_MEDIA_GROUPED_BY}"
                    )
            spec.grouped_by = grouped_by
            if descriptor.resource_name in {"anthropometry", "metadata"}:
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

        index_by = ("subject",) if len(indexed_specs) == 0 else sanitize_index_by(indexed_specs[0].index_by)
        if index_by == ("subject",) and any(
            "ear" in sanitize_grouped_by(spec.grouped_by) for spec in grouped_specs
        ):
            index_by = ("subject", "ear")
        if "ear" not in index_by:
            for spec in grouped_specs:
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
            for spec in indexed_specs:
                if "ear" not in sanitize_index_by(spec.index_by):
                    continue
                descriptor = get_spec_descriptor(spec)
                if descriptor.ear_selectable:
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
                for spec in ear_selectable_specs:
                    descriptor = get_spec_descriptor(spec)
                    if descriptor.resource_name not in {"anthropometry", "metadata"}:
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
                for spec in position_selectable_specs
            ),
            position_index=any(
                bool(spec.position_index)
                for spec in position_selectable_specs
            ),
            frequency_one_hot=any(
                bool(spec.frequency_one_hot)
                for spec in indexed_specs
                if "frequency" in get_supported_index(spec)[0]
            ),
            frequency_index=any(
                bool(spec.frequency_index)
                for spec in indexed_specs
                if "frequency" in get_supported_index(spec)[0]
            ),
            sample_one_hot=any(
                bool(spec.sample_one_hot)
                for spec in indexed_specs
                if "samples" in get_supported_index(spec)[0]
            ),
            sample_index=any(
                bool(spec.sample_index)
                for spec in indexed_specs
                if "samples" in get_supported_index(spec)[0]
            ),
            ear_one_hot=any(
                bool(spec.ear_one_hot)
                for spec in ear_selectable_specs
            ),
            ear_index=any(
                bool(spec.ear_index)
                for spec in ear_selectable_specs
            ),
        )

    @staticmethod
    def get_spec_name(
        spec: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
    ) -> str:
        """Resolve the public sample key for a dataset spec.

        The workflow exposes this wrapper so call sites do not depend directly on
        registry internals. It keeps input/target dictionary naming, duplicate-name
        checks, and value assignment aligned with the central spec registry.

        Parameters
        ----------
        spec : dataset spec
            Spec object whose name should be resolved.

        Returns
        -------
        str Explicit spec name when provided, otherwise the registry default.

        Use Cases
        ---------
        - Build ``sample['inputs']`` and ``sample['target']`` dictionaries.
        - Validate duplicate spec names.
        - Preserve custom keys from named specs.
        """

        return get_spec_name(spec)
