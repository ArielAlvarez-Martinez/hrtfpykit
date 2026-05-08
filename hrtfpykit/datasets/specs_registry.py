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


DatasetSpec = HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec


@dataclass(frozen=True)
class DatasetSpecDescriptor:
    """Describe one dataset spec type in the central registry.

    Parameters
    ----------
    spec_type : type
        Concrete spec class.
    name : str
        Default public sample key.
    resource_name : str
        Resource family used by the spec.
    indexed, acoustic, position_selectable, media, grouped, path_based, ear_selectable : bool
        Behavioral flags consumed by workflow, resources, and values.
    value_method_name : str
        Selector method name used to resolve sample values.

    Returns
    -------
    DatasetSpecDescriptor Immutable registry entry.
    """
    spec_type: type
    name: str
    resource_name: str
    indexed: bool
    acoustic: bool
    position_selectable: bool
    media: bool
    grouped: bool
    path_based: bool
    ear_selectable: bool
    value_method_name: str


SPEC_DESCRIPTORS = (
    DatasetSpecDescriptor(HRTFSpec, "hrtf", "hrtf", True, True, True, False, False, False, True, "get_hrtf_spec_value"),
    DatasetSpecDescriptor(ITDSpec, "itd", "hrtf", True, True, True, False, False, False, False, "get_itd_spec_value"),
    DatasetSpecDescriptor(ILDSpec, "ild", "hrtf", True, True, True, False, False, False, False, "get_ild_spec_value"),
    DatasetSpecDescriptor(SHSpec, "sh", "hrtf", True, True, False, False, False, False, True, "get_sh_spec_value"),
    DatasetSpecDescriptor(MeshSpec, "mesh", "mesh", False, False, False, False, False, True, False, "get_mesh_spec_value"),
    DatasetSpecDescriptor(AnthropometrySpec, "anthropometry", "anthropometry", False, False, False, False, True, True, True, "get_anthropometry_spec_value"),
    DatasetSpecDescriptor(MetadataSpec, "metadata", "metadata", False, False, False, False, True, True, True, "get_metadata_spec_value"),
    DatasetSpecDescriptor(ImageSpec, "image", "image", False, False, False, True, True, True, True, "get_image_spec_value"),
    DatasetSpecDescriptor(VideoSpec, "video", "video", False, False, False, True, True, True, True, "get_video_spec_value"),
)


def get_spec_descriptor(spec: DatasetSpec) -> DatasetSpecDescriptor:
    """Return the registry descriptor for a spec object.

    This is the central type-to-behavior lookup for dataset specs. Workflow
    validation, resource scanning, split intersection, and value dispatch all use
    descriptors so spec taxonomy stays in one module instead of scattered
    ``isinstance`` branches.

    Parameters
    ----------
    spec : dataset spec
        Spec object to classify.

    Returns
    -------
    DatasetSpecDescriptor Descriptor matching the spec type.

    """
    for descriptor in SPEC_DESCRIPTORS:
        if isinstance(spec, descriptor.spec_type):
            return descriptor
    raise TypeError(f"Unsupported dataset spec: {type(spec)!r}")


def get_spec_name(spec: DatasetSpec) -> str:
    """Resolve the public name for a spec.

    A spec can provide an explicit ``name`` to control its sample key; otherwise
    the registry default is used. Centralizing this rule keeps duplicate-name
    validation and sample dictionary construction use the same naming rule.

    Parameters
    ----------
    spec : dataset spec
        Spec object to name.

    Returns
    -------
    str Explicit spec name or descriptor default.

    """
    explicit_name = getattr(spec, "name", None)
    if explicit_name is not None:
        name = str(explicit_name).strip()
        if name == "":
            raise ValueError("Dataset spec name must not be empty")
        return name
    return get_spec_descriptor(spec).name


def get_specs(
    specs: tuple[DatasetSpec, ...],
    *,
    resource_name: str | None = None,
    indexed: bool | None = None,
    acoustic: bool | None = None,
    position_selectable: bool | None = None,
    media: bool | None = None,
    grouped: bool | None = None,
    path_based: bool | None = None,
    ear_selectable: bool | None = None,
) -> tuple[DatasetSpec, ...]:
    """Filter specs by descriptor properties.

    Many build phases need to ask questions such as which specs are acoustic,
    which require media, or which use a resource family. This helper performs
    those descriptor-based filters without duplicating spec-type logic in each
    phase.

    Parameters
    ----------
    specs : tuple of specs
        Specs to filter.
    resource_name : str or None, default=None
        Resource family filter.
    indexed, acoustic, position_selectable : bool or None, default=None
        Descriptor property filters.
    media, grouped, path_based, ear_selectable : bool or None, default=None
        Descriptor property filters.

    Returns
    -------
    tuple of specs
        Specs matching all provided filters.

    """
    selected_specs: list[DatasetSpec] = []
    for spec in specs:
        descriptor = get_spec_descriptor(spec)
        if resource_name is not None and descriptor.resource_name != resource_name:
            continue
        if indexed is not None and descriptor.indexed != indexed:
            continue
        if acoustic is not None and descriptor.acoustic != acoustic:
            continue
        if position_selectable is not None and descriptor.position_selectable != position_selectable:
            continue
        if media is not None and descriptor.media != media:
            continue
        if grouped is not None and descriptor.grouped != grouped:
            continue
        if path_based is not None and descriptor.path_based != path_based:
            continue
        if ear_selectable is not None and descriptor.ear_selectable != ear_selectable:
            continue
        selected_specs.append(spec)
    return tuple(selected_specs)


def has_specs(
    specs: tuple[DatasetSpec, ...],
    *,
    resource_name: str | None = None,
    indexed: bool | None = None,
    acoustic: bool | None = None,
    position_selectable: bool | None = None,
    media: bool | None = None,
    grouped: bool | None = None,
    path_based: bool | None = None,
    ear_selectable: bool | None = None,
) -> bool:
    """Return whether any specs match descriptor filters.

    This convenience wrapper is used when a phase only needs a yes/no answer, such
    as whether to scan a resource or include a summary entry. It keeps boolean
    resource decisions aligned with ``get_specs`` filtering.

    Parameters
    ----------
    specs : tuple of specs
        Specs to inspect. **filters Descriptor filters.

    Returns
    -------
    bool ``True`` when at least one spec matches.

    """
    return len(
        get_specs(
            specs,
            resource_name=resource_name,
            indexed=indexed,
            acoustic=acoustic,
            position_selectable=position_selectable,
            media=media,
            grouped=grouped,
            path_based=path_based,
            ear_selectable=ear_selectable,
        )
    ) > 0


def get_supported_index(spec: DatasetSpec) -> tuple[set[str], str]:
    """Return supported row axes for an indexed spec.

    Different specs support different row axes depending on what value they
    produce; for example time-domain HRTF specs support samples while frequency-
    domain specs support frequency. This helper provides both machine-checkable
    axes and human-readable combinations for errors.

    Parameters
    ----------
    spec : dataset spec
        Spec to inspect.

    Returns
    -------
    tuple Supported axis set and human-readable combinations.

    """
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
        return supported_axes, supported_index_by
    if isinstance(spec, ITDSpec):
        return {"position"}, "('subject',), ('subject', 'position')"
    if isinstance(spec, ILDSpec):
        supported_axes = {"position"}
        if str(spec.mode).strip().lower() == "frequency-dependent":
            supported_axes.add("frequency")
            return supported_axes, (
                "('subject',), ('subject', 'position'), ('subject', 'frequency'), "
                "('subject', 'position', 'frequency')"
            )
        return supported_axes, "('subject',), ('subject', 'position')"
    if isinstance(spec, SHSpec):
        return {"ear", "frequency"}, (
            "('subject',), ('subject', 'ear'), ('subject', 'frequency'), "
            "('subject', 'ear', 'frequency')"
        )
    return set(), "()"


def get_axis_compatibility_hint(spec: DatasetSpec, axis_name: str) -> str:
    """Return a hint for unsupported axis usage.

    When validation rejects an axis, the error also needs configuration guidance.
    This helper adds spec-specific guidance, such as switching HRTF domain before
    using frequency indexing.

    Parameters
    ----------
    spec : dataset spec
        Spec being validated.
    axis_name : str
        Unsupported axis name.

    Returns
    -------
    str Additional compatibility hint, or an empty string.

    """
    if isinstance(spec, HRTFSpec):
        domain = str(spec.domain).strip().lower()
        if axis_name == "frequency" and domain == "time":
            return " In HRTFSpec, the 'frequency' axis is available only when domain='frequency'."
        if axis_name == "samples" and domain == "frequency":
            return " In HRTFSpec, the 'samples' axis is available only when domain='time'."
    if isinstance(spec, ILDSpec) and axis_name == "frequency":
        if str(spec.mode).strip().lower() != "frequency-dependent":
            return " In ILDSpec, enable frequency indexing by setting mode='frequency-dependent'."
    return ""


def get_flag_compatibility_hint(spec: DatasetSpec, axis_name: str) -> str:
    """Return a hint for unsupported context flag usage.

    One-hot and index flags require the corresponding row axis. This helper adds
    spec-specific guidance when a flag is incompatible with the current spec
    configuration.

    Parameters
    ----------
    spec : dataset spec
        Spec being validated.
    axis_name : str
        Axis required by a context flag.

    Returns
    -------
    str Additional compatibility hint, or an empty string.

    """
    if isinstance(spec, HRTFSpec):
        domain = str(spec.domain).strip().lower()
        if axis_name == "frequency" and domain == "time":
            return " In HRTFSpec, set domain='frequency' to use frequency-indexed specs."
        if axis_name == "samples" and domain == "frequency":
            return " In HRTFSpec, set domain='time' to use sample-indexed specs."
    if isinstance(spec, ILDSpec) and axis_name == "frequency":
        return " In ILDSpec, set mode='frequency-dependent' to use frequency indexing."
    return ""
