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

def resources_summary(
    *,
    checked: int = 0,
    found: int = 0,
    missing: int = 0,
    missing_subject_ids: tuple[str, ...] | list[str] = tuple(),
) -> dict[str, object]:
    summary: dict[str, object] = {
        "checked": checked,
        "found": found,
        "missing": missing,
        "missing_subject_ids": tuple(missing_subject_ids),
    }
    return summary


def dataset_summary(dataset: object) -> str:
    resource_lines: list[str] = []
    if len(dataset._resource_summary) == 0:
        resource_lines.append("Dataset Resources Summary: none")
    else:
        used_resource_specs = {
            "hrtf": len(dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))) > 0,
            "mesh": len(dataset._get_specs(MeshSpec)) > 0,
            "anthropometry": len(dataset._get_specs(AnthropometrySpec)) > 0,
            "image": len(dataset._get_specs(ImageSpec)) > 0,
            "video": len(dataset._get_specs(VideoSpec)) > 0,
        }
        displayed_lines: list[str] = []
        for resource_name, summary in dataset._resource_summary.items():
            if not used_resource_specs.get(resource_name, False):
                continue
            parts = [str(resource_name)]
            for key in (
                "checked",
                "found",
                "missing",
            ):
                if key in summary:
                    if summary[key] is None:
                        continue
                    parts.append(f"{key}={summary[key]!r}")
            displayed_lines.append("  " + parts[0] + ": " + ", ".join(parts[1:]))
        if len(displayed_lines) > 0:
            resource_lines.append("Dataset Resources Summary:")
            resource_lines.extend(displayed_lines)
        else:
            resource_lines.append("Dataset Resources Summary: none")
    lines = [
        f"{str(dataset._name).upper()} Dataset Summary",
        f"  root: {dataset._root}",
        f"  split: {dataset._split}",
        f"  subjects_loaded: {len(dataset._subject_ids)}",
        f"  available_subjects: {len(dataset._available_subject_ids)}",
        f"  samples: {len(dataset._rows)}",
        f"  inputs: {', '.join(dataset._input_names) if len(dataset._input_specs) > 0 else 'none'}",
        f"  target: {', '.join(dataset._target_names) if len(dataset._target_specs) > 0 else 'none'}",
    ]
    if len(dataset._exclude_subject_ids) > 0:
        lines.append(f"  excluded_subjects: {len(dataset._exclude_subject_ids)}")
    if getattr(dataset, "variant", None) is not None:
        lines.append(f"  dataset_hrtf_variant: {dataset.variant}")
    if dataset._dataset_sample_rate is not None:
        lines.append(f"  dataset_sample_rate: {dataset._dataset_sample_rate}")
    if dataset._dataset_source_positions is not None:
        lines.append(f"  dataset_source_positions: {len(dataset._dataset_source_positions)}")
    return "\n".join(resource_lines + lines)


def download_summary(
    config,
    root,
    download_jobs: list[dict[str, object]],
    downloaded_count: int,
    verified_count: int,
    failures: list[str],
) -> str:
    resources: dict[str, int] = {}
    subject_ids: set[str] = set()
    variants: set[str] = set()
    for job in download_jobs:
        resource = str(job["resource"])
        resources[resource] = resources.get(resource, 0) + 1
        subject_id = job.get("subject_id")
        if subject_id is not None:
            subject_ids.add(str(subject_id))
        variant = job.get("variant")
        if variant is not None:
            variants.add(str(variant))
    lines = [
        f"{config.name} download summary",
        f"  root: {root}",
        f"  planned_files: {len(download_jobs)}",
        f"  downloaded_files: {downloaded_count}",
        f"  verified_existing_files: {verified_count}",
        f"  failed_files: {len(failures)}",
        f"  subjects: {len(subject_ids)}",
    ]
    if len(variants) > 0:
        lines.append(f"  variants: {', '.join(sorted(variants))}")
    if len(resources) > 0:
        lines.append(
            "  resources: "
            + ", ".join(f"{resource}={count}" for resource, count in sorted(resources.items()))
        )
    if len(failures) == 0:
        lines.append(f"  status: {config.name} dataset downloaded successfully")
    else:
        failure_preview = ", ".join(str(value) for value in failures[:5])
        if len(failures) > 5:
            failure_preview = f"{failure_preview}, ..."
        lines.append("  failure_examples: " + failure_preview)
        lines.append(f"  status: {config.name} dataset download finished with errors")
    return "\n".join(lines)
