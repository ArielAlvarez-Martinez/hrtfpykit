from .specs_registry import has_specs


def _summary_title(text: str, width: int = 54, marker: str = "=") -> str:
    cleaned = str(text).strip()
    if width <= 0:
        return cleaned
    content = f" {cleaned} "
    if len(content) >= width:
        return content
    padding = width - len(content)
    left = padding // 2
    right = padding - left
    return f"{marker * left}{content}{marker * right}"


def resources_summary(
    dataset: object | None = None,
    *,
    checked: int = 0,
    found: int = 0,
    missing: int = 0,
    missing_subject_ids: tuple[str, ...] | list[str] = tuple(),
) -> dict[str, object] | str:
    if dataset is None:
        summary: dict[str, object] = {
            "checked": checked,
            "found": found,
            "missing": missing,
            "missing_subject_ids": tuple(missing_subject_ids),
        }
        return summary

    state = dataset._state
    used_resource_specs = {
        "hrtf": has_specs(state.specs, resource_name="hrtf"),
        "mesh": has_specs(state.specs, resource_name="mesh"),
        "anthropometry": has_specs(state.specs, resource_name="anthropometry"),
        "metadata": has_specs(state.specs, resource_name="metadata"),
        "image": has_specs(state.specs, resource_name="image"),
        "video": has_specs(state.specs, resource_name="video"),
    }
    if len(state.resource_summary) == 0:
        return f"{_summary_title('DATASET RESOURCES SUMMARY')}\n  none"
    resource_lines: list[str] = []
    for resource_name, summary in state.resource_summary.items():
        if not used_resource_specs.get(resource_name, False):
            continue
        parts = [str(resource_name)]
        for key in ("checked", "found", "missing"):
            if key in summary:
                value = summary[key]
                if value is None:
                    continue
                parts.append(f"{key}={value!r}")
        resource_lines.append("  " + parts[0] + ": " + ", ".join(parts[1:]))
    if len(resource_lines) == 0:
        return f"{_summary_title('DATASET RESOURCES SUMMARY')}\n  none"
    return "\n".join([_summary_title("DATASET RESOURCES SUMMARY:")] + resource_lines)


def dataset_summary(dataset: object) -> str:
    state = dataset._state
    uses_hrtf = has_specs(state.specs, resource_name="hrtf")
    uses_mesh = has_specs(state.specs, resource_name="mesh")
    lines: list[str] = [_summary_title(f"{str(state.name).upper()} DATASET SUMMARY")]
    lines.extend(
        [
            f"  root: {state.root}",
            f"  split: {state.split}",
            f"  available_subjects: {len(state.available_subjects)}",
            f"  selected_subjects: {len(state.selected_subjects)}",
            f"  excluded_subjects: {len(state.excluded_subjects)}",
            f"  samples: {len(state.rows)}",
            f"  inputs: {', '.join(state.input_names) if len(state.input_specs) > 0 else 'none'}",
            f"  target: {', '.join(state.target_names) if len(state.target_specs) > 0 else 'none'}",
        ]
    )
    if uses_hrtf and state.dataset_hrtf_type is not None:
        lines.append(f"  hrtf_type: {state.dataset_hrtf_type}")
    if uses_hrtf and state.dataset_hrtf_sample_rate is not None:
        lines.append(f"  hrtf_sample_rate: {state.dataset_hrtf_sample_rate}")
    if uses_hrtf and state.dataset_hrtf_version is not None:
        lines.append(f"  hrtf_version: {state.dataset_hrtf_version}")
    if uses_mesh and state.dataset_mesh_type is not None:
        lines.append(f"  mesh_type: {state.dataset_mesh_type}")
    if uses_mesh and state.dataset_mesh_version is not None:
        lines.append(f"  mesh_version: {state.dataset_mesh_version}")
    if state.sample_rate is not None:
        lines.append(f"  sample_rate: {state.sample_rate}")
    if state.positions is not None:
        lines.append(f"  positions: {len(state.positions)}")
    return "\n".join(lines)


def download_summary(
    config,
    root,
    download_jobs: list[dict[str, object]],
    downloaded_count: int,
    verified_count: int,
    failures: list[str],
) -> str:
    planned_files = len(download_jobs)
    resources: dict[str, int] = {}
    subject_ids: set[str] = set()
    hrtf_types: set[str] = set()
    hrtf_sample_rates: set[str] = set()
    hrtf_versions: set[str] = set()
    mesh_types: set[str] = set()
    mesh_versions: set[str] = set()
    for job in download_jobs:
        resource = str(job["resource"])
        resources[resource] = resources.get(resource, 0) + 1
        subject_id = job.get("subject_id")
        if subject_id is not None:
            subject_ids.add(str(subject_id))
        hrtf_type = job.get("hrtf_type")
        if hrtf_type is not None:
            hrtf_types.add(str(hrtf_type))
        hrtf_sample_rate = job.get("hrtf_sample_rate")
        if hrtf_sample_rate is not None:
            hrtf_sample_rates.add(str(hrtf_sample_rate))
        hrtf_version = job.get("hrtf_version")
        if hrtf_version is not None:
            hrtf_versions.add(str(hrtf_version))
        mesh_type = job.get("mesh_type")
        if mesh_type is not None:
            mesh_types.add(str(mesh_type))
        mesh_version = job.get("mesh_version")
        if mesh_version is not None:
            mesh_versions.add(str(mesh_version))
    lines = [
        _summary_title(f"{str(config.name).upper()} DOWNLOAD SUMMARY"),
        f"  root: {root}",
        f"  planned_files: {planned_files}",
    ]
    if planned_files == 0:
        lines.append(f"  status: nothing to download")
    else:
        lines.extend(
            [
                f"  downloaded_files: {downloaded_count}",
                f"  verified_existing_files: {verified_count}",
                f"  failed_files: {len(failures)}",
                f"  subjects: {len(subject_ids)}",
            ]
        )
        if len(hrtf_types) > 0:
            lines.append(f"  hrtf_types: {', '.join(sorted(hrtf_types))}")
        if len(hrtf_sample_rates) > 0:
            lines.append(f"  hrtf_sample_rates: {', '.join(sorted(hrtf_sample_rates))}")
        if len(hrtf_versions) > 0:
            lines.append(f"  hrtf_versions: {', '.join(sorted(hrtf_versions))}")
        if len(mesh_types) > 0:
            lines.append(f"  mesh_types: {', '.join(sorted(mesh_types))}")
        if len(mesh_versions) > 0:
            lines.append(f"  mesh_versions: {', '.join(sorted(mesh_versions))}")
        if len(resources) > 0:
            lines.append(
                "  resources: "
                + ", ".join(
                    f"{resource}={count}" for resource, count in sorted(resources.items())
                )
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
