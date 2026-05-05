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
    if state.hrtf_variant is not None:
        lines.append(f"  hrtf_variant: {state.hrtf_variant}")
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
    hrtf_variants: set[str] = set()
    for job in download_jobs:
        resource = str(job["resource"])
        resources[resource] = resources.get(resource, 0) + 1
        subject_id = job.get("subject_id")
        if subject_id is not None:
            subject_ids.add(str(subject_id))
        hrtf_variant = job.get("hrtf_variant")
        if hrtf_variant is not None:
            hrtf_variants.add(str(hrtf_variant))
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
        if len(hrtf_variants) > 0:
            lines.append(f"  hrtf_variants: {', '.join(sorted(hrtf_variants))}")
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
