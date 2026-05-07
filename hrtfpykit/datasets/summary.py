from .specs_registry import has_specs


def _summary_title(text: str, width: int = 54, marker: str = "=") -> str:
    """Format a centered summary title.

    Summary output is intentionally plain text so it works in terminals, notebooks,
    and logs. This helper formats the shared title line used by resource, dataset,
    and download summaries.

    Parameters
    ----------
    text : str
        Title text.
    width : int
        Desired output width.
    marker : str
        Padding marker.

    Returns
    -------
    str Formatted title line.

    Use Cases
    ---------
    - Format dataset, resource, and download summary titles.
    """
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
    """Create a resource summary dictionary or formatted dataset resource summary.

    The function has two modes: scanner mode returns a dictionary from
    counts, and dataset mode formats the stored summaries for resources actually
    used by selected specs. This keeps scanner bookkeeping and user-facing summary
    text aligned.

    Parameters
    ----------
    dataset : object or None, default=None
        Dataset instance to summarize. If ``None``, returns a resource summary
        dictionary from the count arguments.
    checked : int, default=0
        Number of resource entries checked.
    found : int, default=0
        Number of resource entries found.
    missing : int, default=0
        Number of resource entries missing.
    missing_subject_ids : tuple or list of str, default=()
        Subject IDs missing this resource.

    Returns
    -------
    dict or str Resource summary dictionary when ``dataset`` is ``None``;
    otherwise a formatted summary string.

    Use Cases
    ---------
    - Store scanner count results in dataset state.
    - Print a user-facing resource summary.
    - Keep missing-subject information attached to resource names.
    """

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
    """Create a formatted summary for a constructed dataset.

    The summary is built from final dataset state, not constructor arguments, so
    it reflects exclusions, resource intersection, split selection, selected
    specs, variants, and acoustic metadata after construction. It is the main
    summary view for a dataset instance.

    Parameters
    ----------
    dataset : object
        Dataset instance with initialized dataset state.

    Returns
    -------
    str Human-readable dataset summary.

    Use Cases
    ---------
    - Inspect selected specs, subject counts, and split information.
    - Confirm selected HRTF and mesh variants.
    - Log dataset construction in scripts or notebooks.
    """

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
    if uses_hrtf and state.dataset_hrtf_variant is not None:
        if isinstance(state.dataset_hrtf_variant, dict):
            hrtf_variant = ", ".join(
                f"{key}={value}"
                for key, value in state.dataset_hrtf_variant.items()
                if value is not None
            )
        else:
            hrtf_variant = str(state.dataset_hrtf_variant)
        lines.append(f"  hrtf_variant: {hrtf_variant}")
    if uses_mesh and state.dataset_mesh_variant is not None:
        if isinstance(state.dataset_mesh_variant, dict):
            mesh_variant = ", ".join(
                f"{key}={value}"
                for key, value in state.dataset_mesh_variant.items()
                if value is not None
            )
        else:
            mesh_variant = str(state.dataset_mesh_variant)
        lines.append(f"  mesh_variant: {mesh_variant}")
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
    """Create a formatted summary for a dataset download operation.

    The summary aggregates planned jobs, downloaded files, verified existing
    files, failures, subjects, resources, and selected variants. It is returned
    after successful downloads and embedded in raised errors when any job fails.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration used by the downloader.
    root : str or Path
        Local download root.
    download_jobs : list of dict
        Planned download job records.
    downloaded_count : int
        Number of files downloaded in this run.
    verified_count : int
        Number of existing files verified without download.
    failures : list of str
        Download or verification failure messages.

    Returns
    -------
    str Human-readable download summary.

    Use Cases
    ---------
    - Print download results from concrete dataset constructors.
    - Raise a detailed error when one or more downloads fail.
    - Report selected HRTF or mesh variants in download workflows.
    """

    planned_files = len(download_jobs)
    resources: dict[str, int] = {}
    subject_ids: set[str] = set()
    hrtf_variants: set[str] = set()
    mesh_variants: set[str] = set()
    for job in download_jobs:
        resource = str(job["resource"])
        resources[resource] = resources.get(resource, 0) + 1
        subject_id = job.get("subject_id")
        if subject_id is not None:
            subject_ids.add(str(subject_id))
        hrtf_variant = job.get("hrtf_variant")
        if isinstance(hrtf_variant, dict):
            hrtf_variants.add(
                ", ".join(
                    f"{key}={value}"
                    for key, value in hrtf_variant.items()
                    if value is not None
                )
            )
        elif hrtf_variant is not None:
            hrtf_variants.add(str(hrtf_variant))
        mesh_variant = job.get("mesh_variant")
        if isinstance(mesh_variant, dict):
            mesh_variants.add(
                ", ".join(
                    f"{key}={value}"
                    for key, value in mesh_variant.items()
                    if value is not None
                )
            )
        elif mesh_variant is not None:
            mesh_variants.add(str(mesh_variant))
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
            lines.append(f"  hrtf_variants: {'; '.join(sorted(hrtf_variants))}")
        if len(mesh_variants) > 0:
            lines.append(f"  mesh_variants: {'; '.join(sorted(mesh_variants))}")
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
