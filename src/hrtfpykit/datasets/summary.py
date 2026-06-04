from typing import Any, cast

from .specs_registry import has_specs


def _summary_title(text: str, width: int = 54, marker: str = "=") -> str:
    """Format a centered title line for plain-text summaries.

    The summary helpers return plain text so their output can be printed in
    terminals, notebooks, logs, and exception messages. This helper strips the
    title text, adds one leading and trailing space around it, and pads both sides
    with the selected marker until the requested width is reached.

    Parameters
    ----------
    text : str
        Title text to center. Leading and trailing whitespace is removed before
        formatting.
    width : int, default=54
        Desired output width. If width is zero or negative, the stripped text is
        returned without padding.
    marker : str, default=``=``
        Padding string repeated on both sides of the title.

    Returns
    -------
    str
        Formatted title line. If the spaced title is already at least as wide as
        width, the spaced title is returned without marker padding.

    Notes
    -----
    This function is intentionally independent from dataset state. It is shared by
    resource, dataset, and download summaries to keep their headers visually
    consistent.

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
    """Create resource scan records or a formatted dataset resource summary.

    This function has two modes. Scanner mode is used by resource discovery code
    and returns a compact dictionary containing checked, found, missing, and
    missing subject counts. Dataset mode reads the constructed
    :class:`~hrtfpykit.datasets.state.DatasetState`, filters stored resource
    summaries to the resource families required by active specs, and returns a
    plain text report. Dataset reports show subject coverage first for every
    resource family and physical file counts at the end when available.

    Parameters
    ----------
    dataset : object or None, default=None
        Dataset instance to summarize. When None, scanner mode is used and the
        count arguments are returned as a dictionary.
    checked : int, default=0
        Number of resource entries checked during scanner mode.
    found : int, default=0
        Number of resource entries found during scanner mode.
    missing : int, default=0
        Number of resource entries missing during scanner mode.
    missing_subject_ids : tuple or list of str, default=()
        Subject IDs missing this resource during scanner mode. Values are stored as
        a tuple in the returned dictionary.

    Returns
    -------
    dict or str
        Resource summary dictionary when dataset is None. Otherwise, a formatted
        summary string containing only resources required by the active specs.

    Notes
    -----
    Dataset mode uses :func:`~hrtfpykit.datasets.specs_registry.has_specs` to decide
    whether ``hrtf``, ``mesh``, ``anthropometry``, ``metadata``, ``image``, or
    ``video`` should appear in the formatted output. If the constructed state
    contains no applicable resource records, the summary reports ``none``.

    """

    if dataset is None:
        summary: dict[str, object] = {
            "checked": checked,
            "found": found,
            "missing": missing,
            "missing_subject_ids": tuple(missing_subject_ids),
        }
        return summary

    state = cast(Any, dataset)._state
    used_resource_specs = {
        "hrtf": has_specs(state.specs, resource_name="hrtf"),
        "mesh": has_specs(state.specs, resource_name="mesh"),
        "anthropometry": has_specs(state.specs, resource_name="anthropometry"),
        "metadata": has_specs(state.specs, resource_name="metadata"),
        "image": has_specs(state.specs, resource_name="image"),
        "video": has_specs(state.specs, resource_name="video"),
    }
    if len(state.resource_summary) == 0:
        return (
            f"{_summary_title('DATASET RESOURCES SUMMARY')}\n"
            "  none\n"
            "  status: no resource specs requested"
        )
    resource_lines: list[str] = []
    for resource_name, summary in state.resource_summary.items():
        if not used_resource_specs.get(resource_name, False):
            continue
        if "subjects_checked" in summary:
            resource_lines.append(f"  {resource_name}:")
            subject_parts = [f"checked={summary['subjects_checked']!r}"]
            if "subjects_available" in summary:
                subject_parts.append(f"available={summary['subjects_available']!r}")
            if "subjects_missing" in summary:
                subject_parts.append(f"missing={summary['subjects_missing']!r}")
            resource_lines.append(f"    subjects: {', '.join(subject_parts)}")
            if "files" in summary:
                resource_lines.append(f"    files: {summary['files']!r}")
        else:
            parts = [str(resource_name)]
            for key in ("checked", "found", "missing"):
                if key in summary:
                    value = summary[key]
                    if value is None:
                        continue
                    parts.append(f"{key}={value!r}")
            resource_lines.append("  " + parts[0] + ": " + ", ".join(parts[1:]))
    if len(resource_lines) == 0:
        return (
            f"{_summary_title('DATASET RESOURCES SUMMARY')}\n"
            "  none\n"
            "  status: no resource specs requested"
        )
    return "\n".join([_summary_title("DATASET RESOURCES SUMMARY:")] + resource_lines)


def dataset_summary(dataset: object) -> str:
    """Create a formatted summary for a constructed dataset.

    The summary is built from final
    :class:`~hrtfpykit.datasets.state.DatasetState` values, not from constructor
    arguments. It therefore reflects configured exclusions, resource intersection,
    split selection, normalized input and target specs, selected variants, and
    acoustic metadata after dataset construction has completed.

    Parameters
    ----------
    dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
        Dataset instance with initialized dataset state.

    Returns
    -------
    str
        Human-readable dataset summary containing the root path, split, subject
        counts, sample count, input keys, target keys, selected HRTF or mesh variants
        when relevant, sample rate when known, and position count when known.

    Notes
    -----
    Variant lines are included only when the active specs require the corresponding
    resource family. This keeps summaries for table-only, image-only, or metadata
    workflows from showing unrelated HRTF or mesh configuration.

    """

    state = cast(Any, dataset)._state
    used_resources = tuple(
        resource_name
        for resource_name in ("hrtf", "mesh", "anthropometry", "metadata", "image", "video")
        if has_specs(state.specs, resource_name=resource_name)
    )
    uses_hrtf = "hrtf" in used_resources
    uses_mesh = "mesh" in used_resources
    has_sample_values = len(state.input_specs) > 0 or len(state.target_specs) > 0
    lines: list[str] = [_summary_title(f"{str(state.name).upper()} DATASET SUMMARY")]
    if has_sample_values:
        lines.extend(
            [
                f"  root: {state.root}",
                f"  split: {state.split}",
                f"  available_subjects: {len(state.available_subjects)}",
                f"  selected_subjects: {len(state.selected_subjects)}",
                f"  excluded_subjects: {len(state.excluded_subjects)}",
                f"  required_resources: {', '.join(used_resources) if len(used_resources) > 0 else 'none'}",
                f"  samples: {len(state.rows)}",
                f"  inputs: {', '.join(state.input_names) if len(state.input_specs) > 0 else 'none'}",
                f"  target: {', '.join(state.target_names) if len(state.target_specs) > 0 else 'none'}",
            ]
        )
    else:
        lines.extend(
            [
                f"  root: {state.root}",
                f"  split: {state.split}",
                "  available_subjects: none",
                "  selected_subjects: none",
                "  excluded_subjects: none",
                "  required_resources: none",
                "  samples: none",
                "  inputs: none",
                "  target: none",
                "  status: no input or target specs requested",
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

    The summary aggregates the download plan, downloaded files, verified existing
    files, failures, subjects, requested resources, and selected variants. Download
    code returns it after successful runs and embeds it in raised errors when one or
    more jobs fail.

    Parameters
    ----------
    config : :class:`~hrtfpykit.datasets.config.DatasetConfig`
        Dataset configuration used by the downloader. The summary reads the dataset
        name from this object.
    root : str or Path
        Local download root reported in the summary.
    download_jobs : list of dict
        Planned download job records. Each job is expected to contain a ``resource``
        entry and may contain ``subject_id``, ``subject_ids``, ``resource_count``,
        ``hrtf_variant``, and ``mesh_variant`` entries.
    downloaded_count : int
        Number of files downloaded in this run.
    verified_count : int
        Number of existing files verified without download.
    failures : list of str
        Download or verification failure messages.

    Returns
    -------
    str
        Human-readable download summary. The output always includes root and
        planned-file count. When files were planned, it also includes downloaded,
        verified, failed, subject, resource, variant, and status lines.

    Notes
    -----
    HRTF and mesh variant dictionaries are rendered as comma-separated key-value
    pairs with None values omitted. Failure details are limited to the first five
    messages so the summary remains readable when many jobs fail.

    """

    planned_files = len(download_jobs)
    resources: dict[str, int] = {}
    subject_ids: set[str] = set()
    hrtf_variants: set[str] = set()
    mesh_variants: set[str] = set()
    for job in download_jobs:
        resource = str(job["resource"])
        resource_count = int(cast(Any, job).get("resource_count", 1))
        resources[resource] = resources.get(resource, 0) + resource_count
        subject_id = job.get("subject_id")
        if subject_id is not None:
            subject_ids.add(str(subject_id))
        job_subject_ids = job.get("subject_ids")
        if isinstance(job_subject_ids, (tuple, list, set)):
            subject_ids.update(str(value) for value in job_subject_ids)
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
        lines.append("  status: nothing to download")
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
