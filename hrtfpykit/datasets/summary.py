class DatasetSummary:
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

    def format_load_summary(self) -> str:
        lines = [
            f"{self.name} dataset summary",
            f"  root: {self.root}",
            f"  split: {self.split}",
            f"  subjects_loaded: {len(self.subject_ids)}",
            f"  available_subjects: {len(self.available_subject_ids)}",
            f"  samples: {len(self._rows)}",
            f"  inputs: {', '.join(self.input_names) if len(self.input_specs) > 0 else 'none'}",
            f"  target: {', '.join(self.target_names) if len(self.target_specs) > 0 else 'none'}",
        ]
        if len(self.exclude_subject_ids) > 0:
            lines.append(f"  excluded_subjects: {len(self.exclude_subject_ids)}")
        if getattr(self, "variant", None) is not None:
            lines.append(f"  variant: {self.variant}")
        if self.dataset_sample_rate is not None:
            lines.append(f"  dataset_sample_rate: {self.dataset_sample_rate}")
        if self.dataset_source_positions is not None:
            lines.append(f"  dataset_source_positions: {len(self.dataset_source_positions)}")
        lines.append(self.format_resource_summary(self.resource_summary))
        return "\n".join(lines)
