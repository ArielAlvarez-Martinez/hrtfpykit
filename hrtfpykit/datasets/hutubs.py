from pathlib import Path

from .base import BaseDataset
from .config import HUTUBS_CONFIG
from .download import BaseDownload
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    MeshSpec,
    VideoSpec,
    normalize_anthropometry_ear,
    normalize_anthropometry_select,
)

class HUTUBS(BaseDataset):
    config = HUTUBS_CONFIG

    @staticmethod
    def build_anthropometry_column_maps(
        values: dict[str, float | str | None],
        left_prefix: str,
        right_prefix: str,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
        exact_lookup: dict[str, str] = {}
        neutral_lookup: dict[str, str] = {}
        left_lookup: dict[str, str] = {}
        right_lookup: dict[str, str] = {}
        for name in values:
            lowered_name = str(name).lower()
            exact_lookup[lowered_name] = name
            text = str(name)
            if text.startswith(left_prefix):
                left_lookup[text[len(left_prefix):].lower()] = name
            elif text.startswith(right_prefix):
                right_lookup[text[len(right_prefix):].lower()] = name
            else:
                neutral_lookup[lowered_name] = name
        return exact_lookup, neutral_lookup, left_lookup, right_lookup

    @staticmethod
    def get_anthropometry_search_scope(
        ear: str,
        left_prefix: str,
        right_prefix: str,
    ) -> str:
        if ear == "left":
            return f"shared columns or left-ear columns with prefix {left_prefix!r}"
        if ear == "right":
            return f"shared columns or right-ear columns with prefix {right_prefix!r}"
        return (
            f"shared columns, left-ear columns with prefix {left_prefix!r}, "
            f"or right-ear columns with prefix {right_prefix!r}"
        )

    def __init__(
        self,
        root: str | Path,
        variant: str = "measured",
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_variant: str = "all",
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
        | None = None,
        target: HRTFSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
        | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
    ) -> None:
        self.variant = str(variant).strip().lower()
        if self.variant not in self.config.hrtf.variants:
            raise ValueError(
                f"Unsupported variant {self.variant!r}. Expected one of {self.config.hrtf.variants}"
            )
        resolved_exclude_subject_ids: tuple[str, ...]
        if exclude_subject_ids is None:
            resolved_exclude_subject_ids = tuple()
        elif isinstance(exclude_subject_ids, (str, int)):
            resolved_exclude_subject_ids = (
                type(self).resolve_dataset_subject_id(exclude_subject_ids, tuple(self.config.subject_ids)),
            )
        else:
            resolved_exclude_subject_ids = tuple(
                dict.fromkeys(
                    type(self).resolve_dataset_subject_id(subject_id, tuple(self.config.subject_ids))
                    for subject_id in exclude_subject_ids
                )
            )
        if download:
            BaseDownload(
                config=self.config,
                root=root,
                excluded_subject_ids=resolved_exclude_subject_ids,
            ).download(
                download_resources=download_resources,
                download_hrtf_variant=download_hrtf_variant,
            )
        super().__init__(
            root=root,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )

    def get_anthropometry_value(
        self,
        spec: AnthropometrySpec,
        subject_id: str,
    ) -> dict[str, float | str | None]:
        values = self._anthropometry_rows[subject_id]
        selected = normalize_anthropometry_select(spec.select)
        ear = normalize_anthropometry_ear(spec.ear)
        if self.config is None or self.config.anthropometry is None:
            raise ValueError("HUTUBS anthropometry config is missing")
        left_prefix = str(self.config.anthropometry.left_prefix)
        right_prefix = str(self.config.anthropometry.right_prefix)

        if selected == "complete":
            selected_values: dict[str, float | str | None] = {}
            for name, value in values.items():
                text = str(name)
                if text.startswith(left_prefix):
                    if ear in {"left", "both"}:
                        selected_values[name] = value
                    continue
                if text.startswith(right_prefix):
                    if ear in {"right", "both"}:
                        selected_values[name] = value
                    continue
                selected_values[name] = value
            return selected_values

        exact_lookup, neutral_lookup, left_lookup, right_lookup = (
            self.build_anthropometry_column_maps(
                values,
                left_prefix,
                right_prefix,
            )
        )

        selected_keys: list[str] = []
        seen: set[str] = set()
        missing_messages: list[str] = []
        searched_locations = self.get_anthropometry_search_scope(
            ear,
            left_prefix,
            right_prefix,
        )

        for requested in selected:
            requested_text = str(requested).strip()
            requested_key = requested_text.lower()
            exact_name = exact_lookup.get(requested_key)
            if exact_name is not None:
                exact_text = str(exact_name)
                if exact_text.startswith(left_prefix):
                    column_description = f"left-ear column {exact_name!r}"
                    include_column = ear in {"left", "both"}
                elif exact_text.startswith(right_prefix):
                    column_description = f"right-ear column {exact_name!r}"
                    include_column = ear in {"right", "both"}
                else:
                    column_description = f"shared column {exact_name!r}"
                    include_column = True
                if include_column:
                    if exact_name not in seen:
                        seen.add(exact_name)
                        selected_keys.append(exact_name)
                    continue
                missing_messages.append(
                    f"{requested_text!r} matched "
                    f"{column_description}, "
                    f"but ear={ear!r} excludes it"
                )
                continue

            neutral_name = neutral_lookup.get(requested_key)
            if neutral_name is not None:
                if neutral_name not in seen:
                    seen.add(neutral_name)
                    selected_keys.append(neutral_name)
                continue

            matched = False
            if ear in {"left", "both"}:
                left_name = left_lookup.get(requested_key)
                if left_name is not None:
                    if left_name not in seen:
                        seen.add(left_name)
                        selected_keys.append(left_name)
                    matched = True
            if ear in {"right", "both"}:
                right_name = right_lookup.get(requested_key)
                if right_name is not None:
                    if right_name not in seen:
                        seen.add(right_name)
                        selected_keys.append(right_name)
                    matched = True
            if not matched:
                missing_messages.append(
                    f"{requested_text!r} was not found in {searched_locations}"
                )

        if len(missing_messages) > 0:
            raise ValueError(
                "Anthropometry select values could not be resolved for HUTUBS: "
                + "; ".join(missing_messages)
            )
        return {name: values[name] for name in selected_keys}
