from pathlib import Path

from .base import BaseDataset
from .download import BaseDownload
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    MeshSpec,
    VideoSpec,
)


HUTUBS_SUBJECT_IDS = tuple(f"pp{index}" for index in range(1, 97))
HUTUBS_MESH_SUBJECT_IDS = tuple(
    f"pp{index}"
    for index in (
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        16,
        19,
        20,
        21,
        22,
        23,
        29,
        30,
        31,
        32,
        33,
        40,
        41,
        44,
        45,
        46,
        47,
        48,
        49,
        55,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        76,
        77,
        78,
        80,
        81,
        82,
        88,
        89,
        90,
        91,
        95,
        96,
    )
)


class HUTUBSDownload(BaseDownload):
    dataset_name = "HUTUBS"
    dataset_base_url = "https://sofacoustics.org/data/database/hutubs"
    available_download_resources = ("all", "hrtf", "mesh", "anthropometry")

    def __init__(
        self,
        root: str | Path,
        excluded_subject_ids: tuple[str, ...] = tuple(),
        hrtf_spec: HRTFSpec | None = None,
        mesh_spec: MeshSpec | None = None,
        anthropometry_spec: AnthropometrySpec | None = None,
    ) -> None:
        super().__init__(root=root, excluded_subject_ids=excluded_subject_ids)
        self.hrtf_spec = hrtf_spec
        self.mesh_spec = mesh_spec
        self.anthropometry_spec = anthropometry_spec

    def build_download_plan(
        self,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_version: str = "all",
    ) -> list[tuple[str, Path, str | None]]:
        resources = self.normalize_download_resources(download_resources)
        download_jobs: list[tuple[str, Path, str | None]] = []

        if "hrtf" in resources:
            if self.hrtf_spec is None:
                raise ValueError(f"{self.dataset_name} does not provide official hrtf files")
            if self.hrtf_spec.download_pattern is None or self.hrtf_spec.download_subject_ids is None:
                raise ValueError(f"{self.dataset_name} hrtf spec is missing download metadata")
            subject_ids = self.get_included_subject_ids(self.hrtf_spec.download_subject_ids)
            for version in self.normalize_download_hrtf_versions(download_hrtf_version, self.hrtf_spec):
                for subject_id in subject_ids:
                    filename = self.hrtf_spec.download_pattern.format(
                        subject_id=subject_id,
                        variant=version,
                    )
                    destination = self.resolve_download_path(filename)
                    checksum = (
                        None
                        if self.hrtf_spec.download_checksums is None
                        else self.hrtf_spec.download_checksums.get(filename)
                    )
                    download_jobs.append(
                        (self.build_download_url(filename), destination, checksum)
                    )

        if "mesh" in resources:
            if self.mesh_spec is None:
                raise ValueError(f"{self.dataset_name} does not provide official mesh data")
            if self.mesh_spec.download_pattern is None or self.mesh_spec.download_subject_ids is None:
                raise ValueError(f"{self.dataset_name} mesh spec is missing download metadata")
            subject_ids = self.get_included_subject_ids(self.mesh_spec.download_subject_ids)
            for subject_id in subject_ids:
                filename = self.mesh_spec.download_pattern.format(subject_id=subject_id)
                destination = self.resolve_download_path(filename)
                checksum = (
                    None
                    if self.mesh_spec.download_checksums is None
                    else self.mesh_spec.download_checksums.get(filename)
                )
                download_jobs.append(
                    (self.build_download_url(filename), destination, checksum)
                )

        if "anthropometry" in resources:
            if self.anthropometry_spec is None:
                raise ValueError(f"{self.dataset_name} does not provide official anthropometry")
            if self.anthropometry_spec.download_filename is None:
                raise ValueError(f"{self.dataset_name} anthropometry spec is missing download metadata")
            filename = self.anthropometry_spec.download_filename
            destination = self.resolve_download_path(filename)
            download_jobs.append(
                (
                    self.build_download_url(filename),
                    destination,
                    self.anthropometry_spec.download_checksum,
                )
            )

        return download_jobs

    def get_hrtf_paths(self, variant: str) -> dict[str, Path]:
        if self.hrtf_spec is None:
            return {}
        if self.hrtf_spec.download_pattern is None or self.hrtf_spec.download_subject_ids is None:
            raise ValueError(f"{self.dataset_name} hrtf spec is missing download metadata")
        variant_key = str(variant).strip().lower()
        if self.hrtf_spec.variants is not None and variant_key not in self.hrtf_spec.variants:
            raise ValueError(
                f"Unsupported hrtf_variant {variant!r}. Expected one of {self.hrtf_spec.variants}"
            )
        subject_ids = self.get_included_subject_ids(self.hrtf_spec.download_subject_ids)
        paths: dict[str, Path] = {}
        for subject_id in subject_ids:
            filename = self.hrtf_spec.download_pattern.format(
                subject_id=subject_id,
                variant=variant_key,
            )
            path = self.resolve_download_path(filename)
            if path.is_file():
                paths[subject_id] = path
        return paths

    def get_mesh_paths(self) -> dict[str, Path]:
        if self.mesh_spec is None:
            return {}
        if self.mesh_spec.download_pattern is None or self.mesh_spec.download_subject_ids is None:
            raise ValueError(f"{self.dataset_name} mesh spec is missing download metadata")
        subject_ids = self.get_included_subject_ids(self.mesh_spec.download_subject_ids)
        paths: dict[str, Path] = {}
        for subject_id in subject_ids:
            filename = self.mesh_spec.download_pattern.format(subject_id=subject_id)
            candidates = [self.resolve_download_path(filename)]
            for extension in self.mesh_spec.extensions:
                candidate = self.resolve_download_path(str(Path(filename).with_suffix(extension)))
                if candidate not in candidates:
                    candidates.append(candidate)
            for candidate in candidates:
                if candidate.is_file():
                    paths[subject_id] = candidate
                    break
        return paths

    def get_anthropometry_path(self) -> Path | None:
        if self.anthropometry_spec is None:
            return None
        if self.anthropometry_spec.filename is None:
            raise ValueError(f"{self.dataset_name} anthropometry spec is missing filename")
        path = self.resolve_download_path(self.anthropometry_spec.filename)
        if path.is_file():
            return path
        return None


class HUTUBS(BaseDataset):
    dataset_name = "HUTUBS"
    dataset_subject_ids = HUTUBS_SUBJECT_IDS
    dataset_base_url = HUTUBSDownload.dataset_base_url
    dataset_download_resources = HUTUBSDownload.available_download_resources
    dataset_download_class = HUTUBSDownload
    dataset_hrtf_spec = HRTFSpec(
        aligned_by=("subject", "position", "ear"),
        variants=("measured", "simulated"),
        default_variant="measured",
        filename_pattern=r"^(?P<subject_id>pp\d+)_HRIRs_(?P<variant>measured|simulated)\.sofa$",
        download_pattern="{subject_id}_HRIRs_{variant}.sofa",
        download_subject_ids=HUTUBS_SUBJECT_IDS,
    )
    dataset_mesh_spec = MeshSpec(
        aligned_by=("subject",),
        filename_pattern=r"^(?P<subject_id>pp\d+)_3DheadMesh\.(?:ply|stl)$",
        download_pattern="{subject_id}_3DheadMesh.ply",
        download_subject_ids=HUTUBS_MESH_SUBJECT_IDS,
    )
    dataset_anthropometry_spec = AnthropometrySpec(
        aligned_by=("subject",),
        filename="AntrhopometricMeasures.csv",
        download_filename="AntrhopometricMeasures.csv",
    )
    dataset_image_spec = ImageSpec(
        supported_align_by=(
            ("subject",),
            ("subject", "position"),
            ("subject", "ear"),
            ("subject", "position", "ear"),
        )
    )
    dataset_video_spec = VideoSpec(
        supported_align_by=(
            ("subject",),
            ("subject", "position"),
            ("subject", "ear"),
            ("subject", "position", "ear"),
        )
    )

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_version: str = "all",
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...] = HRTFSpec(),
        target: HRTFSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
        | None = None,
        index_by: str | tuple[str, ...] = ("subject",),
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
    ) -> None:
        super().__init__(
            root=root,
            download=download,
            download_resources=download_resources,
            download_hrtf_version=download_hrtf_version,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            index_by=index_by,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
