import os
import shutil
from collections.abc import Generator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hrtfpykit.datasets import collate_samples
from hrtfpykit.datasets.base import BaseDataset
from hrtfpykit.datasets.config import (
    AnthropometryConfig,
    DatasetConfig,
    DownloadConfig,
    HRTFConfig,
    ImageConfig,
    MeshConfig,
    MetadataConfig,
    ResourceTypeConfig,
    VideoConfig,
)
from hrtfpykit.datasets.download import BaseDownload
from hrtfpykit.datasets.specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    MetadataSpec,
    SHSpec,
    VideoSpec,
)
from hrtfpykit.hrtf import HRTF, load_hrtf
from hrtfpykit.metrics import ild, itd
from hrtfpykit.sofa import load_sofa


SOFA_PATH = os.getenv("HRTFPYKIT_TEST_SOFA_PATH", "")
INTEGRATION_SOFA_PATH = os.getenv("HRTFPYKIT_TEST_INTEGRATION_SOFA_PATH", "")
FIXTURE_SOFA_PATH = Path(__file__).parent / "fixtures" / "integration_hrtf.sofa"
TESTS_SOFA_PATH = Path(__file__).parent / "pp1_HRIRs_measured.sofa"
RESOLVED_SOFA_PATH = (
    INTEGRATION_SOFA_PATH
    if INTEGRATION_SOFA_PATH != ""
    else str(FIXTURE_SOFA_PATH)
    if FIXTURE_SOFA_PATH.exists()
    else str(TESTS_SOFA_PATH)
    if TESTS_SOFA_PATH.exists()
    else SOFA_PATH
)
pytestmark = pytest.mark.skipif(
    RESOLVED_SOFA_PATH == "" or not os.path.exists(RESOLVED_SOFA_PATH),
    reason="Required integration SOFA fixture is not available",
)


@pytest.fixture(autouse=True)
def close_figures() -> Generator[None, None, None]:
    yield
    plt.close("all")


@pytest.fixture
def sofa_path() -> Path:
    return Path(RESOLVED_SOFA_PATH)


@pytest.fixture
def real_hrtf(sofa_path: Path) -> Generator[HRTF, None, None]:
    hrtf = load_hrtf(sofa_path)
    try:
        yield hrtf
    finally:
        if hrtf.Sofa is not None and hrtf.Sofa.netCDF4_dataset is not None:
            hrtf.Sofa.netCDF4_dataset.close()


def test_selected_transformed_hrtf_feeds_metrics_and_plots(
    real_hrtf: HRTF,
) -> None:
    source_positions = real_hrtf.Sources.get_positions(angle_unit="degrees")
    if source_positions.shape[0] < 3:
        pytest.skip(
            reason="Integration SOFA fixture requires at least three source positions"
        )
    sample_count = int(real_hrtf.IR.values.shape[-1])
    if sample_count < 2:
        pytest.skip(reason="Integration SOFA fixture requires at least two IR samples")
    selected_positions = np.asarray(source_positions[:3], dtype=float)
    crop_end = min(128, sample_count)

    selected_hrtf = real_hrtf.select(
        positions=selected_positions,
        position_coordinate_system=real_hrtf.Sources.source_coordinate_system,
        start=0,
        end=crop_end,
    )
    transformed_hrtf = selected_hrtf.transform.apply_window("hann")

    assert transformed_hrtf.IR.values.shape == (3, 2, crop_end)
    assert transformed_hrtf.TF.values.shape[:2] == (3, 2)
    assert (
        transformed_hrtf.TF.values.shape[-1]
        == transformed_hrtf.TF.frequency_bins.shape[0]
    )
    np.testing.assert_allclose(
        transformed_hrtf.TF.frequency_bins,
        np.fft.rfftfreq(
            transformed_hrtf.fft_length,
            d=1.0 / transformed_hrtf.IR.sample_rate,
        ),
    )

    itd_values = itd(
        transformed_hrtf.IR,
        sample_rate=transformed_hrtf.IR.sample_rate,
        method="maxiacce",
    )
    ild_values = ild(
        transformed_hrtf.IR,
        sample_rate=transformed_hrtf.IR.sample_rate,
    )

    assert np.asarray(itd_values).shape == (3,)
    assert np.asarray(ild_values).shape == (3,)
    assert np.all(np.isfinite(itd_values))
    assert np.all(np.isfinite(ild_values))

    result = transformed_hrtf.plot_magnitude(
        positions=selected_positions[0],
        ear="left",
        show=False,
    )
    figure = plt.gcf()

    assert result is None
    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 1


def test_transformed_hrtf_roundtrips_through_sofa_and_reloads(
    real_hrtf: HRTF,
    tmp_path: Path,
) -> None:
    source_positions = real_hrtf.Sources.get_positions(angle_unit="degrees")
    if source_positions.shape[0] < 3:
        pytest.skip(
            reason="Integration SOFA fixture requires at least three source positions"
        )
    selected_positions = np.asarray(source_positions[:3], dtype=float)

    selected_hrtf = real_hrtf.select(
        positions=selected_positions,
        position_coordinate_system=real_hrtf.Sources.source_coordinate_system,
    )
    transformed_hrtf = selected_hrtf.transform.apply_gain(-3.0, scale="db")
    destination = tmp_path / "transformed_roundtrip.sofa"

    saved_path = transformed_hrtf.save(
        path=destination,
        overwrite=True,
        change_sofa_dimensions=True,
        sofa_convention="SimpleFreeFieldHRTF",
    )

    assert saved_path == destination
    assert destination.exists()

    saved_sofa = load_sofa(saved_path)
    try:
        saved_variables = set(saved_sofa.Variables.get_names())
        assert (
            saved_sofa.GlobalAttributes.get("SOFAConventions").value
            == "SimpleFreeFieldHRTF"
        )
        assert saved_sofa.GlobalAttributes.get("DataType").value == "TF"
        assert {"Data.Real", "Data.Imag", "N", "SourcePosition"}.issubset(saved_variables)
        assert "Data.IR" not in saved_variables
        assert "Data.SamplingRate" not in saved_variables
        assert np.asarray(saved_sofa.Variables.get("Data.Real").value).shape[:2] == (
            3,
            2,
        )
        assert np.asarray(saved_sofa.Variables.get("SourcePosition").value).shape == (3, 3)
    finally:
        saved_sofa.netCDF4_dataset.close()

    reloaded_hrtf = load_hrtf(saved_path)
    try:
        assert reloaded_hrtf.SOFAConventions == "SimpleFreeFieldHRTF"
        assert reloaded_hrtf.IR.values.shape[:2] == (3, 2)
        assert reloaded_hrtf.TF.values.shape[:2] == (3, 2)
        np.testing.assert_allclose(
            reloaded_hrtf.TF.frequency_bins,
            transformed_hrtf.TF.frequency_bins,
        )
        np.testing.assert_allclose(
            reloaded_hrtf.TF.values,
            transformed_hrtf.TF.values,
        )

        itd_values = itd(
            reloaded_hrtf.IR,
            sample_rate=reloaded_hrtf.IR.sample_rate,
            method="maxiacce",
        )
        assert np.asarray(itd_values).shape == (3,)
        assert np.all(np.isfinite(itd_values))

        reloaded_hrtf.Sources.source_coordinate_system = "spherical"
        plot_position = reloaded_hrtf.Sources.get_positions(angle_unit="degrees")[0]
        result = reloaded_hrtf.plot_magnitude(
            positions=plot_position,
            ear="right",
            show=False,
        )
        assert result is None
        assert len(plt.gcf().axes) == 1
    finally:
        reloaded_hrtf.Sofa.netCDF4_dataset.close()


def test_dataset_pipeline_resolves_all_spec_families(
    sofa_path: Path,
    tmp_path: Path,
) -> None:
    source_hrtf = load_hrtf(sofa_path)
    try:
        source_count = int(source_hrtf.IR.values.shape[0])
        frequency_count = int(source_hrtf.TF.frequency_bins.shape[0])
    finally:
        if (
            source_hrtf.Sofa is not None
            and source_hrtf.Sofa.netCDF4_dataset is not None
        ):
            source_hrtf.Sofa.netCDF4_dataset.close()
    if source_count < 2:
        pytest.skip(
            reason="Integration SOFA fixture requires at least two source positions"
        )

    subject_ids = ("S001", "S002")
    mesh_root = tmp_path / "meshes"
    image_root = tmp_path / "images"
    video_root = tmp_path / "videos"
    mesh_root.mkdir()
    image_root.mkdir()
    video_root.mkdir()
    for subject_id in subject_ids:
        shutil.copyfile(sofa_path, tmp_path / f"{subject_id}.sofa")
        (mesh_root / f"{subject_id}.ply").write_text(
            "ply\nformat ascii 1.0\nend_header\n",
            encoding="utf-8",
        )
        subject_image_root = image_root / subject_id
        subject_video_root = video_root / subject_id
        subject_image_root.mkdir()
        subject_video_root.mkdir()
        (subject_image_root / "frame_001.png").write_bytes(b"integration-image")
        (subject_video_root / "clip_001.mp4").write_bytes(b"integration-video")

    (tmp_path / "anthropometry.csv").write_text(
        "subject,head_width,pinna_height\n"
        "S001,14.1,5.1\n"
        "S002,14.2,5.2\n",
        encoding="utf-8",
    )
    (tmp_path / "metadata.csv").write_text(
        "subject,age_group,session\n"
        "S001,adult,A\n"
        "S002,adult,B\n",
        encoding="utf-8",
    )

    config = DatasetConfig(
        name="AllSpecIntegrationDataset",
        subject_ids=subject_ids,
        hrtf=HRTFConfig(
            types={
                "measured": ResourceTypeConfig(
                    path_pattern="{subject_id}.sofa",
                ),
            },
        ),
        mesh=MeshConfig(
            types={
                "default": ResourceTypeConfig(
                    path_pattern="meshes/{subject_id}.ply",
                ),
            },
            extensions=(".ply",),
        ),
        anthropometry=AnthropometryConfig(
            path="anthropometry.csv",
            left_prefix="left_",
            right_prefix="right_",
        ),
        metadata=MetadataConfig(path="metadata.csv"),
        image=ImageConfig(extensions=(".png",)),
        video=VideoConfig(extensions=(".mp4",)),
    )
    dataset = BaseDataset(
        root=tmp_path,
        config=config,
        dataset_hrtf_variant="measured",
        dataset_mesh_variant="default",
        inputs=(
            HRTFSpec(
                domain="frequency",
                signal="tf_magnitude_db",
                positions=(0, 1),
                ears="left",
                index_by=("subject",),
                name="hrtf",
            ),
            ITDSpec(
                positions=(0, 1),
                index_by=("subject",),
                method="maxiacce",
                name="itd",
            ),
            ILDSpec(
                positions=(0, 1),
                index_by=("subject",),
                mode="broad-band",
                name="ild",
            ),
            SHSpec(
                sh_order=1,
                ears="left",
                index_by=("subject",),
                name="sh",
            ),
            MeshSpec(name="mesh"),
            AnthropometrySpec(name="anthropometry"),
            MetadataSpec(name="metadata"),
            ImageSpec(path="images", grouped_by="subject", name="image"),
            VideoSpec(path="videos", grouped_by="subject", name="video"),
        ),
    )

    assert len(dataset) == len(subject_ids)
    assert dataset.available_subjects == list(subject_ids)
    assert dataset.selected_subjects == list(subject_ids)
    assert dataset.dataset_hrtf_variant == "measured"
    assert dataset.dataset_mesh_variant == "default"

    sample = dataset[0]
    assert sample["target"] is None
    assert set(sample["inputs"]) == {
        "hrtf",
        "itd",
        "ild",
        "sh",
        "mesh",
        "anthropometry",
        "metadata",
        "image",
        "video",
    }

    assert np.asarray(sample["inputs"]["hrtf"]).shape == (2, frequency_count)
    assert np.asarray(sample["inputs"]["itd"]).shape == (2,)
    assert np.asarray(sample["inputs"]["ild"]).shape == (2,)
    assert np.asarray(sample["inputs"]["sh"]).shape == (4, frequency_count)
    assert np.all(np.isfinite(sample["inputs"]["hrtf"]))
    assert np.all(np.isfinite(sample["inputs"]["itd"]))
    assert np.all(np.isfinite(sample["inputs"]["ild"]))
    assert np.all(np.isfinite(sample["inputs"]["sh"]))

    assert Path(sample["inputs"]["mesh"]).name == "S001.ply"
    assert sample["inputs"]["anthropometry"]["head_width"] == pytest.approx(14.1)
    assert sample["inputs"]["anthropometry"]["pinna_height"] == pytest.approx(5.1)
    assert sample["inputs"]["metadata"]["age_group"] == "adult"
    assert sample["inputs"]["metadata"]["session"] == "A"
    assert Path(sample["inputs"]["image"]).name == "frame_001.png"
    assert Path(sample["inputs"]["video"]).name == "clip_001.mp4"


def test_dataset_pipeline_loads_hrtf_and_derived_acoustic_specs(
    sofa_path: Path,
    tmp_path: Path,
) -> None:
    source_hrtf = load_hrtf(sofa_path)
    try:
        source_count = int(source_hrtf.IR.values.shape[0])
        sample_rate = float(source_hrtf.IR.sample_rate)
        frequency_count = int(source_hrtf.TF.frequency_bins.shape[0])
    finally:
        if (
            source_hrtf.Sofa is not None
            and source_hrtf.Sofa.netCDF4_dataset is not None
        ):
            source_hrtf.Sofa.netCDF4_dataset.close()
    if source_count < 2:
        pytest.skip(reason="Integration SOFA fixture requires at least two source positions")

    subject_ids = ("S001", "S002")
    for subject_id in subject_ids:
        shutil.copyfile(sofa_path, tmp_path / f"{subject_id}.sofa")

    config = DatasetConfig(
        name="IntegrationDataset",
        subject_ids=subject_ids,
        hrtf=HRTFConfig(
            types={
                "measured": ResourceTypeConfig(
                    path_pattern="{subject_id}.sofa",
                ),
            },
        ),
    )
    dataset = BaseDataset(
        root=tmp_path,
        config=config,
        dataset_hrtf_variant="measured",
        inputs=HRTFSpec(
            domain="frequency",
            signal="tf_magnitude_db",
            positions=(0, 1),
            ears="left",
            index_by=("subject", "position"),
            position_index=True,
            name="magnitude_db",
        ),
        target=(
            ITDSpec(
                positions=(0, 1),
                index_by=("subject", "position"),
                output="samples",
                name="itd",
            ),
            ILDSpec(
                positions=(0, 1),
                index_by=("subject", "position"),
                mode="broad-band",
                name="ild",
            ),
        ),
    )

    assert len(dataset) == len(subject_ids) * 2
    assert dataset.available_subjects == list(subject_ids)
    assert dataset.selected_subjects == list(subject_ids)
    assert dataset.sample_rate == sample_rate
    assert dataset.positions.shape[0] == source_count
    assert dataset.frequency_bins.shape == (frequency_count,)
    assert dataset.selected_position_indices == (0, 1)

    loaded_hrtf = dataset.get_subject_hrtf("S001")
    assert dataset.get_subject_hrtf("S001") is loaded_hrtf

    sample = dataset[0]
    next_position_sample = dataset[1]

    assert set(sample) == {"inputs", "target"}
    assert set(sample["inputs"]) == {"magnitude_db", "position_index"}
    assert set(sample["target"]) == {"itd", "ild"}
    assert np.asarray(sample["inputs"]["magnitude_db"]).shape == (frequency_count,)
    assert np.all(np.isfinite(sample["inputs"]["magnitude_db"]))
    assert sample["inputs"]["position_index"] == 0
    assert next_position_sample["inputs"]["position_index"] == 1
    assert np.asarray(sample["target"]["itd"]).shape == ()
    assert np.asarray(sample["target"]["ild"]).shape == ()
    assert np.isfinite(sample["target"]["itd"])
    assert np.isfinite(sample["target"]["ild"])


def test_download_plan_supports_resource_specific_base_urls(tmp_path: Path) -> None:
    subject_ids = ("S001",)
    config = DatasetConfig(
        name="MultiServerIntegrationDataset",
        subject_ids=subject_ids,
        hrtf=HRTFConfig(
            types={
                "measured": ResourceTypeConfig(
                    path_pattern="hrtf/{subject_id}.sofa",
                ),
            },
        ),
        mesh=MeshConfig(
            types={
                "default": ResourceTypeConfig(
                    path_pattern="mesh/{subject_id}.ply",
                ),
            },
            extensions=(".ply",),
        ),
        anthropometry=AnthropometryConfig(
            path="tables/anthropometry.csv",
            left_prefix="left_",
            right_prefix="right_",
        ),
        metadata=MetadataConfig(path="tables/metadata.csv"),
        download=DownloadConfig(
            base_url="https://default.example.org/dataset",
            available_resources=("hrtf", "mesh", "anthropometry", "metadata"),
            resource_base_urls={
                "hrtf": "https://hrtf.example.org/files",
                "mesh": "https://mesh.example.org/releases/v1",
                "anthropometry": "https://tables.example.org/resources",
            },
        ),
    )

    jobs = BaseDownload(
        config=config,
        root=tmp_path,
        verify_checksum=False,
    ).build_download_plan(
        download_resources="all",
        download_hrtf_variant="measured",
        download_mesh_variant="default",
    )

    jobs_by_resource = {str(job["resource"]): job for job in jobs}

    assert set(jobs_by_resource) == {"hrtf", "mesh", "anthropometry", "metadata"}
    assert jobs_by_resource["hrtf"]["relative_path"] == "hrtf/S001.sofa"
    assert jobs_by_resource["hrtf"]["url"] == "https://hrtf.example.org/files/hrtf/S001.sofa"
    assert jobs_by_resource["hrtf"]["destination"] == tmp_path / "hrtf" / "S001.sofa"

    assert jobs_by_resource["mesh"]["relative_path"] == "mesh/S001.ply"
    assert jobs_by_resource["mesh"]["url"] == "https://mesh.example.org/releases/v1/mesh/S001.ply"
    assert jobs_by_resource["mesh"]["destination"] == tmp_path / "mesh" / "S001.ply"

    assert jobs_by_resource["anthropometry"]["relative_path"] == "tables/anthropometry.csv"
    assert (
        jobs_by_resource["anthropometry"]["url"]
        == "https://tables.example.org/resources/tables/anthropometry.csv"
    )
    assert (
        jobs_by_resource["anthropometry"]["destination"]
        == tmp_path / "tables" / "anthropometry.csv"
    )

    assert jobs_by_resource["metadata"]["relative_path"] == "tables/metadata.csv"
    assert (
        jobs_by_resource["metadata"]["url"]
        == "https://default.example.org/dataset/tables/metadata.csv"
    )
    assert jobs_by_resource["metadata"]["destination"] == tmp_path / "tables" / "metadata.csv"
    assert all(job["checksum"] is None for job in jobs)


def test_collate_samples_returns_training_ready_tensor_dtypes() -> None:
    torch = pytest.importorskip("torch")
    batch = [
        {
            "inputs": {
                "magnitude": np.array([1.0, 2.0], dtype=np.float64),
                "position_index": 0,
                "anthropometry": {"head_width": 1.0, "head_height": 2.0},
                "mesh": Path("S001.ply"),
            },
            "target": {
                "sh": np.array([[1.0, 2.0]], dtype=np.float64),
            },
        },
        {
            "inputs": {
                "magnitude": np.array([3.0, 4.0], dtype=np.float64),
                "position_index": 1,
                "anthropometry": {"head_width": 3.0, "head_height": 4.0},
                "mesh": Path("S002.ply"),
            },
            "target": {
                "sh": np.array([[3.0, 4.0]], dtype=np.float64),
            },
        },
    ]

    collated = collate_samples(batch)

    assert collated["inputs"]["magnitude"].dtype == torch.float32
    assert collated["inputs"]["magnitude"].shape == (2, 2)
    assert collated["target"]["sh"].dtype == torch.float32
    assert collated["target"]["sh"].shape == (2, 1, 2)
    assert collated["inputs"]["anthropometry"].dtype == torch.float32
    assert collated["inputs"]["anthropometry"].shape == (2, 2)
    assert collated["inputs"]["position_index"].dtype == torch.int64
    assert collated["inputs"]["position_index"].tolist() == [0, 1]
    assert collated["inputs"]["mesh"] == [Path("S001.ply"), Path("S002.ply")]
