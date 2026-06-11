import os
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

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
    DownloadServerConfig,
    HRTFConfig,
    ImageConfig,
    MeshConfig,
    MetadataConfig,
    ResourceTypeConfig,
    VideoConfig,
)
from hrtfpykit.datasets.download import PathPatternDownload
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
from hrtfpykit.hrtf import HRTF, ild, load_hrtf
from hrtfpykit.plots import plot_magnitude
from hrtfpykit.utils.metrics import itd
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


def _ir_values(hrtf: HRTF) -> np.ndarray:
    values = hrtf.IR.values
    assert values is not None
    return values


def _tf_values(hrtf: HRTF) -> np.ndarray:
    values = hrtf.TF.values
    assert values is not None
    return values


def _frequency_bins(hrtf: HRTF) -> np.ndarray:
    values = hrtf.TF.frequency_bins
    assert values is not None
    return values


def _sample_rate(hrtf: HRTF) -> float:
    sample_rate = hrtf.IR.sample_rate
    assert sample_rate is not None
    return float(sample_rate)


def _source_coordinate_system(hrtf: HRTF) -> str:
    coordinate_system = hrtf.Sources.source_coordinate_system
    assert coordinate_system is not None
    return coordinate_system


def _wrap_value(container: Any, name: str) -> Any:
    wrapper = container.get(name)
    assert wrapper is not None
    return wrapper.value


def _close_sofa_dataset(sofa: Any) -> None:
    dataset = sofa.netCDF4_dataset
    assert dataset is not None
    dataset.close()


def _close_hrtf_sofa(hrtf: HRTF) -> None:
    sofa = hrtf.Sofa
    assert sofa is not None
    _close_sofa_dataset(sofa)


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


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
    sample_count = int(_ir_values(real_hrtf).shape[-1])
    if sample_count < 2:
        pytest.skip(reason="Integration SOFA fixture requires at least two IR samples")
    selected_positions = np.asarray(source_positions[:3], dtype=float)
    crop_end = min(128, sample_count)

    selected_hrtf = real_hrtf.select(
        positions=selected_positions,
        position_coordinate_system=_source_coordinate_system(real_hrtf),
        start_sample=0,
        end_sample=crop_end,
    )
    transformed_hrtf = selected_hrtf.transform.apply_window("hann")

    assert _ir_values(transformed_hrtf).shape == (3, 2, crop_end)
    assert _tf_values(transformed_hrtf).shape[:2] == (3, 2)
    assert (
        _tf_values(transformed_hrtf).shape[-1]
        == _frequency_bins(transformed_hrtf).shape[0]
    )
    fft_length = transformed_hrtf.fft_length
    assert fft_length is not None
    expected_frequency_bins = np.fft.rfftfreq(
        fft_length,
        d=1.0 / _sample_rate(transformed_hrtf),
    )
    np.testing.assert_allclose(
        cast(Any, _frequency_bins(transformed_hrtf)),
        cast(Any, expected_frequency_bins),
    )

    itd_values = itd(
        transformed_hrtf,
        method="maxiacce",
    )
    ild_values = ild(
        transformed_hrtf,
    )

    assert np.asarray(itd_values).shape == (3,)
    assert np.asarray(ild_values).shape == (3,)
    assert np.all(np.isfinite(itd_values))
    assert np.all(np.isfinite(ild_values))

    plot_magnitude(
        transformed_hrtf,
        positions=selected_positions[0],
        ear="left",
        show=False,
    )
    figure = plt.gcf()

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
        position_coordinate_system=_source_coordinate_system(real_hrtf),
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
            _wrap_value(saved_sofa.GlobalAttributes, "SOFAConventions")
            == "SimpleFreeFieldHRTF"
        )
        assert _wrap_value(saved_sofa.GlobalAttributes, "DataType") == "TF"
        assert {"Data.Real", "Data.Imag", "N", "SourcePosition"}.issubset(saved_variables)
        assert "Data.IR" not in saved_variables
        assert "Data.SamplingRate" not in saved_variables
        assert np.asarray(_wrap_value(saved_sofa.Variables, "Data.Real")).shape[:2] == (
            3,
            2,
        )
        assert np.asarray(_wrap_value(saved_sofa.Variables, "SourcePosition")).shape == (3, 3)
    finally:
        _close_sofa_dataset(saved_sofa)

    reloaded_hrtf = load_hrtf(saved_path)
    try:
        assert reloaded_hrtf.SOFAConventions == "SimpleFreeFieldHRTF"
        assert _ir_values(reloaded_hrtf).shape[:2] == (3, 2)
        assert _tf_values(reloaded_hrtf).shape[:2] == (3, 2)
        np.testing.assert_allclose(
            _frequency_bins(reloaded_hrtf),
            _frequency_bins(transformed_hrtf),
        )
        np.testing.assert_allclose(
            _tf_values(reloaded_hrtf),
            _tf_values(transformed_hrtf),
        )

        itd_values = itd(
            reloaded_hrtf,
            method="maxiacce",
        )
        assert np.asarray(itd_values).shape == (3,)
        assert np.all(np.isfinite(itd_values))

        reloaded_hrtf.Sources.source_coordinate_system = "spherical"
        plot_position = reloaded_hrtf.Sources.get_positions(angle_unit="degrees")[0]
        plot_magnitude(
            reloaded_hrtf,
            positions=plot_position,
            ear="right",
            show=False,
        )
        assert len(plt.gcf().axes) == 1
    finally:
        _close_hrtf_sofa(reloaded_hrtf)


def test_dataset_pipeline_resolves_all_spec_families(
    sofa_path: Path,
    tmp_path: Path,
) -> None:
    source_hrtf = load_hrtf(sofa_path)
    try:
        source_count = int(_ir_values(source_hrtf).shape[0])
        frequency_count = int(_frequency_bins(source_hrtf).shape[0])
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
        image=ImageConfig(path="images", extensions=(".png",)),
        video=VideoConfig(path="videos", extensions=(".mp4",)),
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
            ImageSpec(grouped_by="subject", name="image"),
            VideoSpec(grouped_by="subject", name="video"),
        ),
    )

    assert len(dataset) == len(subject_ids)
    assert dataset.available_subjects == list(subject_ids)
    assert dataset.selected_subjects == list(subject_ids)
    assert dataset.dataset_hrtf_variant == "measured"
    assert dataset.dataset_mesh_variant == "default"

    sample = cast(dict[str, Any], dataset[0])
    inputs = _mapping(sample["inputs"])
    assert sample["target"] is None
    assert sample["meta"] == {
        "dataset": "AllSpecIntegrationDataset",
        "subject_id": "S001",
        "position_index": None,
        "ear": None,
        "ear_index": None,
        "frequency_index": None,
        "sample_index": None,
    }
    assert set(inputs) == {
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

    assert np.asarray(inputs["hrtf"]).shape == (2, frequency_count)
    assert np.asarray(inputs["itd"]).shape == (2,)
    assert np.asarray(inputs["ild"]).shape == (2,)
    assert np.asarray(inputs["sh"]).shape == (4, frequency_count)
    assert np.all(np.isfinite(inputs["hrtf"]))
    assert np.all(np.isfinite(inputs["itd"]))
    assert np.all(np.isfinite(inputs["ild"]))
    assert np.all(np.isfinite(inputs["sh"]))

    assert Path(inputs["mesh"]).name == "S001.ply"
    assert inputs["anthropometry"]["head_width"] == pytest.approx(14.1)
    assert inputs["anthropometry"]["pinna_height"] == pytest.approx(5.1)
    assert inputs["metadata"]["age_group"] == "adult"
    assert inputs["metadata"]["session"] == "A"
    assert Path(inputs["image"]).name == "frame_001.png"
    assert Path(inputs["video"]).name == "clip_001.mp4"


def test_dataset_pipeline_loads_hrtf_and_derived_acoustic_specs(
    sofa_path: Path,
    tmp_path: Path,
) -> None:
    source_hrtf = load_hrtf(sofa_path)
    try:
        source_count = int(_ir_values(source_hrtf).shape[0])
        sample_rate = float(_sample_rate(source_hrtf))
        frequency_count = int(_frequency_bins(source_hrtf).shape[0])
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
    assert cast(np.ndarray, dataset.positions).shape[0] == source_count
    assert cast(np.ndarray, dataset.frequency_bins).shape == (frequency_count,)
    assert dataset.selected_position_indices == (0, 1)

    loaded_hrtf = dataset.get_subject_hrtf("S001")
    assert dataset.get_subject_hrtf("S001") is loaded_hrtf
    assert loaded_hrtf.Sofa is not None
    assert loaded_hrtf.Sofa.is_open() is False

    dataset.clear_cache()
    assert dataset._state.cache == {}
    reloaded_hrtf = dataset.get_subject_hrtf("S001")
    assert reloaded_hrtf is not loaded_hrtf
    assert reloaded_hrtf.Sofa is not None
    assert reloaded_hrtf.Sofa.is_open() is False

    sample = cast(dict[str, Any], dataset[0])
    next_position_sample = cast(dict[str, Any], dataset[1])
    sample_inputs = _mapping(sample["inputs"])
    sample_target = _mapping(sample["target"])
    next_position_inputs = _mapping(next_position_sample["inputs"])

    assert set(sample) == {"inputs", "target", "meta"}
    assert sample["meta"] == {
        "dataset": "IntegrationDataset",
        "subject_id": "S001",
        "position_index": 0,
        "ear": None,
        "ear_index": None,
        "frequency_index": None,
        "sample_index": None,
    }
    assert next_position_sample["meta"] == {
        "dataset": "IntegrationDataset",
        "subject_id": "S001",
        "position_index": 1,
        "ear": None,
        "ear_index": None,
        "frequency_index": None,
        "sample_index": None,
    }
    assert set(sample_inputs) == {"magnitude_db", "position_index"}
    assert set(sample_target) == {"itd", "ild"}
    assert np.asarray(sample_inputs["magnitude_db"]).shape == (frequency_count,)
    assert np.all(np.isfinite(sample_inputs["magnitude_db"]))
    assert sample_inputs["position_index"] == 0
    assert next_position_inputs["position_index"] == 1
    assert np.asarray(sample_target["itd"]).shape == ()
    assert np.asarray(sample_target["ild"]).shape == ()
    assert np.isfinite(sample_target["itd"])
    assert np.isfinite(sample_target["ild"])

    open_dataset = BaseDataset(
        root=tmp_path,
        config=config,
        dataset_hrtf_variant="measured",
        inputs=HRTFSpec(index_by=("subject",)),
        target=(),
        sofa_open=True,
    )
    open_hrtf = open_dataset.get_subject_hrtf("S001")
    assert open_hrtf.Sofa is not None
    assert open_hrtf.Sofa.is_open() is True
    open_dataset.clear_cache()
    assert open_dataset._state.cache == {}
    assert open_hrtf.Sofa.is_open() is False

    preloaded_dataset = BaseDataset(
        root=tmp_path,
        config=config,
        dataset_hrtf_variant="measured",
        inputs=HRTFSpec(index_by=("subject",)),
        target=(),
        preload_hrtfs=True,
    )
    assert ("hrtf", "S001") in preloaded_dataset._state.cache
    assert ("hrtf", "S002") in preloaded_dataset._state.cache
    for subject_id in subject_ids:
        cached_hrtf = cast(HRTF, preloaded_dataset._state.cache[("hrtf", subject_id)])
        assert cached_hrtf.Sofa is not None
        assert cached_hrtf.Sofa.is_open() is False
    preloaded_dataset.clear_cache()
    assert preloaded_dataset._state.cache == {}


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
        download_servers={
            "custom": DownloadServerConfig(
                base_url="https://default.example.org/dataset",
                available_resources=("hrtf", "mesh", "anthropometry", "metadata"),
                resource_base_urls={
                    "hrtf": "https://hrtf.example.org/files",
                    "mesh": "https://mesh.example.org/releases/v1",
                    "anthropometry": "https://tables.example.org/resources",
                },
            ),
        },
    )

    jobs = PathPatternDownload(
        config=config,
        root=tmp_path,
        verify_checksum=False,
        download_server="custom",
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
            "meta": {
                "dataset": "integration",
                "subject_id": "S001",
                "position_index": 0,
                "ear": None,
                "ear_index": None,
                "frequency_index": None,
                "sample_index": None,
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
            "meta": {
                "dataset": "integration",
                "subject_id": "S002",
                "position_index": 1,
                "ear": None,
                "ear_index": None,
                "frequency_index": None,
                "sample_index": None,
            },
        },
    ]

    collated = cast(dict[str, Any], collate_samples(batch))
    collated_inputs = _mapping(collated["inputs"])
    collated_target = _mapping(collated["target"])
    collated_meta = _mapping(collated["meta"])

    assert collated_inputs["magnitude"].dtype == torch.float32
    assert collated_inputs["magnitude"].shape == (2, 2)
    assert collated_target["sh"].dtype == torch.float32
    assert collated_target["sh"].shape == (2, 1, 2)
    assert collated_inputs["anthropometry"].dtype == torch.float32
    assert collated_inputs["anthropometry"].shape == (2, 2)
    assert collated_inputs["position_index"].dtype == torch.int64
    assert collated_inputs["position_index"].tolist() == [0, 1]
    assert collated_inputs["mesh"] == [Path("S001.ply"), Path("S002.ply")]
    assert collated_meta["dataset"] == ["integration", "integration"]
    assert collated_meta["subject_id"] == ["S001", "S002"]
    assert collated_meta["position_index"].dtype == torch.int64
    assert collated_meta["position_index"].tolist() == [0, 1]
    assert collated_meta["ear"] is None
    assert collated_meta["ear_index"] is None
    assert collated_meta["frequency_index"] is None
    assert collated_meta["sample_index"] is None
