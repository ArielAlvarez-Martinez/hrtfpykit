import os
import shutil
from collections.abc import Generator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hrtfpykit.datasets.base import BaseDataset
from hrtfpykit.datasets.config import DatasetConfig, HRTFConfig, ResourceTypeConfig
from hrtfpykit.datasets.specs import HRTFSpec, ILDSpec, ITDSpec
from hrtfpykit.hrtf import HRTF, load_hrtf
from hrtfpykit.metrics import ild, itd
from hrtfpykit.sofa import load_sofa


SOFA_PATH = os.getenv("HRTFPYKIT_TEST_SOFA_PATH", "")
pytestmark = pytest.mark.skipif(
    SOFA_PATH == "" or not os.path.exists(SOFA_PATH),
    reason="Required local SOFA file is not available",
)


@pytest.fixture(autouse=True)
def close_figures() -> Generator[None, None, None]:
    yield
    plt.close("all")


@pytest.fixture
def sofa_path() -> Path:
    return Path(SOFA_PATH)


@pytest.fixture
def real_hrtf(sofa_path: Path) -> Generator[HRTF, None, None]:
    hrtf = load_hrtf(sofa_path)
    try:
        yield hrtf
    finally:
        if hrtf.Sofa is not None and hrtf.Sofa.netCDF4_dataset is not None:
            hrtf.Sofa.netCDF4_dataset.close()


def test_selected_transformed_hrtf_roundtrips_through_sofa_metrics_and_plots(
    real_hrtf: HRTF,
    tmp_path: Path,
) -> None:
    selected_hrtf = real_hrtf.select(
        positions=["front", "left", "right"],
        start=0,
        end=128,
    )
    transformed_hrtf = selected_hrtf.transform.apply_window("hann")
    destination = tmp_path / "selected_windowed_roundtrip.sofa"

    saved_path = transformed_hrtf.save(
        path=destination,
        overwrite=True,
        change_sofa_dimensions=True,
        sofa_convention="SimpleFreeFieldHRIR",
    )

    assert saved_path == destination
    assert destination.exists()

    saved_sofa = load_sofa(destination)
    try:
        assert saved_sofa.GlobalAttributes.get("SOFAConventions").value == "SimpleFreeFieldHRIR"
        assert saved_sofa.GlobalAttributes.get("DataType").value == "FIR"
        assert saved_sofa.Variables.get("SourcePosition").value.shape == (3, 3)
        assert saved_sofa.Variables.get("Data.IR").value.shape == (3, 2, 128)
        assert "Data.Real" not in saved_sofa.Variables.get_names()
        assert "Data.Imag" not in saved_sofa.Variables.get_names()
    finally:
        saved_sofa.netCDF4_dataset.close()

    reloaded_hrtf = load_hrtf(destination)
    try:
        assert reloaded_hrtf.SOFAConventions == "SimpleFreeFieldHRIR"
        assert reloaded_hrtf.IR.values.shape == (3, 2, 128)
        assert reloaded_hrtf.TF.values.shape == (3, 2, 65)
        np.testing.assert_allclose(
            reloaded_hrtf.TF.frequency_bins,
            np.fft.rfftfreq(128, d=1.0 / reloaded_hrtf.IR.sample_rate),
        )

        itd_values = itd(
            reloaded_hrtf.IR,
            sample_rate=reloaded_hrtf.IR.sample_rate,
            method="maxiacce",
        )
        ild_values = ild(
            reloaded_hrtf.IR,
            sample_rate=reloaded_hrtf.IR.sample_rate,
        )

        assert np.asarray(itd_values).shape == (3,)
        assert np.asarray(ild_values).shape == (3,)
        assert np.all(np.isfinite(itd_values))
        assert np.all(np.isfinite(ild_values))

        result = reloaded_hrtf.plot_magnitude(
            positions="front",
            ear="left",
            show=False,
        )
        figure = plt.gcf()

        assert result is None
        assert len(figure.axes) == 1
        assert len(figure.axes[0].lines) == 1
    finally:
        if reloaded_hrtf.Sofa is not None and reloaded_hrtf.Sofa.netCDF4_dataset is not None:
            reloaded_hrtf.Sofa.netCDF4_dataset.close()


def test_dataset_pipeline_loads_hrtf_and_derived_acoustic_specs(
    sofa_path: Path,
    tmp_path: Path,
) -> None:
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
    assert dataset.sample_rate == 44100.0
    assert dataset.positions.shape[0] == 440
    assert dataset.frequency_bins.shape == (129,)
    assert dataset.selected_position_indices == (0, 1)

    loaded_hrtf = dataset.get_subject_hrtf("S001")
    assert dataset.get_subject_hrtf("S001") is loaded_hrtf

    sample = dataset[0]
    next_position_sample = dataset[1]

    assert set(sample) == {"inputs", "target"}
    assert set(sample["inputs"]) == {"magnitude_db", "position_index"}
    assert set(sample["target"]) == {"itd", "ild"}
    assert np.asarray(sample["inputs"]["magnitude_db"]).shape == (129,)
    assert np.all(np.isfinite(sample["inputs"]["magnitude_db"]))
    assert sample["inputs"]["position_index"] == 0
    assert next_position_sample["inputs"]["position_index"] == 1
    assert np.asarray(sample["target"]["itd"]).shape == ()
    assert np.asarray(sample["target"]["ild"]).shape == ()
    assert np.isfinite(sample["target"]["itd"])
    assert np.isfinite(sample["target"]["ild"])
