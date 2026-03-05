import numpy as np
import pytest

from ..hrtfpykit import loader


def test_parse_directions_converts_degrees_to_radians():
    source_direction = np.array([[180.0, 45.0], [0.0, -45.0]])

    out = loader._parse_directions(source_direction, "spherical")

    np.testing.assert_allclose(out[:, 0], np.array([np.pi, 0.0]), atol=1e-7)
    np.testing.assert_allclose(out[:, 1], np.array([np.pi / 4.0, -np.pi / 4.0]), atol=1e-7)


def test_parse_directions_rejects_non_spherical_type():
    with pytest.raises(ValueError, match="Only 'spherical' is supported"):
        loader._parse_directions(np.array([[0.0, 0.0]]), "cartesian")


def test_hrir_to_hrtf_magnitude_shape_and_freq_bins():
    hrir = np.ones((2, 2, 8), dtype=float)
    mag, freqs = loader._hrir_to_hrtf_magnitude(hrir, sampling_rate=48000, fft_length=8)

    assert mag.shape == (2, 2, 5)
    np.testing.assert_allclose(freqs, np.array([0.0, 6000.0, 12000.0, 18000.0, 24000.0]))


def test_load_multiple_hrtfs_from_folder_with_mocked_loader(monkeypatch):
    src_dirs = np.array([[0.0, 0.0], [np.pi / 2.0, 0.0]])
    freqs = np.array([0.0, 1000.0, 2000.0])
    hrtf_mag = np.ones((2, 2, 3))

    def fake_listdir(_folder):
        return ["b.sofa", "ignore.txt", "a.sofa"]

    def fake_load_hrtf(path, fft_length=256):
        return hrtf_mag, src_dirs, freqs, 48000

    monkeypatch.setattr(loader.os, "listdir", fake_listdir)
    monkeypatch.setattr(loader, "load_hrtf", fake_load_hrtf)

    out = loader.load_from_folder("dummy")

    hrtf_mag_list, source_directions, frequency_vector, fs, file_names = out
    assert len(hrtf_mag_list) == 2
    np.testing.assert_allclose(source_directions, src_dirs)
    np.testing.assert_allclose(frequency_vector, freqs)
    assert fs == 48000
    assert file_names == ["a.sofa", "b.sofa"]


