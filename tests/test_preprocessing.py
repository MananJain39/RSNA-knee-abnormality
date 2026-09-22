"""Tests for image preprocessing utilities."""

import numpy as np
import pytest

from src.imaging.preprocessing import (
    create_25d_groups,
    extract_middle_slices,
    from_uint8,
    normalize,
    percentile_normalize_series,
    physical_crop,
    resize,
    sample_central_band,
    to_tensor,
    to_uint8,
)


def test_normalize_minmax():
    image = np.array([0, 50, 100], dtype=np.float32)
    result = normalize(image, method="minmax")
    np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])


def test_uint8_round_trip_error_bounded():
    rng = np.random.default_rng(0)
    volume = rng.random((4, 16, 16)).astype(np.float32)
    restored = from_uint8(to_uint8(volume))
    assert restored.dtype == np.float32
    max_err = float(np.abs(restored - volume).max())
    # Half an LSB of an 8-bit scale: 0.5 / 255.
    assert max_err <= 0.5 / 255 + 1e-6


def test_uint8_rejects_out_of_range():
    with pytest.raises(ValueError):
        to_uint8(np.array([-0.1, 0.5], dtype=np.float32))
    with pytest.raises(ValueError):
        to_uint8(np.array([0.5, 1.5], dtype=np.float32))


def test_uint8_quantization_size_gain():
    volume = np.ones((32, 336, 336), dtype=np.float32)
    assert to_uint8(volume).nbytes == volume.nbytes // 4


def test_normalize_minmax_constant():
    image = np.array([5, 5, 5], dtype=np.float32)
    result = normalize(image, method="minmax")
    np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])


def test_normalize_zscore():
    image = np.array([1, 2, 3], dtype=np.float32)
    result = normalize(image, method="zscore")
    assert abs(result.mean()) < 1e-5
    assert abs(result.std() - 1.0) < 1e-5


def test_normalize_clip():
    image = np.arange(100, dtype=np.float32)
    result = normalize(image, method="clip")
    assert result.min() >= 0
    assert result.max() <= 100


def test_normalize_unknown_method():
    with pytest.raises(ValueError):
        normalize(np.array([1, 2, 3]), method="unknown")


def test_resize_3d():
    image = np.random.rand(10, 20, 20).astype(np.float32)
    result = resize(image, (5, 10, 10))
    assert result.shape == (5, 10, 10)


def test_resize_2d():
    image = np.random.rand(20, 20).astype(np.float32)
    result = resize(image, (10, 10))
    assert result.shape == (10, 10)


def test_physical_crop_2d():
    image = np.random.rand(256, 256).astype(np.float32)
    result = physical_crop(image, pixel_spacing=1.0, target_extent_mm=130)
    assert result.shape == (130, 130)


def test_physical_crop_3d():
    image = np.random.rand(10, 256, 256).astype(np.float32)
    result = physical_crop(image, pixel_spacing=1.0, target_extent_mm=130)
    assert result.shape == (10, 130, 130)


def test_physical_crop_no_op():
    image = np.random.rand(100, 100).astype(np.float32)
    result = physical_crop(image, pixel_spacing=1.0, target_extent_mm=130)
    assert result.shape == image.shape


def test_percentile_normalize_series():
    volume = np.random.rand(10, 50, 50).astype(np.float32) * 1000
    result = percentile_normalize_series(volume)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_percentile_normalize_series_constant():
    volume = np.ones((10, 50, 50), dtype=np.float32) * 5
    result = percentile_normalize_series(volume)
    np.testing.assert_array_almost_equal(result, np.zeros_like(result))


def test_to_tensor_3d_channel_first():
    pytest.importorskip("torch")
    image = np.random.rand(10, 20, 20).astype(np.float32)
    tensor = to_tensor(image, channel_first=True)
    assert tensor.shape == (1, 10, 20, 20)


def test_to_tensor_3d_channel_last():
    pytest.importorskip("torch")
    image = np.random.rand(10, 20, 20).astype(np.float32)
    tensor = to_tensor(image, channel_first=False)
    assert tensor.shape == (10, 20, 20)


def test_extract_middle_slices():
    volume = np.arange(100).reshape(10, 2, 5)
    result = extract_middle_slices(volume, n_slices=3)
    assert result.shape == (3, 2, 5)


def test_extract_middle_slices_odd():
    volume = np.arange(100).reshape(10, 2, 5)
    result = extract_middle_slices(volume, n_slices=5)
    assert result.shape[0] == 5


def test_sample_central_band():
    indices = sample_central_band(100, n_samples=12, low_frac=0.12, high_frac=0.88)
    assert len(indices) == 12
    assert indices[0] >= 12
    assert indices[-1] <= 87


def test_sample_central_band_small_volume():
    indices = sample_central_band(10, n_samples=5, low_frac=0.2, high_frac=0.8)
    assert len(indices) == 5
    assert all(0 <= i < 10 for i in indices)


def test_create_25d_groups():
    volume = np.arange(30).reshape(10, 3, 1)
    groups = create_25d_groups(volume, group_size=3)
    assert groups.shape == (8, 3, 3, 1)


def test_create_25d_groups_small_volume():
    volume = np.arange(6).reshape(2, 3, 1)
    groups = create_25d_groups(volume, group_size=3)
    assert groups.shape[0] == 1
