"""Tests for DICOM geometry utilities."""

import numpy as np

from src.imaging.geometry import (
    compute_image_center,
    compute_physical_position,
    compute_slice_normal,
    determine_laterality,
    normalize_laterality,
    sort_by_physical_position,
)


class MockDICOM:
    def __init__(self, position, orientation):
        self.ImagePositionPatient = position
        self.ImageOrientationPatient = orientation


def test_compute_slice_normal():
    orientation = [1, 0, 0, 0, 1, 0]
    normal = compute_slice_normal(orientation)
    np.testing.assert_array_almost_equal(normal, [0, 0, 1])


def test_compute_physical_position():
    position = [0, 0, 10]
    orientation = [1, 0, 0, 0, 1, 0]
    result = compute_physical_position(position, orientation)
    assert result == 10.0


def test_sort_by_physical_position():
    ds1 = MockDICOM([0, 0, 30], [1, 0, 0, 0, 1, 0])
    ds2 = MockDICOM([0, 0, 10], [1, 0, 0, 0, 1, 0])
    ds3 = MockDICOM([0, 0, 20], [1, 0, 0, 0, 1, 0])

    sorted_ds = sort_by_physical_position([ds1, ds2, ds3])
    assert sorted_ds[0] is ds2
    assert sorted_ds[1] is ds3
    assert sorted_ds[2] is ds1


def test_compute_image_center():
    position = [0, 0, 0]
    orientation = [1, 0, 0, 0, 1, 0]
    pixel_spacing = [1.0, 1.0]
    rows, columns = 256, 256

    center = compute_image_center(position, orientation, pixel_spacing, rows, columns)
    np.testing.assert_array_almost_equal(center, [128, 128, 0])


def test_determine_laterality_right():
    position = [100, 0, 0]
    orientation = [1, 0, 0, 0, 1, 0]
    pixel_spacing = [1.0, 1.0]

    laterality = determine_laterality(position, orientation, pixel_spacing, 256, 256)
    assert laterality == "R"


def test_determine_laterality_left():
    position = [-300, 0, 0]
    orientation = [1, 0, 0, 0, 1, 0]
    pixel_spacing = [1.0, 1.0]

    laterality = determine_laterality(position, orientation, pixel_spacing, 256, 256)
    assert laterality == "L"


def test_determine_laterality_dead_zone_uses_tag():
    position = [-128, 0, 0]
    orientation = [1, 0, 0, 0, 1, 0]
    pixel_spacing = [1.0, 1.0]

    laterality = determine_laterality(
        position, orientation, pixel_spacing, 256, 256, laterality_tag="L"
    )
    assert laterality == "L"


def test_normalize_laterality_right_unchanged():
    pixel_array = np.array([[1, 2], [3, 4]])
    result = normalize_laterality(pixel_array, "R", "Coronal")
    np.testing.assert_array_equal(result, pixel_array)


def test_normalize_laterality_left_coronal():
    pixel_array = np.array([[1, 2], [3, 4]])
    result = normalize_laterality(pixel_array, "L", "Coronal")
    expected = np.array([[2, 1], [4, 3]])
    np.testing.assert_array_equal(result, expected)


def test_normalize_laterality_left_sagittal():
    pixel_array = np.array([[1, 2], [3, 4]])
    result = normalize_laterality(pixel_array, "L", "Sagittal")
    np.testing.assert_array_equal(result, pixel_array)
