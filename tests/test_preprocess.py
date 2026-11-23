"""Synthetic tests for canonical MRI volume preprocessing."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("pydicom")
preprocess = importlib.import_module("scripts.preprocess")


def _dicom(z, pixels, *, orientation=(1, 0, 0, 0, 1, 0), x=0):
    return SimpleNamespace(
        ImageOrientationPatient=orientation,
        ImagePositionPatient=(x, 0, z),
        Rows=pixels.shape[0],
        Columns=pixels.shape[1],
        pixel_array=pixels,
        PixelSpacing=(1.0, 1.0),
        SliceThickness=1.0,
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        Laterality="",
        ImageLaterality="",
        BodyPartExamined="",
        SeriesDescription="",
        Manufacturer="",
        MagneticFieldStrength=1.5,
    )


def _meta(*, tag="", desc="", body_part="", center_x=0):
    return {
        "laterality_tag": tag,
        "desc": desc,
        "body_part": body_part,
        "center_x": center_x,
    }


def test_series_volume_sorts_axial_slices_along_normal():
    first = _dicom(0, np.zeros((2, 2), dtype=np.int16))
    second = _dicom(1, np.ones((2, 2), dtype=np.int16))

    volume, meta = preprocess.series_volume([second, first], "Axial")

    assert volume.shape == (2, 2, 2)
    assert np.array_equal(volume[0], np.zeros((2, 2)))
    assert np.array_equal(volume[1], np.ones((2, 2)))
    assert meta["n_slices"] == 2
    assert meta["rows"] == 2
    assert meta["cols"] == 2


def test_series_volume_canonicalizes_axial_slice_and_row_directions():
    orientation = (-1, 0, 0, 0, 1, 0)
    first = _dicom(0, np.array([[1, 2], [3, 4]]), orientation=orientation)
    second = _dicom(1, np.array([[5, 6], [7, 8]]), orientation=orientation)

    volume, _ = preprocess.series_volume([first, second], "Axial")

    assert np.array_equal(volume[0], np.array([[2, 1], [4, 3]]))
    assert np.array_equal(volume[1], np.array([[6, 5], [8, 7]]))


def test_to_uint8_outputs_fixed_size_and_full_dynamic_range():
    volume = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)

    out = preprocess.to_uint8(volume)

    assert out.shape == (3, preprocess.SIZE, preprocess.SIZE)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_to_uint8_handles_constant_volume_without_nans():
    volume = np.ones((2, 4, 4), dtype=np.float32)

    out = preprocess.to_uint8(volume)

    assert out.dtype == np.uint8
    assert np.array_equal(out, np.zeros_like(out))


def test_resample_depth_returns_exact_count_and_preserves_endpoints():
    volume = np.arange(5 * 2 * 2).reshape(5, 2, 2)

    out = preprocess.resample_depth(volume)

    expected_indices = np.round(np.linspace(0, 4, preprocess.DEPTH)).astype(int)
    assert out.shape == (preprocess.DEPTH, 2, 2)
    assert np.array_equal(out, volume[expected_indices])
    assert np.array_equal(out[0], volume[0])
    assert np.array_equal(out[-1], volume[-1])


def test_laterality_prefers_majority_dicom_tag():
    metas = [
        _meta(tag="L", desc="RIGHT KNEE"),
        _meta(tag="R"),
        _meta(tag="L"),
    ]

    assert preprocess.laterality(metas) == ("L", "tag")


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("LEFT KNEE", "L"),
        ("RODILLA IZQUIERDA", "L"),
        ("LINKES KNIE", "L"),
        ("GENOU GAUCHE", "L"),
        ("RIGHT KNEE", "R"),
        ("RODILLA DERECHA", "R"),
        ("RECHTES KNIE", "R"),
        ("GENOU DROIT", "R"),
    ],
)
def test_laterality_reads_multilingual_descriptions(description, expected):
    assert preprocess.laterality([_meta(desc=description)]) == (expected, "desc")


def test_laterality_uses_patient_coordinate_fallback():
    metas = [_meta(center_x=10), _meta(center_x=20)]

    assert preprocess.laterality(metas) == ("L", "x=15")
    assert preprocess.laterality([_meta(center_x=-10)]) == ("R", "x=-10")
