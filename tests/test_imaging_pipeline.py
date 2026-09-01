import numpy as np

from src.imaging.pipeline import preprocess_series


def test_preprocess_series_normalizes_and_flips_left_coronal_volume():
    volume = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    metadata = {
        "pixel_spacing": [1.0, 1.0],
        "image_position": [-300.0, 0.0, 0.0],
        "image_orientation": [1, 0, 0, 0, 1, 0],
        "rows": 2,
        "columns": 2,
        "laterality": None,
    }
    result = preprocess_series(volume, metadata, "Coronal", image_size=2)
    assert result.shape == volume.shape
    assert result.dtype == np.float32
    assert result.min() == 0.0
    assert result.max() == 1.0
    assert result[0, 0, 0] > result[0, 0, 1]
