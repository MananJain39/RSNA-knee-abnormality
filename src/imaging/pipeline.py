"""Reusable per-series preprocessing for the disk-backed classical pipeline."""

from __future__ import annotations

import numpy as np

from src.imaging.geometry import determine_laterality, normalize_laterality
from src.imaging.preprocessing import percentile_normalize_series, physical_crop, resize


def preprocess_series(
    volume: np.ndarray,
    metadata: dict,
    anatomical_plane: str,
    image_size: int = 336,
    crop_extent_mm: float = 130.0,
) -> np.ndarray:
    """Normalize one sorted MRI series into a float32 ``(N, H, W)`` volume.

    ``metadata`` is the dictionary returned by
    :func:`src.imaging.dicom_loader.get_series_metadata`.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D series volume, got {volume.shape}")

    spacing = metadata["pixel_spacing"]
    laterality = determine_laterality(
        metadata["image_position"],
        metadata["image_orientation"],
        spacing,
        int(metadata["rows"]),
        int(metadata["columns"]),
        laterality_tag=metadata.get("laterality"),
    )
    volume = normalize_laterality(volume, laterality, anatomical_plane)
    volume = physical_crop(volume, float(spacing[0]), crop_extent_mm)
    volume = resize(volume, (volume.shape[0], image_size, image_size))
    return percentile_normalize_series(volume)
