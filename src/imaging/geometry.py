"""DICOM geometry utilities for physical position sorting and laterality normalization."""

import numpy as np


def compute_slice_normal(image_orientation: list[float] | np.ndarray) -> np.ndarray:
    row_cosine = np.array(image_orientation[:3])
    col_cosine = np.array(image_orientation[3:])
    return np.cross(row_cosine, col_cosine)


def compute_physical_position(
    image_position: list[float] | np.ndarray,
    image_orientation: list[float] | np.ndarray,
) -> float:
    position = np.array(image_position)
    normal = compute_slice_normal(image_orientation)
    return float(np.dot(position, normal))


def sort_by_physical_position(dicom_datasets: list) -> list:
    positions = [
        compute_physical_position(ds.ImagePositionPatient, ds.ImageOrientationPatient)
        for ds in dicom_datasets
    ]
    sorted_indices = np.argsort(positions)
    return [dicom_datasets[i] for i in sorted_indices]


def compute_image_center(
    image_position: list[float] | np.ndarray,
    image_orientation: list[float] | np.ndarray,
    pixel_spacing: list[float] | np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    position = np.array(image_position)
    row_cosine = np.array(image_orientation[:3])
    col_cosine = np.array(image_orientation[3:])
    row_spacing = pixel_spacing[0]
    col_spacing = pixel_spacing[1]

    center = (
        position
        + row_cosine * row_spacing * (rows / 2)
        + col_cosine * col_spacing * (columns / 2)
    )
    return center


def determine_laterality(
    image_position: list[float] | np.ndarray,
    image_orientation: list[float] | np.ndarray,
    pixel_spacing: list[float] | np.ndarray,
    rows: int,
    columns: int,
    laterality_tag: str | None = None,
    dead_zone_threshold: float = 25.0,
) -> str:
    center = compute_image_center(
        image_position, image_orientation, pixel_spacing, rows, columns
    )

    if abs(center[0]) > dead_zone_threshold:
        return "L" if center[0] < 0 else "R"

    if laterality_tag in ("L", "R"):
        return laterality_tag

    return "R"


def normalize_laterality(
    pixel_array: np.ndarray,
    laterality: str,
    anatomical_plane: str,
) -> np.ndarray:
    if laterality != "L":
        return pixel_array

    if anatomical_plane in ("Coronal", "Axial"):
        return np.fliplr(pixel_array)

    return pixel_array
