"""DICOM series loading utilities."""

from pathlib import Path

import numpy as np
import pydicom

from src.imaging.geometry import sort_by_physical_position


def _read_dicom(filepath: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(filepath), stop_before_pixels=False)


def _get_pixel_array(ds: pydicom.Dataset) -> np.ndarray:
    return ds.pixel_array.astype(np.float32)


def load_series(
    study_uid: str,
    series_uid: str,
    data_dir: Path | None = None,
) -> np.ndarray:
    if data_dir is None:
        from src.config import DICOM_DIR
        data_dir = DICOM_DIR

    series_dir = data_dir / study_uid / series_uid
    if not series_dir.exists():
        raise FileNotFoundError(f"Series directory not found: {series_dir}")

    dicom_files = sorted(series_dir.glob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    datasets = [_read_dicom(f) for f in dicom_files]
    sorted_datasets = sort_by_physical_position(datasets)

    slices = [_get_pixel_array(ds) for ds in sorted_datasets]
    return np.stack(slices, axis=0)


def load_study(
    study_uid: str,
    data_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    if data_dir is None:
        from src.config import DICOM_DIR
        data_dir = DICOM_DIR

    study_dir = data_dir / study_uid
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    series_uids = list_study_series(study_uid, data_dir)
    result = {}
    for series_uid in series_uids:
        result[series_uid] = load_series(study_uid, series_uid, data_dir)

    return result


def get_series_metadata(
    study_uid: str,
    series_uid: str,
    data_dir: Path | None = None,
) -> dict:
    if data_dir is None:
        from src.config import DICOM_DIR
        data_dir = DICOM_DIR

    series_dir = data_dir / study_uid / series_uid
    dicom_files = sorted(series_dir.glob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    ds = _read_dicom(dicom_files[0])

    return {
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "pixel_spacing": [float(x) for x in ds.PixelSpacing],
        "slice_thickness": float(ds.SliceThickness),
        "image_orientation": [float(x) for x in ds.ImageOrientationPatient],
        "image_position": [float(x) for x in ds.ImagePositionPatient],
        "modality": str(ds.Modality),
        "transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
        "n_slices": len(dicom_files),
    }


def list_study_series(
    study_uid: str,
    data_dir: Path | None = None,
) -> list[str]:
    if data_dir is None:
        from src.config import DICOM_DIR
        data_dir = DICOM_DIR

    study_dir = data_dir / study_uid
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    return [d.name for d in sorted(study_dir.iterdir()) if d.is_dir()]
