"""DICOM series loading utilities.

This is a stub module. Full implementation requires downloading the ~570 GB
train_series/ DICOM dataset from Kaggle.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def load_series(
    study_uid: str,
    series_uid: str,
    data_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load a DICOM series as a 3D numpy array.

    Args:
        study_uid: StudyInstanceUID
        series_uid: SeriesInstanceUID
        data_dir: Path to train_series/ directory

    Returns:
        3D numpy array of shape (n_slices, height, width)

    Raises:
        NotImplementedError: DICOM data not yet downloaded
    """
    raise NotImplementedError(
        "DICOM loading requires downloading the train_series/ dataset (~570 GB). "
        "Run: kaggle competitions download rsna-knee-abnormality-detection"
    )


def load_study(
    study_uid: str,
    data_dir: Optional[Path] = None,
) -> dict[str, np.ndarray]:
    """Load all series for a study.

    Args:
        study_uid: StudyInstanceUID
        data_dir: Path to train_series/ directory

    Returns:
        Dict mapping SeriesInstanceUID to 3D numpy arrays

    Raises:
        NotImplementedError: DICOM data not yet downloaded
    """
    raise NotImplementedError(
        "DICOM loading requires downloading the train_series/ dataset (~570 GB)."
    )


def get_series_metadata(
    study_uid: str,
    series_uid: str,
    data_dir: Optional[Path] = None,
) -> dict:
    """Extract metadata from a DICOM series.

    Args:
        study_uid: StudyInstanceUID
        series_uid: SeriesInstanceUID
        data_dir: Path to train_series/ directory

    Returns:
        Dict with metadata fields (spacing, orientation, dimensions, etc.)

    Raises:
        NotImplementedError: DICOM data not yet downloaded
    """
    raise NotImplementedError(
        "DICOM loading requires downloading the train_series/ dataset (~570 GB)."
    )


def list_study_series(
    study_uid: str,
    data_dir: Optional[Path] = None,
) -> list[str]:
    """List all series UIDs for a given study.

    Args:
        study_uid: StudyInstanceUID
        data_dir: Path to train_series/ directory

    Returns:
        List of SeriesInstanceUIDs

    Raises:
        NotImplementedError: DICOM data not yet downloaded
    """
    raise NotImplementedError(
        "DICOM loading requires downloading the train_series/ dataset (~570 GB)."
    )
