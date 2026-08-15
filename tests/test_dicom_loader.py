"""Tests for DICOM loader."""

from pathlib import Path

import pytest

from src.imaging.dicom_loader import (
    get_series_metadata,
    list_study_series,
    load_series,
    load_study,
)


def test_load_series_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_series("nonexistent_study", "nonexistent_series", Path("/tmp"))


def test_load_study_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_study("nonexistent_study", Path("/tmp"))


def test_get_series_metadata_missing_directory():
    with pytest.raises(FileNotFoundError):
        get_series_metadata("nonexistent_study", "nonexistent_series", Path("/tmp"))


def test_list_study_series_missing_directory():
    with pytest.raises(FileNotFoundError):
        list_study_series("nonexistent_study", Path("/tmp"))
