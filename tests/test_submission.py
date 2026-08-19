"""Tests for schema-safe submission writing."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.classical.submission import SubmissionWriter
from src.config import LABEL_COLUMNS


def _predictions():
    return pd.DataFrame(
        {
            "StudyInstanceUID": ["study_b", "study_a"],
            **{label: [0.2, 0.8] for label in LABEL_COLUMNS},
        }
    )


def test_submission_validates_schema(tmp_path):
    with patch("src.classical.submission.validate_submission_schema", return_value=[]) as validate:
        SubmissionWriter().write(_predictions(), tmp_path / "submission.csv")
    validate.assert_called_once()


def test_submission_sorts_by_study_uid(tmp_path):
    path = SubmissionWriter().write(_predictions(), tmp_path / "submission.csv")
    assert pd.read_csv(path)["StudyInstanceUID"].tolist() == ["study_a", "study_b"]


def test_submission_roundtrip(tmp_path):
    path = SubmissionWriter().write(_predictions(), tmp_path / "nested" / "submission.csv")
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == ["StudyInstanceUID", *LABEL_COLUMNS]
    assert len(loaded) == 2


def test_submission_rejects_nan(tmp_path):
    predictions = _predictions()
    predictions.loc[0, LABEL_COLUMNS[0]] = float("nan")
    with pytest.raises(ValueError, match="null"):
        SubmissionWriter().write(predictions, tmp_path / "submission.csv")


def test_submission_rejects_out_of_range_values(tmp_path):
    predictions = _predictions()
    predictions.loc[0, LABEL_COLUMNS[0]] = 1.1
    with pytest.raises(ValueError, match="outside"):
        SubmissionWriter().write(predictions, tmp_path / "submission.csv")


def test_submission_aligns_with_sample_order(tmp_path):
    sample = pd.DataFrame({
        "StudyInstanceUID": ["study_a", "study_c", "study_b"],
        **{label: [0.0] * 3 for label in LABEL_COLUMNS},
    })
    sample_path = tmp_path / "sample.csv"
    sample.to_csv(sample_path, index=False)

    predictions = pd.DataFrame({
        "StudyInstanceUID": ["study_b", "study_a", "study_c"],
        **{label: [0.1, 0.2, 0.3] for label in LABEL_COLUMNS},
    })
    path = SubmissionWriter().write(
        predictions,
        tmp_path / "submission.csv",
        sample_submission_path=sample_path,
    )
    assert pd.read_csv(path)["StudyInstanceUID"].tolist() == [
        "study_a", "study_c", "study_b",
    ]


def test_submission_handles_predictions_not_in_sample(tmp_path):
    """UIDs missing from the sample submission get appended at the end."""
    sample = pd.DataFrame({
        "StudyInstanceUID": ["study_a"],
        **{label: [0.0] for label in LABEL_COLUMNS},
    })
    sample_path = tmp_path / "sample.csv"
    sample.to_csv(sample_path, index=False)

    predictions = pd.DataFrame({
        "StudyInstanceUID": ["study_b", "study_a"],
        **{label: [0.5, 0.5] for label in LABEL_COLUMNS},
    })
    path = SubmissionWriter().write(
        predictions,
        tmp_path / "submission.csv",
        sample_submission_path=sample_path,
    )
    assert pd.read_csv(path)["StudyInstanceUID"].tolist() == ["study_a", "study_b"]
