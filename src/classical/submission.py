"""Submission CSV validation and deterministic study-ID ordering."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from src.config import LABEL_COLUMNS
from src.data.schema import validate_submission_schema


class SubmissionProtocol(Protocol):
    """Validate and write prediction rows in deterministic submission order."""

    def write(self, predictions: pd.DataFrame, output_path: Path) -> Path: ...


class SubmissionWriter:
    """Write a schema-valid competition submission CSV.

    The writer sorts StudyInstanceUID lexicographically. The competition's
    submission join key is the UID, so this deterministic order also makes
    generated files straightforward to compare and reproduce.
    """

    def write(self, predictions: pd.DataFrame, output_path: Path) -> Path:
        """Validate, sort, and persist predictions; raise ValueError if invalid."""
        expected = ["StudyInstanceUID", *LABEL_COLUMNS]
        if list(predictions.columns) != expected:
            raise ValueError(f"Submission columns must be exactly {expected}")
        errors = validate_submission_schema(predictions)
        if errors:
            raise ValueError("Invalid submission: " + "; ".join(errors))
        if predictions["StudyInstanceUID"].duplicated().any():
            raise ValueError("Invalid submission: StudyInstanceUID has duplicates")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.sort_values("StudyInstanceUID").to_csv(output_path, index=False)
        return output_path


__all__ = ["SubmissionProtocol", "SubmissionWriter"]
