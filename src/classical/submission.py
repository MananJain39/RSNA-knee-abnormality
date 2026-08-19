"""Submission CSV validation and deterministic study-ID ordering."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from src.config import LABEL_COLUMNS
from src.data.schema import validate_submission_schema


class SubmissionProtocol(Protocol):
    """Validate and write prediction rows in deterministic submission order."""

    def write(
        self,
        predictions: pd.DataFrame,
        output_path: Path,
        sample_submission_path: Path | None = None,
    ) -> Path: ...


class SubmissionWriter:
    """Write a schema-valid competition submission CSV.

    When ``sample_submission_path`` is provided, the writer preserves the
    exact StudyInstanceUID order from the reference file. When not provided,
    it falls back to lexicographic order. Either way the submission is
    deterministic and reproducible.
    """

    def write(
        self,
        predictions: pd.DataFrame,
        output_path: Path,
        sample_submission_path: Path | None = None,
    ) -> Path:
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

        sorted_predictions = self._sort(predictions, sample_submission_path)
        sorted_predictions.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def _sort(
        predictions: pd.DataFrame,
        sample_submission_path: Path | None,
    ) -> pd.DataFrame:
        if sample_submission_path is not None and Path(sample_submission_path).exists():
            sample_order = pd.read_csv(sample_submission_path)["StudyInstanceUID"].tolist()
            order_map = {uid: i for i, uid in enumerate(sample_order)}
            ordered = predictions.copy()
            ordered["_order"] = ordered["StudyInstanceUID"].map(order_map)
            unknown_mask = ordered["_order"].isna()
            ordered.loc[unknown_mask, "_order"] = len(order_map) + (
                ordered.loc[unknown_mask].index.to_series()
                - ordered.index[0]
            )
            return ordered.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
        return predictions.sort_values("StudyInstanceUID").reset_index(drop=True)


__all__ = ["SubmissionProtocol", "SubmissionWriter"]
