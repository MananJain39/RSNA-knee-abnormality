"""Base protocol for label extractors."""

from typing import Protocol
from src.config import LABEL_COLUMNS


LabelDict = dict[str, float]


class LabelExtractor(Protocol):
    """Protocol for extracting labels from radiology reports."""

    def extract(self, report: str) -> LabelDict:
        """Extract labels from a single report.

        Returns dict mapping label name to confidence score.
        Use 0.5 for "not addressed" / uncertain.
        """
        ...

    def extract_batch(self, reports: list[str]) -> list[LabelDict]:
        """Extract labels from multiple reports."""
        return [self.extract(report) for report in reports]


def labels_to_dataframe(
    study_uids: list[str], label_dicts: list[LabelDict]
) -> "pd.DataFrame":
    """Convert list of label dicts to DataFrame."""
    import pandas as pd

    rows = []
    for uid, labels in zip(study_uids, label_dicts):
        row = {"StudyInstanceUID": uid}
        for col in LABEL_COLUMNS:
            row[col] = labels.get(col, 0.5)
        rows.append(row)
    return pd.DataFrame(rows)
