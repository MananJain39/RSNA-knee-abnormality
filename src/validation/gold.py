"""Utilities for working with the 58 expert-labeled gold studies."""

import pandas as pd

from src.config import LABEL_COLUMNS, GOLD_STUDY_COUNT
from src.data.loader import load_train


def load_gold_labels(train_df: pd.DataFrame = None) -> pd.DataFrame:
    """Load the 58 studies with expert labels.

    Returns DataFrame with StudyInstanceUID and 12 label columns.
    """
    if train_df is None:
        train_df = load_train()

    gold_mask = train_df[LABEL_COLUMNS[0]].notna()
    gold_df = train_df[gold_mask][["StudyInstanceUID"] + LABEL_COLUMNS].copy()

    assert len(gold_df) == GOLD_STUDY_COUNT, (
        f"Expected {GOLD_STUDY_COUNT} gold studies, got {len(gold_df)}"
    )

    return gold_df


def get_gold_uids(train_df: pd.DataFrame = None) -> list[str]:
    """Get list of gold study UIDs."""
    gold_df = load_gold_labels(train_df)
    return gold_df["StudyInstanceUID"].tolist()


def get_gold_label_matrix(train_df: pd.DataFrame = None) -> pd.DataFrame:
    """Get gold labels as a matrix (studies x labels)."""
    gold_df = load_gold_labels(train_df)
    return gold_df.set_index("StudyInstanceUID")[LABEL_COLUMNS]


def get_label_prevalence(train_df: pd.DataFrame = None) -> pd.Series:
    """Get prevalence of each label among gold studies."""
    gold_df = load_gold_labels(train_df)
    prevalence = gold_df[LABEL_COLUMNS].sum() / len(gold_df)
    return prevalence.sort_values(ascending=False)


def compare_labels(
    predicted: pd.DataFrame,
    gold: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compare predicted labels against gold labels.

    Args:
        predicted: DataFrame with StudyInstanceUID and label columns (values 0-1)
        gold: DataFrame with StudyInstanceUID and label columns (values 0/1)
        threshold: Threshold for converting predictions to binary

    Returns:
        DataFrame with per-study, per-label comparison
    """
    merged = predicted.merge(gold, on="StudyInstanceUID", suffixes=("_pred", "_gold"))

    results = []
    for _, row in merged.iterrows():
        for col in LABEL_COLUMNS:
            pred_val = row[f"{col}_pred"]
            gold_val = row[f"{col}_gold"]
            pred_binary = 1 if pred_val >= threshold else 0

            results.append({
                "StudyInstanceUID": row["StudyInstanceUID"],
                "Label": col,
                "Predicted": pred_val,
                "Predicted_Binary": pred_binary,
                "Gold": gold_val,
                "Correct": pred_binary == gold_val,
            })

    return pd.DataFrame(results)
