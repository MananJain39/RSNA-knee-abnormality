"""Load and merge competition data files."""

import pandas as pd

from src.config import (
    LABEL_COLUMNS,
    TRAIN_CSV,
    TRAIN_SERIES_CSV,
    TEST_CSV,
    TEST_SERIES_CSV,
    SAMPLE_SUBMISSION_CSV,
)


def load_train() -> pd.DataFrame:
    """Load training data with reports and labels."""
    return pd.read_csv(TRAIN_CSV)


def load_train_series() -> pd.DataFrame:
    """Load training series metadata."""
    return pd.read_csv(TRAIN_SERIES_CSV)


def load_test() -> pd.DataFrame:
    """Load test study IDs."""
    return pd.read_csv(TEST_CSV)


def load_test_series() -> pd.DataFrame:
    """Load test series metadata."""
    return pd.read_csv(TEST_SERIES_CSV)


def load_sample_submission() -> pd.DataFrame:
    """Load sample submission file."""
    return pd.read_csv(SAMPLE_SUBMISSION_CSV)


def get_gold_studies(train_df: pd.DataFrame) -> pd.DataFrame:
    """Get the 58 studies with expert labels."""
    gold_mask = train_df[LABEL_COLUMNS[0]].notna()
    return train_df[gold_mask].copy()


def get_unlabeled_studies(train_df: pd.DataFrame) -> pd.DataFrame:
    """Get studies without expert labels (need report-based extraction)."""
    unlabeled_mask = train_df[LABEL_COLUMNS[0]].isna()
    return train_df[unlabeled_mask].copy()


def merge_study_series(
    train_df: pd.DataFrame, series_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge study-level data with series metadata."""
    merged = train_df.merge(series_df, on="StudyInstanceUID", how="left")
    return merged


def get_series_summary(series_df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics per study."""
    summary = series_df.groupby("StudyInstanceUID").agg(
        n_series=("SeriesInstanceUID", "count"),
        n_sagittal=("Anatomical_Plane", lambda x: (x == "Sagittal").sum()),
        n_coronal=("Anatomical_Plane", lambda x: (x == "Coronal").sum()),
        n_axial=("Anatomical_Plane", lambda x: (x == "Axial").sum()),
        n_fluid_sensitive=("Fluid_Sensitive", "sum"),
        n_fat_suppressed=("Fat_Suppression", "sum"),
    ).reset_index()
    return summary
