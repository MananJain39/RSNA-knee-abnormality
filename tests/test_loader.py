"""Tests for data loading utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.loader import (
    load_train,
    load_train_series,
    get_gold_studies,
    get_unlabeled_studies,
    merge_study_series,
    get_series_summary,
)
from src.data.schema import validate_train_schema, validate_series_schema
from src.config import GOLD_STUDY_COUNT, LABEL_COLUMNS


def test_load_train():
    df = load_train()
    assert len(df) == 4407
    assert "StudyInstanceUID" in df.columns
    assert "Report" in df.columns
    assert all(col in df.columns for col in LABEL_COLUMNS)


def test_load_train_series():
    df = load_train_series()
    assert len(df) == 24371
    assert "StudyInstanceUID" in df.columns
    assert "SeriesInstanceUID" in df.columns


def test_gold_studies():
    train_df = load_train()
    gold_df = get_gold_studies(train_df)
    assert len(gold_df) == GOLD_STUDY_COUNT
    assert gold_df[LABEL_COLUMNS[0]].notna().all()


def test_unlabeled_studies():
    train_df = load_train()
    unlabeled_df = get_unlabeled_studies(train_df)
    assert len(unlabeled_df) == 4407 - GOLD_STUDY_COUNT
    assert unlabeled_df[LABEL_COLUMNS[0]].isna().all()


def test_merge_study_series():
    train_df = load_train()
    series_df = load_train_series()
    merged = merge_study_series(train_df, series_df)
    assert len(merged) == len(series_df)
    assert "Report" in merged.columns


def test_series_summary():
    series_df = load_train_series()
    summary = get_series_summary(series_df)
    assert len(summary) == series_df["StudyInstanceUID"].nunique()
    assert "n_series" in summary.columns


def test_validate_train_schema():
    df = load_train()
    errors = validate_train_schema(df)
    assert errors == [], f"Schema errors: {errors}"


def test_validate_series_schema():
    df = load_train_series()
    errors = validate_series_schema(df)
    assert errors == [], f"Schema errors: {errors}"


def test_no_patient_sex_column():
    """Verify PatientSex column is not present (contrary to competition description)."""
    df = load_train()
    assert "PatientSex" not in df.columns


def test_all_reports_present():
    """All 4407 studies should have reports."""
    df = load_train()
    assert df["Report"].notna().all()
    assert df["Report"].str.len().min() > 0


def test_label_values():
    """Gold labels should be exactly 0 or 1."""
    train_df = load_train()
    gold_df = get_gold_studies(train_df)
    for col in LABEL_COLUMNS:
        valid = gold_df[col].isin([0.0, 1.0]).all()
        assert valid, f"Column {col} has non-binary values"
