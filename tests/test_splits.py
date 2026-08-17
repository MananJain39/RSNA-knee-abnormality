"""Tests for src.data.splits.

These tests use a synthetic frame and a quick on-disk check against the real
``train.csv`` that verifies the no-leak property.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_COLUMNS
from src.data.loader import load_train
from src.data.splits import (
    _normalize_report,
    make_split,
    persist_split,
)


def _synthetic() -> pd.DataFrame:
    """8 studies: two pairs sharing a report (4→8 cluster), one gold study, normal noise."""
    return pd.DataFrame(
        {
            "StudyInstanceUID": [f"study_{i}" for i in range(8)],
            "Report": [
                "alpha bravo",       # 0
                "ALPHA   BRAVO  ",   # 1  (same group as 0 after normalization)
                "charlie delta",     # 2
                "charlie\ndelta",    # 3  (same group as 2)
                "echo",              # 4
                "foxtrot",           # 5
                "golf",              # 6
                "hotel",             # 7  (gold)
            ],
            "ACL": [np.nan] * 7 + [1.0],
            "MCL": [np.nan] * 7 + [0.0],
            "Medial Meniscus": [np.nan] * 7 + [1.0],
            "Lateral Meniscus": [np.nan] * 7 + [0.0],
            "Medial OA": [np.nan] * 7 + [0.0],
            "Lateral OA": [np.nan] * 7 + [0.0],
            "PF OA": [np.nan] * 7 + [0.0],
            "Effusion": [np.nan] * 7 + [1.0],
            "Synovitis": [np.nan] * 7 + [0.0],
            "Baker's": [np.nan] * 7 + [0.0],
            "Contusion": [np.nan] * 7 + [1.0],
            "Fracture": [np.nan] * 7 + [0.0],
        }
    )


def test_normalize_report_collapses_whitespace_and_case():
    assert _normalize_report("  ALPHA   Bravo  ") == "alpha bravo"
    assert _normalize_report("alpha\nbravo\tcharlie") == "alpha bravo charlie"


def test_normalize_report_handles_nan():
    assert _normalize_report(np.nan) == "<NAN_REPORT>"
    assert _normalize_report(None) == "<NAN_REPORT>"


def test_synthetic_split_has_no_overlap():
    train_df, val_df = make_split(_synthetic(), val_frac=0.25, seed=42, reserve_gold_to_val=True)
    assert set(train_df["StudyInstanceUID"]).isdisjoint(set(val_df["StudyInstanceUID"]))


def test_synthetic_split_keeps_duplicate_reports_together():
    train_df, val_df = make_split(_synthetic(), val_frac=0.25, seed=42, reserve_gold_to_val=True)

    def _groups(df: pd.DataFrame) -> set[str]:
        return {_normalize_report(x) for x in df["Report"]}

    assert _groups(train_df).isdisjoint(_groups(val_df))


def test_synthetic_split_gold_lands_on_val():
    train_df, val_df = make_split(_synthetic(), val_frac=0.25, seed=42, reserve_gold_to_val=True)
    assert (train_df[LABEL_COLUMNS[0]].notna()).sum() == 0
    assert (val_df[LABEL_COLUMNS[0]].notna()).sum() == 1


def test_synthetic_split_rejects_bad_val_frac():
    with pytest.raises(ValueError):
        make_split(_synthetic(), val_frac=0.0)
    with pytest.raises(ValueError):
        make_split(_synthetic(), val_frac=1.0)


def test_real_split_no_overlap_no_leak():
    """Smoke test against the real train.csv — guards the leak invariant."""
    train_df, val_df = make_split(load_train(), val_frac=0.20, seed=42, reserve_gold_to_val=True)

    assert set(train_df["StudyInstanceUID"]).isdisjoint(set(val_df["StudyInstanceUID"]))

    def _groups(df: pd.DataFrame) -> set[str]:
        return {_normalize_report(x) for x in df["Report"]}

    overlap = _groups(train_df) & _groups(val_df)
    assert not overlap, (
        f"{len(overlap)} reports span both splits — report-grouping broken"
    )

    assert (train_df[LABEL_COLUMNS[0]].notna()).sum() == 0
    assert (val_df[LABEL_COLUMNS[0]].notna()).sum() == 58

    print(
        f"split: train={len(train_df)} val={len(val_df)} "
        f"train_unique_reports={len(_groups(train_df))} "
        f"val_unique_reports={len(_groups(val_df))}"
    )


def test_persist_split_roundtrip(tmp_path):
    train_df, val_df = make_split(_synthetic(), val_frac=0.25, seed=42)
    paths = persist_split(train_df, val_df, output_dir=tmp_path / "splits", tag="test")

    train_loaded = pd.read_csv(paths["train"])
    val_loaded = pd.read_csv(paths["val"])

    assert len(train_loaded) == len(train_df)
    assert len(val_loaded) == len(val_df)
    assert paths["train"].exists() and paths["val"].exists()
