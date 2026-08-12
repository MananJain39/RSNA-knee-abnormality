"""Tests for evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.validation.metrics import (
    macro_auc,
    per_label_auc,
    classification_report_detailed,
)
from src.config import LABEL_COLUMNS


def test_macro_auc_perfect():
    y_true = pd.DataFrame({
        "ACL": [0, 0, 1, 1],
        "MCL": [0, 1, 0, 1],
    })
    y_pred = pd.DataFrame({
        "ACL": [0.0, 0.1, 0.9, 1.0],
        "MCL": [0.1, 0.8, 0.2, 0.9],
    })
    auc = macro_auc(y_true, y_pred, ["ACL", "MCL"])
    assert auc == 1.0


def test_macro_auc_random():
    np.random.seed(42)
    y_true = pd.DataFrame({
        "ACL": np.random.randint(0, 2, 100),
    })
    y_pred = pd.DataFrame({
        "ACL": np.random.rand(100),
    })
    auc = macro_auc(y_true, y_pred, ["ACL"])
    assert 0.3 < auc < 0.7


def test_macro_auc_constant_label():
    """AUC should handle constant labels gracefully."""
    y_true = pd.DataFrame({
        "ACL": [0, 0, 0, 0],
    })
    y_pred = pd.DataFrame({
        "ACL": [0.1, 0.2, 0.3, 0.4],
    })
    auc = macro_auc(y_true, y_pred, ["ACL"])
    assert auc == 0.0


def test_per_label_auc():
    y_true = pd.DataFrame({
        "ACL": [0, 0, 1, 1],
        "MCL": [0, 1, 0, 1],
    })
    y_pred = pd.DataFrame({
        "ACL": [0.0, 0.1, 0.9, 1.0],
        "MCL": [0.1, 0.8, 0.2, 0.9],
    })
    aucs = per_label_auc(y_true, y_pred, ["ACL", "MCL"])
    assert aucs["ACL"] == 1.0
    assert aucs["MCL"] == 1.0


def test_classification_report():
    y_true = pd.DataFrame({
        "ACL": [0, 0, 1, 1],
    })
    y_pred = pd.DataFrame({
        "ACL": [0.1, 0.2, 0.8, 0.9],
    })
    report = classification_report_detailed(y_true, y_pred, label_columns=["ACL"])
    assert len(report) == 1
    assert report.iloc[0]["Label"] == "ACL"
    assert report.iloc[0]["AUC"] == 1.0
    assert report.iloc[0]["Accuracy"] == 1.0


def test_macro_auc_with_nan():
    """Per-label AUC should return NaN for constant labels."""
    y_true = pd.DataFrame({
        "ACL": [0, 0, 0, 0],
        "MCL": [0, 1, 0, 1],
    })
    y_pred = pd.DataFrame({
        "ACL": [0.1, 0.2, 0.3, 0.4],
        "MCL": [0.1, 0.8, 0.2, 0.9],
    })
    aucs = per_label_auc(y_true, y_pred, ["ACL", "MCL"])
    assert np.isnan(aucs["ACL"])
    assert aucs["MCL"] == 1.0
