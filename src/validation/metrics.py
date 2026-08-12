"""Evaluation metrics for label quality and model performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.config import LABEL_COLUMNS


def macro_auc(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    label_columns: list[str] = None,
) -> float:
    """Calculate macro-averaged AUC across all labels.

    This is the competition metric.

    Args:
        y_true: DataFrame with true binary labels (0/1)
        y_pred: DataFrame with predicted probabilities (0-1)
        label_columns: List of label column names (defaults to LABEL_COLUMNS)

    Returns:
        Macro-averaged AUC score
    """
    if label_columns is None:
        label_columns = LABEL_COLUMNS

    auc_scores = []
    for col in label_columns:
        if col in y_true.columns and col in y_pred.columns:
            true_vals = y_true[col].values
            pred_vals = y_pred[col].values

            if len(np.unique(true_vals)) < 2:
                continue

            try:
                auc = roc_auc_score(true_vals, pred_vals)
                auc_scores.append(auc)
            except ValueError:
                continue

    if not auc_scores:
        return 0.0

    return float(np.mean(auc_scores))


def per_label_auc(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    label_columns: list[str] = None,
) -> dict[str, float]:
    """Calculate AUC for each label individually.

    Args:
        y_true: DataFrame with true binary labels
        y_pred: DataFrame with predicted probabilities
        label_columns: List of label column names

    Returns:
        Dict mapping label name to AUC score
    """
    if label_columns is None:
        label_columns = LABEL_COLUMNS

    results = {}
    for col in label_columns:
        if col in y_true.columns and col in y_pred.columns:
            true_vals = y_true[col].values
            pred_vals = y_pred[col].values

            if len(np.unique(true_vals)) < 2:
                results[col] = float("nan")
                continue

            try:
                auc = roc_auc_score(true_vals, pred_vals)
                results[col] = float(auc)
            except ValueError:
                results[col] = float("nan")

    return results


def classification_report_detailed(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    threshold: float = 0.5,
    label_columns: list[str] = None,
) -> pd.DataFrame:
    """Generate detailed classification report for each label.

    Args:
        y_true: DataFrame with true binary labels
        y_pred: DataFrame with predicted probabilities
        threshold: Threshold for converting probabilities to binary
        label_columns: List of label column names

    Returns:
        DataFrame with metrics per label
    """
    if label_columns is None:
        label_columns = LABEL_COLUMNS

    results = []
    for col in label_columns:
        if col not in y_true.columns or col not in y_pred.columns:
            continue

        true_vals = y_true[col].values
        pred_probs = y_pred[col].values
        pred_binary = (pred_probs >= threshold).astype(int)

        try:
            auc = roc_auc_score(true_vals, pred_probs)
        except ValueError:
            auc = float("nan")

        acc = accuracy_score(true_vals, pred_binary)
        f1 = f1_score(true_vals, pred_binary, zero_division=0)

        results.append({
            "Label": col,
            "AUC": auc,
            "Accuracy": acc,
            "F1": f1,
            "Prevalence": true_vals.mean(),
            "N_Positive": int(true_vals.sum()),
            "N_Negative": int(len(true_vals) - true_vals.sum()),
        })

    return pd.DataFrame(results)


def evaluate_labels_vs_gold(
    predicted_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    label_columns: list[str] = None,
) -> dict:
    """Evaluate predicted labels against gold standard.

    Args:
        predicted_df: DataFrame with StudyInstanceUID and predicted labels
        gold_df: DataFrame with StudyInstanceUID and gold labels
        label_columns: List of label column names

    Returns:
        Dict with macro_auc, per_label_auc, and detailed report
    """
    merged = predicted_df.merge(gold_df, on="StudyInstanceUID", suffixes=("_pred", "_gold"))

    if label_columns is None:
        label_columns = LABEL_COLUMNS

    y_true = merged[[f"{col}_gold" for col in label_columns]]
    y_true.columns = label_columns
    y_pred = merged[[f"{col}_pred" for col in label_columns]]
    y_pred.columns = label_columns

    return {
        "macro_auc": macro_auc(y_true, y_pred, label_columns),
        "per_label_auc": per_label_auc(y_true, y_pred, label_columns),
        "detailed_report": classification_report_detailed(y_true, y_pred, label_columns=label_columns),
    }
