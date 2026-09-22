"""Reliability-weighted label fusion from multiple extraction sources.

Fuses graded label DataFrames (one row per StudyInstanceUID, the 12 label
columns in [0, 1]) into a single soft-label set:

1. Per-label reliability: each source's per-label AUC against the gold-58
   studies (rank-based, matching the competition metric).
2. Per-cell fusion: reliability-weighted average of every source's value.
3. Hard-vote snapping: when every source binarized at 0.5 agrees, the fused
   value snaps to that hard vote (0 or 1).
4. Gold override: gold studies keep their expert labels exactly.

Fusion is fully symmetric: every source's graded value enters the average
with its own per-label weight, and no truncation is applied anywhere. (An
earlier ``int()``-truncation fusion bug documented by a top team made errors
asymmetric — one direction only — which is why symmetry is asserted in
tests here.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMNS


def per_label_reliability(
    sources: dict[str, pd.DataFrame],
    gold: pd.DataFrame,
) -> pd.DataFrame:
    """Per-label AUC of each source against gold studies.

    Returns a DataFrame indexed by source name with one column per label.
    Labels that are constant within gold fall back to 0.5 (no information).
    """
    from sklearn.metrics import roc_auc_score

    aucs: dict[str, dict[str, float]] = {}
    for name, df in sources.items():
        merged = df.merge(gold, on="StudyInstanceUID", suffixes=("_pred", "_gold"))
        row: dict[str, float] = {}
        for col in LABEL_COLUMNS:
            y_true = merged[f"{col}_gold"].to_numpy(dtype=float)
            y_pred = merged[f"{col}_pred"].to_numpy(dtype=float)
            if len(np.unique(y_true)) < 2:
                row[col] = 0.5
                continue
            try:
                row[col] = float(roc_auc_score(y_true, y_pred))
            except ValueError:
                row[col] = 0.5
        aucs[name] = row
    return pd.DataFrame(aucs).T


def fuse_labels(
    sources: dict[str, pd.DataFrame],
    reliability: pd.DataFrame,
) -> pd.DataFrame:
    """Fuse sources into one soft-label DataFrame.

    Every source contributes its graded value weighted by its per-label
    reliability; absent (NaN) cells abstain from both the average and the
    hard-vote check. When two or more sources are present and their hard
    votes (binarized at 0.5) all agree, the fused value snaps to that hard
    vote (0 or 1). Single-source cells keep their graded value.
    """
    names = list(sources)
    cols = ["StudyInstanceUID", *LABEL_COLUMNS]
    aligned = {
        name: sources[name][cols].set_index("StudyInstanceUID").astype(float)
        for name in names
    }

    all_index = sorted(set().union(*(df.index for df in aligned.values())))
    values = np.stack(
        [aligned[name].reindex(all_index)[LABEL_COLUMNS].to_numpy() for name in names]
    )  # (S, N, L)
    source_weights = np.stack(
        [reliability.loc[name].to_numpy(dtype=float) for name in names]
    )  # (S, L)

    present = np.isfinite(values)
    cell_weights = np.broadcast_to(source_weights[:, None, :], values.shape) * present
    weight_sum = cell_weights.sum(axis=0)  # (N, L)

    with np.errstate(invalid="ignore"):
        weighted_sum = np.nansum(values * cell_weights, axis=0)
    fused = np.where(weight_sum > 0, weighted_sum / np.maximum(weight_sum, 1e-9), 0.5)
    fused = fused.clip(0.0, 1.0)

    hard_votes = (values >= 0.5).astype(float)
    n_present = present.sum(axis=0)
    agree = n_present >= 2
    for i in range(len(names) - 1):
        pair_agrees = (hard_votes[i] == hard_votes[i + 1]) | ~present[i] | ~present[i + 1]
        agree &= pair_agrees
    first_present = np.argmax(present, axis=0)
    vote_value = np.take_along_axis(hard_votes, first_present[None, ...], axis=0)[0]
    fused = np.where(agree, vote_value, fused)

    result = pd.DataFrame(fused, columns=LABEL_COLUMNS)
    result.insert(0, "StudyInstanceUID", all_index)
    return result


def apply_gold_override(fused: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Replace fused labels with expert labels wherever gold exists."""
    merged = fused.merge(gold, on="StudyInstanceUID", suffixes=("", "_gold"), how="left")
    for col in LABEL_COLUMNS:
        gold_col = f"{col}_gold"
        if gold_col in merged.columns:
            mask = merged[gold_col].notna()
            merged.loc[mask, col] = merged.loc[mask, gold_col]
            merged.drop(columns=[gold_col], inplace=True)
    return merged


__all__ = ["per_label_reliability", "fuse_labels", "apply_gold_override"]
