"""Rank-mean ensemble for blending the main DINOv2 branch with per-cluster specialists.

The competition metric (macro-AUC) is **rank-invariant** — only the relative
order of scores matters.  Therefore the right way to blend probabilities from
different models is rank-averaging, not probability averaging.

Inputs:
    ``logits_per_branch`` — dict mapping branch name to ``np.ndarray`` of
    shape ``(n_studies, n_diagnoses)``.  Branches can be logits (will be
    turned into ranks) or already-probability arrays (will be used as-is).

Output:
    ``np.ndarray`` of shape ``(n_studies, n_diagnoses)`` in ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata

from src.config import CLASSICAL_ENSEMBLE_WEIGHTS, LABEL_COLUMNS


def _to_rank(arr: np.ndarray) -> np.ndarray:
    """Per-column rank-normalize to ``[0, 1]``."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    out = np.zeros_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = rankdata(arr[:, j]) - 1
    denom = max(arr.shape[0] - 1, 1)
    return out / denom


def rank_mean_logits(
    logits_per_branch: Mapping[str, np.ndarray],
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Compute the weighted rank-mean across branches.

    Args:
        logits_per_branch: Each branch is a ``(n, 12)`` array. Each branch
            contributes ``weights[name] * rank(arr)`` to the final score.
        weights: Optional weighting dict; defaults to
            ``CLASSICAL_ENSEMBLE_WEIGHTS``. Missing keys fall back to weight 0
            (so omitting a branch zeroes it out cleanly).

    Returns:
        ``(n, 12)`` array in ``[0, 1]`` — the per-study, per-diagnosis
        rank-mean ensemble score.
    """
    if not logits_per_branch:
        raise ValueError("Need at least one branch")

    keys = list(logits_per_branch.keys())
    sample = logits_per_branch[keys[0]]
    if sample.ndim != 2 or sample.shape[1] != len(LABEL_COLUMNS):
        raise ValueError(
            f"Each branch must be (n, {len(LABEL_COLUMNS)}), "
            f"got shape {sample.shape}"
        )

    if weights is None:
        weights = CLASSICAL_ENSEMBLE_WEIGHTS

    total_weight = 0.0
    accum = np.zeros_like(sample, dtype=float)
    for name, arr in logits_per_branch.items():
        w = float(weights.get(name, 0.0))
        if w == 0:
            continue
        accum += w * _to_rank(np.asarray(arr, dtype=float))
        total_weight += w

    if total_weight <= 0:
        raise ValueError("Total ensemble weight must be > 0")

    return accum / total_weight


def to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Stable sigmoid for use when callers want probabilities, not ranks."""
    import scipy.special as sp

    return sp.expit(logits)


__all__ = ["rank_mean_logits", "to_probabilities"]
