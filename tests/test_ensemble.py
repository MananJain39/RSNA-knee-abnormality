"""Tests for src.classical.ensemble — works without torch/GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.ensemble import rank_mean_logits, to_probabilities


def _synthetic_binary(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, 12))
    X_main = rng.standard_normal((n, 12)) + y * 1.5
    X_spec = rng.standard_normal((n, 12)) + y * 1.0
    return y, {"main": X_main, "specialist": X_spec}


def test_rank_mean_logits_shapes_match_labels():
    y, logits = _synthetic_binary(n=50)
    out = rank_mean_logits(logits)
    assert out.shape == (50, 12)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_rank_mean_logits_falls_back_to_main_when_specialist_zero():
    y, logits = _synthetic_binary(n=80)
    out_no_spec = rank_mean_logits(logits, weights={"main": 1.0, "specialist": 0.0})
    out_all_main = rank_mean_logits({"main": logits["main"]}, weights={"main": 1.0})
    np.testing.assert_allclose(out_no_spec, out_all_main, atol=1e-9)


def test_rank_mean_logits_rejects_bad_shape():
    with pytest.raises(ValueError):
        rank_mean_logits({"main": np.zeros((10, 7))})


def test_rank_mean_logits_rejects_empty():
    with pytest.raises(ValueError):
        rank_mean_logits({})


def test_rank_mean_logits_beats_either_branch_alone():
    """Pure signal check: the rank-mean should at least not tank AUC vs singletons."""
    y, logits = _synthetic_binary(n=400, seed=123)

    auc_main = roc_auc_score(y, logits["main"], average="macro")
    auc_spec = roc_auc_score(y, logits["specialist"], average="macro")
    ens = rank_mean_logits(logits, weights={"main": 0.5, "specialist": 0.5})
    auc_ens = roc_auc_score(y, ens, average="macro")

    assert auc_ens >= min(auc_main, auc_spec) - 1e-6
    print(f"main={auc_main:.3f} spec={auc_spec:.3f} ens={auc_ens:.3f}")


def test_to_probabilities_monotone():
    """Sigmoid must be monotonically increasing in its input."""
    x = np.linspace(-3, 3, 50)
    p = to_probabilities(x)
    assert np.all(np.diff(p) > 0)
    assert p[0] < 0.05
    assert p[-1] > 0.95
