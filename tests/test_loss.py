"""Tests for src.classical.loss — runs without torch/GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.loss import confidence_weighted_bce


def test_perfect_prediction_loss_near_zero():
    """logits=10 with label=1 → p≈1 → loss≈0."""
    logits = np.full((1, 12), 10.0)
    labels = np.ones((1, 12))
    weights = np.ones((1, 12))
    loss = confidence_weighted_bce(logits, labels, weights)
    assert loss < 1e-3


def test_worst_prediction_loss_high():
    """logits=10 (predict 1) with label=0 → loss ≈ 10."""
    logits = np.full((1, 12), 10.0)
    labels = np.zeros((1, 12))
    weights = np.ones((1, 12))
    loss = confidence_weighted_bce(logits, labels, weights)
    assert loss > 5.0


def test_zero_weight_excludes_cell():
    """A cell with weight=0 contributes nothing — others unaffected."""
    logits = np.array([[10.0, 10.0]])
    labels = np.array([[1.0, 0.0]])
    weights = np.array([[1.0, 0.0]])
    loss = confidence_weighted_bce(logits, labels, weights)
    assert loss < 1e-3


def test_silent_findings_excluded_from_loss():
    """NaN labels become 0.5 + 0 weight and contribute nothing."""
    logits = np.full((1, 12), 10.0)
    labels = np.full((1, 12), np.nan)
    weights = np.full((1, 12), 0.3)
    loss = confidence_weighted_bce(logits, labels, weights)
    assert loss == 0.0


def test_handles_negative_logits():
    """Sigmoid branch for x<0 must not overflow."""
    logits = np.full((1, 12), -100.0)
    labels = np.zeros((1, 12))
    weights = np.ones((1, 12))
    loss = confidence_weighted_bce(logits, labels, weights)
    assert loss < 1e-3


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confidence_weighted_bce(np.zeros((2, 12)), np.zeros((2, 11)), np.zeros((2, 12)))


def test_empty_weight_sum_returns_zero():
    """If total weight is zero, return 0 (not NaN)."""
    logits = np.zeros((1, 12))
    labels = np.ones((1, 12))
    weights = np.zeros((1, 12))
    loss = confidence_weighted_bce(logits, labels, weights)
    assert loss == 0.0


def test_batch_size_independence():
    """Same per-cell loss should be returned regardless of batch size."""
    logits = np.array([[2.0, -2.0]])
    labels = np.array([[1.0, 0.0]])
    weights = np.array([[0.7, 0.7]])

    single = confidence_weighted_bce(logits, labels, weights)
    tiled_logits = np.tile(logits, (4, 1))
    tiled_labels = np.tile(labels, (4, 1))
    tiled_weights = np.tile(weights, (4, 1))
    batched = confidence_weighted_bce(tiled_logits, tiled_labels, tiled_weights)
    assert abs(single - batched) < 1e-9
