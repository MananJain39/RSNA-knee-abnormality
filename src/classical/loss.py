"""Confidence-weighted binary cross-entropy loss.

Reads logits[B, 12] + labels[B, 12] + weights[B, 12] → scalar.

NaN handling:
  - 'Not addressed' (NaN label) is mapped to 0.5 in ``KneeMRIDataset`` and
    given weight 0.0 there. The loss reduction therefore excludes those
    cells naturally.
  - Any remaining NaN in logits/labels is treated as 0 weight.

The confidence weight formula matches ``process_labels.compute_confidence_weights``:
    weight = 0.3          if label == 0.5
             abs(label - 0.5) * 2 + 0.5   otherwise

The function is pure-numpy so tests run without torch. The trainer can wrap
it with ``torch.from_numpy`` to get autograd through numpy ops, or
re-implement in torch for performance — both honor the contract.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


# === CONTRACT (top of file for both agents to read) ===
class LossProtocol(Protocol):
    """Confidence-weighted BCE. Returns scalar float."""

    def __call__(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> float: ...


# === IMPLEMENTATION ===

_EPS = 1e-7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def confidence_weighted_bce(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    eps: float = _EPS,
) -> float:
    """Numerically stable weighted BCE.

    Args:
        logits: [B, 12] raw model outputs (any float).
        labels: [B, 12] targets in [0, 1] (NaN already converted to 0.5
            upstream; this function tolerates any remaining NaN by zeroing
            the corresponding weight).
        weights: [B, 12] confidence weights in [0, 1].
        eps: Numerically-stable clamp.

    Returns:
        Scalar mean over all (b, k) where weight[b, k] > 0.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if logits.shape != labels.shape or logits.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch: logits {logits.shape}, "
            f"labels {labels.shape}, weights {weights.shape}"
        )

    safe_labels = np.where(np.isnan(labels), 0.5, labels)
    safe_weights = np.where(np.isnan(labels), 0.0, weights)
    safe_weights = np.clip(safe_weights, 0.0, 1.0)

    p = _sigmoid(logits)
    p = np.clip(p, eps, 1.0 - eps)
    bce = -(safe_labels * np.log(p) + (1.0 - safe_labels) * np.log(1.0 - p))

    total_loss = (bce * safe_weights).sum()
    total_weight = safe_weights.sum()
    if total_weight <= 0:
        return 0.0
    return float(total_loss / total_weight)


__all__ = ["LossProtocol", "confidence_weighted_bce"]
