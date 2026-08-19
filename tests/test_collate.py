"""Tests for src.classical.collate — runs without torch/GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.collate import collate_batch
from src.classical.dataset import Sample
from src.imaging.slot_builder import SLOT_NAMES


def _sample(study_uid, slot_subset=None, n_groups=10, group_size=3, h=8, w=8):
    slots = {}
    for slot in SLOT_NAMES:
        if slot_subset is None or slot in slot_subset:
            slots[slot] = np.random.rand(n_groups, group_size, h, w).astype(np.float32)
    labels = np.random.rand(12).astype(np.float32)
    weights = np.ones(12, dtype=np.float32)
    return slots, labels, weights


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        collate_batch([])


def test_collate_full_batch_shape_and_mask():
    samples = [_sample(f"s{i}") for i in range(4)]
    slots, labels, weights, mask = collate_batch(samples, image_size=8)

    assert slots.shape == (4, 5, 10, 3, 8, 8)
    assert labels.shape == (4, 12)
    assert weights.shape == (4, 12)
    assert mask.shape == (4, 5)
    assert mask.sum() == 4 * 5


def test_collate_missing_slot_zeroed_and_masked():
    s0 = _sample("s0", slot_subset={"Sagittal_Fluid", "Coronal_Fluid"})
    s1 = _sample("s1", slot_subset={"Sagittal_Fluid"})
    slots, _, _, mask = collate_batch([s0, s1], image_size=8)

    assert mask[0, 0] == 1.0
    assert mask[0, 1] == 0.0
    assert mask[0, 2] == 1.0
    assert mask[0, 3] == 0.0
    assert mask[0, 4] == 0.0

    assert mask[1, 0] == 1.0
    assert mask[1].sum() == 1.0

    assert (slots[0, 1] == 0).all()
    assert (slots[1, 2] == 0).all()


def test_collate_raises_on_shape_mismatch():
    s_good = _sample("s0", slot_subset={"Sagittal_Fluid"})
    s_bad = _sample("s0", slot_subset={"Sagittal_Fluid"}, n_groups=11)
    with pytest.raises(ValueError):
        collate_batch([s_good, s_bad], image_size=8)


def test_collate_handles_unknown_slot_gracefully():
    s = _sample("s0", slot_subset={"Sagittal_Fluid", "MadeUpSlot"})
    slots, _, _, mask = collate_batch([s], image_size=8)
    assert mask[0, 0] == 1.0
    assert mask[0].sum() == 1.0
