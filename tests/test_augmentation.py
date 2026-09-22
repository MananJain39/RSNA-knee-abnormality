"""Tests for slot-group data augmentation."""

import numpy as np

from src.classical.augmentation import augment_slot_group, augment_slots


def _fake_slot(n_groups=4, group_size=3, h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n_groups, group_size, h, w)).astype(np.float32)


def test_augmentation_preserves_shape_and_range():
    slot = _fake_slot()
    out = augment_slot_group(slot, rng=np.random.default_rng(1))
    assert out.shape == slot.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_augmentation_changes_data():
    slot = _fake_slot()
    out = augment_slot_group(
        slot,
        rotation_deg=10.0,
        gamma_range=(0.8, 1.2),
        rescale_range=(0.95, 1.05),
        rng=np.random.default_rng(2),
    )
    assert not np.allclose(out, slot)


def test_augmentation_identity_transform_is_exact():
    slot = _fake_slot()
    out = augment_slot_group(
        slot,
        rotation_deg=0.0,
        gamma_range=(1.0, 1.0),
        rescale_range=(1.0, 1.0),
        rng=np.random.default_rng(3),
    )
    np.testing.assert_allclose(out, slot, atol=1e-6)


def test_augmentation_is_deterministic_with_seed():
    slot = _fake_slot()
    out_a = augment_slot_group(slot, rng=np.random.default_rng(42))
    out_b = augment_slot_group(slot, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(out_a, out_b)


def test_augmentation_applies_same_transform_within_slot():
    slot = np.tile(_fake_slot(n_groups=1), (2, 1, 1, 1))  # identical groups
    out = augment_slot_group(slot, rng=np.random.default_rng(5))
    np.testing.assert_allclose(out[0], out[1], atol=1e-6)


def test_augment_slots_covers_every_slot():
    slots = {
        "Sagittal_Fluid": _fake_slot(seed=1),
        "Axial_Fluid": _fake_slot(seed=2),
    }
    out = augment_slots(slots, rng=np.random.default_rng(7))
    assert set(out.keys()) == set(slots.keys())
    for name, arr in out.items():
        assert arr.shape == slots[name].shape
