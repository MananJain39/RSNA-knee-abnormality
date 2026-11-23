"""Tests for laterality mirroring and training batch construction."""

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
laterality = importlib.import_module("scripts.laterality")


def test_mirror_volume_uses_axis_specific_transforms_without_mutating_input():
    volume = np.arange(5 * 3 * 2 * 2, dtype=np.uint8).reshape(5, 3, 2, 2)
    original = volume.copy()

    mirrored = laterality.mirror_volume(volume)

    assert np.array_equal(volume, original)
    for slot in [0, 1]:
        assert np.array_equal(mirrored[slot], volume[slot, ::-1])
    for slot in [2, 3, 4]:
        assert np.array_equal(mirrored[slot], volume[slot, :, :, ::-1])
    assert mirrored.dtype == volume.dtype


def test_batch_builds_normalized_three_channel_images_and_flip_targets():
    rng = np.random.default_rng(4)
    volumes = rng.integers(0, 255, size=(1, 5, 32, 4, 4), dtype=np.uint8)
    masks = np.zeros((1, 5), dtype=bool)
    masks[0, 2] = True

    np.random.seed(3)
    canonical, canonical_target = laterality.batch(volumes, masks, [0], [False], "cpu")
    np.random.seed(3)
    flipped, flipped_target = laterality.batch(volumes, masks, [0], [True], "cpu")

    assert canonical.shape == (1, 3, 224, 224)
    assert flipped.shape == canonical.shape
    assert canonical_target.tolist() == [0.0]
    assert flipped_target.tolist() == [1.0]
    assert torch.allclose(flipped, canonical.flip(-1), atol=2e-6)
    assert torch.isfinite(canonical).all()
