"""CPU-only tests for the rebuilt 2.5D training pipeline."""

import importlib

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
train = importlib.import_module("scripts.train")


def _fold_input():
    index = [f"study-{index}" for index in range(9)]
    return pd.DataFrame(
        {
            "is_gold": [0] * 8 + [1],
            "Report": ["duplicate", "duplicate", "a", "b", "c", "d", "e", "f", "gold"],
        },
        index=index,
    )


def test_make_folds_excludes_gold_and_keeps_duplicate_reports_together():
    folds = train.make_folds(_fold_input(), k=3, seed=42)

    assert len(folds) == 8
    assert "study-8" not in folds.index
    assert folds.loc["study-0"] == folds.loc["study-1"]
    assert set(folds) == {0, 1, 2}


def test_make_folds_is_deterministic_for_same_seed():
    first = train.make_folds(_fold_input(), k=3, seed=42)
    second = train.make_folds(_fold_input(), k=3, seed=42)

    assert first.equals(second)


def test_macro_auc_computes_per_label_scores():
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, len(train.LABELS)))
    p = np.tile(np.array([0.1, 0.2, 0.8, 0.9])[:, None], (1, len(train.LABELS)))

    macro, per_label = train.macro_auc(y, p)

    assert macro == 1.0
    assert set(per_label) == set(train.LABELS)
    assert all(score == 1.0 for score in per_label.values())


def test_macro_auc_skips_single_class_labels():
    y = np.zeros((2, len(train.LABELS)))
    y[:, 0] = [0, 1]
    p = np.tile(np.array([0.1, 0.9])[:, None], (1, len(train.LABELS)))

    macro, per_label = train.macro_auc(y, p)

    assert macro == 1.0
    assert per_label == {"ACL": 1.0}


def test_gpu_batcher_eval_is_deterministic_and_preserves_masks():
    rng = np.random.default_rng(7)
    volumes = rng.integers(0, 255, size=(2, 2, 8, 8, 8), dtype=np.uint8)
    masks = np.array([[True, False], [True, True]])
    batcher = train.GpuBatcher(volumes, masks, n_slices=3, crop=4, device="cpu")

    x1, mask1, pos1 = batcher.get([0, 1], train=False)
    x2, mask2, pos2 = batcher.get([0, 1], train=False)

    assert x1.shape == (2, 2, 3, 3, 4, 4)
    assert torch.equal(x1, x2)
    assert torch.equal(mask1, mask2)
    assert torch.equal(pos1, pos2)
    assert torch.equal(mask1, torch.from_numpy(masks))
    assert torch.isfinite(x1).all()
    assert pos1.min() >= 0
    assert pos1.max() <= 31


def test_gpu_batcher_training_shapes_and_slot_dropout_keep_one_slot():
    rng = np.random.default_rng(8)
    volumes = rng.integers(0, 255, size=(2, 2, 8, 8, 8), dtype=np.uint8)
    masks = np.array([[True, False], [True, True]])
    batcher = train.GpuBatcher(volumes, masks, n_slices=3, crop=4, device="cpu")
    torch.manual_seed(0)

    x, mask, pos = batcher.get([0, 1], train=True)

    assert x.shape == (2, 2, 3, 3, 4, 4)
    assert pos.shape == (2, 2, 3)
    assert torch.isfinite(x).all()
    assert (mask.sum(dim=1) >= 1).all()


@pytest.mark.parametrize("head", ["pool", "attn"])
def test_kneenet_heads_return_finite_multilabel_logits(head):
    model = train.KneeNet(
        "resnet18",
        n_slots=2,
        max_d=8,
        dim=32,
        layers=1,
        pretrained=False,
        head=head,
    )
    x = torch.randn(1, 2, 3, 3, 16, 16)
    mask = torch.tensor([[True, False]])
    pos = torch.tensor([[[0, 1, 2], [0, 1, 2]]])

    with torch.no_grad():
        output = model(x, mask, pos)

    assert output.shape == (1, len(train.LABELS))
    assert torch.isfinite(output).all()
