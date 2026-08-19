"""End-to-end smoke test for the classical training/inference pipeline.

Exercises the full chain on synthetic data:

  dataset build → collate → train main branch → train 4 cluster specialists →
  predictor (main + specialists) → submission writer → schema validation.

Uses tiny ``nn.Linear``-based model stubs in place of DINOv2 so the test
runs on CPU without any HuggingFace download. The real DINOv2 modules
load identically — only the inner network is swapped.

This is the integration test that catches "the modules exist but don't
actually wire together". Without it, Kaggle runs would fail at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.classical import (  # noqa: E402
    ClassicalPredictor,
    ClassicalTrainer,
    KneeMRIDataset,
    SubmissionWriter,
)
from src.classical.collate import collate_batch  # noqa: E402
from src.config import LABEL_COLUMNS  # noqa: E402
from src.imaging.slot_builder import SLOT_NAMES  # noqa: E402

# === Synthetic data builders ===

class _TinyModel(nn.Module):
    """Stand-in for KneeModel. Output shape matches: (B, 12)."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3 * 8 * 8, len(LABEL_COLUMNS))

    def encode_slot(self, images):
        return self.layer(images.flatten(1))

    def parameter_groups(self, lr, weight_decay):
        return [{"params": self.parameters(), "lr": lr, "weight_decay": weight_decay}]


def _volume(n_slices: int = 12) -> np.ndarray:
    return np.random.rand(n_slices, 8, 8).astype(np.float32)


def _labels_df(study_uids, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for uid in study_uids:
        row = {"StudyInstanceUID": uid}
        for col in LABEL_COLUMNS:
            row[col] = float(rng.integers(0, 2))
            row[f"{col}_conf"] = 0.9
        rows.append(row)
    return pd.DataFrame(rows)


def _volume_index(study_uids) -> dict[str, dict[str, np.ndarray]]:
    return {
        uid: {slot: _volume(12) for slot in SLOT_NAMES[:3]}
        for uid in study_uids
    }


def _collate(samples):
    return collate_batch(samples, image_size=8)


def _write_checkpoint(path: Path) -> None:
    """Write a checkpoint compatible with ClassicalTrainer / ClassicalPredictor."""
    torch.save({
        "model_state_dict": _TinyModel().state_dict(),
        "ema_state_dict": None,
        "epoch": 0,
        "best_val_macro_auc": 0.5,
    }, path)


# === The smoke test ===

def test_pipeline_end_to_end(tmp_path):
    """Train main + 4 specialists, predict, write submission, validate schema."""
    study_uids = [f"study_{i}" for i in range(8)]
    cluster_assignments = pd.DataFrame({
        "StudyInstanceUID": study_uids,
        "cluster_id": [0, 0, 1, 1, 2, 2, 3, 3],
    })
    labels_df = _labels_df(study_uids)
    volume_index = _volume_index(study_uids)

    train_ds = KneeMRIDataset(study_uids, labels_df, volume_index=volume_index)
    val_ds = KneeMRIDataset(study_uids, labels_df, volume_index=volume_index)

    sample = collate_batch([train_ds[0]], image_size=8)
    assert sample[0].shape[1] == len(SLOT_NAMES)

    output_dir = tmp_path / "ckpts"
    trainer = ClassicalTrainer(
        model_factory=_TinyModel,
        specialist_factory=lambda cluster_id, image_size: _TinyModel(),
        batch_size=2,
        max_epochs=1,
        device="cpu",
        collate_fn=_collate,
        cluster_assignments=cluster_assignments,
    )

    main_result = trainer.train_main(train_ds, val_ds, output_dir / "main")
    assert main_result.checkpoint_path.exists()
    assert main_result.best_val_macro_auc >= 0

    specialist_paths: dict[int, Path] = {}
    for cluster_id in (0, 1, 2, 3):
        result = trainer.train_specialist(
            cluster_id, train_ds, val_ds, output_dir / f"cluster_{cluster_id}"
        )
        assert result.checkpoint_path.exists()
        specialist_paths[cluster_id] = result.checkpoint_path

    _write_checkpoint(main_result.checkpoint_path)

    predictor = ClassicalPredictor(
        device="cpu",
        model_factory=_TinyModel,
        specialist_factory=lambda cluster_id, image_size: _TinyModel(),
        dataset_factory=lambda uids, ldf, vi: KneeMRIDataset(uids, ldf, volume_index=vi),
        collate_fn=_collate,
    )

    predictions = predictor.predict_studies(
        model_paths={
            "main": main_result.checkpoint_path,
            "specialists": {cid: p for cid, p in specialist_paths.items()},
        },
        study_uids=study_uids,
        volume_index=volume_index,
        labels_df=labels_df,
        cluster_assignments=cluster_assignments,
    )

    assert list(predictions.columns) == ["StudyInstanceUID", *LABEL_COLUMNS]
    assert len(predictions) == len(study_uids)
    assert ((predictions[LABEL_COLUMNS] >= 0) & (predictions[LABEL_COLUMNS] <= 1)).all().all()

    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({
        "StudyInstanceUID": study_uids,
        **{label: [0.0] * len(study_uids) for label in LABEL_COLUMNS},
    }).to_csv(sample_path, index=False)

    submission_path = SubmissionWriter().write(
        predictions,
        tmp_path / "submission.csv",
        sample_submission_path=sample_path,
    )
    assert submission_path.exists()

    written = pd.read_csv(submission_path)
    assert list(written.columns) == ["StudyInstanceUID", *LABEL_COLUMNS]
    assert written["StudyInstanceUID"].tolist() == study_uids
