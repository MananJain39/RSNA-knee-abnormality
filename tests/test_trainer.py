"""Torch-gated tests for classical training orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.classical.trainer import ClassicalTrainer, TrainResult  # noqa: E402
from src.config import LABEL_COLUMNS  # noqa: E402
from src.imaging.slot_builder import SLOT_NAMES  # noqa: E402


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3 * 4 * 4, len(LABEL_COLUMNS))

    def encode_slot(self, images):
        return self.layer(images.flatten(1))

    def parameter_groups(self, lr, weight_decay):
        return [{"params": self.parameters(), "lr": lr, "weight_decay": weight_decay}]


class _Dataset:
    def __init__(self, uids=("s1", "s2")):
        self.study_uids = list(uids)
        columns = {name: [0.0, 1.0][: len(uids)] for name in LABEL_COLUMNS}
        self.labels_df = pd.DataFrame(
            {"StudyInstanceUID": self.study_uids, **columns}
        ).set_index("StudyInstanceUID")

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, index):
        slots = {SLOT_NAMES[0]: np.ones((1, 3, 4, 4), dtype=np.float32)}
        labels = self.labels_df.loc[self.study_uids[index], LABEL_COLUMNS].to_numpy(np.float32)
        return slots, labels, np.ones(len(LABEL_COLUMNS), dtype=np.float32)


def _collate(samples):
    batch = len(samples)
    slots = np.zeros((batch, len(SLOT_NAMES), 1, 3, 4, 4), dtype=np.float32)
    mask = np.zeros((batch, len(SLOT_NAMES)), dtype=np.float32)
    for index, (sample_slots, _, _) in enumerate(samples):
        slots[index, 0] = sample_slots[SLOT_NAMES[0]]
        mask[index, 0] = 1.0
    return slots, np.stack([s[1] for s in samples]), np.stack([s[2] for s in samples]), mask


def _trainer(**kwargs):
    return ClassicalTrainer(
        model_factory=_TinyModel,
        specialist_factory=lambda cluster_id, image_size: _TinyModel(),
        batch_size=2,
        max_epochs=1,
        device="cpu",
        collate_fn=_collate,
        **kwargs,
    )


def test_train_result_dataclass_shape():
    result = TrainResult(1, 0.5, Path("best.pt"), {"train_loss": 0.1})
    assert result.epoch == 1 and result.checkpoint_path.name == "best.pt"


def test_trainer_creates_output_dir_and_saves_checkpoint(tmp_path):
    result = _trainer().train_main(_Dataset(), _Dataset(), tmp_path / "nested")
    assert result.checkpoint_path.exists()
    assert result.checkpoint_path.parent == tmp_path / "nested"


def test_trainer_logs_to_experiment_tracker(tmp_path):
    tracker = Mock()
    result = _trainer(tracker_factory=lambda name: tracker).train_main(
        _Dataset(), _Dataset(), tmp_path
    )
    assert result.metrics_dict["val_macro_auc"] >= 0
    tracker.log_step.assert_called_once()
    tracker.save.assert_called_once()


def test_parameter_groups_passed_to_optimizer(tmp_path):
    with patch("src.classical.trainer.torch.optim.AdamW", wraps=torch.optim.AdamW) as adamw:
        _trainer().train_main(_Dataset(), _Dataset(), tmp_path)
    assert adamw.call_args.args[0][0]["lr"] > 0


def test_trainer_resumes_from_checkpoint_if_provided(tmp_path):
    checkpoint = tmp_path / "resume.pt"
    checkpoint.touch()
    trainer = _trainer(resume_from=checkpoint)
    model = _TinyModel()
    ema = type("EMA", (), {"model": _TinyModel()})()
    payload = {
        "model_state_dict": model.state_dict(),
        "ema_state_dict": None,
        "epoch": 2,
        "best_val_macro_auc": 0.7,
    }
    with patch("src.classical.trainer.torch.load", return_value=payload):
        start, best = trainer._resume(model, ema)
    assert (start, best) == (3, 0.7)


def test_train_specialist_filters_rows(tmp_path):
    assignments = pd.DataFrame({"StudyInstanceUID": ["s1", "s2"], "cluster_id": [1, 2]})
    result = _trainer(cluster_assignments=assignments).train_specialist(
        1, _Dataset(), _Dataset(), tmp_path
    )
    assert result.checkpoint_path.exists()
