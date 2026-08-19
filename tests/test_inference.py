"""Torch-gated tests for main-plus-specialist inference routing."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.classical.inference import ClassicalPredictor  # noqa: E402
from src.config import LABEL_COLUMNS  # noqa: E402
from src.imaging.slot_builder import SLOT_NAMES  # noqa: E402


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3 * 4 * 4, len(LABEL_COLUMNS))

    def encode_slot(self, images):
        return self.layer(images.flatten(1))


class _Dataset:
    def __init__(self, study_uids, labels_df, volume_index):
        self.study_uid = study_uids[0]

    def __getitem__(self, index):
        return {SLOT_NAMES[0]: np.ones((1, 3, 4, 4), dtype=np.float32)}, None, None


def _collate(samples):
    slots = np.zeros((1, len(SLOT_NAMES), 1, 3, 4, 4), dtype=np.float32)
    slots[0, 0] = samples[0][0][SLOT_NAMES[0]]
    return slots, np.zeros((1, 12)), np.zeros((1, 12)), np.array([[1, 0, 0, 0, 0]])


def _write_checkpoint(path):
    torch.save({"model_state_dict": _TinyModel().state_dict(), "ema_state_dict": None}, path)


def _predictor():
    return ClassicalPredictor(
        device="cpu",
        model_factory=_TinyModel,
        specialist_factory=lambda cluster_id, image_size: _TinyModel(),
        dataset_factory=_Dataset,
        collate_fn=_collate,
    )


def _assignments():
    return pd.DataFrame({"StudyInstanceUID": ["s1", "s2"], "cluster_id": [3, 3]})


def test_predictor_returns_dataframe_with_12_columns(tmp_path):
    main = tmp_path / "main.pt"
    _write_checkpoint(main)
    out = _predictor().predict_studies({"main": main}, ["s1"], {}, pd.DataFrame(), _assignments())
    assert list(out.columns) == ["StudyInstanceUID", *LABEL_COLUMNS]


def test_predictor_routes_via_specialist_router(tmp_path):
    main, specialist = tmp_path / "main.pt", tmp_path / "specialist.pt"
    _write_checkpoint(main)
    _write_checkpoint(specialist)
    with patch("src.classical.inference.specialist_router", return_value=3) as router:
        _predictor().predict_studies(
            {"main": main, "specialists": {3: specialist}},
            ["s1"],
            {},
            pd.DataFrame(),
            _assignments(),
        )
    assert router.call_count >= 1


def test_predictor_falls_back_to_main_when_no_specialist(tmp_path):
    main = tmp_path / "main.pt"
    _write_checkpoint(main)
    out = _predictor().predict_studies({"main": main}, ["s1"], {}, pd.DataFrame(), _assignments())
    assert len(out) == 1


def test_predictor_caches_loaded_models(tmp_path):
    main = tmp_path / "main.pt"
    _write_checkpoint(main)
    with patch("src.classical.inference.torch.load", wraps=torch.load) as load:
        _predictor().predict_studies(
            {"main": main}, ["s1", "s2"], {}, pd.DataFrame(), _assignments()
        )
    assert load.call_count == 1


def test_predictor_returns_values_in_0_1_range(tmp_path):
    main = tmp_path / "main.pt"
    _write_checkpoint(main)
    out = _predictor().predict_studies({"main": main}, ["s1"], {}, pd.DataFrame(), _assignments())
    assert ((out[LABEL_COLUMNS] >= 0) & (out[LABEL_COLUMNS] <= 1)).all().all()
