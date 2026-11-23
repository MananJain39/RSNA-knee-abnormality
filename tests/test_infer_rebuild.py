"""Tests for rebuilt checkpoint inference and submission writing."""

import importlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("cv2")
infer = importlib.import_module("scripts.infer")


def test_discover_checkpoints_sorts_and_deduplicates(tmp_path):
    for name in ["b.pt", "a.pt"]:
        (tmp_path / name).touch()

    checkpoints = infer.discover_checkpoints([str(tmp_path / "*.pt"), str(tmp_path / "a.pt")])

    assert checkpoints == [str(tmp_path / "a.pt"), str(tmp_path / "b.pt")]


def test_discover_checkpoints_rejects_empty_matches(tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        infer.discover_checkpoints([str(tmp_path / "*.pt")])


def test_write_submission_preserves_requested_uid_order(tmp_path):
    output = tmp_path / "submission.csv"
    predictions = {
        "study-1": np.linspace(0, 1, len(infer.LABELS)),
        "study-2": np.full(len(infer.LABELS), 0.5),
    }

    infer.write_submission(["study-2", "study-1", "study-2"], predictions, output)

    frame = pd.read_csv(output)
    assert frame["StudyInstanceUID"].tolist() == ["study-2", "study-1", "study-2"]
    assert list(frame.columns) == ["StudyInstanceUID", *infer.LABELS]
    assert frame.iloc[0][infer.LABELS].eq(0.5).all()
    assert frame.iloc[1]["ACL"] == 0.0
    assert frame.iloc[1]["Fracture"] == 1.0


def test_prep_returns_failure_payload_on_series_error(monkeypatch):
    rows = [
        {
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": 1,
            "SeriesInstanceUID": "series-1",
        }
    ]

    def fail(*_):
        raise RuntimeError("unreadable series")

    monkeypatch.setattr(infer.P, "list_series", fail)

    assert infer.prep(("study-1", rows, "root")) == ("study-1", None, None)


def test_prep_without_series_returns_empty_volume_and_mask(monkeypatch):
    monkeypatch.setattr(infer.P, "DEPTH", 2)
    monkeypatch.setattr(infer.P, "SIZE", 4)

    study, volume, mask = infer.prep(("study-1", [], "root"))

    assert study == "study-1"
    assert volume.shape == (len(infer.SLOTS), 2, 4, 4)
    assert not volume.any()
    assert not mask.any()


def test_prep_applies_left_sagittal_slice_reversal(monkeypatch):
    rows = [
        {
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": 1,
            "SeriesInstanceUID": "series-1",
        }
    ]
    volume = np.stack([np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)])
    monkeypatch.setattr(infer.P, "DEPTH", 2)
    monkeypatch.setattr(infer.P, "SIZE", 2)
    monkeypatch.setattr(infer.P, "list_series", lambda *_: ["a.dcm", "b.dcm", "c.dcm"])
    monkeypatch.setattr(infer.P, "read_series", lambda _: object())
    monkeypatch.setattr(infer.P, "series_volume", lambda *_: (volume, {}))
    monkeypatch.setattr(infer.P, "laterality", lambda _: ("L", "tag"))
    monkeypatch.setattr(infer.P, "to_uint8", lambda value: value)
    monkeypatch.setattr(infer.P, "resample_depth", lambda value: value)

    _, packed, mask = infer.prep(("study-1", rows, "root"))

    slot = infer.SLOTS.index("sag_fs")
    assert mask[slot]
    assert np.array_equal(packed[slot, 0], np.ones((2, 2)))
    assert np.array_equal(packed[slot, 1], np.zeros((2, 2)))
    assert mask.sum() == 1
