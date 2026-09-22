"""Tests for src.classical.dataset — runs without torch/GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.dataset import KneeMRIDataset, _make_groups, build_default_volume_index
from src.imaging.slot_builder import SLOT_NAMES


def _fake_volume(n_slices: int = 40, h: int = 16, w: int = 16) -> np.ndarray:
    return np.random.rand(n_slices, h, w).astype(np.float32)


def _fake_labels_df(study_uids, with_conf=True):
    rows = []
    for uid in study_uids:
        row = {"StudyInstanceUID": uid}
        for i, label in enumerate([
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Baker's", "Contusion", "Fracture",
        ]):
            row[label] = float(i % 2)
            if with_conf:
                row[f"{label}_conf"] = 0.9
        rows.append(row)
    return pd.DataFrame(rows)


def _full_volume_index(study_uids):
    return {
        uid: {slot: _fake_volume(40) for slot in SLOT_NAMES}
        for uid in study_uids
    }


def _missing_axial_volume_index(study_uids):
    """All studies missing Axial_Fluid."""
    return {
        uid: {slot: _fake_volume(40) for slot in SLOT_NAMES if slot != "Axial_Fluid"}
        for uid in study_uids
    }


def test_dataset_rejects_empty_study_list():
    labels_df = _fake_labels_df(["dummy"])
    with pytest.raises(ValueError):
        KneeMRIDataset([], labels_df, volume_index={})


def test_dataset_rejects_zero_groups():
    labels_df = _fake_labels_df(["dummy"])
    with pytest.raises(ValueError):
        KneeMRIDataset(["dummy"], labels_df, volume_index={}, n_groups_per_slot=0)


def test_dataset_rejects_zero_group_size():
    labels_df = _fake_labels_df(["dummy"])
    with pytest.raises(ValueError):
        KneeMRIDataset(["dummy"], labels_df, volume_index={}, group_size=0)


def test_dataset_full_volume_index_emits_all_slots():
    uids = ["s1", "s2", "s3"]
    labels_df = _fake_labels_df(uids)
    ds = KneeMRIDataset(uids, labels_df, volume_index=_full_volume_index(uids))

    assert len(ds) == 3
    slots, labels, weights = ds[0]
    assert set(slots.keys()) == set(SLOT_NAMES)
    assert labels.shape == (12,)
    assert weights.shape == (12,)
    assert labels.dtype == np.float32
    assert weights.dtype == np.float32
    for slot_name, vol in slots.items():
        assert vol.shape == (10, 3, 16, 16), f"{slot_name}: {vol.shape}"


def test_dataset_missing_slot_is_absent_from_dict():
    uids = ["s1"]
    labels_df = _fake_labels_df(uids)
    ds = KneeMRIDataset(
        uids, labels_df, volume_index=_missing_axial_volume_index(uids)
    )
    slots, _, _ = ds[0]
    assert "Axial_Fluid" not in slots
    assert "Sagittal_Fluid" in slots


def test_dataset_unknown_study_uid_returns_zero_labels():
    uids = ["known"]
    labels_df = _fake_labels_df(uids)
    ds = KneeMRIDataset(["GHOST_NOT_IN_LABELS"], labels_df, volume_index={"GHOST_NOT_IN_LABELS": {}})
    _, labels, weights = ds[0]
    assert (labels == 0.0).all()
    assert (weights == 0.0).all()


def test_dataset_nan_label_becomes_zero_weight():
    rows = [{
        "StudyInstanceUID": "s1",
        "ACL": np.nan, "ACL_conf": 0.3,
        "MCL": 1.0, "MCL_conf": 0.9,
    }]
    for label in [
        "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
        "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]:
        rows[0][label] = 0.0
        rows[0][f"{label}_conf"] = 0.5
    labels_df = pd.DataFrame(rows)

    ds = KneeMRIDataset(["s1"], labels_df, volume_index={})
    _, labels, weights = ds[0]
    assert labels[0] == 0.5
    assert weights[0] == 0.0
    assert weights[1] == 0.9


def test_dataset_falls_back_to_formula_weight_when_conf_missing():
    rows = [{"StudyInstanceUID": "s1"}]
    for label in [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]:
        rows[0][label] = 1.0
    labels_df = pd.DataFrame(rows)

    ds = KneeMRIDataset(["s1"], labels_df, volume_index={})
    _, _, weights = ds[0]
    assert (weights > 0.99).all()
    assert (weights <= 1.5).all()


def test_make_groups_pads_short_volumes():
    short = _fake_volume(n_slices=2, h=4, w=4)
    groups = _make_groups(short, n_groups=3, group_size=3)
    assert groups.shape == (3, 3, 4, 4)


def test_make_groups_long_volume_central_band():
    long_vol = _fake_volume(n_slices=100, h=8, w=8)
    groups = _make_groups(long_vol, n_groups=10, group_size=3)
    assert groups.shape == (10, 3, 8, 8)


def test_build_default_volume_index_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_default_volume_index(tmp_path / "ghost")


def test_build_default_volume_index_roundtrip(tmp_path):
    study_dir = tmp_path / "study1"
    study_dir.mkdir()
    for slot in SLOT_NAMES[:2]:
        np.save(study_dir / f"{slot}.npy", _fake_volume(20))

    idx = build_default_volume_index(tmp_path)
    assert "study1" in idx
    assert set(idx["study1"].keys()) == {"Sagittal_Fluid", "Sagittal_Structural"}


def test_build_default_volume_index_reads_uint8_npz(tmp_path):
    from src.imaging.preprocessing import to_uint8

    study_dir = tmp_path / "study2"
    study_dir.mkdir()
    originals = {}
    for slot in SLOT_NAMES[:2]:
        originals[slot] = _fake_volume(20)
        np.savez_compressed(
            study_dir / f"{slot}.npz", volume=to_uint8(originals[slot])
        )

    idx = build_default_volume_index(tmp_path)
    assert "study2" in idx
    for slot in SLOT_NAMES[:2]:
        loaded = idx["study2"][slot]
        assert loaded.dtype == np.float32
        max_err = float(np.abs(loaded - originals[slot]).max())
        assert max_err <= 0.5 / 255 + 1e-6


def test_build_default_volume_index_npz_overrides_npy(tmp_path):
    from src.imaging.preprocessing import to_uint8

    study_dir = tmp_path / "study3"
    study_dir.mkdir()
    old = _fake_volume(20)
    new = _fake_volume(20)
    np.save(study_dir / "Sagittal_Fluid.npy", old)
    np.savez_compressed(study_dir / "Sagittal_Fluid.npz", volume=to_uint8(new))

    idx = build_default_volume_index(tmp_path)
    max_err = float(np.abs(idx["study3"]["Sagittal_Fluid"] - new).max())
    assert max_err <= 0.5 / 255 + 1e-6
