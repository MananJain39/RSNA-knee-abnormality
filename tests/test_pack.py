"""Tests for packing study NPZ files into the training memmap format."""

import importlib

import numpy as np
import pandas as pd
import pytest

pack = importlib.import_module("scripts.pack")


def _labels(count=12):
    uids = [f"gold-{index}" for index in range(count)]
    uids += [f"plain-{index}" for index in range(count)]
    return pd.DataFrame(
        {
            "StudyInstanceUID": uids,
            "is_gold": [1] * count + [0] * count,
        }
    )


def _small_pack(monkeypatch):
    monkeypatch.setattr(pack, "D", 2)
    monkeypatch.setattr(pack, "H", 4)


def test_fake_uid_selection_honors_requested_count():
    labels = _labels()

    assert pack.select_fake_uids(labels, 0) == []
    assert pack.select_fake_uids(labels, 1) == ["gold-0"]
    assert pack.select_fake_uids(labels, 12) == [
        *[f"gold-{index}" for index in range(10)],
        "plain-0",
        "plain-1",
    ]
    with pytest.raises(ValueError, match="non-negative"):
        pack.select_fake_uids(labels, -1)


def test_fake_pack_writes_requested_shape_masks_and_index(tmp_path, monkeypatch):
    _small_pack(monkeypatch)
    labels_path = tmp_path / "labels.csv"
    _labels().to_csv(labels_path, index=False)
    out = tmp_path / "fake-pack"
    monkeypatch.setattr(
        "sys.argv",
        ["pack.py", "--fake", "3", "--labels", str(labels_path), "--out", str(out)],
    )

    pack.main()

    volumes = np.load(out / "vols.npy", mmap_mode="r")
    index = pd.read_csv(out / "index.csv")
    assert volumes.shape == (3, len(pack.SLOTS), 2, 4, 4)
    assert volumes.dtype == np.uint8
    assert index["StudyInstanceUID"].tolist() == ["gold-0", "gold-1", "gold-2"]
    assert index["row"].tolist() == [0, 1, 2]
    assert index["mask_cor_t1"].tolist() == [0, 1, 1]
    for name in ["sag_fs", "sag_t1", "cor_fs", "ax_fs"]:
        assert index[f"mask_{name}"].eq(1).all()
    assert np.all(volumes[0, pack.SLOTS.index("cor_t1")] == 0)


def test_real_pack_sorts_studies_and_zero_fills_missing_slots(tmp_path, monkeypatch):
    _small_pack(monkeypatch)
    volume_dir = tmp_path / "volumes"
    volume_dir.mkdir()
    np.savez_compressed(
        volume_dir / "study-b.npz",
        cor_fs=np.full((2, 4, 4), 3, dtype=np.uint8),
    )
    np.savez_compressed(
        volume_dir / "study-a.npz",
        sag_fs=np.ones((2, 4, 4), dtype=np.uint8),
        ax_fs=np.full((2, 4, 4), 2, dtype=np.uint8),
    )
    out = tmp_path / "real-pack"
    monkeypatch.setattr(
        "sys.argv",
        ["pack.py", "--vol-dir", str(volume_dir), "--out", str(out)],
    )

    pack.main()

    volumes = np.load(out / "vols.npy", mmap_mode="r")
    index = pd.read_csv(out / "index.csv").set_index("StudyInstanceUID")
    assert list(index.index) == ["study-a", "study-b"]
    assert index.loc["study-a", "mask_sag_fs"] == 1
    assert index.loc["study-a", "mask_ax_fs"] == 1
    assert index.loc["study-a", "mask_cor_fs"] == 0
    assert np.all(volumes[0, pack.SLOTS.index("sag_fs")] == 1)
    assert np.all(volumes[0, pack.SLOTS.index("ax_fs")] == 2)
    assert np.all(volumes[0, pack.SLOTS.index("cor_fs")] == 0)
    assert np.all(volumes[1, pack.SLOTS.index("cor_fs")] == 3)
