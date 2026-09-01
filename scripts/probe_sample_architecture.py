#!/usr/bin/env python3
"""Validate the real sample volumes and, optionally, the actual DINOv2 model.

This is a *contract and trainability* probe, not a performance evaluation:
three studies are far too few to yield a meaningful AUC.  It answers the
questions that unit tests and synthetic smoke tests cannot:

* Are the cached real MRI volumes readable, normalized, and slot-compatible?
* Do labels cover every sampled study with sensible confidence weights?
* Can the production DINOv2 + attention head run forward and backward on
  real 2.5D MRI groups, and does an optimizer update its head?

Run the data contract anywhere:

    python scripts/probe_sample_architecture.py

Run the model contract on a GPU after installing ``.[classical]`` and making
the DINOv2 weights available (internet or a local HuggingFace model path):

    python scripts/probe_sample_architecture.py --with-model --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSICAL_MAIN_BACKBONE, LABEL_COLUMNS  # noqa: E402
from src.imaging.slot_builder import SLOT_NAMES  # noqa: E402

DEFAULT_VOLUMES = PROJECT_ROOT / "data/processed/volumes_sample3"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data/processed/splits/sample3"
DEFAULT_LABELS = (
    PROJECT_ROOT
    / "data/external/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes-dir", type=Path, default=DEFAULT_VOLUMES)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--model-name", default=CLASSICAL_MAIN_BACKBONE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_study_uids(split_dir: Path) -> list[str]:
    frames = []
    for name in ("train.csv", "val.csv"):
        path = split_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Sample split is missing {path}")
        frames.append(pd.read_csv(path, usecols=["StudyInstanceUID"]))
    return pd.concat(frames, ignore_index=True)["StudyInstanceUID"].tolist()


def _labels_with_confidence(path: Path, study_uids: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = pd.read_csv(path)
    required = ["StudyInstanceUID", *LABEL_COLUMNS]
    missing = set(required) - set(labels.columns)
    if missing:
        raise ValueError(f"Labels file is missing columns: {sorted(missing)}")
    labels = labels.loc[labels["StudyInstanceUID"].isin(study_uids), required].copy()
    missing_uids = set(study_uids) - set(labels["StudyInstanceUID"])
    if missing_uids:
        raise ValueError(f"Labels do not cover {len(missing_uids)} sampled studies")
    values = labels[LABEL_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Sample labels must be finite probabilities in [0, 1]")
    for label in LABEL_COLUMNS:
        labels[f"{label}_conf"] = np.abs(labels[label] - 0.5) * 2.0 + 0.5
    return labels


def _load_volume_index(
    volumes_dir: Path, study_uids: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    index: dict[str, dict[str, np.ndarray]] = {}
    for uid in study_uids:
        study_dir = volumes_dir / uid
        if not study_dir.is_dir():
            raise FileNotFoundError(f"Missing cached volume directory: {study_dir}")
        slots: dict[str, np.ndarray] = {}
        for slot in SLOT_NAMES:
            path = study_dir / f"{slot}.npy"
            if path.exists():
                volume = np.load(path)
                if volume.ndim != 3 or volume.dtype != np.float32:
                    raise ValueError(
                        f"{path} must be a float32 (N, H, W) volume, "
                        f"got {volume.shape}/{volume.dtype}"
                    )
                if not np.isfinite(volume).all() or volume.min() < 0 or volume.max() > 1:
                    raise ValueError(f"{path} is not finite and normalized to [0, 1]")
                slots[slot] = volume
        if not slots:
            raise ValueError(f"No recognized slots found for {uid}")
        index[uid] = slots
    return index


def _data_contract(
    study_uids: list[str],
    labels: pd.DataFrame,
    volume_index: dict[str, dict[str, np.ndarray]],
):
    # Importing the classical package is intentionally delayed: this portion should
    # work on CPU-only machines that do not install the model extra.
    from src.classical.collate import collate_batch
    from src.classical.dataset import KneeMRIDataset

    dataset = KneeMRIDataset(study_uids, labels, volume_index=volume_index)
    batch = collate_batch([dataset[i] for i in range(len(dataset))])
    slots, targets, weights, mask = batch
    if slots.shape[:3] != (len(study_uids), len(SLOT_NAMES), dataset.n_groups_per_slot):
        raise AssertionError(f"Unexpected collated slot shape: {slots.shape}")
    if (
        not np.isfinite(slots).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(weights).all()
    ):
        raise AssertionError("Collated batch contains non-finite values")
    if not (mask.sum(axis=1) > 0).all():
        raise AssertionError("Every sample must have at least one populated slot")
    return batch, {
        "batch_shape": list(slots.shape),
        "slot_fill_counts": {slot: int(mask[:, i].sum()) for i, slot in enumerate(SLOT_NAMES)},
        "label_weight_sum": float(weights.sum()),
    }


def _model_contract(batch, model_name: str, device_name: str | None) -> dict:
    try:
        import torch

        from src.classical.dinov2_base import KneeModel
    except ImportError as exc:
        raise RuntimeError("Model probe requires `pip install -e '.[classical]'`.") from exc

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    slots, targets, weights, mask = (torch.as_tensor(value, device=device) for value in batch)
    model = KneeModel(backbone_name=model_name, freeze_backbone=True).to(device)
    model.train()

    # One real 2.5D group from every present slot. This keeps the probe bounded
    # while still validating all five acquisition paths through the shared model.
    per_slot_logits = []
    for slot_index in range(len(SLOT_NAMES)):
        logits = model.encode_slot(slots[:, slot_index, 0])
        per_slot_logits.append(logits * mask[:, slot_index : slot_index + 1])
    logits = torch.stack(per_slot_logits).sum(dim=0) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise AssertionError("Model produced non-finite logits or loss")

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    before = next(model.head.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=float("inf"))
    )
    if not np.isfinite(grad_norm) or grad_norm == 0:
        raise AssertionError("Attention head received no finite, non-zero gradient")
    optimizer.step()
    changed = not torch.equal(before, next(model.head.parameters()).detach())
    if not changed:
        raise AssertionError("Optimizer step did not change an attention-head parameter")
    return {
        "device": str(device),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "head_gradient_norm": grad_norm,
    }


def main() -> None:
    args = _parse_args()
    study_uids = _load_study_uids(args.split_dir)
    labels = _labels_with_confidence(args.labels_csv, study_uids)
    volumes = _load_volume_index(args.volumes_dir, study_uids)
    batch, report = _data_contract(study_uids, labels, volumes)
    report.update(
        {
            "study_count": len(study_uids),
            "studies": study_uids,
            "model_checked": args.with_model,
        }
    )
    if args.with_model:
        report["model"] = _model_contract(batch, args.model_name, args.device)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
