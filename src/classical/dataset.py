"""PyTorch-ready Dataset for knee MRI studies.

Each sample is one StudyInstanceUID. The dataset emits:

  Sample = (slots, labels, weights)
    slots:   dict[slot_name, np.ndarray[N, 3, H, W]]
             N = n_groups_per_slot (default 10), 3 = adjacent slices per group
    labels:  np.ndarray[12]   per-finding targets, NaN -> 0.5 sentinel
    weights: np.ndarray[12]   per-finding confidence weights

The dataset is torch-optional at import time and at __getitem__ time. Tests
construct an in-memory volume index; production loads from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.config import (
    CLASSICAL_IMAGE_SIZE,
    CLASSICAL_N_GROUPS_PER_SLOT,
    LABEL_COLUMNS,
    PROCESSED_DATA_DIR,
)
from src.imaging.slot_builder import SLOT_NAMES


# === CONTRACT (top of file for both agents to read) ===
class KneeMRIDatasetProtocol(Protocol):
    """One StudyInstanceUID → one training sample.

    Returns:
        Sample = (slots, labels, weights)
            slots:   dict[slot_name, np.ndarray[N, 3, H, W]]
            labels:  np.ndarray[12]
            weights: np.ndarray[12]
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> "Sample": ...


# Type alias so the trainer file can import it.
Sample = tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]


# === IMPLEMENTATION ===

class KneeMRIDataset:
    """Reads preprocessed volumes and emits (slots, labels, weights)."""

    def __init__(
        self,
        study_uids: list[str],
        labels_df: pd.DataFrame,
        volume_index: dict[str, dict[str, np.ndarray]] | None = None,
        n_groups_per_slot: int = CLASSICAL_N_GROUPS_PER_SLOT,
        image_size: int = CLASSICAL_IMAGE_SIZE,
        group_size: int = 3,
        augment: bool = False,
        augmentation_config: dict | None = None,
    ):
        if not study_uids:
            raise ValueError("study_uids must be non-empty")
        if n_groups_per_slot < 1:
            raise ValueError(f"n_groups_per_slot must be >= 1, got {n_groups_per_slot}")
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")

        self.study_uids = list(study_uids)
        if "StudyInstanceUID" in labels_df.columns:
            self.labels_df = labels_df.set_index("StudyInstanceUID")
        else:
            self.labels_df = labels_df
        self.volume_index = volume_index or {}
        self.n_groups_per_slot = n_groups_per_slot
        self.image_size = image_size
        self.group_size = group_size
        self.augment = augment
        self.augmentation_config = augmentation_config or {
            "rotation_deg": 10.0,
            "gamma_range": (0.8, 1.2),
            "rescale_range": (0.95, 1.05),
        }
        self._rng = np.random.default_rng()

        self._label_cols = list(LABEL_COLUMNS)
        self._weight_cols = [f"{c}_conf" for c in self._label_cols]

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int) -> Sample:
        study_uid = self.study_uids[idx]
        slots = self._load_slots(study_uid)
        
        # Apply augmentation if enabled (training mode)
        if self.augment and slots:
            from src.classical.augmentation import augment_slots
            slots = augment_slots(
                slots,
                rotation_deg=self.augmentation_config.get("rotation_deg", 10.0),
                gamma_range=self.augmentation_config.get("gamma_range", (0.8, 1.2)),
                rescale_range=self.augmentation_config.get("rescale_range", (0.95, 1.05)),
                rng=self._rng,
            )
        
        labels, weights = self._load_labels_and_weights(study_uid)
        return slots, labels, weights

    def _load_slots(self, study_uid: str) -> dict[str, np.ndarray]:
        """Load all available slots for this study and group into 3-slice stacks.

        Returns dict keyed by slot_name. Each value is shape
        (n_groups_per_slot, group_size, H, W) float32 in [0, 1].
        Missing slots are absent from the dict (the collate handles that).
        """
        study_volumes = self.volume_index.get(study_uid, {})
        out: dict[str, np.ndarray] = {}
        for slot_name in SLOT_NAMES:
            vol = study_volumes.get(slot_name)
            if vol is None:
                continue
            vol = np.asarray(vol, dtype=np.float32)
            if vol.ndim != 3:
                raise ValueError(
                    f"Volume for {study_uid}/{slot_name} must be 3D (N,H,W), got {vol.shape}"
                )
            groups = _make_groups(vol, self.n_groups_per_slot, self.group_size)
            out[slot_name] = groups
        return out

    def _load_labels_and_weights(self, study_uid: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels[12], weights[12]) for this study.

        NaN labels are filled with 0.5 (the 'not addressed' sentinel from
        process_labels.py). The corresponding weight is zeroed so silent
        findings don't dominate the loss. If the study UID is not in the
        labels DataFrame, returns zeros — the loss reduction will then
        exclude every cell from contributing.
        """
        labels = np.zeros(len(self._label_cols), dtype=np.float32)
        weights = np.zeros(len(self._label_cols), dtype=np.float32)

        if study_uid not in self.labels_df.index:
            return labels, weights

        if isinstance(self.labels_df.index, pd.MultiIndex):
            row_series = self.labels_df.loc[study_uid]
            if isinstance(row_series, pd.DataFrame):
                row = row_series.iloc[0]
            else:
                row = row_series
        else:
            row = self.labels_df.loc[study_uid]

        for i, col in enumerate(self._label_cols):
            label_val = row.get(col, np.nan) if hasattr(row, "get") else row[col]
            if pd.isna(label_val):
                labels[i] = 0.5
                weights[i] = 0.0
            else:
                labels[i] = float(label_val)
                if col + "_conf" in (row.index if hasattr(row, "index") else []):
                    weights[i] = float(row[col + "_conf"])
                else:
                    weights[i] = abs(float(label_val) - 0.5) * 2 + 0.5
        return labels, weights


# === HELPERS ===

def _make_groups(volume: np.ndarray, n_groups: int, group_size: int) -> np.ndarray:
    """Sample ``n_groups`` adjacent ``group_size`` slices from the central band.

    Output shape: (n_groups, group_size, H, W).

    Uses the same central-band heuristic as
    ``src.imaging.preprocessing.sample_central_band`` so behavior matches
    the existing DICOM preprocessing pipeline.

    Volumes shorter than ``group_size`` are zero-padded to exactly
    ``group_size`` so every group is well-formed. Start indices are
    clamped so no group straddles the volume boundary.
    """
    n_slices = volume.shape[0]
    if n_slices < group_size:
        pad = np.zeros(
            (group_size - n_slices,) + volume.shape[1:], dtype=volume.dtype
        )
        volume = np.concatenate([volume, pad], axis=0)
        n_slices = volume.shape[0]

    low = int(n_slices * 0.12)
    high = int(n_slices * 0.88)
    high = max(high, low + n_groups)
    max_start = n_slices - group_size
    start_indices = np.clip(
        np.linspace(low, high - 1, n_groups, dtype=int),
        0,
        max_start,
    )

    groups = np.stack(
        [volume[i : i + group_size] for i in start_indices], axis=0
    ).astype(np.float32)
    return groups


def build_default_volume_index(studies_dir: Path | None = None) -> dict:
    """Load a disk-backed volume index from ``data/processed/volumes/``.

    Expected on disk:
        studies_dir/<StudyInstanceUID>/<slot_name>.npz
            uint8-quantized compressed volume (new default; ``volume`` key)
        studies_dir/<StudyInstanceUID>/<slot_name>.npy
            legacy float32 volume (backward compatible)

    uint8 files are restored to float32 ``[0, 1]`` on load, so downstream
    code cannot tell the storage format apart.
    """
    from src.imaging.preprocessing import from_uint8

    if studies_dir is None:
        studies_dir = PROCESSED_DATA_DIR / "volumes"
    studies_dir = Path(studies_dir)
    if not studies_dir.exists():
        raise FileNotFoundError(f"Volume directory not found: {studies_dir}")

    out: dict = {}
    for study_dir in studies_dir.iterdir():
        if not study_dir.is_dir():
            continue
        per_study: dict = {}
        for npy_path in study_dir.glob("*.npy"):
            per_study[npy_path.stem] = np.load(npy_path)
        for npz_path in study_dir.glob("*.npz"):
            per_study[npz_path.stem] = from_uint8(
                np.load(npz_path)["volume"]
            )
        if per_study:
            out[study_dir.name] = per_study
    return out


__all__ = [
    "KneeMRIDatasetProtocol",
    "Sample",
    "KneeMRIDataset",
    "build_default_volume_index",
]
