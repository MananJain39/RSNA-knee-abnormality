"""Collate function for knee-MRI batches with missing-slot tolerance.

Real studies don't always have all 5 slots (Axial_Fluid fill rate ~92.9%,
Axial_Structural dropped, etc.). The collate function emits a 5-slot tensor
plus a mask so the trainer/model can ignore absent slots cleanly.

Output shapes:
  slot_tensors : np.ndarray[B, 5, n_groups, group_size, H, W]
  labels       : np.ndarray[B, 12]
  weights      : np.ndarray[B, 12]
  slot_mask    : np.ndarray[B, 5]   (1 = present, 0 = absent)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.classical.dataset import Sample
from src.config import CLASSICAL_IMAGE_SIZE, CLASSICAL_N_GROUPS_PER_SLOT
from src.imaging.slot_builder import SLOT_NAMES


# === CONTRACT (top of file for both agents to read) ===
class CollateProtocol(Protocol):
    """Batch of samples → (slot_tensors, labels, weights, slot_mask)."""

    def __call__(self, samples: list[Sample]) -> "Batch": ...


Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# === IMPLEMENTATION ===

def collate_batch(
    samples: list[Sample],
    n_groups_per_slot: int = CLASSICAL_N_GROUPS_PER_SLOT,
    image_size: int = CLASSICAL_IMAGE_SIZE,
) -> Batch:
    """Collate a list of Samples into a batch with mask for missing slots.

    All samples must have the same (n_groups, group_size, H, W) per slot —
    this is guaranteed by ``KneeMRIDataset`` using shared hyperparams.
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    batch_size = len(samples)
    n_slots = len(SLOT_NAMES)
    slot_to_idx = {name: i for i, name in enumerate(SLOT_NAMES)}

    group_size = 1
    if samples:
        first_slot = next(iter(samples[0][0].values()), None)
        if first_slot is not None:
            group_size = int(first_slot.shape[1])

    slot_tensors = np.zeros(
        (batch_size, n_slots, n_groups_per_slot, group_size, image_size, image_size),
        dtype=np.float32,
    )
    slot_mask = np.zeros((batch_size, n_slots), dtype=np.float32)
    labels = np.stack([s[1] for s in samples], axis=0).astype(np.float32)
    weights = np.stack([s[2] for s in samples], axis=0).astype(np.float32)

    for i, (slots, _, _) in enumerate(samples):
        for slot_name, vol in slots.items():
            if slot_name not in slot_to_idx:
                continue
            j = slot_to_idx[slot_name]
            arr = np.asarray(vol, dtype=np.float32)
            if arr.shape != slot_tensors[i, j].shape:
                raise ValueError(
                    f"Slot {slot_name} shape {arr.shape} does not match "
                    f"expected {slot_tensors[i, j].shape}"
                )
            slot_tensors[i, j] = arr
            slot_mask[i, j] = 1.0

    return slot_tensors, labels, weights, slot_mask


__all__ = ["CollateProtocol", "Batch", "collate_batch"]
