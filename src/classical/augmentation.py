"""Data augmentation for knee MRI volumes.

Applies geometric and intensity transformations to 2.5D slot groups.
All slices within a slot share ONE transform (a slot is one MRI series, and
the reference augmentation policy applies per-series). CRITICAL: never
horizontally flip — that would swap left/right anatomy without updating
laterality-specific labels (medial vs lateral meniscus, MCL, etc.).

Augmentations (sampled once per slot):
- Random rotation: ±rotation_deg (default 10°)
- Brightness-gamma jitter: gamma in [gamma_lo, gamma_hi] (default [0.8, 1.2])
- Slight rescale: scale in [scale_lo, scale_hi] (default [0.95, 1.05])

The whole (N, 3, H, W) stack is interpolated in one vectorized pass: the
transform is identical for every slice, so coordinate maps are computed once.
"""

from __future__ import annotations

import numpy as np


def augment_slot_group(
    slot_group: np.ndarray,
    rotation_deg: float = 10.0,
    gamma_range: tuple[float, float] = (0.8, 1.2),
    rescale_range: tuple[float, float] = (0.95, 1.05),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply one shared rotation + rescale + gamma transform to a slot group.

    Args:
        slot_group: Shape (N, 3, H, W) float32 in [0, 1].
        rotation_deg: Max rotation in degrees (±rotation_deg).
        gamma_range: Gamma jitter bounds (gamma_lo, gamma_hi).
        rescale_range: Spatial scale bounds (scale_lo, scale_hi).
        rng: Random generator (for reproducibility).

    Returns:
        Augmented slot_group, same shape, clipped to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    slot_group = np.asarray(slot_group, dtype=np.float32)
    n_groups, group_size, h, w = slot_group.shape

    angle = rng.uniform(-rotation_deg, rotation_deg)
    gamma = rng.uniform(gamma_range[0], gamma_range[1])
    scale = rng.uniform(rescale_range[0], rescale_range[1])

    cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    transform = np.array(
        [[scale * cos_a, -scale * sin_a], [scale * sin_a, scale * cos_a]]
    )
    inv_transform = np.linalg.inv(transform)

    coords_y, coords_x = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.stack(
        [coords_y - h / 2.0, coords_x - w / 2.0], axis=-1
    )  # (H, W, 2)
    src = coords @ inv_transform.T.astype(np.float32)
    src_y = np.clip(src[..., 0] + h / 2.0, 0, h - 1)
    src_x = np.clip(src[..., 1] + w / 2.0, 0, w - 1)

    y0 = np.floor(src_y).astype(np.int32)
    x0 = np.floor(src_x).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (src_y - y0)[None, ...]
    wx = (src_x - x0)[None, ...]

    stack = slot_group.reshape(n_groups * group_size, h, w)
    top_left = stack[:, y0, x0]
    top_right = stack[:, y0, x1]
    bottom_left = stack[:, y1, x0]
    bottom_right = stack[:, y1, x1]
    interpolated = (
        top_left * (1 - wy) * (1 - wx)
        + top_right * (1 - wy) * wx
        + bottom_left * wy * (1 - wx)
        + bottom_right * wy * wx
    )
    augmented = np.power(interpolated, gamma, dtype=np.float32)
    return np.clip(augmented, 0.0, 1.0).reshape(n_groups, group_size, h, w)


def augment_slots(
    slots: dict[str, np.ndarray],
    rotation_deg: float = 10.0,
    gamma_range: tuple[float, float] = (0.8, 1.2),
    rescale_range: tuple[float, float] = (0.95, 1.05),
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Augment every slot of a study, each with its own sampled transform.

    Each slot is one MRI series, so per-slot transforms simulate natural
    variation between sequences/planes while keeping slices within a series
    anatomically consistent.
    """
    if rng is None:
        rng = np.random.default_rng()

    return {
        slot_name: augment_slot_group(
            slot_group,
            rotation_deg=rotation_deg,
            gamma_range=gamma_range,
            rescale_range=rescale_range,
            rng=rng,
        )
        for slot_name, slot_group in slots.items()
    }


__all__ = ["augment_slot_group", "augment_slots"]
