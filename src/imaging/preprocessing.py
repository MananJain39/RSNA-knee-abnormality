"""Image preprocessing utilities."""

import numpy as np
from scipy.ndimage import zoom


def normalize(
    image: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    if method == "minmax":
        img_min, img_max = image.min(), image.max()
        if img_max - img_min == 0:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - img_min) / (img_max - img_min)).astype(np.float32)

    elif method == "zscore":
        mean, std = image.mean(), image.std()
        if std == 0:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - mean) / std).astype(np.float32)

    elif method == "clip":
        p_low, p_high = np.percentile(image, [1, 99])
        return np.clip(image, p_low, p_high).astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize(
    image: np.ndarray,
    target_shape: tuple[int, ...],
    order: int = 1,
) -> np.ndarray:
    zoom_factors = tuple(t / s for t, s in zip(target_shape, image.shape))
    return zoom(image, zoom_factors, order=order).astype(np.float32)


def physical_crop(
    image: np.ndarray,
    pixel_spacing: float,
    target_extent_mm: float = 130.0,
) -> np.ndarray:
    target_pixels = int(target_extent_mm / pixel_spacing)
    h, w = image.shape[-2:]

    if target_pixels >= min(h, w):
        return image

    crop_h = (h - target_pixels) // 2
    crop_w = (w - target_pixels) // 2

    if image.ndim == 2:
        return image[crop_h:crop_h + target_pixels, crop_w:crop_w + target_pixels]
    elif image.ndim == 3:
        return image[:, crop_h:crop_h + target_pixels, crop_w:crop_w + target_pixels]
    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim}")


def percentile_normalize_series(
    volume: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    p_low = np.percentile(volume, low)
    p_high = np.percentile(volume, high)

    if p_high - p_low == 0:
        return np.zeros_like(volume, dtype=np.float32)

    clipped = np.clip(volume, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


def to_tensor(
    image: np.ndarray,
    channel_first: bool = True,
):
    import torch

    tensor = torch.from_numpy(image.astype(np.float32))

    if channel_first and tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif channel_first and tensor.ndim == 4:
        tensor = tensor.permute(3, 0, 1, 2)

    return tensor


def extract_middle_slices(
    image: np.ndarray,
    n_slices: int = 3,
) -> np.ndarray:
    depth = image.shape[0]
    center = depth // 2
    half = n_slices // 2

    start = max(0, center - half)
    end = min(depth, start + n_slices)

    if end - start < n_slices:
        start = max(0, end - n_slices)

    return image[start:end]


def sample_central_band(
    n_slices: int,
    n_samples: int,
    low_frac: float = 0.12,
    high_frac: float = 0.88,
) -> np.ndarray:
    low = int(n_slices * low_frac)
    high = int(n_slices * high_frac)
    high = max(high, low + 1)
    return np.linspace(low, high - 1, n_samples, dtype=int)


def create_25d_groups(
    volume: np.ndarray,
    group_size: int = 3,
) -> np.ndarray:
    n_slices = volume.shape[0]
    n_groups = n_slices - group_size + 1

    if n_groups <= 0:
        return volume[np.newaxis, ...]

    groups = np.stack(
        [volume[i:i + group_size] for i in range(n_groups)],
        axis=0,
    )
    return groups
