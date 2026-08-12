"""Image preprocessing utilities.

This is a stub module. Full implementation requires pydicom and the DICOM dataset.
"""

from typing import Optional
import numpy as np


def normalize(
    image: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Normalize image intensities.

    Args:
        image: Input image array
        method: Normalization method ('minmax', 'zscore', 'clip')

    Returns:
        Normalized image array
    """
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
    target_shape: tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """Resize 3D image to target shape.

    Args:
        image: Input 3D array
        target_shape: Target (depth, height, width)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resized image array

    Note:
        Requires scipy for interpolation. Stub implementation returns input.
    """
    raise NotImplementedError(
        "Image resizing requires scipy. Install with: pip install scipy"
    )


def to_tensor(
    image: np.ndarray,
    channel_first: bool = True,
) -> "torch.Tensor":
    """Convert numpy array to PyTorch tensor.

    Args:
        image: Input numpy array
        channel_first: If True, return (C, D, H, W), else (D, H, W, C)

    Returns:
        PyTorch tensor

    Note:
        Requires PyTorch. Stub raises NotImplementedError.
    """
    raise NotImplementedError(
        "Tensor conversion requires PyTorch. Install with: pip install torch"
    )


def extract_middle_slices(
    image: np.ndarray,
    n_slices: int = 3,
) -> np.ndarray:
    """Extract middle N slices from a 3D volume.

    Args:
        image: 3D array of shape (depth, height, width)
        n_slices: Number of slices to extract

    Returns:
        Array of shape (n_slices, height, width)
    """
    depth = image.shape[0]
    center = depth // 2
    half = n_slices // 2

    start = max(0, center - half)
    end = min(depth, start + n_slices)

    if end - start < n_slices:
        start = max(0, end - n_slices)

    return image[start:end]
