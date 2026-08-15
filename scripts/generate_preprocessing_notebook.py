"""Generate the DICOM preprocessing notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "kaggle": {
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
    },
}

cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


md("""# RSNA Knee — DICOM Preprocessing Pipeline

This notebook implements the complete DICOM preprocessing pipeline
for the RSNA 2026 Knee Abnormality Detection competition.

## Pipeline Steps
1. Load DICOM series with physical position sorting
2. Normalize laterality (left/right knee)
3. Physical crop (130mm → 336px)
4. Per-series percentile normalization
5. Slot assignment (5 slots: plane × contrast)
6. Slice sampling (central band)
7. 2.5D group creation (groups of 3 adjacent slices)
8. Cache processed volumes

## Output
Processed volumes saved as a Kaggle dataset for use in training notebooks.
""")

code("""import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import json

# Competition paths
COMP_DIR = Path("/kaggle/input/rsna-knee-abnormality-detection")
TRAIN_DIR = COMP_DIR / "train_series"
OUTPUT_DIR = Path("/kaggle/working/processed_volumes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
train_series = pd.read_csv(COMP_DIR / "train_series.csv")
print(f"Total series: {len(train_series)}")
print(f"Total studies: {train_series['StudyInstanceUID'].nunique()}")
""")

md("""## 1. Geometry Utilities

Physical position sorting and laterality normalization.
""")

code("""def compute_slice_normal(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    col_cosine = np.array(image_orientation[3:])
    return np.cross(row_cosine, col_cosine)


def compute_physical_position(image_position, image_orientation):
    position = np.array(image_position)
    normal = compute_slice_normal(image_orientation)
    return float(np.dot(position, normal))


def sort_by_physical_position(datasets):
    positions = [compute_physical_position(ds.ImagePositionPatient, ds.ImageOrientationPatient)
                 for ds in datasets]
    sorted_indices = np.argsort(positions)
    return [datasets[i] for i in sorted_indices]


def compute_image_center(image_position, image_orientation, pixel_spacing, rows, columns):
    position = np.array(image_position)
    row_cosine = np.array(image_orientation[:3])
    col_cosine = np.array(image_orientation[3:])
    center = (position
              + row_cosine * pixel_spacing[0] * (rows / 2)
              + col_cosine * pixel_spacing[1] * (columns / 2))
    return center


def determine_laterality(image_position, image_orientation, pixel_spacing, rows, columns,
                         laterality_tag=None, dead_zone=25.0):
    center = compute_image_center(image_position, image_orientation, pixel_spacing, rows, columns)
    if abs(center[0]) > dead_zone:
        return "L" if center[0] < 0 else "R"
    if laterality_tag in ("L", "R"):
        return laterality_tag
    return "R"


def normalize_laterality(pixel_array, laterality, anatomical_plane):
    if laterality != "L":
        return pixel_array
    if anatomical_plane in ("Coronal", "Axial"):
        return np.fliplr(pixel_array)
    return pixel_array
""")

md("""## 2. Preprocessing Functions

Physical crop, normalization, and 2.5D grouping.
""")

code("""def physical_crop(image, pixel_spacing, target_extent_mm=130.0):
    target_pixels = int(target_extent_mm / pixel_spacing)
    h, w = image.shape[-2:]
    if target_pixels >= min(h, w):
        return image
    crop_h = (h - target_pixels) // 2
    crop_w = (w - target_pixels) // 2
    if image.ndim == 2:
        return image[crop_h:crop_h + target_pixels, crop_w:crop_w + target_pixels]
    return image[:, crop_h:crop_h + target_pixels, crop_w:crop_w + target_pixels]


def resize(image, target_shape, order=1):
    zoom_factors = tuple(t / s for t, s in zip(target_shape, image.shape))
    return zoom(image, zoom_factors, order=order).astype(np.float32)


def percentile_normalize_series(volume, low=1.0, high=99.0):
    p_low = np.percentile(volume, low)
    p_high = np.percentile(volume, high)
    if p_high - p_low == 0:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(volume, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


def sample_central_band(n_slices, n_samples, low_frac=0.12, high_frac=0.88):
    low = int(n_slices * low_frac)
    high = int(n_slices * high_frac)
    high = max(high, low + 1)
    return np.linspace(low, high - 1, n_samples, dtype=int)


def create_25d_groups(volume, group_size=3):
    n_slices = volume.shape[0]
    n_groups = n_slices - group_size + 1
    if n_groups <= 0:
        return volume[np.newaxis, ...]
    return np.stack([volume[i:i + group_size] for i in range(n_groups)], axis=0)
""")

md("""## 3. Slot Definitions

Organize series into 5 slots based on plane × contrast.
""")

code("""SLOT_DEFINITIONS = {
    "Sagittal_Fluid": {"plane": "Sagittal", "fluid": 1},
    "Sagittal_Structural": {"plane": "Sagittal", "fluid": 0},
    "Coronal_Fluid": {"plane": "Coronal", "fluid": 1},
    "Coronal_Structural": {"plane": "Coronal", "fluid": 0},
    "Axial_Fluid": {"plane": "Axial", "fluid": 1},
}

def assign_slot(plane, fluid_sensitive):
    for slot_name, defn in SLOT_DEFINITIONS.items():
        if defn["plane"] == plane and defn["fluid"] == fluid_sensitive:
            return slot_name
    return None

print("Slot definitions:")
for name, defn in SLOT_DEFINITIONS.items():
    count = len(train_series[
        (train_series["Anatomical_Plane"] == defn["plane"]) &
        (train_series["Fluid_Sensitive"] == defn["fluid"])
    ])
    print(f"  {name}: {count} series")
""")

md("""## 4. Process Single Series

Load, sort, normalize, crop, resize.
""")

code("""TARGET_SIZE = 336
TARGET_EXTENT_MM = 130.0


def process_series(study_uid, series_uid, anatomical_plane):
    series_dir = TRAIN_DIR / study_uid / series_uid
    dicom_files = sorted(series_dir.glob("*.dcm"))

    if not dicom_files:
        return None, None

    datasets = [pydicom.dcmread(str(f), stop_before_pixels=False) for f in dicom_files]
    sorted_datasets = sort_by_physical_position(datasets)

    first_ds = sorted_datasets[0]
    pixel_spacing = float(first_ds.PixelSpacing[0])
    laterality = determine_laterality(
        first_ds.ImagePositionPatient,
        first_ds.ImageOrientationPatient,
        first_ds.PixelSpacing,
        first_ds.Rows,
        first_ds.Columns,
        laterality_tag=first_ds.get("Laterality"),
    )

    slices = []
    for ds in sorted_datasets:
        pixel_array = ds.pixel_array.astype(np.float32)
        pixel_array = normalize_laterality(pixel_array, laterality, anatomical_plane)
        slices.append(pixel_array)

    volume = np.stack(slices, axis=0)
    volume = physical_crop(volume, pixel_spacing, TARGET_EXTENT_MM)
    volume = resize(volume, (volume.shape[0], TARGET_SIZE, TARGET_SIZE))
    volume = percentile_normalize_series(volume)

    return volume.astype(np.uint8), {
        "laterality": laterality,
        "n_slices": len(sorted_datasets),
        "pixel_spacing": pixel_spacing,
    }
""")

md("""## 5. Process All Studies

Build slots, sample slices, create 2.5D groups, cache volumes.
""")

code("""N_SLICES_PER_SLOT = 12
GROUP_SIZE = 3

study_uids = train_series["StudyInstanceUID"].unique()
print(f"Processing {len(study_uids)} studies...")

results = {}
errors = []

for study_uid in tqdm(study_uids[:100]):
    study_series = train_series[train_series["StudyInstanceUID"] == study_uid]
    slots = {}

    for _, row in study_series.iterrows():
        slot_name = assign_slot(row["Anatomical_Plane"], int(row["Fluid_Sensitive"]))
        if slot_name is None or slot_name in slots:
            continue

        volume, meta = process_series(study_uid, row["SeriesInstanceUID"], row["Anatomical_Plane"])
        if volume is None:
            continue

        indices = sample_central_band(volume.shape[0], N_SLICES_PER_SLOT)
        sampled = volume[indices]
        groups = create_25d_groups(sampled, GROUP_SIZE)

        slots[slot_name] = {
            "groups": groups,
            "series_uid": row["SeriesInstanceUID"],
            "meta": meta,
        }

    results[study_uid] = slots

print(f"Processed {len(results)} studies")
print(f"Sample study slots: {list(results[list(results.keys())[0]].keys())}")
""")

md("""## 6. Save Processed Volumes

Save as numpy arrays for use in training notebooks.
""")

code("""import pickle

output_file = OUTPUT_DIR / "processed_volumes_sample.pkl"
with open(output_file, "wb") as f:
    pickle.dump(results, f)

print(f"Saved to {output_file}")
print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

# Verify
with open(output_file, "rb") as f:
    loaded = pickle.load(f)

study_uid = list(loaded.keys())[0]
print(f"\\nVerification:")
print(f"  Study: {study_uid}")
print(f"  Slots: {list(loaded[study_uid].keys())}")
for slot_name, slot_data in loaded[study_uid].items():
    print(f"    {slot_name}: groups shape = {slot_data['groups'].shape}")
""")

md("""## 7. Slot Fill Rates

Check how often each slot is present.
""")

code("""slot_counts = {name: 0 for name in SLOT_DEFINITIONS}

for study_uid, slots in results.items():
    for slot_name in slots:
        slot_counts[slot_name] += 1

n_studies = len(results)
print("Slot fill rates:")
for name, count in slot_counts.items():
    rate = count / n_studies
    bar = "█" * int(rate * 20)
    print(f"  {name:25s}: {rate:5.1%} {bar}")
""")

md("""## Summary

This notebook implements the complete DICOM preprocessing pipeline:
- Physical position sorting (not filename)
- Laterality normalization (geometry-based)
- Physical crop (130mm → 336px)
- Per-series percentile normalization
- Slot assignment (5 slots)
- Central band slice sampling
- 2.5D group creation

For the full dataset (~4400 studies), run this notebook with GPU enabled
and save the output as a Kaggle dataset.
""")

nb.cells = cells

with open("notebooks/01_dicom_preprocessing.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created: notebooks/01_dicom_preprocessing.ipynb")
