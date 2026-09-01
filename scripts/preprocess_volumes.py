#!/usr/bin/env python3
"""Preprocess RSNA DICOM series into disk-backed five-slot study volumes.

Run this on Kaggle, where the competition DICOM tree is mounted.  The output
layout is directly consumable by ``scripts/train_classical.py``:
``<output>/<StudyInstanceUID>/<slot>.npy``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_train_series  # noqa: E402
from src.imaging.clustering import _syntax_family  # noqa: E402
from src.imaging.dicom_loader import get_series_metadata, load_series  # noqa: E402
from src.imaging.pipeline import preprocess_series  # noqa: E402
from src.imaging.slot_builder import build_slots_for_study  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--series-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--max-studies", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=336)
    return parser.parse_args()


def _metadata_row(study_uid: str, series_uid: str, series_row: pd.Series, meta: dict) -> dict:
    return {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "pixel_spacing_mm": float(meta["pixel_spacing"][0]),
        "rows": int(meta["rows"]),
        "cols": int(meta["columns"]),
        "n_slices": int(meta["n_slices"]),
        "slice_thickness_mm": float(meta["slice_thickness"]),
        "fov_extent_mm": float(meta["pixel_spacing"][0]) * int(meta["rows"]),
        "fluid_sensitive": int(series_row["Fluid_Sensitive"]),
        "fat_suppression": int(series_row["Fat_Suppression"]),
        "transfer_syntax_family": _syntax_family(meta["transfer_syntax"]),
    }


def main() -> None:
    args = parse_args()
    series_df = pd.read_csv(args.series_csv) if args.series_csv else load_train_series()
    study_uids = series_df["StudyInstanceUID"].drop_duplicates().tolist()
    if args.max_studies is not None:
        study_uids = study_uids[: args.max_studies]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows, failures = [], []
    for study_uid in tqdm(study_uids, desc="Preprocessing studies"):
        slots = build_slots_for_study(study_uid, series_df)
        study_dir = args.output_dir / study_uid
        saved_any = False
        for slot_name, slot in slots.items():
            if slot is None:
                continue
            try:
                meta = get_series_metadata(study_uid, slot.series_uid, args.dicom_dir)
                volume = load_series(study_uid, slot.series_uid, args.dicom_dir)
                processed = preprocess_series(volume, meta, slot.anatomical_plane, args.image_size)
                study_dir.mkdir(parents=True, exist_ok=True)
                np.save(study_dir / f"{slot_name}.npy", processed)
                series_row = series_df.loc[
                    (series_df["StudyInstanceUID"] == study_uid)
                    & (series_df["SeriesInstanceUID"] == slot.series_uid)
                ].iloc[0]
                metadata_rows.append(_metadata_row(study_uid, slot.series_uid, series_row, meta))
                saved_any = True
            except Exception as exc:  # preserve the rest of a long corpus run
                failures.append(
                    {"StudyInstanceUID": study_uid, "slot": slot_name, "error": str(exc)}
                )
        if not saved_any and study_dir.exists():
            study_dir.rmdir()

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_rows).to_csv(args.metadata_output, index=False)
    failure_path = args.metadata_output.with_name("preprocessing_failures.csv")
    pd.DataFrame(failures).to_csv(failure_path, index=False)
    print(f"Saved {len(metadata_rows)} series across {len(study_uids)} requested studies")
    print(f"Metadata: {args.metadata_output}; failures: {len(failures)} ({failure_path})")


if __name__ == "__main__":
    main()
