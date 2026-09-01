#!/usr/bin/env python3
"""Create the leakage-safe, gold-reserved train/validation split.

Examples
--------

Full corpus::

    python scripts/build_classical_split.py --tag v1

Restricted to a sample/dev set::

    python scripts/build_classical_split.py --tag sample200 \\
        --study-uids-json data/processed/volumes_sample200/manifest.json

The manifest file is expected to have a top-level ``study_uids`` key. CSV is
also accepted via ``--study-uids-csv``; the CSV must contain a
``StudyInstanceUID`` column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import make_split, persist_split  # noqa: E402


def _load_study_uids(args: argparse.Namespace) -> list[str] | None:
    if args.study_uids_json:
        payload = json.loads(Path(args.study_uids_json).read_text())
        uids = payload.get("study_uids")
        if uids is None:
            raise ValueError(
                f"{args.study_uids_json} has no 'study_uids' key; "
                "expected a manifest dict with study_uids list"
            )
        return list(uids)
    if args.study_uids_csv:
        import pandas as pd

        df = pd.read_csv(args.study_uids_csv)
        if "StudyInstanceUID" not in df.columns:
            raise ValueError(
                f"{args.study_uids_csv} has no 'StudyInstanceUID' column"
            )
        return df["StudyInstanceUID"].drop_duplicates().tolist()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--study-uids-json",
        type=Path,
        default=None,
        help="Manifest JSON with a 'study_uids' list (e.g. Kaggle sample manifest).",
    )
    parser.add_argument(
        "--study-uids-csv",
        type=Path,
        default=None,
        help="CSV with a 'StudyInstanceUID' column.",
    )
    args = parser.parse_args()
    study_uids = _load_study_uids(args)
    train_df, val_df = make_split(study_uids=study_uids)
    paths = persist_split(train_df, val_df, output_dir=args.output_dir, tag=args.tag)
    print(f"train={len(train_df)} val={len(val_df)}")
    print(f"train CSV: {paths['train']}\nval CSV: {paths['val']}")


if __name__ == "__main__":
    main()
