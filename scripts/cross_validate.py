#!/usr/bin/env python3
"""Generate leak-safe, gold-reserved K-fold CV splits and optionally run them.

Each fold holds out a report-grouped slice of the non-gold studies as the
development set and keeps all gold studies on the validation side. Training
never sees gold, so scoring each fold's checkpoint against gold-58 yields a
cross-fitted estimate (guarding against the single-split variance and the
fold-leakage failure mode documented in RESEARCH_SUMMARY.md).

Default action generates and validates the splits (no training). Pass
``--run`` to execute one ``train_classical.py`` run per fold sequentially.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.config import LABEL_COLUMNS, PROCESSED_DATA_DIR  # noqa: E402
from src.data.loader import load_train  # noqa: E402
from src.data.splits import _build_group_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tag", default="cv5", help="Split tag under data/processed/splits/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--study-uids-csv",
        type=Path,
        default=None,
        help="Optional CSV with a StudyInstanceUID column restricting the corpus.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="After generating splits, run train_classical.py for each fold.",
    )
    parser.add_argument(
        "--extra-train-args",
        default="",
        help="Extra args passed through to train_classical.py with --run.",
    )
    return parser.parse_args()


def build_cv_splits(
    folds: int,
    tag: str,
    seed: int,
    study_uids_csv: Path | None,
) -> Path:
    train_df = load_train()
    if study_uids_csv is not None:
        uids = set(pd.read_csv(study_uids_csv)["StudyInstanceUID"])
        train_df = train_df[train_df["StudyInstanceUID"].isin(uids)].reset_index(drop=True)

    gold_mask = train_df[LABEL_COLUMNS[0]].notna()
    gold = train_df[gold_mask].copy()
    pool = train_df[~gold_mask].reset_index(drop=True)

    if pool.empty:
        sys.exit("No non-gold studies available for the CV training pool")
    if len(gold) == 0:
        sys.exit("No gold studies found; cannot build gold-reserved CV splits")

    pool = pool.copy()
    pool["_group_id"] = _build_group_ids(pool, "Report").to_numpy()

    splitter = GroupKFold(n_splits=folds)
    groups = pool["_group_id"].to_numpy()
    if len(pd.unique(groups)) < folds:
        sys.exit(f"Need at least {folds} report groups, found {len(pd.unique(groups))}")

    out_root = PROCESSED_DATA_DIR / "splits" / tag
    out_root.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, dev_idx) in enumerate(splitter.split(pool, groups=groups)):
        fold_train = pool.iloc[train_idx]
        fold_dev = pool.iloc[dev_idx]
        fold_val = pd.concat([fold_dev, gold], ignore_index=True)

        overlap = set(fold_train["_group_id"]) & set(fold_dev["_group_id"])
        if overlap:
            sys.exit(f"Fold {fold}: report-group leak across train/dev ({len(overlap)} groups)")
        train_uids = set(fold_train["StudyInstanceUID"])
        val_uids = set(fold_val["StudyInstanceUID"])
        if train_uids & val_uids:
            sys.exit(f"Fold {fold}: UID overlap between train and val")

        fold_dir = out_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_train.drop(columns=["_group_id"]).to_csv(fold_dir / "train.csv", index=False)
        fold_val.to_csv(fold_dir / "val.csv", index=False)
        print(
            f"fold{fold}: train={len(fold_train)} dev={len(fold_dev)} "
            f"gold={len(gold)} val={len(fold_val)}"
        )

    print(f"Saved {folds}-fold splits to {out_root}")
    return out_root


def run_folds(args: argparse.Namespace, out_root: Path) -> None:
    fold_dirs = sorted(p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("fold"))
    if not fold_dirs:
        sys.exit("No fold directories found")
    for fold_dir in fold_dirs:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_classical.py"),
            "--mode", "main",
            "--split-tag", f"{out_root.name}/{fold_dir.name}",
            "--output-dir", str(fold_dir / "ckpt"),
        ]
        if args.extra_train_args:
            cmd.extend(args.extra_train_args.split())
        print(f"\n=== {fold_dir.name}: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(f"{fold_dir.name} training failed rc={rc}")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        sys.exit("--folds must be >= 2")
    out_root = build_cv_splits(args.folds, args.tag, args.seed, args.study_uids_csv)
    if args.run:
        run_folds(args, out_root)


if __name__ == "__main__":
    main()
