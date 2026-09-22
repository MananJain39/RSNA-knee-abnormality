#!/usr/bin/env python3
"""CLI: Fuse multiple label sources into labels_final_v2.csv.

Sources are given as ``--source name=path.csv`` (repeatable) and must contain
``StudyInstanceUID`` plus the 12 label columns with values in [0, 1]. The
gold studies come from ``data/raw/train.csv`` (expert labels).

Pipeline:
1. Measure per-label reliability (AUC vs gold-58) for every source.
2. Fuse with reliability-weighted averaging + hard-vote snapping
   (``src/labels/fusion.py``).
3. Validate every candidate (each source alone, plain mean, weighted fuse)
   against gold-58 and print a comparison table.
4. Write the weighted fusion (with gold override and the Synovitis imputation
   from ``process_labels.py``) plus confidence columns to ``--output``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import LABEL_COLUMNS  # noqa: E402
from src.labels.fusion import (  # noqa: E402
    apply_gold_override,
    fuse_labels,
    per_label_reliability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Label source as name=path.csv (repeat for each source).",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "train.csv",
        help="Competition train.csv holding the 58 gold rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "labels_final_v2.csv",
        help="Output path for the fused label set.",
    )
    return parser.parse_args()


def load_gold(train_csv: Path) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    gold = train.dropna(subset=[LABEL_COLUMNS[0]])
    return gold[["StudyInstanceUID", *LABEL_COLUMNS]]


def selective_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Same rule as scripts/process_labels.py (keep the two in sync)."""
    df = df.copy()
    synovitis_mask = df["Synovitis"] == 0.5
    df.loc[synovitis_mask & (df["Effusion"] > 0.5), "Synovitis"] = 0.63
    df.loc[synovitis_mask & (df["Effusion"] <= 0.5), "Synovitis"] = 0.22
    return df


def add_confidence_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Same formula as scripts/process_labels.py (keep the two in sync)."""
    df = df.copy()
    for col in LABEL_COLUMNS:
        df[f"{col}_conf"] = np.where(
            df[col] == 0.5,
            0.3,
            np.abs(df[col] - 0.5) * 2 + 0.5,
        )
    return df


def macro_auc_vs_gold(pred_df: pd.DataFrame, gold: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    merged = pred_df.merge(gold, on="StudyInstanceUID", suffixes=("_pred", "_gold"))
    aucs = []
    for col in LABEL_COLUMNS:
        y_true = merged[f"{col}_gold"].to_numpy(dtype=float)
        y_pred = merged[f"{col}_pred"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true, y_pred))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else 0.0


def main() -> None:
    args = parse_args()
    sources: dict[str, pd.DataFrame] = {}
    for spec in args.source:
        name, _, path = spec.partition("=")
        if not name or not path:
            sys.exit(f"--source expects NAME=PATH, got: {spec}")
        df = pd.read_csv(path)
        missing = {"StudyInstanceUID", *LABEL_COLUMNS} - set(df.columns)
        if missing:
            sys.exit(f"Source '{name}' ({path}) is missing columns: {sorted(missing)}")
        sources[name] = df
        print(f"Loaded source '{name}': {len(df)} rows from {path}")

    gold = load_gold(args.train_csv)
    print(f"Gold studies: {len(gold)}")

    reliability = per_label_reliability(sources, gold)
    print("\nPer-label reliability (AUC vs gold):")
    print(reliability.round(4).to_string())

    print("\nCandidate validation (macro AUC vs gold-58):")
    candidates: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        candidates[name] = df
    for name, df in candidates.items():
        print(f"  {name:30s}: {macro_auc_vs_gold(df, gold):.4f}")

    names = list(sources)
    stacked = pd.concat(
        [sources[n][["StudyInstanceUID", *LABEL_COLUMNS]] for n in names]
    ).groupby("StudyInstanceUID")[LABEL_COLUMNS].mean().reset_index()
    print(f"  {'plain-mean':30s}: {macro_auc_vs_gold(stacked, gold):.4f}")

    fused = fuse_labels(sources, reliability)
    print(f"  {'reliability-weighted fuse':30s}: {macro_auc_vs_gold(fused, gold):.4f}")

    fused = selective_imputation(fused)
    fused = apply_gold_override(fused, gold)
    fused = add_confidence_weights(fused)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(args.output, index=False)
    print(f"\nSaved fused labels to {args.output} ({len(fused)} rows)")


if __name__ == "__main__":
    main()
