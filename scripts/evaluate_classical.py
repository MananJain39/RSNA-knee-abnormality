#!/usr/bin/env python3
"""Evaluate a classical checkpoint against the gold studies in the validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.dataset import build_default_volume_index  # noqa: E402
from src.classical.dinov2_base import training_hyperparams  # noqa: E402
from src.classical.inference import ClassicalPredictor  # noqa: E402
from src.config import LABEL_COLUMNS  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.validation.gold import load_gold_labels  # noqa: E402
from src.validation.metrics import evaluate_labels_vs_gold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-checkpoint", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=Path("data/processed/labels_final.csv"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-tag", default="v1")
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    _, val_df = load_split(args.split_tag, args.split_dir)
    gold = load_gold_labels()
    gold = gold[gold["StudyInstanceUID"].isin(val_df["StudyInstanceUID"])]
    if gold.empty:
        raise ValueError("The validation split contains no gold studies")

    labels = pd.read_csv(args.labels_csv)
    volumes = build_default_volume_index(args.data_dir)
    missing = set(gold["StudyInstanceUID"]) - set(volumes)
    if missing:
        raise FileNotFoundError(f"Missing preprocessed volumes for {len(missing)} gold studies")
    predictions = ClassicalPredictor(device=args.device).predict_studies(
        model_paths={"main": args.main_checkpoint},
        study_uids=gold["StudyInstanceUID"].tolist(),
        volume_index=volumes,
        labels_df=labels,
        cluster_assignments=pd.DataFrame(columns=["StudyInstanceUID", "cluster_id"]),
    )
    result = evaluate_labels_vs_gold(predictions, gold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "gold_predictions.csv", index=False)
    result["detailed_report"].to_csv(args.output_dir / "gold_per_label_metrics.csv", index=False)
    summary = {
        "n_gold_studies": len(gold),
        "macro_auc": result["macro_auc"],
        "per_label_auc": result["per_label_auc"],
        "labels": LABEL_COLUMNS,
        "checkpoint": str(args.main_checkpoint),
        "training_hyperparameters": training_hyperparams(),
    }
    (args.output_dir / "gold_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Gold macro AUC ({len(gold)} studies): {result['macro_auc']:.4f}")
    print(f"Metrics written to {args.output_dir}")


if __name__ == "__main__":
    main()
