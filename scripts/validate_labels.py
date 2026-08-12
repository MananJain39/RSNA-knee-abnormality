#!/usr/bin/env python3
"""CLI: Validate extracted labels against the 58 gold studies."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import PROCESSED_DATA_DIR
from src.data.loader import load_train
from src.validation.gold import load_gold_labels
from src.validation.metrics import evaluate_labels_vs_gold


def main():
    parser = argparse.ArgumentParser(description="Validate extracted labels against gold standard")
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Path to predicted labels CSV (default: data/processed/labels_regex.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification (default: 0.5)",
    )
    args = parser.parse_args()

    labels_path = args.labels_file or PROCESSED_DATA_DIR / "labels_regex.csv"
    if not Path(labels_path).exists():
        print(f"Labels file not found: {labels_path}")
        print("Run: python scripts/extract_labels.py --method regex")
        sys.exit(1)

    print(f"Loading predicted labels from {labels_path}...")
    predicted_df = pd.read_csv(labels_path)

    print("Loading gold labels...")
    gold_df = load_gold_labels()

    gold_uids = set(gold_df["StudyInstanceUID"])
    pred_uids = set(predicted_df["StudyInstanceUID"])
    common_uids = gold_uids & pred_uids

    if not common_uids:
        print("No common study UIDs between predicted and gold labels.")
        print("The predicted labels must include the 58 gold studies for validation.")
        sys.exit(1)

    print(f"Found {len(common_uids)} common studies between predicted and gold.")

    pred_subset = predicted_df[predicted_df["StudyInstanceUID"].isin(common_uids)]
    gold_subset = gold_df[gold_df["StudyInstanceUID"].isin(common_uids)]

    results = evaluate_labels_vs_gold(pred_subset, gold_subset)

    print(f"\n{'='*50}")
    print(f"Macro AUC: {results['macro_auc']:.4f}")
    print(f"{'='*50}")

    print(f"\nPer-label AUC:")
    for label, auc in results["per_label_auc"].items():
        if pd.notna(auc):
            print(f"  {label:20s}: {auc:.4f}")
        else:
            print(f"  {label:20s}: N/A (constant label)")

    print(f"\nDetailed report:")
    print(results["detailed_report"].to_string(index=False))


if __name__ == "__main__":
    main()
