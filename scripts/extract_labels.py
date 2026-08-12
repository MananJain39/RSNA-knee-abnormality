#!/usr/bin/env python3
"""CLI: Extract labels from radiology reports."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import LABEL_COLUMNS, PROCESSED_DATA_DIR
from src.data.loader import load_train, get_unlabeled_studies
from src.labels.regex_extractor import RegexExtractor
from src.labels.extractor import labels_to_dataframe


def main():
    parser = argparse.ArgumentParser(description="Extract labels from radiology reports")
    parser.add_argument(
        "--method",
        choices=["regex", "llm"],
        default="regex",
        help="Extraction method (default: regex)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model name (for --method llm)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/processed/labels_<method>.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of reports to process (for testing)",
    )
    args = parser.parse_args()

    train_df = load_train()
    unlabeled = get_unlabeled_studies(train_df)

    if args.limit:
        unlabeled = unlabeled.head(args.limit)

    print(f"Extracting labels from {len(unlabeled)} unlabeled reports using {args.method}...")

    if args.method == "regex":
        extractor = RegexExtractor()
        reports = unlabeled["Report"].tolist()
        label_dicts = extractor.extract_batch(reports)
    elif args.method == "llm":
        from src.labels.llm_extractor import LLMExtractor
        extractor = LLMExtractor(model=args.model)
        reports = unlabeled["Report"].tolist()
        label_dicts = extractor.extract_batch(reports)

    labels_df = labels_to_dataframe(unlabeled["StudyInstanceUID"].tolist(), label_dicts)

    output_path = args.output or PROCESSED_DATA_DIR / f"labels_{args.method}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_path, index=False)
    print(f"Saved labels to {output_path}")

    na_rates = (labels_df[LABEL_COLUMNS] == 0.5).mean()
    print(f"\n'Not addressed' rates per label:")
    for col, rate in na_rates.items():
        print(f"  {col}: {rate:.1%}")


if __name__ == "__main__":
    main()
