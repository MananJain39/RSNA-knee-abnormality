#!/usr/bin/env python3
"""CLI: Extract labels from radiology reports."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import LABEL_COLUMNS, PROCESSED_DATA_DIR
from src.data.loader import load_train, get_unlabeled_studies
from src.labels.regex_extractor import RegexExtractor
from src.labels.extractor import labels_to_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract labels from radiology reports")
    parser.add_argument(
        "--method",
        choices=["regex", "llm"],
        default="regex",
        help="Extraction method (default: regex)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Worker threads for --method llm (default: 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the JSONL checkpoint next to --output "
            "(<output>.jsonl); completed reports are skipped."
        ),
    )
    parser.add_argument(
        "--include-gold",
        action="store_true",
        help=(
            "Also extract for the 58 gold studies (needed to measure "
            "extractor agreement against expert labels; default skips them)."
        ),
    )
    return parser.parse_args()


def _resume_uids(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["StudyInstanceUID"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    args = parse_args()
    train_df = load_train()
    unlabeled = get_unlabeled_studies(train_df)
    if args.include_gold:
        gold = train_df.dropna(subset=[LABEL_COLUMNS[0]])
        unlabeled = pd.concat([unlabeled, gold], ignore_index=True)

    if args.limit:
        unlabeled = unlabeled.head(args.limit)

    output_path = args.output or PROCESSED_DATA_DIR / f"labels_{args.method}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")

    if args.method == "regex":
        extractor = RegexExtractor()
        reports = unlabeled["Report"].tolist()
        label_dicts = extractor.extract_batch(reports)
        labels_df = labels_to_dataframe(unlabeled["StudyInstanceUID"].tolist(), label_dicts)
        labels_df.to_csv(output_path, index=False)
        print(f"Saved labels to {output_path}")
    elif args.method == "llm":
        from src.labels.llm_extractor import LLMExtractor

        done_uids = _resume_uids(jsonl_path) if args.resume else set()
        pending = unlabeled[
            ~unlabeled["StudyInstanceUID"].isin(done_uids)
        ]
        print(
            f"Extracting labels from {len(pending)} reports with "
            f"{args.concurrency} workers"
            + (f" (resuming past {len(done_uids)} done)" if done_uids else "")
            + "..."
        )

        extractor = LLMExtractor(model=args.model)
        items = list(
            zip(pending["StudyInstanceUID"].tolist(), pending["Report"].tolist())
        )

        mode = "a" if done_uids else "w"
        with open(jsonl_path, mode) as fh:

            def on_result(uid: str, labels: dict) -> None:
                fh.write(
                    json.dumps({"StudyInstanceUID": uid, **labels}) + "\n"
                )
                fh.flush()

            extractor.extract_batch_parallel(
                items, on_result=on_result, concurrency=args.concurrency
            )

        records = []
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        labels_df = pd.DataFrame.from_records(records)
        for col in ("StudyInstanceUID", *LABEL_COLUMNS):
            if col not in labels_df.columns:
                labels_df[col] = 0.5
        labels_df.to_csv(output_path, index=False)
        print(f"Saved labels to {output_path} ({len(labels_df)} rows)")
        print(f"Checkpoint (resume-safe): {jsonl_path}")

    na_rates = (labels_df[LABEL_COLUMNS] == 0.5).mean()
    print(f"\n'Not addressed' rates per label:")
    for col, rate in na_rates.items():
        print(f"  {col}: {rate:.1%}")


if __name__ == "__main__":
    main()
