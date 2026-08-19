#!/usr/bin/env python3
"""CLI: Run inference with the trained main branch + cluster specialists.

Loads checkpoints, runs the predictor on the test set, and writes a
schema-valid submission CSV. Use ``--evaluate-against-gold`` to also
report macro-AUC against the 58 expert-labeled studies (internal
sanity check; not used during actual competition inference).

Examples:

    # Main branch only
    python scripts/infer_classical.py --main-checkpoint runs/main/best.pt

    # Main + cluster specialists (filename pattern cluster_<id>.pt)
    python scripts/infer_classical.py \\
        --main-checkpoint runs/main/best.pt \\
        --specialist-checkpoints-dir runs/specialists/

    # Use a sample_submission.csv to fix row order
    python scripts/infer_classical.py \\
        --main-checkpoint runs/main/best.pt \\
        --sample-submission data/raw/sample_submission.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.classical import ClassicalPredictor, KneeMRIDataset, SubmissionWriter  # noqa: E402
from src.classical.dataset import build_default_volume_index  # noqa: E402
from src.classical.inference import collate_batch  # noqa: E402
from src.config import LABEL_COLUMNS  # noqa: E402
from src.data.loader import load_test  # noqa: E402
from src.imaging.clustering import load_clusters, majority_cluster_per_study  # noqa: E402
from src.validation.gold import load_gold_labels  # noqa: E402
from src.validation.metrics import evaluate_labels_vs_gold  # noqa: E402

_CLUSTER_PATTERN = re.compile(r"cluster_(\d+)\.pt$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--main-checkpoint",
        required=True,
        help="Path to the main branch best.pt.",
    )
    parser.add_argument(
        "--specialist-checkpoints-dir",
        default=None,
        help="Directory of cluster_<id>.pt files. Optional.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root of preprocessed volumes (default: data/processed/volumes).",
    )
    parser.add_argument(
        "--labels-csv",
        default="data/processed/labels_final.csv",
        help="Used to supply labels for the predictor (default: labels_final.csv).",
    )
    parser.add_argument(
        "--clusters-csv",
        default="data/processed/series_clusters.csv",
        help="Per-series cluster assignments CSV.",
    )
    parser.add_argument(
        "--test-csv",
        default="data/raw/test.csv",
        help="Study UID source for inference (default: test.csv).",
    )
    parser.add_argument(
        "--sample-submission",
        default="data/raw/sample_submission.csv",
        help="Used to fix submission row order (default: sample_submission.csv).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/submission.csv",
        help="Submission CSV output path.",
    )
    parser.add_argument(
        "--evaluate-against-gold",
        action="store_true",
        help="Also report macro-AUC on the 58 gold studies (internal check).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (default: auto cuda/cpu).",
    )
    return parser.parse_args()


def _discover_specialists(specialist_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in specialist_dir.glob("cluster_*.pt"):
        match = _CLUSTER_PATTERN.search(path.name)
        if match:
            out[int(match.group(1))] = path
    return out


def main() -> None:
    args = _parse_args()

    main_path = Path(args.main_checkpoint)
    if not main_path.exists():
        sys.exit(f"Main checkpoint not found: {main_path}")

    specialist_paths: dict[int, Path] = {}
    if args.specialist_checkpoints_dir:
        specialist_dir = Path(args.specialist_checkpoints_dir)
        if not specialist_dir.exists():
            sys.exit(f"Specialist directory not found: {specialist_dir}")
        specialist_paths = _discover_specialists(specialist_dir)
        print(f"Discovered {len(specialist_paths)} specialist checkpoint(s)")

    print(f"Loading labels from {args.labels_csv}...")
    labels_df = pd.read_csv(args.labels_csv)

    print("Loading cluster assignments...")
    study_clusters = majority_cluster_per_study(load_clusters(Path(args.clusters_csv)))

    print(f"Loading volume index from {args.data_dir or 'data/processed/volumes'}...")
    volume_index = build_default_volume_index(
        Path(args.data_dir) if args.data_dir else None
    )
    print(f"  {len(volume_index)} studies")

    print(f"Loading test UIDs from {args.test_csv}...")
    test_df = load_test()
    study_uids = test_df["StudyInstanceUID"].tolist()
    print(f"  {len(study_uids)} test studies")

    predictor = ClassicalPredictor(
        device=args.device,
        dataset_factory=lambda uids, ldf, vi: KneeMRIDataset(uids, ldf, volume_index=vi),
        collate_fn=lambda samples: collate_batch(samples, image_size=336),
    )

    model_paths = {"main": main_path, "specialists": specialist_paths}
    predictions = predictor.predict_studies(
        model_paths=model_paths,
        study_uids=study_uids,
        volume_index=volume_index,
        labels_df=labels_df,
        cluster_assignments=study_clusters,
    )

    print(f"Writing submission to {args.output}...")
    sample_path = Path(args.sample_submission) if args.sample_submission else None
    if sample_path is not None and not sample_path.exists():
        print(f"  (sample_submission not found at {sample_path}; using lex order)")
        sample_path = None

    submission_path = SubmissionWriter().write(
        predictions,
        Path(args.output),
        sample_submission_path=sample_path,
    )
    print(f"  done: {submission_path}")

    if args.evaluate_against_gold:
        gold = load_gold_labels()
        merged = predictions.merge(
            gold[["StudyInstanceUID"] + LABEL_COLUMNS],
            on="StudyInstanceUID",
            suffixes=("_pred", "_gold"),
        )
        if merged.empty:
            print("No overlap with gold 58; skipping evaluation")
        else:
            y_pred = merged[[f"{c}_pred" for c in LABEL_COLUMNS]]
            y_pred.columns = LABEL_COLUMNS
            y_true = merged[[f"{c}_gold" for c in LABEL_COLUMNS]]
            y_true.columns = LABEL_COLUMNS
            result = evaluate_labels_vs_gold(y_pred, y_true)
            print(f"Macro AUC on overlapping gold studies: {result['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
