#!/usr/bin/env python3
"""CLI: Train the global DINOv2 branch or a cluster-routed specialist.

Examples:

    # Train the main branch (DINOv2-base) on the full 80% split
    python scripts/train_classical.py --mode main

    # Train specialist for cluster 3 on its filtered rows
    python scripts/train_classical.py --mode specialist --cluster-id 3

    # Resume a previous run
    python scripts/train_classical.py --mode main --resume-from runs/main/best.pt

Outputs:
    <output-dir>/best.pt   — model + EMA + epoch + best_val_macro_auc
    experiments/<run>.jsonl — per-epoch metrics (via ExperimentTracker)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.classical import ClassicalTrainer, KneeMRIDataset  # noqa: E402
from src.config import (  # noqa: E402
    CLASSICAL_CHECKPOINT_DIR,
    CLASSICAL_MAX_EPOCHS,
)
from src.data.splits import load_split  # noqa: E402
from src.imaging.clustering import load_clusters, majority_cluster_per_study  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--mode",
        choices=["main", "specialist"],
        required=True,
        help="Train the global branch or one cluster specialist.",
    )
    parser.add_argument(
        "--cluster-id",
        type=int,
        default=None,
        help="Cluster ID to train (required when --mode specialist).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root of preprocessed volumes (defaults to data/processed/volumes).",
    )
    parser.add_argument(
        "--labels-csv",
        default="data/processed/labels_final.csv",
        help="Path to the labels CSV with StudyInstanceUID + 12 labels + conf cols.",
    )
    parser.add_argument(
        "--split-tag",
        default="v1",
        help="Split tag under data/processed/splits/ (default: v1).",
    )
    parser.add_argument(
        "--clusters-csv",
        default="data/processed/series_clusters.csv",
        help="Path to per-series cluster assignments CSV.",
    )
    parser.add_argument(
        "--study-clusters-csv",
        default=None,
        help="Path to per-study cluster assignments (default: derived from clusters-csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Where to write best.pt (default: {CLASSICAL_CHECKPOINT_DIR}/<run-name>).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a previous best.pt to resume training from.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=CLASSICAL_MAX_EPOCHS,
        help="Override CLASSICAL_MAX_EPOCHS for this run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (default: auto cuda/cpu).",
    )
    return parser.parse_args()


def _load_labels(labels_csv: Path) -> pd.DataFrame:
    return pd.read_csv(labels_csv)


def _build_volume_index(data_dir: Path | None):
    from src.classical.dataset import build_default_volume_index

    if data_dir is None:
        return build_default_volume_index()
    return build_default_volume_index(Path(data_dir))


def main() -> None:
    args = _parse_args()

    if args.mode == "specialist" and args.cluster_id is None:
        sys.exit("--cluster-id is required when --mode specialist")

    print(f"Loading labels from {args.labels_csv}...")
    labels_df = _load_labels(Path(args.labels_csv))

    print(f"Loading train/val split (tag={args.split_tag})...")
    train_df, val_df = load_split(args.split_tag)

    print(f"Loading cluster assignments from {args.clusters_csv}...")
    series_clusters = load_clusters(Path(args.clusters_csv))
    study_clusters = (
        pd.read_csv(args.study_clusters_csv)
        if args.study_clusters_csv
        else majority_cluster_per_study(series_clusters)
    )

    print("Loading volume index...")
    volume_index = _build_volume_index(args.data_dir)
    print(f"  loaded volumes for {len(volume_index)} studies")

    train_uids = train_df["StudyInstanceUID"].tolist()
    val_uids = val_df["StudyInstanceUID"].tolist()

    train_ds = KneeMRIDataset(train_uids, labels_df, volume_index=volume_index)
    val_ds = KneeMRIDataset(val_uids, labels_df, volume_index=volume_index)

    run_name = (
        "main"
        if args.mode == "main"
        else f"specialist_cluster_{args.cluster_id}"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else CLASSICAL_CHECKPOINT_DIR / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_kwargs = {
        "batch_size": 1,
        "max_epochs": args.max_epochs,
        "device": args.device,
        "cluster_assignments": study_clusters,
    }
    if args.resume_from:
        trainer_kwargs["resume_from"] = Path(args.resume_from)
    trainer = ClassicalTrainer(**trainer_kwargs)

    print(f"Starting {args.mode} training → {output_dir}")
    if args.mode == "main":
        result = trainer.train_main(train_ds, val_ds, output_dir)
    else:
        result = trainer.train_specialist(
            args.cluster_id, train_ds, val_ds, output_dir
        )

    print(
        f"Done. best_val_macro_auc={result.best_val_macro_auc:.4f} "
        f"checkpoint={result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
