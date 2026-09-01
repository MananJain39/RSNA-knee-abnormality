"""Per-cluster specialist model and cluster-aware dataset filter.

A ``ClusterSpecialist`` is a small DINOv2 model trained **only** on rows whose
``cluster_id`` matches the specialist's id.  At inference time, the
``specialist_router`` decides which specialist gets the study.

This module deliberately reuses ``DINOv2Backbone`` and ``PerDiagnosisHead``
from ``dinov2_base.py`` so the resulting logits are in the same output space
and can be blended with the main branch via ``ensemble.rank_mean_logits``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.config import CLASSICAL_SPECIALIST_BACKBONE
from src.imaging.clustering import load_clusters, majority_cluster_per_study

if TYPE_CHECKING:
    from src.classical.dinov2_base import KneeModel


def make_specialist(cluster_id: int, image_size: int) -> KneeModel:
    """Build a fresh, small specialist for a single cluster.

    The specialist uses the smaller DINOv2 backbone (``dinov2-small`` by
    default) to keep the cluster-level training cost in check; with K≈3-5
    clusters the total param count stays well below the main branch.
    """
    # Keep routing/filter helpers torch-free; load the optional model stack only
    # when a caller actually requests a specialist instance.
    from src.classical.dinov2_base import KneeModel

    return KneeModel(
        backbone_name=CLASSICAL_SPECIALIST_BACKBONE,
        image_size=image_size,
        freeze_backbone=True,
    )


def filter_train_rows_for_cluster(
    train_df: pd.DataFrame,
    cluster_assignments: pd.DataFrame | None = None,
    cluster_id: int = 0,
) -> pd.DataFrame:
    """Return the subset of training studies assigned to ``cluster_id``.

    Args:
        train_df: Studies-level training DataFrame (must include
            ``StudyInstanceUID``).
        cluster_assignments: DataFrame from ``load_clusters`` /
            ``majority_cluster_per_study``. If None, the function falls back
            to loading from the default path on disk.
        cluster_id: Which cluster to keep.

    Returns:
        Filtered copy of ``train_df`` containing only studies that landed in
        the requested cluster.
    """
    if cluster_assignments is None:
        study_clusters = majority_cluster_per_study(load_clusters())
    else:
        if "cluster_id" not in cluster_assignments.columns:
            study_clusters = majority_cluster_per_study(cluster_assignments)
        else:
            study_clusters = cluster_assignments

    uids_in_cluster = set(
        study_clusters.loc[
            study_clusters["cluster_id"] == cluster_id, "StudyInstanceUID"
        ]
    )
    return train_df[train_df["StudyInstanceUID"].isin(uids_in_cluster)].reset_index(
        drop=True
    )


def specialist_router(study_uid: str, study_clusters: pd.DataFrame) -> int:
    """Look up the cluster_id for a single study; raises if not present."""
    rows = study_clusters[study_clusters["StudyInstanceUID"] == study_uid]
    if rows.empty:
        raise KeyError(f"StudyUID {study_uid} not in cluster assignments")
    if rows["cluster_id"].nunique() > 1:
        raise ValueError(
            f"StudyUID {study_uid} appears in multiple clusters: "
            f"{rows['cluster_id'].unique().tolist()}"
        )
    return int(rows["cluster_id"].iloc[0])


__all__ = [
    "make_specialist",
    "filter_train_rows_for_cluster",
    "specialist_router",
]
