"""Tests for src.classical.cluster_specialist routing — no torch needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.cluster_specialist import (
    filter_train_rows_for_cluster,
    specialist_router,
)


def _study_clusters() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "StudyInstanceUID": [f"s{i}" for i in range(10)],
            "cluster_id": [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
        }
    )


def _train_df(uids):
    return pd.DataFrame(
        {
            "StudyInstanceUID": uids,
            "ACL": [0.5] * len(uids),
        }
    )


def test_filter_train_rows_for_cluster_returns_only_members():
    clusters = _study_clusters()
    train_df = _train_df([f"s{i}" for i in range(10)])

    out = filter_train_rows_for_cluster(train_df, clusters, cluster_id=1)
    assert set(out["StudyInstanceUID"]) == {"s2", "s3", "s4"}


def test_filter_train_rows_for_cluster_returns_empty_when_none_in_cluster():
    clusters = _study_clusters()
    train_df = _train_df([f"s{i}" for i in range(10)])

    out = filter_train_rows_for_cluster(train_df, clusters, cluster_id=99)
    assert out.empty


def test_specialist_router_returns_cluster_id():
    clusters = _study_clusters()
    assert specialist_router("s5", clusters) == 2
    assert specialist_router("s9", clusters) == 3


def test_specialist_router_raises_for_unknown_study():
    clusters = _study_clusters()
    with pytest.raises(KeyError):
        specialist_router("ghost", clusters)
