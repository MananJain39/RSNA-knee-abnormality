"""Tests for src.imaging.clustering.

These tests run on synthetic metadata only — they do not require real DICOM
data.  The full DICOM-driven clustering path is exercised on Kaggle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.imaging.clustering import (
    CLUSTER_FEATURE_COLS,
    _syntax_family,
    build_feature_matrix,
    cluster_series,
    gather_series_metadata,
    load_clusters,
    majority_cluster_per_study,
    persist_clusters,
)


def _synthetic_metadata(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.choice([128, 256, 384, 512], size=n)
    pixel_spacings = rng.choice([0.3, 0.4, 0.5, 0.7], size=n)
    fluid = rng.integers(0, 2, size=n)
    fat = rng.integers(0, 2, size=n)
    n_slices = rng.integers(10, 60, size=n)
    slice_thickness = rng.choice([2.0, 3.0, 4.0, 5.0], size=n)
    families = rng.choice(
        ["explicit_vr", "jpeg_lossless", "jpeg2000"], size=n
    )
    return pd.DataFrame(
        {
            "StudyInstanceUID": [f"study_{i // 3}" for i in range(n)],
            "SeriesInstanceUID": [f"series_{i}" for i in range(n)],
            "pixel_spacing_mm": pixel_spacings,
            "rows": sizes,
            "cols": sizes,
            "n_slices": n_slices,
            "slice_thickness_mm": slice_thickness,
            "fov_extent_mm": pixel_spacings * sizes,
            "fluid_sensitive": fluid,
            "fat_suppression": fat,
            "transfer_syntax_family": families,
        }
    )


def test_syntax_family_buckets():
    assert _syntax_family("1.2.840.10008.1.2.1") == "explicit_vr"
    assert _syntax_family("1.2.840.10008.1.2") == "implicit_vr"
    assert _syntax_family("1.2.840.10008.1.2.4.70") == "jpeg_lossless"
    assert _syntax_family("1.2.840.10008.1.2.4.50") == "jpeg_baseline"
    assert _syntax_family("1.2.840.10008.1.2.4.90") == "jpeg2000"
    assert _syntax_family("1.2.840.10008.1.99.99") == "other"
    assert _syntax_family(None) == "unknown"


def test_build_feature_matrix_returns_scaled_numpy():
    md = _synthetic_metadata(50)
    X, used_cols = build_feature_matrix(md)
    assert X.shape[0] == 50
    assert X.shape[1] >= len(CLUSTER_FEATURE_COLS)
    assert all(c.startswith("syntax_") or c in CLUSTER_FEATURE_COLS for c in used_cols)


def test_cluster_series_kmeans_assigns_to_each_bucket():
    md = _synthetic_metadata(200)
    out = cluster_series(md, k=4, method="kmeans", seed=42)
    assert "cluster_id" in out.columns
    assert out["cluster_id"].nunique() <= 4
    assert set(out["cluster_id"].unique()) <= {0, 1, 2, 3}


def test_cluster_series_gmm_returns_probabilities():
    md = _synthetic_metadata(200)
    out = cluster_series(md, k=3, method="gmm", seed=42)
    assert "cluster_id" in out.columns
    assert "cluster_prob" in out.columns
    assert (out["cluster_prob"] > 0).all()
    assert (out["cluster_prob"] <= 1.0).all()


def test_cluster_series_rejects_bad_k():
    md = _synthetic_metadata(50)
    with pytest.raises(ValueError):
        cluster_series(md, k=1, method="kmeans")


def test_cluster_series_rejects_unknown_method():
    md = _synthetic_metadata(50)
    with pytest.raises(ValueError):
        cluster_series(md, k=4, method="foo")


def test_majority_cluster_per_study_aggregates():
    md = _synthetic_metadata(60)
    clustered = cluster_series(md, k=4, seed=42)
    study_cluster = majority_cluster_per_study(clustered)

    assert "StudyInstanceUID" in study_cluster.columns
    assert "cluster_id" in study_cluster.columns
    assert study_cluster["cluster_id"].nunique() >= 1
    assert len(study_cluster) <= clustered["StudyInstanceUID"].nunique()


def test_persist_and_load_roundtrip(tmp_path):
    md = _synthetic_metadata(40)
    clustered = cluster_series(md, k=3, seed=42)
    path = persist_clusters(clustered, output_path=tmp_path / "clusters.csv")
    assert path.exists()

    loaded = load_clusters(path)
    assert "cluster_id" in loaded.columns
    assert len(loaded) == 40


def test_gather_series_metadata_returns_empty_when_dicom_missing(tmp_path):
    series_df = pd.DataFrame(
        {
            "StudyInstanceUID": ["ghost_study"],
            "SeriesInstanceUID": ["ghost_series"],
            "Fluid_Sensitive": [1],
            "Fat_Suppression": [0],
        }
    )
    out = gather_series_metadata(series_df, dicom_dir=tmp_path)
    assert out.empty
