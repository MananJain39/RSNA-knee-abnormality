"""Cluster MRI series by raw acquisition features.

The motivation comes from the hierarchical-clustering plan (``plan.txt``):
studies acquired with very different physics (PixelSpacing varying 3.4×,
slice counts 3-14, different transfer syntaxes) are routed into separate
specialist branches so each branch sees a more homogeneous input distribution
and adds less noise during training.

This module produces ``data/processed/series_clusters.csv`` with columns
``StudyInstanceUID, SeriesInstanceUID, cluster_id, …features``.  The clustering
is done once per corpus pass; downstream code joins on ``SeriesInstanceUID``.

Two operating modes:

* **Metadata mode** (default): use features from a precomputed DICOM-metadata
  table (the preferred path on Kaggle, after running
  ``notebooks/01_dicom_preprocessing.ipynb`` to populate the table).
* **Direct mode**: extract features on the fly by reading DICOM headers; slow
  but works when no metadata table is available yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURE_COLS = [
    "pixel_spacing_mm",
    "rows",
    "cols",
    "n_slices",
    "slice_thickness_mm",
    "fov_extent_mm",
    "fluid_sensitive",
    "fat_suppression",
    "transfer_syntax_family",
]


def load_series_metadata_table(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the flat per-series metadata table produced by DICOM preprocessing.

    The table must contain ``SeriesInstanceUID`` and the ``CLUSTER_FEATURE_COLS``
    columns.  Use ``gather_series_metadata`` to build this table from the
    raw DICOM tree.
    """
    if path is None:
        path = Path("data/processed/series_metadata.csv")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Series metadata table not found at {path}. "
            "Run the DICOM preprocessing notebook first."
        )
    return pd.read_csv(path)


def gather_series_metadata(
    series_df: pd.DataFrame,
    dicom_dir: Path | None = None,
    n_studies: int | None = None,
) -> pd.DataFrame:
    """Read DICOM headers for each series and assemble a flat feature table.

    Args:
        series_df: DataFrame with ``StudyInstanceUID`` and ``SeriesInstanceUID``
            (matches ``src.data.loader.load_train_series``).
        dicom_dir: Root directory containing ``<study>/<series>/*.dcm``.
            Defaults to ``src.config.DICOM_DIR``.
        n_studies: Optional cap for smoke-testing — process only this many
            unique studies.

    Returns:
        DataFrame indexed on ``SeriesInstanceUID`` with columns from
        ``CLUSTER_FEATURE_COLS``.
    """
    from src.imaging.dicom_loader import get_series_metadata

    if dicom_dir is None:
        from src.config import DICOM_DIR

        dicom_dir = DICOM_DIR

    if n_studies is not None:
        studies = series_df["StudyInstanceUID"].unique()[:n_studies]
        series_df = series_df[series_df["StudyInstanceUID"].isin(studies)]

    rows = []
    for study_uid, series_uid in zip(
        series_df["StudyInstanceUID"].values, series_df["SeriesInstanceUID"].values
    ):
        try:
            meta = get_series_metadata(study_uid, series_uid, data_dir=dicom_dir)
        except (FileNotFoundError, AttributeError, ValueError):
            continue

        rows.append(
            {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "pixel_spacing_mm": float(meta["pixel_spacing"][0]),
                "rows": int(meta["rows"]),
                "cols": int(meta["columns"]),
                "n_slices": int(meta["n_slices"]),
                "slice_thickness_mm": float(meta["slice_thickness"]),
                "fov_extent_mm": float(meta["pixel_spacing"][0])
                * int(meta["rows"]),
                "fluid_sensitive": int(series_df.loc[
                    (series_df["StudyInstanceUID"] == study_uid)
                    & (series_df["SeriesInstanceUID"] == series_uid),
                    "Fluid_Sensitive",
                ].iloc[0]),
                "fat_suppression": int(series_df.loc[
                    (series_df["StudyInstanceUID"] == study_uid)
                    & (series_df["SeriesInstanceUID"] == series_uid),
                    "Fat_Suppression",
                ].iloc[0]),
                "transfer_syntax_family": _syntax_family(meta["transfer_syntax"]),
            }
        )

    return pd.DataFrame(rows)


def _syntax_family(transfer_syntax_uid: str) -> str:
    """Bucket transfer-syntax UIDs into one of a few families.

    Order matters: more specific patterns must come first because
    ``1.2.840.10008.1.2.4.70`` also contains the substring ``1.2.840.10008.1.2``.
    """
    if transfer_syntax_uid is None:
        return "unknown"
    s = str(transfer_syntax_uid).upper()
    if "1.2.840.10008.1.2.4.90" in s or "1.2.840.10008.1.2.4.91" in s:
        return "jpeg2000"
    if "1.2.840.10008.1.2.4.70" in s:
        return "jpeg_lossless"
    if "1.2.840.10008.1.2.4.50" in s or "1.2.840.10008.1.2.4.51" in s:
        return "jpeg_baseline"
    if "1.2.840.10008.1.2.1" in s:
        return "explicit_vr"
    if "1.2.840.10008.1.2" in s:
        return "implicit_vr"
    return "other"


def build_feature_matrix(metadata_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Materialize a numeric feature matrix ready for clustering.

    Returns:
        (X, used_cols) where ``X`` is shape ``(n_series, n_features)`` and
        ``used_cols`` lists which CLUSTER_FEATURE_COLS ended up in the matrix
        (stringy columns are one-hot encoded; their one-hot names appear here).
    """
    work = metadata_df.copy()
    family_dummies = pd.get_dummies(work["transfer_syntax_family"], prefix="syntax")
    work = pd.concat([work.drop(columns=["transfer_syntax_family"]), family_dummies], axis=1)

    used_cols = [c for c in CLUSTER_FEATURE_COLS if c in work.columns]
    other_cols = [c for c in work.columns if c not in used_cols and c not in {
        "StudyInstanceUID", "SeriesInstanceUID"
    }]
    used_cols = used_cols + other_cols

    X = work[used_cols].astype(float).to_numpy()
    return X, used_cols


def cluster_series(
    metadata_df: pd.DataFrame,
    k: int = 4,
    method: str = "kmeans",
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster each row (a series) into one of ``k`` buckets.

    Args:
        metadata_df: Output of ``load_series_metadata_table`` or
            ``gather_series_metadata``.
        k: Number of clusters.
        method: Either ``"kmeans"`` or ``"gmm"`` (Gaussian Mixture, soft
            assignments).
        seed: Random seed.

    Returns:
        The input DataFrame plus a ``cluster_id`` column. For GMM the column
        ``cluster_prob`` is also added (max component probability).
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    X, used_cols = build_feature_matrix(metadata_df)
    X_scaled = StandardScaler().fit_transform(X)

    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(X_scaled)
        out = metadata_df.copy()
        out["cluster_id"] = labels
    elif method == "gmm":
        model = GaussianMixture(n_components=k, random_state=seed, n_init=3)
        labels = model.fit(X_scaled).predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        out = metadata_df.copy()
        out["cluster_id"] = labels
        out["cluster_prob"] = probs.max(axis=1)
    else:
        raise ValueError(f"Unknown method '{method}', use 'kmeans' or 'gmm'")

    out.attrs["used_cols"] = used_cols
    out.attrs["model"] = model
    return out


def majority_cluster_per_study(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate series-level clusters to a study-level ``cluster_id``.

    Each study is assigned the cluster that contains the most of its series.
    The 12 specialists can then read this column for routing.
    """
    counts = (
        cluster_df.groupby(["StudyInstanceUID", "cluster_id"])
        .size()
        .reset_index(name="n")
    )
    idx = counts.groupby("StudyInstanceUID")["n"].idxmax()
    return counts.loc[idx].reset_index(drop=True)[["StudyInstanceUID", "cluster_id"]]


def persist_clusters(
    cluster_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Write the per-series cluster assignments to disk."""
    if output_path is None:
        from src.config import PROCESSED_DATA_DIR

        output_path = PROCESSED_DATA_DIR / "series_clusters.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(output_path, index=False)
    return output_path


def load_clusters(path: Path | None = None) -> pd.DataFrame:
    """Read per-series cluster assignments produced by ``persist_clusters``."""
    if path is None:
        from src.config import PROCESSED_DATA_DIR

        path = PROCESSED_DATA_DIR / "series_clusters.csv"
    return pd.read_csv(Path(path))
