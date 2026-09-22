"""Tests for reliability-weighted label fusion."""

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMNS
from src.labels.fusion import (
    apply_gold_override,
    fuse_labels,
    per_label_reliability,
)


def _labels_df(values: dict[str, list[float]]) -> pd.DataFrame:
    """Build a 12-label DataFrame; unspecified labels default to 0.5."""
    n = len(next(iter(values.values())))
    df = pd.DataFrame({col: [0.5] * n for col in LABEL_COLUMNS})
    for col, vals in values.items():
        df[col] = vals
    df.insert(0, "StudyInstanceUID", [f"s{i}" for i in range(n)])
    return df


def _two_sources():
    # source_a is right on ACL, weak on Effusion; source_b the reverse.
    a = _labels_df({"ACL": [0.0, 1.0, 1.0], "Effusion": [1.0, 0.0, 0.0]})
    b = _labels_df({"ACL": [1.0, 0.0, 0.0], "Effusion": [1.0, 1.0, 0.0]})
    return a, b


def _gold():
    return _labels_df({"ACL": [0.0, 1.0, 1.0], "Effusion": [1.0, 1.0, 0.0]})


def test_per_label_reliability_direction():
    a, b = _two_sources()
    rel = per_label_reliability({"a": a, "b": b}, _gold())
    assert rel.loc["a", "ACL"] > rel.loc["b", "ACL"]
    assert rel.loc["b", "Effusion"] > rel.loc["a", "Effusion"]


def test_fusion_is_symmetric_under_renaming():
    # Regression guard: a top-team fusion bug truncated one source's values
    # with int(), making a confident positive unable to override a negative.
    # Renaming the sources must not change the fused output.
    a, b = _two_sources()
    fused_ab = fuse_labels(
        {"a": a, "b": b}, per_label_reliability({"a": a, "b": b}, _gold())
    )
    fused_ba = fuse_labels(
        {"a": b, "b": a}, per_label_reliability({"a": b, "b": a}, _gold())
    )
    for col in LABEL_COLUMNS:
        assert np.allclose(fused_ab[col], fused_ba[col]), f"asymmetric for {col}"


def test_fusion_snaps_when_hard_votes_agree():
    a = _labels_df({"ACL": [0.9, 0.1]})
    b = _labels_df({"ACL": [0.7, 0.2]})
    rel = pd.DataFrame(
        [[0.9] * len(LABEL_COLUMNS)] * 2, index=["a", "b"], columns=LABEL_COLUMNS
    )
    fused = fuse_labels({"a": a, "b": b}, rel)
    assert fused["ACL"].iloc[0] == 1.0
    assert fused["ACL"].iloc[1] == 0.0


def test_fusion_single_source_keeps_graded_value():
    a = _labels_df({"ACL": [0.6, 0.3]})
    rel = pd.DataFrame([[0.9] * len(LABEL_COLUMNS)], index=["a"], columns=LABEL_COLUMNS)
    fused = fuse_labels({"a": a}, rel)
    assert abs(fused["ACL"].iloc[0] - 0.6) < 1e-6
    assert abs(fused["ACL"].iloc[1] - 0.3) < 1e-6


def test_fusion_missing_cell_abstains():
    a = _labels_df({"ACL": [0.0, 1.0]})
    b = _labels_df({"ACL": [1.0, 1.0]})
    a.loc[0, "ACL"] = np.nan  # source a abstains on s0

    rel = pd.DataFrame(
        [[0.9] * len(LABEL_COLUMNS)] * 2, index=["a", "b"], columns=LABEL_COLUMNS
    )
    fused = fuse_labels({"a": a, "b": b}, rel)
    assert abs(fused["ACL"].iloc[0] - 1.0) < 1e-6  # only b voted: keep graded 1.0
    assert fused["ACL"].iloc[1] == 1.0  # both present and agree: hard snap


def test_gold_override_replaces_matched_rows():
    fused = _labels_df({"ACL": [0.9, 0.1]})
    gold = _labels_df({"ACL": [0.0, 1.0]})
    overridden = apply_gold_override(fused, gold)
    assert overridden["ACL"].iloc[0] == 0.0
    assert overridden["ACL"].iloc[1] == 1.0
