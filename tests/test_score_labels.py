"""Tests for mapping and scoring closed-vocabulary LLM labels."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("anthropic")
score_labels = importlib.import_module("scripts.score_labels")


def _states(index, value="not_mentioned"):
    return pd.DataFrame(
        {label: [value] * len(index) for label in score_labels.LABELS}, index=index
    )


def _gold(index, positive=False):
    return pd.DataFrame(
        int(positive), index=index, columns=score_labels.LABELS, dtype=int
    )


def test_load_states_skips_errors_and_preserves_label_order(tmp_path):
    path = tmp_path / "states.jsonl"
    rows = [
        {"uid": "study-1", "states": _states(["study-1"]).iloc[0].to_dict()},
        {"uid": "study-2", "error": "refusal"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    states = score_labels.load_states(path)

    assert list(states.index) == ["study-1"]
    assert list(states.columns) == score_labels.LABELS
    assert states.loc["study-1", "ACL"] == "not_mentioned"


def test_load_states_returns_empty_schema_when_no_rows_succeed(tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text(json.dumps({"uid": "study-1", "error": "refusal"}) + "\n")

    states = score_labels.load_states(path)

    assert states.empty
    assert list(states.columns) == score_labels.LABELS
    assert score_labels.to_soft(states).empty


def test_to_soft_uses_defaults_and_per_label_overrides():
    states = _states(["study-1", "study-2"])
    states["ACL"] = ["negated", "positive"]
    states["MCL"] = ["negated", "positive"]

    soft = score_labels.to_soft(states, {"ACL": {"negated": 0.2, "positive": 0.8}})

    assert soft["ACL"].tolist() == [0.2, 0.8]
    assert soft["MCL"].tolist() == [0.02, 0.95]
    assert soft["Effusion"].tolist() == [0.1, 0.1]


def test_to_soft_marks_unknown_states_as_missing():
    states = _states(["study-1"])
    states.loc["study-1", "ACL"] = "unexpected"

    soft = score_labels.to_soft(states)

    assert np.isnan(soft.loc["study-1", "ACL"])
    assert soft.loc["study-1", "MCL"] == 0.1


def test_score_computes_auc_and_positive_cut_accuracy():
    index = ["study-1", "study-2", "study-3", "study-4"]
    states = _states(index)
    pattern = ["negated", "not_mentioned", "equivocal", "positive"]
    for label in score_labels.LABELS:
        states[label] = pattern
    gold = _gold(index)
    for label in score_labels.LABELS:
        gold[label] = [0, 0, 1, 1]

    metrics, count = score_labels.score(states, gold)

    assert count == 4
    assert np.allclose(metrics["auc"], 1.0)
    assert np.allclose(metrics["acc_pos"], 0.75)


def test_score_uses_custom_mapping_order():
    states = _states(["study-1", "study-2"])
    for label in score_labels.LABELS:
        states[label] = ["negated", "positive"]
    gold = _gold(states.index)
    for label in score_labels.LABELS:
        gold[label] = [0, 1]

    metrics, _ = score_labels.score(
        states, gold, {"ACL": {"negated": 1.0, "positive": 0.0}}
    )

    assert metrics.loc["ACL", "auc"] == 0.0
    assert metrics.loc["MCL", "auc"] == 1.0


def test_score_returns_nan_auc_for_one_class_gold():
    states = _states(["study-1", "study-2"])

    metrics, count = score_labels.score(states, _gold(states.index))

    assert count == 2
    assert metrics["auc"].isna().all()
    assert (metrics["acc_pos"] == 1.0).all()


def test_score_handles_no_overlapping_studies():
    states = _states(["study-1"])
    gold = _gold(["study-2"])

    metrics, count = score_labels.score(states, gold)

    assert count == 0
    assert metrics.isna().all().all()
