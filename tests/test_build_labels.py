"""Tests for blending LLM labels and restoring expert annotations."""

import importlib

import pandas as pd
import pytest

pytest.importorskip("anthropic")
build_labels = importlib.import_module("scripts.build_labels")


def _states(index, value="not_mentioned"):
    return pd.DataFrame(
        {label: [value] * len(index) for label in build_labels.LABELS}, index=index
    )


@pytest.mark.parametrize(
    ("weight", "expected"), [(0.0, 0.02), (0.7, 0.671), (1.0, 0.95)]
)
def test_combine_blends_primary_and_secondary_weights(weight, expected):
    son = _states(["study-1"])
    hai = _states(["study-1"])
    son.loc["study-1", "ACL"] = "positive"
    hai.loc["study-1", "ACL"] = "negated"

    combined = build_labels.combine(son, hai, w=weight, impute_syn=False)

    assert combined.loc["study-1", "ACL"] == pytest.approx(expected)
    assert combined.loc["study-1", "MCL"] == pytest.approx(0.1)
    assert list(combined.columns) == build_labels.LABELS


def test_combine_fills_missing_secondary_studies_with_not_mentioned():
    son = _states(["study-1", "study-2"], value="positive")
    hai = _states(["study-1"], value="negated")

    combined = build_labels.combine(son, hai, w=0.0, impute_syn=False)

    assert combined.loc["study-1", "ACL"] == pytest.approx(0.02)
    assert combined.loc["study-2", "ACL"] == pytest.approx(0.1)
    assert combined.loc["study-2", "Effusion"] == pytest.approx(0.1)


def test_combine_imputes_only_silent_primary_synovitis():
    son = _states(["study-1", "study-2"])
    hai = _states(son.index)
    son.loc["study-1", "Effusion"] = "positive"
    son.loc["study-2", "Synovitis"] = "positive"

    combined = build_labels.combine(son, hai, w=1.0, impute_syn=True)

    assert combined.loc["study-1", "Effusion"] == pytest.approx(0.95)
    assert combined.loc["study-1", "Synovitis"] == pytest.approx(0.625)
    assert combined.loc["study-2", "Synovitis"] == pytest.approx(0.95)


def test_combine_can_disable_synovitis_imputation():
    son = _states(["study-1"])
    hai = _states(["study-1"])

    combined = build_labels.combine(son, hai, w=1.0, impute_syn=False)

    assert combined.loc["study-1", "Synovitis"] == pytest.approx(0.1)


def test_restore_gold_overwrites_overlap_without_mutating_input():
    soft = pd.DataFrame(
        0.4, index=["study-1", "study-2"], columns=build_labels.LABELS
    )
    gold = pd.DataFrame(
        1, index=["study-1", "study-3"], columns=build_labels.LABELS
    )

    restored = build_labels.restore_gold(soft, gold)

    assert restored.loc["study-1", build_labels.LABELS].eq(1).all()
    assert restored.loc["study-2", build_labels.LABELS].eq(0.4).all()
    assert restored["is_gold"].tolist() == [1, 0]
    assert "study-3" not in restored.index
    assert "is_gold" not in soft
    assert soft.loc["study-1", "ACL"] == 0.4


def test_gold_auc_returns_scores_in_label_order():
    soft = pd.DataFrame(
        {label: [0.1, 0.9] for label in build_labels.LABELS},
        index=["study-1", "study-2"],
    )
    gold = pd.DataFrame(
        {label: [0, 1] for label in build_labels.LABELS}, index=soft.index
    )

    scores = build_labels.gold_auc(soft, gold)

    assert list(scores.index) == build_labels.LABELS
    assert scores.eq(1.0).all()
