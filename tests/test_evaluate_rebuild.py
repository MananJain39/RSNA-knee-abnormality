"""Tests for fold loading and ensemble evaluation reports."""

import importlib

import numpy as np
import pandas as pd
import pytest

evaluate = importlib.import_module("scripts.evaluate")


def _write_fold(run, fold, val_uids, val_predictions, gold_uids, gold_predictions):
    fold_path = run / fold
    fold_path.mkdir(parents=True)
    np.save(fold_path / "oof_val.npy", np.asarray(val_predictions, dtype=float))
    np.save(fold_path / "pred_gold.npy", np.asarray(gold_predictions, dtype=float))
    pd.Series(val_uids).to_csv(fold_path / "val_uids.csv", index=False)
    pd.Series(gold_uids).to_csv(fold_path / "gold_uids.csv", index=False)


def _predictions(values):
    return np.tile(np.asarray(values, dtype=float)[:, None], (1, len(evaluate.LABELS)))


def test_aucs_skips_single_class_labels():
    y = np.zeros((2, len(evaluate.LABELS)))
    y[:, 0] = [0, 1]
    predictions = _predictions([0.1, 0.9])

    scores = evaluate.aucs(y, predictions)

    assert scores.to_dict() == {"ACL": 1.0}


def test_load_run_combines_complete_folds(tmp_path):
    run = tmp_path / "run"
    gold_uids = ["gold-0", "gold-1"]
    _write_fold(run, "fold0", ["study-0"], [[0.1] * 12], gold_uids, _predictions([0.1, 0.9]))
    _write_fold(run, "fold1", ["study-1"], [[0.9] * 12], gold_uids, _predictions([0.2, 0.8]))
    (run / "fold2").mkdir()
    (run / "fold2" / "pred_gold.npy").touch()

    oof, gold = evaluate.load_run(run, None)

    assert list(oof.index) == ["study-0", "study-1"]
    assert oof.shape == (2, len(evaluate.LABELS))
    assert len(gold) == 2
    assert gold[0].iloc[0, 0] == 0.1
    assert gold[1].iloc[1, 0] == 0.8


def test_load_run_reports_missing_complete_folds(tmp_path):
    (tmp_path / "fold0").mkdir()

    with pytest.raises(ValueError, match="No complete folds"):
        evaluate.load_run(tmp_path, None)


def test_main_writes_single_and_rank_ensemble_metrics(tmp_path, monkeypatch):
    labels_path = tmp_path / "labels.csv"
    val_uids = [f"study-{index}" for index in range(4)]
    gold_uids = ["gold-0", "gold-1"]
    uids = val_uids + gold_uids
    labels = pd.DataFrame(
        {
            "StudyInstanceUID": uids,
            **{label: [0, 0, 1, 1, 0, 1] for label in evaluate.LABELS},
        }
    )
    labels.to_csv(labels_path, index=False)
    runs = []
    for index, offset in enumerate([0.0, 0.05]):
        run = tmp_path / f"run-{index}"
        _write_fold(
            run,
            "fold0",
            val_uids,
            _predictions([0.1 + offset, 0.2, 0.8, 0.9 - offset]),
            gold_uids,
            _predictions([0.1 + offset, 0.9 - offset]),
        )
        runs.append(run)
    output = tmp_path / "results.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            *map(str, runs),
            "--labels",
            str(labels_path),
            "--out",
            str(output),
        ],
    )

    evaluate.main()

    report = output.read_text()
    assert "Rank-mean ensemble of 2 runs" in report
    assert "ENSEMBLE gold" in report
    assert "ENSEMBLE OOF" in report
    assert "gold-58 macro AUC" in report
