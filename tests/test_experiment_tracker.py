"""Tests for the JSONL experiment tracker."""

import json

from src.experiments.tracker import ExperimentTracker


def test_tracker_saves_and_loads_complete_run(tmp_path):
    tracker = ExperimentTracker("training", experiments_dir=tmp_path)
    tracker.log_params({"learning_rate": 0.001})
    tracker.log_metrics({"macro_auc": 0.75})

    tracker.save()

    runs = ExperimentTracker.load_runs("training", experiments_dir=tmp_path)
    assert len(runs) == 1
    assert runs[0]["run_id"] == tracker.run_id
    assert runs[0]["experiment"] == "training"
    assert runs[0]["params"] == {"learning_rate": 0.001}
    assert runs[0]["metrics"] == {"macro_auc": 0.75}
    assert runs[0]["elapsed_seconds"] >= 0
    assert runs[0]["timestamp"]


def test_tracker_appends_structured_step_entries(tmp_path):
    tracker = ExperimentTracker("steps", experiments_dir=tmp_path)

    tracker.log_step(1, {"loss": 0.8})
    tracker.log_step(2, {"loss": 0.4})

    lines = (tmp_path / "steps.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    assert [entry["step"] for entry in entries] == [1, 2]
    assert [entry["metrics"]["loss"] for entry in entries] == [0.8, 0.4]
    assert all(entry["run_id"] == tracker.run_id for entry in entries)


def test_tracker_selects_best_run_in_either_direction(tmp_path):
    for run_id, score in [("run-a", 0.6), ("run-b", 0.9), ("run-c", 0.7)]:
        tracker = ExperimentTracker("validation", experiments_dir=tmp_path)
        tracker.run_id = run_id
        tracker.log_metrics({"macro_auc": score, "loss": 1.0 - score})
        tracker.save()

    best_auc = ExperimentTracker.get_best_run(
        "validation", "macro_auc", experiments_dir=tmp_path
    )
    best_loss = ExperimentTracker.get_best_run(
        "validation", "loss", maximize=False, experiments_dir=tmp_path
    )

    assert best_auc["run_id"] == "run-b"
    assert best_loss["run_id"] == "run-b"


def test_tracker_summarizes_completed_runs(tmp_path):
    for run_id, score in [("run-a", 0.6), ("run-b", 0.9)]:
        tracker = ExperimentTracker("validation", experiments_dir=tmp_path)
        tracker.run_id = run_id
        tracker.log_metrics({"macro_auc": score})
        tracker.save()

    summaries = ExperimentTracker.summary(experiments_dir=tmp_path)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["experiment"] == "validation"
    assert summary["n_runs"] == 2
    assert summary["last_run_id"] == "run-b"
    assert summary["last_timestamp"]
    assert summary["metrics"] == {"macro_auc": 0.9}


def test_tracker_handles_missing_and_empty_logs(tmp_path):
    assert ExperimentTracker.load_runs("missing", experiments_dir=tmp_path) == []
    assert ExperimentTracker.get_best_run("missing", "auc", experiments_dir=tmp_path) is None
    assert ExperimentTracker.summary(experiments_dir=tmp_path) == []

    (tmp_path / "empty.jsonl").touch()
    assert ExperimentTracker.load_runs("empty", experiments_dir=tmp_path) == []
    assert ExperimentTracker.summary(experiments_dir=tmp_path) == []
