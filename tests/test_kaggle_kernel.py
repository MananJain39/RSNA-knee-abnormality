"""Tests for the import-safe offline Kaggle inference entry point."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "kaggle/kernel/infer_kernel.py"
SPEC = importlib.util.spec_from_file_location("knee_infer_kernel", KERNEL_PATH)
kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kernel)


def _competition(root):
    path = root / "competitions" / kernel.COMPETITION
    path.mkdir(parents=True)
    return path


def _models(root, owner="test-owner"):
    path = root / "datasets" / owner / "rsna-knee-claude-models"
    path.mkdir(parents=True)
    return path


def test_discover_inputs_supports_nested_competition_and_dataset_paths(tmp_path):
    competition = _competition(tmp_path)
    models = _models(tmp_path)

    assert kernel.discover_inputs(tmp_path) == (competition, models)


def test_discover_inputs_rejects_missing_competition(tmp_path):
    _models(tmp_path)

    with pytest.raises(RuntimeError, match="Expected one competition input"):
        kernel.discover_inputs(tmp_path)


def test_discover_inputs_rejects_ambiguous_model_datasets(tmp_path):
    _competition(tmp_path)
    _models(tmp_path, "owner-a")
    _models(tmp_path, "owner-b")

    with pytest.raises(RuntimeError, match="Expected one model dataset, found 2"):
        kernel.discover_inputs(tmp_path)


def test_discover_checkpoints_returns_sorted_fold_paths(tmp_path):
    for relative in ["models/run-b/fold1/best.pt", "models/run-a/fold0/best.pt"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.touch()

    checkpoints = kernel.discover_checkpoints(tmp_path)

    assert checkpoints == [
        str(tmp_path / "models/run-a/fold0/best.pt"),
        str(tmp_path / "models/run-b/fold1/best.pt"),
    ]


def test_discover_checkpoints_rejects_empty_model_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="No fold checkpoints"):
        kernel.discover_checkpoints(tmp_path)


def test_build_command_connects_inputs_checkpoints_and_output(tmp_path):
    competition = tmp_path / "competition"
    models = tmp_path / "models"
    checkpoints = ["fold0.pt", "fold1.pt"]
    output = tmp_path / "submission.csv"

    command = kernel.build_command(competition, models, checkpoints, output, 3)

    assert command[1] == str(models / "scripts/infer.py")
    checkpoint_index = command.index("--ckpts")
    assert command[checkpoint_index + 1 : checkpoint_index + 3] == checkpoints
    assert command[command.index("--series-root") + 1] == str(competition / "test_series")
    assert command[command.index("--series-csv") + 1] == str(competition / "test_series.csv")
    assert command[command.index("--test-csv") + 1] == str(competition / "test.csv")
    assert command[command.index("--out") + 1] == str(output)
    assert command[command.index("--workers") + 1] == "3"


def test_kaggle_metadata_is_offline_and_references_existing_files():
    kernel_metadata = json.loads((ROOT / "kaggle/kernel/kernel-metadata.json").read_text())
    dataset_metadata = json.loads((ROOT / "kaggle/models/dataset-metadata.json").read_text())

    assert kernel_metadata["enable_internet"] is False
    assert kernel_metadata["enable_gpu"] is True
    assert (KERNEL_PATH.parent / kernel_metadata["code_file"]).exists()
    assert kernel_metadata["competition_sources"] == [kernel.COMPETITION]
    assert kernel_metadata["dataset_sources"] == [dataset_metadata["id"]]
