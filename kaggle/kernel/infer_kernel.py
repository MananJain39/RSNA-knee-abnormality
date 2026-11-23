"""Offline Kaggle inference entry point; code and weights come from a dataset."""

import glob
import os
import subprocess
import sys
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
MODEL_PATTERN = "*knee-claude-models*"


def _one_match(patterns, description):
    matches = sorted({path for pattern in patterns for path in glob.glob(str(pattern))})
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {description}, found {len(matches)}: {matches}")
    return Path(matches[0])


def discover_inputs(input_root="/kaggle/input"):
    root = Path(input_root)
    competition = _one_match(
        [root / "competitions" / COMPETITION, root / COMPETITION],
        "competition input",
    )
    models = _one_match(
        [root / MODEL_PATTERN, root / "datasets" / "*" / MODEL_PATTERN],
        "model dataset",
    )
    return competition, models


def discover_checkpoints(model_dataset):
    checkpoints = sorted(glob.glob(str(Path(model_dataset) / "models/*/fold*/best.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found under {model_dataset}")
    return checkpoints


def build_command(competition, model_dataset, checkpoints, output, workers):
    return [
        sys.executable,
        str(Path(model_dataset) / "scripts/infer.py"),
        "--series-root",
        str(Path(competition) / "test_series"),
        "--series-csv",
        str(Path(competition) / "test_series.csv"),
        "--test-csv",
        str(Path(competition) / "test.csv"),
        "--ckpts",
        *checkpoints,
        "--out",
        str(output),
        "--workers",
        str(workers),
    ]


def main():
    competition, models = discover_inputs()
    print(f"comp: {competition} | models: {models}", flush=True)
    print(sorted(os.listdir(competition)), flush=True)
    checkpoints = discover_checkpoints(models)
    print(len(checkpoints), "checkpoints", checkpoints, flush=True)
    output = Path("/kaggle/working/submission.csv")
    command = build_command(competition, models, checkpoints, output, os.cpu_count() or 1)
    subprocess.run(command, check=True)

    import pandas as pd  # noqa: PLC0415

    submission = pd.read_csv(output)
    print(submission.shape, submission.describe().T[["mean", "std"]], flush=True)


if __name__ == "__main__":
    main()
