"""Download public LLM label datasets from Kaggle."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"

DATASETS = [
    "pilkwang/rsna-knee-llm-labels",
    "stevenleehans/rsna-knee-llm-report-labels",
    "barun2104/rsna-knee-stratified-folds-and-llm-soft-labels",
]


def download_dataset(dataset_ref: str, output_dir: Path) -> None:
    slug = dataset_ref.replace("/", "_")
    dest = output_dir / slug
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_ref} -> {dest}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(dest), "--unzip"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
    else:
        files = list(dest.iterdir())
        print(f"  OK: {len(files)} file(s) extracted")


def main() -> None:
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        download_dataset(dataset, EXTERNAL_DATA_DIR)

    print("\nAll datasets downloaded.")
    print(f"Location: {EXTERNAL_DATA_DIR}")


if __name__ == "__main__":
    main()
