"""Configuration constants and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

GOLD_STUDY_COUNT = 58

TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TRAIN_SERIES_CSV = RAW_DATA_DIR / "train_series.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
TEST_SERIES_CSV = RAW_DATA_DIR / "test_series.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

NOT_ADDRESSED_VALUE = 0.5
