"""Configuration constants and paths loaded from configs/default.yaml."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

DATA_DIR = PROJECT_ROOT / "data"

_raw = _cfg["data"]["raw_dir"]
_processed = _cfg["data"]["processed_dir"]
_external = _cfg["data"]["external_dir"]

RAW_DATA_DIR = PROJECT_ROOT / _raw
PROCESSED_DATA_DIR = PROJECT_ROOT / _processed
EXTERNAL_DATA_DIR = PROJECT_ROOT / _external

LABEL_COLUMNS = _cfg["labels"]["columns"]
GOLD_STUDY_COUNT = _cfg["labels"]["gold_study_count"]
NOT_ADDRESSED_VALUE = _cfg["labels"]["not_addressed_value"]

TRAIN_CSV = PROJECT_ROOT / _cfg["data"]["train_csv"]
TRAIN_SERIES_CSV = PROJECT_ROOT / _cfg["data"]["train_series_csv"]
TEST_CSV = PROJECT_ROOT / _cfg["data"]["test_csv"]
TEST_SERIES_CSV = PROJECT_ROOT / _cfg["data"]["test_series_csv"]
SAMPLE_SUBMISSION_CSV = PROJECT_ROOT / _cfg["data"]["sample_submission_csv"]

EXTRACTION_DEFAULT_METHOD = _cfg["extraction"]["default_method"]
LLM_PROVIDER = _cfg["extraction"]["llm"]["provider"]
LLM_MODEL = _cfg["extraction"]["llm"]["model"]
LLM_TEMPERATURE = _cfg["extraction"]["llm"]["temperature"]
LLM_MAX_RETRIES = _cfg["extraction"]["llm"]["max_retries"]

VALIDATION_THRESHOLD = _cfg["validation"]["threshold"]
VALIDATION_GOLD_COMPARISON = _cfg["validation"]["gold_comparison"]

DICOM_DIR = PROJECT_ROOT / _cfg["imaging"]["dicom_dir"]
TARGET_SHAPE = tuple(_cfg["imaging"]["target_shape"])
NORMALIZATION_METHOD = _cfg["imaging"]["normalization"]

EXPERIMENTS_DIR = PROJECT_ROOT / _cfg["experiments"]["dir"]
EXPERIMENTS_FORMAT = _cfg["experiments"]["format"]
