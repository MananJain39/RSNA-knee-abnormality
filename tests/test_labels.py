"""Tests for label extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels.regex_extractor import RegexExtractor
from src.config import LABEL_COLUMNS, NOT_ADDRESSED_VALUE


def test_regex_extractor_init():
    extractor = RegexExtractor()
    assert extractor is not None


def test_regex_extractor_empty_report():
    extractor = RegexExtractor()
    labels = extractor.extract("")
    assert len(labels) == 12
    assert all(v == NOT_ADDRESSED_VALUE for v in labels.values())


def test_regex_extractor_none_report():
    extractor = RegexExtractor()
    labels = extractor.extract(None)
    assert len(labels) == 12


def test_regex_extractor_english_positive():
    extractor = RegexExtractor()

    report = "MRI of the knee shows ACL tear with bone marrow edema."
    labels = extractor.extract(report)
    assert labels["ACL"] == 1.0
    assert labels["Contusion"] == 1.0


def test_regex_extractor_english_negative():
    extractor = RegexExtractor()

    report = "ACL is intact and normal. No effusion seen."
    labels = extractor.extract(report)
    assert labels["ACL"] == 0.0
    assert labels["Effusion"] == 0.0


def test_regex_extractor_spanish():
    extractor = RegexExtractor()

    report = "Rotura de menisco interno. Derrame articular."
    labels = extractor.extract(report)
    assert labels["Medial Meniscus"] == 1.0
    assert labels["Effusion"] == 1.0


def test_regex_extractor_french():
    extractor = RegexExtractor()

    report = "Déchirure du ménisque médial. Épanchement articulaire."
    labels = extractor.extract(report)
    assert labels["Medial Meniscus"] == 1.0
    assert labels["Effusion"] == 1.0


def test_regex_extractor_german():
    extractor = RegexExtractor()

    report = "Riss des Meniskus medial. Gelenkerguss."
    labels = extractor.extract(report)
    assert labels["Medial Meniscus"] == 1.0
    assert labels["Effusion"] == 1.0


def test_regex_extractor_batch():
    extractor = RegexExtractor()
    reports = [
        "ACL tear.",
        "Normal knee.",
        "Meniscus tear lateral.",
    ]
    labels_list = extractor.extract_batch(reports)
    assert len(labels_list) == 3
    assert labels_list[0]["ACL"] == 1.0
    assert labels_list[2]["Lateral Meniscus"] == 1.0


def test_regex_extractor_all_labels():
    """Verify all 12 labels are returned."""
    extractor = RegexExtractor()
    labels = extractor.extract("Some report text.")
    assert set(labels.keys()) == set(LABEL_COLUMNS)


def test_regex_extractor_not_addressed():
    """Labels not mentioned should be 0.5."""
    extractor = RegexExtractor()
    report = "The knee shows no abnormalities."
    labels = extractor.extract(report)
    for col in LABEL_COLUMNS:
        assert labels[col] in (0.0, 0.5, 1.0)
