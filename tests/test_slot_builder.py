"""Tests for slot builder."""

import numpy as np
import pandas as pd

from src.imaging.slot_builder import (
    SLOT_NAMES,
    assign_slot,
    build_slot_mask,
    build_slots_for_study,
    compute_slot_fill_rates,
    get_slot_for_label,
)


def _make_series_df(rows):
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    return pd.DataFrame(rows, columns=cols)


def test_assign_slot_sagittal_fluid():
    assert assign_slot("Sagittal", 1) == "Sagittal_Fluid"


def test_assign_slot_coronal_structural():
    assert assign_slot("Coronal", 0) == "Coronal_Structural"


def test_assign_slot_axial_fluid():
    assert assign_slot("Axial", 1) == "Axial_Fluid"


def test_assign_slot_axial_structural_returns_none():
    assert assign_slot("Axial", 0) is None


def test_build_slots_for_study():
    df = _make_series_df([
        ("study1", "s1", "Sagittal", 1),
        ("study1", "s2", "Sagittal", 0),
        ("study1", "s3", "Coronal", 1),
        ("study1", "s4", "Coronal", 0),
        ("study1", "s5", "Axial", 1),
    ])

    slots = build_slots_for_study("study1", df)
    assert slots["Sagittal_Fluid"] is not None
    assert slots["Sagittal_Fluid"].series_uid == "s1"
    assert slots["Sagittal_Structural"] is not None
    assert slots["Coronal_Fluid"] is not None
    assert slots["Coronal_Structural"] is not None
    assert slots["Axial_Fluid"] is not None


def test_build_slots_missing_slot():
    df = _make_series_df([
        ("study1", "s1", "Sagittal", 1),
    ])

    slots = build_slots_for_study("study1", df)
    assert slots["Sagittal_Fluid"] is not None
    assert slots["Sagittal_Structural"] is None
    assert slots["Coronal_Fluid"] is None


def test_build_slots_duplicate_plane_contrast():
    df = _make_series_df([
        ("study1", "s1", "Sagittal", 1),
        ("study1", "s2", "Sagittal", 1),
    ])

    slots = build_slots_for_study("study1", df)
    assert slots["Sagittal_Fluid"].series_uid == "s1"


def test_compute_slot_fill_rates():
    df = _make_series_df([
        ("study1", "s1", "Sagittal", 1),
        ("study1", "s2", "Sagittal", 0),
        ("study1", "s3", "Coronal", 1),
        ("study1", "s4", "Coronal", 0),
        ("study1", "s5", "Axial", 1),
        ("study2", "s6", "Sagittal", 1),
        ("study2", "s7", "Axial", 1),
    ])

    rates = compute_slot_fill_rates(df)
    assert rates["Sagittal_Fluid"] == 1.0
    assert rates["Sagittal_Structural"] == 0.5
    assert rates["Axial_Fluid"] == 1.0
    assert rates["Coronal_Structural"] == 0.5


def test_get_slot_for_label_acl():
    slots = get_slot_for_label("ACL")
    assert "Sagittal_Fluid" in slots
    assert "Sagittal_Structural" in slots
    assert "Axial_Fluid" in slots


def test_get_slot_for_label_bakers():
    slots = get_slot_for_label("Baker's")
    assert slots == ["Sagittal_Fluid"]


def test_get_slot_for_label_fracture():
    slots = get_slot_for_label("Fracture")
    assert slots == SLOT_NAMES


def test_get_slot_for_label_unknown():
    slots = get_slot_for_label("Unknown")
    assert slots == SLOT_NAMES


def test_build_slot_mask():
    from src.imaging.slot_builder import SlotInfo

    slots = {
        "Sagittal_Fluid": SlotInfo("Sagittal_Fluid", "s1", "Sagittal", 1, 30),
        "Sagittal_Structural": None,
        "Coronal_Fluid": SlotInfo("Coronal_Fluid", "s3", "Coronal", 1, 25),
        "Coronal_Structural": None,
        "Axial_Fluid": SlotInfo("Axial_Fluid", "s5", "Axial", 1, 20),
    }

    mask = build_slot_mask(slots)
    np.testing.assert_array_equal(mask, [1, 0, 1, 0, 1])
