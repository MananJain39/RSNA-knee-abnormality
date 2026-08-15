"""Slot builder for organizing MRI series into plane × contrast slots."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SLOT_DEFINITIONS = {
    "Sagittal_Fluid": {"plane": "Sagittal", "fluid": 1},
    "Sagittal_Structural": {"plane": "Sagittal", "fluid": 0},
    "Coronal_Fluid": {"plane": "Coronal", "fluid": 1},
    "Coronal_Structural": {"plane": "Coronal", "fluid": 0},
    "Axial_Fluid": {"plane": "Axial", "fluid": 1},
}

SLOT_NAMES = list(SLOT_DEFINITIONS.keys())


@dataclass
class SlotInfo:
    slot_name: str
    series_uid: str
    anatomical_plane: str
    fluid_sensitive: int
    n_slices: int


def assign_slot(plane: str, fluid_sensitive: int) -> str | None:
    for slot_name, defn in SLOT_DEFINITIONS.items():
        if defn["plane"] == plane and defn["fluid"] == fluid_sensitive:
            return slot_name
    return None


def build_slots_for_study(
    study_uid: str,
    series_df: pd.DataFrame,
) -> dict[str, SlotInfo | None]:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]

    slots = {slot_name: None for slot_name in SLOT_NAMES}

    for _, row in study_series.iterrows():
        slot_name = assign_slot(row["Anatomical_Plane"], int(row["Fluid_Sensitive"]))
        if slot_name is not None and slots[slot_name] is None:
            slots[slot_name] = SlotInfo(
                slot_name=slot_name,
                series_uid=row["SeriesInstanceUID"],
                anatomical_plane=row["Anatomical_Plane"],
                fluid_sensitive=int(row["Fluid_Sensitive"]),
                n_slices=0,
            )

    return slots


def compute_slot_fill_rates(
    series_df: pd.DataFrame,
) -> dict[str, float]:
    study_uids = series_df["StudyInstanceUID"].unique()
    n_studies = len(study_uids)

    slot_counts = {slot_name: 0 for slot_name in SLOT_NAMES}

    for study_uid in study_uids:
        slots = build_slots_for_study(study_uid, series_df)
        for slot_name, slot_info in slots.items():
            if slot_info is not None:
                slot_counts[slot_name] += 1

    return {slot_name: count / n_studies for slot_name, count in slot_counts.items()}


def get_slot_for_label(label: str) -> list[str]:
    label_slot_map = {
        "ACL": ["Sagittal_Fluid", "Sagittal_Structural", "Axial_Fluid"],
        "MCL": ["Coronal_Fluid", "Coronal_Structural"],
        "Medial Meniscus": [
            "Sagittal_Fluid", "Sagittal_Structural", "Coronal_Fluid", "Coronal_Structural",
        ],
        "Lateral Meniscus": ["Sagittal_Fluid", "Sagittal_Structural", "Coronal_Fluid"],
        "Medial OA": ["Coronal_Fluid", "Coronal_Structural", "Sagittal_Structural"],
        "Lateral OA": ["Coronal_Fluid", "Coronal_Structural", "Sagittal_Structural"],
        "PF OA": ["Sagittal_Fluid", "Axial_Fluid", "Sagittal_Structural"],
        "Effusion": ["Sagittal_Fluid", "Axial_Fluid"],
        "Synovitis": ["Sagittal_Fluid", "Axial_Fluid"],
        "Baker's": ["Sagittal_Fluid"],
        "Contusion": ["Sagittal_Fluid", "Coronal_Fluid", "Axial_Fluid"],
        "Fracture": SLOT_NAMES,
    }
    return label_slot_map.get(label, SLOT_NAMES)


def build_slot_mask(slots: dict[str, SlotInfo | None]) -> np.ndarray:
    return np.array([1 if slots[name] is not None else 0 for name in SLOT_NAMES])
