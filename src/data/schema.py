"""Schema validation for data files."""

import pandas as pd

from src.config import LABEL_COLUMNS


def validate_train_schema(df: pd.DataFrame) -> list[str]:
    """Validate train.csv schema. Returns list of errors."""
    errors = []

    expected_cols = ["StudyInstanceUID", "Report"] + LABEL_COLUMNS
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if "StudyInstanceUID" in df.columns:
        if df["StudyInstanceUID"].isna().any():
            errors.append("StudyInstanceUID contains null values")
        if df["StudyInstanceUID"].duplicated().any():
            errors.append("StudyInstanceUID has duplicates")

    if "Report" in df.columns:
        if df["Report"].isna().any():
            errors.append("Report contains null values")

    for col in LABEL_COLUMNS:
        if col in df.columns:
            valid_values = df[col].dropna()
            if not valid_values.isin([0, 1]).all():
                errors.append(f"Column {col} contains values other than 0, 1, or NaN")

    return errors


def validate_series_schema(df: pd.DataFrame) -> list[str]:
    """Validate train_series.csv schema. Returns list of errors."""
    errors = []

    expected_cols = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    ]
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if "StudyInstanceUID" in df.columns:
        if df["StudyInstanceUID"].isna().any():
            errors.append("StudyInstanceUID contains null values")

    if "SeriesInstanceUID" in df.columns:
        if df["SeriesInstanceUID"].isna().any():
            errors.append("SeriesInstanceUID contains null values")
        if df["SeriesInstanceUID"].duplicated().any():
            errors.append("SeriesInstanceUID has duplicates")

    if "Fluid_Sensitive" in df.columns:
        if not df["Fluid_Sensitive"].isin([0, 1]).all():
            errors.append("Fluid_Sensitive contains values other than 0 or 1")

    if "Fat_Suppression" in df.columns:
        if not df["Fat_Suppression"].isin([0, 1]).all():
            errors.append("Fat_Suppression contains values other than 0 or 1")

    if "Anatomical_Plane" in df.columns:
        valid_planes = {"Sagittal", "Coronal", "Axial"}
        actual_planes = set(df["Anatomical_Plane"].dropna().unique())
        invalid_planes = actual_planes - valid_planes
        if invalid_planes:
            errors.append(f"Invalid Anatomical_Plane values: {invalid_planes}")

    return errors


def validate_submission_schema(df: pd.DataFrame) -> list[str]:
    """Validate submission file schema. Returns list of errors."""
    errors = []

    expected_cols = ["StudyInstanceUID"] + LABEL_COLUMNS
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if "StudyInstanceUID" in df.columns:
        if df["StudyInstanceUID"].isna().any():
            errors.append("StudyInstanceUID contains null values")

    for col in LABEL_COLUMNS:
        if col in df.columns:
            if df[col].isna().any():
                errors.append(f"Column {col} contains null values")
            if (df[col] < 0).any() or (df[col] > 1).any():
                errors.append(f"Column {col} contains values outside [0, 1]")

    return errors
