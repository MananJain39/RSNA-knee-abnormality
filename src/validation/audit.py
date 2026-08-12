"""Audit report-label consistency and identify discrepancies."""

import pandas as pd

from src.config import LABEL_COLUMNS
from src.data.loader import load_train


def audit_report_label_agreement(
    train_df: pd.DataFrame = None,
    sample_size: int = 20,
) -> pd.DataFrame:
    """Audit agreement between reports and provided labels for gold studies.

    This helps identify potential inconsistencies in the labeling.

    Args:
        train_df: Training DataFrame (loaded if None)
        sample_size: Number of gold studies to audit

    Returns:
        DataFrame with audit results per study
    """
    if train_df is None:
        train_df = load_train()

    gold_mask = train_df[LABEL_COLUMNS[0]].notna()
    gold_df = train_df[gold_mask].copy()

    if len(gold_df) > sample_size:
        gold_df = gold_df.sample(sample_size, random_state=42)

    results = []
    for _, row in gold_df.iterrows():
        report = row["Report"]
        report_lower = report.lower() if report else ""

        study_result = {
            "StudyInstanceUID": row["StudyInstanceUID"],
            "Report_Length": len(report) if report else 0,
        }

        for col in LABEL_COLUMNS:
            label_val = row[col]
            study_result[f"{col}_Label"] = label_val

            if label_val == 1.0:
                study_result[f"{col}_Mentioned"] = _check_mention(col, report_lower, positive=True)
            elif label_val == 0.0:
                study_result[f"{col}_Mentioned"] = _check_mention(col, report_lower, positive=False)
            else:
                study_result[f"{col}_Mentioned"] = None

        results.append(study_result)

    return pd.DataFrame(results)


def _check_mention(label: str, report_lower: str, positive: bool) -> bool:
    """Check if a label is mentioned in the report."""
    label_keywords = {
        "ACL": ["acl", "anterior cruciate", "ligamento cruzado anterior"],
        "MCL": ["mcl", "medial collateral", "ligamento colateral medial"],
        "Medial Meniscus": ["medial meniscus", "menisco medial", "menisco interno"],
        "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "menisco externo"],
        "Medial OA": ["medial oa", "medial arthrosis", "artrosis medial"],
        "Lateral OA": ["lateral oa", "lateral arthrosis", "artrosis lateral"],
        "PF OA": ["patellofemoral", "patelofemoral", "retropatellar"],
        "Effusion": ["effusion", "derrame", "ergot", "fluid"],
        "Synovitis": ["synovitis", "sinovitis", "synovite"],
        "Baker's": ["baker", "popliteal cyst", "quiste poplíteo"],
        "Contusion": ["contusion", "bone bruise", "marrow edema", "contusión"],
        "Fracture": ["fracture", "fractura"],
    }

    keywords = label_keywords.get(label, [])
    return any(kw in report_lower for kw in keywords)


def find_labeling_discrepancies(
    train_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Find studies where labels may contradict report text.

    Returns DataFrame of potential discrepancies for manual review.
    """
    if train_df is None:
        train_df = load_train()

    gold_mask = train_df[LABEL_COLUMNS[0]].notna()
    gold_df = train_df[gold_mask].copy()

    discrepancies = []
    for _, row in gold_df.iterrows():
        report = row["Report"].lower() if row["Report"] else ""

        for col in LABEL_COLUMNS:
            label_val = row[col]

            if label_val == 1.0:
                mentioned = _check_mention(col, report, positive=True)
                if not mentioned:
                    discrepancies.append({
                        "StudyInstanceUID": row["StudyInstanceUID"],
                        "Label": col,
                        "Label_Value": label_val,
                        "Issue": "Label=1 but finding not clearly mentioned in report",
                        "Report_Preview": row["Report"][:200] if row["Report"] else "",
                    })
            elif label_val == 0.0:
                negated = _check_negation(col, report)
                if not negated:
                    pass

    return pd.DataFrame(discrepancies)


def _check_negation(label: str, report_lower: str) -> bool:
    """Check if a label is explicitly negated in the report."""
    negation_patterns = {
        "ACL": ["no acl tear", "acl intact", "acl normal"],
        "MCL": ["no mcl tear", "mcl intact", "mcl normal"],
        "Effusion": ["no effusion", "without effusion", "sin derrame"],
    }

    patterns = negation_patterns.get(label, [])
    return any(pattern in report_lower for pattern in patterns)


def compute_not_addressed_rate(
    predicted_df: pd.DataFrame,
    not_addressed_value: float = 0.5,
) -> pd.Series:
    """Compute the rate of "not addressed" (0.5) predictions per label.

    Args:
        predicted_df: DataFrame with predicted labels
        not_addressed_value: Value indicating "not addressed"

    Returns:
        Series with not-addressed rate per label
    """
    rates = {}
    for col in LABEL_COLUMNS:
        if col in predicted_df.columns:
            na_count = (predicted_df[col] == not_addressed_value).sum()
            rates[col] = na_count / len(predicted_df)

    return pd.Series(rates).sort_values(ascending=False)
