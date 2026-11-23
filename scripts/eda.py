"""EDA on competition CSVs: labels, reports, series layout, and previous labels."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data/raw"
LABELS = [
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

CUES = {
    "es": [" el ", " la ", " de ", " sin "],
    "en": [" the ", " no ", " with ", " of "],
    "de": [" der ", " die ", " und ", " mit "],
    "fr": [" le ", " les ", " du ", " avec "],
    "pt": [" do ", " da ", " com ", " não "],
    "it": [" il ", " della ", " con ", " del "],
    "tr": [" ve ", " ile ", " bir ", " yok"],
    "el": [" και ", " του ", " της "],
    "nl": [" het ", " een ", " van ", " met "],
}


def lang(text):
    """Guess a report language using simple stopword frequencies."""
    text = " " + text.lower() + " "
    scores = {key: sum(text.count(cue) for cue in values) for key, values in CUES.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "?"


def main():
    tr = pd.read_csv(RAW / "train.csv")
    se = pd.read_csv(RAW / "train_series.csv")
    gold = tr[tr["ACL"].notna()].copy()

    print("== train.csv", tr.shape, "| gold:", len(gold), "| reports null:", tr.Report.isna().sum())
    print("\n== gold-58 positive rates")
    print(gold[LABELS].mean().round(3).to_string())

    report_lengths = tr.Report.fillna("").str.len()
    print(
        "\n== report length chars: p5/p50/p95",
        np.percentile(report_lengths, [5, 50, 95]).astype(int),
    )
    duplicate_reports = tr.Report.duplicated(keep=False).sum()
    print("duplicate report texts (rows):", duplicate_reports)

    tr["lang"] = tr.Report.fillna("").map(lang)
    print("\n== language guess\n", tr.lang.value_counts().to_string())

    print(
        "\n== series per study:",
        se.groupby("StudyInstanceUID").size().describe().round(2).to_dict(),
    )
    print("Fluid==FatSup identical:", (se.Fluid_Sensitive == se.Fat_Suppression).mean())
    combo = se.groupby(["Anatomical_Plane", "Fluid_Sensitive"]).StudyInstanceUID.nunique()
    print("studies having plane x fluid:\n", (combo / tr.shape[0]).round(3).to_string())
    duplicate_slots = se.groupby(["StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]).size()
    print("multiple series in same slot (frac of slots):", (duplicate_slots > 1).mean().round(3))

    prev = pd.read_csv(RAW / "labels_final_prev.csv").set_index("StudyInstanceUID")
    gold = gold.set_index("StudyInstanceUID")
    print(
        "\n== previous labels_final vs gold "
        "(note: prev file overrides gold, so check report-derived conf instead)"
    )
    same = (prev.loc[gold.index, LABELS].values == gold[LABELS].values).mean()
    print(
        "prev labels equal to gold on gold rows:",
        round(same, 3),
        "(1.0 => gold override, can't evaluate)",
    )
    print("\n== prev labels on non-gold: mean soft label")
    non_gold = prev.drop(index=gold.index, errors="ignore")
    print(non_gold[LABELS].mean().round(3).to_string())
    print("value 0.25 share (not-addressed?)", (non_gold[LABELS] == 0.25).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
