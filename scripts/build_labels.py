"""Merge LLM states (Sonnet primary, Haiku secondary) into soft training labels.

Soft value per state is ordinal; Sonnet and Haiku values are averaged with weight w.
Synovitis 'not_mentioned' is imputed from the Effusion score (silence is uninformative
for synovitis: P(gold+|silent)=0.31 on gold-58), per community finding.
Evaluated on gold-58 (gold labels are never used to fit anything here except reading the table).
Writes data/labels_v1.csv (all 4407 studies; gold rows keep gold 0/1 + is_gold flag).
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from llm_labels import LABELS  # noqa: E402
from score_labels import load_states, to_soft  # noqa: E402

E = ROOT / "exp/01_labels"


def combine(son, hai, w=0.7, impute_syn=True):
    s, h = to_soft(son), to_soft(hai.reindex(son.index).fillna("not_mentioned"))
    out = w * s + (1 - w) * h
    if impute_syn:
        silent = son["Synovitis"] == "not_mentioned"
        out.loc[silent, "Synovitis"] = 0.15 + 0.5 * out.loc[silent, "Effusion"]
    return out


def gold_auc(soft, gold):
    idx = gold.index.intersection(soft.index)
    return pd.Series(
        {label: roc_auc_score(gold.loc[idx, label], soft.loc[idx, label]) for label in LABELS}
    )


def restore_gold(soft, gold):
    """Return labels with available gold rows restored to binary values."""
    out = soft.copy()
    out["is_gold"] = out.index.isin(gold.index).astype(int)
    overlap = out.index.intersection(gold.index)
    out.loc[overlap, LABELS] = gold.loc[overlap, LABELS].values
    return out


def main():
    tr = pd.read_csv(ROOT / "data/raw/train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][LABELS].astype(int)
    son_g, hai_g = load_states(E / "raw_sonnet_gold.jsonl"), load_states(E / "raw_haiku_gold.jsonl")
    rows = {}
    for w in [1.0, 0.8, 0.7, 0.5]:
        for imp in [False, True]:
            a = gold_auc(combine(son_g, hai_g, w, imp), gold)
            rows[f"w={w} impSyn={imp}"] = a
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string())
    print(tab.mean().round(4).to_string())

    son = pd.concat([son_g, load_states(E / "raw_sonnet_nongold.jsonl")])
    hai = pd.concat([hai_g, load_states(E / "raw_haiku_nongold.jsonl")])
    soft = combine(son, hai, 0.7, True).reindex(tr.index)
    print("missing studies:", soft.isna().any(axis=1).sum())
    soft = restore_gold(soft, gold)
    soft.index.name = "StudyInstanceUID"
    soft.to_csv(ROOT / "data/labels_v1.csv")
    print(
        "non-gold mean soft label:\n", soft[soft.is_gold == 0][LABELS].mean().round(3).to_string()
    )


if __name__ == "__main__":
    main()
