"""Score closed-vocab LLM states against gold-58 and map states -> soft labels.

AUC per label uses the ordinal state rank (rank-only metric, so mapping values
only matter through their order). Also reports accuracy at the 'positive' cut and
the empirical P(gold+ | state) table that we use to set the soft-label mapping.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from llm_labels import LABELS, STATES  # noqa: E402

# default ordinal mapping: negated < not_mentioned < below < equivocal < positive
DEFAULT_MAP = {
    "negated": 0.02,
    "not_mentioned": 0.10,
    "below_threshold": 0.30,
    "equivocal": 0.60,
    "positive": 0.95,
}


def load_states(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            j = json.loads(line)
            if "error" in j:
                continue
            rows[j["uid"]] = j["states"]
    if not rows:
        return pd.DataFrame(columns=LABELS, dtype=object)
    return pd.DataFrame.from_dict(rows, orient="index")[LABELS]


def to_soft(states, mapping=None):
    mapping = mapping or {}
    out = pd.DataFrame(index=states.index)
    for lab in LABELS:
        m = {**DEFAULT_MAP, **mapping.get(lab, {})}
        out[lab] = states[lab].map(m).astype(float)
    return out


def _binary_auc(y, p):
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def score(states, gold, mapping=None):
    soft = to_soft(states, mapping)
    idx = gold.index.intersection(soft.index)
    res = {}
    for lab in LABELS:
        y, p = gold.loc[idx, lab].values, soft.loc[idx, lab].values
        acc_pos = float("nan")
        if len(idx):
            acc_pos = float(((states.loc[idx, lab] == "positive") == (y == 1)).mean())
        res[lab] = {"auc": _binary_auc(y, p), "acc_pos": acc_pos}
    df = pd.DataFrame(res).T
    return df, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/raw/train.csv")
    gold = tr[tr.ACL.notna()].set_index("StudyInstanceUID")[LABELS].astype(int)
    summary = {}
    for f in a.files:
        st = load_states(f)
        df, n = score(st, gold)
        summary[Path(f).stem] = df.auc
        print(
            f"\n=== {Path(f).stem}  n={n}  macroAUC={df.auc.mean():.4f}  mean acc@positive={df.acc_pos.mean():.3f}"
        )
        print(df.round(3).to_string())
        # P(gold+ | state) table
        idx = gold.index.intersection(st.index)
        tab = {}
        for lab in LABELS:
            s, y = st.loc[idx, lab], gold.loc[idx, lab]
            tab[lab] = {
                k: f"{y[s == k].mean():.2f}/{(s == k).sum()}" if (s == k).any() else "-"
                for k in STATES
            }
        print("P(gold+|state)/count:\n" + pd.DataFrame(tab).T.to_string())
    print("\n=== macro AUC summary")
    print(pd.DataFrame(summary).round(3).to_string())
    print(pd.DataFrame(summary).mean().round(4).to_string())


if __name__ == "__main__":
    main()
