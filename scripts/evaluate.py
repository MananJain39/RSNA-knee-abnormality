"""Evaluate a K-fold run (and optional ensembles of runs).

  python evaluate.py /data/exp/B1_resnet34d_pool [/data/exp/B2_... ...] --out results.md

Reports per run: OOF AUC vs LLM labels (binarised at 0.5, ~4.3k studies), gold-58 AUC per fold and for
the fold-mean ensemble, per-label gold AUC. If several runs are given, also the rank-mean ensemble across runs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

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


def aucs(y, p, binarize=None):
    out = {}
    for j, lab in enumerate(LABELS):
        t = y[:, j] if binarize is None else (y[:, j] >= binarize).astype(int)
        if 0 < t.sum() < len(t):
            out[lab] = roc_auc_score(t, p[:, j])
    return pd.Series(out, dtype=float)


def load_run(d, lab):
    folds = sorted(Path(d).glob("fold*"))
    required = ("pred_gold.npy", "oof_val.npy", "val_uids.csv", "gold_uids.csv")
    oof_u, oof_p, gold = [], [], []
    for f in folds:
        if not all((f / name).exists() for name in required):
            continue
        oof_u += pd.read_csv(f / "val_uids.csv").iloc[:, 0].tolist()
        oof_p.append(np.load(f / "oof_val.npy"))
        gu = pd.read_csv(f / "gold_uids.csv").iloc[:, 0].tolist()
        gold.append(pd.DataFrame(np.load(f / "pred_gold.npy"), index=gu, columns=LABELS))
    if not oof_p:
        raise ValueError(f"No complete folds found in {d}; required files: {', '.join(required)}")
    oof = pd.DataFrame(np.concatenate(oof_p), index=oof_u, columns=LABELS)
    return oof, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", default="/data/code/data/labels_v1.csv")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lab = pd.read_csv(a.labels).set_index("StudyInstanceUID")
    lines, ens_gold, ens_oof = [], [], []
    table = {}
    for d in a.runs:
        oof, gold = load_run(d, lab)
        y_oof = lab.loc[oof.index, LABELS].values
        oof_auc = aucs(y_oof, oof.values, binarize=0.5)
        gidx = gold[0].index
        y_g = lab.loc[gidx, LABELS].values
        per_fold = [aucs(y_g, g.loc[gidx].values).mean() for g in gold]
        gmean = sum(g.loc[gidx] for g in gold) / len(gold)
        g_auc = aucs(y_g, gmean.values)
        name = Path(d).name
        table[f"{name} gold(ens)"] = g_auc
        table[f"{name} OOF"] = oof_auc
        ens_gold.append(gmean)
        ens_oof.append(oof)
        lines.append(
            f"### {name}\n- folds: {len(gold)} | OOF studies: {len(oof)}\n"
            f"- **OOF macro AUC vs LLM labels: {oof_auc.mean():.4f}**\n"
            f"- gold-58 macro AUC per fold: {', '.join(f'{x:.4f}' for x in per_fold)} "
            f"(mean {np.mean(per_fold):.4f})\n"
            f"- **gold-58 macro AUC, {len(gold)}-fold ensemble: {g_auc.mean():.4f}**\n"
        )
    if len(a.runs) > 1:
        gidx = ens_gold[0].index
        r = sum(rankdata(g.loc[gidx].values, axis=0) for g in ens_gold)
        g_auc = aucs(lab.loc[gidx, LABELS].values, r)
        common = ens_oof[0].index
        for o in ens_oof[1:]:
            common = common.intersection(o.index)
        ro = sum(rankdata(o.loc[common].values, axis=0) for o in ens_oof)
        o_auc = aucs(lab.loc[common, LABELS].values, ro, binarize=0.5)
        table["ENSEMBLE gold"] = g_auc
        table["ENSEMBLE OOF"] = o_auc
        lines.append(
            f"### Rank-mean ensemble of {len(a.runs)} runs\n- OOF: {o_auc.mean():.4f}\n"
            f"- **gold-58: {g_auc.mean():.4f}**\n"
        )
    t = pd.DataFrame(table)
    t.loc["MACRO"] = t.mean()
    txt = "\n".join(lines) + "\n#### Per-label AUC\n" + t.round(3).to_markdown() + "\n"
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
