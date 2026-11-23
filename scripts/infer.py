"""Inference for Kaggle: DICOM test series -> submission.csv, same preprocessing as training.

python infer.py --series-root /kaggle/input/.../test_series --series-csv .../test_series.csv \
    --test-csv .../test.csv --ckpts /kaggle/input/knee-models/*.pt --out submission.csv
Each checkpoint dir must have args.json next to it (backbone, n_slices, crop).
Failed studies get 0.5 everywhere; submission is rewritten every 25 studies (crash safety).
"""

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as P  # noqa: E402
from train import LABELS, SLOTS, GpuBatcher, KneeNet  # noqa: E402


def discover_checkpoints(patterns):
    checkpoints = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not checkpoints:
        raise FileNotFoundError("No checkpoints matched the supplied patterns")
    return checkpoints


def write_submission(uids, predictions, out):
    """Write predictions in the exact order of the supplied study UIDs."""
    frame = pd.DataFrame(
        [[uid, *predictions[uid]] for uid in uids],
        columns=["StudyInstanceUID", *LABELS],
    )
    frame.to_csv(out, index=False)


def prep(args):
    study, rows, root = args
    try:
        cand = {}
        for r in rows:
            key = (r["Anatomical_Plane"], int(r["Fluid_Sensitive"]))
            if key not in P.SLOTS:
                continue
            files = P.list_series(root, study, r["SeriesInstanceUID"])
            if len(files) >= 3 and (key not in cand or len(files) > len(cand[key])):
                cand[key] = files
        arrays, metas = {}, []
        for key, files in cand.items():
            try:
                v, m = P.series_volume(P.read_series(files), key[0])
                arrays[P.SLOTS[key]] = v
                metas.append(m)
            except Exception:
                pass
        lat, _ = P.laterality(metas) if metas else ("R", "")
        vol = np.zeros((len(SLOTS), P.DEPTH, P.SIZE, P.SIZE), np.uint8)
        mask = np.zeros(len(SLOTS), bool)
        for name, v in arrays.items():
            u = P.to_uint8(v)
            if lat == "L":
                u = u[::-1] if name.startswith("sag") else u[:, :, ::-1]
            vol[SLOTS.index(name)] = P.resample_depth(u)
            mask[SLOTS.index(name)] = True
        return study, vol, mask
    except Exception:
        return study, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", required=True)
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ckpts = discover_checkpoints(a.ckpts)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    global AMP
    AMP = (
        torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    )  # T4 -> fp16
    models = []
    for c in ckpts:
        cfg = json.load(open(Path(c).parent / "args.json"))
        m = KneeNet(cfg["backbone"], pretrained=False, head=cfg.get("head", "attn"))
        m.load_state_dict(torch.load(c, map_location="cpu"))
        models.append((m.to(dev).eval(), cfg))
    print(f"{len(models)} models on {dev}", flush=True)
    test = pd.read_csv(a.test_csv)
    se = pd.read_csv(a.series_csv)
    groups = {s: g.to_dict("records") for s, g in se.groupby("StudyInstanceUID")}
    preds = {u: np.full(len(LABELS), 0.5) for u in test.StudyInstanceUID}

    def write():
        write_submission(test.StudyInstanceUID, preds, a.out)

    write()
    t0 = time.time()
    jobs = [(u, groups.get(u, []), a.series_root) for u in test.StudyInstanceUID]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (u, vol, mask) in enumerate(ex.map(prep, jobs), 1):
            if vol is not None and mask.any():
                ps = []
                for m, cfg in models:
                    b = GpuBatcher(vol[None], mask[None], cfg["n_slices"], cfg["crop"], dev)
                    x, mm, pos = b.get([0], train=False)
                    with torch.no_grad(), torch.autocast(dev, dtype=AMP, enabled=dev == "cuda"):
                        ps.append(torch.sigmoid(m(x, mm, pos).float()).cpu().numpy()[0])
                preds[u] = np.mean(ps, 0)
            if i % 25 == 0:
                write()
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    write()
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
