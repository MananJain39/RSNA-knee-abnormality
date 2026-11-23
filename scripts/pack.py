"""Pack per-study npz volumes into one uint8 memmap (N, 5, 32, 256, 256) + index.csv.

Missing slots are zero-filled and flagged in mask_<slot>. With --fake N, writes random volumes
for the first N labelled studies (for smoke tests before real data exists).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = ["sag_fs", "sag_t1", "cor_fs", "cor_t1", "ax_fs"]
D, H = 32, 256


def select_fake_uids(labels, count):
    """Select at most 10 gold studies and fill the remainder with non-gold."""
    if count < 0:
        raise ValueError("fake study count must be non-negative")
    gold_count = min(10, count)
    gold = list(labels[labels.is_gold == 1].StudyInstanceUID[:gold_count])
    non_gold = list(labels[labels.is_gold == 0].StudyInstanceUID[: count - len(gold)])
    return gold + non_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-dir", default="/data/vol")
    ap.add_argument("--out", default="/data/pack")
    ap.add_argument("--labels", default="/data/code/data/labels_v1.csv")
    ap.add_argument("--fake", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.fake:
        lab = pd.read_csv(a.labels)
        uids = select_fake_uids(lab, a.fake)
    else:
        uids = sorted(p.stem for p in Path(a.vol_dir).glob("*.npz"))
    N = len(uids)
    vols = np.lib.format.open_memmap(
        out / "vols.npy", mode="w+", dtype=np.uint8, shape=(N, len(SLOTS), D, H, H)
    )
    masks = np.zeros((N, len(SLOTS)), bool)

    def load(i):
        if a.fake:
            rng = np.random.default_rng(i)
            for s in range(len(SLOTS)):
                if s == 3 and i % 4 == 0:
                    continue
                vols[i, s] = rng.integers(0, 255, (D, H, H), dtype=np.uint8)
                masks[i, s] = True
            return
        z = np.load(Path(a.vol_dir) / f"{uids[i]}.npz")
        for s, name in enumerate(SLOTS):
            if name in z.files:
                vols[i, s] = z[name]
                masks[i, s] = True

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(load, range(N)))
    vols.flush()
    idx = pd.DataFrame({"row": range(N), "StudyInstanceUID": uids})
    for s, name in enumerate(SLOTS):
        idx[f"mask_{name}"] = masks[:, s].astype(int)
    idx.to_csv(out / "index.csv", index=False)
    print(N, "studies packed; slot fill:", masks.mean(0).round(3).tolist())


if __name__ == "__main__":
    main()
