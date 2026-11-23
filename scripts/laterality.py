"""Self-supervised laterality fix.

Studies whose left/right came from a DICOM tag or series description are trusted ("tagged"). Their stored
volumes are canonical (left knees already mirrored). We train a small CNN to tell canonical vs horizontally
mirrored coronal/axial slices, check it on held-out tagged studies (separately for L and R knees, which also
validates our flip logic), then apply it to the untagged studies and mirror those predicted "mirrored".

Writes a corrected copy of the pack: /data/pack_lat/{vols.npy,index.csv} and laterality_report.json.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

SLOTS = ["sag_fs", "sag_t1", "cor_fs", "cor_t1", "ax_fs"]
USE = [2, 3, 4]  # coronal fs, coronal t1, axial fs
SL = list(range(10, 23, 3))  # central slices


def mirror_volume(volume):
    """Mirror a packed study while preserving the input array."""
    out = np.array(volume, copy=True)
    out[0:2] = out[0:2, ::-1]
    out[2:5] = out[2:5, :, :, ::-1]
    return out


def batch(vols, masks, rows, flip, dev):
    xs, ys = [], []
    for r, fl in zip(rows, flip):
        s = next(s for s in USE if masks[r, s])
        k = np.random.choice(SL)
        img = torch.from_numpy(np.ascontiguousarray(vols[r, s, k - 1 : k + 2])).float() / 255
        if fl:
            img = img.flip(-1)
        xs.append(img)
        ys.append(float(fl))
    x = torch.stack(xs).to(dev)
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    return (x - 0.45) / 0.25, torch.tensor(ys, device=dev)


@torch.no_grad()
def score(model, vols, masks, rows, dev):
    out = []
    for r in rows:
        ims = []
        for s in USE:
            if not masks[r, s]:
                continue
            for k in SL:
                ims.append(
                    torch.from_numpy(np.ascontiguousarray(vols[r, s, k - 1 : k + 2])).float() / 255
                )
        x = F.interpolate(torch.stack(ims).to(dev), size=224, mode="bilinear", align_corners=False)
        x = (x - 0.45) / 0.25
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x).float().squeeze(1)).mean().item()
            pf = torch.sigmoid(model(x.flip(-1)).float().squeeze(1)).mean().item()
        out.append((p + (1 - pf)) / 2)  # P(mirrored), symmetrised
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/data/pack")
    ap.add_argument("--meta", default="/data/vol_meta.jsonl")
    ap.add_argument("--out", default="/data/pack_lat")
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    dev = "cuda"
    idx = pd.read_csv(Path(a.pack) / "index.csv")
    vols = np.load(Path(a.pack) / "vols.npy", mmap_mode="r")
    masks = idx[[f"mask_{s}" for s in SLOTS]].values.astype(bool)
    meta = {j["study"]: j for j in map(json.loads, open(a.meta))}
    how = idx.StudyInstanceUID.map(lambda u: meta[u].get("lat_how", ""))
    lat = idx.StudyInstanceUID.map(lambda u: meta[u].get("lat", ""))
    has = masks[:, USE].any(1)
    tagged = np.where(how.isin(["tag", "desc"]) & has)[0]
    untagged = np.where(~how.isin(["tag", "desc"]) & has)[0]
    rng = np.random.RandomState(0)
    rng.shuffle(tagged)
    n_val = len(tagged) // 5
    val, tr = tagged[:n_val], tagged[n_val:]
    print(
        f"tagged {len(tagged)} (train {len(tr)}, val {len(val)}), untagged {len(untagged)}",
        flush=True,
    )

    model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps)
    model.train()
    for it in range(a.steps):
        rows = rng.choice(tr, 64)
        flip = rng.rand(64) < 0.5
        x, y = batch(vols, masks, rows, flip, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.binary_cross_entropy_with_logits(model(x).float().squeeze(1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 250 == 0:
            print(f"it {it} loss {loss.item():.4f}", flush=True)
    model.eval()
    # validation: canonical held-out studies should score ~0, their mirrors ~1
    p_val = score(model, vols, masks, val, dev)
    acc = float((p_val < 0.5).mean())
    acc_L = float((p_val[lat.values[val] == "L"] < 0.5).mean())
    acc_R = float((p_val[lat.values[val] == "R"] < 0.5).mean())
    print(
        f"held-out tagged: acc {acc:.4f} | left-knee acc {acc_L:.4f} | right-knee acc {acc_R:.4f}",
        flush=True,
    )
    p_un = score(model, vols, masks, untagged, dev)
    flip_rows = untagged[p_un > 0.5]
    conf = float(np.mean(np.abs(p_un - 0.5) > 0.4))
    print(
        f"untagged: {len(flip_rows)}/{len(untagged)} predicted mirrored; confident(>0.9 or <0.1): {conf:.3f}",
        flush=True,
    )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(a.pack) / "index.csv", out / "index.csv")
    shutil.copy(Path(a.pack) / "vols.npy", out / "vols.npy")
    v2 = np.load(out / "vols.npy", mmap_mode="r+")
    for r in flip_rows:
        v2[r] = mirror_volume(v2[r])
    v2.flush()
    torch.save(model.state_dict(), out / "laterality_resnet18.pt")
    json.dump(
        {
            "tagged": int(len(tagged)),
            "val_acc": acc,
            "val_acc_left": acc_L,
            "val_acc_right": acc_R,
            "untagged": int(len(untagged)),
            "flipped": int(len(flip_rows)),
            "confident_frac": conf,
            "flipped_uids": idx.StudyInstanceUID.values[flip_rows].tolist(),
        },
        open(out / "laterality_report.json", "w"),
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
