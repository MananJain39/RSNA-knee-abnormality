"""2.5D CNN + slice/series transformer + per-label attention head, multi-label knee MRI.

Data: /data/pack/vols.npy  uint8 (N, S=5, D=32, 256, 256) memmap, /data/pack/index.csv (row -> uid, slot mask)
Labels: labels csv (soft 0..1 for non-gold, 0/1 gold) with is_gold.
CV: K folds over non-gold studies (grouped by duplicate report text); gold-58 never trained on,
scored by every fold model (fold-mean = ensemble estimate, per-fold = single-model estimate).

Usage:
  python train.py --fold 0 --folds 4 --backbone tf_efficientnetv2_b0 --epochs 10 --out /data/exp/A1
  python train.py --bench --backbone ...      # synthetic throughput test
  python train.py --smoke ...                 # 5 train studies, 1 epoch, end-to-end
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
SLOTS = ["sag_fs", "sag_t1", "cor_fs", "cor_t1", "ax_fs"]


# ---------------------------------------------------------------- model
class KneeNet(nn.Module):
    def __init__(
        self,
        backbone,
        n_slots=5,
        max_d=32,
        dim=256,
        layers=2,
        pretrained=True,
        drop_path=0.1,
        head="pool",
    ):
        super().__init__()
        self.head = head
        self.enc = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3, drop_path_rate=drop_path
        )
        c = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(c, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, dim))
        self.pos_emb = nn.Parameter(torch.zeros(max_d, dim))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        el = nn.TransformerEncoderLayer(
            dim, 8, dim * 4, dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.tf = nn.TransformerEncoder(el, layers)
        self.queries = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, 8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(len(LABELS)))
        # "pool" head: per-series masked max+mean over slices -> concat series -> linear
        self.pool_fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_slots * 2 * dim, len(LABELS)))

    def forward(self, x, slot_mask, pos_idx):
        # x: (B, S, D, 3, H, W) float; slot_mask: (B, S) bool; pos_idx: (B, S, D) long
        B, S, D = x.shape[:3]
        feats = torch.zeros(B, S, D, self.proj[0].out_features, device=x.device, dtype=x.dtype)
        sel = slot_mask.nonzero(as_tuple=False)  # only encode present slots
        xs = x[sel[:, 0], sel[:, 1]].flatten(0, 1)  # (n*D, 3, H, W)
        f = self.proj(self.enc(xs)).view(len(sel), D, -1)
        feats[sel[:, 0], sel[:, 1]] = f.to(feats.dtype)
        if self.head == "pool":
            mx = feats.amax(2) * slot_mask[..., None]
            mn = feats.mean(2) * slot_mask[..., None]
            return self.pool_fc(torch.cat([mx, mn], -1).flatten(1))
        tok = feats + self.slot_emb[None, :S, None] + self.pos_emb[pos_idx]
        tok = tok.flatten(1, 2)  # (B, S*D, dim)
        pad = (~slot_mask)[:, :, None].expand(B, S, D).flatten(1)
        tok = self.tf(tok, src_key_padding_mask=pad)
        q = self.queries[None].expand(B, -1, -1)
        o, _ = self.attn(q, tok, tok, key_padding_mask=pad)
        o = self.norm(o) if self.head == "attn2" else self.norm(o + q)
        return (o * self.head_w).sum(-1) + self.head_b  # (B, 12)


# ---------------------------------------------------------------- data on GPU
class GpuBatcher:
    """Reads uint8 volumes from a memmap, does 2.5D stacking + augmentation on GPU."""

    def __init__(self, vols, masks, n_slices, crop, device):
        self.vols, self.masks, self.n, self.crop, self.dev = vols, masks, n_slices, crop, device

    def get(self, rows, train):
        D = self.vols.shape[2]
        arr = torch.from_numpy(np.stack([self.vols[r] for r in rows])).to(
            self.dev, non_blocking=True
        )
        m = torch.from_numpy(self.masks[rows]).to(self.dev)
        B, S = arr.shape[:2]
        # choose slice centres: stride sampling over the full depth, random offset in train
        step = D / self.n
        off = (
            torch.rand(B, S, 1, device=self.dev) * step
            if train
            else torch.full((B, S, 1), step / 2, device=self.dev)
        )
        centre = (
            (off + torch.arange(self.n, device=self.dev) * step).long().clamp(0, D - 1)
        )  # (B,S,n)
        nb = torch.stack(
            [(centre - 1).clamp(0, D - 1), centre, (centre + 1).clamp(0, D - 1)], -1
        )  # (B,S,n,3)
        bi = torch.arange(B, device=self.dev)[:, None, None, None]
        si = torch.arange(S, device=self.dev)[None, :, None, None]
        x = arr[bi, si, nb].float() / 255.0  # (B,S,n,3,H,W)
        H = x.shape[-1]
        x = x.flatten(0, 2)  # (B*S*n, 3, H, W)
        if train:
            # per-series affine (same for all slices in a series): scale, rotation, shift
            g = B * S
            ang = (torch.rand(g, device=self.dev) - 0.5) * 2 * math.radians(12)
            sc = 1 + (torch.rand(g, device=self.dev) - 0.5) * 0.25
            tx, ty = [(torch.rand(g, device=self.dev) - 0.5) * 0.16 for _ in range(2)]
            cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
            theta = torch.stack(
                [torch.stack([cos, -sin, tx], -1), torch.stack([sin, cos, ty], -1)], 1
            )
            theta = theta.repeat_interleave(self.n, 0)
            grid = F.affine_grid(theta, (x.shape[0], 3, self.crop, self.crop), align_corners=False)
            grid = grid * (self.crop / H)  # sample a crop-sized window
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            # intensity: gamma + contrast + brightness per series
            gam = torch.exp((torch.rand(g, device=self.dev) - 0.5) * 0.5).repeat_interleave(self.n)[
                :, None, None, None
            ]
            con = (1 + (torch.rand(g, device=self.dev) - 0.5) * 0.3).repeat_interleave(self.n)[
                :, None, None, None
            ]
            bri = ((torch.rand(g, device=self.dev) - 0.5) * 0.1).repeat_interleave(self.n)[
                :, None, None, None
            ]
            x = (x.clamp(0, 1) ** gam * con + bri).clamp(0, 1)
        else:
            o = (H - self.crop) // 2
            x = x[..., o : o + self.crop, o : o + self.crop]
        x = (x - 0.45) / 0.25
        x = x.view(B, S, self.n, 3, self.crop, self.crop)
        if train:  # slot dropout: randomly hide one present slot (robustness to missing series)
            drop = torch.rand(B, device=self.dev) < 0.3
            if drop.any():
                cand = torch.where(
                    m, torch.rand(B, S, device=self.dev), torch.full((B, S), -1.0, device=self.dev)
                )
                j = cand.argmax(1)
                keep_one = m.sum(1) > 1
                kill = drop & keep_one
                m = m.clone()
                m[kill.nonzero().flatten(), j[kill]] = False
        pos = (centre.float() * (32 / D)).long().clamp(0, 31)
        return x.to(memory_format=torch.contiguous_format), m, pos


# ---------------------------------------------------------------- utils
def make_folds(df, k, seed=42):
    """Fold id per non-gold study, grouped by identical report text."""
    from sklearn.model_selection import GroupKFold

    ng = df[df.is_gold == 0]
    groups = ng["Report"].fillna(ng.index.to_series()).factorize()[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ng))
    fold = np.empty(len(ng), int)
    for f, (_, va) in enumerate(GroupKFold(k).split(perm, groups=groups[perm])):
        fold[perm[va]] = f
    return pd.Series(fold, index=ng.index)


def macro_auc(y, p, binarize=None):
    out = {}
    for j, lab in enumerate(LABELS):
        t = y[:, j] if binarize is None else (y[:, j] >= binarize).astype(int)
        if 0 < t.sum() < len(t):
            out[lab] = roc_auc_score(t, p[:, j])
    return float(np.mean(list(out.values()))), out


@torch.no_grad()
def predict(model, batcher, rows, bs):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        x, m, pos = batcher.get(rows[i : i + bs], train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(x, m, pos).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/data/pack")
    ap.add_argument("--labels", default="/data/code/data/labels_v1.csv")
    ap.add_argument("--train-csv", default="/data/code/data/raw/train.csv")
    ap.add_argument("--backbone", default="tf_efficientnetv2_b0.in1k")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--n-slices", type=int, default=16)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--slots", default=",".join(SLOTS))
    ap.add_argument("--out", default="/data/exp/debug")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--head", default="pool", choices=["pool", "attn", "attn2"])
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument(
        "--eval-train", action="store_true", help="also report AUC on (a subset of) train rows"
    )
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    slot_ids = [SLOTS.index(s) for s in a.slots.split(",")]

    if a.bench:
        model = (
            KneeNet(a.backbone, n_slots=len(slot_ids)).to(dev).to(memory_format=torch.channels_last)
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        vols = np.random.randint(0, 255, (64, len(slot_ids), 32, 256, 256), dtype=np.uint8)
        masks = np.ones((64, len(slot_ids)), bool)
        b = GpuBatcher(vols, masks, a.n_slices, a.crop, dev)
        for it in range(12):
            if it == 4:
                torch.cuda.synchronize()
                t0 = time.time()
            x, m, pos = b.get(list(range(a.bs)), True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, m, pos).float().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 8
        print(
            json.dumps(
                {
                    "backbone": a.backbone,
                    "bs": a.bs,
                    "n_slices": a.n_slices,
                    "slots": len(slot_ids),
                    "s_per_step": round(dt, 3),
                    "studies_per_s": round(a.bs / dt, 2),
                    "epoch_min_4300": round(4300 / (a.bs / dt) / 60, 1),
                    "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                }
            )
        )
        return

    out = Path(a.out) / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    idx = pd.read_csv(Path(a.pack) / "index.csv")  # row, StudyInstanceUID, mask_<slot>...
    vols = np.load(Path(a.pack) / "vols.npy", mmap_mode="r")
    if slot_ids != list(range(len(SLOTS))):
        vols = vols[:, slot_ids] if False else vols  # slot subset handled via mask below
    masks_all = idx[[f"mask_{s}" for s in SLOTS]].values.astype(bool)
    keep = np.zeros(len(SLOTS), bool)
    keep[slot_ids] = True
    masks_all = masks_all & keep[None]
    lab = pd.read_csv(a.labels).set_index("StudyInstanceUID")
    rep = pd.read_csv(a.train_csv).set_index("StudyInstanceUID")["Report"]
    lab = lab.join(rep)
    lab = lab.loc[lab.index.intersection(idx.StudyInstanceUID)]
    row_of = dict(zip(idx.StudyInstanceUID, idx.row))
    folds = make_folds(lab, a.folds)
    tr_uid = folds.index[folds != a.fold]
    va_uid = folds.index[folds == a.fold]
    gold_uid = lab.index[lab.is_gold == 1]
    if a.smoke:
        tr_uid, va_uid, gold_uid, a.epochs = tr_uid[:5], va_uid[:5], gold_uid[:8], 1
    if a.max_train:
        tr_uid = tr_uid[: a.max_train]
    if a.max_val:
        va_uid = va_uid[: a.max_val]
    Y = lab[LABELS].values.astype(np.float32)
    yrow = {u: i for i, u in enumerate(lab.index)}
    tr_rows = np.array([row_of[u] for u in tr_uid])
    tr_y = Y[[yrow[u] for u in tr_uid]]
    va_rows = [row_of[u] for u in va_uid]
    va_y = Y[[yrow[u] for u in va_uid]]
    gd_rows = [row_of[u] for u in gold_uid]
    gd_y = Y[[yrow[u] for u in gold_uid]]
    print(f"fold {a.fold}: train {len(tr_rows)} val {len(va_rows)} gold {len(gd_rows)}", flush=True)

    batcher = GpuBatcher(vols, masks_all, a.n_slices, a.crop, dev)
    model = KneeNet(a.backbone, head=a.head).to(dev).to(memory_format=torch.channels_last)
    enc_params = list(model.enc.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}],
        weight_decay=a.wd,
    )
    steps_per_epoch = max(1, len(tr_rows) // a.bs)
    total = steps_per_epoch * a.epochs
    warm = min(steps_per_epoch, 300)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))
    )
    log = open(out / "log.jsonl", "a")
    best, t0 = -1, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(tr_rows))
        tl, n = 0.0, 0
        for it in range(steps_per_epoch):
            b = perm[it * a.bs : (it + 1) * a.bs]
            x, m, pos = batcher.get(tr_rows[b].tolist(), True)
            y = torch.from_numpy(tr_y[b]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, m, pos).float()
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tl += loss.item()
            n += 1
            if it % 100 == 0:
                print(
                    f"ep {ep} it {it}/{steps_per_epoch} loss {tl / n:.4f} {time.time() - t0:.0f}s",
                    flush=True,
                )
        pv = predict(model, batcher, va_rows, a.bs * 2)
        pg = predict(model, batcher, gd_rows, a.bs * 2)
        va_auc, va_per = macro_auc(va_y, pv, binarize=0.5)
        if a.eval_train:
            ptr = predict(model, batcher, tr_rows[:200].tolist(), a.bs * 2)
            tr_auc, _ = macro_auc(tr_y[:200], ptr, binarize=0.5)
            print(f"TRAIN-subset auc {tr_auc:.4f}", flush=True)
        va_loss = float(
            F.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6), torch.tensor(va_y))
        )
        gd_auc, gd_per = macro_auc(gd_y, pg)
        rec = {
            "epoch": ep,
            "train_loss": tl / n,
            "val_loss": va_loss,
            "val_auc_llm": va_auc,
            "gold_auc": gd_auc,
            "gold_per": gd_per,
            "val_per": va_per,
            "min": (time.time() - t0) / 60,
            "pred_std": float(pv.std(0).mean()),
        }
        print(
            json.dumps(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in rec.items()
                    if k not in ("gold_per", "val_per")
                }
            ),
            flush=True,
        )
        log.write(json.dumps(rec) + "\n")
        log.flush()
        # model selection on the LLM-labelled validation fold only (never on gold)
        if va_auc > best:
            best = va_auc
            torch.save(model.state_dict(), out / "best.pt")
            np.save(out / "oof_val.npy", pv)
            np.save(out / "pred_gold.npy", pg)
            pd.Series(list(va_uid)).to_csv(out / "val_uids.csv", index=False)
            pd.Series(list(gold_uid)).to_csv(out / "gold_uids.csv", index=False)
    torch.save(model.state_dict(), out / "last.pt")
    print("DONE best val_auc_llm", best, flush=True)


if __name__ == "__main__":
    main()
