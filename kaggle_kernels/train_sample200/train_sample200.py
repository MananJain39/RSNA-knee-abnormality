# ruff: noqa: E402
"""Stage 1 (Track A): gated baseline training on the 200-study sample.

Runs entirely inside a Kaggle GPU kernel. Pipeline, in order:
  0. Install repo from the rsna-knee-repo-snapshot dataset + [classical] extra.
  1. Integrity scan of every slot .npy in the attached volumes dataset.
  2. Leak-safe gold-reserved split (Report-grouped) over usable studies.
  3. B2 model contract probe (DINOv2 forward/backward on GPU, 8 studies).
  4. B3 smoke training (2 epochs) - pass if validation AUC is finite and
     scoreable on the binary gold studies.
  5. C1 full main-branch training (20 epochs) -> best.pt.
  6. Gold evaluation (per-label + macro AUC vs gold studies held in val).

All small artifacts are written to /kaggle/working/out for `kaggle kernels output`.
"""

import json
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ast

import numpy as np
import pandas as pd

REPO_URL = "https://github.com/MananJain39/RSNA-knee-abnormality.git"
COMP_DIR = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
DATASET_DIR = Path("/kaggle/input/datasets/mananjain39/rsna-knee-processed-sample200")
LABELS_SRC = Path("/kaggle/input/datasets/mananjain39/rsna-knee-labels-final/labels_final.csv")
SNAPSHOT_DIR = Path("/kaggle/input/datasets/mananjain39/rsna-knee-repo-snapshot")
INPUT_ROOT = Path("/kaggle/input")
for p in sorted(INPUT_ROOT.iterdir()):
    print("[input]", p.name, "->", [q.name for q in sorted(p.iterdir())][:8], flush=True)
REPO_DIR = Path("/kaggle/working/rsna-knee-abnormality")
OUT = Path("/kaggle/working/out")
TMP = Path("/kaggle/tmp")
OUT.mkdir(parents=True, exist_ok=True)

KERNEL_VERSION = "train-sample200-v18"
SEED = 42
VAL_FRAC = 0.20
SPLIT_TAG = "sample200"
PROBE_STUDIES = 8
SMOKE_EPOCHS = 2
FULL_EPOCHS = 20

stage_log: list[dict] = []


# Checkpoints go to TMP so failed runs ship small outputs; copied to OUT at the end.
CKPT_ROOT = TMP / "ckpts"
CKPT_ROOT.mkdir(parents=True, exist_ok=True)


def log_stage(name: str, status: str, **extra) -> None:
    entry = {"stage": name, "status": status,
             "utc": datetime.now(timezone.utc).isoformat(), **extra}
    stage_log.append(entry)
    print(f"\n===== [{len(stage_log)}] {name}: {status} =====", flush=True)
    if extra:
        print(json.dumps(extra, indent=2, default=str), flush=True)


def run(cmd: list[str], cwd: Path | None = None, desc: str = "") -> int:
    print(f"\n>>> {' '.join(cmd)}\n(cwd={cwd}) {desc}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd)
    print(f"<<< rc={proc.returncode} elapsed={time.time() - t0:.0f}s", flush=True)
    return proc.returncode


PY = sys.executable

# ---------------------------------------------------------------- 0. repo
t0 = time.time()
try:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    print("[repo] snapshot mount:", sorted(q.name for q in SNAPSHOT_DIR.iterdir()), flush=True)
    payload_bundle = next(iter(SNAPSHOT_DIR.rglob("rsna-knee-repo-snapshot.bundle")), None)
    if payload_bundle is not None:
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["unzip", "-q", "-o", str(payload_bundle), "-d", str(REPO_DIR)], check=True)
    else:
        shutil.copytree(SNAPSHOT_DIR, REPO_DIR, ignore=shutil.ignore_patterns("dataset-metadata.json"))
    print("[repo] snapshot contents:", sorted(q.name for q in REPO_DIR.iterdir()), flush=True)
    sys.path.insert(0, str(REPO_DIR))
    rc = run([PY, "-m", "pip", "install", "-q", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu124"],
             cwd=REPO_DIR, desc="pin torch+torchvision cu124 build (sm60-sm90 kernels)")
    if rc != 0:
        raise RuntimeError(f"torch pin failed rc={rc}")
    rc = run([PY, "-m", "pip", "install", "-q", "-e", ".[classical]"], cwd=REPO_DIR)
    if rc != 0:
        raise RuntimeError(f"pip install -e .[classical] failed rc={rc}")
    import torch  # noqa: E402
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} "
          f"available={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)
    log_stage("0_repo_install", "ok", seconds=round(time.time() - t0))
except Exception as exc:  # noqa: BLE001
    log_stage("0_repo_install", "fail", error=str(exc))
    raise

import pandas as pd  # noqa: E402
from src.config import LABEL_COLUMNS  # noqa: E402
from src.data.splits import make_split  # noqa: E402
from src.imaging.slot_builder import SLOT_NAMES  # noqa: E402

# ------------------------------------------------- 1. layout + integrity
def resolve_volume_root() -> Path:
    single = DATASET_DIR / "rsna-knee-processed-sample200"
    if (single / "manifest.json").exists():
        return single
    zips = list(DATASET_DIR.glob("*.zip"))
    if zips:
        target = TMP / "volumes_payload"
        if not (target / "manifest.json").exists():
            target.mkdir(parents=True, exist_ok=True)
            rc = subprocess.run(
                ["unzip", "-q", "-n", "-d", str(target), str(zips[0])]
            )
            if rc.returncode != 0:
                raise RuntimeError("unzip failed")
        return target / "rsna-knee-processed-sample200"
    raise FileNotFoundError("No manifest and no payload zip found in dataset")


t0 = time.time()
try:
    def resolve_volume_root() -> Path:
        single = DATASET_DIR / "rsna-knee-processed-sample200"
        if (single / "manifest.json").exists():
            return single
        zips = list(DATASET_DIR.glob("*.zip"))
        if zips:
            target = TMP / "volumes_payload"
            if not (target / "manifest.json").exists():
                target.mkdir(parents=True, exist_ok=True)
                rc = subprocess.run(["unzip", "-q", "-n", "-d", str(target), str(zips[0])])
                if rc.returncode != 0:
                    raise RuntimeError("unzip failed")
            return target / "rsna-knee-processed-sample200"
        raise FileNotFoundError("No manifest and no payload zip found in dataset")

    def inspect_npy(f: Path) -> str:
        size = f.stat().st_size
        with open(f, "rb") as fh:
            magic = fh.read(6)
            if magic != b"\x93NUMPY":
                return "bad-magic"
            major = fh.read(1)[0]
            fh.read(1)
            hlen_len = 4 if major >= 2 else 2
            raw = fh.read(hlen_len)
            hlen = struct.unpack("<I", raw)[0] if hlen_len == 4 else struct.unpack("<H", raw)[0]
            data_start = fh.tell() + hlen
            header = fh.read(hlen).decode("latin1").rstrip().rstrip(",")
        d = ast.literal_eval(header)
        shape, itemsize = d["shape"], np.dtype(d["descr"]).itemsize
        expected = int(np.prod(shape)) * itemsize
        if size - data_start != expected:
            return f"truncated ({size - data_start}/{expected})"
        arr = np.load(f)
        if not np.isfinite(arr).all():
            return "non-finite"
        return "ok"


    VOLUME_ROOT = resolve_volume_root()
    records = []
    for f in sorted(VOLUME_ROOT.glob("1.*/*.npy")):
        try:
            status = inspect_npy(f)
        except Exception as exc:  # noqa: BLE001
            status = f"error: {type(exc).__name__}"
        records.append((f.parent.name, f.name, status, f.stat().st_size))

    bad = [r for r in records if r[2] != "ok"]
    counts = {}
    for uid, slot, status, _ in records:
        counts.setdefault(uid, 0)
        if status == "ok":
            counts[uid] += 1
    usable_uids = sorted(u for u, n in counts.items() if n >= 1)
    integrity = {
        "volume_root": str(VOLUME_ROOT),
        "n_files": len(records),
        "n_bad_files": len(bad),
        "bad_files": [{"uid": u, "slot": s, "status": st, "bytes": b} for u, s, st, b in bad],
        "n_studies_seen": len(counts),
        "n_usable": len(usable_uids),
        "slot_histogram": {str(n): sum(1 for m in counts.values() if m == n)
                           for n in sorted(set(counts.values()))},
    }
    (OUT / "integrity.json").write_text(json.dumps(integrity, indent=2))
    log_stage("1_integrity_scan", "ok", seconds=round(time.time() - t0),
              usable_studies=len(usable_uids), bad_files=len(bad))
except Exception as exc:  # noqa: BLE001
    log_stage("1_integrity_scan", "fail", error=str(exc))
    raise

# ------------------------------------------------------------------ 2. split
t0 = time.time()
try:
    train_df_comp = pd.read_csv(COMP_DIR / "train.csv")
    (REPO_DIR / "data/raw").mkdir(parents=True, exist_ok=True)
    train_df_comp.to_csv(REPO_DIR / "data/raw/train.csv", index=False)
    (REPO_DIR / "data/processed/labels_final.csv").parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LABELS_SRC, REPO_DIR / "data/processed/labels_final.csv")
    train_split, val_split = make_split(
        train_df=train_df_comp, val_frac=VAL_FRAC, seed=SEED, study_uids=usable_uids
    )
    split_dir = REPO_DIR / "data/processed/splits" / SPLIT_TAG
    split_dir.mkdir(parents=True, exist_ok=True)
    train_split.to_csv(split_dir / "train.csv", index=False)
    val_split.to_csv(split_dir / "val.csv", index=False)
    gold_in_train = int(train_split[LABEL_COLUMNS[0]].notna().sum())
    gold_in_val = int(val_split[LABEL_COLUMNS[0]].notna().sum())
    if gold_in_train > 0:
        raise RuntimeError("gold studies leaked into the train split")
    log_stage("2_build_split", "ok", seconds=round(time.time() - t0),
              train=len(train_split), val=len(val_split),
              gold_in_val=gold_in_val)
except Exception as exc:  # noqa: BLE001
    log_stage("2_build_split", "fail", error=str(exc))
    raise

# ------------------------------------------------- 2.5 series clusters
# extract kernel v7 did not ship series_metadata.csv, so real acquisition
# clustering is impossible yet. For the main branch (Stage 1) the trainer only
# needs a schema-compatible study->cluster map; assign every study cluster 0.
try:
    rows = []
    for uid in usable_uids:
        rows.append({"StudyInstanceUID": uid, "SeriesInstanceUID": f"synthetic-{uid}-0",
                     "cluster_id": 0})
    clusters_csv = REPO_DIR / "data/processed/series_clusters.csv"
    pd.DataFrame(rows).to_csv(clusters_csv, index=False)
    log_stage("2b_synthesize_clusters", "ok", rows=len(rows),
              note="uniform cluster 0; real clustering deferred to Stage 2")
except Exception as exc:  # noqa: BLE001
    log_stage("2b_synthesize_clusters", "fail", error=str(exc))
    raise

# -------------------------------------------------------------- 3. B2 probe
t0 = time.time()
try:
    rc = run(
        [PY, "scripts/probe_sample_architecture.py",
         "--volumes-dir", str(VOLUME_ROOT),
         "--split-dir", str(split_dir),
         "--labels-csv", str(REPO_DIR / "data/processed/labels_final.csv"),
         "--with-model", "--max-studies", str(PROBE_STUDIES), "--device", "cuda",
         "--output-json", str(REPO_DIR / "probe_gpu_sample200.json")],
        cwd=REPO_DIR,
    )
    probe_json = REPO_DIR / "probe_gpu_sample200.json"
    if probe_json.exists():
        shutil.copy2(probe_json, OUT / "probe_gpu_sample200.json")
    if rc != 0:
        raise RuntimeError(f"probe exited rc={rc} (see log)")
    log_stage("3_b2_gpu_probe", "ok", seconds=round(time.time() - t0))
except Exception as exc:  # noqa: BLE001
    shutil.copytree(REPO_DIR / "probe_gpu_sample200.json", OUT / "probe_gpu_sample200.json",
                    dirs_exist_ok=True) if (REPO_DIR / "probe_gpu_sample200.json").exists() else None
    log_stage("3_b2_gpu_probe", "fail", error=str(exc))
    raise

# ---------------------------------------------------- 4. B3 smoke (2 epochs)
t0 = time.time()
try:
    rc = run(
        [PY, "scripts/train_classical.py", "--mode", "main",
         "--data-dir", str(VOLUME_ROOT), "--labels-csv", str(REPO_DIR / "data/processed/labels_final.csv"),
         "--split-tag", SPLIT_TAG, "--output-dir", str(CKPT_ROOT / "smoke_main"),
         "--max-epochs", str(SMOKE_EPOCHS), "--device", "cuda"],
        cwd=REPO_DIR,
    )
    entries = [json.loads(line) for line in
               (REPO_DIR / "experiments/main.jsonl").read_text().splitlines() if line.strip()]
    last = entries[-3:] if len(entries) >= 3 else entries
    (OUT / "smoke_metrics.json").write_text(json.dumps(last, indent=2))
    smoke_aucs = [e.get("metrics", {}).get("val_macro_auc") for e in last
                  if e.get("metrics", {}).get("val_macro_auc") is not None]
    if rc != 0:
        raise RuntimeError(f"smoke train exited rc={rc}")
    if (not smoke_aucs or
            not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in smoke_aucs) or
            all(value == 0.0 for value in smoke_aucs)):
        raise RuntimeError(f"degenerate smoke run: aucs={smoke_aucs}")
    log_stage("4_b3_smoke_train", "ok", seconds=round(time.time() - t0), aucs=smoke_aucs)
except Exception as exc:  # noqa: BLE001
    log_stage("4_b3_smoke_train", "fail", error=str(exc))
    raise

# ---------------------------------------------------------- 5. C1 full train
t0 = time.time()
try:
    rc = run(
        [PY, "scripts/train_classical.py", "--mode", "main",
         "--data-dir", str(VOLUME_ROOT), "--labels-csv", str(REPO_DIR / "data/processed/labels_final.csv"),
         "--split-tag", SPLIT_TAG, "--output-dir", str(CKPT_ROOT / "main"),
         "--max-epochs", str(FULL_EPOCHS), "--device", "cuda"],
        cwd=REPO_DIR,
    )
    entries = [json.loads(line) for line in
               (REPO_DIR / "experiments/main.jsonl").read_text().splitlines() if line.strip()]
    (OUT / "main_metrics_history.json").write_text(json.dumps(entries, indent=2))
    if rc != 0:
        raise RuntimeError(f"full train exited rc={rc}")
    ckpt = CKPT_ROOT / "main" / "best.pt"
    if not ckpt.exists():
        raise RuntimeError("no best.pt produced")
    log_stage("5_c1_full_train", "ok", seconds=round(time.time() - t0),
              checkpoint=str(ckpt), size_mb=round(ckpt.stat().st_size / 1e6, 1))
except Exception as exc:  # noqa: BLE001
    log_stage("5_c1_full_train", "fail", error=str(exc))
    raise

# --------------------------------------------------------------- 6. gold eval
t0 = time.time()
try:
    rc = run(
        [PY, "scripts/evaluate_classical.py",
         "--main-checkpoint", str(CKPT_ROOT / "main" / "best.pt"),
         "--labels-csv", str(REPO_DIR / "data/processed/labels_final.csv"),
         "--data-dir", str(VOLUME_ROOT), "--split-tag", SPLIT_TAG,
         "--output-dir", str(OUT / "eval"), "--device", "cuda"],
        cwd=REPO_DIR,
    )
    if rc != 0:
        raise RuntimeError(f"eval exited rc={rc}")
    gold_metrics = json.loads((OUT / "eval" / "gold_metrics.json").read_text())
    log_stage("6_gold_eval", "ok", seconds=round(time.time() - t0),
              macro_auc=gold_metrics.get("macro_auc"),
              n_gold=gold_metrics.get("n_gold_studies"))
except Exception as exc:  # noqa: BLE001
    log_stage("6_gold_eval", "fail", error=str(exc))
    raise

summary = {
    "kernel_version": KERNEL_VERSION,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "val_frac": VAL_FRAC,
    "probe_studies": PROBE_STUDIES,
    "smoke_epochs": SMOKE_EPOCHS,
    "full_epochs": FULL_EPOCHS,
    "train_studies": len(train_split),
    "val_studies": len(val_split),
    "usable_volume_studies": len(usable_uids),
    "stages": stage_log,
    "gold_macro_auc": gold_metrics.get("macro_auc"),
}
try:
    shutil.copy2(CKPT_ROOT / "main" / "best.pt", OUT / "best_main.pt")
except Exception as exc:  # noqa: BLE001
    log_stage("7_publish_checkpoint", "fail", error=str(exc))
else:
    log_stage("7_publish_checkpoint", "ok")
(OUT / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
print("\n===== STAGE 1 COMPLETE =====")
print(f"gold macro AUC: {summary['gold_macro_auc']}")
print(json.dumps(stage_log, indent=2, default=str))
