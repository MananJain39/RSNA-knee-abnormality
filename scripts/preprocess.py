"""DICOM -> canonical uint8 volumes, one .npz per study.

Per study we fill up to 5 slots (plane x fluid):
  sag_fs, sag_t1 (non-fluid), cor_fs, cor_t1, ax_fs
When several series compete for a slot we take the one with the most slices.

Canonicalisation (DICOM patient coords are LPS: +x = patient left, +y = posterior, +z = superior):
  - slices sorted along the slice normal
  - in-plane axes reoriented so that
      sagittal: image down = inferior, image right = posterior, slices ordered by +x
      coronal : image down = inferior, image right = +x
      axial   : image down = posterior, image right = +x
  - laterality: left knees are mirrored in x (flip image for cor/ax, reverse slice order for sag),
    so medial/lateral appear on the same side for every study.
  - pad to square, resize to SIZE x SIZE, clip [p0.5, p99.5] -> uint8
  - slices resampled (nearest index) to exactly DEPTH

Usage:
  python preprocess.py --series-root /data/train_series --series-csv train_series.csv \
      --out /data/vol --workers 90 [--limit 5]
"""

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402

cv2.setNumThreads(1)
SIZE = 256
DEPTH = 32
SLOTS = {
    ("Sagittal", 1): "sag_fs",
    ("Sagittal", 0): "sag_t1",
    ("Coronal", 1): "cor_fs",
    ("Coronal", 0): "cor_t1",
    ("Axial", 1): "ax_fs",
}


_ZIP = None
_MEMBERS = None  # series uid -> list of zip member names (when reading from zip)


def _init_zip(zpath, members):
    global _ZIP, _MEMBERS
    import zipfile

    _ZIP = zipfile.ZipFile(zpath) if zpath else None
    _MEMBERS = members


def list_series(series_root, study, suid):
    if _MEMBERS is not None:
        return _MEMBERS.get(suid, [])
    folder = os.path.join(series_root, study, suid)
    return [os.path.join(folder, f) for f in os.listdir(folder)] if os.path.isdir(folder) else []


def read_series(files):
    import io

    dss = []
    for f in files:
        if not f.endswith(".dcm"):
            continue
        try:
            src = io.BytesIO(_ZIP.read(f)) if _ZIP is not None else f
            dss.append(pydicom.dcmread(src))
        except Exception:
            pass
    return dss


def series_volume(dss, plane):
    ds0 = dss[0]
    iop = np.array(ds0.ImageOrientationPatient, dtype=float)
    r, c = iop[:3], iop[3:]  # r: direction of increasing column index; c: increasing row index
    n = np.cross(r, c)
    pos = np.array([float(np.dot(np.array(d.ImagePositionPatient, dtype=float), n)) for d in dss])
    order = np.argsort(pos)
    dss = [dss[i] for i in order]
    # keep only the modal image shape (rare mixed-size series)
    shapes = [(int(d.Rows), int(d.Columns)) for d in dss]
    mode = max(set(shapes), key=shapes.count)
    imgs = []
    for d in dss:
        if (int(d.Rows), int(d.Columns)) != mode:
            continue
        a = d.pixel_array.astype(np.float32)
        if a.ndim == 3:  # multi-frame or RGB; take first channel/frame
            a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
        slope = float(getattr(d, "RescaleSlope", 1) or 1)
        icpt = float(getattr(d, "RescaleIntercept", 0) or 0)
        imgs.append(a * slope + icpt)
    vol = np.stack(imgs)  # (D, H, W); axis1 ~ c, axis2 ~ r
    # slice axis: want increasing along canonical +axis of the normal's dominant component
    k = int(np.argmax(np.abs(n)))
    if n[k] < 0:
        vol = vol[::-1]
    # in-plane target axes
    if plane == "Sagittal":
        want_down, want_right = (2, -1), (1, +1)  # down=-z (inferior), right=+y (posterior)
    elif plane == "Coronal":
        want_down, want_right = (2, -1), (0, +1)  # down=-z, right=+x
    else:
        want_down, want_right = (1, +1), (0, +1)  # down=+y (posterior), right=+x
    ri, ci = int(np.argmax(np.abs(r))), int(np.argmax(np.abs(c)))
    if ri == want_down[0] and ci == want_right[0]:  # transposed image
        vol = vol.transpose(0, 2, 1)
        r, c = c, r
        ri, ci = ci, ri
    if ci == want_down[0] and np.sign(c[ci]) != want_down[1]:
        vol = vol[:, ::-1, :]
    if ri == want_right[0] and np.sign(r[ri]) != want_right[1]:
        vol = vol[:, :, ::-1]
    center_x = float(np.mean([np.array(d.ImagePositionPatient, dtype=float)[0] for d in dss]))
    meta = {
        "n_slices": int(vol.shape[0]),
        "rows": mode[0],
        "cols": mode[1],
        "spacing": [float(x) for x in getattr(ds0, "PixelSpacing", [0, 0])],
        "thick": float(getattr(ds0, "SliceThickness", 0) or 0),
        "laterality_tag": str(
            getattr(ds0, "Laterality", "") or getattr(ds0, "ImageLaterality", "") or ""
        ),
        "body_part": str(getattr(ds0, "BodyPartExamined", "") or ""),
        "desc": str(getattr(ds0, "SeriesDescription", "") or ""),
        "manufacturer": str(getattr(ds0, "Manufacturer", "") or ""),
        "field": float(getattr(ds0, "MagneticFieldStrength", 0) or 0),
        "center_x": center_x,
    }
    return np.ascontiguousarray(vol), meta


def to_uint8(vol):
    D, H, W = vol.shape
    s = max(H, W)
    pad = np.zeros((D, s, s), np.float32)
    pad[:, (s - H) // 2 : (s - H) // 2 + H, (s - W) // 2 : (s - W) // 2 + W] = vol
    out = np.stack([cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for x in pad])
    lo, hi = np.percentile(out[out > 0] if (out > 0).any() else out, [0.5, 99.5])
    out = np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def resample_depth(vol):
    idx = np.round(np.linspace(0, vol.shape[0] - 1, DEPTH)).astype(int)
    return vol[idx]


def laterality(metas):
    tags = [m["laterality_tag"].upper()[:1] for m in metas if m["laterality_tag"]]
    if tags:
        return max(set(tags), key=tags.count), "tag"
    for m in metas:
        d = (m["desc"] + " " + m["body_part"]).upper()
        if any(t in d for t in ["LEFT", "IZQUIERD", "LINK", "GAUCHE"]):
            return "L", "desc"
        if any(t in d for t in ["RIGHT", "DERECH", "RECHT", "DROIT"]):
            return "R", "desc"
    xs = [m["center_x"] for m in metas]
    x = float(np.median(xs))
    return ("L" if x > 0 else "R"), f"x={x:.0f}"


def process_study(study, rows, series_root, out_dir):
    out = Path(out_dir) / f"{study}.npz"
    if out.exists():
        return study, {"skipped": True}
    cand = {}
    for r in rows:
        key = (r["Anatomical_Plane"], int(r["Fluid_Sensitive"]))
        if key not in SLOTS:
            continue
        files = list_series(series_root, study, r["SeriesInstanceUID"])
        n = len(files)
        if n < 3:
            continue
        if key not in cand or n > cand[key][1]:
            cand[key] = (files, n, r["SeriesInstanceUID"])
    arrays, metas = {}, {}
    for key, (folder, n, suid) in cand.items():
        try:
            vol, meta = series_volume(read_series(folder), key[0])
            meta["series"] = suid
            arrays[SLOTS[key]] = vol
            metas[SLOTS[key]] = meta
        except Exception as e:
            metas[SLOTS[key]] = {"error": f"{type(e).__name__}: {e}"[:300], "series": suid}
    ok = [m for m in metas.values() if "error" not in m]
    lat, how = laterality(ok) if ok else ("R", "none")
    save = {}
    for name, vol in arrays.items():
        v = to_uint8(vol)
        if lat == "L":
            v = v[::-1] if name.startswith("sag") else v[:, :, ::-1]
        save[name] = resample_depth(v)
    np.savez_compressed(out, **save)
    return study, {"lat": lat, "lat_how": how, "slots": metas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", default="")
    ap.add_argument("--zip", default="", help="read DICOMs directly from the competition zip")
    ap.add_argument("--zip-prefix", default="train_series/")
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    se = pd.read_csv(a.series_csv)
    groups = {s: g.to_dict("records") for s, g in se.groupby("StudyInstanceUID")}
    studies = sorted(groups)
    members = None
    if a.zip:
        import zipfile

        members = {}
        have = set()
        for name in zipfile.ZipFile(a.zip).namelist():
            if name.startswith(a.zip_prefix) and name.endswith(".dcm"):
                parts = name[len(a.zip_prefix) :].split("/")
                if len(parts) >= 3:
                    members.setdefault(parts[1], []).append(name)
                    have.add(parts[0])
        studies = [s for s in studies if s in have]
        print(f"zip index: {len(members)} series", flush=True)
    else:
        studies = [s for s in studies if os.path.isdir(os.path.join(a.series_root, s))]
    if a.limit:
        studies = studies[: a.limit]
    print(f"{len(studies)} studies, {a.workers} workers", flush=True)
    meta_path = Path(a.out).parent / (Path(a.out).name + "_meta.jsonl")
    t0 = time.time()
    with (
        ProcessPoolExecutor(a.workers, initializer=_init_zip, initargs=(a.zip, members)) as ex,
        open(meta_path, "a") as mf,
    ):
        futs = {ex.submit(process_study, s, groups[s], a.series_root, a.out): s for s in studies}
        for i, fu in enumerate(as_completed(futs), 1):
            s = futs[fu]
            try:
                _, meta = fu.result()
            except Exception:
                meta = {"fatal": traceback.format_exc()[-500:]}
            mf.write(json.dumps({"study": s, **meta}) + "\n")
            if i % 100 == 0 or i == len(studies):
                el = time.time() - t0
                print(
                    f"{i}/{len(studies)} {el:.0f}s eta={el / i * (len(studies) - i):.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
