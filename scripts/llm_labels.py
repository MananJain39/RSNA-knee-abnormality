"""Report -> closed-vocabulary finding states via Claude on Bedrock.

Usage:
  python scripts/llm_labels.py --model us.anthropic.claude-sonnet-5-5 --subset gold --out exp/01_labels/raw_sonnet_gold.jsonl
  python scripts/llm_labels.py --model ... --subset all --out ... --workers 24

Output: one JSON line per study {uid, model, states{label: state}, evidence{label: str}, usage{...}}.
Resume-safe: skips uids already present in --out.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from anthropic import AnthropicBedrock, APIConnectionError, APIStatusError, RateLimitError

ROOT = Path(__file__).resolve().parents[1]
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
STATES = ["negated", "not_mentioned", "below_threshold", "equivocal", "positive"]

SYSTEM = """You are a musculoskeletal radiologist converting knee MRI reports into structured labels for a research dataset. Reports may be in English, Spanish, Turkish, Greek, German, Dutch, Portuguese, French, Italian or other languages; read them in the original language.

For each of 12 findings, choose exactly one state:
- "positive": the report affirms the finding AND it meets the positive definition below.
- "equivocal": the report raises it as possible/suspected/cannot be excluded/probable, or severity is ambiguous right at the threshold.
- "below_threshold": the report mentions an abnormality of that structure that does NOT meet the positive definition (e.g. low-grade sprain, trace or small effusion, degeneration without tear, mild chondral thinning/grade 1-2 chondromalacia, small cyst, old/healed/chronic injury, intact post-operative graft).
- "negated": the report explicitly says the structure is normal/intact/absent (e.g. "no effusion", "ACL intact", "menisco sin roturas", "kırık yok").
- "not_mentioned": the report is silent about it.

Positive definitions (expert annotation criteria):
- ACL: high-grade partial or complete (full-thickness) tear, including tear/rupture of an ACL graft. Low-grade sprain, mucoid degeneration, ganglion = below_threshold.
- MCL: high-grade (grade 2-3) acute tear/rupture. Low-grade sprain, periligamentous edema only, chronic thickening = below_threshold.
- Medial Meniscus: tear (signal reaching an articular surface), including complex, radial, bucket-handle, root tear, flap, displaced fragment, maceration/extrusion with tear, post-meniscectomy re-tear. Intrasubstance degeneration (grade 1-2 signal not reaching surface) = below_threshold.
- Lateral Meniscus: same as medial, for the lateral meniscus. Discoid meniscus without tear = below_threshold.
- Medial OA: medial tibiofemoral compartment osteoarthritis with substantial cartilage loss (>50% thickness / high-grade or full-thickness chondral loss, grade 3-4 chondromalacia, bone-on-bone, moderate-severe arthrosis/gonarthrosis). Mild thinning, grade 1-2 chondropathy, small osteophytes only = below_threshold.
- Lateral OA: same as medial OA, lateral tibiofemoral compartment.
- PF OA: same, patellofemoral compartment (patella and/or trochlea). Grade 1-2 chondromalacia patellae = below_threshold; grade 3-4 / high-grade / full-thickness patellar or trochlear cartilage loss = positive.
- Effusion: moderate or large joint effusion. Small/minimal/trace/physiological fluid = below_threshold. Unqualified "effusion"/"derrame" without size = equivocal.
- Synovitis: synovial thickening/inflammation, synovitis, Hoffa fat pad edema/inflammation (Hoffa-synovitis), proliferative synovium, plica with inflammation, effusion-synovitis.
- Baker's: moderate or large popliteal (Baker's) cyst. Small Baker's cyst = below_threshold. Unqualified size = equivocal.
- Contusion: bone marrow edema / bone bruise / contusion of traumatic or reactive nature WITHOUT a fracture line at that site. Subchondral edema associated with OA counts as below_threshold unless described as contusion/bruise.
- Fracture: acute fracture, including osteochondral fracture, impaction fracture, Segond, tibial plateau, avulsion, stress/insufficiency fracture, subchondral insufficiency fracture. Old healed fracture = below_threshold.

Rules:
- Use the findings AND the impression/conclusion; if they conflict, trust the impression.
- Only the current examination counts. Ignore clinical history/indication text unless it describes the current findings.
- Laterality of the knee (left/right) is irrelevant; medial vs lateral compartment matters.
- "Internal"/"interno" meniscus = medial; "external"/"externo" = lateral.
- Give a very short evidence quote (<=12 words, original language) for every state except not_mentioned (use "")."""


def build_schema():
    props = {}
    for lab in LABELS:
        props[lab] = {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": STATES},
                "evidence": {"type": "string"},
            },
            "required": ["state", "evidence"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": props,
        "required": LABELS,
        "additionalProperties": False,
    }


SCHEMA = build_schema()
_lock = threading.Lock()


def call(client, model, effort, uid, report, retries=6):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=model,
                max_tokens=8000,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"<report>\n{report}\n</report>"}],
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
            )
            if r.stop_reason == "refusal":
                return {"uid": uid, "model": model, "error": "refusal"}
            text = next(b.text for b in r.content if b.type == "text")
            data = json.loads(text)
            u = r.usage
            return {
                "uid": uid,
                "model": model,
                "effort": effort,
                "states": {k: v["state"] for k, v in data.items()},
                "evidence": {k: v["evidence"] for k, v in data.items()},
                "usage": {
                    "in": u.input_tokens,
                    "out": u.output_tokens,
                    "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                    "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                },
            }
        except (RateLimitError, APIConnectionError):
            time.sleep(min(60, 2**attempt + 1))
        except APIStatusError as e:
            if e.status_code >= 500 or e.status_code == 429:
                time.sleep(min(60, 2**attempt + 1))
            else:
                return {"uid": uid, "model": model, "error": f"{e.status_code}: {str(e)[:200]}"}
        except (json.JSONDecodeError, StopIteration):
            time.sleep(1)
    return {"uid": uid, "model": model, "error": "retries_exhausted"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--subset", choices=["gold", "all", "nongold"], default="gold")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tr = pd.read_csv(ROOT / "data/raw/train.csv")
    if a.subset == "gold":
        tr = tr[tr["ACL"].notna()]
    elif a.subset == "nongold":
        tr = tr[tr["ACL"].isna()]
    if a.limit:
        tr = tr.head(a.limit)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            j = json.loads(line)
            if "error" not in j:
                done.add(j["uid"])
    todo = [
        (r.StudyInstanceUID, r.Report) for r in tr.itertuples() if r.StudyInstanceUID not in done
    ]
    print(f"{len(todo)} to do ({len(done)} already done) model={a.model}", flush=True)

    client = AnthropicBedrock(aws_region="us-east-1", max_retries=0)
    t0, n, errs, tok = time.time(), 0, 0, {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        futs = [ex.submit(call, client, a.model, a.effort, u, rep) for u, rep in todo]
        for fu in as_completed(futs):
            res = fu.result()
            with _lock:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
            n += 1
            if "error" in res:
                errs += 1
            else:
                for k in tok:
                    tok[k] += res["usage"][k]
            if n % 100 == 0 or n == len(todo):
                el = time.time() - t0
                print(
                    f"{n}/{len(todo)} errs={errs} {el:.0f}s eta={el / n * (len(todo) - n):.0f}s tokens={tok}",
                    flush=True,
                )
    print("DONE tokens", tok)


if __name__ == "__main__":
    main()
