# 01 — Report → label extraction (LLM, closed vocabulary)

**Goal:** turn 4,349 free-text multilingual radiology reports into training targets for the 12 findings.
Only 58 studies ("gold-58") carry expert labels; they are used **only for evaluation**.

**Method**
- `scripts/llm_labels.py`: Claude on Bedrock reads each report and assigns every finding one of 5 states:
  `negated < not_mentioned < below_threshold < equivocal < positive`, using the expert annotation
  definitions (e.g. ACL = high-grade partial/complete tear; Effusion = moderate/large only).
  JSON-schema-constrained output, evidence quote per finding, system prompt cached.
- `scripts/score_labels.py`: AUC of the ordinal states vs gold-58 + P(gold+ | state) table.
- `scripts/build_labels.py`: states → soft labels (0.02 / 0.10 / 0.30 / 0.60 / 0.95), Sonnet 0.7 + Haiku 0.3 blend,
  Synovitis-when-silent imputed from Effusion. Output `data/labels_v1.csv` (gold rows keep their 0/1 labels).

**Why closed vocabulary instead of asking for a probability:** LLMs classify reliably but are badly
calibrated; a deterministic state→value map is reproducible and tunable.

**Smoke test:** 5 gold reports on Sonnet 5.5 → valid JSON, sensible evidence (`smoke.jsonl`).

Files: `raw_{haiku,sonnet,opus}_gold.jsonl`, `raw_{sonnet,haiku}_nongold.jsonl`, `bakeoff_gold.txt`, `build_labels.txt`.
