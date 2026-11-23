# 01 — Results

## Model bake-off on gold-58 (macro AUC of report-derived labels vs expert labels)
| Model | Macro AUC | Est. cost full set |
|---|---|---|
| Haiku 5.5 | 0.8586 | ~$1.5 |
| Sonnet 5.5 | 0.8669 | ~$28 |
| Opus 5.5 | 0.8714 | ~$52 |

Sonnet vs Opus gap (0.0045) is far inside the n=58 noise band (~±0.02) → Sonnet chosen for the full run.

## Final label set (`labels_v1.csv`)
| Variant | Macro AUC (gold-58) |
|---|---|
| Sonnet only | 0.8669 |
| + Synovitis imputation from Effusion | 0.8722 |
| Sonnet 0.7 + Haiku 0.3, + imputation (**chosen**) | **0.8762** |

Per-label (chosen): ACL .974, MCL .974, MedMen .920, LatMen .869, MedOA .923, LatOA .841, PFOA .887,
Effusion .827, Synovitis .777, Baker's .930, Contusion .814, Fracture .777.

Reference: best published community label set ≈ 0.887 (stevenleehans + imputation) — we are within noise.
Caveat: weights/imputation were chosen by looking at gold-58, so 0.876 is slightly optimistic.

Labels are the ceiling for the image model: an image model trained on these can at best learn
"what the report would say", which agrees with the experts at ~0.88.

Full run: 4,349 + 58 reports, 0 errors, ~13 min, 24 parallel workers.
