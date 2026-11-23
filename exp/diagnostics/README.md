# Diagnostics — why the first model (A1) did not learn, + backbone benchmark + laterality check

## A1 collapse and the D/E diagnostic runs
All on fold 0; D/E runs use 1,200 training studies, 300 validation studies, 3 epochs unless noted.

| Run | Backbone | Head | LR (CNN / head) | Other | Val AUC | Gold AUC | pred_std |
|---|---|---|---|---|---|---|---|
| A1 | ConvNeXt-Nano (in12k→in1k) | attn | 3e-4 / 1e-3 | full data, 12 ep planned | 0.51 @ ep 2 | 0.46 | ~0 |
| D0 | ConvNeXt-Nano | attn | 1e-4 / 5e-4 | | 0.633 | 0.630 | 0.0004 |
| D1 | ConvNeXt-Nano | attn | 3e-5 / 3e-4 | | 0.658 | 0.673 | 0.039 |
| D2 | ResNet34d | attn | 3e-4 / 1e-3 | | 0.659 | 0.602 | 0.000 |
| D3 | ResNet34d | attn | 3e-4 / 1e-3 | **overfit**: 48 train studies, 25 ep | train 0.83 | — | 0.044 |
| **E1** | **ResNet34d** | **pool** | 3e-4 / 1e-3 | | **0.767** | **0.745** | 0.118 |
| E3 | ResNet34d | pool | 3e-4 / 1e-3 | **overfit**: 48 studies, 15 ep | train **0.985** | — | 0.116 |
| E0/E2 | ConvNeXt-Nano | pool / attn2 | 1e-4 / 5e-4 | | stopped (too slow; GPUs needed for B1) | | |

`pred_std` = standard deviation of predictions across validation patients, averaged over labels. ~0 ⇒ the model gives
everyone the same answer.

**Root cause:** the attention head computed `LayerNorm(attention_output + query)`. The query vector is the same for every
patient and the attention output starts tiny, so after normalisation every patient's vector was ~identical; the output
weights were also zero-initialised. Signal from the images was drowned → near-constant predictions.
**Fix:** "pool" head — masked max+mean over slices per series, concatenate series, linear layer. (`attn2` = attention
without the query residual is kept in the code but untested at full scale.)

## Backbone throughput (L40S, 480 images 224², fwd+bwd, bf16, steady state)
| Backbone | img/s | Notes |
|---|---|---|
| resnet34d.ra2_in1k | ~2,660 (channels-last) / ~1,690 (NCHW) | chosen for B1 |
| tf_efficientnetv2_b0.in1k | ~1,890 | B2 |
| convnext_nano.in12k_ft_in1k | ~1,520 | |
| efficientnet_b3 (first-pass bench) | much slower, 42 GB at bs 6 | dropped |

(The first benchmark reported 5× lower numbers — those included cuDNN auto-tuning warm-up; the profile in
`/tmp/claude/prof.py` measured steady state.)

## Laterality (left/right) check — `scripts/laterality.py`
- Laterality source after preprocessing: DICOM tag 2,201 studies, series description 74, position fallback 2,132
  (unreliable: knees are scanned at the magnet centre, x ≈ 0).
- Self-supervised test: ResNet18 trained (1,500 steps, bs 64, lr 3e-4) to tell canonical vs mirrored coronal/axial
  slices, on 1,820 tagged studies; evaluated on 455 held-out tagged studies.
- **Held-out accuracy 99.1% (left knees 99.1%, right knees 99.1%)** ⇒ our canonicalisation/flip logic is correct.
- On the 2,132 untagged studies it was confident (p<0.1 or >0.9) for only 21% and flagged 182 as mirrored ⇒ those
  studies look different (other sites/scanners). Correction not applied — small expected gain, uncertain correctness.
