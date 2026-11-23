# Leading solutions compared with this repository

Research date: October 30, 2026. Source and implementation review only; no model training or paid annotation was run. Interpret the requested gain provisionally as +0.005 macro ROC AUC. This metric is not classification accuracy.

## Top-ten retrieval coverage

| Rank | Evidence retrieved |
| --- | --- |
| 1 | Full [first-party write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution) |
| 2 | [Write-up located](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752370); body unavailable to retrieval |
| 3 | No verifiable solution retrieved in this search |
| 4 | No verifiable solution retrieved in this search |
| 5 | [Write-up located](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752190); body unavailable |
| 6 | No verifiable solution retrieved in this search |
| 7 | [Write-up located](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752217); body unavailable |
| 8 | [Write-up located](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752219); body unavailable |
| 9 | No verifiable solution retrieved in this search |
| 10 | First-party methods summary from the team's [NDA study record](https://nda.nih.gov/study.html?id=3430) |

Ranks reflect publication claims, not an independently exported final leaderboard. This is an incomplete top-ten comparison. Located titles do not establish that code or checkpoints were retrieved.

## What explains strong scores

The first-place team generated image/report probabilities, anatomy ROIs and lesion boxes. A ConvNeXt slice encoder pooled within finding-specific ROIs; lesion and report-embedding objectives supplied training-only supervision. Physical crops and laterality normalization preserved detail. Their 256-pixel efficiency model scored private 0.954; a 384-pixel five-fold ConvNeXt component scored private 0.958. These are component scores, not the overall winning ensemble score. Sequential changes improved gold-58 AUC from 0.9319 to 0.9432, which does not forecast our gain. See the first-place source above.

The tenth-place team's NDA record reports private 0.957. OAI images plus semiquantitative readings supplied seven findings with validity masks. Image-assisted mapping improved public AUC roughly 0.002–0.003 versus a rule-based mapping; additional follow-up data did not consistently help. This supports targeted external supervision rather than indiscriminate dataset expansion.

Outside the top ten, the [21st-place matched comparison](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752125) is particularly useful: changing report-consensus targets to image-informed soft labels increased ConvNeXt-Small gold AUC from 0.91472 to 0.94282 with training settings held fixed. This is development evidence, not a private leaderboard gain. Their probability blend also demonstrates that rank averaging is not mandatory for AUC.

## Code findings and experiment priorities

| Priority | Observed implementation | Recommended comparison |
| --- | --- | --- |
| 0 | No immutable real-data checkpoint/gold-metrics bundle was found. Local logs include short smoke-style runs; planning documents still describe a baseline gate. | Establish actual baseline, sample count, split, label version, checkpoint and per-label AUC. Synthetic scores do not establish current accuracy. |
| 1 | `src/labels/llm_extractor.py` says both unmentioned=0.5 and unmentioned=0.0. Broad ACL/MCL sprains and small effusions are counted as positive, conflicting with stricter task descriptions in project research. | Reconcile with official annotation guidance. Represent explicit negative, below-threshold, uncertain, unmentioned and extraction failure separately. Compare versioned targets under identical training. |
| 2 | `KneeModel` freezes DINOv2 by default; the default trainer factory uses it. | Compare frozen features against partial unfreezing, then full fine-tuning with a smaller backbone learning rate. Verify nonzero backbone gradients. |
| 3 | Dataset samples ten central-band three-slice groups per slot. Trainer averages group logits and then available slot logits. Patch attention exists, but learned slice/series pooling does not. | Compare broader slice coverage and finding-specific slice/series attention; inspect excluded slices for missed evidence. |
| 4 | Preprocessing already normalizes laterality and uses a physical 130-mm center crop at 336 px. It does not localize finding-specific anatomy. | Compare global context plus anatomy crops, then ROI pooling. More pixels cannot recover anatomy removed before resizing. |
| 5 | No image-reading annotation pipeline or lesion/report auxiliary heads were found in the classical branch. | Pilot image-assisted relabeling on random cases plus disagreement cases. Expand only after a matched student comparison. Add report-embedding supervision as a separate experiment. |
| 6 | Cluster specialists and fixed 50/50 rank blending exist, with unverified real-data benefit. | Focus specialists on measured weak findings; compare probability, logit and rank blends on held-out predictions. |

Changing our extraction prompt alone will not change the existing `labels_final.csv`: `scripts/process_labels.py` constructs it from public-source label tables. Track target provenance before attributing any improvement to a prompt change.

Augmentation, confidence-weighted BCE, EMA, physical crops, laterality normalization and duplicate-report grouping already exist. Older unchecked planning items are not reliable evidence that these features are missing.

The claim in `src/classical/ensemble.py` and project research that AUC requires rank averaging is incorrect. Monotonic transforms preserve one model's ranking; combining transformed models can change the ensemble ranking. Choose the operator experimentally.

## Measuring another 0.005

Freeze the baseline and change one factor per experiment. Use paired study-level bootstrap intervals and several seeds for close comparisons. Keep duplicate-study/report groups together. Repeatedly selecting prompts, checkpoints and weights against the same 58 expert cases overfits that development set.

If all gold cases are excluded from every training fold, all fold models can predict them. If gold cases participate in training, each evaluation case requires genuinely held-out ancestry, including annotation guides and teacher targets. Weak-label OOF AUC and expert-label AUC answer different questions.

Three labels each improving by 0.020 yield +0.005 macro AUC if the other nine remain unchanged. This is arithmetic, not a predicted gain. Start with label consistency, backbone adaptation and anatomy coverage; add external data or large ensembles after measuring remaining errors.

The [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) lists October 22, 2026 as the final submission deadline. Further improvements concern project development or late evaluation rather than changing the original final standing.
