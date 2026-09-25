# Commit guide — RSNA Knee MRI project

This is the plain-English history of the repository. It is written for
explaining the project in a review, interview, or handoff. Each section says
what a commit changed, why it matters, and the short version to say out loud.

## Project in one minute

The project predicts 12 knee abnormalities from MRI studies. The pipeline is:

```text
Competition CSVs + DICOMs
  -> validate/load studies and series
  -> put physical slices in order, normalize and resize them
  -> organise series into five anatomical/contrast slots
  -> build labels from reports, with confidence weights and gold overrides
  -> make a leak-safe train/validation split
  -> cluster scans with similar acquisition characteristics
  -> train a global DINOv2 model and small cluster specialists
  -> route each study to its specialist and rank-ensemble predictions
  -> validate and write a competition-format submission
```

Important terms:

- **Study**: one patient examination, identified by `StudyInstanceUID`.
- **Series**: one acquisition inside a study, identified by `SeriesInstanceUID`.
- **Slot**: a useful plane × contrast category, such as `Sagittal_Fluid`.
- **Gold labels**: the 58 expert-labelled studies. These are the most trusted
  evaluation signal.
- **Pseudo/LLM labels**: labels inferred from reports. They expand training
  data but are confidence-weighted because they are less certain.
- **2.5D input**: three neighbouring MRI slices are treated as a 3-channel
  image for a 2D visual model.

## How to keep this file current

When adding a commit, append an entry with:

1. short hash and subject;
2. changed files;
3. what changed and why;
4. the important control/data flow; and
5. a one-sentence explanation for non-specialists.

Do not edit old entries merely because hashes change after a history rewrite;
update the hash and preserve the explanation.

---

## Foundation and data contracts

### `a53c877` — Initial project setup

**Changed:** initial package layout, YAML config, data loaders/schema checks,
label extraction modules, validation modules, placeholder imaging modules,
small raw/sample CSVs, and baseline tests.

**What it does:** establishes a conventional `src/`, `tests/`, `scripts/`,
`configs/`, and `data/` project. `src/data/loader.py` reads the competition
tables; `src/data/schema.py` checks their required columns and legal values.
It also establishes the 12 target labels and basic label-quality metrics.

**Why it matters:** later work can import a stable project interface instead
of every notebook reimplementing paths, column names, and validation rules.

**Say it like this:** “This commit created the project skeleton and the
contracts that keep data and labels consistent across the pipeline.”

### `c27b46e` — Wire YAML config into `src/config.py`

**Changed:** `src/config.py` now reads `configs/default.yaml` rather than
duplicating settings in Python.

**What it does:** resolves project-relative paths, label names, extraction
settings, validation settings, image shape, and experiment settings into
importable constants.

**Why it matters:** experiments are reproducible: changing a value in the
YAML configuration changes the whole application consistently.

**Say it like this:** “We moved experiment settings out of scattered code
and into one reproducible YAML configuration.”

### `ed93f51` — Add Kaggle label downloader script

**Changed:** `scripts/download_labels.py`.

**What it does:** calls the Kaggle CLI to download and unzip three public
report-label datasets into `data/external/`, keeping each source separate.

**Why it matters:** external label sources are inputs to the label-selection
workflow, but downloading them is explicit and repeatable rather than manual.

**Say it like this:** “This made third-party report-label data reproducible to
fetch instead of a manual prerequisite.”

### `dc70a90` — Add label processing and validation pipeline

**Changed:** `scripts/process_labels.py`.

**What it does:** loads three public LLM/pseudo-label sources, compares them
with the 58 expert labels using macro AUC, selects the strongest non-leaking
source, imputes unaddressed synovitis from effusion, and writes
`data/processed/labels_final.csv`. It also adds `<label>_conf` confidence
columns. Labels at 0.5 (“not addressed”) receive low confidence; more certain
labels receive higher loss weight. Gold labels override LLM labels when both
exist.

**Why it matters:** the model can train on many report-derived labels while
preserving expert labels as the final authority where available.

**Say it like this:** “We turned several noisy report-label sources into one
training table, measured against the gold set and weighted by confidence.”

### `187f7d0` — Add processed LLM labels

**Changed:** `data/processed/labels_final.csv`.

**What it does:** commits the generated label table used by downstream data
and model code.

**Why it matters:** downstream experiments have a concrete, versioned label
artifact rather than depending on an unrecorded preprocessing run.

**Say it like this:** “This checks in the selected processed labels so model
runs use a known version of the training targets.”

---

## MRI geometry, loading, and preprocessing

### `22b6e6a` — Implement DICOM geometry utilities

**Changed:** `src/imaging/geometry.py`, `tests/test_geometry.py`.

**What it does:** derives slice normal vectors and physical positions from
DICOM orientation and position metadata, sorts slices in anatomical order,
computes image centres, and normalizes laterality labels.

**Why it matters:** filesystem names are not a reliable anatomical ordering.
Correct slice order is essential before making volumes or adjacent-slice
groups.

**Say it like this:** “We use DICOM geometry—not filename order—to reconstruct
each MRI volume in its real physical slice order.”

### `55e2dd2` — Implement DICOM loader

**Changed:** `src/imaging/dicom_loader.py`, `tests/test_dicom_loader.py`.

**What it does:** safely reads DICOM files, turns pixel arrays into volumes,
loads a series or study from the expected directory layout, exposes selected
metadata, and lists a study’s series.

**Why it matters:** it is the boundary between raw medical-image files and
NumPy arrays used by image preprocessing and clustering.

**Say it like this:** “This is the DICOM-to-volume adapter: it reads MRI files,
orders them correctly, and exposes both pixels and acquisition metadata.”

### `c27a300` — Complete image preprocessing pipeline

**Changed:** `src/imaging/preprocessing.py`, `tests/test_preprocessing.py`.

**What it does:** adds image normalization, resize/crop helpers, percentile
normalization for whole series, tensor conversion, central-band sampling, and
2.5D group creation.

**Important flow:** raw intensity volume → percentile normalize → crop/resize
to the configured size → select central anatomy → group adjacent slices into
three-channel inputs.

**Why it matters:** MRI intensity ranges and dimensions vary across scanners.
The model needs inputs on a consistent numerical and spatial scale.

**Say it like this:** “We standardize every scan and convert it into the
three-slice image groups DINOv2 can consume.”

### `8436898` — Add slot builder for MRI series

**Changed:** `src/imaging/slot_builder.py`, `tests/test_slot_builder.py`.

**What it does:** maps series into five clinically useful plane × contrast
slots: sagittal fluid/structural, coronal fluid/structural, and axial fluid.
It reports slot fill rates, produces a presence mask, and maps diagnoses to
the slots likely to show them.

**Why it matters:** a study may not contain every acquisition. The slot mask
lets the model aggregate available evidence without treating missing series as
real images.

**Say it like this:** “Instead of treating every scan as identical, we organise
it by the anatomical view and contrast that are relevant to each finding.”

### `0efd228` — Add DICOM preprocessing notebook

**Changed:** `notebooks/01_dicom_preprocessing.ipynb`, generator script.

**What it does:** provides a runnable, documented workflow that loads DICOMs,
applies the preprocessing logic, and creates reusable processed artifacts.

**Why it matters:** it makes the raw-to-processed imaging step inspectable and
easy to run on Kaggle or a GPU environment.

**Say it like this:** “This is the practical notebook version of the imaging
pipeline, so preprocessing can be reproduced and inspected visually.”

### `ea99640` — Ignore local documentation files

**Changed:** `.gitignore`.

**What it does:** prevents local/generated documentation from accidentally
entering commits.

**Why it matters:** keeps the repository focused on source, tests, and
deliberately versioned artifacts.

---

## Evaluation safety and acquisition clustering

### `a6ef159` — Add study-level split with report-grouping guard

**Changed:** `src/data/splits.py`, `tests/test_splits.py`.

**What it does:** creates reproducible train/validation splits using
`GroupShuffleSplit`. It normalizes reports (case and whitespace) and keeps all
studies with the same report together. It also reserves gold-labelled studies
for validation by default, validates UID/report-group disjointness, and can
persist/reload the split CSVs.

**Why it matters:** repeated report templates can leak label information into
validation. Keeping duplicates together makes macro-AUC believable rather than
optimistically biased.

**Say it like this:** “We made validation honest by ensuring no duplicated
report template can appear on both sides of the split.”

### `90dbe1c` — Cluster series by raw acquisition features

**Changed:** `src/imaging/clustering.py`, `tests/test_clustering.py`.

**What it does:** builds per-series features from DICOM metadata—pixel spacing,
dimensions, slice count/thickness, field of view, contrast flags, and transfer
syntax. It standardizes them and assigns K-means or Gaussian-mixture clusters.
It can extract headers directly or use a saved metadata table, persist
assignments, and reduce series clusters to a majority cluster per study.

**Why it matters:** MRI protocols vary substantially. Cluster specialists can
learn on more homogeneous inputs instead of one small model seeing every
scanner/protocol variation at once.

**Say it like this:** “We group scans by how they were acquired, so specialist
models learn less noisy, more consistent image distributions.”

### `1b58a1e` — Wire split, clustering, and classical hyperparameters into config

**Changed:** `configs/default.yaml`, `src/config.py`.

**What it does:** adds configurable split values, clustering settings, and
classical-model settings. Defaults include an 80/20 split, four K-means
clusters, DINOv2-base for the global model, DINOv2-small for specialists, and
training/ensemble values.

**Why it matters:** the whole hierarchical strategy can be tuned without
editing implementation code.

---

## Hierarchical DINOv2 modeling

### `4440364` — Add classical package and DINOv2 main-branch model

**Changed:** `src/classical/__init__.py`, `src/classical/dinov2_base.py`.

**What it does:** introduces a DINOv2 backbone wrapper and `KneeModel`. The
backbone returns patch tokens; `PerDiagnosisHead` gives each of the 12 labels
its own learned query over those patches. Each slot’s groups are encoded and
later aggregated to study-level logits.

**Why it matters:** different abnormalities are visible in different regions;
a per-diagnosis attention head is more expressive than one shared classifier
for all findings.

**Say it like this:** “The global model uses pretrained visual features and
lets every diagnosis attend to its own relevant MRI patches.”

### `cdad113` — Add cluster specialist factory and study-to-cluster router

**Changed:** `src/classical/cluster_specialist.py`, tests.

**What it does:** creates a small DINOv2 specialist for a cluster, filters
training rows to that cluster, and routes a study UID to exactly one cluster.
The module reuses the global model’s output head, so both branches predict the
same 12 logits and can be combined.

**Why it matters:** this turns acquisition clustering into a usable modeling
branch rather than only an analysis table.

**Say it like this:** “After identifying a scan’s acquisition type, we send it
to the expert model trained specifically for that type.”

### `309b81b` — Add rank-invariant ensemble for main and cluster branches

**Changed:** `src/classical/ensemble.py`, tests.

**What it does:** converts each branch’s score for each label into a 0–1 rank
across studies, then takes a configurable weighted mean. A stable sigmoid
helper is also provided when regular probabilities are required.

**Why it matters:** macro-AUC depends on score ordering, not calibration. Rank
averaging therefore blends models in the same space as the competition metric.

**Say it like this:** “Because AUC rewards correct ordering, we ensemble models
by their ranks rather than trusting incompatible raw score scales.”

### `983af7f` — Scaffold dataset, collate, and loss with inline protocols

**Changed:** `src/classical/dataset.py`, `collate.py`, `loss.py`, tests.

**What it does:** defines typed protocols and the data contract:

- a dataset sample is `(slots, labels, weights)`;
- slots are grouped 2.5D images per named slot;
- the collate function produces `[B, 5, N, 3, H, W]`, labels, weights, and a
  `[B, 5]` missing-slot mask; and
- the loss is confidence-weighted binary cross entropy.

Unlabelled/not-addressed targets are represented as 0.5 with zero/low loss
weight, so they do not pretend to be reliable negatives.

**Why it matters:** this is the precise interface between image preparation and
PyTorch training.

### `270ac3b` — Register neural classes as `nn.Module` and correct attention einsum

**Changed:** `src/classical/dinov2_base.py`, `tests/test_dinov2_forward.py`.

**What it does:** converts the backbone, diagnosis head, and composite model
to proper PyTorch modules. It corrects the attention equations so patch tokens
`[B, patches, hidden]` are compared to diagnosis queries
`[12, hidden]`, producing distinct `[B, 12]` logits.

**Why it matters:** before this, parameters would not be safely registered for
optimizers/checkpoints, and the head’s tensor math did not represent
per-diagnosis attention correctly.

**Say it like this:** “This made the model genuinely trainable in PyTorch and
fixed the attention calculation so each diagnosis gets its own output.”

### `105fee1` — Give each missing report its own split group

**Changed:** `src/data/splits.py`, tests.

**What it does:** replaces every missing report with a unique sentinel before
group encoding. A missing report no longer causes unrelated studies to be
merged into one split group.

**Why it matters:** report grouping should prevent genuine duplicates leaking;
it should not accidentally distort the split because metadata is absent.

---

## Training, inference, and delivery

### `e0d06f4` — Add config keys/constants for EMA, AMP, cosine, checkpoint

**Changed:** `configs/default.yaml`, `src/config.py`.

**What it does:** configures separate EMA decay for main/specialist models,
AMP enablement, cosine schedule length, and checkpoint directory. Existing
warmup settings are used by the trainer.

**Why it matters:** performance and stability choices are experiment settings,
not hard-coded behavior.

### `5223edc` — Add trainer with AdamW, cosine, EMA, and AMP

**Changed:** `src/classical/trainer.py`, tests.

**What it does:** adds `ClassicalTrainer` and `TrainResult`.

**Training flow:**

1. build a global or cluster-specialist `KneeModel`;
2. pass `model.parameter_groups()` to AdamW, so a trainable backbone (if
   unfrozen) has a lower learning rate than the diagnosis head;
3. apply a linear warmup followed by cosine learning-rate decay;
4. use AMP and gradient scaling on CUDA, but make both harmless on CPU;
5. encode every available slot/group, average group logits, mask absent slots,
   and average present slot logits into one `[B, 12]` study prediction;
6. train with differentiable confidence-weighted BCE;
7. update an exponential moving-average copy of the model;
8. evaluate EMA predictions with macro AUC each epoch; and
9. save the best checkpoint and JSONL experiment metrics.

Specialist mode uses `filter_train_rows_for_cluster` before training.

**Why it matters:** this is the first reusable loop that turns the dataset and
model definitions into actual checkpoints.

### `0ec62ce` — Add inference pipeline with main and cluster routing

**Changed:** `src/classical/inference.py`, tests.

**What it does:** adds `ClassicalPredictor`. It caches loaded checkpoints,
constructs a one-study dataset, produces masked slot-averaged logits, routes a
study to its specialist according to its cluster, and falls back to main-only
when no specialist assignment/checkpoint exists. It preferentially loads EMA
weights from a checkpoint and blends main/specialist scores through the rank
ensemble.

**Why it matters:** training a model is not enough; this converts checkpoints
and volumes into the 12 prediction columns required for submission.

### `51dc833` — Add submission writer with schema validation

**Changed:** `src/classical/submission.py`, tests.

**What it does:** adds `SubmissionWriter`. It requires exactly
`StudyInstanceUID + 12 label columns`, calls the shared submission-schema
validator, rejects duplicates, NaNs, and values outside `[0, 1]`, then writes
a deterministic CSV.

**Why it matters:** prevents silent submission-format mistakes after expensive
inference runs.

### `de838f0` — Migrate trainer to non-deprecated `torch.amp` API

**Changed:** `src/classical/trainer.py`.

**What it does:** replaces deprecated `torch.cuda.amp` calls with the current
`torch.amp` autocast/scaler API while retaining CUDA-only AMP behaviour.

**Why it matters:** removes framework deprecation warnings and keeps the
trainer compatible with modern PyTorch.

### `5d2adda` — Align submission sort with `sample_submission.csv` order

**Changed:** `src/classical/submission.py`, tests.

**What it does:** extends the writer with an optional sample-submission path.
When present, output rows follow that file’s exact UID order; otherwise it
falls back to lexical UID order. Unknown UIDs are deterministically placed
after known sample rows.

**Why it matters:** the submission has the expected row order even if the
competition evaluator joins by UID.

### `5cd512c` — Re-export classical pipeline classes

**Changed:** `src/classical/__init__.py`.

**What it does:** exposes dataset, collate, model, loss, specialists,
ensemble, trainer, predictor, and submission writer through `src.classical`.

**Why it matters:** scripts and users import one public package rather than
depending on internal file locations.

### `84c6754` — Add end-to-end pipeline smoke test

**Changed:** `tests/test_pipeline_smoke.py`.

**What it does:** exercises the six classical modules together with synthetic
data and a tiny mocked model: dataset → collate → trainer → checkpoint →
predictor → submission writer.

**Why it matters:** unit tests can all pass while interfaces disagree. This
test checks the full handoff chain without DICOM data, GPUs, or Hugging Face
network access.

### `444cb43` — Add training CLI entry point

**Changed:** `scripts/train_classical.py`.

**What it does:** provides `--mode main` and `--mode specialist` commands.
It loads processed labels, a persisted split, series clusters (converted to
study clusters), volume data, creates datasets, supports resume/checkpoint
paths, and invokes `ClassicalTrainer`.

**Why it matters:** the pipeline is usable from a terminal or Kaggle notebook
without custom glue code.

**Example:**

```bash
python scripts/train_classical.py --mode main
python scripts/train_classical.py --mode specialist --cluster-id 3
```

### `c6ccc86` — Add inference CLI entry point

**Changed:** `scripts/infer_classical.py`.

**What it does:** accepts a main checkpoint and optional `cluster_<id>.pt`
specialist directory, discovers specialist files, loads preprocessed volumes
and study clusters, runs `ClassicalPredictor`, and writes a submission through
`SubmissionWriter`. It can also compare overlapping predictions with gold
labels as an internal macro-AUC sanity check.

**Why it matters:** it closes the operational loop from trained checkpoints to
a valid submission file.

**Example:**

```bash
python scripts/infer_classical.py \
  --main-checkpoint data/processed/checkpoints/main/best.pt \
  --specialist-checkpoints-dir data/processed/checkpoints/specialists
```

### `824e208` — Support an overridable DICOM root

**Changed:** `src/config.py`.

**What it does:** lets `RSNA_DICOM_DIR` override the configured DICOM path at
runtime. When the environment variable is absent, the project still uses the
path from `configs/default.yaml` relative to the repository root.

**Why it matters:** the same code can run locally, in a mounted-volume setup,
or in Kaggle without editing the tracked experiment configuration.

**Say it like this:** “We made the raw MRI location portable, while keeping a
safe project default.”

### `9343b13` — Add a Kaggle DICOM sample-extraction kernel

**Changed:** `kaggle_kernels/dicom_sample_extract/extract.ipynb`,
`kernel-metadata.json`.

**What it does:** adds a private, internet-disabled Kaggle notebook attached
to the RSNA competition data. The initial version copied a deterministic
sample of 50 training-study DICOM trees and packaged it as an extract-only
artifact.

**Why it matters:** a small real-DICOM payload makes it possible to test the
loader and preprocessing workflow outside the full competition dataset.

**Say it like this:** “This gives us a compact, repeatable slice of the
competition images for real-world pipeline checks.”

### `04a080c` — Make Kaggle sample extraction reliable to download

**Changed:** `.gitignore`, `kaggle_kernels/dicom_sample_extract/extract.ipynb`,
`list_layout.ipynb`.

**What it does:** updates the extractor for Kaggle's competition-mounted path,
reduces the working sample to three studies while troubleshooting payload
delivery, writes a manifest with the selected UIDs and creation time, and
reports file count and total size. It also adds a layout-inspection notebook
and probes the kernel environment's Kaggle SDK/CLI authentication behavior.
`data/local_dicoms/` is now ignored so downloaded samples cannot be committed
accidentally.

**Why it matters:** the extraction workflow now verifies both its input layout
and its output payload, making failures to retrieve a sample diagnosable rather
than silent.

**Say it like this:** “We hardened the sample-download workflow and kept the
large local DICOM copies out of version control.”

### `55ca7eb` — Align Kaggle kernel metadata with its slug

**Changed:** `kaggle_kernels/dicom_sample_extract/kernel-metadata.json`.

**What it does:** changes the kernel title to match the published Kaggle slug,
`rsna-knee-dicom-sample-extract`.

**Why it matters:** consistent metadata prevents ambiguity when uploading,
updating, or locating the kernel on Kaggle.

**Say it like this:** “We made the kernel's displayed identity match the
artifact it publishes.”

### `7881334` — Track the Decision Log decisions 13–18

**Changed:** `.md Files/DECISION.md`.

**What it does:** adds the project decision record for the dataset and loss
contracts, CLI shape, main/specialist training modes, EMA use, portable
`RSNA_DICOM_DIR` configuration, and the Kaggle DICOM-sample workflow. The file
is force-added because the repository intentionally ignores most local
documentation under `.md Files/`.

**Why it matters:** implementation alone does not preserve the reasons behind
important modeling and operational choices. This keeps those decisions
available after the code is shared remotely.

**Say it like this:** “We versioned the reasoning behind the newest pipeline
and local-validation decisions, so future changes have their original context.”

---

## The complete explanation to use with others

“We built an end-to-end knee MRI classification pipeline. We first make the
data trustworthy: DICOM slices are physically ordered, normalized, converted
into clinically meaningful slots, and report-derived labels are compared
against expert labels and confidence-weighted. Our validation split prevents
duplicate-report leakage and reserves gold studies as a benchmark. Since MRI
acquisition protocols vary, we cluster series by raw scanner/protocol features.
The global DINOv2 model sees all data, while lightweight specialists learn the
more homogeneous acquisition clusters. Each diagnosis has its own attention
over image patches. During inference we route a study to its matching
specialist, blend it with the global prediction in rank space because the
metric is macro-AUC, and validate/write a competition-ready submission. The
whole flow has unit tests, an end-to-end synthetic smoke test, and CLI entry
points for training and inference.”
