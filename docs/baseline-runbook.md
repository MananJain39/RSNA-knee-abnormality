# Classical Baseline Runbook

Run the first real experiment on Kaggle, where the complete competition DICOM
tree is available. Do not start specialists until this global main-branch run
has a recorded gold-58 score.

## 1. Prepare the runtime

Clone this repository in a Kaggle notebook with Internet enabled, then install
the GPU-side dependencies:

```bash
!git clone https://github.com/MananJain39/RSNA-knee-abnormality.git
%cd RSNA-knee-abnormality
!pip install -q torch transformers scipy pydicom pandas scikit-learn pyyaml tqdm
```

Set `COMP_DIR` to Kaggle's competition mount and use its CSV files with the
repository's label files. The repository's `data/raw/` CSVs and
`data/processed/labels_final.csv` must be present; attach or create a private
Kaggle dataset containing them if the clone does not include them.

## 2. Create inputs

```bash
python scripts/build_classical_split.py --tag v1
python scripts/preprocess_volumes.py \
  --dicom-dir "$COMP_DIR/train_series" \
  --series-csv "$COMP_DIR/train_series.csv" \
  --output-dir /kaggle/working/volumes \
  --metadata-output /kaggle/working/series_metadata.csv
```

Confirm the preprocessing failures CSV is empty or investigate every failure
before training. The output must contain all 58 gold-validation studies.

## 3. Train the main baseline

```bash
python scripts/train_classical.py \
  --mode main \
  --data-dir /kaggle/working/volumes \
  --split-tag v1 \
  --output-dir /kaggle/working/checkpoints/main \
  --device cuda
```

## 4. Evaluate the actual benchmark

```bash
python scripts/evaluate_classical.py \
  --main-checkpoint /kaggle/working/checkpoints/main/best.pt \
  --data-dir /kaggle/working/volumes \
  --split-tag v1 \
  --output-dir /kaggle/working/metrics/main \
  --device cuda
```

Record `gold_metrics.json`, `gold_per_label_metrics.csv`, the checkpoint, and
the experiment JSONL as one immutable baseline artifact. Only compare later
specialist/ensemble experiments against this same split and gold evaluation.
