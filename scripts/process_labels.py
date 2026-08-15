"""Process and validate downloaded LLM label datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

LABEL_COLUMNS = [
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


def macro_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame, label_columns: list) -> float:
    from sklearn.metrics import roc_auc_score

    auc_scores = []
    for col in label_columns:
        if col in y_true.columns and col in y_pred.columns:
            true_vals = y_true[col].values
            pred_vals = y_pred[col].values

            if len(np.unique(true_vals)) < 2:
                continue

            try:
                auc = roc_auc_score(true_vals, pred_vals)
                auc_scores.append(auc)
            except ValueError:
                continue

    return float(np.mean(auc_scores)) if auc_scores else 0.0


def load_gold_labels() -> pd.DataFrame:
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    gold = train.dropna(subset=[LABEL_COLUMNS[0]])
    return gold[["StudyInstanceUID"] + LABEL_COLUMNS]


def load_pilkwang() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_DIR / "pilkwang_rsna-knee-llm-labels" / "report_labels_v2.csv")
    return df[["StudyInstanceUID"] + LABEL_COLUMNS]


def load_stevenleehans() -> pd.DataFrame:
    df = pd.read_csv(
        EXTERNAL_DIR / "stevenleehans_rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv"
    )
    return df[["StudyInstanceUID"] + LABEL_COLUMNS]


def load_barun2104() -> pd.DataFrame:
    df = pd.read_csv(
        EXTERNAL_DIR
        / "barun2104_rsna-knee-stratified-folds-and-llm-soft-labels"
        / "train_folds_with_pseudo.csv"
    )
    pseudo_cols = [f"pseudo_{col}" for col in LABEL_COLUMNS]
    result = df[["StudyInstanceUID"] + pseudo_cols].copy()
    result.columns = ["StudyInstanceUID"] + LABEL_COLUMNS
    return result


def validate_against_gold(pred_df: pd.DataFrame, gold_df: pd.DataFrame, name: str) -> float:
    merged = pred_df.merge(gold_df, on="StudyInstanceUID", suffixes=("_pred", "_gold"))

    y_true = merged[[f"{col}_gold" for col in LABEL_COLUMNS]]
    y_pred = merged[[f"{col}_pred" for col in LABEL_COLUMNS]]

    y_true.columns = LABEL_COLUMNS
    y_pred.columns = LABEL_COLUMNS

    auc = macro_auc(y_true, y_pred, LABEL_COLUMNS)
    print(f"{name:30s}: Macro AUC = {auc:.4f} (n={len(merged)})")
    return auc


def compute_not_addressed_rates(df: pd.DataFrame) -> dict:
    rates = {}
    for col in LABEL_COLUMNS:
        na_count = (df[col] == 0.5).sum()
        rates[col] = na_count / len(df)
    return rates


def selective_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    effusion = df["Effusion"]
    synovitis_mask = df["Synovitis"] == 0.5

    df.loc[synovitis_mask & (effusion > 0.5), "Synovitis"] = 0.63
    df.loc[synovitis_mask & (effusion <= 0.5), "Synovitis"] = 0.22

    return df


def compute_confidence_weights(df: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(index=df.index, columns=LABEL_COLUMNS, dtype=float)

    for col in LABEL_COLUMNS:
        weights[col] = np.where(
            df[col] == 0.5,
            0.3,
            np.abs(df[col] - 0.5) * 2 + 0.5,
        )

    return weights


def main() -> None:
    print("Loading datasets...")
    gold = load_gold_labels()
    pilkwang = load_pilkwang()
    stevenleehans = load_stevenleehans()
    barun2104 = load_barun2104()

    print("\nValidating against gold 58...")
    auc_pilkwang = validate_against_gold(pilkwang, gold, "pilkwang")
    auc_stevenleehans = validate_against_gold(stevenleehans, gold, "stevenleehans v4_blend")
    validate_against_gold(barun2104, gold, "barun2104 pseudo")
    print("  (barun2104 pseudo labels for gold studies = gold labels, so AUC=1.0 is expected)")

    best_name = max(
        [("pilkwang", auc_pilkwang), ("stevenleehans", auc_stevenleehans)],
        key=lambda x: x[1],
    )[0]
    print(f"\nBest dataset (excluding barun2104): {best_name}")

    if best_name == "pilkwang":
        best_df = pilkwang
    else:
        best_df = stevenleehans

    print("\nComputing 'not addressed' rates...")
    na_rates = compute_not_addressed_rates(best_df)
    for col, rate in na_rates.items():
        print(f"  {col:20s}: {rate:.1%}")

    print("\nApplying selective imputation (synovitis from effusion)...")
    imputed_df = selective_imputation(best_df)

    print("\nComputing confidence weights...")
    weights = compute_confidence_weights(imputed_df)

    print("\nMerging with gold labels (override where available)...")
    final_df = imputed_df.merge(gold, on="StudyInstanceUID", suffixes=("", "_gold"), how="left")
    for col in LABEL_COLUMNS:
        gold_col = f"{col}_gold"
        if gold_col in final_df.columns:
            mask = final_df[gold_col].notna()
            final_df.loc[mask, col] = final_df.loc[mask, gold_col]
            final_df.drop(columns=[gold_col], inplace=True)

    print("\nAdding confidence weights...")
    for col in LABEL_COLUMNS:
        final_df[f"{col}_conf"] = weights[col]

    output_path = PROCESSED_DATA_DIR / "labels_final.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"\nSaved final labels to {output_path}")
    print(f"Shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")


if __name__ == "__main__":
    main()
