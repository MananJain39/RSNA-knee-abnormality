"""Study-level train/validation split builder.

The split invariant: studies that share a verbatim report text MUST land on the
same side of the split.  We use the normalized report text as the grouping key
when ``group_col='Report'`` is requested; this prevents label leakage through
identical radiology-report templates that occur 5-37 times across the corpus.

The gold-58 expert-labeled studies are always reserved on the validation side
unless explicitly opted out, because they are the only ground-truth signal in
the competition and serve as the held-out benchmark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import GOLD_STUDY_COUNT, LABEL_COLUMNS
from src.data.loader import load_train


def _normalize_report(text: str | float) -> str:
    """Normalize a report string for grouping.

    Whitespace is collapsed and the text is case-folded so trivial formatting
    differences do not split a duplicate-report group into separate groups.

    NaN reports are pre-substituted upstream by ``make_split`` with a unique
    sentinel (``<NAN_REPORT_<index>>``) so each missing-report study gets
    its own group. By the time we get here, no row should have a NaN
    ``text`` — but the NaN branch is kept as defense-in-depth and returns
    a sentinel string (not NaN) so downstream ``category`` encoding
    cannot blow up.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "<NAN_REPORT>"
    return " ".join(str(text).split()).casefold()


def _isolate_nan_reports(df: pd.DataFrame, group_col: str) -> None:
    """Replace NaN reports in-place with unique sentinels.

    Two NaN-report studies therefore produce two distinct normalized
    strings, and the splitter treats them as two distinct groups rather
    than merging them into one.
    """
    nan_mask = df[group_col].isna()
    if nan_mask.any():
        nan_indices = df.index[nan_mask].tolist()
        df.loc[nan_mask, group_col] = [
            f"<NAN_REPORT_{i}>" for i in nan_indices
        ]


def _build_group_ids(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Compute integer group IDs for each row, isolating NaN groups.

    Mutates ``df`` in place to substitute NaN values in ``group_col`` with
    unique sentinels, then returns the resulting integer group IDs.
    """
    _isolate_nan_reports(df, group_col)
    normalized = df[group_col].map(_normalize_report)
    return normalized.astype("category").cat.codes


def make_split(
    train_df: pd.DataFrame | None = None,
    val_frac: float = 0.20,
    seed: int = 42,
    group_col: str = "Report",
    reserve_gold_to_val: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split studies into train/val respecting report groups.

    Args:
        train_df: Training DataFrame (loaded from disk if None).
        val_frac: Fraction of studies to put in validation.
        seed: Random seed for reproducibility.
        group_col: Column whose value identifies a leak group. The same value
            guarantees the row lands on the same split side.
        reserve_gold_to_val: If True, gold-labeled studies are forced to the
            validation side so they are never used for training targets.

    Returns:
        (train_df, val_df) — disjoint subsets of ``train_df``'s StudyInstanceUIDs.
    """
    if train_df is None:
        train_df = load_train()

    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    work = train_df.copy()
    work["_group_id"] = _build_group_ids(work, group_col).to_numpy()

    if reserve_gold_to_val:
        gold_mask = work[LABEL_COLUMNS[0]].notna()
        gold_count = int(gold_mask.sum())
        if gold_count != GOLD_STUDY_COUNT:
            print(
                f"warning: expected {GOLD_STUDY_COUNT} gold studies, found {gold_count}"
            )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=seed
    )

    train_idx, val_idx = next(
        splitter.split(work, groups=work["_group_id"].to_numpy())
    )

    if reserve_gold_to_val:
        gold_uids = set(work.loc[work[LABEL_COLUMNS[0]].notna(), "StudyInstanceUID"])
        gold_group_ids = set(
            work.loc[work["StudyInstanceUID"].isin(gold_uids), "_group_id"].tolist()
        )

        train_groups = set(work.iloc[train_idx]["_group_id"].tolist())
        val_groups = set(work.iloc[val_idx]["_group_id"].tolist())

        groups_to_move = (train_groups & gold_group_ids) - val_groups

        if groups_to_move:
            move_mask = work["_group_id"].isin(groups_to_move)
            move_idx = work.index[move_mask].tolist()

            train_idx = [i for i in train_idx if i not in set(move_idx)]
            val_idx = list(val_idx) + move_idx

    train_split = work.iloc[train_idx].drop(columns=["_group_id"]).reset_index(drop=True)
    val_split = work.iloc[val_idx].drop(columns=["_group_id"]).reset_index(drop=True)

    _validate_split(train_split, val_split, group_col, reserve_gold_to_val)

    return train_split, val_split


def _validate_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_col: str,
    reserve_gold_to_val: bool,
) -> None:
    """Assert the split is safe: no UID overlap, no report-group leak, gold on val."""
    train_uids = set(train_df["StudyInstanceUID"])
    val_uids = set(val_df["StudyInstanceUID"])
    overlap = train_uids & val_uids
    if overlap:
        raise AssertionError(f"{len(overlap)} study UIDs appear in both splits")

    def _norm(s: pd.Series) -> set[str]:
        return {_normalize_report(x) for x in s.tolist()}

    leaked_groups = _norm(train_df[group_col]) & _norm(val_df[group_col])
    if leaked_groups:
        raise AssertionError(
            f"{len(leaked_groups)} '{group_col}' groups span both splits "
            "(expected with raw GroupShuffleSplit — this should be 0)"
        )

    if reserve_gold_to_val:
        gold_in_train = train_df[LABEL_COLUMNS[0]].notna().sum()
        if gold_in_train > 0:
            raise AssertionError(
                f"{gold_in_train} gold studies ended up in train split"
            )

    if "<NAN_REPORT>" in leaked_groups:
        raise AssertionError("NaN report grouped into multiple studies (impossible)")


def persist_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path | None = None,
    tag: str = "v1",
) -> dict[str, Path]:
    """Write train/val CSVs to ``data/processed/splits/{tag}/``.

    Returns a dict with the paths written.
    """
    if output_dir is None:
        from src.config import PROCESSED_DATA_DIR

        output_dir = PROCESSED_DATA_DIR / "splits" / tag
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    return {"train": train_path, "val": val_path}


def load_split(
    tag: str = "v1", output_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a persisted split."""
    if output_dir is None:
        from src.config import PROCESSED_DATA_DIR

        output_dir = PROCESSED_DATA_DIR / "splits" / tag
    output_dir = Path(output_dir)
    train_df = pd.read_csv(output_dir / "train.csv")
    val_df = pd.read_csv(output_dir / "val.csv")
    return train_df, val_df
