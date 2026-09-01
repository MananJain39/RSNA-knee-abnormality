"""Public classical-pipeline API with lazy model imports.

Dataset, collation, loss, and submission utilities are intentionally usable on
CPU-only machines. Model/trainer symbols load only when requested, so a data
contract check does not require the optional ``classical`` dependency extra.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DINOv2Backbone": ("src.classical.dinov2_base", "DINOv2Backbone"),
    "KneeModel": ("src.classical.dinov2_base", "KneeModel"),
    "PerDiagnosisHead": ("src.classical.dinov2_base", "PerDiagnosisHead"),
    "KneeMRIDataset": ("src.classical.dataset", "KneeMRIDataset"),
    "build_default_volume_index": ("src.classical.dataset", "build_default_volume_index"),
    "collate_batch": ("src.classical.collate", "collate_batch"),
    "confidence_weighted_bce": ("src.classical.loss", "confidence_weighted_bce"),
    "filter_train_rows_for_cluster": (
        "src.classical.cluster_specialist",
        "filter_train_rows_for_cluster",
    ),
    "make_specialist": ("src.classical.cluster_specialist", "make_specialist"),
    "specialist_router": ("src.classical.cluster_specialist", "specialist_router"),
    "rank_mean_logits": ("src.classical.ensemble", "rank_mean_logits"),
    "to_probabilities": ("src.classical.ensemble", "to_probabilities"),
    "ClassicalPredictor": ("src.classical.inference", "ClassicalPredictor"),
    "ClassicalTrainer": ("src.classical.trainer", "ClassicalTrainer"),
    "SubmissionWriter": ("src.classical.submission", "SubmissionWriter"),
    "TrainResult": ("src.classical.trainer", "TrainResult"),
}


def __getattr__(name: str):
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
