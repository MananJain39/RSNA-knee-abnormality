"""Classical training and inference pipeline.

Re-exports the public surface so callers can do
``from src.classical import KneeMRIDataset, ClassicalTrainer, ...``
without knowing which submodule each symbol lives in.
"""

from src.classical.cluster_specialist import (
    filter_train_rows_for_cluster,
    make_specialist,
    specialist_router,
)
from src.classical.collate import collate_batch
from src.classical.dataset import (
    KneeMRIDataset,
    build_default_volume_index,
)
from src.classical.dinov2_base import (
    DINOv2Backbone,
    KneeModel,
    PerDiagnosisHead,
)
from src.classical.ensemble import rank_mean_logits, to_probabilities
from src.classical.inference import ClassicalPredictor
from src.classical.loss import confidence_weighted_bce
from src.classical.submission import SubmissionWriter
from src.classical.trainer import ClassicalTrainer, TrainResult

__all__ = [
    "DINOv2Backbone",
    "KneeModel",
    "PerDiagnosisHead",
    "KneeMRIDataset",
    "build_default_volume_index",
    "collate_batch",
    "confidence_weighted_bce",
    "filter_train_rows_for_cluster",
    "make_specialist",
    "rank_mean_logits",
    "specialist_router",
    "to_probabilities",
    "ClassicalPredictor",
    "ClassicalTrainer",
    "SubmissionWriter",
    "TrainResult",
]
